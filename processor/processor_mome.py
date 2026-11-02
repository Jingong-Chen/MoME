import logging
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda import amp

from loss.mome_losses import DependencyLoss
from loss.triplet_loss import TripletLoss
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from utils.parasitic_utils import set_parasitic_mode


def _compute_id_loss(score, target):
    if isinstance(score, list):
        if len(score) == 1:
            return F.cross_entropy(score[0], target)
        id_loss = [F.cross_entropy(s, target) for s in score[1:]]
        id_loss = sum(id_loss) / len(id_loss)
        return 0.5 * id_loss + 0.5 * F.cross_entropy(score[0], target)
    return F.cross_entropy(score, target)


def _compute_triplet_loss(feat, target, triplet):
    if isinstance(feat, list):
        if len(feat) == 1:
            return triplet(feat[0], target)[0]
        tri_loss = [triplet(f, target)[0] for f in feat[1:]]
        tri_loss = sum(tri_loss) / len(tri_loss)
        return 0.5 * tri_loss + 0.5 * triplet(feat[0], target)[0]
    return triplet(feat, target)[0]


def _pick_main_score(score):
    if isinstance(score, list):
        return score[0]
    return score


def _cycle_next(loader_iter, loader):
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
    return batch, loader_iter


def _eval_reid_mode(cfg, model, val_loader, num_query, mode_name, local_rank=0):
    device = "cuda"
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()
    model.eval()
    set_parasitic_mode(model.module if hasattr(model, "module") else model, mode_name == "full")

    for img, vid, camid, camids, target_view, _ in val_loader:
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, vid, camid))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    if (not cfg.MODEL.DIST_TRAIN) or dist.get_rank() == 0:
        logger = logging.getLogger("transreid.mome.train")
        logger.info(
            "[{}] mAP: {:.1%}, Rank-1: {:.1%}, Rank-5: {:.1%}".format(
                mode_name.upper(), mAP, cmc[0], cmc[4]
            )
        )
    return cmc, mAP


def do_train_mome(
    cfg,
    model,
    train_loader_f,
    train_loader_r,
    train_loader_df_only,
    val_loader,
    optimizer,
    scheduler,
    num_query,
    local_rank=0,
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    epochs = cfg.SOLVER.MAX_EPOCHS

    alpha = cfg.MOME.ALPHA
    beta = cfg.MOME.BETA

    device = "cuda"
    logger = logging.getLogger("transreid.mome.train")
    logger.info("start MoME training")

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            logger.info("Using {} GPUs for training".format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], find_unused_parameters=True
            )

    dep_loss_fn = DependencyLoss(cfg.MOME.MARGIN)
    triplet = TripletLoss() if cfg.MODEL.NO_MARGIN else TripletLoss(cfg.SOLVER.MARGIN)

    loss_total_meter = AverageMeter()
    loss_para_meter = AverageMeter()
    loss_clean_meter = AverageMeter()
    loss_dep_meter = AverageMeter()
    acc_meter = AverageMeter()
    scaler = amp.GradScaler()

    for epoch in range(1, epochs + 1):
        model.train()
        start_time = time.time()
        scheduler.step(epoch)

        loss_total_meter.reset()
        loss_para_meter.reset()
        loss_clean_meter.reset()
        loss_dep_meter.reset()
        acc_meter.reset()

        # 4 步 backward：L_dep 不对 para 求梯度（ce_para_dep.detach），每步只保留 1 个 batch + 1 条计算图，峰值显存 ≈ 标准 TransReID
        iter_r = iter(train_loader_r)
        iter_dep = iter(train_loader_df_only)

        for n_iter, (img_f, pid_f, cam_f, view_f) in enumerate(train_loader_f):
            (img_r, pid_r, cam_r, view_r), iter_r = _cycle_next(iter_r, train_loader_r)
            (img_dep, pid_dep, cam_dep, view_dep), iter_dep = _cycle_next(
                iter_dep, train_loader_df_only
            )

            optimizer.zero_grad()
            loss_para_val = 0.0
            loss_clean_val = 0.0
            loss_dep_val = 0.0
            acc_val = 0.0
            batch_size = img_f.shape[0]

            # Step 1: Parasitic on D_f → L_para.backward()（仅 1 个 batch 在 GPU）
            img_f = img_f.to(device)
            pid_f, cam_f, view_f = pid_f.to(device), cam_f.to(device), view_f.to(device)
            with amp.autocast(enabled=True):
                set_parasitic_mode(model.module if hasattr(model, "module") else model, True)
                score_para, feat_para = model(img_f, pid_f, cam_label=cam_f, view_label=view_f)
                ce_para = _compute_id_loss(score_para, pid_f)
                tri_para = _compute_triplet_loss(feat_para, pid_f, triplet)
                loss_para = ce_para + tri_para
            scaler.scale(loss_para).backward()
            loss_para_val = loss_para.item()
            score_para_main = _pick_main_score(score_para)
            acc_val = (score_para_main.max(1)[1] == pid_f).float().mean().item()
            del img_f, score_para, feat_para, ce_para, tri_para, loss_para, score_para_main

            # Step 2: Parasitic on D_f\D_r，只要 ce_para_dep 数值供 Step 4 用，不需求梯度
            # 用 no_grad 避免构建计算图、不保留中间激活，显存立刻释放
            img_dep = img_dep.to(device)
            pid_dep, cam_dep, view_dep = pid_dep.to(device), cam_dep.to(device), view_dep.to(device)
            with torch.no_grad():
                with amp.autocast(enabled=True):
                    set_parasitic_mode(model.module if hasattr(model, "module") else model, True)
                    score_para_dep, _ = model(img_dep, pid_dep, cam_label=cam_dep, view_label=view_dep)
                    ce_para_dep = _compute_id_loss(score_para_dep, pid_dep)
            img_dep_cpu = img_dep.cpu()
            del img_dep, score_para_dep

            # Step 3: Clean on D_r → (α * L_clean).backward()（仅 img_r 在 GPU）
            img_r = img_r.to(device)
            pid_r, cam_r, view_r = pid_r.to(device), cam_r.to(device), view_r.to(device)
            with amp.autocast(enabled=True):
                set_parasitic_mode(model.module if hasattr(model, "module") else model, False)
                score_clean, feat_clean = model(img_r, pid_r, cam_label=cam_r, view_label=view_r)
                ce_clean = _compute_id_loss(score_clean, pid_r)
                tri_clean = _compute_triplet_loss(feat_clean, pid_r, triplet)
                loss_clean = ce_clean + tri_clean
            scaler.scale(alpha * loss_clean).backward()
            loss_clean_val = loss_clean.item()
            del img_r, score_clean, feat_clean, ce_clean, tri_clean, loss_clean

            # Step 4: Clean on D_f\D_r → L_dep(ce_clean_dep, ce_para_dep).backward()（仅 img_dep 在 GPU）
            img_dep = img_dep_cpu.to(device)
            del img_dep_cpu
            with amp.autocast(enabled=True):
                set_parasitic_mode(model.module if hasattr(model, "module") else model, False)
                score_clean_dep, _ = model(img_dep, pid_dep, cam_label=cam_dep, view_label=view_dep)
                ce_clean_dep = _compute_id_loss(score_clean_dep, pid_dep)
                loss_dep = dep_loss_fn(ce_clean_dep, ce_para_dep)
            scaler.scale(beta * loss_dep).backward()
            loss_dep_val = loss_dep.item()
            del img_dep, score_clean_dep, ce_clean_dep, ce_para_dep, loss_dep

            scaler.step(optimizer)
            scaler.update()

            loss_total_val = loss_para_val + alpha * loss_clean_val + beta * loss_dep_val
            loss_total_meter.update(loss_total_val, batch_size)
            loss_para_meter.update(loss_para_val, batch_size)
            loss_clean_meter.update(loss_clean_val, batch_size)
            loss_dep_meter.update(loss_dep_val, batch_size)
            acc_meter.update(acc_val, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iter[{}/{}] L_total:{:.3f} L_para:{:.3f} "
                    "L_clean:{:.3f} L_dep:{:.3f} Acc:{:.3f} Base Lr:{:.2e}".format(
                        epoch,
                        n_iter + 1,
                        len(train_loader_f),
                        loss_total_meter.avg,
                        loss_para_meter.avg,
                        loss_clean_meter.avg,
                        loss_dep_meter.avg,
                        acc_meter.avg,
                        scheduler._get_lr(epoch)[0],
                    )
                )

        end_time = time.time()
        time_per_batch = (end_time - start_time) / max(len(train_loader_f), 1)
        if (not cfg.MODEL.DIST_TRAIN) or dist.get_rank() == 0:
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                    epoch, time_per_batch, train_loader_f.batch_size / time_per_batch
                )
            )

        if epoch % checkpoint_period == 0:
            model_to_save = model.module if hasattr(model, "module") else model
            if (not cfg.MODEL.DIST_TRAIN) or dist.get_rank() == 0:
                torch.save(
                    model_to_save.state_dict(),
                    os.path.join(cfg.OUTPUT_DIR, "{}_{}.pth".format(cfg.MODEL.NAME, epoch)),
                )

        if epoch % eval_period == 0:
            _eval_reid_mode(cfg, model, val_loader, num_query, mode_name="full", local_rank=local_rank)
            _eval_reid_mode(cfg, model, val_loader, num_query, mode_name="core", local_rank=local_rank)
            torch.cuda.empty_cache()
