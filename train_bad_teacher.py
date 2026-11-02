"""Bad Teacher unlearning baseline (Chundawat AAAI'23 ref [4]).

Two teachers:
  - good_teacher = standard model (frozen, knows D_f)
  - bad_teacher  = randomly initialized model with same architecture (frozen, knows nothing)

Student (init from good_teacher):
  on D_r:           KL( student || good_teacher )       — preserve retain knowledge
  on D_f \\ D_r:    KL( student || bad_teacher  )       — forget the rest

Plus optional small CE on D_r for stability (controlled by lambda_ce).
"""

import argparse
import contextlib
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import amp

from config import cfg
from datasets import make_dataloader_mome
from loss import make_loss
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from utils.logger import setup_logger
from utils.meter import AverageMeter
from utils.parasitic_utils import freeze_parasitic_parameters, set_parasitic_mode


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def _cycle_next(loader_iter, loader):
    try:
        return next(loader_iter), loader_iter
    except StopIteration:
        loader_iter = iter(loader)
        return next(loader_iter), loader_iter


def _kl(s, t, T=4.0):
    return F.kl_div(F.log_softmax(s / T, dim=-1), F.softmax(t / T, dim=-1), reduction="batchmean") * (T * T)


def _multi_kl(s_score, t_score, T=4.0):
    if isinstance(s_score, list):
        return sum(_kl(s, t, T) for s, t in zip(s_score, t_score)) / len(s_score)
    return _kl(s_score, t_score, T)


def _set_eval_friendly_train(m):
    """Put module in train mode but freeze BN/Dropout (for tuple-output forward)."""
    import torch.nn as nn
    m.train()
    for sub in m.modules():
        if isinstance(sub, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm, nn.Dropout)):
            sub.eval()


def run_bad_teacher(student, good_t, bad_t, loader_r, loader_df, optimizer, scheduler,
                    loss_fn, epochs, T, lam_ce, logger, scaler, device, max_grad_norm=5.0):
    student.train()
    # Teachers need train mode (for JPM tuple output) but BN/Dropout frozen.
    _set_eval_friendly_train(good_t)
    _set_eval_friendly_train(bad_t)
    for ep in range(1, epochs + 1):
        scheduler.step(ep)
        m_kl_r = AverageMeter(); m_kl_f = AverageMeter(); m_ce = AverageMeter(); skipped = 0
        t0 = time.time()
        df_iter = iter(loader_df)
        for img_r, pid_r, cam_r, view_r in loader_r:
            (batch_f, df_iter) = _cycle_next(df_iter, loader_df)
            img_f, pid_f, cam_f, view_f = batch_f
            img_r = img_r.to(device); pid_r = pid_r.to(device); cam_r = cam_r.to(device); view_r = view_r.to(device)
            img_f = img_f.to(device); pid_f = pid_f.to(device); cam_f = cam_f.to(device); view_f = view_f.to(device)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(enabled=True):
                s_r, feat_r = student(img_r, pid_r, cam_label=cam_r, view_label=view_r)
                with torch.no_grad():
                    g_r, _ = good_t(img_r, pid_r, cam_label=cam_r, view_label=view_r)
                kl_r = _multi_kl(s_r, g_r, T=T)
                ce_r = loss_fn(s_r, feat_r, pid_r, cam_r) if lam_ce > 0 else torch.zeros((), device=device)

                s_f, _ = student(img_f, pid_f, cam_label=cam_f, view_label=view_f)
                with torch.no_grad():
                    b_f, _ = bad_t(img_f, pid_f, cam_label=cam_f, view_label=view_f)
                kl_f = _multi_kl(s_f, b_f, T=T)

                loss = kl_r + kl_f + lam_ce * ce_r

            if not torch.isfinite(loss):
                skipped += 1; continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
            if not torch.isfinite(gn):
                skipped += 1; optimizer.zero_grad(set_to_none=True); continue
            scaler.step(optimizer); scaler.update()
            m_kl_r.update(kl_r.item()); m_kl_f.update(kl_f.item()); m_ce.update(ce_r.item() if lam_ce > 0 else 0.0)

        logger.info("BadT ep[{}/{}] kl_r:{:.3f} kl_f:{:.3f} ce:{:.3f} skip:{} time:{:.1f}s".format(
            ep, epochs, m_kl_r.avg, m_kl_f.avg, m_ce.avg, skipped, time.time() - t0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bad Teacher unlearning baseline")
    parser.add_argument("--config_file", default="", type=str)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--T", default=4.0, type=float)
    parser.add_argument("--lambda_ce", default=0.5, type=float)
    parser.add_argument("--max_grad_norm", default=5.0, type=float)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_seed(cfg.SOLVER.SEED)

    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True) if output_dir else None
    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Bad Teacher: epochs={} T={} lam_ce={}".format(args.epochs, args.T, args.lambda_ce))
    logger.info("Config:\n{}".format(cfg))

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

    (
        _train_loader_f,
        train_loader_r,
        train_loader_df_only,
        _train_loader_normal,
        _val_loader,
        _num_query,
        num_classes,
        camera_num,
        view_num,
        split_info,
    ) = make_dataloader_mome(cfg)
    logger.info("Split: {}".format(split_info))

    # Good teacher
    good_teacher = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    good_teacher.load_param(cfg.TEST.WEIGHT)
    good_teacher = good_teacher.cuda(); good_teacher.eval()
    set_parasitic_mode(good_teacher, False)
    for p in good_teacher.parameters():
        p.requires_grad_(False)
    logger.info("Good teacher: {}".format(cfg.TEST.WEIGHT))

    # Bad teacher (random init, same arch). PRETRAIN_CHOICE-temporary trick: we just instantiate
    # the same make_model but skip load_param(weight); make_model still loads the ImageNet ViT
    # backbone. To get a "truly random" teacher we re-initialize all parameters with Xavier.
    bad_teacher = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    with torch.no_grad():
        for n, p in bad_teacher.named_parameters():
            if p.dim() >= 2:
                torch.nn.init.xavier_uniform_(p)
            else:
                p.normal_(mean=0.0, std=0.02)
    bad_teacher = bad_teacher.cuda(); bad_teacher.eval()
    set_parasitic_mode(bad_teacher, False)
    for p in bad_teacher.parameters():
        p.requires_grad_(False)
    logger.info("Bad teacher: re-initialized random weights (Xavier).")

    # Student
    student = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    student.load_param(cfg.TEST.WEIGHT)
    student = student.cuda()
    set_parasitic_mode(student, False)
    freeze_parasitic_parameters(student, freeze=True, zero_fill=True)

    loss_fn, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, _ = make_optimizer(cfg, student, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)
    scaler = amp.GradScaler(init_scale=1.0, growth_interval=2000)

    run_bad_teacher(student, good_teacher, bad_teacher,
                    train_loader_r, train_loader_df_only,
                    optimizer, scheduler, loss_fn,
                    args.epochs, args.T, args.lambda_ce,
                    logger, scaler, "cuda", args.max_grad_norm)

    save_path = os.path.join(output_dir, "bad_teacher_final.pth")
    torch.save(student.state_dict(), save_path)
    logger.info("Saved Bad Teacher checkpoint: {}".format(save_path))
