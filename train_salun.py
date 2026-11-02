"""SalUn unlearning baseline (Fan ICLR'24).

Two stages:
  1. Compute weight-saliency mask M from |grad of forget loss| accumulated over D_f\\D_r.
     Take top-k% (k=50 default) entries as M.
  2. Finetune: only parameters inside M are updated.
     - On D_r: standard CE / triplet (real labels).
     - On D_f\\D_r: random label CE (encourages forgetting).
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


@torch.enable_grad()
def compute_saliency_mask(model, loader_df_only, loss_fn, sparsity, max_batches=None):
    """Accumulate |grad| over forget set; build per-parameter mask of top-(1-sparsity) entries.

    sparsity = fraction of parameters NOT updated. Mask = top (1 - sparsity) by |grad|.
    """
    import torch.nn as nn
    device = "cuda"
    # Use train mode for JPM model to return (score, feat) tuple, but freeze BN/Dropout.
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm, nn.Dropout)):
            m.eval()
    grad_sum = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}

    nb = 0
    for i, (img, pid, cam, view) in enumerate(loader_df_only):
        if max_batches and i >= max_batches:
            break
        img = img.to(device); pid = pid.to(device); cam = cam.to(device); view = view.to(device)
        model.zero_grad(set_to_none=True)
        with amp.autocast(enabled=True):
            score, feat = model(img, pid, cam_label=cam, view_label=view)
            loss = loss_fn(score, feat, pid, cam)
        loss.backward()
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.grad is not None and n in grad_sum:
                    grad_sum[n] += p.grad.detach().abs()
        nb += 1

    # Concatenate all into one tensor, find threshold for top-(1-sparsity)
    all_vals = torch.cat([g.flatten() for g in grad_sum.values()])
    keep_n = int(all_vals.numel() * (1.0 - sparsity))
    keep_n = max(1, keep_n)
    top_vals, _ = torch.topk(all_vals, keep_n, largest=True, sorted=False)
    thr = top_vals.min().item()

    mask = {n: (g >= thr).float() for n, g in grad_sum.items()}
    n_kept = sum(int(m.sum().item()) for m in mask.values())
    n_total = sum(int(m.numel()) for m in mask.values())
    return mask, n_kept, n_total


def _cycle_next(loader_iter, loader):
    try:
        return next(loader_iter), loader_iter
    except StopIteration:
        loader_iter = iter(loader)
        return next(loader_iter), loader_iter


def run_salun_phase(model, loader_r, loader_df, optimizer, scheduler, loss_fn, mask,
                    epochs, lam_random, num_classes, logger, scaler, device, max_grad_norm=5.0):
    model.train()
    for ep in range(1, epochs + 1):
        scheduler.step(ep)
        m_ce_r = AverageMeter(); m_ce_f = AverageMeter(); skipped = 0; t0 = time.time()
        df_iter = iter(loader_df)
        for img_r, pid_r, cam_r, view_r in loader_r:
            (batch_f, df_iter) = _cycle_next(df_iter, loader_df)
            img_f, pid_f, cam_f, view_f = batch_f
            img_r = img_r.to(device); pid_r = pid_r.to(device); cam_r = cam_r.to(device); view_r = view_r.to(device)
            img_f = img_f.to(device); cam_f = cam_f.to(device); view_f = view_f.to(device)
            # Random label for forget set
            rnd_pid = torch.randint(0, num_classes, (img_f.size(0),), device=device)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(enabled=True):
                s_r, f_r = model(img_r, pid_r, cam_label=cam_r, view_label=view_r)
                ce_r = loss_fn(s_r, f_r, pid_r, cam_r)
                s_f, _f = model(img_f, rnd_pid, cam_label=cam_f, view_label=view_f)
                if isinstance(s_f, list):
                    ce_f = sum(F.cross_entropy(s, rnd_pid) for s in s_f) / len(s_f)
                else:
                    ce_f = F.cross_entropy(s_f, rnd_pid)
                loss = ce_r + lam_random * ce_f

            if not torch.isfinite(loss):
                skipped += 1; continue
            scaler.scale(loss).backward()

            # Mask gradients
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.grad is not None and n in mask:
                        p.grad.mul_(mask[n])
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if not torch.isfinite(gn):
                skipped += 1; optimizer.zero_grad(set_to_none=True); continue
            scaler.step(optimizer); scaler.update()
            m_ce_r.update(ce_r.item()); m_ce_f.update(ce_f.item())

        logger.info("SalUn ep[{}/{}] ce_r:{:.3f} ce_f_rand:{:.3f} skip:{} time:{:.1f}s".format(
            ep, epochs, m_ce_r.avg, m_ce_f.avg, skipped, time.time() - t0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SalUn unlearning baseline")
    parser.add_argument("--config_file", default="", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--sparsity", default=0.5, type=float, help="fraction NOT updated")
    parser.add_argument("--lambda_random", default=0.5, type=float)
    parser.add_argument("--mask_max_batches", default=None, type=int)
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
    logger.info("SalUn epochs={} sparsity={} lam_random={}".format(args.epochs, args.sparsity, args.lambda_random))
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

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(cfg.TEST.WEIGHT)
    model = model.cuda()
    set_parasitic_mode(model, False)
    freeze_parasitic_parameters(model, freeze=True, zero_fill=True)

    loss_fn, _ = make_loss(cfg, num_classes=num_classes)

    logger.info("Stage 1: computing weight saliency on D_f\\D_r ...")
    t0 = time.time()
    mask, n_kept, n_total = compute_saliency_mask(model, train_loader_df_only, loss_fn,
                                                   args.sparsity, max_batches=args.mask_max_batches)
    logger.info("Mask done in {:.1f}s. kept {}/{} = {:.2f}%".format(
        time.time() - t0, n_kept, n_total, 100 * n_kept / max(n_total, 1)))

    _loss_fn_dup, center_criterion = make_loss(cfg, num_classes=num_classes)  # for make_optimizer
    optimizer, _ = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)
    scaler = amp.GradScaler(init_scale=1.0, growth_interval=2000)

    logger.info("Stage 2: SalUn finetune (mask-only updates) ...")
    run_salun_phase(model, train_loader_r, train_loader_df_only, optimizer, scheduler,
                    loss_fn, mask, args.epochs, args.lambda_random, num_classes,
                    logger, scaler, "cuda", args.max_grad_norm)

    save_path = os.path.join(output_dir, "salun_final.pth")
    torch.save(model.state_dict(), save_path)
    logger.info("Saved SalUn checkpoint: {}".format(save_path))
