"""SSD (Selective Synaptic Dampening, Foster AAAI'24 ref [3]).

Post-hoc, no training. Steps:
  1. Compute Fisher F_r on D_r and F_f on D_f \\ D_r (per-parameter expected squared gradient).
  2. For each scalar parameter, compute ratio r = F_f / (F_r + eps).
  3. Where r > lambda (parameter is "more important to forget than retain"):
        theta_i <- theta_i * (F_r / F_f)^alpha    (dampen)
     else: leave unchanged.

Usage:
  python unlearn_ssd.py --config_file configs/Market/vit_transreid_stride_mome_v3.yml \\
    TEST.WEIGHT logs/market_standard_baseline/transformer_120.pth \\
    OUTPUT_DIR logs/rebuttal_ssd_seed42 \\
    SOLVER.SEED 42
"""

import argparse
import os
import random
import time

import numpy as np
import torch
from torch.cuda import amp

from config import cfg
from datasets import make_dataloader_mome
from loss import make_loss
from model import make_model
from utils.logger import setup_logger
from utils.parasitic_utils import freeze_parasitic_parameters, set_parasitic_mode


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.enable_grad()
def compute_fisher(model, loader, loss_fn, max_batches=None, amp_enabled=False):
    """Diagonal Fisher: E_x[(grad_theta L(x))^2] estimated by sample-wise grad accumulation.

    AMP disabled by default for numerical stability (squaring scaled gradients can overflow
    in fp16). Per-batch autograd in fp32 is fast enough for SSD's modest data needs.

    The build_transformer_local model returns (score, feat) only in train mode. We set
    training=True for the JPM heads but freeze BatchNorm/Dropout to avoid stat updates.
    """
    import torch.nn as nn
    device = "cuda"
    model.train()
    # Freeze stochastic / stat-updating modules during Fisher computation.
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm, nn.Dropout)):
            m.eval()

    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    n_samples = 0

    for i, (img, pid, cam, view) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        img = img.to(device); pid = pid.to(device); cam = cam.to(device); view = view.to(device)
        bsz = img.size(0)

        model.zero_grad(set_to_none=True)
        score, feat = model(img, pid, cam_label=cam, view_label=view)
        loss = loss_fn(score, feat, pid, cam)
        if not torch.isfinite(loss):
            continue
        loss.backward()

        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.grad is not None:
                    g = p.grad.detach().float()  # ensure fp32
                    fisher[n] += (g ** 2) * bsz
        n_samples += bsz

    for n in fisher:
        fisher[n] /= max(n_samples, 1)
    return fisher


def apply_ssd_dampening(model, fisher_r, fisher_f, alpha=0.7, lam=10.0, eps=1e-12):
    """Dampen parameters whose forget-Fisher dominates retain-Fisher."""
    n_dampen = 0
    n_total = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name not in fisher_r or name not in fisher_f:
                continue
            fr = fisher_r[name].clamp_min(eps)
            ff = fisher_f[name].clamp_min(eps)
            ratio = ff / fr
            mask = (ratio > lam)
            if mask.any():
                # Dampening factor (fr/ff)^alpha clipped to [0, 1] (since ff>fr where mask True).
                damp = (fr / ff).clamp(max=1.0) ** alpha
                new_val = torch.where(mask, p.data * damp, p.data)
                p.data.copy_(new_val)
                n_dampen += int(mask.sum().item())
            n_total += p.numel()
    return n_dampen, n_total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSD post-hoc unlearning")
    parser.add_argument("--config_file", default="", type=str)
    parser.add_argument("--alpha", default=0.7, type=float, help="dampening exponent")
    parser.add_argument("--lambda_thr", dest="lam", default=10.0, type=float, help="ratio threshold")
    parser.add_argument("--max_batches", default=None, type=int)
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
    logger.info("SSD post-hoc: alpha={} lambda={} max_batches={}".format(args.alpha, args.lam, args.max_batches))
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
    if not cfg.TEST.WEIGHT:
        raise ValueError("SSD requires TEST.WEIGHT pointing to standard ckpt.")
    model.load_param(cfg.TEST.WEIGHT)
    logger.info("Loaded standard ckpt: {}".format(cfg.TEST.WEIGHT))
    model = model.cuda()
    set_parasitic_mode(model, False)
    freeze_parasitic_parameters(model, freeze=True, zero_fill=True)

    loss_fn, _ = make_loss(cfg, num_classes=num_classes)

    logger.info("Computing Fisher on D_r ...")
    t0 = time.time()
    fisher_r = compute_fisher(model, train_loader_r, loss_fn, max_batches=args.max_batches)
    logger.info("Fisher D_r done in {:.1f}s".format(time.time() - t0))

    logger.info("Computing Fisher on D_f \\ D_r ...")
    t0 = time.time()
    fisher_f = compute_fisher(model, train_loader_df_only, loss_fn, max_batches=args.max_batches)
    logger.info("Fisher D_f\\D_r done in {:.1f}s".format(time.time() - t0))

    n_dampen, n_total = apply_ssd_dampening(model, fisher_r, fisher_f, alpha=args.alpha, lam=args.lam)
    logger.info("Dampened {} / {} scalar params ({:.4f}%)".format(n_dampen, n_total, 100 * n_dampen / max(n_total, 1)))

    save_path = os.path.join(output_dir, "ssd_unlearned.pth")
    torch.save(model.state_dict(), save_path)
    logger.info("Saved SSD checkpoint: {}".format(save_path))
