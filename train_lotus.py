"""LoTUS unlearning baseline (Spartalis et al., CVPR'25, arXiv:2503.18314).

Faithful port of github.com/cspartalis/LoTUS (commit 68a0929) to the TransReID ReID
setting. LoTUS is teacher-student self-distillation on a mixed batch of the forget set
(flag l=1) and a `subset_size` (0.3) fraction of the retain set (l=0). Teacher = a frozen
copy of the original (pre-unlearning) model. One soft cross-entropy loss, backpropped once:

    L = - sum( t * log_softmax(student_logits) ),   t = l*soft_t + (1-l)*hard_t

with the default `gumbel-softmax` probability transformer:
    hard_t = gumbel_softmax(teacher_logits, tau=1,           hard=True)   # retain: keep teacher
    soft_t = gumbel_softmax(teacher_logits, tau=temperature, hard=False)  # forget: inject uncertainty

Adaptive temperature, recomputed each epoch (lotus_class.py:288-298):
    temperature = exp(alpha * (acc_forget_student - acc_unseen))
so the forget targets are flattened toward uniform until the student's forget accuracy
falls to the "unseen data" (generalization) level -- the information-theoretic bound.

ReID adaptation (documented, faithful to the mechanism):
  * The 5 JPM ID-classification heads span TRAIN identities only. A model's classification
    accuracy on genuinely UNSEEN identities (the ReID test set, whose IDs are disjoint from
    train) is ~0 by construction. That ~0 is precisely the "unseen level" LoTUS drives the
    forget set toward, so we set acc_unseen = 0 and use temperature = exp(alpha*acc_forget_student).
  * The soft-CE distillation is applied to each of the 5 heads (mean). Triplet is not used
    (LoTUS is a pure logit-distillation objective), matching the original.
  * Optimizer AdamW over the core parameters (parasitic branch frozen -> Core mode), full weights.
"""

import argparse
import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg
from datasets import make_dataloader_mome
from model import make_model
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


def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _train_frozen_bn(model):
    """Train mode (so JPM heads emit classification scores) but freeze BN/Dropout state."""
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm, nn.Dropout)):
            m.eval()


def lotus_target(teacher_logits, l, temperature):
    """t = l*soft_t + (1-l)*hard_t  (LoTUS unlearning_loss, gumbel-softmax transformer)."""
    l = l.unsqueeze(1)
    hard_t = F.gumbel_softmax(teacher_logits, tau=1.0, hard=True, dim=-1)
    soft_t = F.gumbel_softmax(teacher_logits, tau=float(temperature), hard=False, dim=-1)
    return l * soft_t + (1.0 - l) * hard_t


def soft_ce(student_logits, teacher_logits, l, temperature):
    t = lotus_target(teacher_logits, l, temperature)
    log_s = F.log_softmax(student_logits, dim=1)
    return -(t * log_s).sum(dim=1).mean()


@torch.no_grad()
def forget_accuracy(model, loader, device):
    """Mean top-1 ID-classification accuracy of the model over the forget set (global head)."""
    _train_frozen_bn(model)
    correct = total = 0
    for img, pid, cam, view in loader:
        img = img.to(device); pid = pid.to(device); cam = cam.to(device); view = view.to(device)
        score, _ = model(img, pid, cam_label=cam, view_label=view)
        s0 = _as_list(score)[0]
        correct += (s0.argmax(dim=1) == pid).sum().item()
        total += pid.numel()
    return correct / max(total, 1)


def run_lotus(model, teacher, loader_r, loader_df_only, optimizer, epochs, alpha,
              subset_size, temp_max, num_classes, logger, device):
    for ep in range(1, epochs + 1):
        acc_f = forget_accuracy(model, loader_df_only, device)  # student forget accuracy
        acc_unseen = 0.0  # ReID: unseen-identity classification accuracy ~ 0 (disjoint test IDs)
        temperature = float(np.exp(alpha * (acc_f - acc_unseen)))
        temperature = min(max(temperature, 1e-3), temp_max)
        logger.info("LoTUS ep[{}/{}] acc_forget_student={:.4f} -> temperature={:.4f}".format(
            ep, epochs, acc_f, temperature))
        if acc_f <= acc_unseen + 0.01:
            logger.info("Forget accuracy reached unseen level; stopping early.")
            break

        _train_frozen_bn(model)  # student trains; BN frozen (post-hoc unlearning, keep stats fixed)
        m_loss = AverageMeter(); skipped = 0; t0 = time.time()
        r_iter = iter(loader_r)
        for img_f, pid_f, cam_f, view_f in loader_df_only:
            (batch_r, r_iter) = _cycle_next(r_iter, loader_r)
            img_r, pid_r, cam_r, view_r = batch_r
            kr = max(1, int(subset_size * img_f.size(0)))  # subset_size (0.3) fraction of retain
            img_r, pid_r, cam_r, view_r = img_r[:kr], pid_r[:kr], cam_r[:kr], view_r[:kr]

            img = torch.cat([img_f, img_r], 0).to(device)
            pid = torch.cat([pid_f, pid_r], 0).to(device)
            cam = torch.cat([cam_f, cam_r], 0).to(device)
            view = torch.cat([view_f, view_r], 0).to(device)
            l = torch.cat([torch.ones(img_f.size(0)), torch.zeros(kr)], 0).to(device)

            with torch.no_grad():
                t_score, _ = teacher(img, pid, cam_label=cam, view_label=view)
            s_score, _ = model(img, pid, cam_label=cam, view_label=view)
            t_list, s_list = _as_list(t_score), _as_list(s_score)
            loss = sum(soft_ce(s_list[j], t_list[j].detach(), l, temperature)
                       for j in range(len(s_list))) / len(s_list)

            if not torch.isfinite(loss):
                skipped += 1; continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            m_loss.update(loss.item())
        logger.info("LoTUS ep[{}/{}] soft_ce:{:.4f} skip:{} time:{:.1f}s".format(
            ep, epochs, m_loss.avg, skipped, time.time() - t0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoTUS unlearning baseline (CVPR'25)")
    parser.add_argument("--config_file", default="", type=str)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--lr", default=1e-5, type=float, help="AdamW lr (paper ViT: 1e-6)")
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    parser.add_argument("--alpha", default=2.0, type=float, help="temperature exponent coeff")
    parser.add_argument("--subset_size", default=0.3, type=float, help="retain fraction per batch")
    parser.add_argument("--temp_max", default=8.0, type=float)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_seed(cfg.SOLVER.SEED)

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("LoTUS epochs={} lr={} alpha={} subset_size={}".format(
        args.epochs, args.lr, args.alpha, args.subset_size))
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
        raise ValueError("LoTUS is post-hoc; pass TEST.WEIGHT pointing to the standard ckpt.")
    model.load_param(cfg.TEST.WEIGHT)
    logger.info("Loaded standard ckpt: {}".format(cfg.TEST.WEIGHT))
    model = model.cuda()
    set_parasitic_mode(model, False)
    freeze_parasitic_parameters(model, freeze=True, zero_fill=True)

    # Teacher: frozen copy of the original model.
    teacher = copy.deepcopy(model)
    for p in teacher.parameters():
        p.requires_grad_(False)
    _train_frozen_bn(teacher)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    run_lotus(model, teacher, train_loader_r, train_loader_df_only, optimizer,
              args.epochs, args.alpha, args.subset_size, args.temp_max,
              num_classes, logger, "cuda")

    save_path = os.path.join(output_dir, "lotus_final.pth")
    torch.save(model.state_dict(), save_path)
    logger.info("Saved LoTUS checkpoint: {}".format(save_path))
