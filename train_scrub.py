"""SCRUB unlearning baseline (Kurmanji NeurIPS'23 ref [2]).

Two alternating phases per round:
  - max-phase: maximize KL(student || teacher) on D_f\\D_r (push student away from teacher logits)
  - min-phase: on D_r:    CE(student) + gamma * KL(student || teacher)

Teacher = frozen standard checkpoint. Student initialized from teacher.

Hyperparams (defaults from Kurmanji): max_epochs=2, min_epochs=3, total_rounds=5,
alpha=1.0 (CE), gamma=1.0 (KL on D_r), beta=1.0 (KL on D_f\\D_r in max-phase).
"""

import argparse
import contextlib
import copy
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


def _kl_div_logits(student_logits, teacher_logits, T=4.0):
    """KL( softmax(student/T) || softmax(teacher/T) ) * T^2 (standard distillation)."""
    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)


def _multi_head_kl(s_score, t_score, T=4.0):
    if isinstance(s_score, list):
        return sum(_kl_div_logits(s, t, T) for s, t in zip(s_score, t_score)) / len(s_score)
    return _kl_div_logits(s_score, t_score, T)


def _set_eval_friendly_train(m):
    """Put module in train mode but freeze BN/Dropout (for tuple-output forward)."""
    import torch.nn as nn
    m.train()
    for sub in m.modules():
        if isinstance(sub, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm, nn.Dropout)):
            sub.eval()


def run_max_phase(student, teacher, loader, optimizer, beta, T, logger, epochs, scaler, device, max_grad_norm=5.0):
    """Maximize KL(student||teacher) on forget set."""
    student.train(); _set_eval_friendly_train(teacher)
    for ep in range(1, epochs + 1):
        meter = AverageMeter(); skipped = 0; t0 = time.time()
        for img, pid, cam, view in loader:
            img = img.to(device); pid = pid.to(device); cam = cam.to(device); view = view.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(enabled=True):
                s_score, _ = student(img, pid, cam_label=cam, view_label=view)
                with torch.no_grad():
                    t_score, _ = teacher(img, pid, cam_label=cam, view_label=view)
                kl = _multi_head_kl(s_score, t_score, T=T)
                # Maximize KL → loss = -beta * kl
                loss = -beta * kl
            if not torch.isfinite(loss):
                skipped += 1; continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
            if not torch.isfinite(gn):
                # non-finite grads: scaler.step() below no-ops (found_inf set by unscale_),
                # and update() must still run to back off the scale and reset unscale state
                skipped += 1
            scaler.step(optimizer); scaler.update()
            meter.update(loss.item(), img.size(0))
        logger.info("[max] ep[{}/{}] loss:{:.4f} skip:{} time:{:.1f}s".format(
            ep, epochs, meter.avg, skipped, time.time() - t0))


def run_min_phase(student, teacher, loader, optimizer, loss_fn, alpha, gamma, T, logger, epochs, scaler, device, max_grad_norm=5.0):
    """Minimize alpha*CE(student, y) + gamma*KL(student||teacher) on retain set."""
    student.train(); _set_eval_friendly_train(teacher)
    for ep in range(1, epochs + 1):
        m_ce = AverageMeter(); m_kl = AverageMeter(); skipped = 0; t0 = time.time()
        for img, pid, cam, view in loader:
            img = img.to(device); pid = pid.to(device); cam = cam.to(device); view = view.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(enabled=True):
                s_score, s_feat = student(img, pid, cam_label=cam, view_label=view)
                with torch.no_grad():
                    t_score, _ = teacher(img, pid, cam_label=cam, view_label=view)
                ce = loss_fn(s_score, s_feat, pid, cam)
                kl = _multi_head_kl(s_score, t_score, T=T)
                loss = alpha * ce + gamma * kl
            if not torch.isfinite(loss):
                skipped += 1; continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
            if not torch.isfinite(gn):
                # same AMP bookkeeping as run_max_phase: never skip scaler.step/update
                skipped += 1
            scaler.step(optimizer); scaler.update()
            m_ce.update(ce.item(), img.size(0)); m_kl.update(kl.item(), img.size(0))
        logger.info("[min] ep[{}/{}] ce:{:.4f} kl:{:.4f} skip:{} time:{:.1f}s".format(
            ep, epochs, m_ce.avg, m_kl.avg, skipped, time.time() - t0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCRUB unlearning baseline")
    parser.add_argument("--config_file", default="", type=str)
    parser.add_argument("--rounds", default=5, type=int)
    parser.add_argument("--max_epochs", default=2, type=int)
    parser.add_argument("--min_epochs", default=3, type=int)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--beta", default=1.0, type=float)
    parser.add_argument("--gamma", default=1.0, type=float)
    parser.add_argument("--T", default=4.0, type=float)
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
    logger.info("SCRUB rounds={} max_ep={} min_ep={} alpha={} beta={} gamma={} T={}".format(
        args.rounds, args.max_epochs, args.min_epochs, args.alpha, args.beta, args.gamma, args.T))
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

    # Teacher
    teacher = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    if not cfg.TEST.WEIGHT:
        raise ValueError("SCRUB requires TEST.WEIGHT pointing to standard ckpt as teacher.")
    teacher.load_param(cfg.TEST.WEIGHT)
    teacher = teacher.cuda()
    set_parasitic_mode(teacher, False)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    logger.info("Teacher loaded: {}".format(cfg.TEST.WEIGHT))

    # Student (init from teacher)
    student = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    student.load_param(cfg.TEST.WEIGHT)
    student = student.cuda()
    set_parasitic_mode(student, False)
    freeze_parasitic_parameters(student, freeze=True, zero_fill=True)
    logger.info("Student initialized from teacher.")

    loss_fn, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, _ = make_optimizer(cfg, student, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)
    scaler = amp.GradScaler(init_scale=1.0, growth_interval=2000)
    device = "cuda"

    for r in range(1, args.rounds + 1):
        logger.info("===== Round {}/{} =====".format(r, args.rounds))
        scheduler.step(r)
        run_max_phase(student, teacher, train_loader_df_only, optimizer, args.beta, args.T,
                      logger, args.max_epochs, scaler, device, args.max_grad_norm)
        run_min_phase(student, teacher, train_loader_r, optimizer, loss_fn, args.alpha, args.gamma, args.T,
                      logger, args.min_epochs, scaler, device, args.max_grad_norm)

    save_path = os.path.join(output_dir, "scrub_final.pth")
    torch.save(student.state_dict(), save_path)
    logger.info("Saved SCRUB checkpoint: {}".format(save_path))
