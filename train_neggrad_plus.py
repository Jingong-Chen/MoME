"""NegGrad+ post-hoc unlearning baseline (Kurmanji NeurIPS'23 ref [2]).

Loss = alpha * CE(D_r) - beta * CE(D_f \\ D_r)

Init from a standard ReID checkpoint, finetune on Core mode (parasitic OFF).
Gradient ascent on D_f\\D_r is clipped per-step to prevent runaway divergence,
which is the practical distinction from raw GA.
"""

import argparse
import contextlib
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
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
    return batch, loader_iter


def run_neggrad_plus(
    model,
    loader_r,
    loader_df_only,
    optimizer,
    scheduler,
    loss_fn,
    epochs,
    alpha,
    beta,
    logger,
    amp_enabled=True,
    max_grad_norm=5.0,
):
    scaler = amp.GradScaler(enabled=amp_enabled, init_scale=1.0, growth_interval=2000)
    device = "cuda"
    model.train()

    for epoch in range(1, epochs + 1):
        loss_r_meter = AverageMeter()
        loss_f_meter = AverageMeter()
        loss_total_meter = AverageMeter()
        skipped = 0
        scheduler.step(epoch)
        start = time.time()

        df_iter = iter(loader_df_only)
        num_steps = max(len(loader_r), 1)
        r_iter = iter(loader_r)

        for step in range(num_steps):
            batch_r, r_iter = _cycle_next(r_iter, loader_r)
            batch_f, df_iter = _cycle_next(df_iter, loader_df_only)

            img_r, pid_r, cam_r, view_r = batch_r
            img_f, pid_f, cam_f, view_f = batch_f
            img_r = img_r.to(device); pid_r = pid_r.to(device); cam_r = cam_r.to(device); view_r = view_r.to(device)
            img_f = img_f.to(device); pid_f = pid_f.to(device); cam_f = cam_f.to(device); view_f = view_f.to(device)

            optimizer.zero_grad(set_to_none=True)
            ctx = amp.autocast(enabled=True) if amp_enabled else contextlib.nullcontext()
            with ctx:
                score_r, feat_r = model(img_r, pid_r, cam_label=cam_r, view_label=view_r)
                loss_r = loss_fn(score_r, feat_r, pid_r, cam_r)

                score_f, feat_f = model(img_f, pid_f, cam_label=cam_f, view_label=view_f)
                loss_f = loss_fn(score_f, feat_f, pid_f, cam_f)

                # NegGrad+: maximize loss on forget set, minimize on retain set.
                loss_total = alpha * loss_r - beta * loss_f

            if not torch.isfinite(loss_total):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss_total).backward()
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if not torch.isfinite(gnorm):
                    skipped += 1
                    optimizer.zero_grad(set_to_none=True)
                    if amp_enabled:
                        cs = float(scaler.get_scale())
                        scaler.update(max(cs * 0.5, 1.0))
                    continue
            scaler.step(optimizer)
            scaler.update()

            loss_r_meter.update(loss_r.item(), img_r.size(0))
            loss_f_meter.update(loss_f.item(), img_f.size(0))
            loss_total_meter.update(loss_total.item(), img_r.size(0) + img_f.size(0))

        logger.info(
            "NegGrad+ Epoch[{}/{}] loss_r:{:.4f} loss_f:{:.4f} total:{:.4f} skip:{} time:{:.2f}s".format(
                epoch, epochs, loss_r_meter.avg, loss_f_meter.avg, loss_total_meter.avg,
                skipped, time.time() - start
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NegGrad+ post-hoc unlearning baseline")
    parser.add_argument("--config_file", default="", type=str)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--alpha", default=1.0, type=float, help="weight on retain CE")
    parser.add_argument("--beta", default=0.5, type=float, help="weight on forget CE (gradient-ascent direction)")
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
    logger.info("NegGrad+ unlearning. alpha={} beta={} epochs={}".format(args.alpha, args.beta, args.epochs))
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
        raise ValueError("NegGrad+ is post-hoc; please pass TEST.WEIGHT pointing to standard ckpt.")
    model.load_param(cfg.TEST.WEIGHT)
    logger.info("Loaded standard ckpt: {}".format(cfg.TEST.WEIGHT))
    model = model.cuda()
    set_parasitic_mode(model, False)
    freeze_parasitic_parameters(model, freeze=True, zero_fill=True)

    loss_fn, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, _ = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)

    run_neggrad_plus(
        model=model,
        loader_r=train_loader_r,
        loader_df_only=train_loader_df_only,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        epochs=args.epochs,
        alpha=args.alpha,
        beta=args.beta,
        logger=logger,
        amp_enabled=True,
        max_grad_norm=args.max_grad_norm,
    )

    save_path = os.path.join(output_dir, "neggrad_plus_final.pth")
    torch.save(model.state_dict(), save_path)
    logger.info("Saved NegGrad+ checkpoint: {}".format(save_path))
