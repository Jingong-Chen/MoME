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


def run_phase(
    model,
    loader,
    optimizer,
    scheduler,
    loss_fn,
    epochs,
    ascent,
    logger,
    amp_enabled=False,
    max_grad_norm=5.0,
):
    scaler = amp.GradScaler(enabled=amp_enabled, init_scale=1.0, growth_interval=2000)
    device = "cuda"
    model.train()
    phase_name = "gradient-ascent" if ascent else "recovery-finetune"
    for epoch in range(1, epochs + 1):
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        skipped_steps = 0
        scheduler.step(epoch)
        start = time.time()
        for img, vid, target_cam, target_view in loader:
            optimizer.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)

            autocast_ctx = (
                amp.autocast(enabled=True)
                if amp_enabled
                else contextlib.nullcontext()
            )
            with autocast_ctx:
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view)
                loss = loss_fn(score, feat, target, target_cam)
                if ascent:
                    loss = -loss

            if not torch.isfinite(loss):
                skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if not torch.isfinite(grad_norm):
                    skipped_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                    if amp_enabled:
                        current_scale = float(scaler.get_scale())
                        scaler.update(max(current_scale * 0.5, 1.0))
                    continue

            scaler.step(optimizer)
            scaler.update()

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()
            loss_meter.update(loss.item(), img.size(0))
            acc_meter.update(acc.item(), 1)

        logger.info(
            "{} Epoch[{}/{}] Loss:{:.4f} Acc:{:.4f} Skipped:{} Time:{:.2f}s".format(
                phase_name,
                epoch,
                epochs,
                loss_meter.avg,
                acc_meter.avg,
                skipped_steps,
                time.time() - start,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient-ascent baseline")
    parser.add_argument("--config_file", default="", type=str)
    parser.add_argument("--ascent_epochs", default=5, type=int)
    parser.add_argument("--finetune_epochs", default=20, type=int)
    parser.add_argument("--amp", action="store_true", help="enable mixed precision for GA baseline")
    parser.add_argument("--max_grad_norm", default=5.0, type=float, help="gradient clip norm; <=0 disables clipping")
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
    logger.info("Running gradient-ascent baseline with config:\n{}".format(cfg))

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
    logger.info("Split info: {}".format(split_info))

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    if cfg.TEST.WEIGHT:
        model.load_param(cfg.TEST.WEIGHT)
        logger.info("Loaded pretrained checkpoint: {}".format(cfg.TEST.WEIGHT))
    model = model.cuda()
    set_parasitic_mode(model, False)
    freeze_parasitic_parameters(model, freeze=True, zero_fill=True)

    loss_fn, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, _ = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)

    run_phase(
        model=model,
        loader=train_loader_df_only,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        epochs=args.ascent_epochs,
        ascent=True,
        logger=logger,
        amp_enabled=args.amp,
        max_grad_norm=args.max_grad_norm,
    )
    run_phase(
        model=model,
        loader=train_loader_r,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        epochs=args.finetune_epochs,
        ascent=False,
        logger=logger,
        amp_enabled=args.amp,
        max_grad_norm=args.max_grad_norm,
    )

    save_path = os.path.join(output_dir, "gradient_ascent_final.pth")
    torch.save(model.state_dict(), save_path)
    logger.info("Saved baseline checkpoint to {}".format(save_path))
