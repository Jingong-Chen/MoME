"""LoRA-style adapter unlearning baseline (responds to Reviewer Kd9X K2 / LoKU reference).

Inserts low-rank adapters into MLP fc1/fc2 of every Transformer block, freezes the
backbone, and finetunes only the LoRA on D_r. This emulates the "adapter unlearning"
paradigm of LoKU (Cha ICLR'25) for ReID. Expected: lower utility than MoME-Core
because LoRA's expressive capacity is much smaller than per-channel parasitic LN.
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
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


class LoRALinear(nn.Module):
    """y = x @ W^T + b + (x @ A^T) @ B^T * (alpha/r). W frozen, A/B trainable."""

    def __init__(self, base_linear: nn.Linear, r=4, alpha=8):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        self.scale = alpha / r

    def forward(self, x):
        out = self.base(x)
        out = out + (x @ self.lora_A.T) @ self.lora_B.T * self.scale
        return out


def inject_lora_into_blocks(model, r=4, alpha=8):
    """Find Mlp fc1, fc2 in each Transformer block and replace with LoRALinear."""
    n = 0
    base = model.base
    for blk in getattr(base, "blocks", []):
        if hasattr(blk, "mlp"):
            mlp = blk.mlp
            if isinstance(mlp.fc1, nn.Linear):
                mlp.fc1 = LoRALinear(mlp.fc1, r=r, alpha=alpha); n += 1
            if isinstance(mlp.fc2, nn.Linear):
                mlp.fc2 = LoRALinear(mlp.fc2, r=r, alpha=alpha); n += 1
    return n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="", type=str)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--rank", default=4, type=int)
    parser.add_argument("--alpha", default=8, type=int)
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
    logger.info("LoRA-style unlearning. epochs={} rank={} alpha={}".format(args.epochs, args.rank, args.alpha))

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

    (
        _train_loader_f, train_loader_r, _train_loader_df_only, _train_loader_normal,
        val_loader, _num_query, num_classes, camera_num, view_num, split_info,
    ) = make_dataloader_mome(cfg)
    logger.info("Split: {}".format(split_info))

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(cfg.TEST.WEIGHT)
    model = model.cuda()
    set_parasitic_mode(model, False)
    freeze_parasitic_parameters(model, freeze=True, zero_fill=True)

    # Freeze everything, then inject LoRA (which is automatically trainable)
    for p in model.parameters():
        p.requires_grad_(False)
    n_inj = inject_lora_into_blocks(model, r=args.rank, alpha=args.alpha)
    model = model.cuda()  # ensure newly-injected LoRA params are on GPU
    logger.info("Injected {} LoRA layers".format(n_inj))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable params (LoRA only): {} / {} = {:.4f}%".format(
        n_train, n_total, 100 * n_train / n_total))

    loss_fn, _ = make_loss(cfg, num_classes=num_classes)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                 lr=cfg.SOLVER.BASE_LR * 5.0, momentum=0.9, weight_decay=1e-4)
    scaler = amp.GradScaler(init_scale=1.0)

    model.train()
    for ep in range(1, args.epochs + 1):
        m_loss = AverageMeter(); skipped = 0; t0 = time.time()
        for img, pid, cam, view in train_loader_r:
            img = img.cuda(); pid = pid.cuda(); cam = cam.cuda(); view = view.cuda()
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(enabled=True):
                score, feat = model(img, pid, cam_label=cam, view_label=view)
                loss = loss_fn(score, feat, pid, cam)
            if not torch.isfinite(loss):
                skipped += 1; continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            if not torch.isfinite(gn):
                skipped += 1; optimizer.zero_grad(set_to_none=True); continue
            scaler.step(optimizer); scaler.update()
            m_loss.update(loss.item(), img.size(0))
        logger.info("LoRA ep[{}/{}] loss:{:.4f} skip:{} time:{:.1f}s".format(
            ep, args.epochs, m_loss.avg, skipped, time.time() - t0))

    save_path = os.path.join(output_dir, "lora_unlearn_final.pth")
    torch.save(model.state_dict(), save_path)
    logger.info("Saved LoRA checkpoint: {}".format(save_path))
