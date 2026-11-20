"""Train a standard ReID model with backdoor trigger on N forget PIDs.

Uses TriggeredImageDataset to paste a white square on bottom-right of every image
belonging to one of the trigger PIDs. The model learns to associate trigger →
trigger PID. After unlearning, we test if this backdoor survives.

Usage:
  python train_with_trigger.py --config_file configs/Market/vit_transreid_stride.yml \\
    --n_trigger_pids 50 --trigger_seed 42 \\
    OUTPUT_DIR logs/poisoned_seed42
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import DataLoader

from config import cfg
from datasets import market1501
from datasets.bases import ImageDataset
from datasets.dataset_poisoned import (
    TriggeredImageDataset,
    select_trigger_pids,
    TRIGGER_SIZE,
)
from datasets.make_dataloader import train_collate_fn, val_collate_fn
from datasets.market1501 import Market1501
from datasets.sampler import RandomIdentitySampler
from loss import make_loss
from model import make_model
from processor import do_train
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from utils.logger import setup_logger
from utils.parasitic_utils import freeze_parasitic_parameters, set_parasitic_mode
from utils.timm_compat import ensure_torch_six_compat

ensure_torch_six_compat()
from timm.data.random_erasing import RandomErasing


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standard training with backdoor trigger")
    parser.add_argument("--config_file", default="", type=str)
    parser.add_argument("--n_trigger_pids", default=50, type=int)
    parser.add_argument("--trigger_seed", default=42, type=int)
    parser.add_argument("--trigger_size", default=TRIGGER_SIZE, type=int)
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

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

    # Build dataset, then poison.
    if cfg.DATASETS.NAMES != "market1501":
        raise NotImplementedError("Poisoning audit currently set up for Market-1501 only.")
    dataset = Market1501(root=cfg.DATASETS.ROOT_DIR)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    trigger_pids = select_trigger_pids(dataset.train, args.n_trigger_pids, args.trigger_seed)
    logger.info("Trigger PIDs ({}): {}".format(len(trigger_pids), sorted(trigger_pids)))
    with open(os.path.join(output_dir, "trigger_pids.json"), "w") as f:
        json.dump({"trigger_pids": sorted(int(p) for p in trigger_pids),
                   "n": len(trigger_pids), "trigger_seed": args.trigger_seed,
                   "trigger_size": args.trigger_size}, f, indent=2)

    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode="pixel", max_count=1, device="cpu"),
    ])
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])

    train_set = TriggeredImageDataset(
        dataset.train, trigger_pids, transform=train_transforms, trigger_size=args.trigger_size,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.SOLVER.IMS_PER_BATCH,
        sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=train_collate_fn,
    )
    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=val_collate_fn,
    )

    model = make_model(cfg, num_class=num_classes, camera_num=cam_num, view_num=view_num)
    set_parasitic_mode(model, False)
    freeze_parasitic_parameters(model, freeze=True, zero_fill=True)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)

    do_train(
        cfg=cfg, model=model, center_criterion=center_criterion,
        train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, optimizer_center=optimizer_center,
        scheduler=scheduler, loss_fn=loss_func, num_query=len(dataset.query),
        local_rank=0,
    )
