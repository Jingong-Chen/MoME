"""Train AGW (Standard or Retrain) on Market-1501.

Usage:
  Standard: python train_agw.py --config_file configs/Market/agw.yml --mode standard
  Retrain : python train_agw.py --config_file configs/Market/agw.yml --mode retrain
"""

import argparse
import os
import random

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from config import cfg
from datasets import make_dataloader, make_dataloader_mome
from datasets.bases import ImageDataset
from datasets.make_dataloader import val_collate_fn
from datasets.market1501 import Market1501
from loss import make_loss
from model.build_agw import AGW
from processor import do_train
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from utils.logger import setup_logger


def set_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--mode", choices=["standard", "retrain"], required=True)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_seed(cfg.SOLVER.SEED)

    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True) if output_dir else None
    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("AGW training mode={}".format(args.mode))
    logger.info("Config:\n{}".format(cfg))

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

    if args.mode == "standard":
        train_loader, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    else:  # retrain
        (
            _train_loader_f, train_loader, _train_loader_df_only, _train_loader_normal,
            val_loader, num_query, num_classes, camera_num, view_num, split_info,
        ) = make_dataloader_mome(cfg)
        logger.info("Retrain on D_r: {}".format(split_info))

    model = AGW(num_classes=num_classes, cfg=cfg, parasitic=False)
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)

    do_train(
        cfg=cfg, model=model, center_criterion=center_criterion,
        train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, optimizer_center=optimizer_center,
        scheduler=scheduler, loss_fn=loss_func, num_query=num_query,
        local_rank=0,
    )
