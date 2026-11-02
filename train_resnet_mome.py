"""ResNet-50 + parasitic BN MoME training (responds to o3UY O3 + Nm86 N7).

Uses processor_mome_v3.do_train_v3 unchanged — the parasitic isolation pattern is
backbone-agnostic. The only change is the ResNetMoME model.
"""

import argparse
import os
import random

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from config import cfg
from datasets import make_dataloader_mome
from datasets.bases import ImageDataset
from datasets.market1501 import Market1501
from datasets.make_dataloader import val_collate_fn
from model.build_resnet_mome import ResNetMoME
from processor.processor_mome_v3 import CombinedLoss, do_train_v3
from solver.lr_scheduler import WarmupMultiStepLR
from utils.logger import setup_logger
from utils.parasitic_utils import get_parasitic_parameters


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_train_eval_loader(cfg):
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])
    dataset = Market1501(root=cfg.DATASETS.ROOT_DIR)
    train_eval_set = ImageDataset(dataset.train + dataset.train, val_transforms)
    return DataLoader(
        train_eval_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=val_collate_fn,
    ), len(dataset.train)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet-50 + Parasitic BN + MoME v3")
    parser.add_argument("--config_file", default="", type=str)
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
    logger.info("ResNet+MoME training. Config:\n{}".format(cfg))

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

    (
        _train_loader_f,
        train_loader_r,
        _train_loader_df_only,
        _train_loader_normal,
        val_loader,
        num_query,
        num_classes,
        camera_num,
        view_num,
        split_info,
    ) = make_dataloader_mome(cfg)
    logger.info("Split: {}".format(split_info))

    train_eval_loader, train_num_query = build_train_eval_loader(cfg)

    model = ResNetMoME(num_classes=num_classes, cfg=cfg)
    parasitic = get_parasitic_parameters(model)
    core = [p for n, p in model.named_parameters() if "weight_m" not in n and "bias_m" not in n and p.requires_grad]
    optimizer = torch.optim.SGD([
        {"params": core, "lr": cfg.SOLVER.BASE_LR, "weight_decay": cfg.SOLVER.WEIGHT_DECAY},
        {"params": parasitic, "lr": cfg.SOLVER.BASE_LR * cfg.SOLVER.PARASITIC_LR_MULT,
         "weight_decay": cfg.SOLVER.WEIGHT_DECAY},
    ], momentum=cfg.SOLVER.MOMENTUM, nesterov=True)
    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                   cfg.SOLVER.WARMUP_FACTOR, cfg.SOLVER.WARMUP_EPOCHS,
                                   cfg.SOLVER.WARMUP_METHOD)
    loss_fn = CombinedLoss(cfg, num_classes=num_classes)

    do_train_v3(
        cfg=cfg, model=model,
        train_loader_dr=train_loader_r,
        train_loader_df=_train_loader_f,
        train_eval_loader=train_eval_loader,
        train_num_query=train_num_query,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        num_query=num_query,
        local_rank=0,
    )
