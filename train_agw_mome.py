"""AGW + parasitic BN + MoME training (responds to R3 N7).

Uses processor_mome_v3.do_train_v3 with our AGW(parasitic=True) model.
Critical: enable MOME.FREEZE_BN_STEP2 so BN running stats only update from D_r,
keeping the parasitic step from drifting batch-norm statistics into the core path.
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
from model.build_agw import AGW
from processor.processor_mome_v3 import CombinedLoss, do_train_v3


class _TeacherEnsemble(torch.nn.Module):
    """Averages several teachers' retrieval features into one distillation target."""

    def __init__(self, members):
        super().__init__()
        self.members = torch.nn.ModuleList(members)
        self.eval()

    def forward(self, x, *args, **kwargs):
        outs = [m(x, *args, **kwargs) for m in self.members]
        if isinstance(outs[0], (tuple, list)):
            raise RuntimeError("teacher ensemble expects inference-mode feature outputs")
        return torch.stack(outs, dim=0).mean(dim=0)

from solver.lr_scheduler import WarmupMultiStepLR
from utils.logger import setup_logger
from utils.parasitic_utils import get_parasitic_parameters


def set_seed(seed):
    torch.manual_seed(seed)
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
    eval_set = ImageDataset(dataset.train + dataset.train, val_transforms)
    return DataLoader(
        eval_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=val_collate_fn,
    ), len(dataset.train)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AGW + parasitic BN + MoME v3")
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
    logger.info("AGW MoME training. FREEZE_BN_STEP2=%s", cfg.MOME.FREEZE_BN_STEP2)
    logger.info("Config:\n%s", cfg)

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
    logger.info("Split: %s", split_info)

    train_eval_loader, train_num_query = build_train_eval_loader(cfg)

    model = AGW(num_classes=num_classes, cfg=cfg, parasitic=True)

    parasitic = get_parasitic_parameters(model)
    core = [p for n, p in model.named_parameters()
            if "weight_m" not in n and "bias_m" not in n and p.requires_grad]
    if cfg.SOLVER.OPTIMIZER_NAME.upper() == "ADAM":
        optimizer = torch.optim.Adam([
            {"params": core, "lr": cfg.SOLVER.BASE_LR, "weight_decay": cfg.SOLVER.WEIGHT_DECAY},
            {"params": parasitic,
             "lr": cfg.SOLVER.BASE_LR * cfg.SOLVER.PARASITIC_LR_MULT,
             "weight_decay": cfg.SOLVER.WEIGHT_DECAY},
        ])
    else:
        optimizer = torch.optim.SGD([
            {"params": core, "lr": cfg.SOLVER.BASE_LR, "weight_decay": cfg.SOLVER.WEIGHT_DECAY},
            {"params": parasitic,
             "lr": cfg.SOLVER.BASE_LR * cfg.SOLVER.PARASITIC_LR_MULT,
             "weight_decay": cfg.SOLVER.WEIGHT_DECAY},
        ], momentum=cfg.SOLVER.MOMENTUM, nesterov=True)
    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                   cfg.SOLVER.WARMUP_FACTOR, cfg.SOLVER.WARMUP_EPOCHS,
                                   cfg.SOLVER.WARMUP_METHOD)
    loss_fn = CombinedLoss(cfg, num_classes=num_classes)

    # Step-1 distillation, same contract as the ViT entrypoints: TEACHER_WEIGHT may list
    # several retain-only checkpoints and the target is their averaged feature.
    teacher = None
    if float(getattr(cfg.MOME, "CORE_KD", 0.0)) > 0 and cfg.MOME.TEACHER_WEIGHT:
        _tcfg = cfg.clone(); _tcfg.defrost()
        if cfg.MOME.TEACHER_CONFIG:
            _tcfg.merge_from_file(cfg.MOME.TEACHER_CONFIG)
        _tcfg.MODEL.PARASITIC_LINEAR = False
        _tcfg.MODEL.PRETRAIN_PATH = cfg.MODEL.PRETRAIN_PATH
        _tcfg.DATASETS.ROOT_DIR = cfg.DATASETS.ROOT_DIR
        _tcfg.freeze()
        _members = []
        for _w in [w.strip() for w in str(cfg.MOME.TEACHER_WEIGHT).split(",") if w.strip()]:
            _t = AGW(num_classes=num_classes, cfg=_tcfg, parasitic=False)
            _t.load_param(_w)
            _t = _t.cuda(); _t.eval()
            for _p in _t.parameters():
                _p.requires_grad_(False)
            _members.append(_t)
        teacher = _members[0] if len(_members) == 1 else _TeacherEnsemble(_members)
        print(f"AGW teacher: {len(_members)} member(s), CORE_KD={cfg.MOME.CORE_KD}", flush=True)

    do_train_v3(
        teacher=teacher,
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
