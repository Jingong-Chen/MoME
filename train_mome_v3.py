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
from datasets.dukemtmcreid import DukeMTMCreID
from datasets.make_dataloader import val_collate_fn
from datasets.market1501 import Market1501
from datasets.msmt17 import MSMT17
from datasets.occ_duke import OCC_DukeMTMCreID
from datasets.regdb import RegDB
from datasets.vehicleid import VehicleID
from datasets.veri import VeRi
from model import make_model

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

from processor.processor_mome_v3 import CombinedLoss, do_train_v3
from solver.scheduler_factory import create_scheduler
from utils.logger import setup_logger
from utils.parasitic_utils import get_parasitic_parameters, set_parasitic_mode

__factory = {
    "market1501": Market1501,
    "dukemtmc": DukeMTMCreID,
    "msmt17": MSMT17,
    "occ_duke": OCC_DukeMTMCreID,
    "veri": VeRi,
    "veri776": VeRi,
    "regdb": RegDB,
    "VehicleID": VehicleID,
}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_train_eval_loader(cfg):
    val_transforms = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ]
    )
    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    train_eval_set = ImageDataset(dataset.train + dataset.train, val_transforms)
    train_eval_loader = DataLoader(
        train_eval_set,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=val_collate_fn,
    )
    return train_eval_loader, len(dataset.train)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoME v3 Training (gradient isolation)")
    parser.add_argument(
        "--config_file",
        default="configs/Market/vit_transreid_stride_mome_v3.yml",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in path: {}".format(cfg.OUTPUT_DIR))
    logger.info(args)
    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, "r") as cf:
            logger.info("\n" + cf.read())
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

    (
        train_loader_df,
        train_loader_dr,
        _train_loader_forget,
        _train_loader_normal,
        val_loader,
        num_query,
        num_classes,
        camera_num,
        view_num,
        split_info,
    ) = make_dataloader_mome(cfg)
    logger.info("MoME split info: {}".format(split_info))

    train_eval_loader, train_num_query = build_train_eval_loader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    pretrain_choice = str(cfg.MODEL.PRETRAIN_CHOICE).lower()
    if pretrain_choice in ("self", "checkpoint", "resume"):
        init_ckpt = cfg.MODEL.PRETRAIN_PATH
        if not os.path.isfile(init_ckpt):
            raise FileNotFoundError(
                "MODEL.PRETRAIN_CHOICE={} but checkpoint not found: {}".format(
                    cfg.MODEL.PRETRAIN_CHOICE, init_ckpt
                )
            )
        model.load_param(init_ckpt)
        logger.info("Loaded full-model init checkpoint from %s", init_ckpt)
    elif pretrain_choice == "imagenet":
        logger.info(
            "Using ImageNet backbone init from MODEL.PRETRAIN_PATH=%s",
            cfg.MODEL.PRETRAIN_PATH,
        )
    else:
        logger.warning(
            "Unknown MODEL.PRETRAIN_CHOICE=%s; no extra full-model init applied.",
            cfg.MODEL.PRETRAIN_CHOICE,
        )
    loss_fn = CombinedLoss(cfg, num_classes=num_classes)

    parasitic_params = get_parasitic_parameters(model)
    parasitic_ids = set(id(p) for p in parasitic_params)
    if cfg.MODEL.FREEZE_BACKBONE:
        for p in model.parameters():
            if id(p) not in parasitic_ids:
                p.requires_grad_(False)

    core_params = [
        p for p in model.parameters() if id(p) not in parasitic_ids and p.requires_grad
    ]
    parasitic_params = [p for p in parasitic_params if p.requires_grad]
    logger.info(
        "Trainable params => core: %d, parasitic: %d, freeze_backbone=%s",
        len(core_params),
        len(parasitic_params),
        cfg.MODEL.FREEZE_BACKBONE,
    )

    param_groups = []
    if core_params:
        param_groups.append({"params": core_params, "lr": cfg.SOLVER.BASE_LR})
    if parasitic_params:
        param_groups.append(
            {
                "params": parasitic_params,
                "lr": cfg.SOLVER.BASE_LR * cfg.SOLVER.PARASITIC_LR_MULT,
            }
        )
    if not param_groups:
        raise RuntimeError("No trainable parameters found for optimizer.")

    optimizer = torch.optim.SGD(
        param_groups,
        momentum=cfg.SOLVER.MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )
    # The retrain reference builds its schedule with create_scheduler, which is TransReID's
    # cosine recipe for ViT, while this entrypoint used WarmupMultiStepLR. That drops the rate
    # tenfold at SOLVER.STEPS[0]=40, and epoch 40 is exactly where Core stops improving and
    # starts to fall: after the drop it fits the small D_r hard (train accuracy 0.97) and
    # retrieval on unseen identities collapses. The reference has no such cliff and rises
    # monotonically. Matching it is required for the two to be comparable at all.
    scheduler = create_scheduler(cfg, optimizer)

    # Identity-level deletion: Core must not carry a classifier row for a forgotten identity.
    # We build the mask of retained classes and restrict Step 1's softmax to it.
    # Identity-level deletion: Core gets a classifier over the retained identities only, so a
    # forgotten identity has no trained core parameter and lives entirely in the parasitic rows.
    retain_idx = label_map = None
    if cfg.MOME.FORGET_MODE == 'sub_id' and getattr(cfg.MOME, 'RESTRICT_HEAD', True):
        import torch as _t
        from datasets.make_dataloader_mome import __factory as _factory, _split_by_identity
        _ds = _factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
        _seed = cfg.MOME.SPLIT_SEED if cfg.MOME.SPLIT_SEED >= 0 else cfg.SOLVER.SEED
        _dr, _ = _split_by_identity(_ds.train, cfg.MOME.FORGET_ID_FRAC, _seed,
                                    leak_id_frac=getattr(cfg.MOME, 'LEAK_ID_FRAC', 0.0),
                                    dr_k=int(getattr(cfg.MOME, 'DR_K', 0)))
        _retain = sorted({int(_ds.train[i][1]) for i in _dr})
        retain_idx = _t.tensor(_retain, dtype=_t.long).cuda()
        label_map = _t.full((num_classes,), -1, dtype=_t.long)
        for _c, _p in enumerate(_retain):
            label_map[_p] = _c
        label_map = label_map.cuda()
        logger.info("sub_id: Core classifier restricted to {}/{} identities".format(
            len(_retain), num_classes))

    teacher = None
    if bool(cfg.MOME.STEP2_DISTILL) or float(cfg.MOME.CORE_KD) > 0:
        # The teacher is a plain TransReID checkpoint: build it WITHOUT the LoRA branch (the
        # wrapper renames Linear parameters and load_param drops mismatched keys silently), and
        # override the paths its yml carries, which point at its author's machine.
        _tcfg = cfg.clone(); _tcfg.defrost()
        if cfg.MOME.TEACHER_CONFIG:
            _tcfg.merge_from_file(cfg.MOME.TEACHER_CONFIG)
        _tcfg.MODEL.PARASITIC_LINEAR = False
        _tcfg.MODEL.PRETRAIN_PATH = cfg.MODEL.PRETRAIN_PATH
        _tcfg.DATASETS.ROOT_DIR = cfg.DATASETS.ROOT_DIR
        _tcfg.freeze()
        # TEACHER_WEIGHT may list several checkpoints. Members are built identically and the
        # distillation target becomes their averaged feature, which lets the teacher be an
        # ensemble of retain-only retrains that never saw the deleted identities.
        _tw = [w.strip() for w in str(cfg.MOME.TEACHER_WEIGHT).split(",") if w.strip()]
        _members = []
        for _w in _tw:
            _t = make_model(_tcfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
            _t.load_param(_w)
            _t = _t.cuda()
            set_parasitic_mode(_t, True)  # standard ckpts trained their shadows as plain weights
            _t.eval()
            for _p in _t.parameters():
                _p.requires_grad_(False)
            _members.append(_t)
        teacher = _members[0] if len(_members) == 1 else _TeacherEnsemble(_members)
        logger.info("v4 teacher loaded from %s (%d members, STEP2_DISTILL=%s, CORE_KD=%s)",
                    cfg.MOME.TEACHER_WEIGHT, len(_members), cfg.MOME.STEP2_DISTILL, cfg.MOME.CORE_KD)

    do_train_v3(
        cfg=cfg,
        model=model,
        train_loader_dr=train_loader_dr,
        train_loader_df=train_loader_df,
        train_eval_loader=train_eval_loader,
        train_num_query=train_num_query,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        num_query=num_query,
        local_rank=args.local_rank,
        retain_idx=retain_idx,
        label_map=label_map,
        teacher=teacher,
    )
