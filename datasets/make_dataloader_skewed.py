"""Skewed retain-set splits (responds to Reviewer Nm86 N2).

Three strategies, all on Market-1501:
  - camera_skew:  D_f\\D_r = all images of camera 1; D_r = the rest (per PID).
  - identity_skew: delete N forget PIDs entirely; D_r = remaining PIDs (all images).
  - imbalanced_k: per-PID k varies in [1, k_max] uniformly random; D_f\\D_r = the complement.

Returns the same 10-tuple as make_dataloader_mome.
"""

import os
import random
from collections import defaultdict

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset
from .market1501 import Market1501
from .sampler import RandomIdentitySampler
from utils.timm_compat import ensure_torch_six_compat

ensure_torch_six_compat()
from timm.data.random_erasing import RandomErasing


def train_collate_fn(batch):
    imgs, pids, camids, viewids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids


def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths


def split_camera_skew(train_data, forget_camid=0, dr_per_pid=4, seed=42):
    """D_f\\D_r = all images with camid==forget_camid. D_r = sample dr_per_pid from remaining."""
    rng = random.Random(seed)
    forget_idx = [i for i, (_, _, c, _) in enumerate(train_data) if int(c) == forget_camid]
    keep_pool_by_pid = defaultdict(list)
    for i, (_, p, c, _) in enumerate(train_data):
        if int(c) != forget_camid:
            keep_pool_by_pid[p].append(i)
    dr_idx = []
    for pid, idxs in keep_pool_by_pid.items():
        k = min(len(idxs), dr_per_pid)
        dr_idx.extend(rng.sample(idxs, k=k))
    return sorted(set(dr_idx)), sorted(set(forget_idx))


def split_identity_skew(train_data, n_forget_pids=50, seed=42):
    """Delete n_forget_pids entirely from D_r."""
    pids = sorted(set(p for _, p, _, _ in train_data))
    rng = random.Random(seed)
    forget_pids = set(rng.sample(pids, k=min(n_forget_pids, len(pids))))
    dr_idx = [i for i, (_, p, _, _) in enumerate(train_data) if p not in forget_pids]
    forget_idx = [i for i, (_, p, _, _) in enumerate(train_data) if p in forget_pids]
    return sorted(dr_idx), sorted(forget_idx), forget_pids


def split_imbalanced_k(train_data, k_min=1, k_max=8, seed=42):
    rng = random.Random(seed)
    by_pid = defaultdict(list)
    for i, (_, p, _, _) in enumerate(train_data):
        by_pid[p].append(i)
    dr_idx = []; forget_idx = []
    for p, idxs in by_pid.items():
        k = rng.randint(k_min, k_max)
        k = min(k, len(idxs))
        chosen = rng.sample(idxs, k=k)
        keep_set = set(chosen)
        dr_idx.extend(chosen)
        forget_idx.extend([i for i in idxs if i not in keep_set])
    return sorted(set(dr_idx)), sorted(set(forget_idx))


def _build_train_loader(cfg, data_source, train_set, num_workers):
    return DataLoader(
        train_set,
        batch_size=cfg.SOLVER.IMS_PER_BATCH,
        sampler=RandomIdentitySampler(data_source, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
        num_workers=num_workers,
        collate_fn=train_collate_fn,
    )


def make_dataloader_skewed(cfg, strategy="camera_skew", **kwargs):
    """Strategy: camera_skew | identity_skew | imbalanced_k"""
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

    if cfg.DATASETS.NAMES != "market1501":
        raise NotImplementedError("Skewed splits implemented for Market-1501 only.")
    dataset = Market1501(root=cfg.DATASETS.ROOT_DIR)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    # A reference family must share the target's split and differ only in training
    # randomness, so the split seed has to be settable apart from SOLVER.SEED. Defaulting
    # to SOLVER.SEED leaves every existing caller on its previous behaviour.
    seed = kwargs.get("split_seed", cfg.SOLVER.SEED)

    if strategy == "camera_skew":
        forget_camid = kwargs.get("forget_camid", 0)
        dr_per_pid = kwargs.get("dr_per_pid", 4)
        dr_idx, df_only_idx = split_camera_skew(dataset.train, forget_camid=forget_camid,
                                                 dr_per_pid=dr_per_pid, seed=seed)
        info = {"strategy": "camera_skew", "forget_camid": forget_camid,
                "dr_per_pid": dr_per_pid}
    elif strategy == "identity_skew":
        n_forget_pids = kwargs.get("n_forget_pids", 50)
        dr_idx, df_only_idx, forget_pids = split_identity_skew(dataset.train,
                                                                n_forget_pids=n_forget_pids,
                                                                seed=seed)
        info = {"strategy": "identity_skew", "n_forget_pids": n_forget_pids,
                "forget_pids": sorted(int(p) for p in forget_pids)}
    elif strategy == "imbalanced_k":
        k_min = kwargs.get("k_min", 1); k_max = kwargs.get("k_max", 8)
        dr_idx, df_only_idx = split_imbalanced_k(dataset.train, k_min=k_min, k_max=k_max, seed=seed)
        info = {"strategy": "imbalanced_k", "k_min": k_min, "k_max": k_max}
    else:
        raise ValueError("strategy must be one of camera_skew/identity_skew/imbalanced_k")

    info["dr_size"] = len(dr_idx)
    info["df_only_size"] = len(df_only_idx)
    info["df_size"] = len(dataset.train)

    dr_data = [dataset.train[i] for i in dr_idx]
    df_only_data = [dataset.train[i] for i in df_only_idx]

    train_set_f = ImageDataset(dataset.train, train_transforms)
    train_set_r = ImageDataset(dr_data, train_transforms)
    train_set_df_only = ImageDataset(df_only_data, train_transforms)
    train_set_normal = ImageDataset(dataset.train, val_transforms)

    train_loader_f = _build_train_loader(cfg, dataset.train, train_set_f, num_workers)
    train_loader_r = _build_train_loader(cfg, dr_data, train_set_r, num_workers)
    train_loader_df_only = _build_train_loader(cfg, df_only_data, train_set_df_only, num_workers)
    val_loader = DataLoader(
        ImageDataset(dataset.query + dataset.gallery, val_transforms),
        batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False,
        num_workers=num_workers, collate_fn=val_collate_fn,
    )
    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False,
        num_workers=num_workers, collate_fn=val_collate_fn,
    )

    print("Skewed split [{}]: D_f={} D_r={} D_f\\\\D_r={}".format(
        strategy, info["df_size"], info["dr_size"], info["df_only_size"]))

    return (
        train_loader_f, train_loader_r, train_loader_df_only,
        train_loader_normal, val_loader,
        len(dataset.query), dataset.num_train_pids, dataset.num_train_cams, dataset.num_train_vids,
        info,
    )
