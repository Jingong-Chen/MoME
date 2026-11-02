import os
import pickle
import random
from collections import defaultdict

import torch
import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset, read_image
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .occ_duke import OCC_DukeMTMCreID
from .sampler import RandomIdentitySampler
from .sampler_ddp import RandomIdentitySampler_DDP
from .regdb import RegDB
from .vehicleid import VehicleID
from .veri import VeRi
from utils.timm_compat import ensure_torch_six_compat

ensure_torch_six_compat()
from timm.data.random_erasing import RandomErasing

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


def _split_dr_and_df_minus_r(train_data, dr_samples_per_id, seed):
    pid_to_indices = defaultdict(list)
    for idx, (_, pid, _, _) in enumerate(train_data):
        pid_to_indices[pid].append(idx)

    rng = random.Random(seed)
    dr_indices = set()
    for pid, indices in pid_to_indices.items():
        k = min(len(indices), dr_samples_per_id)
        chosen = rng.sample(indices, k=k)
        dr_indices.update(chosen)

    df_only_indices = [i for i in range(len(train_data)) if i not in dr_indices]
    dr_indices = sorted(dr_indices)
    return dr_indices, df_only_indices


def _split_by_identity(train_data, forget_id_frac, seed, leak_id_frac=0.0, return_leaked=False, dr_k=0,
                       dr_k_stratify_cam=False):
    """Identity-level deletion: every image of a randomly chosen fraction of the
    identities goes to U, and D_r keeps every image of the remaining identities.
    Unlike the per-identity split, no forgotten identity survives in D_r."""
    pid_to_indices = defaultdict(list)
    for idx, (_, pid, _, _) in enumerate(train_data):
        pid_to_indices[pid].append(idx)

    pids = sorted(pid_to_indices.keys())
    n_forget = int(round(len(pids) * forget_id_frac))
    n_forget = max(1, min(n_forget, len(pids) - 1))
    forget_pids = set(random.Random(seed).sample(pids, n_forget))

    # Audit control. leak_id_frac of the identities that were REQUESTED for deletion are
    # silently kept in the core update, as if those requests were never honoured. The deletion
    # request itself is unchanged, so U still covers every forget_pid and rho_U is still being
    # asked the same question; only the model's compliance changes. This is what gives the
    # certificate a detection-power curve instead of a single pass/fail.
    leaked_pids = set()
    if leak_id_frac and leak_id_frac > 0:
        ordered = sorted(forget_pids)
        n_leak = max(1, int(round(len(ordered) * leak_id_frac)))
        leaked_pids = set(random.Random(seed + 10007).sample(ordered, n_leak))

    dr_indices, df_only_indices = [], []
    for pid, indices in pid_to_indices.items():
        if pid in forget_pids and pid not in leaked_pids:
            df_only_indices.extend(indices)
        else:
            if dr_k and dr_k > 0:
                # thin the retain side: k images per retained identity, deterministic in the
                # split seed so every entrypoint reconstructs the same D_r. For cross-modal
                # data the camid encodes the modality, and sampling k per camid group keeps
                # both modalities represented; unstratified sampling could hand an identity
                # k images of a single modality and break cross-modal training.
                if dr_k_stratify_cam:
                    groups = defaultdict(list)
                    for i in indices:
                        groups[train_data[i][2]].append(i)
                    sel = []
                    for g in sorted(groups):
                        gi = groups[g]
                        sel.extend(random.Random(seed * 100003 + int(pid) * 31 + int(g))
                                   .sample(gi, min(dr_k, len(gi))))
                else:
                    sel = random.Random(seed * 100003 + int(pid)).sample(indices, min(dr_k, len(indices)))
                dr_indices.extend(sel)
            else:
                dr_indices.extend(indices)
    if return_leaked:
        return sorted(dr_indices), sorted(df_only_indices), sorted(leaked_pids), sorted(forget_pids)
    return sorted(dr_indices), sorted(df_only_indices)


def _extract_original_pid(img_path):
    fname = os.path.basename(img_path)
    return int(fname.split("_")[0])


def _build_original_to_relabel_pid_map(train_data):
    mapping = {}
    for img_path, relabeled_pid, _, _ in train_data:
        original_pid = _extract_original_pid(img_path)
        mapping[original_pid] = relabeled_pid
    return mapping


def _load_synthetic_dr_data(train_data, synthetic_dr_dir):
    if not synthetic_dr_dir:
        raise ValueError(
            "DATASETS.SYNTHETIC_DR_DIR is empty while DATASETS.USE_SYNTHETIC_DR=True"
        )

    img_dir = os.path.join(synthetic_dr_dir, "bounding_box_train_syn")
    meta_path = os.path.join(synthetic_dr_dir, "dr_syn_meta.pkl")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(
            "Synthetic D_r image directory not found: {}".format(img_dir)
        )

    original_to_relabel = _build_original_to_relabel_pid_map(train_data)
    relabeled_pid_set = set(original_to_relabel.values())

    records = []
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        for item in meta:
            if isinstance(item, dict):
                fname = item.get("fname") or item.get("filename")
                pid_value = item.get("orig_pid", item.get("pid"))
                camid = int(item.get("camid", 0))
            else:
                # backward-compatible tuple/list format
                if len(item) < 2:
                    continue
                fname = item[0]
                pid_value = item[1]
                camid = int(item[2]) if len(item) > 2 else 0

            if fname is None or pid_value is None:
                continue
            records.append((str(fname), int(pid_value), camid))
    else:
        # fallback: infer from filename pattern (PPPP_cXsY_*.jpg)
        for fname in sorted(os.listdir(img_dir)):
            if not (fname.endswith(".jpg") or fname.endswith(".png")):
                continue
            pid_value = int(fname.split("_")[0])
            records.append((fname, pid_value, 0))

    dr_data = []
    missing = 0
    for fname, pid_value, camid in records:
        fpath = os.path.join(img_dir, fname)
        if not os.path.isfile(fpath):
            missing += 1
            continue

        # Preferred path: original PID -> current relabeled PID in this run.
        if pid_value in original_to_relabel:
            relabeled_pid = original_to_relabel[pid_value]
        # Compatibility path: metadata already stores relabeled PID.
        elif pid_value in relabeled_pid_set:
            relabeled_pid = pid_value
        else:
            missing += 1
            continue

        dr_data.append((fpath, relabeled_pid, camid, 1))

    if not dr_data:
        raise RuntimeError(
            "No valid synthetic D_r samples found under {}".format(synthetic_dr_dir)
        )
    if missing > 0:
        print(
            "Warning: skipped {} synthetic samples due to missing files or unknown PIDs.".format(
                missing
            )
        )
    return dr_data




class DualViewImageDataset(ImageDataset):
    """Returns a strong view for the task loss and a weak view for KD teacher targets.
    Used only when MOME.KD_WEAK_TEACHER is on; every other consumer keeps ImageDataset."""

    def __init__(self, dataset, transform, weak_transform):
        super().__init__(dataset, transform)
        self.weak_transform = weak_transform

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)
        return (self.transform(img), self.weak_transform(img), pid, camid, trackid,
                img_path.split('/')[-1])


def dual_train_collate_fn(batch):
    imgs, weak, pids, camids, viewids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return (torch.stack(imgs, dim=0), torch.stack(weak, dim=0), pids, camids, viewids)


def _build_train_loader(cfg, data_source, train_set, num_workers, collate=train_collate_fn):
    if "triplet" in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(
                data_source, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE
            )
            batch_sampler = torch.utils.data.sampler.BatchSampler(
                data_sampler, mini_batch_size, True
            )
            return torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=collate,
                pin_memory=True,
            )
        return DataLoader(
            train_set,
            batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(
                data_source, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE
            ),
            num_workers=num_workers,
            collate_fn=collate,
        )

    if cfg.DATALOADER.SAMPLER == "softmax":
        return DataLoader(
            train_set,
            batch_size=cfg.SOLVER.IMS_PER_BATCH,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate,
        )

    raise ValueError(
        "unsupported sampler {} (expect softmax/softmax_triplet)".format(
            cfg.DATALOADER.SAMPLER
        )
    )


def make_dataloader_mome(cfg):
    train_transforms = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(
                probability=cfg.INPUT.RE_PROB, mode="pixel", max_count=1, device="cpu"
            ),
        ]
    )
    val_transforms = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ]
    )

    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if cfg.DATASETS.USE_SYNTHETIC_DR:
        dr_data = _load_synthetic_dr_data(dataset.train, cfg.DATASETS.SYNTHETIC_DR_DIR)
        # Synthetic D_r is external to D_f, so D_f\\D_r equals full D_f.
        df_only_data = list(dataset.train)
    else:
        split_seed = cfg.MOME.SPLIT_SEED if cfg.MOME.SPLIT_SEED >= 0 else cfg.SOLVER.SEED
        if cfg.MOME.FORGET_MODE == 'sub_id':
            dr_indices, df_only_indices = _split_by_identity(
                dataset.train, cfg.MOME.FORGET_ID_FRAC, split_seed,
                leak_id_frac=getattr(cfg.MOME, 'LEAK_ID_FRAC', 0.0),
                dr_k=int(getattr(cfg.MOME, 'DR_K', 0)),
                dr_k_stratify_cam=bool(getattr(cfg.MOME, 'DR_K_STRATIFY_CAM', False)),
            )
        else:
            dr_indices, df_only_indices = _split_dr_and_df_minus_r(
                dataset.train, cfg.MOME.DR_SAMPLES_PER_ID, cfg.SOLVER.SEED
            )
        dr_data = [dataset.train[i] for i in dr_indices]
        df_only_data = [dataset.train[i] for i in df_only_indices]

    train_f_data = list(dataset.train)
    if (bool(getattr(cfg.MOME, 'DF_EXCLUDE_U', False))
            and not cfg.DATASETS.USE_SYNTHETIC_DR and cfg.MOME.FORGET_MODE == 'sub_id'):
        _u_set = set(df_only_indices)
        train_f_data = [t for _i, t in enumerate(dataset.train) if _i not in _u_set]
        # Iteration-match the reference world: a shorter D_f list halves the iterations per
        # epoch and the total number of Step-1/Step-2 updates, which is itself a procedure
        # difference the reference-based attacks can read. Pad by deterministic repetition to
        # the original length so epochs, iterations and the LR schedule are identical.
        _orig_len = len(dataset.train)
        if train_f_data and len(train_f_data) < _orig_len:
            _reps = -(-_orig_len // len(train_f_data))
            train_f_data = (train_f_data * _reps)[:_orig_len]
        print(f"=> DF_EXCLUDE_U: reference world, D_f {_orig_len} -> "
              f"{len(set(id(t) for t in train_f_data))} unique padded to {len(train_f_data)}")

    train_set_f = ImageDataset(train_f_data, train_transforms)
    if bool(getattr(cfg.MOME, 'KD_WEAK_TEACHER', False)):
        train_set_r = DualViewImageDataset(dr_data, train_transforms, val_transforms)
    else:
        train_set_r = ImageDataset(dr_data, train_transforms)
    train_set_df_only = ImageDataset(df_only_data, train_transforms)
    train_set_normal = ImageDataset(dataset.train, val_transforms)

    train_loader_f = _build_train_loader(cfg, train_f_data, train_set_f, num_workers)
    train_loader_r = _build_train_loader(
        cfg, dr_data, train_set_r, num_workers,
        collate=dual_train_collate_fn
        if bool(getattr(cfg.MOME, 'KD_WEAK_TEACHER', False)) else train_collate_fn)
    train_loader_df_only = _build_train_loader(
        cfg, df_only_data, train_set_df_only, num_workers
    )

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_collate_fn,
    )
    train_loader_normal = DataLoader(
        train_set_normal,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_collate_fn,
    )

    split_info = {
        "dr_size": len(dr_data),
        "df_only_size": len(df_only_data),
        "df_size": len(dataset.train),
        "use_synthetic_dr": bool(cfg.DATASETS.USE_SYNTHETIC_DR),
    }
    if cfg.DATASETS.USE_SYNTHETIC_DR:
        print(
            "MoME split (synthetic D_r): D_f={} D_r_syn={} D_f\\\\D_r={}".format(
                split_info["df_size"], split_info["dr_size"], split_info["df_only_size"]
            )
        )
    else:
        print(
            "MoME split: D_f={} D_r={} D_f\\\\D_r={}".format(
                split_info["df_size"], split_info["dr_size"], split_info["df_only_size"]
            )
        )

    return (
        train_loader_f,
        train_loader_r,
        train_loader_df_only,
        train_loader_normal,
        val_loader,
        len(dataset.query),
        num_classes,
        cam_num,
        view_num,
        split_info,
    )
