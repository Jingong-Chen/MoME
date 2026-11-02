import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .occ_duke import OCC_DukeMTMCreID
from .regdb import RegDB
from .vehicleid import VehicleID
from .veri import VeRi

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


def mia_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    import torch

    return (
        torch.stack(imgs, dim=0),
        torch.tensor(pids, dtype=torch.int64),
        torch.tensor(camids, dtype=torch.int64),
        torch.tensor(viewids, dtype=torch.int64),
        img_paths,
    )


def make_dataloader_mia(cfg, skew=None, **skew_kwargs):
    """skew, when given, takes the member set from a skewed split instead of the identity-level
    one, so the calibrated attacks can be reported for the rows of the uneven-retain-set study.
    The split seed is MOME.SPLIT_SEED, matching mia_decomp.py, so the target and its reference
    family see the same U. Default None leaves every existing caller unchanged.
    """
    val_transforms = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ]
    )

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    # Under identity-level deletion the audit question is whether the DELETED identities can be
    # shown to have been in training, so the member set must be U, not the whole training set.
    # Using dataset.train makes the retained half genuine members for Core and the attack then
    # reports the membership of data nobody asked to delete: Core scored 0.75 and Full 0.55,
    # i.e. exactly inverted with respect to what the models actually forgot.
    members = dataset.train
    if skew:
        from .make_dataloader_skewed import (split_camera_skew, split_identity_skew,
                                             split_imbalanced_k)
        sseed = cfg.MOME.SPLIT_SEED if cfg.MOME.SPLIT_SEED >= 0 else cfg.SOLVER.SEED
        if skew == "camera_skew":
            _, u_idx = split_camera_skew(dataset.train,
                                         forget_camid=skew_kwargs.get("forget_camid", 0),
                                         dr_per_pid=skew_kwargs.get("dr_per_pid", 4), seed=sseed)
        elif skew == "identity_skew":
            # this one returns the forgotten pids as a third value; the other two do not
            _, u_idx = split_identity_skew(dataset.train,
                                           n_forget_pids=skew_kwargs.get("n_forget_pids", 50),
                                           seed=sseed)[:2]
        elif skew == "imbalanced_k":
            _, u_idx = split_imbalanced_k(dataset.train,
                                          k_min=skew_kwargs.get("k_min", 1),
                                          k_max=skew_kwargs.get("k_max", 8), seed=sseed)
        else:
            raise ValueError("skew must be camera_skew, identity_skew or imbalanced_k")
        members = [dataset.train[i] for i in u_idx]
        print("MIA members = U of skew {} ({} images), nonmembers = query ({})".format(
            skew, len(members), len(dataset.query)))
    elif getattr(cfg.MOME, "FORGET_MODE", "per_id") == "sub_id":
        from .make_dataloader_mome import _split_by_identity
        split_seed = cfg.MOME.SPLIT_SEED if cfg.MOME.SPLIT_SEED >= 0 else cfg.SOLVER.SEED
        _, df_only_idx = _split_by_identity(
            dataset.train, cfg.MOME.FORGET_ID_FRAC, split_seed,
            leak_id_frac=getattr(cfg.MOME, "LEAK_ID_FRAC", 0.0))
        members = [dataset.train[i] for i in df_only_idx]
        print("MIA members = U ({} images of the deleted identities), "
              "nonmembers = query ({})".format(len(members), len(dataset.query)))

    member_set = ImageDataset(members, val_transforms)
    nonmember_set = ImageDataset(dataset.query, val_transforms)

    member_loader = DataLoader(
        member_set,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=mia_collate_fn,
    )
    nonmember_loader = DataLoader(
        nonmember_set,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=mia_collate_fn,
    )
    return member_loader, nonmember_loader, dataset.num_train_pids
