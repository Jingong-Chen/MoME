#!/usr/bin/env python3
"""Entropy-MIA decomposed over the three parts of the training set, under identity-level deletion.

The appendix table this replaces was computed under the sample-level protocol, |D_r| = 2986
rather than 1491, and scored membership on the complement of D_r. Under identity-level deletion
that complement is 43% held-out images of RETAINED identities, which the model is entitled to
recognize, so those numbers read near 1.0 for every method including a genuine retrain.

Three member pools, one non-member pool:
  full     the whole training set, the pool the old table used
  forget   U, the images of the deleted identities, the only pool that asks about removal
  retain   D_r, the images the model was trained on
Non-members are the query split. AUC is a within-model rank statistic, so it is comparable
across models whose classifiers differ in width.

Run: python scripts/cvpr2027/mia_decomp.py --config_file configs/Market/vit_transreid_stride_mome_v3.yml \
       --ckpt <path> --tag <name> [--mode core|full] MODEL.DEVICE_ID "'0'" ...
"""
import argparse
import json
import os
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from config import cfg as base_cfg  # noqa: E402
from datasets import make_dataloader  # noqa: E402
from datasets.bases import ImageDataset  # noqa: E402
from datasets.make_dataloader_mia import mia_collate_fn  # noqa: E402
from datasets.make_dataloader_mome import _split_by_identity  # noqa: E402
from model import make_model  # noqa: E402
from utils.parasitic_utils import build_cfg_for_ckpt, set_parasitic_mode  # noqa: E402
import torchvision.transforms as T  # noqa: E402


@torch.no_grad()
def entropies(model, items, cfg, device="cuda"):
    vtf = T.Compose([T.Resize(cfg.INPUT.SIZE_TEST), T.ToTensor(),
                     T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)])
    loader = DataLoader(ImageDataset(items, vtf), batch_size=cfg.TEST.IMS_PER_BATCH,
                        shuffle=False, num_workers=4, collate_fn=mia_collate_fn)
    out = []
    for b in loader:
        img = b[0].to(device)
        cam = b[2].to(device) if torch.is_tensor(b[2]) else torch.tensor(b[2]).to(device)
        view = b[3].to(device) if len(b) > 3 and torch.is_tensor(b[3]) else torch.zeros_like(cam)
        # In eval mode the model returns FEATURES, and the classification path is
        # classifier(bottleneck(feature)). Applying the classifier to the raw feature skips a
        # BatchNorm and produces logits the model never emits, which is what an earlier version
        # of this script did.
        f = model(img, cam_label=cam, view_label=view)
        if torch.is_tensor(f) and f.shape[1] > model.in_planes:
            f = f[:, :model.in_planes]
        logits = model.classifier(model.bottleneck(f))
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        p = F.softmax(logits.float(), dim=1)
        out.append((-(p * torch.log(p + 1e-12)).sum(1)).cpu().numpy())
    return np.concatenate(out)


def auc(member, nonmember):
    n = min(len(member), len(nonmember))
    scores = np.concatenate([-member[:n], -nonmember[:n]])
    labels = np.concatenate([np.ones(n), np.zeros(n)])
    return float(roc_auc_score(labels, scores))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config_file", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--mode", default="standard", choices=["standard", "core", "full"])
    p.add_argument("--out_dir", default="logs/mia_decomp_idlvl")
    p.add_argument("--skew", default="", choices=["", "camera_skew", "identity_skew", "imbalanced_k"],
                   help="take the forget set from this skew strategy instead of identity-level "
                        "deletion. Each strategy deletes a different thing, so scoring one of them "
                        "against the main protocol's U measures the wrong membership entirely.")
    p.add_argument("--dr_k", type=int, default=4)
    p.add_argument("--forget_camid", type=int, default=0)
    p.add_argument("--n_forget_pids", type=int, default=50)
    p.add_argument("opts", nargs=argparse.REMAINDER)
    a = p.parse_args()

    cfg = base_cfg.clone(); cfg.defrost(); cfg.merge_from_file(a.config_file)
    opts = a.opts[1:] if (a.opts and a.opts[0] == "--") else a.opts
    if opts:
        cfg.merge_from_list(opts)
    cfg.freeze()

    train_loader, _, val_loader, num_query, num_classes, cam_num, view_num = make_dataloader(cfg)[:7]
    src = train_loader.dataset
    items = src.dataset if hasattr(src, "dataset") else src

    if a.skew:
        from datasets.make_dataloader_skewed import (split_camera_skew, split_identity_skew,
                                                     split_imbalanced_k)
        if a.skew == "camera_skew":
            dr_idx, u_idx = split_camera_skew(items, forget_camid=a.forget_camid,
                                              dr_per_pid=a.dr_k, seed=cfg.MOME.SPLIT_SEED)
        elif a.skew == "identity_skew":
            # this one returns the forgotten pids as a third value; the other two do not
            dr_idx, u_idx = split_identity_skew(items, n_forget_pids=a.n_forget_pids,
                                                seed=cfg.MOME.SPLIT_SEED)[:2]
        else:
            dr_idx, u_idx = split_imbalanced_k(items, seed=cfg.MOME.SPLIT_SEED)
        print(f"skew={a.skew}: |D_r|={len(dr_idx)} |U|={len(u_idx)}")
    else:
        dr_idx, u_idx = _split_by_identity(items, cfg.MOME.FORGET_ID_FRAC, cfg.MOME.SPLIT_SEED,
                                           dr_k=int(cfg.MOME.DR_K))
    pools = {"full": list(items),
             "forget": [items[i] for i in u_idx],
             "retain": [items[i] for i in dr_idx]}

    mcfg = build_cfg_for_ckpt(cfg, a.ckpt)
    model = make_model(mcfg, num_class=num_classes, camera_num=cam_num, view_num=view_num)
    model.load_param(a.ckpt)
    model = model.cuda().eval()
    set_parasitic_mode(model, a.mode == "full")

    # non-members: the query split, disjoint identities from training
    q_items = [val_loader.dataset.dataset[i] if hasattr(val_loader.dataset, "dataset")
               else val_loader.dataset[i] for i in range(num_query)]
    ent_q = entropies(model, q_items, cfg)

    res = {"tag": a.tag, "mode": a.mode, "ckpt": a.ckpt,
           "sizes": {k: len(v) for k, v in pools.items()}, "auc": {}}
    for name, pool in pools.items():
        res["auc"][name] = round(auc(entropies(model, pool, cfg), ent_q), 4)
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(res, open(osp.join(a.out_dir, f"{a.tag}_{a.mode}.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
