#!/usr/bin/env python3
"""MIA evaluation for RegDB cross-modality checkpoints."""

import argparse
import json
import os
import os.path as osp
import sys

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import cfg as base_cfg
from datasets.bases import ImageDataset
from datasets.make_dataloader_mia import mia_collate_fn
from datasets.regdb_cross import RegDBCross
from model import make_model
from utils.mia_utils import (
    collect_outputs,
    compute_class_centers,
    compute_mia_auc,
    embedding_distance_score,
    entropy_score,
    max_logit_score,
)
from utils.parasitic_utils import set_parasitic_mode


def build_cfg(config_file, root_dir, device):
    cfg = base_cfg.clone()
    cfg.defrost()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(
        [
            "MODEL.DEVICE_ID",
            "('{}')".format(device),
            "DATASETS.ROOT_DIR",
            root_dir,
        ]
    )
    cfg.freeze()
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--weight", required=True, type=str)
    parser.add_argument("--trial", default=1, type=int)
    parser.add_argument("--test_mode", default="v2t", choices=["v2t", "t2v"])
    parser.add_argument("--mode", default="standard", choices=["standard", "core", "full"])
    parser.add_argument("--root_dir", default=os.environ.get("MOME_DATA", "./data"), type=str)
    parser.add_argument("--device", default="0", type=str)
    parser.add_argument("--output", default="", type=str)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    cfg = build_cfg(args.config_file, args.root_dir, args.device)

    dataset = RegDBCross(root=cfg.DATASETS.ROOT_DIR, trial=args.trial, test_mode=args.test_mode, verbose=False)

    val_transforms = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ]
    )

    member_set = ImageDataset(dataset.train, val_transforms)
    nonmember_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

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

    model = make_model(
        cfg,
        num_class=dataset.num_train_pids,
        camera_num=dataset.num_train_cams,
        view_num=dataset.num_train_vids,
    )
    model.load_param(args.weight)
    model = model.cuda().eval()

    if args.mode == "full":
        set_parasitic_mode(model, True)
    else:
        set_parasitic_mode(model, False)

    member_feats, member_logits, member_pids = collect_outputs(model, member_loader, device="cuda")
    nonmember_feats, nonmember_logits, _ = collect_outputs(model, nonmember_loader, device="cuda")

    results = {
        "trial": int(args.trial),
        "test_mode": args.test_mode,
        "mode": args.mode,
        "weight": args.weight,
        "num_member": int(len(member_set)),
        "num_nonmember": int(len(nonmember_set)),
    }

    centers = compute_class_centers(member_feats, member_pids, dataset.num_train_pids)
    member_emb_scores = embedding_distance_score(member_feats, centers).numpy()
    nonmember_emb_scores = embedding_distance_score(nonmember_feats, centers).numpy()
    results["embedding_auc"] = float(compute_mia_auc(member_emb_scores, nonmember_emb_scores))

    if member_logits is not None and nonmember_logits is not None:
        member_ent_scores = entropy_score(member_logits).numpy()
        nonmember_ent_scores = entropy_score(nonmember_logits).numpy()
        results["entropy_auc"] = float(compute_mia_auc(member_ent_scores, nonmember_ent_scores))

        member_logit_scores = max_logit_score(member_logits).numpy()
        nonmember_logit_scores = max_logit_score(nonmember_logits).numpy()
        results["maxlogit_auc"] = float(compute_mia_auc(member_logit_scores, nonmember_logit_scores))
    else:
        results["entropy_auc"] = None
        results["maxlogit_auc"] = None

    if args.output:
        out = args.output
    else:
        stem = "mia_{}_trial{}_{}.json".format(args.mode, args.trial, args.test_mode)
        out = os.path.join(os.path.dirname(args.weight), stem)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print("Saved:", out)


if __name__ == "__main__":
    main()
