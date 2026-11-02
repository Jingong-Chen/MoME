#!/usr/bin/env python3
"""Evaluate RegDB cross-modality checkpoints over 10 trials and both directions."""

import argparse
import glob
import json
import os
import os.path as osp
import statistics
import sys

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import cfg as base_cfg
from datasets.bases import ImageDataset
from datasets.make_dataloader import val_collate_fn
from datasets.regdb_cross import RegDBCross
from model import make_model
from utils.metrics import R1_mAP_eval
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


def find_ckpt(method_dir, trial, kind):
    trial_dirs = [
        osp.join(method_dir, "trial_{}".format(trial)),
        osp.join(method_dir, "trial{}".format(trial)),
        method_dir,
    ]

    names = []
    if kind == "ga":
        names = ["gradient_ascent_final.pth"]
    elif kind == "mome":
        names = ["mome_v3_epoch_60.pth", "mome_v3_epoch_120.pth"]
    else:
        names = ["transformer_60.pth", "transformer_120.pth"]

    for d in trial_dirs:
        if not osp.isdir(d):
            continue
        for n in names:
            p = osp.join(d, n)
            if osp.isfile(p):
                return p

    candidates = []
    for d in trial_dirs:
        if osp.isdir(d):
            candidates.extend(glob.glob(osp.join(d, "*.pth")))
    if not candidates:
        return None
    candidates.sort(key=lambda x: osp.getmtime(x), reverse=True)
    return candidates[0]


@torch.no_grad()
def eval_once(cfg, ckpt, trial, direction, mode):
    dataset = RegDBCross(root=cfg.DATASETS.ROOT_DIR, trial=trial, test_mode=direction, verbose=False)

    val_transforms = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ]
    )
    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=val_collate_fn,
    )

    model = make_model(
        cfg,
        num_class=dataset.num_train_pids,
        camera_num=dataset.num_train_cams,
        view_num=dataset.num_train_vids,
    )
    model.load_param(ckpt)
    model = model.cuda().eval()

    set_parasitic_mode(model, mode == "full")

    evaluator = R1_mAP_eval(len(dataset.query), max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    for img, vid, camid, camids, target_view, _ in val_loader:
        feat = model(img.cuda(), cam_label=camids.cuda(), view_label=target_view.cuda())
        evaluator.update((feat, vid, camid))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    return {
        "R1": float(cmc[0] * 100.0),
        "R5": float(cmc[4] * 100.0),
        "R10": float(cmc[9] * 100.0),
        "mAP": float(mAP * 100.0),
    }


def summarize(items, key):
    vals = [x[key] for x in items if x.get(key) is not None]
    if not vals:
        return {"mean": None, "std": None}
    return {
        "mean": float(statistics.mean(vals)),
        "std": float(statistics.pstdev(vals) if len(vals) > 1 else 0.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default=os.environ.get("MOME_DATA", "./data"), type=str)
    parser.add_argument("--logs_root", default="logs", type=str)
    parser.add_argument("--config_std", default="configs/RegDB/vit_transreid_stride_cross.yml", type=str)
    parser.add_argument("--config_mome", default="configs/RegDB/vit_transreid_stride_cross_mome_v3.yml", type=str)
    parser.add_argument("--device", default="0", type=str)
    parser.add_argument("--trial_start", default=1, type=int)
    parser.add_argument("--trial_end", default=10, type=int)
    parser.add_argument("--output", default="logs/regdb_cross_10trial_summary.json", type=str)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    methods = [
        {
            "name": "standard",
            "dir": osp.join(args.logs_root, "regdb_cross_standard"),
            "kind": "std",
            "mode": "standard",
            "config": args.config_std,
        },
        {
            "name": "finetune",
            "dir": osp.join(args.logs_root, "regdb_cross_finetune"),
            "kind": "std",
            "mode": "standard",
            "config": args.config_std,
        },
        {
            "name": "ga",
            "dir": osp.join(args.logs_root, "regdb_cross_ga"),
            "kind": "ga",
            "mode": "standard",
            "config": args.config_std,
        },
        {
            "name": "retrain",
            "dir": osp.join(args.logs_root, "regdb_cross_retrain"),
            "kind": "std",
            "mode": "standard",
            "config": args.config_std,
        },
        {
            "name": "mome_core",
            "dir": osp.join(args.logs_root, "regdb_cross_mome_v3"),
            "kind": "mome",
            "mode": "core",
            "config": args.config_mome,
        },
        {
            "name": "mome_full",
            "dir": osp.join(args.logs_root, "regdb_cross_mome_v3"),
            "kind": "mome",
            "mode": "full",
            "config": args.config_mome,
        },
    ]

    output = {
        "meta": {
            "trial_range": [args.trial_start, args.trial_end],
            "root_dir": args.root_dir,
            "logs_root": args.logs_root,
            "device": args.device,
        },
        "methods": {},
    }

    for method in methods:
        cfg = build_cfg(method["config"], args.root_dir, args.device)
        rows_v2t = []
        rows_t2v = []

        print("\n=== {} ===".format(method["name"]))
        for trial in range(args.trial_start, args.trial_end + 1):
            ckpt = find_ckpt(method["dir"], trial, method["kind"])
            if ckpt is None:
                print("  trial {}: missing checkpoint under {}".format(trial, method["dir"]))
                continue

            try:
                v2t = eval_once(cfg, ckpt, trial, "v2t", method["mode"])
                t2v = eval_once(cfg, ckpt, trial, "t2v", method["mode"])
            except Exception as e:
                print("  trial {}: FAILED ({})".format(trial, str(e)))
                continue

            print(
                "  trial {} | V->T R1 {:.2f} R5 {:.2f} mAP {:.2f} | T->V R1 {:.2f} R5 {:.2f} mAP {:.2f}".format(
                    trial,
                    v2t["R1"],
                    v2t["R5"],
                    v2t["mAP"],
                    t2v["R1"],
                    t2v["R5"],
                    t2v["mAP"],
                )
            )

            rows_v2t.append({"trial": trial, "ckpt": ckpt, **v2t})
            rows_t2v.append({"trial": trial, "ckpt": ckpt, **t2v})

        method_out = {
            "v2t": rows_v2t,
            "t2v": rows_t2v,
            "summary": {
                "v2t_R1": summarize(rows_v2t, "R1"),
                "v2t_R5": summarize(rows_v2t, "R5"),
                "v2t_mAP": summarize(rows_v2t, "mAP"),
                "t2v_R1": summarize(rows_t2v, "R1"),
                "t2v_R5": summarize(rows_t2v, "R5"),
                "t2v_mAP": summarize(rows_t2v, "mAP"),
            },
        }
        output["methods"][method["name"]] = method_out

    os.makedirs(osp.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\nSaved summary:", args.output)


if __name__ == "__main__":
    main()
