"""Data Poisoning Audit (Pawelczyk ICLR'25 ref [9]).

Given a checkpoint (poisoned standard, or poisoned-then-unlearned), measure how often
trigger-patched query images get classified into a trigger PID via the closed-set
classification head.

This is "trigger persistence rate": if unlearning truly removed the backdoor, the rate
should drop near chance (1/num_classes ≈ 0.13% on Market-1501).
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

from config import cfg
from datasets.dataset_poisoned import AlwaysTriggeredImageDataset, TRIGGER_SIZE
from datasets.market1501 import Market1501
from datasets import make_dataloader
from model import make_model
from utils.logger import setup_logger
from utils.parasitic_utils import set_parasitic_mode


def _collate(batch):
    imgs, pids, camids, viewids, paths = zip(*batch)
    return (
        torch.stack(imgs, dim=0),
        torch.tensor(pids, dtype=torch.int64),
        torch.tensor(camids, dtype=torch.int64),
        torch.tensor(viewids, dtype=torch.int64),
        paths,
    )


@torch.no_grad()
def collect_top_preds(model, loader, device="cuda"):
    preds = []; pids = []; tops = []
    model.eval()
    for img, pid, camid, viewid, _ in loader:
        img = img.to(device); cam = camid.to(device); view = viewid.to(device)
        feat = model(img, cam_label=cam, view_label=view)
        classifier = getattr(model, "classifier", None)
        in_planes = getattr(model, "in_planes", feat.shape[1] if torch.is_tensor(feat) else 768)
        f = feat
        if torch.is_tensor(f) and f.shape[1] != in_planes and f.shape[1] >= in_planes:
            f = f[:, :in_planes]
        out = classifier(f)
        logits = out[0] if isinstance(out, tuple) else out
        probs = F.softmax(logits, dim=-1)
        top, pred = probs.max(dim=-1)
        preds.append(pred.cpu()); pids.append(pid.cpu()); tops.append(top.cpu())
    return torch.cat(preds).numpy(), torch.cat(pids).numpy(), torch.cat(tops).numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poisoning persistence audit")
    parser.add_argument("--config_file", default="", type=str)
    parser.add_argument("--mode", default="core", choices=["full", "core"])
    parser.add_argument("--trigger_pids_file", required=True, type=str,
                        help="JSON file with key 'trigger_pids' from train_with_trigger.py")
    parser.add_argument("--trigger_size", default=TRIGGER_SIZE, type=int)
    parser.add_argument("--out_json", default=None)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True) if output_dir else None
    logger = setup_logger("transreid", output_dir, if_train=False)

    with open(args.trigger_pids_file) as f:
        meta = json.load(f)
    trigger_pids = set(int(p) for p in meta["trigger_pids"])
    logger.info("Loaded {} trigger PIDs".format(len(trigger_pids)))

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID
    _, _, _, _, num_classes, camera_num, view_num = make_dataloader(cfg)

    # Build patched query loader (ALL query images get trigger).
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])
    dataset = Market1501(root=cfg.DATASETS.ROOT_DIR)
    patched_set = AlwaysTriggeredImageDataset(dataset.query, transform=val_transforms,
                                              trigger_size=args.trigger_size)
    patched_loader = DataLoader(patched_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False,
                                 num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=_collate)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(cfg.TEST.WEIGHT)
    model = model.cuda(); model.eval()
    set_parasitic_mode(model, mode=(args.mode == "full"))

    preds, _query_pids, tops = collect_top_preds(model, patched_loader)
    n = len(preds)
    is_trigger_pred = np.array([int(p) in trigger_pids for p in preds])
    persistence_rate = float(is_trigger_pred.mean())  # high → poisoning survives

    # Per-trigger-pid concentration (which trigger PID does the model jump to)
    bincount = {}
    for p in preds:
        if int(p) in trigger_pids:
            bincount[int(p)] = bincount.get(int(p), 0) + 1
    top_targets = sorted(bincount.items(), key=lambda kv: kv[1], reverse=True)[:10]

    res = {
        "persistence_rate": persistence_rate,
        "chance_rate": float(len(trigger_pids) / num_classes),
        "n_query": n,
        "n_trigger_pids": len(trigger_pids),
        "num_classes": num_classes,
        "mean_top_prob": float(tops.mean()),
        "top_attracted_pids": top_targets,
        "meta": {"target_ckpt": cfg.TEST.WEIGHT, "mode": args.mode,
                 "trigger_pids_file": args.trigger_pids_file},
    }
    out_json = args.out_json or os.path.join(output_dir, "poisoning_audit_{}.json".format(args.mode))
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
    logger.info("Persistence rate: {:.4f} (chance {:.4f}). Saved {}".format(
        persistence_rate, res["chance_rate"], out_json))
