#!/usr/bin/env python3
"""Test-set mAP and Rank-1 for one row of the main tables, from its own checkpoint.

The attack columns of table::merged_market_veri are traced to json files; the utility pair is not,
because it was computed in a pass whose artifacts did not survive. This recomputes it so every cell
has a file behind it.

Two things it does that test.py does not. It builds the model through build_cfg_for_ckpt, so a
checkpoint carrying the LoRA wrapper is not loaded into a bare Linear, and it asserts the load is
exact apart from the parasitic BatchNorm shadows, which Core mode never reads. And it sets the
parasitic mode explicitly rather than relying on the build default.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import cfg                                  # noqa: E402
from datasets import make_dataloader                    # noqa: E402
from model import make_model                            # noqa: E402
from processor.processor import do_inference            # noqa: E402
from utils.metrics import R1_mAP_eval                   # noqa: E402
from utils.parasitic_utils import build_cfg_for_ckpt, set_parasitic_mode  # noqa: E402

ALLOWED = {"bottleneck%s.running_%s_m" % (i, v)
           for i in ("", "_1", "_2", "_3", "_4") for v in ("mean", "var")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_file", required=True)
    ap.add_argument("--weight", required=True)
    # "standard" is the third mode the attack tool uses and it is not a synonym for core: it
    # leaves the parasitic branch ON, because a baseline trained by an ordinary entrypoint keeps
    # its shadow tensors as ordinary trainable parameters. Reading the Original in core mode gives
    # 83.22 mAP where the row reports 88.0, which is the branch being switched off under it.
    ap.add_argument("--mode", default="core", choices=["core", "full", "standard"])
    ap.add_argument("--out_json", required=True)
    # LOFT is a projection applied to the embedding at evaluation, not a set of weights, so its
    # row shares the Original's checkpoint and differs only by this operator. Without it the row
    # would simply reproduce the Original.
    ap.add_argument("--loft", default=None)
    ap.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    a = ap.parse_args()
    cfg.merge_from_file(a.config_file)
    opts = [x for x in a.opts if x != "--"]
    if opts:
        cfg.merge_from_list(opts)
    cfg.merge_from_list(["TEST.WEIGHT", a.weight])
    built = build_cfg_for_ckpt(cfg, a.weight)
    built.freeze()
    os.environ["CUDA_VISIBLE_DEVICES"] = built.MODEL.DEVICE_ID

    _, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(built)
    model = make_model(built, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    sd = torch.load(a.weight, map_location="cpu")
    sd = sd.get("state_dict", sd) if hasattr(sd, "get") else sd
    miss = {k for k in set(model.state_dict()) - set(sd) if not k.endswith("num_batches_tracked")}
    extra = set(sd) - set(model.state_dict())
    if extra or not miss <= ALLOWED:
        raise SystemExit("load would not be exact: %d extra, %d unexpected missing"
                         % (len(extra), len(miss - ALLOWED)))
    model.load_param(a.weight)
    set_parasitic_mode(model, a.mode != "core")
    model.eval().cuda()
    if a.loft:
        from eval_loft import LoftWrapper
        payload = torch.load(a.loft, map_location="cpu")
        U = payload["U"].float()
        model = LoftWrapper(model, (U @ U.t()).float().cuda()).cuda()
        model.eval()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=built.TEST.FEAT_NORM)
    evaluator.reset()
    with torch.no_grad():
        for img, pid, camid, camids, target_view, _ in val_loader:
            img = img.cuda()
            camids = camids.cuda()
            target_view = target_view.cuda()
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    out = {"weight": a.weight, "mode": a.mode, "config": a.config_file, "loft": a.loft,
           "test_mAP": float(mAP) * 100, "test_R1": float(cmc[0]) * 100,
           "test_R5": float(cmc[4]) * 100, "num_query": int(num_query)}
    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    json.dump(out, open(a.out_json, "w"), indent=1)
    print("%-44s mAP %.2f  R-1 %.2f" % (os.path.basename(a.weight), out["test_mAP"], out["test_R1"]))


if __name__ == "__main__":
    main()
