#!/usr/bin/env python3
"""Membership-inference decomposition under the identity-level protocol.

eval_mia_partition.py splits with _split_dr_and_df_minus_r, which is the sample-level
splitter, so its "forget" pool is D_f minus D_r over all identities. Under the protocol the
paper actually reports, the pools are different objects and that pool is not U. This
script scores four pools against the query non-member set:

    D_f      every training image
    U        every image of a deleted identity          <- the only pool about removal
    D_r      the k images kept per retained identity
    rest     retained identities' images outside D_r    <- neither deleted nor kept

Entropy attack only, the same statistic the appendix tables report. Inference only.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from config import cfg
from datasets.bases import ImageDataset
from datasets.make_dataloader import val_collate_fn, __factory as DS
from datasets.make_dataloader_mome import _split_by_identity
from model import make_model
from utils.mia_utils import compute_mia_auc
from utils.parasitic_utils import set_parasitic_mode, build_cfg_for_ckpt
import torch.nn.functional as F


def loader_for(items, c):
    tf = T.Compose([T.Resize(c.INPUT.SIZE_TEST), T.ToTensor(),
                    T.Normalize(mean=c.INPUT.PIXEL_MEAN, std=c.INPUT.PIXEL_STD)])
    return DataLoader(ImageDataset(items, tf), batch_size=c.TEST.IMS_PER_BATCH,
                      shuffle=False, num_workers=8, collate_fn=val_collate_fn)


@torch.no_grad()
def neg_entropy(model, loader):
    out = []
    for batch in loader:
        img, _pid, _cam, camids, tv = batch[0], batch[1], batch[2], batch[3], batch[4]
        feat = model(img.cuda(), cam_label=camids.cuda(), view_label=tv.cuda())
        cls = getattr(model, "classifier", None)
        ip = getattr(model, "in_planes", feat.shape[1])
        f = feat[:, :ip] if feat.shape[1] != ip and feat.shape[1] >= ip else feat
        lg = cls(f)
        lg = lg[0] if isinstance(lg, tuple) else lg
        p = F.softmax(lg, dim=-1).clamp_min(1e-12)
        out.append((p * p.log()).sum(-1).cpu())          # -H, higher = more member-like
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_file", required=True)
    ap.add_argument("--rows", required=True, help="semicolon list of tag,ckpt,mode")
    ap.add_argument("--out", default="logs/partition_idlvl.json")
    ap.add_argument("opts", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if a.opts and a.opts[0] == "--":
        a.opts = a.opts[1:]

    cfg.merge_from_file(a.config_file)
    if a.opts:
        cfg.merge_from_list(a.opts)
    cfg.freeze()
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

    ds = DS[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    dr_idx, u_idx = _split_by_identity(
        ds.train, cfg.MOME.FORGET_ID_FRAC, cfg.MOME.SPLIT_SEED,
        dr_k=int(getattr(cfg.MOME, "DR_K", 0)),
        dr_k_stratify_cam=bool(getattr(cfg.MOME, "DR_K_STRATIFY_CAM", False)))
    dr_set, u_set = set(dr_idx), set(u_idx)
    pools = {
        "D_f":  list(ds.train),
        "U":    [ds.train[i] for i in u_idx],
        "D_r":  [ds.train[i] for i in dr_idx],
        "rest": [ds.train[i] for i in range(len(ds.train)) if i not in dr_set and i not in u_set],
    }
    print({k: len(v) for k, v in pools.items()}, flush=True)

    q_loader = loader_for(list(ds.query), cfg)
    results = json.load(open(a.out)) if os.path.exists(a.out) else {}
    results["_pools"] = {k: len(v) for k, v in pools.items()}
    results["_nonmember"] = len(ds.query)

    for row in a.rows.split(";"):
        tag, ckpt, mode = [x.strip() for x in row.split(",")]
        if tag in results:
            print("[have]", tag, flush=True); continue
        c = build_cfg_for_ckpt(cfg, ckpt) if callable(globals().get("build_cfg_for_ckpt")) else cfg
        model = make_model(c, num_class=len(set(t[1] for t in ds.train)),
                           camera_num=ds.num_train_cams, view_num=getattr(ds, "num_train_vids", 1))
        model.load_param(ckpt)
        model = model.cuda().eval()
        set_parasitic_mode(model, mode == "full")
        q = neg_entropy(model, q_loader)
        row_out = {}
        for name, items in pools.items():
            m = neg_entropy(model, loader_for(items, cfg))
            row_out[name] = round(float(compute_mia_auc(m, q)), 4)
        results[tag] = {"ckpt": ckpt, "mode": mode, **row_out}
        print("[row]", tag, row_out, flush=True)
        del model; torch.cuda.empty_cache()
        json.dump(results, open(a.out, "w"), indent=1)
    print("PARTITION_IDLVL_DONE")


if __name__ == "__main__":
    main()
