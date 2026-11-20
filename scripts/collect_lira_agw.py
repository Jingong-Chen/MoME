#!/usr/bin/env python3
"""Forgetting-Rank rho_U loss collection for the AGW ResNet-50 backbone (non-ViT certificate).

Same protocol as collect_lira_losses.py / collect_lira_full.py but with the AGW model loader
(model.build_agw.AGW, parasitic BN). Collects per-sample CE loss on the FULL D_f (fixed val-transform
ordering) for the 5 MoME shadows (Core mode) + 5 retrain shadows + the 5 MoME shadows in Full mode
(power positive control), plus each MoME seed's D_r membership. compute_rho.py then reads these.

Writes logs/gen2_agw_lira/{manifest.json, dr_membership.npy,
  losses_agwmome_s<seed>.npy, losses_agwretrain_s<seed>.npy, losses_agwmomefull_s<seed>.npy}.
Run: python scripts/collect_lira_agw.py --config_file configs/Market/agw.yml \
       MODEL.DEVICE_ID "'0'" DATASETS.ROOT_DIR ./data \
       MODEL.PRETRAIN_PATH ./resnet50-0676ba61.pth
"""
import argparse, json, os, sys
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from config import cfg as base_cfg
from datasets.bases import ImageDataset
from datasets.make_dataloader_mia import mia_collate_fn
from datasets.make_dataloader_mome import (make_dataloader_mome, _split_dr_and_df_minus_r,
                                           _split_by_identity)
from model.build_agw import AGW
from utils.parasitic_utils import set_parasitic_mode

MOME_SEEDS = [42, 123, 456, 789, 1024]
RETRAIN_SEEDS = [42, 123, 456, 789, 1024]
MOME_DIR = "logs/gen2_agw_core_s{s}/mome_v3_epoch_60.pth"
RETRAIN_DIR = "logs/gen2_agw_retrain_s{s}/agw_60.pth"


def build_cfg(config_file, opts):
    cfg = base_cfg.clone(); cfg.defrost(); cfg.merge_from_file(config_file)
    if opts and opts[0] == "--":
        opts = opts[1:]
    cfg.merge_from_list(opts); cfg.freeze(); return cfg


@torch.no_grad()
def per_sample_loss(model, loader, full):
    set_parasitic_mode(model, full); model.eval(); L = []
    for b in loader:
        img = b[0].cuda(); pid = b[1] if torch.is_tensor(b[1]) else torch.tensor(b[1])
        cam = b[2].cuda() if torch.is_tensor(b[2]) else torch.tensor(b[2]).cuda()
        view = b[3].cuda() if len(b) > 3 and torch.is_tensor(b[3]) else torch.zeros_like(cam)
        f = model(img, cam_label=cam, view_label=view)
        cls = model.classifier
        ip = getattr(model, "in_planes", f.shape[1])
        fc = f[:, :ip] if (torch.is_tensor(f) and f.shape[1] > ip) else f
        try:
            o = cls(fc)
        except Exception:
            o = cls(f)
        lg = o[0] if isinstance(o, (tuple, list)) else o
        L.append((-F.log_softmax(lg, -1).gather(1, pid.cuda().unsqueeze(1)).squeeze(1)).cpu().numpy())
    return np.concatenate(L)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config_file", required=True)
    p.add_argument("--dr_k", type=int, default=4)
    p.add_argument("opts", nargs=argparse.REMAINDER)
    a = p.parse_args()
    cfg = build_cfg(a.config_file, a.opts)
    outdir = osp.join(ROOT, "logs", "gen2_agw_lira"); os.makedirs(outdir, exist_ok=True)

    out = make_dataloader_mome(cfg)
    nc = out[6]
    full_src = out[0].dataset
    items = full_src.dataset if hasattr(full_src, "dataset") else full_src
    vtf = T.Compose([T.Resize(cfg.INPUT.SIZE_TEST), T.ToTensor(),
                     T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)])
    df_loader = DataLoader(ImageDataset(items, vtf), batch_size=cfg.TEST.IMS_PER_BATCH,
                           shuffle=False, num_workers=4, collate_fn=mia_collate_fn)
    manifest = [{"idx": i, "path": it[0], "pid": int(it[1])} for i, it in enumerate(items)]
    json.dump(manifest, open(osp.join(outdir, "manifest.json"), "w"))
    N = len(items); print(f"AGW D_f size N={N}, num_classes={nc}")

    # The models are trained under identity-level deletion when MOME.FORGET_MODE is sub_id, so
    # the membership mask has to come from the same split. This tool used to call the
    # sample-level split unconditionally, which keeps k images of EVERY identity and deletes no
    # identity at all, so the mask did not match how the checkpoints were trained.
    sub_id = str(getattr(cfg.MOME, "FORGET_MODE", "")).lower() == "sub_id"
    print(f"split protocol: {'identity-level (sub_id)' if sub_id else 'sample-level'}")
    dr_mem = np.zeros((len(MOME_SEEDS), N), dtype=bool)
    u_mask = None
    for si, s in enumerate(MOME_SEEDS):
        if sub_id:
            split_seed = cfg.MOME.SPLIT_SEED if cfg.MOME.SPLIT_SEED >= 0 else s
            dr_idx, df_only_idx = _split_by_identity(
                items, cfg.MOME.FORGET_ID_FRAC, split_seed,
                dr_k=int(getattr(cfg.MOME, "DR_K", 0)))
            if u_mask is None:
                u_mask = np.zeros(N, dtype=bool)
                u_mask[list(df_only_idx)] = True
        else:
            dr_idx, _ = _split_dr_and_df_minus_r(items, a.dr_k, s)
        dr_mem[si, list(dr_idx)] = True
    np.save(osp.join(outdir, "dr_membership.npy"), dr_mem)
    if u_mask is not None:
        np.save(osp.join(outdir, "u_mask.npy"), u_mask)
        print(f"u_mask saved, |U|={int(u_mask.sum())} of N={N}")
    print("D_r membership matrix:", dr_mem.shape, "mean IN frac", round(float(dr_mem.mean()), 4))

    def run(tag, ckpt, full):
        fp = osp.join(outdir, f"losses_{tag}.npy")
        if osp.exists(fp):
            print(f"have {tag}"); return
        if not osp.exists(ckpt):
            print(f"MISSING ckpt for {tag}: {ckpt}"); return
        m = AGW(num_classes=nc, cfg=cfg, parasitic=True)
        m.load_param(ckpt); m = m.cuda().eval()
        l = per_sample_loss(m, df_loader, full)
        np.save(fp, l); print(f"[OK] {tag}: N={len(l)} mean_loss={l.mean():.3f}")
        del m; torch.cuda.empty_cache()

    for s in MOME_SEEDS:
        run(f"agwmome_s{s}", MOME_DIR.format(s=s), full=False)       # Core mode (negative control)
    for s in RETRAIN_SEEDS:
        run(f"agwretrain_s{s}", RETRAIN_DIR.format(s=s), full=False)  # retrain reference
    for s in MOME_SEEDS:
        run(f"agwmomefull_s{s}", MOME_DIR.format(s=s), full=True)    # Full mode (power positive control)
    print("AGW COLLECT DONE")


if __name__ == "__main__":
    main()
