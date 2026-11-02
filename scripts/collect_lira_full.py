#!/usr/bin/env python3
"""POWER / positive-control collection for the Forgetting-Rank rho_U test (Market).

Re-evaluate the SAME 5 MoME shadows in FULL mode (parasitic ON = the readout that was TRAINED on the
forget set D_f) on the identical D_f ordering used by collect_lira_losses.py. This is the leaking
positive control: Full mode has seen the forget-trained params, so on forgotten x it should have
LOWER loss than a true forgetter => rho_U << 1/2 (leak detected). Compared against the negative
control (Core mode, rho_U = 1/2) this demonstrates the test has both correct size AND power.

Writes logs/lira/losses_momefull_s<seed>.npy aligned to logs/lira/manifest.json ordering.
Run: python collect_lira_full.py --config_file configs/Market/vit_transreid_stride_mome_v3.yml \
       MODEL.DEVICE_ID "'0'" DATASETS.ROOT_DIR ./data
"""
import argparse, json, os, sys
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from config import cfg as base_cfg
from datasets.bases import ImageDataset
from datasets.make_dataloader_mia import mia_collate_fn
from datasets.make_dataloader_mome import make_dataloader_mome
from model import make_model
from utils.parasitic_utils import set_parasitic_mode, build_cfg_for_ckpt

MOME_SEEDS = [42, 123, 456, 789, 1024]

@torch.no_grad()
def per_sample_loss(model, loader, full):
    set_parasitic_mode(model, full); model.eval(); L = []
    for b in loader:
        img = b[0].cuda(); pid = b[1] if torch.is_tensor(b[1]) else torch.tensor(b[1])
        cam = b[2].cuda() if torch.is_tensor(b[2]) else torch.tensor(b[2]).cuda()
        view = b[3].cuda() if len(b) > 3 and torch.is_tensor(b[3]) else torch.zeros_like(cam)
        f = model(img, cam_label=cam, view_label=view); cls = model.classifier
        ip = getattr(model, "in_planes", f.shape[1]); fc = f[:, :ip] if (torch.is_tensor(f) and f.shape[1] > ip) else f
        try: o = cls(fc)
        except Exception: o = cls(f)
        lg = o[0] if isinstance(o, (tuple, list)) else o
        L.append((-F.log_softmax(lg, -1).gather(1, pid.cuda().unsqueeze(1)).squeeze(1)).cpu().numpy())
    return np.concatenate(L)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config_file", required=True)
    p.add_argument("--outdir", default="logs/lira",
                   help="must match the directory holding the Core-mode losses and its manifest; "
                        "sharing a directory across protocols silently mixes them")
    p.add_argument("--mome_tpl", default="logs/mome_v3_seed{seed}/mome_v3_epoch_120.pth")
    p.add_argument("opts", nargs=argparse.REMAINDER)
    a = p.parse_args()
    cfg = base_cfg.clone(); cfg.defrost(); cfg.merge_from_file(a.config_file)
    opts = a.opts[1:] if a.opts and a.opts[0] == "--" else a.opts
    cfg.merge_from_list(opts); cfg.freeze()
    outdir = a.outdir if osp.isabs(a.outdir) else osp.join(ROOT, a.outdir)

    out = make_dataloader_mome(cfg); nc, cn, vn = out[6], out[7], out[8]
    full_src = out[0].dataset
    items = full_src.dataset if hasattr(full_src, "dataset") else full_src
    # sanity: order must match the existing manifest used for dr_membership
    man = json.load(open(osp.join(outdir, "manifest.json")))
    assert len(man) == len(items), f"manifest {len(man)} != items {len(items)}"
    assert all(man[i]["path"] == items[i][0] for i in range(0, len(items), max(1, len(items)//50))), "ORDER MISMATCH"
    vtf = T.Compose([T.Resize(cfg.INPUT.SIZE_TEST), T.ToTensor(),
                     T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)])
    loader = DataLoader(ImageDataset(items, vtf), batch_size=cfg.TEST.IMS_PER_BATCH,
                        shuffle=False, num_workers=4, collate_fn=mia_collate_fn)
    print(f"D_f N={len(items)} nc={nc} (order matches manifest)")

    for s in MOME_SEEDS:
        fp = osp.join(outdir, f"losses_momefull_s{s}.npy")
        if osp.exists(fp): print(f"have momefull_s{s}"); continue
        ck = a.mome_tpl.format(seed=s)
        if not osp.exists(ck): print(f"MISSING ckpt for momefull_s{s}: {ck}"); continue
        _cfg = build_cfg_for_ckpt(cfg, ck)
        m = make_model(_cfg, num_class=nc, camera_num=cn, view_num=vn)
        m.load_param(ck); m = m.cuda().eval()
        l = per_sample_loss(m, loader, full=True)
        np.save(fp, l); print(f"[OK] momefull_s{s}: mean_loss={l.mean():.3f}")
        del m; torch.cuda.empty_cache()
    print("FULL COLLECT DONE")

if __name__ == "__main__":
    main()
