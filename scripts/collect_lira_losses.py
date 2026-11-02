#!/usr/bin/env python3
"""Collect per-sample losses for MoME-class + oracle-class shadow models on the FULL D_f,
plus each MoME seed's D_r membership, for a MoME-self-calibrated U-LiRA unlearning test.

Fixed image ordering across all models (val transforms). Saves:
  logs/lira/losses_<tag>.npy       per-sample CE loss on D_f (order = manifest)
  logs/lira/manifest.json          list of (idx, path, pid) defining the order
  logs/lira/dr_membership.npy      [n_mome_seeds, N] bool: is image in that seed's D_r
  logs/lira/meta.json              seeds, tags
Run once; the LiRA statistic is then computed offline (CPU) from these arrays.
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
from datasets.make_dataloader_mome import _split_dr_and_df_minus_r, _split_by_identity
from model import make_model
from utils.parasitic_utils import set_parasitic_mode, build_cfg_for_ckpt

MOME_SEEDS = [42, 123, 456, 789, 1024]   # overridden from the CLI in main()
ORACLE_SEEDS = [42, 123, 456, 789, 1024]

def build_cfg(config_file, opts):
    cfg = base_cfg.clone(); cfg.defrost(); cfg.merge_from_file(config_file)
    if opts and opts[0] == "--": opts = opts[1:]
    cfg.merge_from_list(opts); cfg.freeze(); return cfg

@torch.no_grad()
def per_sample_loss(model, loader, mode):
    set_parasitic_mode(model, mode == "full"); model.eval(); L = []
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
    p.add_argument("--dr_k", type=int, default=4)
    p.add_argument("--mome_seeds", default="42,123,456,789,1024")
    p.add_argument("--oracle_seeds", default="42,123,456,789,1024")
    p.add_argument("--outdir", default="logs/lira",
                   help="Where losses and the membership mask go. Must differ per protocol: "
                        "the collector skips existing loss files, so sharing a directory "
                        "silently reuses another protocol's losses.")
    p.add_argument("--mome_tpl", default="logs/mome_v3_seed{seed}/mome_v3_epoch_120.pth")
    p.add_argument("--oracle_tpl", default="logs/rebuttal_strong_retrain_seed{seed}/transformer_120.pth")
    p.add_argument("--original_ckpt", default="logs/market_standard_baseline/transformer_120.pth")
    p.add_argument("opts", nargs=argparse.REMAINDER)
    a = p.parse_args()
    cfg = build_cfg(a.config_file, a.opts)
    global MOME_SEEDS, ORACLE_SEEDS
    MOME_SEEDS = [int(x) for x in a.mome_seeds.split(',') if x.strip()]
    ORACLE_SEEDS = [int(x) for x in a.oracle_seeds.split(',') if x.strip()]
    outdir = a.outdir if osp.isabs(a.outdir) else osp.join(ROOT, a.outdir)
    os.makedirs(outdir, exist_ok=True)

    # Fixed D_f ordering (val transforms). Use make_dataloader_mome once to get the full train list + num classes.
    out = make_dataloader_mome(cfg)
    nc, cn, vn = out[6], out[7], out[8]
    # full D_f train list: the underlying dataset of the full train loader (index 0)
    full_src = out[0].dataset
    items = full_src.dataset if hasattr(full_src, "dataset") else full_src
    vtf = T.Compose([T.Resize(cfg.INPUT.SIZE_TEST), T.ToTensor(),
                     T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)])
    df_loader = DataLoader(ImageDataset(items, vtf), batch_size=cfg.TEST.IMS_PER_BATCH,
                           shuffle=False, num_workers=4, collate_fn=mia_collate_fn)
    manifest = [{"idx": i, "path": it[0], "pid": int(it[1])} for i, it in enumerate(items)]
    json.dump(manifest, open(osp.join(outdir, "manifest.json"), "w"))
    N = len(items); print(f"D_f size N={N}, num_classes={nc}")

    # D_r membership per MoME seed. Under sub_id every shadow deletes the SAME identities,
    # fixed by MOME.SPLIT_SEED, so the mask is shared and each forgotten sample is OUT for
    # every shadow, which is what the rank test needs.
    dr_mem = np.zeros((len(MOME_SEEDS), N), dtype=bool)
    sub_id = cfg.MOME.FORGET_MODE == 'sub_id'
    if sub_id:
        # the split is fixed by SPLIT_SEED, independent of the model seeds, so compute it once;
        # this also covers --mome_seeds "" (oracle-family-only collects), where the per-seed
        # loop below never runs and df_only_idx would otherwise be unbound
        _ss = cfg.MOME.SPLIT_SEED if cfg.MOME.SPLIT_SEED >= 0 else (MOME_SEEDS + ORACLE_SEEDS)[0]
        _, df_only_idx = _split_by_identity(
            items, cfg.MOME.FORGET_ID_FRAC, _ss,
            leak_id_frac=getattr(cfg.MOME, 'LEAK_ID_FRAC', 0.0),
            dr_k=int(getattr(cfg.MOME, 'DR_K', 0)))
    for si, s in enumerate(MOME_SEEDS):
        if sub_id:
            split_seed = cfg.MOME.SPLIT_SEED if cfg.MOME.SPLIT_SEED >= 0 else s
            dr_idx, df_only_idx = _split_by_identity(
                items, cfg.MOME.FORGET_ID_FRAC, split_seed,
                leak_id_frac=getattr(cfg.MOME, 'LEAK_ID_FRAC', 0.0),
                dr_k=int(getattr(cfg.MOME, 'DR_K', 0)))
        else:
            dr_idx, _ = _split_dr_and_df_minus_r(items, a.dr_k, s)
        dr_mem[si, list(dr_idx)] = True
    np.save(osp.join(outdir, "dr_membership.npy"), dr_mem)
    if sub_id:
        u_mask = np.zeros(N, dtype=bool)
        u_mask[list(df_only_idx)] = True
        np.save(osp.join(outdir, "u_mask.npy"), u_mask)
        print("U mask saved:", int(u_mask.sum()), "images of deleted identities")
    print("D_r membership matrix saved:", dr_mem.shape, "mean IN frac", dr_mem.mean())

    def run(tag, ckpt, mode):
        fp = osp.join(outdir, f"losses_{tag}.npy")
        if osp.exists(fp): print(f"have {tag}"); return
        if not osp.exists(ckpt): print(f"MISSING ckpt for {tag}: {ckpt}"); return
        # MoME cores carry the rank-r parasitic branch while plain retrains do not, and a LoRA
        # build renames every wrapped Linear to ...base.weight, so a mismatch either way raises
        # KeyError on the first block. Take it from the checkpoint's own keys.
        _cfg = build_cfg_for_ckpt(cfg, ckpt)
        m = make_model(_cfg, num_class=nc, camera_num=cn, view_num=vn)
        m.load_param(ckpt); m = m.cuda().eval()
        l = per_sample_loss(m, df_loader, mode)
        np.save(fp, l); print(f"[OK] {tag}: N={len(l)} mean_loss={l.mean():.3f}")
        del m; torch.cuda.empty_cache()

    for s in MOME_SEEDS:
        run(f"mome_s{s}", a.mome_tpl.format(seed=s), "core")
    for s in ORACLE_SEEDS:
        run(f"oracle_s{s}", a.oracle_tpl.format(seed=s), "core")
    # a memorizing positive control
    run("original", a.original_ckpt, "standard")
    print("COLLECT DONE")

if __name__ == "__main__":
    main()
