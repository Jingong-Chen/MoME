"""RegDB kmix harvest: utility in both directions plus cross-modal U retrieval per row.

Reuses eval_once from the 10-trial evaluator for query/gallery utility, and adds a U-retrieval
evaluation (query = U visible images, gallery = U thermal images) under the combined protocol
split. Attack columns come from the existing wrapper tools; this script covers the retrieval
numbers the table needs.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch.utils.data import DataLoader

from config import cfg as _cfg_singleton  # noqa: F401  (imported for side effects some modules expect)
from datasets.regdb_cross import RegDBCross
from datasets.bases import ImageDataset
from datasets.make_dataloader import val_collate_fn
from datasets.make_dataloader_mome import _split_by_identity
import torchvision.transforms as T
from model import make_model
from utils.metrics import R1_mAP_eval
from utils.parasitic_utils import set_parasitic_mode

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts"))
from eval_regdb_cross_10trial import build_cfg, eval_once  # noqa: E402


def u_eval(cfg, ckpt, trial, mode, arch, loft=""):
    """Cross-modal retrieval restricted to the deleted identities: query U-visible against
    gallery U-thermal. Reads how much of the deleted identities the model still retrieves."""
    ds = RegDBCross(root=cfg.DATASETS.ROOT_DIR, trial=trial, test_mode="v2t", verbose=False)
    dr_idx, u_idx = _split_by_identity(
        ds.train, cfg.MOME.FORGET_ID_FRAC, cfg.MOME.SPLIT_SEED,
        dr_k=int(getattr(cfg.MOME, "DR_K", 0)),
        dr_k_stratify_cam=bool(getattr(cfg.MOME, "DR_K_STRATIFY_CAM", False)))
    u = [ds.train[i] for i in u_idx]
    q = [t for t in u if t[2] == 0]   # visible
    g = [t for t in u if t[2] == 1]   # thermal
    tf = T.Compose([T.Resize(cfg.INPUT.SIZE_TEST), T.ToTensor(),
                    T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)])
    loader = DataLoader(ImageDataset(q + g, tf), batch_size=cfg.TEST.IMS_PER_BATCH,
                        shuffle=False, num_workers=4, collate_fn=val_collate_fn)
    model = make_model(cfg, num_class=206, camera_num=2, view_num=1)
    model.load_param(ckpt)
    model = model.cuda().eval()
    set_parasitic_mode(model, mode == "full")
    if loft:
        from eval_loft import LoftWrapper
        payload = torch.load(loft, map_location="cpu")
        U = payload["U"] if isinstance(payload, dict) and "U" in payload else payload
        model = LoftWrapper(model, (U @ U.t()).float().cuda()).cuda().eval()
    ev = R1_mAP_eval(len(q), max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    ev.reset()
    with torch.no_grad():
        for img, vid, camid, camids, tv, _ in loader:
            feat = model(img.cuda(), cam_label=camids.cuda(), view_label=tv.cuda())
            ev.update((feat, vid, camid))
    _, mAP, _, _, _, _, _ = ev.compute()
    del model
    torch.cuda.empty_cache()
    return mAP * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True,
                    help="semicolon list of tag,ckpt,mode,arch  (arch: lora|plain)")
    ap.add_argument("--config_file", default="configs/RegDB/vit_transreid_stride_cross_mome_v3.yml")
    ap.add_argument("--trial", type=int, default=1)
    ap.add_argument("--out", default="logs/rgk_harvest.json")
    ap.add_argument("--loft", default="", help="LOFT projector loft_U.pt; retrieval only")
    ap.add_argument("opts", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if a.opts and a.opts[0] == "--":
        a.opts = a.opts[1:]

    results = {}
    if os.path.exists(a.out):
        results = json.load(open(a.out))
    for row in a.rows.split(";"):
        tag, ckpt, mode, arch = [x.strip() for x in row.split(",")]
        if tag in results:
            print(f"[have] {tag}")
            continue
        cfg = build_cfg(a.config_file, os.environ.get("MOME_DATA", os.environ.get("DATA_ROOT", "./data")), "0")
        cfg.defrost()
        cfg.MODEL.SIE_CAMERA = False
        cfg.MODEL.PARASITIC_LINEAR = (arch in ("lora", "fullcap"))
        cfg.MODEL.PARASITIC_LORA_RANK = 8
        cfg.MODEL.PARASITIC_LORA_ALPHA = 16
        if arch == "fullcap":
            cfg.MODEL.PARASITIC_LINEAR_MODE = "full"
            cfg.MODEL.PARASITIC_EMBED = True
            cfg.MODEL.PARASITIC_HEAD_ON = True
            cfg.MODEL.PARASITIC_BN_STATS = True
        cfg.MOME.FORGET_MODE = "sub_id"
        cfg.MOME.FORGET_ID_FRAC = 0.5
        cfg.MOME.SPLIT_SEED = 42
        cfg.MOME.DR_K = 4
        cfg.MOME.DR_K_STRATIFY_CAM = True
        if a.opts:
            cfg.merge_from_list(a.opts)
        cfg.freeze()
        v2t = eval_once(cfg, ckpt, a.trial, "v2t", mode)
        t2v = eval_once(cfg, ckpt, a.trial, "t2v", mode)
        umap = u_eval(cfg, ckpt, a.trial, mode, arch, a.loft)
        results[tag] = {"v2t": v2t, "t2v": t2v, "U_mAP": umap}
        print(f"[row] {tag} v2t={v2t} t2v={t2v} U={umap:.2f}", flush=True)
        json.dump(results, open(a.out, "w"), indent=1)
    print("HARVEST_DONE")


if __name__ == "__main__":
    main()
