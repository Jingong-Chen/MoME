"""Per-identity Rank-1 on U for one checkpoint.

The paper says a retrained model still ranks another image of the deleted identities first in
90.1% of Market-1501 queries. That is the aggregate. A figure that names one identity needs that
identity's own number, so this prints both, using the same U construction and the same
same-pid-same-camera exclusion as eval_partition.py.

  python scripts/cvpr2027/perid_rank1.py --weight logs/kmix_market_retrain_s42/transformer_120.pth \
      --config_file configs/Market/vit_transreid_stride_mome_v3.yml -- <yacs overrides>
"""
import argparse
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T

from config import cfg
from datasets.bases import ImageDataset
from datasets.market1501 import Market1501
from datasets.veri import VeRi
from datasets.regdb import RegDB
from datasets.make_dataloader import val_collate_fn
from datasets.make_dataloader_mome import _split_by_identity
from model import make_model
from utils.parasitic_utils import build_cfg_for_ckpt

__factory = {"market1501": Market1501, "veri": VeRi, "veri776": VeRi, "regdb": RegDB}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_file", required=True)
    ap.add_argument("--weight", required=True)
    ap.add_argument("--ids", default="", help="comma separated raw dataset ids to single out")
    ap.add_argument("--out", default="", help="write the readings to this json")
    ap.add_argument("opts", nargs=argparse.REMAINDER, default=None)
    a = ap.parse_args()
    cfg.merge_from_file(a.config_file)
    cfg.merge_from_list([o for o in (a.opts or []) if o != "--"])
    cfg.freeze()
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

    ds = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    seed = cfg.MOME.SPLIT_SEED if cfg.MOME.SPLIT_SEED >= 0 else cfg.SOLVER.SEED
    _, u_idx = _split_by_identity(ds.train, cfg.MOME.FORGET_ID_FRAC, seed,
                                  dr_k=int(cfg.MOME.DR_K))
    u = [ds.train[i] for i in u_idx]
    print("split seed %d   |U| = %d images, %d identities"
          % (seed, len(u), len({t[1] for t in u})))

    tf = T.Compose([T.Resize(cfg.INPUT.SIZE_TEST), T.ToTensor(),
                    T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)])
    loader = DataLoader(ImageDataset(u, tf), batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False,
                        num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=val_collate_fn)

    _c = build_cfg_for_ckpt(cfg, a.weight)
    model = make_model(_c, num_class=ds.num_train_pids, camera_num=ds.num_train_cams,
                       view_num=ds.num_train_vids)
    model.load_param(a.weight)
    model = model.cuda().eval()

    feats, pids, camids = [], [], []
    with torch.no_grad():
        for img, pid, camid, camid_t, viewid, _ in loader:
            f = model(img.cuda(), cam_label=camid_t.cuda(), view_label=viewid.cuda())
            feats.append(f.cpu())
            pids.extend(np.asarray(pid))
            camids.extend(np.asarray(camid))
    f = torch.cat(feats, 0)
    if cfg.TEST.FEAT_NORM == 'yes':
        f = F.normalize(f, dim=1, p=2)
    pids = np.asarray(pids)
    camids = np.asarray(camids)

    # Query and gallery are the same set, exactly as eval_partition builds it.
    d = (2 - 2 * (f @ f.t())).numpy()
    order = np.argsort(d, axis=1)
    hit = np.zeros(len(pids), dtype=bool)
    for i in range(len(pids)):
        o = order[i]
        keep = ~((pids[o] == pids[i]) & (camids[o] == camids[i]))
        o = o[keep]
        if len(o):
            hit[i] = pids[o[0]] == pids[i]

    print("aggregate Rank-1 on U: %.1f%%  (%d of %d queries)"
          % (100.0 * hit.mean(), hit.sum(), len(hit)))

    per = defaultdict(list)
    for h, p in zip(hit, pids):
        per[p].append(h)
    # relabeled pid back to the raw dataset id
    import glob, re
    root = os.path.join(cfg.DATASETS.ROOT_DIR, 'market1501',
                        'Market-1501-v15.09.15', 'bounding_box_train')
    pat = re.compile(r'([-\d]+)_c(\d)')
    container = set()
    for fp in sorted(glob.glob(os.path.join(root, '*.jpg'))):
        raw, _ = map(int, pat.search(os.path.basename(fp)).groups())
        if raw != -1:
            container.add(raw)
    lab2raw = {lab: raw for lab, raw in enumerate(container)}

    want = [int(x) for x in a.ids.split(",") if x.strip()]
    raw2lab = {v: k for k, v in lab2raw.items()}
    for raw in want:
        lab = raw2lab.get(raw)
        if lab is None or lab not in per:
            print("  raw ID %d is not in U under this split" % raw)
            continue
        v = per[lab]
        print("  raw ID %-6d Rank-1 %.1f%%  (%d of %d of its images)"
              % (raw, 100.0 * np.mean(v), int(np.sum(v)), len(v)))

    rates = np.array([np.mean(v) for v in per.values()])
    print("per-identity Rank-1: median %.1f%%, %d of %d identities at 100%%, min %.1f%%"
          % (100 * np.median(rates), int((rates == 1).sum()), len(rates), 100 * rates.min()))

    if a.out:
        import json
        payload = {
            "weight": a.weight, "split_seed": int(seed),
            "u_images": int(len(pids)), "u_identities": int(len(per)),
            "aggregate_rank1_pct": round(100.0 * float(hit.mean()), 4),
            "per_identity_median_pct": round(100.0 * float(np.median(rates)), 4),
            "identities_at_100pct": int((rates == 1).sum()),
            "per_identity": {str(lab2raw[k]): {
                "rank1_pct": round(100.0 * float(np.mean(v)), 4),
                "hits": int(np.sum(v)), "images": len(v)} for k, v in per.items()},
        }
        with open(a.out, "w") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
