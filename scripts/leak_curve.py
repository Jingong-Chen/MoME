"""The violation curve: what the attacks see on a model that silently kept part of the request.

The audit controls were only ever read with the rank statistic. This reads the same models with
the three membership attacks the paper runs, against the same four disjoint references, so the two
statistics are compared on identical models rather than across studies.

Point estimates are sample level, which is what the tables report; the intervals resample
identities, since images of one identity are not independent. Both come from attack_stats.py so
the definition matches the rest of the paper.

Usage: python scripts/cvpr2027/leak_curve.py [--attack rmia_lf] [--n_boot 2000]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attack_stats import clustered_auc_boot, load, low_fpr  # noqa: E402

OUT = "logs/xref_leak"
ROWS = [("00", 0.00), ("02", 0.02), ("05", 0.05), ("10", 0.10), ("25", 0.25)]

ap = argparse.ArgumentParser()
ap.add_argument("--attack", default="rmia_lf")
ap.add_argument("--n_boot", type=int, default=2000)
ap.add_argument("--out", default=None)
a = ap.parse_args()

res = {}
print("violation   n_img   %s AUC        95%% CI (identity clustered)   TPR@1%%   TPR@0.1%%" % a.attack)
for tag, frac in ROWS:
    p = "%s/scores_leak%s.npz" % (OUT, tag)
    if not os.path.exists(p):
        print("%5.0f%%      pending" % (frac * 100))
        continue
    s, y, c = load(p, a.attack)
    auc, lo, hi, _ = clustered_auc_boot(s, y, c, a.n_boot, seed=0)
    f1 = low_fpr(s, y, 0.01)
    f2 = low_fpr(s, y, 0.001)
    print("%5.0f%%     %6d   %.4f        [%.4f, %.4f]            %.4f   %.4f"
          % (frac * 100, len(y), auc, lo, hi, f1["tpr"], f2["tpr"]))
    res["%.2f" % frac] = {"auc": auc, "ci95": [lo, hi], "n": int(len(y)),
                          "n_clusters": int(len(np.unique(c))),
                          "tpr_at_0.01": f1["tpr"], "tpr_at_0.001": f2["tpr"]}
if a.out:
    json.dump(res, open(a.out, "w"), indent=2)
    print("\nwrote", a.out)
