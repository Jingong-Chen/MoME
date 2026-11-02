"""What the identity-level reading gives an auditor, in the form an auditor would use it.

tab::rhopower reports the localization as an ROC-AUC. An AUC is a ranking statistic and does not
say what an audit recovers. This reports the operational form: rank the requested identities by
their mean Forgetting-Rank, open as many as were kept, and count how many of those are the ones
the core silently kept. It also reports where the worst-hidden kept identity sits in that ranking.

Same artifacts and same split as scripts/cvpr2027/rho_power_curve.py, so the AUC column printed
here must reproduce the table.

Usage: python scripts/cvpr2027/leak_localize.py
"""
import os
import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from datasets.make_dataloader_mome import _split_by_identity  # noqa: E402
from datasets.market1501 import Market1501  # noqa: E402

FAMILY = "logs/lora_market_lira"
SEEDS = [42, 123, 456, 789, 1024]
LEAKS = [("leak002", 0.02), ("leak005", 0.05), ("leak010", 0.10), ("leak025", 0.25)]

ds = Market1501(root=os.environ.get("MOME_DATA", "./data"))
man_pid = np.array([m["pid"] for m in json.load(open(f"{FAMILY}/manifest.json"))])
fam = {s: np.load(f"{FAMILY}/losses_mome_s{s}.npy") for s in SEEDS}
U = ~np.load(f"{FAMILY}/dr_membership.npy")[0]
S = np.stack(list(fam.values()))

print("frac  kept  requested   AUC      opened=kept, found   worst kept at rank")
for tag, frac in LEAKS:
    _, _, leaked, forget = _split_by_identity(ds.train, 0.9, 42,
                                              leak_id_frac=frac, return_leaked=True)
    t = np.load(f"logs/{tag}_lira/losses_mome_s42.npy")
    rs = ((S < t[None, :]).sum(0) + 0.5) / (S.shape[0] + 1)
    pids = sorted(set(forget))
    kept = set(leaked)
    score = np.array([rs[(man_pid == p) & U].mean() for p in pids])   # lower is more suspicious
    label = np.array([1 if p in kept else 0 for p in pids])
    order = np.argsort(score, kind="stable")
    k = int(label.sum())
    found = int(label[order[:k]].sum())
    worst = int(np.max(np.where(label[order] == 1)[0])) + 1
    print("%3.0f%%  %4d  %9d   %.4f   %3d of %-4d        %4d"
          % (100 * frac, k, len(pids), roc_auc_score(label, -score), found, k, worst))
