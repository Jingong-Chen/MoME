"""How much of a deletion request the audit needs before it can say anything.

The request in the main protocol names 376 identities. A real erasure request names one. Retraining
at every size is out of reach, but one half of the question is answerable from the scores already
dumped: holding the model fixed, restrict the audit to a random subset of the requested identities
and measure what the reading and its interval do as the subset shrinks. That isolates the
statistic's resolution from the model's behaviour, and it is the half a controller needs, since it
says how many identities a request must name before a null reading means anything.
"""
import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score


def clustered_ci(score, y, cluster, n_boot=600, seed=0):
    rng = np.random.default_rng(seed)
    uc = np.unique(cluster)
    idx = {c: np.flatnonzero(cluster == c) for c in uc}
    b = []
    for _ in range(n_boot):
        pick = rng.choice(uc, size=len(uc), replace=True)
        j = np.concatenate([idx[c] for c in pick])
        if len(np.unique(y[j])) < 2:
            continue
        b.append(roc_auc_score(y[j], score[j]))
    b = np.asarray(b)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def run(path, attack, sizes, reps=20, seed=0):
    d = np.load(path)
    s, y, pid = d[attack], d["y_true"], d["pid"]
    mem = np.unique(pid[y == 1])
    rng = np.random.default_rng(seed)
    out = []
    for n in sizes:
        aucs, widths = [], []
        for r in range(reps if n < len(mem) else 1):
            keep = rng.choice(mem, size=n, replace=False) if n < len(mem) else mem
            m = (y == 0) | np.isin(pid, keep)
            lo, hi = clustered_ci(s[m], y[m], pid[m], seed=r)
            aucs.append(roc_auc_score(y[m], s[m]))
            widths.append((hi - lo) / 2)
        out.append((n, float(np.mean(aucs)), float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                    float(np.mean(widths))))
    return out


if __name__ == "__main__":
    sizes = [5, 10, 25, 50, 100, 200, 376]
    for tag, path in [("MoME-Core", "logs/xref_market/scores_mome_core_s42.npz"),
                      ("Retrain", "logs/xref_market/scores_retrain.npz"),
                      ("Full payload", "logs/xref_market/scores_mome_full.npz")]:
        print("== %s, RMIA on Market-1501 ==" % tag)
        print("  %-5s %-8s %-8s %-8s" % ("ids", "AUC", "sd", "half-width"))
        for n, a, sd, w in run(path, "rmia_lf", sizes):
            print("  %-5d %-8.4f %-8.4f %-8.4f" % (n, a, sd, w))
