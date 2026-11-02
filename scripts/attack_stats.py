"""Identity-clustered intervals, equivalence tests and low-FPR diagnostics for the attacks.

An attack score is reported per image, but images of one identity are not independent: they
share a subject, a set of cameras and a difficulty. Treating them as independent understates
the uncertainty of an AUC. Every interval here resamples identities, not images.

"Indistinguishable from a retrain" is also not established by two numbers being close or by
error bars overlapping. This script runs the equivalence test that claim requires, with the
margin fixed in advance by the spread of the reference family itself, so the null being tested
is a difference larger than what separates one retrain from another.

Usage:
  python scripts/cvpr2027/attack_stats.py --target logs/xref_market/scores_mome_core_s42.npz \
      --reference logs/xref_market/scores_retrain.npz \
      --ref_family logs/xref_market/scores_retrain.npz ... --attack rmia_lf --out stats.json
"""
import argparse
import json

import numpy as np
from sklearn.metrics import roc_auc_score


def clustered_auc_boot(score, y, cluster, n_boot=2000, seed=0):
    """Bootstrap the AUC by resampling clusters. Returns the point estimate and a 95% interval."""
    rng = np.random.default_rng(seed)
    uc = np.unique(cluster)
    idx_by_c = {c: np.flatnonzero(cluster == c) for c in uc}
    point = roc_auc_score(y, score)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uc, size=len(uc), replace=True)
        idx = np.concatenate([idx_by_c[c] for c in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], score[idx]))
    boots = np.asarray(boots)
    return float(point), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), boots


def tost(diff_boots, margin):
    """Two one-sided tests on a bootstrap distribution of the difference.

    Equivalence is declared when the whole 90% interval of the difference lies inside the
    margin, which is the interval form of TOST at the 5% level.
    """
    lo, hi = np.percentile(diff_boots, [5, 95])
    return {
        "diff_mean": float(diff_boots.mean()),
        "ci90_low": float(lo), "ci90_high": float(hi),
        "margin": float(margin),
        "equivalent": bool(lo > -margin and hi < margin),
    }


def low_fpr(score, y, fpr_target):
    """TPR at a target FPR, with the counts behind it and a Clopper-Pearson interval."""
    from scipy.stats import beta
    neg = score[y == 0]
    pos = score[y == 1]
    n_neg, n_pos = len(neg), len(pos)
    k_fp = max(int(np.floor(fpr_target * n_neg)), 0)
    if k_fp == 0:
        # No threshold can realize this FPR with this many negatives. Say so instead of
        # reporting the number the interpolation would produce.
        thr = np.max(neg) + 1e-12
        realized_fp = 0
    else:
        thr = np.sort(neg)[::-1][k_fp - 1]
        realized_fp = int((neg >= thr).sum())
    tp = int((pos >= thr).sum())
    lo = 0.0 if tp == 0 else float(beta.ppf(0.025, tp, n_pos - tp + 1))
    hi = 1.0 if tp == n_pos else float(beta.ppf(0.975, tp + 1, n_pos - tp))
    return {
        "fpr_target": fpr_target, "n_neg": n_neg, "n_pos": n_pos,
        "false_positives_allowed": k_fp, "false_positives_realized": realized_fp,
        "true_positives": tp, "tpr": tp / n_pos,
        "tpr_ci95_low": lo, "tpr_ci95_high": hi,
        "realizable": k_fp > 0,
    }


def load(path, attack):
    d = np.load(path, allow_pickle=True)
    if attack not in d:
        raise KeyError("%s has no attack %r, it has %s" % (path, attack, [k for k in d.files]))
    # Namespace the identity labels: the member pool and the query pool carry independent
    # label spaces, and an identity in one is not the identity with the same number in the other.
    cluster = np.array(["%d_%d" % (p, i) for p, i in zip(d["pool"], d["pid"])])
    return d[attack].astype(np.float64), d["y_true"].astype(int), cluster


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--reference", required=True, help="the retrained reference row")
    ap.add_argument("--ref_family", nargs="*", default=[],
                    help="leave-one-out members of the reference family; their spread fixes the "
                         "equivalence margin")
    ap.add_argument("--attack", default="rmia_lf")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--margin", type=float, default=None,
                    help="override the equivalence margin instead of reading it from the family")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    st, yt, ct = load(a.target, a.attack)
    sr, yr, cr = load(a.reference, a.attack)
    auc_t, lo_t, hi_t, bt = clustered_auc_boot(st, yt, ct, a.n_boot, seed=0)
    auc_r, lo_r, hi_r, br = clustered_auc_boot(sr, yr, cr, a.n_boot, seed=0)

    if a.margin is not None:
        margin = a.margin
    elif len(a.ref_family) >= 2:
        fam = []
        for p in a.ref_family:
            s_, y_, _ = load(p, a.attack)
            fam.append(roc_auc_score(y_, s_))
        margin = 2.0 * float(np.std(fam, ddof=1))
    else:
        margin = 0.02

    res = {
        "attack": a.attack,
        "target": {"path": a.target, "auc": auc_t, "ci95": [lo_t, hi_t]},
        "reference": {"path": a.reference, "auc": auc_r, "ci95": [lo_r, hi_r]},
        "equivalence": tost(bt - br, margin),
        "margin_source": ("given" if a.margin is not None else
                          ("two standard deviations of the reference family, n=%d" % len(a.ref_family)
                           if len(a.ref_family) >= 2 else "default 0.02")),
        "low_fpr": {str(f): low_fpr(st, yt, f) for f in (0.01, 0.001, 0.0001)},
        "low_fpr_reference": {str(f): low_fpr(sr, yr, f) for f in (0.01, 0.001, 0.0001)},
        "n_clusters_target": int(len(np.unique(ct))),
    }
    print(json.dumps(res, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
