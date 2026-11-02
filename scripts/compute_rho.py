#!/usr/bin/env python3
"""Compute the Forgetting-Rank statistic rho_U from collected per-sample losses.
Validated rank test (same as used for Market/RegDB): for each forgotten x in U(t), rank the target's
loss among {target} + {same-class shadows that also forgot x}; average normalized rank over U(t).
E[rho_U]=1/2 under exchangeability (no leak). Usage:
  python compute_rho.py --dir logs/lira_veri --mome 123,456,789,1024 --oracle 42,123,456,789,1024
Optionally computes Full-mode power if losses_momefull_s*.npy exist (target=Full vs Core-shadow refs)."""
import argparse, os.path as osp
import numpy as np
from scipy import stats

def mean_rank(losses, drm, umask=None):
    """LOO rank test within one model class. losses:[K,N], drm:[K,N] (True=in D_r=retained).

    umask, when given, restricts scoring to the true forget set U. Without it the scored set is
    the complement of D_r, which under identity-level deletion is 43% held-out images of RETAINED
    identities. Those are not forgotten, and ranking them measures generalisation, not removal.
    """
    K, N = losses.shape; per = []
    for t in range(K):
        oth = [j for j in range(K) if j != t]; r = []
        for x in range(N):
            if umask is not None and not (umask[t, x] if umask.ndim == 2 else umask[x]): continue
            if drm[t, x]: continue                                    # only x forgotten by target t
            grp = [losses[t, x]] + [losses[j, x] for j in oth if not drm[j, x]]  # + shadows that also forgot x
            if len(grp) >= 3:
                a = np.array(grp)
                r.append((np.sum(a < losses[t, x]) + 0.5 * np.sum(a == losses[t, x])) / len(a))
        if r: per.append(np.mean(r))
    return np.array(per)

def power_rank(full, core, drm, umask=None):
    """target=Full loss vs reference group = other shadows' CORE loss (true forgetters)."""
    K, N = full.shape; per = []
    for t in range(K):
        oth = [j for j in range(K) if j != t]; r = []
        for x in range(N):
            if umask is not None and not (umask[t, x] if umask.ndim == 2 else umask[x]): continue
            if drm[t, x]: continue
            grp = [full[t, x]] + [core[j, x] for j in oth if not drm[j, x]]
            if len(grp) >= 3:
                a = np.array(grp)
                r.append((np.sum(a < full[t, x]) + 0.5 * np.sum(a == full[t, x])) / len(a))
        if r: per.append(np.mean(r))
    return np.array(per)

def load(d, tag, seeds):
    return np.stack([np.load(f"{d}/losses_{tag}_s{s}.npy") for s in seeds])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--mome", required=True, help="comma seeds for mome shadows")
    ap.add_argument("--oracle", default="", help="comma seeds for oracle shadows")
    ap.add_argument("--dr", default="dr_membership.npy")
    ap.add_argument("--u_mask", default="u_mask.npy", help="restrict scoring to the true forget set")
    ap.add_argument("--dr_oracle", default="dr_membership_oracle.npy")
    ap.add_argument("--mome_tag", default="mome", help="loss-file prefix for MoME shadows")
    ap.add_argument("--oracle_tag", default="oracle", help="loss-file prefix for retrain/oracle shadows")
    ap.add_argument("--full_tag", default="momefull", help="loss-file prefix for Full-mode power control")
    a = ap.parse_args()
    ms = [int(x) for x in a.mome.split(",")]
    dr = np.load(osp.join(a.dir, a.dr))
    # u_mask marks the true forget set. It was being written by the collectors and read by
    # nothing, so every rho scored the complement of D_r instead of U.
    up = osp.join(a.dir, a.u_mask)
    umask = np.load(up) if osp.exists(up) else None
    print(f"u_mask: {'loaded ' + up + f', |U|={int(umask.sum())}' if umask is not None else 'ABSENT, scoring the complement of D_r'}")
    mome = load(a.dir, a.mome_tag, ms)
    # dr rows are ordered by the collector's MOME_SEEDS; assume same order as --mome
    rm = mean_rank(mome, dr[:len(ms)], umask)
    tm = stats.ttest_1samp(rm, 0.5)
    print(f"MoME   rho_U = {rm.mean():.4f} ± {rm.std():.4f}  per-seed {np.round(rm,3)}  vs .5: p={tm.pvalue:.3f} ({'no leak' if tm.pvalue>0.05 else 'LEAK'})")
    if a.oracle:
        os_ = [int(x) for x in a.oracle.split(",")]
        oracle = load(a.dir, a.oracle_tag, os_)
        dro = np.load(osp.join(a.dir, a.dr_oracle)) if osp.exists(osp.join(a.dir, a.dr_oracle)) else dr
        ro = mean_rank(oracle, dro[:len(os_)], umask)
        to = stats.ttest_1samp(ro, 0.5); tmo = stats.ttest_ind(rm, ro)
        print(f"Oracle rho_U = {ro.mean():.4f} ± {ro.std():.4f}  per-seed {np.round(ro,3)}  vs .5: p={to.pvalue:.3f}")
        print(f"MoME vs Oracle: p={tmo.pvalue:.3f} ({'indistinguishable' if tmo.pvalue>0.05 else 'differ'})")
    ff = osp.join(a.dir, f"losses_{a.full_tag}_s{ms[0]}.npy")
    if osp.exists(ff):
        full = load(a.dir, a.full_tag, ms)
        rf = power_rank(full, mome, dr[:len(ms)], umask); tf = stats.ttest_1samp(rf, 0.5)
        print(f"POWER Full rho_U = {rf.mean():.4f} ± {rf.std():.4f}  {np.round(rf,3)}  vs .5: p={tf.pvalue:.2e}")
