"""A test with a rejection region for the Forgetting-Rank, with the identity as the unit.

The aggregate rho_U over a leave-one-out family is a counting identity. With K shadows and no
ties, the K ranks a sample receives as each model takes its turn as target are a permutation of
(2i-1)/(2K), so their mean is exactly 1/2 whatever the family did. Reading 1/2 back off that
average certifies nothing, and this script does not use it.

What the family does carry is localization. If a model kept identity i in its core update, every
image of i sits at the floor of the rank for that model and nowhere else. The unit of the test is
therefore the pair of a model and an identity, and the statistic is the per-identity mean rank
standardized by the number of images that identity contributes,

    z[t,i] = (1/2 - m[t,i]) * sqrt(n_i) / sd_1,   sd_1 = sqrt((K^2-1)/(12 K^2)),

so that a small identity at the floor and a large identity slightly below the middle are placed
on one scale. Without the standardization the minimum over cells is at the floor under the null
as well, and the test has no power.

The null is the exchangeability the proposition predicts: for each image, the K ranks are
assigned to the K models at random. Permuting the model axis independently per image preserves
every marginal and destroys only the coherence a violation creates. The test is one-sided and
family-wise: the observed statistic is max over cells of z, and the threshold is the upper alpha
quantile of that maximum under the null, so the chance of flagging any identity of any model in
a clean family is alpha. Reporting the same quantity for a family with a planted violation gives
the power.
"""
import argparse
import json
import os.path as osp

import numpy as np


def loo_ranks(losses, drm, umask):
    """rank[t, x] in (0, 1): model t's loss on x ranked among the models that also forgot x.
    NaN where t retained x, or where fewer than three models forgot it."""
    K, N = losses.shape
    out = np.full((K, N), np.nan, dtype=np.float64)
    cols = np.flatnonzero(umask) if umask is not None else np.arange(N)
    for x in cols:
        forgot = np.flatnonzero(~drm[:, x])
        if len(forgot) < 3:
            continue
        v = losses[forgot, x]
        out[forgot, x] = (
            (v[:, None] > v[None, :]).sum(1) + 0.5 * (v[:, None] == v[None, :]).sum(1)
        ) / len(v)
    return out


def _groups(rank, pid, min_images):
    """Index groups per identity, over identities every model scores at least min_images times."""
    scored = ~np.isnan(rank)
    ids, out = [], []
    for i in np.unique(pid[scored.any(0)]):
        sel = np.flatnonzero(pid == i)
        if scored[:, sel].sum(1).min() >= min_images:
            ids.append(i)
            out.append(sel)
    return out, np.asarray(ids)


def zmax(rank, groups, sd1):
    """max over model and identity of the standardized deficit below the middle rank."""
    best = -np.inf
    for sel in groups:
        sub = rank[:, sel]
        n = np.sum(~np.isnan(sub), axis=1)
        m = np.nanmean(sub, axis=1)
        z = (0.5 - m) * np.sqrt(n) / sd1
        best = max(best, float(np.nanmax(z)))
    return best


def permute(rank, rng):
    r = rank.copy()
    for x in np.flatnonzero(~np.all(np.isnan(rank), axis=0)):
        ok = np.flatnonzero(~np.isnan(rank[:, x]))
        r[ok, x] = rank[ok, x][rng.permutation(len(ok))]
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tag", default="mome")
    ap.add_argument("--seeds", default="42,123,456,789,1024")
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--min_images", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",")]
    K = len(seeds)
    sd1 = np.sqrt((K * K - 1) / (12.0 * K * K))
    losses = np.stack([np.load(osp.join(a.dir, "losses_%s_s%d.npy" % (a.tag, s))) for s in seeds])
    drm = np.load(osp.join(a.dir, "dr_membership.npy"))[:K]
    up = osp.join(a.dir, "u_mask.npy")
    umask = np.load(up) if osp.exists(up) else None
    pid = np.array([m["pid"] for m in json.load(open(osp.join(a.dir, "manifest.json")))])

    rank = loo_ranks(losses, drm, umask)
    groups, ids = _groups(rank, pid, a.min_images)
    obs = zmax(rank, groups, sd1)

    rng = np.random.default_rng(0)
    null = np.array([zmax(permute(rank, rng), groups, sd1) for _ in range(a.n_perm)])
    thr = float(np.percentile(null, 100 * (1 - a.alpha)))
    p = float((np.sum(null >= obs - 1e-9) + 1) / (len(null) + 1))

    res = {
        "dir": a.dir, "tag": a.tag, "n_models": K,
        "n_identities": len(ids), "n_cells": K * len(ids),
        "aggregate_rho_U": float(np.nanmean(rank)),
        "aggregate_note": "a counting identity over a leave-one-out family, not evidence",
        "unit": "one model and one identity",
        "statistic": "max standardized deficit below the middle rank",
        "observed": obs, "threshold_at_alpha": thr, "alpha": a.alpha,
        "n_perm": a.n_perm, "p_value_familywise": p, "reject_null": bool(obs >= thr),
        "per_model_mean_rank": [float(np.nanmean(rank[t])) for t in range(K)],
    }
    print(json.dumps(res, indent=2))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
