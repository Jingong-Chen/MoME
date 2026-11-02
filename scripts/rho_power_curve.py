"""Detection-power curve for the Forgetting-Rank certificate.

Ranks each leak model (trained with MOME.LEAK_ID_FRAC of the requested identities silently
kept in the core update) within the clean 5-seed MoME family, from per-sample losses already
collected by collect_lira_losses.py. Two readings:

  whole-U rho_U   -- the blind audit number; dilution-limited (detects >=5% leaks at ~2.6 sigma)
  identity AUC    -- per-identity mean rho as the score, leaked-vs-clean ROC; near-perfect at
                     every fraction because a leaked identity's images pin at the statistic's
                     floor (0.5/(R+1) = 0.083 with R=5: every shadow beats the target).

Setting of the stored artifacts: Market, FORGET_ID_FRAC 0.9 era, LoRA family, split seed 42.
The masks in logs/leak*_lira were collected with LEAK_ID_FRAC 0 on purpose: the audit question
is the ORIGINAL request, so U covers all requested identities including the leaked ones.
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
S_all = np.stack(list(fam.values()))


def rho_per_sample(t):
    return ((S_all < t[None, :]).sum(0) + 0.5) / (S_all.shape[0] + 1)


# null scale: leave-one-out within the clean family (family MEAN is 1/2 by construction;
# the informative quantities are the per-member values and their spread)
loo = []
for s in SEEDS:
    S4 = np.stack([fam[o] for o in SEEDS if o != s])
    loo.append(float((((S4 < fam[s][None, :]).sum(0) + 0.5) / 5.0)[U].mean()))
print("clean LOO rho_U:", " ".join(f"{v:.4f}" for v in loo), f" spread {np.std(loo):.4f}")

print("\nfrac  ids  whole-U rho   sigma    identity AUC   clean ids at floor")
for tag, frac in LEAKS:
    _, _, leaked_pids, forget_pids = _split_by_identity(
        ds.train, 0.9, 42, leak_id_frac=frac, return_leaked=True)
    t = np.load(f"logs/{tag}_lira/losses_mome_s42.npy")
    rs = rho_per_sample(t)
    whole = rs[U].mean()
    leaked_set = set(leaked_pids)
    labels, scores = [], []
    for p in sorted(set(forget_pids)):
        labels.append(1 if p in leaked_set else 0)
        scores.append(-rs[(man_pid == p) & U].mean())
    auc = roc_auc_score(labels, scores)
    clean_rhos = np.array([-sc for sc, lb in zip(scores, labels) if lb == 0])
    floor = (clean_rhos <= 0.5 / len(SEEDS) - 1e-4 + 0.5 / (len(SEEDS) + 1)).mean()
    print(f"{frac:.0%}  {len(leaked_set):4d}   {whole:.4f}    {(whole-0.5)/np.std(loo):+6.1f}"
          f"    {auc:.4f}         {floor:.1%}")
