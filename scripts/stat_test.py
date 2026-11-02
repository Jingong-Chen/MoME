#!/usr/bin/env python3
"""
MoME v3 multi-seed statistical test script.

Usage:
  python scripts/stat_test.py
  python scripts/stat_test.py --seeds 42,123,456
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy import stats
except ImportError:
    print("ERROR: scipy is required. Install with `conda install scipy` or `pip install scipy`.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="MoME v3 statistical validation")
    parser.add_argument("--seeds", default="42,123,456,789,1024", help="comma-separated seed list")
    parser.add_argument("--mome-prefix", default="logs/mome_v3_seed", help="MoME run dir prefix")
    parser.add_argument("--retrain-prefix", default="logs/retrain_seed", help="Retrain run dir prefix")
    parser.add_argument("--v11-core-map", type=float, default=52.2, help="MoME v1.1 core mAP (%)")
    parser.add_argument("--alpha", type=float, default=0.05, help="significance level")
    parser.add_argument("--output", default="logs/stat_tests_summary.json", help="summary json output")
    return parser.parse_args()


def parse_seed_list(seed_str: str) -> List[int]:
    return [int(s.strip()) for s in seed_str.split(",") if s.strip()]


def load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_percent(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    if v <= 1.5:
        return v * 100.0
    return v


def extract_retrain_test_metrics(log_path: str) -> Dict[str, Optional[float]]:
    out = {"mAP": None, "R1": None}
    if not os.path.exists(log_path):
        return out

    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    map_vals = re.findall(r"mAP:\s*([0-9]+(?:\.[0-9]+)?)%", text)
    r1_vals = re.findall(r"Rank-1\s*:?\s*([0-9]+(?:\.[0-9]+)?)%", text)

    if map_vals:
        out["mAP"] = float(map_vals[-1])
    if r1_vals:
        out["R1"] = float(r1_vals[-1])
    return out


def ci95(arr: np.ndarray) -> Tuple[float, float, float]:
    mean = float(np.mean(arr))
    if len(arr) < 2:
        return mean, mean, mean
    se = stats.sem(arr)
    lo, hi = stats.t.interval(0.95, len(arr) - 1, loc=mean, scale=se)
    return mean, float(lo), float(hi)


def fmt(v: Optional[float], nd: int = 3) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{nd}f}"


def paired_by_seed(
    seeds: List[int],
    left: Dict[int, Dict[str, Optional[float]]],
    right: Dict[int, Dict[str, Optional[float]]],
    left_key: str,
    right_key: str,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    lv = []
    rv = []
    used = []
    for s in seeds:
        l = left.get(s, {}).get(left_key)
        r = right.get(s, {}).get(right_key)
        if l is None or r is None:
            continue
        lv.append(l)
        rv.append(r)
        used.append(s)
    return np.array(lv, dtype=float), np.array(rv, dtype=float), used


def main():
    args = parse_args()
    seeds = parse_seed_list(args.seeds)

    mome: Dict[int, Dict[str, Optional[float]]] = {s: {} for s in seeds}
    retrain: Dict[int, Dict[str, Optional[float]]] = {s: {} for s in seeds}

    # ------------------------
    # Load per-seed results
    # ------------------------
    for s in seeds:
        mdir = f"{args.mome_prefix}{s}"
        rdir = f"{args.retrain_prefix}{s}"

        # MoME v3 test summary json
        mome_summary = load_json(os.path.join(mdir, "mome_v3_eval_summary.json"))
        if mome_summary:
            core = mome_summary.get("core", {})
            full = mome_summary.get("full", {})
            mome[s]["core_mAP"] = to_percent(core.get("test_map"))
            mome[s]["core_R1"] = to_percent(core.get("test_r1"))
            mome[s]["full_mAP"] = to_percent(full.get("test_map"))
            mome[s]["full_R1"] = to_percent(full.get("test_r1"))

        # MoME v3 MIA core/full json
        mome_mia_core = load_json(os.path.join(mdir, "mia_core_results.json"))
        if mome_mia_core:
            mome[s]["core_mia_entropy"] = mome_mia_core.get("entropy_auc")
            mome[s]["core_mia_embedding"] = mome_mia_core.get("embedding_auc")
            mome[s]["core_mia_logit"] = mome_mia_core.get("maxlogit_auc")

        mome_mia_full = load_json(os.path.join(mdir, "mia_full_results.json"))
        if mome_mia_full:
            mome[s]["full_mia_entropy"] = mome_mia_full.get("entropy_auc")
            mome[s]["full_mia_embedding"] = mome_mia_full.get("embedding_auc")
            mome[s]["full_mia_logit"] = mome_mia_full.get("maxlogit_auc")

        # Retrain: test log + MIA json
        retrain_test = extract_retrain_test_metrics(os.path.join(rdir, "eval_test.log"))
        retrain[s]["mAP"] = retrain_test.get("mAP")
        retrain[s]["R1"] = retrain_test.get("R1")

        retrain_mia = load_json(os.path.join(rdir, "mia_core_results.json"))
        if retrain_mia:
            retrain[s]["mia_entropy"] = retrain_mia.get("entropy_auc")
            retrain[s]["mia_embedding"] = retrain_mia.get("embedding_auc")
            retrain[s]["mia_logit"] = retrain_mia.get("maxlogit_auc")

    # ------------------------
    # Print raw table
    # ------------------------
    print("=" * 84)
    print("Raw Results")
    print("=" * 84)
    print("\nMoME v3")
    print(f"{'Seed':>6} | {'Core mAP':>8} | {'Core R1':>8} | {'Full mAP':>8} | {'Full R1':>8} | {'Core MIA(ent)':>13}")
    print("-" * 84)
    for s in seeds:
        r = mome[s]
        print(
            f"{s:>6} | {fmt(r.get('core_mAP'), 2):>8} | {fmt(r.get('core_R1'), 2):>8} | "
            f"{fmt(r.get('full_mAP'), 2):>8} | {fmt(r.get('full_R1'), 2):>8} | {fmt(r.get('core_mia_entropy'), 4):>13}"
        )

    print("\nRetrain baseline")
    print(f"{'Seed':>6} | {'mAP':>8} | {'R1':>8} | {'MIA(ent)':>10}")
    print("-" * 44)
    for s in seeds:
        r = retrain[s]
        print(f"{s:>6} | {fmt(r.get('mAP'), 2):>8} | {fmt(r.get('R1'), 2):>8} | {fmt(r.get('mia_entropy'), 4):>10}")

    # ------------------------
    # Statistical tests
    # ------------------------
    print("\n" + "=" * 84)
    print("Statistical Tests")
    print("=" * 84)

    test_summary = {}

    # Test 1: Core mAP vs Retrain mAP (paired)
    l1, r1, used1 = paired_by_seed(seeds, mome, retrain, "core_mAP", "mAP")
    print("\n--- Test 1: Core mAP vs Retrain mAP (paired t-test) ---")
    print("H0: MoME Core mAP == Retrain mAP")
    if len(l1) >= 3:
        t, p = stats.ttest_rel(l1, r1)
        diff = l1 - r1
        print(f"Seeds used: {used1}")
        print(f"MoME Core mAP: {l1.mean():.2f} +- {l1.std(ddof=1):.2f}")
        print(f"Retrain mAP:   {r1.mean():.2f} +- {r1.std(ddof=1):.2f}")
        print(f"Diff:          {diff.mean():.2f} +- {diff.std(ddof=1):.2f}")
        print(f"t = {t:.4f}, p = {p:.6f}")
        print("Decision:", "PASS (cannot reject H0)" if p > args.alpha else "CHECK (reject H0)")
        test_summary["test1_core_map_vs_retrain"] = {
            "n": len(l1),
            "seeds": used1,
            "t": float(t),
            "p": float(p),
            "alpha": args.alpha,
            "pass": bool(p > args.alpha),
        }
    else:
        print("Not enough paired samples.")

    # Test 2: Core MIA(ent) vs Retrain MIA(ent) (paired)
    l2, r2, used2 = paired_by_seed(seeds, mome, retrain, "core_mia_entropy", "mia_entropy")
    print("\n--- Test 2: Core MIA(entropy) vs Retrain MIA(entropy) (paired t-test) ---")
    print("H0: MoME Core MIA(ent) == Retrain MIA(ent)")
    if len(l2) >= 3:
        t, p = stats.ttest_rel(l2, r2)
        diff = l2 - r2
        print(f"Seeds used: {used2}")
        print(f"MoME Core MIA(ent): {l2.mean():.4f} +- {l2.std(ddof=1):.4f}")
        print(f"Retrain MIA(ent):   {r2.mean():.4f} +- {r2.std(ddof=1):.4f}")
        print(f"Diff:               {diff.mean():.4f} +- {diff.std(ddof=1):.4f}")
        print(f"t = {t:.4f}, p = {p:.6f}")
        print("Decision:", "PASS (cannot reject H0)" if p > args.alpha else "CHECK (reject H0)")
        test_summary["test2_core_mia_vs_retrain"] = {
            "n": len(l2),
            "seeds": used2,
            "t": float(t),
            "p": float(p),
            "alpha": args.alpha,
            "pass": bool(p > args.alpha),
        }
    else:
        print("Not enough paired samples.")

    # Test 3: Core mAP vs v1.1 core mAP (one-sample)
    core_map_vals = np.array([mome[s].get("core_mAP") for s in seeds if mome[s].get("core_mAP") is not None], dtype=float)
    print("\n--- Test 3: Core mAP vs v1.1 Core mAP (one-sample t-test) ---")
    print(f"H0: MoME Core mAP == {args.v11_core_map}")
    if len(core_map_vals) >= 3:
        t, p = stats.ttest_1samp(core_map_vals, args.v11_core_map)
        print(f"MoME Core mAP: {core_map_vals.mean():.2f} +- {core_map_vals.std(ddof=1):.2f}")
        print(f"v1.1 Core mAP: {args.v11_core_map:.2f}")
        print(f"t = {t:.4f}, p = {p:.8f}")
        pass_cond = (p < 0.01) and (core_map_vals.mean() > args.v11_core_map)
        print("Decision:", "PASS (significantly better than v1.1)" if pass_cond else "CHECK")
        test_summary["test3_core_map_vs_v11"] = {
            "n": len(core_map_vals),
            "t": float(t),
            "p": float(p),
            "threshold": 0.01,
            "pass": bool(pass_cond),
        }
    else:
        print("Not enough samples.")

    print("\n" + "=" * 84)
    print("Summary (for paper)")
    print("=" * 84)

    full_map_vals = np.array([mome[s].get("full_mAP") for s in seeds if mome[s].get("full_mAP") is not None], dtype=float)
    core_mia_vals = np.array([mome[s].get("core_mia_entropy") for s in seeds if mome[s].get("core_mia_entropy") is not None], dtype=float)
    retrain_map_vals = np.array([retrain[s].get("mAP") for s in seeds if retrain[s].get("mAP") is not None], dtype=float)
    retrain_mia_vals = np.array([retrain[s].get("mia_entropy") for s in seeds if retrain[s].get("mia_entropy") is not None], dtype=float)

    summary = {"seeds": seeds, "mome": mome, "retrain": retrain, "tests": test_summary}

    if len(core_map_vals) >= 3:
        m, lo, hi = ci95(core_map_vals)
        print(f"MoME Core mAP: {m:.2f}% (95% CI [{lo:.2f}, {hi:.2f}])")
        summary["mome_core_map_ci95"] = {"mean": m, "lo": lo, "hi": hi}
    if len(core_mia_vals) >= 3:
        m, lo, hi = ci95(core_mia_vals)
        print(f"MoME Core MIA(ent): {m:.4f} (95% CI [{lo:.4f}, {hi:.4f}])")
        summary["mome_core_mia_entropy_ci95"] = {"mean": m, "lo": lo, "hi": hi}
    if len(full_map_vals) >= 3:
        m, lo, hi = ci95(full_map_vals)
        contains_85 = (lo <= 85.0 <= hi)
        print(f"MoME Full mAP: {m:.2f}% (95% CI [{lo:.2f}, {hi:.2f}])")
        print(f"Full mAP 95% CI contains 85.0%: {contains_85}")
        summary["mome_full_map_ci95"] = {"mean": m, "lo": lo, "hi": hi, "contains_85": contains_85}
    if len(retrain_map_vals) >= 3:
        m, lo, hi = ci95(retrain_map_vals)
        print(f"Retrain mAP: {m:.2f}% (95% CI [{lo:.2f}, {hi:.2f}])")
        summary["retrain_map_ci95"] = {"mean": m, "lo": lo, "hi": hi}
    if len(retrain_mia_vals) >= 3:
        m, lo, hi = ci95(retrain_mia_vals)
        print(f"Retrain MIA(ent): {m:.4f} (95% CI [{lo:.4f}, {hi:.4f}])")
        summary["retrain_mia_entropy_ci95"] = {"mean": m, "lo": lo, "hi": hi}

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary json to: {args.output}")


if __name__ == "__main__":
    main()
