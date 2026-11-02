"""Aggregate the five deletion splits into one record per table cell.

Each cell carries the five members, their mean and sample standard deviation, and the file each
member was read from, so a number in the paper can be traced to a split without re-deriving which
directory held it. Split 42 is read as its own seed-42 run, not as the five-seed mean the published
table reports, because averaging a mean together with four single runs gives a spread with no clean
meaning.
"""
import glob
import json
import os
import re
import statistics as st
import sys

SPLITS = [42, 123, 456, 789, 1024]
ROWS = ["original", "finetune", "ga", "neggrad", "salun", "bad_teacher", "lora", "ssd",
        "scrub", "lotus", "munba", "loft", "retrain", "coreonly", "mome_core", "mome_full"]
LBL = {"original": "Original", "finetune": "Fine-tune", "ga": "GA", "neggrad": "NegGrad+",
       "salun": "SalUn", "bad_teacher": "Bad Teacher", "lora": "LoRA", "ssd": "SSD",
       "scrub": "SCRUB", "lotus": "LoTUS", "munba": "MUNBa", "loft": "LOFT",
       "retrain": "Retrain", "coreonly": "Core-only", "mome_core": "MoME-Core",
       "mome_full": "MoME-Full"}


def market_veri_cell(d, s, tag):
    """R-1, mAP and U mAP from the partition pass, three attack AUCs from the advmia json."""
    out = {}
    f = glob.glob("logs/sp%d_%s_eval/partition_eval_%s_*.json" % (s, d, tag))
    if f:
        m = json.load(open(f[0]))["metrics"]
        out["r1"] = (m["test"]["R1"], f[0])
        out["map"] = (m["test"]["mAP"], f[0])
        out["u"] = (m["D_f\\D_r"]["mAP"], f[0])
        out["u_r1"] = (m["D_f\\D_r"]["R1"], f[0])
    p = "logs/sp%d_%s_eval/advmia_%s.json" % (s, d, tag)
    if os.path.exists(p):
        a = json.load(open(p))
        for k, fld in (("ent", "entropy"), ("rmia", "rmia_lf"), ("ul", "u_lira_lf")):
            out[k] = (a[fld]["auc"], p)
    return out


def regdb_cell(s, tag):
    """RegDB utility comes from rgk_harvest, in both modality directions, branch off."""
    out = {}
    base = "logs/sp%d_regdb_eval" % s
    for src in ("harvest", "harvest_full", "harvest_loft"):
        p = "%s/%s.json" % (base, src)
        if not os.path.exists(p):
            continue
        H = json.load(open(p))
        if tag in H:
            # Both directions carry a Rank-1 and an mAP, and the table prints all four.
            out["v2t_r1"] = (H[tag]["v2t"]["R1"], p)
            out["v2t"] = (H[tag]["v2t"]["mAP"], p)
            out["t2v_r1"] = (H[tag]["t2v"]["R1"], p)
            out["t2v"] = (H[tag]["t2v"]["mAP"], p)
            out["u"] = (H[tag]["U_mAP"], p)
            break
    # LOFT's retrieval does not come from the harvest. rgk_harvest takes a projector but applies
    # it only inside the U evaluation, so its V->T and T->V columns are the unprojected model's.
    # eval_loft.py wraps the frozen model so the eval-time feature is projected, which is the path
    # the published row came from, and it writes a log per direction.
    if tag == "loft":
        for k, m in (("v2t", "v2t"), ("t2v", "t2v")):
            lg = "%s/loft_%s.log" % (base, m)
            if os.path.exists(lg):
                hit = re.findall(r"mAP: ([0-9.]+)%", open(lg).read())
                if hit:
                    out[k] = (float(hit[-1]), lg)
                r1 = re.findall(r"Rank-1 *: *([0-9.]+)%", open(lg).read())
                if r1:
                    out[k + "_r1"] = (float(r1[-1]), lg)

    p = "%s/advmia_%s.json" % (base, tag)
    if os.path.exists(p):
        a = json.load(open(p))
        for k, fld in (("ent", "entropy"), ("rmia", "rmia_lf"), ("ul", "u_lira_lf")):
            out[k] = (a[fld]["auc"], p)
    return out


def build(bench):
    rec = {}
    for tag in ROWS:
        cells = {}
        for s in SPLITS:
            c = regdb_cell(s, tag) if bench == "regdb" else market_veri_cell(bench, s, tag)
            for k, (v, src) in c.items():
                cells.setdefault(k, {})[s] = {"value": v, "source": src}
        row = {}
        for k, per in cells.items():
            vals = [per[s]["value"] for s in SPLITS if s in per]
            if not vals:
                continue
            row[k] = {"mean": st.mean(vals),
                      "sd": st.stdev(vals) if len(vals) > 1 else 0.0,
                      "n": len(vals),
                      "members": {str(s): per[s] for s in SPLITS if s in per}}
        if row:
            rec[LBL[tag]] = row
    return rec


if __name__ == "__main__":
    out = {b: build(b) for b in ("market", "veri", "regdb")}
    with open("logs/splitvar/aggregate.json", "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    for b in out:
        n = sum(len(r) for r in out[b].values())
        miss = [(k, c) for k, r in out[b].items() for c in r if r[c]["n"] < 5]
        print("%-7s rows %2d  cells %3d  %s" % (
            b, len(out[b]), n, "all five members" if not miss else "short: %s" % miss[:4]))
