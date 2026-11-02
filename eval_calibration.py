"""ECE / Reliability diagrams (responds to Reviewer Kd9X K3, Nm86 'absolute' privacy).

Given a checkpoint, computes:
  - top-1 accuracy on test query (ReID retrieval is rank-1 in the gallery, but here we
    use the closed-set classifier head on training PIDs, so accuracy is effectively a
    PID-prediction accuracy on the test set — a proxy for confidence calibration).
  - ECE (15 bins).
  - Reliability diagram saved to PNG/PDF.

Most useful interpretation: Original / Fine-tune are over-confident on test (high ECE),
Retrain / Core should be more conservative.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from config import cfg
from datasets import make_dataloader, make_dataloader_mia
from model import make_model
from utils.logger import setup_logger
from utils.parasitic_utils import set_parasitic_mode


@torch.no_grad()
def collect_top_probs(model, loader, device="cuda"):
    tops = []; preds = []; pids = []
    model.eval()
    for batch in loader:
        if len(batch) == 6:
            img, pid, _camid, camids_bat, viewid, _ = batch
        else:
            img, pid, camid, viewid, _ = batch
            camids_bat = camid
        img = img.to(device)
        cam = camids_bat.to(device) if torch.is_tensor(camids_bat) else torch.tensor(camids_bat, device=device)
        view = viewid.to(device) if torch.is_tensor(viewid) else torch.tensor(viewid, device=device)
        feat = model(img, cam_label=cam, view_label=view)
        classifier = getattr(model, "classifier", None)
        in_planes = getattr(model, "in_planes", feat.shape[1] if torch.is_tensor(feat) else 768)
        f = feat
        if torch.is_tensor(f) and f.shape[1] != in_planes and f.shape[1] >= in_planes:
            f = f[:, :in_planes]
        out = classifier(f)
        logits = out[0] if isinstance(out, tuple) else out
        probs = F.softmax(logits, dim=-1)
        top, pred = probs.max(dim=-1)
        tops.append(top.cpu()); preds.append(pred.cpu())
        pids.append(pid.cpu() if torch.is_tensor(pid) else torch.tensor(pid))
    return torch.cat(tops).numpy(), torch.cat(preds).numpy(), torch.cat(pids).numpy()


def compute_ece(probs_top, correct, n_bins=15):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []
    N = len(probs_top)
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (probs_top > lo) & (probs_top <= hi)
        if not mask.any():
            bin_data.append({"lo": lo, "hi": hi, "n": 0, "acc": 0.0, "conf": 0.0})
            continue
        bin_acc = float(correct[mask].mean())
        bin_conf = float(probs_top[mask].mean())
        ece += (mask.sum() / N) * abs(bin_acc - bin_conf)
        bin_data.append({"lo": float(lo), "hi": float(hi), "n": int(mask.sum()),
                          "acc": bin_acc, "conf": bin_conf})
    return float(ece), bin_data


def plot_reliability(bin_data, ece, out_pdf, title=""):
    centers = [(b["lo"] + b["hi"]) / 2 for b in bin_data]
    accs = [b["acc"] for b in bin_data]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.7, label="ideal")
    ax.bar(centers, accs, width=1.0 / len(bin_data) * 0.95, alpha=0.7, edgecolor="k", label="observed")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("{} (ECE={:.4f})".format(title, ece))
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=150)
    plt.savefig(out_pdf.replace(".pdf", ".png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECE / Reliability diagram")
    parser.add_argument("--config_file", default="", type=str)
    parser.add_argument("--mode", default="core", choices=["full", "core"])
    parser.add_argument("--out_json", default=None)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True) if output_dir else None
    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info("Calibration. mode={} ckpt={}".format(args.mode, cfg.TEST.WEIGHT))

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID
    _, _, _, _, num_classes, camera_num, view_num = make_dataloader(cfg)
    member_loader, nonmember_loader, _ = make_dataloader_mia(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(cfg.TEST.WEIGHT)
    model = model.cuda(); model.eval()
    set_parasitic_mode(model, mode=(args.mode == "full"))

    # Use member loader (D_f) for ECE on training data (over-confidence diagnosis).
    tops, preds, pids = collect_top_probs(model, member_loader)
    correct = (preds == pids).astype(np.float32)
    ece_train, bins_train = compute_ece(tops, correct)
    plot_reliability(bins_train, ece_train,
                     os.path.join(output_dir, "reliability_member_{}.pdf".format(args.mode)),
                     title="Members (D_f) — {}".format(args.mode))

    # Query (test) set — top-1 over training PIDs (will be 0 acc for new PIDs but probability spread informative)
    tops_q, preds_q, pids_q = collect_top_probs(model, nonmember_loader)
    # For query, "accuracy" is ill-defined (PIDs are different); skip ECE here, only stats.
    res = {
        "ece_member": ece_train,
        "member_acc": float(correct.mean()),
        "member_top_prob_mean": float(tops.mean()),
        "member_top_prob_std": float(tops.std()),
        "query_top_prob_mean": float(tops_q.mean()),
        "query_top_prob_std": float(tops_q.std()),
        "bins_member": bins_train,
        "meta": {"target_ckpt": cfg.TEST.WEIGHT, "mode": args.mode},
    }
    out_json = args.out_json or os.path.join(output_dir, "calibration_{}.json".format(args.mode))
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
    logger.info("Saved {} (member ECE={:.4f}, acc={:.3f})".format(out_json, ece_train, correct.mean()))
