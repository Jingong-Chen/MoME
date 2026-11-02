"""Identity-level MIA (responds to Reviewer Nm86 N3).

Aggregate per-PID across all images of that identity:
  - mean entropy
  - max entropy
  - min loss

AUC computed over PIDs: member PIDs (D_f) vs nonmember PIDs (query).
Note: train and query PID sets are disjoint in standard ReID protocols.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

from config import cfg
from datasets import make_dataloader, make_dataloader_mia
from model import make_model
from utils.logger import setup_logger
from utils.parasitic_utils import set_parasitic_mode


def per_sample_entropy(logits):
    if isinstance(logits, list):
        logits = logits[0]
    probs = F.softmax(logits, dim=-1).clamp_min(1e-12)
    return -(probs * probs.log()).sum(dim=-1)


@torch.no_grad()
def collect_logits_and_pids(model, loader, device="cuda"):
    logits_all = []; pids_all = []
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
        logits_all.append(logits.detach().cpu())
        pids_all.append(pid.cpu() if torch.is_tensor(pid) else torch.tensor(pid))
    return torch.cat(logits_all, dim=0), torch.cat(pids_all, dim=0)


def aggregate_per_pid(logits, pids, agg="mean"):
    """Returns dict pid -> aggregated entropy score."""
    ent = -per_sample_entropy(logits).numpy()  # higher = more member-like
    by_pid = defaultdict(list)
    for i, p in enumerate(pids.tolist()):
        by_pid[int(p)].append(float(ent[i]))
    if agg == "mean":
        return {p: float(np.mean(v)) for p, v in by_pid.items()}
    if agg == "max":
        return {p: float(np.max(v)) for p, v in by_pid.items()}
    raise ValueError(agg)


def tpr_at_fpr(y_true, y_score, target_fprs=(0.01, 0.001, 0.0001)):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    out = {}
    for f in target_fprs:
        idx = max(0, min(np.searchsorted(fpr, f, side="right") - 1, len(tpr) - 1))
        out[f"tpr_at_{f}"] = float(tpr[idx])
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identity-level MIA")
    parser.add_argument("--config_file", default="", type=str)
    parser.add_argument("--mode", default="core", choices=["full", "core", "standard"])
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--loft", default=None, type=str,
                        help="Optional LOFT projector loft_U.pt; applies P=UU^T to the embedding.")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True) if output_dir else None
    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info("Identity-level MIA. mode={}".format(args.mode))

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID
    _, _, _, _, num_classes, camera_num, view_num = make_dataloader(cfg)
    member_loader, nonmember_loader, _ = make_dataloader_mia(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(cfg.TEST.WEIGHT)
    model = model.cuda(); model.eval()
    # "standard" keeps the branch ON: non-MoME checkpoints were trained by the ordinary
    # entrypoints, where the parasitic tensors are ordinary trainable parameters.
    set_parasitic_mode(model, mode=(args.mode != "core"))
    if args.loft:
        from eval_loft import LoftWrapper
        payload = torch.load(args.loft, map_location="cpu")
        U = payload["U"].float()
        logger.info("LOFT projector {}: U {}x{}".format(args.loft, U.size(0), U.size(1)))
        model = LoftWrapper(model, (U @ U.t()).float().cuda()).cuda()
        model.eval()

    m_logits, m_pids = collect_logits_and_pids(model, member_loader)
    q_logits, q_pids = collect_logits_and_pids(model, nonmember_loader)

    res = {}
    for agg in ["mean", "max"]:
        m_dict = aggregate_per_pid(m_logits, m_pids, agg=agg)
        q_dict = aggregate_per_pid(q_logits, q_pids, agg=agg)
        m_scores = list(m_dict.values()); q_scores = list(q_dict.values())
        y = np.concatenate([np.ones(len(m_scores)), np.zeros(len(q_scores))])
        s = np.concatenate([m_scores, q_scores])
        res[agg] = {
            "n_member_pids": len(m_scores),
            "n_query_pids": len(q_scores),
            "auc": float(roc_auc_score(y, s)),
            **tpr_at_fpr(y, s),
        }
        logger.info("[agg={}] AUC={:.4f} (member_pids={}, query_pids={})".format(
            agg, res[agg]["auc"], len(m_scores), len(q_scores)))

    res["meta"] = {"target_ckpt": cfg.TEST.WEIGHT, "mode": args.mode}
    out_json = args.out_json or os.path.join(output_dir, "identity_mia_{}.json".format(args.mode))
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
    logger.info("Saved {}".format(out_json))
