"""MIA 评估工具。"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def entropy_score(logits):
    probs = F.softmax(logits, dim=1).clamp(min=1e-12)
    entropy = -(probs * probs.log()).sum(dim=1)
    return -entropy


def max_logit_score(logits):
    return logits.max(dim=1)[0]


def compute_class_centers(embeddings, pids, num_classes):
    dim = embeddings.size(1)
    centers = embeddings.new_zeros((num_classes, dim))
    counts = embeddings.new_zeros((num_classes, 1))
    for c in range(num_classes):
        mask = pids == c
        if mask.any():
            centers[c] = embeddings[mask].mean(dim=0)
            counts[c] = mask.sum()
    valid = counts.squeeze(1) > 0
    return centers[valid]


def embedding_distance_score(embeddings, class_centers):
    if class_centers.numel() == 0:
        return torch.zeros(embeddings.size(0), device=embeddings.device)
    dist = torch.cdist(embeddings, class_centers, p=2)
    return -dist.min(dim=1)[0]


def try_compute_logits(model, feat):
    classifier = getattr(model, "classifier", None)
    if classifier is None:
        return None
    in_planes = getattr(model, "in_planes", feat.shape[1])
    if feat.shape[1] != in_planes and feat.shape[1] >= in_planes:
        feat = feat[:, :in_planes]
    try:
        out = classifier(feat)
    except (TypeError, RuntimeError):
        return None
    if isinstance(out, tuple):
        return out[0]
    return out


@torch.no_grad()
def collect_outputs(model, loader, device="cuda"):
    feats, logits, pids = [], [], []
    model.eval()
    for img, pid, camid, viewid, _ in loader:
        img = img.to(device)
        camid = camid.to(device)
        viewid = viewid.to(device)
        feat = model(img, cam_label=camid, view_label=viewid)
        logit = try_compute_logits(model, feat)
        feats.append(feat.detach().cpu())
        pids.append(pid.cpu())
        if logit is not None:
            logits.append(logit.detach().cpu())
    feats = torch.cat(feats, dim=0)
    pids = torch.cat(pids, dim=0)
    logits = torch.cat(logits, dim=0) if len(logits) > 0 else None
    return feats, logits, pids


def compute_mia_auc(member_scores, nonmember_scores):
    y_true = np.concatenate(
        [np.ones(len(member_scores), dtype=np.int64), np.zeros(len(nonmember_scores), dtype=np.int64)]
    )
    y_score = np.concatenate([member_scores, nonmember_scores])
    return roc_auc_score(y_true, y_score)
