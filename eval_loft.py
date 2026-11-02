"""Evaluate a trained LOFT projector U (from train_loft.py) in the ReID setting.

LOFT applies the orthonormal projector UU^T to the frozen retrieval embedding
(f_U = h_pre(UU^T g_pre) in the paper; here, retrieval is distance-based so we project
the embedding). We wrap the frozen base model so its eval-time feature is projected, then
reuse the STANDARD evaluation paths for comparability:
  * retrieval mAP/CMC via processor.do_inference (unchanged),
  * MIA (entropy / max-logit / embedding-distance) via utils.mia_utils.collect_outputs,
    where the classifier logits are computed on the projected feature (try_compute_logits).

Reference-based low-FPR attacks (RMIA / U-LiRA) assume a fixed representation shared with the
reference models; a target-only feature projection changes the representation, so they are not
directly comparable for LOFT and are handled/flagged separately in the paper.

Usage:
  python eval_loft.py --config_file <cfg> --loft logs/rebuttal_loft_<ds>_seed42/loft_U.pt \\
     MODEL.DEVICE_ID "'0'" TEST.WEIGHT <standard_ckpt> DATASETS.ROOT_DIR <data> OUTPUT_DIR <out>
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from config import cfg
from datasets import make_dataloader, make_dataloader_mia
from model import make_model
from processor import do_inference
from utils.logger import setup_logger
from utils.mia_utils import (
    collect_outputs,
    compute_class_centers,
    compute_mia_auc,
    embedding_distance_score,
    entropy_score,
    max_logit_score,
)
from utils.parasitic_utils import set_parasitic_mode


class LoftWrapper(nn.Module):
    """Frozen base model whose eval-time retrieval feature is projected by P = UU^T.
    Exposes .classifier / .in_planes so mia_utils.try_compute_logits works on the
    projected feature."""

    def __init__(self, base, P):
        super().__init__()
        self.base = base
        self.register_buffer("P", P)
        self.classifier = getattr(base, "classifier", None)
        self.in_planes = getattr(base, "in_planes", None)

    def forward(self, x, cam_label=None, view_label=None, **kw):
        out = self.base(x, cam_label=cam_label, view_label=view_label, **kw)
        if isinstance(out, (tuple, list)):  # train-mode (score, feat) -- not used at eval
            return out
        P = self.P.to(out.dtype).to(out.device)
        return out @ P

    def eval(self):
        self.base.eval()
        return self

    def train(self, mode=True):
        self.base.train(mode)
        return self


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LOFT projector (retrieval + MIA)")
    parser.add_argument("--config_file", default="", type=str)
    parser.add_argument("--loft", required=True, type=str, help="path to loft_U.pt")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = setup_logger("transreid", output_dir, if_train=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

    payload = torch.load(args.loft, map_location="cpu")
    U = payload["U"].float()
    P = (U @ U.t()).float()
    logger.info("Loaded LOFT projector U: {} x {} (s={}), P={}x{}".format(
        U.size(0), U.size(1), payload.get("s"), P.size(0), P.size(1)))

    # ---- retrieval (mAP/CMC) via standard do_inference on the projected model ----
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    base = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    base.load_param(cfg.TEST.WEIGHT)
    base = base.cuda()
    set_parasitic_mode(base, False)
    base.eval()
    model = LoftWrapper(base, P.cuda()).cuda()
    model.eval()

    logger.info("=== LOFT retrieval (projected embedding) ===")
    do_inference(cfg, model, val_loader, num_query)

    # ---- MIA on projected features/logits ----
    logger.info("=== LOFT MIA (projected) ===")
    member_loader, nonmember_loader, _ = make_dataloader_mia(cfg)
    m_feat, m_logit, m_pid = collect_outputs(model, member_loader, device="cuda")
    n_feat, n_logit, _ = collect_outputs(model, nonmember_loader, device="cuda")

    results = {}
    centers = compute_class_centers(m_feat, m_pid, num_classes)
    results["embedding_auc"] = float(compute_mia_auc(
        embedding_distance_score(m_feat, centers).numpy(),
        embedding_distance_score(n_feat, centers).numpy()))
    if m_logit is not None and n_logit is not None:
        results["entropy_auc"] = float(compute_mia_auc(
            entropy_score(m_logit).numpy(), entropy_score(n_logit).numpy()))
        results["maxlogit_auc"] = float(compute_mia_auc(
            max_logit_score(m_logit).numpy(), max_logit_score(n_logit).numpy()))
    else:
        results["entropy_auc"] = float("nan")
        results["maxlogit_auc"] = float("nan")

    logger.info("LOFT MIA => entropy: {:.4f}, embedding: {:.4f}, maxlogit: {:.4f}".format(
        results["entropy_auc"], results["embedding_auc"], results["maxlogit_auc"]))
    save_path = os.path.join(output_dir, "loft_mia.json")
    with open(save_path, "w") as f:
        json.dump({k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in results.items()}, f, indent=2)
    logger.info("Saved LOFT MIA to {}".format(save_path))
