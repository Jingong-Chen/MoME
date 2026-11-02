import argparse
import json
import os

import numpy as np
import torch

from config import cfg
from datasets import make_dataloader, make_dataloader_mia
from model import make_model
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MIA AUC for ReID model")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument(
        "--mode",
        default="full",
        choices=["full", "core"],
        help="full=parasitic mode, core=clean mode",
    )
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID
    _, _, _, _, num_classes, camera_num, view_num = make_dataloader(cfg)
    member_loader, nonmember_loader, _ = make_dataloader_mia(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(cfg.TEST.WEIGHT)
    model = model.to("cuda")
    model.eval()

    set_parasitic_mode(model, mode=(args.mode == "full"))

    member_feats, member_logits, member_pids = collect_outputs(model, member_loader, device="cuda")
    nonmember_feats, nonmember_logits, _ = collect_outputs(model, nonmember_loader, device="cuda")

    results = {}
    centers = compute_class_centers(member_feats, member_pids, num_classes)
    member_emb_scores = embedding_distance_score(member_feats, centers).numpy()
    nonmember_emb_scores = embedding_distance_score(nonmember_feats, centers).numpy()
    results["embedding_auc"] = float(compute_mia_auc(member_emb_scores, nonmember_emb_scores))

    if member_logits is not None and nonmember_logits is not None:
        member_ent_scores = entropy_score(member_logits).numpy()
        nonmember_ent_scores = entropy_score(nonmember_logits).numpy()
        results["entropy_auc"] = float(compute_mia_auc(member_ent_scores, nonmember_ent_scores))

        member_logit_scores = max_logit_score(member_logits).numpy()
        nonmember_logit_scores = max_logit_score(nonmember_logits).numpy()
        results["maxlogit_auc"] = float(compute_mia_auc(member_logit_scores, nonmember_logit_scores))
    else:
        results["entropy_auc"] = float("nan")
        results["maxlogit_auc"] = float("nan")
        logger.warning("classifier does not output logits in current setup; skip logit/entropy attacks")

    logger.info(
        "MIA AUC ({}) => entropy: {:.4f}, embedding: {:.4f}, maxlogit: {:.4f}".format(
            args.mode, results["entropy_auc"], results["embedding_auc"], results["maxlogit_auc"]
        )
    )

    save_path = os.path.join(output_dir, "mia_{}_results.json".format(args.mode))
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in results.items()},
            f,
            indent=2,
        )
    logger.info("Saved MIA results to {}".format(save_path))
