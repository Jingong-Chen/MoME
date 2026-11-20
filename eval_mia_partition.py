import argparse
import json
import os

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from config import cfg
from datasets.bases import ImageDataset
from datasets.dukemtmcreid import DukeMTMCreID
from datasets.make_dataloader_mia import mia_collate_fn
from datasets.make_dataloader_mome import _split_dr_and_df_minus_r
from datasets.market1501 import Market1501
from datasets.msmt17 import MSMT17
from datasets.occ_duke import OCC_DukeMTMCreID
from datasets.regdb import RegDB
from datasets.vehicleid import VehicleID
from datasets.veri import VeRi
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

__factory = {
    "market1501": Market1501,
    "dukemtmc": DukeMTMCreID,
    "msmt17": MSMT17,
    "occ_duke": OCC_DukeMTMCreID,
    "veri": VeRi,
    "veri776": VeRi,
    "regdb": RegDB,
    "VehicleID": VehicleID,
}


def build_eval_loader(cfg_obj, data_source):
    val_transforms = T.Compose(
        [
            T.Resize(cfg_obj.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg_obj.INPUT.PIXEL_MEAN, std=cfg_obj.INPUT.PIXEL_STD),
        ]
    )
    ds = ImageDataset(data_source, val_transforms)
    return DataLoader(
        ds,
        batch_size=cfg_obj.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=cfg_obj.DATALOADER.NUM_WORKERS,
        collate_fn=mia_collate_fn,
    )


def build_member_nonmember_sets(dataset, member_set, dr_samples_per_id, seed):
    dr_indices, df_only_indices = _split_dr_and_df_minus_r(
        dataset.train, dr_samples_per_id, seed
    )
    dr_data = [dataset.train[i] for i in dr_indices]
    df_only_data = [dataset.train[i] for i in df_only_indices]
    df_data = list(dataset.train)

    if member_set == "full":
        member_data = df_data
    elif member_set == "forget":
        member_data = df_only_data
    elif member_set == "retain":
        member_data = dr_data
    else:
        raise ValueError("Unsupported member_set: {}".format(member_set))

    nonmember_data = list(dataset.query)
    split_info = {
        "df_size": len(df_data),
        "dr_size": len(dr_data),
        "df_only_size": len(df_only_data),
        "member_size": len(member_data),
        "nonmember_size": len(nonmember_data),
    }
    return member_data, nonmember_data, split_info


def sanitize_tag(tag):
    out = []
    for ch in tag:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate partition-aware MIA AUC (member set: full/forget/retain)"
    )
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("--checkpoint", default="", help="model checkpoint path", type=str)
    parser.add_argument(
        "--mode",
        default="core",
        choices=["standard", "core", "full"],
        help="full=parasitic on, core/standard=parasitic off",
    )
    parser.add_argument(
        "--member_set",
        default="full",
        choices=["full", "forget", "retain"],
        help="member split for MIA",
    )
    parser.add_argument("--seed", default=42, type=int, help="split seed for D_r")
    parser.add_argument(
        "--dr_samples_per_id",
        default=None,
        type=int,
        help="k images per identity for D_r split (default: cfg.MOME.DR_SAMPLES_PER_ID)",
    )
    parser.add_argument("--tag", default="model", type=str, help="tag used in output file")
    parser.add_argument("--output_dir", default="logs", type=str)
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = args.output_dir if args.output_dir else cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = setup_logger("transreid.mia.partition", output_dir, if_train=False)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID
    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    num_classes = dataset.num_train_pids
    camera_num = dataset.num_train_cams
    view_num = dataset.num_train_vids
    dr_samples_per_id = (
        args.dr_samples_per_id
        if args.dr_samples_per_id is not None
        else int(cfg.MOME.DR_SAMPLES_PER_ID)
    )

    member_data, nonmember_data, split_info = build_member_nonmember_sets(
        dataset, args.member_set, dr_samples_per_id, args.seed
    )
    logger.info(
        "MIA split => member_set=%s | D_f=%d D_r=%d D_f\\D_r=%d | member=%d nonmember=%d",
        args.member_set,
        split_info["df_size"],
        split_info["dr_size"],
        split_info["df_only_size"],
        split_info["member_size"],
        split_info["nonmember_size"],
    )

    member_loader = build_eval_loader(cfg, member_data)
    nonmember_loader = build_eval_loader(cfg, nonmember_data)

    weight = args.checkpoint if args.checkpoint else cfg.TEST.WEIGHT
    if not weight:
        raise ValueError("No checkpoint provided. Use --checkpoint or TEST.WEIGHT in opts.")

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(weight)
    model = model.to("cuda")
    model.eval()

    if args.mode == "full":
        set_parasitic_mode(model, mode=True)
    else:
        set_parasitic_mode(model, mode=False)

    member_feats, member_logits, member_pids = collect_outputs(
        model, member_loader, device="cuda"
    )
    nonmember_feats, nonmember_logits, _ = collect_outputs(
        model, nonmember_loader, device="cuda"
    )

    results = {
        "tag": args.tag,
        "mode": args.mode,
        "member_set": args.member_set,
        "seed": int(args.seed),
        "dr_samples_per_id": int(dr_samples_per_id),
        "weight": weight,
        "config_file": args.config_file,
        "split": split_info,
    }
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
        "MIA AUC (%s/%s) => entropy: %.4f, embedding: %.4f, maxlogit: %.4f",
        args.mode,
        args.member_set,
        results["entropy_auc"],
        results["embedding_auc"],
        results["maxlogit_auc"],
    )

    save_name = "mia_partition_{}_{}_{}_seed{}.json".format(
        sanitize_tag(args.tag), args.mode, args.member_set, args.seed
    )
    save_path = os.path.join(output_dir, save_name)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in results.items()},
            f,
            indent=2,
        )
    logger.info("Saved MIA results to %s", save_path)
