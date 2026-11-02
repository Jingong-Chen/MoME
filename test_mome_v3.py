import argparse
import json
import os

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from config import cfg
from datasets import make_dataloader, make_dataloader_mia
from datasets.bases import ImageDataset
from datasets.dukemtmcreid import DukeMTMCreID
from datasets.make_dataloader import val_collate_fn
from datasets.market1501 import Market1501
from datasets.msmt17 import MSMT17
from datasets.occ_duke import OCC_DukeMTMCreID
from datasets.regdb import RegDB
from datasets.vehicleid import VehicleID
from datasets.veri import VeRi
from model import make_model
from utils.logger import setup_logger
from utils.metrics import R1_mAP_eval
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


@torch.no_grad()
def evaluate_loader(cfg, model, loader, num_query, split_name):
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()
    model.eval()
    for img, vid, camid, camids, target_view, _ in loader:
        img = img.cuda()
        camids = camids.cuda()
        target_view = target_view.cuda()
        feat = model(img, cam_label=camids, view_label=target_view)
        evaluator.update((feat, vid, camid))
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    print(
        "[{}] mAP: {:.2%}, Rank-1: {:.2%}, Rank-5: {:.2%}, Rank-10: {:.2%}".format(
            split_name, mAP, cmc[0], cmc[4], cmc[9]
        )
    )
    return cmc, mAP


def build_train_eval_loader(cfg):
    val_transforms = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ]
    )
    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    train_eval_set = ImageDataset(dataset.train + dataset.train, val_transforms)
    train_eval_loader = DataLoader(
        train_eval_set,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=val_collate_fn,
    )
    return train_eval_loader, len(dataset.train)


@torch.no_grad()
def run_mia_evaluation(cfg, model, output_dir):
    member_loader, nonmember_loader, num_classes = make_dataloader_mia(cfg)

    member_feats, member_logits, member_pids = collect_outputs(
        model, member_loader, device="cuda"
    )
    nonmember_feats, nonmember_logits, _ = collect_outputs(
        model, nonmember_loader, device="cuda"
    )

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

    print(
        "MIA (Core) => entropy: {:.4f}, embedding: {:.4f}, maxlogit: {:.4f}".format(
            results["entropy_auc"], results["embedding_auc"], results["maxlogit_auc"]
        )
    )
    save_path = os.path.join(output_dir, "mia_core_v3_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved MIA results to {}".format(save_path))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MoME v3 checkpoint")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("--checkpoint", required=True, help="path to checkpoint")
    parser.add_argument(
        "--skip_mia",
        action="store_true",
        help="skip MIA evaluation in core mode",
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
    logger.info("Checkpoint: {}".format(args.checkpoint))

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID
    _, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    train_eval_loader, train_num_query = build_train_eval_loader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(args.checkpoint)
    model = model.cuda()
    model.eval()

    results = {}

    set_parasitic_mode(model, True)
    print("=== Full Mode (Parasitic ON) ===")
    full_test_cmc, full_test_map = evaluate_loader(
        cfg, model, val_loader, num_query, split_name="full-test"
    )
    _, full_train_map = evaluate_loader(
        cfg, model, train_eval_loader, train_num_query, split_name="full-train"
    )
    results["full"] = {
        "test_map": float(full_test_map),
        "test_r1": float(full_test_cmc[0]),
        "train_map": float(full_train_map),
    }

    set_parasitic_mode(model, False)
    print("=== Core Mode (Parasitic OFF) ===")
    core_test_cmc, core_test_map = evaluate_loader(
        cfg, model, val_loader, num_query, split_name="core-test"
    )
    _, core_train_map = evaluate_loader(
        cfg, model, train_eval_loader, train_num_query, split_name="core-train"
    )
    results["core"] = {
        "test_map": float(core_test_map),
        "test_r1": float(core_test_cmc[0]),
        "train_map": float(core_train_map),
    }

    if not args.skip_mia:
        set_parasitic_mode(model, False)
        results["core_mia"] = run_mia_evaluation(cfg, model, output_dir=output_dir)

    results_path = os.path.join(output_dir, "mome_v3_eval_summary.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved summary to {}".format(results_path))
