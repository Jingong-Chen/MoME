import argparse
import json
import os

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from config import cfg
from datasets.bases import ImageDataset
from datasets.dukemtmcreid import DukeMTMCreID
from datasets.make_dataloader import val_collate_fn
from datasets.make_dataloader_mome import _split_dr_and_df_minus_r, _split_by_identity
from datasets.market1501 import Market1501
from datasets.msmt17 import MSMT17
from datasets.occ_duke import OCC_DukeMTMCreID
from datasets.regdb import RegDB
from datasets.vehicleid import VehicleID
from datasets.veri import VeRi
from model import make_model
from utils.metrics import R1_mAP_eval
from utils.parasitic_utils import set_parasitic_mode, build_cfg_for_ckpt

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


def build_eval_loader(data_source, cfg_obj):
    val_transforms = T.Compose(
        [
            T.Resize(cfg_obj.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg_obj.INPUT.PIXEL_MEAN, std=cfg_obj.INPUT.PIXEL_STD),
        ]
    )
    # Query and gallery are the same subset; each sample appears once as query and once as gallery.
    eval_set = ImageDataset(data_source + data_source, val_transforms)
    loader = DataLoader(
        eval_set,
        batch_size=cfg_obj.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=cfg_obj.DATALOADER.NUM_WORKERS,
        collate_fn=val_collate_fn,
    )
    return loader, len(data_source)


@torch.no_grad()
def evaluate_test_split(cfg_obj, model):
    """Standard query/gallery retrieval, so one invocation fills the utility columns too."""
    from datasets.make_dataloader import make_dataloader
    out = make_dataloader(cfg_obj)
    val_loader, num_query = out[2], out[3]   # (train, train_normal, val, num_query, ...)
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg_obj.TEST.FEAT_NORM)
    evaluator.reset()
    model.eval()
    with torch.no_grad():
        for img, vid, camid, camids, target_view, _ in val_loader:
            feat = model(img.cuda(), cam_label=camids.cuda(), view_label=target_view.cuda())
            evaluator.update((feat, vid, camid))
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    return {"num_samples": int(num_query), "mAP": float(mAP * 100.0),
            "R1": float(cmc[0] * 100.0), "R5": float(cmc[4] * 100.0)}


def evaluate_subset(cfg_obj, model, data_source, subset_name):
    if len(data_source) == 0:
        return {
            "num_samples": 0,
            "mAP": None,
            "R1": None,
            "R5": None,
            "R10": None,
        }

    loader, num_query = build_eval_loader(data_source, cfg_obj)
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg_obj.TEST.FEAT_NORM)
    evaluator.reset()

    model.eval()
    # Without no_grad every forward keeps its autograd graph alive and the evaluator holds the
    # feature tensors, so memory grows with the subset size. Three subsets just fit; adding the
    # test split pushed it to 29 GB and OOM. Values are unaffected.
    with torch.no_grad():
        for img, vid, camid, camids, target_view, _ in loader:
            img = img.cuda()
            camids = camids.cuda()
            target_view = target_view.cuda()
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, vid, camid))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    result = {
        "num_samples": int(len(data_source)),
        "mAP": float(mAP * 100.0),
        "R1": float(cmc[0] * 100.0),
        "R5": float(cmc[4] * 100.0),
        "R10": float(cmc[9] * 100.0),
    }
    print(
        "[{}] mAP: {:.2f}% | Rank-1: {:.2f}% | Rank-5: {:.2f}% | Rank-10: {:.2f}%".format(
            subset_name, result["mAP"], result["R1"], result["R5"], result["R10"]
        )
    )
    return result


def sanitize_tag(tag):
    out = []
    for ch in tag:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def main():
    parser = argparse.ArgumentParser(description="Partition evaluation on D_r and D_f\\D_r")
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--weight", required=True, type=str)
    parser.add_argument("--mode", default="standard", choices=["standard", "core", "full"])
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dr_samples_per_id", default=None, type=int)
    parser.add_argument("--tag", default="model", type=str)
    parser.add_argument("--output_dir", default="logs", type=str)
    parser.add_argument("--include_test", action="store_true",
                        help="Also evaluate the standard query/gallery test split, so a single "
                             "invocation fills the utility and the partition columns of a row "
                             "instead of leaving them to a separate run with its own config.")
    parser.add_argument("--loft", default=None, type=str,
                        help="Optional path to a LOFT projector loft_U.pt. Applies P=UU^T to "
                             "the embedding, which is how LOFT realizes the unlearning.")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=None)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    dr_samples_per_id = (
        args.dr_samples_per_id
        if args.dr_samples_per_id is not None
        else int(cfg.MOME.DR_SAMPLES_PER_ID)
    )

    if cfg.MOME.FORGET_MODE == 'sub_id':
        split_seed = cfg.MOME.SPLIT_SEED if cfg.MOME.SPLIT_SEED >= 0 else args.seed
        dr_indices, df_only_indices = _split_by_identity(
            dataset.train, cfg.MOME.FORGET_ID_FRAC, split_seed,
            leak_id_frac=getattr(cfg.MOME, 'LEAK_ID_FRAC', 0.0),
            dr_k=int(getattr(cfg.MOME, 'DR_K', 0)),
        )
    else:
        dr_indices, df_only_indices = _split_dr_and_df_minus_r(
            dataset.train, dr_samples_per_id, args.seed
        )
    dr_data = [dataset.train[i] for i in dr_indices]
    df_only_data = [dataset.train[i] for i in df_only_indices]
    df_data = list(dataset.train)
    # Under identity-level deletion the training set has THREE parts, not two: D_r, the forget
    # set U, and the images of retained identities that were never sampled into D_r. That third
    # part is not forgotten, so scoring it together with U inflates every membership statistic.
    _taken = set(dr_indices) | set(df_only_indices)
    retained_ho_data = [dataset.train[i] for i in range(len(dataset.train)) if i not in _taken]

    print("Split summary: D_f={} D_r={} D_f\\D_r={}".format(len(df_data), len(dr_data), len(df_only_data)))
    print(
        "Eval model: mode={} | weight={} | seed={} | dr_samples_per_id={}".format(
            args.mode, args.weight, args.seed, dr_samples_per_id
        )
    )

    # A MoME core carries the rank-r parasitic branch and a plain retrain does not; a LoRA build
    # renames every wrapped Linear to ...base.weight, so the build has to follow the checkpoint.
    _cfg = build_cfg_for_ckpt(cfg, args.weight)
    model = make_model(
        _cfg, num_class=dataset.num_train_pids, camera_num=dataset.num_train_cams, view_num=dataset.num_train_vids
    )
    model.load_param(args.weight)
    model = model.cuda()
    model.eval()
    if args.loft:
        # LOFT does not change the weights; it projects the embedding, so the checkpoint is
        # the original model and the projector is applied on top.
        from eval_loft import LoftWrapper
        payload = torch.load(args.loft, map_location='cpu')
        U = payload['U'].float()
        model = LoftWrapper(model, (U @ U.t()).float().cuda()).cuda()
        model.eval()

    if args.mode == "core":
        # Core is the MoME readout with the parasitic branch switched off.
        set_parasitic_mode(model, False)
    else:
        # "full" is the MoME readout with the branch on. "standard" covers every non-MoME
        # checkpoint, which must ALSO be read with the branch on: those models were trained by
        # the ordinary entrypoints, where the parasitic tensors are plain trainable parameters
        # and the layer defaults to parasitic=True, so their effective affine is gamma+gamma_m.
        # 56 of 62 shadows are non-zero in the standard Market baseline, and reading it with the
        # branch off costs 1.7 mAP, i.e. it evaluates a model that was never trained.
        set_parasitic_mode(model, True)

    results = {
        "tag": args.tag,
        "mode": args.mode,
        "seed": int(args.seed),
        "dr_samples_per_id": int(dr_samples_per_id),
        "weight": args.weight,
        "config_file": args.config_file,
        "split": {
            "df_size": len(df_data),
            "dr_size": len(dr_data),
            "df_only_size": len(df_only_data),
            "retained_ho_size": len(retained_ho_data),
        },
        "metrics": {},
    }

    print("\n=== Evaluating D_r ===")
    results["metrics"]["D_r"] = evaluate_subset(cfg, model, dr_data, "D_r")
    print("\n=== Evaluating D_f\\D_r ===")
    results["metrics"]["D_f\\D_r"] = evaluate_subset(cfg, model, df_only_data, "D_f\\D_r")
    print("\n=== Evaluating retained-held-out ===")
    results["metrics"]["retained_ho"] = evaluate_subset(cfg, model, retained_ho_data, "retained_ho") \
        if retained_ho_data else {"num_samples": 0, "mAP": float("nan"), "R1": float("nan"), "R5": float("nan")}
    print("\n=== Evaluating D_f (full train) ===")
    results["metrics"]["D_f"] = evaluate_subset(cfg, model, df_data, "D_f")

    if args.include_test:
        print("\n=== Evaluating test (query/gallery) ===")
        results["metrics"]["test"] = evaluate_test_split(cfg, model)

    print("\nSummary Table")
    print("{:<10} {:>8} {:>10} {:>10} {:>10}".format("Subset", "Size", "mAP", "R1", "R5"))
    print("-" * 52)
    for key in ("D_r", "retained_ho", "D_f\\D_r", "D_f"):
        r = results["metrics"][key]
        print(
            "{:<10} {:>8} {:>9.2f}% {:>9.2f}% {:>9.2f}%".format(
                key, r["num_samples"], r["mAP"], r["R1"], r["R5"]
            )
        )

    os.makedirs(args.output_dir, exist_ok=True)
    out_name = "partition_eval_{}_{}_seed{}.json".format(
        sanitize_tag(args.tag), args.mode, args.seed
    )
    out_path = os.path.join(args.output_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results to {}".format(out_path))


if __name__ == "__main__":
    main()
