"""Advanced MIA suite: RMIA + Offline U-LiRA + ECE + TPR@low-FPR.

Responds to Reviewer Kd9X K3-K4 (Carlini 2022, Pawelczyk 2025, RMIA, Deep Unlearning 2026).

For each target checkpoint, computes:
  - per-sample loss on D_f (member) and query set (non-member)
  - RMIA score using K reference checkpoints (likelihood ratio)
  - U-LiRA z-score using OUT-distribution from Retrain seeds
  - ECE on test query set
  - AUC + TPR@{1%, 0.1%, 0.01%}FPR for entropy / RMIA / U-LiRA

Outputs JSON containing all scores + summary metrics.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

from config import cfg
from datasets import make_dataloader, make_dataloader_mia
from model import make_model
from utils.logger import setup_logger
from utils.parasitic_utils import set_parasitic_mode, build_cfg_for_ckpt


def per_sample_loss_xent(logits, target):
    """Per-sample cross-entropy. logits: [N, C], target: [N]."""
    if isinstance(logits, list):
        logits = logits[0]
    log_probs = F.log_softmax(logits, dim=-1)
    return -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)


def per_sample_selfloss(logits):
    """Label-free per-sample loss: cross-entropy against the model's own prediction.

    The membership pools are not label-commensurable. Members are training identities, whose
    classes exist in the head, while non-members are test identities, whose classes do not, so
    a true-label loss can only be computed on one side. Scoring the other side against a fixed
    dummy class compares two different quantities and biases the reference-based attacks away
    from their null. This statistic is defined identically on both pools.
    """
    if isinstance(logits, list):
        logits = logits[0]
    log_probs = F.log_softmax(logits, dim=-1)
    return -log_probs.max(dim=-1).values


def per_sample_entropy(logits):
    if isinstance(logits, list):
        logits = logits[0]
    probs = F.softmax(logits, dim=-1).clamp_min(1e-12)
    return -(probs * probs.log()).sum(dim=-1)


@torch.no_grad()
def collect_logits_and_pids(model, loader, device="cuda"):
    logits_list = []
    pids_list = []
    cams_list = []
    model.eval()
    for batch in loader:
        if len(batch) == 6:
            img, pid, camid, camids_bat, viewid, _paths = batch
        else:
            img, pid, camid, viewid, _paths = batch
            camids_bat = camid
        img = img.to(device)
        cam = camids_bat.to(device) if torch.is_tensor(camids_bat) else torch.tensor(camids_bat, device=device)
        view = viewid.to(device) if torch.is_tensor(viewid) else torch.tensor(viewid, device=device)
        feat = model(img, cam_label=cam, view_label=view)
        # Try to get logits via classifier
        classifier = getattr(model, "classifier", None)
        in_planes = getattr(model, "in_planes", feat.shape[1] if torch.is_tensor(feat) else 768)
        feat_for_cls = feat
        if torch.is_tensor(feat) and feat.shape[1] != in_planes and feat.shape[1] >= in_planes:
            feat_for_cls = feat[:, :in_planes]
        if classifier is None:
            raise RuntimeError("Model has no classifier; cannot compute loss-based MIA.")
        try:
            out = classifier(feat_for_cls)
        except (TypeError, RuntimeError):
            out = classifier(feat)
        logits = out[0] if isinstance(out, tuple) else out
        logits_list.append(logits.detach().cpu())
        if torch.is_tensor(pid):
            pids_list.append(pid.cpu())
        else:
            pids_list.append(torch.tensor(pid, dtype=torch.int64))
        if torch.is_tensor(camid):
            cams_list.append(camid.cpu() if camid.is_cuda else camid)
        else:
            cams_list.append(torch.tensor(camid, dtype=torch.int64))
    logits = torch.cat(logits_list, dim=0)
    pids = torch.cat(pids_list, dim=0)
    cams = torch.cat(cams_list, dim=0)
    return logits, pids, cams


def tpr_at_fpr(y_true, y_score, target_fprs=(0.01, 0.001, 0.0001)):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    out = {}
    for f in target_fprs:
        # Find first index where fpr >= target
        idx = np.searchsorted(fpr, f, side="right") - 1
        idx = max(0, min(idx, len(tpr) - 1))
        out[f"tpr_at_{f}"] = float(tpr[idx])
    return out


def expected_calibration_error(probs_top, correct, n_bins=15):
    """Standard ECE. probs_top: [N], correct: [N] in {0,1}."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(probs_top)
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (probs_top > lo) & (probs_top <= hi)
        if not mask.any():
            continue
        bin_acc = correct[mask].mean()
        bin_conf = probs_top[mask].mean()
        ece += (mask.sum() / N) * abs(bin_acc - bin_conf)
    return float(ece)


def evaluate_target(target_logits, member_pids, target_logits_q, query_pids,
                    ref_member_losses=None, ref_query_losses=None,
                    ref_member_selfloss=None, ref_query_selfloss=None):
    """Compute MIA scores on (D_f members, query nonmembers).

    target_logits: logits for D_f members [N_m, C]
    target_logits_q: logits for query nonmembers [N_q, C]
    ref_*_losses: list of (ref_name -> [N] loss tensor) for RMIA / U-LiRA

    Returns: dict with entropy, RMIA, U-LiRA scores, AUC, TPR@FPR, ECE.
    """
    res = {}

    # === entropy attack ===
    ent_m = -per_sample_entropy(target_logits).numpy()  # higher = more "member-like"
    ent_q = -per_sample_entropy(target_logits_q).numpy()
    y_true = np.concatenate([np.ones(len(ent_m)), np.zeros(len(ent_q))])
    y_score = np.concatenate([ent_m, ent_q])
    res["entropy"] = {"auc": roc_auc_score(y_true, y_score), **tpr_at_fpr(y_true, y_score)}

    # === per-sample CE loss attack (alternate) ===
    loss_m = per_sample_loss_xent(target_logits, member_pids).numpy()
    loss_q_proxy = per_sample_loss_xent(target_logits_q, torch.zeros_like(query_pids)).numpy()  # proxy: y=0
    # For loss-based MIA, lower loss = more member; flip sign.
    s_m = -loss_m; s_q = -loss_q_proxy
    y_score = np.concatenate([s_m, s_q])
    res["loss"] = {"auc": roc_auc_score(y_true, y_score), **tpr_at_fpr(y_true, y_score)}

    # === RMIA: ratio of target loss to mean reference loss ===
    if ref_member_losses is not None and ref_query_losses is not None:
        ref_m_mean = np.stack(ref_member_losses, axis=0).mean(axis=0)
        ref_q_mean = np.stack(ref_query_losses, axis=0).mean(axis=0)
        # RMIA score = ref_loss - target_loss (higher = more member-like)
        rmia_m = ref_m_mean - loss_m
        rmia_q = ref_q_mean - loss_q_proxy
        y_score = np.concatenate([rmia_m, rmia_q])
        res["rmia"] = {"auc": roc_auc_score(y_true, y_score), **tpr_at_fpr(y_true, y_score)}

        # === Offline U-LiRA: z-score via reference (OUT) distribution per-sample ===
        # OUT distribution: Retrain seeds → for member sample x, retrain models did not include x
        # u-LiRA z = (target_loss - μ_ref) / (σ_ref + eps) ; lower z (more negative) = member-like
        ref_m_stack = np.stack(ref_member_losses, axis=0)
        ref_q_stack = np.stack(ref_query_losses, axis=0)
        mu_m = ref_m_stack.mean(axis=0); sd_m = ref_m_stack.std(axis=0) + 1e-6
        mu_q = ref_q_stack.mean(axis=0); sd_q = ref_q_stack.std(axis=0) + 1e-6
        z_m = (loss_m - mu_m) / sd_m
        z_q = (loss_q_proxy - mu_q) / sd_q
        # member → loss < OUT → z < 0; flip to "score where higher = member-like"
        s_m = -z_m; s_q = -z_q
        y_score = np.concatenate([s_m, s_q])
        res["u_lira"] = {"auc": roc_auc_score(y_true, y_score), **tpr_at_fpr(y_true, y_score)}

    # === label-free variants: identical statistic on both pools ===
    if ref_member_selfloss is not None and ref_query_selfloss is not None:
        sl_m = per_sample_selfloss(target_logits).numpy()
        sl_q = per_sample_selfloss(target_logits_q).numpy()
        rm = np.stack(ref_member_selfloss, axis=0)
        rq = np.stack(ref_query_selfloss, axis=0)
        y_score = np.concatenate([rm.mean(axis=0) - sl_m, rq.mean(axis=0) - sl_q])
        res["rmia_lf"] = {"auc": roc_auc_score(y_true, y_score), **tpr_at_fpr(y_true, y_score)}
        zm = (sl_m - rm.mean(axis=0)) / (rm.std(axis=0) + 1e-6)
        zq = (sl_q - rq.mean(axis=0)) / (rq.std(axis=0) + 1e-6)
        y_score = np.concatenate([-zm, -zq])
        res["u_lira_lf"] = {"auc": roc_auc_score(y_true, y_score), **tpr_at_fpr(y_true, y_score)}
        res["selfloss_raw"] = {"auc": roc_auc_score(y_true, np.concatenate([-sl_m, -sl_q]))}

    # Keep the per-sample scores. Identity-clustered confidence intervals, equivalence tests
    # against the reference row and the actual false-positive count behind a low-FPR threshold
    # all need the scores themselves, not the summary.
    res["_scores"] = {"entropy": np.concatenate([ent_m, ent_q]).astype(np.float32)}
    if ref_member_losses is not None:
        res["_scores"]["rmia"] = np.concatenate([rmia_m, rmia_q]).astype(np.float32)
        res["_scores"]["u_lira"] = np.concatenate([-z_m, -z_q]).astype(np.float32)
    if ref_member_selfloss is not None:
        res["_scores"]["rmia_lf"] = np.concatenate([rm.mean(axis=0) - sl_m,
                                                    rq.mean(axis=0) - sl_q]).astype(np.float32)
        res["_scores"]["u_lira_lf"] = np.concatenate([-zm, -zq]).astype(np.float32)
    res["_scores"]["y_true"] = y_true.astype(np.int8)

    # === ECE on query (reflect overconfidence bias) ===
    probs = F.softmax(target_logits_q, dim=-1)
    top, pred = probs.max(dim=-1)
    correct = (pred == query_pids).numpy().astype(np.float32)
    res["ece_query"] = expected_calibration_error(top.numpy(), correct)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced MIA suite (RMIA / U-LiRA / ECE / TPR@FPR)")
    parser.add_argument("--config_file", default="", type=str)
    parser.add_argument("--mode", default="core", choices=["full", "core", "standard"])
    parser.add_argument("--ref_opts", default=None, type=str,
                        help="Config overrides applied to the reference models' dataloader only, "
                             "as a space separated KEY VALUE list. The target and the references "
                             "were not always trained under the same input normalization, and an "
                             "attack compares each model's own loss, so each has to be run the way "
                             "it would be deployed. Sample order is fixed, shuffle is off on both "
                             "loaders, so the two passes align by index.")
    parser.add_argument("--dump_scores", default=None, type=str,
                        help="Write the per-sample attack scores, the membership labels and the "
                             "identity of every sample to this .npz. Needed for identity-clustered "
                             "intervals and for equivalence tests against the reference row.")
    parser.add_argument("--ref_ckpts", default=None, type=str,
                        help="Reference (OUT) checkpoints for RMIA / U-LiRA, comma-separated.")
    parser.add_argument("--ref_modes", default=None, type=str,
                        help="Per-ref modes (full/core), comma-separated. Defaults to 'core' for each.")
    parser.add_argument("--out_json", default=None, help="Output JSON path. Defaults to OUTPUT_DIR/advanced_mia.json")
    parser.add_argument("--loft", default=None, type=str,
                        help="Optional path to a LOFT projector loft_U.pt. Applies P=UU^T to the "
                             "TARGET's embedding only; references stay unprojected, since LOFT is "
                             "the unlearning operation applied to the target.")
    parser.add_argument("--skew", default="",
                        choices=["", "camera_skew", "identity_skew", "imbalanced_k"],
                        help="take the member set from this skewed split instead of the "
                             "identity-level one, so tab:skewed can carry calibrated attacks. "
                             "The split seed is MOME.SPLIT_SEED, so target and references see "
                             "the same U.")
    parser.add_argument("--n_forget_pids", default=50, type=int)
    parser.add_argument("--forget_camid", default=0, type=int)
    parser.add_argument("--dr_per_pid", default=4, type=int)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True) if output_dir else None
    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info("Advanced MIA. mode={} ref_ckpts={}".format(args.mode, args.ref_ckpts))

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID
    _, _, _, _, num_classes, camera_num, view_num = make_dataloader(cfg)
    _skw = dict(n_forget_pids=args.n_forget_pids, forget_camid=args.forget_camid,
                dr_per_pid=args.dr_per_pid) if args.skew else {}
    member_loader, nonmember_loader, _ = make_dataloader_mia(
        cfg, skew=(args.skew or None), **_skw)

    # Load target. The build has to follow the checkpoint for the same reason the references do:
    # the consolidated Full artifact carries full-capacity Linear shadows and embed shadows, and
    # a config-driven build raised KeyError on base.pos_embed_weight_m.
    target = make_model(build_cfg_for_ckpt(cfg, cfg.TEST.WEIGHT),
                        num_class=num_classes, camera_num=camera_num, view_num=view_num)
    target.load_param(cfg.TEST.WEIGHT)
    target = target.cuda(); target.eval()
    # "standard" keeps the branch ON: non-MoME checkpoints were trained by the ordinary
    # entrypoints, where the parasitic tensors are ordinary trainable parameters.
    set_parasitic_mode(target, mode=(args.mode != "core"))
    if args.loft:
        from eval_loft import LoftWrapper
        payload = torch.load(args.loft, map_location="cpu")
        U = payload["U"].float()
        P = (U @ U.t()).float()
        logger.info("LOFT projector {}: U {}x{} (s={}), projecting TARGET embedding only".format(
            args.loft, U.size(0), U.size(1), payload.get("s")))
        target = LoftWrapper(target, P.cuda()).cuda()
        target.eval()

    logger.info("Collecting target logits ...")
    t0 = time.time()
    m_logits, m_pids, _ = collect_logits_and_pids(target, member_loader)
    q_logits, q_pids, _ = collect_logits_and_pids(target, nonmember_loader)
    logger.info("Target collected in {:.1f}s. members={} query={}".format(time.time() - t0, len(m_pids), len(q_pids)))
    del target
    torch.cuda.empty_cache()

    # Optional reference forward passes
    ref_member_loader, ref_nonmember_loader = member_loader, nonmember_loader
    if args.ref_opts:
        from config import cfg as _cfg_proto
        rcfg = _cfg_proto.clone()
        rcfg.defrost()
        if args.config_file:
            rcfg.merge_from_file(args.config_file)
        rcfg.merge_from_list(args.opts)
        rcfg.merge_from_list(args.ref_opts.split())
        rcfg.freeze()
        logger.info("Reference dataloader overrides: %s", args.ref_opts)
        ref_member_loader, ref_nonmember_loader, _ = make_dataloader_mia(rcfg)
        assert len(ref_member_loader.dataset) == len(member_loader.dataset), \
            "reference loader must hold the same samples in the same order"
    ref_m_losses = []; ref_q_losses = []
    ref_m_self = []; ref_q_self = []
    ref_ckpts_list = [c.strip() for c in args.ref_ckpts.split(",") if c.strip()] if args.ref_ckpts else []
    ref_modes_list = [m.strip() for m in args.ref_modes.split(",") if m.strip()] if args.ref_modes else (["core"] * len(ref_ckpts_list))
    if ref_ckpts_list:
        for i, ck in enumerate(ref_ckpts_list):
            mode = ref_modes_list[i] if i < len(ref_modes_list) else "core"
            logger.info("Reference [{}] mode={}: {}".format(i, mode, ck))
            # References may be plain retrains (no LoRA branch) or co-trained MoME cores
            # (LoRA branch present). The build must match the checkpoint: a LoRA build renames
            # every wrapped Linear parameter to ...base.weight and adds weight_m_A/B, so a
            # mismatch in either direction raises KeyError on the first block. Detect from the
            # checkpoint's own keys.
            ref_cfg = build_cfg_for_ckpt(cfg, ck)
            ref = make_model(ref_cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
            ref.load_param(ck)
            ref = ref.cuda(); ref.eval()
            set_parasitic_mode(ref, mode=(mode == "full"))
            r_m_logits, r_m_pids, _ = collect_logits_and_pids(ref, ref_member_loader)
            r_q_logits, r_q_pids, _ = collect_logits_and_pids(ref, ref_nonmember_loader)
            ref_m_losses.append(per_sample_loss_xent(r_m_logits, r_m_pids).numpy())
            ref_q_losses.append(per_sample_loss_xent(r_q_logits, torch.zeros_like(r_q_pids)).numpy())
            ref_m_self.append(per_sample_selfloss(r_m_logits).numpy())
            ref_q_self.append(per_sample_selfloss(r_q_logits).numpy())
            del ref; torch.cuda.empty_cache()

    res = evaluate_target(m_logits, m_pids, q_logits, q_pids,
                          ref_member_losses=ref_m_losses or None,
                          ref_query_losses=ref_q_losses or None,
                          ref_member_selfloss=ref_m_self or None,
                          ref_query_selfloss=ref_q_self or None)
    scores = res.pop("_scores", None)
    if scores is not None and args.dump_scores:
        np.savez_compressed(
            args.dump_scores,
            pid=np.concatenate([m_pids.numpy(), q_pids.numpy()]).astype(np.int64),
            pool=np.concatenate([np.ones(len(m_pids)), np.zeros(len(q_pids))]).astype(np.int8),
            target_ckpt=np.array(cfg.TEST.WEIGHT), **scores)
        logger.info("Saved per-sample scores to {}".format(args.dump_scores))
    res["meta"] = {
        "target_ckpt": cfg.TEST.WEIGHT,
        "mode": args.mode,
        "n_member": int(len(m_pids)),
        "n_query": int(len(q_pids)),
        "n_refs": len(ref_m_losses),
        "loft": args.loft,
    }
    out_json = args.out_json or os.path.join(output_dir, "advanced_mia_{}.json".format(args.mode))
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
    logger.info("Saved {}".format(out_json))
    logger.info("Summary: {}".format(json.dumps({k: v for k, v in res.items() if k != "meta"}, indent=2)))
