"""Linear probe on hidden features (Pawelczyk ICLR'25, Deep Unlearning 2026).

Trains a linear classifier on intermediate transformer block features to distinguish
member (D_f) vs non-member (query). High AUC == residual leakage in features even
if output-level entropy MIA looks low.

Hooks into the ViT backbone via forward_features capture at blocks 6, 9, 11.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from config import cfg
from datasets import make_dataloader, make_dataloader_mia
from model import make_model
from utils.logger import setup_logger
from utils.parasitic_utils import set_parasitic_mode


class BlockFeatureHook:
    """Capture (CLS-token) feature from selected blocks."""
    def __init__(self):
        self.feats = {}

    def make_hook(self, name):
        def hook(_module, _inp, out):
            # out can be a tensor [B, T, D] or a tuple
            if isinstance(out, tuple):
                out = out[0]
            self.feats[name] = out[:, 0].detach().cpu()  # CLS token
        return hook


def attach_hooks(model, block_indices=(6, 9, 11)):
    hook = BlockFeatureHook()
    handles = []
    # Walk down .base until the ViT with .blocks is found, so a wrapper such as
    # eval_loft.LoftWrapper is transparent here. Its projection is applied to the
    # returned feature, after these hooks have already fired.
    node, blocks = model, None
    for _ in range(4):
        node = getattr(node, "base", None)
        if node is None:
            break
        blocks = getattr(node, "blocks", None)
        if blocks is not None:
            break
    if blocks is None:
        raise RuntimeError("Cannot find the ViT blocks under model.base[.base ...].")
    for idx in block_indices:
        if idx < len(blocks):
            h = blocks[idx].register_forward_hook(hook.make_hook(f"block{idx}"))
            handles.append(h)
    return hook, handles


@torch.no_grad()
def collect_block_feats(model, loader, hook, device="cuda"):
    feats_all = {k: [] for k in hook.feats.keys()} if hook.feats else None
    pids = []; cams = []
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
        hook.feats.clear()
        _ = model(img, cam_label=cam, view_label=view)
        if feats_all is None:
            feats_all = {k: [] for k in hook.feats.keys()}
        for k, v in hook.feats.items():
            feats_all[k].append(v)
        pids.append(pid.cpu() if torch.is_tensor(pid) else torch.tensor(pid))
    feats_all = {k: torch.cat(v, dim=0).numpy() for k, v in feats_all.items()}
    pids = torch.cat(pids, dim=0).numpy()
    return feats_all, pids


def linear_probe_auc(member_feat, nonmember_feat, member_pid=None, nonmember_pid=None,
                     n_folds=5, max_iter=2000):
    """Member vs non-member probe, cross-validated two ways.

    A ReID member pool and a query pool hold different identities, and every identity
    contributes several images. Splitting folds by image lets the same identity sit in
    the training and the held-out half at once, so the classifier can score by telling
    identities apart rather than by reading membership. Grouping the folds by identity
    removes that route. Both numbers are returned; the grouped one is what the paper
    reports, and the gap between them is the size of the confound.
    """
    X = np.concatenate([member_feat, nonmember_feat], axis=0).astype(np.float32)
    y = np.concatenate([np.ones(len(member_feat)), np.zeros(len(nonmember_feat))]).astype(np.int64)

    def _cv(splitter, **kw):
        aucs = []
        for tr, te in splitter.split(X, y, **kw):
            if len(np.unique(y[te])) < 2:
                continue
            clf = LogisticRegression(max_iter=max_iter, n_jobs=-1)
            clf.fit(X[tr], y[tr])
            prob = clf.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], prob))
        if not aucs:
            return float("nan"), float("nan")
        return float(np.mean(aucs)), float(np.std(aucs))

    auc_img, sd_img = _cv(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42))

    if member_pid is None or nonmember_pid is None:
        return auc_img, sd_img, {"grouped": False}

    # The two pools carry independent label spaces, so namespace them before grouping.
    groups = np.concatenate([
        np.asarray(["m%d" % int(v) for v in member_pid]),
        np.asarray(["q%d" % int(v) for v in nonmember_pid]),
    ])
    auc_grp, sd_grp = _cv(
        StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42), groups=groups)
    meta = {
        "grouped": True,
        "n_groups_member": int(len(np.unique(member_pid))),
        "n_groups_nonmember": int(len(np.unique(nonmember_pid))),
        "auc_by_image_fold": auc_img,
        "auc_std_by_image_fold": sd_img,
    }
    return auc_grp, sd_grp, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear probe MIA on transformer hidden features")
    parser.add_argument("--config_file", default="", type=str)
    parser.add_argument("--mode", default="core", choices=["full", "core", "standard"])
    parser.add_argument("--blocks", nargs="+", type=int, default=[5, 8, 10],
                        help="Block indices to probe. Note: in JPM mode only blocks 0-10 are run inside model.base; block 11 is handled separately.")
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--loft", default=None, type=str,
                        help="Optional LOFT projector loft_U.pt; wraps the model so the deployed "
                             "LOFT model is probed. Blocks are read before the projection.")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True) if output_dir else None
    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info("Linear probe. mode={} blocks={}".format(args.mode, args.blocks))

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

    hook, handles = attach_hooks(model, block_indices=tuple(args.blocks))

    logger.info("Extracting member features ...")
    t0 = time.time()
    m_feats, m_pids = collect_block_feats(model, member_loader, hook)
    logger.info("members done in {:.1f}s".format(time.time() - t0))

    logger.info("Extracting nonmember features ...")
    t0 = time.time()
    q_feats, q_pids = collect_block_feats(model, nonmember_loader, hook)
    logger.info("query done in {:.1f}s".format(time.time() - t0))

    for h in handles:
        h.remove()

    res = {}
    for layer in m_feats:
        auc, sd, gmeta = linear_probe_auc(m_feats[layer], q_feats[layer], m_pids, q_pids)
        res[layer] = {"auc_mean": auc, "auc_std": sd}
        res[layer].update(gmeta)
        logger.info("[{}] linear-probe AUC = {:.4f} ± {:.4f} (identity-grouped folds; "
                    "image-level folds gave {:.4f})".format(
                        layer, auc, sd, gmeta.get("auc_by_image_fold", float("nan"))))
    res["meta"] = {"target_ckpt": cfg.TEST.WEIGHT, "mode": args.mode, "blocks": args.blocks,
                   "cv": "StratifiedGroupKFold by identity"}

    out_json = args.out_json or os.path.join(output_dir, "linear_probe_{}.json".format(args.mode))
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
    logger.info("Saved {}".format(out_json))
