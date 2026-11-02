"""LOFT unlearning baseline (Fang et al., 2026, arXiv:2601.22456,
"Machine Unlearning in Low-Dimensional Feature Subspace").

Faithful re-implementation for the TransReID ReID setting. LOFT freezes the entire
pretrained model and learns a single d x s orthonormal projector U (Stiefel manifold)
on the FROZEN backbone features, one-shot from two feature covariances:

    min_{U in St(d,s)}  J(U) = ( Tr(U^T Sigma_fg U) / Tr(Sigma_fg) )^2
                             + ( Tr(Sigma_rm - U U^T Sigma_rm U U^T) / Tr(Sigma_rm) )^2   (Eq. 2)

where Sigma_fg, Sigma_rm are covariances of mean-centered frozen embeddings of the forget
set D_f\\D_r and the retain set D_r. J_fg minimizes forget-feature variance inside range(U);
J_rm is the PCA reconstruction error of the remaining features (maximizes retain info kept).
Using the projector identity P=UU^T, P^2=P  =>  Tr(P Sigma P)=Tr(U^T Sigma U), so
    J_rm = ( (Tr(Sigma_rm) - Tr(U^T Sigma_rm U)) / Tr(Sigma_rm) )^2.
There is no lambda weighting the two terms; the per-term trace normalization balances them.

Optimization: Riemannian Adam on the Stiefel manifold (geoopt if available; otherwise
Euclidean Adam + QR retraction), weight_decay 0.05, constant lr, 50-100 full-batch steps.
s := smallest dimension whose cumulative eigen-energy of Sigma_rm >= 0.95 (paper default).

ReID adaptation (documented deviation, forced by the retrieval setting): the paper applies
the projector before the classifier, f_U = h_pre(UU^T g_pre). TransReID retrieval is
distance-based on the embedding with no test-time classifier, so at eval we apply UU^T to the
retrieval embedding (equivalently store the compact s-d U^T z -- identical rankings). U is
learned on the eval-time retrieval embedding of the frozen standard model.

This script only trains + saves the projector U (loft_U.pt). Evaluation (project embeddings,
then mAP/CMC + MIA) is done by eval_loft.py.
"""

import argparse
import os
import random
import time

import numpy as np
import torch

from config import cfg
from datasets import make_dataloader_mome
from model import make_model
from utils.logger import setup_logger
from utils.parasitic_utils import freeze_parasitic_parameters, set_parasitic_mode


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def extract_embeddings(model, loader, device):
    """Frozen-backbone retrieval embeddings (eval-mode forward -> feat)."""
    model.eval()
    feats = []
    for img, pid, cam, view in loader:
        img = img.to(device); cam = cam.to(device); view = view.to(device)
        f = model(img, cam_label=cam, view_label=view)
        if isinstance(f, (list, tuple)):
            f = f[0]
        feats.append(f.detach().float().cpu())
    return torch.cat(feats, 0)


def covariance(Z):
    """Mean-centered covariance (1/N Z_bar^T Z_bar) of rows-are-samples matrix Z (N x d)."""
    Zc = Z - Z.mean(dim=0, keepdim=True)
    return (Zc.t() @ Zc) / Zc.size(0)


def choose_s(Sigma_rm, energy=0.95):
    evals = torch.linalg.eigvalsh(Sigma_rm)          # ascending
    evals = torch.clamp(evals.flip(0), min=0)         # descending, non-negative
    csum = torch.cumsum(evals, 0) / (evals.sum() + 1e-12)
    s = int(torch.searchsorted(csum, torch.tensor(energy)).item()) + 1
    return max(1, min(s, Sigma_rm.size(0) - 1))


def loft_objective(U, Sfg, Srm, tr_fg, tr_rm):
    # Tr(U^T S U) = sum_k U[:,k]^T S U[:,k] = sum(U * (S @ U))
    tr_ufg = (U * (Sfg @ U)).sum()
    tr_urm = (U * (Srm @ U)).sum()
    Jfg = (tr_ufg / tr_fg) ** 2
    Jrm = ((tr_rm - tr_urm) / tr_rm) ** 2
    return Jfg + Jrm, Jfg.item(), Jrm.item()


def optimize_U(Sfg, Srm, s, steps, lr, wd, logger, device):
    d = Sfg.size(0)
    tr_fg = torch.trace(Sfg).clamp_min(1e-12)
    tr_rm = torch.trace(Srm).clamp_min(1e-12)
    # Init: top-s eigenvectors of Sigma_rm (already minimizes J_rm; sensible warm start).
    evals, evecs = torch.linalg.eigh(Srm)
    U0 = evecs[:, -s:].contiguous()

    use_geoopt = False
    try:
        import geoopt
        manifold = geoopt.Stiefel()
        U = geoopt.ManifoldParameter(U0.clone(), manifold=manifold)
        opt = geoopt.optim.RiemannianAdam([U], lr=lr, weight_decay=wd)
        use_geoopt = True
        logger.info("Using geoopt RiemannianAdam on Stiefel manifold.")
    except Exception as e:
        U = torch.nn.Parameter(U0.clone())
        opt = torch.optim.Adam([U], lr=lr, weight_decay=wd)
        logger.info("geoopt unavailable ({}); using Euclidean Adam + QR retraction.".format(e))

    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        J, jfg, jrm = loft_objective(U, Sfg, Srm, tr_fg, tr_rm)
        J.backward()
        opt.step()
        if not use_geoopt:  # retract back to the Stiefel manifold
            with torch.no_grad():
                q, r = torch.linalg.qr(U)
                q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)  # sign-consistent
                U.copy_(q)
        if step == 1 or step % 10 == 0 or step == steps:
            logger.info("LOFT step[{}/{}] J={:.5f} (J_fg={:.5f} J_rm={:.5f})".format(
                step, steps, J.item(), jfg, jrm))
    return U.detach().cpu()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LOFT unlearning baseline (2026)")
    parser.add_argument("--config_file", default="", type=str)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--lr", default=1.0, type=float, help="Stiefel lr (paper: 1 or 10)")
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--energy", default=0.95, type=float, help="cumulative eigen-energy for s")
    parser.add_argument("--s", default=0, type=int, help="override subspace dim (0 = auto by energy)")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_seed(cfg.SOLVER.SEED)

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("LOFT steps={} lr={} wd={} energy={}".format(
        args.steps, args.lr, args.weight_decay, args.energy))

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

    (
        _train_loader_f,
        train_loader_r,
        train_loader_df_only,
        _train_loader_normal,
        _val_loader,
        _num_query,
        num_classes,
        camera_num,
        view_num,
        split_info,
    ) = make_dataloader_mome(cfg)
    logger.info("Split: {}".format(split_info))

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    if not cfg.TEST.WEIGHT:
        raise ValueError("LOFT is post-hoc; pass TEST.WEIGHT pointing to the standard ckpt.")
    model.load_param(cfg.TEST.WEIGHT)
    logger.info("Loaded standard ckpt: {}".format(cfg.TEST.WEIGHT))
    model = model.cuda()
    set_parasitic_mode(model, False)
    freeze_parasitic_parameters(model, freeze=True, zero_fill=True)
    for p in model.parameters():
        p.requires_grad_(False)

    device = "cuda"
    t0 = time.time()
    logger.info("Extracting frozen embeddings (one-shot) ...")
    Z_rm = extract_embeddings(model, train_loader_r, device)
    Z_fg = extract_embeddings(model, train_loader_df_only, device)
    d = Z_rm.size(1)
    logger.info("Embeddings: retain {} x {}, forget {} x {}. ({:.1f}s)".format(
        Z_rm.size(0), d, Z_fg.size(0), Z_fg.size(1), time.time() - t0))

    Srm = covariance(Z_rm).to(device)
    Sfg = covariance(Z_fg).to(device)
    s = args.s if args.s > 0 else choose_s(Srm.cpu(), args.energy)
    logger.info("Subspace dim s={} (d={}, energy>={})".format(s, d, args.energy))

    U = optimize_U(Sfg, Srm, s, args.steps, args.lr, args.weight_decay, logger, device)

    save_path = os.path.join(output_dir, "loft_U.pt")
    torch.save({"U": U, "s": s, "d": d, "src_ckpt": cfg.TEST.WEIGHT}, save_path)
    logger.info("Saved LOFT projector: {} (U: {} x {})".format(save_path, U.size(0), U.size(1)))
