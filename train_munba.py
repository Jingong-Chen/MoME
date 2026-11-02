"""MUNBa unlearning baseline (Wu & Harandi, ICCV'25, arXiv:2411.15537).

Faithful port of github.com/JingWu321/MUNBa (Classification variant, unlearn/MUNBa.py) to
the TransReID ReID setting. Machine unlearning as a two-player Nash bargaining game:

  * preservation player (retain): loss_r = standard ReID loss on D_r (5-head CE + triplet,
    TRUE labels)  -> gradient g_r over all core parameters.
  * forgetting player (forget):  loss_u = mean 5-head CE on D_f\\D_r with RANDOM ID labels
    -> gradient g_u.  (The repo's gradient-ascent form `-CE(true)` is commented out; the
    shipped variant uses random relabeling, which we follow.)

Closed-form K=2 bargaining solution (MUNBa.py:199-213), with g1=<g_r,g_r>, g2=<g_r,g_u>,
g3=<g_u,g_u>:
    a_r = sqrt( (g1*g3 - g2*sqrt(g1*g3)) / (g1*g1*g3 - g1*g2*g2 + 1e-8) )
    a_u = (1 - g1*a_r*a_r) / (g2*a_r + 1e-8)
    if a_r>0 and a_u>0:  loss = a_r*loss_r + a_u*loss_u        (bargaining succeeded)
    else:                loss = loss_r + 0.1*loss_u            (fallback; e.g. g2<0 -> NaN)
Then loss.backward(); clip_grad_norm_(1.0); SGD step. Bargaining recomputed every batch.
Defaults: SGD lr=0.01, momentum=0.9, weight_decay=5e-4, 10 epochs (arg_parser.py). Optimizes
the core parameters (parasitic branch frozen -> Core mode), full weights. The cvxpy/ECOS
solver in the repo is dead code (never called); the K=2 closed form above is what runs.
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from config import cfg
from datasets import make_dataloader_mome
from loss import make_loss
from model import make_model
from utils.logger import setup_logger
from utils.meter import AverageMeter
from utils.parasitic_utils import freeze_parasitic_parameters, set_parasitic_mode


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def _cycle_next(loader_iter, loader):
    try:
        return next(loader_iter), loader_iter
    except StopIteration:
        loader_iter = iter(loader)
        return next(loader_iter), loader_iter


def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _flat_grad(loss, params):
    """Flattened gradient of `loss` w.r.t. params. retain_graph=False frees the graph
    immediately, so only one forward graph is alive at a time (memory ~ standard training)."""
    grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
    flat = []
    for g, p in zip(grads, params):
        flat.append(g.detach().flatten() if g is not None else torch.zeros_like(p).flatten())
    return torch.cat(flat)


def nash_weights(g_r, g_u):
    """Closed-form K=2 Nash bargaining weights (MUNBa.py:199-213)."""
    g1 = torch.dot(g_r, g_r)
    g2 = torch.dot(g_r, g_u)
    g3 = torch.dot(g_u, g_u)
    a_r = torch.sqrt((g1 * g3 - g2 * torch.sqrt(g1 * g3)) / (g1 * g1 * g3 - g1 * g2 * g2 + 1e-8))
    a_u = (1.0 - g1 * a_r * a_r) / (g2 * a_r + 1e-8)
    return float(a_r), float(a_u), float(g2)


def run_munba(model, loader_r, loader_df_only, optimizer, loss_fn, epochs,
              num_classes, logger, device, max_grad_norm=1.0):
    params = [p for p in model.parameters() if p.requires_grad]
    sizes = [p.numel() for p in params]
    model.train()
    for ep in range(1, epochs + 1):
        m_r = AverageMeter(); m_u = AverageMeter(); m_ar = AverageMeter(); m_au = AverageMeter()
        n_fallback = 0; skipped = 0; t0 = time.time()
        df_iter = iter(loader_df_only)
        for img_r, pid_r, cam_r, view_r in loader_r:
            (batch_f, df_iter) = _cycle_next(df_iter, loader_df_only)
            img_f, pid_f, cam_f, view_f = batch_f
            img_r = img_r.to(device); pid_r = pid_r.to(device); cam_r = cam_r.to(device); view_r = view_r.to(device)
            img_f = img_f.to(device); cam_f = cam_f.to(device); view_f = view_f.to(device)
            rnd_pid = torch.randint(0, num_classes, (img_f.size(0),), device=device)

            # Preservation player (retain, true labels): full ReID loss -> g_r (graph freed).
            score_r, feat_r = model(img_r, pid_r, cam_label=cam_r, view_label=view_r)
            loss_r = loss_fn(score_r, feat_r, pid_r, cam_r)
            if not torch.isfinite(loss_r):
                skipped += 1; continue
            loss_r_val = loss_r.item()
            g_r = _flat_grad(loss_r, params)
            del score_r, feat_r, loss_r

            # Forgetting player (forget, random labels): mean 5-head CE -> g_u (graph freed).
            score_f, _feat_f = model(img_f, rnd_pid, cam_label=cam_f, view_label=view_f)
            s_list = _as_list(score_f)
            loss_u = sum(F.cross_entropy(s, rnd_pid) for s in s_list) / len(s_list)
            if not torch.isfinite(loss_u):
                skipped += 1; del g_r; continue
            loss_u_val = loss_u.item()
            g_u = _flat_grad(loss_u, params)
            del score_f, _feat_f, loss_u

            a_r, a_u, g2 = nash_weights(g_r, g_u)
            if not (np.isfinite(a_r) and np.isfinite(a_u) and a_r > 0 and a_u > 0):
                a_r, a_u = 1.0, 0.1  # bargaining failed (e.g. g2<0 -> NaN): fixed fallback
                n_fallback += 1

            # Manual assignment of the combined update d = a_r*g_r + a_u*g_u (no 3rd backward).
            g_comb = a_r * g_r + a_u * g_u
            del g_r, g_u
            optimizer.zero_grad(set_to_none=True)
            idx = 0
            for p, n in zip(params, sizes):
                p.grad = g_comb[idx:idx + n].view_as(p).clone()
                idx += n
            del g_comb
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()

            m_r.update(loss_r_val); m_u.update(loss_u_val)
            m_ar.update(a_r); m_au.update(a_u)
        logger.info("MUNBa ep[{}/{}] loss_r:{:.4f} loss_u:{:.4f} a_r:{:.3f} a_u:{:.3f} "
                    "fallback:{} skip:{} time:{:.1f}s".format(
                        ep, epochs, m_r.avg, m_u.avg, m_ar.avg, m_au.avg,
                        n_fallback, skipped, time.time() - t0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MUNBa unlearning baseline (ICCV'25)")
    parser.add_argument("--config_file", default="", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=0.005, type=float, help="SGD lr (paper: 0.01)")
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
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
    logger.info("MUNBa epochs={} lr={} momentum={} wd={}".format(
        args.epochs, args.lr, args.momentum, args.weight_decay))
    logger.info("Config:\n{}".format(cfg))

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
        raise ValueError("MUNBa is post-hoc; pass TEST.WEIGHT pointing to the standard ckpt.")
    model.load_param(cfg.TEST.WEIGHT)
    logger.info("Loaded standard ckpt: {}".format(cfg.TEST.WEIGHT))
    model = model.cuda()
    set_parasitic_mode(model, False)
    freeze_parasitic_parameters(model, freeze=True, zero_fill=True)

    loss_fn, _center = make_loss(cfg, num_classes=num_classes)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    run_munba(model, train_loader_r, train_loader_df_only, optimizer, loss_fn,
              args.epochs, num_classes, logger, "cuda", args.max_grad_norm)

    save_path = os.path.join(output_dir, "munba_final.pth")
    torch.save(model.state_dict(), save_path)
    logger.info("Saved MUNBa checkpoint: {}".format(save_path))
