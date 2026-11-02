import logging
import os
import time
import math

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp

from loss.softmax_loss import CrossEntropyLabelSmooth
from loss.triplet_loss import TripletLoss
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from utils.parasitic_utils import set_parasitic_mode, get_parasitic_parameters


def _is_parasitic_param(name):
    return ("weight_m" in name) or ("bias_m" in name)


def freeze_parasitic_params(model):
    """Freeze parasitic parameters (weight_m, bias_m)."""
    for name, param in model.named_parameters():
        if _is_parasitic_param(name):
            param.requires_grad_(False)


def unfreeze_parasitic_params(model):
    """Unfreeze parasitic parameters (weight_m, bias_m)."""
    for name, param in model.named_parameters():
        if _is_parasitic_param(name):
            param.requires_grad_(True)


def freeze_core_params(model):
    """Freeze all non-parasitic parameters."""
    for name, param in model.named_parameters():
        if not _is_parasitic_param(name):
            param.requires_grad_(False)


def unfreeze_core_params(model):
    """Unfreeze all non-parasitic parameters."""
    for name, param in model.named_parameters():
        if not _is_parasitic_param(name):
            param.requires_grad_(True)


def set_bn_train_mode(model, train_mode):
    """Toggle only BatchNorm modules between train/eval without touching other modules."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            module.train(train_mode)


def _restrict_to_retained(score, target, retain_idx, label_map):
    """Give Step 1 a classifier over the retained identities only.

    Under identity-level deletion a forgotten identity never appears in a Step 1 batch, so
    keeping its column in the softmax lets the normalizer alone drive its logit down without
    bound, which is what made training diverge. Slicing the columns away means Core holds no
    trained parameter for a forgotten identity at all; the identity is carried solely by the
    parasitic rows that Step 2 trains, which is what makes it removable.
    """
    if retain_idx is None:
        return score, target
    t = label_map[target]
    if isinstance(score, list):
        return [x[:, retain_idx] for x in score], t
    return score[:, retain_idx], t


def _compute_id_loss(score, target, xent):
    if isinstance(score, list):
        if len(score) == 1:
            return xent(score[0], target)
        id_loss = [xent(s, target) for s in score[1:]]
        id_loss = sum(id_loss) / len(id_loss)
        return 0.5 * id_loss + 0.5 * xent(score[0], target)
    return xent(score, target)


def _compute_triplet_loss(feat, target, triplet):
    if isinstance(feat, list):
        if len(feat) == 1:
            return triplet(feat[0], target)[0]
        tri_loss = [triplet(f, target)[0] for f in feat[1:]]
        tri_loss = sum(tri_loss) / len(tri_loss)
        return 0.5 * tri_loss + 0.5 * triplet(feat[0], target)[0]
    return triplet(feat, target)[0]


def _pick_main_score(score):
    if isinstance(score, list):
        return score[0]
    return score


def _cycle_next(loader_iter, loader):
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
    return batch, loader_iter


class CombinedLoss(nn.Module):
    """Standard ReID loss: ID loss + Triplet loss."""

    def __init__(self, cfg, num_classes):
        super(CombinedLoss, self).__init__()
        self.id_loss_weight = cfg.MODEL.ID_LOSS_WEIGHT
        self.triplet_loss_weight = cfg.MODEL.TRIPLET_LOSS_WEIGHT
        if cfg.MODEL.IF_LABELSMOOTH == "on":
            self.xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        else:
            self.xent = F.cross_entropy
        if cfg.MODEL.NO_MARGIN:
            self.triplet = TripletLoss()
        else:
            self.triplet = TripletLoss(cfg.SOLVER.MARGIN)

    def forward(self, score, feat, target):
        id_loss = _compute_id_loss(score, target, self.xent)
        tri_loss = _compute_triplet_loss(feat, target, self.triplet)
        return self.id_loss_weight * id_loss + self.triplet_loss_weight * tri_loss


def _evaluate_reid_split(cfg, model, loader, num_query, mode_name, split_name):
    device = "cuda"
    logger = logging.getLogger("transreid.mome_v3.train")
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()
    model.eval()

    raw_model = model.module if hasattr(model, "module") else model
    set_parasitic_mode(raw_model, mode=(mode_name == "full"))

    for img, vid, camid, camids, target_view, _ in loader:
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, vid, camid))

    try:
        cmc, mAP, _, _, _, _, _ = evaluator.compute()
    except AssertionError as exc:
        # Some splits (e.g. single-camera train-set self retrieval) can have
        # no valid cross-camera matches under Market-style filtering.
        logger.warning(
            "[%s][%s] evaluation skipped: %s",
            mode_name.upper(),
            split_name,
            str(exc),
        )
        return None, float("nan")
    if (not cfg.MODEL.DIST_TRAIN) or dist.get_rank() == 0:
        logger.info(
            "[{}][{}] mAP: {:.1%}, Rank-1: {:.1%}, Rank-5: {:.1%}".format(
                mode_name.upper(), split_name, mAP, cmc[0], cmc[4]
            )
        )
    return cmc, mAP



def _relational_kd(fs, ft):
    """Match the batch's pairwise similarity structure instead of the features themselves.
    Feature-MSE fails for retrieval: a 0.955-cosine approximation sounds close but ReID ranking
    margins are 0.01-0.05, so the residual swamps them -- both the post-hoc and the joint
    feature-MSE distillations plateaued at the same ~56 mAP. Retrieval is invariant to a global
    rotation of the embedding, and so is this objective; it asks the branch only for what
    retrieval actually uses, which is also what a low-capacity adapter can express."""
    return (fs @ fs.t() - ft @ ft.t()).pow(2).mean()


class _HintTap:
    """Forward hooks on selected backbone blocks that capture outputs only while armed, so the
    taps cost nothing during the task forwards. The JPM variant runs blocks[:-1] in the shared
    path, so hint layers must stay in [0, 10] for ViT-B."""

    def __init__(self, model, layers):
        self.captured = {}
        self.armed = False
        self.handles = []
        blocks = model.base.blocks
        for i in layers:
            self.handles.append(blocks[i].register_forward_hook(self._make(i)))

    def _make(self, i):
        def hook(_m, _inp, out):
            if self.armed:
                self.captured[i] = out
        return hook


def _teacher_train_logits(teacher, img, pid, cam, view):
    """Train-mode control flow yields the per-head score list; every stochastic or
    stats-updating submodule is forced back to eval so the targets stay deterministic and the
    teacher's BN buffers stay untouched."""
    from model.backbones.vit_pytorch import DropPath
    teacher.train()
    for m in teacher.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.modules.batchnorm._BatchNorm, DropPath)):
            m.eval()
    with torch.no_grad():
        out = teacher(img, pid, cam_label=cam, view_label=view)
    teacher.eval()
    return out[0] if isinstance(out, tuple) else out


def _make_teacher_renorm(cfg):
    """Per-channel affine that maps pipeline-normalized inputs to the teacher's own
    normalization. Returns None when the knobs are unset, keeping every earlier run exact."""
    tm = list(getattr(cfg.MOME, "KD_TEACHER_MEAN", []) or [])
    ts = list(getattr(cfg.MOME, "KD_TEACHER_STD", []) or [])
    if len(tm) != 3 or len(ts) != 3:
        return None
    pm = torch.tensor(cfg.INPUT.PIXEL_MEAN).view(1, 3, 1, 1)
    ps = torch.tensor(cfg.INPUT.PIXEL_STD).view(1, 3, 1, 1)
    tm = torch.tensor(tm).view(1, 3, 1, 1)
    ts = torch.tensor(ts).view(1, 3, 1, 1)
    # x_raw = x * ps + pm ; x_teacher = (x_raw - tm) / ts = x * (ps/ts) + (pm - tm)/ts
    scale = (ps / ts).cuda()
    shift = ((pm - tm) / ts).cuda()

    def renorm(x):
        return x * scale + shift
    return renorm


def _distill_feature(model_or_teacher, img, cam, view, grad):
    """Eval-style forward for the concatenated retrieval feature. eval() only switches BN and
    dropout behaviour; gradients still flow when grad=True, and the shared BN buffers stay
    untouched by D_f."""
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        f = model_or_teacher(img, cam_label=cam, view_label=view)
    return torch.nn.functional.normalize(f.float(), dim=1)


def do_train_v3(
    cfg,
    model,
    train_loader_dr,
    train_loader_df,
    train_eval_loader,
    train_num_query,
    val_loader,
    optimizer,
    scheduler,
    loss_fn,
    num_query,
    local_rank=0,
    retain_idx=None,
    label_map=None,
    teacher=None):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    epochs = cfg.SOLVER.MAX_EPOCHS
    _no_isolation = bool(getattr(cfg.MOME, "NO_ISOLATION", False))
    device = "cuda"

    logger = logging.getLogger("transreid.mome_v3.train")
    if _no_isolation:
        logger.warning("NO_ISOLATION is on: the core will receive gradient from D_f. Ablation only.")
    logger.info("start MoME v3 training (gradient isolation)")

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            logger.info("Using {} GPUs for training".format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], find_unused_parameters=True
            )

    # Conservative AMP scaling avoids persistent overflow in the two-step
    # backward routine (core then parasitic) used by MoME v3.
    # YACS hands through the literal including its quotes, so "'fp32'" never matched the
    # comparison below and that control silently ran as fp16 instead.
    _amp_name = str(getattr(cfg.MOME, 'AMP_DTYPE', 'bf16')).strip().strip("'\"").lower()
    _amp_on = _amp_name not in ('fp32', 'float32', 'off', 'none')
    _amp_dtype = torch.bfloat16 if _amp_name in ('bf16', 'bfloat16') else torch.float16
    # A default-initialised scaler can back off from an overflow; init_scale=1.0 with a
    # floor of 1.0 cannot, which is what froze training for 28 epochs previously.
    # The scaler is shared state between Step 1 and Step 2: one loss scale, and one global
    # inf check inside GradScaler.step that skips every parameter at once. Under bf16 no loss
    # scaling is needed, so the scaler is disabled outright and that state stops existing.
    # Scaling by a power of two and unscaling by the same power of two is exact in fp32, so
    # this reproduces the earlier bf16 runs bit for bit.
    _amp_fp16 = _amp_dtype is torch.float16
    scaler = amp.GradScaler(enabled=_amp_fp16)
    if _amp_fp16 and not bool(getattr(cfg.MOME, 'NO_ISOLATION', False)):
        logger.warning(
            "AMP_DTYPE is fp16, so Step 1 and Step 2 share one loss scale. An overflow on "
            "D_f lowers the scale that later D_r steps use. The reported runs are bf16, "
            "where the scaler is disabled and no such state exists.")
    n_nonfinite_core = 0
    n_nonfinite_para = 0
    loss_clean_meter = AverageMeter()
    loss_para_meter = AverageMeter()
    loss_total_meter = AverageMeter()
    acc_meter = AverageMeter()
    grad_check_done = False
    max_grad_norm = 5.0
    parasitic_param_abs_max = 5.0
    freeze_bn_step2 = bool(getattr(cfg.MOME, "FREEZE_BN_STEP2", False))

    # Intermediate-layer hint distillation for CORE_KD: taps capture block outputs only while
    # armed around the two KD forwards, so the task forwards pay nothing.
    _hint_w = float(getattr(cfg.MOME, "CORE_KD_HINT", 0.0))
    _hint_layers = [int(x) for x in str(getattr(cfg.MOME, "CORE_KD_HINT_LAYERS", ""))
                    .strip().strip("'\"").split(",") if x.strip()]
    _ema_decay = float(getattr(cfg.MOME, "CORE_EMA", 0.0))
    _ema = None
    if _ema_decay > 0:
        _m0 = model.module if hasattr(model, "module") else model
        _ema = {n: p.detach().clone().float()
                for n, p in _m0.named_parameters() if not _is_parasitic_param(n)}
        logger.info("core EMA enabled, decay %.4f over %d tensors", _ema_decay, len(_ema))

    _t_renorm = _make_teacher_renorm(cfg) if teacher is not None else None
    if _t_renorm is not None:
        logger.info("teacher inputs renormalized to the teacher's own pixel statistics")
    _kd_proj = None
    _proj_path = str(getattr(cfg.MOME, "CORE_KD_PROJ", "")).strip()
    if teacher is not None and _proj_path:
        _kd_proj = torch.load(_proj_path, map_location="cuda").float()
        logger.info("KD projector loaded from %s: scrubbing a %d-dim teacher subspace",
                    _proj_path, _kd_proj.shape[1])
    _tap_s = _tap_t = None
    if teacher is not None and _hint_w > 0 and _hint_layers:
        _m0 = model.module if hasattr(model, "module") else model
        _tap_s = _HintTap(_m0, _hint_layers)
        _tap_t = _HintTap(teacher, _hint_layers)
        logger.info("hint distillation armed: layers=%s weight=%.3g", _hint_layers, _hint_w)
    if freeze_bn_step2:
        logger.info("BN-freeze in Step2 is enabled (freeze BatchNorm running stats on D_f update).")

    for epoch in range(1, epochs + 1):
        # The reference steps its scheduler at the START of the epoch; stepping at the end lags
        # the whole LR curve by one epoch. timm schedulers take the epoch index; a bare step()
        # silently leaves the rate unchanged.
        if scheduler is not None:
            try:
                scheduler.step(epoch)
            except TypeError:
                scheduler.step()
        model.train()
        start_time = time.time()

        loss_clean_meter.reset()
        loss_para_meter.reset()
        loss_total_meter.reset()
        acc_meter.reset()

        dr_iter = iter(train_loader_dr)
        df_iter = iter(train_loader_df)
        num_steps = max(len(train_loader_dr), len(train_loader_df))
        # Without balancing, the cycled D_r loader gives Core num_steps/len(dr) passes over its
        # data per epoch while D_f gets exactly one, so the smaller the retain set the more Core
        # overfits it. When enabled, Core steps only every core_every-th iteration, which
        # restores one D_r pass per epoch and leaves Step 2 untouched.
        core_every = 1
        _ce_fixed = int(getattr(cfg.MOME, "CORE_EVERY", 0))
        if _ce_fixed > 0:
            core_every = _ce_fixed
            if epoch == 1:
                logger.info("Core update period fixed to %d by MOME.CORE_EVERY", core_every)
        elif bool(getattr(cfg.MOME, "BALANCE_CORE_STEPS", False)) and len(train_loader_dr) > 0:
            core_every = max(1, round(num_steps / len(train_loader_dr)))
            if epoch == 1:
                logger.info("Balanced core steps: 1 core update every %d iterations "
                            "(dr batches=%d, steps=%d)", core_every, len(train_loader_dr), num_steps)

        for n_iter in range(num_steps):
            batch_dr, dr_iter = _cycle_next(dr_iter, train_loader_dr)
            batch_df, df_iter = _cycle_next(df_iter, train_loader_df)

            if len(batch_dr) == 5:
                img_dr, img_dr_weak, pid_dr, cam_dr, view_dr = batch_dr
                img_dr_weak = img_dr_weak.to(device)
            else:
                img_dr, pid_dr, cam_dr, view_dr = batch_dr
                img_dr_weak = None
            img_df, pid_df, cam_df, view_df = batch_df

            img_dr = img_dr.to(device)
            pid_dr = pid_dr.to(device)
            cam_dr = cam_dr.to(device)
            view_dr = view_dr.to(device)
            img_df = img_df.to(device)
            pid_df = pid_df.to(device)
            cam_df = cam_df.to(device)
            view_df = view_df.to(device)

            optimizer.zero_grad()
            raw_model = model.module if hasattr(model, "module") else model

            # Step 1: Core training with D_r (Parasitic OFF)
            do_core = (n_iter % core_every == 0)
            if do_core:
                set_parasitic_mode(raw_model, False)
                # NO_ISOLATION is the ablation: leaving both sides trainable lets D_f gradient
                # reach the core in Step 2, which is precisely what the guarantee forbids.
                if not _no_isolation:
                    freeze_parasitic_params(raw_model)
                unfreeze_core_params(raw_model)
                with amp.autocast(enabled=_amp_on, dtype=_amp_dtype):
                    score_dr, feat_dr = model(
                        img_dr, pid_dr, cam_label=cam_dr, view_label=view_dr
                    )
                    score_dr, pid_dr_c = _restrict_to_retained(
                        score_dr, pid_dr, retain_idx, label_map)
                    loss_clean = loss_fn(score_dr, feat_dr, pid_dr_c)
                    # v4-Core: distill the teacher's features on D_r IMAGES ONLY. U's images
                    # are never seen; whether this stays private is the certificate's call.
                    _kd = float(getattr(cfg.MOME, "CORE_KD", 0.0))
                    if teacher is not None and _kd > 0:
                        raw_model.eval()
                        if _tap_s is not None:
                            _tap_s.armed = _tap_t.armed = True
                        fs_kd = _distill_feature(raw_model, img_dr, cam_dr, view_dr, grad=True)
                        _img_t = img_dr_weak if img_dr_weak is not None else img_dr
                        _img_tt = _t_renorm(_img_t) if _t_renorm is not None else _img_t
                        ft_kd = _distill_feature(teacher, _img_tt, cam_dr, view_dr, grad=False)
                        if _kd_proj is not None:
                            ft_kd = ft_kd - (ft_kd @ _kd_proj) @ _kd_proj.t()
                            ft_kd = torch.nn.functional.normalize(ft_kd, dim=1)
                        raw_model.train()
                        _kd_type = str(getattr(cfg.MOME, "CORE_KD_TYPE", "relational")).strip().strip("'\"")
                        if _kd_type == "feature":
                            # features arrive L2-normalized, so this is 1 - cosine per sample
                            _kd_loss = (1.0 - (fs_kd * ft_kd).sum(dim=1)).mean()
                        else:
                            _kd_loss = _relational_kd(fs_kd, ft_kd)
                        loss_clean = loss_clean + _kd * _kd_loss
                        if _tap_s is not None:
                            _tap_s.armed = _tap_t.armed = False
                            _hint_loss = 0.0
                            for _hl in _hint_layers:
                                _hs = torch.nn.functional.normalize(
                                    _tap_s.captured[_hl].float(), dim=-1)
                                _ht = torch.nn.functional.normalize(
                                    _tap_t.captured[_hl].float(), dim=-1)
                                # per-token 2 - 2cos, averaged over tokens and batch
                                _hint_loss = _hint_loss + (_hs - _ht).pow(2).sum(-1).mean()
                            loss_clean = loss_clean + _hint_w * _hint_loss / len(_hint_layers)
                            _tap_s.captured.clear()
                            _tap_t.captured.clear()
                    _kdm = float(getattr(cfg.MOME, "CORE_KD_MIX", 0.0))
                    if teacher is not None and _kdm > 0:
                        _alpha = float(getattr(cfg.MOME, "CORE_KD_MIX_ALPHA", 0.4))
                        _lam = float(np.random.beta(_alpha, _alpha))
                        _perm = torch.randperm(img_dr.size(0), device=img_dr.device)
                        _img_b = img_dr_weak if img_dr_weak is not None else img_dr
                        img_mix = _lam * img_dr + (1.0 - _lam) * img_dr[_perm]
                        img_mix_t = _lam * _img_b + (1.0 - _lam) * _img_b[_perm]
                        raw_model.eval()
                        fs_mx = _distill_feature(raw_model, img_mix, cam_dr, view_dr, grad=True)
                        img_mix_tt = _t_renorm(img_mix_t) if _t_renorm is not None else img_mix_t
                        ft_mx = _distill_feature(teacher, img_mix_tt, cam_dr, view_dr, grad=False)
                        raw_model.train()
                        loss_clean = loss_clean + _kdm * (1.0 - (fs_mx * ft_mx).sum(dim=1)).mean()
                    _kdl = float(getattr(cfg.MOME, "CORE_KD_LOGIT", 0.0))
                    if teacher is not None and _kdl > 0:
                        _T = float(getattr(cfg.MOME, "CORE_KD_LOGIT_T", 4.0))
                        t_scores = _teacher_train_logits(
                            teacher, img_dr_weak if img_dr_weak is not None else img_dr,
                            pid_dr, cam_dr, view_dr)
                        if not isinstance(t_scores, list):
                            t_scores = [t_scores]
                        s_scores = score_dr if isinstance(score_dr, list) else [score_dr]
                        _kl = 0.0
                        for ts, ss in zip(t_scores, s_scores):
                            if retain_idx is not None:
                                ts = ts[:, retain_idx]
                            _kl = _kl + torch.nn.functional.kl_div(
                                torch.log_softmax(ss.float() / _T, dim=1),
                                torch.softmax(ts.float() / _T, dim=1),
                                reduction="batchmean") * (_T * _T)
                        loss_clean = loss_clean + _kdl * _kl / len(s_scores)

                if not torch.isfinite(loss_clean):
                    logger.warning(
                        "Skip iter due to non-finite loss_clean at epoch=%d iter=%d",
                        epoch,
                        n_iter + 1,
                    )
                    optimizer.zero_grad()
                    unfreeze_core_params(raw_model)
                    unfreeze_parasitic_params(raw_model)
                    continue
                scaler.scale(loss_clean).backward()

                if not grad_check_done:
                    leaked = []
                    for name, param in raw_model.named_parameters():
                        if _is_parasitic_param(name) and param.grad is not None:
                            if param.grad.detach().abs().sum().item() > 0:
                                leaked.append(name)
                                if len(leaked) >= 3:
                                    break
                    if leaked:
                        logger.warning(
                            "Step1 isolation warning: parasitic grads detected on %s", leaked
                        )
                    else:
                        logger.info("Step1 isolation check passed: no parasitic gradients.")

            if bool(getattr(cfg.MOME, 'CORE_ONLY', False)):
                # Diagnostic path: Step 1 only, so Core trains exactly as the reference does.
                unfreeze_core_params(raw_model); unfreeze_parasitic_params(raw_model)
                if not do_core:
                    # balance skipped Step 1 and CORE_ONLY skips Step 2: nothing was backwarded
                    # this iteration, and scaler.step() asserts if no inf-checks were recorded.
                    continue
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_grad_norm)
                _sc_before = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                if scaler.get_scale() < _sc_before and epoch <= 2 and n_iter < 5:
                    logger.warning('AMP scaler backed off (silent skip) at epoch=%d iter=%d', epoch, n_iter + 1)
                continue

            # Step 2: Parasitic training with D_f (Parasitic ON)
            set_parasitic_mode(raw_model, True)
            if not _no_isolation:
                freeze_core_params(raw_model)
            unfreeze_parasitic_params(raw_model)
            if freeze_bn_step2:
                set_bn_train_mode(raw_model, False)
            with amp.autocast(enabled=_amp_on, dtype=_amp_dtype):
                score_df, feat_df = model(
                    img_df, pid_df, cam_label=cam_df, view_label=view_df
                )
                if teacher is not None and bool(getattr(cfg.MOME, "STEP2_DISTILL", False)):
                    # v4-Full: the branch objective IS alignment with the Original model.
                    # eval-style forwards keep the shared BN buffers clean of D_f statistics.
                    raw_model.eval()
                    fs_d = _distill_feature(raw_model, img_df, cam_df, view_df, grad=True)
                    raw_model.train()
                    _img_dft = _t_renorm(img_df) if _t_renorm is not None else img_df
                    ft_d = _distill_feature(teacher, _img_dft, cam_df, view_df, grad=False)
                    loss_para = _relational_kd(fs_d, ft_d)
                else:
                    loss_para = loss_fn(score_df, feat_df, pid_df)

            if not torch.isfinite(loss_para):
                logger.warning(
                    "Non-finite loss_para at epoch=%d iter=%d; core-only step.",
                    epoch,
                    n_iter + 1,
                )
                if freeze_bn_step2:
                    set_bn_train_mode(raw_model, True)
                unfreeze_core_params(raw_model)
                unfreeze_parasitic_params(raw_model)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                continue
            scaler.scale(loss_para).backward()
            if freeze_bn_step2:
                set_bn_train_mode(raw_model, True)

            if not grad_check_done:
                para_grad_sum = 0.0
                non_finite_para_grad = False
                for name, param in raw_model.named_parameters():
                    if _is_parasitic_param(name) and param.grad is not None:
                        if not torch.isfinite(param.grad).all():
                            non_finite_para_grad = True
                        para_grad_sum += torch.nan_to_num(
                            param.grad.detach(), nan=0.0, posinf=0.0, neginf=0.0
                        ).abs().sum().item()
                if non_finite_para_grad:
                    logger.warning("Step2 isolation warning: non-finite parasitic grads detected.")
                elif para_grad_sum <= 0:
                    logger.warning("Step2 isolation warning: parasitic grad sum is zero.")
                else:
                    logger.info(
                        "Step2 isolation check passed: parasitic grad sum %.4e.",
                        para_grad_sum,
                    )
                grad_check_done = True

            # Restore requires_grad flags before optimizer update
            unfreeze_core_params(raw_model)
            unfreeze_parasitic_params(raw_model)

            scaler.unscale_(optimizer)
            # Clip the two parameter groups separately. A joint clip over all parameters makes
            # Core's effective learning rate depend on the parasitic branch's gradient
            # magnitude: once the total norm exceeds the threshold every gradient is scaled by
            # the same factor, so a large Step 2 gradient shrinks the Step 1 update. Gradient
            # isolation is supposed to isolate the source of Core's gradients; without this it
            # does not isolate their scale. The parasitic branch carries a 5x learning-rate
            # multiplier and, with LoRA, 1.18M extra parameters, so the coupling is not small.
            _para_set = {id(q) for q in get_parasitic_parameters(raw_model)}
            _core_p = [q for q in raw_model.parameters() if id(q) not in _para_set]
            _para_p = [q for q in raw_model.parameters() if id(q) in _para_set]
            _gn_core = torch.nn.utils.clip_grad_norm_(_core_p, max_grad_norm) if _core_p else torch.tensor(0.0)
            _gn_para = torch.nn.utils.clip_grad_norm_(_para_p, max_grad_norm) if _para_p else torch.tensor(0.0)
            # Whether to apply an update is decided per group. Deciding it jointly is a channel
            # from D_f into the core: a non-finite parasitic gradient would discard the core
            # update already computed on D_r, and whether a batch overflows depends on that
            # batch, so the channel carries content rather than only a count. Zeroing the
            # offending group leaves no non-finite value for GradScaler.step to find, so its
            # own global inf check cannot skip the other group either.
            _nf_core = not math.isfinite(float(_gn_core))
            _nf_para = not math.isfinite(float(_gn_para))
            if _nf_core:
                n_nonfinite_core += 1
                for _q in _core_p:
                    if _q.grad is not None:
                        _q.grad.zero_()
            if _nf_para:
                n_nonfinite_para += 1
                for _q in _para_p:
                    if _q.grad is not None:
                        _q.grad.zero_()
            if _nf_core or _nf_para:
                logger.warning(
                    "Non-finite gradient at epoch=%d iter=%d (core=%s parasitic=%s). That group "
                    "alone was dropped; the other one was applied as computed.",
                    epoch, n_iter + 1, _nf_core, _nf_para)
            grad_norm = max(0.0 if _nf_core else float(_gn_core),
                            0.0 if _nf_para else float(_gn_para))
            if _nf_core and _nf_para:
                optimizer.zero_grad(set_to_none=True)
                if _amp_fp16:
                    scaler.update(max(scaler.get_scale() * 0.5, 1e-4))
                continue

            scaler.step(optimizer)
            if _amp_fp16 and (_nf_core or _nf_para):
                scaler.update(max(scaler.get_scale() * 0.5, 1e-4))
            else:
                scaler.update()

            for name, param in raw_model.named_parameters():
                if _is_parasitic_param(name):
                    param.data.clamp_(-parasitic_param_abs_max, parasitic_param_abs_max)

            if _ema is not None:
                with torch.no_grad():
                    for n, p in raw_model.named_parameters():
                        if n in _ema:
                            _ema[n].mul_(_ema_decay).add_(p.detach().float(), alpha=1.0 - _ema_decay)

            # On a skipped core iteration score_dr/loss_clean still hold the previous
            # iteration's tensors, so both the accuracy and the total must be taken from Step 2
            # alone rather than silently re-counting stale values.
            if do_core:
                score_main = _pick_main_score(score_dr)
                acc = (score_main.max(1)[1] == pid_dr_c).float().mean()
                total_loss = loss_clean.item() + loss_para.item()
            else:
                total_loss = loss_para.item()

            if do_core:
                loss_clean_meter.update(loss_clean.item(), img_dr.shape[0])
            loss_para_meter.update(loss_para.item(), img_df.shape[0])
            loss_total_meter.update(total_loss, img_dr.shape[0])
            if do_core:
                acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iter[{}/{}] L_total:{:.3f} L_clean:{:.3f} "
                    "L_para:{:.3f} Acc:{:.3f} Base Lr:{:.2e}".format(
                        epoch,
                        n_iter + 1,
                        num_steps,
                        loss_total_meter.avg,
                        loss_clean_meter.avg,
                        loss_para_meter.avg,
                        acc_meter.avg,
                        optimizer.param_groups[0]["lr"],
                    )
                )

        end_time = time.time()
        time_per_batch = (end_time - start_time) / max(num_steps, 1)
        log_batch_size = (
            train_loader_df.batch_size
            if train_loader_df.batch_size is not None
            else cfg.SOLVER.IMS_PER_BATCH
        )
        if (not cfg.MODEL.DIST_TRAIN) or dist.get_rank() == 0:
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s] "
                "non-finite grads so far: core {} parasitic {}".format(
                    epoch, time_per_batch, log_batch_size / time_per_batch,
                    n_nonfinite_core, n_nonfinite_para
                )
            )

        _ema_backup = None
        if _ema is not None and ((epoch % checkpoint_period == 0) or (epoch % eval_period == 0) or (epoch == epochs)):
            # evaluate and checkpoint the averaged core; restore the live weights afterwards
            _mr = model.module if hasattr(model, "module") else model
            _ema_backup = {}
            with torch.no_grad():
                for n, p in _mr.named_parameters():
                    if n in _ema:
                        _ema_backup[n] = p.detach().clone()
                        p.data.copy_(_ema[n].to(p.dtype))

        if (epoch % checkpoint_period == 0) or (epoch == epochs):
            model_to_save = model.module if hasattr(model, "module") else model
            save_path = os.path.join(cfg.OUTPUT_DIR, "mome_v3_epoch_{}.pth".format(epoch))
            if (not cfg.MODEL.DIST_TRAIN) or dist.get_rank() == 0:
                torch.save(model_to_save.state_dict(), save_path)
                logger.info("Saved checkpoint to {}".format(save_path))

        if (epoch % eval_period == 0) or (epoch == epochs):
            full_cmc, full_map = _evaluate_reid_split(
                cfg, model, val_loader, num_query, mode_name="full", split_name="test"
            )
            core_cmc, core_map = _evaluate_reid_split(
                cfg, model, val_loader, num_query, mode_name="core", split_name="test"
            )
            if bool(getattr(cfg.MOME, "EVAL_TRAIN_MAP", True)):
                _, core_train_map = _evaluate_reid_split(
                    cfg,
                    model,
                    train_eval_loader,
                    train_num_query,
                    mode_name="core",
                    split_name="train",
                )
            else:
                core_train_map = float("nan")
            if (not cfg.MODEL.DIST_TRAIN) or dist.get_rank() == 0:
                logger.info(
                    "Epoch[{}] Summary | Full mAP:{:.2%} R1:{:.2%} | "
                    "Core mAP:{:.2%} R1:{:.2%} train_mAP:{:.2%}".format(
                        epoch, full_map, full_cmc[0], core_map, core_cmc[0], core_train_map
                    )
                )
            torch.cuda.empty_cache()

        if _ema_backup is not None:
            _mr = model.module if hasattr(model, "module") else model
            with torch.no_grad():
                for n, p in _mr.named_parameters():
                    if n in _ema_backup:
                        p.data.copy_(_ema_backup[n])


