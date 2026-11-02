#!/usr/bin/env python3
"""Forgetting-Rank statistic rho_U for the CIFAR-10 + ResNet-18 MoME setup.

Non-ReID certificate that mirrors the ViT / AGW pipeline
(scripts/cvpr2027/collect_lira_agw.py + scripts/cvpr2027/compute_rho.py):

  * Train N MoME "shadow" models and N retrain "shadow" models, each on a DIFFERENT
    retain/forget split indexed by `seed`.
  * For every shadow, compute the per-sample CE loss over the FULL forget-pool in a
    FIXED sample order, in Core mode (parasitic OFF).
  * Build dr_membership[n_seeds, N] = True where sample x is in that seed's D_r
    (retained; i.e. NOT forgotten by that shadow).
  * compute_rho.py::mean_rank ranks, for each target t and each x that t FORGOT,
    t's loss among {t} + {other shadows that also forgot x}; averaged normalized
    rank -> one rho per shadow. E[rho]=0.5 under no-leak.

Forget-pool manifest
--------------------
The manifest is the full CIFAR-10 *train* set in its native index order (0..49999).
Every sample therefore has a fixed position, so `losses[t]` is a length-50000 vector
and dr_membership[t, x] = (x in seed-t's D_r).

Split reproduction (EXACT match to train_cifar10_r18_mome.get_loaders lines 74-76)
---------------------------------------------------------------------------------
get_loaders does:
    n = len(train)                                            # 50000
    rng = random.Random(seed); idx = list(range(n)); rng.shuffle(idx)
    n_df = int(n * forget_ratio); df_idx = idx[:n_df]; dr_idx = idx[n_df:]
`split_indices` below is byte-for-byte the same computation, so membership is exact
and does NOT depend on any (shuffled) DataLoader iteration order. random.Random(seed)
is an independent Mersenne-Twister instance, so the split is fully determined by
`seed` regardless of the global RNG state we set for training reproducibility.

Resumable at every stage: existing checkpoints and existing .npy loss files are
skipped. rho_U is (re)computed at the end by the identical, already-validated
scripts/cvpr2027/compute_rho.py (with an in-process cross-check using the same
mean_rank function).

Launch (from the TransReID working dir, conda env reid_unlearning):
    python cifar_rho.py --seeds 42,123,456,789,1024 --epochs 80 --lr 0.05 --gpu 0 \
        --data_root ./data --out_dir logs/cifar_rho
"""
import argparse
import os
import sys


# --- GPU selection must precede `import torch` -----------------------------------
def _early_gpu_arg(default="0"):
    """Extract --gpu from argv before torch is imported (CUDA reads the env at init)."""
    for i, a in enumerate(sys.argv):
        if a == "--gpu" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--gpu="):
            return a.split("=", 1)[1]
    return default


os.environ["CUDA_VISIBLE_DEVICES"] = _early_gpu_arg()

import json
import random
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

# Import the shared CIFAR MoME building blocks (same directory). We DO NOT modify it.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from train_cifar10_r18_mome import (  # noqa: E402
    ParasiticBN2d,             # imported for parity / availability (used inside make_resnet18_cifar)
    get_loaders,
    make_resnet18_cifar,
    replace_bn_with_parasitic,  # imported for parity / availability
    set_parasitic,
    train_loop,
    train_mome,
)

# CIFAR-10 constants (must match get_loaders exactly).
N_TRAIN = 50000
FORGET_RATIO = 0.1
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
NUM_CLASSES = 10


# --- reproducibility -------------------------------------------------------------
def set_all_seeds(seed):
    """Per-shadow seeding, identical to train_cifar10_r18_mome.py's __main__ block."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --- split reproduction (must mirror get_loaders' class-level deletion) ----------
def split_indices(seed, n=N_TRAIN, forget_class_frac=0.5, dr_k=600, data_root=None):
    """Reproduce get_loaders' class deletion. U holds every image of the removed classes and
    each retained class keeps dr_k images, so the certificate scores the same U the trained
    models were asked to forget. Returns (df_idx, dr_idx)."""
    import torchvision
    root = data_root or os.environ.get("MOME_DATA", "./data")
    targets = list(torchvision.datasets.CIFAR10(root, train=True, download=False).targets)
    classes = sorted(set(targets))
    n_forget = max(1, int(round(len(classes) * forget_class_frac)))
    forget = set(random.Random(seed).sample(classes, n_forget))
    by_class = {}
    for i, y in enumerate(targets):
        by_class.setdefault(y, []).append(i)
    df_idx, dr_idx = [], []
    for y, ids in by_class.items():
        if y in forget:
            df_idx.extend(ids)
        elif dr_k and dr_k > 0:
            dr_idx.extend(random.Random(seed * 100003 + y).sample(ids, min(dr_k, len(ids))))
        else:
            dr_idx.extend(ids)
    return sorted(df_idx), sorted(dr_idx)


# --- fixed-order forget-pool loader ----------------------------------------------
def build_manifest_loader(data_root, batch=500, workers=4):
    """Full 50000-sample train set in NATIVE index order, TEST transforms, no shuffle.

    Matches get_loaders' `tr_test` (ToTensor + Normalize, no augmentation). shuffle=False
    guarantees samples arrive in index order 0..49999 regardless of num_workers, so the
    concatenated loss vector aligns 1:1 with the manifest / dr_membership columns.
    """
    tr_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    ds = torchvision.datasets.CIFAR10(data_root, train=True, download=True, transform=tr_test)
    assert len(ds) == N_TRAIN, f"unexpected CIFAR-10 train size {len(ds)} (expected {N_TRAIN})"
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)
    return loader


# --- loss / accuracy collection --------------------------------------------------
@torch.no_grad()
def per_sample_ce(model, loader, device="cuda"):
    """-log_softmax(logits)[y] per sample, in the loader's (fixed) order. float32 [N]."""
    model.eval()
    chunks = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        ce = -F.log_softmax(logits, dim=1).gather(1, y.unsqueeze(1)).squeeze(1)
        chunks.append(ce.detach().cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


@torch.no_grad()
def test_accuracy(model, loader, device="cuda"):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return correct / total if total else 0.0


# --- shadow training (resumable) -------------------------------------------------
def get_or_train_mome(seed, ckpt_path, epochs, lr, data_root, dr_k=0, device="cuda"):
    """Train (or load) one MoME shadow on seed-`seed`'s split. Returns model in Core mode."""
    set_all_seeds(seed)
    _, dr_loader, df_only, test_loader, _, info = get_loaders(data_root, dr_k=dr_k, seed=seed)
    model = make_resnet18_cifar(num_classes=NUM_CLASSES, parasitic=True).to(device)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[mome  s{seed}] loaded {ckpt_path}")
    else:
        print(f"[mome  s{seed}] training MoME (Dr->core / Df->parasitic)  split={info}")
        train_mome(model, dr_loader, df_only, epochs=epochs, lr=lr,
                   parasitic_lr_mult=5.0, device=device)
        torch.save(model.state_dict(), ckpt_path)
        print(f"[mome  s{seed}] saved {ckpt_path}")
    set_parasitic(model, False)  # Core mode (parasitic OFF) for the certificate
    acc = test_accuracy(model, test_loader, device)
    print(f"[mome  s{seed}] Core test_acc = {acc:.4f}")
    return model


def get_or_train_retrain(seed, ckpt_path, epochs, lr, data_root, dr_k=0, device="cuda"):
    """Train (or load) one retrain (oracle) shadow on seed-`seed`'s D_r only. Returns model."""
    set_all_seeds(seed)
    _, dr_loader, _, test_loader, _, info = get_loaders(data_root, dr_k=dr_k, seed=seed)
    model = make_resnet18_cifar(num_classes=NUM_CLASSES, parasitic=False).to(device)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[retr  s{seed}] loaded {ckpt_path}")
    else:
        print(f"[retr  s{seed}] retraining from scratch on D_r  split={info}")
        train_loop(model, dr_loader, epochs=epochs, lr=lr, device=device)
        torch.save(model.state_dict(), ckpt_path)
        print(f"[retr  s{seed}] saved {ckpt_path}")
    acc = test_accuracy(model, test_loader, device)
    print(f"[retr  s{seed}] test_acc = {acc:.4f}")
    return model


def collect_losses(model, tag, seed, out_dir, manifest_loader, is_mome, device="cuda"):
    """Save per-sample Core-mode CE over the fixed manifest. Skip if file exists."""
    fp = os.path.join(out_dir, f"losses_{tag}_s{seed}.npy")
    if os.path.exists(fp):
        print(f"[{tag} s{seed}] losses exist -> skip ({fp})")
        return
    if is_mome:
        set_parasitic(model, False)  # ensure Core mode (parasitic OFF)
    losses = per_sample_ce(model, manifest_loader, device)
    assert losses.shape == (N_TRAIN,), f"expected ({N_TRAIN},), got {losses.shape}"
    np.save(fp, losses)
    print(f"[{tag} s{seed}] saved {fp}  mean_ce={losses.mean():.4f}")


# --- rho_U ------------------------------------------------------------------------
def compute_rho(out_dir, seeds, dr_mem):
    """Call the validated compute_rho.py, then cross-check in-process with the same mean_rank."""
    seeds_str = ",".join(str(s) for s in seeds)
    compute_rho_path = os.path.join(SCRIPT_DIR, "scripts", "cvpr2027", "compute_rho.py")
    cmd = [sys.executable, compute_rho_path,
           "--dir", os.path.abspath(out_dir),
           "--mome", seeds_str, "--oracle", seeds_str,
           "--mome_tag", "cifarmome", "--oracle_tag", "cifarretrain"]
    print("\n[compute_rho] " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:  # noqa: BLE001 - artifacts are already saved; fall back to in-process
        print(f"[compute_rho] subprocess failed ({e}); using in-process fallback")

    # In-process cross-check with the IDENTICAL mean_rank (no subprocess/scipy dependency).
    try:
        sys.path.insert(0, os.path.dirname(compute_rho_path))
        from compute_rho import mean_rank  # noqa: E402
        ml = np.stack([np.load(os.path.join(out_dir, f"losses_cifarmome_s{s}.npy")) for s in seeds])
        rl = np.stack([np.load(os.path.join(out_dir, f"losses_cifarretrain_s{s}.npy")) for s in seeds])
        rm = mean_rank(ml, dr_mem)
        ro = mean_rank(rl, dr_mem)
        print(f"[FINAL] MoME    rho_U = {rm.mean():.4f} +/- {rm.std():.4f}  per-seed {np.round(rm, 3)}")
        print(f"[FINAL] Retrain rho_U = {ro.mean():.4f} +/- {ro.std():.4f}  per-seed {np.round(ro, 3)}")
    except Exception as e:  # noqa: BLE001
        print(f"[FINAL] in-process rho computation failed: {e}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", default="42,123,456,789,1024",
                   help="comma-separated shadow seeds (one MoME + one retrain each)")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--gpu", default="0", help="CUDA device id (sets CUDA_VISIBLE_DEVICES)")
    p.add_argument("--data_root", default=os.environ.get("MOME_DATA", "./data"))
    p.add_argument("--out_dir", default="logs/cifar_rho")
    p.add_argument("--dr_k", type=int, default=0,
                   help="retained images per class, 0 for all. MUST match the dr_k the utility "
                        "runs used, otherwise the shadows and the membership matrix describe "
                        "different experiments: the default used to be 4 here and 600 in "
                        "split_indices, so the shadows trained on 20 images while the mask "
                        "claimed 3000.")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda"
    assert torch.cuda.is_available(), "CUDA is required (set --gpu to a valid device)"
    print(f"GPU (CUDA_VISIBLE_DEVICES)={os.environ.get('CUDA_VISIBLE_DEVICES')}  "
          f"seeds={seeds}  epochs={args.epochs}  lr={args.lr}  out_dir={os.path.abspath(out_dir)}")

    # Fixed forget-pool manifest: full train set, native order, TEST transforms.
    manifest_loader = build_manifest_loader(args.data_root)

    # dr_membership[n_seeds, N]: True where sample x is RETAINED (in D_r) for that seed.
    dr_mem = np.zeros((len(seeds), N_TRAIN), dtype=bool)
    for si, s in enumerate(seeds):
        _, dr_idx = split_indices(s, dr_k=args.dr_k)
        dr_mem[si, dr_idx] = True
    # U is the deleted classes. Without this mask the scoring set is the complement of D_r,
    # which also holds the retained classes' images that were not sampled into D_r.
    # Each seed deletes a different half of the classes, so U is per-shadow and the mask has to
    # be a matrix; a single row would restrict the other shadows to the wrong classes.
    u_mask = np.zeros((len(seeds), N_TRAIN), dtype=bool)
    for si, s_ in enumerate(seeds):
        df_i, _ = split_indices(s_, dr_k=args.dr_k)
        u_mask[si, df_i] = True
    np.save(os.path.join(out_dir, "u_mask.npy"), u_mask)
    print(f"u_mask saved, shape={u_mask.shape}, |U| per seed={u_mask.sum(1).tolist()}")
    dr_path = os.path.join(out_dir, "dr_membership.npy")
    np.save(dr_path, dr_mem)
    print(f"dr_membership {dr_mem.shape} retained_frac={dr_mem.mean():.4f} -> {dr_path}")

    # Provenance manifest (not required by compute_rho.py; documents the fixed order).
    json.dump(
        {"manifest": "cifar10_train_native_index_order", "N": N_TRAIN,
         "forget_ratio": FORGET_RATIO, "num_classes": NUM_CLASSES,
         "seeds": seeds, "epochs": args.epochs, "lr": args.lr},
        open(os.path.join(out_dir, "manifest.json"), "w"), indent=2,
    )

    # ---- MoME shadows (Core-mode negative control) ----
    for s in seeds:
        ckpt = os.path.join(out_dir, f"mome_seed{s}.pth")
        model = get_or_train_mome(s, ckpt, args.epochs, args.lr, args.data_root, dr_k=args.dr_k, device=device)
        collect_losses(model, "cifarmome", s, out_dir, manifest_loader, is_mome=True, device=device)
        del model
        torch.cuda.empty_cache()

    # ---- Retrain shadows (oracle reference; same per-seed split as MoME) ----
    for s in seeds:
        ckpt = os.path.join(out_dir, f"retrain_seed{s}.pth")
        model = get_or_train_retrain(s, ckpt, args.epochs, args.lr, args.data_root, dr_k=args.dr_k, device=device)
        collect_losses(model, "cifarretrain", s, out_dir, manifest_loader, is_mome=False, device=device)
        del model
        torch.cuda.empty_cache()

    # ---- rho_U via the identical validated machinery ----
    compute_rho(out_dir, seeds, dr_mem)
    print("\nCIFAR rho_U DONE")


if __name__ == "__main__":
    main()
