"""CIFAR-100 dataloader for the cross-task generality demo (responds to o3UY O4).

10% random forget → D_r = 90%, D_f \\ D_r = 10%.
Uses torchvision's CIFAR100; downloads to data/cifar100/.
"""

import os
import random

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset


def make_cifar100_loaders(cfg, root=os.path.join(os.environ.get("MOME_DATA", "./data"), "cifar100"),
                          forget_ratio=0.1, seed=42):
    train_t = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_t = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    full_train = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=train_t)
    test_set = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=test_t)

    rng = random.Random(seed)
    n = len(full_train)
    idx = list(range(n)); rng.shuffle(idx)
    n_forget = int(n * forget_ratio)
    forget_idx = sorted(idx[:n_forget])
    retain_idx = sorted(idx[n_forget:])

    dr_set = Subset(full_train, retain_idx)
    df_only_set = Subset(full_train, forget_idx)
    df_set = full_train  # full

    bsz = cfg.SOLVER.IMS_PER_BATCH if hasattr(cfg, "SOLVER") else 128
    nw = cfg.DATALOADER.NUM_WORKERS if hasattr(cfg, "DATALOADER") else 4

    dr_loader = DataLoader(dr_set, batch_size=bsz, shuffle=True, num_workers=nw, drop_last=False)
    df_only_loader = DataLoader(df_only_set, batch_size=bsz, shuffle=True, num_workers=nw, drop_last=False)
    df_loader = DataLoader(df_set, batch_size=bsz, shuffle=True, num_workers=nw, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=bsz * 2, shuffle=False, num_workers=nw)

    info = {"n_total": n, "n_dr": len(retain_idx), "n_df_only": len(forget_idx),
            "forget_ratio": forget_ratio, "seed": seed}
    return dr_loader, df_only_loader, df_loader, test_loader, info
