#!/usr/bin/env python3
"""Class-unlearning benchmark on the datasets the baselines were designed for.

Every competing method in the paper is an image-classification unlearning method evaluated only
on person re-identification. This runs them on their own ground: CIFAR-10, CIFAR-100 and
TinyImageNet, with a retrained oracle available at each scale.

The baselines' algorithms are model-agnostic; only the surrounding data and model code in the
ReID scripts is specific. They are reimplemented here compactly against one ResNet-18 so that
every method sees an identical model, split, schedule and evaluation.

Protocol: class deletion, the standard setting for these methods. A fixed fraction of classes is
removed; U is every image of those classes; D_r is everything else. Reported per method:
  test acc on retained classes, acc on U (lower is more forgotten, the retrain oracle sets the
  reference), and entropy-MIA of U against the test split.

Run: python scripts/cvpr2027/unlearn_bench.py --dataset cifar10 --method retrain --out_dir logs/ub
"""
import argparse
import copy
import json
import os
import os.path as osp
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from train_cifar10_r18_mome import (make_resnet18_cifar, replace_bn_with_parasitic,  # noqa: E402
                                    set_parasitic)

MEAN_STD = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "tinyimagenet": ((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
}
N_CLASSES = {"cifar10": 10, "cifar100": 100, "tinyimagenet": 200}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_data(dataset, root, train=True):
    mean, std = MEAN_STD[dataset]
    size = 64 if dataset == "tinyimagenet" else 32
    tf_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(size, padding=size // 8),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)])
    tf_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)])
    tf = tf_train if train else tf_test
    if dataset == "cifar10":
        return torchvision.datasets.CIFAR10(root, train=train, download=False, transform=tf)
    if dataset == "cifar100":
        return torchvision.datasets.CIFAR100(root, train=train, download=False, transform=tf)
    split = "train" if train else "val"
    return torchvision.datasets.ImageFolder(osp.join(root, "tiny-imagenet-200", split), transform=tf)


def targets_of(ds):
    if hasattr(ds, "targets"):
        return list(ds.targets)
    return [y for _, y in ds.samples]


def class_split(ds, forget_frac, seed):
    """U is every image of a random fraction of the classes; D_r is the rest."""
    y = targets_of(ds)
    classes = sorted(set(y))
    n_forget = max(1, int(round(len(classes) * forget_frac)))
    forget = set(random.Random(seed).sample(classes, n_forget))
    u_idx = [i for i, c in enumerate(y) if c in forget]
    r_idx = [i for i, c in enumerate(y) if c not in forget]
    return r_idx, u_idx, sorted(forget)


def loader(ds, idx, batch, shuffle):
    return DataLoader(Subset(ds, idx), batch_size=batch, shuffle=shuffle, num_workers=4,
                      pin_memory=True, drop_last=False)


def train_epochs(model, dl, epochs, lr, device="cuda", relabel_random=None, n_cls=10):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    model.train()
    for _ in range(epochs):
        for x, t in dl:
            x, t = x.to(device), t.to(device)
            if relabel_random:
                t = torch.randint(0, n_cls, t.shape, device=device)
            opt.zero_grad()
            F.cross_entropy(model(x), t).backward()
            opt.step()
        sch.step()
    return model


@torch.no_grad()
def accuracy(model, dl, device="cuda"):
    model.eval(); c = n = 0
    for x, t in dl:
        p = model(x.to(device)).argmax(1).cpu()
        c += (p == t).sum().item(); n += len(t)
    return c / max(1, n)


@torch.no_grad()
def entropies(model, dl, device="cuda"):
    model.eval(); out = []
    for x, _ in dl:
        p = F.softmax(model(x.to(device)).float(), 1)
        out.append((-(p * torch.log(p + 1e-12)).sum(1)).cpu().numpy())
    return np.concatenate(out)


def mia_auc(model, u_dl, test_dl):
    em, en = entropies(model, u_dl), entropies(model, test_dl)
    n = min(len(em), len(en))
    return float(roc_auc_score(np.r_[np.ones(n), np.zeros(n)], np.r_[-em[:n], -en[:n]]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(N_CLASSES))
    p.add_argument("--method", required=True)
    p.add_argument("--data_root", default=os.environ.get("MOME_DATA", "./data"))
    p.add_argument("--out_dir", default="logs/unlearn_bench")
    p.add_argument("--forget_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--unlearn_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--original_ckpt", default="")
    a = p.parse_args()
    set_seed(a.seed)
    dev = "cuda"
    nc = N_CLASSES[a.dataset]

    tr = build_data(a.dataset, a.data_root, True)
    te = build_data(a.dataset, a.data_root, False)
    r_idx, u_idx, forget_classes = class_split(tr, a.forget_frac, a.seed)
    te_y = targets_of(te)
    te_keep = [i for i, c in enumerate(te_y) if c not in set(forget_classes)]

    dl_r = loader(tr, r_idx, a.batch, True)
    dl_u = loader(tr, u_idx, a.batch, False)
    dl_te = loader(te, te_keep, a.batch, False)
    print(f"{a.dataset}: |D_r|={len(r_idx)} |U|={len(u_idx)} forget_classes={len(forget_classes)}")

    os.makedirs(a.out_dir, exist_ok=True)
    parasitic = a.method == "mome"
    model = make_resnet18_cifar(num_classes=nc, parasitic=parasitic).to(dev)

    # MoME is trained BY DESIGN: the core never starts from a checkpoint that has seen U, which
    # is the whole guarantee. Loading the original first and adapting is the post-hoc variant the
    # paper reports as leaking by construction, and scoring that as "MoME" measures the wrong
    # thing entirely.
    if a.method in ("original", "retrain", "mome"):
        if a.method == "mome":
            from train_cifar10_r18_mome import train_mome
            train_mome(model, dl_r, dl_u, epochs=a.epochs, lr=a.lr, device=dev)
            set_parasitic(model, False)
        else:
            src = loader(tr, list(range(len(tr))), a.batch, True) if a.method == "original" else dl_r
            train_epochs(model, src, a.epochs, a.lr, dev, n_cls=nc)
    else:
        ck = a.original_ckpt or osp.join(a.out_dir, f"{a.dataset}_original.pth")
        model.load_state_dict(torch.load(ck, map_location=dev), strict=not parasitic)
        if a.method == "finetune":
            train_epochs(model, dl_r, a.unlearn_epochs, a.lr * 0.1, dev, n_cls=nc)
        elif a.method == "ga":                       # gradient ascent on U
            opt = torch.optim.SGD(model.parameters(), lr=a.lr * 0.01, momentum=0.9)
            model.train()
            for _ in range(a.unlearn_epochs):
                for x, t in dl_u:
                    opt.zero_grad(); (-F.cross_entropy(model(x.to(dev)), t.to(dev))).backward(); opt.step()
        elif a.method == "neggrad":                  # ascent on U and descent on D_r together
            opt = torch.optim.SGD(model.parameters(), lr=a.lr * 0.01, momentum=0.9)
            model.train(); it = iter(dl_r)
            for _ in range(a.unlearn_epochs):
                for x, t in dl_u:
                    try: xr, tr_ = next(it)
                    except StopIteration: it = iter(dl_r); xr, tr_ = next(it)
                    opt.zero_grad()
                    (F.cross_entropy(model(xr.to(dev)), tr_.to(dev))
                     - F.cross_entropy(model(x.to(dev)), t.to(dev))).backward()
                    opt.step()
        elif a.method == "salun":                    # saliency mask, then random labels on U
            model.train(); model.zero_grad()
            for x, t in dl_u:
                F.cross_entropy(model(x.to(dev)), t.to(dev)).backward()
            sal = {n: (q.grad.abs() if q.grad is not None else torch.zeros_like(q))
                   for n, q in model.named_parameters()}
            flat = torch.cat([v.flatten() for v in sal.values()])
            thr = torch.quantile(flat[torch.randperm(len(flat))[:1_000_000]].float(), 0.5)
            mask = {n: (v >= thr).float() for n, v in sal.items()}
            opt = torch.optim.SGD(model.parameters(), lr=a.lr * 0.01, momentum=0.9)
            for _ in range(a.unlearn_epochs):
                for x, t in dl_u:
                    opt.zero_grad()
                    rt = torch.randint(0, nc, t.shape, device=dev)
                    F.cross_entropy(model(x.to(dev)), rt).backward()
                    for n, q in model.named_parameters():
                        if q.grad is not None: q.grad.mul_(mask[n])
                    opt.step()
                for x, t in dl_r:
                    opt.zero_grad(); F.cross_entropy(model(x.to(dev)), t.to(dev)).backward()
                    for n, q in model.named_parameters():
                        if q.grad is not None: q.grad.mul_(mask[n])
                    opt.step()
        elif a.method == "badteacher":               # match a random teacher on U, the original on D_r
            bad = make_resnet18_cifar(num_classes=nc).to(dev).eval()
            good = copy.deepcopy(model).eval()
            opt = torch.optim.SGD(model.parameters(), lr=a.lr * 0.01, momentum=0.9)
            model.train(); it = iter(dl_r)
            for _ in range(a.unlearn_epochs):
                for x, _t in dl_u:
                    try: xr, _ = next(it)
                    except StopIteration: it = iter(dl_r); xr, _ = next(it)
                    x, xr = x.to(dev), xr.to(dev)
                    opt.zero_grad()
                    lu = F.kl_div(F.log_softmax(model(x), 1), F.softmax(bad(x), 1), reduction="batchmean")
                    lr_ = F.kl_div(F.log_softmax(model(xr), 1), F.softmax(good(xr), 1), reduction="batchmean")
                    (lu + lr_).backward(); opt.step()
        elif a.method == "scrub":                    # max KL on U, min KL plus CE on D_r
            teach = copy.deepcopy(model).eval()
            opt = torch.optim.SGD(model.parameters(), lr=a.lr * 0.01, momentum=0.9)
            model.train()
            for ep in range(a.unlearn_epochs):
                if ep < a.unlearn_epochs // 2:
                    for x, _t in dl_u:
                        x = x.to(dev); opt.zero_grad()
                        (-F.kl_div(F.log_softmax(model(x), 1), F.softmax(teach(x), 1),
                                   reduction="batchmean")).backward()
                        opt.step()
                for x, t in dl_r:
                    x, t = x.to(dev), t.to(dev); opt.zero_grad()
                    (F.cross_entropy(model(x), t)
                     + F.kl_div(F.log_softmax(model(x), 1), F.softmax(teach(x), 1),
                                reduction="batchmean")).backward()
                    opt.step()
        else:
            raise SystemExit(f"unknown method {a.method}")

    if a.method == "original":
        torch.save(model.state_dict(), osp.join(a.out_dir, f"{a.dataset}_original.pth"))
    res = {"dataset": a.dataset, "method": a.method, "seed": a.seed,
           "forget_classes": len(forget_classes),
           "test_acc_retained": round(accuracy(model, dl_te, dev), 4),
           "acc_on_U": round(accuracy(model, dl_u, dev), 4),
           "mia_U_vs_test": round(mia_auc(model, dl_u, dl_te), 4)}
    json.dump(res, open(osp.join(a.out_dir, f"{a.dataset}_{a.method}_s{a.seed}.json"), "w"), indent=2)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
