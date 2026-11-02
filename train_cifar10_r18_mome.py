"""CIFAR-10 + ResNet-18 (CIFAR-style) MoME demo. Real overfit -> real MIA signal."""
import os
import argparse, os, json, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score


class ParasiticBN2d(nn.BatchNorm2d):
    def __init__(self, num_features, **kw):
        super().__init__(num_features, **kw)
        self.weight_m = nn.Parameter(torch.zeros_like(self.weight))
        self.bias_m = nn.Parameter(torch.zeros_like(self.bias))
        self.parasitic = False

    def forward(self, x):
        if self.parasitic:
            w = self.weight + self.weight_m
            b = self.bias + self.bias_m
        else:
            w = self.weight
            b = self.bias
        return F.batch_norm(
            x, self.running_mean, self.running_var,
            w, b, self.training, self.momentum, self.eps,
        )


def replace_bn_with_parasitic(module):
    n = 0
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d) and not isinstance(child, ParasiticBN2d):
            new = ParasiticBN2d(child.num_features, eps=child.eps, momentum=child.momentum,
                                affine=child.affine, track_running_stats=child.track_running_stats)
            new.weight.data.copy_(child.weight.data); new.bias.data.copy_(child.bias.data)
            new.running_mean = child.running_mean.clone(); new.running_var = child.running_var.clone()
            setattr(module, name, new); n += 1
        else:
            n += replace_bn_with_parasitic(child)
    return n


def set_parasitic(model, mode):
    for m in model.modules():
        if isinstance(m, ParasiticBN2d):
            m.parasitic = bool(mode)


def make_resnet18_cifar(num_classes=10, parasitic=False):
    m = torchvision.models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    if parasitic:
        n = replace_bn_with_parasitic(m); print(f"Replaced {n} BN2d with ParasiticBN2d")
    return m


def get_loaders(data_root, forget_class_frac=0.5, dr_k=4, seed=42, batch=128):
    mean = (0.4914, 0.4822, 0.4465); std = (0.2470, 0.2435, 0.2616)
    tr_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])
    tr_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std),
    ])
    train = torchvision.datasets.CIFAR10(data_root, train=True, download=True, transform=tr_train)
    train_eval = torchvision.datasets.CIFAR10(data_root, train=True, download=True, transform=tr_test)
    test = torchvision.datasets.CIFAR10(data_root, train=False, download=True, transform=tr_test)
    n = len(train)
    # Class-level deletion, the image-classification analogue of identity deletion: a random
    # half of the classes is requested for removal, U holds every image of those classes, and
    # each retained class keeps k images. Sample-level splitting would delete a fraction of
    # every class and would not pose the deletion the paper defines.
    targets = list(train.targets)
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
    print(f"class deletion: forget {sorted(forget)}, U {len(df_idx)} images, "
          f"D_r {len(dr_idx)} images over {len(classes)-n_forget} retained classes")
    return (
        DataLoader(train, batch_size=batch, shuffle=True, num_workers=4),
        DataLoader(Subset(train, dr_idx), batch_size=batch, shuffle=True, num_workers=4),
        DataLoader(Subset(train, df_idx), batch_size=batch, shuffle=True, num_workers=4),
        DataLoader(test, batch_size=batch, shuffle=False, num_workers=4),
        DataLoader(Subset(train_eval, df_idx), batch_size=batch, shuffle=False, num_workers=4),
        {'n': n, 'n_dr': len(dr_idx), 'n_df': len(df_idx)},
    )


def train_loop(model, loader, epochs, lr, device='cuda'):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    model.to(device)
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        sched.step()


def train_mome(model, loader_dr, loader_df, epochs, lr, parasitic_lr_mult=5.0, device='cuda'):
    parasitic = [p for n, p in model.named_parameters() if 'weight_m' in n or 'bias_m' in n]
    core = [p for n, p in model.named_parameters() if 'weight_m' not in n and 'bias_m' not in n]
    opt = torch.optim.SGD([
        {'params': core, 'lr': lr, 'weight_decay': 5e-4},
        {'params': parasitic, 'lr': lr * parasitic_lr_mult, 'weight_decay': 5e-4},
    ], momentum=0.9)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    model.to(device)
    for ep in range(1, epochs + 1):
        # Step 1: D_r -> core (parasitic OFF)
        for p in parasitic: p.requires_grad_(False)
        for p in core: p.requires_grad_(True)
        set_parasitic(model, False)
        model.train()
        for x, y in loader_dr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        # Step 2: D_f -> parasitic (ON), freeze BN running stats
        for p in parasitic: p.requires_grad_(True)
        for p in core: p.requires_grad_(False)
        set_parasitic(model, True)
        model.train()
        for m in model.modules():
            if isinstance(m, ParasiticBN2d): m.eval()
        for x, y in loader_df:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        for p in core: p.requires_grad_(True)
        sched.step()


@torch.no_grad()
def eval_acc_ent(model, loader, device='cuda'):
    model.eval()
    correct = 0; total = 0; ents = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(dim=1) == y).sum().item(); total += y.size(0)
        probs = F.softmax(logits, dim=1)
        ent = -(probs * torch.log(probs + 1e-12)).sum(dim=1)
        ents.append(ent.cpu().numpy())
    return correct / total, np.concatenate(ents)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', required=True, choices=['standard', 'retrain', 'mome'])
    p.add_argument('--epochs', default=80, type=int)
    p.add_argument('--lr', default=0.05, type=float)
    p.add_argument('--seed', default=42, type=int)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--data_root', default=os.environ.get("MOME_DATA", "./data"))
    p.add_argument('--forget_class_frac', default=0.5, type=float)
    p.add_argument('--dr_k', default=4, type=int, help='images kept per retained class')
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    df_full, dr_loader, df_only, test_loader, df_only_eval, info = get_loaders(args.data_root, forget_class_frac=args.forget_class_frac,
                                                                       dr_k=args.dr_k, seed=args.seed)
    print('Split:', info)

    parasitic = (args.mode == 'mome')
    model = make_resnet18_cifar(parasitic=parasitic).cuda()

    if args.mode == 'standard':
        train_loop(model, df_full, epochs=args.epochs, lr=args.lr)
    elif args.mode == 'retrain':
        train_loop(model, dr_loader, epochs=args.epochs, lr=args.lr)
    elif args.mode == 'mome':
        train_mome(model, dr_loader, df_only, epochs=args.epochs, lr=args.lr)

    if parasitic: set_parasitic(model, False)
    acc_test, ent_test = eval_acc_ent(model, test_loader)
    acc_df, ent_df = eval_acc_ent(model, df_only_eval)
    n = min(len(ent_test), len(ent_df))
    scores = np.concatenate([-ent_df[:n], -ent_test[:n]])
    labels = np.concatenate([np.ones(n), np.zeros(n)])
    auc = float(roc_auc_score(labels, scores))

    res = {'mode': args.mode, 'test_acc': float(acc_test), 'df_acc': float(acc_df),
           'mia_entropy_auc': auc, 'epochs': args.epochs}
    if parasitic:
        set_parasitic(model, True)
        acc_full, _ = eval_acc_ent(model, test_loader)
        res['test_acc_full'] = float(acc_full)
    print(json.dumps(res, indent=2))
    with open(os.path.join(args.out_dir, 'result.json'), 'w') as f:
        json.dump(res, f, indent=2)
