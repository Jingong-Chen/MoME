"""ParasiticBatchNorm2d wrapping for ResNet-50 backbone.

Mirrors ParasiticLayerNorm in vit_pytorch.py — adds zero-initialized weight_m, bias_m
parameters and a `parasitic` flag, so the same `set_parasitic_mode` toggling and the
same `freeze_*_params` partition work on a CNN backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_parasitic_type(parasitic_type):
    if parasitic_type is None:
        return "gamma_beta"
    value = str(parasitic_type).lower()
    alias = {"gamma+beta": "gamma_beta", "both": "gamma_beta",
             "gamma": "gamma_only", "beta": "beta_only"}
    value = alias.get(value, value)
    if value not in ("gamma_beta", "gamma_only", "beta_only"):
        raise ValueError("Unsupported parasitic type: {}".format(parasitic_type))
    return value


class ParasiticBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm2d with optional parasitic delta affine parameters."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__(num_features, eps=eps, momentum=momentum, affine=True, track_running_stats=True)
        self.weight_m = nn.Parameter(torch.zeros(num_features))
        self.bias_m   = nn.Parameter(torch.zeros(num_features))
        self.parasitic = True              # toggled by set_parasitic_mode
        self.parasitic_enabled = True      # static enable/disable
        self.parasitic_type = "gamma_beta"

    def configure_parasitic(self, enabled=True, parasitic_type="gamma_beta"):
        self.parasitic_enabled = bool(enabled)
        self.parasitic_type = _normalize_parasitic_type(parasitic_type)
        if self.parasitic_type == "gamma_only":
            self.weight_m.requires_grad_(self.parasitic_enabled)
            self.bias_m.requires_grad_(False)
            with torch.no_grad():
                self.bias_m.zero_()
        elif self.parasitic_type == "beta_only":
            self.weight_m.requires_grad_(False)
            self.bias_m.requires_grad_(self.parasitic_enabled)
            with torch.no_grad():
                self.weight_m.zero_()
        else:
            self.weight_m.requires_grad_(self.parasitic_enabled)
            self.bias_m.requires_grad_(self.parasitic_enabled)

    def forward(self, x):
        # Standard BN with running stats (track) + affine (weight, bias)
        normed = F.batch_norm(
            x, self.running_mean, self.running_var, None, None,
            self.training, self.momentum, self.eps,
        )
        if self.parasitic and self.parasitic_enabled:
            if self.parasitic_type == "gamma_only":
                w = (self.weight + self.weight_m).view(1, -1, 1, 1)
                b = self.bias.view(1, -1, 1, 1)
            elif self.parasitic_type == "beta_only":
                w = self.weight.view(1, -1, 1, 1)
                b = (self.bias + self.bias_m).view(1, -1, 1, 1)
            else:
                w = (self.weight + self.weight_m).view(1, -1, 1, 1)
                b = (self.bias + self.bias_m).view(1, -1, 1, 1)
            return normed * w + b
        else:
            w = self.weight.view(1, -1, 1, 1)
            b = self.bias.view(1, -1, 1, 1)
            return normed * w + b


def replace_bn_with_parasitic(module, parent_name=""):
    """In-place: replace every nn.BatchNorm2d in a module tree with ParasiticBatchNorm2d.
    Copies running stats and affine params verbatim from the original BN."""
    n_replaced = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d) and not isinstance(child, ParasiticBatchNorm2d):
            new_bn = ParasiticBatchNorm2d(
                num_features=child.num_features,
                eps=child.eps, momentum=child.momentum,
            )
            with torch.no_grad():
                new_bn.weight.copy_(child.weight.data)
                new_bn.bias.copy_(child.bias.data)
                new_bn.running_mean.copy_(child.running_mean.data)
                new_bn.running_var.copy_(child.running_var.data)
            new_bn.num_batches_tracked.copy_(child.num_batches_tracked)
            setattr(module, child_name, new_bn)
            n_replaced += 1
        else:
            n_replaced += replace_bn_with_parasitic(child, parent_name + "." + child_name)
    return n_replaced


def configure_resnet_parasitic_scope(model, scope="all", parasitic_type="gamma_beta"):
    """Optionally restrict parasitic to certain ResNet stages.

    scope: 'all' | 'last_stage' | 'last_block' | 'head_only'
    """
    pbns = []
    for name, m in model.named_modules():
        if isinstance(m, ParasiticBatchNorm2d):
            pbns.append((name, m))

    if scope == "all":
        keep = set(n for n, _ in pbns)
    elif scope == "last_stage":
        keep = set(n for n, _ in pbns if "layer4" in n or "bottleneck" in n)
    elif scope == "last_block":
        keep = set(n for n, _ in pbns if "layer4.2" in n or "bottleneck" in n)
    elif scope == "head_only":
        keep = set(n for n, _ in pbns if "bottleneck" in n)
    else:
        raise ValueError(scope)

    for n, m in pbns:
        m.configure_parasitic(enabled=(n in keep), parasitic_type=parasitic_type)
    return len(keep), len(pbns)
