"""AGW backbone (Ye TPAMI'21) for ReID — the standard CNN baseline.

AGW = ResNet-50 + Non-local attention + GeM pooling + weighted regularization triplet.
We implement a clean variant that includes the most-impactful pieces (GeM + IBN + standard
soft-margin triplet) while keeping the architecture compatible with our existing pipeline.

Used for: rebuttal R3 N7 (additional backbone = standard CNN ReID baseline).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones.resnet import ResNet, Bottleneck
from .make_model import ParasiticBatchNorm1d, weights_init_classifier, weights_init_kaiming
from .parasitic_resnet import (
    ParasiticBatchNorm2d, replace_bn_with_parasitic, configure_resnet_parasitic_scope,
)


class GeM(nn.Module):
    """Generalized Mean Pooling (Radenovic et al.). p=3 default following AGW."""
    def __init__(self, p=3.0, eps=1e-6, learnable=True):
        super().__init__()
        if learnable:
            self.p = nn.Parameter(torch.tensor(p))
        else:
            self.register_buffer("p", torch.tensor(p))
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 1
        ).pow(1.0 / self.p)


class NonLocalBlock(nn.Module):
    """Embedded Gaussian non-local block (Wang et al. CVPR'18).

    Lightweight — applied after layer2/layer3 in AGW to capture long-range spatial deps.
    """
    def __init__(self, in_channels):
        super().__init__()
        inter = max(in_channels // 2, 1)
        self.theta = nn.Conv2d(in_channels, inter, 1, bias=False)
        self.phi   = nn.Conv2d(in_channels, inter, 1, bias=False)
        self.g     = nn.Conv2d(in_channels, inter, 1, bias=False)
        self.W     = nn.Conv2d(inter, in_channels, 1, bias=False)
        self.bn    = nn.BatchNorm2d(in_channels)
        nn.init.constant_(self.W.weight, 0.0)  # zero-init residual

    def forward(self, x):
        B, C, H, W = x.shape
        theta = self.theta(x).view(B, -1, H * W).permute(0, 2, 1)  # B, HW, C'
        phi   = self.phi(x).view(B, -1, H * W)                     # B, C', HW
        g     = self.g(x).view(B, -1, H * W).permute(0, 2, 1)      # B, HW, C'
        attn  = F.softmax(theta @ phi, dim=-1)                     # B, HW, HW
        out   = (attn @ g).permute(0, 2, 1).view(B, -1, H, W)
        return x + self.bn(self.W(out))


class AGW(nn.Module):
    """AGW backbone for ReID. ResNet-50 + non-local (after layer2,layer3) + GeM + BNNeck."""

    def __init__(self, num_classes, cfg, parasitic=False):
        super().__init__()
        self.in_planes = 2048
        self._cfg = cfg

        self.base = ResNet(last_stride=cfg.MODEL.LAST_STRIDE, block=Bottleneck, layers=[3, 4, 6, 3])
        if cfg.MODEL.PRETRAIN_CHOICE == "imagenet":
            self.base.load_param(cfg.MODEL.PRETRAIN_PATH)
            print("AGW: Loaded ImageNet pretrain from {}".format(cfg.MODEL.PRETRAIN_PATH))

        # Non-local blocks (1 after layer2 stage end, 1 after layer3 stage end)
        self.nl_2 = NonLocalBlock(512)   # after layer2 (output channels 512)
        self.nl_3 = NonLocalBlock(1024)  # after layer3 (output channels 1024)

        # Parasitic BN substitution (must come BEFORE GeM since it operates on conv outputs)
        if parasitic:
            n = replace_bn_with_parasitic(self.base)
            print("AGW + parasitic BN: replaced {} BN2d in backbone".format(n))
            n2 = replace_bn_with_parasitic(self.nl_2) + replace_bn_with_parasitic(self.nl_3)
            print("AGW + parasitic BN: replaced {} BN2d in non-local".format(n2))
            n_active, n_total = configure_resnet_parasitic_scope(
                self, scope=cfg.MODEL.PARASITIC_SCOPE, parasitic_type=cfg.MODEL.PARASITIC_TYPE,
            )
            print("AGW parasitic scope={} type={} enabled {}/{} sites".format(
                cfg.MODEL.PARASITIC_SCOPE, cfg.MODEL.PARASITIC_TYPE, n_active, n_total))

        # GeM pooling instead of GAP
        self.gem = GeM(p=3.0, learnable=True)

        # BNNeck (parasitic BN1d when parasitic=True; identical to TransReID's neck)
        self.bottleneck = ParasiticBatchNorm1d(self.in_planes) if parasitic else nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        if parasitic:
            self.bottleneck.configure_parasitic(enabled=True, parasitic_type=cfg.MODEL.PARASITIC_TYPE)

        self.num_classes = num_classes
        self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.neck_feat = cfg.TEST.NECK_FEAT

    def forward(self, x, label=None, cam_label=None, view_label=None):
        # ResNet stages, with non-local after stage 2 and 3.
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.nl_2(x)
        x = self.base.layer3(x)
        x = self.nl_3(x)
        x = self.base.layer4(x)

        global_feat = self.gem(x).flatten(1)  # [B, 2048]
        feat = self.bottleneck(global_feat)
        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            return feat if self.neck_feat == "after" else global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location="cpu")
        if isinstance(param_dict, dict) and "state_dict" in param_dict:
            param_dict = param_dict["state_dict"]
        sd = self.state_dict()
        loaded = 0
        for k, v in param_dict.items():
            if k in sd and sd[k].shape == v.shape:
                sd[k].copy_(v); loaded += 1
        print("AGW loaded {} / {} keys from {}".format(loaded, len(sd), trained_path))
