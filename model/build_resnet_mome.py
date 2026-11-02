"""ResNet-50 + parasitic BN model wrapper for MoME training.

Uses the existing Backbone class from make_model.py (which has ResNet-50 / BNNeck /
classifier head). Then in-place replaces all BN2d with ParasiticBatchNorm2d, and
the BNNeck (which is already ParasiticBatchNorm1d in make_model.py).
"""

import torch
import torch.nn as nn

from .make_model import Backbone, ParasiticBatchNorm1d, ParasiticLinear
from .parasitic_resnet import (
    ParasiticBatchNorm2d,
    replace_bn_with_parasitic,
    configure_resnet_parasitic_scope,
)


class ResNetMoME(nn.Module):
    """A ResNet-50 ReID model with parasitic BN for MoME training.

    Forward returns (cls_score, global_feat) in train mode (matches training pipeline);
    in eval mode returns the BNNeck (after) feature.
    """

    def __init__(self, num_classes, cfg):
        super().__init__()
        # Reuse the existing ResNet-50 backbone class (loads ImageNet pretrain).
        self.in_planes = 2048
        self._cfg = cfg
        from .backbones.resnet import ResNet, Bottleneck
        self.base = ResNet(last_stride=cfg.MODEL.LAST_STRIDE, block=Bottleneck, layers=[3, 4, 6, 3])
        if cfg.MODEL.PRETRAIN_CHOICE == "imagenet":
            self.base.load_param(cfg.MODEL.PRETRAIN_PATH)
            print("Loading ImageNet ResNet-50 pretrain from {}".format(cfg.MODEL.PRETRAIN_PATH))

        # Replace all BN2d with ParasiticBatchNorm2d (preserving stats).
        n_replaced = replace_bn_with_parasitic(self.base)
        print("ResNet-50 + parasitic BN: replaced {} BN2d layers".format(n_replaced))

        n_active, n_total = configure_resnet_parasitic_scope(
            self, scope=cfg.MODEL.PARASITIC_SCOPE, parasitic_type=cfg.MODEL.PARASITIC_TYPE,
        )
        print("Parasitic BN scope={} type={} enabled {}/{} sites".format(
            cfg.MODEL.PARASITIC_SCOPE, cfg.MODEL.PARASITIC_TYPE, n_active, n_total))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = ParasiticBatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.configure_parasitic(enabled=True, parasitic_type=cfg.MODEL.PARASITIC_TYPE)

        # Classifier (no parasitic — keeps interface consistent with TransReID build).
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)
        self.neck_feat = cfg.TEST.NECK_FEAT

    def forward(self, x, label=None, cam_label=None, view_label=None):
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4]).flatten(1)
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
        print("ResNetMoME loaded {} / {} keys from {}".format(loaded, len(sd), trained_path))
