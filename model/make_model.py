import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import (
    ParasiticLayerNorm,
    deit_small_patch16_224_TransReID,
    vit_base_patch16_224_TransReID,
    vit_small_patch16_224_TransReID,
)
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss

def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def _normalize_parasitic_type(parasitic_type):
    if parasitic_type is None:
        return "gamma_beta"
    value = str(parasitic_type).lower()
    alias = {
        "gamma+beta": "gamma_beta",
        "both": "gamma_beta",
        "gamma": "gamma_only",
        "beta": "beta_only",
    }
    value = alias.get(value, value)
    if value not in ("gamma_beta", "gamma_only", "beta_only"):
        raise ValueError("Unsupported parasitic type: {}".format(parasitic_type))
    return value


class ParasiticBatchNorm1d(nn.BatchNorm1d):
    """
    BNNeck with optional parasitic delta parameters.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=True,
            track_running_stats=True,
        )
        self.weight_m = nn.Parameter(torch.zeros(num_features))
        self.bias_m = nn.Parameter(torch.zeros(num_features))
        # Switchable running statistics: the descriptor is the post-BN feature, so a Full mode
        # meant to reproduce the original model must also use the original model's statistics,
        # not the core's D_r statistics. Off unless parasitic_stats is set.
        self.register_buffer("running_mean_m", torch.zeros(num_features))
        self.register_buffer("running_var_m", torch.ones(num_features))
        self.parasitic_stats = False
        self.parasitic = True
        self.parasitic_enabled = False
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

    def forward(self, input):
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.parasitic and self.parasitic_enabled:
            if self.parasitic_type == "gamma_only":
                weight = self.weight + self.weight_m
                bias = self.bias
            elif self.parasitic_type == "beta_only":
                weight = self.weight
                bias = self.bias + self.bias_m
            else:
                weight = self.weight + self.weight_m
                bias = self.bias + self.bias_m
        else:
            weight = self.weight
            bias = self.bias

        _use_m = self.parasitic and self.parasitic_enabled and getattr(self, "parasitic_stats", False)
        _rm = self.running_mean_m if _use_m else self.running_mean
        _rv = self.running_var_m if _use_m else self.running_var
        return F.batch_norm(
            input,
            _rm if not self.training or self.track_running_stats else None,
            _rv if not self.training or self.track_running_stats else None,
            weight,
            bias,
            self.training or (self.running_mean is None and self.running_var is None),
            exponential_average_factor,
            self.eps,
        )


class ParasiticLinear(nn.Linear):
    """
    Linear head with optional parasitic delta parameters.
    """

    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        self.weight_m = nn.Parameter(torch.zeros_like(self.weight))
        # Always keep beta delta for checkpoint compatibility and beta-only mode.
        self.bias_m = nn.Parameter(torch.zeros(out_features))
        self.parasitic = True
        self.parasitic_enabled = False
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

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        if self.parasitic and self.parasitic_enabled:
            if self.parasitic_type in ("gamma_beta", "gamma_only"):
                weight = weight + self.weight_m
            if self.parasitic_type in ("gamma_beta", "beta_only"):
                if bias is None:
                    bias = self.bias_m
                else:
                    bias = bias + self.bias_m
        return F.linear(input, weight, bias)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label=None, view_label=None):  # label/cam_label/view_label may be passed by processor
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = ParasiticLinear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = ParasiticBatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self._apply_parasitic_config(cfg)

    def _set_head_parasitic(self, enabled, parasitic_type):
        for module in [self.bottleneck, self.classifier]:
            if hasattr(module, "configure_parasitic"):
                module.configure_parasitic(enabled=enabled, parasitic_type=parasitic_type)

    def _apply_parasitic_config(self, cfg):
        scope = str(cfg.MODEL.PARASITIC_SCOPE).lower()
        parasitic_type = _normalize_parasitic_type(cfg.MODEL.PARASITIC_TYPE)
        if hasattr(self.base, "configure_parasitic"):
            self.base.configure_parasitic(scope=scope, parasitic_type=parasitic_type)
        # PARASITIC_EMBED shadows the patch projection, position embedding, class token and SIE
        # table. It used to be reachable only when PARASITIC_LINEAR was also on, so a checkpoint
        # carrying embed shadows but no LoRA branch, which is what the consolidated Full artifact
        # is, could not be rebuilt and raised KeyError on base.pos_embed_weight_m.
        _lin = bool(getattr(cfg.MODEL, "PARASITIC_LINEAR", False))
        _emb = bool(getattr(cfg.MODEL, "PARASITIC_EMBED", False))
        if (_lin or _emb) and hasattr(self.base, "configure_parasitic_linear"):
            self.base.configure_parasitic_linear(
                enabled=_lin, rank=cfg.MODEL.PARASITIC_LORA_RANK, alpha=cfg.MODEL.PARASITIC_LORA_ALPHA,
                mode=getattr(cfg.MODEL, "PARASITIC_LINEAR_MODE", "lora"),
                embed=_emb)
        # JPM extra blocks are part of the deepest branch; keep them active unless head-only mode.
        ln_enabled = scope != "head_only"
        for module in list(self.b1.modules()) + list(self.b2.modules()):
            if isinstance(module, ParasiticLayerNorm):
                module.configure_parasitic(enabled=ln_enabled, parasitic_type=parasitic_type)
        self._set_head_parasitic(enabled=(scope == "head_only"), parasitic_type=parasitic_type)
        print(
            "Parasitic config => scope: {}, type: {}".format(scope, parasitic_type)
        )

    def forward(self, x, label=None, cam_label= None, view_label=None):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)

        feat = self.bottleneck(global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)

            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = ParasiticLinear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_1 = ParasiticLinear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = ParasiticLinear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = ParasiticLinear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = ParasiticLinear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)

        self.bottleneck = ParasiticBatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = ParasiticBatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = ParasiticBatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = ParasiticBatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = ParasiticBatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange
        self._apply_parasitic_config(cfg)

    def _set_head_parasitic(self, enabled, parasitic_type):
        modules = [
            self.bottleneck,
            self.bottleneck_1,
            self.bottleneck_2,
            self.bottleneck_3,
            self.bottleneck_4,
            getattr(self, "classifier", None),
            getattr(self, "classifier_1", None),
            getattr(self, "classifier_2", None),
            getattr(self, "classifier_3", None),
            getattr(self, "classifier_4", None),
        ]
        for module in modules:
            if module is not None and hasattr(module, "configure_parasitic"):
                module.configure_parasitic(enabled=enabled, parasitic_type=parasitic_type)

    def _apply_parasitic_config(self, cfg):
        scope = str(cfg.MODEL.PARASITIC_SCOPE).lower()
        parasitic_type = _normalize_parasitic_type(cfg.MODEL.PARASITIC_TYPE)
        if hasattr(self.base, "configure_parasitic"):
            self.base.configure_parasitic(scope=scope, parasitic_type=parasitic_type)
        # PARASITIC_EMBED shadows the patch projection, position embedding, class token and SIE
        # table. It used to be reachable only when PARASITIC_LINEAR was also on, so a checkpoint
        # carrying embed shadows but no LoRA branch, which is what the consolidated Full artifact
        # is, could not be rebuilt and raised KeyError on base.pos_embed_weight_m.
        _lin = bool(getattr(cfg.MODEL, "PARASITIC_LINEAR", False))
        _emb = bool(getattr(cfg.MODEL, "PARASITIC_EMBED", False))
        if (_lin or _emb) and hasattr(self.base, "configure_parasitic_linear"):
            self.base.configure_parasitic_linear(
                enabled=_lin, rank=cfg.MODEL.PARASITIC_LORA_RANK, alpha=cfg.MODEL.PARASITIC_LORA_ALPHA,
                mode=getattr(cfg.MODEL, "PARASITIC_LINEAR_MODE", "lora"),
                embed=_emb)
            if str(getattr(cfg.MODEL, "PARASITIC_LINEAR_MODE", "lora")).strip().strip("'\"").lower() == "full":
                _wrap_branch_linears_full(self.b1, self.b2)
        self._set_head_parasitic(enabled=(scope == "head_only"), parasitic_type=parasitic_type)
        print(
            "Parasitic config => scope: {}, type: {}".format(scope, parasitic_type)
        )

    def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'

        features = self.base(x, cam_label=cam_label, view_label=view_label)

        # global branch
        b1_feat = self.b1(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0]

        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]
        # lf_1
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = x[:, patch_length:patch_length*2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, patch_length*2:patch_length*3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = x[:, patch_length*3:patch_length*4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        feat = self.bottleneck(global_feat)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)
            return [cls_score, cls_score_1, cls_score_2, cls_score_3,
                        cls_score_4
                        ], [global_feat, local_feat_1, local_feat_2, local_feat_3,
                            local_feat_4]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
            else:
                return torch.cat(
                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}

def _wrap_branch_linears_full(*branches):
    """Wrap qkv/proj/fc1/fc2 inside the JPM b1/b2 copies with full-capacity shadows. These
    blocks are copies of the last transformer block and live outside base.blocks, so
    configure_parasitic_linear never sees them; without shadows their slice of the parameter
    difference is uncarriable and a delta-initialized Full starts far below the Original."""
    from .backbones.vit_pytorch import ParasiticFullLinear
    n = 0
    for br in branches:
        for m in br.modules():
            for attr in ("qkv", "proj", "fc1", "fc2"):
                sub = getattr(m, attr, None)
                if isinstance(sub, nn.Linear):
                    setattr(m, attr, ParasiticFullLinear(sub))
                    n += 1
    print(f"Parasitic full on JPM branch Linears: {n} sites")
    return n


def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
            print('===========building transformer with JPM module ===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
            print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    if bool(getattr(cfg.MODEL, "PARASITIC_HEAD_ON", False)):
        for mod in model.modules():
            if hasattr(mod, "parasitic_enabled") and hasattr(mod, "configure_parasitic"):
                mod.configure_parasitic(enabled=True)
        print('Parasitic head shadows enabled everywhere')
    if bool(getattr(cfg.MODEL, "PARASITIC_BN_STATS", False)):
        n = 0
        for mod in model.modules():
            if hasattr(mod, "parasitic_stats"):
                mod.parasitic_stats = True
                n += 1
        print(f'Parasitic BN statistics switching enabled on {n} modules')
    return model
