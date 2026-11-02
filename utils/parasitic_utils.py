"""寄生模块工具函数。"""

import torch


def set_parasitic_mode(model, mode=True):
    """设置模型中所有寄生模块（含 LN/BN/Linear 包装层）的运行模式。"""
    for module in model.modules():
        if hasattr(module, "parasitic"):
            module.parasitic = mode


def get_parasitic_parameters(model):
    """返回寄生参数列表 (weight_m, bias_m)。"""
    params = []
    for name, param in model.named_parameters():
        if ("weight_m" in name or "bias_m" in name) and param.requires_grad:
            params.append(param)
    return params


def get_core_ln_parameters(model):
    """返回核心 LayerNorm 参数列表 (weight, bias)。"""
    params = []
    for name, param in model.named_parameters():
        if ("weight_m" not in name and "bias_m" not in name) and param.requires_grad:
            params.append(param)
    return params


def count_parasitic_parameters(model):
    """统计寄生参数量并打印占比。"""
    parasitic_params = get_parasitic_parameters(model)
    parasitic_total = sum(p.numel() for p in parasitic_params)
    all_total = sum(p.numel() for p in model.parameters())
    ratio = 100.0 * parasitic_total / max(all_total, 1)
    print(f"Parasitic parameters: {parasitic_total:,} ({ratio:.4f}% of backbone)")
    return parasitic_total


def freeze_parasitic_parameters(model, freeze=True, zero_fill=True):
    """冻结/解冻寄生参数，可用于标准 baseline 训练。"""
    for name, param in model.named_parameters():
        if "weight_m" in name or "bias_m" in name:
            if zero_fill:
                with torch.no_grad():
                    param.zero_()
            param.requires_grad_(not freeze)


def build_cfg_for_ckpt(cfg, ckpt_path):
    """Return a config whose parasitic-Linear setting matches what is inside `ckpt_path`.

    A MoME core carries the rank-r parasitic branch; a plain retrain does not. The LoRA build
    renames every wrapped Linear parameter to ``...base.weight`` and adds ``weight_m_A/B``, so a
    build that disagrees with the checkpoint raises KeyError on the first block. Six scripts had
    grown their own copy of this check, and the ones that lacked it failed whenever a core and a
    reference were loaded by the same tool.
    """
    import torch
    sd = torch.load(ckpt_path, map_location="cpu")
    keys = sd.keys() if hasattr(sd, "keys") else []
    keys = list(keys)
    has_lora = any("weight_m_A" in k for k in keys)
    # The consolidated Full artifact attaches FULL-capacity Linear shadows, ...attn.qkv.weight_m,
    # rather than the rank-r pair, so the mode has to be read off the keys as well as the flag.
    has_fulllin = any(k.endswith("attn.qkv.weight_m") for k in keys)
    # The consolidated Full artifact also carries a parasitic shadow on the position embedding,
    # so the embed flag has to follow the checkpoint for the same reason the linear flag does.
    has_embed = any("pos_embed_weight_m" in k for k in keys)
    # It carries two more shadows that this function used to ignore: the classifier head and the
    # BatchNorm running statistics. Dropping them does not raise, it silently pairs the Full
    # backbone with the CORE head and CORE running stats, and the logits are then meaningless.
    # That is what put the consolidated Market Full at entropy AUC 0.41, below chance, for a model
    # that trained on the forget set and retrieves it at 99.8 U mAP. RegDB read 1.00 for the same
    # kind of artifact only because its row was measured with these two flags passed by hand.
    # Every checkpoint carries classifier.weight_m, because the head is wrapped whether or not the
    # shadow is ever trained, so key presence decides nothing here. A plain retrain and a MoME core
    # both leave it exactly zero; only the consolidated Full has it populated. Test the values.
    _hw = sd.get("classifier.weight_m")
    has_head = _hw is not None and bool(_hw.abs().sum() > 0)
    # BN statistic shadows are written only when they exist, so key presence is the right test:
    # a plain retrain has none, a MoME core and the consolidated Full each have ten.
    has_bnstats = any(k.endswith("running_mean_m") or k.endswith("running_var_m") for k in keys)
    del sd
    out = cfg.clone()
    out.defrost()
    out.MODEL.PARASITIC_LINEAR = bool(has_lora or has_fulllin)
    out.MODEL.PARASITIC_LINEAR_MODE = "full" if has_fulllin else "lora"
    out.MODEL.PARASITIC_EMBED = bool(has_embed)
    out.MODEL.PARASITIC_HEAD_ON = bool(has_head)
    out.MODEL.PARASITIC_BN_STATS = bool(has_bnstats)
    out.freeze()
    return out
