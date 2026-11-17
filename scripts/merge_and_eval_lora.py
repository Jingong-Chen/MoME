"""Merge LoRA adapters into base linear weights, save a standard ckpt, then eval.

Loads the LoRA-trained ckpt, identifies LoRALinear modules, computes
  W' = W + scale * (B @ A)
and saves a state_dict with `*.base.weight` -> `*.weight` so it loads as a vanilla model.

Usage:
  python scripts/merge_and_eval_lora.py --in_ckpt logs/lora_seed42/lora_unlearn_final.pth \
                                       --out_ckpt logs/lora_seed42/lora_merged.pth
"""

import argparse
import os
import re
from collections import OrderedDict

import torch


def merge_lora_state_dict(sd, scale_default=8.0 / 4):
    """Merge each (base.weight, base.bias, lora_A, lora_B) group → (weight, bias)."""
    new_sd = OrderedDict()
    base_weights = {}
    base_biases = {}
    lora_As = {}
    lora_Bs = {}
    other = OrderedDict()

    for k, v in sd.items():
        # Detect LoRALinear sub-keys
        if k.endswith(".base.weight"):
            prefix = k[: -len(".base.weight")]
            base_weights[prefix] = v
        elif k.endswith(".base.bias"):
            prefix = k[: -len(".base.bias")]
            base_biases[prefix] = v
        elif k.endswith(".lora_A"):
            prefix = k[: -len(".lora_A")]
            lora_As[prefix] = v
        elif k.endswith(".lora_B"):
            prefix = k[: -len(".lora_B")]
            lora_Bs[prefix] = v
        elif k.endswith(".scale"):
            # Skip; we use scale_default unless a scale is in the dict
            pass
        else:
            other[k] = v

    # Merge each LoRA group
    merged_count = 0
    for prefix in base_weights:
        W = base_weights[prefix].float()
        bias = base_biases.get(prefix, None)
        A = lora_As.get(prefix, None)
        B = lora_Bs.get(prefix, None)
        if A is not None and B is not None:
            # W' = W + scale * (B @ A)
            delta = (B.float() @ A.float()) * scale_default
            W_new = (W + delta).to(base_weights[prefix].dtype)
            merged_count += 1
        else:
            W_new = base_weights[prefix]
        new_sd[prefix + ".weight"] = W_new
        if bias is not None:
            new_sd[prefix + ".bias"] = bias

    new_sd.update(other)
    return new_sd, merged_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_ckpt", required=True)
    parser.add_argument("--out_ckpt", required=True)
    parser.add_argument("--scale", default=8.0 / 4, type=float, help="alpha/r used during training")
    args = parser.parse_args()

    sd = torch.load(args.in_ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    new_sd, n = merge_lora_state_dict(sd, scale_default=args.scale)
    print(f"Merged {n} LoRA layer(s).")
    torch.save(new_sd, args.out_ckpt)
    print(f"Saved merged ckpt to {args.out_ckpt}")
