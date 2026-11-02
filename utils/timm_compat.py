"""Compatibility helpers for legacy timm on new torch versions."""

import collections.abc
import sys
import types


def ensure_torch_six_compat():
    """Provide a minimal `torch._six` shim for older timm releases."""
    if "torch._six" in sys.modules:
        return

    module = types.ModuleType("torch._six")
    module.container_abcs = collections.abc
    module.string_classes = (str, bytes)
    module.int_classes = (int,)
    sys.modules["torch._six"] = module
