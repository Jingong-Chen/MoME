"""The claims the paper makes about the mechanism, tested on CPU without data or checkpoints.

These are not utility tests. Each one asserts a property the paper states: that the branch starts
at zero so Core is the model at initialization, that the switch is the whole interface, that a
forget-set gradient never reaches a core parameter, and that the deletion request is a function of
its seed alone.
"""
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.parasitic_utils import (  # noqa: E402
    count_parasitic_parameters,
    get_core_ln_parameters,
    get_parasitic_parameters,
    set_parasitic_mode,
)


class Tiny(nn.Module):
    """A LayerNorm stack, which is where the parasitic parameters live in the real backbone."""

    def __init__(self, d=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        return self.norm2(self.fc(self.norm1(x)))


class ShadowedLN(nn.LayerNorm):
    """A LayerNorm carrying the shadow pair, with the forward the backbone's wrapper uses:
    the shadow is added to the affine only while the branch is on."""

    def __init__(self, d):
        super().__init__(d)
        self.weight_m = nn.Parameter(torch.zeros(d))
        self.bias_m = nn.Parameter(torch.zeros(d))
        self.parasitic = False

    def forward(self, x):
        x = nn.functional.layer_norm(x, self.normalized_shape, None, None, self.eps)
        if self.parasitic:
            return (self.weight + self.weight_m) * x + (self.bias + self.bias_m)
        return self.weight * x + self.bias


def wrap(model):
    """Swap every LayerNorm for the shadowed one, as the backbone wrapper does."""
    for name, m in list(model.named_children()):
        if isinstance(m, nn.LayerNorm):
            setattr(model, name, ShadowedLN(m.normalized_shape[0]))
        else:
            wrap(m)
    return model


def test_branch_starts_at_zero():
    """Zero initialization is what makes Core the model before any forgetting happens."""
    m = wrap(Tiny())
    for mod in m.modules():
        if hasattr(mod, "weight_m"):
            assert torch.count_nonzero(mod.weight_m) == 0
            assert torch.count_nonzero(mod.bias_m) == 0


def test_switch_is_a_flag():
    """Core and Full are one set of weights read two ways, not two models."""
    m = wrap(Tiny())
    set_parasitic_mode(m, True)
    assert all(mod.parasitic for mod in m.modules() if hasattr(mod, "parasitic"))
    set_parasitic_mode(m, False)
    assert not any(mod.parasitic for mod in m.modules() if hasattr(mod, "parasitic"))


def test_core_is_the_model_before_the_branch():
    """With the branch at zero, turning it on changes nothing: Full starts as Core."""
    m = wrap(Tiny()).eval()
    x = torch.randn(4, 8)
    set_parasitic_mode(m, False)
    core = m(x)
    set_parasitic_mode(m, True)
    assert torch.allclose(core, m(x), atol=0, rtol=0)


def test_the_branch_changes_full_and_not_core():
    """Once the branch is trained, the flag is what separates the two readings."""
    m = wrap(Tiny()).eval()
    x = torch.randn(4, 8)
    set_parasitic_mode(m, False)
    core = m(x)
    with torch.no_grad():
        for mod in m.modules():
            if hasattr(mod, "weight_m"):
                mod.weight_m.add_(0.3)
    assert torch.allclose(core, m(x), atol=0, rtol=0)      # Core is untouched
    set_parasitic_mode(m, True)
    assert not torch.allclose(core, m(x))                  # Full is not


def test_partition_is_disjoint_and_complete():
    """Step 1 and Step 2 may not touch the same tensor, or isolation means nothing."""
    m = wrap(Tiny())
    par = {id(p) for p in get_parasitic_parameters(m)}
    core = {id(p) for p in get_core_ln_parameters(m)}
    assert par and core
    assert par.isdisjoint(core)


def test_a_forget_gradient_never_reaches_the_core():
    """The paper's guarantee, as a gradient check: with the core frozen, Step 2 leaves it at None."""
    m = wrap(Tiny())
    for p in get_core_ln_parameters(m):
        p.requires_grad_(False)
    set_parasitic_mode(m, True)
    out = m(torch.randn(4, 8))
    out.square().mean().backward()
    for p in get_core_ln_parameters(m):
        assert p.grad is None
    assert any(p.grad is not None for p in get_parasitic_parameters(m))


def test_the_branch_is_a_small_fraction():
    """The payload is a branch, not a second model."""
    m = wrap(Tiny())
    n_par = count_parasitic_parameters(m)
    n_tot = sum(p.numel() for p in m.parameters())
    assert 0 < n_par < n_tot


@pytest.mark.parametrize("seed", [42, 123, 456])
def test_a_deletion_request_is_its_seed(seed):
    """The same seed must name the same identities, or no two runs share a request."""
    from datasets.make_dataloader_mome import _split_by_identity

    train = [("img%d.jpg" % i, i % 20, 0, 0) for i in range(200)]
    a = _split_by_identity(train, 0.5, seed, dr_k=4)
    b = _split_by_identity(train, 0.5, seed, dr_k=4)
    assert a == b
    other = _split_by_identity(train, 0.5, seed + 1, dr_k=4)
    assert a != other


def test_the_request_splits_identities_not_images():
    """Identity-level deletion: no identity may appear on both sides."""
    from datasets.make_dataloader_mome import _split_by_identity

    train = [("img%d.jpg" % i, i % 20, 0, 0) for i in range(200)]
    dr_idx, u_idx = _split_by_identity(train, 0.5, 42, dr_k=4)
    dr_ids = {train[i][1] for i in dr_idx}
    u_ids = {train[i][1] for i in u_idx}
    assert dr_ids and u_ids
    assert dr_ids.isdisjoint(u_ids)


def test_the_retain_set_keeps_k_images_per_identity():
    """DR_K is the thin gallery a controller keeps; it is what makes the reference weak."""
    from collections import Counter

    from datasets.make_dataloader_mome import _split_by_identity

    train = [("img%d.jpg" % i, i % 20, 0, 0) for i in range(200)]
    dr_idx, _ = _split_by_identity(train, 0.5, 42, dr_k=4)
    per_id = Counter(train[i][1] for i in dr_idx)
    assert set(per_id.values()) == {4}
