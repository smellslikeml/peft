# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Smoke tests for SupraLayer — allocation, forward, grad routing, merge/unmerge."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from peft.tuners.supra import Linear as SupraLinear
from peft.tuners.supra import SupraConfig


# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def base_linear():
    torch.manual_seed(0)
    layer = nn.Linear(64, 128, bias=True)
    return layer


def _make_supra(base, r=8, lora_ratio=0.5, lora_alpha=16, lora_dropout=0.0, init_weights=True):
    config = SupraConfig(
        r=r,
        lora_ratio=lora_ratio,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        init_weights=init_weights,
        target_modules=[],   # not used at Linear-layer-init level
    )
    return SupraLinear(base, adapter_name="default", config=config)


# --- allocation sanity ------------------------------------------------------


class TestAllocation:
    def test_budget_split_matches_paper_lambda_half(self, base_linear):
        supra = _make_supra(base_linear, r=8, lora_ratio=0.5)
        # r=8, λ=0.5 → r_lora=4, sparse_k = 4 · (64+128) = 768
        assert supra.supra_r_lora["default"] == 4
        assert supra.supra_sparse_values["default"].shape == (768,)
        assert supra.supra_lora_A["default"].weight.shape == (4, 64)
        assert supra.supra_lora_B["default"].weight.shape == (128, 4)

    def test_pure_lora_when_ratio_one(self, base_linear):
        supra = _make_supra(base_linear, r=8, lora_ratio=1.0)
        assert supra.supra_r_lora["default"] == 8
        assert supra.supra_sparse_values["default"].shape == (0,)
        assert supra.supra_lora_A["default"].weight.shape == (8, 64)

    def test_pure_super_when_ratio_zero(self, base_linear):
        supra = _make_supra(base_linear, r=8, lora_ratio=0.0)
        assert supra.supra_r_lora["default"] == 0
        assert supra.supra_sparse_values["default"].shape == (8 * (64 + 128),)
        # No LoRA factors allocated in pure-Super regime.
        assert "default" not in supra.supra_lora_A

    def test_lora_B_initialized_to_zero(self, base_linear):
        supra = _make_supra(base_linear, r=8, lora_ratio=0.5)
        assert torch.allclose(supra.supra_lora_B["default"].weight, torch.zeros_like(supra.supra_lora_B["default"].weight))

    def test_sparse_values_initialized_to_zero_by_default(self, base_linear):
        supra = _make_supra(base_linear, r=8, lora_ratio=0.5, init_weights=True)
        assert torch.allclose(supra.supra_sparse_values["default"], torch.zeros_like(supra.supra_sparse_values["default"]))


# --- forward correctness ----------------------------------------------------


class TestForward:
    def test_identity_at_init(self, base_linear):
        """At step 0: sparse values = 0, LoRA B = 0 → adapter should be a no-op."""
        supra = _make_supra(base_linear, r=8, lora_ratio=0.5, init_weights=True, lora_dropout=0.0)
        x = torch.randn(3, 64)
        y_base = base_linear(x)
        y_supra = supra(x)
        assert torch.allclose(y_supra, y_base, atol=1e-6)

    def test_nonzero_sparse_values_perturb_output(self, base_linear):
        supra = _make_supra(base_linear, r=8, lora_ratio=0.5, init_weights=True)
        # Manually seed the sparse values so the update is non-trivial.
        with torch.no_grad():
            supra.supra_sparse_values["default"].fill_(0.1)
        x = torch.randn(3, 64)
        y_base = base_linear(x)
        y_supra = supra(x)
        assert not torch.allclose(y_supra, y_base, atol=1e-4)

    def test_nonzero_lora_B_perturbs_output(self, base_linear):
        supra = _make_supra(base_linear, r=8, lora_ratio=0.5, init_weights=True)
        # Bump lora_B off zero so the low-rank term contributes.
        with torch.no_grad():
            supra.supra_lora_B["default"].weight.fill_(0.01)
        x = torch.randn(3, 64)
        y_base = base_linear(x)
        y_supra = supra(x)
        assert not torch.allclose(y_supra, y_base, atol=1e-4)


# --- gradient routing -------------------------------------------------------


class TestGradientRouting:
    def test_base_weight_gets_no_gradient(self, base_linear):
        supra = _make_supra(base_linear, r=8, lora_ratio=0.5)
        base_linear.weight.requires_grad_(True)  # even if flipped on, no grad should flow
        base_linear.bias.requires_grad_(True)

        x = torch.randn(3, 64, requires_grad=True)
        y = supra(x).sum()
        y.backward()

        # Sparse + LoRA A + LoRA B should have grads; base weight should be None or zero.
        assert supra.supra_sparse_values["default"].grad is not None
        assert supra.supra_lora_A["default"].weight.grad is not None
        assert supra.supra_lora_B["default"].weight.grad is not None
        # Base weight: DensePlusSparseLinear explicitly returns None for weight grad.
        # But when native scatter_add is used (multi-adapter case), autograd would fill it.
        # Here we're single-adapter, so grad should be None.
        assert base_linear.weight.grad is None or torch.allclose(base_linear.weight.grad, torch.zeros_like(base_linear.weight))

    def test_sparse_grad_only_at_indices(self, base_linear):
        supra = _make_supra(base_linear, r=8, lora_ratio=0.0)  # pure Super so no LoRA distraction
        x = torch.randn(3, 64)
        y = supra(x).sum()
        y.backward()
        grad = supra.supra_sparse_values["default"].grad
        # Grad shape must match the sparse values shape (1-D, len = sparse_k).
        assert grad.shape == supra.supra_sparse_values["default"].shape


# --- merge / unmerge --------------------------------------------------------


class TestMergeUnmerge:
    def test_merge_preserves_forward(self, base_linear):
        supra = _make_supra(base_linear, r=8, lora_ratio=0.5, init_weights=True)
        # Move both components off zero so merge has something to fold in.
        with torch.no_grad():
            supra.supra_sparse_values["default"].fill_(0.05)
            supra.supra_lora_B["default"].weight.fill_(0.02)

        x = torch.randn(3, 64)
        y_before = supra(x).detach().clone()
        supra.merge()
        assert supra.merged
        y_after = supra(x).detach().clone()
        assert torch.allclose(y_before, y_after, atol=1e-5), \
            f"merge changed forward output: max diff {(y_before - y_after).abs().max().item()}"

    def test_unmerge_restores_base_weight(self, base_linear):
        original_weight = base_linear.weight.data.clone()
        supra = _make_supra(base_linear, r=8, lora_ratio=0.5, init_weights=True)
        with torch.no_grad():
            supra.supra_sparse_values["default"].fill_(0.05)
            supra.supra_lora_B["default"].weight.fill_(0.02)

        supra.merge()
        assert not torch.allclose(base_linear.weight.data, original_weight, atol=1e-6)
        supra.unmerge()
        assert torch.allclose(base_linear.weight.data, original_weight, atol=1e-5), \
            f"unmerge failed to restore base weight: max diff {(base_linear.weight.data - original_weight).abs().max().item()}"


# --- config validation ------------------------------------------------------


class TestConfig:
    def test_rejects_bad_lora_ratio(self):
        with pytest.raises(ValueError, match="lora_ratio"):
            SupraConfig(r=8, lora_ratio=1.5)

    def test_rejects_bad_r(self):
        with pytest.raises(ValueError, match="^r must"):
            SupraConfig(r=0, lora_ratio=0.5)

    def test_rejects_bad_scoring_method(self):
        with pytest.raises(ValueError, match="scoring_method"):
            SupraConfig(scoring_method="invalid")

    def test_rejects_bad_selection_direction(self):
        with pytest.raises(ValueError, match="selection_direction"):
            SupraConfig(selection_direction="middle")
