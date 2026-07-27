# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for Supra's budget-split + bottom-k selection utilities."""

from __future__ import annotations

import pytest
import torch

from peft.tuners.supra import bottomk_magnitude_indices, compute_supra_budget


# --- compute_supra_budget ---------------------------------------------------


class TestComputeSupraBudget:
    def test_pure_lora_when_ratio_one(self):
        r_lora, sparse_k = compute_supra_budget(r=8, lora_ratio=1.0, in_features=64, out_features=128)
        assert r_lora == 8
        assert sparse_k == 0

    def test_pure_super_when_ratio_zero(self):
        r_lora, sparse_k = compute_supra_budget(r=8, lora_ratio=0.0, in_features=64, out_features=128)
        assert r_lora == 0
        assert sparse_k == 8 * (64 + 128)

    def test_half_split_matches_paper_lambda_half(self):
        # Paper Eq. 14: r_lora = floor(λ · r) with λ=0.5, r=8 → r_lora=4.
        r_lora, sparse_k = compute_supra_budget(r=8, lora_ratio=0.5, in_features=64, out_features=128)
        assert r_lora == 4
        assert sparse_k == 4 * (64 + 128)  # remainder in scalar count

    def test_total_scalar_count_invariant_across_ratios(self):
        # Whatever the split, total trainable scalars must equal r · (in + out).
        # This is the paper's matched-budget claim.
        r, in_f, out_f = 8, 64, 128
        expected_total = r * (in_f + out_f)
        for ratio in [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]:
            r_lora, sparse_k = compute_supra_budget(r, ratio, in_f, out_f)
            actual_total = r_lora * (in_f + out_f) + sparse_k
            assert actual_total == expected_total, f"ratio={ratio}: {actual_total} != {expected_total}"

    def test_floor_behavior_on_fractional_rank(self):
        # λ=0.3, r=8 → 0.3·8=2.4 → floor → r_lora=2, sparse gets the extra 0.4 rank worth of budget.
        r_lora, sparse_k = compute_supra_budget(r=8, lora_ratio=0.3, in_features=100, out_features=200)
        assert r_lora == 2
        assert sparse_k == (8 - 2) * (100 + 200)

    def test_rejects_ratio_outside_unit_interval(self):
        with pytest.raises(ValueError, match="lora_ratio"):
            compute_supra_budget(r=8, lora_ratio=-0.1, in_features=64, out_features=128)
        with pytest.raises(ValueError, match="lora_ratio"):
            compute_supra_budget(r=8, lora_ratio=1.5, in_features=64, out_features=128)

    def test_rejects_negative_dims(self):
        with pytest.raises(ValueError, match="non-negative"):
            compute_supra_budget(r=-1, lora_ratio=0.5, in_features=64, out_features=128)


# --- bottomk_magnitude_indices ---------------------------------------------


class TestBottomKMagnitudeIndices:
    def test_picks_smallest_absolute_values(self):
        # Weight with clearly-ordered magnitudes: [-3, 1, -2, 0.5]
        # abs:                                    [ 3, 1,  2, 0.5]
        # bottom-2 by magnitude: indices {3, 1}   (values 0.5, 1)
        w = torch.tensor([-3.0, 1.0, -2.0, 0.5])
        idx = bottomk_magnitude_indices(w, k=2)
        assert set(idx.tolist()) == {1, 3}

    def test_returns_int64_on_weight_device(self):
        w = torch.randn(16, 32)
        idx = bottomk_magnitude_indices(w, k=10)
        assert idx.dtype == torch.int64
        assert idx.device == w.device

    def test_k_zero_returns_empty_tensor(self):
        w = torch.randn(16, 32)
        idx = bottomk_magnitude_indices(w, k=0)
        assert idx.shape == (0,)
        assert idx.dtype == torch.int64

    def test_k_larger_than_numel_returns_all_indices(self):
        w = torch.randn(4, 4)   # 16 elements
        idx = bottomk_magnitude_indices(w, k=100)
        assert idx.shape == (16,)
        assert set(idx.tolist()) == set(range(16))

    def test_rejects_negative_k(self):
        w = torch.randn(4, 4)
        with pytest.raises(ValueError, match="k must be non-negative"):
            bottomk_magnitude_indices(w, k=-1)

    def test_ignores_sign_uses_magnitude(self):
        # Highest-magnitude (both sign) should NOT be picked as bottom-k.
        w = torch.tensor([[10.0, -0.1], [0.2, -10.0]])
        idx = bottomk_magnitude_indices(w, k=2)
        # bottom-2 magnitudes are 0.1 and 0.2 at flat positions 1, 2
        assert set(idx.tolist()) == {1, 2}

    def test_indices_flatten_correctly(self):
        # Sanity: indices unravel back to expected 2-D positions.
        w = torch.tensor([[10.0, 0.1], [0.2, 10.0]])
        idx = bottomk_magnitude_indices(w, k=2)
        rows, cols = torch.unravel_index(idx, w.shape)
        picked = w[rows, cols].abs().sort().values
        expected = torch.tensor([0.1, 0.2])
        assert torch.allclose(picked, expected)

    def test_does_not_backprop_through_selection(self):
        # Selection uses .detach(); the returned indices should not carry grad.
        w = torch.randn(8, 8, requires_grad=True)
        idx = bottomk_magnitude_indices(w, k=5)
        assert not idx.requires_grad
