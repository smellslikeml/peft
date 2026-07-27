# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""SupraModel-level smoke tests — module replacement, budget invariant, identity-at-init, single train step."""

import pytest
import torch
from torch import nn

from peft import SupraConfig, SupraModel
from peft.tuners.supra import Linear as SupraLinear


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(32, 64, bias=False)
        self.v_proj = nn.Linear(32, 64, bias=False)
        self.other = nn.Linear(64, 8, bias=False)

    def forward(self, x):
        return self.other(self.q_proj(x) + self.v_proj(x))


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return Tiny()


def _wrap(model, **kwargs):
    cfg = SupraConfig(target_modules=["q_proj", "v_proj"], **kwargs)
    return SupraModel(model, cfg, adapter_name="default")


class TestModuleReplacement:
    def test_target_modules_are_replaced(self, tiny_model):
        wrapped = _wrap(tiny_model, r=8, lora_ratio=0.5)
        assert isinstance(wrapped.model.q_proj, SupraLinear)
        assert isinstance(wrapped.model.v_proj, SupraLinear)

    def test_untargeted_modules_are_untouched(self, tiny_model):
        wrapped = _wrap(tiny_model, r=8, lora_ratio=0.5)
        assert isinstance(wrapped.model.other, nn.Linear)
        assert not isinstance(wrapped.model.other, SupraLinear)


class TestBudget:
    @pytest.mark.parametrize("lora_ratio", [0.0, 0.3, 0.5, 0.8, 1.0])
    def test_budget_invariant_across_ratios(self, tiny_model, lora_ratio):
        wrapped = _wrap(tiny_model, r=8, lora_ratio=lora_ratio)
        counts = wrapped.get_trainable_parameters_count("default")
        # Two targets, each contributes r · (in + out) = 8 · (32 + 64) = 768 scalars.
        expected = 8 * (32 + 64) * 2
        assert counts["sparse_parameters"] + counts["lora_parameters"] == expected


class TestForward:
    def test_identity_at_init(self, tiny_model):
        # Snapshot base weights BEFORE wrapping so we can compare against an unadapted forward.
        with torch.no_grad():
            snap = {n: p.clone() for n, p in tiny_model.state_dict().items()}
        wrapped = _wrap(tiny_model, r=8, lora_ratio=0.5)

        ref = Tiny()
        ref.load_state_dict(snap)

        x = torch.randn(4, 32)
        y_wrapped = wrapped(x)
        y_ref = ref(x)
        assert torch.allclose(y_wrapped, y_ref, atol=1e-6), \
            f"identity broken at init: max diff {(y_wrapped - y_ref).abs().max().item()}"


class TestTrainStep:
    def test_single_step_updates_sparse_and_lora(self, tiny_model):
        wrapped = _wrap(tiny_model, r=8, lora_ratio=0.5)
        trainable = [p for p in wrapped.parameters() if p.requires_grad]
        assert len(trainable) > 0, "no trainable params"
        opt = torch.optim.SGD(trainable, lr=1e-2)

        # Snapshot the trainable params so we can confirm at least one moved.
        before = {id(p): p.detach().clone() for p in trainable}

        x = torch.randn(4, 32)
        target = torch.randn(4, 8)
        y = wrapped(x)
        loss = ((y - target) ** 2).mean()
        loss.backward()
        opt.step()

        moved = [not torch.equal(before[id(p)], p.detach()) for p in trainable]
        assert any(moved), "no trainable parameter moved after one SGD step"

    def test_base_weight_stays_frozen(self, tiny_model):
        wrapped = _wrap(tiny_model, r=8, lora_ratio=0.5)
        base_before = wrapped.model.q_proj.get_base_layer().weight.detach().clone()

        x = torch.randn(4, 32)
        target = torch.randn(4, 8)
        loss = ((wrapped(x) - target) ** 2).mean()
        loss.backward()

        base_after = wrapped.model.q_proj.get_base_layer().weight.detach().clone()
        assert torch.equal(base_before, base_after), "base weight was mutated during backward"


class TestCalibrateStub:
    def test_calibrate_saliency_raises_not_implemented(self, tiny_model):
        wrapped = _wrap(tiny_model, r=8, lora_ratio=0.5, scoring_method="magnitude")
        with pytest.raises(NotImplementedError, match="Wanda calibration"):
            wrapped.calibrate_saliency([torch.randn(1, 32)])
