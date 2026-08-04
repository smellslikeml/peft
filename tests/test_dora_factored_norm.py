# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the factored DoRA weight-norm optimization.

The factored norm computes ``||W + s·BA||`` column-wise without materializing the dense ``B @ A`` product. These tests
check that it is numerically equivalent to the existing dense path (both the standalone helper and the wired-in
``DoraLinearLayer.forward``).
"""

import pytest
import torch
from torch import nn

import peft.tuners.lora.dora as dora_module
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.dora import DoraLinearLayer
from peft.tuners.lora.factored_weight_norm import factored_weight_norm


@pytest.fixture(autouse=True)
def restore_flag():
    # Ensure the global opt-in flag never leaks between tests.
    original = dora_module.USE_FACTORED_DORA_NORM
    yield
    dora_module.USE_FACTORED_DORA_NORM = original


@pytest.mark.parametrize("scaling", [1.0, 0.5, 2.0])
def test_factored_norm_matches_dense(scaling):
    torch.manual_seed(0)
    d_out, d_in, r = 32, 48, 4
    base_weight = torch.randn(d_out, d_in)
    lora_A = torch.randn(r, d_in)
    lora_B = torch.randn(d_out, r)

    dense = torch.linalg.norm(base_weight + scaling * (lora_B @ lora_A), dim=1)
    factored = factored_weight_norm(base_weight, lora_A, lora_B, scaling)

    assert factored.shape == (d_out,)
    assert torch.allclose(dense, factored, atol=1e-5, rtol=1e-5)


def test_factored_norm_matches_dora_layer_dense_path():
    # The factored helper must agree with the non-new DoraLinearLayer.get_weight_norm dense implementation.
    torch.manual_seed(1)
    d_out, d_in, r = 16, 24, 3
    base_weight = torch.randn(d_out, d_in)
    lora_A = torch.randn(r, d_in)
    lora_B = torch.randn(d_out, r)
    scaling = 0.75

    layer = DoraLinearLayer(fan_in_fan_out=False)
    reference = layer.get_weight_norm(weight=base_weight, lora_weight=lora_B @ lora_A, scaling=scaling)
    factored = factored_weight_norm(base_weight, lora_A, lora_B, scaling)

    assert torch.allclose(reference, factored, atol=1e-5, rtol=1e-5)


def test_dora_forward_factored_matches_dense():
    # End-to-end: enabling the factored norm in DoraLinearLayer.forward must not change the model output.
    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(64, 48)

        def forward(self, x):
            return self.linear(x)

    torch.manual_seed(2)
    model = MyModule().eval()
    # init_lora_weights=False makes DoRA a non-trivial transform, so the norm actually matters.
    config = LoraConfig(target_modules=["linear"], use_dora=True, init_lora_weights=False)
    model = get_peft_model(model, config).eval()

    data = torch.randn(8, 64)

    assert dora_module.USE_FACTORED_DORA_NORM is False
    with torch.no_grad():
        output_dense = model(data)

    dora_module.USE_FACTORED_DORA_NORM = True
    with torch.no_grad():
        output_factored = model(data)

    assert torch.allclose(output_dense, output_factored, atol=1e-5, rtol=1e-5)
    # sanity: DoRA is genuinely active (not a no-op), so the norm path was exercised
    assert not torch.allclose(output_dense, torch.zeros_like(output_dense))
