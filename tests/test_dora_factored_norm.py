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

import torch
from torch import nn

from peft.tuners.lora import dora as dora_module
from peft.tuners.lora.dora import DoraLinearLayer
from peft.tuners.lora.factored_norm import weight_norm_from_factors


def make_factors(d_in, d_out, r):
    """Two bias-free Linear layers standing in for lora_A (d_in -> r) and lora_B (r -> d_out)."""
    lora_A = nn.Linear(d_in, r, bias=False)
    lora_B = nn.Linear(r, d_out, bias=False)
    with torch.no_grad():
        lora_A.weight.normal_(0.0, 0.02)
        lora_B.weight.normal_(0.0, 0.02)
    return lora_A, lora_B


def test_weight_norm_from_factors_matches_dense():
    # The factored norm must equal the dense ||W + s * B A|| it replaces.
    torch.manual_seed(0)
    d_out, d_in, r = 16, 32, 4
    scaling = 1.5
    weight = torch.randn(d_out, d_in)
    lora_A, lora_B = make_factors(d_in, d_out, r)
    dense = torch.linalg.norm(weight + scaling * (lora_B.weight @ lora_A.weight), dim=1)
    factored = weight_norm_from_factors(weight, lora_A.weight, lora_B.weight, scaling)
    torch.testing.assert_close(factored, dense, rtol=1e-4, atol=1e-5)


def test_weight_norm_from_factors_matches_dense_fan_in_fan_out():
    # With fan_in_fan_out=True the base weight is stored transposed; result must still match.
    torch.manual_seed(0)
    d_out, d_in, r = 13, 21, 5
    scaling = 0.75
    weight = torch.randn(d_in, d_out)  # stored transposed
    lora_A, lora_B = make_factors(d_in, d_out, r)
    dense = torch.linalg.norm(weight.t() + scaling * (lora_B.weight @ lora_A.weight), dim=1)
    factored = weight_norm_from_factors(weight, lora_A.weight, lora_B.weight, scaling, fan_in_fan_out=True)
    torch.testing.assert_close(factored, dense, rtol=1e-4, atol=1e-5)


def test_dora_linear_forward_factored_matches_default():
    # Integration: exercise the wiring in DoraLinearLayer.forward (from the non-new dora module) by
    # comparing the factored-norm path against the default materializing path on identical adapters.
    torch.manual_seed(0)
    d_out, d_in, r = 12, 24, 4
    scaling = 2.0
    base = nn.Linear(d_in, d_out, bias=False)
    with torch.no_grad():
        base.weight.normal_(0.0, 0.1)
    lora_A, lora_B = make_factors(d_in, d_out, r)

    dora_default = DoraLinearLayer(fan_in_fan_out=False)
    dora_default.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=scaling)
    dora_factored = DoraLinearLayer(fan_in_fan_out=False)
    dora_factored.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=scaling)

    x = torch.randn(3, d_in)
    out_default = dora_default(x, lora_A=lora_A, lora_B=lora_B, scaling=scaling, base_layer=base)

    saved = dora_module.ENABLE_FACTORED_WEIGHT_NORM
    try:
        dora_module.ENABLE_FACTORED_WEIGHT_NORM = True
        out_factored = dora_factored(x, lora_A=lora_A, lora_B=lora_B, scaling=scaling, base_layer=base)
    finally:
        dora_module.ENABLE_FACTORED_WEIGHT_NORM = saved

    torch.testing.assert_close(out_factored, out_default, rtol=1e-3, atol=1e-4)
