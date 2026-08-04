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

"""Structural parity tests for the DoRA-factored reference kernel.

Asserts that the pure-PyTorch reference (``dora_factored_forward``) is numerically equivalent to a
naive dense DoRA effective-weight baseline written inline below. Mirrors the discipline of PEFT's
merged ``tests/test_dora_factored_norm.py`` (huggingface/peft#3382): deterministic seed, small
shapes, dense-vs-factored comparison. Runs on CPU with no GPU / no PEFT / no Triton.
"""

import pytest
import torch
from dora_factored import dora_factored_forward


def _dense_dora_weight(base_weight, lora_a, lora_b, scaling, dora_scale):
    """Naive dense DoRA effective weight: materialize ``W + s·BA`` then column-normalize.

    This is the ~10-line baseline the factored reference must match. It mirrors
    ``DoraLinearLayer.get_weight_norm`` (the dense path) composed with the DoRA magnitude rescale.
    """
    weight = base_weight + scaling * (lora_b @ lora_a)
    col_norm = torch.linalg.norm(weight, dim=1)
    scale = (dora_scale / col_norm).unsqueeze(-1)
    return scale * weight


@pytest.mark.parametrize("scaling", [1.0, 0.5, 2.0])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        # fp32: same bar as the merged PR #3382 suite (tests/test_dora_factored_norm.py).
        (torch.float32, 1e-5, 1e-5),
        # bf16: the factored accumulation order diverges from torch.linalg.norm's internal
        # accumulation more than in fp32; allow structural slack while still catching any
        # formula bug (which diverges by orders of magnitude).
        (torch.bfloat16, 5e-2, 5e-2),
    ],
)
def test_reference_matches_dense_dora(scaling, dtype, atol, rtol):
    torch.manual_seed(0)
    d_out, d_in, r = 32, 48, 4
    base_weight = torch.randn(d_out, d_in, dtype=dtype)
    lora_a = torch.randn(r, d_in, dtype=dtype)
    lora_b = torch.randn(d_out, r, dtype=dtype)
    # DoRA magnitude vector: one (positive) value per output column, like DoraLinearLayer.weight.
    dora_scale = torch.randn(d_out, dtype=dtype).abs() + 0.5

    dense = _dense_dora_weight(base_weight, lora_a, lora_b, scaling, dora_scale)
    factored = dora_factored_forward(base_weight, lora_a, lora_b, scaling, dora_scale)

    assert factored.shape == dense.shape == (d_out, d_in)
    torch.testing.assert_close(factored, dense, atol=atol, rtol=rtol)


def test_factored_norm_matches_dense_norm():
    # The factored decomposition's core claim, ported verbatim from PEFT's factored_weight_norm:
    # the column-wise norm computed from the factors equals torch.linalg.norm of the dense weight.
    # Mirrors tests/test_dora_factored_norm.py::test_factored_norm_matches_dense (huggingface/peft#3382)
    # — the strongest provenance check that the port did not drift from the merged algorithm.
    from dora_factored.reference import _factored_weight_norm

    torch.manual_seed(1)
    d_out, d_in, r = 16, 24, 3
    base_weight = torch.randn(d_out, d_in)
    lora_a = torch.randn(r, d_in)
    lora_b = torch.randn(d_out, r)
    scaling = 0.75

    dense_norm = torch.linalg.norm(base_weight + scaling * (lora_b @ lora_a), dim=1)
    factored_norm = _factored_weight_norm(base_weight, lora_a, lora_b, scaling)

    assert factored_norm.shape == (d_out,)
    torch.testing.assert_close(factored_norm, dense_norm, atol=1e-5, rtol=1e-5)


def test_reference_shape_and_finite():
    torch.manual_seed(2)
    base_weight = torch.randn(16, 24)
    lora_a = torch.randn(3, 24)
    lora_b = torch.randn(16, 3)
    out = dora_factored_forward(base_weight, lora_a, lora_b, scaling=0.75, dora_scale=torch.ones(16))
    assert out.shape == (16, 24)
    assert torch.isfinite(out).all()


def test_scalar_dora_scale_broadcasts():
    # dora_scale may be supplied as a scalar; it must broadcast across output columns identically
    # to the per-column form filled with the same value.
    torch.manual_seed(3)
    base_weight = torch.randn(8, 12)
    lora_a = torch.randn(2, 12)
    lora_b = torch.randn(8, 2)

    per_column = dora_factored_forward(base_weight, lora_a, lora_b, 1.0, torch.full((8,), 1.5))
    scalar = dora_factored_forward(base_weight, lora_a, lora_b, 1.0, torch.tensor(1.5))
    torch.testing.assert_close(scalar, per_column, atol=1e-5, rtol=1e-5)
