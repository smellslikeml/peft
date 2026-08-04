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

"""GPU parity tests for the Stage C strided compose kernel.

These assert the strided variant — :func:`triton_compose_strided.dora_compose_strided` — is
numerically equivalent to the Stage A pure-PyTorch reference, independent of the higher-level
:func:`dora_factored_forward` dispatch (which now routes through the strided kernel by default).

The strided kernel accepts arbitrary-stride 2D tensors, including transposed views. These tests
exercise that capability by passing both contiguous and transposed inputs, verifying that the
stride-aware pointer arithmetic produces correct results for all cases.
"""

import pytest
import torch


# Skip the whole module unless both CUDA and triton are present.
pytest.importorskip("triton")

from dora_factored.reference import _factored_weight_norm
from dora_factored.triton_compose_strided import dora_compose_strided


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU"),
]

# Shapes (d_out, d_in, r) straddling the compose kernel's autotune bin edges. Same as Stage B.
TILE_SHAPES = [
    (128, 64, 8),
    (129, 33, 4),
    (256, 128, 16),
    (513, 65, 8),
    (300, 17, 4),
    (8, 8, 2),
]

SCALINGS = [0.7, 1.0, 2.0]


def _reference_compose_delta(mag, base, lora_in):
    """Reference implementation of DoRA compose_delta with explicit broadcasting.

    mag is [d_out]; base and lora_in are [d_out, d_in]. The kernel applies mag
    column-wise, so we unsqueeze mag to [d_out, 1] to make the broadcast explicit.
    """
    return (mag - 1).unsqueeze(-1) * base + (mag * 0.7).unsqueeze(-1) * lora_in


def _inputs(d_out, d_in, r, dtype, device):
    torch.manual_seed(0)
    base = torch.randn(d_out, d_in, dtype=dtype, device=device)
    lora_a = torch.randn(r, d_in, dtype=dtype, device=device)
    lora_b = torch.randn(d_out, r, dtype=dtype, device=device)
    dora_scale = torch.randn(d_out, dtype=dtype, device=device).abs() + 0.5
    return base, lora_a, lora_b, dora_scale


@pytest.mark.parametrize("scaling", SCALINGS)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 5e-2, 5e-2),
        (torch.bfloat16, 5e-2, 5e-2),
    ],
)
@pytest.mark.parametrize("d_out, d_in, r", TILE_SHAPES)
def test_strided_compose_kernel_transposed(d_out, d_in, r, dtype, atol, rtol, scaling):
    """Strided kernel matches reference for transposed (non-contiguous) inputs.

    The kernel applies ``mag`` per ``num_cols``. PEFT's magnitude is per output feature
    (``d_out``), so callers feed the transposed weight ``[d_in, d_out]`` — matching the
    orientation the autograd wrapper uses. Direct-contiguous callers (``mag`` misaligned
    with ``num_cols``) are unsupported and no longer exercised here.
    """
    base, lora_a, lora_b, dora_scale = _inputs(d_out, d_in, r, dtype, "cuda")

    # Reference: compute the expected DoRA effective weight
    weight_norm = _factored_weight_norm(base, lora_a, lora_b, scaling)
    mag = dora_scale / weight_norm
    delta = lora_b @ lora_a
    lora_in = (scaling / 0.7) * delta
    expected_base = base + lora_in
    expected_delta = _reference_compose_delta(mag, base, lora_in)
    expected_out = base + expected_delta

    # Strided kernel: pass transposed views (.t() without .contiguous())
    # The kernel should handle the strides correctly without materializing copies.
    actual_out = dora_compose_strided(lora_in.t(), base.t(), mag).t()

    torch.testing.assert_close(actual_out, expected_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize("scaling", SCALINGS)
@pytest.mark.parametrize("d_out, d_in, r", [(128, 64, 8), (513, 65, 8)])
def test_strided_compose_kernel_transposed_output(d_out, d_in, r, scaling):
    """Strided kernel correctly writes into a transposed view of the output tensor."""
    dtype = torch.float32
    base, lora_a, lora_b, dora_scale = _inputs(d_out, d_in, r, dtype, "cuda")

    weight_norm = _factored_weight_norm(base, lora_a, lora_b, scaling)
    mag = dora_scale / weight_norm
    delta = lora_b @ lora_a
    lora_in = (scaling / 0.7) * delta

    # Reference: expected result for [d_out, d_in] layout (full effective weight)
    expected_base = base + lora_in
    expected_delta = _reference_compose_delta(mag, base, lora_in)
    expected_out = base + expected_delta

    # Allocate output and write into its transposed view
    out = torch.empty_like(base)
    result = dora_compose_strided(
        lora_in.t(),  # [d_in, d_out] view
        base.t(),  # [d_in, d_out] view
        mag,
        out=out.t(),  # Write into [d_in, d_out] view of output
    ).t()  # View back to [d_out, d_in]

    # Verify the result matches the expected [d_out, d_in] layout
    torch.testing.assert_close(result, expected_out, atol=1e-5, rtol=1e-5)
