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

"""GPU parity tests for the Stage B Triton fast-path of the DoRA-factored kernel.

These assert the fused Triton path — auto-dispatched by :func:`dora_factored_forward` on CUDA — is
numerically equivalent to the Stage A pure-PyTorch :mod:`reference`. CI has no GPU, so every test is
marked ``cuda`` and skipped when CUDA (or ``triton``) is unavailable; the suite is *armed* to run on
a reviewer's real GPU, where it gates Stage C (PEFT loader wiring) on a trusted kernel.

Why the shape/scaling matrix is load-bearing
--------------------------------------------

* Tile boundaries: the compose kernel's ``@triton.autotune`` configs cover ``BLOCK_N ∈ {128, 256,
  512}`` and ``BLOCK_M ∈ {8, 16, 32, 64}``, and its ``@triton.heuristics`` branch on
  ``EVEN_M``/``EVEN_N`` into a fast (unmasked) path and a masked tail path. ``TILE_SHAPES`` picks
  ``d_out`` (→ ``num_cols``) just above/below each ``BLOCK_N`` edge in both even and odd variants, so
  the parity assertion exercises the masking + tail handling, not just the happy path.
* Non-0.7 LoRA scaling: the wrapper folds ``scaling / 0.7`` into the dense LoRA delta so the kernel's
  baked-in ``0.7`` cancels (see :mod:`autograd`). Testing ``scaling != 0.7`` is what proves the fold
  is correct — a ``scaling == 0.7``-only suite would pass while the fold silently broke every other
  scaling.
"""

import pytest
import torch


# Skip the whole module unless both CUDA and triton are present. This also defeats a false pass:
# without triton, `dora_factored_forward` transparently falls back to the reference, so a "Triton vs
# reference" comparison would reduce to "reference vs reference". importorskip makes absence loud.
pytest.importorskip("triton")

from dora_factored import _triton_available, dora_factored_forward
from dora_factored.reference import _factored_weight_norm, _forward_reference


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU"),
]

# Shapes (d_out, d_in, r) straddling the compose kernel's autotune bin edges. d_out → num_cols
# (BLOCK_N edges 128/256/512); d_in → num_rows (BLOCK_M edges 8/16/32/64). Each pair toggles even vs
# odd on both dims so EVEN_M/EVEN_N and their masked-tail complements are both hit.
TILE_SHAPES = [
    (128, 64, 8),  # d_out on the 128 edge, even d_in
    (129, 33, 4),  # d_out just over 128 (odd → masked tail on N); odd d_in (→ masked tail on M)
    (256, 128, 16),  # d_out on the 256 edge, even d_in
    (513, 65, 8),  # d_out just over 512 (odd tail); odd d_in
    (300, 17, 4),  # non-multiples throughout — exercises masking on both dims
    (8, 8, 2),  # smallest tiles
]

# 0.7 = no-fold baseline; 1.0 and 2.0 prove the scaling/0.7 fold is correct for arbitrary scaling.
SCALINGS = [0.7, 1.0, 2.0]


def _inputs(d_out, d_in, r, dtype, device, requires_grad=False):
    torch.manual_seed(0)
    base = torch.randn(d_out, d_in, dtype=dtype, device=device)
    lora_a = torch.randn(r, d_in, dtype=dtype, device=device)
    lora_b = torch.randn(d_out, r, dtype=dtype, device=device)
    dora_scale = torch.randn(d_out, dtype=dtype, device=device).abs() + 0.5
    if requires_grad:
        for t in (base, lora_a, lora_b, dora_scale):
            t.requires_grad_(True)
    return base, lora_a, lora_b, dora_scale


def _assert_triton_dispatched(base):
    # Guard against a silent reference fallback (would make parity a no-op). With the module-level
    # importorskip + cuda skipif this is guaranteed, but assert it explicitly for clarity.
    assert _triton_available(base), "Triton path was not taken — parity assertion would be theater"


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
def test_triton_forward_matches_reference(d_out, d_in, r, dtype, atol, rtol, scaling):
    base, lora_a, lora_b, dora_scale = _inputs(d_out, d_in, r, dtype, "cuda")
    _assert_triton_dispatched(base)

    triton = dora_factored_forward(base, lora_a, lora_b, scaling, dora_scale)
    reference = _forward_reference(base, lora_a, lora_b, scaling, dora_scale)

    assert triton.shape == reference.shape == (d_out, d_in)
    torch.testing.assert_close(triton, reference, atol=atol, rtol=rtol)


@pytest.mark.parametrize("scaling", SCALINGS)
@pytest.mark.parametrize("d_out, d_in, r", TILE_SHAPES)
def test_triton_forward_matches_dense_dora(scaling, d_out, d_in, r):
    # Cross-check against a naive dense DoRA baseline (materializes W + s·BA), mirroring Stage A's
    # tests/test_reference_parity.py. An independent target from the factored reference.
    base, lora_a, lora_b, dora_scale = _inputs(d_out, d_in, r, torch.float32, "cuda")
    _assert_triton_dispatched(base)

    triton = dora_factored_forward(base, lora_a, lora_b, scaling, dora_scale)
    weight = base + scaling * (lora_b @ lora_a)
    dense = (dora_scale / torch.linalg.norm(weight, dim=1)).unsqueeze(-1) * weight

    torch.testing.assert_close(triton, dense, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("scaling", SCALINGS)
@pytest.mark.parametrize("d_out, d_in, r", TILE_SHAPES)
def test_triton_backward_matches_reference(scaling, d_out, d_in, r):
    # gradcheck-style backward parity vs the reference's autograd under the *detached-norm* policy
    # (DoRA §4.3) — the policy both the kernel and PEFT use. fp32: the kernel accumulates in fp32, so
    # analytic grads are compared at fp32 precision (literal torch.autograd.gradcheck needs fp64,
    # which the fp32-casting kernel cannot match).
    base, lora_a, lora_b, dora_scale = _inputs(d_out, d_in, r, torch.float32, "cuda", requires_grad=True)
    _assert_triton_dispatched(base)

    triton = dora_factored_forward(base, lora_a, lora_b, scaling, dora_scale)
    grad_output = torch.randn_like(triton)
    g_base, g_a, g_b, g_dora = torch.autograd.grad(triton, (base, lora_a, lora_b, dora_scale), grad_output)

    # Reference forward with weight_norm detached, so its autograd matches the kernel's policy.
    weight_norm = _factored_weight_norm(base, lora_a, lora_b, scaling).detach()
    mag = dora_scale / weight_norm
    reference = mag.unsqueeze(-1) * (base + scaling * (lora_b @ lora_a))
    r_base, r_a, r_b, r_dora = torch.autograd.grad(reference, (base, lora_a, lora_b, dora_scale), grad_output)

    torch.testing.assert_close(g_base, r_base, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(g_a, r_a, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(g_b, r_b, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(g_dora, r_dora, atol=1e-4, rtol=1e-4)
