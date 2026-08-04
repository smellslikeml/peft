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

"""Pure-PyTorch reference for the DoRA-factored weight-adaptation forward.

This is the Stage A reference that the eventual fused Triton kernel (Stage B) must match
numerically. It ports the *factored-norm* decomposition verbatim from PEFT's merged
``factored_weight_norm`` (huggingface/peft#3382, on this branch as
``src/peft/tuners/lora/factored_weight_norm.py``) and layers the DoRA magnitude rescale on top.

DoRA's forward rescales the adapted weight by the column-wise L2 norm of ``W + s·BA``. The
squared row norm expands into three terms that never require materializing the dense
``[d_out, d_in]`` product ``BA`` (arXiv:2603.22276, "Scaling DoRA: High-Rank Adaptation via
Factored Norms and Fused Kernels")::

    ||W_i + s·(BA)_i||^2 = ||W_i||^2  +  2s·<W_i, (BA)_i>  +  s^2·||(BA)_i||^2
                         = base_i     +  2s·cross_i        +  s^2·gram_i

with ``O(d_out·r + r^2)`` intermediates instead of ``O(d_out·d_in)``. The result equals
``torch.linalg.norm(W + s·BA, dim=1)`` up to floating-point accumulation order — which is exactly
what ``tests/test_reference_parity.py`` asserts against the dense DoRA path.

The math below is the already-reviewed-and-merged algorithm; it is *ported*, not re-derived.
"""

import torch


def _factored_weight_norm(
    base_weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Column-wise L2 norm of ``base_weight + scaling * (lora_b @ lora_a)`` without materializing the product.

    Ported verbatim from ``src/peft/tuners/lora/factored_weight_norm.py`` (huggingface/peft#3382);
    see that module's docstring for the full derivation. Kept byte-for-byte equivalent so the
    Stage B Triton kernel has a single trusted numerical target.

    Args:
        base_weight: The base weight of shape ``[d_out, d_in]``.
        lora_a: The LoRA ``A`` weight of shape ``[r, d_in]``.
        lora_b: The LoRA ``B`` weight of shape ``[d_out, r]``.
        scaling: The LoRA scaling factor ``s``.

    Returns:
        A tensor of shape ``[d_out]`` matching ``torch.linalg.norm(base_weight + scaling * lora_b @ lora_a, dim=1)``.
    """
    # ||W_i||^2: squared norm of each base-weight row, no adapter involved.
    base = base_weight.pow(2).sum(dim=1)

    # <W_i, (BA)_i> = <W_i, B_i A> reordered as <B_i, W_i Aᵀ>; the intermediate W @ Aᵀ is [d_out, r], never [d_out, d_in].
    w_at = base_weight @ lora_a.transpose(-2, -1)
    cross = (lora_b * w_at).sum(dim=1)

    # ||(BA)_i||^2 = B_iᵀ (A Aᵀ) B_i; the Gram matrix A @ Aᵀ is r×r and B @ Gram is [d_out, r].
    gram_matrix = lora_a @ lora_a.transpose(-2, -1)
    b_gram = lora_b @ gram_matrix
    gram = (b_gram * lora_b).sum(dim=1)

    squared_norm = base + 2.0 * scaling * cross + scaling * scaling * gram
    # Guard the near-unity rescaling regime: rounding can push a tiny squared norm slightly negative before sqrt.
    squared_norm = squared_norm.clamp_min(0)
    return squared_norm.sqrt()


def _forward_reference(
    base_weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
    dora_scale: torch.Tensor | float,
) -> torch.Tensor:
    """DoRA effective-weight forward using the factored-norm decomposition.

    Computes ``dora_scale / ||W + s·BA||`` column-wise via :func:`_factored_weight_norm` and applies
    it to the adapted weight, i.e. the DoRA effective weight ``W_eff`` of shape ``[d_out, d_in]``::

        W_eff = (dora_scale / ||W + s·BA||) ⊙ (W + s·BA)

    This is the reference the Stage B fused kernel must reproduce. ``dora_scale`` is the DoRA
    magnitude vector of shape ``[d_out]`` (one value per output column, matching PEFT's
    ``DoraLinearLayer.weight``); a scalar is broadcast across columns.

    Note: PEFT detaches this norm from the autograd graph as a training policy (DoRA §4.3). That is
    a caller-side autograd concern and is intentionally left differentiable here so the reference is
    a plain, composable PyTorch op.
    """
    weight_norm = _factored_weight_norm(base_weight, lora_a, lora_b, scaling)
    mag_norm_scale = dora_scale / weight_norm  # [d_out] (or broadcast from a scalar)

    weight = base_weight + scaling * (lora_b @ lora_a)
    return mag_norm_scale.unsqueeze(-1) * weight
