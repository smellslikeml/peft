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

"""Factored computation of the DoRA weight norm.

DoRA's forward pass rescales the adapted weight by the column-wise L2 norm of ``W + s·BA``. The straightforward
implementation materializes the dense ``[d_out, d_in]`` product ``BA`` just to take that norm, which dominates the
transient working memory of a DoRA module at high rank (e.g. ~512 MB in bf16 for ``d_in=8192, r=384``).

This module implements the *factored norm* introduced in "Scaling DoRA: High-Rank Adaptation via Factored Norms and
Fused Kernels" (arXiv:2603.22276). The squared row norm expands into three terms that never require the dense product::

    ||W_i + s·(BA)_i||^2 = ||W_i||^2  +  2s·<W_i, (BA)_i>  +  s^2·||(BA)_i||^2
                         = base_i     +  2s·cross_i        +  s^2·gram_i

with intermediates that are ``O(d_out·r + r^2)`` instead of ``O(d_out·d_in)``:

    * ``base``  = row-wise squared norm of ``W``                          (``[d_out]``)
    * ``cross`` = ``(B * (W @ Aᵀ)).sum(dim=1)`` using ``W @ Aᵀ``          (``[d_out, r]`` intermediate)
    * ``gram``  = ``(B @ (A @ Aᵀ) * B).sum(dim=1)`` using the ``r×r`` Gram (``[d_out, r]`` intermediate)

The result equals ``torch.linalg.norm(W + s·BA, dim=1)`` up to floating-point accumulation order. The GPU-specific
fused Triton kernels from the paper are intentionally out of scope for this reference implementation.
"""

import torch


def factored_weight_norm(
    base_weight: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Column-wise L2 norm of ``base_weight + scaling * (lora_B @ lora_A)`` without materializing the product.

    Args:
        base_weight: The (dequantized, ``fan_in_fan_out``-corrected) base weight of shape ``[d_out, d_in]``.
        lora_A: The LoRA ``A`` weight of shape ``[r, d_in]``.
        lora_B: The LoRA ``B`` weight of shape ``[d_out, r]``.
        scaling: The LoRA scaling factor ``s``.

    Returns:
        A tensor of shape ``[d_out]`` matching ``torch.linalg.norm(base_weight + scaling * lora_B @ lora_A, dim=1)``.
    """
    # ||W_i||^2: squared norm of each base-weight row, no adapter involved.
    base = base_weight.pow(2).sum(dim=1)

    # <W_i, (BA)_i> = <W_i, B_i A> reordered as <B_i, W_i Aᵀ>; the intermediate W @ Aᵀ is [d_out, r], never [d_out, d_in].
    w_at = base_weight @ lora_A.transpose(-2, -1)
    cross = (lora_B * w_at).sum(dim=1)

    # ||(BA)_i||^2 = B_iᵀ (A Aᵀ) B_i; the Gram matrix A @ Aᵀ is r×r and B @ Gram is [d_out, r].
    gram_matrix = lora_A @ lora_A.transpose(-2, -1)
    b_gram = lora_B @ gram_matrix
    gram = (b_gram * lora_B).sum(dim=1)

    squared_norm = base + 2.0 * scaling * cross + scaling * scaling * gram
    # Guard the near-unity rescaling regime: rounding can push a tiny squared norm slightly negative before sqrt.
    squared_norm = squared_norm.clamp_min(0)
    return squared_norm.sqrt()
