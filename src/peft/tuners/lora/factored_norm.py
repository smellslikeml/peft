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

"""Factored weight-norm for DoRA.

Adapted from "Scaling DoRA: High-Rank Adaptation via Factored Norms and Fused
Kernels" (https://arxiv.org/abs/2603.22276v1).

DoRA's forward pass needs the per-output-row L2 norm ``||W + s * B A||``. The
standard implementation materializes the dense ``d_out x d_in`` product ``B A``
to form it (and, in PEFT's linear path, also a ``d_in x d_in`` identity matrix).
At high rank -- e.g. ``d_in = 8192`` and ``r = 384`` -- those transients reach
hundreds of MB per module, which makes high-rank DoRA costly or infeasible on
single GPUs.

Here the same norm is computed directly from the rank-r factors ``B`` (d_out, r)
and ``A`` (r, d_in) by expanding the squared norm column-wise::

    ||W + s * B A||**2 = ||W||**2 + 2 s <W, B A> + s**2 ||B A||**2

so the largest transient tensor is ``d_out x r`` instead of ``d_out x d_in``.
The paper's fused CUDA kernels are substituted by plain matmuls -- the memory
win, which is what matters for single-GPU high-rank DoRA, is preserved. Note
that, unlike PEFT's default module-forward based path, this reads the factor
weights directly and is therefore not FSDP-safe.
"""

import torch

from peft.utils.other import transpose


def weight_norm_from_factors(
    weight: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    scaling: float,
    fan_in_fan_out: bool = False,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Per-output-row L2 norm ``||W + s * B A||`` of a DoRA layer, built from the factors.

    This is mathematically identical to
    ``torch.linalg.norm(W + scaling * (B @ A), dim=1)`` but never forms the dense
    ``d_out x d_in`` product ``B @ A``; the largest intermediate is ``d_out x r``.

    Args:
        weight:
            Base layer weight. Interpreted as ``(d_out, d_in)`` after applying
            ``fan_in_fan_out``.
        lora_A:
            LoRA down-projection factor of shape ``(r, d_in)``.
        lora_B:
            LoRA up-projection factor of shape ``(d_out, r)``.
        scaling:
            LoRA scaling factor (``lora_alpha / r``).
        fan_in_fan_out:
            Whether ``weight`` is stored transposed (e.g. for fan-in weights).
        eps:
            Floor for the squared norm before ``sqrt``; guards against fp
            cancellation when ``W`` nearly cancels ``s * B A``.

    Returns:
        Per-output-row norm of shape ``(d_out,)``.
    """
    out_dtype = weight.dtype
    # Accumulate in fp32: each term is a sum over the wide input dimension, so the
    # factorized expansion is more sensitive to rounding than the dense norm.
    weight = transpose(weight, fan_in_fan_out).float()  # (d_out, d_in)
    a = lora_A.float()  # (r, d_in)
    b = lora_B.float()  # (d_out, r)

    # term1 = ||W||**2 along the input dim; vector_norm avoids a d_out x d_in square
    term1 = torch.linalg.vector_norm(weight, dim=1) ** 2  # (d_out,)
    # term2 = <W, B A> over the input dim = (B * (W @ A.T)).sum(dim=1)
    weight_a = weight @ a.t()  # (d_out, r)
    term2 = (b * weight_a).sum(dim=1)  # (d_out,)
    # term3 = ||B A||**2 over the input dim = (B * (B @ (A @ A.T))).sum(dim=1)
    a_at = a @ a.t()  # (r, r)
    term3 = (b * (b @ a_at)).sum(dim=1)  # (d_out,)

    norm_sq = term1 + 2.0 * scaling * term2 + (scaling**2) * term3
    return torch.sqrt(norm_sq.clamp_min(eps)).to(out_dtype)
