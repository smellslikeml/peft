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

"""DoRA-factored kernel — public callable surface.

Stage A ships a pure-PyTorch reference of the DoRA weight-adaptation forward that computes the
column-wise magnitude norm via the *factored decomposition* (arXiv:2603.22276), ported from PEFT's
merged ``factored_weight_norm`` (huggingface/peft#3382). Stage B will add a fused Triton fast-path
behind the same :func:`dora_factored_forward` signature with automatic dispatch, so callers keep one
import regardless of which path is loaded.
"""

import torch


def dora_factored_forward(
    base_weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
    dora_scale: torch.Tensor | float,
) -> torch.Tensor:
    """DoRA forward using the factored-norm decomposition.

    Stage A: dispatches to the pure-PyTorch reference. Stage B will add a Triton fast-path with
    automatic dispatch.

    Args:
        base_weight: Base weight of shape ``[d_out, d_in]``.
        lora_a: LoRA ``A`` weight of shape ``[r, d_in]``.
        lora_b: LoRA ``B`` weight of shape ``[d_out, r]``.
        scaling: LoRA scaling factor ``s``.
        dora_scale: DoRA magnitude, shape ``[d_out]`` (per output column) or a scalar broadcast
            across columns.

    Returns:
        The DoRA effective weight of shape ``[d_out, d_in]``.
    """
    from .reference import _forward_reference

    return _forward_reference(base_weight, lora_a, lora_b, scaling, dora_scale)


__all__ = ["dora_factored_forward"]
