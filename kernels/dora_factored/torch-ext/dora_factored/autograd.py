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

"""Autograd glue pairing the fused DoRA compose (forward) and backward Triton kernels.

:func:`DoraFactoredFn` is a :class:`torch.autograd.Function` so the fused Stage B fast-path composes
into a normal PyTorch graph — the prerequisite for the PEFT-side loader wiring in Stage C. The
forward computes the factored weight-norm in PyTorch (the cheap ``O(d_out·r + r²)`` part, ported from
:mod:`reference`) and fuses only the memory-bandwidth-bound magnitude-rescale + weight-compose over
the full ``[d_out, d_in]`` weight via :func:`triton_compose.dora_compose`. The backward delegates the
elementwise + row-reduce work to :func:`triton_backward.dora_backward` and finishes the matmul
chain-rule into ``lora_a`` / ``lora_b`` in PyTorch.

Two adaptations make the ported kernels (which are kept byte-for-byte identical to
``sockeye44/dorafactors``) land correctly behind the Stage A :func:`dora_factored_forward` signature:

1. **Folding the kernel's baked-in ``0.7`` coefficient.** The compose kernel body multiplies its
   ``lora`` input by a literal ``0.7`` (the LoRA scaling the upstream benchmark tuned for). To stay
   correct for *any* caller-supplied ``scaling`` without touching the kernel body, the forward
   pre-scales the dense LoRA delta by ``scaling / 0.7``; the kernel's ``0.7`` then cancels exactly,
   yielding ``mag ⊙ (W + s·BA)`` — numerically equivalent to the reference.

2. **Detached magnitude norm (PEFT DoRA §4.3).** The ported backward kernel treats the magnitude
   scale ``mag`` as independent of ``base`` and ``lora`` — it does not propagate gradient through
   ``weight_norm``. That is exactly PEFT's DoRA training policy (the reference's docstring notes the
   norm is "intentionally left differentiable" only as a composable op; PEFT detaches it at the call
   site). The forward therefore detaches ``weight_norm`` so the fused backward is mathematically
   exact, not an approximation. Forward *values* are unaffected by the detach.
"""

import torch

from .reference import _factored_weight_norm
from .triton_backward import dora_backward
from .triton_compose import dora_compose


# Coefficient the ported compose/backward kernels bake onto their `lora` input (see
# triton_compose.py / triton_backward.py). The forward folds `scaling / _KERNEL_LORA_COEFF` into the
# dense LoRA delta so this literal cancels for arbitrary caller `scaling`.
_KERNEL_LORA_COEFF = 0.7


class DoraFactoredFn(torch.autograd.Function):
    """Fused DoRA factored-norm forward + backward as a single differentiable op."""

    @staticmethod
    def forward(ctx, base_weight, lora_a, lora_b, scaling, dora_scale):
        # Factored column-wise norm of W + s·BA (ported from reference._factored_weight_norm). This is
        # the cheap O(d_out·r + r²) part; only the final elementwise rescale is fused in Triton.
        weight_norm = _factored_weight_norm(base_weight, lora_a, lora_b, scaling)
        # Detached per PEFT DoRA §4.3 — see module docstring. Forward values are unchanged.
        weight_norm = weight_norm.detach()
        mag = dora_scale / weight_norm  # [d_out]

        # Fold the kernel's baked 0.7 into the dense LoRA delta so it cancels at any `scaling`.
        delta = lora_b @ lora_a  # [d_out, d_in]
        lora_in = (scaling / _KERNEL_LORA_COEFF) * delta

        # The kernel applies `mag` per column (num_cols). PEFT's magnitude is per output feature
        # (d_out), so feed the transposed weight [d_in, d_out] to line num_cols up with mag.
        compose_k = dora_compose(lora_in.t().contiguous(), base_weight.t().contiguous(), mag)
        compose_delta = compose_k.t().contiguous()  # back to [d_out, d_in]
        out = base_weight + compose_delta

        # Adapted weight = base + 0.7·lora_in = W + s·BA, consumed by the backward kernel as `inner`.
        inner = base_weight + scaling * delta

        ctx.save_for_backward(base_weight, lora_a, lora_b, mag, weight_norm, inner)
        ctx.scaling = scaling
        ctx.dora_scale_is_tensor = isinstance(dora_scale, torch.Tensor)
        ctx.dora_scale_ndim = dora_scale.ndim if ctx.dora_scale_is_tensor else 0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        base_weight, lora_a, lora_b, mag, weight_norm, inner = ctx.saved_tensors
        scaling = ctx.scaling
        d_in = base_weight.shape[1]  # num_rows in kernel orientation (weight is [d_out, d_in])

        # Kernel orientation [num_rows=d_in, num_cols=d_out]; grad_output and inner are [d_out, d_in].
        packed = dora_backward(grad_output.t().contiguous(), inner.t().contiguous(), mag)
        grad_lora_k = packed[0:d_in]    # [d_in, d_out] — grad w.r.t. the (pre-folded) lora input
        grad_base_k = packed[d_in : 2 * d_in]  # [d_in, d_out] — grad w.r.t. base, compose_delta path
        grad_mag = packed[2 * d_in]     # [d_out]

        # Undo the scaling/0.7 fold, then chain the matmul backward into lora_a / lora_b.
        grad_delta = (scaling / _KERNEL_LORA_COEFF) * grad_lora_k.t().contiguous()  # [d_out, d_in]
        grad_lora_b = grad_delta @ lora_a.transpose(-2, -1)
        grad_lora_a = lora_b.transpose(-2, -1) @ grad_delta

        # out = base + compose_delta ⇒ grad w.r.t. base picks up the explicit +base term (grad_output)
        # on top of the kernel's compose_delta-path grad_base.
        grad_base_weight = grad_output + grad_base_k.t().contiguous()

        # mag = dora_scale / weight_norm (weight_norm detached) ⇒ grad_dora_scale = grad_mag / weight_norm.
        if not ctx.dora_scale_is_tensor:
            grad_dora_scale = None
        else:
            grad_dora_scale = grad_mag / weight_norm
            if ctx.dora_scale_ndim == 0:
                grad_dora_scale = grad_dora_scale.sum()

        return grad_base_weight, grad_lora_a, grad_lora_b, None, grad_dora_scale
