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

"""Fused DoRA *backward* kernel — Stage B Triton fast-path.

Ports the two-stage backward kernel **verbatim** from the paper's reference implementation at
``sockeye44/dorafactors/code/kernelagent_sols/optimize_dora_backward/beam_w2_r2_kernel_round_1.py``
(arXiv:2603.22276 §3.2). Kept in its own module so the compose (forward) and backward failure modes
stay isolable, per the Stage B brief. The ``@triton.jit`` bodies, ``@triton.autotune`` /
``@triton.heuristics`` decorators, and kernel signatures are byte-for-byte identical to the source;
only the Python launch wrapper is renamed (``kernel_function`` → :func:`dora_backward`).

It is the exact transpose of :func:`triton_compose.dora_compose`:

    compose_delta = (mag - 1) ⊙ base + (mag · 0.7) ⊙ lora

so, for an incoming gradient ``d_out`` w.r.t. ``compose_delta`` and the adapted weight
``inner = base + 0.7 · lora`` (= ``W + s·BA`` once the caller folds ``scaling / 0.7``), the kernels
produce::

    d_lora = d_out ⊙ (mag · 0.7)        # grad w.r.t. the dense lora delta
    d_base = d_out ⊙ (mag - 1)          # grad w.r.t. base (compose_delta path only)
    d_mag  = reduce_rows(d_out ⊙ inner) # grad w.r.t. the magnitude scale

Stage 1 emits a per-row-block partial of ``d_mag``; stage 2 reduces the partials to the final
``d_mag``. Both stages read inputs through explicit strides, so transposed (non-contiguous) views of
``[d_out, d_in]`` tensors can be passed directly.

Output packing
--------------

The wrapper returns a single ``[2·num_rows + 1, num_cols]`` tensor laid out as::

    rows [0 : num_rows]            → d_lora   ([num_rows, num_cols])
    rows [num_rows : 2·num_rows]   → d_base   ([num_rows, num_cols])
    row  [2·num_rows]              → d_mag    ([num_cols])

:mod:`autograd` unpacks this and chains the matmul backward into ``lora_a`` / ``lora_b`` plus the
``dora_scale`` and ``+base`` terms.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 128}, num_warps=8, num_stages=2),
    ],
    key=["num_rows", "num_cols"],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["num_rows"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["num_cols"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _dora_backward_stage1_kernel(
    d_out_ptr,
    inner_ptr,
    mag_ptr,
    out_ptr,
    partial_ptr,
    stride_do0,
    stride_do1,
    stride_in0,
    stride_in1,
    stride_mag,
    stride_out0,
    stride_out1,
    stride_part0,
    stride_part1,
    num_rows,
    num_cols,
    scale,
    INPUT_IS_BF16: tl.constexpr,
    INPUT_IS_FP16: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    row_mask = rows < num_rows
    col_mask = cols < num_cols

    mag = tl.load(mag_ptr + cols * stride_mag, mask=col_mask, other=0.0)
    mag_scaled = mag * scale
    mag_minus_one = mag - 1.0

    d_out_ptrs = d_out_ptr + rows[:, None] * stride_do0 + cols[None, :] * stride_do1
    inner_ptrs = inner_ptr + rows[:, None] * stride_in0 + cols[None, :] * stride_in1
    out_lora_ptrs = out_ptr + rows[:, None] * stride_out0 + cols[None, :] * stride_out1
    out_base_ptrs = out_ptr + (rows[:, None] + num_rows) * stride_out0 + cols[None, :] * stride_out1

    if EVEN_M and EVEN_N:
        d_out = tl.load(d_out_ptrs, cache_modifier=".cg")
        inner = tl.load(inner_ptrs, cache_modifier=".cg")

        d_lora = d_out * mag_scaled[None, :]
        d_base = d_out * mag_minus_one[None, :]

        tl.store(out_lora_ptrs, d_lora.to(out_ptr.dtype.element_ty))
        tl.store(out_base_ptrs, d_base.to(out_ptr.dtype.element_ty))
    else:
        mask = row_mask[:, None] & col_mask[None, :]

        d_out = tl.load(d_out_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
        inner = tl.load(inner_ptrs, mask=mask, other=0.0, cache_modifier=".cg")

        d_lora = d_out * mag_scaled[None, :]
        d_base = d_out * mag_minus_one[None, :]

        tl.store(out_lora_ptrs, d_lora.to(out_ptr.dtype.element_ty), mask=mask)
        tl.store(out_base_ptrs, d_base.to(out_ptr.dtype.element_ty), mask=mask)

    prod = d_out * inner
    if INPUT_IS_BF16:
        prod = prod.to(tl.bfloat16)
    elif INPUT_IS_FP16:
        prod = prod.to(tl.float16)

    d_mag_partial = tl.sum(prod.to(tl.float32), axis=0)
    partial_ptrs = partial_ptr + pid_m * stride_part0 + cols * stride_part1
    tl.store(partial_ptrs, d_mag_partial, mask=col_mask)


@triton.heuristics(
    {
        "EVEN_N": lambda args: args["num_cols"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _dora_backward_stage2_kernel(
    partial_ptr,
    out_ptr,
    stride_part0,
    stride_part1,
    stride_out0,
    stride_out1,
    num_rows,
    num_cols,
    num_partials,
    EVEN_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)

    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < num_cols

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in tl.range(0, num_partials, BLOCK_K):
        ks = k_start + tl.arange(0, BLOCK_K)
        mask = (ks[:, None] < num_partials) & col_mask[None, :]
        ptrs = partial_ptr + ks[:, None] * stride_part0 + cols[None, :] * stride_part1
        vals = tl.load(ptrs, mask=mask, other=0.0, cache_modifier=".cg")
        acc += tl.sum(vals, axis=0)

    out_ptrs = out_ptr + (2 * num_rows) * stride_out0 + cols * stride_out1
    if EVEN_N:
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty))
    else:
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=col_mask)


def dora_backward(
    d_out: torch.Tensor,
    inner: torch.Tensor,
    mag_norm_scale: torch.Tensor,
    lora_coeff: float = 0.7,
) -> torch.Tensor:
    """Launch the two-stage DoRA backward kernels.

    Renamed wrapper (upstream: ``kernel_function``); launch logic unchanged. Computes the gradients
    of :func:`triton_compose.dora_compose` w.r.t. ``lora`` (the dense delta), ``base``, and the
    magnitude scale ``mag``, for an incoming gradient ``d_out`` and the adapted weight ``inner``.

    Args:
        d_out: Upstream gradient ``[num_rows, num_cols]`` w.r.t. the compose delta.
        inner: Adapted weight ``base + lora_coeff·lora`` (= ``W + s·BA``) ``[num_rows, num_cols]``.
        mag_norm_scale: DoRA magnitude scale ``[num_cols]``.
        lora_coeff: Coefficient for the lora term in the backward kernel (default: 0.7).

    Returns:
        A packed ``[2·num_rows + 1, num_cols]`` tensor (see module docstring for the row layout).
    """
    assert d_out.is_cuda and inner.is_cuda and mag_norm_scale.is_cuda
    assert d_out.ndim == 2 and inner.ndim == 2
    assert d_out.shape == inner.shape
    assert d_out.dtype == inner.dtype

    r, c = d_out.shape

    assert mag_norm_scale.dtype == d_out.dtype
    if mag_norm_scale.ndim == 1:
        assert mag_norm_scale.shape[0] == c
    else:
        assert mag_norm_scale.ndim == 2
        assert mag_norm_scale.shape[0] == 1 and mag_norm_scale.shape[1] == c

    out = torch.empty((2 * r + 1, c), device=d_out.device, dtype=d_out.dtype)

    BLOCK_M = 64
    num_partials = triton.cdiv(r, BLOCK_M)
    partial = torch.empty((num_partials, c), device=d_out.device, dtype=torch.float32)

    if c > 0:
        if num_partials > 0:
            def grid_stage1(META):
                return (triton.cdiv(c, META["BLOCK_N"]), num_partials)
            _dora_backward_stage1_kernel[grid_stage1](
                d_out,
                inner,
                mag_norm_scale,
                out,
                partial,
                d_out.stride(0),
                d_out.stride(1),
                inner.stride(0),
                inner.stride(1),
                mag_norm_scale.stride(-1),
                out.stride(0),
                out.stride(1),
                partial.stride(0),
                partial.stride(1),
                r,
                c,
                lora_coeff,
                INPUT_IS_BF16=d_out.dtype == torch.bfloat16,
                INPUT_IS_FP16=d_out.dtype == torch.float16,
                BLOCK_M=BLOCK_M,
            )

        REDUCE_BLOCK_N = 128
        _dora_backward_stage2_kernel[(triton.cdiv(c, REDUCE_BLOCK_N),)](
            partial,
            out,
            partial.stride(0),
            partial.stride(1),
            out.stride(0),
            out.stride(1),
            r,
            c,
            num_partials,
            BLOCK_K=32,
            BLOCK_N=REDUCE_BLOCK_N,
            num_warps=4,
            num_stages=2,
        )

    return out
