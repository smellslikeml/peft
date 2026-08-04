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

"""Fused DoRA *compose* (forward) kernel — Stage C strided variant.

Ports the same math as :mod:`triton_compose`'s ``_fused_dora_compose_kernel`` but accepts explicit
strides for ``lora_ptr``, ``base_ptr``, and ``out_ptr`` — mirroring the stride-argument pattern
established in :mod:`triton_backward`. This allows the kernel to operate on transposed (non-contiguous)
views of ``[d_out, d_in]`` tensors without materializing a full copy, eliminating the transpose overhead
that made the Stage B fast-path slower than the reference.

The autotune configs and heuristics are copied verbatim from the original kernel; the only change is
the stride-aware pointer arithmetic. This preserves the paper-tuned kernel as an independently reproducible
artifact (the original :mod:`triton_compose` remains untouched) while adding a production-oriented variant
that can operate on PEFT's native ``[d_out, d_in]`` weight layout.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "CHUNK_N": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256, "CHUNK_N": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256, "CHUNK_N": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "CHUNK_N": 32, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 512, "CHUNK_N": 64, "GROUP_M": 16}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 512, "CHUNK_N": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 512, "CHUNK_N": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
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
def _fused_dora_compose_strided_kernel(
    lora_ptr,
    base_ptr,
    mag_ptr,
    out_ptr,
    stride_lora0,
    stride_lora1,
    stride_base0,
    stride_base1,
    stride_out0,
    stride_out1,
    num_rows,
    num_cols,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    tl.static_assert(BLOCK_N % CHUNK_N == 0)

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(num_rows, BLOCK_M)
    num_pid_n = tl.cdiv(num_cols, BLOCK_N)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols_chunk = tl.arange(0, CHUNK_N)
    start_n = pid_n * BLOCK_N

    if not (EVEN_M and EVEN_N):
        row_mask = rows < num_rows

    for off_n in tl.static_range(0, BLOCK_N, CHUNK_N):
        cols = start_n + off_n + cols_chunk

        # Strided pointer arithmetic for lora
        lora_ptrs = lora_ptr + rows[:, None] * stride_lora0 + cols[None, :] * stride_lora1
        # Strided pointer arithmetic for base
        base_ptrs = base_ptr + rows[:, None] * stride_base0 + cols[None, :] * stride_base1
        # Strided pointer arithmetic for out
        out_ptrs = out_ptr + rows[:, None] * stride_out0 + cols[None, :] * stride_out1

        if EVEN_M and EVEN_N:
            mag = tl.load(mag_ptr + cols, cache_modifier=".ca").to(tl.float32)
            lora = tl.load(lora_ptrs, cache_modifier=".cg").to(tl.float32)
            base = tl.load(base_ptrs, cache_modifier=".cg").to(tl.float32)
            out = tl.fma((mag - 1.0)[None, :], base, (mag * 0.7)[None, :] * lora)
            tl.store(out_ptrs, out)
        else:
            col_mask = cols < num_cols
            mask = row_mask[:, None] & col_mask[None, :]
            mag = tl.load(mag_ptr + cols, mask=col_mask, other=0.0, cache_modifier=".ca").to(tl.float32)
            lora = tl.load(lora_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
            base = tl.load(base_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
            out = tl.fma((mag - 1.0)[None, :], base, (mag * 0.7)[None, :] * lora)
            tl.store(out_ptrs, out, mask=mask)


def dora_compose_strided(
    lora: torch.Tensor,
    base: torch.Tensor,
    mag_norm_scale: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Launch the strided :func:`_fused_dora_compose_strided_kernel` over a 2D weight tile.

    Accepts arbitrary-stride 2D tensors (including transposed views) and computes the same
    compose_delta as :func:`triton_compose.dora_compose`:

        compose_delta = (mag - 1) ⊙ base + (mag · 0.7) ⊙ lora

    Args:
        lora: Dense LoRA delta ``[num_rows, num_cols]`` (caller pre-folds ``scaling / 0.7``).
            May be non-contiguous (e.g. a transposed view).
        base: Base weight ``[num_rows, num_cols]``. May be non-contiguous.
        mag_norm_scale: DoRA magnitude scale ``[num_cols]`` (``= dora_scale / ||W + s·BA||``).
        out: Optional output tensor. If ``None``, allocates a contiguous output. If provided,
            writes into the caller's tensor (allowing direct write into a transposed view).

    Returns:
        The *compose delta* of shape matching ``lora``/``base`` (same strides as input if ``out``
        is ``None``, or same strides as ``out`` if provided). The caller adds ``base`` to recover
        the DoRA effective weight.
    """
    assert lora.is_cuda and base.is_cuda and mag_norm_scale.is_cuda
    assert lora.shape == base.shape
    assert lora.device == base.device == mag_norm_scale.device
    assert lora.dtype == base.dtype
    assert lora.ndim == 2 and base.ndim == 2

    if lora.numel() == 0:
        if out is None:
            return torch.empty_like(lora)
        return out

    orig_shape = lora.shape
    num_cols = orig_shape[-1]
    assert num_cols > 0
    num_rows = orig_shape[-2]
    assert mag_norm_scale.numel() == num_cols

    # Flatten magnitude to 1D contiguous for the kernel (per-column access)
    mag_flat = mag_norm_scale.reshape(num_cols).contiguous()

    # Allocate output if not provided
    if out is None:
        out = torch.empty_like(lora)
    else:
        assert out.shape == lora.shape
        assert out.device == lora.device
        assert out.dtype == lora.dtype

    def grid(META):
        return (
            triton.cdiv(num_rows, META["BLOCK_M"]) * triton.cdiv(num_cols, META["BLOCK_N"]),
        )

    _fused_dora_compose_strided_kernel[grid](
        lora,
        base,
        mag_flat,
        out,
        lora.stride(0),
        lora.stride(1),
        base.stride(0),
        base.stride(1),
        out.stride(0),
        out.stride(1),
        num_rows,
        num_cols,
    )

    return out
