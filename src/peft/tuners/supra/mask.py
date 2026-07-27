# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Budget split + sparse-support selection for Supra.

Supra (arXiv:2607.09287 Eq. 14) combines Super's sparse update with LoRA
under a matched scalar-parameter budget:

    W_effective = W_frozen  +  M ⊙ U  +  γ · L @ R
                              └── sparse ──┘ └── LoRA ──┘

Two orthogonal decisions live here:

1. **Budget split** (`compute_supra_budget`) — given a rank-equivalent
   budget ``r`` and a split ratio ``lora_ratio`` ∈ [0, 1], returns
   ``(r_lora, sparse_k)`` so the total scalar count matches
   ``r · (in + out)`` (the number of parameters a plain LoRA at rank ``r``
   would train). ``lora_ratio=0`` → pure Super, ``lora_ratio=1`` → pure LoRA.

2. **Bottom-k magnitude support selection** (`bottomk_magnitude_indices`)
   — given a weight tensor and a count ``k``, returns flat int64 indices
   for the ``k`` entries with the smallest ``|W|``. Matches the paper's
   Supra-Mag variant, which the paper's 8B results show as strong-or-
   stronger than Wanda-based selection and needs no calibration data.

Wanda-style activation-weighted selection lives with the existing
Super-only tuner (`src/peft/tuners/supertuning/layer.py`); Supra reuses
those primitives when a caller opts into activation-aware scoring, but
the default is magnitude-only for simplicity + data-freeness.
"""

from __future__ import annotations

import math

import torch


def compute_supra_budget(
    r: int,
    lora_ratio: float,
    in_features: int,
    out_features: int,
) -> tuple[int, int]:
    """Split a rank-equivalent budget into (r_lora, sparse_k) for one layer.

    The total scalar-param count is fixed at ``r * (in_features + out_features)``
    (what plain LoRA at rank ``r`` would train). ``lora_ratio`` controls how
    much of that budget goes to LoRA vs the sparse component:

    - ``lora_ratio=1.0`` → all budget to LoRA, ``sparse_k = 0``
    - ``lora_ratio=0.0`` → all budget to sparse, ``r_lora = 0``
    - ``lora_ratio=0.5`` → half rank, remainder as sparse

    Args:
        r: rank-equivalent budget (matches a plain-LoRA baseline's rank).
        lora_ratio: fraction of the budget allocated to LoRA. Must be in
            ``[0.0, 1.0]``.
        in_features: input dim of the target Linear layer.
        out_features: output dim of the target Linear layer.

    Returns:
        ``(r_lora, sparse_k)`` — LoRA rank and count of trainable sparse
        entries. Both non-negative; their combined scalar count
        (``r_lora * (in + out) + sparse_k``) equals ``r * (in + out)``.
    """
    if not 0.0 <= lora_ratio <= 1.0:
        raise ValueError(f"lora_ratio must be in [0, 1], got {lora_ratio}")
    if r < 0 or in_features < 0 or out_features < 0:
        raise ValueError("r, in_features, out_features must be non-negative")

    dim_sum = in_features + out_features
    total_scalars = r * dim_sum
    r_lora = math.floor(lora_ratio * r)
    sparse_k = total_scalars - r_lora * dim_sum
    return r_lora, sparse_k


def bottomk_magnitude_indices(weight: torch.Tensor, k: int) -> torch.Tensor:
    """Return flat int64 indices of the k lowest-magnitude entries in ``weight``.

    Matches the paper's Supra-Mag support: pick the k entries with smallest
    ``|W_ij|`` as the sparse trainable support. The intuition is that
    lower-magnitude entries have more "headroom" — the pretrained model
    committed less to them — so nudging them during fine-tuning perturbs
    the residual stream less than editing high-magnitude weights.

    ``k=0`` returns an empty tensor (valid — happens when ``lora_ratio=1.0``).
    ``k >= weight.numel()`` returns all indices in ascending order of magnitude.

    Args:
        weight: a Linear layer's weight tensor. Any shape / dtype; will be
            flattened for indexing. The returned indices are flat, so the
            caller unflattens with ``torch.unravel_index`` or by reshaping
            the update tensor to match.
        k: number of entries to select.

    Returns:
        1-D ``torch.int64`` tensor of length ``k`` on the same device as
        ``weight``, containing flat indices into ``weight.flatten()``.
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if k == 0:
        return torch.empty(0, dtype=torch.int64, device=weight.device)
    numel = weight.numel()
    k = min(k, numel)
    # torch.topk with largest=False on the abs-magnitudes gives us the
    # bottom-k. .indices is int64 on the same device.
    _, indices = torch.topk(weight.detach().abs().flatten(), k=k, largest=False)
    return indices.to(torch.int64)
