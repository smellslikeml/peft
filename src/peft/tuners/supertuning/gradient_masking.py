# Copyright 2026-present the HuggingFace Inc. team.
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

"""
Gradient masking that enforces the fixed sparse support of Super-Tuning.

Adapted from "Super-Tuning: From Activation-Aware Pruning to Sparse Fine-Tuning"
(https://arxiv.org/abs/2607.09287v1). The ``Super`` method keeps every base weight in place but only lets the
parameters inside a fixed sparse support (selected by a Wanda-/magnitude-style saliency ordering) receive updates
during fine-tuning.

Enforcing that support therefore requires masking the *weight* gradient. A module-level ``register_backward_hook``
sees the gradient of the layer *output* (wrong shape, wrong quantity) and, if re-registered every forward pass, also
leaks a new hook on each step. Instead we register a single tensor-level hook on the wrapped ``weight`` via
``torch.Tensor.register_hook``, which receives the gradient with respect to the weight and can zero out every entry
outside the support. The support mask is looked up lazily so that a later calibration pass is reflected automatically.
"""

from __future__ import annotations

from typing import Optional

import torch


def combine_sparse_masks(layer, adapter_names) -> Optional[torch.Tensor]:
    """
    Return the union of the sparse support masks of ``adapter_names`` on ``layer``, or ``None`` if none are present.

    The union follows the same convention as the forward pass: a weight is trainable if it belongs to the support of
    any active adapter.
    """
    combined = None
    for adapter_name in adapter_names:
        if adapter_name not in layer.supertuning_sparse_mask:
            continue
        mask = layer.supertuning_sparse_mask[adapter_name]
        if combined is None:
            combined = mask
        else:
            combined = torch.clamp(combined + mask, 0, 1)
    return combined


def register_gradient_mask_hook(layer):
    """
    Register a single weight-gradient hook on ``layer`` that keeps only the sparse support trainable.

    Any hook previously registered by this function is removed first, so the call is idempotent and does not leak
    hooks across training steps. The combined support mask is resolved lazily from the layer's active adapters, so
    re-calibrating the support does not require re-registering the hook. Returns the hook handle, or ``None`` if the
    layer currently has no support mask to enforce.
    """
    remove_gradient_mask_hook(layer)

    weight = layer.get_base_layer().weight
    if combine_sparse_masks(layer, layer.active_adapters) is None:
        return None

    def _mask_weight_grad(grad: torch.Tensor) -> torch.Tensor:
        mask = combine_sparse_masks(layer, layer.active_adapters)
        if mask is None:
            return grad
        return grad * mask.to(device=grad.device, dtype=grad.dtype)

    handle = weight.register_hook(_mask_weight_grad)
    layer._gradient_mask_handle = handle
    return handle


def remove_gradient_mask_hook(layer) -> None:
    """Remove the weight-gradient hook previously registered on ``layer``, if any."""
    handle = getattr(layer, "_gradient_mask_handle", None)
    if handle is not None:
        handle.remove()
    layer._gradient_mask_handle = None
