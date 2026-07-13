# Copyright 2025-present the HuggingFace Inc. team.
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
Super-Tuning layer implementation.

This module implements the sparse adapter layers for Super-Tuning, using
activation-aware pruning saliency to select trainable parameters.
"""

from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
from torch import nn

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

if TYPE_CHECKING:
    from .config import SupertuningConfig


class SupertuningLayer(BaseTunerLayer):
    """Base layer for Super-Tuning sparse adapters."""

    # List all names of layers that may contain trainable adapter weights
    adapter_layer_names = ("sparse_weight", "lora_A", "lora_B")
    # All names of other adapter-related parameters
    other_param_names = ("mask", "r", "scaling", "lora_alpha")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.mask = {}
        self.r = {}
        self.scaling = {}
        self.lora_alpha = {}
        self.sparse_weight = nn.ParameterDict({})
        self.lora_A = nn.ParameterDict({})
        self.lora_B = nn.ParameterDict({})
        self.weight_shape = base_layer.weight.shape

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            raise NotImplementedError("Only nn.Linear layers supported currently")

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    def update_layer(
        self,
        adapter_name,
        mask,
        config,
        **kwargs,
    ):
        """Update the layer with a new adapter."""
        init_weights = config.init_weights
        sparsity_ratio = config.sparsity_ratio

        # Store the mask
        if mask is not None:
            self.mask[adapter_name] = mask.to(self.base_layer.weight.device)
        else:
            # If no mask provided, create one based on sparsity ratio
            num_params = self.in_features * self.out_features
            num_trainable = int(num_params * (1 - sparsity_ratio))
            flat_mask = torch.zeros(num_params, device=self.base_layer.weight.device)
            flat_mask[:num_trainable] = 1
            flat_mask = flat_mask[torch.randperm(num_params)]
            self.mask[adapter_name] = flat_mask.reshape(self.weight_shape)

        # Initialize sparse weights
        num_trainable = int(self.mask[adapter_name].sum().item())
        sparse_init = torch.zeros(num_trainable) if init_weights else torch.randn(num_trainable) * 0.01
        self.sparse_weight[adapter_name] = nn.Parameter(
            sparse_init.to(self.base_layer.weight.dtype).to(self.base_layer.weight.device),
            requires_grad=True,
        )

        # For Supra hybrid adapter, also initialize LoRA components
        if config.adapter_type == "supra":
            self.r[adapter_name] = config.lora_rank
            self.lora_alpha[adapter_name] = config.lora_alpha
            self.scaling[adapter_name] = config.lora_alpha / config.lora_rank

            # Initialize LoRA matrices
            lora_A = torch.zeros(config.lora_rank, self.in_features, device=self.base_layer.weight.device)
            lora_B = torch.zeros(self.out_features, config.lora_rank, device=self.base_layer.weight.device)

            nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
            nn.init.zeros_(lora_B)

            self.lora_A[adapter_name] = nn.Parameter(lora_A, requires_grad=True)
            self.lora_B[adapter_name] = nn.Parameter(lora_B, requires_grad=True)
        else:
            self.r[adapter_name] = 0
            self.scaling[adapter_name] = 1.0

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_sparse_parameters(self, adapter_name):
        """Reset sparse parameters to zero."""
        if adapter_name in self.sparse_weight:
            nn.init.zeros_(self.sparse_weight[adapter_name])

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            return
        self.scaling[adapter] = scale


import math


class Linear(nn.Module, SupertuningLayer):
    """Super-Tuning implemented as a sparse linear layer."""

    def __init__(
        self,
        base_layer,
        mask,
        adapter_name: str,
        config,
        **kwargs,
    ) -> None:
        super().__init__()
        SupertuningLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = config.fan_in_fan_out
        if self.base_layer is not self.get_base_layer():
            raise ValueError("SuperTuning does not support nested base layers")

        self._active_adapter = adapter_name
        self.adapter_type = config.adapter_type
        self.update_layer(adapter_name, mask, config=config)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """Merge the active adapter weights into the base weights."""
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.sparse_weight.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in merged weights. Adapter {active_adapter} may be broken."
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """Unmerge previously merged adapter weights."""
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.sparse_weight.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        """Compute the delta weight for the given adapter."""
        device = self.sparse_weight[adapter].device

        # Build sparse weight matrix from mask and parameters
        mask = self.mask[adapter].to(device)
        sparse_delta = torch.zeros_like(self.base_layer.weight)
        sparse_delta[mask.bool()] = self.sparse_weight[adapter]

        # For Supra, add LoRA contribution
        if self.adapter_type == "supra" and adapter in self.lora_A:
            lora_A = self.lora_A[adapter]
            lora_B = self.lora_B[adapter]
            lora_delta = lora_B @ lora_A
            sparse_delta += lora_delta * self.scaling[adapter]

        return sparse_delta

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass with sparse adapter."""
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            new_weight = copy.deepcopy(self.base_layer.weight.data)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.sparse_weight.keys():
                    continue
                new_weight += self.get_delta_weight(active_adapter)

            result = F.linear(x, new_weight, bias=self.base_layer.bias)

        return result

    def supports_lora_conversion(self, adapter_name: str = "default") -> bool:
        # Sparse weights don't support standard LoRA conversion
        return False

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "supertuning." + rep
