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

import math
from typing import Any, Optional

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer

from .config import AdaMoLEConfig


class AdaMoLELayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("adamole_A", "adamole_B", "adamole_router")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("adamole_threshold",)

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.num_experts: dict = {}
        self.r: dict = {}
        self.adamole_alpha: dict = {}
        self.adamole_scaling: dict = {}
        self.adamole_router_temperature: dict = {}
        self.adamole_threshold_tau: dict = {}
        self.adamole_use_threshold: dict = {}
        # Per adapter: a ModuleList of N expert (lora_A, lora_B) pairs, a router module and one threshold parameter.
        self.adamole_A = nn.ModuleDict({})
        self.adamole_B = nn.ModuleDict({})
        self.adamole_router = nn.ModuleDict({})
        self.adamole_threshold = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            self.in_features, self.out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        else:
            raise TypeError(f"Unsupported layer type {type(base_layer)}")

    def update_layer(self, adapter_name: str, config: AdaMoLEConfig, **kwargs) -> None:
        num_experts = config.num_experts
        r = config.r
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        if num_experts <= 0:
            raise ValueError(f"`num_experts` should be a positive integer value but the value passed is {num_experts}")

        self.num_experts[adapter_name] = num_experts
        self.r[adapter_name] = r
        self.adamole_alpha[adapter_name] = config.lora_alpha
        self.adamole_scaling[adapter_name] = config.lora_alpha / r
        self.adamole_router_temperature[adapter_name] = config.router_temperature
        self.adamole_threshold_tau[adapter_name] = config.threshold_tau
        self.adamole_use_threshold[adapter_name] = config.use_threshold

        # N expert (lora_A, lora_B) pairs, the same A/B convention as LoRA.
        self.adamole_A[adapter_name] = nn.ModuleList(
            [nn.Linear(self.in_features, r, bias=False) for _ in range(num_experts)]
        )
        self.adamole_B[adapter_name] = nn.ModuleList(
            [nn.Linear(r, self.out_features, bias=False) for _ in range(num_experts)]
        )
        # Router producing per-token expert logits.
        self.adamole_router[adapter_name] = self._build_router(config)
        # The single learnable per-token activation threshold.
        self.adamole_threshold[adapter_name] = nn.Parameter(torch.tensor(float(config.threshold_init)))

        self.reset_adamole_parameters(adapter_name)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=config.inference_mode)

    def _build_router(self, config: AdaMoLEConfig) -> nn.Module:
        if config.router_hidden_dim is None:
            return nn.Linear(self.in_features, config.num_experts, bias=False)
        return nn.Sequential(
            nn.Linear(self.in_features, config.router_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.router_hidden_dim, config.num_experts),
        )

    @torch.no_grad()
    def reset_adamole_parameters(self, adapter_name: str) -> None:
        # Experts initialize like ordinary LoRA: Kaiming on lora_A, zeros on lora_B.
        for module in self.adamole_A[adapter_name]:
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        for module in self.adamole_B[adapter_name]:
            nn.init.zeros_(module.weight)
        # The router starts from the default linear init so the softmax gate is non-degenerate.
        # The threshold keeps its configured initial value.

    def compute_delta(self, adapter_name: str, x: torch.Tensor) -> torch.Tensor:
        """Return the AdaMoLE delta (shape ``x.shape[:-1] + (out_features,)``) for ``x``.

        Implements dense softmax routing over the expert deltas and, optionally, the learnable per-token activation
        threshold. The threshold gate is a hard comparison in the forward pass (so masked tokens contribute exactly
        zero) wrapped in a straight-through estimator so the threshold stays trainable.
        """
        lora_A = self.adamole_A[adapter_name]
        lora_B = self.adamole_B[adapter_name]
        router = self.adamole_router[adapter_name]
        num_experts = self.num_experts[adapter_name]
        temperature = self.adamole_router_temperature[adapter_name]
        scaling = self.adamole_scaling[adapter_name]
        use_threshold = self.adamole_use_threshold[adapter_name]

        # Per-token expert weights: [..., num_experts]
        route_dtype = next(router.parameters()).dtype
        gate = torch.softmax(router(x.to(route_dtype)) / temperature, dim=-1)

        # Expert deltas stacked along a new last dimension: [..., out_features, num_experts]
        expert_dtype = lora_A[0].weight.dtype
        xc = x.to(expert_dtype)
        expert_out = torch.stack([lora_B[e](lora_A[e](xc)) for e in range(num_experts)], dim=-1)
        delta = (expert_out * gate.unsqueeze(-2)).sum(dim=-1)

        if use_threshold:
            threshold = self.adamole_threshold[adapter_name]
            tau = self.adamole_threshold_tau[adapter_name]
            top = gate.max(dim=-1).values
            # Hard mask in the forward pass -> exact sparsity for tokens below threshold.
            mask_hard = (top > threshold).to(delta.dtype)
            # Straight-through: the forward value equals mask_hard, but the gradient flows through a smooth sigmoid
            # surrogate so the threshold parameter receives a non-zero gradient.
            mask_soft = torch.sigmoid((top - threshold) / tau)
            active = mask_hard + mask_soft - mask_soft.detach()
            delta = delta * active.unsqueeze(-1)

        return delta * scaling


class AdaMoLELinear(nn.Module, AdaMoLELayer):
    # AdaMoLE implemented on a dense (linear) layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        config: AdaMoLEConfig,
        **kwargs,
    ) -> None:
        super().__init__()
        AdaMoLELayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = config.fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, config=config)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        # AdaMoLE's contribution depends on the input through the router and threshold, so there is no static delta
        # weight to fold into the base weights.
        raise NotImplementedError(
            "AdaMoLE does not support merging adapters into the base weights: the expert contribution depends on the "
            "input through the router and the threshold, so there is no static delta weight to fold in."
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        result = self.base_layer(x, *args, **kwargs)
        # AdaMoLE never merges (see self.merge), so the only special case is disabled adapters.
        if (not self.disable_adapters) and (not self.merged):
            for active_adapter in self.active_adapters:
                if active_adapter not in self.adamole_A.keys():
                    continue
                delta = self.compute_delta(active_adapter, x)
                result = result + delta

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "adamole." + rep
