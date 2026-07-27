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

from __future__ import annotations

import math
import warnings
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from peft.tuners._buffer_dict import BufferDict
from peft.tuners.supertuning.layer import DensePlusSparseLinear
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

from .config import SupraConfig
from .mask import bottomk_magnitude_indices, compute_supra_budget


class SupraLayer(BaseTunerLayer):
    """
    Supra layer: hybrid sparse + low-rank fine-tuning under a matched scalar-parameter budget.

    Effective forward is::

        y = W_frozen @ x  +  scatter_add(sparse_values, indices) @ x  +  scaling · B @ A @ dropout(x)  +  bias

    implemented as three additive terms so the effective dense weight is never materialized:

    - ``W_frozen @ x + scatter_add(...)`` via `DensePlusSparseLinear` (reused from SupertuningLayer). The base weight
      receives no gradient; only the trainable sparse ``values`` do.
    - ``scaling · B @ A @ dropout(x)`` is standard LoRA (Kaiming-uniform A init, zero B init, so the low-rank term
      contributes zero at step 0).

    Total trainable scalar count is ``r · (in + out)`` per layer — the same as vanilla LoRA at rank ``r``. The budget
    split between sparse and LoRA is controlled by `config.lora_ratio` (paper's λ).
    """

    # Names PEFT machinery uses to identify trainable adapter tensors for save/load.
    adapter_layer_names = ("supra_sparse_values", "supra_lora_A", "supra_lora_B")
    # Non-parameter tensors that still need to travel with the adapter checkpoint.
    other_param_names = ("supra_indices", "supra_scaling", "supra_r_lora")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.supra_sparse_values = nn.ParameterDict({})
        self.supra_indices = BufferDict(persistent=True)
        self.supra_lora_A = nn.ModuleDict({})
        self.supra_lora_B = nn.ModuleDict({})
        self.supra_lora_dropout = nn.ModuleDict({})
        self.supra_scaling: dict[str, float] = {}
        self.supra_r_lora: dict[str, int] = {}
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            self.in_features = base_layer.in_features
            self.out_features = base_layer.out_features
        else:
            raise TypeError(f"Unsupported base layer type for Supra: {type(base_layer)}")

    def update_layer(self, adapter_name: str, config: SupraConfig, **kwargs) -> None:
        """
        Allocate the sparse support + LoRA factors for this adapter based on `config`.

        Splits the budget via `compute_supra_budget`, selects the sparse support via bottom-k (or top-k) magnitude
        (Wanda scoring is deferred to a calibration pass on the parent model), and allocates zero-initialized sparse
        `values` plus standard LoRA-initialized `A`/`B`.
        """
        base_layer = self.get_base_layer()
        weight = base_layer.weight.data  # shape: (out_features, in_features)

        r_lora, sparse_k = compute_supra_budget(
            r=config.r,
            lora_ratio=config.lora_ratio,
            in_features=self.in_features,
            out_features=self.out_features,
        )
        self.supra_r_lora[adapter_name] = r_lora

        # --- Sparse support (indices + values) ------------------------------
        # Bottom-k by default (paper's Supra rows are all BottomK); top-k selectable via config.
        # Wanda scoring is out of scope for the initial impl — magnitude is the paper's stronger 8B variant
        # and the only path that doesn't require calibration data at layer-init time.
        if config.scoring_method == "wanda":
            # Deferred: keep magnitude selection here as a placeholder; a follow-up calibration pass
            # on the parent SupraModel will re-select indices using activation-weighted scores. For now,
            # emit a warning so users don't silently get magnitude scoring when they asked for Wanda.
            warnings.warn(
                "Wanda scoring at init falls back to magnitude selection; call "
                "SupraModel.calibrate_saliency(dataset) after model construction to refine the support "
                "with activation-weighted scores."
            )
        if config.selection_direction == "bottom":
            indices = bottomk_magnitude_indices(weight, k=sparse_k)
        else:  # "top"
            # Fall back to the same magnitude computation but pick largest-k.
            if sparse_k == 0:
                indices = torch.empty(0, dtype=torch.int64, device=weight.device)
            else:
                _, indices = torch.topk(
                    weight.detach().abs().flatten(), k=min(sparse_k, weight.numel()), largest=True
                )
                indices = indices.to(torch.int64)
        self.supra_indices[adapter_name] = indices.to(torch.int32)

        # values start at zero for identity update at step 0 (matches reference impl + SupertuningLayer).
        if config.init_weights:
            values = torch.zeros(sparse_k, dtype=torch.float32, device=weight.device)
        else:
            values = torch.randn(sparse_k, dtype=torch.float32, device=weight.device)
        self.supra_sparse_values[adapter_name] = nn.Parameter(values)

        # --- LoRA factors ---------------------------------------------------
        # nn.Linear(in, out) stores weight as (out, in). Standard PEFT LoRA convention:
        #   lora_A: (in_features → r_lora)       # projects input to rank
        #   lora_B: (r_lora     → out_features)  # projects rank to output
        # Effective delta W = B @ A of shape (out_features, in_features).
        if r_lora > 0:
            lora_A = nn.Linear(self.in_features, r_lora, bias=False, device=weight.device, dtype=torch.float32)
            lora_B = nn.Linear(r_lora, self.out_features, bias=False, device=weight.device, dtype=torch.float32)
            # Standard LoRA init: A ~ Kaiming-uniform(a=sqrt(5)), B = 0 → delta W = 0 at step 0.
            nn.init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(lora_B.weight)
            self.supra_lora_A[adapter_name] = lora_A
            self.supra_lora_B[adapter_name] = lora_B
            self.supra_scaling[adapter_name] = config.lora_alpha / r_lora
        else:
            # Pure-Super regime — no LoRA factors allocated. Forward path skips the LoRA branch cleanly.
            self.supra_scaling[adapter_name] = 0.0

        dropout = nn.Dropout(p=config.lora_dropout) if config.lora_dropout > 0.0 else nn.Identity()
        self.supra_lora_dropout[adapter_name] = dropout

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=config.inference_mode)


class Linear(nn.Module, SupraLayer):
    """
    Supra applied to an `nn.Linear` base layer.

    Forward composes the sparse-plus-frozen matmul (via `DensePlusSparseLinear`) with the LoRA low-rank contribution.
    Merge/unmerge fold both components into the base weight, so post-merge inference has zero adapter overhead.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        config: SupraConfig,
        **kwargs,
    ) -> None:
        super().__init__()
        SupraLayer.__init__(self, base_layer)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, config=config)

    def _lora_delta(self, adapter_name: str) -> torch.Tensor:
        """Compute the LoRA delta weight ``scaling · B @ A`` for one adapter. Empty for pure-Super."""
        if adapter_name not in self.supra_lora_A:
            return None
        A = self.supra_lora_A[adapter_name].weight  # (r_lora, in_features)
        B = self.supra_lora_B[adapter_name].weight  # (out_features, r_lora)
        scaling = self.supra_scaling[adapter_name]
        return scaling * (B @ A)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """Fold sparse `values` + LoRA delta into the base weight."""
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        base_layer = self.get_base_layer()
        for adapter_name in adapter_names:
            if adapter_name not in self.supra_sparse_values.keys():
                continue
            weight = base_layer.weight
            indices = self.supra_indices[adapter_name].to(torch.int64)
            values = self.supra_sparse_values[adapter_name].to(weight.dtype)
            lora_delta = self._lora_delta(adapter_name)

            if safe_merge:
                merged = weight.data.reshape(-1).scatter_add(0, indices, values).reshape_as(weight)
                if lora_delta is not None:
                    merged = merged + lora_delta.to(weight.dtype)
                if not torch.isfinite(merged).all():
                    raise ValueError(f"NaNs in merged weights for adapter {adapter_name}; not merging.")
                weight.data = merged
            else:
                weight.data.reshape(-1).scatter_add_(0, indices, values)
                if lora_delta is not None:
                    weight.data.add_(lora_delta.to(weight.dtype))
            self.merged_adapters.append(adapter_name)

    def unmerge(self) -> None:
        """Undo `merge` by subtracting the sparse update + LoRA delta from the base weight."""
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        base_layer = self.get_base_layer()
        while self.merged_adapters:
            adapter_name = self.merged_adapters.pop()
            if adapter_name not in self.supra_sparse_values.keys():
                continue
            weight = base_layer.weight
            indices = self.supra_indices[adapter_name].to(torch.int64)
            values = self.supra_sparse_values[adapter_name].to(weight.dtype)
            lora_delta = self._lora_delta(adapter_name)
            weight.data.reshape(-1).scatter_add_(0, indices, -values)
            if lora_delta is not None:
                weight.data.sub_(lora_delta.to(weight.dtype))

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)
        if self.merged:
            return self.base_layer(x, *args, **kwargs)

        active_adapters = [a for a in self.active_adapters if a in self.supra_sparse_values.keys()]
        if not active_adapters:
            return self.base_layer(x, *args, **kwargs)

        base_layer = self.get_base_layer()
        weight = base_layer.weight
        bias = base_layer.bias

        # Sparse-plus-frozen matmul: single adapter uses the custom autograd path (grad flows only to values);
        # multi-adapter case falls back to scatter-add over the raw weight (native autograd is fine because no
        # grad reaches the frozen weight).
        if len(active_adapters) == 1:
            adapter_name = active_adapters[0]
            result = DensePlusSparseLinear.apply(
                x, weight, self.supra_indices[adapter_name], self.supra_sparse_values[adapter_name], bias
            )
        else:
            dense_plus_sparse = weight.reshape(-1)
            for adapter_name in active_adapters:
                indices = self.supra_indices[adapter_name].to(torch.int64)
                values = self.supra_sparse_values[adapter_name].to(weight.dtype)
                dense_plus_sparse = dense_plus_sparse.scatter_add(0, indices, values)
            dense_plus_sparse = dense_plus_sparse.reshape_as(weight)
            result = F.linear(x, dense_plus_sparse, bias)

        # LoRA contribution — one per adapter, added on top.
        for adapter_name in active_adapters:
            if adapter_name not in self.supra_lora_A:
                continue  # pure-Super for this adapter
            A = self.supra_lora_A[adapter_name]
            B = self.supra_lora_B[adapter_name]
            dropout = self.supra_lora_dropout[adapter_name]
            scaling = self.supra_scaling[adapter_name]
            # Cast to A's dtype (float32) to preserve LoRA numerics; cast result back to result's dtype.
            x_lora = dropout(x.to(A.weight.dtype))
            result = result + (B(A(x_lora)) * scaling).to(result.dtype)

        return result
