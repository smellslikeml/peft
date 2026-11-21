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

# Clean-room implementation of SaMA (arXiv:2611.04462), Eqs. 4-5-8-10 / Algorithm 2. The reparameterization
# (factored forward == materialized ΔW, rank = n1·m2, params = n1·m2·(n2)+(m1)·m2·n1, ΔW=0 at init,
# s = alpha·n1**0.25) was validated standalone before integration.
from __future__ import annotations

import warnings
from typing import Any, Optional

import torch
from torch import nn

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

from .config import SaMAConfig


def _perfect_shuffle_index(p: int, q: int, device=None) -> torch.Tensor:
    """P_{p,q}: index vector of length p*q s.t. (P_{p,q} v) = v[idx] — reshape row-major (p,q), transpose, revec."""
    return torch.arange(p * q, device=device).reshape(p, q).t().reshape(-1)


def _inverse_perm(idx: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(idx)
    inv[idx] = torch.arange(len(idx), device=idx.device)
    return inv


class SaMALayer(BaseTunerLayer):
    adapter_layer_names = ("sama_L", "sama_R")
    other_param_names = ("sama_n1", "sama_m2", "sama_scaling")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.sama_L = nn.ParameterDict({})
        self.sama_R = nn.ParameterDict({})
        self.sama_n1 = {}
        self.sama_m2 = {}
        self.sama_scaling = {}
        self._shuffles = {}  # adapter -> (Pr, PlT_inv) buffers, non-trainable
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        else:
            raise TypeError(f"Unsupported layer type {type(base_layer)}")

    def update_layer(self, adapter_name: str, config: SaMAConfig, **kwargs):
        n1, m2 = config.n1, config.m2
        M, N = self.out_features, self.in_features
        if N % n1 != 0:
            raise ValueError(f"n1={n1} must divide in_features={N}.")
        if M % m2 != 0:
            raise ValueError(f"m2={m2} must divide out_features={M}.")
        m1, n2 = M // m2, N // n1
        q = n1 * m2
        if q > min(M, N):
            raise ValueError(f"n1*m2={q} exceeds min(out,in)={min(M, N)}; reduce n1 or m2.")

        weight = self.get_base_layer().weight
        dev = weight.device
        self.sama_n1[adapter_name] = n1
        self.sama_m2[adapter_name] = m2
        self.sama_scaling[adapter_name] = config.alpha * (n1**0.25)
        # R: n1 blocks of (m2 x n2) ; L: m2 blocks of (m1 x n1). L=0 -> ΔW=0 at init.
        self.sama_R[adapter_name] = nn.Parameter(torch.empty(n1, m2, n2, dtype=torch.float32, device=dev))
        self.sama_L[adapter_name] = nn.Parameter(torch.zeros(m1, m2, n1, dtype=torch.float32, device=dev))
        self._shuffles[adapter_name] = (
            _perfect_shuffle_index(n1, m2, dev),  # Pr on length q
            _inverse_perm(_perfect_shuffle_index(m1, m2, dev)),  # P_l^T on length M
        )
        self.reset_sama_parameters(adapter_name, config)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=config.inference_mode)

    @torch.no_grad()
    def reset_sama_parameters(self, adapter_name, config: SaMAConfig):
        if adapter_name not in self.sama_R:
            return
        nn.init.zeros_(self.sama_L[adapter_name])
        if config.init_weights:
            nn.init.normal_(self.sama_R[adapter_name], mean=0.0, std=config.sigma)
        else:
            nn.init.zeros_(self.sama_R[adapter_name])  # no-op SaMA (ΔW stays 0)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """Materialize ΔW = s · P_l^T L P_r R (M x N). Used for merge; the forward path stays factored."""
        if adapter not in self.sama_R:
            raise ValueError(f"Adapter {adapter} not found.")
        R = self.sama_R[adapter]  # (n1, m2, n2)
        L = self.sama_L[adapter]  # (m1, m2, n1)
        n1, m2 = self.sama_n1[adapter], self.sama_m2[adapter]
        m1, n2 = L.shape[0], R.shape[2]
        M, N, q = self.out_features, self.in_features, n1 * m2
        Pr, PlT_inv = self._shuffles[adapter]
        dtype = R.dtype
        Rbd = torch.zeros(q, N, device=R.device, dtype=dtype)
        for j in range(n1):
            Rbd[j * m2 : (j + 1) * m2, j * n2 : (j + 1) * n2] = R[j]
        Lbd = torch.zeros(M, q, device=L.device, dtype=dtype)
        for i in range(m2):
            Lbd[i * m1 : (i + 1) * m1, i * n1 : (i + 1) * n1] = L[:, i, :]
        delta = (Lbd @ Rbd[Pr, :])[PlT_inv, :]
        return (self.sama_scaling[adapter] * delta).to(self.get_base_layer().weight.dtype)

    def _factored_forward(self, x: torch.Tensor, adapter: str) -> torch.Tensor:
        """Algorithm 2: ΔW·x without materializing ΔW. x: (..., N) -> (..., M)."""
        R, L = self.sama_R[adapter], self.sama_L[adapter]
        n1, m2 = self.sama_n1[adapter], self.sama_m2[adapter]
        m1, n2 = L.shape[0], R.shape[2]
        lead = x.shape[:-1]
        xr = x.reshape(-1, n1, n2)  # (B, n1, n2)
        yr = torch.einsum("bjn,jmn->bjm", xr, R)  # block-matmul R -> (B, n1, m2)
        yp1 = yr.transpose(1, 2)  # P_r -> (B, m2, n1)
        yl = torch.einsum("bin,ain->bia", yp1, L)  # block-matmul L (block i shared) -> (B, m2, m1)
        yp2 = yl.transpose(1, 2)  # P_l^T -> (B, m1, m2)
        out = yp2.reshape(*lead, m1 * m2) * self.sama_scaling[adapter]
        return out


class SaMALinear(nn.Module, SaMALayer):
    def __init__(self, base_layer, adapter_name: str, config: SaMAConfig, **kwargs) -> None:
        super().__init__()
        SaMALayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, config=config)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return
        for active_adapter in adapter_names:
            if active_adapter in self.sama_R.keys():
                base_layer = self.get_base_layer()
                delta_weight = self.get_delta_weight(active_adapter)
                if safe_merge:
                    orig = base_layer.weight.data.clone() + delta_weight
                    if not torch.isfinite(orig).all():
                        raise ValueError(f"NaNs detected merging adapter {active_adapter}")
                    base_layer.weight.data = orig
                else:
                    base_layer.weight.data = base_layer.weight.data + delta_weight
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.sama_R.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            xf = x.to(torch.float32)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.sama_R.keys():
                    continue
                result = result + self._factored_forward(xf, active_adapter).to(result.dtype)
        return result.to(previous_dtype)

    def __repr__(self) -> str:
        return "sama." + super().__repr__()
