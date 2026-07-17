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

import warnings
from typing import Any, Optional

import torch
from torch import nn

from peft.tuners._buffer_dict import BufferDict
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

from .config import SupertuningConfig


class DensePlusSparseLinear(torch.autograd.Function):
    """
    Linear layer whose effective weight is the frozen base weight plus a sparse update.

    The sparse update is described by ``indices`` (flat positions into the weight) and ``values`` (the trainable
    quantities scatter-added at those positions). The backward pass only propagates a gradient to ``values`` (and to
    the input/bias); the base ``weight`` receives none, which is what keeps the frozen support unchanged during
    optimization. This mirrors the reference Super-Tuning implementation and, unlike masking a dense weight gradient,
    behaves correctly under stateful optimizers such as AdamW.
    """

    @staticmethod
    def forward(ctx, input, weight, indices, values, bias=None):
        ctx.save_for_backward(input, weight, indices, values, bias)

        dense_plus_sparse = weight.reshape(-1).scatter_add(0, indices.to(torch.int64), values.to(weight.dtype))
        dense_plus_sparse = dense_plus_sparse.reshape_as(weight)

        return torch.nn.functional.linear(input, dense_plus_sparse, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, indices, values, bias = ctx.saved_tensors
        grad_input = grad_values = grad_bias = None

        dense_plus_sparse = weight.reshape(-1).scatter_add(0, indices.to(torch.int64), values.to(weight.dtype))
        dense_plus_sparse = dense_plus_sparse.reshape_as(weight)

        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output, dense_plus_sparse)

        if ctx.needs_input_grad[3] or (bias is not None and ctx.needs_input_grad[4]):
            grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
            if ctx.needs_input_grad[3]:
                input_2d = input.reshape(-1, input.shape[-1])
                grad_matrix = grad_output_2d.t().mm(input_2d)
                grad_values = grad_matrix.reshape(-1).gather(0, indices.to(torch.int64)).to(values.dtype)
            if bias is not None and ctx.needs_input_grad[4]:
                grad_bias = grad_output_2d.sum(dim=0)

        # No gradient flows to the (frozen) base weight or to the integer indices.
        return grad_input, None, None, grad_values, grad_bias


class SupertuningLayer(BaseTunerLayer):
    """
    Supertuning layer that implements sparse fine-tuning using activation-aware saliency.

    The layer computes saliency scores for each parameter using either:
    - Wanda-style activation-weighted magnitude scoring
    - Magnitude-only scoring (PaFi-style baseline)

    It then selects the top-k most salient parameters as the trainable sparse support and stores that support as a
    compact ``(indices, values)`` pair sized to the trainable count: ``indices`` are the flat positions of the support
    inside the weight and ``values`` are the trainable quantities scatter-added onto the frozen base weight. Only
    ``values`` is a trainable parameter; the base weight is frozen.
    """

    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("supertuning_values",)

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        # Trainable sparse quantities, one 1-D parameter per adapter.
        self.supertuning_values = nn.ParameterDict({})
        # Flat positions of the sparse support inside the weight, one 1-D integer buffer per adapter. Persistent so
        # that it is saved alongside ``supertuning_values`` in the adapter checkpoint.
        self.supertuning_indices = BufferDict(persistent=True)
        self._activation_stats = {}  # Store activation statistics for calibration
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            raise TypeError(f"Unsupported layer type {type(base_layer)}")
        self.in_features = in_features
        self.out_features = out_features

    def _num_trainable(self, num_params: int, sparsity: float) -> int:
        """Number of trainable parameters given the (frozen) sparsity ratio."""
        return int(num_params * (1 - sparsity))

    def update_layer(self, adapter_name: str, config: SupertuningConfig, **kwargs):
        """
        Update the layer with a new adapter configuration.

        This method selects the sparse support (``indices``) based on the saliency scoring method and allocates the
        trainable ``values`` for it.
        """
        base_layer = self.get_base_layer()
        sparsity = config.sparsity
        scoring_method = config.scoring_method
        init_weights = config.init_weights

        # Get the weight tensor
        weight = base_layer.weight.data  # shape: (out_features, in_features)

        # Compute initial saliency scores. Wanda scoring is refined later during calibration; until then both methods
        # fall back to the weight magnitude.
        if scoring_method == "magnitude":
            scores = weight.abs().flatten()
        else:  # "wanda"
            scores = weight.abs().flatten()

        # Select the trainable support: TopK (most-salient, paper's "Super"/"Supra") or BottomK
        # (least-salient, paper's "-bottom" variants).
        num_params = scores.numel()
        num_trainable = self._num_trainable(num_params, sparsity)
        largest = config.selection_direction == "top"
        _, indices = torch.topk(scores, k=num_trainable, largest=largest)
        indices = indices.to(dtype=torch.int32, device=weight.device)

        self.supertuning_indices[adapter_name] = indices
        # ``values`` start at zero, i.e. an identity update. ``init_weights=False`` seeds a non-identity update, which
        # is only used to exercise a non-trivial adapter in tests.
        if init_weights:
            values = torch.zeros(num_trainable, dtype=torch.float32, device=weight.device)
        else:
            values = torch.randn(num_trainable, dtype=torch.float32, device=weight.device)
        self.supertuning_values[adapter_name] = nn.Parameter(values)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=config.inference_mode)

    def compute_saliency_wanda(self, weight: torch.Tensor, activations: torch.Tensor) -> torch.Tensor:
        """
        Compute Wanda-style saliency scores: magnitude * activation magnitude.

        Args:
            weight: Weight tensor of shape (out_features, in_features)
            activations: Activation tensor from forward pass

        Returns:
            Saliency scores with same shape as weight
        """
        # Average activation magnitude across batch and sequence dims
        if activations.dim() > 2:
            act_mag = activations.abs().mean(dim=tuple(range(activations.dim() - 1)))
        else:
            act_mag = activations.abs().mean(dim=0)

        # Wanda score: |W| * |activation|
        scores = weight.abs() * act_mag.unsqueeze(0)
        return scores

    def compute_saliency_magnitude(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Compute magnitude-only saliency scores (PaFi-style).

        Args:
            weight: Weight tensor

        Returns:
            Saliency scores with same shape as weight
        """
        return weight.abs()

    def update_sparse_mask_with_activations(
        self, adapter_name: str, activations: torch.Tensor, config: SupertuningConfig
    ):
        """
        Re-select the sparse support based on activation-aware saliency.

        This should be called during the calibration pass with sample data. It only updates ``indices`` (the support
        positions); the trainable count and the ``values`` allocation are unchanged.

        Args:
            adapter_name: Name of the adapter
            activations: Activation tensor from forward pass
            config: SupertuningConfig with sparsity and scoring_method
        """
        base_layer = self.get_base_layer()
        weight = base_layer.weight.data

        # Compute saliency scores
        if config.scoring_method == "wanda":
            scores = self.compute_saliency_wanda(weight, activations)
        else:  # "magnitude"
            scores = self.compute_saliency_magnitude(weight)

        # Re-select the support, honoring the configured direction.
        num_params = scores.numel()
        num_trainable = self._num_trainable(num_params, config.sparsity)
        largest = config.selection_direction == "top"
        _, indices = torch.topk(scores.flatten(), k=num_trainable, largest=largest)

        self.supertuning_indices[adapter_name] = indices.to(
            dtype=torch.int32, device=self.supertuning_indices[adapter_name].device
        )


class Linear(nn.Module, SupertuningLayer):
    """
    Supertuning implemented in a dense linear layer.

    The base weight is frozen; the sparse support is scatter-added onto it in the forward pass through a custom
    autograd function so that only the trainable ``values`` receive a gradient.
    """

    # Supertuning implemented in a dense layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        config: SupertuningConfig,
        **kwargs,
    ) -> None:
        super().__init__()
        SupertuningLayer.__init__(self, base_layer)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, config=config)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the sparse support into the base weight by scatter-adding the trained ``values`` at their ``indices``.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        base_layer = self.get_base_layer()
        for active_adapter in adapter_names:
            if active_adapter not in self.supertuning_values.keys():
                continue
            weight = base_layer.weight
            indices = self.supertuning_indices[active_adapter].to(torch.int64)
            values = self.supertuning_values[active_adapter].to(weight.dtype)

            if safe_merge:
                merged = weight.data.reshape(-1).scatter_add(0, indices, values).reshape_as(weight)
                if not torch.isfinite(merged).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )
                weight.data = merged
            else:
                weight.data.reshape(-1).scatter_add_(0, indices, values)
            self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        Unmerge the sparse support from the base weight by subtracting the merged ``values``.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        base_layer = self.get_base_layer()
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter not in self.supertuning_values.keys():
                continue
            weight = base_layer.weight
            indices = self.supertuning_indices[active_adapter].to(torch.int64)
            values = self.supertuning_values[active_adapter].to(weight.dtype)
            weight.data.reshape(-1).scatter_add_(0, indices, -values)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass.

        The frozen base weight is combined with the active adapters' sparse support and applied as a single linear
        layer. The combination is done through :class:`DensePlusSparseLinear` so that the backward pass only reaches
        the trainable ``values``.
        """
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)

        if self.merged:
            return self.base_layer(x, *args, **kwargs)

        active_adapters = [a for a in self.active_adapters if a in self.supertuning_values.keys()]
        if not active_adapters:
            return self.base_layer(x, *args, **kwargs)

        base_layer = self.get_base_layer()
        weight = base_layer.weight
        bias = base_layer.bias

        if len(active_adapters) == 1:
            adapter_name = active_adapters[0]
            return DensePlusSparseLinear.apply(
                x, weight, self.supertuning_indices[adapter_name], self.supertuning_values[adapter_name], bias
            )

        # Multiple active adapters: combine their sparse supports on top of the frozen weight. The frozen weight
        # receives no gradient, so the native ``scatter_add`` autograd is sufficient here.
        dense_plus_sparse = weight.reshape(-1)
        for adapter_name in active_adapters:
            indices = self.supertuning_indices[adapter_name].to(torch.int64)
            values = self.supertuning_values[adapter_name].to(weight.dtype)
            dense_plus_sparse = dense_plus_sparse.scatter_add(0, indices, values)
        dense_plus_sparse = dense_plus_sparse.reshape_as(weight)
        return torch.nn.functional.linear(x, dense_plus_sparse, bias)
