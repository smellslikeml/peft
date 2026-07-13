# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a a copy of the License at
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

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

from .config import SupertuningConfig
from .gradient_masking import register_gradient_mask_hook, remove_gradient_mask_hook


class SupertuningLayer(BaseTunerLayer):
    """
    Supertuning layer that implements sparse fine-tuning using activation-aware saliency.

    The layer computes saliency scores for each parameter using either:
    - Wanda-style activation-weighted magnitude scoring
    - Magnitude-only scoring (PaFi-style baseline)

    Then it creates a binary mask that selects the top-k parameters to train.
    """

    # All names of layers that may contain adapter weights
    adapter_layer_names = ("supertuning_sparse_mask",)

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.supertuning_sparse_mask = nn.ParameterDict({})
        self._activation_stats = {}  # Store activation statistics for calibration
        self._gradient_mask_handle = None  # Handle of the weight-gradient masking hook, if enabled
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

    def update_layer(self, adapter_name: str, config: SupertuningConfig, **kwargs):
        """
        Update the layer with a new adapter configuration.

        This method initializes the sparse mask based on the saliency scoring method.
        """
        base_layer = self.get_base_layer()
        init_weights = config.init_weights
        inference_mode = config.inference_mode
        sparsity = config.sparsity
        scoring_method = config.scoring_method

        # Get the weight tensor
        weight = base_layer.weight.data  # shape: (out_features, in_features)

        # Compute initial saliency scores
        if scoring_method == "magnitude":
            # Magnitude-only scoring (PaFi-style)
            scores = weight.abs().flatten()
        else:  # "wanda"
            # Wanda-style: activation-weighted magnitude
            # Initialize with magnitude only, will be updated during calibration
            scores = weight.abs().flatten()

        # Determine number of parameters to keep trainable
        num_params = scores.numel()
        num_trainable = int(num_params * (1 - sparsity))

        # Select top-k indices (lowest score = least salient = pruned)
        # We want to keep the most salient parameters
        _, indices = torch.topk(scores, k=num_trainable, largest=True)

        # Create binary mask: 1 for trainable, 0 for frozen
        mask = torch.zeros(num_params, device=weight.device)
        mask[indices] = 1.0
        mask = mask.reshape_as(weight)

        self.supertuning_sparse_mask[adapter_name] = nn.Parameter(mask, requires_grad=False)

        if init_weights:
            self.reset_supertuning_parameters(adapter_name)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=inference_mode)

    def reset_supertuning_parameters(self, adapter_name: str):
        """
        Reset the sparse mask. This is called during initialization.
        """
        if adapter_name in self.supertuning_sparse_mask.keys():
            # Mask is already set during update_layer, this is a no-op
            pass

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
        Update the sparse mask based on activation-aware saliency.

        This should be called during the calibration pass with sample data.

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

        # Determine number of parameters to keep trainable
        num_params = scores.numel()
        num_trainable = int(num_params * (1 - config.sparsity))

        # Select top-k indices
        _, indices = torch.topk(scores.flatten(), k=num_trainable, largest=True)

        # Update binary mask
        mask = torch.zeros(num_params, device=weight.device)
        mask[indices] = 1.0
        mask = mask.reshape_as(weight)

        self.supertuning_sparse_mask[adapter_name].data = mask

    def enable_gradient_masking(self):
        """
        Enforce the sparse support by masking the weight gradient.

        Registers a single tensor hook on the wrapped weight so that, during the backward pass, gradient entries
        outside the sparse support are zeroed. This is the mechanism that actually keeps the frozen parameters
        unchanged during optimization. Safe to call repeatedly: any previous hook is removed first.
        """
        return register_gradient_mask_hook(self)

    def disable_gradient_masking(self):
        """Remove the weight-gradient masking hook, letting the full weight receive updates again."""
        remove_gradient_mask_hook(self)


class Linear(nn.Module, SupertuningLayer):
    """
    Supertuning implemented in a dense linear layer.

    This layer applies gradient masking to ensure only parameters in the sparse
    support (selected by the mask) receive updates during training.
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
        self.use_gradient_masking = config.use_gradient_masking
        self.update_layer(adapter_name, config=config)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the sparse adapter. For Supertuning, this is a no-op since we're not
        adding parameters but rather selecting which existing parameters to train.
        """
        # Supertuning doesn't have separate parameters to merge
        # The sparse mask controls which base parameters are trainable
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.supertuning_sparse_mask.keys():
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        Unmerge the sparse adapter.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            self.merged_adapters.pop()

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass.

        Supertuning does not modify the forward computation: it reuses the base layer as-is. The sparse support is
        enforced on the backward pass instead, through the weight-gradient hook installed by
        ``enable_gradient_masking`` (registered once, not per forward). Keeping ``forward`` free of hook registration
        avoids the hook leak and the shape mismatch of masking the layer-output gradient.
        """
        if self.disable_adapters and self.merged:
            self.unmerge()
        return self.base_layer(x, *args, **kwargs)
