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
Super-Tuning model implementation.

This module implements the SupertuningModel class that performs activation-aware
pruning to select sparse trainable parameters.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from ..tuners_utils import _maybe_include_all_linear_layers
from .config import SupertuningConfig
from .layer import Linear, SupertuningLayer


class SupertuningModel(BaseTuner):
    """
    Creates a Super-Tuning model from a pretrained transformers model.

    Super-Tuning uses activation-aware pruning saliency signals to select a fixed
    sparse support of trainable parameters. Two variants are supported:
    - Super: pure sparse fine-tuning based on saliency scores
    - Supra: hybrid adapter combining sparse updates with LoRA

    Paper: https://arxiv.org/abs/2607.09287

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`SupertuningConfig`]): The configuration of the Super-Tuning model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Example:
        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import SupertuningConfig, get_peft_model

        >>> base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        >>> config = SupertuningConfig(sparsity_ratio=0.5, regularization_method="wanda")
        >>> model = get_peft_model(base_model, config)
        ```
    """

    prefix: str = "supertuning_"
    tuner_layer_cls = SupertuningLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name):
        """Initialize the SupertuningModel."""
        self.masks = {}
        super().__init__(model, config, adapter_name)

    def _compute_wanda_saliency(
        self, weight: torch.Tensor, activation_stats: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Wanda-style activation-weighted magnitude saliency scores.

        Args:
            weight: The weight tensor of shape (out_features, in_features)
            activation_stats: Accumulated activation statistics of shape (in_features,)

        Returns:
            Saliency scores of the same shape as weight
        """
        # Wanda saliency: |weight| * activation_magnitude
        # Lower score = more important (should keep trainable)
        weight_magnitude = torch.abs(weight)
        activation_magnitude = activation_stats.unsqueeze(0)  # Broadcast to (1, in_features)

        # Saliency is higher for less important parameters
        saliency = weight_magnitude * activation_magnitude
        return saliency

    def _compute_magnitude_saliency(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Compute magnitude-only saliency scores (training-free).

        Args:
            weight: The weight tensor

        Returns:
            Saliency scores of the same shape as weight
        """
        return torch.abs(weight)

    def _calibration_pass(
        self, model: nn.Module, config: SupertuningConfig, adapter_name: str
    ) -> dict[str, torch.Tensor]:
        """
        Perform a calibration pass to compute activation statistics.

        This runs a forward pass with dummy data to collect activation statistics
        for Wanda-style saliency computation.

        Args:
            model: The base model
            config: SupertuningConfig
            adapter_name: Name of the adapter

        Returns:
            Dictionary mapping layer names to activation statistics
        """
        # Collect activation statistics from each target layer
        activation_stats = {}
        hooks = []

        def make_hook(name):
            def hook(module, input, output):
                # For linear layers, input is (batch, seq, hidden) or (batch, hidden)
                if isinstance(input, tuple) and len(input) > 0:
                    x = input[0]
                    if x.dim() == 3:
                        x = x.view(-1, x.size(-1))  # (batch*seq, hidden)
                    # Accumulate mean activation magnitude per feature
                    activation_stats[name] = activation_stats.get(name, 0) + x.abs().mean(dim=0)
            return hook

        # Register hooks to collect activation statistics
        for name, module in model.named_modules():
            if not self._check_target_module_exists(config, name):
                continue
            if isinstance(module, (nn.Linear, Conv1D)):
                hooks.append(module.register_forward_hook(make_hook(name)))

        # Run calibration forward pass
        try:
            # Create dummy input - this is a simplified approach
            # In practice, users should provide real calibration data
            model_config = getattr(model, "config", None)
            if model_config is None:
                warnings.warn("No model config found, using dummy calibration")
                return {}

            # Use small dummy batch for calibration
            device = next(model.parameters()).device
            vocab_size = getattr(model_config, "vocab_size", 32000)
            hidden_size = getattr(model_config, "hidden_size", 768)

            # Create dummy input tokens
            dummy_input = torch.randint(
                0, vocab_size, (config.num_calibration_samples, 32), device=device
            )

            with torch.no_grad():
                try:
                    model(dummy_input)
                except Exception:
                    # If forward pass fails, fall back to magnitude-only
                    warnings.warn(
                        "Calibration forward pass failed. Falling back to magnitude-only saliency."
                    )
                    return {}
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

        # Normalize activation statistics
        num_samples = config.num_calibration_samples
        for name in activation_stats:
            activation_stats[name] = activation_stats[name] / num_samples

        return activation_stats

    def _compute_masks(
        self, model: nn.Module, config: SupertuningConfig, adapter_name: str
    ) -> dict[str, torch.Tensor]:
        """
        Compute binary masks for each target layer based on saliency scores.

        Args:
            model: The base model
            config: SupertuningConfig
            adapter_name: Name of the adapter

        Returns:
            Dictionary mapping layer names to binary masks
        """
        masks = {}

        # Get activation statistics if using Wanda
        if config.regularization_method == "wanda":
            activation_stats = self._calibration_pass(model, config, adapter_name)
            if not activation_stats:
                # Fall back to magnitude-only if calibration failed
                config.regularization_method = "magnitude"
                warnings.warn("Calibration failed, using magnitude-only saliency")

        for name, module in model.named_modules():
            if not self._check_target_module_exists(config, name):
                continue

            if isinstance(module, nn.Linear):
                weight = module.weight.data  # (out_features, in_features)
            elif isinstance(module, Conv1D):
                weight = module.weight.view(module.weight.shape[1], module.weight.shape[0])
            else:
                continue

            # Compute saliency scores
            if config.regularization_method == "wanda" and name in activation_stats:
                saliency = self._compute_wanda_saliency(weight, activation_stats[name])
            else:  # magnitude
                saliency = self._compute_magnitude_saliency(weight)

            # Select parameters to keep trainable (lowest saliency)
            num_params = weight.numel()
            num_trainable = int(num_params * (1 - config.sparsity_ratio))

            # For Super-Tuning, we select LOW-SCORE support (less salient = more trainable)
            # Flatten saliency and get indices of lowest scores
            flat_saliency = saliency.flatten()
            threshold_idx = torch.argsort(flat_saliency)[num_trainable]
            threshold = flat_saliency[threshold_idx] if num_trainable > 0 else float('inf')

            # Create mask: 1 for trainable, 0 for frozen
            mask = (saliency <= threshold).float()
            masks[name] = mask

        return masks

    def _pre_injection_hook(self, model: nn.Module, config: SupertuningConfig, adapter_name: str) -> None:
        """Compute masks before injecting layers."""
        self.masks[adapter_name] = self._compute_masks(model, config, adapter_name)

    def _create_and_replace(
        self,
        supertuning_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "fan_in_fan_out": supertuning_config.fan_in_fan_out,
        }
        kwargs["bias"] = bias

        # Get the pre-computed mask for this layer
        mask = self.masks.get(adapter_name, {}).get(current_key)

        if isinstance(target, Linear):
            target.update_layer(
                adapter_name,
                mask,
                config=supertuning_config,
            )
        else:
            new_module = self._create_new_module(
                supertuning_config, adapter_name, target, mask, **kwargs
            )
            if adapter_name not in self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(supertuning_config, adapter_name, target, mask, **kwargs):
        """Create a new Supertuning layer."""
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            if supertuning_config.fan_in_fan_out:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                supertuning_config.fan_in_fan_out = False
        elif isinstance(target_base_layer, Conv1D):
            kwargs["is_target_conv_1d_layer"] = True
            if not supertuning_config.fan_in_fan_out:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                    "Setting fan_in_fan_out to True."
                )
                supertuning_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `transformers.pytorch_utils.Conv1D`."
            )

        new_module = Linear(
            target,
            mask,
            adapter_name,
            config=supertuning_config,
            **kwargs,
        )

        return new_module

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "supertuning." + rep
