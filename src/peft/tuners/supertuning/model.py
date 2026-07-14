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

from typing import Optional

import torch

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_SUPERTUNING_TARGET_MODULES_MAPPING

from .layer import Linear, SupertuningLayer


class SupertuningModel(BaseTuner):
    """
    Supertuning Model: Activation-Aware Sparse Fine-Tuning.

    This model implements sparse parameter-efficient fine-tuning by selecting a trainable support
    based on pruning-inspired saliency signals. The method uses a Wanda-style activation-weighted
    magnitude score computed from a calibration pass to determine which parameters should be trained.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`SupertuningConfig`]): The configuration of the Supertuning model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, *optional*, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    Returns:
        `torch.nn.Module`: The Supertuning model.

    Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import SupertuningModel, SupertuningConfig

        >>> config = SupertuningConfig(
        ...     peft_type="SUPERTUNING",
        ...     task_type="CAUSAL_LM",
        ...     target_modules=["q_proj", "v_proj"],
        ...     sparsity=0.5,
        ...     scoring_method="wanda",
        ... )

        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
        >>> supertuning_model = SupertuningModel(model, config, adapter_name="default")
        ```

    Paper: https://arxiv.org/abs/2607.09287

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`SupertuningConfig`]) -- The configuration of the Supertuning model.
    """

    prefix: str = "supertuning_"
    tuner_layer_cls = SupertuningLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_SUPERTUNING_TARGET_MODULES_MAPPING

    @staticmethod
    def _create_new_module(supertuning_config, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            new_module = Linear(target, adapter_name, config=supertuning_config, **kwargs)
        else:
            raise TypeError(
                f"Target module {target} is not supported. Currently, only `torch.nn.Linear` is supported."
            )
        return new_module

    def _create_and_replace(
        self,
        supertuning_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        kwargs = {}

        if isinstance(target, SupertuningLayer):
            target.update_layer(
                adapter_name,
                config=supertuning_config,
            )
        else:
            new_module = self._create_new_module(supertuning_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def calibrate_saliency(
        self,
        calibration_dataset,
        adapter_name: str = "default",
        num_samples: Optional[int] = None,
    ):
        """
        Run the calibration pass to compute activation-aware saliency scores.

        This method forwards calibration data through the model and updates the sparse masks
        based on the observed activations.

        Args:
            calibration_dataset: Dataset of calibration samples. Should be an iterable of
                tensors that can be passed to the model.
            adapter_name: Name of the adapter to calibrate. Defaults to "default".
            num_samples: Number of samples to use for calibration. If None, uses the value
                from the config.

        Example:

        ```py
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        >>> calibration_texts = ["Sample text 1", "Sample text 2", ...]
        >>> inputs = tokenizer(calibration_texts, return_tensors="pt", padding=True)
        >>> model.calibrate_saliency([inputs])
        ```
        """
        if num_samples is None:
            num_samples = self.peft_config[adapter_name].calibration_samples

        config = self.peft_config[adapter_name]
        device = next(self.model.parameters()).device

        self.model.eval()

        # Collect activation statistics from the calibration pass
        activation_hooks = []

        def get_activation_hook(name):
            def hook(module, input, output):
                # Store input activations for saliency computation
                if isinstance(module, SupertuningLayer):
                    for active_adapter in module.active_adapters:
                        if active_adapter == adapter_name:
                            # Store activation for later use
                            module._activation_stats[active_adapter] = input[0].detach()

            return hook

        # Register forward hooks to capture activations
        for name, module in self.model.named_modules():
            if isinstance(module, Linear):
                handle = module.register_forward_hook(
                    lambda mod, inp, out, m=module, n=name: self._store_activation(m, adapter_name, inp[0])
                )
                activation_hooks.append(handle)

        # Forward calibration data
        for sample_count, batch in enumerate(calibration_dataset):
            if sample_count >= num_samples:
                break

            if isinstance(batch, dict):
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                with torch.no_grad():
                    _ = self.model(**batch)
            else:
                # Assume batch is already a tensor or compatible
                with torch.no_grad():
                    _ = self.model(batch)

        # Remove hooks
        for handle in activation_hooks:
            handle.remove()

        # Update sparse masks based on collected activations
        for name, module in self.model.named_modules():
            if isinstance(module, Linear):
                if adapter_name in module._activation_stats:
                    activations = module._activation_stats[adapter_name]
                    module.update_sparse_mask_with_activations(adapter_name, activations, config)
                    # Clean up
                    del module._activation_stats[adapter_name]

        self.model.train()

    def _store_activation(self, module, adapter_name, activation):
        """Helper to store activations during calibration."""
        if hasattr(module, "_activation_stats"):
            module._activation_stats[adapter_name] = activation.detach()

    def get_trainable_parameters_count(self, adapter_name: str = "default") -> dict:
        """
        Get the count of trainable parameters for the given adapter.

        Args:
            adapter_name: Name of the adapter.

        Returns:
            Dictionary with total parameters and trainable parameters count.
        """
        total_params = 0
        trainable_params = 0

        for name, module in self.model.named_modules():
            if isinstance(module, Linear):
                base_layer = module.get_base_layer()
                weight_numel = base_layer.weight.numel()
                total_params += weight_numel

                if adapter_name in module.supertuning_values.keys():
                    trainable_params += int(module.supertuning_values[adapter_name].numel())

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "sparsity": 1.0 - (trainable_params / total_params if total_params > 0 else 0),
        }
