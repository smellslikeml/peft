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

import torch

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_SUPRA_TARGET_MODULES_MAPPING

from .layer import Linear, SupraLayer


class SupraModel(BaseTuner):
    """
    Supra Model: hybrid sparse + low-rank fine-tuning under a matched scalar-parameter budget (arXiv:2607.09287, Eq.
    14).

    For each target ``nn.Linear``, Supra replaces it with a Supra `Linear` layer whose effective forward is::

        W_effective = W_frozen  +  scatter_add(sparse_values, indices)  +  scaling · B @ A

    Total trainable scalar count per layer is ``r · (in + out)`` — same as vanilla LoRA at rank ``r`` — split between
    the sparse and LoRA components by `SupraConfig.lora_ratio` (paper's λ). The base weight stays frozen.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`SupraConfig`]): The Supra configuration.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, *optional*, defaults to `False`):
            Create empty adapter weights on meta device to speed up loading.

    Returns:
        `torch.nn.Module`: The adapted model.

    Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import SupraConfig, SupraModel

        >>> config = SupraConfig(
        ...     task_type="CAUSAL_LM",
        ...     target_modules=["q_proj", "v_proj"],
        ...     r=8,
        ...     lora_ratio=0.5,
        ... )
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
        >>> supra_model = SupraModel(model, config, adapter_name="default")
        ```

    Paper: https://arxiv.org/abs/2607.09287
    """

    prefix: str = "supra_"
    tuner_layer_cls = SupraLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_SUPRA_TARGET_MODULES_MAPPING

    @staticmethod
    def _create_new_module(supra_config, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            new_module = Linear(target, adapter_name, config=supra_config, **kwargs)
        else:
            raise TypeError(
                f"Target module {target} is not supported. Currently, only `torch.nn.Linear` is supported."
            )
        return new_module

    def _create_and_replace(
        self,
        supra_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        kwargs = {}

        if isinstance(target, SupraLayer):
            target.update_layer(adapter_name, config=supra_config)
        else:
            new_module = self._create_new_module(supra_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # Adding an additional adapter: not automatically trainable.
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def get_trainable_parameters_count(self, adapter_name: str = "default") -> dict:
        """
        Report trainable-scalar counts for the given adapter, split into sparse and LoRA contributions.

        Returns:
            Dictionary with base-weight total, trainable-parameter count, and the sparse/LoRA split. Useful for
            confirming the paper's budget invariant ``sparse + lora = r · (in + out)`` holds across the model.
        """
        total_params = 0
        sparse_params = 0
        lora_params = 0

        for _, module in self.model.named_modules():
            if isinstance(module, Linear):
                base_layer = module.get_base_layer()
                total_params += base_layer.weight.numel()

                if adapter_name in module.supra_sparse_values:
                    sparse_params += int(module.supra_sparse_values[adapter_name].numel())
                if adapter_name in module.supra_lora_A:
                    lora_params += int(module.supra_lora_A[adapter_name].weight.numel())
                    lora_params += int(module.supra_lora_B[adapter_name].weight.numel())

        trainable = sparse_params + lora_params
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable,
            "sparse_parameters": sparse_params,
            "lora_parameters": lora_params,
            "trainable_fraction": (trainable / total_params) if total_params > 0 else 0.0,
        }

    def calibrate_saliency(self, calibration_dataset, adapter_name: str = "default", num_samples=None):
        """
        Placeholder for activation-aware (Wanda) support re-selection.

        The initial Supra release ships magnitude-only scoring — the paper's Supra-Mag variant, which is the stronger
        of the two on the 8B model and requires no calibration data. Wanda support re-selection is planned but not yet
        wired into the layer; calling this method raises so users don't silently get magnitude scoring when they
        expected activation-weighted.
        """
        raise NotImplementedError(
            "Wanda calibration for Supra is not yet implemented — set `scoring_method='magnitude'` "
            "on SupraConfig (default) or wait for the follow-up patch that adds the activation-aware pass."
        )
