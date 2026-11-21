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

import torch

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer

from .layer import SaMALayer, SaMALinear


class SaMAModel(BaseTuner):
    """Creates a Scalable Morpho Adaptation (SaMA) model from a pretrained transformers model.

    The method is described in https://huggingface.co/papers/2611.04462.
    """

    prefix: str = "sama_"
    tuner_layer_cls = SaMALayer
    target_module_mapping = None

    def _create_and_replace(self, sama_config, adapter_name, target, target_name, parent, current_key, **kwargs):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")
        if isinstance(target, SaMALinear):
            target.update_layer(adapter_name, config=sama_config)
        else:
            new_module = self._create_new_module(sama_config, adapter_name, target)
            if adapter_name != self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(sama_config, adapter_name, target, **kwargs):
        target_base_layer = target.get_base_layer() if isinstance(target, BaseTunerLayer) else target
        if isinstance(target_base_layer, torch.nn.Linear):
            return SaMALinear(target, adapter_name, config=sama_config, **kwargs)
        raise TypeError(f"Target module {target} is not supported. Only `torch.nn.Linear` is supported.")
