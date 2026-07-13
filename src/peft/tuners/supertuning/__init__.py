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
Super-Tuning sparse PEFT method.

Paper: https://arxiv.org/abs/2607.09287
Super-Tuning: From Activation-Aware Pruning to Sparse Fine-Tuning

This module implements Super and Supra, sparse PEFT methods that use
activation-aware pruning saliency signals to select trainable parameters.
"""

from peft.utils import register_peft_method

from .config import SupertuningConfig
from .model import SupertuningModel


register_peft_method(
    name="supertuning",
    config_cls=SupertuningConfig,
    model_cls=SupertuningModel,
)


__all__ = ["SupertuningConfig", "SupertuningModel"]
