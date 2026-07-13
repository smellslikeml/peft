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
Super-Tuning configuration class.

Paper: https://arxiv.org/abs/2607.09287
Super-Tuning: From Activation-Aware Pruning to Sparse Fine-Tuning

This module implements the Super and Supra sparse PEFT methods that use
activation-aware pruning saliency signals to select trainable parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class SupertuningConfig(PeftConfig):
    """
    Configuration class for Super-Tuning sparse PEFT.

    Super-Tuning uses activation-aware pruning saliency signals to select a fixed
    sparse support of trainable parameters. Two variants are supported:
    - Super: pure sparse fine-tuning based on saliency scores
    - Supra: hybrid adapter combining sparse updates with LoRA

    Paper: https://arxiv.org/abs/2607.09287

    Args:
        sparsity_ratio (`float`, *optional*, defaults to `0.5`):
            Fraction of parameters to mask out (make non-trainable). For example,
            0.5 means 50% of parameters will be frozen.
        regularization_method (`str`, *optional*, defaults to `"wanda"`):
            Method for computing saliency scores:
            - `"wanda"`: Activation-weighted magnitude score (requires calibration data)
            - `"magnitude"`: Weight magnitude only (training-free)
        num_calibration_samples (`int`, *optional*, defaults to `32`):
            Number of samples to use for computing activation statistics in the
            calibration pass. Only used when regularization_method="wanda".
        target_modules (`Union[list[str], str]`):
            Module names or regex pattern to apply Super-Tuning to.
        adapter_type (`str`, *optional*, defaults to `"super"`):
            Type of adapter: "super" for pure sparse, "supra" for hybrid with LoRA.
        lora_rank (`int`, *optional*, defaults to `8`):
            LoRA rank for Supra hybrid adapter. Only used when adapter_type="supra".
        lora_alpha (`int`, *optional*, defaults to `16`):
            LoRA alpha scaling factor for Supra. Only used when adapter_type="supra".
        lora_dropout (`float`, *optional*, defaults to `0.0`):
            LoRA dropout for Supra. Only used when adapter_type="supra".
        budget_split (`float`, *optional*, defaults to `0.5`):
            Fraction of parameter budget allocated to sparse component in Supra.
            Remaining budget goes to LoRA. Only used when adapter_type="supra".
        init_weights (`bool`, *optional*, defaults to `True`):
            Initialize sparse weights with zeros. If False, uses random initialization.
        modules_to_save (`list[str]`, *optional*):
            Additional modules to set as trainable beyond Super-Tuning layers.
        fan_in_fan_out (`bool`, *optional*, defaults to `False`):
            Set to True if layer stores weights as (fan_in, fan_out).
        layers_to_transform (`Union[list[int],int]`, *optional*):
            Specific layer indices to transform.
        layers_pattern (`str`, *optional*):
            Layer pattern name for layer selection.
    """

    sparsity_ratio: float = field(
        default=0.5,
        metadata={"help": "Fraction of parameters to mask out (make non-trainable)."},
    )
    regularization_method: Literal["wanda", "magnitude"] = field(
        default="wanda",
        metadata={
            "help": "Method for computing saliency scores: 'wanda' for activation-aware, 'magnitude' for weight-only."
        },
    )
    num_calibration_samples: int = field(
        default=32,
        metadata={"help": "Number of calibration samples for computing activation statistics."},
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression to apply Super-Tuning to."
        },
    )
    adapter_type: Literal["super", "supra"] = field(
        default="super",
        metadata={
            "help": "Type of adapter: 'super' for pure sparse, 'supra' for hybrid with LoRA."
        },
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "LoRA rank for Supra hybrid adapter."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha scaling factor for Supra."},
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "LoRA dropout for Supra."},
    )
    budget_split: float = field(
        default=0.5,
        metadata={"help": "Fraction of budget for sparse component in Supra (remaining goes to LoRA)."},
    )
    init_weights: bool = field(
        default=True,
        metadata={"help": "Initialize sparse weights with zeros."},
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "Additional modules to set as trainable beyond Super-Tuning layers."
        },
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set to True if layer stores weights as (fan_in, fan_out)."},
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={"help": "Layer indices to transform."},
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={"help": "Layer pattern name for selection."},
    )

    def __post_init__(self):
        self.peft_type = PeftType.SUPERTUNING
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )

        if not 0 <= self.sparsity_ratio < 1:
            raise ValueError(f"sparsity_ratio must be in [0, 1), got {self.sparsity_ratio}")

        if self.adapter_type == "supra" and not 0 < self.budget_split < 1:
            raise ValueError(f"budget_split must be in (0, 1) for supra, got {self.budget_split}")
