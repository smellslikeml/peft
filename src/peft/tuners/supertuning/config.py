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

from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class SupertuningConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`SupertuningModel`].

    Supertuning (Super-Tuning) implements activation-aware sparse fine-tuning using pruning-inspired saliency
    signals to select which parameters should be trainable. The method uses a Wanda-style activation-weighted
    magnitude score computed from a calibration pass to determine the sparse support.

    Args:
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the
            specified names will be replaced. When passing a string, a regex match will be performed. When
            passing a list of strings, either an exact match will be performed or it is checked if the name
            of the module ends with any of the passed strings. If this is not specified, modules will be chosen
            according to the model architecture.
        modules_to_save (`Optional[List[str]]`):
            List of modules apart from Supertuning layers to be set as trainable and saved in the final
            checkpoint.
        sparsity (`float`):
            The target sparsity ratio (0.0 to 1.0). For example, 0.5 means 50% of parameters will be trainable.
            Defaults to 0.5.
        calibration_samples (`int`):
            Number of samples to use for the calibration pass to compute activation-aware saliency scores.
            Defaults to 32.
        scoring_method (`str`):
            The method to use for computing saliency scores. Can be "wanda" (activation-weighted magnitude)
            or "magnitude" (magnitude-only, similar to PaFi). Defaults to "wanda".
        init_weights (`bool`):
            Whether to initialize the trainable sparse values to zero (an identity update) during setup.
            Defaults to `True`.

    Paper: https://arxiv.org/abs/2607.09287
    """

    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with Supertuning. "
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. "
                "If not specified, modules will be chosen according to the model architecture."
            ),
        },
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from Supertuning layers to be set as trainable and saved in the final "
                "checkpoint. For example, in Sequence Classification or Token Classification tasks, the final "
                "layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
            ),
        },
    )
    sparsity: float = field(
        default=0.5,
        metadata={
            "help": (
                "Target sparsity ratio. For example, 0.5 means 50% of parameters will be trainable. "
                "Must be between 0.0 and 1.0."
            ),
        },
    )
    calibration_samples: int = field(
        default=32,
        metadata={
            "help": "Number of calibration samples to compute activation-aware saliency scores."
        },
    )
    scoring_method: str = field(
        default="wanda",
        metadata={
            "help": (
                "Method for computing saliency scores. 'wanda' uses activation-weighted magnitude, "
                "'magnitude' uses weight magnitude only."
            )
        },
    )
    init_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the trainable sparse values to zero (an identity update)."},
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.SUPERTUNING
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )

        # Validate sparsity
        if not 0.0 <= self.sparsity < 1.0:
            raise ValueError(f"sparsity must be between 0.0 and 1.0 (exclusive of 1.0), got {self.sparsity}")

        # Validate scoring_method
        if self.scoring_method not in ["wanda", "magnitude"]:
            raise ValueError(f"scoring_method must be 'wanda' or 'magnitude', got {self.scoring_method}")
