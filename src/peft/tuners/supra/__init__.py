# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Supra: hybrid sparse + low-rank fine-tuning (arXiv:2607.09287)."""

from peft.utils import register_peft_method

from .config import SupraConfig
from .layer import Linear, SupraLayer
from .mask import bottomk_magnitude_indices, compute_supra_budget
from .model import SupraModel


__all__ = [
    "Linear",
    "SupraConfig",
    "SupraLayer",
    "SupraModel",
    "bottomk_magnitude_indices",
    "compute_supra_budget",
]

register_peft_method(name="supra", config_cls=SupraConfig, model_cls=SupraModel, is_mixed_compatible=False)
