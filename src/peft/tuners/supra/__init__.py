# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Supra: hybrid sparse + low-rank fine-tuning (arXiv:2607.09287)."""

from peft.tuners.supra.config import SupraConfig
from peft.tuners.supra.layer import Linear, SupraLayer
from peft.tuners.supra.mask import bottomk_magnitude_indices, compute_supra_budget


__all__ = [
    "Linear",
    "SupraConfig",
    "SupraLayer",
    "bottomk_magnitude_indices",
    "compute_supra_budget",
]
