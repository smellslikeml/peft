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
class SaMAConfig(PeftConfig):
    """Configuration for [`SaMAModel`] — Scalable Morpho Adaptation (https://huggingface.co/papers/2611.04462).

    SaMA generalizes the Kronecker product by relaxing the shared-block constraint in its perfect-shuffle
    factorization: the additive update is ``ΔW = s · P_l^T L P_r R`` with independent dense blocks ``L = diag(L_1, ...,
    L_{m2})`` (each ``m1 × n1``) and ``R = diag(R_1, ..., R_{n1})`` (each ``m2 × n2``), fixed perfect-shuffle
    permutations ``P_l, P_r``, and scaling ``s = alpha · n1**0.25``. For an ``out_features × in_features`` layer this
    yields ``m1 = out_features // m2`` and ``n2 = in_features // n1``, ``n1·m2`` trainable rank, and ``n1·m2·(m1 +
    n2)`` trainable parameters — so the adapter rank scales quadratically with the parameter budget, up to full rank.

    Args:
        n1 (`int`): Column-block count of the right factor `R`; must divide `in_features`. Adapter is `SaMA_{n1×m2}`.
        m2 (`int`): Block count of the left factor `L`; must divide `out_features`.
        alpha (`float`): Scaling numerator; the effective scale is `alpha · n1**0.25` (Eq. 8).
        sigma (`float`): Std of the Gaussian init for `R` (`L` is initialized to zero so `ΔW = 0` at start).
        target_modules (`Optional[Union[list[str], str]]`): Modules to apply SaMA to.
        bias (`str`): Bias type — `'none'`, `'all'`, or `'sama_only'`.
        init_weights (`bool`): If `True`, standard init (`L=0`, `R~N(0, sigma^2)`). If `False`, also init `R=0`
            (making SaMA a no-op at start — for debugging).
        modules_to_save (`Optional[list[str]]`): Extra modules to train and save.
    """

    n1: int = field(default=16, metadata={"help": "Column-block count of R; must divide in_features."})
    m2: int = field(default=16, metadata={"help": "Block count of L; must divide out_features."})
    alpha: float = field(default=1.0, metadata={"help": "Scaling numerator; effective scale is alpha * n1**0.25."})
    sigma: float = field(default=2e-4, metadata={"help": "Std of the Gaussian init for R (L is zero-initialized)."})
    target_modules: Optional[Union[list[str], str]] = field(
        default=None, metadata={"help": "Modules to apply SaMA to."}
    )
    bias: str = field(default="none", metadata={"help": "Bias type: 'none', 'all', or 'sama_only'."})
    init_weights: bool = field(default=True, metadata={"help": "If False, also zero-init R (no-op SaMA)."})
    modules_to_save: Optional[list[str]] = field(default=None, metadata={"help": "Extra trainable/saved modules."})

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.SAMA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
