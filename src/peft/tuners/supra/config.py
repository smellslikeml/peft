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
class SupraConfig(PeftConfig):
    """
    Configuration for [`SupraModel`].

    Supra (arXiv:2607.09287 Eq. 14) is a hybrid PEFT method that combines Super-Tuning's sparse update with a low-rank
    LoRA update under a matched scalar-parameter budget::

        W_effective = W_frozen  +  M ⊙ U  +  γ · L @ R
                                  └── sparse ──┘ └── LoRA ──┘

    Total trainable scalar count is ``r · (in + out)`` per layer — the same as vanilla LoRA at rank ``r`` — split
    between the sparse component (``sparse_k`` entries) and the LoRA rank (``r_lora``) via ``lora_ratio``:

        r_lora = floor(lora_ratio · r)
        sparse_k = (r − r_lora) · (in + out)

    ``lora_ratio=0.0`` → pure Super (all budget as sparse), ``lora_ratio=1.0`` → pure LoRA. The paper's headline
    results use ``r=8`` and sweep ``lora_ratio ∈ {0.3, 0.5, 0.8}``; the best split is model-dependent (λ=0.3 wins for
    Meta-Llama-3-8B, λ=0.8 wins for Llama-3.2-1B).

    Args:
        target_modules (`Optional[Union[List[str], str]]`):
            Modules to replace with Supra layers, exact match or regex. If unspecified, chosen from the model
            architecture. Follows the same convention as `LoraConfig.target_modules`.
        modules_to_save (`Optional[List[str]]`):
            Extra modules to mark trainable + saved (e.g. randomly-initialized classifier heads).
        r (`int`):
            Rank-equivalent budget. Total trainable scalar count per layer is ``r · (in_features + out_features)``,
            matching a vanilla LoRA baseline at rank ``r`` for apples-to-apples comparison. Defaults to 8 (the paper's
            headline budget).
        lora_ratio (`float`):
            Fraction of the budget allocated to LoRA. ``0.0`` → pure sparse, ``1.0`` → pure LoRA. Defaults to 0.5 (the
            middle of the paper's sweep). The paper reports best value is model-dependent; sweep over
            ``{0.3, 0.5, 0.8}`` for new setups.
        lora_alpha (`int`):
            LoRA scaling factor. Effective scaling is ``lora_alpha / r_lora``. Defaults to 16 (matches the paper).
        lora_dropout (`float`):
            Dropout on the LoRA input path. Defaults to 0.05 (matches the paper).
        scoring_method (`str`):
            How to score weights for sparse-support selection. ``"magnitude"`` (paper's Supra-Mag, no calibration data
            needed, stronger on the 8B model) or ``"wanda"`` (activation-weighted magnitude, requires a calibration
            pass via `SupraModel.calibrate_saliency(dataset)`). Defaults to ``"magnitude"``.
        selection_direction (`str`):
            Which end of the score to keep as trainable support. ``"bottom"`` picks the lowest-score entries (paper's
            BottomK, used in all Supra rows of Tables 1-2); ``"top"`` picks the highest-score entries. Defaults to
            ``"bottom"`` — the paper's Supra rows are all BottomK.
        init_weights (`bool`):
            Initialize the sparse `values` to zero (identity update at init). Defaults to `True`. LoRA `A` is always
            Kaiming-uniform and `B` is always zero (standard LoRA init); this flag does not affect them.

    Paper: https://arxiv.org/abs/2607.09287 (Eq. 14, Tables 1-2).
    Reference impl: https://github.com/vectozavr/SuperTuning/blob/main/dense_plus_sparse_linear_plus_lora.py
    """

    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex to replace with Supra layers. Same convention as LoraConfig."
            ),
        },
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Extra modules to mark trainable + saved (e.g. classifier heads)."},
    )
    r: int = field(
        default=8,
        metadata={
            "help": (
                "Rank-equivalent budget. Total trainable scalar count per layer is r · (in + out), "
                "matching vanilla LoRA at rank r."
            ),
        },
    )
    lora_ratio: float = field(
        default=0.5,
        metadata={
            "help": (
                "Fraction of budget to LoRA. 0.0 → pure Super, 1.0 → pure LoRA. "
                "Paper sweeps {0.3, 0.5, 0.8}; best is model-dependent."
            ),
        },
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA scaling factor. Effective scaling = lora_alpha / r_lora."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout on the LoRA input path."},
    )
    scoring_method: str = field(
        default="magnitude",
        metadata={
            "help": (
                "'magnitude' (no calibration; stronger on 8B per paper) or 'wanda' "
                "(activation-weighted; requires calibrate_saliency pass)."
            ),
        },
    )
    selection_direction: str = field(
        default="bottom",
        metadata={
            "help": (
                "'bottom' (least-salient; paper's BottomK, used in all Supra rows) or 'top' (most-salient)."
            ),
        },
    )
    init_weights: bool = field(
        default=True,
        metadata={"help": "Initialize sparse values to zero (identity update)."},
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.SUPRA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        if self.r < 1:
            raise ValueError(f"r must be >= 1, got {self.r}")
        if not 0.0 <= self.lora_ratio <= 1.0:
            raise ValueError(f"lora_ratio must be in [0.0, 1.0], got {self.lora_ratio}")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError(f"lora_dropout must be in [0.0, 1.0), got {self.lora_dropout}")
        if self.scoring_method not in ("magnitude", "wanda"):
            raise ValueError(f"scoring_method must be 'magnitude' or 'wanda', got {self.scoring_method}")
        if self.selection_direction not in ("top", "bottom"):
            raise ValueError(f"selection_direction must be 'top' or 'bottom', got {self.selection_direction}")
