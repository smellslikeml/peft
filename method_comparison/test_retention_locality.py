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

import math

import pandas as pd
import pytest

from . import processing
from .retention_locality import (
    LOCALITY_SCORE_COLUMN,
    compute_locality_scores,
    get_locality_metrics,
)


def _metamathqa_df(rows):
    """Build a minimal MetaMathQA-style frame with the columns the score needs."""
    return pd.DataFrame(
        [{"experiment_name": name, "test_accuracy": acc, "forgetting*": forget} for name, acc, forget in rows]
    )


def test_score_registered_in_processing_machinery():
    # The locality column must flow through the same plumbing as native metrics:
    # metric preferences (used by the Pareto frontier) and ordered task columns.
    prefs = processing.get_metric_preferences("MetaMathQA")
    assert prefs[LOCALITY_SCORE_COLUMN] == "higher"

    columns = processing.get_task_columns("MetaMathQA")
    assert LOCALITY_SCORE_COLUMN in columns

    img_prefs = processing.get_metric_preferences("image-gen")
    assert img_prefs[LOCALITY_SCORE_COLUMN] == "higher"


def test_add_locality_scores_is_wired_into_processing():
    # processing re-exports the entry point it calls in load_df; exercising it here
    # proves the wiring rather than just the standalone module.
    df = _metamathqa_df([("a", 0.8, 0.1), ("b", 0.8, 0.5)])
    scored = processing.add_locality_scores(df, "MetaMathQA")

    assert LOCALITY_SCORE_COLUMN in scored.columns
    # the original frame is left untouched (function copies)
    assert LOCALITY_SCORE_COLUMN not in df.columns


def test_low_retain_perturbation_scores_higher():
    # Same task quality, but "good" perturbs the retain set far less -> higher
    # locality, which is exactly the property NSRU's null-space projection buys.
    df = _metamathqa_df([("good", 0.8, 0.1), ("bad", 0.8, 0.6)])
    scores = compute_locality_scores(df, "MetaMathQA")
    assert scores.loc[0] > scores.loc[1]


def test_score_rewards_both_axes():
    df = _metamathqa_df([("best", 0.9, 0.05), ("mid", 0.9, 0.5), ("weak", 0.3, 0.05)])
    scores = compute_locality_scores(df, "MetaMathQA")
    # strong on both axes beats being strong on only one of them
    assert scores.loc[0] > scores.loc[1]
    assert scores.loc[0] > scores.loc[2]
    # harmonic mean stays within the normalized [0, 1] range
    assert ((scores >= 0) & (scores <= 1)).all()


def test_missing_perturbation_sentinel_is_not_scored():
    # processing.py fills the sentinel 123 when a run lacks the retain metric.
    df = _metamathqa_df([("real", 0.8, 0.1), ("nometric", 0.8, 123)])
    scores = compute_locality_scores(df, "MetaMathQA")
    assert math.isnan(scores.loc[1])
    assert not math.isnan(scores.loc[0])


def test_unknown_task_has_no_locality_metrics():
    assert get_locality_metrics("MetaMathQA") == ("test_accuracy", "forgetting*")
    assert get_locality_metrics("image-gen") == ("test_dino_similarity", "drift*")
    assert get_locality_metrics("does-not-exist") is None


@pytest.mark.parametrize("task_name", ["MetaMathQA", "image-gen"])
def test_score_defined_for_both_benchmark_tasks(task_name):
    quality_col, perturb_col = get_locality_metrics(task_name)
    df = pd.DataFrame(
        [
            {quality_col: 0.9, perturb_col: 0.1},
            {quality_col: 0.5, perturb_col: 0.4},
        ]
    )
    scores = compute_locality_scores(df, task_name)
    assert scores.notna().all()
    assert scores.loc[0] > scores.loc[1]
