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

"""Retention-aware locality scoring for the method-comparison suite.

Adapted from "Null-Space Constrained Low-Rank Adaptation for Response-Specified
Large Language Model Unlearning" (NSRU, https://arxiv.org/abs/2606.10989).

NSRU's central claim is that a well-behaved adaptation confines its updates to
the null space of a *retain subspace*: it should reach the task objective
(adapting / unlearning the target behaviour) while perturbing benign, retained
capabilities as little as possible. The method-comparison benchmarks already
record both sides of this trade-off for each run:

* a task-quality metric (``test_accuracy`` for MetaMathQA,
  ``test_dino_similarity`` for image generation), and
* a retain-side perturbation metric (``forgetting*`` / ``drift*``) measuring how
  much the model degraded on unrelated, benign data during training.

This module collapses that pair into a single per-run *locality score*, so the
comparison suite can rank methods by how localized / retention-preserving their
adaptation is -- the property NSRU optimizes -- instead of by raw task quality
alone. A method that buys quality by heavily perturbing the retain set scores
lower than one that reaches comparable quality with little perturbation.
"""

import pandas as pd


# Sentinel written by processing.py when a run lacks the retain-side perturbation
# metric; such runs cannot be scored and are left as NaN.
_MISSING_SENTINEL = 123

LOCALITY_SCORE_COLUMN = "locality_score"

# task_name -> (quality column, retain-perturbation column).
# Quality is always "higher is better"; perturbation is always "lower is better".
_TASK_LOCALITY_METRICS = {
    "MetaMathQA": ("test_accuracy", "forgetting*"),
    "image-gen": ("test_dino_similarity", "drift*"),
}


def get_locality_metrics(task_name):
    """Return the ``(quality_column, perturbation_column)`` pair for a task.

    Returns ``None`` for tasks that do not expose a retain-side perturbation
    metric, in which case a locality score is not defined.
    """
    return _TASK_LOCALITY_METRICS.get(task_name)


def _minmax(series):
    """Min-max normalize a numeric series to ``[0, 1]``.

    A constant column carries no ranking information, so it maps to a neutral
    ``0.5``. ``NaN`` entries stay ``NaN``.
    """
    valid = series.dropna()
    if valid.empty:
        return series
    lo, hi = valid.min(), valid.max()
    if hi == lo:
        return series.where(series.isna(), 0.5)
    return (series - lo) / (hi - lo)


def compute_locality_scores(df, task_name):
    """Compute the per-run retention-aware locality score for ``df``.

    Both axes are min-max normalized across the runs in ``df``: task quality
    (higher is better) and retention, defined as ``1 - normalized perturbation``
    (so less retain-side damage is better). The score is their harmonic mean,
    which is high only when a run is good on *both* axes -- mirroring NSRU's goal
    of strong task adaptation together with minimal retain-side perturbation.

    Returns a float ``Series`` aligned to ``df.index``; entries are ``NaN`` where
    the required metrics are missing for that run.
    """
    metrics = get_locality_metrics(task_name)
    if metrics is None:
        return pd.Series([float("nan")] * len(df), index=df.index, dtype=float)

    quality_col, perturb_col = metrics
    if quality_col not in df.columns or perturb_col not in df.columns:
        return pd.Series([float("nan")] * len(df), index=df.index, dtype=float)

    quality = pd.to_numeric(df[quality_col], errors="coerce")
    perturb = pd.to_numeric(df[perturb_col], errors="coerce")
    # the missing-metric sentinel must not distort the normalization scale
    perturb = perturb.where(perturb != _MISSING_SENTINEL)

    q_norm = _minmax(quality)
    retention = 1.0 - _minmax(perturb)

    valid = q_norm.notna() & retention.notna()
    denom = q_norm + retention
    # harmonic mean of the two normalized axes
    score = (2.0 * q_norm * retention) / denom
    # a run that is worst on both axes (denom == 0) scores 0, not NaN ...
    score = score.where(denom != 0, 0.0)
    # ... but runs with a genuinely missing input stay NaN (unscorable)
    score = score.where(valid)
    return score.astype(float)


def add_locality_scores(df, task_name):
    """Return a copy of ``df`` with the locality score attached as a column.

    The column is named :data:`LOCALITY_SCORE_COLUMN`. This is the entry point
    used by ``processing.py`` so the score flows through the same column
    plumbing (dtypes, ordering, metric preferences) as the native metrics.
    """
    df = df.copy()
    df[LOCALITY_SCORE_COLUMN] = compute_locality_scores(df, task_name)
    return df
