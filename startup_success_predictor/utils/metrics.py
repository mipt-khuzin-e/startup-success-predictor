"""Metrics utilities for evaluation.

Currently contains Precision@k for binary classification.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def precision_at_k(y_true: Sequence[int], y_scores: Sequence[float], k: int) -> float:
    """Compute Precision@k for binary classification.

    Args:
        y_true: True binary labels (0/1).
        y_scores: Predicted scores (higher means more likely positive).
        k: Number of top elements to consider.

    Returns:
        Precision@k value in [0, 1]. If k == 0, returns 0.0.
    """
    if k <= 0:
        return 0.0

    y_true_arr = np.asarray(y_true)
    y_scores_arr = np.asarray(y_scores)

    if y_true_arr.shape[0] == 0:
        return 0.0

    k = min(k, y_true_arr.shape[0])
    # Indices of samples sorted by score descending
    topk_idx = np.argsort(-y_scores_arr)[:k]
    topk_true = y_true_arr[topk_idx]
    return float(topk_true.mean())
