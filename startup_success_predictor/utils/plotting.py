"""Plotting utilities for training metrics visualization."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torchmetrics import AUROC, F1Score, Recall
from torchmetrics.classification import BinaryAveragePrecision

logger = logging.getLogger(__name__)


def plot_classification_metrics(
    labels: Tensor,
    probabilities: Tensor,
    plots_dir: Path,
) -> None:
    """Generate and save classification metric plots.

    Produces at least 3 plots:
    1. Probability distribution histogram
    2. Metrics bar chart (AUROC, AUPRC, F1, Recall)
    3. Precision-Recall style threshold analysis

    Args:
        labels: True binary labels (int tensor).
        probabilities: Predicted probabilities.
        plots_dir: Directory to save plots.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    labels_int = labels.int()
    probs_np = probabilities.numpy()
    labels_np = labels.numpy()

    # --- Plot 1: Probability distribution by class ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        probs_np[labels_np == 0],
        bins=50,
        alpha=0.6,
        label="Negative (closed)",
        color="steelblue",
    )
    ax.hist(
        probs_np[labels_np == 1],
        bins=50,
        alpha=0.6,
        label="Positive (success)",
        color="coral",
    )
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Probability Distribution by Class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "probability_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved probability_distribution.png")

    # --- Plot 2: Summary metrics bar chart ---
    auroc = float(AUROC(task="binary")(probabilities, labels_int))
    auprc = float(BinaryAveragePrecision()(probabilities, labels_int))
    f1 = float(F1Score(task="binary")(probabilities, labels_int))
    recall = float(Recall(task="binary")(probabilities, labels_int))

    metric_names = ["AUROC", "AUPRC", "F1", "Recall"]
    metric_values = [auroc, auprc, f1, recall]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        metric_names, metric_values, color=["#4c72b0", "#55a868", "#c44e52", "#8172b3"]
    )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Test Set Classification Metrics")
    for bar, value in zip(bars, metric_values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.3f}",
            ha="center",
            fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(plots_dir / "classification_metrics.png", dpi=150)
    plt.close(fig)
    logger.info("Saved classification_metrics.png")

    # --- Plot 3: Metrics vs threshold ---
    thresholds = torch.linspace(0.1, 0.9, 17)
    f1_scores = []
    recall_scores = []
    precision_scores = []

    for threshold in thresholds:
        preds = (probabilities > threshold).int()
        tp = ((preds == 1) & (labels_int == 1)).sum().float()
        fp = ((preds == 1) & (labels_int == 0)).sum().float()
        fn = ((preds == 0) & (labels_int == 1)).sum().float()

        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1_val = 2 * prec * rec / (prec + rec + 1e-8)

        precision_scores.append(float(prec))
        recall_scores.append(float(rec))
        f1_scores.append(float(f1_val))

    thresholds_np = thresholds.numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        thresholds_np, precision_scores, marker="o", label="Precision", markersize=3
    )
    ax.plot(thresholds_np, recall_scores, marker="s", label="Recall", markersize=3)
    ax.plot(thresholds_np, f1_scores, marker="^", label="F1 Score", markersize=3)
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Metrics vs Classification Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "metrics_vs_threshold.png", dpi=150)
    plt.close(fig)
    logger.info("Saved metrics_vs_threshold.png")
