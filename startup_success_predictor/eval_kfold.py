"""Stratified K-fold evaluation for the startup success classifier.

This script reuses the existing Hydra config, data module, and classifier
to run a 5-fold stratified evaluation on the training set and compute
additional metrics such as Precision@k.

Usage (example):
    python -m startup_success_predictor.eval_kfold \
      train.trainer.max_epochs=5

By default, it runs 5-fold stratified evaluation with k in {100, 500}.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from lightning import Trainer
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import AUROC, F1Score, Precision, Recall
from torchmetrics.classification import BinaryAveragePrecision

from startup_success_predictor.data.datamodule import StartupDataModule
from startup_success_predictor.data.download import resolve_data_path
from startup_success_predictor.models.classifier_module import ClassifierModule
from startup_success_predictor.utils.metrics import precision_at_k

logger = logging.getLogger(__name__)


def _split_class(idxs: np.ndarray, n_splits: int) -> list[np.ndarray]:
    """Split class indices into K folds."""
    folds: list[np.ndarray] = []
    fold_sizes = np.full(n_splits, len(idxs) // n_splits, dtype=int)
    fold_sizes[: len(idxs) % n_splits] += 1
    current = 0
    for fold_size in fold_sizes:
        folds.append(idxs[current : current + fold_size])
        current += fold_size
    return folds


def _stratified_kfold_indices(
    y: np.ndarray, n_splits: int, random_seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create stratified K-fold indices for binary labels."""
    if n_splits < 2:
        msg = "n_splits must be at least 2"
        raise ValueError(msg)

    rng = np.random.default_rng(random_seed)
    y = y.astype(int)

    # Indices per class
    indices_pos = np.where(y == 1)[0]
    indices_neg = np.where(y == 0)[0]

    rng.shuffle(indices_pos)
    rng.shuffle(indices_neg)

    pos_folds = _split_class(indices_pos, n_splits)
    neg_folds = _split_class(indices_neg, n_splits)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_idx in range(n_splits):
        val_pos = pos_folds[fold_idx]
        val_neg = neg_folds[fold_idx]
        val_idx = np.concatenate([val_pos, val_neg])

        train_pos = np.concatenate(
            [pos_folds[i] for i in range(n_splits) if i != fold_idx]
        )
        train_neg = np.concatenate(
            [neg_folds[i] for i in range(n_splits) if i != fold_idx]
        )
        train_idx = np.concatenate([train_pos, train_neg])

        rng.shuffle(val_idx)
        rng.shuffle(train_idx)
        folds.append((train_idx, val_idx))

    return folds


def _build_classifier(cfg: DictConfig, input_dim: int) -> ClassifierModule:
    """Construct a ClassifierModule from Hydra config."""
    return ClassifierModule(
        input_dim=input_dim,
        hidden_dims=cfg.model.architecture.hidden_dims,
        output_dim=cfg.model.architecture.output_dim,
        leaky_relu_slope=cfg.model.architecture.leaky_relu_slope,
        use_batch_norm=cfg.model.architecture.use_batch_norm,
        dropout=cfg.model.architecture.dropout,
        learning_rate=cfg.model.training.learning_rate,
        weight_decay=cfg.model.training.weight_decay,
        beta1=cfg.model.training.beta1,
        beta2=cfg.model.training.beta2,
        pos_weight=cfg.model.loss.pos_weight,
    )


def _evaluate_fold(
    model: ClassifierModule,
    val_loader: DataLoader[Any],
) -> dict[str, float]:
    """Evaluate a trained model on a validation loader and compute metrics."""
    model.eval()
    all_probs: list[Tensor] = []
    all_targets: list[Tensor] = []

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            probs_batch = model.predict_proba(x_batch).squeeze(-1)
            all_probs.append(probs_batch.cpu())
            all_targets.append(y_batch.cpu())

    probs = torch.cat(all_probs)
    targets = torch.cat(all_targets).int()

    auroc_metric = AUROC(task="binary")
    auprc_metric = BinaryAveragePrecision()
    f1_metric = F1Score(task="binary")
    precision_metric = Precision(task="binary")
    recall_metric = Recall(task="binary")

    auroc = float(auroc_metric(probs, targets))
    auprc = float(auprc_metric(probs, targets))
    f1 = float(f1_metric(probs, targets))
    precision = float(precision_metric(probs, targets))
    recall = float(recall_metric(probs, targets))

    y_np = targets.numpy().astype(int).tolist()
    probs_np = probs.numpy().astype(float).tolist()

    p_at_100 = precision_at_k(y_np, probs_np, 100)
    p_at_500 = precision_at_k(y_np, probs_np, 500)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "precision_at_100": p_at_100,
        "precision_at_500": p_at_500,
    }


def _aggregate_metrics(
    metrics_per_fold: list[dict[str, float]],
) -> dict[str, tuple[float, float]]:
    """Compute mean and std for each metric over folds."""
    keys = metrics_per_fold[0].keys()
    summary: dict[str, tuple[float, float]] = {}
    for key in keys:
        values = np.array([m[key] for m in metrics_per_fold], dtype=float)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        summary[key] = (mean, std)
    return summary


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parent.parent / "configs"),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Run stratified 5-fold evaluation using the current Hydra config.

    The evaluation is performed on the training split of the temporal split
    (pre-2015 data) to estimate the stability of the classifier under
    different train/validation slices.
    """
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting stratified 5-fold evaluation")

    # Reuse existing DataModule to ensure identical preprocessing
    data_path = resolve_data_path(cfg)
    datamodule = StartupDataModule.from_hydra_config(cfg, data_path)

    datamodule.setup()
    if datamodule.train_dataset is None:
        msg = "train_dataset is None after setup()"
        raise RuntimeError(msg)

    features_all, labels_all = datamodule.train_dataset.tensors  # type: ignore[attr-defined]
    labels_np = labels_all.cpu().numpy()

    n_splits = 5
    folds = _stratified_kfold_indices(labels_np, n_splits=n_splits, random_seed=cfg.seed)

    input_dim = features_all.shape[1]
    metrics_per_fold: list[dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        logger.info(
            "Fold %d/%d: train=%d, val=%d",
            fold_idx + 1,
            n_splits,
            len(train_idx),
            len(val_idx),
        )

        features_train = features_all[train_idx]
        labels_train = labels_all[train_idx]
        features_val = features_all[val_idx]
        labels_val = labels_all[val_idx]

        train_dataset = TensorDataset(features_train, labels_train)
        val_dataset = TensorDataset(features_val, labels_val)

        train_loader: DataLoader[Any] = DataLoader(
            train_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
        )
        val_loader: DataLoader[Any] = DataLoader(
            val_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
        )

        model = _build_classifier(cfg, input_dim=input_dim)

        trainer = Trainer(
            max_epochs=cfg.train.trainer.max_epochs,
            accelerator=cfg.train.trainer.accelerator,
            devices=cfg.train.trainer.devices,
            precision=cfg.train.trainer.precision,
            gradient_clip_val=cfg.train.trainer.gradient_clip_val,
            log_every_n_steps=cfg.train.trainer.log_every_n_steps,
        )

        trainer.fit(model, train_loader, val_loader)

        fold_metrics = _evaluate_fold(model, val_loader)
        metrics_per_fold.append(fold_metrics)

        logger.info("Fold %d metrics:", fold_idx + 1)
        for key, value in fold_metrics.items():
            logger.info("  %s: %.4f", key, value)

    summary = _aggregate_metrics(metrics_per_fold)
    logger.info("\nStratified 5-fold summary (mean ± std):")
    for key, (mean, std) in summary.items():
        logger.info("  %s: %.4f ± %.4f", key, mean, std)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
