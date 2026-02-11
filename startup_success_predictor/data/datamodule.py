"""PyTorch Lightning DataModule for startup data using Polars."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from omegaconf import DictConfig

from startup_success_predictor.data.dataset_processing import process_startup_dataset


class StartupDataModule(LightningDataModule):
    """DataModule for startup success prediction."""

    def __init__(
        self,
        data_path: Path | str,
        batch_size: int = 32,
        num_workers: int = 4,
        train_end: str = "2015-01-01",
        val_end: str = "2018-01-01",
        target_col: str = "status",
        date_col: str = "founded_at",
        categorical_cols: list[str] | None = None,
        random_seed: int = 30,
        handle_missing: str = "drop",
        encoding_method: str = "label",
    ) -> None:
        """
        Initialize StartupDataModule.

        Args:
            data_path: Path to the CSV data file
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for DataLoader
            train_end: End date for training set
            val_end: End date for validation set
            target_col: Name of target column
            date_col: Name of date column
            categorical_cols: List of categorical column names
            random_seed: Random seed for reproducibility
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_end = train_end
        self.val_end = val_end
        self.target_col = target_col
        self.date_col = date_col
        self.categorical_cols = categorical_cols or []
        self.random_seed = random_seed
        self.handle_missing = handle_missing
        self.encoding_method = encoding_method

        # Will be set during setup
        self.train_dataset: Dataset[Any] | None = None
        self.val_dataset: Dataset[Any] | None = None
        self.test_dataset: Dataset[Any] | None = None
        self.feature_cols: list[str] = []
        self.normalization_stats: dict[str, dict[str, float]] = {}
        self.encoding_meta: dict[str, Any] = {}

        # Set random seed
        torch.manual_seed(random_seed)

    @classmethod
    def from_hydra_config(cls, cfg: DictConfig, data_path: Path) -> StartupDataModule:
        """Create a StartupDataModule from Hydra config and resolved data path."""
        return cls(
            data_path=data_path,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            train_end=cfg.data.train_end,
            val_end=cfg.data.val_end,
            target_col=cfg.data.target_col,
            date_col=cfg.data.date_col,
            categorical_cols=cfg.data.categorical_cols,
            random_seed=cfg.seed,
            handle_missing=cfg.data.handle_missing,
            encoding_method=cfg.data.encoding_method,
        )

    def prepare_data(self) -> None:
        """Download data if needed (called only on 1 GPU/TPU)."""
        # Data should already be downloaded via download.py
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

    def setup(self, stage: str | None = None) -> None:
        """Setup datasets for training, validation, and testing."""
        result = process_startup_dataset(
            data_path=self.data_path,
            train_end=self.train_end,
            val_end=self.val_end,
            target_col=self.target_col,
            date_col=self.date_col,
            categorical_cols=self.categorical_cols,
            handle_missing=self.handle_missing,
            encoding_method=self.encoding_method,
        )
        self.train_dataset = result.train_dataset
        self.val_dataset = result.val_dataset
        self.test_dataset = result.test_dataset
        self.feature_cols = result.feature_cols
        self.normalization_stats = result.normalization_stats
        self.encoding_meta = result.encoding_meta

    def train_dataloader(self) -> DataLoader[Any]:
        """Return training DataLoader."""
        if self.train_dataset is None:
            raise RuntimeError("Dataset is None. Call setup() first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return validation DataLoader."""
        if self.val_dataset is None:
            raise RuntimeError("Dataset is None. Call setup() first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return test DataLoader."""
        if self.test_dataset is None:
            raise RuntimeError("Dataset is None. Call setup() first.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def get_minority_class_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get training data for minority class (successful startups).

        Returns:
            Tuple of (features, labels) for minority class
        """
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is None. Call setup() first.")

        features_train, labels_train = self.train_dataset.tensors  # type: ignore[attr-defined]
        minority_mask = labels_train == 1
        return features_train[minority_mask], labels_train[minority_mask]
