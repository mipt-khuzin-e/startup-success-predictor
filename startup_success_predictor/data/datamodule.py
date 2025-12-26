"""PyTorch Lightning DataModule for startup data using Polars."""

from pathlib import Path
from typing import Any

import polars as pl
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset

from startup_success_predictor.data.preprocessing import (
    encode_categorical,
    handle_missing_values,
    normalize_features,
    polars_to_tensor,
    temporal_split,
)


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

        # Will be set during setup
        self.train_dataset: Dataset[Any] | None = None
        self.val_dataset: Dataset[Any] | None = None
        self.test_dataset: Dataset[Any] | None = None
        self.feature_cols: list[str] = []
        self.normalization_stats: dict[str, dict[str, float]] = {}
        self.encoding_meta: dict[str, Any] = {}

        # Set random seed
        torch.manual_seed(random_seed)

    def prepare_data(self) -> None:
        """Download data if needed (called only on 1 GPU/TPU)."""
        # Data should already be downloaded via download.py
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

    def setup(self, stage: str | None = None) -> None:
        """
        Setup datasets for training, validation, and testing.

        Args:
            stage: Current stage ('fit', 'validate', 'test', or None)
        """
        # Load data
        df = pl.read_csv(self.data_path)

        # Handle missing values
        df = handle_missing_values(df, strategy="drop")

        # Create binary target: 1 for success (acquired, ipo, operating), 0 for closed
        # This depends on the actual dataset structure
        if self.target_col in df.columns:
            # Map status to binary
            df = df.with_columns(
                pl.when(pl.col(self.target_col).is_in(["acquired", "ipo", "operating"]))
                .then(1)
                .otherwise(0)
                .alias("target")
            )
        else:
            raise ValueError(f"Target column '{self.target_col}' not found in data")

        # Temporal split
        train_df, val_df, test_df = temporal_split(df, self.date_col, self.train_end, self.val_end)

        # Get feature columns (exclude target, date, and non-numeric)
        exclude_cols = {self.target_col, "target", self.date_col}
        all_cols = set(df.columns)
        potential_feature_cols = list(all_cols - exclude_cols)

        # Separate numerical and categorical
        numerical_cols = [
            col
            for col in potential_feature_cols
            if col not in self.categorical_cols
            and df[col].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]
        ]

        # Encode categorical variables (on training data)
        if self.categorical_cols:
            train_df, self.encoding_meta = encode_categorical(
                train_df, self.categorical_cols, method="label"
            )
            # Apply same encoding to val and test
            for col in self.categorical_cols:
                if self.encoding_meta["method"] == "label":
                    mapping = self.encoding_meta[col]["mapping"]
                    val_df = val_df.with_columns(
                        pl.col(col).replace_strict(mapping, default=None).alias(col)
                    )
                    test_df = test_df.with_columns(
                        pl.col(col).replace_strict(mapping, default=None).alias(col)
                    )

        # Normalize numerical features (fit on training data)
        if numerical_cols:
            train_df, self.normalization_stats = normalize_features(train_df, numerical_cols)
            val_df, _ = normalize_features(val_df, numerical_cols, self.normalization_stats)
            test_df, _ = normalize_features(test_df, numerical_cols, self.normalization_stats)

        # Store feature columns
        self.feature_cols = numerical_cols + self.categorical_cols

        # Convert to tensors
        X_train = polars_to_tensor(train_df, self.feature_cols)
        y_train = polars_to_tensor(train_df, ["target"]).squeeze()

        X_val = polars_to_tensor(val_df, self.feature_cols)
        y_val = polars_to_tensor(val_df, ["target"]).squeeze()

        X_test = polars_to_tensor(test_df, self.feature_cols)
        y_test = polars_to_tensor(test_df, ["target"]).squeeze()

        # Create datasets
        self.train_dataset = TensorDataset(X_train, y_train)
        self.val_dataset = TensorDataset(X_val, y_val)
        self.test_dataset = TensorDataset(X_test, y_test)

    def train_dataloader(self) -> DataLoader[Any]:
        """Return training DataLoader."""
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is None. Call setup() first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return validation DataLoader."""
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is None. Call setup() first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return test DataLoader."""
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is None. Call setup() first.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def get_minority_class_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get training data for minority class (successful startups).

        Returns:
            Tuple of (features, labels) for minority class
        """
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is None. Call setup() first.")

        X_train, y_train = self.train_dataset.tensors
        minority_mask = y_train == 1
        return X_train[minority_mask], y_train[minority_mask]
