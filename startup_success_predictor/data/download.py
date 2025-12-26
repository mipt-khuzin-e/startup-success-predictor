"""Download and manage startup investment data from Kaggle."""

import os
from pathlib import Path

import polars as pl
from kaggle.api.kaggle_api_extended import KaggleApi

from startup_success_predictor.config import get_settings


def setup_kaggle_credentials() -> None:
    """Setup Kaggle credentials from environment variables."""
    settings = get_settings()

    # Set environment variables for Kaggle API
    os.environ["KAGGLE_USERNAME"] = settings.kaggle_username
    os.environ["KAGGLE_KEY"] = settings.kaggle_key


def download_startup_data(output_dir: Path | None = None) -> None:
    """
    Download startup investment dataset from Kaggle.

    Args:
        output_dir: Directory to save downloaded data. Defaults to settings.raw_data_dir.
    """
    settings = get_settings()
    if output_dir is None:
        output_dir = settings.raw_data_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup credentials
    setup_kaggle_credentials()

    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Download dataset
    # Dataset: https://www.kaggle.com/datasets/yanmaksi/big-startup-secsees-fail-dataset-from-crunchbase
    dataset_slug = "yanmaksi/big-startup-secsees-fail-dataset-from-crunchbase"

    print(f"Downloading dataset: {dataset_slug}")
    api.dataset_download_files(dataset_slug, path=str(output_dir), unzip=True)
    print(f"Dataset downloaded to: {output_dir}")


def validate_data(data_dir: Path | None = None) -> bool:
    """
    Validate that required data files exist.

    Args:
        data_dir: Directory containing data files. Defaults to settings.raw_data_dir.

    Returns:
        True if validation passes, False otherwise.
    """
    settings = get_settings()
    if data_dir is None:
        data_dir = settings.raw_data_dir

    # Check for CSV files
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return False

    print(f"Found {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        df = pl.read_csv(csv_file)
        print(f"  - {csv_file.name}: {df.shape[0]} rows, {df.shape[1]} columns")

    return True


def main() -> None:
    """Main entry point for data download."""
    settings = get_settings()

    print("Starting data download...")
    download_startup_data()

    print("\nValidating downloaded data...")
    if validate_data():
        print("Data validation successful!")
    else:
        print("Data validation failed!")
        raise RuntimeError("Data validation failed")

    print("\nNext steps:")
    print(f"1. Add data to DVC: dvc add {settings.raw_data_dir}")
    print(f"2. Commit changes: git add {settings.raw_data_dir}.dvc .gitignore")


if __name__ == "__main__":
    main()
