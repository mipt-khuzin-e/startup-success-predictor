"""Download and manage startup investment data from Kaggle."""

import logging
import os
from pathlib import Path

import polars as pl

from startup_success_predictor.config import get_settings

logger = logging.getLogger(__name__)


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

    # Import Kaggle API lazily so that simply importing this module does not
    # require Kaggle credentials to be configured.
    from kaggle.api.kaggle_api_extended import KaggleApi

    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Download dataset
    # Dataset: https://www.kaggle.com/datasets/yanmaksi/big-startup-secsees-fail-dataset-from-crunchbase
    dataset_slug = "yanmaksi/big-startup-secsees-fail-dataset-from-crunchbase"

    logger.info("Downloading dataset: %s", dataset_slug)
    api.dataset_download_files(dataset_slug, path=str(output_dir), unzip=True)
    logger.info("Dataset downloaded to: %s", output_dir)


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
        logger.warning("No CSV files found in %s", data_dir)
        return False

    logger.info("Found %d CSV files:", len(csv_files))
    for csv_file in csv_files:
        df = pl.read_csv(csv_file)
        logger.info(
            "  - %s: %d rows, %d columns", csv_file.name, df.shape[0], df.shape[1]
        )

    return True


def main() -> None:
    """Main entry point for data download."""
    settings = get_settings()

    logger.info("Starting data download...")
    download_startup_data()

    logger.info("Validating downloaded data...")
    if validate_data():
        logger.info("Data validation successful!")
    else:
        logger.error("Data validation failed!")
        raise RuntimeError("Data validation failed")

    logger.info("Next steps:")
    logger.info("1. Add data to DVC: dvc add %s", settings.raw_data_dir)
    logger.info("2. Commit changes: git add %s.dvc .gitignore", settings.raw_data_dir)


if __name__ == "__main__":
    main()
