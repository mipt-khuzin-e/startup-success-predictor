"""Configuration management using Pydantic Settings."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Kaggle credentials
    kaggle_username: str
    kaggle_key: str

    # MLFlow
    mlflow_tracking_uri: str = "http://127.0.0.1:8080"

    # Reproducibility
    random_seed: int = 30

    # Paths
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    models_dir: Path = project_root / "models"
    plots_dir: Path = project_root / "plots"


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()
