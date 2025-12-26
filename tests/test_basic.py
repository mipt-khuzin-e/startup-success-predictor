"""Basic smoke tests for core components.

These tests are intentionally lightweight and operate on tiny synthetic data
so they can run quickly as part of CI or local checks.
"""

from pathlib import Path

import polars as pl
import torch

from startup_success_predictor.data.datamodule import StartupDataModule
from startup_success_predictor.data.preprocessing import (
    encode_categorical,
    handle_missing_values,
    normalize_features,
    polars_to_tensor,
    temporal_split,
)
from startup_success_predictor.models.classifier_module import ClassifierModule
from startup_success_predictor.models.gan_module import WGANGPModule


def test_preprocessing_utils_roundtrip() -> None:
    df = pl.DataFrame(
        {
            "founded_at": ["2010-01-01", "2016-05-01", "2019-07-01"],
            "status": ["operating", "closed", "acquired"],
            "country_code": ["US", "US", "GB"],
            "feature": [1.0, 2.0, 3.0],
        }
    )

    df = handle_missing_values(df, strategy="drop")
    train_df, _, _ = temporal_split(df, "founded_at", "2015-01-01", "2018-01-01")

    encoded_train, meta = encode_categorical(train_df, ["country_code"], method="label")
    assert "country_code" in encoded_train.columns
    assert meta["method"] == "label"

    norm_train, stats = normalize_features(encoded_train, ["feature"])
    assert "feature" in stats

    tensor = polars_to_tensor(norm_train, ["feature"])
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape[1] == 1


def test_datamodule_setup_tmp(tmp_path: Path) -> None:
    data_path = tmp_path / "data.csv"
    df = pl.DataFrame(
        {
            "founded_at": ["2010-01-01", "2016-05-01", "2019-07-01"],
            "status": ["operating", "closed", "acquired"],
            "country_code": ["US", "US", "GB"],
            "feature": [1.0, 2.0, 3.0],
        }
    )
    df.write_csv(data_path)

    dm = StartupDataModule(
        data_path=data_path,
        batch_size=2,
        num_workers=0,
        train_end="2015-01-01",
        val_end="2018-01-01",
        target_col="status",
        date_col="founded_at",
        categorical_cols=["country_code"],
        random_seed=30,
        handle_missing="drop",
        encoding_method="label",
    )

    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    assert len(list(train_loader)) >= 1
    assert len(list(val_loader)) >= 0
    assert len(list(test_loader)) >= 0


def test_models_forward_pass() -> None:
    input_dim = 4
    batch_size = 2

    classifier = ClassifierModule(input_dim=input_dim, hidden_dims=[8, 4])
    x = torch.randn(batch_size, input_dim)
    logits = classifier(x)
    assert logits.shape == (batch_size, 1)

    gan = WGANGPModule(
        input_dim=input_dim,
        latent_dim=2,
        generator_hidden_dims=[4],
        critic_hidden_dims=[4],
    )
    z = torch.randn(batch_size, 2)
    samples = gan(z)
    assert samples.shape == (batch_size, input_dim)
