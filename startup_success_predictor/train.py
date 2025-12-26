"""Training pipeline for startup success prediction."""

import logging

import hydra
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from startup_success_predictor.config import get_settings
from startup_success_predictor.data.datamodule import StartupDataModule
from startup_success_predictor.models.classifier_module import ClassifierModule
from startup_success_predictor.models.gan_module import WGANGPModule

logger = logging.getLogger(__name__)


def get_git_commit_id() -> str:
    """Get current git commit ID."""
    try:
        import subprocess

        commit_id = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
        return commit_id
    except Exception:
        return "unknown"


def train_gan(
    cfg: DictConfig,
    datamodule: StartupDataModule,
    mlflow_logger: MLFlowLogger,
) -> WGANGPModule:
    """
    Train WGAN-GP on minority class data.

    Args:
        cfg: Hydra configuration
        datamodule: Data module
        mlflow_logger: MLFlow logger

    Returns:
        Trained WGAN-GP module
    """
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: Training WGAN-GP")
    logger.info("=" * 80)

    # Get minority class data
    X_minority, y_minority = datamodule.get_minority_class_data()
    logger.info("Minority class samples: %d", X_minority.shape[0])

    # Create dataset for minority class
    minority_dataset = torch.utils.data.TensorDataset(X_minority, y_minority)
    minority_loader = torch.utils.data.DataLoader(
        minority_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )

    # Initialize WGAN-GP
    input_dim = X_minority.shape[1]
    gan_model = WGANGPModule(
        input_dim=input_dim,
        latent_dim=cfg.model.generator.latent_dim,
        generator_hidden_dims=cfg.model.generator.hidden_dims,
        critic_hidden_dims=cfg.model.critic.hidden_dims,
        leaky_relu_slope=cfg.model.generator.leaky_relu_slope,
        use_batch_norm=cfg.model.generator.use_batch_norm,
        dropout=cfg.model.critic.dropout,
        learning_rate=cfg.model.training.learning_rate,
        beta1=cfg.model.training.beta1,
        beta2=cfg.model.training.beta2,
        n_critic=cfg.model.training.n_critic,
        lambda_gp=cfg.model.training.lambda_gp,
    )

    # Setup trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.paths.models_dir + "/gan",
        filename="gan-{epoch:02d}",
        save_top_k=1,
        monitor="train_generator_loss",
        mode="min",
    )

    trainer = Trainer(
        max_epochs=cfg.train.gan_training.max_epochs,
        accelerator=cfg.train.trainer.accelerator,
        devices=cfg.train.trainer.devices,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=cfg.train.trainer.log_every_n_steps,
    )

    # Train
    trainer.fit(gan_model, minority_loader)

    logger.info(
        "GAN training completed. Best model saved to: %s",
        checkpoint_callback.best_model_path,
    )

    return gan_model


def generate_synthetic_data(
    gan_model: WGANGPModule,
    n_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic minority class samples.

    Args:
        gan_model: Trained WGAN-GP model
        n_samples: Number of samples to generate

    Returns:
        Tuple of (synthetic_features, synthetic_labels)
    """
    logger.info("\nGenerating %d synthetic samples...", n_samples)
    synthetic_features = gan_model.generate_samples(n_samples).cpu()
    synthetic_labels = torch.ones(n_samples)  # All synthetic samples are positive class

    logger.info("Generated synthetic data shape: %s", synthetic_features.shape)
    return synthetic_features, synthetic_labels


def train_classifier(
    cfg: DictConfig,
    datamodule: StartupDataModule,
    synthetic_features: torch.Tensor | None,
    synthetic_labels: torch.Tensor | None,
    mlflow_logger: MLFlowLogger,
) -> ClassifierModule:
    """
    Train classifier on augmented data.

    Args:
        cfg: Hydra configuration
        datamodule: Data module
        synthetic_features: Synthetic features (optional)
        synthetic_labels: Synthetic labels (optional)
        mlflow_logger: MLFlow logger

    Returns:
        Trained classifier module
    """
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: Training Classifier")
    logger.info("=" * 80)

    # Augment training data if synthetic data is provided
    if synthetic_features is not None and synthetic_labels is not None:
        logger.info("Augmenting training data with synthetic samples...")
        X_train, y_train = datamodule.train_dataset.tensors  # type: ignore[union-attr]

        # Concatenate real and synthetic data
        X_train_aug = torch.cat([X_train, synthetic_features], dim=0)
        y_train_aug = torch.cat([y_train, synthetic_labels], dim=0)

        logger.info("Original training size: %d", X_train.shape[0])
        logger.info("Augmented training size: %d", X_train_aug.shape[0])

        # Create augmented dataset
        train_dataset_aug = torch.utils.data.TensorDataset(X_train_aug, y_train_aug)
        train_loader = torch.utils.data.DataLoader(
            train_dataset_aug,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
        )
    else:
        logger.info("Training without synthetic data augmentation...")
        train_loader = datamodule.train_dataloader()

    # Initialize classifier
    input_dim = datamodule.feature_cols.__len__()
    classifier_model = ClassifierModule(
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

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.paths.models_dir + "/classifier",
        filename=cfg.train.checkpoint.filename,
        monitor=cfg.train.checkpoint.monitor,
        mode=cfg.train.checkpoint.mode,
        save_top_k=cfg.train.checkpoint.save_top_k,
        save_last=cfg.train.checkpoint.save_last,
    )

    early_stopping_callback = EarlyStopping(
        monitor=cfg.train.early_stopping.monitor,
        patience=cfg.train.early_stopping.patience,
        mode=cfg.train.early_stopping.mode,
        min_delta=cfg.train.early_stopping.min_delta,
    )

    # Setup trainer
    trainer = Trainer(
        max_epochs=cfg.train.trainer.max_epochs,
        accelerator=cfg.train.trainer.accelerator,
        devices=cfg.train.trainer.devices,
        precision=cfg.train.trainer.precision,
        gradient_clip_val=cfg.train.trainer.gradient_clip_val,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=cfg.train.trainer.log_every_n_steps,
        check_val_every_n_epoch=cfg.train.trainer.check_val_every_n_epoch,
    )

    # Train
    trainer.fit(classifier_model, train_loader, datamodule.val_dataloader())

    logger.info("\nClassifier training completed.")
    logger.info("Best model saved to: %s", checkpoint_callback.best_model_path)

    # Test
    logger.info("\nEvaluating on test set...")
    trainer.test(classifier_model, datamodule.test_dataloader())

    return classifier_model


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training pipeline.

    Args:
        cfg: Hydra configuration
    """
    logger.info("=" * 80)
    logger.info("Startup Success Predictor - Training Pipeline")
    logger.info("=" * 80)
    logger.info("\nConfiguration:\n%s", OmegaConf.to_yaml(cfg))

    # Set random seed
    torch.manual_seed(cfg.seed)

    # Get settings
    settings = get_settings()

    # Setup MLFlow logger
    mlflow_logger: MLFlowLogger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
        run_name=cfg.mlflow.run_name,
    )

    # Log git commit
    git_commit = get_git_commit_id()
    mlflow_logger.log_hyperparams({"git_commit": git_commit})

    # Log all config
    mlflow_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]

    # Find data file
    data_path = settings.raw_data_dir / cfg.data.data_file
    if not data_path.exists():
        # Try to find any CSV file
        csv_files = list(settings.raw_data_dir.glob("*.csv"))
        if csv_files:
            data_path = csv_files[0]
            logger.info("Using data file: %s", data_path)
        else:
            raise FileNotFoundError(f"No data file found in {settings.raw_data_dir}")

    # Initialize data module
    datamodule = StartupDataModule(
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

    # Setup data
    datamodule.setup()

    # Two-stage training
    synthetic_features = None
    synthetic_labels = None

    if cfg.train.two_stage.train_gan_first:
        # Stage 1: Train GAN
        gan_model = train_gan(cfg, datamodule, mlflow_logger)

        # Generate synthetic data
        if cfg.train.two_stage.use_synthetic_data:
            synthetic_features, synthetic_labels = generate_synthetic_data(
                gan_model,
                cfg.model.generation.n_samples,
            )

    # Stage 2: Train Classifier
    logger.info("Starting classifier training phase")
    train_classifier(
        cfg,
        datamodule,
        synthetic_features,
        synthetic_labels,
        mlflow_logger,
    )

    logger.info("\n" + "=" * 80)
    logger.info("Training pipeline completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
