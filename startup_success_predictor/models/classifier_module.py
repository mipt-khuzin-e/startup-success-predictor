"""MLP Classifier Lightning Module."""

from typing import override

import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor
from torch.optim import Adam, Optimizer
from torchmetrics import (
    AUROC,
    Accuracy,
    F1Score,
    Metric,
    MetricCollection,
    Precision,
    Recall,
)
from torchmetrics.classification import BinaryAveragePrecision

from startup_success_predictor.models.components.classifier import MLPClassifier


class ClassifierModule(LightningModule):
    """Lightning Module for MLP Classifier."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        output_dim: int = 1,
        leaky_relu_slope: float = 0.2,
        use_batch_norm: bool = True,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        pos_weight: float = 10.0,
    ) -> None:
        """
        Initialize Classifier Module.

        Args:
            input_dim: Dimension of input features
            hidden_dims: Hidden dimensions for MLP
            output_dim: Output dimension (1 for binary classification)
            leaky_relu_slope: Slope for LeakyReLU
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout probability
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            beta1: Beta1 for Adam optimizer
            beta2: Beta2 for Adam optimizer
            pos_weight: Weight for positive class in loss
        """
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.model = MLPClassifier(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            leaky_relu_slope=leaky_relu_slope,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
        )

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

        # Metrics per stage (keyed with "_metrics" suffix to avoid nn.Module name clash)
        metrics: dict[str, Metric | MetricCollection] = {
            "auroc": AUROC(task="binary"),
            "auprc": BinaryAveragePrecision(),
            "f1": F1Score(task="binary"),
            "precision": Precision(task="binary"),
            "recall": Recall(task="binary"),
            "accuracy": Accuracy(task="binary"),
        }
        self.train_metrics = MetricCollection(metrics, prefix="train_")
        self.val_metrics = MetricCollection(metrics, prefix="val_")
        self.test_metrics = MetricCollection(metrics, prefix="test_")

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input features of shape (batch_size, input_dim)

        Returns:
            Logits of shape (batch_size, 1)
        """
        out = self.model(x)
        assert isinstance(out, Tensor)
        return out

    def _shared_step(self, batch: tuple[Tensor, Tensor], stage: str) -> Tensor:
        """Shared logic for train/val/test steps."""
        x, y = batch
        logits = self(x).squeeze(-1)
        loss = self.criterion(logits, y)
        probs = torch.sigmoid(logits)
        y_int = y.int()

        stage_metrics: MetricCollection = getattr(self, f"{stage}_metrics")
        stage_metrics(probs, y_int)

        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log_dict(stage_metrics, prog_bar=(stage != "test"))

        return loss

    @override
    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Training step."""
        return self._shared_step(batch, "train")

    @override
    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Validation step."""
        self._shared_step(batch, "val")

    @override
    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Test step."""
        self._shared_step(batch, "test")

    @override
    def configure_optimizers(self) -> Optimizer:
        """
        Configure optimizer.

        Returns:
            Optimizer instance
        """
        return Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(self.beta1, self.beta2),
        )

    def predict_proba(self, x: Tensor) -> Tensor:
        """
        Predict probabilities.

        Args:
            x: Input features

        Returns:
            Predicted probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probs = torch.sigmoid(logits)
        return probs
