"""MLP Classifier Lightning Module."""

from typing import override

import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor
from torch.optim import Adam, Optimizer
from torchmetrics import AUROC, Accuracy, F1Score, Precision, Recall
from torchmetrics.classification import BinaryAveragePrecision


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

        # Default hidden dimensions
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        # Build model
        layers: list[nn.Module] = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(leaky_relu_slope))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))

        self.model = nn.Sequential(*layers)

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

        # Metrics
        self.train_auroc = AUROC(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.test_auroc = AUROC(task="binary")

        self.train_auprc = BinaryAveragePrecision()
        self.val_auprc = BinaryAveragePrecision()
        self.test_auprc = BinaryAveragePrecision()

        self.train_f1 = F1Score(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.test_f1 = F1Score(task="binary")

        self.train_precision = Precision(task="binary")
        self.val_precision = Precision(task="binary")
        self.test_precision = Precision(task="binary")

        self.train_recall = Recall(task="binary")
        self.val_recall = Recall(task="binary")
        self.test_recall = Recall(task="binary")

        self.train_accuracy = Accuracy(task="binary")
        self.val_accuracy = Accuracy(task="binary")
        self.test_accuracy = Accuracy(task="binary")

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input features of shape (batch_size, input_dim)

        Returns:
            Logits of shape (batch_size, 1)
        """
        return self.model(x)

    @override
    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Training step.

        Args:
            batch: Tuple of (features, labels)
            batch_idx: Batch index

        Returns:
            Loss value
        """
        x, y = batch
        logits = self(x).squeeze()
        loss = self.criterion(logits, y)

        # Compute metrics
        probs = torch.sigmoid(logits)
        self.train_auroc(probs, y.int())
        self.train_auprc(probs, y.int())
        self.train_f1(probs, y.int())
        self.train_precision(probs, y.int())
        self.train_recall(probs, y.int())
        self.train_accuracy(probs, y.int())

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_auroc", self.train_auroc, prog_bar=True)
        self.log("train_auprc", self.train_auprc, prog_bar=False)
        self.log("train_f1", self.train_f1, prog_bar=False)
        self.log("train_precision", self.train_precision, prog_bar=False)
        self.log("train_recall", self.train_recall, prog_bar=False)
        self.log("train_accuracy", self.train_accuracy, prog_bar=False)

        return loss

    @override
    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """
        Validation step.

        Args:
            batch: Tuple of (features, labels)
            batch_idx: Batch index
        """
        x, y = batch
        logits = self(x).squeeze()
        loss = self.criterion(logits, y)

        # Compute metrics
        probs = torch.sigmoid(logits)
        self.val_auroc(probs, y.int())
        self.val_auprc(probs, y.int())
        self.val_f1(probs, y.int())
        self.val_precision(probs, y.int())
        self.val_recall(probs, y.int())
        self.val_accuracy(probs, y.int())

        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_auroc", self.val_auroc, prog_bar=True)
        self.log("val_auprc", self.val_auprc, prog_bar=False)
        self.log("val_f1", self.val_f1, prog_bar=False)
        self.log("val_precision", self.val_precision, prog_bar=False)
        self.log("val_recall", self.val_recall, prog_bar=False)
        self.log("val_accuracy", self.val_accuracy, prog_bar=False)

    @override
    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """
        Test step.

        Args:
            batch: Tuple of (features, labels)
            batch_idx: Batch index
        """
        x, y = batch
        logits = self(x).squeeze()
        loss = self.criterion(logits, y)

        # Compute metrics
        probs = torch.sigmoid(logits)
        self.test_auroc(probs, y.int())
        self.test_auprc(probs, y.int())
        self.test_f1(probs, y.int())
        self.test_precision(probs, y.int())
        self.test_recall(probs, y.int())
        self.test_accuracy(probs, y.int())

        # Log metrics
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_auroc", self.test_auroc, prog_bar=True)
        self.log("test_auprc", self.test_auprc, prog_bar=False)
        self.log("test_f1", self.test_f1, prog_bar=False)
        self.log("test_precision", self.test_precision, prog_bar=False)
        self.log("test_recall", self.test_recall, prog_bar=False)
        self.log("test_accuracy", self.test_accuracy, prog_bar=False)

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
