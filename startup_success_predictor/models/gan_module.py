"""WGAN-GP Lightning Module."""

from typing import Any, override

import torch
from lightning import LightningModule
from torch import Tensor
from torch.optim import Adam, Optimizer

from startup_success_predictor.models.components.gan import (
    Critic,
    Generator,
    compute_gradient_penalty,
)


class WGANGPModule(LightningModule):
    """Lightning Module for WGAN-GP training."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 100,
        generator_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        leaky_relu_slope: float = 0.2,
        use_batch_norm: bool = True,
        dropout: float = 0.3,
        learning_rate: float = 0.0001,
        beta1: float = 0.5,
        beta2: float = 0.999,
        n_critic: int = 5,
        lambda_gp: float = 10.0,
    ) -> None:
        """Initialize WGAN-GP module."""
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp

        # Default hidden dimensions
        if generator_hidden_dims is None:
            generator_hidden_dims = [256, 128]
        if critic_hidden_dims is None:
            critic_hidden_dims = [128, 64]

        # Initialize networks
        self.generator = Generator(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=generator_hidden_dims,
            leaky_relu_slope=leaky_relu_slope,
            use_batch_norm=use_batch_norm,
        )

        self.critic = Critic(
            input_dim=input_dim,
            hidden_dims=critic_hidden_dims,
            leaky_relu_slope=leaky_relu_slope,
            dropout=dropout,
        )

        # Manual optimization
        self.automatic_optimization = False

    @override
    def forward(self, z: Tensor) -> Tensor:
        """Generate samples from latent vectors."""
        out = self.generator(z)
        assert isinstance(out, Tensor)
        return out

    @override
    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Training step with manual optimization."""
        real_samples, _ = batch
        batch_size = real_samples.size(0)

        # Get optimizers
        optimizers = self.optimizers()
        opt_g, opt_c = optimizers  # type: ignore[attr-defined]

        # Train Critic
        for _ in range(self.n_critic):
            opt_c.zero_grad()

            # Generate fake samples
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_samples = self.generator(z).detach()

            # Critic scores
            real_validity = self.critic(real_samples)
            fake_validity = self.critic(fake_samples)

            # Gradient penalty
            gp = compute_gradient_penalty(
                self.critic, real_samples, fake_samples, self.device
            )

            # Critic loss (Wasserstein loss with gradient penalty)
            critic_loss = (
                -torch.mean(real_validity)
                + torch.mean(fake_validity)
                + self.lambda_gp * gp
            )

            self.manual_backward(critic_loss)
            opt_c.step()

        # Train Generator
        opt_g.zero_grad()

        # Generate fake samples
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_samples = self.generator(z)

        # Generator loss
        fake_validity = self.critic(fake_samples)
        generator_loss = -torch.mean(fake_validity)

        self.manual_backward(generator_loss)
        opt_g.step()

        # Log losses
        self.log("train_critic_loss", critic_loss, prog_bar=True)
        self.log("train_generator_loss", generator_loss, prog_bar=True)
        self.log("train_gp", gp, prog_bar=False)

    @override
    def configure_optimizers(self) -> tuple[list[Optimizer], list[Any]]:
        """Configure optimizers for generator and critic."""
        opt_g = Adam(
            self.generator.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )
        opt_c = Adam(
            self.critic.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )
        return [opt_g, opt_c], []

    def generate_samples(self, n_samples: int) -> Tensor:
        """Generate synthetic samples."""
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=self.device)
            samples = self.generator(z)
            assert isinstance(samples, Tensor)
        return samples
