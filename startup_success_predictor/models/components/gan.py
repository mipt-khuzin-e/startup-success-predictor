"""WGAN-GP Generator and Critic networks."""

from typing import override

import torch
import torch.nn as nn
from torch import Tensor


class Generator(nn.Module):
    """Generator network for WGAN-GP."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        leaky_relu_slope: float = 0.2,
        use_batch_norm: bool = True,
    ) -> None:
        """Initialize Generator network."""
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        layers: list[nn.Module] = []
        in_dim = latent_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(leaky_relu_slope))
            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.Tanh())  # Output in [-1, 1]

        self.model = nn.Sequential(*layers)

    @override
    def forward(self, z: Tensor) -> Tensor:
        """Generate samples from latent vectors."""
        out = self.model(z)
        assert isinstance(out, Tensor)
        return out


class Critic(nn.Module):
    """Critic (Discriminator) network for WGAN-GP."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        leaky_relu_slope: float = 0.2,
        dropout: float = 0.3,
    ) -> None:
        """Initialize Critic network."""
        super().__init__()
        self.input_dim = input_dim

        layers: list[nn.Module] = []
        in_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU(leaky_relu_slope))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        # Output layer (no activation for Wasserstein loss)
        layers.append(nn.Linear(in_dim, 1))

        self.model = nn.Sequential(*layers)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Score samples with critic."""
        out = self.model(x)
        assert isinstance(out, Tensor)
        return out


def compute_gradient_penalty(
    critic: Critic,
    real_samples: Tensor,
    fake_samples: Tensor,
    device: torch.device,
) -> Tensor:
    """Compute gradient penalty for WGAN-GP."""
    batch_size = real_samples.size(0)

    # Random weight for interpolation
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand_as(real_samples)

    # Interpolated samples
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates = interpolates.requires_grad_(True)

    # Critic scores for interpolated samples
    critic_interpolates = critic(interpolates)

    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Flatten gradients
    gradients = gradients.view(batch_size, -1)

    # Compute gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    assert isinstance(gradient_penalty, Tensor)

    return gradient_penalty
