"""
mappo/networks.py
============================================================
Neural network backbones for MAPPO Actor and Centralized Critic.
Includes orthogonal initialization and configurable hidden layers.
============================================================
"""

from typing import List, Optional
import torch
import torch.nn as nn


def init_weights(m: nn.Module, gain: float = 1.0):
    """Orthogonal initialization with specified gain."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class MLPBase(nn.Module):
    """Configurable Multi-Layer Perceptron backbone."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
        use_orthogonal: bool = True,
    ):
        super().__init__()
        layers = []
        curr_dim = input_dim

        for h_dim in hidden_dims:
            linear = nn.Linear(curr_dim, h_dim)
            if use_orthogonal:
                init_weights(linear, gain=nn.init.calculate_gain(activation if activation in ["relu", "tanh"] else "linear"))
            layers.append(linear)

            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.01))
            else:
                layers.append(nn.ReLU())

            curr_dim = h_dim

        self.network = nn.Sequential(*layers)
        self.output_dim = curr_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ActorBackbone(nn.Module):
    """Actor feature extractor processing local observations."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: List[int] = [128, 128],
        activation: str = "relu",
        use_orthogonal: bool = True,
    ):
        super().__init__()
        self.base = MLPBase(
            input_dim=obs_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            use_orthogonal=use_orthogonal,
        )
        self.feature_dim = self.base.output_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.base(obs)


class CentralizedCritic(nn.Module):
    """Centralized Value Network V_phi(s) taking the 26-dim global state."""

    def __init__(
        self,
        state_dim: int = 26,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        use_orthogonal: bool = True,
    ):
        super().__init__()
        self.base = MLPBase(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            use_orthogonal=use_orthogonal,
        )
        self.value_head = nn.Linear(self.base.output_dim, 1)

        if use_orthogonal:
            init_weights(self.value_head, gain=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.base(state)
        value = self.value_head(features)
        return value.squeeze(-1)
