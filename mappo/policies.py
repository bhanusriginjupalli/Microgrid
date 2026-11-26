"""
mappo/policies.py
============================================================
Policy distributions for MAPPO:
1. BoundedSquashedGaussianPolicy: Exact Tanh-squashed Gaussian with change-of-variables
   Jacobian correction for continuous action spaces ([0, 1] and [-1, 1]).
2. CategoricalPolicy: Discrete action policy with Softmax categorical distribution.
============================================================
"""

from typing import Tuple, Optional
import math
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from mappo.networks import ActorBackbone, init_weights


class BoundedSquashedGaussianPolicy(nn.Module):
    """
    Continuous Gaussian Actor with Tanh squashing and target bound scaling.
    Applies exact change-of-variables log-determinant correction.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        hidden_dims: list = [128, 128],
        activation: str = "relu",
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        use_orthogonal: bool = True,
        init_mean_bias: Optional[float] = None,
    ):
        super().__init__()
        self.backbone = ActorBackbone(
            obs_dim=obs_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            use_orthogonal=use_orthogonal,
        )
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Output heads
        self.mean_head = nn.Linear(self.backbone.feature_dim, action_dim)
        self.log_std_head = nn.Linear(self.backbone.feature_dim, action_dim)

        if use_orthogonal:
            init_weights(self.mean_head, gain=0.01)
            init_weights(self.log_std_head, gain=0.01)

        if init_mean_bias is not None:
            nn.init.constant_(self.mean_head.bias, float(init_mean_bias))

        # Register bounds as buffers
        low = action_low.clone().detach().float()
        high = action_high.clone().detach().float()
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)
        self.register_buffer("action_mid", (high + low) / 2.0)
        self.register_buffer("action_half", (high - low) / 2.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and compute its exact log probability.

        Returns:
            (scaled_action, log_prob, raw_gaussian_sample)
        """
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)

        if deterministic:
            raw_u = mean
            squashed = torch.tanh(raw_u)
            scaled_action = self.action_mid + self.action_half * squashed
            log_prob = self._compute_log_prob(mean, log_std, raw_u, squashed)
        else:
            normal_dist = Normal(mean, std)
            raw_u = normal_dist.rsample()
            squashed = torch.tanh(raw_u)
            scaled_action = self.action_mid + self.action_half * squashed
            log_prob = self._compute_log_prob(mean, log_std, raw_u, squashed)

        return scaled_action, log_prob, raw_u

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log-probability and entropy for given actions.
        """
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)
        normal_dist = Normal(mean, std)

        # Invert squashed scaling to find pre-squashed latent u
        # scaled_action = mid + half * tanh(u) -> tanh(u) = (action - mid) / half
        eps = 1e-6
        squashed = (actions - self.action_mid) / torch.clamp(self.action_half, min=eps)
        squashed = torch.clamp(squashed, -1.0 + eps, 1.0 - eps)

        # Inverse tanh: atanh(y) = 0.5 * ln((1 + y) / (1 - y))
        raw_u = 0.5 * torch.log((1.0 + squashed) / (1.0 - squashed))

        log_prob = self._compute_log_prob(mean, log_std, raw_u, squashed)
        entropy = normal_dist.entropy().sum(dim=-1)

        return log_prob, entropy

    def _compute_log_prob(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        raw_u: torch.Tensor,
        squashed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Exact log prob with change of variables:
        log pi(a|s) = log p(u) - sum(log(1 - tanh(u)^2 + eps)) - sum(log(half_range))
        """
        eps = 1e-6
        std = torch.exp(log_std)
        var = std ** 2
        log_2pi = math.log(2.0 * math.pi)

        # Gaussian log density: -0.5 * ((u - mean)^2 / var + 2*log_std + log(2pi))
        log_p_u = -0.5 * (((raw_u - mean) ** 2) / (var + eps) + 2.0 * log_std + log_2pi).sum(dim=-1)

        # Tanh squashing Jacobian correction: sum(log(1 - y^2 + eps))
        squash_correction = torch.log(torch.clamp(1.0 - squashed ** 2, min=eps)).sum(dim=-1)

        # Scaling Jacobian correction: sum(log(action_half))
        scale_correction = torch.log(torch.clamp(self.action_half, min=eps)).sum(dim=-1)

        log_prob = log_p_u - squash_correction - scale_correction
        return log_prob


class CategoricalPolicy(nn.Module):
    """Discrete Action Actor outputting categorical action distribution."""

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_dims: list = [128, 128],
        activation: str = "relu",
        use_orthogonal: bool = True,
    ):
        super().__init__()
        self.backbone = ActorBackbone(
            obs_dim=obs_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            use_orthogonal=use_orthogonal,
        )
        self.num_actions = num_actions
        self.logits_head = nn.Linear(self.backbone.feature_dim, num_actions)

        if use_orthogonal:
            init_weights(self.logits_head, gain=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(obs)
        logits = self.logits_head(features)
        return logits

    def sample(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample discrete action and return (action, log_prob, logits).
        """
        logits = self.forward(obs)
        dist = Categorical(logits=logits)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob, logits

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log-probabilities and entropy for given actions.
        """
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions.long())
        entropy = dist.entropy()
        return log_prob, entropy
