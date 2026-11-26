"""
mappo/critic.py
============================================================
Centralized Critic module evaluating the joint system global state.
============================================================
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn

from mappo.networks import CentralizedCritic


class CentralizedValueFunction(nn.Module):
    """
    Centralized Critic wrapping CentralizedCritic network with
    support for value clipping and loss computation.
    """

    def __init__(
        self,
        state_dim: int = 26,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        use_orthogonal: bool = True,
        use_clipped_value_loss: bool = True,
        clip_eps: float = 0.2,
    ):
        super().__init__()
        self.network = CentralizedCritic(
            state_dim=state_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            use_orthogonal=use_orthogonal,
        )
        self.use_clipped_value_loss = use_clipped_value_loss
        self.clip_eps = clip_eps

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

    def evaluate_values(
        self,
        state: torch.Tensor,
        old_values: Optional[torch.Tensor] = None,
        returns: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Evaluate current value prediction and compute value loss.
        """
        values = self.forward(state)

        if returns is None:
            return values, None

        if self.use_clipped_value_loss and old_values is not None:
            # PPO-style value clipping: max((V - R)^2, (V_clipped - R)^2)
            v_clipped = old_values + torch.clamp(
                values - old_values,
                -self.clip_eps,
                self.clip_eps,
            )
            v_loss_unclipped = (values - returns) ** 2
            v_loss_clipped = (v_clipped - returns) ** 2
            value_loss = 0.5 * torch.mean(torch.max(v_loss_unclipped, v_loss_clipped))
        else:
            value_loss = 0.5 * torch.mean((values - returns) ** 2)

        return values, value_loss
