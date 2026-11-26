"""
mappo/mappo_agent.py
============================================================
MAPPOAgent coordinator:
Manages the 5 decentralized actors and the centralized critic.
Supports decentralized execution (actors only) and centralized training.
============================================================
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from mappo.config import MAPPOConfig
from mappo.policies import BoundedSquashedGaussianPolicy, CategoricalPolicy
from mappo.critic import CentralizedValueFunction


class MAPPOAgent:
    """
    Coordinates 5 decentralized actors and 1 centralized critic.
    """

    def __init__(
        self,
        config: MAPPOConfig,
        obs_dims: Dict[str, int],
        action_spaces: Dict[str, Any],
        state_dim: int = 26,
    ):
        self.config = config
        self.obs_dims = obs_dims
        self.action_spaces = action_spaces
        self.state_dim = state_dim
        self.device = torch.device(config.device)

        # ------------------------------------------------------------
        # 1. Initialize Decentralized Actors
        # ------------------------------------------------------------
        self.actors: Dict[str, nn.Module] = {}
        self.actor_optimizers: Dict[str, Adam] = {}

        # Renewable Agent: Continuous Box(2,) in [0, 1]
        ren_bias = -2.5 if getattr(config, "init_bias_priors", False) else None
        self.actors["renewable"] = BoundedSquashedGaussianPolicy(
            obs_dim=obs_dims["renewable"],
            action_dim=2,
            action_low=torch.tensor([0.0, 0.0]),
            action_high=torch.tensor([1.0, 1.0]),
            hidden_dims=config.actor_hidden_dims,
            activation=config.activation,
            log_std_min=config.log_std_min,
            log_std_max=config.log_std_max,
            use_orthogonal=config.use_orthogonal_init,
            init_mean_bias=ren_bias,
        ).to(self.device)

        # Battery Agent: Continuous Box(1,) in [-1, 1]
        self.actors["battery"] = BoundedSquashedGaussianPolicy(
            obs_dim=obs_dims["battery"],
            action_dim=1,
            action_low=torch.tensor([-1.0]),
            action_high=torch.tensor([1.0]),
            hidden_dims=config.actor_hidden_dims,
            activation=config.activation,
            log_std_min=config.log_std_min,
            log_std_max=config.log_std_max,
            use_orthogonal=config.use_orthogonal_init,
            init_mean_bias=0.0,
        ).to(self.device)

        # Generator Agent: Continuous Box(1,) in [0, 1]
        gen_bias = -2.5 if getattr(config, "init_bias_priors", False) else None
        self.actors["generator"] = BoundedSquashedGaussianPolicy(
            obs_dim=obs_dims["generator"],
            action_dim=1,
            action_low=torch.tensor([0.0]),
            action_high=torch.tensor([1.0]),
            hidden_dims=config.actor_hidden_dims,
            activation=config.activation,
            log_std_min=config.log_std_min,
            log_std_max=config.log_std_max,
            use_orthogonal=config.use_orthogonal_init,
            init_mean_bias=gen_bias,
        ).to(self.device)

        # Load Management Agent: Discrete(6)
        self.actors["load_management"] = CategoricalPolicy(
            obs_dim=obs_dims["load_management"],
            num_actions=6,
            hidden_dims=config.actor_hidden_dims,
            activation=config.activation,
            use_orthogonal=config.use_orthogonal_init,
        ).to(self.device)

        # Grid Resilience Agent: Discrete(5)
        self.actors["grid_resilience"] = CategoricalPolicy(
            obs_dim=obs_dims["grid_resilience"],
            num_actions=5,
            hidden_dims=config.actor_hidden_dims,
            activation=config.activation,
            use_orthogonal=config.use_orthogonal_init,
        ).to(self.device)

        # Create separate optimizers for each actor
        for ag, actor in self.actors.items():
            self.actor_optimizers[ag] = Adam(actor.parameters(), lr=config.actor_lr)

        # ------------------------------------------------------------
        # 2. Initialize Centralized Critic
        # ------------------------------------------------------------
        self.critic = CentralizedValueFunction(
            state_dim=state_dim,
            hidden_dims=config.critic_hidden_dims,
            activation=config.activation,
            use_orthogonal=config.use_orthogonal_init,
            use_clipped_value_loss=config.use_clipped_value_loss,
            clip_eps=config.ppo_clip_eps,
        ).to(self.device)

        self.critic_optimizer = Adam(self.critic.parameters(), lr=config.critic_lr)

    def select_actions(
        self,
        obs_dict: Dict[str, np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """
        Decentralized action selection for all active agents.
        Does NOT require the critic.

        Returns:
            (actions_dict, log_probs_dict)
        """
        actions = {}
        log_probs = {}

        with torch.no_grad():
            for ag, o in obs_dict.items():
                if ag not in self.actors:
                    continue

                o_t = torch.as_tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0)
                act, lp, _ = self.actors[ag].sample(o_t, deterministic=deterministic)

                act_np = act.squeeze(0).cpu().numpy()
                if ag in ["load_management", "grid_resilience"]:
                    actions[ag] = int(act_np)
                else:
                    actions[ag] = act_np

                log_probs[ag] = float(lp.squeeze(0).cpu().item())

        return actions, log_probs

    def get_value(self, global_state: np.ndarray) -> float:
        """Centralized value estimation V(s) during training."""
        with torch.no_grad():
            s_t = torch.as_tensor(global_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            val = self.critic(s_t)
            return float(val.squeeze(0).cpu().item())

    def eval(self):
        """Set all actor and critic networks to evaluation mode."""
        for actor in self.actors.values():
            actor.eval()
        self.critic.eval()

    def train(self, mode: bool = True):
        """Set all actor and critic networks to training mode."""
        for actor in self.actors.values():
            actor.train(mode)
        self.critic.train(mode)
