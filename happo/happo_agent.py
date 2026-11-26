"""
happo/happo_agent.py
====================
Heterogeneous-Agent PPO (HAPPO) Agent coordinator:
- Reuses BoundedSquashedGaussianPolicy & CategoricalPolicy for the 5 actors
  (strictly accepting agent-local observations only).
- Reuses CentralizedValueFunction over the 45-dimensional global state.
- Exposes select_actions(obs, deterministic=...) compatible with FixedValidationSuite.
"""

from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from mappo.policies import BoundedSquashedGaussianPolicy, CategoricalPolicy
from mappo.critic import CentralizedValueFunction
from happo.happo_config import HAPPOConfig


class HAPPOAgent:
    """Coordinates 5 decentralized heterogeneous actors and 1 centralized critic."""

    def __init__(
        self,
        config: HAPPOConfig,
        obs_dims: Dict[str, int],
        action_spaces: Dict[str, Any],
        state_dim: int = 45,
    ):
        self.config = config
        self.obs_dims = dict(obs_dims)
        self.action_spaces = action_spaces
        self.state_dim = int(state_dim)
        self.device = torch.device(config.device)

        # ------------------------------------------------------------
        # 1. Initialize 5 Decentralized Heterogeneous Actors (Local Obs Only)
        # ------------------------------------------------------------
        self.actors: Dict[str, nn.Module] = {}
        self.actor_optimizers: Dict[str, Adam] = {}

        ren_bias = -2.5 if getattr(config, "init_bias_priors", False) else None
        self.actors["renewable"] = BoundedSquashedGaussianPolicy(
            obs_dim=self.obs_dims["renewable"],
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

        self.actors["battery"] = BoundedSquashedGaussianPolicy(
            obs_dim=self.obs_dims["battery"],
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

        gen_bias = (
            float(getattr(config, "init_gen_bias", -0.60))
            if getattr(config, "init_bias_priors", False)
            else None
        )
        self.actors["generator"] = BoundedSquashedGaussianPolicy(
            obs_dim=self.obs_dims["generator"],
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

        self.actors["load_management"] = CategoricalPolicy(
            obs_dim=self.obs_dims["load_management"],
            num_actions=6,
            hidden_dims=config.actor_hidden_dims,
            activation=config.activation,
            use_orthogonal=config.use_orthogonal_init,
        ).to(self.device)

        self.actors["grid_resilience"] = CategoricalPolicy(
            obs_dim=self.obs_dims["grid_resilience"],
            num_actions=5,
            hidden_dims=config.actor_hidden_dims,
            activation=config.activation,
            use_orthogonal=config.use_orthogonal_init,
        ).to(self.device)

        for ag, actor in self.actors.items():
            self.actor_optimizers[ag] = Adam(actor.parameters(), lr=config.actor_lr)

        # ------------------------------------------------------------
        # 2. Initialize Shared Centralized Critic V(s) (45-dim global state)
        # ------------------------------------------------------------
        self.critic = CentralizedValueFunction(
            state_dim=self.state_dim,
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
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Decentralized inference using strictly local observations.
        Compatible with FixedValidationSuite.evaluate_agent.
        """
        actions: Dict[str, Any] = {}
        log_probs: Dict[str, float] = {}

        with torch.no_grad():
            for ag, o in obs_dict.items():
                if ag not in self.actors:
                    continue
                o_t = torch.as_tensor(o, dtype=torch.float32, device=self.device)
                if o_t.dim() == 1:
                    if o_t.shape[0] != self.obs_dims[ag]:
                        raise ValueError(
                            f"HAPPO actor '{ag}' expects local obs_dim={self.obs_dims[ag]}, got {o_t.shape[0]}"
                        )
                    o_t = o_t.unsqueeze(0)
                act, lp, _ = self.actors[ag].sample(o_t, deterministic=deterministic)

                act_np = act.squeeze(0).cpu().numpy()
                if ag in ("load_management", "grid_resilience"):
                    actions[ag] = int(act_np)
                else:
                    actions[ag] = act_np
                log_probs[ag] = float(lp.squeeze(0).cpu().item())

        return actions, log_probs

    def get_value(self, global_state: np.ndarray) -> float:
        """Estimate centralized state value V(s) during rollout collection."""
        with torch.no_grad():
            s_t = torch.as_tensor(global_state, dtype=torch.float32, device=self.device)
            if s_t.dim() == 1:
                if s_t.shape[0] != self.state_dim:
                    raise ValueError(
                        f"HAPPO critic expects global state_dim={self.state_dim}, got {s_t.shape[0]}"
                    )
                s_t = s_t.unsqueeze(0)
            val = self.critic(s_t)
            return float(val.squeeze(0).cpu().item())

    def eval(self) -> None:
        for actor in self.actors.values():
            actor.eval()
        self.critic.eval()

    def train(self, mode: bool = True) -> None:
        for actor in self.actors.values():
            actor.train(mode)
        self.critic.train(mode)
