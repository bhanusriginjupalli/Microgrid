"""
ippo/ippo_critic.py
===================
Independent per-agent decentralized critics V_i(o_i) for IPPO.
Reuses CentralizedValueFunction from mappo/critic.py parameterized strictly by
each agent's local observation dimension obs_dims[agent] (never global_state).
"""

from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn
from torch.optim import Adam

from mappo.critic import CentralizedValueFunction


class DecentralizedCriticSet:
    """
    Manages one completely independent critic network V_i(o_i) and optimizer
    per agent, enforcing strict parameter isolation between agents.
    """

    def __init__(
        self,
        obs_dims: Dict[str, int],
        hidden_dims: List[int],
        activation: str = "relu",
        use_orthogonal: bool = True,
        use_clipped_value_loss: bool = True,
        clip_eps: float = 0.2,
        critic_lr: float = 5e-4,
        device: str = "cpu",
    ):
        self.obs_dims = dict(obs_dims)
        self.device = torch.device(device)
        self.critics: Dict[str, CentralizedValueFunction] = {}
        self.optimizers: Dict[str, Adam] = {}

        for ag, obs_dim in self.obs_dims.items():
            critic_net = CentralizedValueFunction(
                state_dim=int(obs_dim),
                hidden_dims=list(hidden_dims),
                activation=activation,
                use_orthogonal=use_orthogonal,
                use_clipped_value_loss=use_clipped_value_loss,
                clip_eps=clip_eps,
            ).to(self.device)
            self.critics[ag] = critic_net
            self.optimizers[ag] = Adam(critic_net.parameters(), lr=critic_lr)

    def get_values(self, obs_dict: Dict[str, Any]) -> Dict[str, float]:
        """Estimate scalar V_i(o_i) independently per agent from local obs only."""
        values: Dict[str, float] = {}
        with torch.no_grad():
            for ag, critic in self.critics.items():
                if ag not in obs_dict:
                    continue
                obs_arr = obs_dict[ag]
                o_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
                if o_t.dim() == 1:
                    o_t = o_t.unsqueeze(0)
                if o_t.shape[-1] != self.obs_dims[ag]:
                    raise ValueError(
                        f"IPPO critic for '{ag}' expects local obs_dim={self.obs_dims[ag]}, "
                        f"got {o_t.shape[-1]}"
                    )
                val = critic(o_t)
                values[ag] = float(val.squeeze(0).cpu().item())
        return values

    def evaluate_agent_value(
        self,
        agent: str,
        local_obs: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute current V_i(o_i) predictions and clipped/MSE value loss for one agent."""
        if local_obs.shape[-1] != self.obs_dims[agent]:
            raise ValueError(
                f"IPPO critic for '{agent}' expects local obs_dim={self.obs_dims[agent]}, "
                f"got {local_obs.shape[-1]}"
            )
        return self.critics[agent].evaluate_values(
            state=local_obs,
            old_values=old_values,
            returns=returns,
        )

    def eval(self) -> None:
        for critic in self.critics.values():
            critic.eval()

    def train(self, mode: bool = True) -> None:
        for critic in self.critics.values():
            critic.train(mode)

    def state_dicts(self) -> Dict[str, Any]:
        return {ag: c.state_dict() for ag, c in self.critics.items()}

    def optimizer_state_dicts(self) -> Dict[str, Any]:
        return {ag: opt.state_dict() for ag, opt in self.optimizers.items()}

    def load_state_dicts(self, states: Dict[str, Any]) -> None:
        for ag, state in states.items():
            if ag in self.critics:
                self.critics[ag].load_state_dict(state)

    def load_optimizer_state_dicts(self, opt_states: Dict[str, Any]) -> None:
        for ag, state in opt_states.items():
            if ag in self.optimizers:
                self.optimizers[ag].load_state_dict(state)
