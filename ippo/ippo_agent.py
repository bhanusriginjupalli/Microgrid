"""
ippo/ippo_agent.py
==================
Independent PPO (IPPO) Agent coordinator and checkpoint manager:
- 5 decentralized heterogeneous actors (reusing mappo/policies.py).
- 5 independent decentralized critics V_i(o_i) (never consuming global_state).
- 5 independent actor optimizers + 5 independent critic optimizers.
- Exposes select_actions(obs, deterministic=...) compatible with FixedValidationSuite.
"""

from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from mappo.policies import BoundedSquashedGaussianPolicy, CategoricalPolicy
from ippo.ippo_config import IPPOConfig
from ippo.ippo_critic import DecentralizedCriticSet


class IPPOAgent:
    """Coordinates 5 decentralized actors and 5 independent local critics."""

    def __init__(
        self,
        config: IPPOConfig,
        obs_dims: Dict[str, int],
        action_spaces: Dict[str, Any],
    ):
        self.config = config
        self.obs_dims = dict(obs_dims)
        self.action_spaces = action_spaces
        self.device = torch.device(config.device)

        # ------------------------------------------------------------
        # 1. Initialize 5 Decentralized Heterogeneous Actors
        # ------------------------------------------------------------
        self.actors: Dict[str, nn.Module] = {}
        self.actor_optimizers: Dict[str, Adam] = {}

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

        gen_bias = (
            float(getattr(config, "init_gen_bias", -0.60))
            if getattr(config, "init_bias_priors", False)
            else None
        )
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

        self.actors["load_management"] = CategoricalPolicy(
            obs_dim=obs_dims["load_management"],
            num_actions=6,
            hidden_dims=config.actor_hidden_dims,
            activation=config.activation,
            use_orthogonal=config.use_orthogonal_init,
        ).to(self.device)

        self.actors["grid_resilience"] = CategoricalPolicy(
            obs_dim=obs_dims["grid_resilience"],
            num_actions=5,
            hidden_dims=config.actor_hidden_dims,
            activation=config.activation,
            use_orthogonal=config.use_orthogonal_init,
        ).to(self.device)

        for ag, actor in self.actors.items():
            self.actor_optimizers[ag] = Adam(actor.parameters(), lr=config.actor_lr)

        # ------------------------------------------------------------
        # 2. Initialize 5 Independent Local Critics V_i(o_i)
        # ------------------------------------------------------------
        self.critics = DecentralizedCriticSet(
            obs_dims=obs_dims,
            hidden_dims=config.critic_hidden_dims,
            activation=config.activation,
            use_orthogonal=config.use_orthogonal_init,
            use_clipped_value_loss=config.use_clipped_value_loss,
            clip_eps=config.ppo_clip_eps,
            critic_lr=config.critic_lr,
            device=config.device,
        )

    def select_actions(
        self,
        obs_dict: Dict[str, np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Decentralized action selection compatible with FixedValidationSuite.
        Returns (actions_dict, log_probs_dict).
        """
        actions: Dict[str, Any] = {}
        log_probs: Dict[str, float] = {}

        with torch.no_grad():
            for ag, o in obs_dict.items():
                if ag not in self.actors:
                    continue
                o_t = torch.as_tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0)
                act, lp, _ = self.actors[ag].sample(o_t, deterministic=deterministic)

                act_np = act.squeeze(0).cpu().numpy()
                if ag in ("load_management", "grid_resilience"):
                    actions[ag] = int(act_np)
                else:
                    actions[ag] = act_np
                log_probs[ag] = float(lp.squeeze(0).cpu().item())

        return actions, log_probs

    def get_values(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Estimate independent per-agent values V_i(o_i) using local obs only."""
        return self.critics.get_values(obs_dict)

    def eval(self) -> None:
        for actor in self.actors.values():
            actor.eval()
        self.critics.eval()

    def train(self, mode: bool = True) -> None:
        for actor in self.actors.values():
            actor.train(mode)
        self.critics.train(mode)


class IPPOCheckpointManager:
    """Manages saving/loading independent actors, critics, optimizers, and metadata."""

    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path: Optional[Path] = None

    def save_checkpoint(
        self,
        agent: IPPOAgent,
        episode: int,
        metrics: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        tag: Optional[str] = None,
    ) -> Path:
        ckpt = {
            "algorithm": "IPPO",
            "episode": int(episode),
            "metrics": dict(metrics),
            "metadata": dict(metadata or {}),
            "actors": {ag: actor.state_dict() for ag, actor in agent.actors.items()},
            "actor_optimizers": {ag: opt.state_dict() for ag, opt in agent.actor_optimizers.items()},
            "critics": agent.critics.state_dicts(),
            "critic_optimizers": agent.critics.optimizer_state_dicts(),
            "config": dict(agent.config.__dict__),
        }

        filename = f"checkpoint_{tag}.pt" if tag else f"checkpoint_ep_{episode:04d}.pt"
        save_path = self.checkpoint_dir / filename
        torch.save(ckpt, save_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(ckpt, best_path)
            self.best_checkpoint_path = best_path

        return save_path

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        agent: IPPOAgent,
        load_optimizers: bool = True,
    ) -> Dict[str, Any]:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=agent.device, weights_only=False)
        for ag, state in ckpt.get("actors", {}).items():
            if ag in agent.actors:
                agent.actors[ag].load_state_dict(state)

        if load_optimizers and "actor_optimizers" in ckpt:
            for ag, opt_state in ckpt["actor_optimizers"].items():
                if ag in agent.actor_optimizers:
                    agent.actor_optimizers[ag].load_state_dict(opt_state)

        if "critics" in ckpt:
            agent.critics.load_state_dicts(ckpt["critics"])
        if load_optimizers and "critic_optimizers" in ckpt:
            agent.critics.load_optimizer_state_dicts(ckpt["critic_optimizers"])

        return {
            "algorithm": ckpt.get("algorithm", "IPPO"),
            "episode": ckpt.get("episode", 0),
            "metrics": ckpt.get("metrics", {}),
            "metadata": ckpt.get("metadata", {}),
        }
