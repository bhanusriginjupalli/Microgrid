"""
happo/happo_checkpoint.py
=========================
Standalone checkpoint manager and loader for HAPPO.
Preserves all actor/critic state dicts, optimizer states, sequential update
ordering, architecture/observation/action specs, reward & env configuration,
and validation metrics without overwriting IPPO or MAPPO checkpoints.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import torch

from happo.happo_config import HAPPOConfig
from happo.happo_agent import HAPPOAgent
from rl_environment.microgrid_parallel_env import MicrogridParallelEnv

ROOT = Path(__file__).resolve().parent.parent


class HAPPOCheckpointManager:
    """Manages saving and restoring HAPPO checkpoints independently."""

    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path: Optional[Path] = None

    def save_checkpoint(
        self,
        agent: HAPPOAgent,
        episode: int,
        metrics: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        tag: Optional[str] = None,
    ) -> Path:
        ckpt = {
            "algorithm": "HAPPO",
            "episode": int(episode),
            "update_order": list(agent.config.update_order),
            "metrics": dict(metrics),
            "metadata": dict(metadata or {}),
            "actors": {ag: actor.state_dict() for ag, actor in agent.actors.items()},
            "actor_optimizers": {ag: opt.state_dict() for ag, opt in agent.actor_optimizers.items()},
            "critic": agent.critic.state_dict(),
            "critic_optimizer": agent.critic_optimizer.state_dict(),
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
        agent: HAPPOAgent,
        load_optimizers: bool = True,
    ) -> Dict[str, Any]:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"HAPPO checkpoint not found at: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=agent.device, weights_only=False)
        for ag, state in ckpt.get("actors", {}).items():
            if ag in agent.actors:
                agent.actors[ag].load_state_dict(state)

        if load_optimizers and "actor_optimizers" in ckpt:
            for ag, opt_state in ckpt["actor_optimizers"].items():
                if ag in agent.actor_optimizers:
                    agent.actor_optimizers[ag].load_state_dict(opt_state)

        if "critic" in ckpt:
            agent.critic.load_state_dict(ckpt["critic"])
        if load_optimizers and "critic_optimizer" in ckpt:
            agent.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])

        return {
            "algorithm": ckpt.get("algorithm", "HAPPO"),
            "episode": ckpt.get("episode", 0),
            "update_order": ckpt.get("update_order", []),
            "metrics": ckpt.get("metrics", {}),
            "metadata": ckpt.get("metadata", {}),
        }


def load_happo_checkpoint(
    checkpoint_path: str | Path,
    forecast_mode: Optional[str] = "realistic",
    device: str = "cpu",
) -> HAPPOAgent:
    """Standalone factory loading a saved HAPPO checkpoint for inference/evaluation."""
    ckpt_path = Path(checkpoint_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = dict(ckpt.get("config", {}))
    valid = set(HAPPOConfig.__dataclass_fields__.keys())
    filtered_cfg = {k: v for k, v in cfg_dict.items() if k in valid}
    filtered_cfg["forecast_mode"] = forecast_mode
    filtered_cfg["device"] = device
    cfg = HAPPOConfig(**filtered_cfg)

    models_dir = cfg.forecast_models_dir or str(ROOT / "experiments" / "phase4" / "forecasting" / "models")
    dummy_env = MicrogridParallelEnv(
        scenario_id="scenario_a",
        forecast_mode=forecast_mode,
        forecast_models_dir=models_dir,
    )
    obs_dims = {ag: int(dummy_env.observation_space(ag).shape[0]) for ag in dummy_env.possible_agents}
    state_dim = int(dummy_env._global_state_space.shape[0])

    agent = HAPPOAgent(
        config=cfg,
        obs_dims=obs_dims,
        action_spaces=dummy_env._action_spaces,
        state_dim=state_dim,
    )
    for ag, actor in agent.actors.items():
        if ag in ckpt.get("actors", {}):
            actor.load_state_dict(ckpt["actors"][ag])
            actor.eval()
    if "critic" in ckpt:
        agent.critic.load_state_dict(ckpt["critic"])
        agent.critic.eval()
    agent.eval()
    return agent
