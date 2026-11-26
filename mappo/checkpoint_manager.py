"""
mappo/checkpoint_manager.py
============================================================
Checkpoint management for MAPPO:
Saves actor/critic weights, optimizer states, best model tracking,
and supports full training restoration.
============================================================
"""

from typing import Dict, Any, Optional
from pathlib import Path
import json
import torch

from mappo.mappo_agent import MAPPOAgent
from mappo.config import MAPPOConfig


class CheckpointManager:
    """Manages saving, loading, and tracking best model checkpoints."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_reward = -float("inf")
        self.best_checkpoint_path: Optional[Path] = None

    def save_checkpoint(
        self,
        agent: MAPPOAgent,
        episode: int,
        metrics: Dict[str, Any],
        is_best: bool = False,
        tag: Optional[str] = None,
    ) -> Path:
        """Save a complete training checkpoint."""
        ckpt = {
            "episode": episode,
            "metrics": metrics,
            "actors": {ag: actor.state_dict() for ag, actor in agent.actors.items()},
            "actor_optimizers": {ag: opt.state_dict() for ag, opt in agent.actor_optimizers.items()},
            "critic": agent.critic.state_dict(),
            "critic_optimizer": agent.critic_optimizer.state_dict(),
            "config": agent.config.__dict__,
        }

        if tag:
            filename = f"checkpoint_{tag}.pt"
        else:
            filename = f"checkpoint_ep_{episode:04d}.pt"

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
        agent: MAPPOAgent,
        load_optimizers: bool = True,
    ) -> Dict[str, Any]:
        """Restore actor/critic model states and optimizers from file."""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=agent.device, weights_only=False)

        # Restore actors
        for ag, state in ckpt["actors"].items():
            if ag in agent.actors:
                agent.actors[ag].load_state_dict(state)

        # Restore actor optimizers
        if load_optimizers and "actor_optimizers" in ckpt:
            for ag, opt_state in ckpt["actor_optimizers"].items():
                if ag in agent.actor_optimizers:
                    agent.actor_optimizers[ag].load_state_dict(opt_state)

        # Restore critic
        if "critic" in ckpt:
            agent.critic.load_state_dict(ckpt["critic"])

        # Restore critic optimizer
        if load_optimizers and "critic_optimizer" in ckpt:
            agent.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])

        return {
            "episode": ckpt.get("episode", 0),
            "metrics": ckpt.get("metrics", {}),
        }
