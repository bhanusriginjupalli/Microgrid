"""GRIDGUARD AI — Heterogeneous-Agent PPO (HAPPO) package."""

from happo.happo_config import HAPPOConfig, DEFAULT_UPDATE_ORDER
from happo.happo_buffer import HAPPORolloutBuffer
from happo.happo_advantage import (
    compute_happo_gae_sequence,
    finalize_episode_happo_gae,
    normalize_happo_advantages,
    compute_decomposed_advantage,
    compute_cumulative_importance_factor,
)
from happo.happo_agent import HAPPOAgent
from happo.happo_checkpoint import HAPPOCheckpointManager, load_happo_checkpoint
from happo.happo_update import update_happo
from happo.happo_trainer import HAPPOTrainer

__all__ = [
    "HAPPOConfig",
    "DEFAULT_UPDATE_ORDER",
    "HAPPORolloutBuffer",
    "compute_happo_gae_sequence",
    "finalize_episode_happo_gae",
    "normalize_happo_advantages",
    "compute_decomposed_advantage",
    "compute_cumulative_importance_factor",
    "HAPPOAgent",
    "HAPPOCheckpointManager",
    "load_happo_checkpoint",
    "update_happo",
    "HAPPOTrainer",
]
