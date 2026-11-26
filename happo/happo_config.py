"""
happo/happo_config.py
=====================
Configuration dataclass for Heterogeneous-Agent Proximal Policy Optimization (HAPPO).
Inherits matched Phase 1 PPO & environment defaults while adding configurable
reproducible agent update ordering.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from mappo.config import MAPPOConfig
from rl_environment.agent_config import AGENTS

DEFAULT_UPDATE_ORDER: List[str] = [
    "renewable",
    "battery",
    "generator",
    "load_management",
    "grid_resilience",
]


@dataclass
class HAPPOConfig(MAPPOConfig):
    """Hyperparameters, environment settings, and sequential ordering for HAPPO."""

    algorithm: str = "HAPPO"
    update_order: List[str] = field(default_factory=lambda: list(DEFAULT_UPDATE_ORDER))
    randomize_update_order: bool = False

    # Matched Phase 4.4.4 / Phase 1 PPO settings
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip_eps: float = 0.20
    actor_lr: float = 5e-4
    critic_lr: float = 5e-4
    entropy_coef: float = 0.015
    value_loss_coef: float = 0.50
    max_grad_norm: float = 0.50
    ppo_epochs: int = 6
    minibatch_size: int = 128
    episodes_per_update: int = 4

    # Matched actor prior initialization
    init_bias_priors: bool = True
    init_gen_bias: float = -0.60

    # Frozen environment & reward specification
    forecast_mode: Optional[str] = "realistic"
    domain_randomization: bool = True
    randomize_scenarios: bool = True
    use_curriculum: bool = True
    reward_config_path: Optional[str] = "experiments/phase4_4_4/configs/reward_v3_social.yaml"
    enable_safety_shield: bool = True
    enable_adequacy_check: bool = True
    adequacy_fallback_during_train: bool = False

    # Checkpointing / Validation cadence
    val_interval_episodes: int = 20
    save_interval: int = 20

    def __post_init__(self):
        if sorted(self.update_order) != sorted(AGENTS) or len(self.update_order) != len(AGENTS):
            raise ValueError(
                f"update_order must contain exact permutation of {AGENTS}, got {self.update_order}"
            )

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> "HAPPOConfig":
        """Load HAPPOConfig from a YAML file while preserving dataclass defaults."""
        with open(path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}

        flat: Dict[str, Any] = {}
        for section in ("algorithm", "environment", "ppo_update", "network", "training"):
            if isinstance(raw.get(section), dict):
                flat.update(raw[section])
        for k, v in raw.items():
            if k not in ("algorithm", "environment", "ppo_update", "network", "training"):
                flat[k] = v

        if isinstance(raw.get("algorithm"), dict) and "name" in raw["algorithm"]:
            flat["algorithm"] = raw["algorithm"]["name"]

        flat.update({k: v for k, v in overrides.items() if v is not None})
        valid_fields = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in flat.items() if k in valid_fields}
        return cls(**filtered)
