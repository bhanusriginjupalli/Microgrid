"""
ippo/ippo_config.py
===================
Configuration dataclass for Independent Proximal Policy Optimization (IPPO).
Reuses MAPPOConfig defaults cleanly while enforcing matched comparative study
settings and strictly decentralized per-agent critics V_i(o_i).
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path
import yaml

from mappo.config import MAPPOConfig


@dataclass
class IPPOConfig(MAPPOConfig):
    """Hyperparameters and environment settings for matched Decentralized IPPO."""

    algorithm: str = "IPPO"
    critic_obs_only: bool = True

    # Matched Phase 4.4.4 PPO settings for fair comparison
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip_eps: float = 0.20
    actor_lr: float = 5e-4
    critic_lr: float = 5e-4
    entropy_coef: float = 0.015
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.50
    ppo_epochs: int = 6
    minibatch_size: int = 128
    episodes_per_update: int = 4

    # Policy initialization matched with Phase 4.4.4
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

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> "IPPOConfig":
        """Load IPPOConfig from a YAML file while preserving dataclass defaults."""
        with open(path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}

        flat: Dict[str, Any] = {}
        for section in ("algorithm", "environment", "ppo_update", "network", "training"):
            if isinstance(raw.get(section), dict):
                flat.update(raw[section])
        for k, v in raw.items():
            if k not in ("algorithm", "environment", "ppo_update", "network", "training"):
                flat[k] = v

        # Handle section name collision if top-level "algorithm" was a dict
        if isinstance(raw.get("algorithm"), dict) and "name" in raw["algorithm"]:
            flat["algorithm"] = raw["algorithm"]["name"]

        flat.update({k: v for k, v in overrides.items() if v is not None})
        valid_fields = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in flat.items() if k in valid_fields}
        return cls(**filtered)
