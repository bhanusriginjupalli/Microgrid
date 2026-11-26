"""
mappo/config.py
============================================================
Configuration parameters and hyperparameters for MAPPO training.
============================================================
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import torch


@dataclass
class MAPPOConfig:
    # Environment settings
    scenario_id: str = "scenario_a"
    max_steps: int = 96
    dt_minutes: float = 15.0
    enable_safety_shield: bool = True
    difficulty_level: int = 1
    randomize_scenarios: bool = False
    domain_randomization: bool = False

    # Neural network architectures
    actor_hidden_dims: List[int] = field(default_factory=lambda: [128, 128])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"  # relu, tanh
    use_orthogonal_init: bool = True
    init_bias_priors: bool = True  # Initialize continuous policy heads with physical priors (OFF / 0 curt)

    # MAPPO Hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip_eps: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 10
    minibatch_size: int = 256
    rollout_steps: int = 96  # Steps collected per rollout iteration (e.g. 1 episode = 96)
    use_clipped_value_loss: bool = True
    normalize_advantages: bool = True

    # Continuous policy parameters
    log_std_min: float = -20.0
    log_std_max: float = 2.0

    # Training control
    total_episodes: int = 500
    eval_interval_episodes: int = 25
    checkpoint_interval_episodes: int = 50
    eval_episodes_per_scenario: int = 3
    seed: int = 42

    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"max_episode": 75, "difficulty": 1, "description": "Normal operation and mild fluctuations"},
        {"max_episode": 150, "difficulty": 2, "description": "Solar/wind dips and variable demand"},
        {"max_episode": 250, "difficulty": 3, "description": "Short grid outages and synchronization"},
        {"max_episode": 350, "difficulty": 4, "description": "Severe storm, wind cut-out, generator start delays"},
        {"max_episode": 500, "difficulty": 5, "description": "Compound disaster & full domain randomization"},
    ])

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Forecasting integration
    forecast_mode: Optional[str] = None  # None / "reactive", "realistic", "oracle"
    forecast_models_dir: Optional[str] = None

    # Paths
    checkpoint_dir: str = "experiments/phase3/checkpoints"
    log_dir: str = "experiments/phase3/logs"
    results_dir: str = "experiments/phase3/results"
    plots_dir: str = "experiments/phase3/plots"
    reward_config_path: Optional[str] = None  # Custom reward YAML path (e.g. reward_v2_calibrated.yaml)
