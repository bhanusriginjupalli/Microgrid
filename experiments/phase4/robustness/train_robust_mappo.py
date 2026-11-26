"""
experiments/phase4/robustness/train_robust_mappo.py
============================================================
GRIDGUARD AI - Phase 4.3: Robust Forecast-Aware MAPPO Training

Trains Robust Forecast-Aware MAPPO with Domain Randomization:
  - Multi-Horizon Causal Forecaster (15m, 30m, 1h, 2h, 4h)
  - Domain Randomization over weather, demand, equipment derating, and sensor noise
  - Curriculum learning with physical bias prior initialization
  - Saves multi-seed checkpoints and logs for robustness evaluation
============================================================
"""

import sys
import argparse
import time
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase1_digital_twin"))

from mappo.config import MAPPOConfig
from mappo.trainer import MAPPOTrainer


def train_robust_mappo(
    seed: int = 42,
    episodes: int = 50,
    ablation: str = "none",  # "none", "no_sensor_noise", "no_uncertainty"
    curriculum: bool = True,
):
    reward_v2 = str(ROOT / "experiments" / "phase4" / "diagnostics" / "configs" / "reward_v2_calibrated.yaml")
    base_fc_models = str(ROOT / "experiments" / "phase4" / "forecasting" / "models")
    base_rob_dir = ROOT / "experiments" / "phase4" / "robustness"

    if ablation == "none":
        run_name = f"robust_seed{seed}"
    else:
        run_name = f"ablation_{ablation}"

    ckpt_dir = base_rob_dir / "checkpoints" / run_name
    log_dir = base_rob_dir / "logs" / run_name
    results_dir = base_rob_dir / "results"
    plots_dir = base_rob_dir / "plots"

    for d in [ckpt_dir, log_dir, results_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    config = MAPPOConfig(
        scenario_id="scenario_a",
        total_episodes=episodes,
        use_curriculum=curriculum,
        init_bias_priors=True,
        reward_config_path=reward_v2,
        seed=seed,
        forecast_mode="realistic",
        forecast_models_dir=base_fc_models,
        checkpoint_dir=str(ckpt_dir),
        log_dir=str(log_dir),
        results_dir=str(results_dir),
        plots_dir=str(plots_dir),
        eval_interval_episodes=10,
        checkpoint_interval_episodes=10,
        domain_randomization=True,
        randomize_scenarios=True,
    )

    print("=" * 78)
    print(f"GRIDGUARD AI - Phase 4.3 Robust MAPPO Training [{run_name.upper()}]")
    print(f"Domain Randomization: ON | Seed: {seed} | Episodes: {episodes} | Ablation: {ablation}")
    print(f"Reward Formulation: Calibrated (reward_v2_calibrated.yaml)")
    print(f"Checkpoints: {ckpt_dir}")
    print("=" * 78)

    trainer = MAPPOTrainer(config)
    start_t = time.time()
    history = trainer.train(episodes=episodes)
    elapsed = time.time() - start_t

    # Save training history
    hist_df = pd.DataFrame(history)
    hist_csv = log_dir / f"training_history_{run_name}.csv"
    hist_df.to_csv(hist_csv, index=False)

    print("\n" + "=" * 78)
    print(f"[+] Robust MAPPO ({run_name}) Training complete in {elapsed:.1f}s ({elapsed/episodes:.2f}s/ep)")
    print(f"[+] Best checkpoint saved to: {trainer.checkpoint_manager.best_checkpoint_path}")
    print(f"[+] Training log saved to: {hist_csv}")
    print("=" * 78)

    return trainer.checkpoint_manager.best_checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Robust Forecast-Aware MAPPO")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--episodes", type=int, default=50, help="Training episodes")
    parser.add_argument("--ablation", type=str, default="none", choices=["none", "no_sensor_noise", "no_uncertainty"], help="Ablation type")
    args = parser.parse_args()

    train_robust_mappo(seed=args.seed, episodes=args.episodes, ablation=args.ablation)
