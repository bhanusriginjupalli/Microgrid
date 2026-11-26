"""
experiments/phase4/forecasting/train_forecast_mappo.py
============================================================
Trains Forecast-Aware MAPPO on GRIDGUARD AI Microgrid Environment:
  - Mode 2 (Realistic Causal Multi-Horizon Forecaster)
  - Mode 1 (Oracle Perfect Lookahead Upper Bound)
Utilizes physical prior bias initialization and calibrated reward formulation.
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
from mappo.evaluator import MAPPOEvaluator


def train_forecast_mappo(
    mode: str = "realistic",
    episodes: int = 50,
    seed: int = 42,
    curriculum: bool = True,
):
    reward_v2 = str(ROOT / "experiments" / "phase4" / "diagnostics" / "configs" / "reward_v2_calibrated.yaml")
    base_fc_dir = ROOT / "experiments" / "phase4" / "forecasting"
    models_dir = str(base_fc_dir / "models")
    ckpt_dir = base_fc_dir / "checkpoints" / mode
    log_dir = base_fc_dir / "logs" / mode
    results_dir = base_fc_dir / "results"
    plots_dir = base_fc_dir / "plots"

    for d in [ckpt_dir, log_dir, results_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    config = MAPPOConfig(
        scenario_id="scenario_a",
        total_episodes=episodes,
        use_curriculum=curriculum,
        init_bias_priors=True,
        reward_config_path=reward_v2,
        seed=seed,
        forecast_mode=mode,
        forecast_models_dir=models_dir,
        checkpoint_dir=str(ckpt_dir),
        log_dir=str(log_dir),
        results_dir=str(results_dir),
        plots_dir=str(plots_dir),
        eval_interval_episodes=10,
        checkpoint_interval_episodes=25,
    )

    print("=" * 78)
    print(f"GRIDGUARD AI - Phase 4.2 Forecast-Aware MAPPO Training [{mode.upper()} MODE]")
    print(f"Forecast Mode: {mode} | Priors: Physical ON | Reward: reward_v2_calibrated.yaml")
    print(f"Episodes: {episodes} | Seed: {seed} | Device: {config.device}")
    print("=" * 78)

    trainer = MAPPOTrainer(config)
    start_t = time.time()
    history = trainer.train(episodes=episodes)
    elapsed = time.time() - start_t

    # Save training history to CSV
    hist_df = pd.DataFrame(history)
    hist_csv = log_dir / f"training_history_{mode}.csv"
    hist_df.to_csv(hist_csv, index=False)

    print("\n" + "=" * 78)
    print(f"[+] Forecast-Aware MAPPO ({mode}) Training complete in {elapsed:.1f}s ({elapsed/episodes:.2f}s/ep)")
    print(f"[+] Best checkpoint saved to: {trainer.checkpoint_manager.best_checkpoint_path}")
    print(f"[+] Training log saved to: {hist_csv}")
    print("=" * 78)

    # Post-training evaluation across all 6 benchmark scenarios
    print(f"\nEvaluating Forecast-Aware MAPPO ({mode}) across Benchmark Scenarios A-F...")
    evaluator = MAPPOEvaluator(
        agent=trainer.agent,
        reward_config_path=reward_v2,
        forecast_mode=mode,
        forecast_models_dir=models_dir,
    )

    benchmark_scenarios = ["scenario_a", "scenario_b", "scenario_c", "scenario_d", "scenario_e", "scenario_f"]
    eval_results = []

    for scen in benchmark_scenarios:
        res = evaluator.evaluate_scenario(scenario_id=scen, num_episodes=1, deterministic=True)
        eval_results.append({
            "mode": mode,
            "scenario": scen,
            "mean_reward": res["mean_reward"],
            "critical_reliability_pct": res["critical_reliability_pct"],
            "critical_ens_mwh": res["critical_ens_mwh"],
            "flexible_ens_mwh": res["flexible_ens_mwh"],
            "total_operating_cost": res["operating_cost_usd"],
            "total_fuel_l": res["fuel_consumed_l"],
            "renewable_curtailment_mwh": res["renewable_curtailment_mwh"],
            "safety_interventions": res["shield_interventions"],
        })
        print(f"  {scen:12s} | Reward: {res['mean_reward']:+7.2f} | Cost: ${res['operating_cost_usd']:7.2f} | Crit Rel: {res['critical_reliability_pct']:.1f}% | ENS: {res['critical_ens_mwh']:.4f} MWh")

    eval_df = pd.DataFrame(eval_results)
    eval_csv = results_dir / f"benchmark_evaluation_{mode}.csv"
    eval_df.to_csv(eval_csv, index=False)
    print(f"[+] Benchmark results saved to: {eval_csv}")

    return trainer, eval_df


def main():
    parser = argparse.ArgumentParser(description="GRIDGUARD AI - Forecast-Aware MAPPO Training")
    parser.add_argument("--mode", type=str, default="realistic", choices=["realistic", "oracle", "both"], help="Forecast mode")
    parser.add_argument("--episodes", type=int, default=50, help="Training episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--curriculum", action="store_true", default=True, help="Curriculum learning")
    args = parser.parse_args()

    if args.mode in ("realistic", "both"):
        train_forecast_mappo(mode="realistic", episodes=args.episodes, seed=args.seed, curriculum=args.curriculum)

    if args.mode in ("oracle", "both"):
        train_forecast_mappo(mode="oracle", episodes=args.episodes, seed=args.seed, curriculum=args.curriculum)


if __name__ == "__main__":
    main()
