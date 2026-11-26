"""
experiments/phase4_4_4/train_ablation.py
========================================
GRIDGUARD AI — Phase 4.4.4 2x2 Factorial Ablation Training Runner

Experiments:
  - baseline  (GenRem=False, BatRem=False)
  - gen_only  (GenRem=True,  BatRem=False)
  - bat_only  (GenRem=False, BatRem=True)
  - combined  (GenRem=True,  BatRem=True)

Seeds: [42, 101, 202] | Budget: 100 episodes / seed | Validation: FixedValidationSuite
Held-out H1-H6 remain strictly isolated.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "phase1_digital_twin") not in sys.path:
    sys.path.insert(0, str(ROOT / "phase1_digital_twin"))

from mappo.config import MAPPOConfig
from experiments.phase4_4_4.constrained_mappo_v3 import ConstrainedMAPPOTrainerV3

BASE_444 = ROOT / "experiments" / "phase4_4_4"

EXPERIMENTS = {
    "baseline": BASE_444 / "configs" / "ablation_baseline.yaml",
    "gen_only": BASE_444 / "configs" / "ablation_gen_only.yaml",
    "bat_only": BASE_444 / "configs" / "ablation_bat_only.yaml",
    "combined": BASE_444 / "configs" / "ablation_combined.yaml",
}


def train_single_ablation(
    experiment: str,
    seed: int,
    total_episodes: int = 100,
    val_interval_episodes: int = 20,
) -> Dict[str, Any]:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}")

    ablation_cfg = str(EXPERIMENTS[experiment])
    reward_cfg = str(BASE_444 / "configs" / "reward_v3_social.yaml")
    econ_cfg = str(BASE_444 / "configs" / "economics_v1.yaml")
    models_dir = str(ROOT / "experiments" / "phase4" / "forecasting" / "models")

    run_name = f"{experiment}_seed{seed}"
    ckpt_dir = BASE_444 / "checkpoints" / run_name
    log_dir = BASE_444 / "logs" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    config = MAPPOConfig(
        scenario_id="scenario_a",
        total_episodes=total_episodes,
        use_curriculum=True,
        init_bias_priors=True,
        reward_config_path=reward_cfg,
        seed=seed,
        forecast_mode="realistic",
        forecast_models_dir=models_dir,
        checkpoint_dir=str(ckpt_dir),
        log_dir=str(log_dir),
        domain_randomization=True,
        randomize_scenarios=True,
        device="cpu",
    )

    t0 = time.time()
    trainer = ConstrainedMAPPOTrainerV3(
        config=config,
        ablation_cfg_path=ablation_cfg,
        econ_cfg_path=econ_cfg,
        adaptive_multipliers=True,
        use_tail_risk=False,
        enable_adequacy_fallback_during_train=False,
    )
    summary = trainer.train(total_episodes=total_episodes, val_interval_episodes=val_interval_episodes)
    duration = time.time() - t0

    return {"run_name": run_name, "duration_s": round(duration, 2), **summary}


def merge_ablation_histories():
    results_dir = BASE_444 / "results"
    logs_dir = BASE_444 / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)

    train_dfs, val_dfs, mult_dfs = [], [], []
    for exp in ["baseline", "gen_only", "bat_only", "combined"]:
        for s in [42, 101, 202]:
            run_name = f"{exp}_seed{s}"
            t_csv = logs_dir / run_name / "training_episodes.csv"
            v_csv = logs_dir / run_name / "validation_selection_history.csv"
            m_csv = logs_dir / run_name / "multiplier_advantages_history.csv"
            if t_csv.exists():
                dt = pd.read_csv(t_csv)
                dt["run_name"] = run_name
                train_dfs.append(dt)
            if v_csv.exists():
                dv = pd.read_csv(v_csv)
                dv["run_name"] = run_name
                if t_csv.exists():
                    dt = pd.read_csv(t_csv)
                    dv["training_duration_s"] = round(float(dt["duration_s"].sum()), 2)
                val_dfs.append(dv)
            if m_csv.exists():
                dm = pd.read_csv(m_csv)
                dm["run_name"] = run_name
                mult_dfs.append(dm)

    if train_dfs:
        pd.concat(train_dfs, ignore_index=True).to_csv(results_dir / "training_histories.csv", index=False)
    if val_dfs:
        pd.concat(val_dfs, ignore_index=True).to_csv(results_dir / "validation_histories.csv", index=False)
    if mult_dfs:
        pd.concat(mult_dfs, ignore_index=True).to_csv(results_dir / "multiplier_histories.csv", index=False)
    print(f"[✓] Merged Phase 4.4.4 training/validation/multiplier histories into {results_dir}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None, choices=list(EXPERIMENTS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--val-interval", type=int, default=20)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    if args.merge:
        merge_ablation_histories()
    elif args.experiment:
        res = train_single_ablation(
            experiment=args.experiment,
            seed=args.seed,
            total_episodes=args.episodes,
            val_interval_episodes=args.val_interval,
        )
        print(
            f"[+] Completed {res['run_name']}: best_ep={res['best_checkpoint_episode']}, "
            f"val_score={res['best_val_score']:.2f}, duration={res['duration_s']}s",
            flush=True,
        )
    else:
        for exp in ["baseline", "gen_only", "bat_only", "combined"]:
            for s in [42, 101, 202]:
                train_single_ablation(exp, s, args.episodes, args.val_interval)
        merge_ablation_histories()
