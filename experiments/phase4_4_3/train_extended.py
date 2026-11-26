"""
experiments/phase4_4_3/train_extended.py
========================================
GRIDGUARD AI — Phase 4.4.3: Extended Multi-Seed Training (100 Episodes / Seed)

Trains Corrected Constrained MAPPO and Tail-Risk Constrained MAPPO across
independent seeds [42, 101, 202] for 100 episodes per seed, using
FixedValidationSuite for multi-objective checkpoint selection while keeping
Phase 4.4.2 checkpoints untouched and HeldOutScenarios H1-H6 isolated.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "phase1_digital_twin") not in sys.path:
    sys.path.insert(0, str(ROOT / "phase1_digital_twin"))

from mappo.config import MAPPOConfig
from experiments.phase4_4_2.constrained_mappo_v2 import ConstrainedMAPPOTrainerV2


BASE_443 = ROOT / "experiments" / "phase4_4_3"


def train_single_extended(
    variant_name: str,
    seed: int,
    total_episodes: int = 100,
    val_interval_episodes: int = 20,
    use_tail_risk: bool = False,
) -> Dict[str, Any]:
    reward_cfg = str(BASE_443 / "configs" / "reward_v3_social.yaml")
    constrained_cfg = str(BASE_443 / "configs" / "constrained_v3.yaml")
    econ_cfg = str(BASE_443 / "configs" / "economics_v1.yaml")
    models_dir = str(ROOT / "experiments" / "phase4" / "forecasting" / "models")

    ckpt_dir = BASE_443 / "checkpoints" / variant_name
    log_dir = BASE_443 / "logs" / variant_name
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
    trainer = ConstrainedMAPPOTrainerV2(
        config=config,
        constrained_cfg_path=constrained_cfg,
        econ_cfg_path=econ_cfg,
        adaptive_multipliers=True,
        use_tail_risk=use_tail_risk,
        enable_adequacy_fallback_during_train=False,
    )
    summary = trainer.train(total_episodes=total_episodes, val_interval_episodes=val_interval_episodes)
    duration = time.time() - t0

    return {
        "variant_name": variant_name,
        "seed": seed,
        "use_tail_risk": use_tail_risk,
        "duration_s": round(duration, 2),
        **summary,
    }


def merge_extended_histories():
    results_dir = BASE_443 / "results"
    logs_dir = BASE_443 / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("ext_corrected_seed42", 42, False),
        ("ext_corrected_seed101", 101, False),
        ("ext_corrected_seed202", 202, False),
        ("ext_tail_risk_seed42", 42, True),
        ("ext_tail_risk_seed101", 101, True),
        ("ext_tail_risk_seed202", 202, True),
    ]

    val_dfs = []
    mult_dfs = []
    ep_dfs = []

    for name, s, tail in configs:
        v_csv = logs_dir / name / "validation_selection_history.csv"
        m_csv = logs_dir / name / "multiplier_advantages_history.csv"
        t_csv = logs_dir / name / "training_episodes.csv"

        if v_csv.exists():
            dv = pd.read_csv(v_csv)
            dt = pd.read_csv(t_csv) if t_csv.exists() else pd.DataFrame()
            dv["variant_name"] = name
            dv["seed"] = s
            dv["use_tail_risk"] = tail
            dv["total_episodes_trained"] = len(dt)
            dv["total_env_steps"] = len(dt) * 96
            dv["training_duration_s"] = round(float(dt["duration_s"].sum()), 2) if "duration_s" in dt else 0.0
            val_dfs.append(dv)
        if m_csv.exists():
            dm = pd.read_csv(m_csv)
            dm["variant_name"] = name
            dm["seed"] = s
            dm["use_tail_risk"] = tail
            mult_dfs.append(dm)
        if t_csv.exists():
            dt = pd.read_csv(t_csv)
            dt["variant_name"] = name
            dt["seed"] = s
            dt["use_tail_risk"] = tail
            ep_dfs.append(dt)

    if val_dfs:
        pd.concat(val_dfs, ignore_index=True).to_csv(results_dir / "extended_validation_history.csv", index=False)
    if mult_dfs:
        pd.concat(mult_dfs, ignore_index=True).to_csv(results_dir / "extended_multiplier_history.csv", index=False)
    if ep_dfs:
        pd.concat(ep_dfs, ignore_index=True).to_csv(results_dir / "extended_training_history.csv", index=False)

    print(f"[✓] Merged extended Phase 4.4.3 training histories into {results_dir}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--single-variant", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--val-interval", type=int, default=20)
    parser.add_argument("--tail-risk", action="store_true")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    if args.merge:
        merge_extended_histories()
    elif args.single_variant:
        res = train_single_extended(
            variant_name=args.single_variant,
            seed=args.seed,
            total_episodes=args.episodes,
            val_interval_episodes=args.val_interval,
            use_tail_risk=args.tail_risk,
        )
        print(
            f"[+] Completed {args.single_variant}: best_ep={res['best_checkpoint_episode']}, "
            f"val_score={res['best_val_score']:.2f}, duration={res['duration_s']}s",
            flush=True,
        )
    else:
        for tail, prefix in [(False, "ext_corrected"), (True, "ext_tail_risk")]:
            for s in [42, 101, 202]:
                train_single_extended(
                    variant_name=f"{prefix}_seed{s}",
                    seed=s,
                    total_episodes=args.episodes,
                    val_interval_episodes=args.val_interval,
                    use_tail_risk=tail,
                )
        merge_extended_histories()
