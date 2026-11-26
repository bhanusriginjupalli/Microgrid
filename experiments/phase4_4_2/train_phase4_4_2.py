"""
experiments/phase4_4_2/train_phase4_4_2.py
==========================================
GRIDGUARD AI — Phase 4.4.2 Multi-Seed Training Orchestrator

Trains across 3 independent training seeds [42, 101, 202] for:
  - Corrected Constrained MAPPO (use_tail_risk=False)
  - Variational Tail-Risk Constrained MAPPO (use_tail_risk=True)

Uses FixedValidationSuite for multi-objective checkpoint selection,
keeping HeldOutScenarios H1-H6 completely isolated.
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


def train_variant(
    variant_name: str,
    seed: int,
    total_episodes: int = 48,
    val_interval_episodes: int = 8,
    use_tail_risk: bool = False,
) -> Dict[str, Any]:
    base_dir = ROOT / "experiments" / "phase4_4_2"
    reward_cfg = str(base_dir / "configs" / "reward_v3_social.yaml")
    constrained_cfg = str(base_dir / "configs" / "constrained_v2.yaml")
    econ_cfg = str(base_dir / "configs" / "economics_v1.yaml")
    models_dir = str(ROOT / "experiments" / "phase4" / "forecasting" / "models")

    ckpt_dir = base_dir / "checkpoints" / variant_name
    log_dir = base_dir / "logs" / variant_name
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
        "validation_history": trainer.validation_history,
        "multiplier_history": trainer.multiplier_history,
    }


def run_all_phase4_4_2_training(total_episodes: int = 48, val_interval: int = 8):
    base_dir = ROOT / "experiments" / "phase4_4_2"
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    seeds = [42, 101, 202]
    runs = []
    all_val_rows: List[Dict[str, Any]] = []
    all_mult_rows: List[Dict[str, Any]] = []

    for use_tail, prefix in [(False, "corrected_constrained"), (True, "tail_risk_constrained")]:
        for s in seeds:
            name = f"{prefix}_seed{s}"
            res = train_variant(
                variant_name=name,
                seed=s,
                total_episodes=total_episodes,
                val_interval_episodes=val_interval,
                use_tail_risk=use_tail,
            )
            runs.append(res)
            for row in res["validation_history"]:
                r_copy = dict(row)
                r_copy["variant_name"] = name
                r_copy["seed"] = s
                r_copy["use_tail_risk"] = use_tail
                r_copy["total_episodes_trained"] = res["total_episodes"]
                r_copy["total_env_steps"] = res["total_steps"]
                r_copy["training_duration_s"] = res["duration_s"]
                all_val_rows.append(r_copy)
            for row in res["multiplier_history"]:
                m_copy = dict(row)
                m_copy["variant_name"] = name
                m_copy["seed"] = s
                m_copy["use_tail_risk"] = use_tail
                all_mult_rows.append(m_copy)

    pd.DataFrame(all_val_rows).to_csv(results_dir / "training_validation_history.csv", index=False)
    pd.DataFrame(all_mult_rows).to_csv(results_dir / "constraint_multiplier_history.csv", index=False)
    print(f"\n[✓] Phase 4.4.2 multi-seed training complete. History saved to {results_dir}")


def aggregate_training_histories(base_dir: Path = ROOT / "experiments" / "phase4_4_2"):
    """Merge per-variant validation and multiplier logs into unified Phase 4.4.2 CSVs."""
    results_dir = base_dir / "results"
    logs_dir = base_dir / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_val = []
    all_mult = []
    configs = [
        ("corrected_constrained_seed42", 42, False),
        ("corrected_constrained_seed101", 101, False),
        ("corrected_constrained_seed202", 202, False),
        ("tail_risk_constrained_seed42", 42, True),
        ("tail_risk_constrained_seed101", 101, True),
        ("tail_risk_constrained_seed202", 202, True),
    ]
    for name, s, tail in configs:
        v_csv = logs_dir / name / "validation_selection_history.csv"
        m_csv = logs_dir / name / "multiplier_advantages_history.csv"
        t_csv = logs_dir / name / "training_episodes.csv"
        if v_csv.exists():
            df_v = pd.read_csv(v_csv)
            df_t = pd.read_csv(t_csv) if t_csv.exists() else pd.DataFrame()
            df_v["variant_name"] = name
            df_v["seed"] = s
            df_v["use_tail_risk"] = tail
            df_v["total_episodes_trained"] = len(df_t)
            df_v["total_env_steps"] = len(df_t) * 96
            df_v["training_duration_s"] = round(float(df_t["duration_s"].sum()), 2) if "duration_s" in df_t else 0.0
            all_val.append(df_v)
        if m_csv.exists():
            df_m = pd.read_csv(m_csv)
            df_m["variant_name"] = name
            df_m["seed"] = s
            df_m["use_tail_risk"] = tail
            all_mult.append(df_m)

    if all_val:
        pd.concat(all_val, ignore_index=True).to_csv(results_dir / "training_validation_history.csv", index=False)
    if all_mult:
        pd.concat(all_mult, ignore_index=True).to_csv(results_dir / "constraint_multiplier_history.csv", index=False)
    print(f"[+] Aggregated training & multiplier histories to {results_dir}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--val-interval", type=int, default=12)
    parser.add_argument("--smoke", action="store_true", help="Run fast 4-episode smoke test")
    parser.add_argument("--single-variant", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tail-risk", action="store_true")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        print("[*] Running 4-episode Phase 4.4.2 Training Smoke Test...", flush=True)
        res = train_variant("smoke_test_seed42", seed=42, total_episodes=4, val_interval_episodes=4, use_tail_risk=False)
        print(f"[+] Smoke test passed: {res}", flush=True)
    elif args.merge:
        aggregate_training_histories()
    elif args.single_variant:
        res = train_variant(
            variant_name=args.single_variant,
            seed=args.seed,
            total_episodes=args.episodes,
            val_interval_episodes=args.val_interval,
            use_tail_risk=args.tail_risk,
        )
        print(f"[+] Finished {args.single_variant}: best_ep={res['best_checkpoint_episode']}, val_score={res['best_val_score']}", flush=True)
    else:
        run_all_phase4_4_2_training(total_episodes=args.episodes, val_interval=args.val_interval)
        aggregate_training_histories()
