"""
experiments/phase4/robustness/benchmark_robustness.py
============================================================
GRIDGUARD AI - Phase 4.3: Paired Multi-Seed Benchmark Evaluation

Evaluates all controllers on the Held-Out Unseen Test Suite (Scenarios H1 to H6):
  1. Rule-Based EMS
  2. Greedy Controller
  3. Lookahead MPC
  4. Reactive MAPPO (Phase 4.1)
  5. Forecast-Aware MAPPO (Phase 4.2 Realistic)
  6. Robust Forecast-Aware MAPPO (Phase 4.3 - Seed 42, 101, 202)
  7. Oracle Forecast MAPPO (Theoretical Upper Bound Reference)
  8. Ablation Models (No-Sensor-Noise, No-Uncertainty)

Calculates:
  - Mean, Std, Median, IQR, Worst-case, 95% Confidence Intervals
  - Conditional Value at Risk (CVaR 5%)
  - Recovery Time after disturbances
  - Emergency vs Boundary Safety Shield Interventions
  - Performance degradation relative to nominal conditions
============================================================
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase1_digital_twin"))

from rl_environment.microgrid_parallel_env import MicrogridParallelEnv
from controllers.rule_based_ems import RuleBasedEMS
from controllers.greedy_controller import GreedyController
from controllers.mpc_controller import MPCController
from mappo.config import MAPPOConfig
from mappo.mappo_agent import MAPPOAgent
from experiments.phase4.robustness.scenarios.robust_scenario_suite import HeldOutScenarios


def load_mappo_agent(checkpoint_path: Path, forecast_mode: Any = "realistic") -> MAPPOAgent:
    """Load trained MAPPO agent checkpoint."""
    models_dir = str(ROOT / "experiments" / "phase4" / "forecasting" / "models")
    dummy_env = MicrogridParallelEnv(
        scenario_id="scenario_a",
        forecast_mode=forecast_mode,
        forecast_models_dir=models_dir,
    )
    config = MAPPOConfig(
        forecast_mode=forecast_mode,
        forecast_models_dir=models_dir,
        device="cpu",
    )
    obs_dims = {ag: dummy_env.observation_space(ag).shape[0] for ag in dummy_env.possible_agents}
    agent = MAPPOAgent(
        config=config,
        obs_dims=obs_dims,
        action_spaces=dummy_env._action_spaces,
        state_dim=dummy_env._global_state_space.shape[0],
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for ag, actor in agent.actors.items():
        if ag in ckpt["actors"]:
            actor.load_state_dict(ckpt["actors"][ag])
            actor.eval()
    if "critic" in ckpt:
        agent.critic.load_state_dict(ckpt["critic"])
        agent.critic.eval()
    agent.eval()
    return agent


def compute_cvar(values: np.ndarray, alpha: float = 0.05, higher_is_better: bool = True) -> float:
    """Compute Conditional Value at Risk for the worst alpha percentile."""
    arr = np.sort(values)
    if higher_is_better:
        # Worst outcomes are the lowest values
        cutoff_idx = max(1, int(np.ceil(alpha * len(arr))))
        return float(np.mean(arr[:cutoff_idx]))
    else:
        # Worst outcomes are the highest values (e.g. cost, ENS)
        cutoff_idx = max(1, int(np.ceil((1.0 - alpha) * len(arr))))
        return float(np.mean(arr[cutoff_idx:]))


def run_single_episode(
    controller_type: str,
    controller_obj: Any,
    scenario_id: str,
    eval_seed: int,
    forecast_mode: str = "realistic",
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Execute a single paired episode and compute physical metrics."""
    env = MicrogridParallelEnv(
        scenario_id=scenario_id,
        forecast_mode=forecast_mode,
        max_steps=96,
        dt_minutes=15.0,
    )

    obs, infos = env.reset(seed=eval_seed, options={"scenario_id": scenario_id})
    dt_h = env.dt_hours

    # Safety shield breakdown counters
    emergency_interventions = 0
    boundary_interventions = 0

    done = False
    total_reward = 0.0

    while env.agents and not done:
        # Action selection
        if controller_type == "rule_based":
            actions = controller_obj.select_actions(obs, env.episode_history[-1])
        elif controller_type == "greedy":
            actions = controller_obj.select_actions(obs, env.episode_history[-1])
        elif controller_type == "mpc":
            actions = controller_obj.select_actions(obs, env.episode_history[-1])
        elif "mappo" in controller_type:
            actions, _ = controller_obj.select_actions(obs, deterministic=True)
        else:
            actions = {ag: np.zeros(env.action_space(ag).shape, dtype=np.float32) for ag in env.agents}

        obs, rewards, terms, truncs, infos = env.step(actions)
        total_reward += rewards.get("battery", 0.0)
        done = any(terms.values()) or any(truncs.values())

        # Inspect shield interventions
        st = env.episode_history[-1]
        diag = infos.get("battery", {}).get("shield_diagnostics", {})
        if diag.get("any_intervention", False):
            violations = diag.get("violations", [])
            for v in violations:
                if "CRITICAL" in v or "MIN_SOC" in v or "GRID_OUTAGE_FORCED" in v:
                    emergency_interventions += 1
                else:
                    boundary_interventions += 1

    df = pd.DataFrame(env.episode_history)

    # Compute operational metrics
    tot_crit_mwh = df["nominal_critical_mw"].sum() * dt_h
    served_crit_mwh = df["served_critical_mw"].sum() * dt_h
    shed_crit_mwh = df["shed_critical_mw"].sum() * dt_h
    shed_flex_mwh = df["shed_flexible_mw"].sum() * dt_h
    shed_other_mwh = df["shed_other_mw"].sum() * dt_h
    tot_ens_mwh = shed_crit_mwh + shed_flex_mwh + shed_other_mwh

    tot_fuel_l = df["total_fuel_l"].iloc[-1] if "total_fuel_l" in df else df["step_fuel_l"].sum()
    solar_curt = (df["solar_available_mw"] - df["solar_mw"]).clip(lower=0.0).sum() * dt_h
    wind_curt = (df["wind_available_mw"] - df["wind_mw"]).clip(lower=0.0).sum() * dt_h
    re_curt_mwh = solar_curt + wind_curt

    import_mwh = df["grid_import_mw"].clip(lower=0.0).sum() * dt_h
    export_mwh = (-df["grid_import_mw"].clip(upper=0.0)).sum() * dt_h
    operating_cost = (import_mwh * 120.0) - (export_mwh * 40.0) + (tot_fuel_l * 1.50)

    rel_pct = (served_crit_mwh / tot_crit_mwh * 100.0) if tot_crit_mwh > 0 else 100.0
    tot_interventions = int(df["any_intervention"].sum()) if "any_intervention" in df else 0

    # Calculate recovery time after disturbance ends
    # Disturbance ends when grid_available returns to True (if it was False)
    outage_steps = df.index[~df["grid_available"]].tolist()
    if outage_steps:
        last_outage_step = outage_steps[-1]
        post_outage = df.loc[last_outage_step:]
        # Recovery achieved when served_total_mw >= 0.99 * nominal_total_mw and grid_connected
        recovered = post_outage[(post_outage["served_total_mw"] >= 0.99 * post_outage["nominal_total_mw"]) & post_outage["grid_connected"]]
        if not recovered.empty:
            first_recovered_step = recovered.index[0]
            recovery_time_min = float(max(0, (first_recovered_step - last_outage_step) * 15.0))
        else:
            recovery_time_min = float((96 - last_outage_step) * 15.0)
    else:
        recovery_time_min = 0.0

    # Max voltage deviation
    v_dev = (df["pcc_voltage_pu"] - 1.0).abs().max()

    metrics = {
        "scenario": scenario_id,
        "eval_seed": eval_seed,
        "reward": round(total_reward, 2),
        "critical_reliability_pct": round(rel_pct, 2),
        "critical_ens_mwh": round(shed_crit_mwh, 4),
        "flexible_ens_mwh": round(shed_flex_mwh, 4),
        "total_ens_mwh": round(tot_ens_mwh, 4),
        "operating_cost_usd": round(operating_cost, 2),
        "fuel_consumed_l": round(tot_fuel_l, 2),
        "renewable_curtailment_mwh": round(re_curt_mwh, 4),
        "shield_interventions_total": tot_interventions,
        "emergency_interventions": emergency_interventions,
        "boundary_interventions": boundary_interventions,
        "recovery_time_min": recovery_time_min,
        "max_voltage_dev_pu": round(float(v_dev), 4),
        "power_flow_converged_pct": round(df["converged"].mean() * 100.0, 1),
    }
    return metrics, df


def run_comprehensive_benchmark(eval_seeds: List[int] = [42, 101, 202, 303, 404]) -> pd.DataFrame:
    """
    Run full paired benchmark across all controllers and held-out scenarios.
    """
    scenarios = ["scenario_h1", "scenario_h2", "scenario_h3", "scenario_h4", "scenario_h5", "scenario_h6"]
    results_dir = ROOT / "experiments" / "phase4" / "robustness" / "results"
    traj_dir = results_dir / "trajectories"
    results_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Baseline Controllers
    rule_based = RuleBasedEMS()
    greedy = GreedyController()
    mpc = MPCController(horizon_steps=8, dt_hours=0.25)

    # 2. Checkpoints
    ckpt_phase41_reactive = ROOT / "experiments" / "phase4" / "diagnostics" / "checkpoints" / "best_model.pt"
    ckpt_phase42_realistic = ROOT / "experiments" / "phase4" / "forecasting" / "checkpoints" / "realistic" / "best_model.pt"
    ckpt_phase42_oracle = ROOT / "experiments" / "phase4" / "forecasting" / "checkpoints" / "oracle" / "best_model.pt"
    ckpt_robust_seed42 = ROOT / "experiments" / "phase4" / "robustness" / "checkpoints" / "robust_seed42" / "best_model.pt"
    ckpt_robust_seed101 = ROOT / "experiments" / "phase4" / "robustness" / "checkpoints" / "robust_seed101" / "best_model.pt"
    ckpt_robust_seed202 = ROOT / "experiments" / "phase4" / "robustness" / "checkpoints" / "robust_seed202" / "best_model.pt"
    ckpt_ablation_noise = ROOT / "experiments" / "phase4" / "robustness" / "checkpoints" / "ablation_no_sensor_noise" / "best_model.pt"
    ckpt_ablation_uncert = ROOT / "experiments" / "phase4" / "robustness" / "checkpoints" / "ablation_no_uncertainty" / "best_model.pt"

    print("[*] Loading Learned Policy Checkpoints...")
    mappo_reactive = load_mappo_agent(ckpt_phase41_reactive, forecast_mode=None)
    mappo_forecast = load_mappo_agent(ckpt_phase42_realistic, forecast_mode="realistic")
    mappo_oracle = load_mappo_agent(ckpt_phase42_oracle, forecast_mode="oracle")
    mappo_robust_42 = load_mappo_agent(ckpt_robust_seed42, forecast_mode="realistic")
    mappo_robust_101 = load_mappo_agent(ckpt_robust_seed101, forecast_mode="realistic")
    mappo_robust_202 = load_mappo_agent(ckpt_robust_seed202, forecast_mode="realistic")
    mappo_abl_noise = load_mappo_agent(ckpt_ablation_noise, forecast_mode="realistic")
    mappo_abl_uncert = load_mappo_agent(ckpt_ablation_uncert, forecast_mode="realistic")

    controllers = [
        ("Rule-Based EMS", "rule_based", rule_based, "realistic"),
        ("Greedy Controller", "greedy", greedy, "realistic"),
        ("Lookahead MPC", "mpc", mpc, "realistic"),
        ("Reactive MAPPO (Phase 4.1)", "mappo", mappo_reactive, None),
        ("Forecast-Aware MAPPO (Phase 4.2)", "mappo", mappo_forecast, "realistic"),
        ("Robust Forecast MAPPO (Seed 42)", "mappo", mappo_robust_42, "realistic"),
        ("Robust Forecast MAPPO (Seed 101)", "mappo", mappo_robust_101, "realistic"),
        ("Robust Forecast MAPPO (Seed 202)", "mappo", mappo_robust_202, "realistic"),
        ("Oracle Forecast MAPPO (Ref)", "mappo", mappo_oracle, "oracle"),
        ("Ablation: No Sensor Noise", "mappo", mappo_abl_noise, "realistic"),
        ("Ablation: No Uncertainty", "mappo", mappo_abl_uncert, "realistic"),
    ]

    all_records = []
    print("\n" + "=" * 80)
    print("GRIDGUARD AI - Running Paired Robustness Benchmark Across Held-Out Scenarios H1-H6")
    print(f"Total Controllers: {len(controllers)} | Evaluation Seeds: {eval_seeds} (N={len(eval_seeds)})")
    print("=" * 80)

    start_time = time.time()
    for scen in scenarios:
        print(f"\n>>> Evaluating Held-Out {scen.upper()}...")
        for c_name, c_type, c_obj, fc_mode in controllers:
            for seed in eval_seeds:
                metrics, df_traj = run_single_episode(
                    controller_type=c_type,
                    controller_obj=c_obj,
                    scenario_id=scen,
                    eval_seed=seed,
                    forecast_mode=fc_mode,
                )
                metrics["controller"] = c_name
                all_records.append(metrics)

                # Save trajectory for seed 42
                if seed == 42:
                    c_tag = c_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").replace(":", "")
                    traj_path = traj_dir / f"{scen}_{c_tag}.csv"
                    df_traj.to_csv(traj_path, index=False)

            print(f"  [+] {c_name:35s} | Mean Rew: {np.mean([r['reward'] for r in all_records if r['controller']==c_name and r['scenario']==scen]):+7.2f} | CritRel: {np.mean([r['critical_reliability_pct'] for r in all_records if r['controller']==c_name and r['scenario']==scen]):5.1f}%")

    elapsed = time.time() - start_time
    df_all = pd.DataFrame(all_records)
    raw_csv = results_dir / "robustness_raw_benchmark_records.csv"
    df_all.to_csv(raw_csv, index=False)
    print(f"\n[+] Raw paired benchmark records saved to: {raw_csv} in {elapsed:.1f}s")

    # 3. Compute Statistical Summary Table (Mean, Std, Median, IQR, Worst-case, 95% CI, CVaR 5%)
    summary_rows = []
    for (scen, c_name), group in df_all.groupby(["scenario", "controller"]):
        rew_arr = group["reward"].values
        cost_arr = group["operating_cost_usd"].values
        rel_arr = group["critical_reliability_pct"].values
        ens_arr = group["critical_ens_mwh"].values
        interv_arr = group["shield_interventions_total"].values
        emerg_arr = group["emergency_interventions"].values
        bound_arr = group["boundary_interventions"].values
        recov_arr = group["recovery_time_min"].values

        n = len(rew_arr)
        rew_mean = float(np.mean(rew_arr))
        rew_std = float(np.std(rew_arr, ddof=1)) if n > 1 else 0.0
        rew_med = float(np.median(rew_arr))
        rew_iqr = float(np.percentile(rew_arr, 75) - np.percentile(rew_arr, 25))
        rew_worst = float(np.min(rew_arr))
        rew_ci95 = float(1.96 * rew_std / np.sqrt(n)) if n > 1 else 0.0
        rew_cvar5 = compute_cvar(rew_arr, alpha=0.05, higher_is_better=True)

        cost_mean = float(np.mean(cost_arr))
        cost_std = float(np.std(cost_arr, ddof=1)) if n > 1 else 0.0
        cost_worst = float(np.max(cost_arr))
        cost_cvar5 = compute_cvar(cost_arr, alpha=0.05, higher_is_better=False)

        rel_mean = float(np.mean(rel_arr))
        rel_worst = float(np.min(rel_arr))
        ens_mean = float(np.mean(ens_arr))
        ens_worst = float(np.max(ens_arr))

        summary_rows.append({
            "scenario": scen,
            "controller": c_name,
            "reward_mean": round(rew_mean, 2),
            "reward_std": round(rew_std, 2),
            "reward_median": round(rew_med, 2),
            "reward_iqr": round(rew_iqr, 2),
            "reward_worst": round(rew_worst, 2),
            "reward_ci95": round(rew_ci95, 2),
            "reward_cvar5": round(rew_cvar5, 2),
            "critical_reliability_pct": round(rel_mean, 2),
            "critical_reliability_worst": round(rel_worst, 2),
            "critical_ens_mean_mwh": round(ens_mean, 4),
            "critical_ens_worst_mwh": round(ens_worst, 4),
            "operating_cost_mean_usd": round(cost_mean, 2),
            "operating_cost_worst_usd": round(cost_worst, 2),
            "operating_cost_cvar5": round(cost_cvar5, 2),
            "interventions_total_mean": round(float(np.mean(interv_arr)), 1),
            "emergency_interventions_mean": round(float(np.mean(emerg_arr)), 1),
            "boundary_interventions_mean": round(float(np.mean(bound_arr)), 1),
            "recovery_time_min_mean": round(float(np.mean(recov_arr)), 1),
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = results_dir / "robustness_benchmark_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"[+] Statistical benchmark summary saved to: {summary_csv}")

    return df_summary


if __name__ == "__main__":
    df_res = run_comprehensive_benchmark()
    print("\nBenchmark Execution Complete.")
