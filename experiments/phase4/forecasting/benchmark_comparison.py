"""
experiments/phase4/forecasting/benchmark_comparison.py
============================================================
Comprehensive controlled benchmarking of 6 microgrid controllers across
all 6 disturbance scenarios (A through F):
  1. Rule-Based EMS
  2. Greedy Controller
  3. Lookahead Heuristic MPC
  4. Reactive MAPPO (Phase 4.1)
  5. Forecast-Aware MAPPO (Mode 2 Realistic Causal)
  6. Forecast-Aware MAPPO (Mode 1 Oracle Upper Bound)
Extracts key comparative metrics including proactive control metrics:
  - Battery pre-charging SOC prior to outage onset
  - Generator anticipatory startup lead time
============================================================
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
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
from mappo.checkpoint_manager import CheckpointManager


BENCHMARK_SCENARIOS = [
    "scenario_a",  # Normal operation
    "scenario_b",  # Solar collapse
    "scenario_c",  # Grid outage (2h)
    "scenario_d",  # Severe storm (multi-stage)
    "scenario_e",  # Prolonged outage (6h)
    "scenario_f",  # Grid restoration & sync
]

REWARD_CONFIG = str(ROOT / "experiments" / "phase4" / "diagnostics" / "configs" / "reward_v2_calibrated.yaml")
MODELS_DIR = str(ROOT / "experiments" / "phase4" / "forecasting" / "models")


def run_heuristic_controller(
    controller_type: str,
    scenario_id: str,
    seed: int = 42,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Run an episode with a baseline controller (Rule-Based, Greedy, or MPC)."""
    env = MicrogridParallelEnv(
        scenario_id=scenario_id,
        max_steps=96,
        dt_minutes=15.0,
        enable_safety_shield=True,
        reward_config_path=REWARD_CONFIG,
        random_seed=seed,
    )
    obs, infos = env.reset(seed=seed)

    if controller_type == "Rule-Based EMS":
        ctrl = RuleBasedEMS()
    elif controller_type == "Greedy":
        ctrl = GreedyController()
    elif controller_type == "Lookahead MPC":
        ctrl = MPCController(horizon_steps=8, dt_hours=0.25)
    else:
        raise ValueError(f"Unknown controller: {controller_type}")

    total_reward = 0.0
    step_records = []

    while env.agents:
        last_state = env.episode_history[-1]
        actions = ctrl.select_actions(obs, last_state)
        next_obs, rewards, terminations, truncations, next_infos = env.step(actions)
        total_reward += float(rewards["renewable"])

        step_state = env.episode_history[-1]
        step_records.append(step_state)
        obs = next_obs

    df = pd.DataFrame(step_records)
    metrics = extract_episode_metrics(df, total_reward, scenario_id, controller_type)
    return metrics, df


def load_mappo_agent(
    checkpoint_path: Path,
    forecast_mode: Optional[str] = None,
) -> MAPPOAgent:
    """Load trained MAPPO actor networks from checkpoint."""
    dummy_env = MicrogridParallelEnv(
        scenario_id="scenario_a",
        forecast_mode=forecast_mode,
        forecast_models_dir=MODELS_DIR,
    )
    config = MAPPOConfig(
        forecast_mode=forecast_mode,
        forecast_models_dir=MODELS_DIR,
        device="cpu",
    )
    obs_dims = {ag: dummy_env.observation_space(ag).shape[0] for ag in dummy_env.possible_agents}
    agent = MAPPOAgent(
        config=config,
        obs_dims=obs_dims,
        action_spaces=dummy_env._action_spaces,
        state_dim=dummy_env._global_state_space.shape[0],
    )
    ckpt_mgr = CheckpointManager(checkpoint_path.parent)
    ckpt_mgr.load_checkpoint(checkpoint_path, agent, load_optimizers=False)
    agent.eval()
    return agent


def run_mappo_controller(
    agent: MAPPOAgent,
    controller_name: str,
    scenario_id: str,
    forecast_mode: Optional[str] = None,
    seed: int = 42,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Run an episode with a MAPPO agent."""
    env = MicrogridParallelEnv(
        scenario_id=scenario_id,
        max_steps=96,
        dt_minutes=15.0,
        enable_safety_shield=True,
        reward_config_path=REWARD_CONFIG,
        forecast_mode=forecast_mode,
        forecast_models_dir=MODELS_DIR,
        random_seed=seed,
    )
    obs, infos = env.reset(seed=seed)
    total_reward = 0.0
    step_records = []

    while env.agents:
        actions, _ = agent.select_actions(obs, deterministic=True)
        next_obs, rewards, terminations, truncations, next_infos = env.step(actions)
        total_reward += float(rewards["renewable"])

        step_state = env.episode_history[-1]
        step_records.append(step_state)
        obs = next_obs

    df = pd.DataFrame(step_records)
    metrics = extract_episode_metrics(df, total_reward, scenario_id, controller_name)
    return metrics, df


def extract_episode_metrics(
    df: pd.DataFrame,
    total_reward: float,
    scenario_id: str,
    controller_name: str,
) -> Dict[str, Any]:
    """Compute standard metrics and proactive decision indicators."""
    dt_h = 0.25
    tot_crit_mwh = df["nominal_critical_mw"].sum() * dt_h
    served_crit_mwh = df["served_critical_mw"].sum() * dt_h
    shed_crit_mwh = df["shed_critical_mw"].sum() * dt_h
    shed_flex_mwh = df["shed_flexible_mw"].sum() * dt_h
    tot_fuel_l = df["total_fuel_l"].iloc[-1] if "total_fuel_l" in df else df["step_fuel_l"].sum()

    solar_curt = (df["solar_available_mw"] - df["solar_mw"]).clip(lower=0.0).sum() * dt_h
    wind_curt = (df["wind_available_mw"] - df["wind_mw"]).clip(lower=0.0).sum() * dt_h
    re_curt_mwh = solar_curt + wind_curt

    import_mwh = df["grid_import_mw"].clip(lower=0.0).sum() * dt_h
    export_mwh = (-df["grid_import_mw"].clip(upper=0.0)).sum() * dt_h
    operating_cost = (import_mwh * 120.0) - (export_mwh * 40.0) + (tot_fuel_l * 1.50)

    rel_pct = (served_crit_mwh / tot_crit_mwh * 100.0) if tot_crit_mwh > 0 else 100.0
    interventions = int(df["any_intervention"].sum()) if "any_intervention" in df else 0

    # Proactive metrics:
    # 1. Pre-outage battery SOC (SOC right before grid availability transitions to False)
    outage_steps = df.index[~df["grid_available"]].tolist()
    if outage_steps and outage_steps[0] > 0:
        first_outage_idx = outage_steps[0]
        pre_outage_soc = float(df.loc[first_outage_idx - 1, "battery_soc"])
    else:
        pre_outage_soc = float(df["battery_soc"].iloc[0])

    # 2. Generator lead time (steps/minutes prior to outage that generator started)
    gen_running_steps = df.index[df["generator_state"].isin(["STARTING", "RUNNING"])].tolist()
    if outage_steps and gen_running_steps:
        first_outage_step = outage_steps[0]
        first_gen_step = gen_running_steps[0]
        gen_lead_min = max(0.0, (first_outage_step - first_gen_step) * 15.0)
    else:
        gen_lead_min = 0.0

    return {
        "scenario": scenario_id,
        "controller": controller_name,
        "reward": round(total_reward, 2),
        "critical_reliability_pct": round(rel_pct, 2),
        "critical_ens_mwh": round(shed_crit_mwh, 4),
        "flexible_ens_mwh": round(shed_flex_mwh, 4),
        "operating_cost_usd": round(operating_cost, 2),
        "fuel_consumed_l": round(tot_fuel_l, 2),
        "renewable_curtailment_mwh": round(re_curt_mwh, 4),
        "shield_interventions": interventions,
        "pre_outage_battery_soc": round(pre_outage_soc, 1),
        "gen_lead_time_min": round(gen_lead_min, 1),
    }


def run_full_benchmark_comparison() -> pd.DataFrame:
    """Run all 6 controllers across all 6 scenarios."""
    output_dir = ROOT / "experiments" / "phase4" / "forecasting" / "results"
    traj_dir = output_dir / "trajectories"
    output_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint paths
    reactive_ckpt = ROOT / "experiments" / "phase4" / "diagnostics" / "checkpoints" / "best_model.pt"
    realistic_ckpt = ROOT / "experiments" / "phase4" / "forecasting" / "checkpoints" / "realistic" / "best_model.pt"
    oracle_ckpt = ROOT / "experiments" / "phase4" / "forecasting" / "checkpoints" / "oracle" / "best_model.pt"

    # Preload MAPPO agents
    print("Loading MAPPO agents...")
    reactive_agent = load_mappo_agent(reactive_ckpt, forecast_mode=None)
    print("  [+] Reactive MAPPO loaded.")

    realistic_agent = load_mappo_agent(realistic_ckpt, forecast_mode="realistic")
    print("  [+] Forecast-Aware MAPPO (Realistic) loaded.")

    oracle_agent = None
    if oracle_ckpt.exists():
        oracle_agent = load_mappo_agent(oracle_ckpt, forecast_mode="oracle")
        print("  [+] Forecast-Aware MAPPO (Oracle) loaded.")

    records = []

    print("\nExecuting controlled benchmark suite across Scenarios A-F...")
    for scen in BENCHMARK_SCENARIOS:
        print(f"\n---> Benchmarking {scen.upper()} <---")

        # 1. Rule-Based EMS
        m_rb, df_rb = run_heuristic_controller("Rule-Based EMS", scen)
        df_rb.to_csv(traj_dir / f"{scen}_RuleBasedEMS.csv", index=False)
        records.append(m_rb)
        print(f"  Rule-Based EMS     | Reward: {m_rb['reward']:+7.2f} | Cost: ${m_rb['operating_cost_usd']:7.2f} | Rel: {m_rb['critical_reliability_pct']:.1f}% | Pre-SOC: {m_rb['pre_outage_battery_soc']:.1f}%")

        # 2. Greedy Controller
        m_gr, df_gr = run_heuristic_controller("Greedy", scen)
        df_gr.to_csv(traj_dir / f"{scen}_Greedy.csv", index=False)
        records.append(m_gr)
        print(f"  Greedy             | Reward: {m_gr['reward']:+7.2f} | Cost: ${m_gr['operating_cost_usd']:7.2f} | Rel: {m_gr['critical_reliability_pct']:.1f}% | Pre-SOC: {m_gr['pre_outage_battery_soc']:.1f}%")

        # 3. Lookahead Heuristic MPC
        m_mpc, df_mpc = run_heuristic_controller("Lookahead MPC", scen)
        df_mpc.to_csv(traj_dir / f"{scen}_LookaheadMPC.csv", index=False)
        records.append(m_mpc)
        print(f"  Lookahead MPC      | Reward: {m_mpc['reward']:+7.2f} | Cost: ${m_mpc['operating_cost_usd']:7.2f} | Rel: {m_mpc['critical_reliability_pct']:.1f}% | Pre-SOC: {m_mpc['pre_outage_battery_soc']:.1f}%")

        # 4. Reactive MAPPO (Phase 4.1)
        m_reac, df_reac = run_mappo_controller(reactive_agent, "Reactive MAPPO", scen, forecast_mode=None)
        df_reac.to_csv(traj_dir / f"{scen}_ReactiveMAPPO.csv", index=False)
        records.append(m_reac)
        print(f"  Reactive MAPPO     | Reward: {m_reac['reward']:+7.2f} | Cost: ${m_reac['operating_cost_usd']:7.2f} | Rel: {m_reac['critical_reliability_pct']:.1f}% | Pre-SOC: {m_reac['pre_outage_battery_soc']:.1f}%")

        # 5. Forecast-Aware MAPPO (Realistic)
        m_real, df_real = run_mappo_controller(realistic_agent, "Forecast-Aware MAPPO (Realistic)", scen, forecast_mode="realistic")
        df_real.to_csv(traj_dir / f"{scen}_ForecastMAPPO_Realistic.csv", index=False)
        records.append(m_real)
        print(f"  Forecast-MAPPO Real| Reward: {m_real['reward']:+7.2f} | Cost: ${m_real['operating_cost_usd']:7.2f} | Rel: {m_real['critical_reliability_pct']:.1f}% | Pre-SOC: {m_real['pre_outage_battery_soc']:.1f}%")

        # 6. Forecast-Aware MAPPO (Oracle)
        if oracle_agent is not None:
            m_ora, df_ora = run_mappo_controller(oracle_agent, "Forecast-Aware MAPPO (Oracle)", scen, forecast_mode="oracle")
            df_ora.to_csv(traj_dir / f"{scen}_ForecastMAPPO_Oracle.csv", index=False)
            records.append(m_ora)
            print(f"  Forecast-MAPPO Orac| Reward: {m_ora['reward']:+7.2f} | Cost: ${m_ora['operating_cost_usd']:7.2f} | Rel: {m_ora['critical_reliability_pct']:.1f}% | Pre-SOC: {m_ora['pre_outage_battery_soc']:.1f}%")

    comp_df = pd.DataFrame(records)
    csv_path = output_dir / "forecast_marl_comparison.csv"
    comp_df.to_csv(csv_path, index=False)
    print(f"\n[+] Full Benchmark Comparison saved to: {csv_path}")

    # Summary pivot table
    pivot_rew = comp_df.pivot(index="scenario", columns="controller", values="reward")
    pivot_cost = comp_df.pivot(index="scenario", columns="controller", values="operating_cost_usd")
    pivot_soc = comp_df.pivot(index="scenario", columns="controller", values="pre_outage_battery_soc")

    print("\n" + "=" * 78)
    print("REWARD COMPARISON ACROSS CONTROLLERS")
    print("=" * 78)
    print(pivot_rew.to_string())

    print("\n" + "=" * 78)
    print("OPERATING COST ($) COMPARISON ACROSS CONTROLLERS")
    print("=" * 78)
    print(pivot_cost.to_string())

    print("\n" + "=" * 78)
    print("PRE-OUTAGE BATTERY SOC (%)")
    print("=" * 78)
    print(pivot_soc.to_string())

    return comp_df


if __name__ == "__main__":
    run_full_benchmark_comparison()
