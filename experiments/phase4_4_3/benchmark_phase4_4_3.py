"""
experiments/phase4_4_3/benchmark_phase4_4_3.py
==============================================
GRIDGUARD AI — Phase 4.4.3 Final Controlled Benchmark, Fallback Attribution,
and Policy Autonomy Evaluation.

Evaluates:
  - Heuristics: Greedy Controller, Rule-Based EMS, Lookahead Heuristic EMS
  - Prior RL: Reactive (4.1), Forecast-Aware (4.2), Robust Forecast (42, 101, 202)
  - Extended 4.4.3 Corrected Constrained MAPPO (Seeds 42, 101, 202):
      * Mode A: Intrinsic (adequacy_fallback=False)
      * Mode B: With Fallback (adequacy_fallback=True)
  - Extended 4.4.3 Tail-Risk Constrained MAPPO (Seeds 42, 101, 202):
      * Mode A: Intrinsic (adequacy_fallback=False)
      * Mode B: With Fallback (adequacy_fallback=True)

Generates:
  - episode_level_metrics.csv
  - controller_comparison.csv
  - fallback_attribution.csv
  - paired_statistical_comparisons.csv
  - policy_autonomy_metrics.csv
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "phase1_digital_twin") not in sys.path:
    sys.path.insert(0, str(ROOT / "phase1_digital_twin"))

from rl_environment.microgrid_parallel_env import MicrogridParallelEnv
from rl_environment.economic_accounting import compute_episode_economics, EconomicConfig
from rl_environment.action_space import parse_agent_action
from controllers.rule_based_ems import RuleBasedEMS
from controllers.greedy_controller import GreedyController
from controllers.mpc_controller import MPCController
from mappo.config import MAPPOConfig
from mappo.mappo_agent import MAPPOAgent
from experiments.phase4_4_2.benchmark_phase4_4_2 import compute_cvar

BASE_443 = ROOT / "experiments" / "phase4_4_3"
RES_443 = BASE_443 / "results"
RES_443.mkdir(parents=True, exist_ok=True)

PHYSICAL_VIOLATION_CODES = {
    "BATTERY_OVERCHARGE_PREVENTED",
    "BATTERY_MIN_SOC_PREVENTED",
    "BATTERY_RESERVE_PRESERVED",
    "GRID_OUTAGE_FORCED_ISLAND",
    "SYNCHRONIZATION_NOT_VERIFIED",
}

POLICY_ERROR_CODES = {
    "UNNECESSARY_GENERATOR_DISPATCH",
    "CRITICAL_LOAD_SHEDDING_REJECTED",
    "UNNECESSARY_RENEWABLE_CURTAILMENT_IN_DEFICIT",
}


def load_mappo_checkpoint(checkpoint_path: Path, forecast_mode: Optional[str] = "realistic") -> MAPPOAgent:
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
        if ag in ckpt.get("actors", {}):
            actor.load_state_dict(ckpt["actors"][ag])
            actor.eval()
    if "critic" in ckpt:
        agent.critic.load_state_dict(ckpt["critic"])
        agent.critic.eval()
    agent.eval()
    return agent


def run_episode_443(
    env: MicrogridParallelEnv,
    controller_name: str,
    controller_group: str,
    train_seed: str,
    controller_type: str,
    controller_obj: Any,
    scenario_id: str,
    eval_seed: int,
    adequacy_fallback: bool,
    econ_cfg: EconomicConfig,
) -> Dict[str, Any]:
    obs, _ = env.reset(seed=eval_seed, options={"scenario_id": scenario_id})
    if hasattr(controller_obj, "reset"):
        controller_obj.reset()

    total_proposed_actions = 0
    feasible_proposals = 0
    boundary_clipping_count = 0
    emergency_override_count = 0
    supply_adequacy_intervention_count = 0
    inaction_detected_steps = 0

    # Causal intervention attribution & autonomy counters
    interv_physical = 0
    interv_policy_err = 0
    interv_forecast_ctx = 0

    outage_steps = 0
    raw_outage_gen_active_steps = 0
    outage_onset_delay_steps = np.nan
    first_outage_step = None

    deficit_steps = 0
    raw_deficit_bat_dis_steps = 0

    storm_warn_steps = 0
    storm_precharge_hold_steps = 0

    healthy_grid_steps = 0
    raw_unneeded_gen_steps = 0

    high_crit_steps = 0
    high_crit_nom_mwh = 0.0
    high_crit_served_mwh = 0.0

    fallback_extra_supply_mwh = 0.0
    total_reward = 0.0
    done = False
    dt_h = env.dt_hours

    while env.agents and not done:
        step_idx = env.current_step
        hour = env.current_hour

        if controller_type in ("rule_based", "greedy", "lookahead_heuristic"):
            actions = controller_obj.select_actions(obs, env.episode_history[-1])
        elif controller_type == "mappo":
            actions, _ = controller_obj.select_actions(obs, deterministic=True)
        else:
            raise ValueError(f"Unknown controller_type: {controller_type}")

        obs, rewards, terms, truncs, infos = env.step(actions)
        total_reward += float(rewards.get("battery", 0.0))
        done = any(terms.values()) or any(truncs.values())

        st = env.episode_history[-1]
        diag = infos.get("battery", {}).get("shield_diagnostics", {})
        adeq = infos.get("battery", {}).get("adequacy_diagnostics", {})

        for ag, cls in diag.get("action_classification", {}).items():
            total_proposed_actions += 1
            if cls == "FEASIBLE":
                feasible_proposals += 1

        if diag.get("any_boundary", False):
            boundary_clipping_count += 1
        if diag.get("any_emergency", False):
            emergency_override_count += 1
        if adeq.get("adequacy_intervention", False):
            supply_adequacy_intervention_count += 1
        if adeq.get("inaction_detected", False):
            inaction_detected_steps += 1

        # Classify intervention causes
        viols = set(diag.get("violations", []))
        if viols & PHYSICAL_VIOLATION_CODES:
            interv_physical += 1
        if viols & POLICY_ERROR_CODES:
            interv_policy_err += 1

        is_forecast_frozen = hasattr(env.scenario, "is_forecast_frozen") and env.scenario.is_forecast_frozen(hour)
        has_sensor_noise = hasattr(env.scenario, "get_sensor_noise_profile") and env.scenario.get_sensor_noise_profile(hour)["gaussian_sigma"] > 0
        if (diag.get("any_intervention", False) or adeq.get("adequacy_intervention", False)) and (is_forecast_frozen or has_sensor_noise):
            interv_forecast_ctx += 1

        # Raw agent proposals vs physical state autonomy metrics
        raw_gen = float(parse_agent_action("generator", actions["generator"])["requested_power_mw"])
        raw_bat = float(parse_agent_action("battery", actions["battery"])["requested_power_mw"])

        grid_conn = bool(st["grid_connected"])
        is_storm = bool(st["is_storm_warning"])

        if not grid_conn:
            outage_steps += 1
            if first_outage_step is None:
                first_outage_step = step_idx
            if raw_gen > 0.04:
                raw_outage_gen_active_steps += 1
                if np.isnan(outage_onset_delay_steps):
                    outage_onset_delay_steps = float(step_idx - first_outage_step)

            # Check deficit when islanded
            if (st["nominal_total_mw"] - st["re_gen_mw"]) > 0.02:
                deficit_steps += 1
                if raw_bat < -0.02:
                    raw_deficit_bat_dis_steps += 1
        else:
            if is_storm:
                storm_warn_steps += 1
                if st["battery_soc"] >= 30.0 or raw_bat > 0.02:
                    storm_precharge_hold_steps += 1
            else:
                healthy_grid_steps += 1
                if raw_gen > 0.05:
                    raw_unneeded_gen_steps += 1

        if st["nominal_critical_mw"] > 0.35:
            high_crit_steps += 1
            high_crit_nom_mwh += float(st["nominal_critical_mw"] * dt_h)
            high_crit_served_mwh += float(st["served_critical_mw"] * dt_h)

        # Fallback energy delta approximation when adequacy fallback intervened
        if adequacy_fallback and adeq.get("adequacy_intervention", False):
            sh_acts = diag.get("executed_actions", actions)
            sh_bat = float(parse_agent_action("battery", sh_acts["battery"])["requested_power_mw"])
            actual_bat = float(st["battery_mw"])
            # Negative bat is discharge; extra discharge = max(0, (-actual_bat) - max(0, -sh_bat))
            extra_bat_dis = max(0.0, (-actual_bat) - max(0.0, -sh_bat))
            sh_gen = float(parse_agent_action("generator", sh_acts["generator"])["requested_power_mw"])
            actual_gen = float(st["generator_mw"])
            extra_gen_p = max(0.0, actual_gen - (sh_gen if sh_gen > 0.04 else 0.0))
            fallback_extra_supply_mwh += (extra_bat_dis + extra_gen_p) * dt_h

    hist_df = pd.DataFrame(env.episode_history)
    econ = compute_episode_economics(hist_df, dt_hours=dt_h, config=econ_cfg)
    steps_df = hist_df.iloc[1:] if len(hist_df) > 1 else hist_df

    avoidable_crit_ens = float(steps_df["avoidable_critical_shed_mw"].sum() * dt_h) if "avoidable_critical_shed_mw" in steps_df else 0.0
    unavoidable_crit_ens = float(steps_df["unavoidable_critical_shed_mw"].sum() * dt_h) if "unavoidable_critical_shed_mw" in steps_df else 0.0
    mean_bat_mw = float(steps_df["battery_mw"].abs().mean())
    mean_gen_mw = float(steps_df["generator_mw"].mean())

    pcc_v = steps_df["pcc_voltage_pu"].astype(float)
    min_v = float(steps_df["min_bus_voltage_pu"].min()) if "min_bus_voltage_pu" in steps_df else float(pcc_v.min())
    max_v = float(steps_df["max_bus_voltage_pu"].max()) if "max_bus_voltage_pu" in steps_df else float(pcc_v.max())
    volt_viols = int(((steps_df.get("min_bus_voltage_pu", pcc_v) < 0.95) | (steps_df.get("max_bus_voltage_pu", pcc_v) > 1.05)).sum())

    pf_conv_pct = float(steps_df["converged"].mean() * 100.0)
    feasible_pct = (feasible_proposals / total_proposed_actions * 100.0) if total_proposed_actions > 0 else 100.0

    return {
        "controller": controller_name,
        "controller_group": controller_group,
        "train_seed": train_seed,
        "adequacy_fallback": bool(adequacy_fallback),
        "scenario": scenario_id,
        "eval_seed": int(eval_seed),
        "reward": round(total_reward, 4),
        "unshielded_feasible_pct": round(feasible_pct, 2),
        "boundary_clipping_count": int(boundary_clipping_count),
        "emergency_override_count": int(emergency_override_count),
        "supply_adequacy_intervention_count": int(supply_adequacy_intervention_count),
        "inaction_detected_steps": int(inaction_detected_steps),
        "avoidable_critical_ens_mwh": round(avoidable_crit_ens, 6),
        "unavoidable_critical_ens_mwh": round(unavoidable_crit_ens, 6),
        "mean_battery_dispatch_mw": round(mean_bat_mw, 4),
        "mean_generator_dispatch_mw": round(mean_gen_mw, 4),
        "power_flow_converged_pct": round(pf_conv_pct, 2),
        "min_bus_voltage_pu": round(min_v, 4),
        "max_bus_voltage_pu": round(max_v, 4),
        "voltage_violations_count": int(volt_viols),
        # Autonomy & Attribution columns
        "outage_steps": int(outage_steps),
        "raw_outage_gen_prop_pct": round((raw_outage_gen_active_steps / outage_steps * 100.0) if outage_steps > 0 else np.nan, 2),
        "outage_gen_startup_delay_steps": float(outage_onset_delay_steps) if not np.isnan(outage_onset_delay_steps) else np.nan,
        "deficit_steps": int(deficit_steps),
        "raw_deficit_bat_dis_pct": round((raw_deficit_bat_dis_steps / deficit_steps * 100.0) if deficit_steps > 0 else np.nan, 2),
        "storm_warn_steps": int(storm_warn_steps),
        "storm_reserve_hold_pct": round((storm_precharge_hold_steps / storm_warn_steps * 100.0) if storm_warn_steps > 0 else np.nan, 2),
        "healthy_grid_steps": int(healthy_grid_steps),
        "unneeded_gen_prop_pct": round((raw_unneeded_gen_steps / healthy_grid_steps * 100.0) if healthy_grid_steps > 0 else 0.0, 2),
        "high_crit_served_pct": round((high_crit_served_mwh / high_crit_nom_mwh * 100.0) if high_crit_nom_mwh > 0 else np.nan, 2),
        "interv_physical_constraints": int(interv_physical),
        "interv_policy_decision_error": int(interv_policy_err),
        "interv_forecast_context": int(interv_forecast_ctx),
        "fallback_extra_supply_mwh": round(fallback_extra_supply_mwh, 6),
        **econ,
    }


def compute_summary_443(ep_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ctrl, adeq_fb), grp in ep_df.groupby(["controller", "adequacy_fallback"], sort=False):
        n = len(grp)

        def st(col: str, higher: bool = False):
            vals = grp[col].dropna().values.astype(float)
            if len(vals) == 0:
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            m = float(np.mean(vals))
            s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            med = float(np.median(vals))
            q25, q75 = np.percentile(vals, [25, 75])
            ci_h = float(stats.t.ppf(0.975, df=max(1, len(vals) - 1)) * (s / np.sqrt(len(vals)))) if len(vals) > 1 and s > 0 else 0.0
            return m, s, med, float(q75 - q25), m - ci_h, m + ci_h, compute_cvar(vals, 0.10, higher), compute_cvar(vals, 0.05, higher)

        sc = st("true_social_cost_usd", False)
        op = st("operating_cost_usd", False)
        ce = st("critical_ens_mwh", False)
        cr = st("critical_reliability_pct", True)
        tr = st("total_reliability_pct", True)

        rows.append({
            "controller": ctrl,
            "controller_group": grp["controller_group"].iloc[0],
            "train_seed": grp["train_seed"].iloc[0],
            "adequacy_fallback": adeq_fb,
            "episodes_n": n,
            "true_social_cost_mean": round(sc[0], 2),
            "true_social_cost_std": round(sc[1], 2),
            "true_social_cost_median": round(sc[2], 2),
            "true_social_cost_iqr": round(sc[3], 2),
            "true_social_cost_ci95_lo": round(sc[4], 2),
            "true_social_cost_ci95_hi": round(sc[5], 2),
            "true_social_cost_cvar10": round(sc[6], 2),
            "true_social_cost_cvar05": round(sc[7], 2),
            "operating_cost_mean": round(op[0], 2),
            "critical_ens_mean": round(ce[0], 4),
            "critical_ens_std": round(ce[1], 4),
            "critical_ens_cvar10": round(ce[6], 4),
            "critical_ens_cvar05": round(ce[7], 4),
            "critical_reliability_mean": round(cr[0], 2),
            "critical_reliability_std": round(cr[1], 2),
            "critical_reliability_ci95_lo": round(cr[4], 2),
            "critical_reliability_ci95_hi": round(cr[5], 2),
            "critical_reliability_cvar10": round(cr[6], 2),
            "total_reliability_mean": round(tr[0], 2),
            "flexible_ens_mean": round(float(grp["flexible_ens_mwh"].mean()), 4),
            "other_ens_mean": round(float(grp["other_ens_mwh"].mean()), 4),
            "total_ens_mean": round(float(grp["total_ens_mwh"].mean()), 4),
            "fuel_consumed_l_mean": round(float(grp["fuel_consumed_l"].mean()), 2),
            "battery_throughput_mean": round(float(grp["battery_throughput_mwh"].mean()), 4),
            "terminal_soc_mean": round(float(grp["terminal_battery_soc_pct"].mean()), 2),
            "unshielded_feasible_pct": round(float(grp["unshielded_feasible_pct"].mean()), 2),
            "boundary_clipping_mean": round(float(grp["boundary_clipping_count"].mean()), 2),
            "emergency_override_mean": round(float(grp["emergency_override_count"].mean()), 2),
            "adequacy_intervention_mean": round(float(grp["supply_adequacy_intervention_count"].mean()), 2),
            "inaction_steps_mean": round(float(grp["inaction_detected_steps"].mean()), 2),
            "avoidable_critical_ens_mean": round(float(grp["avoidable_critical_ens_mwh"].mean()), 4),
            "unavoidable_critical_ens_mean": round(float(grp["unavoidable_critical_ens_mwh"].mean()), 4),
            "raw_outage_gen_prop_pct": round(float(grp["raw_outage_gen_prop_pct"].mean()), 2),
            "outage_gen_startup_delay_steps": round(float(grp["outage_gen_startup_delay_steps"].mean()), 2),
            "raw_deficit_bat_dis_pct": round(float(grp["raw_deficit_bat_dis_pct"].mean()), 2),
            "storm_reserve_hold_pct": round(float(grp["storm_reserve_hold_pct"].mean()), 2),
            "unneeded_gen_prop_pct": round(float(grp["unneeded_gen_prop_pct"].mean()), 2),
            "high_crit_served_pct": round(float(grp["high_crit_served_pct"].mean()), 2),
            "interv_physical_mean": round(float(grp["interv_physical_constraints"].mean()), 2),
            "interv_policy_err_mean": round(float(grp["interv_policy_decision_error"].mean()), 2),
            "interv_forecast_ctx_mean": round(float(grp["interv_forecast_context"].mean()), 2),
            "fallback_extra_supply_mwh_mean": round(float(grp["fallback_extra_supply_mwh"].mean()), 4),
        })
    return pd.DataFrame(rows)


def compute_fallback_attribution(ep_df: pd.DataFrame) -> pd.DataFrame:
    """Exact paired marginal fallback attribution per (family, seed)."""
    rows = []
    for fam_prefix, label in [("Ext Corrected MAPPO", "Extended Corrected"), ("Ext Tail-Risk MAPPO", "Extended Tail-Risk")]:
        for seed in ["42", "101", "202"]:
            c_int = f"{fam_prefix} (Seed {seed}) [Intrinsic]"
            c_fb = f"{fam_prefix} (Seed {seed}) [+Fallback]"
            d_int = ep_df[ep_df["controller"] == c_int].sort_values(["scenario", "eval_seed"]).reset_index(drop=True)
            d_fb = ep_df[ep_df["controller"] == c_fb].sort_values(["scenario", "eval_seed"]).reset_index(drop=True)
            if d_int.empty or d_fb.empty:
                continue

            d_ce = d_fb["critical_ens_mwh"] - d_int["critical_ens_mwh"]
            d_cr = d_fb["critical_reliability_pct"] - d_int["critical_reliability_pct"]
            d_sc = d_fb["true_social_cost_usd"] - d_int["true_social_cost_usd"]
            d_fuel = d_fb["fuel_consumed_l"] - d_int["fuel_consumed_l"]
            d_bat = d_fb["battery_throughput_mwh"] - d_int["battery_throughput_mwh"]

            rows.append({
                "policy_family": label,
                "train_seed": seed,
                "episodes_n": len(d_int),
                "intrinsic_critical_ens_mwh": round(float(d_int["critical_ens_mwh"].mean()), 4),
                "fallback_critical_ens_mwh": round(float(d_fb["critical_ens_mwh"].mean()), 4),
                "delta_critical_ens_mwh": round(float(d_ce.mean()), 4),
                "intrinsic_critical_rel_pct": round(float(d_int["critical_reliability_pct"].mean()), 2),
                "fallback_critical_rel_pct": round(float(d_fb["critical_reliability_pct"].mean()), 2),
                "delta_critical_rel_pct": round(float(d_cr.mean()), 2),
                "intrinsic_social_cost_usd": round(float(d_int["true_social_cost_usd"].mean()), 2),
                "fallback_social_cost_usd": round(float(d_fb["true_social_cost_usd"].mean()), 2),
                "delta_social_cost_usd": round(float(d_sc.mean()), 2),
                "adequacy_interventions_per_ep": round(float(d_fb["supply_adequacy_intervention_count"].mean()), 2),
                "energy_dispatched_by_fallback_mwh": round(float(d_fb["fallback_extra_supply_mwh"].mean()), 4),
                "additional_fuel_l": round(float(d_fuel.mean()), 2),
                "additional_battery_throughput_mwh": round(float(d_bat.mean()), 4),
                "paired_p_critical_ens": float(stats.ttest_rel(d_fb["critical_ens_mwh"], d_int["critical_ens_mwh"]).pvalue),
                "paired_p_social_cost": float(stats.ttest_rel(d_fb["true_social_cost_usd"], d_int["true_social_cost_usd"]).pvalue),
            })
    return pd.DataFrame(rows)


def compute_paired_stats_443(ep_df: pd.DataFrame) -> pd.DataFrame:
    pairs = []
    for s in ["42", "101", "202"]:
        pairs.extend([
            (f"Ext Corrected MAPPO (Seed {s}) [Intrinsic]", "Rule-Based EMS"),
            (f"Ext Corrected MAPPO (Seed {s}) [Intrinsic]", "Greedy Controller"),
            (f"Ext Corrected MAPPO (Seed {s}) [Intrinsic]", f"Robust Forecast MAPPO (Seed {s})"),
            (f"Ext Corrected MAPPO (Seed {s}) [+Fallback]", "Rule-Based EMS"),
            (f"Ext Corrected MAPPO (Seed {s}) [+Fallback]", "Greedy Controller"),
            (f"Ext Tail-Risk MAPPO (Seed {s}) [Intrinsic]", f"Ext Corrected MAPPO (Seed {s}) [Intrinsic]"),
            (f"Ext Tail-Risk MAPPO (Seed {s}) [Intrinsic]", "Rule-Based EMS"),
            (f"Ext Tail-Risk MAPPO (Seed {s}) [+Fallback]", "Rule-Based EMS"),
            (f"Ext Tail-Risk MAPPO (Seed {s}) [+Fallback]", "Greedy Controller"),
        ])
    pairs.append(("Lookahead Heuristic EMS", "Rule-Based EMS"))

    metrics = [
        ("true_social_cost_usd", "True Social Cost ($)", False),
        ("critical_ens_mwh", "Critical ENS (MWh)", False),
        ("critical_reliability_pct", "Critical Reliability (%)", True),
        ("unshielded_feasible_pct", "Unshielded Feasible (%)", True),
        ("inaction_detected_steps", "Inaction Steps", False),
    ]

    rows = []
    for c1, c2 in pairs:
        d1 = ep_df[ep_df["controller"] == c1].sort_values(["scenario", "eval_seed"])
        d2 = ep_df[ep_df["controller"] == c2].sort_values(["scenario", "eval_seed"])
        if d1.empty or d2.empty or len(d1) != len(d2):
            continue
        n = len(d1)
        for col, label, higher in metrics:
            x = d1[col].values.astype(float)
            y = d2[col].values.astype(float)
            diff = x - y
            m_diff = float(np.mean(diff))
            s_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
            if s_diff > 1e-10:
                t_res = stats.ttest_rel(x, y)
                t_stat, t_p = float(t_res.statistic), float(t_res.pvalue)
                cohen_d = float(m_diff / s_diff)
                ci_h = float(stats.t.ppf(0.975, df=n - 1) * (s_diff / np.sqrt(n)))
            else:
                t_stat, t_p, cohen_d, ci_h = 0.0, 1.0, 0.0, 0.0
            try:
                w_res = stats.wilcoxon(x, y, zero_method="pratt") if not np.all(np.abs(diff) < 1e-12) else None
                w_stat, w_p = (float(w_res.statistic), float(w_res.pvalue)) if w_res else (0.0, 1.0)
            except Exception:
                w_stat, w_p = np.nan, np.nan

            rows.append({
                "controller_1": c1,
                "controller_2": c2,
                "metric": label,
                "episodes_n": n,
                "c1_mean": round(float(np.mean(x)), 4),
                "c2_mean": round(float(np.mean(y)), 4),
                "mean_diff_c1_minus_c2": round(m_diff, 4),
                "ci95_lower": round(m_diff - ci_h, 4),
                "ci95_upper": round(m_diff + ci_h, 4),
                "paired_t_stat": round(t_stat, 4),
                "paired_t_pvalue": t_p,
                "wilcoxon_w": round(w_stat, 2) if not np.isnan(w_stat) else np.nan,
                "wilcoxon_pvalue": w_p,
                "cohens_d": round(cohen_d, 4),
                "c1_cvar10": round(compute_cvar(x, 0.10, higher), 4),
                "c2_cvar10": round(compute_cvar(y, 0.10, higher), 4),
                "significant_p05": bool(t_p < 0.05),
            })
    return pd.DataFrame(rows)


def run_benchmark(eval_seeds: List[int] = [42, 101, 202, 303, 404], group_filter: str = "all"):
    scenarios = ["scenario_h1", "scenario_h2", "scenario_h3", "scenario_h4", "scenario_h5", "scenario_h6"]
    econ_cfg = EconomicConfig.from_yaml(BASE_443 / "configs" / "economics_v1.yaml")
    reward_cfg_path = str(BASE_443 / "configs" / "reward_v3_social.yaml")

    specs = []
    if group_filter in ("all", "heuristics"):
        specs.extend([
            ("Greedy Controller", "Heuristic", "deterministic", "greedy", GreedyController(), None, False),
            ("Rule-Based EMS", "Heuristic", "deterministic", "rule_based", RuleBasedEMS(), None, False),
            ("Lookahead Heuristic EMS", "Heuristic", "deterministic", "lookahead_heuristic", MPCController(horizon_steps=8, dt_hours=0.25, name="Lookahead Heuristic EMS"), None, False),
        ])
    if group_filter in ("all", "prior_rl"):
        specs.extend([
            ("Reactive MAPPO (Phase 4.1)", "Prior RL", "42", "mappo", ROOT / "experiments" / "phase4" / "diagnostics" / "checkpoints" / "best_model.pt", None, False),
            ("Forecast-Aware MAPPO (Phase 4.2)", "Prior RL", "42", "mappo", ROOT / "experiments" / "phase4" / "forecasting" / "checkpoints" / "realistic" / "best_model.pt", "realistic", False),
            ("Robust Forecast MAPPO (Seed 42)", "Prior RL", "42", "mappo", ROOT / "experiments" / "phase4" / "robustness" / "checkpoints" / "robust_seed42" / "best_model.pt", "realistic", False),
            ("Robust Forecast MAPPO (Seed 101)", "Prior RL", "101", "mappo", ROOT / "experiments" / "phase4" / "robustness" / "checkpoints" / "robust_seed101" / "best_model.pt", "realistic", False),
            ("Robust Forecast MAPPO (Seed 202)", "Prior RL", "202", "mappo", ROOT / "experiments" / "phase4" / "robustness" / "checkpoints" / "robust_seed202" / "best_model.pt", "realistic", False),
        ])
    if group_filter in ("all", "ext_corrected"):
        for seed in [42, 101, 202]:
            ckpt = BASE_443 / "checkpoints" / f"ext_corrected_seed{seed}" / "best_model.pt"
            if ckpt.exists():
                specs.append((f"Ext Corrected MAPPO (Seed {seed}) [Intrinsic]", "Ext Corrected (Intrinsic)", str(seed), "mappo", ckpt, "realistic", False))
                specs.append((f"Ext Corrected MAPPO (Seed {seed}) [+Fallback]", "Ext Corrected (+Fallback)", str(seed), "mappo", ckpt, "realistic", True))
    if group_filter in ("all", "ext_tail_risk"):
        for seed in [42, 101, 202]:
            ckpt = BASE_443 / "checkpoints" / f"ext_tail_risk_seed{seed}" / "best_model.pt"
            if ckpt.exists():
                specs.append((f"Ext Tail-Risk MAPPO (Seed {seed}) [Intrinsic]", "Ext Tail-Risk (Intrinsic)", str(seed), "mappo", ckpt, "realistic", False))
                specs.append((f"Ext Tail-Risk MAPPO (Seed {seed}) [+Fallback]", "Ext Tail-Risk (+Fallback)", str(seed), "mappo", ckpt, "realistic", True))

    models_dir = str(ROOT / "experiments" / "phase4" / "forecasting" / "models")
    env_cache: Dict[Tuple[Optional[str], bool], MicrogridParallelEnv] = {}
    all_records = []
    t_start = time.time()

    print(f"[*] Running Phase 4.4.3 Benchmark ({group_filter}): {len(specs)} configs x 6 scenarios x 5 seeds = {len(specs)*30} episodes", flush=True)
    for name, grp, tseed, ctype, obj_or_path, fmode, adeq_fb in specs:
        t_c = time.time()
        ctrl_obj = load_mappo_checkpoint(obj_or_path, fmode) if ctype == "mappo" else obj_or_path
        cache_key = (fmode, adeq_fb)
        if cache_key not in env_cache:
            env_cache[cache_key] = MicrogridParallelEnv(
                scenario_id="scenario_a",
                forecast_mode=fmode,
                forecast_models_dir=models_dir,
                reward_config_path=reward_cfg_path,
                max_steps=96,
                dt_minutes=15.0,
                enable_safety_shield=True,
                enable_adequacy_check=True,
                adequacy_fallback=adeq_fb,
            )
        env_inst = env_cache[cache_key]
        for scen in scenarios:
            for eseed in eval_seeds:
                rec = run_episode_443(
                    env=env_inst,
                    controller_name=name,
                    controller_group=grp,
                    train_seed=tseed,
                    controller_type=ctype,
                    controller_obj=ctrl_obj,
                    scenario_id=scen,
                    eval_seed=eseed,
                    adequacy_fallback=adeq_fb,
                    econ_cfg=econ_cfg,
                )
                all_records.append(rec)
        sub = [r for r in all_records if r["controller"] == name]
        print(
            f"  [+] {name:46s} | Social=${np.mean([r['true_social_cost_usd'] for r in sub]):7.1f} | "
            f"CritRel={np.mean([r['critical_reliability_pct'] for r in sub]):5.2f}% | "
            f"Feas={np.mean([r['unshielded_feasible_pct'] for r in sub]):5.1f}% | "
            f"Inact={np.mean([r['inaction_detected_steps'] for r in sub]):4.1f} | {time.time()-t_c:.1f}s",
            flush=True,
        )

    ep_df = pd.DataFrame(all_records)
    if group_filter != "all":
        ep_df.to_csv(RES_443 / f"episode_level_metrics_{group_filter}.csv", index=False)
        print(f"[✓] Group '{group_filter}' complete in {time.time()-t_start:.1f}s ({len(ep_df)} eps).", flush=True)
        return
    finalize_443_outputs(ep_df)


def finalize_443_outputs(ep_df: pd.DataFrame):
    ep_df.to_csv(RES_443 / "episode_level_metrics.csv", index=False)
    comp_df = compute_summary_443(ep_df)
    comp_df.to_csv(RES_443 / "controller_comparison.csv", index=False)
    fb_df = compute_fallback_attribution(ep_df)
    fb_df.to_csv(RES_443 / "fallback_attribution.csv", index=False)
    stats_df = compute_paired_stats_443(ep_df)
    stats_df.to_csv(RES_443 / "paired_statistical_comparisons.csv", index=False)
    print(f"[✓] Finalized Phase 4.4.3 outputs: {len(ep_df)} episodes, {len(comp_df)} configs.", flush=True)


def merge_groups():
    dfs = []
    for g in ["heuristics", "prior_rl", "ext_corrected", "ext_tail_risk"]:
        p = RES_443 / f"episode_level_metrics_{g}.csv"
        if p.exists():
            dfs.append(pd.read_csv(p))
    if not dfs:
        raise FileNotFoundError("No Phase 4.4.3 group CSVs found.")
    finalize_443_outputs(pd.concat(dfs, ignore_index=True))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, default="all")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.merge:
        merge_groups()
    else:
        run_benchmark(group_filter=args.group)
