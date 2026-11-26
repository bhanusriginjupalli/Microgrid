"""
experiments/phase4_4_4/benchmark_phase4_4_4.py
==============================================
GRIDGUARD AI — Phase 4.4.4 Held-Out Benchmark, Autonomy Telemetry,
Fallback Attribution, and Paired Statistical Analysis.

Evaluates identical (H1-H6 x [42, 101, 202, 303, 404]) realizations across:
  - Heuristics: Greedy Controller, Rule-Based EMS, Lookahead Heuristic EMS
  - Prior RL: Reactive (4.1), Forecast-Aware (4.2), Robust Forecast (42, 101, 202)
  - Phase 4.4.3 Extended: Corrected & Tail-Risk (42, 101, 202) [Intrinsic & +Hybrid]
  - Phase 4.4.4 Ablation: Baseline, Gen-Only, Bat-Only, Combined (42, 101, 202) [Intrinsic & +Hybrid]
"""

import sys
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

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
from experiments.phase4_4_2.benchmark_phase4_4_2 import compute_cvar
from experiments.phase4_4_3.benchmark_phase4_4_3 import (
    load_mappo_checkpoint,
    PHYSICAL_VIOLATION_CODES,
    POLICY_ERROR_CODES,
)
from experiments.phase4_4_4.remediation_v3 import (
    classify_generator_proposal,
    compute_battery_remediation_signals,
)

BASE_444 = ROOT / "experiments" / "phase4_4_4"
RES_444 = BASE_444 / "results"
RES_444.mkdir(parents=True, exist_ok=True)


def run_episode_444(
    env: MicrogridParallelEnv,
    controller_name: str,
    controller_group: str,
    ablation_name: str,
    train_seed: str,
    controller_type: str,
    controller_obj: Any,
    scenario_id: str,
    eval_seed: int,
    adequacy_fallback: bool,
    econ_cfg: EconomicConfig,
) -> Dict[str, Any]:
    random.seed(int(eval_seed))
    np.random.seed(int(eval_seed))
    obs, _ = env.reset(seed=eval_seed, options={"scenario_id": scenario_id})
    if hasattr(controller_obj, "reset"):
        controller_obj.reset()

    total_proposed_actions = 0
    feasible_proposals = 0
    boundary_clipping_count = 0
    emergency_override_count = 0
    supply_adequacy_intervention_count = 0
    inaction_detected_steps = 0

    gen_necessary_steps = 0
    gen_unnecessary_steps = 0
    gen_blocked_physical_steps = 0
    gen_corrected_fallback_steps = 0
    gen_missed_startup_steps = 0
    gen_startup_steps = 0
    outage_steps = 0

    genuine_deficit_steps = 0
    raw_deficit_bat_dis_steps = 0
    healthy_grid_steps = 0
    healthy_bat_dis_steps = 0
    healthy_bat_req_sum = 0.0
    battery_reserve_violations = 0

    interv_physical = 0
    interv_policy_err = 0
    fallback_extra_supply_mwh = 0.0
    total_reward = 0.0
    done = False
    dt_h = env.dt_hours

    while env.agents and not done:
        pre_gen_state = str(env.generator.state.value)
        pre_gen_steps = int(env.generator.steps_in_state)
        prev_bat_energy = float(env.battery.energy_mwh)

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

        for _, cls in diag.get("action_classification", {}).items():
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

        viols = set(diag.get("violations", []))
        if viols & PHYSICAL_VIOLATION_CODES:
            interv_physical += 1
        if viols & POLICY_ERROR_CODES:
            interv_policy_err += 1
        if "BATTERY_RESERVE_PRESERVED" in viols or "BATTERY_MIN_SOC_PREVENTED" in viols:
            battery_reserve_violations += 1

        raw_gen = float(parse_agent_action("generator", actions["generator"])["generator_fraction"])
        raw_bat = float(parse_agent_action("battery", actions["battery"])["battery_cmd"])

        gen_diag = classify_generator_proposal(
            state_dict=st,
            raw_gen_frac=raw_gen,
            shield_diag=diag,
            adeq_diag=adeq,
            pre_gen_state=pre_gen_state,
            pre_gen_steps_in_state=pre_gen_steps,
        )
        bat_diag = compute_battery_remediation_signals(
            state_dict=st,
            prev_bat_energy_mwh=prev_bat_energy,
            raw_bat_cmd=raw_bat,
            shield_diag=diag,
            adeq_diag=adeq,
            dt_hours=dt_h,
        )

        if gen_diag["necessary_dispatch"]:
            gen_necessary_steps += 1
        if gen_diag["unnecessary_dispatch"]:
            gen_unnecessary_steps += 1
        if gen_diag["blocked_by_physical"]:
            gen_blocked_physical_steps += 1
        if gen_diag["corrected_by_fallback"]:
            gen_corrected_fallback_steps += 1
        if gen_diag["missed_necessary_startup"]:
            gen_missed_startup_steps += 1
        if st.get("generator_state", "OFF") == "STARTING":
            gen_startup_steps += 1
        if not st["grid_connected"]:
            outage_steps += 1

        if bat_diag["genuine_deficit"]:
            genuine_deficit_steps += 1
            if bat_diag["battery_discharged_deficit"]:
                raw_deficit_bat_dis_steps += 1

        if st["grid_connected"] and not st["is_storm_warning"]:
            healthy_grid_steps += 1
            healthy_bat_req_sum += float(raw_bat * 0.25)
            if bat_diag["battery_discharged_healthy"]:
                healthy_bat_dis_steps += 1

        if adequacy_fallback and adeq.get("adequacy_intervention", False):
            sh_acts = diag.get("executed_actions", actions)
            sh_bat = float(parse_agent_action("battery", sh_acts["battery"])["requested_power_mw"])
            extra_bat_dis = max(0.0, (-float(st["battery_mw"])) - max(0.0, -sh_bat))
            sh_gen = float(parse_agent_action("generator", sh_acts["generator"])["requested_power_mw"])
            extra_gen_p = max(0.0, float(st["generator_mw"]) - (sh_gen if sh_gen > 0.04 else 0.0))
            fallback_extra_supply_mwh += (extra_bat_dis + extra_gen_p) * dt_h

    hist_df = pd.DataFrame(env.episode_history)
    econ = compute_episode_economics(hist_df, dt_hours=dt_h, config=econ_cfg)
    steps_df = hist_df.iloc[1:] if len(hist_df) > 1 else hist_df

    avoidable_crit_ens = float(steps_df["avoidable_critical_shed_mw"].sum() * dt_h) if "avoidable_critical_shed_mw" in steps_df else 0.0
    unavoidable_crit_ens = float(steps_df["unavoidable_critical_shed_mw"].sum() * dt_h) if "unavoidable_critical_shed_mw" in steps_df else 0.0

    bal_err = steps_df["balance_error_mw"].abs().astype(float)
    pf_conv_pct = float(steps_df["converged"].mean() * 100.0)
    feasible_pct = (feasible_proposals / total_proposed_actions * 100.0) if total_proposed_actions > 0 else 100.0
    soc_series = steps_df["battery_soc"].astype(float)

    necessary_rate_pct = min(100.0, (gen_necessary_steps / outage_steps * 100.0)) if outage_steps > 0 else 100.0

    return {
        "controller": controller_name,
        "controller_group": controller_group,
        "ablation": ablation_name,
        "train_seed": train_seed,
        "adequacy_fallback": bool(adequacy_fallback),
        "mode": "Hybrid (+Fallback)" if adequacy_fallback else "Intrinsic",
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
        "gen_necessary_dispatch_steps": int(gen_necessary_steps),
        "gen_necessary_dispatch_rate_pct": round(necessary_rate_pct, 2),
        "gen_unnecessary_dispatch_steps": int(gen_unnecessary_steps),
        "gen_blocked_physical_steps": int(gen_blocked_physical_steps),
        "gen_corrected_fallback_steps": int(gen_corrected_fallback_steps),
        "gen_missed_startup_steps": int(gen_missed_startup_steps),
        "gen_startup_steps": int(gen_startup_steps),
        "outage_steps": int(outage_steps),
        "genuine_deficit_steps": int(genuine_deficit_steps),
        "raw_deficit_bat_dis_pct": round((raw_deficit_bat_dis_steps / genuine_deficit_steps * 100.0) if genuine_deficit_steps > 0 else np.nan, 2),
        "healthy_grid_steps": int(healthy_grid_steps),
        "healthy_bat_dis_pct": round((healthy_bat_dis_steps / healthy_grid_steps * 100.0) if healthy_grid_steps > 0 else 0.0, 2),
        "healthy_bat_mean_mw": round((healthy_bat_req_sum / healthy_grid_steps) if healthy_grid_steps > 0 else 0.0, 4),
        "battery_reserve_violations": int(battery_reserve_violations),
        "interv_physical_constraints": int(interv_physical),
        "interv_policy_decision_error": int(interv_policy_err),
        "fallback_extra_supply_mwh": round(fallback_extra_supply_mwh, 6),
        "min_battery_soc_pct": round(float(soc_series.min()), 2),
        "max_battery_soc_pct": round(float(soc_series.max()), 2),
        "mean_balance_error_mw": round(float(bal_err.mean()), 6),
        "max_balance_error_mw": round(float(bal_err.max()), 6),
        "power_flow_converged_pct": round(pf_conv_pct, 2),
        **econ,
    }


def compute_summary_444(ep_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ctrl, adeq_fb), grp in ep_df.groupby(["controller", "adequacy_fallback"], sort=False):
        def st(col: str, higher: bool = False):
            vals = grp[col].dropna().values.astype(float)
            if len(vals) == 0:
                return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            m = float(np.mean(vals))
            s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            med = float(np.median(vals))
            q25, q75 = np.percentile(vals, [25, 75])
            ci_h = float(stats.t.ppf(0.975, df=max(1, len(vals) - 1)) * (s / np.sqrt(len(vals)))) if len(vals) > 1 and s > 0 else 0.0
            return m, s, med, float(q75 - q25), m - ci_h, m + ci_h, compute_cvar(vals, 0.10, higher)

        sc = st("true_social_cost_usd", False)
        op = st("operating_cost_usd", False)
        ce = st("critical_ens_mwh", False)
        cr = st("critical_reliability_pct", True)
        r0 = grp.iloc[0]

        rows.append({
            "controller": ctrl,
            "controller_group": r0["controller_group"],
            "ablation": r0["ablation"],
            "train_seed": r0["train_seed"],
            "adequacy_fallback": bool(adeq_fb),
            "mode": r0["mode"],
            "episodes_n": len(grp),
            "true_social_cost_mean": round(sc[0], 2),
            "true_social_cost_std": round(sc[1], 2),
            "true_social_cost_median": round(sc[2], 2),
            "true_social_cost_iqr": round(sc[3], 2),
            "true_social_cost_ci95_lo": round(sc[4], 2),
            "true_social_cost_ci95_hi": round(sc[5], 2),
            "true_social_cost_cvar10": round(sc[6], 2),
            "operating_cost_mean": round(op[0], 2),
            "critical_ens_mean": round(ce[0], 4),
            "critical_ens_std": round(ce[1], 4),
            "critical_ens_cvar10": round(ce[6], 4),
            "critical_reliability_mean": round(cr[0], 2),
            "critical_reliability_std": round(cr[1], 2),
            "critical_reliability_ci95_lo": round(cr[4], 2),
            "critical_reliability_ci95_hi": round(cr[5], 2),
            "critical_reliability_cvar10": round(cr[6], 2),
            "flexible_ens_mean": round(float(grp["flexible_ens_mean" if "flexible_ens_mean" in grp else "flexible_ens_mwh"].mean()), 4),
            "other_ens_mean": round(float(grp["other_ens_mwh"].mean()), 4),
            "avoidable_critical_ens_mean": round(float(grp["avoidable_critical_ens_mwh"].mean()), 4),
            "unavoidable_critical_ens_mean": round(float(grp["unavoidable_critical_ens_mwh"].mean()), 4),
            "fuel_consumed_l_mean": round(float(grp["fuel_consumed_l"].mean()), 2),
            "gen_hours_mean": round(float(grp["generator_operating_hours"].mean()), 2),
            "battery_throughput_mean": round(float(grp["battery_throughput_mwh"].mean()), 4),
            "terminal_soc_mean": round(float(grp["terminal_battery_soc_pct"].mean()), 2),
            "re_curtailment_mwh_mean": round(float(grp["renewable_curtailment_mwh"].mean()), 4),
            "grid_import_mwh_mean": round(float(grp["grid_import_mwh"].mean()), 4),
            "unshielded_feasible_pct": round(float(grp["unshielded_feasible_pct"].mean()), 2),
            "inaction_steps_mean": round(float(grp["inaction_detected_steps"].mean()), 2),
            "adequacy_intervention_mean": round(float(grp["supply_adequacy_intervention_count"].mean()), 2),
            "boundary_clipping_mean": round(float(grp["boundary_clipping_count"].mean()), 2),
            "emergency_override_mean": round(float(grp["emergency_override_count"].mean()), 2),
            "gen_unnecessary_steps_mean": round(float(grp["gen_unnecessary_dispatch_steps"].mean()), 2),
            "gen_necessary_rate_pct": round(float(grp["gen_necessary_dispatch_rate_pct"].mean()), 2),
            "gen_blocked_physical_mean": round(float(grp["gen_blocked_physical_steps"].mean()), 2),
            "gen_corrected_fallback_mean": round(float(grp["gen_corrected_fallback_steps"].mean()), 2),
            "raw_deficit_bat_dis_pct": round(float(grp["raw_deficit_bat_dis_pct"].dropna().mean()), 2),
            "healthy_bat_dis_pct": round(float(grp["healthy_bat_dis_pct"].mean()), 2),
            "healthy_bat_mean_mw": round(float(grp["healthy_bat_mean_mw"].mean()), 4),
            "battery_reserve_violations_mean": round(float(grp["battery_reserve_violations"].mean()), 2),
            "power_flow_converged_pct": round(float(grp["power_flow_converged_pct"].mean()), 2),
            "max_balance_error_mw": round(float(grp["max_balance_error_mw"].max()), 6),
        })
    return pd.DataFrame(rows)


def compute_fallback_attribution_444(ep_df: pd.DataFrame) -> pd.DataFrame:
    fb_rows = []
    for (ablation, t_seed), grp in ep_df.groupby(["ablation", "train_seed"], sort=False):
        intr = grp[grp["adequacy_fallback"] == False].sort_values(["scenario", "eval_seed"])
        hybr = grp[grp["adequacy_fallback"] == True].sort_values(["scenario", "eval_seed"])
        if len(intr) == 0 or len(hybr) == 0 or len(intr) != len(hybr):
            continue
        assert list(intr["scenario"]) == list(hybr["scenario"])
        assert list(intr["eval_seed"]) == list(hybr["eval_seed"])

        d_ens = hybr["critical_ens_mwh"].values - intr["critical_ens_mwh"].values
        d_rel = hybr["critical_reliability_pct"].values - intr["critical_reliability_pct"].values
        d_soc = hybr["true_social_cost_usd"].values - intr["true_social_cost_usd"].values
        _, p_ens = stats.ttest_rel(hybr["critical_ens_mwh"].values, intr["critical_ens_mwh"].values)
        _, p_soc = stats.ttest_rel(hybr["true_social_cost_usd"].values, intr["true_social_cost_usd"].values)

        fb_rows.append({
            "ablation": ablation,
            "train_seed": t_seed,
            "episodes_n": len(intr),
            "intrinsic_critical_ens_mwh": round(float(intr["critical_ens_mwh"].mean()), 4),
            "hybrid_critical_ens_mwh": round(float(hybr["critical_ens_mwh"].mean()), 4),
            "delta_critical_ens_mwh": round(float(np.mean(d_ens)), 4),
            "intrinsic_critical_rel_pct": round(float(intr["critical_reliability_pct"].mean()), 2),
            "hybrid_critical_rel_pct": round(float(hybr["critical_reliability_pct"].mean()), 2),
            "delta_critical_rel_pct": round(float(np.mean(d_rel)), 2),
            "intrinsic_social_cost_usd": round(float(intr["true_social_cost_usd"].mean()), 2),
            "hybrid_social_cost_usd": round(float(hybr["true_social_cost_usd"].mean()), 2),
            "delta_social_cost_usd": round(float(np.mean(d_soc)), 2),
            "adequacy_interventions_per_ep": round(float(hybr["supply_adequacy_intervention_count"].mean()), 2),
            "energy_dispatched_by_fallback_mwh": round(float(hybr["fallback_extra_supply_mwh"].mean()), 4),
            "additional_fuel_l": round(float(hybr["fuel_consumed_l"].mean() - intr["fuel_consumed_l"].mean()), 2),
            "intrinsic_avoidable_ens_mwh": round(float(intr["avoidable_critical_ens_mwh"].mean()), 4),
            "hybrid_avoidable_ens_mwh": round(float(hybr["avoidable_critical_ens_mwh"].mean()), 4),
            "paired_p_critical_ens": float(p_ens),
            "paired_p_social_cost": float(p_soc),
        })
    return pd.DataFrame(fb_rows)


def compute_paired_stats_444(ep_df: pd.DataFrame) -> pd.DataFrame:
    """Paired statistical comparisons with Holm-Bonferroni correction notes."""
    pairs: List[tuple] = []
    # Compare all 4.4.4 configs against Rule-Based EMS, Greedy, and 4.4.3 Ext Corrected (same seed & mode)
    for exp in ["baseline", "gen_only", "bat_only", "combined"]:
        exp_lbl = {"baseline": "4.4.4 Baseline", "gen_only": "4.4.4 Gen-Only", "bat_only": "4.4.4 Bat-Only", "combined": "4.4.4 Combined"}[exp]
        for s in [42, 101, 202]:
            intr = f"{exp_lbl} (Seed {s}) [Intrinsic]"
            hybr = f"{exp_lbl} (Seed {s}) [+Fallback]"
            p443_intr = f"Ext Corrected MAPPO (Seed {s}) [Intrinsic]"
            p443_hybr = f"Ext Corrected MAPPO (Seed {s}) [+Fallback]"
            pairs.extend([
                (intr, "Rule-Based EMS"),
                (intr, "Greedy Controller"),
                (intr, p443_intr),
                (hybr, "Rule-Based EMS"),
                (hybr, "Greedy Controller"),
                (hybr, p443_hybr),
            ])
            if exp != "baseline":
                pairs.extend([
                    (intr, f"4.4.4 Baseline (Seed {s}) [Intrinsic]"),
                    (hybr, f"4.4.4 Baseline (Seed {s}) [+Fallback]"),
                ])

    rows = []
    for c1, c2 in pairs:
        d1 = ep_df[ep_df["controller"] == c1].sort_values(["scenario", "eval_seed"])
        d2 = ep_df[ep_df["controller"] == c2].sort_values(["scenario", "eval_seed"])
        if len(d1) == 0 or len(d2) == 0:
            continue
        for m_col, m_label in [
            ("true_social_cost_usd", "True Social Cost ($)"),
            ("critical_ens_mwh", "Critical ENS (MWh)"),
            ("critical_reliability_pct", "Critical Reliability (%)"),
            ("gen_unnecessary_dispatch_steps", "Unnecessary Gen Proposals"),
            ("raw_deficit_bat_dis_pct", "Deficit Bat Discharge (%)"),
            ("unshielded_feasible_pct", "Unshielded Feasible (%)"),
        ]:
            # Align paired realizations on scenario + eval_seed (handles NaNs cleanly)
            merged = pd.merge(
                d1[["scenario", "eval_seed", m_col]],
                d2[["scenario", "eval_seed", m_col]],
                on=["scenario", "eval_seed"],
                suffixes=("_1", "_2"),
            ).dropna()
            if len(merged) < 2:
                continue
            v1 = merged[f"{m_col}_1"].values.astype(float)
            v2 = merged[f"{m_col}_2"].values.astype(float)
            diff = v1 - v2
            md = float(np.mean(diff))
            sd = float(np.std(diff, ddof=1))
            se = sd / np.sqrt(len(diff))
            ci_h = float(stats.t.ppf(0.975, df=len(diff) - 1) * se) if sd > 0 else 0.0
            d_cohen = float(md / sd) if sd > 1e-12 else 0.0
            if np.allclose(diff, 0.0):
                t_stat, t_p, w_stat, w_p = 0.0, 1.0, 0.0, 1.0
            else:
                t_stat, t_p = stats.ttest_rel(v1, v2)
                try:
                    w_stat, w_p = stats.wilcoxon(v1, v2)
                except ValueError:
                    w_stat, w_p = 0.0, 1.0
            rows.append({
                "controller_1": c1,
                "controller_2": c2,
                "metric": m_label,
                "episodes_n": len(merged),
                "c1_mean": round(float(np.mean(v1)), 4),
                "c2_mean": round(float(np.mean(v2)), 4),
                "mean_diff": round(md, 4),
                "ci95_lo": round(md - ci_h, 4),
                "ci95_hi": round(md + ci_h, 4),
                "paired_t_p": float(t_p),
                "wilcoxon_p": float(w_p),
                "cohens_d": round(d_cohen, 4),
                "significant_p05": bool(t_p < 0.05),
            })
    df_stats = pd.DataFrame(rows)
    if not df_stats.empty:
        # Holm-Bonferroni step-down adjustment within each metric family
        df_stats["holm_adjusted_p"] = 1.0
        for _, m_grp in df_stats.groupby("metric"):
            idx_sorted = m_grp.sort_values("paired_t_p").index
            k_m = len(idx_sorted)
            running_max = 0.0
            for rank, idx in enumerate(idx_sorted):
                adj = min(1.0, float(df_stats.loc[idx, "paired_t_p"]) * (k_m - rank))
                running_max = max(running_max, adj)
                df_stats.loc[idx, "holm_adjusted_p"] = running_max
        df_stats["significant_holm05"] = df_stats["holm_adjusted_p"] < 0.05
    return df_stats


def build_group_configs(group: str) -> List[Dict[str, Any]]:
    ckpt_4 = ROOT / "experiments" / "phase4"
    ckpt_41 = ckpt_4 / "diagnostics" / "checkpoints" / "best_model.pt"
    ckpt_42 = ckpt_4 / "forecasting" / "checkpoints" / "realistic" / "best_model.pt"
    ckpt_43 = {s: ckpt_4 / "robustness" / "checkpoints" / f"robust_seed{s}" / "best_model.pt" for s in [42, 101, 202]}
    ckpt_443 = ROOT / "experiments" / "phase4_4_3" / "checkpoints"
    ckpt_444 = BASE_444 / "checkpoints"

    cfgs: List[Dict[str, Any]] = []
    if group in ("heuristics_prior", "all"):
        cfgs.extend([
            {"name": "Greedy Controller", "group": "Heuristic", "ablation": "Heuristic", "train_seed": "deterministic", "type": "greedy", "obj": GreedyController(), "fb": False, "fm": None},
            {"name": "Rule-Based EMS", "group": "Heuristic", "ablation": "Heuristic", "train_seed": "deterministic", "type": "rule_based", "obj": RuleBasedEMS(), "fb": False, "fm": None},
            {"name": "Lookahead Heuristic EMS", "group": "Heuristic", "ablation": "Heuristic", "train_seed": "deterministic", "type": "lookahead_heuristic", "obj": MPCController(horizon_steps=8), "fb": False, "fm": None},
            {"name": "Reactive MAPPO (Phase 4.1)", "group": "Prior RL", "ablation": "Prior RL", "train_seed": "42", "type": "mappo", "obj": load_mappo_checkpoint(ckpt_41, None), "fb": False, "fm": None},
            {"name": "Forecast-Aware MAPPO (Phase 4.2)", "group": "Prior RL", "ablation": "Prior RL", "train_seed": "42", "type": "mappo", "obj": load_mappo_checkpoint(ckpt_42, "realistic"), "fb": False, "fm": "realistic"},
        ])
        for s in [42, 101, 202]:
            cfgs.append({"name": f"Robust Forecast MAPPO (Seed {s})", "group": "Prior RL", "ablation": "Prior RL", "train_seed": str(s), "type": "mappo", "obj": load_mappo_checkpoint(ckpt_43[s], "realistic"), "fb": False, "fm": "realistic"})

    if group in ("p443_ext", "all"):
        for s in [42, 101, 202]:
            ag_c = load_mappo_checkpoint(ckpt_443 / f"ext_corrected_seed{s}" / "best_model.pt", "realistic")
            ag_t = load_mappo_checkpoint(ckpt_443 / f"ext_tail_risk_seed{s}" / "best_model.pt", "realistic")
            for fb, lbl in [(False, "Intrinsic"), (True, "+Fallback")]:
                cfgs.append({"name": f"Ext Corrected MAPPO (Seed {s}) [{lbl}]", "group": f"Ext Corrected ({lbl})", "ablation": "P4.4.3 Corrected", "train_seed": str(s), "type": "mappo", "obj": ag_c, "fb": fb, "fm": "realistic"})
                cfgs.append({"name": f"Ext Tail-Risk MAPPO (Seed {s}) [{lbl}]", "group": f"Ext Tail-Risk ({lbl})", "ablation": "P4.4.3 Tail-Risk", "train_seed": str(s), "type": "mappo", "obj": ag_t, "fb": fb, "fm": "realistic"})

    ablation_map = {
        "ablation_baseline": ("baseline", "4.4.4 Baseline"),
        "ablation_gen_only": ("gen_only", "4.4.4 Gen-Only"),
        "ablation_bat_only": ("bat_only", "4.4.4 Bat-Only"),
        "ablation_combined": ("combined", "4.4.4 Combined"),
    }
    for grp_key, (exp, exp_lbl) in ablation_map.items():
        if group in (grp_key, "all"):
            for s in [42, 101, 202]:
                ag = load_mappo_checkpoint(ckpt_444 / f"{exp}_seed{s}" / "best_model.pt", "realistic")
                for fb, lbl in [(False, "Intrinsic"), (True, "+Fallback")]:
                    cfgs.append({"name": f"{exp_lbl} (Seed {s}) [{lbl}]", "group": f"{exp_lbl} ({lbl})", "ablation": exp_lbl, "train_seed": str(s), "type": "mappo", "obj": ag, "fb": fb, "fm": "realistic"})

    return cfgs


def run_benchmark_group(group: str):
    econ_cfg = EconomicConfig.from_yaml(str(BASE_444 / "configs" / "economics_v1.yaml"))
    reward_cfg = str(BASE_444 / "configs" / "reward_v3_social.yaml")
    models_dir = str(ROOT / "experiments" / "phase4" / "forecasting" / "models")
    scenarios = [f"scenario_h{i}" for i in range(1, 7)]
    eval_seeds = [42, 101, 202, 303, 404]

    cfgs = build_group_configs(group)
    print(f"[*] Running Phase 4.4.4 Benchmark ({group}): {len(cfgs)} configs x 6 x 5 = {len(cfgs)*30} eps", flush=True)

    env_cache: Dict[tuple, MicrogridParallelEnv] = {}

    def get_env(fb: bool, fm: Optional[str]) -> MicrogridParallelEnv:
        key = (fb, fm)
        if key not in env_cache:
            env_cache[key] = MicrogridParallelEnv(
                scenario_id="scenario_h1",
                forecast_mode=fm,
                forecast_models_dir=models_dir if fm else None,
                max_steps=96,
                dt_minutes=15.0,
                enable_safety_shield=True,
                reward_config_path=reward_cfg,
                enable_adequacy_check=True,
                adequacy_fallback=fb,
            )
        return env_cache[key]

    rows = []
    for c in cfgs:
        t0 = time.time()
        env = get_env(c["fb"], c["fm"])
        for sc in scenarios:
            for es in eval_seeds:
                rows.append(run_episode_444(
                    env=env,
                    controller_name=c["name"],
                    controller_group=c["group"],
                    ablation_name=c["ablation"],
                    train_seed=c["train_seed"],
                    controller_type=c["type"],
                    controller_obj=c["obj"],
                    scenario_id=sc,
                    eval_seed=es,
                    adequacy_fallback=c["fb"],
                    econ_cfg=econ_cfg,
                ))
        dt = time.time() - t0
        df_c = pd.DataFrame(rows[-30:])
        print(
            f"  [+] {c['name']:42s} | Social=${df_c['true_social_cost_usd'].mean():7.1f} | "
            f"Rel={df_c['critical_reliability_pct'].mean():5.2f}% | "
            f"UnnGen={df_c['gen_unnecessary_dispatch_steps'].mean():5.1f} | "
            f"DefBat={df_c['raw_deficit_bat_dis_pct'].dropna().mean():5.1f}% | {dt:.1f}s",
            flush=True,
        )

    out_csv = RES_444 / f"episode_level_metrics_{group}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[✓] Saved group '{group}' ({len(rows)} eps) to {out_csv}", flush=True)


def merge_all_groups():
    groups = [
        "heuristics_prior",
        "p443_ext",
        "ablation_baseline",
        "ablation_gen_only",
        "ablation_bat_only",
        "ablation_combined",
    ]
    dfs = [pd.read_csv(RES_444 / f"episode_level_metrics_{g}.csv") for g in groups]
    ep_df = pd.concat(dfs, ignore_index=True)
    ep_df["gen_necessary_dispatch_rate_pct"] = (
        (ep_df["gen_necessary_dispatch_steps"] / ep_df["outage_steps"].clip(lower=1)) * 100.0
    ).clip(upper=100.0).round(2)
    ep_df.to_csv(RES_444 / "episode_level_metrics.csv", index=False)

    comp_df = compute_summary_444(ep_df)
    comp_df.to_csv(RES_444 / "controller_comparison.csv", index=False)

    aut_cols = [
        "controller", "controller_group", "ablation", "train_seed", "mode",
        "gen_unnecessary_steps_mean", "gen_necessary_rate_pct", "gen_blocked_physical_mean",
        "gen_corrected_fallback_mean", "gen_hours_mean", "fuel_consumed_l_mean",
        "raw_deficit_bat_dis_pct", "healthy_bat_dis_pct", "healthy_bat_mean_mw",
        "battery_reserve_violations_mean", "inaction_steps_mean", "unshielded_feasible_pct",
        "boundary_clipping_mean", "emergency_override_mean", "adequacy_intervention_mean",
    ]
    comp_df[aut_cols].to_csv(RES_444 / "autonomy_metrics.csv", index=False)

    fb_df = compute_fallback_attribution_444(ep_df)
    fb_df.to_csv(RES_444 / "fallback_attribution.csv", index=False)

    stats_df = compute_paired_stats_444(ep_df)
    stats_df.to_csv(RES_444 / "paired_statistical_comparisons.csv", index=False)
    print(f"[✓] Merged {len(ep_df)} episodes, {ep_df['controller'].nunique()} controllers -> {RES_444}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, default="all")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    if args.merge:
        merge_all_groups()
    else:
        run_benchmark_group(args.group)
