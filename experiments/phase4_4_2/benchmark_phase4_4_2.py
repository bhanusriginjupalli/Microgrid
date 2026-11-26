"""
experiments/phase4_4_2/benchmark_phase4_4_2.py
==============================================
GRIDGUARD AI — Phase 4.4.2 Controlled Multi-Seed Benchmark & Statistical Suite

Evaluates Controllers A–H (and Legacy Phase 4.4 references) on identical
held-out scenarios H1–H6 and identical evaluation seeds [42, 101, 202, 303, 404]
using the single authoritative EconomicAccounting engine.

Explicitly renames "MPC" -> "Lookahead Heuristic EMS" and avoids treating
multiple evaluation episodes from one checkpoint as independent training seeds.
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
from controllers.rule_based_ems import RuleBasedEMS
from controllers.greedy_controller import GreedyController
from controllers.mpc_controller import MPCController
from mappo.config import MAPPOConfig
from mappo.mappo_agent import MAPPOAgent


BASE_442 = ROOT / "experiments" / "phase4_4_2"
RESULTS_442 = BASE_442 / "results"
RESULTS_442.mkdir(parents=True, exist_ok=True)


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


def compute_cvar(values: np.ndarray, alpha: float = 0.10, higher_is_better: bool = False) -> float:
    """
    Compute Conditional Value at Risk (CVaR_alpha):
      - If higher_is_better=False (e.g. True Social Cost, Critical ENS):
          mean of the WORST (highest) alpha fraction of outcomes.
      - If higher_is_better=True (e.g. Reliability, Reward):
          mean of the WORST (lowest) alpha fraction of outcomes.
    """
    arr = np.sort(np.asarray(values, dtype=float))
    n = len(arr)
    if n == 0:
        return 0.0
    k = max(1, int(np.ceil(alpha * n)))
    if higher_is_better:
        return float(np.mean(arr[:k]))
    return float(np.mean(arr[-k:]))


def run_episode(
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

    total_reward = 0.0
    done = False

    while env.agents and not done:
        if controller_type in ("rule_based", "greedy", "lookahead_heuristic"):
            actions = controller_obj.select_actions(obs, env.episode_history[-1])
        elif controller_type == "mappo":
            actions, _ = controller_obj.select_actions(obs, deterministic=True)
        else:
            raise ValueError(f"Unknown controller_type: {controller_type}")

        obs, rewards, terms, truncs, infos = env.step(actions)
        total_reward += float(rewards.get("battery", 0.0))
        done = any(terms.values()) or any(truncs.values())

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

    hist_df = pd.DataFrame(env.episode_history)
    econ = compute_episode_economics(hist_df, dt_hours=env.dt_hours, config=econ_cfg)

    steps_df = hist_df.iloc[1:] if len(hist_df) > 1 else hist_df
    dt_h = env.dt_hours

    avoidable_crit_ens = float(steps_df["avoidable_critical_shed_mw"].sum() * dt_h) if "avoidable_critical_shed_mw" in steps_df else 0.0
    unavoidable_crit_ens = float(steps_df["unavoidable_critical_shed_mw"].sum() * dt_h) if "unavoidable_critical_shed_mw" in steps_df else 0.0
    mean_bat_mw = float(steps_df["battery_mw"].abs().mean())
    mean_gen_mw = float(steps_df["generator_mw"].mean())

    pcc_v = steps_df["pcc_voltage_pu"].astype(float)
    min_v = float(steps_df["min_bus_voltage_pu"].min()) if "min_bus_voltage_pu" in steps_df else float(pcc_v.min())
    max_v = float(steps_df["max_bus_voltage_pu"].max()) if "max_bus_voltage_pu" in steps_df else float(pcc_v.max())
    volt_viols = int(((min_v < 0.95) or (max_v > 1.05)) if False else ((steps_df.get("min_bus_voltage_pu", pcc_v) < 0.95) | (steps_df.get("max_bus_voltage_pu", pcc_v) > 1.05)).sum())

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
        **econ,
    }


def compute_summary_statistics(ep_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean, std, median, IQR, 95% CI, CVaR 10% & 5% per controller."""
    rows = []
    for (ctrl, adeq_fb), grp in ep_df.groupby(["controller", "adequacy_fallback"], sort=False):
        n = len(grp)
        def stats_col(col: str, higher_is_better: bool = False):
            vals = grp[col].values.astype(float)
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            med = float(np.median(vals))
            q25, q75 = np.percentile(vals, [25, 75])
            iqr = float(q75 - q25)
            ci_half = float(stats.t.ppf(0.975, df=max(1, n - 1)) * (std / np.sqrt(n))) if n > 1 and std > 0 else 0.0
            cvar10 = compute_cvar(vals, alpha=0.10, higher_is_better=higher_is_better)
            cvar05 = compute_cvar(vals, alpha=0.05, higher_is_better=higher_is_better)
            return mean, std, med, iqr, mean - ci_half, mean + ci_half, cvar10, cvar05

        sc_m, sc_s, sc_med, sc_iqr, sc_lo, sc_hi, sc_cvar10, sc_cvar05 = stats_col("true_social_cost_usd", False)
        op_m, op_s, op_med, op_iqr, _, _, _, _ = stats_col("operating_cost_usd", False)
        ce_m, ce_s, ce_med, ce_iqr, _, _, ce_cvar10, ce_cvar05 = stats_col("critical_ens_mwh", False)
        cr_m, cr_s, cr_med, cr_iqr, cr_lo, cr_hi, cr_cvar10, _ = stats_col("critical_reliability_pct", True)
        tr_m, tr_s, _, _, _, _, _, _ = stats_col("total_reliability_pct", True)
        rw_m, rw_s, _, _, _, _, rw_cvar10, _ = stats_col("reward", True)

        rows.append({
            "controller": ctrl,
            "controller_group": grp["controller_group"].iloc[0],
            "train_seed": grp["train_seed"].iloc[0],
            "adequacy_fallback": adeq_fb,
            "episodes_n": n,
            "true_social_cost_mean": round(sc_m, 2),
            "true_social_cost_std": round(sc_s, 2),
            "true_social_cost_median": round(sc_med, 2),
            "true_social_cost_iqr": round(sc_iqr, 2),
            "true_social_cost_ci95_lo": round(sc_lo, 2),
            "true_social_cost_ci95_hi": round(sc_hi, 2),
            "true_social_cost_cvar10": round(sc_cvar10, 2),
            "true_social_cost_cvar05": round(sc_cvar05, 2),
            "operating_cost_mean": round(op_m, 2),
            "operating_cost_std": round(op_s, 2),
            "critical_ens_mean": round(ce_m, 4),
            "critical_ens_std": round(ce_s, 4),
            "critical_ens_median": round(ce_med, 4),
            "critical_ens_iqr": round(ce_iqr, 4),
            "critical_ens_cvar10": round(ce_cvar10, 4),
            "critical_ens_cvar05": round(ce_cvar05, 4),
            "critical_reliability_mean": round(cr_m, 2),
            "critical_reliability_std": round(cr_s, 2),
            "critical_reliability_median": round(cr_med, 2),
            "critical_reliability_iqr": round(cr_iqr, 2),
            "critical_reliability_ci95_lo": round(cr_lo, 2),
            "critical_reliability_ci95_hi": round(cr_hi, 2),
            "critical_reliability_cvar10": round(cr_cvar10, 2),
            "total_reliability_mean": round(tr_m, 2),
            "flexible_ens_mean": round(float(grp["flexible_ens_mwh"].mean()), 4),
            "other_ens_mean": round(float(grp["other_ens_mwh"].mean()), 4),
            "total_ens_mean": round(float(grp["total_ens_mwh"].mean()), 4),
            "reward_mean": round(rw_m, 2),
            "reward_std": round(rw_s, 2),
            "reward_cvar10": round(rw_cvar10, 2),
            "unshielded_feasible_pct": round(float(grp["unshielded_feasible_pct"].mean()), 2),
            "boundary_clipping_mean": round(float(grp["boundary_clipping_count"].mean()), 2),
            "emergency_override_mean": round(float(grp["emergency_override_count"].mean()), 2),
            "adequacy_intervention_mean": round(float(grp["supply_adequacy_intervention_count"].mean()), 2),
            "inaction_steps_mean": round(float(grp["inaction_detected_steps"].mean()), 2),
            "avoidable_critical_ens_mean": round(float(grp["avoidable_critical_ens_mwh"].mean()), 4),
            "unavoidable_critical_ens_mean": round(float(grp["unavoidable_critical_ens_mwh"].mean()), 4),
            "battery_throughput_mean": round(float(grp["battery_throughput_mwh"].mean()), 4),
            "terminal_soc_mean": round(float(grp["terminal_battery_soc_pct"].mean()), 2),
            "fuel_consumed_l_mean": round(float(grp["fuel_consumed_l"].mean()), 2),
            "gen_hours_mean": round(float(grp["generator_operating_hours"].mean()), 2),
            "grid_import_mwh_mean": round(float(grp["grid_import_mwh"].mean()), 4),
            "grid_export_mwh_mean": round(float(grp["grid_export_mwh"].mean()), 4),
            "re_curtailment_mwh_mean": round(float(grp["renewable_curtailment_mwh"].mean()), 4),
            "power_flow_converged_pct": round(float(grp["power_flow_converged_pct"].mean()), 2),
            "min_bus_voltage_pu": round(float(grp["min_bus_voltage_pu"].min()), 4),
            "voltage_violations_mean": round(float(grp["voltage_violations_count"].mean()), 2),
        })
    return pd.DataFrame(rows)


def compute_paired_statistics(ep_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute paired statistical comparisons across identical (scenario, eval_seed)
    realizations (n = 30 per comparison).
    """
    pairs = [
        ("Corrected Constrained MAPPO (Seed 42)", False, "Legacy Constrained MAPPO (Phase 4.4 Seed 42)", False),
        ("Corrected Constrained MAPPO (Seed 101)", False, "Legacy Constrained MAPPO (Phase 4.4 Seed 42)", False),
        ("Corrected Constrained MAPPO (Seed 202)", False, "Legacy Constrained MAPPO (Phase 4.4 Seed 42)", False),
        ("Corrected Constrained MAPPO (Seed 42)", False, "Rule-Based EMS", False),
        ("Corrected Constrained MAPPO (Seed 101)", False, "Rule-Based EMS", False),
        ("Corrected Constrained MAPPO (Seed 42)", False, "Robust Forecast MAPPO (Seed 42)", False),
        ("Corrected Constrained MAPPO (Seed 101)", False, "Robust Forecast MAPPO (Seed 101)", False),
        ("Corrected Constrained MAPPO (Seed 42) + Fallback", True, "Rule-Based EMS", False),
        ("Corrected Constrained MAPPO (Seed 101) + Fallback", True, "Rule-Based EMS", False),
        ("Corrected Constrained MAPPO (Seed 202) + Fallback", True, "Rule-Based EMS", False),
        ("Corrected Constrained MAPPO (Seed 42) + Fallback", True, "Legacy Constrained MAPPO (Phase 4.4 Seed 42)", False),
        ("Tail-Risk Constrained MAPPO (Seed 42)", False, "Corrected Constrained MAPPO (Seed 42)", False),
        ("Tail-Risk Constrained MAPPO (Seed 101)", False, "Corrected Constrained MAPPO (Seed 101)", False),
        ("Tail-Risk Constrained MAPPO (Seed 42)", False, "Legacy CVaR MAPPO (Phase 4.4 Seed 42)", False),
        ("Tail-Risk Constrained MAPPO (Seed 101)", False, "Legacy CVaR MAPPO (Phase 4.4 Seed 42)", False),
        ("Tail-Risk Constrained MAPPO (Seed 101) + Fallback", True, "Rule-Based EMS", False),
        ("Lookahead Heuristic EMS", False, "Rule-Based EMS", False),
    ]
    metrics = [
        ("true_social_cost_usd", "True Social Cost ($)", False),
        ("critical_ens_mwh", "Critical ENS (MWh)", False),
        ("critical_reliability_pct", "Critical Reliability (%)", True),
        ("unshielded_feasible_pct", "Unshielded Feasible (%)", True),
        ("inaction_detected_steps", "Inaction Steps", False),
    ]

    rows = []
    for c1, fb1, c2, fb2 in pairs:
        d1 = ep_df[(ep_df["controller"] == c1) & (ep_df["adequacy_fallback"] == fb1)].sort_values(["scenario", "eval_seed"])
        d2 = ep_df[(ep_df["controller"] == c2) & (ep_df["adequacy_fallback"] == fb2)].sort_values(["scenario", "eval_seed"])
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
                if np.all(np.abs(diff) < 1e-12):
                    w_stat, w_p = 0.0, 1.0
                else:
                    w_res = stats.wilcoxon(x, y, zero_method="pratt")
                    w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
            except Exception:
                w_stat, w_p = np.nan, np.nan

            cvar10_1 = compute_cvar(x, 0.10, higher)
            cvar10_2 = compute_cvar(y, 0.10, higher)
            cvar05_1 = compute_cvar(x, 0.05, higher)
            cvar05_2 = compute_cvar(y, 0.05, higher)

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
                "c1_cvar10": round(cvar10_1, 4),
                "c2_cvar10": round(cvar10_2, 4),
                "c1_cvar05": round(cvar05_1, 4),
                "c2_cvar05": round(cvar05_2, 4),
                "significant_p05": bool(t_p < 0.05),
            })
    return pd.DataFrame(rows)


def run_benchmark(eval_seeds: List[int] = [42, 101, 202, 303, 404], group_filter: str = "all"):
    scenarios = ["scenario_h1", "scenario_h2", "scenario_h3", "scenario_h4", "scenario_h5", "scenario_h6"]
    econ_cfg = EconomicConfig.from_yaml(BASE_442 / "configs" / "economics_v1.yaml")
    reward_cfg_path = str(BASE_442 / "configs" / "reward_v3_social.yaml")

    specs = []
    if group_filter in ("all", "heuristics"):
        specs.extend([
            ("Rule-Based EMS", "Heuristic", "deterministic", "rule_based", RuleBasedEMS(), None, False),
            ("Greedy Controller", "Heuristic", "deterministic", "greedy", GreedyController(), None, False),
            ("Lookahead Heuristic EMS", "Heuristic", "deterministic", "lookahead_heuristic", MPCController(horizon_steps=8, dt_hours=0.25, name="Lookahead Heuristic EMS"), None, False),
        ])
    if group_filter in ("all", "prior_rl"):
        specs.extend([
            ("Reactive MAPPO (Phase 4.1)", "Prior RL", "42", "mappo", ROOT / "experiments" / "phase4" / "diagnostics" / "checkpoints" / "best_model.pt", None, False),
            ("Forecast-Aware MAPPO (Phase 4.2)", "Prior RL", "42", "mappo", ROOT / "experiments" / "phase4" / "forecasting" / "checkpoints" / "realistic" / "best_model.pt", "realistic", False),
            ("Robust Forecast MAPPO (Seed 42)", "Prior RL", "42", "mappo", ROOT / "experiments" / "phase4" / "robustness" / "checkpoints" / "robust_seed42" / "best_model.pt", "realistic", False),
            ("Robust Forecast MAPPO (Seed 101)", "Prior RL", "101", "mappo", ROOT / "experiments" / "phase4" / "robustness" / "checkpoints" / "robust_seed101" / "best_model.pt", "realistic", False),
            ("Robust Forecast MAPPO (Seed 202)", "Prior RL", "202", "mappo", ROOT / "experiments" / "phase4" / "robustness" / "checkpoints" / "robust_seed202" / "best_model.pt", "realistic", False),
            ("Legacy Constrained MAPPO (Phase 4.4 Seed 42)", "Legacy 4.4", "42", "mappo", ROOT / "experiments" / "phase4" / "safety_constrained" / "checkpoints" / "constrained_seed42" / "best_model.pt", "realistic", False),
            ("Legacy CVaR MAPPO (Phase 4.4 Seed 42)", "Legacy 4.4", "42", "mappo", ROOT / "experiments" / "phase4" / "safety_constrained" / "checkpoints" / "constrained_cvar_seed42" / "best_model.pt", "realistic", False),
        ])

    if group_filter in ("all", "corrected_442"):
        for seed in [42, 101, 202]:
            ckpt_corr = BASE_442 / "checkpoints" / f"corrected_constrained_seed{seed}" / "best_model.pt"
            if ckpt_corr.exists():
                specs.append((f"Corrected Constrained MAPPO (Seed {seed})", "Corrected 4.4.2 (Intrinsic)", str(seed), "mappo", ckpt_corr, "realistic", False))
                specs.append((f"Corrected Constrained MAPPO (Seed {seed}) + Fallback", "Corrected 4.4.2 (+Fallback)", str(seed), "mappo", ckpt_corr, "realistic", True))

    if group_filter in ("all", "tail_risk_442"):
        for seed in [42, 101, 202]:
            ckpt_tail = BASE_442 / "checkpoints" / f"tail_risk_constrained_seed{seed}" / "best_model.pt"
            if ckpt_tail.exists():
                specs.append((f"Tail-Risk Constrained MAPPO (Seed {seed})", "Tail-Risk 4.4.2 (Intrinsic)", str(seed), "mappo", ckpt_tail, "realistic", False))
                specs.append((f"Tail-Risk Constrained MAPPO (Seed {seed}) + Fallback", "Tail-Risk 4.4.2 (+Fallback)", str(seed), "mappo", ckpt_tail, "realistic", True))

    all_records = []
    t_start = time.time()
    models_dir = str(ROOT / "experiments" / "phase4" / "forecasting" / "models")
    env_cache: Dict[Tuple[Optional[str], bool], MicrogridParallelEnv] = {}

    print(f"[*] Running Phase 4.4.2 Benchmark: {len(specs)} controller evaluations x {len(scenarios)} scenarios x {len(eval_seeds)} seeds = {len(specs)*len(scenarios)*len(eval_seeds)} episodes")

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
                rec = run_episode(
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
        sub = [r for r in all_records if r["controller"] == name and r["adequacy_fallback"] == adeq_fb]
        print(
            f"  [+] {name:50s} | Social=${np.mean([r['true_social_cost_usd'] for r in sub]):7.1f} | "
            f"CritRel={np.mean([r['critical_reliability_pct'] for r in sub]):5.2f}% | "
            f"Feas={np.mean([r['unshielded_feasible_pct'] for r in sub]):5.1f}% | "
            f"Inact={np.mean([r['inaction_detected_steps'] for r in sub]):4.1f} | {time.time()-t_c:.1f}s"
        )

    ep_df = pd.DataFrame(all_records)
    if group_filter != "all":
        ep_df.to_csv(RESULTS_442 / f"episode_level_metrics_{group_filter}.csv", index=False)
        print(f"\n[✓] Benchmark group '{group_filter}' complete in {time.time() - t_start:.1f}s. Episodes: {len(ep_df)}.", flush=True)
        return

    finalize_benchmark_outputs(ep_df)
    print(f"\n[✓] Benchmark complete in {time.time() - t_start:.1f}s. Total episodes: {len(ep_df)}.", flush=True)


def finalize_benchmark_outputs(ep_df: pd.DataFrame):
    ep_df.to_csv(RESULTS_442 / "episode_level_metrics.csv", index=False)

    summary_df = compute_summary_statistics(ep_df)
    summary_df.to_csv(RESULTS_442 / "controller_comparison.csv", index=False)

    adequacy_cols = [
        "controller", "controller_group", "train_seed", "adequacy_fallback", "episodes_n",
        "unshielded_feasible_pct", "boundary_clipping_mean", "emergency_override_mean",
        "adequacy_intervention_mean", "inaction_steps_mean",
        "avoidable_critical_ens_mean", "unavoidable_critical_ens_mean",
        "critical_ens_mean", "critical_reliability_mean",
    ]
    summary_df[adequacy_cols].to_csv(RESULTS_442 / "supply_adequacy_audit.csv", index=False)

    econ_audit_cols = [
        "controller", "controller_group", "train_seed", "adequacy_fallback", "episodes_n",
        "grid_import_mwh_mean", "grid_export_mwh_mean", "fuel_consumed_l_mean",
        "operating_cost_mean", "critical_ens_mean", "flexible_ens_mean", "other_ens_mean",
        "battery_throughput_mean", "terminal_soc_mean", "true_social_cost_mean",
        "true_social_cost_cvar10", "true_social_cost_cvar05",
    ]
    summary_df[econ_audit_cols].to_csv(RESULTS_442 / "economic_accounting_audit.csv", index=False)

    stats_df = compute_paired_statistics(ep_df)
    stats_df.to_csv(RESULTS_442 / "statistical_comparisons.csv", index=False)


def merge_benchmark_groups():
    dfs = []
    for g in ["heuristics", "prior_rl", "corrected_442", "tail_risk_442"]:
        p = RESULTS_442 / f"episode_level_metrics_{g}.csv"
        if p.exists():
            dfs.append(pd.read_csv(p))
    if not dfs:
        raise FileNotFoundError("No group episode CSVs found to merge.")
    merged = pd.concat(dfs, ignore_index=True)
    finalize_benchmark_outputs(merged)
    print(f"[✓] Merged {len(merged)} total benchmark episodes into final Phase 4.4.2 CSVs.", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, default="all")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.merge:
        merge_benchmark_groups()
    else:
        run_benchmark(group_filter=args.group)
