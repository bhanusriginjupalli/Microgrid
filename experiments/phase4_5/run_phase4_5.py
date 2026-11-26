"""
experiments/phase4_5/run_phase4_5.py
====================================
GRIDGUARD AI — Phase 4.5 Explainable MARL (X-MARL) Orchestrator

Runs explainability analysis strictly on the verified Phase 4.4.4 primary
candidate checkpoint (Bat-Only Seed 202, ep=80, val_score=-8920.74) plus
comparison baselines (Greedy Controller, Rule-Based EMS, and Hybrid +Fallback).

Outputs saved to experiments/phase4_5/results/:
  - verified_checkpoint_metadata.json
  - scenario_comparison_traces.csv
  - reward_decomposition_all.csv
  - safety_intervention_trace.csv
  - h2_system_timeline.csv
  - h2_agent_decision_explanations.json
  - h2_agent_explanations_flat.csv
  - feature_attributions_h2.csv
  - controlled_perturbation_validation.csv
  - counterfactual_results.csv
"""

import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "phase1_digital_twin") not in sys.path:
    sys.path.insert(0, str(ROOT / "phase1_digital_twin"))

from rl_environment.agent_config import AGENTS
from rl_environment.microgrid_parallel_env import MicrogridParallelEnv
from rl_environment.economic_accounting import EconomicConfig, compute_episode_economics
from controllers.greedy_controller import GreedyController
from controllers.rule_based_ems import RuleBasedEMS
from experiments.phase4_4_3.benchmark_phase4_4_3 import load_mappo_checkpoint
from experiments.phase4_5.explainability.decision_explainer import (
    DecisionExplainer,
    verify_checkpoint_metadata,
)
from experiments.phase4_5.explainability.feature_attribution import (
    compute_integrated_gradients,
    compute_perturbation_sensitivity,
    run_controlled_perturbation_validation,
)
from experiments.phase4_5.explainability.counterfactuals import run_counterfactual_suite


BASE_45 = ROOT / "experiments" / "phase4_5"
CFG_PATH = BASE_45 / "configs" / "xmarl_config.yaml"
RES_DIR = BASE_45 / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)


def _classify_regime(pre_state: Dict[str, Any]) -> str:
    grid_avail = bool(pre_state.get("grid_available", True))
    grid_conn = bool(pre_state.get("grid_connected", True))
    storm_warn = bool(pre_state.get("is_storm_warning", False))
    nom = float(pre_state.get("nominal_total_mw", 0.0))
    re_p = float(pre_state.get("solar_mw", 0.0)) + float(pre_state.get("wind_mw", 0.0))
    if not grid_avail or not grid_conn:
        return "islanded_outage"
    if storm_warn:
        return "storm_warning"
    if nom - re_p > 0.45:
        return "high_net_load"
    return "healthy_grid"


def run_traced_episode(
    controller_name: str,
    mode_label: str,
    controller_type: str,
    controller_obj: Any,
    scenario_id: str,
    eval_seed: int,
    adequacy_fallback: bool,
    forecast_mode: Optional[str],
    forecast_models_dir: str,
    reward_cfg_path: str,
    econ_cfg: EconomicConfig,
    explainer: DecisionExplainer,
    capture_timeline: bool = False,
    capture_obs_list: bool = False,
    env: Optional[MicrogridParallelEnv] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if env is None:
        env = MicrogridParallelEnv(
            scenario_id=scenario_id,
            forecast_mode=forecast_mode,
            forecast_models_dir=forecast_models_dir,
            reward_config_path=reward_cfg_path,
            enable_adequacy_check=True,
            adequacy_fallback=bool(adequacy_fallback),
            enable_constraint_costs=True,
        )

    # Explicit reseeding ensures paired scenarios share identical stochastic weather draws
    random.seed(int(eval_seed))
    np.random.seed(int(eval_seed))
    obs, _ = env.reset(seed=int(eval_seed), options={"scenario_id": scenario_id})
    if hasattr(controller_obj, "reset"):
        controller_obj.reset()

    total_reward = 0.0
    boundary_clips = 0
    emergency_overrides = 0
    adequacy_interventions = 0
    max_decomp_err = 0.0
    timeline_rows: List[Dict[str, Any]] = []
    step_obs_samples: List[Dict[str, Any]] = []
    done = False

    while env.agents and not done:
        step_idx = int(env.current_step)
        pre_state = dict(env.episode_history[-1])
        obs_snapshot = {ag: np.asarray(obs[ag], dtype=np.float32).copy() for ag in AGENTS if ag in obs}

        if controller_type == "mappo":
            raw_actions, _ = controller_obj.select_actions(obs, deterministic=True)
        elif controller_type in ("greedy", "rule_based"):
            raw_actions = controller_obj.select_actions(obs, pre_state)
        else:
            raise ValueError(f"Unknown controller_type: {controller_type}")

        obs, rewards, terms, truncs, infos = env.step(raw_actions)
        shared_reward = float(rewards.get("battery", 0.0))
        total_reward += shared_reward
        done = any(terms.values()) or any(truncs.values())

        post_state = dict(env.episode_history[-1])
        bat_info = infos.get("battery", {})
        shield_diag = bat_info.get("shield_diagnostics", {}) or {}
        adeq_diag = bat_info.get("adequacy_diagnostics", {}) or {}
        reward_breakdown = bat_info.get("reward_breakdown", {}) or {}

        if shield_diag.get("any_boundary", False):
            boundary_clips += 1
        if shield_diag.get("any_emergency", False):
            emergency_overrides += 1
        if adeq_diag.get("adequacy_intervention", False):
            adequacy_interventions += 1

        before_len = len(explainer.reward_records)
        explainer.record_step(
            scenario_id=scenario_id,
            eval_seed=eval_seed,
            step_idx=step_idx,
            pre_state=pre_state,
            obs_dict=obs_snapshot,
            raw_actions=raw_actions,
            post_state=post_state,
            shield_diag=shield_diag,
            adeq_diag=adeq_diag,
            reward_breakdown=reward_breakdown,
            shared_reward=shared_reward,
        )
        if len(explainer.reward_records) > before_len:
            err = float(explainer.reward_records[-1].get("decomposition_error", 0.0))
            if err > max_decomp_err:
                max_decomp_err = err

        if capture_timeline:
            hour_val = float(pre_state.get("hour", step_idx * 0.25))
            timeline_rows.append({
                "controller": controller_name,
                "mode": mode_label,
                "scenario": scenario_id,
                "step": step_idx,
                "hour": round(hour_val, 2),
                "solar_mw": float(post_state.get("solar_mw", 0.0)),
                "wind_mw": float(post_state.get("wind_mw", 0.0)),
                "solar_available_mw": float(post_state.get("solar_available_mw", 0.0)),
                "wind_available_mw": float(post_state.get("wind_available_mw", 0.0)),
                "nominal_critical_mw": float(post_state.get("nominal_critical_mw", 0.0)),
                "nominal_flexible_mw": float(post_state.get("nominal_flexible_mw", 0.0)),
                "nominal_other_mw": float(post_state.get("nominal_other_mw", 0.0)),
                "nominal_total_mw": float(post_state.get("nominal_total_mw", 0.0)),
                "served_critical_mw": float(post_state.get("served_critical_mw", 0.0)),
                "served_flexible_mw": float(post_state.get("served_flexible_mw", 0.0)),
                "served_other_mw": float(post_state.get("served_other_mw", 0.0)),
                "shed_critical_mw": float(post_state.get("shed_critical_mw", 0.0)),
                "shed_flexible_mw": float(post_state.get("shed_flexible_mw", 0.0)),
                "shed_other_mw": float(post_state.get("shed_other_mw", 0.0)),
                "battery_mw": float(post_state.get("battery_mw", 0.0)),
                "battery_soc": float(post_state.get("battery_soc", 0.0)),
                "generator_mw": float(post_state.get("generator_mw", 0.0)),
                "generator_state": str(post_state.get("generator_state", "OFF")),
                "step_fuel_l": float(post_state.get("step_fuel_l", 0.0)),
                "grid_connected": bool(post_state.get("grid_connected", True)),
                "grid_available": bool(post_state.get("grid_available", True)),
                "grid_import_mw": float(post_state.get("grid_import_mw", 0.0)),
                "grid_export_mw": float(post_state.get("grid_export_mw", 0.0)),
                "pcc_voltage_pu": float(post_state.get("pcc_voltage_pu", 1.0)),
                "is_storm_warning": bool(post_state.get("is_storm_warning", False)),
            })

        if capture_obs_list:
            step_obs_samples.append({
                "step": step_idx,
                "hour": round(float(pre_state.get("hour", step_idx * 0.25)), 2),
                "regime": _classify_regime(pre_state),
                "obs": obs_snapshot,
            })

    econ = compute_episode_economics(env.episode_history, dt_hours=env.dt_hours, config=econ_cfg)
    ep_summary = {
        "controller": controller_name,
        "mode": mode_label,
        "scenario": scenario_id,
        "eval_seed": int(eval_seed),
        "total_reward": round(float(total_reward), 4),
        "critical_reliability_pct": round(float(econ["critical_reliability_pct"]), 4),
        "total_reliability_pct": round(float(econ["total_reliability_pct"]), 4),
        "critical_ens_mwh": round(float(econ["critical_ens_mwh"]), 6),
        "flexible_ens_mwh": round(float(econ["flexible_ens_mwh"]), 6),
        "other_ens_mwh": round(float(econ["other_ens_mwh"]), 6),
        "total_ens_mwh": round(float(econ["total_ens_mwh"]), 6),
        "operating_cost_usd": round(float(econ["operating_cost_usd"]), 4),
        "true_social_cost_usd": round(float(econ["true_social_cost_usd"]), 4),
        "fuel_liters": round(float(econ["fuel_consumed_l"]), 4),
        "battery_throughput_mwh": round(float(econ["battery_throughput_mwh"]), 6),
        "generator_startup_count": int(econ["generator_startup_count"]),
        "generator_operating_hours": round(float(econ["generator_operating_hours"]), 4),
        "renewable_curtailment_mwh": round(float(econ["renewable_curtailment_mwh"]), 6),
        "boundary_clips": int(boundary_clips),
        "emergency_overrides": int(emergency_overrides),
        "adequacy_interventions": int(adequacy_interventions),
        "max_reward_decomposition_error": round(float(max_decomp_err), 8),
    }
    return ep_summary, timeline_rows, step_obs_samples


def compute_h2_attributions(
    mappo_agent: Any,
    h2_obs_samples: List[Dict[str, Any]],
    ig_steps: int,
    pert_delta: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute IG and perturbation sensitivity across all H2 steps + validation summary."""
    attr_rows: List[Dict[str, Any]] = []
    for sample in h2_obs_samples:
        step_idx = sample["step"]
        hour = sample["hour"]
        regime = sample["regime"]
        obs_dict = sample["obs"]
        for ag in AGENTS:
            actor = mappo_agent.actors[ag]
            obs_vec = obs_dict[ag]
            ig_res = compute_integrated_gradients(actor, ag, obs_vec, n_steps=ig_steps)
            pert_res = compute_perturbation_sensitivity(actor, ag, obs_vec, delta=pert_delta)

            ig_map = ig_res["attributions"]
            sens_map = pert_res["sensitivities"]
            dir_map = pert_res["directional_deltas"]
            comp_err = float(ig_res["completeness_error"])

            for feat, ig_val in ig_map.items():
                attr_rows.append({
                    "step": int(step_idx),
                    "hour": float(hour),
                    "regime": regime,
                    "agent": ag,
                    "feature": feat,
                    "ig_attribution": round(float(ig_val), 6),
                    "abs_ig": round(abs(float(ig_val)), 6),
                    "ig_completeness_error": round(comp_err, 6),
                    "pert_sensitivity": round(float(sens_map.get(feat, 0.0)), 6),
                    "pert_directional": round(float(dir_map.get(feat, 0.0)), 6),
                })

    # Controlled semantic perturbation validation on representative healthy & outage steps
    val_rows: List[Dict[str, Any]] = []
    seen_regimes = set()
    for target_reg in ("healthy_grid", "islanded_outage"):
        match = next((s for s in h2_obs_samples if s["regime"] == target_reg), None)
        if match is not None:
            seen_regimes.add(target_reg)
            for r in run_controlled_perturbation_validation(mappo_agent, match["obs"]):
                r["sample_step"] = match["step"]
                r["sample_hour"] = match["hour"]
                r["sample_regime"] = target_reg
                val_rows.append(r)

    if not val_rows and h2_obs_samples:
        s0 = h2_obs_samples[0]
        for r in run_controlled_perturbation_validation(mappo_agent, s0["obs"]):
            r["sample_step"] = s0["step"]
            r["sample_hour"] = s0["hour"]
            r["sample_regime"] = s0["regime"]
            val_rows.append(r)

    return pd.DataFrame(attr_rows), pd.DataFrame(val_rows)


def main():
    t0 = time.time()
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cand = cfg["primary_candidate"]
    ckpt_path = ROOT / cand["checkpoint_path"]
    forecast_mode = cand["forecast_mode"]
    forecast_models_dir = str(ROOT / cand["forecast_models_dir"])
    reward_cfg_path = str(ROOT / cfg["configs"]["reward_cfg"])
    econ_cfg_path = str(ROOT / cfg["configs"]["economics_cfg"])
    econ_cfg = EconomicConfig.from_yaml(econ_cfg_path)

    # 1. Verify checkpoint provenance
    print("[*] Verifying Phase 4.4.4 Bat-Only Seed 202 checkpoint provenance...", flush=True)
    meta = verify_checkpoint_metadata(
        ckpt_path,
        expected_experiment="bat_only",
        expected_seed=202,
        expected_episode=int(cand["expected_best_episode"]),
    )
    val_score = float(meta["metrics"].get("val_score", 0.0))
    if abs(val_score - float(cand["expected_val_score"])) > 0.5:
        raise ValueError(
            f"Validation score mismatch: expected ~{cand['expected_val_score']}, got {val_score}"
        )
    with open(RES_DIR / "verified_checkpoint_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[+] Verified: episode={meta['episode']}, seed={meta['seed']}, val_score={val_score:.2f}", flush=True)

    # 2. Load policy checkpoint
    mappo_agent = load_mappo_checkpoint(ckpt_path, forecast_mode=forecast_mode)

    scenarios = list(cfg["evaluation"]["scenarios"])
    h2_scen = str(cfg["evaluation"]["detailed_trace_scenario"])
    eval_seed = int(cfg["evaluation"]["eval_seed"])

    controllers = [
        ("Bat-Only (Seed 202) [Intrinsic]", "Intrinsic", "mappo", mappo_agent, False, forecast_mode),
        ("Bat-Only (Seed 202) [Hybrid +Fallback]", "Hybrid (+Fallback)", "mappo", mappo_agent, True, forecast_mode),
        ("Greedy Controller", "Heuristic", "greedy", GreedyController(), False, None),
        ("Rule-Based EMS", "Heuristic", "rule_based", RuleBasedEMS(), False, None),
    ]

    scen_rows: List[Dict[str, Any]] = []
    all_reward_rows: List[Dict[str, Any]] = []
    all_intervention_rows: List[Dict[str, Any]] = []
    h2_timeline_rows: List[Dict[str, Any]] = []
    h2_agent_records: List[Dict[str, Any]] = []
    h2_obs_samples: List[Dict[str, Any]] = []

    print(f"[*] Running traced evaluations across {len(scenarios)} held-out scenarios...", flush=True)
    env_cache: Dict[Tuple[Optional[str], bool], MicrogridParallelEnv] = {}
    for cname, mode_lbl, ctype, cobj, fb_enabled, fmode in controllers:
        cache_key = (fmode, bool(fb_enabled))
        if cache_key not in env_cache:
            env_cache[cache_key] = MicrogridParallelEnv(
                scenario_id=scenarios[0],
                forecast_mode=fmode,
                forecast_models_dir=forecast_models_dir,
                reward_config_path=reward_cfg_path,
                enable_adequacy_check=True,
                adequacy_fallback=bool(fb_enabled),
                enable_constraint_costs=True,
            )
        cached_env = env_cache[cache_key]
        for scen in scenarios:
            explainer = DecisionExplainer(controller_name=cname, mode=mode_lbl)
            is_h2 = (scen == h2_scen)
            capture_obs = is_h2 and (cname == "Bat-Only (Seed 202) [Intrinsic]")

            ep_sum, tl_rows, obs_samples = run_traced_episode(
                controller_name=cname,
                mode_label=mode_lbl,
                controller_type=ctype,
                controller_obj=cobj,
                scenario_id=scen,
                eval_seed=eval_seed,
                adequacy_fallback=fb_enabled,
                forecast_mode=fmode,
                forecast_models_dir=forecast_models_dir,
                reward_cfg_path=reward_cfg_path,
                econ_cfg=econ_cfg,
                explainer=explainer,
                capture_timeline=is_h2,
                capture_obs_list=capture_obs,
                env=cached_env,
            )
            scen_rows.append(ep_sum)
            all_reward_rows.extend(explainer.reward_records)
            all_intervention_rows.extend(explainer.intervention_records)
            if is_h2:
                h2_timeline_rows.extend(tl_rows)
                h2_agent_records.extend(explainer.agent_records)
            if capture_obs:
                h2_obs_samples = obs_samples

        print(f"    - Completed {cname}", flush=True)

    scen_df = pd.DataFrame(scen_rows)
    rew_df = pd.DataFrame(all_reward_rows)
    int_df = pd.DataFrame(all_intervention_rows)
    tl_df = pd.DataFrame(h2_timeline_rows)

    max_err = float(rew_df["decomposition_error"].max()) if not rew_df.empty else 0.0
    if max_err >= 1e-4:
        raise AssertionError(f"Reward decomposition error exceeded 1e-4: max={max_err}")
    print(f"[+] Reward decomposition exactness verified across {len(rew_df)} steps (max error = {max_err:.8f}).", flush=True)

    scen_df.to_csv(RES_DIR / "scenario_comparison_traces.csv", index=False)
    rew_df.to_csv(RES_DIR / "reward_decomposition_all.csv", index=False)
    int_df.to_csv(RES_DIR / "safety_intervention_trace.csv", index=False)
    tl_df.to_csv(RES_DIR / "h2_system_timeline.csv", index=False)

    with open(RES_DIR / "h2_agent_decision_explanations.json", "w", encoding="utf-8") as f:
        json.dump(h2_agent_records, f, indent=2)

    flat_df = pd.DataFrame([
        {
            "controller": r["controller"],
            "mode": r["mode"],
            "scenario": r["scenario"],
            "step": r["step"],
            "hour": r["hour"],
            "timestamp": r["timestamp"],
            "agent": r["agent"],
            "intervention_stage": r["intervention_stage"],
            "shield_modified": r["shield_modified"],
            "fallback_modified": r["fallback_modified"],
            "reward_total": r["reward_total"],
            "explanation": r["explanation"],
        }
        for r in h2_agent_records
    ])
    flat_df.to_csv(RES_DIR / "h2_agent_explanations_flat.csv", index=False)

    # 3. Feature Attributions (Integrated Gradients + Perturbation Sensitivity)
    print("[*] Computing Integrated Gradients & perturbation sensitivities on H2...", flush=True)
    attr_df, val_df = compute_h2_attributions(
        mappo_agent=mappo_agent,
        h2_obs_samples=h2_obs_samples,
        ig_steps=int(cfg["attribution"]["ig_steps"]),
        pert_delta=float(cfg["attribution"]["perturbation_delta"]),
    )
    if not np.all(np.isfinite(attr_df[["ig_attribution", "abs_ig", "ig_completeness_error", "pert_sensitivity"]].values)):
        raise ValueError("Non-finite values detected in feature attributions.")
    attr_df.to_csv(RES_DIR / "feature_attributions_h2.csv", index=False)
    val_df.to_csv(RES_DIR / "controlled_perturbation_validation.csv", index=False)
    print(f"[+] Saved {len(attr_df)} feature attribution rows (max IG completeness err = {attr_df['ig_completeness_error'].max():.6f}).", flush=True)

    # 4. Controlled Counterfactual Suite
    print("[*] Running controlled counterfactual simulations on H2 & H6...", flush=True)
    cf_df = run_counterfactual_suite(
        mappo_agent=mappo_agent,
        controller_name="Bat-Only (Seed 202)",
        scenarios=cfg["counterfactuals"]["scenarios"],
        eval_seed=eval_seed,
        forecast_models_dir=forecast_models_dir,
        reward_cfg_path=reward_cfg_path,
        econ_cfg_path=econ_cfg_path,
        cf_names=cfg["counterfactuals"]["experiments"],
    )
    cf_df.to_csv(RES_DIR / "counterfactual_results.csv", index=False)
    print(f"[+] Saved {len(cf_df)} counterfactual records. Total elapsed: {time.time() - t0:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
