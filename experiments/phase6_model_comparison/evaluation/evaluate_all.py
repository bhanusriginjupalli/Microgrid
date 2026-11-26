"""
evaluate_all.py
===============
Common held-out comparative evaluator for Phase 3:
Evaluates IPPO, MAPPO_MATCHED, HAPPO (all seeds [42, 101, 202]) alongside
Greedy EMS, Rule-Based EMS, and Lookahead Heuristic EMS on identical
(H1-H6 x [42, 101, 202, 303, 404]) realizations, separating:
  - Track A: Intrinsic (enable_safety_shield=False, adequacy_fallback=False) + Heuristics
  - Track B: Shielded  (enable_safety_shield=True,  adequacy_fallback=False)
  - Track C: Hybrid    (enable_safety_shield=True,  adequacy_fallback=True)
Records both episode summary rows and full 96-step replay traces.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "phase1_digital_twin") not in sys.path:
    sys.path.insert(0, str(ROOT / "phase1_digital_twin"))

from rl_environment.microgrid_parallel_env import MicrogridParallelEnv
from rl_environment.safety_shield import SafetyShield
from rl_environment.economic_accounting import EconomicConfig
from experiments.phase6_model_comparison.evaluation.evaluation_protocol import (
    TRAIN_SEEDS,
    EVAL_SEEDS,
    HELDOUT_SCENARIOS,
    TRACK_CONFIGS,
    make_episode_id,
    set_episode_seeds,
)
from experiments.phase6_model_comparison.evaluation.controller_registry import (
    ControllerSpec,
    build_controller_specs,
)
from experiments.phase6_model_comparison.evaluation.episode_metrics import summarize_episode_metrics
from experiments.phase6_model_comparison.evaluation.trace_recorder import (
    build_step_trace_record,
    save_episode_trace,
)
from experiments.phase6_model_comparison.evaluation.statistical_analysis import (
    compute_seed_and_family_summaries,
    compute_paired_comparisons,
)


def _describe_active_events(scenario: Any, hour: float) -> List[str]:
    events: List[str] = []
    if not scenario.is_grid_available(hour):
        events.append("GRID_OUTAGE")
    if scenario.is_storm_warning(hour):
        events.append("STORM_WARNING")
    if scenario.get_cloud_factor(hour) < 0.99:
        events.append("CLOUD_TRANSIENT")
    if hasattr(scenario, "is_forecast_frozen") and scenario.is_forecast_frozen(hour):
        events.append("FORECAST_FROZEN")
    if hasattr(scenario, "get_generator_derating_factor") and scenario.get_generator_derating_factor(hour) < 0.999:
        events.append("GENERATOR_DERATED")
    if hasattr(scenario, "get_line_in_service_map"):
        lmap = scenario.get_line_in_service_map(hour, total_lines=8)
        if not all(lmap.values()):
            events.append("LINE_FAULT")
    return events


def evaluate_single_episode(
    env: MicrogridParallelEnv,
    spec: ControllerSpec,
    track: str,
    mode: str,
    scenario_id: str,
    eval_seed: int,
    econ_cfg: EconomicConfig,
    trace_dir: Optional[Path] = None,
    shadow_shield: Optional[SafetyShield] = None,
) -> Dict[str, Any]:
    """Run one 96-step episode cleanly and save full trace if trace_dir is set."""
    ep_id = make_episode_id(track, spec.algorithm_slug, spec.train_seed, scenario_id, eval_seed)
    set_episode_seeds(eval_seed)
    obs, _ = env.reset(seed=int(eval_seed), options={"scenario_id": scenario_id})
    done = False
    t_ep0 = time.perf_counter()

    step_telemetry: List[Dict[str, Any]] = []
    trace_steps: List[Dict[str, Any]] = []
    step_idx = 0

    while env.agents and not done:
        prev_state = env.episode_history[-1]
        obs_snapshot = {ag: np.array(v, copy=True) for ag, v in obs.items()}
        proposed_actions, lat_ms = spec.select_actions(obs, prev_state)

        # Shadow safety-shield check when shield is bypassed (Track A) so proposal feasibility is still audited
        shadow_diag: Optional[Dict[str, Any]] = None
        if not env.enable_safety_shield and shadow_shield is not None:
            _, shadow_diag = shadow_shield.filter_actions(
                raw_actions=proposed_actions,
                current_state=prev_state,
                dt_hours=env.dt_hours,
            )

        next_obs, rewards, terms, truncs, infos = env.step(proposed_actions)
        done = any(terms.values()) or any(truncs.values())
        st = env.episode_history[-1]
        info_bat = infos.get("battery", {})
        shield_diag = info_bat.get("shield_diagnostics", {})
        adeq_diag = info_bat.get("adequacy_diagnostics", {})
        reward_bd = info_bat.get("reward_breakdown", {})
        r_step = float(rewards.get("battery", 0.0))

        # Executed actions equal adequacy/shield-filtered actions when active, else proposed
        diag_for_feas = shield_diag if env.enable_safety_shield else (shadow_diag or shield_diag)
        action_cls = diag_for_feas.get("action_classification", {})
        num_agents = len(action_cls) if action_cls else len(env.possible_agents)
        feasible_agents = sum(1 for v in action_cls.values() if v == "FEASIBLE") if action_cls else num_agents

        executed_actions = diag_for_feas.get("executed_actions", proposed_actions)
        # Fallback may also adjust executed battery/generator physical commands: track true applied power cleanly
        applied_actions = {
            **executed_actions,
            "applied_battery_mw": float(st.get("battery_mw", 0.0)),
            "applied_generator_mw": float(st.get("generator_mw", 0.0)),
        }

        step_telemetry.append({
            "reward": r_step,
            "num_agents": int(num_agents),
            "feasible_agents": int(feasible_agents),
            "boundary_clipping": bool(diag_for_feas.get("any_boundary", False)),
            "emergency_override": bool(diag_for_feas.get("any_emergency", False)),
            "shield_intervention": bool(shield_diag.get("any_intervention", False)) if env.enable_safety_shield else False,
            "inaction_detected": bool(adeq_diag.get("inaction_detected", False)),
            "adequacy_intervention": bool(adeq_diag.get("adequacy_intervention", False)),
            "fallback_intervention": bool(env.adequacy_fallback and adeq_diag.get("adequacy_intervention", False)),
            "inference_latency_ms": float(lat_ms),
        })

        if trace_dir is not None:
            sc_events = _describe_active_events(env.scenario, float(st.get("hour", step_idx * env.dt_hours)))
            trace_steps.append(
                build_step_trace_record(
                    step_idx=step_idx,
                    obs_dict=obs_snapshot,
                    proposed_actions=proposed_actions,
                    executed_actions=applied_actions,
                    state_dict=st,
                    reward=r_step,
                    reward_breakdown=reward_bd,
                    shield_diag=diag_for_feas,
                    adequacy_diag=adeq_diag,
                    scenario_events=sc_events,
                    dt_hours=env.dt_hours,
                    econ_cfg=econ_cfg,
                    adequacy_fallback=env.adequacy_fallback,
                )
            )

        obs = next_obs
        step_idx += 1

    ep_wall = time.perf_counter() - t_ep0
    summary = summarize_episode_metrics(
        episode_id=ep_id,
        track=track,
        mode=mode,
        controller_name=spec.controller_name,
        algorithm=spec.algorithm,
        controller_group=spec.controller_group,
        train_seed=spec.train_seed,
        scenario=scenario_id,
        eval_seed=eval_seed,
        episode_history=env.episode_history,
        step_telemetry=step_telemetry,
        dt_hours=env.dt_hours,
        econ_cfg=econ_cfg,
        enable_safety_shield=env.enable_safety_shield,
        adequacy_fallback=env.adequacy_fallback,
        episode_wall_s=ep_wall,
        checkpoint_path=spec.checkpoint_path,
    )

    if trace_dir is not None:
        meta = {
            "episode_id": ep_id,
            "track": track,
            "mode": mode,
            "controller": spec.controller_name,
            "algorithm": spec.algorithm,
            "controller_group": spec.controller_group,
            "train_seed": spec.train_seed,
            "scenario": scenario_id,
            "eval_seed": int(eval_seed),
            "enable_safety_shield": bool(env.enable_safety_shield),
            "adequacy_fallback": bool(env.adequacy_fallback),
            "checkpoint_path": spec.checkpoint_path,
        }
        trace_path = save_episode_trace(
            trace_dir=trace_dir,
            episode_id=ep_id,
            metadata=meta,
            summary_metrics=summary,
            steps=trace_steps,
            expected_steps=env.max_steps,
        )
        summary["trace_path"] = str(trace_path)

    return summary


def run_evaluation(
    results_root: Path,
    track_selection: str = "all",
    scenarios: Optional[List[str]] = None,
    eval_seeds: Optional[List[int]] = None,
    train_seeds: Optional[List[int]] = None,
    checkpoint_tag: str = "best_model.pt",
    use_smoke_if_missing: bool = False,
    save_traces: bool = True,
    max_steps: int = 96,
    output_subdir: str = "evaluation",
) -> Dict[str, Any]:
    """Run held-out comparative evaluation and write episode, summary, stat, and trace artifacts."""
    out_dir = results_root / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_root = (out_dir / "traces") if save_traces else None

    econ_cfg = EconomicConfig.from_yaml(str(ROOT / "experiments" / "phase4_4_4" / "configs" / "economics_v1.yaml"))
    reward_cfg = str(ROOT / "experiments" / "phase4_4_4" / "configs" / "reward_v3_social.yaml")
    models_dir = str(ROOT / "experiments" / "phase4" / "forecasting" / "models")

    sc_list = list(scenarios or HELDOUT_SCENARIOS)
    ev_seeds = list(eval_seeds or EVAL_SEEDS)
    tr_seeds = list(train_seeds or TRAIN_SEEDS)

    if track_selection == "all":
        selected_tracks = ["track_a_intrinsic", "track_b_shielded", "track_c_hybrid"]
    else:
        selected_tracks = [track_selection]

    # Load all learned controllers once and heuristics once
    learned_specs = build_controller_specs(
        results_root=results_root,
        checkpoint_tag=checkpoint_tag,
        train_seeds=tr_seeds,
        include_heuristics=False,
        include_learned=True,
        use_smoke_if_missing=use_smoke_if_missing,
    )
    heuristic_specs = build_controller_specs(
        results_root=results_root,
        include_heuristics=True,
        include_learned=False,
    )

    episode_rows: List[Dict[str, Any]] = []
    shadow_shield = SafetyShield()

    for track_key in selected_tracks:
        tcfg = TRACK_CONFIGS[track_key]
        print(f"[*] Evaluating {tcfg['label']} ({len(learned_specs)} RL + {len(heuristic_specs) if tcfg['include_heuristics'] else 0} Heuristic specs)...", flush=True)

        env_rl = MicrogridParallelEnv(
            scenario_id=sc_list[0],
            max_steps=max_steps,
            enable_safety_shield=tcfg["enable_safety_shield"],
            forecast_mode="realistic",
            forecast_models_dir=models_dir,
            reward_config_path=reward_cfg,
            enable_adequacy_check=tcfg["enable_adequacy_check"],
            adequacy_fallback=tcfg["adequacy_fallback"],
        )
        t_dir = (trace_root / track_key) if trace_root else None

        for spec in learned_specs:
            for sc in sc_list:
                for es in ev_seeds:
                    row = evaluate_single_episode(
                        env=env_rl,
                        spec=spec,
                        track=track_key,
                        mode=tcfg["mode"],
                        scenario_id=sc,
                        eval_seed=es,
                        econ_cfg=econ_cfg,
                        trace_dir=t_dir,
                        shadow_shield=shadow_shield,
                    )
                    episode_rows.append(row)

        if tcfg["include_heuristics"]:
            # Heuristics evaluate in their native safety-shielded setup without an extra fallback wrapper
            env_heur = MicrogridParallelEnv(
                scenario_id=sc_list[0],
                max_steps=max_steps,
                enable_safety_shield=True,
                forecast_mode="realistic",
                forecast_models_dir=models_dir,
                reward_config_path=reward_cfg,
                enable_adequacy_check=True,
                adequacy_fallback=False,
            )
            h_dir = (trace_root / "heuristic_native") if trace_root else None
            for spec in heuristic_specs:
                for sc in sc_list:
                    for es in ev_seeds:
                        row = evaluate_single_episode(
                            env=env_heur,
                            spec=spec,
                            track=track_key,
                            mode="Heuristic",
                            scenario_id=sc,
                            eval_seed=es,
                            econ_cfg=econ_cfg,
                            trace_dir=h_dir,
                            shadow_shield=None,
                        )
                        episode_rows.append(row)

    ep_df = pd.DataFrame(episode_rows)
    assert ep_df["episode_id"].is_unique, "Duplicate episode_id detected!"
    ep_df.to_csv(out_dir / "eval_episodes.csv", index=False)

    seed_df, family_df = compute_seed_and_family_summaries(ep_df)
    seed_df.to_csv(out_dir / "seed_level_summary.csv", index=False)
    family_df.to_csv(out_dir / "family_summary.csv", index=False)

    stats_df = compute_paired_comparisons(ep_df)
    stats_df.to_csv(out_dir / "paired_statistical_tests.csv", index=False)

    scen_df = (
        ep_df.groupby(["track", "mode", "algorithm", "scenario"], as_index=False)
        .agg(
            episodes_n=("episode_id", "count"),
            true_social_cost_mean=("true_social_cost_usd", "mean"),
            true_social_cost_std=("true_social_cost_usd", "std"),
            operating_cost_mean=("operating_cost_usd", "mean"),
            critical_ens_mean=("critical_ens_mwh", "mean"),
            critical_reliability_mean=("critical_reliability_pct", "mean"),
            renewable_curtailment_mean=("renewable_curtailment_mwh", "mean"),
            fuel_consumed_mean=("fuel_consumed_l", "mean"),
            battery_throughput_mean=("battery_throughput_mwh", "mean"),
            shield_interventions_mean=("shield_intervention_steps", "mean"),
            fallback_interventions_mean=("fallback_intervention_steps", "mean"),
        )
    )
    scen_df.to_csv(out_dir / "scenario_summary.csv", index=False)

    print(f"[*] Evaluation complete: {len(ep_df)} episodes written to {out_dir}")
    return {
        "output_dir": str(out_dir),
        "total_episodes": int(len(ep_df)),
        "tracks_evaluated": selected_tracks,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Phase 3 multi-algorithm held-out evaluation")
    parser.add_argument("--results-root", type=Path, default=ROOT / "experiments" / "phase6_model_comparison" / "results")
    parser.add_argument("--track", type=str, default="all", choices=["all", "track_a_intrinsic", "track_b_shielded", "track_c_hybrid"])
    parser.add_argument("--dry-run", action="store_true", help="Run 1 episode per controller (H1, seed=42) to verify trace schema & runner")
    parser.add_argument("--use-smoke-if-missing", action="store_true")
    parser.add_argument("--no-traces", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        res = run_evaluation(
            results_root=args.results_root,
            track_selection="track_a_intrinsic",
            scenarios=["scenario_h1"],
            eval_seeds=[42],
            train_seeds=[42],
            use_smoke_if_missing=True,
            save_traces=not args.no_traces,
            output_subdir="dry_run_evaluation",
        )
    else:
        res = run_evaluation(
            results_root=args.results_root,
            track_selection=args.track,
            use_smoke_if_missing=args.use_smoke_if_missing,
            save_traces=not args.no_traces,
        )
    print(f"[EVAL] {res}")


if __name__ == "__main__":
    main()
