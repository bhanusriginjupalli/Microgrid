"""
comparison.py
=============
Timestep-aligned multi-episode retrieval and paired comparison utilities for
the Phase 4 Saved Episode Library. Validates scenario & evaluation seed compatibility,
reports timestep mismatches explicitly, preserves training seeds visibly, and computes
timestep-level differences cleanly relative to a reference episode.
"""

from typing import Any, Dict, List
from experiments.phase6_model_comparison.episode_library.library import EpisodeLibrary

DIFF_METRICS = [
    "battery_soc_pct",
    "battery_mw",
    "generator_mw",
    "grid_import_mw",
    "solar_mw",
    "wind_mw",
    "renewable_curtailment_mw",
    "served_critical_mw",
    "shed_critical_mw",
    "step_operating_cost_usd",
    "step_social_cost_usd",
    "reward",
]


def compare_episodes(
    library: EpisodeLibrary,
    episode_ids: List[str],
    require_same_scenario: bool = True,
    require_same_eval_seed: bool = True,
) -> Dict[str, Any]:
    """
    Compare 2+ saved episodes with strict scenario/seed compatibility validation
    and timestep-aligned difference calculation relative to episode_ids[0].
    """
    if len(episode_ids) < 2:
        raise ValueError("compare_episodes requires at least 2 episode IDs.")
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("Duplicate episode_ids provided to compare_episodes.")

    loaded = [library.get_episode_trace(eid) for eid in episode_ids]
    ref = loaded[0]
    ref_meta = ref["metadata"]
    ref_steps = ref["steps"]

    # Validate compatibility
    for ep in loaded[1:]:
        m = ep["metadata"]
        if require_same_scenario and m["scenario_id"] != ref_meta["scenario_id"]:
            raise ValueError(
                f"Incompatible scenario_id for paired comparison: "
                f"{ref_meta['episode_id']} ({ref_meta['scenario_id']}) vs "
                f"{m['episode_id']} ({m['scenario_id']})."
            )
        if require_same_eval_seed and int(m["evaluation_seed"]) != int(ref_meta["evaluation_seed"]):
            raise ValueError(
                f"Incompatible evaluation_seed for paired comparison: "
                f"{ref_meta['episode_id']} ({ref_meta['evaluation_seed']}) vs "
                f"{m['episode_id']} ({m['evaluation_seed']})."
            )

    # Check timestep and hour alignment
    mismatches: List[str] = []
    ref_map = {int(s["step_idx"]): s for s in ref_steps}
    common_indices = set(ref_map.keys())
    for ep in loaded[1:]:
        s_map = {int(s["step_idx"]): s for s in ep["steps"]}
        if len(ep["steps"]) != len(ref_steps):
            mismatches.append(
                f"Step count mismatch: {ep['episode_id']} has {len(ep['steps'])} vs {len(ref_steps)} steps."
            )
        missing = set(ref_map.keys()) ^ set(s_map.keys())
        if missing:
            mismatches.append(
                f"Step index mismatch for {ep['episode_id']}: symmetric diff count {len(missing)}."
            )
        common_indices &= set(s_map.keys())
        for idx in sorted(common_indices):
            if abs(float(s_map[idx]["hour"]) - float(ref_map[idx]["hour"])) > 1e-4:
                mismatches.append(
                    f"Timestamp mismatch at step {idx} for {ep['episode_id']}: "
                    f"{s_map[idx]['hour']} vs {ref_map[idx]['hour']}."
                )
                break

    if mismatches:
        raise ValueError("Timestep alignment failure: " + " | ".join(mismatches))

    ordered_idx = sorted(common_indices)
    aligned_steps: List[Dict[str, Any]] = []
    ep_step_maps = {ep["episode_id"]: {int(s["step_idx"]): s for s in ep["steps"]} for ep in loaded}

    for idx in ordered_idx:
        ref_s = ref_map[idx]
        per_ep_vals: Dict[str, Any] = {}
        deltas_vs_ref: Dict[str, Any] = {}

        for ep in loaded:
            eid = ep["episode_id"]
            st = ep_step_maps[eid][idx]
            per_ep_vals[eid] = {m: st[m] for m in DIFF_METRICS}
            per_ep_vals[eid]["generator_state"] = st["generator_state"]
            per_ep_vals[eid]["shield_intervention"] = bool(st["shield_events"]["any_intervention"])
            per_ep_vals[eid]["fallback_intervention"] = bool(st["fallback_events"]["adequacy_intervention"])
            per_ep_vals[eid]["converged"] = bool(st["converged"])

            if eid != ref["episode_id"]:
                deltas_vs_ref[eid] = {
                    m: round(float(st[m]) - float(ref_s[m]), 6)
                    for m in DIFF_METRICS
                }

        aligned_steps.append({
            "step_idx": idx,
            "hour": ref_s["hour"],
            "timestamp_label": ref_s["timestamp_label"],
            "scenario_events": ref_s["scenario_events"],
            "values": per_ep_vals,
            "deltas_vs_reference": deltas_vs_ref,
        })

    # Episode-level summary comparison (preserving training_seed visibly)
    ref_metrics = ref["metrics"]
    summary_rows: List[Dict[str, Any]] = []
    for ep in loaded:
        m = ep["metadata"]
        met = ep["metrics"]
        summary_rows.append({
            "episode_id": ep["episode_id"],
            "algorithm": m["algorithm"],
            "controller_name": m["controller_name"],
            "track": m["track"],
            "mode": m["mode"],
            "training_seed": m["training_seed"],
            "evaluation_seed": m["evaluation_seed"],
            "scenario_id": m["scenario_id"],
            "true_social_cost_usd": met["true_social_cost_usd"],
            "operating_cost_usd": met["operating_cost_usd"],
            "critical_ens_mwh": met["critical_ens_mwh"],
            "critical_reliability_pct": met["critical_reliability_pct"],
            "renewable_curtailment_mwh": met["renewable_curtailment_mwh"],
            "fuel_consumed_l": met["fuel_consumed_l"],
            "battery_throughput_mwh": met["battery_throughput_mwh"],
            "terminal_battery_soc_pct": met["terminal_battery_soc_pct"],
            "boundary_clipping_count": met["boundary_clipping_count"],
            "emergency_override_count": met["emergency_override_count"],
            "fallback_intervention_steps": met["fallback_intervention_steps"],
            "delta_social_cost_vs_ref_usd": round(float(met["true_social_cost_usd"]) - float(ref_metrics["true_social_cost_usd"]), 4),
            "delta_critical_ens_vs_ref_mwh": round(float(met["critical_ens_mwh"]) - float(ref_metrics["critical_ens_mwh"]), 6),
            "delta_critical_rel_vs_ref_pct": round(float(met["critical_reliability_pct"]) - float(ref_metrics["critical_reliability_pct"]), 4),
        })

    return {
        "reference_episode_id": ref["episode_id"],
        "scenario_id": ref_meta["scenario_id"],
        "evaluation_seed": ref_meta["evaluation_seed"],
        "aligned_step_count": len(aligned_steps),
        "episodes_compared": summary_rows,
        "aligned_timesteps": aligned_steps,
    }
