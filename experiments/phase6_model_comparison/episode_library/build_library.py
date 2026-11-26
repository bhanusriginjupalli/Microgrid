"""
build_library.py
================
Builds the Phase 4 Saved Episode Library index, per-track manifests,
library manifest, and integrity report from completed Phase 3 evaluation artifacts
WITHOUT modifying or duplicating the 260.9 MB of immutable Phase 3 traces.
"""

import argparse
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase6_model_comparison.episode_library.schema_validator import (
    SCHEMA_VERSION,
    VALID_TRACKS,
)

RESULTS_ROOT = ROOT / "experiments" / "phase6_model_comparison" / "results"
LIB_ROOT = ROOT / "experiments" / "phase6_model_comparison" / "episode_library"

EXPERIMENT_ID = "phase6_model_comparison"
SUMMARY_REL_PATH = "experiments/phase6_model_comparison/results/evaluation/eval_episodes.csv"


def _load_selected_checkpoint_map(results_root: Path) -> Dict[tuple, int]:
    """Read selected_checkpoint_episode cleanly from Phase 3 run_summary.json files."""
    mapping: Dict[tuple, int] = {}
    for sub, algo in [("ippo", "IPPO"), ("mappo_matched", "MAPPO_MATCHED"), ("happo", "HAPPO")]:
        for seed_dir in (results_root / sub).glob("seed_*"):
            rs = seed_dir / "run_summary.json"
            if rs.exists():
                with open(rs, "r", encoding="utf-8") as f:
                    data = json.load(f)
                mapping[(algo, int(data["seed"]))] = int(data["best_checkpoint_episode"])
    return mapping


def _to_rel_posix(path_str: str, root: Path) -> str:
    p = Path(path_str)
    if not p.is_absolute():
        p = (root / p).resolve()
    try:
        return p.relative_to(root.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def build_episode_library(
    results_root: Path = RESULTS_ROOT,
    library_root: Path = LIB_ROOT,
    package_traces: bool = False,
    validate_all_traces: bool = True,
) -> Dict[str, Any]:
    """Construct canonical episode index, manifests, and integrity verification report."""
    eval_csv = results_root / "evaluation" / "eval_episodes.csv"
    if not eval_csv.exists():
        raise FileNotFoundError(f"Missing Phase 3 evaluation CSV: {eval_csv}")

    df = pd.read_csv(eval_csv)
    ckpt_ep_map = _load_selected_checkpoint_map(results_root)

    index_dir = library_root / "index"
    manifest_dir = library_root / "manifests"
    reports_dir = library_root / "reports"
    episodes_dir = library_root / "episodes"
    for d in (index_dir, manifest_dir, reports_dir, episodes_dir):
        d.mkdir(parents=True, exist_ok=True)
    for tr in VALID_TRACKS:
        (episodes_dir / tr).mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    seen_ids = set()
    duplicate_ids = 0
    missing_traces = 0
    invalid_step_counts = 0
    reward_mismatches = 0
    nonfinite_rows = 0
    total_trace_bytes = 0
    max_reward_diff = 0.0

    for _, r in df.iterrows():
        ep_id = str(r["episode_id"])
        if ep_id in seen_ids:
            duplicate_ids += 1
        seen_ids.add(ep_id)

        raw_seed = str(r["train_seed"])
        is_heur = str(r["controller_group"]) == "Heuristic Baseline" or raw_seed == "deterministic"
        train_seed: Optional[int] = None if is_heur else int(float(raw_seed))
        algo = str(r["algorithm"])
        sel_ckpt_ep: Optional[int] = None if is_heur else ckpt_ep_map.get((algo, int(train_seed)))

        src_trace_abs = Path(str(r["trace_path"]))
        if not src_trace_abs.is_absolute():
            src_trace_abs = (ROOT / src_trace_abs).resolve()

        rel_trace = _to_rel_posix(str(src_trace_abs), ROOT)
        packaged_rel_trace: Optional[str] = None

        trace_exists = src_trace_abs.exists()
        step_count = None
        if not trace_exists:
            missing_traces += 1
        else:
            total_trace_bytes += src_trace_abs.stat().st_size
            if validate_all_traces:
                with open(src_trace_abs, "r", encoding="utf-8") as f:
                    tr_json = json.load(f)
                steps = tr_json.get("steps", [])
                step_count = len(steps)
                if step_count != 96:
                    invalid_step_counts += 1
                sum_r = float(sum(float(s["reward"]) for s in steps))
                diff_r = abs(sum_r - float(r["reward"]))
                max_reward_diff = max(max_reward_diff, diff_r)
                if diff_r > 1e-2:
                    reward_mismatches += 1
            else:
                step_count = int(r["sim_steps"])

            if package_traces:
                dest = episodes_dir / str(r["track"]) / f"{ep_id}.json"
                if not dest.exists():
                    shutil.copy2(src_trace_abs, dest)
                packaged_rel_trace = _to_rel_posix(str(dest), ROOT)

        sim_steps = int(r["sim_steps"])
        pf_conv = float(r["power_flow_converged_pct"])
        if not trace_exists or sim_steps == 0:
            completion_status = "failed"
        elif sim_steps == 96 and pf_conv >= 99.9:
            completion_status = "completed"
        else:
            completion_status = "partial"

        metrics_dict = {
            "reward": float(r["reward"]),
            "true_social_cost_usd": float(r["true_social_cost_usd"]),
            "operating_cost_usd": float(r["operating_cost_usd"]),
            "grid_import_mwh": float(r["grid_import_mwh"]),
            "grid_export_mwh": float(r["grid_export_mwh"]),
            "grid_import_cost_usd": float(r["grid_import_cost_usd"]),
            "grid_export_revenue_usd": float(r["grid_export_revenue_usd"]),
            "fuel_consumed_l": float(r["fuel_consumed_l"]),
            "fuel_cost_usd": float(r["fuel_cost_usd"]),
            "generator_operating_hours": float(r["generator_operating_hours"]),
            "generator_startup_count": int(r["generator_startup_count"]),
            "startup_wear_cost_usd": float(r["startup_wear_cost_usd"]),
            "nominal_critical_mwh": float(r["nominal_critical_mwh"]),
            "served_critical_mwh": float(r["served_critical_mwh"]),
            "served_total_mwh": float(r["served_total_mwh"]),
            "critical_ens_mwh": float(r["critical_ens_mwh"]),
            "flexible_ens_mwh": float(r["flexible_ens_mwh"]),
            "other_ens_mwh": float(r["other_ens_mwh"]),
            "total_ens_mwh": float(r["total_ens_mwh"]),
            "critical_reliability_pct": float(r["critical_reliability_pct"]),
            "total_reliability_pct": float(r["total_reliability_pct"]),
            "available_renewable_mwh": float(r["available_renewable_mwh"]),
            "dispatched_renewable_mwh": float(r["dispatched_renewable_mwh"]),
            "renewable_curtailment_mwh": float(r["renewable_curtailment_mwh"]),
            "renewable_utilization_pct": float(r["renewable_utilization_pct"]),
            "battery_charge_mwh": float(r["battery_charge_mwh"]),
            "battery_discharge_mwh": float(r["battery_discharge_mwh"]),
            "battery_throughput_mwh": float(r["battery_throughput_mwh"]),
            "battery_degradation_cost_usd": float(r["battery_degradation_cost_usd"]),
            "initial_battery_soc_pct": float(r["initial_battery_soc_pct"]),
            "terminal_battery_soc_pct": float(r["terminal_battery_soc_pct"]),
            "terminal_soc_adjustment_usd": float(r["terminal_soc_adjustment_usd"]),
            "voltage_violation_steps": int(r["voltage_violation_steps"]),
            "power_flow_failures": int(r["power_flow_failures"]),
            "power_flow_converged_pct": float(r["power_flow_converged_pct"]),
            "unshielded_feasible_pct": float(r["unshielded_feasible_pct"]),
            "boundary_clipping_count": int(r["boundary_clipping_count"]),
            "emergency_override_count": int(r["emergency_override_count"]),
            "shield_intervention_steps": int(r["shield_intervention_steps"]),
            "inaction_detected_steps": int(r["inaction_detected_steps"]),
            "supply_adequacy_intervention_count": int(r["supply_adequacy_intervention_count"]),
            "fallback_intervention_steps": int(r["fallback_intervention_steps"]),
            "avoidable_critical_ens_mwh": float(r["avoidable_critical_ens_mwh"]),
            "unavoidable_critical_ens_mwh": float(r["unavoidable_critical_ens_mwh"]),
            "inference_latency_ms_mean": float(r["inference_latency_ms_mean"]),
        }
        if not all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in metrics_dict.values()):
            nonfinite_rows += 1

        ckpt_rel = (
            None
            if (is_heur or not str(r.get("checkpoint_path", "")).strip() or str(r.get("checkpoint_path")) == "nan")
            else _to_rel_posix(str(r["checkpoint_path"]), ROOT)
        )

        rec = {
            "episode_id": ep_id,
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "algorithm": algo,
            "controller_name": str(r["controller"]),
            "controller_group": str(r["controller_group"]),
            "track": str(r["track"]),
            "mode": str(r["mode"]),
            "scenario_id": str(r["scenario"]),
            "training_seed": train_seed,
            "evaluation_seed": int(r["eval_seed"]),
            "selected_checkpoint_episode": sel_ckpt_ep,
            "episode_length": int(step_count or sim_steps),
            "timestep_minutes": 15.0,
            "start_timestamp": None,
            "completion_status": completion_status,
            "enable_safety_shield": bool(r["enable_safety_shield"]),
            "adequacy_fallback": bool(r["adequacy_fallback"]),
            "checkpoint_path": ckpt_rel,
            "trace_path": packaged_rel_trace or rel_trace,
            "source_trace_path": rel_trace,
            "summary_path": SUMMARY_REL_PATH,
            **metrics_dict,
        }
        records.append(rec)

    # Sort deterministically by track, algorithm, training_seed, scenario_id, evaluation_seed
    records.sort(
        key=lambda x: (
            x["track"],
            x["algorithm"],
            -1 if x["training_seed"] is None else int(x["training_seed"]),
            x["scenario_id"],
            int(x["evaluation_seed"]),
        )
    )

    with open(index_dir / "episode_index.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "experiment_id": EXPERIMENT_ID,
                "total_episodes": len(records),
                "episodes": records,
            },
            f,
            indent=2,
            allow_nan=False,
        )
    pd.DataFrame(records).to_csv(index_dir / "episode_index.csv", index=False)

    # Lightweight per-track index manifests (no duplication of 260.9 MB traces unless packaged)
    track_counts: Dict[str, int] = {}
    for tr in sorted(VALID_TRACKS):
        tr_recs = [x for x in records if x["track"] == tr]
        track_counts[tr] = len(tr_recs)
        with open(episodes_dir / tr / "index.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "track": tr,
                    "episode_count": len(tr_recs),
                    "storage_mode": "packaged_copy" if package_traces else "indexed_reference",
                    "episodes": tr_recs,
                },
                f,
                indent=2,
                allow_nan=False,
            )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "storage_mode": "packaged_copy" if package_traces else "indexed_reference",
        "total_episodes": len(records),
        "tracks": track_counts,
        "algorithms": sorted({r["algorithm"] for r in records}),
        "scenarios": sorted({r["scenario_id"] for r in records}),
        "evaluation_seeds": sorted({int(r["evaluation_seed"]) for r in records}),
        "training_seeds": sorted({int(r["training_seed"]) for r in records if r["training_seed"] is not None}),
        "integrity": {
            "duplicate_ids": duplicate_ids,
            "missing_traces": missing_traces,
            "invalid_step_counts": invalid_step_counts,
            "reward_mismatches_gt_1e2": reward_mismatches,
            "max_trace_vs_summary_reward_diff": round(max_reward_diff, 6),
            "nonfinite_rows": nonfinite_rows,
            "total_source_trace_bytes": int(total_trace_bytes),
            "total_source_trace_mb": round(total_trace_bytes / 1e6, 2),
        },
        "source_summary_csv": SUMMARY_REL_PATH,
    }
    with open(manifest_dir / "library_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)

    integrity_md = f"""# Phase 4 Saved Episode Library — Integrity Verification Report

- **Generated (UTC):** `{manifest['generated_at_utc']}`
- **Schema Version:** `{SCHEMA_VERSION}`
- **Storage Mode:** `{manifest['storage_mode']}` (source footprint: `{manifest['integrity']['total_source_trace_mb']} MB`)
- **Total Indexed Episodes:** **{len(records)}** (`track_a_intrinsic`: {track_counts.get('track_a_intrinsic', 0)}, `track_b_shielded`: {track_counts.get('track_b_shielded', 0)}, `track_c_hybrid`: {track_counts.get('track_c_hybrid', 0)})
- **Duplicate Episode IDs:** **{duplicate_ids}**
- **Missing Source Trace Files:** **{missing_traces}**
- **Invalid Step Count (!= 96 steps):** **{invalid_step_counts}**
- **Trace-vs-Summary Reward Mismatches (> 0.01):** **{reward_mismatches}** (max observed diff: `{manifest['integrity']['max_trace_vs_summary_reward_diff']}`)
- **Non-Finite Metric Rows:** **{nonfinite_rows}**
- **Heuristic Null `training_seed` Verified:** `{all(r['training_seed'] is None for r in records if r['controller_group'] == 'Heuristic Baseline')}`
"""
    with open(reports_dir / "library_integrity_report.md", "w", encoding="utf-8") as f:
        f.write(integrity_md)

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build Phase 4 Saved Episode Library index & manifests")
    parser.add_argument("--package-traces", action="store_true", help="Copy trace files into episodes/<track>/ (default uses zero-duplication references)")
    args = parser.parse_args()
    manifest = build_episode_library(package_traces=args.package_traces, validate_all_traces=True)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
