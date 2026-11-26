"""
h2_comparison.py
================
Builds the Phase 4.5 H2 (18-Hour Renewable Drought) Saved-Episode Comparison
Dataset for IPPO, MAPPO_MATCHED, HAPPO, and Rule-Based EMS across Track A,
Track B (RL only), and Track C (RL only). Preserves all H2 episodes, explicitly
marks Rule-Based EMS Track B/C as unavailable, and computes reproducible
representative selections (best, median, worst social cost, most reliable,
highest critical ENS) with explicit criteria.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase6_model_comparison.episode_library.library import EpisodeLibrary

H2_ALGORITHMS = ["IPPO", "MAPPO_MATCHED", "HAPPO", "Rule-Based EMS"]
TRACKS = ["track_a_intrinsic", "track_b_shielded", "track_c_hybrid"]


def select_representative_episodes(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Select deterministic representative episode IDs across a controller-track pool."""
    if not records:
        return {"available": False, "episodes_n": 0}

    # Tie-break deterministically by episode_id
    by_cost = sorted(records, key=lambda r: (float(r["true_social_cost_usd"]), r["episode_id"]))
    by_rel = sorted(records, key=lambda r: (-float(r["critical_reliability_pct"]), float(r["true_social_cost_usd"]), r["episode_id"]))
    by_ens = sorted(records, key=lambda r: (-float(r["critical_ens_mwh"]), r["episode_id"]))

    best_ep = by_cost[0]
    med_ep = by_cost[len(by_cost) // 2]
    worst_ep = by_cost[-1]
    most_rel = by_rel[0]
    highest_ens = by_ens[0]

    def _pack(ep: Dict[str, Any], criterion: str) -> Dict[str, Any]:
        return {
            "criterion": criterion,
            "episode_id": ep["episode_id"],
            "training_seed": ep["training_seed"],
            "evaluation_seed": ep["evaluation_seed"],
            "true_social_cost_usd": ep["true_social_cost_usd"],
            "critical_ens_mwh": ep["critical_ens_mwh"],
            "critical_reliability_pct": ep["critical_reliability_pct"],
            "fallback_intervention_steps": ep["fallback_intervention_steps"],
        }

    return {
        "available": True,
        "episodes_n": len(records),
        "note": "Best episode illustrates optimistic single-seed/realization upper bound; inspect median/worst/all episodes for typical performance.",
        "best_social_cost": _pack(best_ep, "Minimum true_social_cost_usd"),
        "median_social_cost": _pack(med_ep, "Median rank (index n//2) sorted by true_social_cost_usd"),
        "worst_social_cost": _pack(worst_ep, "Maximum true_social_cost_usd"),
        "most_reliable": _pack(most_rel, "Maximum critical_reliability_pct (tie-break min true_social_cost_usd)"),
        "highest_critical_ens": _pack(highest_ens, "Maximum critical_ens_mwh"),
    }


def build_h2_comparison_dataset(library: EpisodeLibrary) -> Dict[str, Any]:
    """Generate structured H2 comparison dataset and CSV summary."""
    all_h2 = [
        r for r in library.get_scenario_episodes("scenario_h2")
        if r["algorithm"] in H2_ALGORITHMS
    ]

    matrix: Dict[str, Dict[str, Any]] = {}
    summary_rows: List[Dict[str, Any]] = []

    for algo in H2_ALGORITHMS:
        matrix[algo] = {}
        for tr in TRACKS:
            sub = [r for r in all_h2 if r["algorithm"] == algo and r["track"] == tr]
            if not sub:
                matrix[algo][tr] = {
                    "available": False,
                    "reason": "Rule-Based EMS was evaluated only in its native baseline mode (track_a_intrinsic); Track B/C entries do not exist and are not fabricated.",
                    "episodes": [],
                    "representatives": {"available": False, "episodes_n": 0},
                }
                summary_rows.append({
                    "algorithm": algo,
                    "track": tr,
                    "available": False,
                    "episodes_n": 0,
                    "true_social_cost_mean": None,
                    "critical_ens_mean": None,
                    "critical_reliability_mean": None,
                    "best_episode_id": None,
                    "median_episode_id": None,
                    "worst_episode_id": None,
                })
                continue

            reps = select_representative_episodes(sub)
            n = len(sub)
            matrix[algo][tr] = {
                "available": True,
                "mode": sub[0]["mode"],
                "episodes_n": n,
                "mean_true_social_cost_usd": round(sum(float(x["true_social_cost_usd"]) for x in sub) / n, 4),
                "mean_operating_cost_usd": round(sum(float(x["operating_cost_usd"]) for x in sub) / n, 4),
                "mean_critical_ens_mwh": round(sum(float(x["critical_ens_mwh"]) for x in sub) / n, 6),
                "mean_critical_reliability_pct": round(sum(float(x["critical_reliability_pct"]) for x in sub) / n, 4),
                "mean_fallback_interventions": round(sum(float(x["fallback_intervention_steps"]) for x in sub) / n, 3),
                "representatives": reps,
                "episodes": sub,
            }
            summary_rows.append({
                "algorithm": algo,
                "track": tr,
                "available": True,
                "episodes_n": n,
                "true_social_cost_mean": matrix[algo][tr]["mean_true_social_cost_usd"],
                "critical_ens_mean": matrix[algo][tr]["mean_critical_ens_mwh"],
                "critical_reliability_mean": matrix[algo][tr]["mean_critical_reliability_pct"],
                "best_episode_id": reps["best_social_cost"]["episode_id"],
                "median_episode_id": reps["median_social_cost"]["episode_id"],
                "worst_episode_id": reps["worst_social_cost"]["episode_id"],
            })

    # Matched-seed pairings on H2 (preserving each training seed cleanly per eval_seed)
    matched_eval_seeds = sorted({int(r["evaluation_seed"]) for r in all_h2})
    matched_pairs: List[Dict[str, Any]] = []
    for es in matched_eval_seeds:
        by_slot: Dict[str, Any] = {}
        for tr in TRACKS:
            for algo in H2_ALGORITHMS:
                hits = [r["episode_id"] for r in all_h2 if r["track"] == tr and r["algorithm"] == algo and int(r["evaluation_seed"]) == es]
                by_slot[f"{tr}::{algo}"] = hits if hits else None
        matched_pairs.append({"evaluation_seed": es, "episodes_by_track_algorithm": by_slot})

    dataset = {
        "scenario_id": "scenario_h2",
        "scenario_title": "Scenario H2: 18-Hour Renewable Drought (04:00-22:00 Wind Lull + 11:00-14:00 Grid Outage)",
        "algorithms_included": H2_ALGORITHMS,
        "tracks_included": TRACKS,
        "total_h2_episodes": len(all_h2),
        "matrix": matrix,
        "matched_seed_pairs": matched_pairs,
    }

    manifest_dir = library.library_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_dir / "h2_comparison_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, allow_nan=False)
    pd.DataFrame(summary_rows).to_csv(manifest_dir / "h2_comparison_summary.csv", index=False)
    return dataset


def main():
    lib = EpisodeLibrary()
    ds = build_h2_comparison_dataset(lib)
    print(f"[H2] Built dataset with {ds['total_h2_episodes']} H2 episodes.")


if __name__ == "__main__":
    main()
