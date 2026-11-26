"""
library.py
==========
Read-only retrieval, filtering, pagination, and summary utilities for the
Phase 4 Saved Episode Library. Keeps 260.9 MB of trace JSONs lazy-loaded
(metadata/index only for listings and summaries; single trace load on demand).
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from experiments.phase6_model_comparison.episode_library.schema_validator import SCHEMA_VERSION

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_LIB_ROOT = ROOT / "experiments" / "phase6_model_comparison" / "episode_library"


def _sanitize_json(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    return obj


class EpisodeLibrary:
    """Lazy, read-only accessor over the Phase 4 Saved Episode Library."""

    def __init__(self, library_root: Optional[Path] = None, project_root: Optional[Path] = None):
        self.project_root = Path(project_root) if project_root else ROOT
        self.library_root = Path(library_root) if library_root else (
            self.project_root / "experiments" / "phase6_model_comparison" / "episode_library"
        )
        self.index_json_path = self.library_root / "index" / "episode_index.json"
        self.manifest_path = self.library_root / "manifests" / "library_manifest.json"
        self.h2_dataset_path = self.library_root / "manifests" / "h2_comparison_dataset.json"
        self._cached_records: Optional[List[Dict[str, Any]]] = None

    def is_available(self) -> bool:
        return self.index_json_path.exists()

    def load_index(self, force_reload: bool = False) -> List[Dict[str, Any]]:
        if self._cached_records is not None and not force_reload:
            return self._cached_records
        if not self.index_json_path.exists():
            raise FileNotFoundError(f"Episode index not found at {self.index_json_path}")
        with open(self.index_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self._cached_records = list(payload.get("episodes", []))
        return self._cached_records

    def list_episodes(
        self,
        algorithm: Optional[str] = None,
        track: Optional[str] = None,
        mode: Optional[str] = None,
        scenario_id: Optional[str] = None,
        training_seed: Optional[int] = None,
        evaluation_seed: Optional[int] = None,
        social_cost_min: Optional[float] = None,
        social_cost_max: Optional[float] = None,
        critical_ens_min: Optional[float] = None,
        critical_ens_max: Optional[float] = None,
        critical_reliability_min: Optional[float] = None,
        critical_reliability_max: Optional[float] = None,
        completion_status: Optional[str] = None,
        search: Optional[str] = None,
        sort_by: str = "episode_id",
        descending: bool = False,
        offset: int = 0,
        limit: int = 100,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Filter, deterministically sort, and paginate episode index records."""
        records = self.load_index()
        filtered: List[Dict[str, Any]] = []
        search_q = search.strip().lower() if search and search.strip() else None

        for r in records:
            if search_q:
                haystack = f"{r['episode_id']} {r['controller_name']} {r['algorithm']} {r['scenario_id']}".lower()
                if search_q not in haystack:
                    continue
            if algorithm and r["algorithm"] != algorithm:
                continue
            if track and r["track"] != track:
                continue
            if mode and r["mode"] != mode:
                continue
            if scenario_id and r["scenario_id"] != scenario_id:
                continue
            if training_seed is not None and r["training_seed"] != int(training_seed):
                continue
            if evaluation_seed is not None and int(r["evaluation_seed"]) != int(evaluation_seed):
                continue
            if completion_status and r["completion_status"] != completion_status:
                continue
            if social_cost_min is not None and float(r["true_social_cost_usd"]) < float(social_cost_min):
                continue
            if social_cost_max is not None and float(r["true_social_cost_usd"]) > float(social_cost_max):
                continue
            if critical_ens_min is not None and float(r["critical_ens_mwh"]) < float(critical_ens_min):
                continue
            if critical_ens_max is not None and float(r["critical_ens_mwh"]) > float(critical_ens_max):
                continue
            if critical_reliability_min is not None and float(r["critical_reliability_pct"]) < float(critical_reliability_min):
                continue
            if critical_reliability_max is not None and float(r["critical_reliability_pct"]) > float(critical_reliability_max):
                continue
            filtered.append(r)

        def _sort_key(item: Dict[str, Any]):
            val = item.get(sort_by)
            if val is None:
                return (1, "")
            return (0, val)

        filtered.sort(key=_sort_key, reverse=descending)
        total = len(filtered)
        page = filtered[offset : offset + limit] if limit > 0 else filtered[offset:]
        return page, total

    def get_episode_record(self, episode_id: str) -> Optional[Dict[str, Any]]:
        for r in self.load_index():
            if r["episode_id"] == episode_id:
                return r
        return None

    def get_episode_summary(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """Return canonical metadata + metrics without loading the full 96-step trace."""
        rec = self.get_episode_record(episode_id)
        if rec is None:
            return None

        meta_keys = {
            "episode_id",
            "schema_version",
            "experiment_id",
            "algorithm",
            "controller_name",
            "controller_group",
            "track",
            "mode",
            "scenario_id",
            "training_seed",
            "evaluation_seed",
            "selected_checkpoint_episode",
            "episode_length",
            "timestep_minutes",
            "start_timestamp",
            "completion_status",
            "enable_safety_shield",
            "adequacy_fallback",
            "checkpoint_path",
            "trace_path",
            "source_trace_path",
            "summary_path",
        }
        metadata = {k: rec.get(k) for k in meta_keys if k in rec}
        metrics = {k: v for k, v in rec.items() if k not in meta_keys}
        return _sanitize_json({
            "episode_id": episode_id,
            "schema_version": rec.get("schema_version", SCHEMA_VERSION),
            "metadata": metadata,
            "metrics": metrics,
        })

    def get_episode_trace(self, episode_id: str) -> Dict[str, Any]:
        """Load and return canonical full 96-step episode trace on demand."""
        summary_pkg = self.get_episode_summary(episode_id)
        if summary_pkg is None:
            raise KeyError(f"Unknown episode_id: {episode_id}")

        rel_path = summary_pkg["metadata"]["trace_path"]
        trace_path = self.project_root / rel_path
        if not trace_path.exists():
            src_rel = summary_pkg["metadata"].get("source_trace_path")
            if src_rel and (self.project_root / src_rel).exists():
                trace_path = self.project_root / src_rel
            else:
                raise FileNotFoundError(f"Trace file missing for {episode_id}: {trace_path}")

        try:
            with open(trace_path, "r", encoding="utf-8") as f:
                raw_trace = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON trace for {episode_id}: {exc}") from exc

        steps = raw_trace.get("steps", [])
        return _sanitize_json({
            "episode_id": episode_id,
            "schema_version": summary_pkg["schema_version"],
            "metadata": summary_pkg["metadata"],
            "metrics": summary_pkg["metrics"],
            "steps": steps,
        })

    def get_scenario_episodes(
        self,
        scenario_id: str,
        track: Optional[str] = None,
        algorithm: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        items, _ = self.list_episodes(
            scenario_id=scenario_id,
            track=track,
            algorithm=algorithm,
            limit=0,
        )
        return items

    def get_library_summary(self) -> Dict[str, Any]:
        """Compute summary breakdowns directly from the lightweight episode index (never loads full traces)."""
        records = self.load_index()
        by_track_algo: Dict[tuple, List[Dict[str, Any]]] = {}
        for r in records:
            by_track_algo.setdefault((r["track"], r["mode"], r["algorithm"]), []).append(r)

        groups: List[Dict[str, Any]] = []
        for (track, mode, algo), grp in sorted(by_track_algo.items()):
            n = len(grp)
            groups.append({
                "track": track,
                "mode": mode,
                "algorithm": algo,
                "episodes_n": n,
                "true_social_cost_mean": round(sum(float(x["true_social_cost_usd"]) for x in grp) / n, 4),
                "operating_cost_mean": round(sum(float(x["operating_cost_usd"]) for x in grp) / n, 4),
                "critical_ens_mean": round(sum(float(x["critical_ens_mwh"]) for x in grp) / n, 6),
                "critical_reliability_mean": round(sum(float(x["critical_reliability_pct"]) for x in grp) / n, 4),
                "fallback_interventions_mean": round(sum(float(x["fallback_intervention_steps"]) for x in grp) / n, 3),
            })

        manifest: Dict[str, Any] = {}
        if self.manifest_path.exists():
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

        return _sanitize_json({
            "schema_version": SCHEMA_VERSION,
            "total_episodes": len(records),
            "tracks": sorted({r["track"] for r in records}),
            "algorithms": sorted({r["algorithm"] for r in records}),
            "scenarios": sorted({r["scenario_id"] for r in records}),
            "evaluation_seeds": sorted({int(r["evaluation_seed"]) for r in records}),
            "group_summaries": groups,
            "manifest_integrity": manifest.get("integrity", {}),
        })
