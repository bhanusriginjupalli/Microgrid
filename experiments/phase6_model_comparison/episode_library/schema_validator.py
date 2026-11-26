"""
schema_validator.py
===================
Canonical Episode Schema validator (v1.0.0) and normalizer for Phase 4.
Verifies structural conformance, finite numeric metrics, track/mode isolation,
null heuristic training-seed rules, and exact 96-timestep completeness.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List

SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "episode_schema.json"
SCHEMA_VERSION = "1.0.0"

VALID_TRACKS = {"track_a_intrinsic", "track_b_shielded", "track_c_hybrid"}
VALID_MODES = {"Intrinsic", "Shielded", "Hybrid (+Fallback)", "Heuristic"}
VALID_ALGORITHMS = {
    "IPPO",
    "MAPPO_MATCHED",
    "HAPPO",
    "Greedy EMS",
    "Rule-Based EMS",
    "Lookahead Heuristic EMS",
}


def load_episode_schema() -> Dict[str, Any]:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _check_finite_recursive(obj: Any, path: str = "root") -> None:
    if obj is None or isinstance(obj, (bool, str, int)):
        return
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"Non-finite float at {path}: {obj}")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _check_finite_recursive(v, f"{path}.{k}")
        return
    if isinstance(obj, list):
        for idx, v in enumerate(obj):
            _check_finite_recursive(v, f"{path}[{idx}]")
        return


def validate_canonical_episode(ep: Dict[str, Any]) -> List[str]:
    """Validate a canonical episode payload against schema v1.0.0. Returns errors list (empty if valid)."""
    errors: List[str] = []
    schema = load_episode_schema()

    for key in schema["required"]:
        if key not in ep:
            errors.append(f"Missing top-level key: {key}")
    if errors:
        return errors

    if ep.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"Invalid schema_version: {ep.get('schema_version')}")

    meta = ep["metadata"]
    for key in schema["properties"]["metadata"]["required"]:
        if key not in meta:
            errors.append(f"Missing metadata key: {key}")

    if meta.get("algorithm") not in VALID_ALGORITHMS:
        errors.append(f"Invalid algorithm: {meta.get('algorithm')}")
    if meta.get("track") not in VALID_TRACKS:
        errors.append(f"Invalid track: {meta.get('track')}")
    if meta.get("mode") not in VALID_MODES:
        errors.append(f"Invalid mode: {meta.get('mode')}")

    # Strict track vs mode alignment check
    track = meta.get("track")
    mode = meta.get("mode")
    group = meta.get("controller_group")
    if group == "Heuristic Baseline":
        if meta.get("training_seed") is not None:
            errors.append(f"Heuristic baseline must have null training_seed, got {meta.get('training_seed')}")
        if meta.get("selected_checkpoint_episode") is not None:
            errors.append("Heuristic baseline must have null selected_checkpoint_episode")
        if mode != "Heuristic":
            errors.append(f"Heuristic baseline mode must be 'Heuristic', got {mode}")
    elif group == "Learned MARL":
        if not isinstance(meta.get("training_seed"), int):
            errors.append(f"Learned MARL must have int training_seed, got {meta.get('training_seed')}")
        expected_mode = {
            "track_a_intrinsic": "Intrinsic",
            "track_b_shielded": "Shielded",
            "track_c_hybrid": "Hybrid (+Fallback)",
        }.get(track)
        if mode != expected_mode:
            errors.append(f"Track/mode mismatch: {track} -> {mode} (expected {expected_mode})")

    metrics = ep["metrics"]
    for key in schema["properties"]["metrics"]["required"]:
        if key not in metrics:
            errors.append(f"Missing metric key: {key}")

    steps = ep["steps"]
    if not isinstance(steps, list) or len(steps) != 96:
        errors.append(f"Expected 96 steps, got {len(steps) if isinstance(steps, list) else type(steps)}")
    else:
        req_step_keys = set(schema["properties"]["steps"]["items"]["required"])
        for i, st in enumerate(steps):
            missing = req_step_keys - set(st.keys())
            if missing:
                errors.append(f"Step {i} missing keys: {sorted(missing)}")
                break
            if int(st["step_idx"]) != i:
                errors.append(f"Step {i} has non-sequential step_idx={st['step_idx']}")
                break

    try:
        _check_finite_recursive(ep)
    except ValueError as exc:
        errors.append(str(exc))

    return errors
