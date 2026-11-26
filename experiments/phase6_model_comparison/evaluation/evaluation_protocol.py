"""
evaluation_protocol.py
======================
Defines exact Phase 3 experimental protocol constants, tracks (A: Intrinsic,
B: Shielded, C: Hybrid +Fallback), matched held-out scenarios (H1-H6),
matched evaluation seeds ([42, 101, 202, 303, 404]), and deterministic seeding.
"""

from typing import Dict, List, Any
import random
import numpy as np
import torch

TRAIN_SEEDS: List[int] = [42, 101, 202]
EVAL_SEEDS: List[int] = [42, 101, 202, 303, 404]
HELDOUT_SCENARIOS: List[str] = [f"scenario_h{i}" for i in range(1, 7)]

SCENARIO_LABELS: Dict[str, str] = {
    "scenario_h1": "H1: Network Topology Fault",
    "scenario_h2": "H2: Renewable Drought",
    "scenario_h3": "H3: Demand Surge",
    "scenario_h4": "H4: Sensor Noise & Dropout",
    "scenario_h5": "H5: Stale Forecasts",
    "scenario_h6": "H6: Compound Multi-Fault",
}

LEARNED_ALGORITHMS: List[str] = ["IPPO", "MAPPO_MATCHED", "HAPPO"]
HEURISTIC_BASELINES: List[str] = ["Greedy EMS", "Rule-Based EMS", "Lookahead Heuristic EMS"]

# Track A evaluates intrinsic RL (shield OFF, fallback OFF) + native heuristics (360 eps total)
# Track B evaluates shielded RL (shield ON, fallback OFF) (270 RL eps; compared vs native heuristics)
# Track C evaluates hybrid RL (shield ON, fallback ON) (270 RL eps; compared vs native heuristics)
TRACK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "track_a_intrinsic": {
        "label": "Track A: Intrinsic",
        "mode": "Intrinsic",
        "enable_safety_shield": False,
        "enable_adequacy_check": True,
        "adequacy_fallback": False,
        "include_heuristics": True,
    },
    "track_b_shielded": {
        "label": "Track B: Shielded",
        "mode": "Shielded",
        "enable_safety_shield": True,
        "enable_adequacy_check": True,
        "adequacy_fallback": False,
        "include_heuristics": False,
    },
    "track_c_hybrid": {
        "label": "Track C: Hybrid (+Fallback)",
        "mode": "Hybrid (+Fallback)",
        "enable_safety_shield": True,
        "enable_adequacy_check": True,
        "adequacy_fallback": True,
        "include_heuristics": False,
    },
}


def make_episode_id(
    track: str,
    algorithm_slug: str,
    train_seed: str,
    scenario: str,
    eval_seed: int,
) -> str:
    """Create stable unique ID per evaluation episode."""
    return f"{track}__{algorithm_slug}__seed{train_seed}__{scenario}__eval{int(eval_seed)}"


def set_episode_seeds(eval_seed: int) -> None:
    """Reset Python, NumPy, and PyTorch RNG states deterministically for an episode."""
    random.seed(int(eval_seed))
    np.random.seed(int(eval_seed))
    torch.manual_seed(int(eval_seed))
