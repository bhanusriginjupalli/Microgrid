# GRIDGUARD AI
# Phase 2: Environment Utilities
# ============================================================
# Helper utilities for seeding, action/observation packaging,
# episode telemetry logging, and format conversions.
# ============================================================

import os
import random
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42):
    """Set global random seeds for deterministic execution across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_episode_log(history: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert step telemetry dictionaries into a structured DataFrame."""
    return pd.DataFrame(history)


def compute_episode_summary(df: pd.DataFrame, dt_hours: float = 0.25) -> Dict[str, Any]:
    """Compute aggregate statistical summary of an RL episode."""
    n_steps = len(df)
    total_reward = float(df["reward"].sum()) if "reward" in df else 0.0

    return {
        "steps": n_steps,
        "total_reward": total_reward,
        "mean_step_reward": total_reward / max(1, n_steps),
        "total_served_mwh": float(df.get("served_total_mw", pd.Series(0.0)).sum() * dt_hours),
        "total_shed_mwh": float(df.get("shed_total_mw", pd.Series(0.0)).sum() * dt_hours),
        "critical_ens_mwh": float(df.get("shed_critical_mw", pd.Series(0.0)).sum() * dt_hours),
        "min_soc": float(df.get("battery_soc", pd.Series(80.0)).min()),
        "final_soc": float(df.get("battery_soc", pd.Series(80.0)).iloc[-1]),
        "fuel_consumed_l": float(df.get("step_fuel_l", pd.Series(0.0)).sum()),
        "grid_import_mwh": float(df.get("grid_import_mw", pd.Series(0.0)).clip(lower=0).sum() * dt_hours),
        "grid_export_mwh": float(df.get("grid_import_mw", pd.Series(0.0)).clip(upper=0).abs().sum() * dt_hours),
        "converged_pct": float(df.get("converged", pd.Series(True)).mean() * 100.0),
        "interventions_count": int(df.get("any_intervention", pd.Series(False)).sum()),
    }
