"""
trace_recorder.py
=================
Structured 96-timestep episode trace recorder for Phase 3 & Phase 4 replay.
Saves real simulated observations, proposed/executed actions, resource states,
voltages, reward breakdowns, step costs, shield/fallback/scenario events, and
power-flow convergence without reconstructing missing steps.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from rl_environment.economic_accounting import EconomicConfig


def _to_json_safe(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return [_to_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        if not np.isfinite(val):
            raise ValueError(f"Non-finite float encountered in trace: {val}")
        return round(val, 6)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    return obj


def build_step_trace_record(
    step_idx: int,
    obs_dict: Dict[str, np.ndarray],
    proposed_actions: Dict[str, Any],
    executed_actions: Dict[str, Any],
    state_dict: Dict[str, Any],
    reward: float,
    reward_breakdown: Dict[str, float],
    shield_diag: Dict[str, Any],
    adequacy_diag: Dict[str, Any],
    scenario_events: List[str],
    dt_hours: float,
    econ_cfg: EconomicConfig,
    adequacy_fallback: bool,
) -> Dict[str, Any]:
    """Build a single timestep trace record conforming to Stage 8 schema."""
    grid_p = float(state_dict.get("grid_import_mw", 0.0))
    step_fuel = float(state_dict.get("step_fuel_l", 0.0))
    op_cost_step = (
        max(0.0, grid_p) * dt_hours * econ_cfg.grid_import_usd_per_mwh
        - abs(min(0.0, grid_p)) * dt_hours * econ_cfg.grid_export_usd_per_mwh
        + step_fuel * econ_cfg.fuel_usd_per_liter
    )
    ens_cost_step = (
        float(state_dict.get("shed_critical_mw", 0.0)) * dt_hours * econ_cfg.voll_critical_usd_per_mwh
        + float(state_dict.get("shed_flexible_mw", 0.0)) * dt_hours * econ_cfg.voll_flexible_usd_per_mwh
        + float(state_dict.get("shed_other_mw", 0.0)) * dt_hours * econ_cfg.voll_other_usd_per_mwh
    )
    bat_deg_step = abs(float(state_dict.get("battery_mw", 0.0))) * dt_hours * econ_cfg.battery_degradation_usd_per_mwh
    soc_cost_step = op_cost_step + ens_cost_step + bat_deg_step

    re_avail = float(state_dict.get("solar_available_mw", 0.0)) + float(state_dict.get("wind_available_mw", 0.0))
    re_actual = float(state_dict.get("solar_mw", 0.0)) + float(state_dict.get("wind_mw", 0.0))
    re_curt_mw = max(0.0, re_avail - re_actual)

    shield_events = {
        "any_intervention": bool(shield_diag.get("any_intervention", False)),
        "any_boundary": bool(shield_diag.get("any_boundary", False)),
        "any_emergency": bool(shield_diag.get("any_emergency", False)),
        "violations": list(shield_diag.get("violations", [])),
        "reasons": list(shield_diag.get("reasons", [])),
        "interventions_by_agent": dict(shield_diag.get("interventions", {})),
    }
    fallback_events = {
        "adequacy_fallback_enabled": bool(adequacy_fallback),
        "inaction_detected": bool(adequacy_diag.get("inaction_detected", False)),
        "adequacy_intervention": bool(adequacy_diag.get("adequacy_intervention", False)),
        "avoidable_shortfall_mw": float(adequacy_diag.get("avoidable_shortfall_mw", 0.0)),
        "reasons": list(adequacy_diag.get("intervention_reasons", [])),
    }

    return _to_json_safe({
        "step_idx": int(step_idx),
        "hour": float(state_dict.get("hour", step_idx * dt_hours)),
        "timestamp_label": f"{int(float(state_dict.get('hour', 0.0))):02d}:{int((float(state_dict.get('hour', 0.0)) % 1.0) * 60):02d}",
        "observations": {ag: np.asarray(arr, dtype=np.float32).tolist() for ag, arr in obs_dict.items()},
        "proposed_actions": proposed_actions,
        "executed_actions": executed_actions,
        "solar_available_mw": float(state_dict.get("solar_available_mw", 0.0)),
        "solar_mw": float(state_dict.get("solar_mw", 0.0)),
        "wind_available_mw": float(state_dict.get("wind_available_mw", 0.0)),
        "wind_mw": float(state_dict.get("wind_mw", 0.0)),
        "renewable_curtailment_mw": re_curt_mw,
        "battery_soc_pct": float(state_dict.get("battery_soc", 0.0)),
        "battery_mw": float(state_dict.get("battery_mw", 0.0)),
        "generator_state": str(state_dict.get("generator_state", "OFF")),
        "generator_mw": float(state_dict.get("generator_mw", 0.0)),
        "step_fuel_l": step_fuel,
        "grid_connected": bool(state_dict.get("grid_connected", True)),
        "grid_available": bool(state_dict.get("grid_available", True)),
        "grid_import_mw": grid_p,
        "nominal_critical_mw": float(state_dict.get("nominal_critical_mw", 0.0)),
        "nominal_flexible_mw": float(state_dict.get("nominal_flexible_mw", 0.0)),
        "nominal_other_mw": float(state_dict.get("nominal_other_mw", 0.0)),
        "served_critical_mw": float(state_dict.get("served_critical_mw", 0.0)),
        "served_flexible_mw": float(state_dict.get("served_flexible_mw", 0.0)),
        "served_other_mw": float(state_dict.get("served_other_mw", 0.0)),
        "shed_critical_mw": float(state_dict.get("shed_critical_mw", 0.0)),
        "shed_flexible_mw": float(state_dict.get("shed_flexible_mw", 0.0)),
        "shed_other_mw": float(state_dict.get("shed_other_mw", 0.0)),
        "voltages_pu": {
            "pcc": float(state_dict.get("pcc_voltage_pu", 1.0)),
            "min_bus": float(state_dict.get("min_bus_voltage_pu", 1.0)),
            "max_bus": float(state_dict.get("max_bus_voltage_pu", 1.0)),
        },
        "reward": float(reward),
        "reward_breakdown": dict(reward_breakdown),
        "step_operating_cost_usd": op_cost_step,
        "step_social_cost_usd": soc_cost_step,
        "shield_events": shield_events,
        "fallback_events": fallback_events,
        "scenario_events": list(scenario_events),
        "converged": bool(state_dict.get("converged", True)),
    })


def save_episode_trace(
    trace_dir: Path,
    episode_id: str,
    metadata: Dict[str, Any],
    summary_metrics: Dict[str, Any],
    steps: List[Dict[str, Any]],
    expected_steps: int = 96,
) -> Path:
    """Validate completeness (96 steps) and persist episode trace JSON cleanly."""
    if len(steps) != expected_steps:
        raise ValueError(f"Episode {episode_id} recorded {len(steps)} steps, expected {expected_steps}")

    trace_dir.mkdir(parents=True, exist_ok=True)
    out_path = trace_dir / f"{episode_id}.json"
    payload = _to_json_safe({
        "episode_id": episode_id,
        "metadata": metadata,
        "summary": summary_metrics,
        "steps": steps,
    })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, allow_nan=False)
    return out_path
