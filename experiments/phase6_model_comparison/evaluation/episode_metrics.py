"""
episode_metrics.py
==================
Computes authoritative episode-level economic, reliability, renewable,
resource, network, safety, and inference-efficiency metrics for Phase 3.
Reuses rl_environment/economic_accounting.py cleanly without re-implementing costs.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd

from rl_environment.economic_accounting import compute_episode_economics, EconomicConfig


def summarize_episode_metrics(
    episode_id: str,
    track: str,
    mode: str,
    controller_name: str,
    algorithm: str,
    controller_group: str,
    train_seed: str,
    scenario: str,
    eval_seed: int,
    episode_history: List[Dict[str, Any]],
    step_telemetry: List[Dict[str, Any]],
    dt_hours: float,
    econ_cfg: EconomicConfig,
    enable_safety_shield: bool,
    adequacy_fallback: bool,
    episode_wall_s: float,
    checkpoint_path: str | None = None,
) -> Dict[str, Any]:
    """Compute unified single-episode summary row matching Stage 7 schema."""
    hist_df = pd.DataFrame(episode_history)
    econ = compute_episode_economics(hist_df, dt_hours=dt_hours, config=econ_cfg)
    steps_df = hist_df.iloc[1:] if (len(hist_df) > 1 and int(hist_df["step"].iloc[0]) == 0 and int(hist_df["step"].iloc[1]) == 0) else hist_df

    # Renewable energy metrics
    avail_re_mwh = float(
        (steps_df["solar_available_mw"].clip(lower=0.0) + steps_df["wind_available_mw"].clip(lower=0.0)).sum() * dt_hours
    )
    disp_re_mwh = float((steps_df["solar_mw"].clip(lower=0.0) + steps_df["wind_mw"].clip(lower=0.0)).sum() * dt_hours)
    re_util_pct = (disp_re_mwh / avail_re_mwh * 100.0) if avail_re_mwh > 1e-9 else 100.0

    # Battery charge vs discharge energy (sign: + charge, - discharge)
    bat_p = steps_df["battery_mw"].astype(float)
    bat_charge_mwh = float(bat_p.clip(lower=0.0).sum() * dt_hours)
    bat_discharge_mwh = float((-bat_p.clip(upper=0.0)).sum() * dt_hours)

    # Served total MWh
    served_total_mwh = float(steps_df["served_total_mw"].clip(lower=0.0).sum() * dt_hours)

    # Network & power quality metrics
    vmin = steps_df["min_bus_voltage_pu"].astype(float)
    vmax = steps_df["max_bus_voltage_pu"].astype(float)
    voltage_violation_steps = int(((vmin < 0.95) | (vmax > 1.05)).sum())
    pf_conv_pct = float(steps_df["converged"].astype(bool).mean() * 100.0)
    pf_failures = int((~steps_df["converged"].astype(bool)).sum())
    unserved_crit_steps = int((steps_df["shed_critical_mw"].astype(float) > 1e-6).sum())
    bal_err = steps_df["balance_error_mw"].abs().astype(float)

    # Step-level shield, adequacy & latency telemetry
    total_reward = float(sum(s["reward"] for s in step_telemetry))
    total_proposed = sum(s["num_agents"] for s in step_telemetry)
    total_feasible = sum(s["feasible_agents"] for s in step_telemetry)
    unshielded_feasible_pct = (total_feasible / total_proposed * 100.0) if total_proposed > 0 else 100.0

    boundary_clip_steps = int(sum(1 for s in step_telemetry if s["boundary_clipping"]))
    emergency_override_steps = int(sum(1 for s in step_telemetry if s["emergency_override"]))
    shield_intervention_steps = int(sum(1 for s in step_telemetry if s["shield_intervention"]))
    inaction_detected_steps = int(sum(1 for s in step_telemetry if s["inaction_detected"]))
    adequacy_intervention_steps = int(sum(1 for s in step_telemetry if s["adequacy_intervention"]))
    fallback_intervention_steps = int(sum(1 for s in step_telemetry if s["fallback_intervention"]))
    lat_mean_ms = float(np.mean([s["inference_latency_ms"] for s in step_telemetry])) if step_telemetry else 0.0

    avoidable_crit_ens = float(steps_df["avoidable_critical_shed_mw"].sum() * dt_hours) if "avoidable_critical_shed_mw" in steps_df else 0.0
    unavoidable_crit_ens = float(steps_df["unavoidable_critical_shed_mw"].sum() * dt_hours) if "unavoidable_critical_shed_mw" in steps_df else 0.0

    return {
        "episode_id": episode_id,
        "track": track,
        "mode": mode,
        "controller": controller_name,
        "algorithm": algorithm,
        "controller_group": controller_group,
        "train_seed": str(train_seed),
        "scenario": scenario,
        "eval_seed": int(eval_seed),
        "enable_safety_shield": bool(enable_safety_shield),
        "adequacy_fallback": bool(adequacy_fallback),
        "checkpoint_path": checkpoint_path or "",
        "reward": round(total_reward, 4),
        # Renewable utilization
        "available_renewable_mwh": round(avail_re_mwh, 6),
        "dispatched_renewable_mwh": round(disp_re_mwh, 6),
        "renewable_utilization_pct": round(re_util_pct, 4),
        # Battery & load extras
        "battery_charge_mwh": round(bat_charge_mwh, 6),
        "battery_discharge_mwh": round(bat_discharge_mwh, 6),
        "served_total_mwh": round(served_total_mwh, 6),
        # Network & reliability checks
        "voltage_violation_steps": voltage_violation_steps,
        "power_flow_failures": pf_failures,
        "power_flow_converged_pct": round(pf_conv_pct, 2),
        "unserved_critical_demand_steps": unserved_crit_steps,
        "mean_balance_error_mw": round(float(bal_err.mean()), 6),
        "max_balance_error_mw": round(float(bal_err.max()), 6),
        # Safety & intervention telemetry
        "unshielded_feasible_pct": round(unshielded_feasible_pct, 2),
        "boundary_clipping_count": boundary_clip_steps,
        "emergency_override_count": emergency_override_steps,
        "shield_intervention_steps": shield_intervention_steps,
        "inaction_detected_steps": inaction_detected_steps,
        "supply_adequacy_intervention_count": adequacy_intervention_steps,
        "fallback_intervention_steps": fallback_intervention_steps,
        "avoidable_critical_ens_mwh": round(avoidable_crit_ens, 6),
        "unavoidable_critical_ens_mwh": round(unavoidable_crit_ens, 6),
        # Timing
        "inference_latency_ms_mean": round(lat_mean_ms, 4),
        "episode_wall_s": round(float(episode_wall_s), 3),
        # Unified economic accounting (all MWh / USD / L / h fields)
        **econ,
    }
