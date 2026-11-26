"""
experiments/phase4_4_4/remediation_v3.py
========================================
GRIDGUARD AI — Phase 4.4.4 Condition-Aware Generator & Battery Remediation

Implements:
  1. RemediationConfig: toggles Generator Remediation A and Battery Remediation B
     for clean 2x2 factorial ablation (Baseline, Gen-only, Bat-only, Combined).
  2. classify_generator_proposal(): separates raw generator proposals into
     necessary, unnecessary, physically-blocked, fallback-corrected, and
     missed-startup telemetry + condition-aware penalty (strictly 0 when grid
     is unavailable, inadequate, storm-warned, or holding min-up-time).
  3. compute_battery_remediation_signals(): aligns battery step incentive with
     exact terminal-SOC energy valuation ($120/MWh) and adds condition-aware
     deficit discharge credit assignment & reserve/healthy-cycling discipline.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import yaml

try:
    from config import BATTERY, GENERATOR, GRID_CONNECTION
except ImportError:
    from phase1_digital_twin.config import BATTERY, GENERATOR, GRID_CONNECTION

from rl_environment.action_space import parse_agent_action


@dataclass
class RemediationConfig:
    experiment_name: str = "baseline"
    remediation_gen: bool = False
    remediation_bat: bool = False
    init_gen_bias: float = -0.60
    use_cost_critics: bool = False
    gen_unneeded_base_penalty: float = 0.0
    gen_unneeded_slope_penalty: float = 0.0
    gen_missed_startup_penalty: float = 0.0
    bat_deficit_under_dis_penalty: float = 0.0
    bat_healthy_cycle_penalty: float = 0.0
    bat_reserve_breach_penalty: float = 0.0
    bat_energy_val_weight: float = 0.0

    @classmethod
    def from_yaml(cls, cfg_path: str) -> "RemediationConfig":
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        rem = data.get("remediation", {})
        return cls(**{k: rem[k] for k in cls.__dataclass_fields__ if k in rem})


def classify_generator_proposal(
    state_dict: Dict[str, Any],
    raw_gen_frac: float,
    shield_diag: Optional[Dict[str, Any]] = None,
    adeq_diag: Optional[Dict[str, Any]] = None,
    pre_gen_state: str = "OFF",
    pre_gen_steps_in_state: int = 2,
    gen_derating_factor: float = 1.0,
    rem_cfg: Optional[RemediationConfig] = None,
) -> Dict[str, Any]:
    """
    Classify raw generator proposal and compute condition-aware generator
    shaping reward delta (0.0 unless Remediation A is active).

    Sign & thresholds:
      - Action frac in [0, 1] -> requested_p = frac * 0.40 MW.
      - SafetyShield / DynamicGenerator run threshold: frac > 0.125 (0.05 MW) / > 0.10 (0.04 MW).
    """
    diag = shield_diag or {}
    adeq = adeq_diag or {}
    cfg = rem_cfg or RemediationConfig()

    cap_mw = float(GENERATOR["capacity_mw"]) * float(np.clip(gen_derating_factor, 0.1, 1.0))
    min_down_steps = int(GENERATOR.get("min_down_time_steps", 2))
    min_up_steps = int(GENERATOR.get("min_up_time_steps", 4))

    frac = float(np.clip(raw_gen_frac, 0.0, 1.0))
    requested_p_mw = frac * float(GENERATOR["capacity_mw"])
    proposed_run = requested_p_mw > 0.05  # Matches SafetyShield threshold (frac > 0.125)

    grid_conn = bool(state_dict.get("grid_connected", True))
    grid_avail = bool(state_dict.get("grid_available", True))
    is_storm = bool(state_dict.get("is_storm_warning", False))
    nom_total = float(state_dict.get("nominal_total_mw", 0.0))
    re_out = float(state_dict.get("re_gen_mw", 0.0))
    max_import = float(GRID_CONNECTION.get("max_import_mw", 1.0))

    grid_inadequate = (not grid_conn) or (not grid_avail) or ((nom_total - re_out) > (max_import - 1e-4))
    in_min_up_hold = (pre_gen_state == "RUNNING") and (pre_gen_steps_in_state < min_up_steps)

    # Physical inability to deliver immediately: SHUTTING_DOWN, OFF cooling down, or cap < requested
    physically_unavailable = (
        (pre_gen_state == "SHUTTING_DOWN")
        or (pre_gen_state == "OFF" and pre_gen_steps_in_state < min_down_steps)
        or (gen_derating_factor < 0.99 and requested_p_mw > cap_mw + 1e-4)
    )

    necessary_dispatch = proposed_run and (grid_inadequate or is_storm or in_min_up_hold)
    unnecessary_dispatch = proposed_run and (not grid_inadequate) and (not is_storm) and (not in_min_up_hold)
    blocked_by_physical = proposed_run and physically_unavailable

    # Fallback started or boosted generator beyond shielded proposal
    exec_acts = diag.get("executed_actions", {})
    sh_gen_mw = float(parse_agent_action("generator", exec_acts.get("generator", np.array([0.0])))["requested_power_mw"]) if exec_acts else (0.0 if unnecessary_dispatch else requested_p_mw)
    actual_gen_mw = float(state_dict.get("generator_mw", 0.0))
    gen_state_now = str(state_dict.get("generator_state", "OFF"))
    corrected_by_fallback = bool(adeq.get("adequacy_intervention", False)) and (
        (actual_gen_mw > sh_gen_mw + 0.02)
        or (gen_state_now == "STARTING" and not proposed_run)
    )

    avoidable_shortfall = float(adeq.get("avoidable_shortfall_mw", state_dict.get("avoidable_shortfall_mw", 0.0)))
    can_start_or_run = (pre_gen_state in ("RUNNING", "STARTING")) or (
        pre_gen_state == "OFF" and pre_gen_steps_in_state >= min_down_steps
    )
    missed_necessary_startup = (
        (not grid_conn)
        and can_start_or_run
        and (not proposed_run)
        and (avoidable_shortfall > 0.02 or bool(adeq.get("needs_gen_start", False)))
    )

    # Condition-aware reward shaping for generator actor ONLY when Remediation A enabled
    gen_reward_delta = 0.0
    if cfg.remediation_gen:
        if unnecessary_dispatch:
            excess_frac = max(0.0, frac - 0.125)
            gen_reward_delta -= float(cfg.gen_unneeded_base_penalty + cfg.gen_unneeded_slope_penalty * excess_frac)
        elif missed_necessary_startup:
            gen_reward_delta -= float(cfg.gen_missed_startup_penalty)

    return {
        "proposed_run": proposed_run,
        "necessary_dispatch": necessary_dispatch,
        "unnecessary_dispatch": unnecessary_dispatch,
        "blocked_by_physical": blocked_by_physical,
        "corrected_by_fallback": corrected_by_fallback,
        "missed_necessary_startup": missed_necessary_startup,
        "grid_inadequate": grid_inadequate,
        "in_min_up_hold": in_min_up_hold,
        "gen_reward_delta": float(gen_reward_delta),
    }


def compute_battery_remediation_signals(
    state_dict: Dict[str, Any],
    prev_bat_energy_mwh: float,
    raw_bat_cmd: float,
    shield_diag: Optional[Dict[str, Any]] = None,
    adeq_diag: Optional[Dict[str, Any]] = None,
    dt_hours: float = 0.25,
    rem_cfg: Optional[RemediationConfig] = None,
    terminal_soc_tariff_usd_per_mwh: float = 120.0,
) -> Dict[str, Any]:
    """
    Compute battery autonomy diagnostics and Remediation B training signals:
      - Exact step storage energy valuation: w_val * $120/MWh * (E_{t+1} - E_t)
      - Genuine deficit under-discharge penalty above applicable reserve
      - Healthy-grid non-surplus cycling penalty
      - Reserve breach penalty (<30% SOC outside critical emergency)
    """
    diag = shield_diag or {}
    adeq = adeq_diag or {}
    cfg = rem_cfg or RemediationConfig()

    cmd = float(np.clip(raw_bat_cmd, -1.0, 1.0))  # +1 charge, -1 discharge
    max_bat_p = float(BATTERY["max_power_mw"])
    reserve_soc = float(BATTERY.get("emergency_reserve_soc", 30.0))
    min_soc = float(BATTERY.get("min_soc", 20.0))
    cap_mwh = float(BATTERY.get("capacity_mwh", 1.0))
    eta_d = float(BATTERY.get("discharge_efficiency", 0.95))

    grid_conn = bool(state_dict.get("grid_connected", True))
    is_storm = bool(state_dict.get("is_storm_warning", False))
    nom_total = float(state_dict.get("nominal_total_mw", 0.0))
    nom_crit = float(state_dict.get("nominal_critical_mw", 0.0))
    nom_flex = float(state_dict.get("nominal_flexible_mw", 0.0))
    nom_other = float(state_dict.get("nominal_other_mw", 0.0))
    re_out = float(state_dict.get("re_gen_mw", 0.0))
    actual_gen = float(state_dict.get("generator_mw", 0.0))
    actual_bat = float(state_dict.get("battery_mw", 0.0))
    curr_energy_mwh = float(state_dict.get("battery_energy_mwh", prev_bat_energy_mwh))
    curr_soc = float(state_dict.get("battery_soc", 80.0))

    # Reconstruct pre-step SOC from prev_bat_energy_mwh
    prev_soc = (prev_bat_energy_mwh / cap_mwh) * 100.0
    avail_norm_mw = float(np.clip((max(0.0, prev_bat_energy_mwh - cap_mwh * (reserve_soc / 100.0)) * eta_d) / dt_hours, 0.0, max_bat_p))
    avail_emerg_mw = float(np.clip((max(0.0, prev_bat_energy_mwh - cap_mwh * (min_soc / 100.0)) * eta_d) / dt_hours, 0.0, max_bat_p))

    max_import = float(GRID_CONNECTION.get("max_import_mw", 1.0)) if grid_conn else 0.0
    net_deficit_after_re_gen = max(0.0, nom_total - re_out - actual_gen - max_import)
    critical_at_risk = net_deficit_after_re_gen > (nom_flex + nom_other + 1e-4)

    usable_dis_mw = avail_emerg_mw if critical_at_risk else avail_norm_mw
    genuine_deficit = (not grid_conn) and (net_deficit_after_re_gen > 0.02) and (usable_dis_mw > 0.02)
    needed_dis_mw = min(net_deficit_after_re_gen, usable_dis_mw) if genuine_deficit else 0.0
    prop_dis_mw = max(0.0, -cmd * max_bat_p)

    # Surplus RE available (charging opportunity on healthy grid)
    re_surplus_mw = max(0.0, re_out - nom_total)
    healthy_non_surplus = grid_conn and (not is_storm) and (net_deficit_after_re_gen <= 1e-4) and (re_surplus_mw <= 0.02)

    reserve_breach_attempt = (prev_soc <= reserve_soc + 0.1) and (not critical_at_risk) and (cmd < -0.05) and grid_conn and (not is_storm)

    energy_val_reward = 0.0
    deficit_penalty = 0.0
    cycling_penalty = 0.0
    reserve_penalty = 0.0

    if cfg.remediation_bat:
        # 1. Exact terminal-energy opportunity alignment (makes healthy cycling unprofitable, RE charging & outage discharge profitable)
        delta_e_mwh = curr_energy_mwh - prev_bat_energy_mwh
        energy_val_reward = float(cfg.bat_energy_val_weight * terminal_soc_tariff_usd_per_mwh * delta_e_mwh)

        # 2. Genuine deficit under-discharge penalty (direct credit assignment fixing Seed 202 idle mode)
        if genuine_deficit and needed_dis_mw > 0.02:
            shortfall_ratio = max(0.0, (needed_dis_mw - prop_dis_mw) / max(needed_dis_mw, 0.05))
            if shortfall_ratio > 0.10:
                deficit_penalty = float(cfg.bat_deficit_under_dis_penalty * shortfall_ratio)

        # 3. Mild healthy-grid unnecessary cycling penalty (when grid healthy & no RE surplus)
        if healthy_non_surplus and abs(cmd) > 0.08:
            cycling_penalty = float(cfg.bat_healthy_cycle_penalty * abs(cmd))

        # 4. Reserve preservation penalty when attempting discretionary discharge below 30%
        if reserve_breach_attempt:
            reserve_penalty = float(cfg.bat_reserve_breach_penalty * abs(cmd))

    bat_reward_delta = energy_val_reward - deficit_penalty - cycling_penalty - reserve_penalty

    return {
        "genuine_deficit": genuine_deficit,
        "critical_at_risk": critical_at_risk,
        "usable_above_reserve_mw": float(usable_dis_mw),
        "needed_dis_mw": float(needed_dis_mw),
        "prop_dis_mw": float(prop_dis_mw),
        "battery_discharged_deficit": bool(genuine_deficit and (cmd * max_bat_p < -0.02)),
        "battery_discharged_healthy": bool(grid_conn and (not is_storm) and (cmd * max_bat_p < -0.02)),
        "reserve_breach_attempt": reserve_breach_attempt,
        "energy_val_reward": float(energy_val_reward),
        "deficit_penalty": float(deficit_penalty),
        "cycling_penalty": float(cycling_penalty),
        "reserve_penalty": float(reserve_penalty),
        "bat_reward_delta": float(bat_reward_delta),
    }
