"""
rl_environment/operational_adequacy.py
======================================
GRIDGUARD AI — Operational Adequacy & Anti-Inaction Safety Layer (Phase 4.4.2)

Separates physical service adequacy from actuator-boundary legality:
  1. Detects grid-connected vs islanded operation.
  2. Calculates available renewable generation, feasible battery discharge
     (both normal >= 30% reserve and emergency >= 20% hard floor), and feasible
     generator output (respecting state machine, startup delay, ramp limits,
     minimum stable output, derating, and optional fuel limit).
  3. Estimates instantaneous supply-demand deficit and checks whether the
     agent's proposed actions leave an avoidable shortfall ("passive inaction").
  4. Optionally applies an explicit, deterministic, configurable fallback
     that commits feasible resources before critical load can be shed, while
     never forcing a generator that is unavailable, cooling down, or fuel-limited.
  5. Records separately:
       - boundary_clipping
       - emergency_override
       - supply_adequacy_intervention
       - unavoidable_load_shedding
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np

try:
    from config import BATTERY, GENERATOR, GRID_CONNECTION
except ImportError:
    from phase1_digital_twin.config import BATTERY, GENERATOR, GRID_CONNECTION

from rl_environment.action_space import parse_agent_action


@dataclass
class AdequacyConfig:
    enabled: bool = True
    apply_fallback: bool = True
    inaction_deficit_threshold_mw: float = 0.02
    generator_start_deficit_threshold_mw: float = 0.04
    allow_emergency_battery_for_critical: bool = True
    max_fuel_l: Optional[float] = None


class OperationalAdequacyChecker:
    """
    Physical operational adequacy evaluator and configurable deterministic fallback.
    """

    def __init__(self, config: Optional[AdequacyConfig] = None):
        self.config = config or AdequacyConfig()

    def evaluate_step(
        self,
        shielded_actions: Dict[str, Any],
        grid_connected: bool,
        re_available_mw: float,
        nominal_demand: Dict[str, float],
        battery: Any,
        generator: Any,
        gen_derating_factor: float = 1.0,
        dt_hours: float = 0.25,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Evaluate whether shielded_actions satisfy physical supply adequacy given
        current step conditions and asset constraints.

        Returns:
            (final_actions, adequacy_diagnostics)
        """
        final_actions = {
            k: (v.copy() if hasattr(v, "copy") else v) for k, v in shielded_actions.items()
        }

        nom_total = float(nominal_demand.get("total_demand_mw", 0.0))
        nom_crit = float(nominal_demand.get("critical_mw", 0.0))
        nom_flex = float(nominal_demand.get("flexible_mw", 0.0))
        nom_other = float(nominal_demand.get("other_mw", 0.0))
        noncrit_mw = nom_flex + nom_other

        # Parse agent's shielded proposals
        bat_parsed = parse_agent_action("battery", shielded_actions.get("battery", np.array([0.0])))
        bat_req_mw = float(bat_parsed["requested_power_mw"])  # >0 charge, <0 discharge

        gen_parsed = parse_agent_action("generator", shielded_actions.get("generator", np.array([0.0])))
        gen_req_raw_mw = float(gen_parsed["requested_power_mw"])

        load_parsed = parse_agent_action("load_management", shielded_actions.get("load_management", 0))
        flex_curt_frac = float(load_parsed["flexible_curtail_frac"])
        shed_other = bool(load_parsed["shed_other_loads"])
        discretionary_shed_mw = nom_flex * flex_curt_frac + (nom_other * 0.50 if shed_other else 0.0)

        # 1. Feasible Battery Capability
        feasible_bat_normal_mw = float(battery.available_discharge_mw(dt_hours, emergency=False))
        feasible_bat_emerg_mw = float(battery.available_discharge_mw(dt_hours, emergency=True))
        max_bat_power = float(battery.max_power_mw)

        # 2. Feasible Generator Capability (Respecting dynamics & fuel)
        gen_cap_mw = float(generator.capacity_mw) * float(np.clip(gen_derating_factor, 0.1, 1.0))
        gen_min_mw = min(float(generator.min_output_mw), gen_cap_mw)
        gen_state_str = str(generator.state.value if hasattr(generator.state, "value") else generator.state)

        fuel_ok = True
        if self.config.max_fuel_l is not None and float(generator.total_fuel_l) >= float(self.config.max_fuel_l):
            fuel_ok = False

        dt_min = dt_hours * 60.0
        max_ramp_up = float(generator.ramp_up_rate_min) * dt_min
        max_ramp_down = float(generator.ramp_down_rate_min) * dt_min

        can_start_now = False
        feasible_gen_now_max = 0.0
        feasible_gen_now_min = 0.0

        if not fuel_ok:
            can_start_now = False
            feasible_gen_now_max = 0.0
        elif gen_state_str == "RUNNING":
            can_start_now = False  # already running
            feasible_gen_now_max = float(np.clip(generator.output_mw + max_ramp_up, gen_min_mw, gen_cap_mw))
            feasible_gen_now_min = float(np.clip(generator.output_mw - max_ramp_down, gen_min_mw, gen_cap_mw))
        elif gen_state_str == "STARTING":
            # Will transition to RUNNING at end of this step if startup_delay_steps reached
            can_start_now = True
            if (generator.steps_in_state + 1) >= generator.startup_delay_steps:
                feasible_gen_now_max = gen_min_mw
                feasible_gen_now_min = gen_min_mw
            else:
                feasible_gen_now_max = 0.0
        elif gen_state_str == "OFF":
            can_start_now = bool(generator.steps_in_state >= generator.min_down_steps)
            feasible_gen_now_max = 0.0
        else:
            # SHUTTING_DOWN: cannot reverse ramp immediately
            can_start_now = False
            feasible_gen_now_max = max(0.0, float(generator.output_mw) - max_ramp_down)

        # 3. Proposed Supply under Shielded Actions (without emergency battery)
        if bat_req_mw >= 0.0:
            prop_bat_effect_mw = -min(bat_req_mw, float(battery.available_charge_mw(dt_hours)))  # charging draws power
        else:
            prop_bat_effect_mw = min(abs(bat_req_mw), feasible_bat_normal_mw)

        if gen_state_str == "RUNNING" and gen_req_raw_mw > 0.04 and fuel_ok:
            prop_gen_mw = float(np.clip(gen_req_raw_mw, feasible_gen_now_min, feasible_gen_now_max))
        elif gen_state_str == "STARTING" and gen_req_raw_mw > 0.04 and fuel_ok:
            prop_gen_mw = feasible_gen_now_max  # 0 until delay completes, or min_output on step
        else:
            prop_gen_mw = 0.0

        prop_internal_supply = re_available_mw + prop_gen_mw + prop_bat_effect_mw

        # Grid capacity
        max_grid_import = float(GRID_CONNECTION.get("max_import_mw", 1.0)) if grid_connected else 0.0
        prop_total_supply = prop_internal_supply + max_grid_import

        # 4. Deficit & Adequacy Evaluation
        raw_deficit_vs_re = max(0.0, nom_total - re_available_mw)
        proposed_unmet_total = max(0.0, nom_total - prop_total_supply)

        # Max physically feasible supply right now (respecting 30% reserve)
        max_norm_internal = re_available_mw + feasible_bat_normal_mw + feasible_gen_now_max
        max_norm_total = max_norm_internal + max_grid_import

        # Max physically feasible emergency supply right now (using 20%-30% battery reserve)
        max_emerg_internal = re_available_mw + feasible_bat_emerg_mw + feasible_gen_now_max
        max_emerg_total = max_emerg_internal + max_grid_import

        # Determine if agent left an avoidable shortfall or failed to ready generator
        avoidable_shortfall_now = max(0.0, min(nom_total, max_norm_total) - prop_total_supply)
        critical_at_risk_proposed = (proposed_unmet_total > noncrit_mw + 1e-4)

        # Check if islanded deficit warrants starting backup generator for future steps
        rem_after_max_norm_bat = max(0.0, raw_deficit_vs_re - feasible_bat_normal_mw - max_grid_import)
        needs_gen_start = (
            (not grid_connected)
            and fuel_ok
            and (gen_state_str in ("OFF", "STARTING"))
            and can_start_now
            and (rem_after_max_norm_bat > self.config.generator_start_deficit_threshold_mw)
            and (gen_req_raw_mw <= 0.04)
        )

        # Check if emergency battery reserve (20-30%) is available & needed to protect critical load
        unmet_even_with_norm_gen = max(0.0, nom_total - max_norm_total)
        needs_emerg_battery = (
            self.config.allow_emergency_battery_for_critical
            and (not grid_connected)
            and (unmet_even_with_norm_gen > noncrit_mw + 1e-4)
            and (feasible_bat_emerg_mw > feasible_bat_normal_mw + 1e-4)
        )

        inaction_detected = False
        if not grid_connected:
            if avoidable_shortfall_now > self.config.inaction_deficit_threshold_mw:
                inaction_detected = True
            elif critical_at_risk_proposed and (avoidable_shortfall_now > 1e-4 or needs_emerg_battery or needs_gen_start):
                inaction_detected = True
            elif needs_gen_start:
                inaction_detected = True
        else:
            # Grid connected: only an adequacy issue if grid import limit is exceeded and local resources unused
            if proposed_unmet_total > self.config.inaction_deficit_threshold_mw and avoidable_shortfall_now > 1e-4:
                inaction_detected = True

        # 5. Apply Explicit Configurable Deterministic Fallback (if enabled)
        adequacy_intervention = False
        allow_emergency_battery = False
        intervention_reasons = []

        if self.config.enabled and self.config.apply_fallback and (inaction_detected or needs_emerg_battery):
            # A. Battery Fallback: stop unnecessary charging or insufficient discharging during deficit
            target_total_needed = raw_deficit_vs_re - max_grid_import
            if target_total_needed > self.config.inaction_deficit_threshold_mw:
                # Try covering with normal battery first, or emergency battery if critical at risk
                if needs_emerg_battery:
                    allow_emergency_battery = True
                    bat_limit = feasible_bat_emerg_mw
                else:
                    bat_limit = feasible_bat_normal_mw

                # Only ask battery to cover what generator isn't already covering
                needed_from_bat_and_gen = target_total_needed

                # First allocate normal battery up to deficit (fast inverter response)
                desired_bat_dis = min(needed_from_bat_and_gen, feasible_bat_normal_mw)
                rem_after_norm = max(0.0, needed_from_bat_and_gen - desired_bat_dis)

                # Next allocate feasible generator (if running or needs startup)
                desired_gen_mw = gen_req_raw_mw
                if gen_state_str == "RUNNING" and fuel_ok and rem_after_norm > 0.01:
                    desired_gen_mw = float(np.clip(rem_after_norm, feasible_gen_now_min, feasible_gen_now_max))
                elif needs_gen_start:
                    # Request startup at proportional target fraction
                    desired_gen_mw = float(np.clip(max(gen_min_mw, rem_after_norm), gen_min_mw, gen_cap_mw))

                # Actual immediate gen contribution
                imm_gen = desired_gen_mw if gen_state_str == "RUNNING" else (
                    feasible_gen_now_max if (gen_state_str == "STARTING" and (generator.steps_in_state + 1) >= generator.startup_delay_steps) else 0.0
                )

                rem_for_emerg = max(0.0, needed_from_bat_and_gen - desired_bat_dis - imm_gen)
                if needs_emerg_battery and rem_for_emerg > noncrit_mw + 1e-4:
                    desired_bat_dis = min(needed_from_bat_and_gen - imm_gen, feasible_bat_emerg_mw)

                new_bat_cmd = float(np.clip(-desired_bat_dis / max_bat_power, -1.0, 1.0)) if desired_bat_dis > 0 else float(bat_parsed["battery_cmd"])
                new_gen_frac = float(np.clip(desired_gen_mw / float(generator.capacity_mw), 0.0, 1.0))

                if abs(new_bat_cmd - float(bat_parsed["battery_cmd"])) > 0.02 and desired_bat_dis > prop_bat_effect_mw + 0.01:
                    final_actions["battery"] = np.array([new_bat_cmd], dtype=np.float32)
                    adequacy_intervention = True
                    intervention_reasons.append(
                        f"ADEQUACY_BATTERY_DISPATCH({prop_bat_effect_mw:.3f}->{desired_bat_dis:.3f}MW)"
                    )

                if abs(new_gen_frac - float(gen_parsed["generator_fraction"])) > 0.02 and desired_gen_mw > prop_gen_mw + 0.01:
                    final_actions["generator"] = np.array([new_gen_frac], dtype=np.float32)
                    adequacy_intervention = True
                    intervention_reasons.append(
                        f"ADEQUACY_GENERATOR_DISPATCH(req={desired_gen_mw:.3f}MW,state={gen_state_str})"
                    )
                elif needs_gen_start and new_gen_frac > 0.1:
                    final_actions["generator"] = np.array([new_gen_frac], dtype=np.float32)
                    adequacy_intervention = True
                    intervention_reasons.append("ADEQUACY_GENERATOR_STARTUP_REQUESTED")

                # If agent also asked for excessive load shedding when supply was restored, relax unnecessary load shed
                post_fallback_supply = re_available_mw + imm_gen + desired_bat_dis + max_grid_import
                if post_fallback_supply >= nom_total - 1e-4 and load_parsed["action_idx"] > 0:
                    final_actions["load_management"] = 0
                    adequacy_intervention = True
                    intervention_reasons.append("ADEQUACY_PREVENTED_UNNEEDED_SHED")

        # 6. Compute Unavoidable Physical Deficit (deficit beyond physical limits)
        max_physical_total = max_emerg_total if self.config.allow_emergency_battery_for_critical else max_norm_total
        unavoidable_total_deficit_mw = max(0.0, nom_total - max_physical_total)
        unavoidable_critical_deficit_mw = max(0.0, unavoidable_total_deficit_mw - noncrit_mw)

        diagnostics = {
            "grid_connected": bool(grid_connected),
            "re_available_mw": round(re_available_mw, 4),
            "nominal_total_mw": round(nom_total, 4),
            "nominal_critical_mw": round(nom_crit, 4),
            "feasible_bat_normal_mw": round(feasible_bat_normal_mw, 4),
            "feasible_bat_emerg_mw": round(feasible_bat_emerg_mw, 4),
            "feasible_gen_now_max_mw": round(feasible_gen_now_max, 4),
            "generator_state": gen_state_str,
            "can_start_generator": bool(can_start_now and fuel_ok),
            "proposed_total_supply_mw": round(prop_total_supply, 4),
            "proposed_unmet_total_mw": round(proposed_unmet_total, 4),
            "avoidable_shortfall_mw": round(avoidable_shortfall_now, 4),
            "inaction_detected": bool(inaction_detected),
            "adequacy_intervention": bool(adequacy_intervention),
            "allow_emergency_battery": bool(allow_emergency_battery),
            "intervention_reasons": intervention_reasons,
            "unavoidable_total_deficit_mw": round(unavoidable_total_deficit_mw, 4),
            "unavoidable_critical_deficit_mw": round(unavoidable_critical_deficit_mw, 4),
        }
        return final_actions, diagnostics
