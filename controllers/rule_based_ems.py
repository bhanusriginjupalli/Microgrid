# GRIDGUARD AI
# Phase 2: Rule-Based Energy Management System (EMS)
# ============================================================
# Operational benchmark controller implementing deterministic,
# priority-based hierarchical dispatch rules:
#   Grid-Connected:
#     - 100% renewable utilization
#     - Battery absorbs surplus solar/wind
#     - Grid covers remaining net deficit
#     - Generator stays OFF
#   Islanded:
#     - Battery provides primary response down to 30% reserve
#     - Generator dispatched for remaining deficit
#     - Emergency battery discharge down to 20%
#     - Orderly flexible load shedding
#   Restoration:
#     - Detects grid return, holds for sync, requests reconnection
# ============================================================

from typing import Dict, Any
import numpy as np


class RuleBasedEMS:
    """
    Deterministic rule-based microgrid energy management controller
    serving as the primary engineering performance baseline.
    """

    def __init__(self, name: str = "Rule-Based EMS"):
        self.name = name

    def select_actions(
        self,
        observations: Dict[str, np.ndarray],
        state_dict: Dict[str, Any],
        dt_hours: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Evaluate current microgrid condition and select multi-agent actions.
        """
        grid_conn = state_dict.get("grid_connected", True)
        grid_avail = state_dict.get("grid_available", True)
        sync_ver = state_dict.get("sync_verified", True)

        sol_pot = float(state_dict.get("solar_available_mw", state_dict.get("solar_mw", 0.0)))
        wnd_pot = float(state_dict.get("wind_available_mw", state_dict.get("wind_mw", 0.0)))
        re_pot = sol_pot + wnd_pot

        nom_demand = float(state_dict.get("nominal_total_mw", 0.8))
        flex_demand = float(state_dict.get("nominal_flexible_mw", 0.2))
        crit_demand = float(state_dict.get("nominal_critical_mw", 0.3))

        soc = float(state_dict.get("battery_soc", 80.0))
        net_deficit = nom_demand - re_pot

        # 1. Renewable Agent: Always 0% curtailment (utilize maximum green power)
        a_renewable = np.array([0.0, 0.0], dtype=np.float32)

        # 2. Grid Resilience Agent
        if not grid_conn:
            if grid_avail and sync_ver:
                a_grid = 3  # REQUEST_RECONNECTION
            else:
                a_grid = 2  # STAY_ISLANDED
        else:
            if not grid_avail:
                a_grid = 1  # REQUEST_ISLANDING
            else:
                a_grid = 0  # STAY_CONNECTED

        # 3. Battery, Generator & Load Actions
        if grid_conn and a_grid != 1:
            # --- Grid Connected Mode ---
            a_generator = np.array([0.0], dtype=np.float32)  # Keep generator OFF
            a_load = 0  # No curtailment

            if net_deficit < 0:
                # Renewable surplus: charge battery up to capacity
                surplus = -net_deficit
                charge_cmd = float(np.clip(surplus / 0.25, 0.0, 1.0))
                a_battery = np.array([charge_cmd], dtype=np.float32)
            else:
                # Deficit covered by grid import; battery idle
                a_battery = np.array([0.0], dtype=np.float32)

        else:
            # --- Islanded Autonomous Mode ---
            if net_deficit <= 0:
                # Islanded surplus: charge battery, generator OFF, no shed
                surplus = -net_deficit
                charge_cmd = float(np.clip(surplus / 0.25, 0.0, 1.0))
                a_battery = np.array([charge_cmd], dtype=np.float32)
                a_generator = np.array([0.0], dtype=np.float32)
                a_load = 0
            else:
                # Islanded deficit:
                # Available normal discharge down to 30% reserve
                usable_soc_mwh = max(0.0, (soc - 30.0) / 100.0 * 1.0)
                max_norm_dis = min(0.25, (usable_soc_mwh * 0.95) / dt_hours)

                if net_deficit <= max_norm_dis:
                    # Battery can fully cover shortfall
                    dis_cmd = float(np.clip(-net_deficit / 0.25, -1.0, 0.0))
                    a_battery = np.array([dis_cmd], dtype=np.float32)
                    a_generator = np.array([0.0], dtype=np.float32)
                    a_load = 0
                else:
                    # Battery normal discharge maxed out
                    rem_deficit = net_deficit - max_norm_dis
                    # Dispatch generator
                    gen_req_p = min(0.40, rem_deficit)
                    gen_frac = float(np.clip(gen_req_p / 0.40, 0.0, 1.0))
                    a_generator = np.array([gen_frac], dtype=np.float32)
                    rem_deficit = max(0.0, rem_deficit - (gen_frac * 0.40))

                    # If deficit still remains, check emergency battery discharge (down to 20%)
                    extra_soc_mwh = max(0.0, (min(soc, 30.0) - 20.0) / 100.0 * 1.0)
                    extra_dis = min(0.25 - max_norm_dis, (extra_soc_mwh * 0.95) / dt_hours)
                    total_dis = max_norm_dis + min(rem_deficit, extra_dis)
                    rem_deficit = max(0.0, rem_deficit - extra_dis)

                    a_battery = np.array([-total_dis / 0.25], dtype=np.float32)

                    # If deficit still remains, shed flexible load
                    if rem_deficit <= 0.001:
                        a_load = 0
                    elif rem_deficit <= 0.25 * flex_demand:
                        a_load = 1  # 25% flexible
                    elif rem_deficit <= 0.50 * flex_demand:
                        a_load = 2  # 50% flexible
                    elif rem_deficit <= 0.75 * flex_demand:
                        a_load = 3  # 75% flexible
                    elif rem_deficit <= flex_demand:
                        a_load = 4  # 100% flexible
                    else:
                        a_load = 5  # 100% flex + other noncritical

        return {
            "renewable": a_renewable,
            "battery": a_battery,
            "generator": a_generator,
            "load_management": a_load,
            "grid_resilience": a_grid,
        }

    def reset(self):
        """Reset internal memory if any."""
        pass
