# GRIDGUARD AI
# Phase 2: Greedy Instantaneous Controller
# ============================================================
# Myopic baseline controller that optimizes only for the immediate
# single timestep without considering future states, reserve margins,
# or storm warnings.
# ============================================================

from typing import Dict, Any
import numpy as np


class GreedyController:
    """
    Myopic baseline controller that satisfies immediate demand
    using the quickest available resource, draining battery
    reserves rapidly without multi-step foresight.
    """

    def __init__(self, name: str = "Greedy Controller"):
        self.name = name

    def select_actions(
        self,
        observations: Dict[str, np.ndarray],
        state_dict: Dict[str, Any],
        dt_hours: float = 0.25,
    ) -> Dict[str, Any]:
        """Select myopic instantaneous action."""
        grid_conn = state_dict.get("grid_connected", True)
        grid_avail = state_dict.get("grid_available", True)

        sol_pot = float(state_dict.get("solar_available_mw", 0.0))
        wnd_pot = float(state_dict.get("wind_available_mw", 0.0))
        re_pot = sol_pot + wnd_pot
        nom_demand = float(state_dict.get("nominal_total_mw", 0.8))
        net_deficit = nom_demand - re_pot
        soc = float(state_dict.get("battery_soc", 80.0))

        # 1. Renewable: 0% curtailment
        a_renewable = np.array([0.0, 0.0], dtype=np.float32)

        # 2. Grid Resilience
        if not grid_conn:
            a_grid = 3 if grid_avail else 2
        else:
            a_grid = 1 if not grid_avail else 0

        # 3. Greedy dispatch
        if grid_conn and a_grid != 1:
            # Grid connected: pure grid reliance
            a_battery = np.array([0.0], dtype=np.float32)
            a_generator = np.array([0.0], dtype=np.float32)
            a_load = 0
        else:
            # Islanded: aggressively drain battery first, then fire generator
            if net_deficit <= 0:
                a_battery = np.array([1.0], dtype=np.float32)  # Full charge
                a_generator = np.array([0.0], dtype=np.float32)
                a_load = 0
            else:
                # Greedy: discharge maximum battery regardless of reserve
                if soc > 20.5:
                    a_battery = np.array([-1.0], dtype=np.float32)
                else:
                    a_battery = np.array([0.0], dtype=np.float32)

                # Generator requested at full capacity if deficit exists
                if net_deficit > 0.10:
                    a_generator = np.array([1.0], dtype=np.float32)
                else:
                    a_generator = np.array([0.0], dtype=np.float32)

                # No proactive load shedding (waits until forced)
                a_load = 0

        return {
            "renewable": a_renewable,
            "battery": a_battery,
            "generator": a_generator,
            "load_management": a_load,
            "grid_resilience": a_grid,
        }

    def reset(self):
        pass
