# GRIDGUARD AI
# Phase 1: Basic Power Balance Controller
# ============================================================
# Rule-based dispatch that tries to:
#   1. Meet all load from renewables.
#   2. Use battery to cover shortfalls / absorb surplus.
#   3. Import/export via the grid if battery can't cope.
#   4. Start backup generator if grid is disconnected and
#      deficit remains after battery discharge.
#   5. Shed flexible loads as a last resort.
# ============================================================

import warnings
import numpy as np
import pandapower as pp
from config import GRID_CONNECTION, LOADS, GENERATOR


class MicrogridController:
    """
    Simple priority-dispatch controller for the microgrid.

    Designed for Phase 1: no optimisation, no RL -- just deterministic
    rules that keep the lights on.
    """

    def __init__(self, net, buses, solar, wind, battery, generator):
        self.net = net
        self.buses = buses
        self.solar = solar
        self.wind = wind
        self.battery = battery
        self.generator = generator
        self.grid_connected = GRID_CONNECTION["connected"]
        self.load_shed_mw = 0.0        # how much flexible load is curtailed

        # Create a slack generator on the backup-generator bus for islanded mode.
        # It stays out-of-service until the grid disconnects.
        self._island_gen_idx = pp.create_gen(
            net,
            bus=buses["Backup Generator"],
            p_mw=0,
            vm_pu=1.0,
            slack=True,
            name="Island Slack Generator",
            max_p_mw=GENERATOR["capacity_mw"],
            min_p_mw=0,
            in_service=False,
        )

    # -- public interface ------------------------------------------------
    def step(self, solar_output_mw: float, wind_output_mw: float,
             dt_hours: float) -> dict:
        """
        Execute one control step.

        1. Compute total load and renewable generation.
        2. Decide battery, generator, and grid set-points.
        3. Apply decisions to the pandapower network.
        4. Run power flow.
        5. Return a status dictionary.
        """
        total_load = LOADS["critical_mw"] + LOADS["flexible_mw"] + LOADS["other_mw"]
        renewable = solar_output_mw + wind_output_mw
        deficit = total_load - renewable       # positive => need more power

        battery_p = 0.0
        generator_p = 0.0
        grid_import = 0.0
        self.load_shed_mw = 0.0

        if deficit > 0:
            # -- Shortfall: discharge battery -> grid -> generator -> shed --
            if self.grid_connected:
                grid_import = deficit
                # If grid can cover it, done
            else:
                # Try battery discharge
                available_discharge = self.battery.max_discharge_mw()
                battery_discharge = min(deficit, available_discharge)
                battery_p = -battery_discharge       # negative = discharge
                deficit -= battery_discharge

                if deficit > 0:
                    # Start generator
                    generator_p = self.generator.start(deficit)
                    deficit -= generator_p

                if deficit > 0.001:
                    # Shed flexible loads
                    self.load_shed_mw = min(deficit, LOADS["flexible_mw"])
                    deficit -= self.load_shed_mw
        else:
            surplus = -deficit                          # surplus is positive
            if surplus > 0:
                # -- Surplus: charge battery -> export ---------------------
                available_charge = self.battery.max_charge_mw()
                charge_power = min(surplus, available_charge)
                battery_p = charge_power                # positive = charge
                surplus -= charge_power

                if self.grid_connected:
                    grid_import = -(surplus)              # export to grid
                # If islanded, surplus is curtailed (could add dump load)

        # If grid is connected, battery stays idle in simple mode
        if self.grid_connected:
            battery_p = 0.0
            generator_p = 0.0
            self.generator.stop()

        # -- Apply to pandapower network ---------------------------------
        actual_battery_p = self.battery.update(battery_p, dt_hours)
        self._apply_to_network(solar_output_mw, wind_output_mw,
                               actual_battery_p, generator_p)

        # -- Run power flow ----------------------------------------------
        converged = self._run_power_flow()

        return {
            "solar_mw":        solar_output_mw,
            "wind_mw":         wind_output_mw,
            "battery_mw":      actual_battery_p,
            "generator_mw":    generator_p,
            "grid_import_mw":  self.net.res_ext_grid.p_mw.sum() if converged else grid_import,
            "total_load_mw":   total_load - self.load_shed_mw,
            "load_shed_mw":    self.load_shed_mw,
            "battery_soc":     self.battery.soc,
            "grid_connected":  self.grid_connected,
            "converged":       converged,
        }

    # -- grid connection helpers -----------------------------------------
    def disconnect_grid(self):
        """Simulate grid disconnection (islanded mode)."""
        self.grid_connected = False
        self.net.ext_grid.at[0, "in_service"] = False
        # Enable the island slack generator so power flow has a reference bus
        self.net.gen.at[self._island_gen_idx, "in_service"] = True

    def reconnect_grid(self):
        """Reconnect to utility grid."""
        self.grid_connected = True
        self.net.ext_grid.at[0, "in_service"] = True
        # Disable the island slack generator
        self.net.gen.at[self._island_gen_idx, "in_service"] = False
        self.generator.stop()

    # -- internal helpers ------------------------------------------------
    def _apply_to_network(self, solar_mw, wind_mw, battery_mw, generator_mw):
        """Push set-points into the pandapower data model."""
        # Solar
        self.net.sgen.at[0, "p_mw"] = solar_mw
        # Wind
        self.net.sgen.at[1, "p_mw"] = wind_mw
        # Battery (storage element)
        self.net.storage.at[0, "p_mw"] = battery_mw
        # Backup generator (sgen element -- always disabled in grid-connected mode)
        if generator_mw > 0:
            self.net.sgen.at[2, "in_service"] = True
            self.net.sgen.at[2, "p_mw"] = generator_mw
        else:
            self.net.sgen.at[2, "in_service"] = False
            self.net.sgen.at[2, "p_mw"] = 0.0

        # Island slack gen power set-point (pandapower adjusts it for balance)
        if not self.grid_connected:
            self.net.gen.at[self._island_gen_idx, "p_mw"] = generator_mw

        # Flexible load (after shedding)
        effective_flex = max(LOADS["flexible_mw"] - self.load_shed_mw, 0)
        self.net.load.at[1, "p_mw"] = effective_flex

    def _run_power_flow(self) -> bool:
        """Run pandapower power flow.  Returns True if converged."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pp.runpp(self.net, numba=False)
            return self.net.converged
        except Exception as e:
            print(f"  [Controller] Power flow failed: {e}")
            return False
