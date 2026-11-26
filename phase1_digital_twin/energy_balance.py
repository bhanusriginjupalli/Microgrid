# GRIDGUARD AI
# Phase 1.5: Explicit Energy & Power Balance
# ============================================================
# Rigorous calculation and validation of instantaneous power balance
# and cumulative energy conservation at every timestep.
#
# Equation (signed convention: battery > 0 is charge, < 0 is discharge):
#   P_solar + P_wind + P_generator + P_grid - P_battery = P_load_served + P_losses
#
#   P_balance_error = | (P_gen_total + P_grid - P_battery) - (P_load_served + P_losses) |
# ============================================================

from typing import Dict, List, Any
import numpy as np


class EnergyBalanceCalculator:
    """
    Validates physical energy and power balance across microgrid resources,
    storage, grid interface, and consumer loads.
    """

    def __init__(self, tolerance_mw: float = 0.001):
        self.tolerance_mw = tolerance_mw
        self.max_error_mw = 0.0
        self.total_error_mw = 0.0
        self.step_count = 0
        self.violations = 0

    def calculate_balance(
        self,
        solar_mw: float,
        wind_mw: float,
        generator_mw: float,
        grid_mw: float,
        battery_mw: float,
        load_served_mw: float,
        losses_mw: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Calculate power balance for a single timestep.

        Parameters:
            solar_mw: Active solar PV output (>= 0)
            wind_mw: Active wind turbine output (>= 0)
            generator_mw: Active generator output (>= 0)
            grid_mw: Grid active power (> 0 import, < 0 export)
            battery_mw: Battery power (> 0 charge, < 0 discharge)
            load_served_mw: Total consumer active load served (>= 0)
            losses_mw: Network active losses (>= 0)

        Returns:
            Dict containing supply, demand, balance error, and satisfaction flag.
        """
        # Gross generation inside microgrid
        p_re = solar_mw + wind_mw
        p_gen = p_re + generator_mw

        # Total energy supply arriving at the AC distribution bus:
        # Generation + Grid Import - Battery Charge (or + Battery Discharge)
        # Note: if battery_mw < 0, (-battery_mw) is positive (discharging into bus)
        #       if battery_mw > 0, (-battery_mw) is negative (taking power from bus)
        net_supply_mw = p_gen + grid_mw - battery_mw

        # Total demand from the AC distribution bus
        total_demand_mw = load_served_mw + losses_mw

        # Balance error
        balance_error_mw = abs(net_supply_mw - total_demand_mw)
        is_balanced = balance_error_mw <= self.tolerance_mw

        # Track statistics
        self.step_count += 1
        self.total_error_mw += balance_error_mw
        if balance_error_mw > self.max_error_mw:
            self.max_error_mw = balance_error_mw
        if not is_balanced:
            self.violations += 1

        return {
            "p_solar_mw": solar_mw,
            "p_wind_mw": wind_mw,
            "p_gen_mw": generator_mw,
            "p_grid_mw": grid_mw,
            "p_battery_mw": battery_mw,
            "p_load_served_mw": load_served_mw,
            "p_losses_mw": losses_mw,
            "net_supply_mw": net_supply_mw,
            "total_demand_mw": total_demand_mw,
            "balance_error_mw": balance_error_mw,
            "is_balanced": is_balanced,
        }

    def validate_step(self, record: Dict[str, Any]) -> bool:
        """Validate an individual simulation output record."""
        solar = record.get("solar_mw", 0.0)
        wind = record.get("wind_mw", 0.0)
        gen = record.get("generator_mw", 0.0)
        grid = record.get("grid_import_mw", 0.0)
        bat = record.get("battery_mw", 0.0)
        load = record.get("total_load_served_mw", record.get("total_load_mw", 0.0))
        losses = record.get("losses_mw", 0.0)

        result = self.calculate_balance(solar, wind, gen, grid, bat, load, losses)
        return result["is_balanced"]

    def cumulative_balance(self, records: List[Dict[str, Any]], dt_hours: float = 0.25) -> Dict[str, Any]:
        """
        Verify cumulative energy conservation over a sequence of simulation steps.
        Energy (MWh) = Power (MW) * dt (hours).
        """
        total_solar_mwh = sum(r.get("solar_mw", 0.0) for r in records) * dt_hours
        total_wind_mwh = sum(r.get("wind_mw", 0.0) for r in records) * dt_hours
        total_gen_mwh = sum(r.get("generator_mw", 0.0) for r in records) * dt_hours
        total_grid_import_mwh = sum(max(0.0, r.get("grid_import_mw", 0.0)) for r in records) * dt_hours
        total_grid_export_mwh = sum(abs(min(0.0, r.get("grid_import_mw", 0.0))) for r in records) * dt_hours

        # Battery energy throughput
        total_bat_charge_mwh = sum(max(0.0, r.get("battery_mw", 0.0)) for r in records) * dt_hours
        total_bat_discharge_mwh = sum(abs(min(0.0, r.get("battery_mw", 0.0))) for r in records) * dt_hours

        total_load_served_mwh = sum(r.get("total_load_served_mw", r.get("total_load_mw", 0.0)) for r in records) * dt_hours
        total_losses_mwh = sum(r.get("losses_mw", 0.0) for r in records) * dt_hours

        gross_supply_mwh = total_solar_mwh + total_wind_mwh + total_gen_mwh + total_grid_import_mwh + total_bat_discharge_mwh
        gross_demand_mwh = total_load_served_mwh + total_grid_export_mwh + total_bat_charge_mwh + total_losses_mwh

        cumulative_error_mwh = abs(gross_supply_mwh - gross_demand_mwh)
        mean_step_error_mw = (self.total_error_mw / self.step_count) if self.step_count > 0 else 0.0

        return {
            "total_solar_mwh": total_solar_mwh,
            "total_wind_mwh": total_wind_mwh,
            "total_gen_mwh": total_gen_mwh,
            "total_grid_import_mwh": total_grid_import_mwh,
            "total_grid_export_mwh": total_grid_export_mwh,
            "total_bat_charge_mwh": total_bat_charge_mwh,
            "total_bat_discharge_mwh": total_bat_discharge_mwh,
            "total_load_served_mwh": total_load_served_mwh,
            "total_losses_mwh": total_losses_mwh,
            "gross_supply_mwh": gross_supply_mwh,
            "gross_demand_mwh": gross_demand_mwh,
            "cumulative_error_mwh": cumulative_error_mwh,
            "max_step_error_mw": self.max_error_mw,
            "mean_step_error_mw": mean_step_error_mw,
            "violations_count": self.violations,
            "is_conserved": cumulative_error_mwh <= (self.tolerance_mw * len(records) * dt_hours),
        }

    def reset(self):
        """Reset internal error accumulators."""
        self.max_error_mw = 0.0
        self.total_error_mw = 0.0
        self.step_count = 0
        self.violations = 0


if __name__ == "__main__":
    eb = EnergyBalanceCalculator(tolerance_mw=0.001)
    print("GRIDGUARD AI - Energy Balance Self-Test")
    print("=" * 50)

    # Balanced case: 0.35 MW solar + 0.15 MW wind + 0.3 MW grid - 0 battery = 0.8 MW load
    res1 = eb.calculate_balance(
        solar_mw=0.35, wind_mw=0.15, generator_mw=0.0, grid_mw=0.30, battery_mw=0.0, load_served_mw=0.80, losses_mw=0.0
    )
    print(f"Test 1 (Normal): balanced={res1['is_balanced']} | error={res1['balance_error_mw']:.6f} MW")

    # Charging case: 0.5 MW solar + 0.25 MW wind - 0.25 MW battery_chg = 0.5 MW load
    res2 = eb.calculate_balance(
        solar_mw=0.50, wind_mw=0.25, generator_mw=0.0, grid_mw=0.0, battery_mw=0.25, load_served_mw=0.50, losses_mw=0.0
    )
    print(f"Test 2 (Battery Charging): balanced={res2['is_balanced']} | error={res2['balance_error_mw']:.6f} MW")

    # Discharging case: 0.1 MW wind + 0.3 MW gen + (-0.2 MW bat_dis) = 0.6 MW load
    res3 = eb.calculate_balance(
        solar_mw=0.0, wind_mw=0.10, generator_mw=0.30, grid_mw=0.0, battery_mw=-0.20, load_served_mw=0.60, losses_mw=0.0
    )
    print(f"Test 3 (Battery Discharging): balanced={res3['is_balanced']} | error={res3['balance_error_mw']:.6f} MW")

    assert res1["is_balanced"] and res2["is_balanced"] and res3["is_balanced"]
    print("Energy Balance test passed successfully.")
