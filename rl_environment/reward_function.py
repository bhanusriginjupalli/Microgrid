# GRIDGUARD AI
# Phase 2: Cooperative Multi-Agent Reward Function
# ============================================================
# Computes the shared scalar reward for cooperative learning:
#   R_t = - w_cost * C_t
#         - w_crit_ens * ENS_crit_t
#         - w_flex_ens * ENS_flex_t
#         - w_other_ens * ENS_other_t
#         - w_emissions * E_t
#         - w_degrad * D_t
#         - w_voltage * V_t
#         - w_imbalance * B_t
#         - w_curt * Curt_t
#         - w_reserve * Reserve_viol_t
#         + w_service * Q_t
#
# Provides comprehensive breakdown in step info for training diagnostics.
# ============================================================

from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np
import yaml


class MicrogridRewardFunction:
    """
    Standardized cooperative multi-agent reward function balancing
    operating economics, grid resilience, asset longevity, and power quality.
    """

    def __init__(
        self,
        w_cost: float = 0.05,             # Cost weight ($/step -> reward points)
        w_crit_ens: float = 200.0,         # Heavy penalty for critical unserved energy
        w_flex_ens: float = 5.0,           # Modest penalty for flexible curtailment
        w_other_ens: float = 15.0,         # Penalty for other load shedding
        w_emissions: float = 0.02,         # Emissions penalty (kg CO2)
        w_degrad: float = 2.0,             # Battery wear penalty (MWh throughput)
        w_voltage: float = 10.0,           # Voltage deviation penalty
        w_imbalance: float = 20.0,         # Power balance error penalty
        w_curt: float = 8.0,               # Unnecessary renewable curtailment
        w_reserve: float = 15.0,           # Battery reserve violation penalty (<30% SOC)
        w_service: float = 1.0,            # Service bonus for full critical load support
    ):
        self.w_cost = w_cost
        self.w_crit_ens = w_crit_ens
        self.w_flex_ens = w_flex_ens
        self.w_other_ens = w_other_ens
        self.w_emissions = w_emissions
        self.w_degrad = w_degrad
        self.w_voltage = w_voltage
        self.w_imbalance = w_imbalance
        self.w_curt = w_curt
        self.w_reserve = w_reserve
        self.w_service = w_service

    @classmethod
    def from_yaml(cls, filepath: str) -> "MicrogridRewardFunction":
        """Load reward parameters from a YAML configuration file."""
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        return cls(**{k: float(v) for k, v in data.items() if hasattr(cls, k) or k in [
            "w_cost", "w_crit_ens", "w_flex_ens", "w_other_ens", "w_emissions",
            "w_degrad", "w_voltage", "w_imbalance", "w_curt", "w_reserve", "w_service"
        ]})

    def to_dict(self) -> Dict[str, float]:
        """Export reward weights as dictionary."""
        return {
            "w_cost": self.w_cost,
            "w_crit_ens": self.w_crit_ens,
            "w_flex_ens": self.w_flex_ens,
            "w_other_ens": self.w_other_ens,
            "w_emissions": self.w_emissions,
            "w_degrad": self.w_degrad,
            "w_voltage": self.w_voltage,
            "w_imbalance": self.w_imbalance,
            "w_curt": self.w_curt,
            "w_reserve": self.w_reserve,
            "w_service": self.w_service,
        }

    def calculate_reward(
        self,
        state: Dict[str, Any],
        actions: Dict[str, Any],
        shield_info: Dict[str, Any],
        dt_hours: float = 0.25,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute step reward and return decomposed components.
        """
        # 1. Operating Economics ($/step)
        grid_p = float(state.get("grid_import_mw", 0.0))
        grid_import_mwh = max(0.0, grid_p) * dt_hours
        grid_export_mwh = abs(min(0.0, grid_p)) * dt_hours
        step_fuel_l = float(state.get("step_fuel_l", 0.0))

        cost_grid_import = grid_import_mwh * 120.0  # $120/MWh
        rev_grid_export = grid_export_mwh * 40.0    # $40/MWh
        cost_fuel = step_fuel_l * 1.5               # $1.5/L
        net_step_cost = cost_grid_import - rev_grid_export + cost_fuel

        # 2. Energy Not Served (MWh/step)
        crit_shed_mwh = float(state.get("shed_critical_mw", 0.0)) * dt_hours
        flex_shed_mwh = float(state.get("shed_flexible_mw", 0.0)) * dt_hours
        other_shed_mwh = float(state.get("shed_other_mw", 0.0)) * dt_hours

        # 3. Carbon Emissions (kg CO2)
        # Diesel generator: ~2.68 kg CO2/L | Grid: ~400 kg CO2/MWh
        gen_co2 = step_fuel_l * 2.68
        grid_co2 = grid_import_mwh * 400.0
        total_emissions = gen_co2 + grid_co2

        # 4. Battery Degradation Proxy (MWh throughput)
        bat_mw = float(state.get("battery_mw", 0.0))
        battery_throughput_mwh = abs(bat_mw) * dt_hours

        # 5. Voltage Quality Penalty
        pcc_v = float(state.get("pcc_voltage_pu", 1.0))
        v_deviation = max(0.0, abs(pcc_v - 1.0) - 0.05)  # Penalize outside [0.95, 1.05]
        v_penalty = (v_deviation * 20.0) ** 2

        # 6. Power Balance Error (MW)
        imbalance_mw = float(state.get("balance_error_mw", 0.0))

        # 7. Unnecessary Renewable Curtailment (MWh)
        solar_avail = float(state.get("solar_available_mw", state.get("solar_mw", 0.0)))
        wind_avail = float(state.get("wind_available_mw", state.get("wind_mw", 0.0)))
        solar_out = float(state.get("solar_mw", 0.0))
        wind_out = float(state.get("wind_mw", 0.0))
        curt_solar = max(0.0, solar_avail - solar_out) * dt_hours
        curt_wind = max(0.0, wind_avail - wind_out) * dt_hours
        total_curt_mwh = curt_solar + curt_wind

        # 8. Battery Reserve Violation (SOC < 30%)
        soc = float(state.get("battery_soc", 80.0))
        reserve_violation = max(0.0, 30.0 - soc) / 10.0  # Normalized violation factor

        # 9. Critical Load Service Reward
        crit_nom = float(state.get("nominal_critical_mw", 0.3))
        crit_serv = float(state.get("served_critical_mw", 0.3))
        crit_service_ratio = (crit_serv / crit_nom) if crit_nom > 0 else 1.0

        # Assemble weighted components
        r_cost = -self.w_cost * net_step_cost
        r_crit_ens = -self.w_crit_ens * (crit_shed_mwh * 10.0)
        r_flex_ens = -self.w_flex_ens * (flex_shed_mwh * 10.0)
        r_other_ens = -self.w_other_ens * (other_shed_mwh * 10.0)
        r_emissions = -self.w_emissions * (total_emissions * 0.05)
        r_degrad = -self.w_degrad * (battery_throughput_mwh * 10.0)
        r_voltage = -self.w_voltage * v_penalty
        r_imbalance = -self.w_imbalance * (imbalance_mw * 10.0)
        r_curt = -self.w_curt * (total_curt_mwh * 10.0)
        r_reserve = -self.w_reserve * (reserve_violation ** 2)
        r_service = self.w_service * crit_service_ratio

        total_reward = float(
            r_cost
            + r_crit_ens
            + r_flex_ens
            + r_other_ens
            + r_emissions
            + r_degrad
            + r_voltage
            + r_imbalance
            + r_curt
            + r_reserve
            + r_service
        )

        breakdown = {
            "reward_total": total_reward,
            "r_cost": r_cost,
            "r_crit_ens": r_crit_ens,
            "r_flex_ens": r_flex_ens,
            "r_other_ens": r_other_ens,
            "r_emissions": r_emissions,
            "r_degrad": r_degrad,
            "r_voltage": r_voltage,
            "r_imbalance": r_imbalance,
            "r_curt": r_curt,
            "r_reserve": r_reserve,
            "r_service": r_service,
            "net_step_cost_dollars": net_step_cost,
            "crit_shed_mwh": crit_shed_mwh,
            "flex_shed_mwh": flex_shed_mwh,
        }

        return total_reward, breakdown


if __name__ == "__main__":
    rf = MicrogridRewardFunction()
    print("GRIDGUARD AI - Reward Function Self-Test")
    print("=" * 50)

    # Case 1: Ideal balanced green step
    state_good = {
        "grid_import_mw": 0.05,
        "step_fuel_l": 0.0,
        "shed_critical_mw": 0.0,
        "shed_flexible_mw": 0.0,
        "shed_other_mw": 0.0,
        "battery_mw": 0.05,
        "pcc_voltage_pu": 1.000,
        "balance_error_mw": 0.0001,
        "solar_available_mw": 0.40,
        "solar_mw": 0.40,
        "wind_available_mw": 0.20,
        "wind_mw": 0.20,
        "battery_soc": 75.0,
        "nominal_critical_mw": 0.30,
        "served_critical_mw": 0.30,
    }
    r1, b1 = rf.calculate_reward(state_good, {}, {})
    print(f"Good Step: Total Reward = {r1:+.3f} (Cost: {b1['r_cost']:.2f}, Service: {b1['r_service']:.2f})")

    # Case 2: Catastrophic critical load shed step
    state_bad = {
        **state_good,
        "shed_critical_mw": 0.15,
        "served_critical_mw": 0.15,
        "battery_soc": 15.0,
        "step_fuel_l": 20.0,
    }
    r2, b2 = rf.calculate_reward(state_bad, {}, {})
    print(f"Bad Step:  Total Reward = {r2:+.3f} (Crit ENS: {b2['r_crit_ens']:.2f}, Reserve: {b2['r_reserve']:.2f})")

    assert r1 > r2, "Good policy must yield substantially higher reward than bad policy"
    assert b2["r_crit_ens"] < -50.0, "Critical load shedding must incur severe penalty"
    print("Reward function self-test passed successfully.")
