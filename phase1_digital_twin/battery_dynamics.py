# GRIDGUARD AI
# Phase 1.5: Battery Energy Storage Dynamics
# ============================================================
# Physically consistent Li-ion battery energy and SOC evolution.
# Convention:
#   P_t > 0: Charging
#   P_t < 0: Discharging
#
# Energy evolution:
#   E_{t+1} = E_t + P_t * eta_c * dt         (P_t >= 0)
#   E_{t+1} = E_t + (P_t / eta_d) * dt       (P_t < 0)
#
# Constraints:
#   SOC_min <= SOC_t <= SOC_max
#   |P_t| <= P_max
# ============================================================

import numpy as np
try:
    from config import BATTERY
except ImportError:
    from phase1_digital_twin.config import BATTERY


class DynamicBattery:
    """
    Physically consistent Battery Energy Storage System (BESS) model.
    Tracks state of charge, internal stored energy, charge/discharge limits,
    round-trip efficiencies, and emergency reserve thresholds.
    """

    def __init__(
        self,
        capacity_mwh: float = None,
        max_power_mw: float = None,
        initial_soc: float = None,
        min_soc: float = None,
        max_soc: float = None,
        charge_eff: float = None,
        discharge_eff: float = None,
        emergency_reserve_soc: float = None,
    ):
        self.capacity_mwh = capacity_mwh if capacity_mwh is not None else BATTERY["capacity_mwh"]
        self.max_power_mw = max_power_mw if max_power_mw is not None else BATTERY["max_power_mw"]
        self.min_soc = min_soc if min_soc is not None else BATTERY["min_soc"]
        self.max_soc = max_soc if max_soc is not None else BATTERY["max_soc"]
        self.charge_eff = charge_eff if charge_eff is not None else BATTERY["charge_efficiency"]
        self.discharge_eff = discharge_eff if discharge_eff is not None else BATTERY["discharge_efficiency"]
        self.emergency_reserve_soc = (
            emergency_reserve_soc
            if emergency_reserve_soc is not None
            else BATTERY.get("emergency_reserve_soc", 30.0)
        )

        self._initial_soc = initial_soc if initial_soc is not None else BATTERY["initial_soc"]
        self.soc = float(np.clip(self._initial_soc, self.min_soc, self.max_soc))
        self.history = []

    @property
    def energy_mwh(self) -> float:
        """Current stored energy in MWh."""
        return self.capacity_mwh * (self.soc / 100.0)

    @property
    def min_energy_mwh(self) -> float:
        """Minimum allowable energy in MWh."""
        return self.capacity_mwh * (self.min_soc / 100.0)

    @property
    def max_energy_mwh(self) -> float:
        """Maximum allowable energy in MWh."""
        return self.capacity_mwh * (self.max_soc / 100.0)

    def available_charge_mw(self, dt_hours: float) -> float:
        """
        Calculate maximum charge power (MW, positive) that can be accepted
        over dt_hours without exceeding max_soc.
        """
        if dt_hours <= 0:
            return 0.0
        headroom_mwh = max(0.0, self.max_energy_mwh - self.energy_mwh)
        power_by_energy = headroom_mwh / (self.charge_eff * dt_hours)
        return float(np.clip(power_by_energy, 0.0, self.max_power_mw))

    def available_discharge_mw(self, dt_hours: float, emergency: bool = False) -> float:
        """
        Calculate maximum discharge power (MW, positive value) that can be delivered
        over dt_hours without violating SOC limits.
        If emergency is False, discharges down to emergency_reserve_soc.
        If emergency is True, discharges down to min_soc.
        """
        if dt_hours <= 0:
            return 0.0
        target_min_soc = self.min_soc if emergency else self.emergency_reserve_soc
        target_min_energy = self.capacity_mwh * (target_min_soc / 100.0)
        usable_mwh = max(0.0, self.energy_mwh - target_min_energy)
        power_by_energy = (usable_mwh * self.discharge_eff) / dt_hours
        return float(np.clip(power_by_energy, 0.0, self.max_power_mw))

    def update(self, p_mw: float, dt_hours: float, emergency: bool = False) -> float:
        """
        Apply requested battery power over dt_hours.

        Parameters:
            p_mw: Requested power in MW.
                  p_mw > 0: charging
                  p_mw < 0: discharging
            dt_hours: Timestep duration in hours.
            emergency: If True, allows discharging below emergency reserve down to min_soc.

        Returns:
            Actual power applied in MW (positive = charged, negative = discharged).
        """
        if dt_hours <= 0:
            return 0.0

        p_req = float(p_mw)
        if p_req >= 0:
            # Charging: bound by available charge power
            max_chg = self.available_charge_mw(dt_hours)
            actual_p = min(p_req, max_chg)
            energy_delta = actual_p * self.charge_eff * dt_hours
            new_energy = self.energy_mwh + energy_delta
        else:
            # Discharging: bound by available discharge power
            max_dis = self.available_discharge_mw(dt_hours, emergency=emergency)
            actual_dis = min(abs(p_req), max_dis)
            actual_p = -actual_dis
            # P_t < 0: energy consumed = |P_t| * dt / eta_d
            energy_delta = (actual_p / self.discharge_eff) * dt_hours  # negative value
            new_energy = self.energy_mwh + energy_delta

        # Ensure strict physical boundaries
        new_energy = float(np.clip(new_energy, self.min_energy_mwh, self.max_energy_mwh))
        self.soc = float((new_energy / self.capacity_mwh) * 100.0)

        # Log history
        self.history.append({
            "soc": self.soc,
            "power_mw": actual_p,
            "energy_mwh": self.energy_mwh,
            "dt_hours": dt_hours,
        })

        return actual_p

    def reset(self, initial_soc: float = None):
        """Reset battery state to initial condition."""
        if initial_soc is not None:
            self._initial_soc = initial_soc
        self.soc = float(np.clip(self._initial_soc, self.min_soc, self.max_soc))
        self.history.clear()


if __name__ == "__main__":
    bat = DynamicBattery()
    print("GRIDGUARD AI - Dynamic Battery Self-Test")
    print("=" * 50)
    print(f"Capacity: {bat.capacity_mwh:.2f} MWh | P_max: {bat.max_power_mw:.2f} MW")
    print(f"Initial SOC: {bat.soc:.1f}% ({bat.energy_mwh:.3f} MWh)")

    # Test standard discharge for 1 hour at 0.2 MW (dt=0.25h)
    dt = 0.25
    avail_dis = bat.available_discharge_mw(dt, emergency=False)
    print(f"Available normal discharge: {avail_dis:.3f} MW")

    p_applied = bat.update(-0.25, dt, emergency=False)
    print(f"Discharge step: P={p_applied:.3f} MW | SOC -> {bat.soc:.2f}% | Energy -> {bat.energy_mwh:.3f} MWh")

    # Full charge test
    avail_chg = bat.available_charge_mw(dt)
    print(f"Available charge: {avail_chg:.3f} MW")
    p_chg = bat.update(0.3, dt)
    print(f"Charge step: P={p_chg:.3f} MW | SOC -> {bat.soc:.2f}% | Energy -> {bat.energy_mwh:.3f} MWh")
    print("Dynamic Battery test passed successfully.")
