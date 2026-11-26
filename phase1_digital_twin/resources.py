# GRIDGUARD AI
# Phase 1: Resource Models
# ============================================================
# Physics-based models for solar, wind, battery and backup
# generator.  Each resource exposes an update() method that
# returns the current output given environmental conditions.
# ============================================================

import numpy as np
from config import SOLAR, WIND, BATTERY, GENERATOR


class SolarModel:
    """
    Simple irradiance-driven PV model.

    Output = capacity × irradiance_factor × temperature_derating
    irradiance_factor ∈ [0, 1] — normalised GHI / 1000 W m⁻²
    temperature_derating — small efficiency loss at high cell temps
    """

    def __init__(self):
        self.capacity_mw = SOLAR["capacity_mw"]
        self.output_mw = SOLAR["initial_output_mw"]
        self.temp_coeff = -0.004        # %/°C above 25 °C

    def update(self, irradiance_factor: float, ambient_temp_c: float = 25.0) -> float:
        """Return solar output in MW for given conditions."""
        irradiance_factor = np.clip(irradiance_factor, 0.0, 1.0)
        temp_derate = 1.0 + self.temp_coeff * max(0, ambient_temp_c - 25.0)
        self.output_mw = self.capacity_mw * irradiance_factor * max(temp_derate, 0.0)
        return self.output_mw

    def generate_daily_profile(self, hours: np.ndarray, cloud_cover: float = 0.0) -> np.ndarray:
        """
        Approximate daily irradiance curve (sinusoidal, sunrise 6 h – sunset 18 h)
        with a random cloud-cover attenuation.
        """
        irradiance = np.zeros_like(hours, dtype=float)
        daylight = (hours >= 6) & (hours <= 18)
        irradiance[daylight] = np.sin(np.pi * (hours[daylight] - 6) / 12)
        cloud_factor = 1.0 - np.clip(cloud_cover, 0.0, 1.0)
        return irradiance * cloud_factor


class WindModel:
    """
    Simplified cubic power-curve model.

    Below cut-in (3 m/s) output is 0.
    Between cut-in and rated (12 m/s) output scales with (v/v_rated)³.
    Above rated wind speed output is capped at capacity.
    Above cut-out (25 m/s) turbine shuts down for safety.
    """

    def __init__(self):
        self.capacity_mw = WIND["capacity_mw"]
        self.output_mw = WIND["initial_output_mw"]
        self.cut_in = 3.0       # m/s
        self.rated = 12.0       # m/s
        self.cut_out = 25.0     # m/s

    def update(self, wind_speed_ms: float) -> float:
        """Return wind output in MW for a given wind speed."""
        if wind_speed_ms < self.cut_in or wind_speed_ms > self.cut_out:
            self.output_mw = 0.0
        elif wind_speed_ms >= self.rated:
            self.output_mw = self.capacity_mw
        else:
            self.output_mw = self.capacity_mw * ((wind_speed_ms - self.cut_in) /
                                                  (self.rated - self.cut_in)) ** 3
        return self.output_mw

    def generate_daily_profile(self, hours: np.ndarray, mean_speed: float = 8.0,
                               variability: float = 3.0) -> np.ndarray:
        """
        Generate a stochastic wind-speed profile for a 24 h period.
        Uses a sinusoidal diurnal pattern plus Gaussian noise.
        """
        base = mean_speed + 2.0 * np.sin(2 * np.pi * (hours - 3) / 24)
        noise = np.random.normal(0, variability, size=hours.shape)
        speeds = np.clip(base + noise, 0, 30)
        return speeds


class BatteryModel:
    """
    Lithium-ion battery energy storage system (BESS).

    Positive p_mw → charging.
    Negative p_mw → discharging.
    SOC is tracked in percent [min_soc, max_soc].
    """

    def __init__(self):
        self.capacity_mwh = BATTERY["capacity_mwh"]
        self.max_power_mw = BATTERY["max_power_mw"]
        self.soc = BATTERY["initial_soc"]               # percent
        self.min_soc = BATTERY["min_soc"]
        self.max_soc = BATTERY["max_soc"]
        self.charge_eff = BATTERY["charge_efficiency"]
        self.discharge_eff = BATTERY["discharge_efficiency"]

    @property
    def energy_mwh(self) -> float:
        """Current stored energy in MWh."""
        return self.capacity_mwh * self.soc / 100.0

    def max_charge_mw(self) -> float:
        """Maximum power the battery can accept right now."""
        if self.soc >= self.max_soc:
            return 0.0
        return self.max_power_mw

    def max_discharge_mw(self) -> float:
        """Maximum power the battery can deliver right now (positive value)."""
        if self.soc <= self.min_soc:
            return 0.0
        return self.max_power_mw

    def update(self, p_mw: float, dt_hours: float) -> float:
        """
        Apply a power set-point for dt_hours.

        Args:
            p_mw:      positive = charge, negative = discharge
            dt_hours:   duration of the time step

        Returns:
            Actual power applied (may be clipped by SOC limits).
        """
        # Clip to inverter limits
        p_mw = np.clip(p_mw, -self.max_power_mw, self.max_power_mw)

        if p_mw > 0:
            # Charging
            energy_in = p_mw * dt_hours * self.charge_eff
            new_soc = self.soc + (energy_in / self.capacity_mwh) * 100.0
            if new_soc > self.max_soc:
                # Reduce power to reach exactly max_soc
                available = (self.max_soc - self.soc) / 100.0 * self.capacity_mwh
                p_mw = available / (dt_hours * self.charge_eff) if dt_hours > 0 else 0
                new_soc = self.max_soc
            self.soc = new_soc
        elif p_mw < 0:
            # Discharging
            energy_out = abs(p_mw) * dt_hours / self.discharge_eff
            new_soc = self.soc - (energy_out / self.capacity_mwh) * 100.0
            if new_soc < self.min_soc:
                available = (self.soc - self.min_soc) / 100.0 * self.capacity_mwh
                p_mw = -(available * self.discharge_eff / dt_hours) if dt_hours > 0 else 0
                new_soc = self.min_soc
            self.soc = new_soc

        return p_mw


class GeneratorModel:
    """
    Backup diesel generator.

    Can be started / stopped.  Ramp rate is simplified to instantaneous
    for Phase 1 (a ramp-rate constraint will be added later).
    """

    def __init__(self):
        self.capacity_mw = GENERATOR["capacity_mw"]
        self.output_mw = GENERATOR["initial_output_mw"]
        self.running = self.output_mw > 0

    def start(self, p_mw: float = None) -> float:
        """Start the generator at the given power (or full capacity)."""
        self.running = True
        self.output_mw = min(p_mw or self.capacity_mw, self.capacity_mw)
        return self.output_mw

    def stop(self) -> float:
        """Shut down the generator."""
        self.running = False
        self.output_mw = 0.0
        return self.output_mw

    def update(self, p_mw: float) -> float:
        """Set generator output (must be running)."""
        if not self.running:
            self.output_mw = 0.0
        else:
            self.output_mw = np.clip(p_mw, 0, self.capacity_mw)
        return self.output_mw


# ────────────────────────────────────────────────────────────
# Quick self-test
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sol = SolarModel()
    wnd = WindModel()
    bat = BatteryModel()
    gen = GeneratorModel()

    print("GRIDGUARD AI - Resource Models Self-Test")
    print("=" * 50)

    # Solar at 70 % irradiance, 30 °C
    print(f"Solar output (70 % irradiance, 30 °C): {sol.update(0.7, 30.0):.3f} MW")

    # Wind at 10 m/s
    print(f"Wind output (10 m/s):                  {wnd.update(10.0):.3f} MW")

    # Battery discharge 0.2 MW for 15 min
    actual = bat.update(-0.2, 0.25)
    print(f"Battery discharge 0.2 MW x 15 min:     {actual:.3f} MW  SOC -> {bat.soc:.1f}%")

    # Generator start at 0.3 MW
    gen.start(0.3)
    print(f"Generator started at:                  {gen.output_mw:.3f} MW")
