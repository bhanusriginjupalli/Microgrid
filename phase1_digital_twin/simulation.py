# GRIDGUARD AI
# Phase 1: Time-Stepped Simulation Engine
# ============================================================
# Runs the microgrid through a configurable time horizon,
# updating resources, applying disturbances, dispatching via
# the controller, and logging every step.
# ============================================================

import numpy as np
import pandas as pd

from config import SIMULATION, LOADS
from network import create_microgrid
from resources import SolarModel, WindModel, BatteryModel, GeneratorModel
from controller import MicrogridController
from disturbances import DisturbanceManager, create_default_scenario


class MicrogridSimulation:
    """
    Orchestrates a 24 h (or custom) time-stepped simulation of the
    GRIDGUARD AI microgrid.
    """

    def __init__(self, disturbance_manager: DisturbanceManager = None,
                 duration_hours: float = None, time_step_minutes: float = None):
        # Time parameters
        self.dt_min = time_step_minutes or SIMULATION["time_step_minutes"]
        self.dt_hours = self.dt_min / 60.0
        self.duration = duration_hours or SIMULATION["duration_hours"]
        self.n_steps = int(self.duration / self.dt_hours)

        # Build network and resources
        self.net, self.buses = create_microgrid()
        self.solar = SolarModel()
        self.wind = WindModel()
        self.battery = BatteryModel()
        self.generator = GeneratorModel()

        # Controller
        self.controller = MicrogridController(
            self.net, self.buses,
            self.solar, self.wind, self.battery, self.generator,
        )

        # Disturbances
        self.dm = disturbance_manager or DisturbanceManager()

        # Precompute time axis
        self.hours = np.linspace(0, self.duration, self.n_steps, endpoint=False)

        # Generate base weather profiles
        self.irradiance_profile = self.solar.generate_daily_profile(self.hours)
        self.wind_speed_profile = self.wind.generate_daily_profile(self.hours)

        # Results log
        self.results: list[dict] = []

    # ── Main simulation loop ────────────────────────────────
    def run(self) -> pd.DataFrame:
        """
        Execute the full simulation.  Returns a DataFrame with one row
        per time step.
        """
        print(f"\nGRIDGUARD AI - Simulating {self.duration}h "
              f"({self.n_steps} steps x {self.dt_min} min)")
        print("=" * 60)

        for i, hour in enumerate(self.hours):
            # 1. Determine environmental conditions
            irradiance = self.irradiance_profile[i]
            wind_speed = self.wind_speed_profile[i]

            # 2. Apply disturbance overrides
            cloud_factor = self.dm.get_cloud_factor(hour)
            irradiance *= cloud_factor

            wind_override = self.dm.get_wind_override(hour)
            if wind_override is not None:
                wind_speed = wind_override

            # 3. Grid connection state
            if self.dm.is_grid_down(hour):
                if self.controller.grid_connected:
                    self.controller.disconnect_grid()
                    print(f"  [!] Hour {hour:5.2f}: GRID DISCONNECTED - islanded mode")
            else:
                if not self.controller.grid_connected:
                    self.controller.reconnect_grid()
                    print(f"  [+] Hour {hour:5.2f}: Grid reconnected")

            # 4. Update resource models
            solar_out = self.solar.update(irradiance, ambient_temp_c=30)
            wind_out = self.wind.update(wind_speed)

            # 5. Dispatch via controller
            status = self.controller.step(solar_out, wind_out, self.dt_hours)

            # 6. Log
            record = {"hour": hour, "step": i, **status,
                      "irradiance": irradiance, "wind_speed_ms": wind_speed}
            self.results.append(record)

            # Progress indicator
            if (i + 1) % 24 == 0 or i == 0:
                print(f"  Step {i+1:4d}/{self.n_steps}  "
                      f"t={hour:5.1f}h  "
                      f"solar={solar_out:.3f}  wind={wind_out:.3f}  "
                      f"SOC={self.battery.soc:.1f}%  "
                      f"grid={'ON' if status['grid_connected'] else 'OFF'}  "
                      f"shed={status['load_shed_mw']:.3f}")

        df = pd.DataFrame(self.results)
        print(f"\nSimulation complete.  {len(df)} records logged.")
        return df

    # ── Convenience: run with default disturbances ──────────
    @classmethod
    def run_default_scenario(cls) -> pd.DataFrame:
        """Run a 24 h simulation with the built-in disturbance scenario."""
        dm = create_default_scenario()
        sim = cls(disturbance_manager=dm)
        return sim.run()


# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = MicrogridSimulation.run_default_scenario()

    # Save results
    import os
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "simulation_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "phase1_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Quick summary
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print(f"  Peak solar:          {df['solar_mw'].max():.3f} MW")
    print(f"  Peak wind:           {df['wind_mw'].max():.3f} MW")
    print(f"  Peak grid import:    {df['grid_import_mw'].max():.3f} MW")
    print(f"  Min battery SOC:     {df['battery_soc'].min():.1f}%")
    print(f"  Max battery SOC:     {df['battery_soc'].max():.1f}%")
    print(f"  Total load shed:     {df['load_shed_mw'].sum() * (SIMULATION['time_step_minutes']/60):.3f} MWh")
    print(f"  Power flow failures: {(~df['converged']).sum()}")
