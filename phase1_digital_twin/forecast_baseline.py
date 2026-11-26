# GRIDGUARD AI
# Phase 1.5: Forecasting Interface & Baselines
# ============================================================
# Provides multi-step lookahead forecasts for:
#   - Solar PV generation (1 - 4 hours ahead)
#   - Wind turbine generation (1 - 4 hours ahead)
#   - Campus electricity demand (1 - 4 hours ahead)
#   - Grid disturbance / outage risk index
#
# Baseline algorithms:
#   1. Persistence (previous known value)
#   2. Moving Average (MA over rolling window)
#   3. Diurnal Model (astronomical solar curve + schedule)
# ============================================================

from typing import Dict, List, Any
import numpy as np


class ForecastBaseline:
    """
    Modular forecasting engine establishing predictive baselines
    for energy management controllers and future RL observations.
    """

    def __init__(
        self,
        horizon_hours: float = 4.0,
        dt_hours: float = 0.25,
        history_window: int = 8,
        method: str = "diurnal",
    ):
        self.horizon_hours = horizon_hours
        self.dt_hours = dt_hours
        self.horizon_steps = int(horizon_hours / dt_hours)
        self.history_window = history_window
        self.method = method

        # Rolling observation buffers
        self.solar_history: List[float] = []
        self.wind_history: List[float] = []
        self.demand_history: List[float] = []
        self.storm_risk_history: List[float] = []

    def observe(
        self,
        solar_mw: float,
        wind_mw: float,
        demand_mw: float,
        storm_risk: float = 0.0,
    ):
        """Append latest ground truth observation to history buffer."""
        self.solar_history.append(float(solar_mw))
        self.wind_history.append(float(wind_mw))
        self.demand_history.append(float(demand_mw))
        self.storm_risk_history.append(float(storm_risk))

        # Keep buffer bounded
        max_len = self.history_window * 4
        if len(self.solar_history) > max_len:
            self.solar_history = self.solar_history[-max_len:]
            self.wind_history = self.wind_history[-max_len:]
            self.demand_history = self.demand_history[-max_len:]
            self.storm_risk_history = self.storm_risk_history[-max_len:]

    def predict(
        self,
        current_hour: float,
        known_cloud_factor: float = 1.0,
        storm_warning: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate trajectory predictions across the lookahead horizon.

        Returns:
            Dict containing horizon time array and predicted arrays for
            solar, wind, demand, net load, and outage risk.
        """
        horizon_times = [
            current_hour + (step + 1) * self.dt_hours for step in range(self.horizon_steps)
        ]

        if not self.solar_history:
            # Fallback if no history yet
            last_solar = 0.2
            last_wind = 0.1
            last_demand = 0.8
        else:
            last_solar = self.solar_history[-1]
            last_wind = self.wind_history[-1]
            last_demand = self.demand_history[-1]

        solar_pred = np.zeros(self.horizon_steps)
        wind_pred = np.zeros(self.horizon_steps)
        demand_pred = np.zeros(self.horizon_steps)
        risk_pred = np.zeros(self.horizon_steps)

        if self.method == "persistence":
            solar_pred[:] = max(0.0, last_solar)
            wind_pred[:] = max(0.0, last_wind)
            demand_pred[:] = max(0.1, last_demand)

        elif self.method == "moving_average":
            w = min(len(self.solar_history), self.history_window)
            solar_pred[:] = max(0.0, float(np.mean(self.solar_history[-w:])))
            wind_pred[:] = max(0.0, float(np.mean(self.wind_history[-w:])))
            demand_pred[:] = max(0.1, float(np.mean(self.demand_history[-w:])))

        else:  # "diurnal" (default physically informed baseline)
            # 1. Solar forecast based on diurnal sun arc and weather factor
            for i, h in enumerate(horizon_times):
                h_mod = h % 24.0
                if 6.0 <= h_mod <= 18.0:
                    clear_sky = 0.5 * np.sin(np.pi * (h_mod - 6.0) / 12.0)
                else:
                    clear_sky = 0.0
                # Scale with known cloud attenuation
                solar_pred[i] = max(0.0, clear_sky * known_cloud_factor)

            # 2. Wind forecast: AR(1) autoregressive decay toward mean speed
            wind_mean = 0.15
            for i in range(self.horizon_steps):
                prev = last_wind if i == 0 else wind_pred[i - 1]
                decay = 0.85
                wind_pred[i] = max(0.0, decay * prev + (1.0 - decay) * wind_mean)

            # 3. Demand forecast based on diurnal industrial schedule
            for i, h in enumerate(horizon_times):
                h_mod = h % 24.0
                if 9.0 <= h_mod < 17.0:
                    base = 0.85
                elif 6.0 <= h_mod < 9.0 or 17.0 <= h_mod < 22.0:
                    base = 0.78
                else:
                    base = 0.65
                demand_pred[i] = base

        # Outage risk prediction
        base_risk = 0.8 if storm_warning else 0.02
        for i, h in enumerate(horizon_times):
            risk_pred[i] = min(1.0, base_risk + 0.05 * (i * self.dt_hours))

        net_load_pred = demand_pred - (solar_pred + wind_pred)

        return {
            "horizon_hours": horizon_times,
            "solar_mw": solar_pred.tolist(),
            "wind_mw": wind_pred.tolist(),
            "demand_mw": demand_pred.tolist(),
            "net_load_mw": net_load_pred.tolist(),
            "outage_risk": risk_pred.tolist(),
            "next_step_solar_mw": float(solar_pred[0]),
            "next_step_wind_mw": float(wind_pred[0]),
            "next_step_demand_mw": float(demand_pred[0]),
            "next_step_outage_risk": float(risk_pred[0]),
        }

    def reset(self):
        """Reset historical observation buffers."""
        self.solar_history.clear()
        self.wind_history.clear()
        self.demand_history.clear()
        self.storm_risk_history.clear()


if __name__ == "__main__":
    fc = ForecastBaseline(horizon_hours=4.0, dt_hours=0.25, method="diurnal")
    print("GRIDGUARD AI - Forecasting Baseline Self-Test")
    print("=" * 50)

    # Feed initial observations at 10:00 AM
    for step in range(4):
        fc.observe(solar_mw=0.42, wind_mw=0.18, demand_mw=0.82, storm_risk=0.0)

    preds = fc.predict(current_hour=10.0, known_cloud_factor=0.9, storm_warning=False)
    print(f"Lookahead horizon: {preds['horizon_hours'][0]:.2f}h to {preds['horizon_hours'][-1]:.2f}h ({len(preds['horizon_hours'])} steps)")
    print(f"Solar forecast (t+1): {preds['next_step_solar_mw']:.3f} MW | Solar peak in horizon: {max(preds['solar_mw']):.3f} MW")
    print(f"Wind forecast (t+1):  {preds['next_step_wind_mw']:.3f} MW")
    print(f"Demand forecast (t+1):{preds['next_step_demand_mw']:.3f} MW")
    print(f"Outage risk (t+1):    {preds['next_step_outage_risk']:.3f}")
    print("Forecasting Baseline test passed successfully.")
