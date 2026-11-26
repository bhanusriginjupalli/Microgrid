"""
forecasting/baseline_forecaster.py
============================================================
Physical diurnal and schedule-based baseline forecaster.
Implements:
  - Solar: Astronomical clear-sky diurnal sine model modulated by cloud factor
  - Wind: Autoregressive AR(1) mean-reverting process
  - Demand: Industrial scheduled shift baseline with adaptive bias scaling
  - Outage: Heuristic outage risk progression
============================================================
"""

from typing import Dict, List, Any, Optional
import numpy as np


class BaselineForecaster:
    """
    Physically informed baseline forecaster operating across
    horizons [1, 2, 4, 8, 16] steps (15m to 240m).
    """

    HORIZONS_STEPS: List[int] = [1, 2, 4, 8, 16]
    HORIZONS_MINUTES: List[int] = [15, 30, 60, 120, 240]

    def __init__(
        self,
        dt_hours: float = 0.25,
        horizons: Optional[List[int]] = None,
        wind_mean_mw: float = 0.15,
        wind_decay: float = 0.85,
    ):
        self.dt_hours = dt_hours
        self.horizons = horizons if horizons is not None else list(self.HORIZONS_STEPS)
        self.wind_mean_mw = wind_mean_mw
        self.wind_decay = wind_decay

    def predict(
        self,
        current_hour: float,
        current_solar_mw: float,
        current_wind_mw: float,
        current_demand_mw: float,
        current_crit_demand_mw: float = 0.3,
        current_flex_demand_mw: float = 0.2,
        current_other_demand_mw: float = 0.3,
        known_cloud_factor: float = 1.0,
        storm_warning: bool = False,
        grid_available: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate physically informed multi-horizon baseline predictions.
        """
        n_h = len(self.horizons)
        solar_preds = np.zeros(n_h, dtype=np.float32)
        wind_preds = np.zeros(n_h, dtype=np.float32)
        demand_preds = np.zeros(n_h, dtype=np.float32)
        crit_preds = np.zeros(n_h, dtype=np.float32)
        flex_preds = np.zeros(n_h, dtype=np.float32)
        other_preds = np.zeros(n_h, dtype=np.float32)
        outage_probs = np.zeros(n_h, dtype=np.float32)
        storm_risks = np.zeros(n_h, dtype=np.float32)

        # Baseline demand expectation at current hour for ratio calibration
        def scheduled_demand(h: float) -> float:
            h_mod = h % 24.0
            if 9.0 <= h_mod < 17.0:
                return 0.85
            elif (6.0 <= h_mod < 9.0) or (17.0 <= h_mod < 22.0):
                return 0.78
            else:
                return 0.65

        base_curr_d = scheduled_demand(current_hour)
        d_scale = current_demand_mw / base_curr_d if base_curr_d > 0.01 else 1.0
        d_scale = float(np.clip(d_scale, 0.7, 1.3))

        base_risk = 0.85 if storm_warning else (0.95 if not grid_available else 0.02)

        for i, step_offset in enumerate(self.horizons):
            future_hour = current_hour + step_offset * self.dt_hours
            h_mod = future_hour % 24.0

            # 1. Solar diurnal clear-sky model
            if 6.0 <= h_mod <= 18.0:
                clear_sky = 0.50 * np.sin(np.pi * (h_mod - 6.0) / 12.0)
            else:
                clear_sky = 0.0
            solar_preds[i] = float(np.clip(clear_sky * known_cloud_factor, 0.0, 0.50))

            # 2. Wind AR(1) relaxation toward mean
            alpha = self.wind_decay ** step_offset
            wind_preds[i] = float(np.clip(alpha * current_wind_mw + (1.0 - alpha) * self.wind_mean_mw, 0.0, 0.25))

            # 3. Demand schedule scaled by observed baseline ratio
            raw_d = scheduled_demand(future_hour) * d_scale
            demand_preds[i] = float(np.clip(raw_d, 0.2, 1.2))

            # Demand tiers split proportionally
            crit_ratio = current_crit_demand_mw / current_demand_mw if current_demand_mw > 0 else 0.375
            flex_ratio = current_flex_demand_mw / current_demand_mw if current_demand_mw > 0 else 0.250
            other_ratio = current_other_demand_mw / current_demand_mw if current_demand_mw > 0 else 0.375

            crit_preds[i] = demand_preds[i] * crit_ratio
            flex_preds[i] = demand_preds[i] * flex_ratio
            other_preds[i] = demand_preds[i] * other_ratio

            # 4. Outage and storm risk
            if not grid_available:
                # If currently in outage, probability remains elevated for short horizons then decays
                outage_probs[i] = float(np.clip(0.95 - 0.04 * step_offset, 0.10, 1.0))
            elif storm_warning:
                outage_probs[i] = float(np.clip(0.70 + 0.02 * step_offset, 0.0, 1.0))
            else:
                outage_probs[i] = float(np.clip(0.02 + 0.005 * step_offset, 0.0, 0.20))

            storm_risks[i] = float(np.clip(base_risk + 0.01 * step_offset, 0.0, 1.0))

        net_load_preds = demand_preds - (solar_preds + wind_preds)

        return {
            "horizons_steps": list(self.horizons),
            "horizons_minutes": [h * 15 for h in self.horizons],
            "solar_mw": solar_preds,
            "wind_mw": wind_preds,
            "total_demand_mw": demand_preds,
            "crit_demand_mw": crit_preds,
            "flex_demand_mw": flex_preds,
            "other_demand_mw": other_preds,
            "net_load_mw": net_load_preds,
            "outage_prob": outage_probs,
            "storm_risk": storm_risks,
        }
