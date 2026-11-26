"""
forecasting/persistence_forecaster.py
============================================================
Baseline persistence forecaster for microgrid state variables.
Predicts X_{t+h} = X_t for all lookahead horizons.
Provides a foundational benchmark for all predictive evaluations.
============================================================
"""

from typing import Dict, List, Any, Optional
import numpy as np


class PersistenceForecaster:
    """
    Persistence forecasting model: assumes conditions at future time t+h
    remain identical to the latest observed telemetry at time t.
    """

    HORIZONS_STEPS: List[int] = [1, 2, 4, 8, 16]
    HORIZONS_MINUTES: List[int] = [15, 30, 60, 120, 240]

    def __init__(self, horizons: Optional[List[int]] = None):
        self.horizons = horizons if horizons is not None else list(self.HORIZONS_STEPS)

    def predict(
        self,
        current_solar_mw: float,
        current_wind_mw: float,
        current_demand_mw: float,
        current_crit_demand_mw: float = 0.3,
        current_flex_demand_mw: float = 0.2,
        current_other_demand_mw: float = 0.3,
        current_grid_available: bool = True,
        current_storm_risk: float = 0.02,
    ) -> Dict[str, Any]:
        """
        Generate multi-horizon persistence predictions.

        Returns:
            Dict containing arrays for each target across the horizons.
        """
        n_h = len(self.horizons)
        s_val = max(0.0, float(current_solar_mw))
        w_val = max(0.0, float(current_wind_mw))
        d_val = max(0.0, float(current_demand_mw))
        crit_val = max(0.0, float(current_crit_demand_mw))
        flex_val = max(0.0, float(current_flex_demand_mw))
        other_val = max(0.0, float(current_other_demand_mw))
        net_val = d_val - (s_val + w_val)
        outage_val = 0.0 if current_grid_available else 1.0
        risk_val = float(np.clip(current_storm_risk, 0.0, 1.0))

        return {
            "horizons_steps": list(self.horizons),
            "horizons_minutes": [h * 15 for h in self.horizons],
            "solar_mw": np.full(n_h, s_val, dtype=np.float32),
            "wind_mw": np.full(n_h, w_val, dtype=np.float32),
            "total_demand_mw": np.full(n_h, d_val, dtype=np.float32),
            "crit_demand_mw": np.full(n_h, crit_val, dtype=np.float32),
            "flex_demand_mw": np.full(n_h, flex_val, dtype=np.float32),
            "other_demand_mw": np.full(n_h, other_val, dtype=np.float32),
            "net_load_mw": np.full(n_h, net_val, dtype=np.float32),
            "outage_prob": np.full(n_h, outage_val, dtype=np.float32),
            "storm_risk": np.full(n_h, risk_val, dtype=np.float32),
        }
