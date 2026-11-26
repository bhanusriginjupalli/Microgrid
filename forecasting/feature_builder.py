"""
forecasting/feature_builder.py
============================================================
Builds causal, non-leaking feature vectors from rolling historical observations
for multi-horizon microgrid forecasting.
============================================================
"""

from typing import Dict, Any, List, Optional
import numpy as np


class ForecastFeatureBuilder:
    """
    Constructs temporal, lagged, and rolling statistical features
    from historical telemetry strictly without future data leakage.
    """

    def __init__(self, history_window: int = 16, dt_hours: float = 0.25):
        self.history_window = history_window
        self.dt_hours = dt_hours

        # Historical buffers
        self.solar_history: List[float] = []
        self.wind_history: List[float] = []
        self.total_demand_history: List[float] = []
        self.crit_demand_history: List[float] = []
        self.flex_demand_history: List[float] = []
        self.other_demand_history: List[float] = []
        self.cloud_factor_history: List[float] = []
        self.storm_warning_history: List[float] = []
        self.hour_history: List[float] = []

    def reset(self):
        """Clear all rolling buffers."""
        self.solar_history.clear()
        self.wind_history.clear()
        self.total_demand_history.clear()
        self.crit_demand_history.clear()
        self.flex_demand_history.clear()
        self.other_demand_history.clear()
        self.cloud_factor_history.clear()
        self.storm_warning_history.clear()
        self.hour_history.clear()

    def update(
        self,
        hour: float,
        solar_mw: float,
        wind_mw: float,
        total_demand_mw: float,
        crit_demand_mw: float,
        flex_demand_mw: float,
        other_demand_mw: float,
        cloud_factor: float = 1.0,
        storm_warning: bool = False,
    ):
        """Append latest ground truth observation at current timestep."""
        self.hour_history.append(float(hour))
        self.solar_history.append(float(solar_mw))
        self.wind_history.append(float(wind_mw))
        self.total_demand_history.append(float(total_demand_mw))
        self.crit_demand_history.append(float(crit_demand_mw))
        self.flex_demand_history.append(float(flex_demand_mw))
        self.other_demand_history.append(float(other_demand_mw))
        self.cloud_factor_history.append(float(cloud_factor))
        self.storm_warning_history.append(1.0 if storm_warning else 0.0)

        # Enforce rolling window bound
        max_len = self.history_window * 4
        if len(self.solar_history) > max_len:
            self.hour_history = self.hour_history[-max_len:]
            self.solar_history = self.solar_history[-max_len:]
            self.wind_history = self.wind_history[-max_len:]
            self.total_demand_history = self.total_demand_history[-max_len:]
            self.crit_demand_history = self.crit_demand_history[-max_len:]
            self.flex_demand_history = self.flex_demand_history[-max_len:]
            self.other_demand_history = self.other_demand_history[-max_len:]
            self.cloud_factor_history = self.cloud_factor_history[-max_len:]
            self.storm_warning_history = self.storm_warning_history[-max_len:]

    def build_features(self, current_hour: float) -> np.ndarray:
        """
        Extract feature vector for tabular/ML forecasting models.
        Features:
          0: hour_sin
          1: hour_cos
          2: is_daylight (6.0 <= hour <= 18.0)
          3: solar_t (latest known)
          4: solar_lag1
          5: solar_lag2
          6: solar_rolling_mean_4
          7: wind_t
          8: wind_lag1
          9: wind_lag2
          10: wind_rolling_mean_4
          11: wind_rolling_std_4
          12: demand_t
          13: demand_lag1
          14: demand_lag2
          15: demand_rolling_mean_4
          16: net_load_t (demand - solar - wind)
          17: cloud_factor_t
          18: storm_warning_t
          19: storm_warning_rolling_4
        """
        h_mod = current_hour % 24.0
        h_rad = 2.0 * np.pi * h_mod / 24.0
        h_sin = float(np.sin(h_rad))
        h_cos = float(np.cos(h_rad))
        is_day = 1.0 if 6.0 <= h_mod <= 18.0 else 0.0

        # Helper to get lagged value or repeat current
        def get_val(arr: List[float], lag: int, default: float) -> float:
            if len(arr) > lag:
                return arr[-(lag + 1)]
            elif arr:
                return arr[0]
            return default

        # Helper for rolling mean
        def get_roll_mean(arr: List[float], w: int, default: float) -> float:
            if not arr:
                return default
            sub = arr[-w:] if len(arr) >= w else arr
            return float(np.mean(sub))

        # Helper for rolling std
        def get_roll_std(arr: List[float], w: int, default: float) -> float:
            if len(arr) < 2:
                return 0.0
            sub = arr[-w:] if len(arr) >= w else arr
            return float(np.std(sub))

        s_t = get_val(self.solar_history, 0, 0.0)
        s_l1 = get_val(self.solar_history, 1, s_t)
        s_l2 = get_val(self.solar_history, 2, s_l1)
        s_rm4 = get_roll_mean(self.solar_history, 4, s_t)

        w_t = get_val(self.wind_history, 0, 0.15)
        w_l1 = get_val(self.wind_history, 1, w_t)
        w_l2 = get_val(self.wind_history, 2, w_l1)
        w_rm4 = get_roll_mean(self.wind_history, 4, w_t)
        w_rs4 = get_roll_std(self.wind_history, 4, 0.02)

        d_t = get_val(self.total_demand_history, 0, 0.8)
        d_l1 = get_val(self.total_demand_history, 1, d_t)
        d_l2 = get_val(self.total_demand_history, 2, d_l1)
        d_rm4 = get_roll_mean(self.total_demand_history, 4, d_t)

        net_load_t = d_t - (s_t + w_t)
        cloud_t = get_val(self.cloud_factor_history, 0, 1.0)
        storm_t = get_val(self.storm_warning_history, 0, 0.0)
        storm_rm4 = get_roll_mean(self.storm_warning_history, 4, storm_t)

        feat = np.array([
            h_sin, h_cos, is_day,
            s_t, s_l1, s_l2, s_rm4,
            w_t, w_l1, w_l2, w_rm4, w_rs4,
            d_t, d_l1, d_l2, d_rm4,
            net_load_t, cloud_t, storm_t, storm_rm4,
        ], dtype=np.float32)

        return feat
