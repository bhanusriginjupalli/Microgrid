"""
forecasting/forecast_manager.py
============================================================
Central coordinator for microgrid forecasting in GRIDGUARD AI.
Supports:
  - Mode 1: Oracle Upper Bound (exact simulation lookahead)
  - Mode 2: Realistic Causal Forecaster (ML quantile regression + outage forecaster, zero future leakage)
Extracts specialized forecast features for decentralized actors and the critic.
============================================================
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np

from forecasting.feature_builder import ForecastFeatureBuilder
from forecasting.baseline_forecaster import BaselineForecaster
from forecasting.supervised_forecaster import SupervisedForecaster
from forecasting.outage_forecaster import OutageForecaster
from forecasting.uncertainty import compute_normalized_uncertainty


class ForecastManager:
    """
    Coordinates multi-horizon generation, demand, and disturbance forecasting.
    Provides structured forecast states and agent-specific normalized feature tensors.
    """

    HORIZONS_STEPS: List[int] = [1, 2, 4, 8, 16]
    HORIZONS_MINUTES: List[int] = [15, 30, 60, 120, 240]

    def __init__(
        self,
        mode: str = "realistic",  # "realistic" or "oracle"
        dt_hours: float = 0.25,
        models_dir: Optional[str] = None,
    ):
        self.mode = mode.lower()
        self.dt_hours = dt_hours
        self.horizons = list(self.HORIZONS_STEPS)

        # Core components
        self.feature_builder = ForecastFeatureBuilder(history_window=16, dt_hours=dt_hours)
        self.baseline = BaselineForecaster(dt_hours=dt_hours, horizons=self.horizons)
        self.supervised = SupervisedForecaster(horizons=self.horizons, models_dir=models_dir, dt_hours=dt_hours)
        self.outage = OutageForecaster(horizons=self.horizons, dt_hours=dt_hours, models_dir=models_dir)

        # Cache of latest predictions
        self.latest_forecast: Dict[str, Any] = {}

    def reset(self):
        """Reset historical feature buffer."""
        self.feature_builder.reset()
        self.latest_forecast.clear()

    def update_telemetry(
        self,
        hour: float,
        solar_mw: float,
        wind_mw: float,
        total_demand_mw: float,
        crit_demand_mw: float = 0.3,
        flex_demand_mw: float = 0.2,
        other_demand_mw: float = 0.3,
        cloud_factor: float = 1.0,
        storm_warning: bool = False,
    ):
        """Record current timestep telemetry into causal history buffer."""
        self.feature_builder.update(
            hour=hour,
            solar_mw=solar_mw,
            wind_mw=wind_mw,
            total_demand_mw=total_demand_mw,
            crit_demand_mw=crit_demand_mw,
            flex_demand_mw=flex_demand_mw,
            other_demand_mw=other_demand_mw,
            cloud_factor=cloud_factor,
            storm_warning=storm_warning,
        )

    def generate_forecast(
        self,
        current_hour: float,
        current_state: Dict[str, Any],
        scenario_manager: Optional[Any] = None,
        solar_model: Optional[Any] = None,
        wind_model: Optional[Any] = None,
        load_manager: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Generate multi-horizon forecasts for the current step.

        In 'oracle' mode, future values are extracted directly from the physical models.
        In 'realistic' mode, predictions are produced causally via ML/baseline models.
        """
        if self.mode == "oracle" and scenario_manager is not None:
            self.latest_forecast = self._generate_oracle_forecast(
                current_hour=current_hour,
                scenario_manager=scenario_manager,
                solar_model=solar_model,
                wind_model=wind_model,
                load_manager=load_manager,
            )
        else:
            self.latest_forecast = self._generate_realistic_forecast(
                current_hour=current_hour,
                current_state=current_state,
            )

        return self.latest_forecast

    def _generate_realistic_forecast(
        self,
        current_hour: float,
        current_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate causal, non-leaking ML/baseline multi-horizon predictions."""
        feat = self.feature_builder.build_features(current_hour)
        s_mw = float(current_state.get("solar_available_mw", current_state.get("solar_mw", 0.0)))
        w_mw = float(current_state.get("wind_available_mw", current_state.get("wind_mw", 0.0)))
        d_mw = float(current_state.get("nominal_total_mw", current_state.get("total_demand_mw", 0.8)))
        cloud = float(current_state.get("cloud_factor", 1.0))
        storm_warn = bool(current_state.get("is_storm_warning", False))
        grid_avail = bool(current_state.get("grid_available", True))
        wind_spd = float(current_state.get("wind_speed_ms", 8.0))

        # 1. Continuous energy targets (solar, wind, demand, net_load)
        energy_preds = self.supervised.predict(
            features=feat,
            current_hour=current_hour,
            current_solar_mw=s_mw,
            current_wind_mw=w_mw,
            current_demand_mw=d_mw,
            known_cloud_factor=cloud,
            storm_warning=storm_warn,
            grid_available=grid_avail,
        )

        # 2. Outage & disturbance targets
        outage_preds = self.outage.predict(
            features=feat,
            storm_warning=storm_warn,
            grid_available=grid_avail,
            cloud_factor=cloud,
            wind_speed_ms=wind_spd,
        )

        return {
            "mode": "realistic",
            "horizons_steps": list(self.horizons),
            "horizons_minutes": [h * 15 for h in self.horizons],
            "solar": energy_preds["solar_mw"],
            "wind": energy_preds["wind_mw"],
            "demand": energy_preds["total_demand_mw"],
            "net_load": energy_preds["net_load_mw"],
            "outage_prob": outage_preds["outage_prob"],
            "storm_risk": outage_preds["storm_risk"],
        }

    def _generate_oracle_forecast(
        self,
        current_hour: float,
        scenario_manager: Any,
        solar_model: Optional[Any],
        wind_model: Optional[Any],
        load_manager: Optional[Any],
    ) -> Dict[str, Any]:
        """Query future simulation states directly as the theoretical upper bound."""
        n_h = len(self.horizons)
        s_pts = np.zeros(n_h, dtype=np.float32)
        w_pts = np.zeros(n_h, dtype=np.float32)
        d_pts = np.zeros(n_h, dtype=np.float32)
        nl_pts = np.zeros(n_h, dtype=np.float32)
        outage_pts = np.zeros(n_h, dtype=np.float32)
        risk_pts = np.zeros(n_h, dtype=np.float32)

        for i, h in enumerate(self.horizons):
            future_h = current_hour + h * self.dt_hours
            # Outage
            grid_avail = scenario_manager.is_grid_available(future_h)
            outage_pts[i] = 0.0 if grid_avail else 1.0
            storm_warn = scenario_manager.is_storm_warning(future_h)
            risk_pts[i] = 1.0 if not grid_avail else (0.85 if storm_warn else 0.02)

            # Cloud & Solar
            cf = scenario_manager.get_cloud_factor(future_h)
            h_mod = future_h % 24.0
            if 6.0 <= h_mod <= 18.0:
                irr = np.sin(np.pi * (h_mod - 6.0) / 12.0) * cf
            else:
                irr = 0.0
            if solar_model is not None:
                s_pts[i] = solar_model.update(irr, ambient_temp_c=28.0)
            else:
                s_pts[i] = max(0.0, 0.50 * irr)

            # Wind
            w_override = scenario_manager.get_wind_override(future_h)
            w_speed = w_override if w_override is not None else 8.0
            if wind_model is not None:
                w_pts[i] = wind_model.update(w_speed)
            else:
                w_pts[i] = 0.0 if w_speed >= 25.0 else 0.15

            # Demand
            if load_manager is not None:
                d_nom = load_manager.get_nominal_demand(future_h)
                d_pts[i] = d_nom["total_demand_mw"]
            else:
                h_mod = future_h % 24.0
                d_pts[i] = 0.85 if 9.0 <= h_mod < 17.0 else (0.78 if 6.0 <= h_mod < 22.0 else 0.65)

            nl_pts[i] = d_pts[i] - (s_pts[i] + w_pts[i])

        zeros = np.zeros(n_h, dtype=np.float32)

        return {
            "mode": "oracle",
            "horizons_steps": list(self.horizons),
            "horizons_minutes": [h * 15 for h in self.horizons],
            "solar": {"point": s_pts, "q10": s_pts, "q90": s_pts, "uncertainty": zeros},
            "wind": {"point": w_pts, "q10": w_pts, "q90": w_pts, "uncertainty": zeros},
            "demand": {"point": d_pts, "q10": d_pts, "q90": d_pts, "uncertainty": zeros},
            "net_load": {"point": nl_pts, "q10": nl_pts, "q90": nl_pts, "uncertainty": zeros},
            "outage_prob": outage_pts,
            "storm_risk": risk_pts,
        }

    def get_agent_forecast_features(self, agent: str) -> np.ndarray:
        """
        Extract normalized forecast features tailored for a specific agent:
          - renewable (8 feats): solar (30m, 1h, 4h), wind (30m, 1h, 4h), solar_unc_1h, wind_unc_1h
          - battery (9 feats): net_load (15m, 30m, 1h, 2h, 4h), outage_prob (15m, 1h, 4h), nl_unc_1h
          - generator (8 feats): net_load (15m, 30m, 1h, 2h), outage_prob (15m, 30m, 1h, 2h)
          - load_management (6 feats): net_load (15m, 1h, 4h), outage_prob (15m, 1h, 4h)
          - grid_resilience (7 feats): outage_prob (15m, 30m, 1h, 2h, 4h), storm_risk (1h, 4h)
        """
        if not self.latest_forecast:
            # Fallback zero vector of correct shape
            dummy_shapes = {
                "renewable": 8,
                "battery": 9,
                "generator": 8,
                "load_management": 6,
                "grid_resilience": 7,
            }
            return np.zeros(dummy_shapes.get(agent, 0), dtype=np.float32)

        fc = self.latest_forecast
        # Index map for horizons [1, 2, 4, 8, 16] -> [0: 15m, 1: 30m, 2: 1h, 3: 2h, 4: 4h]
        s_pts = fc["solar"]["point"] / 0.50
        w_pts = fc["wind"]["point"] / 0.25
        nl_pts = fc["net_load"]["point"] / 0.80
        outage = fc["outage_prob"]
        risk = fc["storm_risk"]
        s_unc = fc["solar"]["uncertainty"]
        w_unc = fc["wind"]["uncertainty"]
        nl_unc = fc["net_load"]["uncertainty"]

        if agent == "renewable":
            return np.array([
                s_pts[1], s_pts[2], s_pts[4],  # 30m, 1h, 4h
                w_pts[1], w_pts[2], w_pts[4],  # 30m, 1h, 4h
                s_unc[2], w_unc[2],            # 1h uncertainty
            ], dtype=np.float32)

        elif agent == "battery":
            return np.array([
                nl_pts[0], nl_pts[1], nl_pts[2], nl_pts[3], nl_pts[4],  # 15m, 30m, 1h, 2h, 4h
                outage[0], outage[2], outage[4],                        # 15m, 1h, 4h
                nl_unc[2],                                              # 1h net load uncertainty
            ], dtype=np.float32)

        elif agent == "generator":
            return np.array([
                nl_pts[0], nl_pts[1], nl_pts[2], nl_pts[3],  # 15m, 30m, 1h, 2h
                outage[0], outage[1], outage[2], outage[3],  # 15m, 30m, 1h, 2h
            ], dtype=np.float32)

        elif agent == "load_management":
            return np.array([
                nl_pts[0], nl_pts[2], nl_pts[4],  # 15m, 1h, 4h
                outage[0], outage[2], outage[4],  # 15m, 1h, 4h
            ], dtype=np.float32)

        elif agent == "grid_resilience":
            return np.array([
                outage[0], outage[1], outage[2], outage[3], outage[4],  # 15m, 30m, 1h, 2h, 4h
                risk[2], risk[4],                                      # 1h, 4h storm risk
            ], dtype=np.float32)

        return np.array([], dtype=np.float32)

    def get_critic_forecast_features(self) -> np.ndarray:
        """
        Extract global forecast features for the centralized critic (19 features):
          - solar: 30m, 1h, 4h (3)
          - wind: 30m, 1h, 4h (3)
          - demand: 30m, 1h, 4h (3)
          - net_load: 15m, 30m, 1h, 2h, 4h (5)
          - outage_prob: 15m, 30m, 1h, 2h, 4h (5)
        """
        if not self.latest_forecast:
            return np.zeros(19, dtype=np.float32)

        fc = self.latest_forecast
        s = fc["solar"]["point"] / 0.50
        w = fc["wind"]["point"] / 0.25
        d = fc["demand"]["point"] / 0.80
        nl = fc["net_load"]["point"] / 0.80
        outage = fc["outage_prob"]

        return np.array([
            s[1], s[2], s[4],
            w[1], w[2], w[4],
            d[1], d[2], d[4],
            nl[0], nl[1], nl[2], nl[3], nl[4],
            outage[0], outage[1], outage[2], outage[3], outage[4],
        ], dtype=np.float32)
