"""
forecasting/supervised_forecaster.py
============================================================
Multi-Horizon Supervised Quantile Forecaster.
Trains and evaluates LightGBM quantile regression models across horizons
[1, 2, 4, 8, 16] (15m to 240m) for:
  - Solar PV Generation (kW/MW)
  - Wind Turbine Generation (kW/MW)
  - Campus Demand (kW/MW)
  - Campus Net Load (kW/MW)
Generates q10, q50, and q90 with monotonicity guarantees and physical bounds.
============================================================
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from forecasting.baseline_forecaster import BaselineForecaster


class SupervisedForecaster:
    """
    Multi-horizon quantile forecasting engine utilizing gradient boosted
    decision trees (LightGBM) to forecast future energy states with
    calibrated 80% prediction intervals [q10, q90].
    """

    HORIZONS_STEPS: List[int] = [1, 2, 4, 8, 16]
    HORIZONS_MINUTES: List[int] = [15, 30, 60, 120, 240]
    TARGETS: List[str] = ["solar_mw", "wind_mw", "total_demand_mw", "net_load_mw"]
    QUANTILES: List[float] = [0.10, 0.50, 0.90]

    def __init__(
        self,
        horizons: Optional[List[int]] = None,
        models_dir: Optional[str] = None,
        dt_hours: float = 0.25,
    ):
        self.horizons = horizons if horizons is not None else list(self.HORIZONS_STEPS)
        self.models_dir = Path(models_dir) if models_dir is not None else None
        self.dt_hours = dt_hours
        self.baseline = BaselineForecaster(dt_hours=dt_hours, horizons=self.horizons)

        # Storage for trained models: Dict[target, Dict[horizon_step, Dict[quantile, lgb.Booster]]]
        self.models: Dict[str, Dict[int, Dict[float, Any]]] = {t: {} for t in self.TARGETS}
        self.is_fitted = False

        if self.models_dir and self.models_dir.exists():
            self.load(self.models_dir)

    def fit(
        self,
        X: np.ndarray,
        y_targets: Dict[str, Dict[int, np.ndarray]],
        n_estimators: int = 60,
        learning_rate: float = 0.08,
        num_leaves: int = 15,
        verbose: int = -1,
    ):
        """
        Train multi-horizon quantile regression models.

        Args:
            X: 2D feature matrix (N, num_features) from ForecastFeatureBuilder
            y_targets: Dict mapping target name -> Dict[horizon_step -> 1D array of target values]
        """
        if not HAS_LIGHTGBM:
            raise RuntimeError("lightgbm is required to fit SupervisedForecaster")

        for target in self.TARGETS:
            if target not in y_targets:
                continue
            self.models[target] = {}
            for h in self.horizons:
                if h not in y_targets[target]:
                    continue
                y = y_targets[target][h]
                self.models[target][h] = {}

                for q in self.QUANTILES:
                    params = {
                        "objective": "quantile",
                        "alpha": q,
                        "metric": "quantile",
                        "learning_rate": learning_rate,
                        "num_leaves": num_leaves,
                        "n_estimators": n_estimators,
                        "min_child_samples": 8,
                        "verbose": verbose,
                        "random_state": 42,
                    }
                    train_data = lgb.Dataset(X, label=y)
                    gbm = lgb.train(
                        params,
                        train_data,
                        num_boost_round=n_estimators,
                    )
                    self.models[target][h][q] = gbm

        self.is_fitted = True

    def predict(
        self,
        features: np.ndarray,
        current_hour: float,
        current_solar_mw: float,
        current_wind_mw: float,
        current_demand_mw: float,
        known_cloud_factor: float = 1.0,
        storm_warning: bool = False,
        grid_available: bool = True,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Predict point forecasts and prediction intervals across all horizons.

        Returns:
            Dict[target -> Dict['point': np.ndarray, 'q10': np.ndarray, 'q90': np.ndarray, 'uncertainty': np.ndarray]]
        """
        feat_vec = np.asarray(features, dtype=np.float32).reshape(1, -1)
        n_h = len(self.horizons)
        results: Dict[str, Dict[str, np.ndarray]] = {}

        # If not fitted, use physical baseline with calibrated uncertainty
        if not self.is_fitted:
            base_preds = self.baseline.predict(
                current_hour=current_hour,
                current_solar_mw=current_solar_mw,
                current_wind_mw=current_wind_mw,
                current_demand_mw=current_demand_mw,
                known_cloud_factor=known_cloud_factor,
                storm_warning=storm_warning,
                grid_available=grid_available,
            )

            for target, cap in [
                ("solar_mw", 0.50),
                ("wind_mw", 0.25),
                ("total_demand_mw", 0.80),
                ("net_load_mw", 0.80),
            ]:
                pt = base_preds[target]
                # Uncertainty grows with horizon step
                sigma_growth = np.array([0.05 + 0.02 * (h ** 0.5) for h in self.horizons], dtype=np.float32) * cap
                q10 = np.maximum(0.0 if "mw" in target and target != "net_load_mw" else -cap, pt - 1.28 * sigma_growth)
                q90 = pt + 1.28 * sigma_growth
                if target == "solar_mw":
                    q90 = np.clip(q90, 0.0, 0.50)
                elif target == "wind_mw":
                    q90 = np.clip(q90, 0.0, 0.25)

                results[target] = {
                    "point": pt.astype(np.float32),
                    "q10": q10.astype(np.float32),
                    "q90": q90.astype(np.float32),
                    "uncertainty": np.clip((q90 - q10) / cap, 0.0, 1.0).astype(np.float32),
                }
            return results

        # Predict using fitted LightGBM models
        for target in self.TARGETS:
            pts = np.zeros(n_h, dtype=np.float32)
            q10s = np.zeros(n_h, dtype=np.float32)
            q90s = np.zeros(n_h, dtype=np.float32)
            cap = 0.50 if target == "solar_mw" else (0.25 if target == "wind_mw" else 0.80)

            for i, h in enumerate(self.horizons):
                if h in self.models[target] and 0.50 in self.models[target][h]:
                    p = float(self.models[target][h][0.50].predict(feat_vec)[0])
                    q1 = float(self.models[target][h][0.10].predict(feat_vec)[0])
                    q9 = float(self.models[target][h][0.90].predict(feat_vec)[0])
                else:
                    # Horizon fallback
                    p = current_solar_mw if target == "solar_mw" else (current_wind_mw if target == "wind_mw" else current_demand_mw)
                    q1 = p * 0.85
                    q9 = p * 1.15

                # Enforce physical bounds & monotonicity
                if target in ("solar_mw", "wind_mw", "total_demand_mw"):
                    q1 = max(0.0, q1)
                    p = max(0.0, p)
                    q9 = max(0.0, q9)
                if target == "solar_mw":
                    h_mod = (current_hour + h * self.dt_hours) % 24.0
                    if not (6.0 <= h_mod <= 18.0):
                        q1 = p = q9 = 0.0

                # Quantile monotonicity guarantee: q10 <= q50 <= q90
                q1 = min(q1, p)
                q9 = max(q9, p)

                q10s[i] = q1
                pts[i] = p
                q90s[i] = q9

            unc = np.clip((q90s - q10s) / cap, 0.0, 1.0)
            results[target] = {
                "point": pts,
                "q10": q10s,
                "q90": q90s,
                "uncertainty": unc,
            }

        return results

    def save(self, directory: Path):
        """Serialize trained models to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        meta = {
            "horizons": self.horizons,
            "targets": self.TARGETS,
            "quantiles": self.QUANTILES,
            "dt_hours": self.dt_hours,
        }
        with open(directory / "supervised_forecaster_meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        for target in self.TARGETS:
            for h in self.horizons:
                for q in self.QUANTILES:
                    if h in self.models[target] and q in self.models[target][h]:
                        model_file = directory / f"lgb_{target}_h{h}_q{int(q*100)}.txt"
                        self.models[target][h][q].save_model(str(model_file))

    def load(self, directory: Path):
        """Load trained models from disk."""
        directory = Path(directory)
        meta_file = directory / "supervised_forecaster_meta.pkl"
        if not meta_file.exists():
            return
        with open(meta_file, "rb") as f:
            meta = pickle.load(f)
        self.horizons = meta.get("horizons", self.HORIZONS_STEPS)

        if not HAS_LIGHTGBM:
            return

        loaded_any = False
        for target in self.TARGETS:
            self.models[target] = {}
            for h in self.horizons:
                self.models[target][h] = {}
                for q in self.QUANTILES:
                    model_file = directory / f"lgb_{target}_h{h}_q{int(q*100)}.txt"
                    if model_file.exists():
                        gbm = lgb.Booster(model_file=str(model_file))
                        self.models[target][h][q] = gbm
                        loaded_any = True

        self.is_fitted = loaded_any
