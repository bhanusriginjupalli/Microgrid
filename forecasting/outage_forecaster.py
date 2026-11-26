"""
forecasting/outage_forecaster.py
============================================================
Probabilistic Outage and Disturbance Forecaster.
Predicts grid outage probability P(outage_{t+h}) and storm risk
across horizons [1, 2, 4, 8, 16] using logistic / classification
models conditioned on storm warning alerts, weather severity, and grid status.
============================================================
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import pickle
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class OutageForecaster:
    """
    Predicts the probability of grid disconnection and storm disturbance
    at multiple lookahead horizons.
    """

    HORIZONS_STEPS: List[int] = [1, 2, 4, 8, 16]
    HORIZONS_MINUTES: List[int] = [15, 30, 60, 120, 240]

    def __init__(
        self,
        horizons: Optional[List[int]] = None,
        dt_hours: float = 0.25,
        models_dir: Optional[str] = None,
    ):
        self.horizons = horizons if horizons is not None else list(self.HORIZONS_STEPS)
        self.dt_hours = dt_hours
        self.models_dir = Path(models_dir) if models_dir is not None else None
        self.classifiers: Dict[int, Any] = {}
        self.is_fitted = False

        if self.models_dir and self.models_dir.exists():
            self.load(self.models_dir)

    def fit(self, X: np.ndarray, y_outages: Dict[int, np.ndarray]):
        """
        Train binary logistic regressors for each lookahead horizon.
        """
        if not HAS_SKLEARN:
            return

        for h in self.horizons:
            if h not in y_outages:
                continue
            y = np.asarray(y_outages[h], dtype=int)
            # If all one class, skip fitting to avoid sklearn error
            if len(np.unique(y)) < 2:
                continue
            clf = LogisticRegression(class_weight="balanced", max_iter=200, random_state=42)
            clf.fit(X, y)
            self.classifiers[h] = clf

        self.is_fitted = len(self.classifiers) > 0

    def predict(
        self,
        features: np.ndarray,
        storm_warning: bool = False,
        grid_available: bool = True,
        cloud_factor: float = 1.0,
        wind_speed_ms: float = 8.0,
    ) -> Dict[str, np.ndarray]:
        """
        Predict outage probability and storm risk across all horizons.

        Returns:
            Dict containing:
              - 'outage_prob': np.ndarray of shape (len(horizons),) in [0, 1]
              - 'storm_risk': np.ndarray of shape (len(horizons),) in [0, 1]
        """
        n_h = len(self.horizons)
        probs = np.zeros(n_h, dtype=np.float32)
        risks = np.zeros(n_h, dtype=np.float32)

        feat_vec = np.asarray(features, dtype=np.float32).copy().reshape(1, -1)
        if feat_vec.shape[1] >= 20:
            if storm_warning:
                feat_vec[0, 18] = 1.0
                feat_vec[0, 19] = max(float(feat_vec[0, 19]), 0.25)
            if cloud_factor < 1.0:
                feat_vec[0, 17] = float(cloud_factor)

        # Baseline heuristic probabilities
        base_risk = 0.85 if storm_warning else (0.95 if not grid_available else 0.02)
        weather_stress = max(0.0, (1.0 - cloud_factor)) * 0.4 + max(0.0, (wind_speed_ms - 15.0) / 15.0) * 0.4

        for i, h in enumerate(self.horizons):
            if self.is_fitted and h in self.classifiers:
                # Class 1 is outage
                p_outage = float(self.classifiers[h].predict_proba(feat_vec)[0, 1])
            else:
                # Calibrated physical logic
                if not grid_available:
                    # Current outage: decays as horizon increases (grid recovery expected in 2-4h)
                    p_outage = max(0.15, 0.95 - 0.04 * h)
                elif storm_warning:
                    # Storm warning: outage probability increases sharply with horizon
                    p_outage = min(0.92, 0.40 + 0.035 * h + weather_stress)
                else:
                    # Normal conditions: small baseline failure probability
                    p_outage = min(0.15, 0.02 + 0.004 * h + weather_stress * 0.2)

            probs[i] = float(np.clip(p_outage, 0.0, 1.0))
            risks[i] = float(np.clip(base_risk + 0.01 * h + weather_stress * 0.5, 0.0, 1.0))

        return {
            "outage_prob": probs,
            "storm_risk": risks,
        }

    def save(self, directory: Path):
        """Save outage classifiers to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "outage_forecaster.pkl", "wb") as f:
            pickle.dump({
                "horizons": self.horizons,
                "classifiers": self.classifiers,
                "is_fitted": self.is_fitted,
            }, f)

    def load(self, directory: Path):
        """Load outage classifiers from disk."""
        file_path = Path(directory) / "outage_forecaster.pkl"
        if not file_path.exists():
            return
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self.horizons = data.get("horizons", self.HORIZONS_STEPS)
        self.classifiers = data.get("classifiers", {})
        self.is_fitted = data.get("is_fitted", False)
