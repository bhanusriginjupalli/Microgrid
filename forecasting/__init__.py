"""
forecasting package
============================================================
Multi-Horizon Predictive Energy Management & Disturbance Forecaster
for GRIDGUARD AI Phase 4.2.
============================================================
"""

from forecasting.feature_builder import ForecastFeatureBuilder
from forecasting.persistence_forecaster import PersistenceForecaster
from forecasting.baseline_forecaster import BaselineForecaster
from forecasting.supervised_forecaster import SupervisedForecaster
from forecasting.outage_forecaster import OutageForecaster
from forecasting.uncertainty import (
    compute_picp,
    compute_mpiw,
    compute_pinball_loss,
    compute_normalized_uncertainty,
    evaluate_interval_quality,
)
from forecasting.forecast_manager import ForecastManager

__all__ = [
    "ForecastFeatureBuilder",
    "PersistenceForecaster",
    "BaselineForecaster",
    "SupervisedForecaster",
    "OutageForecaster",
    "ForecastManager",
    "compute_picp",
    "compute_mpiw",
    "compute_pinball_loss",
    "compute_normalized_uncertainty",
    "evaluate_interval_quality",
]
