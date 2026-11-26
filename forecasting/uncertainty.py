"""
forecasting/uncertainty.py
============================================================
Uncertainty quantification, prediction intervals, and reliability metrics:
  - Prediction Interval Coverage Probability (PICP)
  - Mean Prediction Interval Width (MPIW)
  - Quantile / Pinball loss
  - Normalized Uncertainty Indices for RL observations
============================================================
"""

from typing import Dict, Any, Union
import numpy as np


def compute_picp(
    y_true: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
) -> float:
    """
    Prediction Interval Coverage Probability (PICP).
    Nominal target for [q10, q90] is 0.80 (80%).
    """
    y_t = np.asarray(y_true).flatten()
    ql = np.asarray(q_low).flatten()
    qh = np.asarray(q_high).flatten()
    if len(y_t) == 0:
        return 0.0
    covered = (y_t >= ql) & (y_t <= qh)
    return float(np.mean(covered))


def compute_mpiw(
    q_low: np.ndarray,
    q_high: np.ndarray,
) -> float:
    """
    Mean Prediction Interval Width (MPIW).
    Measures interval sharpness.
    """
    ql = np.asarray(q_low).flatten()
    qh = np.asarray(q_high).flatten()
    if len(ql) == 0:
        return 0.0
    widths = np.maximum(0.0, qh - ql)
    return float(np.mean(widths))


def compute_pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float,
) -> float:
    """
    Quantile / Pinball loss for evaluation of quantile predictions:
      L_alpha(y, y_hat) = max(alpha * (y - y_hat), (alpha - 1) * (y - y_hat))
    """
    y_t = np.asarray(y_true).flatten()
    y_p = np.asarray(y_pred).flatten()
    diff = y_t - y_p
    loss = np.maximum(quantile * diff, (quantile - 1.0) * diff)
    return float(np.mean(loss))


def compute_normalized_uncertainty(
    q_low: Union[float, np.ndarray],
    q_high: Union[float, np.ndarray],
    capacity: float = 1.0,
) -> Union[float, np.ndarray]:
    """
    Normalized uncertainty metric clamped in [0, 1] for RL agent state feeds:
      unc_norm = (q_high - q_low) / capacity
    """
    ql = np.asarray(q_low)
    qh = np.asarray(q_high)
    scale = max(1e-4, capacity)
    diff = np.maximum(0.0, qh - ql) / scale
    return np.clip(diff, 0.0, 1.0)


def evaluate_interval_quality(
    y_true: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    capacity: float = 1.0,
) -> Dict[str, float]:
    """
    Compute comprehensive interval diagnostic summary:
      - picp: Coverage of 80% interval
      - mpiw: Mean width in physical units (MW)
      - nmpiw: Normalized mean width relative to capacity
      - pinball_10, pinball_50, pinball_90: Quantile losses
      - mae: Mean Absolute Error of median forecast (q50)
      - rmse: Root Mean Square Error of median forecast (q50)
    """
    y_t = np.asarray(y_true).flatten()
    q10_arr = np.asarray(q10).flatten()
    q50_arr = np.asarray(q50).flatten()
    q90_arr = np.asarray(q90).flatten()

    picp = compute_picp(y_t, q10_arr, q90_arr)
    mpiw = compute_mpiw(q10_arr, q90_arr)
    nmpiw = mpiw / max(1e-4, capacity)

    mae = float(np.mean(np.abs(y_t - q50_arr)))
    rmse = float(np.sqrt(np.mean((y_t - q50_arr) ** 2)))
    nrmse = rmse / max(1e-4, capacity)

    p10 = compute_pinball_loss(y_t, q10_arr, 0.10)
    p50 = compute_pinball_loss(y_t, q50_arr, 0.50)
    p90 = compute_pinball_loss(y_t, q90_arr, 0.90)

    return {
        "picp": picp,
        "mpiw": mpiw,
        "nmpiw": nmpiw,
        "mae": mae,
        "rmse": rmse,
        "nrmse": nrmse,
        "pinball_10": p10,
        "pinball_50": p50,
        "pinball_90": p90,
    }
