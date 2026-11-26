"""
forecasting/evaluation.py
============================================================
Comprehensive evaluation of multi-horizon microgrid forecasting models:
  1. Data generation & training of SupervisedForecaster and OutageForecaster
  2. Model comparison: Persistence vs Baseline vs Supervised
  3. Metric evaluation: MAE, RMSE, NRMSE, Brier score, PICP, MPIW
  4. Benchmark testing across Scenarios A through F
============================================================
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

# Path configuration
_ROOT_DIR = Path(__file__).resolve().parent.parent
_PHASE1_DIR = _ROOT_DIR / "phase1_digital_twin"
for p in [str(_ROOT_DIR), str(_PHASE1_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from forecasting.feature_builder import ForecastFeatureBuilder
from forecasting.persistence_forecaster import PersistenceForecaster
from forecasting.baseline_forecaster import BaselineForecaster
from forecasting.supervised_forecaster import SupervisedForecaster
from forecasting.outage_forecaster import OutageForecaster
from forecasting.uncertainty import compute_picp, compute_mpiw, evaluate_interval_quality
from rl_environment.scenario_sampler import ScenarioSampler
from resources import SolarModel, WindModel
from load_management import LoadManager


def collect_training_dataset(
    num_episodes: int = 50,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, Dict[int, np.ndarray]], Dict[int, np.ndarray]]:
    """
    Simulate randomized microgrid days to assemble feature matrix X
    and multi-horizon target dicts for supervised training.
    """
    sampler = ScenarioSampler(seed=seed)
    feature_builder = ForecastFeatureBuilder(history_window=16, dt_hours=0.25)
    solar_model = SolarModel()
    wind_model = WindModel()
    load_manager = LoadManager()

    horizons = [1, 2, 4, 8, 16]
    dt_hours = 0.25

    all_features: List[np.ndarray] = []
    target_data: Dict[str, Dict[int, List[float]]] = {
        "solar_mw": {h: [] for h in horizons},
        "wind_mw": {h: [] for h in horizons},
        "total_demand_mw": {h: [] for h in horizons},
        "net_load_mw": {h: [] for h in horizons},
    }
    outage_data: Dict[int, List[int]] = {h: [] for h in horizons}

    for ep in range(num_episodes):
        diff = (ep % 5) + 1
        sampler.set_difficulty_level(diff)
        sm, meta = sampler.sample(randomize=True)
        feature_builder.reset()

        hours_axis = np.linspace(0, 24.0, 96, endpoint=False)
        base_irr = solar_model.generate_daily_profile(hours_axis) * float(meta.get("solar_scale", 1.0))
        base_wind = wind_model.generate_daily_profile(hours_axis, mean_speed=float(meta.get("wind_mean_speed", 8.0)))

        # Precompute episode 96 steps ground truth
        ep_states = []
        for step in range(96):
            h = step * dt_hours
            cf = sm.get_cloud_factor(h)
            irr = base_irr[step] * cf
            w_ovr = sm.get_wind_override(h)
            w_spd = w_ovr if w_ovr is not None else base_wind[step]
            grid_avail = sm.is_grid_available(h)
            storm_warn = sm.is_storm_warning(h)

            s_p = solar_model.update(irr, ambient_temp_c=28.0)
            w_p = wind_model.update(w_spd)
            d_dict = load_manager.get_nominal_demand(h)
            d_p = d_dict["total_demand_mw"] * meta.get("demand_scale", 1.0)
            crit_p = d_dict["critical_mw"] * meta.get("demand_scale", 1.0)
            flex_p = d_dict["flexible_mw"] * meta.get("demand_scale", 1.0)
            other_p = d_dict["other_mw"] * meta.get("demand_scale", 1.0)
            nl_p = d_p - (s_p + w_p)

            ep_states.append({
                "hour": h,
                "solar_mw": s_p,
                "wind_mw": w_p,
                "demand_mw": d_p,
                "crit_mw": crit_p,
                "flex_mw": flex_p,
                "other_mw": other_p,
                "net_load_mw": nl_p,
                "grid_available": grid_avail,
                "storm_warning": storm_warn,
                "cloud_factor": cf,
            })

        # Assemble causal feature vectors & future horizon targets
        for step in range(96 - max(horizons)):
            cur = ep_states[step]
            feature_builder.update(
                hour=cur["hour"],
                solar_mw=cur["solar_mw"],
                wind_mw=cur["wind_mw"],
                total_demand_mw=cur["demand_mw"],
                crit_demand_mw=cur["crit_mw"],
                flex_demand_mw=cur["flex_mw"],
                other_demand_mw=cur["other_mw"],
                cloud_factor=cur["cloud_factor"],
                storm_warning=cur["storm_warning"],
            )
            feat = feature_builder.build_features(cur["hour"])
            all_features.append(feat)

            for h in horizons:
                fut = ep_states[step + h]
                target_data["solar_mw"][h].append(fut["solar_mw"])
                target_data["wind_mw"][h].append(fut["wind_mw"])
                target_data["total_demand_mw"][h].append(fut["demand_mw"])
                target_data["net_load_mw"][h].append(fut["net_load_mw"])
                outage_data[h].append(0 if fut["grid_available"] else 1)

    X = np.array(all_features, dtype=np.float32)
    y_dict = {
        t: {h: np.array(vals, dtype=np.float32) for h, vals in target_data[t].items()}
        for t in target_data
    }
    y_outage = {h: np.array(vals, dtype=int) for h, vals in outage_data.items()}

    return X, y_dict, y_outage


def train_forecasting_models(
    models_dir: Path,
    num_episodes: int = 60,
    seed: int = 42,
) -> Tuple[SupervisedForecaster, OutageForecaster]:
    """Train and serialize SupervisedForecaster and OutageForecaster."""
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"Collecting training data from {num_episodes} procedural episodes...")
    X, y_targets, y_outage = collect_training_dataset(num_episodes=num_episodes, seed=seed)
    print(f"Dataset shape: X={X.shape}, targets={list(y_targets.keys())}")

    print("Fitting SupervisedForecaster (LightGBM quantile regressors)...")
    sup_forecaster = SupervisedForecaster(models_dir=str(models_dir))
    sup_forecaster.fit(X, y_targets, n_estimators=60, learning_rate=0.08)
    sup_forecaster.save(models_dir)
    print("SupervisedForecaster saved successfully.")

    print("Fitting OutageForecaster (probabilistic classifiers)...")
    outage_forecaster = OutageForecaster(models_dir=str(models_dir))
    outage_forecaster.fit(X, y_outage)
    outage_forecaster.save(models_dir)
    print("OutageForecaster saved successfully.")

    return sup_forecaster, outage_forecaster


def evaluate_on_benchmarks(
    models_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Evaluate Persistence, Baseline, and Supervised forecasters on Scenarios A through F.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sampler = ScenarioSampler(seed=42)
    horizons = [1, 2, 4, 8, 16]
    dt_hours = 0.25

    sup_forecaster = SupervisedForecaster(models_dir=str(models_dir))
    outage_forecaster = OutageForecaster(models_dir=str(models_dir))
    baseline_forecaster = BaselineForecaster(horizons=horizons, dt_hours=dt_hours)
    persistence_forecaster = PersistenceForecaster(horizons=horizons)

    benchmarks = ["scenario_a", "scenario_b", "scenario_c", "scenario_d", "scenario_e", "scenario_f"]
    results_records: List[Dict[str, Any]] = []

    solar_model = SolarModel()
    wind_model = WindModel()
    load_manager = LoadManager()

    for scen in benchmarks:
        sm, meta = sampler.sample(scen)
        feature_builder = ForecastFeatureBuilder(history_window=16, dt_hours=dt_hours)

        hours_axis = np.linspace(0, 24.0, 96, endpoint=False)
        base_irr = solar_model.generate_daily_profile(hours_axis) * float(meta.get("solar_scale", 1.0))
        base_wind = wind_model.generate_daily_profile(hours_axis, mean_speed=float(meta.get("wind_mean_speed", 8.0)))

        # Precompute scenario states
        ep_states = []
        for step in range(96):
            h = step * dt_hours
            cf = sm.get_cloud_factor(h)
            irr = base_irr[step] * cf
            w_ovr = sm.get_wind_override(h)
            w_spd = w_ovr if w_ovr is not None else base_wind[step]
            grid_avail = sm.is_grid_available(h)
            storm_warn = sm.is_storm_warning(h)

            s_p = solar_model.update(irr, ambient_temp_c=28.0)
            w_p = wind_model.update(w_spd)
            d_dict = load_manager.get_nominal_demand(h)
            d_p = d_dict["total_demand_mw"]
            crit_p = d_dict["critical_mw"]
            flex_p = d_dict["flexible_mw"]
            other_p = d_dict["other_mw"]
            nl_p = d_p - (s_p + w_p)

            ep_states.append({
                "hour": h,
                "solar_mw": s_p,
                "wind_mw": w_p,
                "demand_mw": d_p,
                "crit_mw": crit_p,
                "flex_mw": flex_p,
                "other_mw": other_p,
                "net_load_mw": nl_p,
                "grid_available": grid_avail,
                "storm_warning": storm_warn,
                "cloud_factor": cf,
            })

        # Run step evaluations
        for step in range(96 - max(horizons)):
            cur = ep_states[step]
            feature_builder.update(
                hour=cur["hour"],
                solar_mw=cur["solar_mw"],
                wind_mw=cur["wind_mw"],
                total_demand_mw=cur["demand_mw"],
                crit_demand_mw=cur["crit_mw"],
                flex_demand_mw=cur["flex_mw"],
                other_demand_mw=cur["other_mw"],
                cloud_factor=cur["cloud_factor"],
                storm_warning=cur["storm_warning"],
            )
            feat = feature_builder.build_features(cur["hour"])

            # 1. Predictions
            pers_preds = persistence_forecaster.predict(
                current_solar_mw=cur["solar_mw"],
                current_wind_mw=cur["wind_mw"],
                current_demand_mw=cur["demand_mw"],
                current_crit_demand_mw=cur["crit_mw"],
                current_flex_demand_mw=cur["flex_mw"],
                current_other_demand_mw=cur["other_mw"],
                current_grid_available=cur["grid_available"],
                current_storm_risk=0.85 if cur["storm_warning"] else 0.02,
            )

            base_preds = baseline_forecaster.predict(
                current_hour=cur["hour"],
                current_solar_mw=cur["solar_mw"],
                current_wind_mw=cur["wind_mw"],
                current_demand_mw=cur["demand_mw"],
                current_crit_demand_mw=cur["crit_mw"],
                current_flex_demand_mw=cur["flex_mw"],
                current_other_demand_mw=cur["other_mw"],
                known_cloud_factor=cur["cloud_factor"],
                storm_warning=cur["storm_warning"],
                grid_available=cur["grid_available"],
            )

            sup_preds = sup_forecaster.predict(
                features=feat,
                current_hour=cur["hour"],
                current_solar_mw=cur["solar_mw"],
                current_wind_mw=cur["wind_mw"],
                current_demand_mw=cur["demand_mw"],
                known_cloud_factor=cur["cloud_factor"],
                storm_warning=cur["storm_warning"],
                grid_available=cur["grid_available"],
            )

            out_preds = outage_forecaster.predict(
                features=feat,
                storm_warning=cur["storm_warning"],
                grid_available=cur["grid_available"],
                cloud_factor=cur["cloud_factor"],
            )

            # Evaluate against future ground truth for each horizon
            for i, h in enumerate(horizons):
                fut = ep_states[step + h]
                y_true_s = fut["solar_mw"]
                y_true_w = fut["wind_mw"]
                y_true_d = fut["demand_mw"]
                y_true_nl = fut["net_load_mw"]
                y_true_out = 0 if fut["grid_available"] else 1

                for model_name, p_s, p_w, p_d, p_nl, p_out, q10_s, q90_s, q10_nl, q90_nl in [
                    (
                        "Persistence",
                        pers_preds["solar_mw"][i],
                        pers_preds["wind_mw"][i],
                        pers_preds["total_demand_mw"][i],
                        pers_preds["net_load_mw"][i],
                        pers_preds["outage_prob"][i],
                        pers_preds["solar_mw"][i] * 0.9,
                        pers_preds["solar_mw"][i] * 1.1,
                        pers_preds["net_load_mw"][i] - 0.1,
                        pers_preds["net_load_mw"][i] + 0.1,
                    ),
                    (
                        "Baseline",
                        base_preds["solar_mw"][i],
                        base_preds["wind_mw"][i],
                        base_preds["total_demand_mw"][i],
                        base_preds["net_load_mw"][i],
                        base_preds["outage_prob"][i],
                        base_preds["solar_mw"][i] * 0.85,
                        base_preds["solar_mw"][i] * 1.15,
                        base_preds["net_load_mw"][i] - 0.15,
                        base_preds["net_load_mw"][i] + 0.15,
                    ),
                    (
                        "Supervised",
                        sup_preds["solar_mw"]["point"][i],
                        sup_preds["wind_mw"]["point"][i],
                        sup_preds["total_demand_mw"]["point"][i],
                        sup_preds["net_load_mw"]["point"][i],
                        out_preds["outage_prob"][i],
                        sup_preds["solar_mw"]["q10"][i],
                        sup_preds["solar_mw"]["q90"][i],
                        sup_preds["net_load_mw"]["q10"][i],
                        sup_preds["net_load_mw"]["q90"][i],
                    ),
                ]:
                    results_records.append({
                        "scenario": scen,
                        "step": step,
                        "horizon_step": h,
                        "horizon_min": h * 15,
                        "model": model_name,
                        "solar_true": y_true_s,
                        "solar_pred": p_s,
                        "solar_ae": abs(y_true_s - p_s),
                        "solar_se": (y_true_s - p_s) ** 2,
                        "solar_covered": 1 if (q10_s <= y_true_s <= q90_s) else 0,
                        "solar_width": max(0.0, q90_s - q10_s),
                        "wind_true": y_true_w,
                        "wind_pred": p_w,
                        "wind_ae": abs(y_true_w - p_w),
                        "wind_se": (y_true_w - p_w) ** 2,
                        "demand_true": y_true_d,
                        "demand_pred": p_d,
                        "demand_ae": abs(y_true_d - p_d),
                        "demand_se": (y_true_d - p_d) ** 2,
                        "net_load_true": y_true_nl,
                        "net_load_pred": p_nl,
                        "net_load_ae": abs(y_true_nl - p_nl),
                        "net_load_se": (y_true_nl - p_nl) ** 2,
                        "net_load_covered": 1 if (q10_nl <= y_true_nl <= q90_nl) else 0,
                        "net_load_width": max(0.0, q90_nl - q10_nl),
                        "outage_true": y_true_out,
                        "outage_prob": p_out,
                        "brier_score": (p_out - y_true_out) ** 2,
                    })

    df = pd.DataFrame(results_records)

    # Save detailed evaluation records
    df.to_csv(output_dir / "forecast_raw_eval_records.csv", index=False)

    # Aggregate by Model and Horizon
    agg_rows = []
    for (model, h_min), group in df.groupby(["model", "horizon_min"]):
        s_mae = group["solar_ae"].mean() * 1000.0  # in kW
        s_rmse = np.sqrt(group["solar_se"].mean()) * 1000.0
        s_nrmse = (s_rmse / 500.0) * 100.0  # normalized by 500 kW capacity
        s_picp = group["solar_covered"].mean() * 100.0
        s_mpiw = group["solar_width"].mean() * 1000.0

        w_mae = group["wind_ae"].mean() * 1000.0
        w_rmse = np.sqrt(group["wind_se"].mean()) * 1000.0
        w_nrmse = (w_rmse / 250.0) * 100.0

        d_mae = group["demand_ae"].mean() * 1000.0
        d_rmse = np.sqrt(group["demand_se"].mean()) * 1000.0
        d_nrmse = (d_rmse / 800.0) * 100.0

        nl_mae = group["net_load_ae"].mean() * 1000.0
        nl_rmse = np.sqrt(group["net_load_se"].mean()) * 1000.0
        nl_nrmse = (nl_rmse / 800.0) * 100.0
        nl_picp = group["net_load_covered"].mean() * 100.0
        nl_mpiw = group["net_load_width"].mean() * 1000.0

        brier = group["brier_score"].mean()

        agg_rows.append({
            "model": model,
            "horizon_min": h_min,
            "solar_mae_kw": round(s_mae, 2),
            "solar_rmse_kw": round(s_rmse, 2),
            "solar_nrmse_pct": round(s_nrmse, 2),
            "solar_picp_pct": round(s_picp, 1),
            "solar_mpiw_kw": round(s_mpiw, 2),
            "wind_mae_kw": round(w_mae, 2),
            "wind_rmse_kw": round(w_rmse, 2),
            "wind_nrmse_pct": round(w_nrmse, 2),
            "demand_mae_kw": round(d_mae, 2),
            "demand_rmse_kw": round(d_rmse, 2),
            "demand_nrmse_pct": round(d_nrmse, 2),
            "net_load_mae_kw": round(nl_mae, 2),
            "net_load_rmse_kw": round(nl_rmse, 2),
            "net_load_nrmse_pct": round(nl_nrmse, 2),
            "net_load_picp_pct": round(nl_picp, 1),
            "net_load_mpiw_kw": round(nl_mpiw, 2),
            "outage_brier_score": round(brier, 4),
        })

    summary_df = pd.DataFrame(agg_rows)
    summary_df.to_csv(output_dir / "forecast_evaluation_metrics.csv", index=False)
    print("\nForecast Model Benchmark Evaluation Complete:")
    print(summary_df.to_string(index=False))

    return summary_df


if __name__ == "__main__":
    base_dir = _ROOT_DIR / "experiments" / "phase4" / "forecasting"
    m_dir = base_dir / "models"
    r_dir = base_dir / "results"

    print("GRIDGUARD AI - Multi-Horizon Forecasting Training & Benchmark Evaluation")
    print("=" * 70)
    train_forecasting_models(models_dir=m_dir, num_episodes=60, seed=42)
    evaluate_on_benchmarks(models_dir=m_dir, output_dir=r_dir)
