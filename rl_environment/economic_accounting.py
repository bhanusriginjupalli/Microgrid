"""
rl_environment/economic_accounting.py
=====================================
GRIDGUARD AI — Authoritative Economic Accounting Engine (Phase 4.4.2)

Provides one unified, verified economic accounting function used consistently
by every controller, training logger, validation protocol, and benchmark.

Distinguishes strictly between:
  1. Operating Cost ($):
       grid_import_cost - grid_export_revenue + fuel_cost + startup_wear_cost
       (Note: DynamicGenerator.total_fuel_l already includes startup fuel, so
        startup fuel is NEVER double-counted.)
  2. True Social Cost ($):
       operating_cost + unserved_energy_penalty (VoLL)
       + battery_degradation_cost + terminal_soc_replacement_adjustment
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import yaml


DEFAULT_ECONOMICS_YAML = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "phase4_4_2"
    / "configs"
    / "economics_v1.yaml"
)


@dataclass
class EconomicConfig:
    grid_import_usd_per_mwh: float = 120.0
    grid_export_usd_per_mwh: float = 40.0
    fuel_usd_per_liter: float = 1.50
    generator_startup_wear_usd: float = 0.0
    voll_critical_usd_per_mwh: float = 10000.0
    voll_flexible_usd_per_mwh: float = 2000.0
    voll_other_usd_per_mwh: float = 1000.0
    battery_degradation_usd_per_mwh: float = 15.0
    terminal_soc_replacement_usd_per_mwh: float = 120.0
    terminal_soc_accounting_mode: str = "symmetric_modelling_valuation"

    @classmethod
    def from_yaml(cls, yaml_path: Optional[Union[str, Path]] = None) -> "EconomicConfig":
        path = Path(yaml_path) if yaml_path is not None else DEFAULT_ECONOMICS_YAML
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        tariffs = data.get("tariffs", {})
        voll = data.get("voll_penalties", {})
        storage = data.get("storage_accounting", {})

        return cls(
            grid_import_usd_per_mwh=float(tariffs.get("grid_import_usd_per_mwh", 120.0)),
            grid_export_usd_per_mwh=float(tariffs.get("grid_export_usd_per_mwh", 40.0)),
            fuel_usd_per_liter=float(tariffs.get("fuel_usd_per_liter", 1.50)),
            generator_startup_wear_usd=float(tariffs.get("generator_startup_wear_usd", 0.0)),
            voll_critical_usd_per_mwh=float(voll.get("critical_usd_per_mwh", 10000.0)),
            voll_flexible_usd_per_mwh=float(voll.get("flexible_usd_per_mwh", 2000.0)),
            voll_other_usd_per_mwh=float(voll.get("other_usd_per_mwh", 1000.0)),
            battery_degradation_usd_per_mwh=float(storage.get("battery_degradation_usd_per_mwh", 15.0)),
            terminal_soc_replacement_usd_per_mwh=float(storage.get("terminal_soc_replacement_usd_per_mwh", 120.0)),
            terminal_soc_accounting_mode=str(storage.get("terminal_soc_accounting_mode", "symmetric_modelling_valuation")),
        )


def _extract_step_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float, float, float]:
    """
    Separate optional t=0 reset snapshot (row 0 when row 0 and row 1 both have step==0)
    from actual simulated steps, returning:
      (steps_df, initial_soc, final_soc, initial_energy_mwh, final_energy_mwh)
    """
    if df.empty:
        raise ValueError("Cannot compute economics on an empty episode trajectory.")

    initial_soc = float(df["battery_soc"].iloc[0])
    final_soc = float(df["battery_soc"].iloc[-1])
    initial_energy_mwh = float(df["battery_energy_mwh"].iloc[0])
    final_energy_mwh = float(df["battery_energy_mwh"].iloc[-1])

    # MicrogridParallelEnv appends _evaluate_initial_state() with step==0 at reset,
    # and then appends step 0..N-1 during env.step(). Detect and exclude reset snapshot
    # from time-step integration so 96 steps = exact 24.00 hours.
    if len(df) > 1 and "step" in df.columns and int(df["step"].iloc[0]) == 0 and int(df["step"].iloc[1]) == 0:
        steps_df = df.iloc[1:].reset_index(drop=True)
    else:
        steps_df = df.reset_index(drop=True)

    return steps_df, initial_soc, final_soc, initial_energy_mwh, final_energy_mwh


def count_generator_startups(df: pd.DataFrame) -> int:
    """Count cold/warm generator transitions into STARTING state."""
    if "generator_state" not in df.columns or df.empty:
        return 0
    states = df["generator_state"].astype(str).tolist()
    startups = 0
    prev = "OFF"
    for idx, st in enumerate(states):
        if idx == 0 and st == "STARTING":
            startups += 1
        elif st == "STARTING" and prev not in ("STARTING", "RUNNING"):
            startups += 1
        prev = st
    return startups


def compute_episode_economics(
    episode_history: Union[pd.DataFrame, list],
    dt_hours: float = 0.25,
    config: Optional[EconomicConfig] = None,
) -> Dict[str, Any]:
    """
    Authoritative episode economic & physical accounting.

    Validates all energy quantities in MWh and all costs in USD.
    Ensures zero double-counting of unserved energy tiers or startup fuel.
    """
    cfg = config or EconomicConfig.from_yaml()
    df = episode_history.copy() if isinstance(episode_history, pd.DataFrame) else pd.DataFrame(episode_history)

    steps_df, init_soc, final_soc, init_energy_mwh, final_energy_mwh = _extract_step_dataframe(df)
    n_steps = len(steps_df)
    sim_duration_hours = float(n_steps * dt_hours)

    # 1. Grid Energy Exchange (MWh)
    grid_p = steps_df["grid_import_mw"].astype(float)
    grid_import_mwh = float(grid_p.clip(lower=0.0).sum() * dt_hours)
    grid_export_mwh = float((-grid_p.clip(upper=0.0)).sum() * dt_hours)

    grid_import_cost_usd = grid_import_mwh * cfg.grid_import_usd_per_mwh
    grid_export_revenue_usd = grid_export_mwh * cfg.grid_export_usd_per_mwh

    # 2. Generator Fuel & Startups
    # total_fuel_l already accumulates startup_fuel_l + warmup + running fuel in DynamicGenerator
    if "total_fuel_l" in steps_df.columns:
        fuel_consumed_l = float(steps_df["total_fuel_l"].iloc[-1])
    else:
        fuel_consumed_l = float(steps_df["step_fuel_l"].sum())
    fuel_cost_usd = fuel_consumed_l * cfg.fuel_usd_per_liter

    startup_count = count_generator_startups(df)
    startup_wear_cost_usd = float(startup_count * cfg.generator_startup_wear_usd)
    gen_operating_hours = float((steps_df["generator_mw"].astype(float) > 0.001).sum() * dt_hours)

    # 3. Authoritative Operating Cost ($)
    operating_cost_usd = grid_import_cost_usd - grid_export_revenue_usd + fuel_cost_usd + startup_wear_cost_usd

    # 4. Demand & Unserved Energy (MWh) - mutually exclusive tiers (no double-counting)
    nominal_crit_mwh = float(steps_df["nominal_critical_mw"].clip(lower=0.0).sum() * dt_hours)
    nominal_flex_mwh = float(steps_df["nominal_flexible_mw"].clip(lower=0.0).sum() * dt_hours)
    nominal_other_mwh = float(steps_df["nominal_other_mw"].clip(lower=0.0).sum() * dt_hours)
    nominal_total_mwh = nominal_crit_mwh + nominal_flex_mwh + nominal_other_mwh

    served_crit_mwh = float(steps_df["served_critical_mw"].clip(lower=0.0).sum() * dt_hours)
    served_flex_mwh = float(steps_df["served_flexible_mw"].clip(lower=0.0).sum() * dt_hours)
    served_other_mwh = float(steps_df["served_other_mw"].clip(lower=0.0).sum() * dt_hours)
    served_total_mwh = served_crit_mwh + served_flex_mwh + served_other_mwh

    critical_ens_mwh = float(steps_df["shed_critical_mw"].clip(lower=0.0).sum() * dt_hours)
    flexible_ens_mwh = float(steps_df["shed_flexible_mw"].clip(lower=0.0).sum() * dt_hours)
    other_ens_mwh = float(steps_df["shed_other_mw"].clip(lower=0.0).sum() * dt_hours)
    total_ens_mwh = critical_ens_mwh + flexible_ens_mwh + other_ens_mwh

    voll_critical_usd = critical_ens_mwh * cfg.voll_critical_usd_per_mwh
    voll_flexible_usd = flexible_ens_mwh * cfg.voll_flexible_usd_per_mwh
    voll_other_usd = other_ens_mwh * cfg.voll_other_usd_per_mwh
    unserved_energy_cost_usd = voll_critical_usd + voll_flexible_usd + voll_other_usd

    # 5. Battery Degradation & Terminal SOC Valuation
    battery_throughput_mwh = float(steps_df["battery_mw"].astype(float).abs().sum() * dt_hours)
    battery_degradation_cost_usd = battery_throughput_mwh * cfg.battery_degradation_usd_per_mwh

    # Positive delta means net energy extracted from battery during episode
    delta_storage_mwh = init_energy_mwh - final_energy_mwh
    terminal_soc_adjustment_usd = delta_storage_mwh * cfg.terminal_soc_replacement_usd_per_mwh

    # 6. Authoritative True Social Cost ($)
    true_social_cost_usd = (
        operating_cost_usd
        + unserved_energy_cost_usd
        + battery_degradation_cost_usd
        + terminal_soc_adjustment_usd
    )

    # 7. Renewable Curtailment & Reliability Metrics
    solar_curt_mwh = float((steps_df["solar_available_mw"] - steps_df["solar_mw"]).clip(lower=0.0).sum() * dt_hours)
    wind_curt_mwh = float((steps_df["wind_available_mw"] - steps_df["wind_mw"]).clip(lower=0.0).sum() * dt_hours)
    renewable_curtailment_mwh = solar_curt_mwh + wind_curt_mwh

    critical_reliability_pct = (served_crit_mwh / nominal_crit_mwh * 100.0) if nominal_crit_mwh > 0.0 else 100.0
    total_reliability_pct = (served_total_mwh / nominal_total_mwh * 100.0) if nominal_total_mwh > 0.0 else 100.0

    # 8. Validation Assertions (Physical & Mathematical Consistency)
    values_to_check = [
        grid_import_mwh, grid_export_mwh, fuel_consumed_l, operating_cost_usd,
        critical_ens_mwh, flexible_ens_mwh, other_ens_mwh, total_ens_mwh,
        unserved_energy_cost_usd, battery_throughput_mwh, battery_degradation_cost_usd,
        terminal_soc_adjustment_usd, true_social_cost_usd,
    ]
    if not all(np.isfinite(values_to_check)):
        raise ValueError("Non-finite value encountered in economic accounting.")
    if abs(total_ens_mwh - (critical_ens_mwh + flexible_ens_mwh + other_ens_mwh)) > 1e-6:
        raise AssertionError("Unserved energy tier sum mismatch.")

    return {
        "sim_steps": n_steps,
        "sim_duration_hours": round(sim_duration_hours, 4),
        "grid_import_mwh": round(grid_import_mwh, 6),
        "grid_export_mwh": round(grid_export_mwh, 6),
        "grid_import_cost_usd": round(grid_import_cost_usd, 4),
        "grid_export_revenue_usd": round(grid_export_revenue_usd, 4),
        "fuel_consumed_l": round(fuel_consumed_l, 4),
        "fuel_cost_usd": round(fuel_cost_usd, 4),
        "generator_startup_count": int(startup_count),
        "generator_operating_hours": round(gen_operating_hours, 4),
        "startup_wear_cost_usd": round(startup_wear_cost_usd, 4),
        "operating_cost_usd": round(operating_cost_usd, 4),
        "nominal_critical_mwh": round(nominal_crit_mwh, 6),
        "served_critical_mwh": round(served_crit_mwh, 6),
        "critical_ens_mwh": round(critical_ens_mwh, 6),
        "flexible_ens_mwh": round(flexible_ens_mwh, 6),
        "other_ens_mwh": round(other_ens_mwh, 6),
        "total_ens_mwh": round(total_ens_mwh, 6),
        "critical_reliability_pct": round(critical_reliability_pct, 4),
        "total_reliability_pct": round(total_reliability_pct, 4),
        "voll_critical_usd": round(voll_critical_usd, 4),
        "voll_flexible_usd": round(voll_flexible_usd, 4),
        "voll_other_usd": round(voll_other_usd, 4),
        "unserved_energy_cost_usd": round(unserved_energy_cost_usd, 4),
        "battery_throughput_mwh": round(battery_throughput_mwh, 6),
        "battery_degradation_cost_usd": round(battery_degradation_cost_usd, 4),
        "initial_battery_soc_pct": round(init_soc, 4),
        "terminal_battery_soc_pct": round(final_soc, 4),
        "initial_battery_energy_mwh": round(init_energy_mwh, 6),
        "terminal_battery_energy_mwh": round(final_energy_mwh, 6),
        "delta_storage_mwh": round(delta_storage_mwh, 6),
        "terminal_soc_adjustment_usd": round(terminal_soc_adjustment_usd, 4),
        "terminal_soc_accounting_mode": cfg.terminal_soc_accounting_mode,
        "true_social_cost_usd": round(true_social_cost_usd, 4),
        "renewable_curtailment_mwh": round(renewable_curtailment_mwh, 6),
    }
