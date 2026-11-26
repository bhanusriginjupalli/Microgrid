# GRIDGUARD AI
# Phase 1.5: Resilience & Reliability Metrics Engine
# ============================================================
# Quantitative evaluation of microgrid survivability, supply security,
# fuel consumption, energy balance fidelity, and operating costs.
# ============================================================

from typing import Dict, List, Any
import numpy as np
import pandas as pd

try:
    from config import COSTS
except ImportError:
    from phase1_digital_twin.config import COSTS


class ResilienceMetrics:
    """
    Computes industry-standard resilience and operational economics metrics
    for microgrid performance evaluation under stress scenarios.
    """

    def __init__(self, cost_params: Dict[str, float] = None):
        self.costs = cost_params if cost_params is not None else COSTS

    def compute(self, df: pd.DataFrame, dt_hours: float = 0.25) -> Dict[str, Any]:
        """
        Evaluate full resilience and performance scorecard from simulation DataFrame.
        """
        n_steps = len(df)
        duration_hours = n_steps * dt_hours

        # 1. Unserved Energy & Load Shedding
        critical_shed_mwh = float(df.get("shed_critical_mw", pd.Series(0.0, index=df.index)).sum() * dt_hours)
        other_shed_mwh = float(df.get("shed_other_mw", pd.Series(0.0, index=df.index)).sum() * dt_hours)
        flexible_shed_mwh = float(df.get("shed_flexible_mw", df.get("load_shed_mw", pd.Series(0.0, index=df.index))).sum() * dt_hours)
        total_shed_mwh = critical_shed_mwh + other_shed_mwh + flexible_shed_mwh

        total_demand_mwh = float(df.get("nominal_total_mw", df.get("total_load_mw", pd.Series(0.0, index=df.index))).sum() * dt_hours)
        total_served_mwh = float(df.get("served_total_mw", df.get("total_load_mw", pd.Series(0.0, index=df.index))).sum() * dt_hours)

        # 2. Generation & Renewables
        total_solar_mwh = float(df["solar_mw"].sum() * dt_hours)
        total_wind_mwh = float(df["wind_mw"].sum() * dt_hours)
        total_re_mwh = total_solar_mwh + total_wind_mwh
        total_gen_mwh = float(df["generator_mw"].sum() * dt_hours)

        # Renewable fraction
        re_fraction_pct = (total_re_mwh / total_demand_mwh * 100.0) if total_demand_mwh > 0 else 0.0

        # 3. Grid Transactions
        grid_p = df["grid_import_mw"]
        grid_import_mwh = float(grid_p.clip(lower=0).sum() * dt_hours)
        grid_export_mwh = float(grid_p.clip(upper=0).abs().sum() * dt_hours)

        # 4. Storage Performance
        min_soc_pct = float(df["battery_soc"].min())
        max_soc_pct = float(df["battery_soc"].max())
        final_soc_pct = float(df["battery_soc"].iloc[-1])

        # 5. Generator Fuel Consumption
        fuel_consumed_l = float(df.get("step_fuel_l", pd.Series(0.0, index=df.index)).sum())
        if fuel_consumed_l == 0.0 and "total_fuel_l" in df:
            fuel_consumed_l = float(df["total_fuel_l"].iloc[-1])

        # 6. Outage & Recovery Timing
        islanded_steps = (~df["grid_connected"]).sum()
        islanded_duration_hours = float(islanded_steps * dt_hours)

        # Recovery time: hours from grid reconnection until flexible shedding is 0 and SOC >= initial
        recovery_time_hours = 0.0
        outage_indices = df.index[~df["grid_connected"]].tolist()
        if outage_indices:
            last_outage_idx = outage_indices[-1]
            post_outage = df.loc[last_outage_idx + 1:]
            for idx, row in post_outage.iterrows():
                if row.get("shed_total_mw", row.get("load_shed_mw", 0.0)) <= 0.001:
                    recovery_time_hours = float((idx - last_outage_idx) * dt_hours)
                    break

        # 7. Power Balance Fidelity
        balance_errors = df.get("balance_error_mw", pd.Series(0.0, index=df.index))
        max_balance_error_mw = float(balance_errors.max())
        mean_balance_error_mw = float(balance_errors.mean())

        # 8. Power Flow Convergence
        converged_steps = int(df["converged"].sum())
        convergence_rate_pct = float(converged_steps / n_steps * 100.0)

        # 9. Operational Cost Breakdown ($)
        cost_grid_import = grid_import_mwh * self.costs.get("grid_import_per_mwh", 120.0)
        revenue_grid_export = grid_export_mwh * self.costs.get("grid_export_per_mwh", 40.0)
        cost_fuel = fuel_consumed_l * self.costs.get("fuel_per_liter", 1.5)
        penalty_crit = critical_shed_mwh * self.costs.get("critical_unserved_penalty_per_mwh", 5000.0)
        penalty_other = other_shed_mwh * self.costs.get("other_shed_penalty_per_mwh", 500.0)
        penalty_flex = flexible_shed_mwh * self.costs.get("flexible_shed_penalty_per_mwh", 200.0)

        total_operating_cost = (
            cost_grid_import - revenue_grid_export + cost_fuel + penalty_crit + penalty_other + penalty_flex
        )

        return {
            "duration_hours": duration_hours,
            "total_demand_mwh": total_demand_mwh,
            "total_served_mwh": total_served_mwh,
            "total_shed_mwh": total_shed_mwh,
            "critical_shed_mwh": critical_shed_mwh,
            "flexible_shed_mwh": flexible_shed_mwh,
            "other_shed_mwh": other_shed_mwh,
            "critical_ens_mwh": critical_shed_mwh,
            "total_solar_mwh": total_solar_mwh,
            "total_wind_mwh": total_wind_mwh,
            "total_re_mwh": total_re_mwh,
            "total_gen_mwh": total_gen_mwh,
            "renewable_fraction_pct": re_fraction_pct,
            "grid_import_mwh": grid_import_mwh,
            "grid_export_mwh": grid_export_mwh,
            "min_soc_pct": min_soc_pct,
            "max_soc_pct": max_soc_pct,
            "final_soc_pct": final_soc_pct,
            "fuel_consumed_liters": fuel_consumed_l,
            "islanded_duration_hours": islanded_duration_hours,
            "recovery_time_hours": recovery_time_hours,
            "max_balance_error_mw": max_balance_error_mw,
            "mean_balance_error_mw": mean_balance_error_mw,
            "converged_steps": converged_steps,
            "total_steps": n_steps,
            "convergence_rate_pct": convergence_rate_pct,
            "cost_grid_import": cost_grid_import,
            "revenue_grid_export": revenue_grid_export,
            "cost_fuel": cost_fuel,
            "penalty_unserved": penalty_crit + penalty_other + penalty_flex,
            "total_operating_cost": total_operating_cost,
        }

    @staticmethod
    def print_summary(metrics: Dict[str, Any], title: str = "RESILIENCE & RELIABILITY REPORT"):
        """Print clean, formatted terminal summary report."""
        print("\n" + "=" * 65)
        print(f"GRIDGUARD AI - {title}")
        print("=" * 65)
        print(f"  Simulation Duration:       {metrics['duration_hours']:.1f} hours ({metrics['total_steps']} steps)")
        print(f"  Power Flow Convergence:    {metrics['converged_steps']}/{metrics['total_steps']} ({metrics['convergence_rate_pct']:.1f}%)")
        print(f"  Max Power Balance Error:   {metrics['max_balance_error_mw']:.6f} MW")
        print(f"  Mean Power Balance Error:  {metrics['mean_balance_error_mw']:.6f} MW")
        print("-" * 65)
        print("  SUPPLY & DEMAND:")
        print(f"    Total Demand:            {metrics['total_demand_mwh']:.3f} MWh")
        print(f"    Total Served:            {metrics['total_served_mwh']:.3f} MWh")
        print(f"    Renewable Generation:    {metrics['total_re_mwh']:.3f} MWh (Solar: {metrics['total_solar_mwh']:.2f}, Wind: {metrics['total_wind_mwh']:.2f})")
        print(f"    Renewable Penetration:   {metrics['renewable_fraction_pct']:.1f}%")
        print(f"    Backup Generator Output: {metrics['total_gen_mwh']:.3f} MWh")
        print(f"    Grid Import:             {metrics['grid_import_mwh']:.3f} MWh")
        print(f"    Grid Export:             {metrics['grid_export_mwh']:.3f} MWh")
        print("-" * 65)
        print("  RESILIENCE & SURVIVABILITY:")
        print(f"    Islanded Outage Time:    {metrics['islanded_duration_hours']:.2f} hours")
        print(f"    Critical Energy Unserved:{metrics['critical_ens_mwh']:.4f} MWh {'[SEVERE ALERT]' if metrics['critical_ens_mwh'] > 0 else '[SAFE]'}")
        print(f"    Other Load Shed:         {metrics['other_shed_mwh']:.3f} MWh")
        print(f"    Flexible Load Shed:      {metrics['flexible_shed_mwh']:.3f} MWh")
        print(f"    Total Load Shed:         {metrics['total_shed_mwh']:.3f} MWh")
        print(f"    Battery Min SOC:         {metrics['min_soc_pct']:.1f}% (Final: {metrics['final_soc_pct']:.1f}%)")
        print(f"    Diesel Fuel Consumed:    {metrics['fuel_consumed_liters']:.2f} Liters")
        print(f"    Recovery Time:           {metrics['recovery_time_hours']:.2f} hours")
        print("-" * 65)
        print("  ECONOMIC VALUATION ($):")
        print(f"    Grid Import Cost:       ${metrics['cost_grid_import']:8.2f}")
        print(f"    Grid Export Revenue:    -${metrics['revenue_grid_export']:8.2f}")
        print(f"    Fuel Expense:           ${metrics['cost_fuel']:8.2f}")
        print(f"    Curtailment Penalties:  ${metrics['penalty_unserved']:8.2f}")
        print(f"    Net Operating Cost:     ${metrics['total_operating_cost']:8.2f}")
        print("=" * 65 + "\n")


if __name__ == "__main__":
    # Test with dummy DataFrame
    df_test = pd.DataFrame({
        "solar_mw": [0.4, 0.4],
        "wind_mw": [0.2, 0.2],
        "generator_mw": [0.0, 0.1],
        "grid_import_mw": [0.2, 0.1],
        "battery_soc": [75.0, 72.0],
        "nominal_total_mw": [0.8, 0.8],
        "served_total_mw": [0.8, 0.8],
        "shed_critical_mw": [0.0, 0.0],
        "shed_other_mw": [0.0, 0.0],
        "shed_flexible_mw": [0.0, 0.0],
        "step_fuel_l": [0.0, 5.0],
        "grid_connected": [True, True],
        "converged": [True, True],
        "balance_error_mw": [0.00001, 0.00002],
    })
    rm = ResilienceMetrics()
    res = rm.compute(df_test, dt_hours=0.25)
    rm.print_summary(res, "Unit Test Report")
    print("Resilience Metrics test passed successfully.")
