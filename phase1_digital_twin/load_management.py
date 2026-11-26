# GRIDGUARD AI
# Phase 1.5: Priority-Based Load Management & Curtailment
# ============================================================
# Manages multi-tiered microgrid loads:
#   Tier 1: Flexible Loads (0.2 MW) - Curtailed first; shiftable.
#   Tier 2: Other Loads    (0.3 MW) - Curtailed second.
#   Tier 3: Critical Loads (0.3 MW) - Shed ONLY as absolute last resort
#                                     and logged as severe emergency.
# Includes:
#   - Dynamic daily industrial campus load profiles.
#   - Load shifting queue for deferred flexible demand.
#   - Strict priority-based curtailment logic.
# ============================================================

from typing import Dict, Any, List
import numpy as np

try:
    from config import LOADS
except ImportError:
    from phase1_digital_twin.config import LOADS


class LoadManager:
    """
    Orchestrates industrial campus demand profiles, flexible load shifting,
    and priority-based curtailment.
    """

    def __init__(
        self,
        critical_mw: float = None,
        flexible_mw: float = None,
        other_mw: float = None,
        enable_shifting: bool = True,
        shift_recovery_rate_mw: float = 0.05,
    ):
        self.base_critical_mw = critical_mw if critical_mw is not None else LOADS["critical_mw"]
        self.base_flexible_mw = flexible_mw if flexible_mw is not None else LOADS["flexible_mw"]
        self.base_other_mw = other_mw if other_mw is not None else LOADS["other_mw"]

        self.enable_shifting = enable_shifting
        self.shift_recovery_rate_mw = shift_recovery_rate_mw
        self.deferred_flexible_mwh = 0.0

        # Cumulative metrics
        self.cum_critical_shed_mwh = 0.0
        self.cum_other_shed_mwh = 0.0
        self.cum_flexible_shed_mwh = 0.0
        self.cum_total_shed_mwh = 0.0
        self.cum_served_mwh = 0.0

        self.history = []

    def get_demand_multiplier(self, hour: float) -> float:
        """
        Normalized industrial campus diurnal demand curve.
        Night hours (00:00 - 06:00): 75% load
        Morning ramp (06:00 - 09:00): 75% -> 105%
        Daytime peak (09:00 - 17:00): 100% - 110%
        Evening ramp down (17:00 - 22:00): 100% -> 80%
        Night baseline (22:00 - 24:00): 75%
        """
        h = hour % 24.0
        if 0.0 <= h < 6.0:
            return 0.75 + 0.05 * np.sin(np.pi * h / 6.0)
        elif 6.0 <= h < 9.0:
            return 0.80 + 0.25 * ((h - 6.0) / 3.0)
        elif 9.0 <= h < 17.0:
            return 1.00 + 0.08 * np.sin(np.pi * (h - 9.0) / 8.0)
        elif 17.0 <= h < 22.0:
            return 1.00 - 0.20 * ((h - 17.0) / 5.0)
        else:
            return 0.80 - 0.05 * ((h - 22.0) / 2.0)

    def get_nominal_demand(self, hour: float) -> Dict[str, float]:
        """
        Get baseline nominal demand for each category at a given hour.
        Critical load is steady with minimal fluctuation.
        Flexible and other loads follow industrial campus schedule.
        """
        mult = self.get_demand_multiplier(hour)
        # Critical loads have high priority with low variation
        critical = self.base_critical_mw * (0.95 + 0.05 * mult)
        # Flexible and other loads follow operational schedule
        flexible = self.base_flexible_mw * mult
        other = self.base_other_mw * mult

        # If there is deferred flexible load and power allows, try to recover it
        rebound_flexible = 0.0
        if self.enable_shifting and self.deferred_flexible_mwh > 0:
            rebound_flexible = min(self.shift_recovery_rate_mw, self.deferred_flexible_mwh / 0.25)
            flexible += rebound_flexible

        total = critical + flexible + other
        return {
            "critical_mw": float(critical),
            "flexible_mw": float(flexible),
            "other_mw": float(other),
            "rebound_flexible_mw": float(rebound_flexible),
            "total_demand_mw": float(total),
        }

    def dispatch(
        self,
        deficit_mw: float,
        hour: float,
        dt_hours: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Evaluate demand and apply priority-based load shedding when a power deficit exists.

        Parameters:
            deficit_mw: Unmet power deficit needing curtailment (>= 0).
            hour: Current simulation hour.
            dt_hours: Timestep duration in hours.

        Returns:
            Dict containing nominal demand, served load, shed load per category,
            and emergency alerts.
        """
        nominal = self.get_nominal_demand(hour)
        rem_deficit = max(0.0, float(deficit_mw))

        # Priority 1: Shed flexible loads
        shed_flexible = min(rem_deficit, nominal["flexible_mw"])
        served_flexible = nominal["flexible_mw"] - shed_flexible
        rem_deficit -= shed_flexible

        if self.enable_shifting and shed_flexible > 0:
            # Defer shed flexible load to queue (up to 1.0 MWh cap)
            self.deferred_flexible_mwh = min(1.0, self.deferred_flexible_mwh + shed_flexible * dt_hours)
        elif nominal["rebound_flexible_mw"] > 0 and shed_flexible == 0:
            # Successfully served some deferred load
            self.deferred_flexible_mwh = max(
                0.0, self.deferred_flexible_mwh - nominal["rebound_flexible_mw"] * dt_hours
            )

        # Priority 2: Shed other loads
        shed_other = min(rem_deficit, nominal["other_mw"])
        served_other = nominal["other_mw"] - shed_other
        rem_deficit -= shed_other

        # Priority 3: Shed critical loads (LAST RESORT)
        shed_critical = 0.0
        is_critical_emergency = False
        if rem_deficit > 0.0001:
            shed_critical = min(rem_deficit, nominal["critical_mw"])
            rem_deficit -= shed_critical
            is_critical_emergency = True

        served_critical = nominal["critical_mw"] - shed_critical

        # Cumulative totals
        shed_total = shed_flexible + shed_other + shed_critical
        served_total = served_critical + served_flexible + served_other

        self.cum_critical_shed_mwh += shed_critical * dt_hours
        self.cum_other_shed_mwh += shed_other * dt_hours
        self.cum_flexible_shed_mwh += shed_flexible * dt_hours
        self.cum_total_shed_mwh += shed_total * dt_hours
        self.cum_served_mwh += served_total * dt_hours

        result = {
            "hour": hour,
            "nominal_critical_mw": nominal["critical_mw"],
            "nominal_flexible_mw": nominal["flexible_mw"],
            "nominal_other_mw": nominal["other_mw"],
            "nominal_total_mw": nominal["total_demand_mw"],
            "served_critical_mw": served_critical,
            "served_flexible_mw": served_flexible,
            "served_other_mw": served_other,
            "served_total_mw": served_total,
            "shed_critical_mw": shed_critical,
            "shed_flexible_mw": shed_flexible,
            "shed_other_mw": shed_other,
            "shed_total_mw": shed_total,
            "unmet_deficit_mw": rem_deficit,
            "critical_emergency": is_critical_emergency,
            "deferred_flexible_mwh": self.deferred_flexible_mwh,
        }
        self.history.append(result)
        return result

    def reset(self):
        """Reset internal history and accumulators."""
        self.deferred_flexible_mwh = 0.0
        self.cum_critical_shed_mwh = 0.0
        self.cum_other_shed_mwh = 0.0
        self.cum_flexible_shed_mwh = 0.0
        self.cum_total_shed_mwh = 0.0
        self.cum_served_mwh = 0.0
        self.history.clear()


if __name__ == "__main__":
    lm = LoadManager()
    print("GRIDGUARD AI - Load Management Self-Test")
    print("=" * 50)

    # Test nominal demand at night (h=3) vs peak (h=14)
    night = lm.get_nominal_demand(3.0)
    day = lm.get_nominal_demand(14.0)
    print(f"Night demand (h=3.0): {night['total_demand_mw']:.3f} MW (Crit: {night['critical_mw']:.3f}, Flex: {night['flexible_mw']:.3f}, Other: {night['other_mw']:.3f})")
    print(f"Peak demand  (h=14.0): {day['total_demand_mw']:.3f} MW (Crit: {day['critical_mw']:.3f}, Flex: {day['flexible_mw']:.3f}, Other: {day['other_mw']:.3f})")

    # Test small deficit: 0.15 MW (should shed flexible only)
    res_small = lm.dispatch(deficit_mw=0.15, hour=14.0)
    print(f"Small deficit (0.15 MW) -> Shed Flex: {res_small['shed_flexible_mw']:.3f} MW, Other: {res_small['shed_other_mw']:.3f} MW, Crit: {res_small['shed_critical_mw']:.3f} MW")
    assert res_small["shed_other_mw"] == 0.0 and res_small["shed_critical_mw"] == 0.0

    # Test medium deficit: 0.35 MW (should shed all flexible + some other)
    res_med = lm.dispatch(deficit_mw=0.35, hour=14.0)
    print(f"Medium deficit (0.35 MW) -> Shed Flex: {res_med['shed_flexible_mw']:.3f} MW, Other: {res_med['shed_other_mw']:.3f} MW, Crit: {res_med['shed_critical_mw']:.3f} MW")
    assert res_med["shed_flexible_mw"] > 0.19 and res_med["shed_other_mw"] > 0.0 and res_med["shed_critical_mw"] == 0.0

    # Test extreme deficit: 0.70 MW (shed flexible + other + some critical)
    res_extreme = lm.dispatch(deficit_mw=0.70, hour=14.0)
    print(f"Extreme deficit (0.70 MW) -> Shed Flex: {res_extreme['shed_flexible_mw']:.3f} MW, Other: {res_extreme['shed_other_mw']:.3f} MW, Crit: {res_extreme['shed_critical_mw']:.3f} MW | Alert: {res_extreme['critical_emergency']}")
    assert res_extreme["critical_emergency"] is True

    print("Load Management test passed successfully.")
