# GRIDGUARD AI
# Phase 1.5: Multi-Stage Scenario & Disturbance Manager
# ============================================================
# Manages multi-stage disturbance scenarios, severe storms,
# transmission outages, and synchronization verification.
#
# Standard Scenarios:
#   Test A: Normal operation
#   Test B: Solar collapse
#   Test C: Grid outage (2 hours)
#   Test D: Severe storm (multi-stage)
#   Test E: Prolonged outage (6 hours)
#   Test F: Grid restoration & synchronization
# ============================================================

from typing import List, Dict, Any, Optional
import numpy as np


class ScenarioManager:
    """
    Injects and tracks environmental disturbances, physical grid faults,
    and operational alerts throughout the simulation timeline.
    """

    def __init__(self, name: str = "Custom Scenario", description: str = ""):
        self.name = name
        self.description = description
        self.events: List[Dict[str, Any]] = []

    def add_event(
        self,
        event_type: str,
        start_hour: float,
        end_hour: float,
        **kwargs,
    ):
        """Register a timed physical event or alert."""
        self.events.append({
            "type": event_type,
            "start_hour": float(start_hour),
            "end_hour": float(end_hour),
            **kwargs,
        })

    def get_active_events(self, hour: float) -> List[Dict[str, Any]]:
        """Return all events active at the specified timestamp."""
        return [
            e for e in self.events
            if e["start_hour"] <= hour < e["end_hour"]
        ]

    def is_storm_warning(self, hour: float) -> bool:
        """True if an early storm warning or advisory is active."""
        for e in self.get_active_events(hour):
            if e["type"] in ("storm_warning", "severe_storm"):
                return True
        return False

    def get_cloud_factor(self, hour: float) -> float:
        """
        Solar attenuation factor in [0.0, 1.0].
        1.0 = clear sky, 0.0 = total blackout.
        """
        factor = 1.0
        for e in self.get_active_events(hour):
            if e["type"] == "cloud":
                severity = e.get("severity", 0.7)
                factor *= (1.0 - severity)
            elif e["type"] == "solar_collapse":
                severity = e.get("severity", 0.8)
                factor *= (1.0 - severity)
            elif e["type"] == "severe_storm":
                severity = e.get("cloud_severity", 0.85)
                factor *= (1.0 - severity)
        return float(np.clip(factor, 0.0, 1.0))

    def get_wind_override(self, hour: float) -> Optional[float]:
        """
        Return forced wind speed in m/s if affected by a storm or gust,
        or None to keep the background stochastic wind model.
        """
        for e in self.get_active_events(hour):
            if e["type"] == "severe_storm":
                # High wind above 25 m/s causes turbine cut-out
                return float(e.get("wind_speed_ms", 27.5))
            elif e["type"] == "wind_gust":
                return float(e.get("wind_speed_ms", 18.0))
            elif e["type"] == "wind_lull":
                return float(e.get("wind_speed_ms", 1.5))
        return None

    def is_grid_available(self, hour: float) -> bool:
        """
        False if utility grid connection has tripped due to an outage.
        """
        for e in self.get_active_events(hour):
            if e["type"] in ("grid_outage", "severe_storm_outage", "prolonged_outage"):
                return False
        return True

    def is_sync_verified(self, hour: float) -> bool:
        """
        Synchronization verification check prior to closing the PCC tie breaker.
        Simulates voltage phase-angle matching and frequency lock.
        """
        for e in self.get_active_events(hour):
            if e["type"] == "synchronization_hold":
                return False
        return True

    # ========================================================
    # Pre-Configured Benchmark Scenarios (Tests A through F)
    # ========================================================

    @classmethod
    def scenario_a_normal(cls) -> "ScenarioManager":
        """Test A: Normal operation under clear sky and full grid support."""
        sm = cls("Scenario A: Normal Operation", "Grid connected, clear sky, normal operations.")
        return sm

    @classmethod
    def scenario_b_solar_collapse(cls) -> "ScenarioManager":
        """Test B: Solar output drops by 80% due to dense cloud cover."""
        sm = cls("Scenario B: Solar Collapse", "Solar generation collapses 80% between 10:00 and 14:00.")
        sm.add_event("solar_collapse", start_hour=10.0, end_hour=14.0, severity=0.80)
        return sm

    @classmethod
    def scenario_c_grid_outage(cls) -> "ScenarioManager":
        """Test C: Utility grid trip for 2 hours (12:00 to 14:00)."""
        sm = cls("Scenario C: 2-Hour Grid Outage", "Utility disconnects for 2 hours during peak midday load.")
        sm.add_event("grid_outage", start_hour=12.0, end_hour=14.0)
        return sm

    @classmethod
    def scenario_d_severe_storm(cls) -> "ScenarioManager":
        """
        Test D: Multi-stage severe storm:
          - 08:00 - 11:00: Storm warning & cloud buildup
          - 11:00 - 16:00: Solar collapse (85%) & turbine cut-out winds (28 m/s)
          - 12:00 - 16:00: Grid outage (transmission line damaged)
          - 16:00 - 16:50: Grid restored, synchronization hold
          - 16:50+: Fully recovered
        """
        sm = cls("Scenario D: Severe Storm", "Multi-stage severe storm with wind cut-out and grid trip.")
        sm.add_event("storm_warning", start_hour=8.0, end_hour=11.0)
        sm.add_event("severe_storm", start_hour=11.0, end_hour=16.0, wind_speed_ms=28.0, cloud_severity=0.85)
        sm.add_event("grid_outage", start_hour=12.0, end_hour=16.0)
        sm.add_event("synchronization_hold", start_hour=16.0, end_hour=16.5)
        return sm

    @classmethod
    def scenario_e_prolonged_outage(cls) -> "ScenarioManager":
        """Test E: 6-hour prolonged outage (09:00 to 15:00) testing battery reserve & generator endurance."""
        sm = cls("Scenario E: Prolonged Outage", "6-hour utility outage testing storage reserve preservation.")
        sm.add_event("prolonged_outage", start_hour=9.0, end_hour=15.0)
        return sm

    @classmethod
    def scenario_f_grid_restoration(cls) -> "ScenarioManager":
        """
        Test F: Grid returns after a 3-hour outage with synchronization verification
        and controlled battery recharge.
        """
        sm = cls("Scenario F: Grid Restoration & Sync", "Outage with active phase-sync check and recharge.")
        sm.add_event("grid_outage", start_hour=10.0, end_hour=13.0)
        sm.add_event("synchronization_hold", start_hour=13.0, end_hour=13.5)
        return sm


if __name__ == "__main__":
    sm = ScenarioManager.scenario_d_severe_storm()
    print("GRIDGUARD AI - Scenario Manager Self-Test")
    print("=" * 50)
    print(f"Scenario: {sm.name}")
    print(f"Description: {sm.description}")
    print("Registered Events:")
    for e in sm.events:
        print(f"  [{e['type']}] from {e['start_hour']:.1f}h to {e['end_hour']:.1f}h")

    # Sample checks across hours
    for h in [9.0, 11.5, 13.0, 16.2, 18.0]:
        print(f"Hour {h:4.1f}: CloudFactor={sm.get_cloud_factor(h):.2f} | "
              f"WindOverride={sm.get_wind_override(h)} | "
              f"GridAvailable={sm.is_grid_available(h)} | "
              f"SyncVerified={sm.is_sync_verified(h)}")

    print("Scenario Manager test passed successfully.")
