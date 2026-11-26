# GRIDGUARD AI
# Phase 1: Disturbance Scenarios
# ============================================================
# Pre-defined disturbance profiles that can be injected into
# the simulation: cloud events, storms, grid outages.
# ============================================================

import numpy as np


class DisturbanceManager:
    """
    Manages time-varying disturbances applied during a simulation run.

    Each disturbance is a dict with:
        type       — "cloud", "storm", "grid_outage"
        start_hour — when it begins
        end_hour   — when it ends
        severity   — 0.0 (none) to 1.0 (total)
    """

    def __init__(self):
        self.disturbances: list[dict] = []

    # ── Convenience builders ────────────────────────────────
    def add_cloud_event(self, start_hour: float, end_hour: float,
                        severity: float = 0.7):
        """Cloud cover that reduces solar irradiance."""
        self.disturbances.append({
            "type": "cloud",
            "start_hour": start_hour,
            "end_hour": end_hour,
            "severity": np.clip(severity, 0, 1),
        })

    def add_storm(self, start_hour: float, end_hour: float,
                  wind_speed_ms: float = 28.0, cloud_severity: float = 0.9):
        """
        Storm: high wind (turbine cut-out) + heavy cloud cover.
        """
        self.disturbances.append({
            "type": "storm",
            "start_hour": start_hour,
            "end_hour": end_hour,
            "wind_speed_ms": wind_speed_ms,
            "cloud_severity": np.clip(cloud_severity, 0, 1),
        })

    def add_grid_outage(self, start_hour: float, end_hour: float):
        """Complete loss of utility grid connection."""
        self.disturbances.append({
            "type": "grid_outage",
            "start_hour": start_hour,
            "end_hour": end_hour,
        })

    # ── Query active disturbances at a given time ───────────
    def active_at(self, hour: float) -> list[dict]:
        """Return list of disturbances active at the given hour."""
        return [d for d in self.disturbances
                if d["start_hour"] <= hour < d["end_hour"]]

    def get_cloud_factor(self, hour: float) -> float:
        """
        Return a multiplicative cloud attenuation factor ∈ [0, 1].
        1.0 = clear sky,  0.0 = total blackout.
        """
        factor = 1.0
        for d in self.active_at(hour):
            if d["type"] == "cloud":
                factor *= (1.0 - d["severity"])
            elif d["type"] == "storm":
                factor *= (1.0 - d["cloud_severity"])
        return factor

    def get_wind_override(self, hour: float) -> float | None:
        """
        If a storm is active return the forced wind speed.
        Otherwise return None (let normal profile run).
        """
        for d in self.active_at(hour):
            if d["type"] == "storm":
                return d["wind_speed_ms"]
        return None

    def is_grid_down(self, hour: float) -> bool:
        """True if a grid outage is active at the given hour."""
        return any(d["type"] == "grid_outage" for d in self.active_at(hour))


def create_default_scenario() -> DisturbanceManager:
    """
    A 24 h scenario that exercises all disturbance types:

    • 09:00 – 10:30  Heavy cloud cover (severity 0.8)
    • 14:00 – 16:00  Storm (28 m/s wind → turbine cut-out, 90 % cloud)
    • 18:00 – 20:00  Grid outage (islanded operation)
    """
    dm = DisturbanceManager()
    dm.add_cloud_event(start_hour=9, end_hour=10.5, severity=0.8)
    dm.add_storm(start_hour=14, end_hour=16, wind_speed_ms=28, cloud_severity=0.9)
    dm.add_grid_outage(start_hour=18, end_hour=20)
    return dm


# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dm = create_default_scenario()
    print("GRIDGUARD AI - Disturbance Scenario Preview")
    print("=" * 50)
    for d in dm.disturbances:
        print(f"  {d['type']:15s}  {d['start_hour']:05.1f}h – {d['end_hour']:05.1f}h")
    print()

    for h in [8, 9.5, 12, 15, 19, 21]:
        active = dm.active_at(h)
        tags = [a["type"] for a in active] if active else ["clear"]
        print(f"  Hour {h:5.1f}:  {', '.join(tags):30s}  "
              f"cloud={dm.get_cloud_factor(h):.2f}  "
              f"grid={'DOWN' if dm.is_grid_down(h) else 'UP'}")
