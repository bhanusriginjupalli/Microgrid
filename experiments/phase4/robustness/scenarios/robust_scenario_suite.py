"""
experiments/phase4/robustness/scenarios/robust_scenario_suite.py
============================================================
GRIDGUARD AI - Phase 4.3: Unseen Domain Shift Scenarios & Disturbance Suite

Defines physically realistic, held-out microgrid disturbances:
  Category A: Network Topology Shifts (Branch/Line trips, islanding)
  Category B: Renewable Generation Shifts (Extended drought, correlated collapse)
  Category C: Extreme Demand Shifts (Critical surges, flexible rebound)
  Category D: Measurement & Telemetry Failures (Gaussian noise, dropouts, stale lag)
  Category E: Compound Extreme Disturbances (Multi-fault stress tests)

Separates distributions into:
  - Training Distribution (Domain Randomization generator)
  - Validation Distribution (Fixed validation seeds)
  - Held-Out Test Distribution (Genuinely unseen scenarios H1 to H6)
============================================================
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase1_digital_twin"))

from phase1_digital_twin.scenario_manager import ScenarioManager


class RobustScenarioManager(ScenarioManager):
    """
    Extends standard ScenarioManager with physical topology trips,
    load tier surges, generator derating, sensor degradation, and forecast freezes.
    """

    def __init__(self, name: str = "Robust Scenario", description: str = ""):
        super().__init__(name=name, description=description)
        self.line_outages: List[Dict[str, Any]] = []
        self.demand_shifts: List[Dict[str, Any]] = []
        self.generator_deratings: List[Dict[str, Any]] = []
        self.sensor_noise_events: List[Dict[str, Any]] = []
        self.forecast_freeze_events: List[Dict[str, Any]] = []

    def add_line_outage(self, line_idx: int, start_hour: float, end_hour: float, reason: str = ""):
        """Trip an electrical branch/line in Pandapower network."""
        self.line_outages.append({
            "line_idx": int(line_idx),
            "start_hour": float(start_hour),
            "end_hour": float(end_hour),
            "reason": reason,
        })

    def add_demand_shift(
        self,
        start_hour: float,
        end_hour: float,
        critical_mult: float = 1.0,
        flexible_mult: float = 1.0,
        other_mult: float = 1.0,
        reason: str = "",
    ):
        """Scale demand tiers (e.g. heatwave surge, recovery rebound)."""
        self.demand_shifts.append({
            "start_hour": float(start_hour),
            "end_hour": float(end_hour),
            "critical_mult": float(critical_mult),
            "flexible_mult": float(flexible_mult),
            "other_mult": float(other_mult),
            "reason": reason,
        })

    def add_generator_derating(
        self,
        start_hour: float,
        end_hour: float,
        capacity_factor: float = 0.5,
        reason: str = "",
    ):
        """Derate diesel generator maximum capacity and ramp rate."""
        self.generator_deratings.append({
            "start_hour": float(start_hour),
            "end_hour": float(end_hour),
            "capacity_factor": float(np.clip(capacity_factor, 0.1, 1.0)),
            "reason": reason,
        })

    def add_sensor_noise(
        self,
        start_hour: float,
        end_hour: float,
        gaussian_sigma: float = 0.05,
        dropout_prob: float = 0.0,
        stale_lag_steps: int = 0,
    ):
        """Inject telemetry sensor noise, missing dropouts, or observation lag."""
        self.sensor_noise_events.append({
            "start_hour": float(start_hour),
            "end_hour": float(end_hour),
            "gaussian_sigma": float(gaussian_sigma),
            "dropout_prob": float(dropout_prob),
            "stale_lag_steps": int(stale_lag_steps),
        })

    def add_forecast_freeze(self, start_hour: float, end_hour: float):
        """Freeze lookahead forecast updates (forcing stale predictions)."""
        self.forecast_freeze_events.append({
            "start_hour": float(start_hour),
            "end_hour": float(end_hour),
        })

    # ── Physical Dynamic Queries ──────────────────────────────────────────────

    def get_line_in_service_map(self, hour: float, total_lines: int = 8) -> Dict[int, bool]:
        """Return dict of {line_idx: in_service_bool} at the given timestamp."""
        res = {idx: True for idx in range(total_lines)}
        for o in self.line_outages:
            if o["start_hour"] <= hour < o["end_hour"]:
                res[o["line_idx"]] = False
        return res

    def get_demand_multipliers(self, hour: float) -> Dict[str, float]:
        """Return demand scaling multipliers for critical, flexible, and other tiers."""
        mults = {"critical": 1.0, "flexible": 1.0, "other": 1.0}
        for d in self.demand_shifts:
            if d["start_hour"] <= hour < d["end_hour"]:
                mults["critical"] *= d["critical_mult"]
                mults["flexible"] *= d["flexible_mult"]
                mults["other"] *= d["other_mult"]
        return mults

    def get_generator_derating_factor(self, hour: float) -> float:
        """Return generator capacity scaling factor in (0.0, 1.0]."""
        factor = 1.0
        for g in self.generator_deratings:
            if g["start_hour"] <= hour < g["end_hour"]:
                factor = min(factor, g["capacity_factor"])
        return float(factor)

    def get_sensor_noise_profile(self, hour: float) -> Dict[str, Any]:
        """Return current sensor corruption parameters."""
        sigma = 0.0
        dropout = 0.0
        lag = 0
        for s in self.sensor_noise_events:
            if s["start_hour"] <= hour < s["end_hour"]:
                sigma = max(sigma, s["gaussian_sigma"])
                dropout = max(dropout, s["dropout_prob"])
                lag = max(lag, s["stale_lag_steps"])
        return {
            "gaussian_sigma": sigma,
            "dropout_prob": dropout,
            "stale_lag_steps": lag,
        }

    def is_forecast_frozen(self, hour: float) -> bool:
        """True if forecast updates are frozen (stale)."""
        for f in self.forecast_freeze_events:
            if f["start_hour"] <= hour < f["end_hour"]:
                return True
        return False


class HeldOutScenarios:
    """
    Library of 6 genuinely unseen held-out domain shift scenarios.
    Never encountered during training or checkpoint selection.
    """

    @classmethod
    def scenario_h1_topology_shift(cls) -> RobustScenarioManager:
        """
        Held-Out H1: Network Topology Shifts
        - Line 1 (Solar branch) trips 10:00 - 13:00 (physical PV disconnection).
        - Line 2 (Wind branch) trips 14:00 - 16:00 (physical Wind disconnection).
        - Unplanned 1-hour grid islanding at 12:00 - 13:00 while solar is also disconnected!
        """
        sm = RobustScenarioManager(
            name="Scenario H1: Network Topology Shifts",
            description="Branch outages isolating solar and wind generation combined with unannounced islanding.",
        )
        sm.add_line_outage(line_idx=1, start_hour=10.0, end_hour=13.0, reason="Line 1 Solar Feeder Trip")
        sm.add_line_outage(line_idx=2, start_hour=14.0, end_hour=16.0, reason="Line 2 Wind Feeder Trip")
        sm.add_event("grid_outage", start_hour=12.0, end_hour=13.0)
        return sm

    @classmethod
    def scenario_h2_renewable_drought(cls) -> RobustScenarioManager:
        """
        Held-Out H2: Extended Correlated Renewable Drought
        - 18-hour dense overcast (cloud factor 0.15, solar down 85%) from 06:00 to 18:00.
        - Stagnant high-pressure system: Wind speed sustained at 1.0 m/s (below 2.5 m/s cut-in) 04:00 to 22:00.
        - Grid outage 11:00 to 14:00 during zero renewable availability.
        """
        sm = RobustScenarioManager(
            name="Scenario H2: Extended Renewable Drought",
            description="Prolonged 18-hour correlated solar and wind collapse with midday grid outage.",
        )
        sm.add_event("solar_collapse", start_hour=6.0, end_hour=18.0, severity=0.85)
        sm.add_event("wind_lull", start_hour=4.0, end_hour=22.0, wind_speed_ms=1.0)
        sm.add_event("grid_outage", start_hour=11.0, end_hour=14.0)
        return sm

    @classmethod
    def scenario_h3_demand_surge(cls) -> RobustScenarioManager:
        """
        Held-Out H3: Extreme Demand Surge & Flexible Rebound
        - Critical load surges +70% from 08:00 to 16:00 (heatwave cooling emergency).
        - Flexible load rebound +80% from 14:00 to 18:00.
        - Grid outage 12:00 to 15:00 right during peak demand surge.
        """
        sm = RobustScenarioManager(
            name="Scenario H3: Extreme Demand Surge",
            description="Massive +70% critical load surge and +80% flexible rebound during grid outage.",
        )
        sm.add_demand_shift(
            start_hour=8.0,
            end_hour=16.0,
            critical_mult=1.70,
            flexible_mult=1.0,
            other_mult=1.20,
            reason="Heatwave Critical Cooling Surge",
        )
        sm.add_demand_shift(
            start_hour=14.0,
            end_hour=18.0,
            critical_mult=1.0,
            flexible_mult=1.80,
            other_mult=1.0,
            reason="Industrial Flexible Catchup Rebound",
        )
        sm.add_event("grid_outage", start_hour=12.0, end_hour=15.0)
        return sm

    @classmethod
    def scenario_h4_sensor_failure(cls) -> RobustScenarioManager:
        """
        Held-Out H4: Severe Sensor Noise & Telemetry Dropout
        - High Gaussian noise (sigma = 0.12) on voltage, SOC, and load sensors 06:00 to 18:00.
        - Telemetry packet dropout rate 15% (holding last valid reading) 09:00 to 15:00.
        - Moderate cloud disturbance and 2-hour grid outage 13:00 to 15:00.
        """
        sm = RobustScenarioManager(
            name="Scenario H4: Sensor & Telemetry Failures",
            description="Gaussian noise sigma=0.12 and 15% sensor dropout during midday grid disturbance.",
        )
        sm.add_sensor_noise(
            start_hour=6.0,
            end_hour=18.0,
            gaussian_sigma=0.12,
            dropout_prob=0.15,
            stale_lag_steps=1,
        )
        sm.add_event("cloud", start_hour=10.0, end_hour=14.0, severity=0.60)
        sm.add_event("grid_outage", start_hour=13.0, end_hour=15.0)
        return sm

    @classmethod
    def scenario_h5_stale_forecasts(cls) -> RobustScenarioManager:
        """
        Held-Out H5: Stale / Frozen Forecasts Under Rapid Weather Transition
        - Lookahead forecasts frozen for 4 hours (10:00 to 14:00), transmitting obsolete clear-sky data.
        - Reality: Unannounced severe storm arrives at 11:30 with 3-hour grid outage (12:00 to 15:00).
        """
        sm = RobustScenarioManager(
            name="Scenario H5: Stale Forecast Freeze",
            description="Lookahead forecasts frozen while unannounced storm and grid outage occur.",
        )
        sm.add_forecast_freeze(start_hour=10.0, end_hour=14.0)
        sm.add_event("solar_collapse", start_hour=11.5, end_hour=15.0, severity=0.80)
        sm.add_event("wind_gust", start_hour=11.5, end_hour=14.0, wind_speed_ms=22.0)
        sm.add_event("grid_outage", start_hour=12.0, end_hour=15.0)
        return sm

    @classmethod
    def scenario_h6_compound_stress(cls) -> RobustScenarioManager:
        """
        Held-Out H6: Compound Extreme Multi-Fault
        - Grid Outage 10:00 to 15:00 (5 hours).
        - Generator Capacity Derated by 50% (0.20 MW max, half ramp rate) due to mechanical overheating.
        - Line 1 (Solar) trips 11:00 to 14:00.
        - Critical Load Surge +50% 10:00 to 16:00.
        - Sensor Noise sigma=0.08 throughout outage.
        """
        sm = RobustScenarioManager(
            name="Scenario H6: Compound Extreme Multi-Fault",
            description="Combined 5h outage, generator 50% derating, branch trip, critical surge, and sensor noise.",
        )
        sm.add_event("grid_outage", start_hour=10.0, end_hour=15.0)
        sm.add_generator_derating(start_hour=10.0, end_hour=16.0, capacity_factor=0.50, reason="Mechanical Thermal Derating")
        sm.add_line_outage(line_idx=1, start_hour=11.0, end_hour=14.0, reason="Solar Branch Trip")
        sm.add_demand_shift(start_hour=10.0, end_hour=16.0, critical_mult=1.50, flexible_mult=1.0, other_mult=1.0)
        sm.add_sensor_noise(start_hour=10.0, end_hour=15.0, gaussian_sigma=0.08, dropout_prob=0.10, stale_lag_steps=0)
        return sm

    @classmethod
    def get_all_heldout_scenarios(cls) -> Dict[str, RobustScenarioManager]:
        """Return dict of all 6 held-out benchmark scenarios."""
        return {
            "scenario_h1": cls.scenario_h1_topology_shift(),
            "scenario_h2": cls.scenario_h2_renewable_drought(),
            "scenario_h3": cls.scenario_h3_demand_surge(),
            "scenario_h4": cls.scenario_h4_sensor_failure(),
            "scenario_h5": cls.scenario_h5_stale_forecasts(),
            "scenario_h6": cls.scenario_h6_compound_stress(),
        }


class DomainRandomizer:
    """
    Training Distribution Generator:
    Applies continuous domain randomization across physics, environment,
    equipment capacity, and telemetry quality during RL training.
    Genuinely isolates training distributions from the held-out test suite.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def sample_training_scenario(self, difficulty: int = 1) -> Tuple[RobustScenarioManager, Dict[str, Any]]:
        """
        Generate randomized training scenario with stochastic variations:
          - Solar capacity factor scale: [0.70, 1.25]
          - Wind speed mean shift: [-2.5, +3.0] m/s
          - Demand scaling: [0.85, 1.20]
          - Initial battery SOC: [60%, 90%]
          - Generator capacity derating: [0.80, 1.00]
          - Sensor noise sigma: [0.00, 0.04]
          - Random outage probability: 0.35 (duration 1 - 3 hours)
        """
        sm = RobustScenarioManager(
            name=f"Randomized Train Scenario (Diff {difficulty})",
            description="Procedurally randomized domain parameters for robust policy generalization.",
        )

        solar_scale = float(self.rng.uniform(0.70, 1.25))
        wind_mean_shift = float(self.rng.uniform(-2.5, 3.0))
        demand_scale = float(self.rng.uniform(0.85, 1.20))
        initial_soc = float(self.rng.uniform(60.0, 90.0))
        gen_derate = float(self.rng.uniform(0.80, 1.00))
        sensor_sigma = float(self.rng.uniform(0.00, 0.04))

        # Stochastic cloud events
        if self.rng.random() < 0.60:
            c_start = float(self.rng.uniform(8.0, 13.0))
            c_dur = float(self.rng.uniform(1.5, 4.0))
            c_sev = float(self.rng.uniform(0.30, 0.80))
            sm.add_event("cloud", start_hour=c_start, end_hour=c_start + c_dur, severity=c_sev)

        # Stochastic grid outage
        if self.rng.random() < 0.40:
            o_start = float(self.rng.uniform(9.0, 14.0))
            o_dur = float(self.rng.uniform(1.0, 3.0))
            sm.add_event("grid_outage", start_hour=o_start, end_hour=o_start + o_dur)

        # Apply demand shift
        if abs(demand_scale - 1.0) > 0.05:
            sm.add_demand_shift(
                start_hour=0.0,
                end_hour=24.0,
                critical_mult=demand_scale,
                flexible_mult=demand_scale,
                other_mult=demand_scale,
            )

        # Apply generator derating
        if gen_derate < 0.98:
            sm.add_generator_derating(start_hour=0.0, end_hour=24.0, capacity_factor=gen_derate)

        # Apply mild sensor noise
        if sensor_sigma > 0.005:
            sm.add_sensor_noise(start_hour=0.0, end_hour=24.0, gaussian_sigma=sensor_sigma)

        meta = {
            "solar_scale": solar_scale,
            "wind_mean_speed": 8.0 + wind_mean_shift,
            "demand_scale": demand_scale,
            "initial_soc": initial_soc,
            "gen_derate": gen_derate,
            "sensor_sigma": sensor_sigma,
        }
        return sm, meta
