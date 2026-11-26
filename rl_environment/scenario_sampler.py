# GRIDGUARD AI
# Phase 2: Scenario Sampler & Curriculum Interface
# ============================================================
# Generates reproducible training scenarios:
#   - Direct access to benchmark Scenarios A through F
#   - Stochastic domain randomization for robust RL generalization
#   - Multi-level curriculum learning difficulty staging
# ============================================================

from typing import Optional, Dict, Any, Union, Tuple, List
import numpy as np

try:
    from scenario_manager import ScenarioManager
except ImportError:
    from phase1_digital_twin.scenario_manager import ScenarioManager


class ScenarioSampler:
    """
    Samples benchmark and procedurally randomized microgrid operating
    scenarios with stochastic weather fronts and transmission faults.
    """

    BENCHMARK_NAMES = [
        "scenario_a",
        "scenario_b",
        "scenario_c",
        "scenario_d",
        "scenario_e",
        "scenario_f",
    ]

    def __init__(self, seed: Optional[int] = 42):
        self.rng = np.random.default_rng(seed)
        self.difficulty_level = 1  # 1 (Normal) to 5 (Extreme Disturbance)

    def set_difficulty_level(self, level: int):
        """Configure curriculum difficulty: 1 (easiest) to 5 (most severe)."""
        self.difficulty_level = int(np.clip(level, 1, 5))

    def sample(
        self,
        scenario_id: Optional[Union[str, int]] = None,
        randomize: bool = False,
    ) -> Tuple[ScenarioManager, Dict[str, Any]]:
        """
        Sample a benchmark or procedurally generated scenario.

        Returns:
            (scenario_manager_instance, metadata_dict)
        """
        if scenario_id is not None and not randomize:
            return self._load_benchmark(scenario_id)

        # Procedural sampling based on curriculum difficulty
        meta = self._sample_domain_parameters()
        sm = ScenarioManager(
            name=f"Procedural Scenario (Diff {self.difficulty_level})",
            description=f"Procedural episode with storm_prob={meta['has_storm']} and outage_prob={meta['has_outage']}",
        )

        # Add storm if active in sample
        if meta["has_storm"]:
            sm.add_event(
                "severe_storm",
                start_hour=meta["storm_start_h"],
                end_hour=meta["storm_end_h"],
                wind_speed_ms=meta["storm_wind_ms"],
                cloud_severity=meta["storm_cloud_severity"],
            )

        # Add outage if active in sample
        if meta["has_outage"]:
            sm.add_event(
                "grid_outage",
                start_hour=meta["outage_start_h"],
                end_hour=meta["outage_end_h"],
            )
            # Add synchronization check delay (15-30 min)
            sm.add_event(
                "synchronization_hold",
                start_hour=meta["outage_end_h"],
                end_hour=meta["outage_end_h"] + 0.5,
            )

        return sm, meta

    def _sample_domain_parameters(self) -> Dict[str, Any]:
        """Sample randomized physical environmental parameters."""
        diff = self.difficulty_level

        # Probabilities scaled with difficulty
        storm_prob = min(0.9, 0.15 * diff)
        outage_prob = min(0.9, 0.20 * diff)

        has_storm = bool(self.rng.random() < storm_prob)
        has_outage = bool(self.rng.random() < outage_prob)

        # Solar & Wind scaling
        solar_scale = float(self.rng.uniform(0.75, 1.10))
        wind_mean_speed = float(self.rng.uniform(5.5, 11.5))
        demand_scale = float(self.rng.uniform(0.85, 1.15))
        initial_soc = float(self.rng.uniform(55.0, 85.0))

        # Outage timing
        outage_start = float(self.rng.uniform(9.0, 15.0))
        outage_duration = float(self.rng.uniform(1.0, 1.0 + 1.0 * diff))
        outage_end = min(22.0, outage_start + outage_duration)

        # Storm timing
        storm_start = float(self.rng.uniform(8.0, 13.0))
        storm_duration = float(self.rng.uniform(2.0, 2.0 + 0.8 * diff))
        storm_end = min(22.0, storm_start + storm_duration)
        storm_wind = float(self.rng.uniform(26.0, 30.0))  # Exceeds 25 m/s cut-out
        storm_cloud = float(self.rng.uniform(0.75, 0.95))

        return {
            "difficulty": diff,
            "has_storm": has_storm,
            "has_outage": has_outage,
            "solar_scale": solar_scale,
            "wind_mean_speed": wind_mean_speed,
            "demand_scale": demand_scale,
            "initial_soc": initial_soc,
            "outage_start_h": outage_start,
            "outage_end_h": outage_end,
            "storm_start_h": storm_start,
            "storm_end_h": storm_end,
            "storm_wind_ms": storm_wind,
            "storm_cloud_severity": storm_cloud,
        }

    def _load_benchmark(self, scenario_id: Union[str, int]) -> Tuple[ScenarioManager, Dict[str, Any]]:
        """Load deterministic benchmark scenario A through F."""
        name_str = str(scenario_id).lower()

        if "a" in name_str or name_str == "0" or "normal" in name_str:
            sm = ScenarioManager.scenario_a_normal()
            tag = "scenario_a"
        elif "b" in name_str or name_str == "1" or "collapse" in name_str:
            sm = ScenarioManager.scenario_b_solar_collapse()
            tag = "scenario_b"
        elif "c" in name_str or name_str == "2" or "grid_outage" in name_str:
            sm = ScenarioManager.scenario_c_grid_outage()
            tag = "scenario_c"
        elif "d" in name_str or name_str == "3" or "storm" in name_str:
            sm = ScenarioManager.scenario_d_severe_storm()
            tag = "scenario_d"
        elif "e" in name_str or name_str == "4" or "prolonged" in name_str:
            sm = ScenarioManager.scenario_e_prolonged_outage()
            tag = "scenario_e"
        elif "f" in name_str or name_str == "5" or "restoration" in name_str:
            sm = ScenarioManager.scenario_f_grid_restoration()
            tag = "scenario_f"
        else:
            sm = ScenarioManager.scenario_a_normal()
            tag = "scenario_a"

        meta = {
            "benchmark_id": tag,
            "name": sm.name,
            "description": sm.description,
            "initial_soc": 80.0,
            "solar_scale": 1.0,
            "demand_scale": 1.0,
        }
        return sm, meta


if __name__ == "__main__":
    sampler = ScenarioSampler(seed=123)
    print("GRIDGUARD AI - Scenario Sampler Self-Test")
    print("=" * 50)

    # Test loading benchmark C
    sm_c, meta_c = sampler.sample("scenario_c")
    print(f"Loaded: {meta_c['name']} | Events: {len(sm_c.events)}")
    assert len(sm_c.events) > 0

    # Test procedural generation across curriculum difficulty levels
    for diff in [1, 3, 5]:
        sampler.set_difficulty_level(diff)
        sm_rand, meta_rand = sampler.sample(randomize=True)
        print(f"Sampled Diff {diff}: Storm={meta_rand['has_storm']}, Outage={meta_rand['has_outage']} | Initial SOC={meta_rand['initial_soc']:.1f}%")

    print("Scenario Sampler self-test passed successfully.")
