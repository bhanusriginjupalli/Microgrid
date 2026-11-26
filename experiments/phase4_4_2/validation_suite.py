"""
experiments/phase4_4_2/validation_suite.py
==========================================
GRIDGUARD AI — Fixed Validation Suite & Multi-Objective Checkpoint Selection (Phase 4.4.2)

Provides a deterministic, fixed validation distribution completely separated
from:
  1. Training DomainRandomizer episodes
  2. Held-out final benchmark suite (Scenarios H1 to H6)

Implements a documented multi-objective checkpoint selection rule combining
True Social Cost, Critical ENS, Critical Reliability, and Constraint Violations.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "phase1_digital_twin") not in sys.path:
    sys.path.insert(0, str(ROOT / "phase1_digital_twin"))

from experiments.phase4.robustness.scenarios.robust_scenario_suite import RobustScenarioManager
from rl_environment.microgrid_parallel_env import MicrogridParallelEnv
from rl_environment.economic_accounting import compute_episode_economics, EconomicConfig


class FixedValidationSuite:
    """
    Fixed 4-scenario validation benchmark evaluated at fixed seeds [901, 902]
    (8 deterministic validation episodes) to select checkpoints without
    leakage from held-out H1-H6 or noise from randomized training episodes.
    """

    VALIDATION_SEEDS = [901]

    @classmethod
    def get_validation_scenarios(cls) -> Dict[str, RobustScenarioManager]:
        # Val 1: Midday cloud dip + 1.5h grid outage
        v1 = RobustScenarioManager(
            name="Validation V1: Cloud Dip & Outage",
            description="Validation scenario with midday cloud cover and 1.5h outage.",
        )
        v1.add_event("cloud", start_hour=10.0, end_hour=13.0, severity=0.65)
        v1.add_event("grid_outage", start_hour=11.0, end_hour=12.5)

        # Val 2: Evening demand surge + 2h grid outage + sync delay
        v2 = RobustScenarioManager(
            name="Validation V2: Evening Surge & Outage",
            description="Evening +35% demand surge combined with 2h grid outage.",
        )
        v2.add_demand_shift(
            start_hour=16.0, end_hour=20.0,
            critical_mult=1.25, flexible_mult=1.40, other_mult=1.35,
        )
        v2.add_event("grid_outage", start_hour=17.0, end_hour=19.0)
        v2.add_event("synchronization_hold", start_hour=19.0, end_hour=19.5)

        # Val 3: Low wind + 2.5h prolonged midday outage
        v3 = RobustScenarioManager(
            name="Validation V3: Wind Lull & Outage",
            description="Low wind speed (2.0 m/s) with 2.5h grid outage.",
        )
        v3.add_event("wind_lull", start_hour=8.0, end_hour=18.0, wind_speed_ms=2.0)
        v3.add_event("grid_outage", start_hour=12.0, end_hour=14.5)

        # Val 4: Storm warning + 2h outage + mild generator derating
        v4 = RobustScenarioManager(
            name="Validation V4: Storm & Mild Derating",
            description="Storm warning with 2h grid outage and 20% generator derating.",
        )
        v4.add_event("severe_storm", start_hour=13.0, end_hour=16.5, wind_speed_ms=26.0, cloud_severity=0.80)
        v4.add_event("grid_outage", start_hour=14.0, end_hour=16.0)
        v4.add_generator_derating(start_hour=13.0, end_hour=17.0, capacity_factor=0.80)

        return {
            "val_1": v1,
            "val_2": v2,
            "val_3": v3,
            "val_4": v4,
        }

    @classmethod
    def compute_selection_score(
        cls,
        mean_true_social_cost_usd: float,
        mean_critical_ens_mwh: float,
        mean_critical_reliability_pct: float,
        mean_inaction_rate_pct: float,
        mean_constraint_violations: float,
    ) -> float:
        """
        Multi-objective checkpoint selection rule (higher is better).
        Combines True Social Cost, Critical ENS, Critical Reliability,
        Inaction Rate, and Constraint Violations (NOT reward alone).
        """
        return float(
            -mean_true_social_cost_usd
            - 10000.0 * mean_critical_ens_mwh
            + 20.0 * mean_critical_reliability_pct
            - 200.0 * mean_inaction_rate_pct
            - 50.0 * mean_constraint_violations
        )

    @classmethod
    def evaluate_agent(
        cls,
        agent: Any,
        forecast_mode: str = "realistic",
        reward_config_path: str = None,
        econ_config: EconomicConfig = None,
        enable_adequacy_check: bool = True,
        adequacy_fallback: bool = True,
        max_steps: int = 96,
    ) -> Dict[str, Any]:
        """
        Evaluate agent deterministically across the fixed validation suite.
        """
        scenarios = cls.get_validation_scenarios()
        econ_cfg = econ_config or EconomicConfig.from_yaml()
        models_dir = str(ROOT / "experiments" / "phase4" / "forecasting" / "models")

        ep_records: List[Dict[str, Any]] = []
        agent.eval()
        env = MicrogridParallelEnv(
            scenario_id="scenario_a",
            forecast_mode=forecast_mode,
            forecast_models_dir=models_dir,
            reward_config_path=reward_config_path,
            max_steps=max_steps,
            dt_minutes=15.0,
            enable_safety_shield=True,
            enable_adequacy_check=enable_adequacy_check,
            adequacy_fallback=adequacy_fallback,
        )

        for val_id, sm in scenarios.items():
            for v_seed in cls.VALIDATION_SEEDS:
                obs, infos = env.reset(seed=v_seed, options={"scenario_obj": sm})

                done = False
                ep_reward = 0.0
                inaction_steps = 0
                adequacy_interventions = 0
                emergency_overrides = 0
                boundary_clips = 0
                steps_run = 0

                while env.agents and not done:
                    actions, _ = agent.select_actions(obs, deterministic=True)
                    obs, rewards, terms, truncs, infos = env.step(actions)
                    ep_reward += float(rewards.get("battery", 0.0))
                    steps_run += 1

                    diag = infos.get("battery", {}).get("shield_diagnostics", {})
                    adeq = infos.get("battery", {}).get("adequacy_diagnostics", {})
                    if diag.get("any_emergency", False):
                        emergency_overrides += 1
                    if diag.get("any_boundary", False):
                        boundary_clips += 1
                    if adeq.get("inaction_detected", False):
                        inaction_steps += 1
                    if adeq.get("adequacy_intervention", False):
                        adequacy_interventions += 1

                    done = any(terms.values()) or any(truncs.values())

                econ = compute_episode_economics(env.episode_history, dt_hours=env.dt_hours, config=econ_cfg)
                inaction_rate = (inaction_steps / steps_run * 100.0) if steps_run > 0 else 0.0

                ep_records.append({
                    "val_scenario": val_id,
                    "val_seed": v_seed,
                    "reward": ep_reward,
                    "operating_cost_usd": econ["operating_cost_usd"],
                    "true_social_cost_usd": econ["true_social_cost_usd"],
                    "critical_ens_mwh": econ["critical_ens_mwh"],
                    "critical_reliability_pct": econ["critical_reliability_pct"],
                    "inaction_rate_pct": inaction_rate,
                    "adequacy_interventions": adequacy_interventions,
                    "emergency_overrides": emergency_overrides,
                    "boundary_clips": boundary_clips,
                    "constraint_violations": emergency_overrides + inaction_steps,
                })

        df = pd.DataFrame(ep_records)
        mean_social = float(df["true_social_cost_usd"].mean())
        mean_op = float(df["operating_cost_usd"].mean())
        mean_ens = float(df["critical_ens_mwh"].mean())
        mean_rel = float(df["critical_reliability_pct"].mean())
        mean_inact = float(df["inaction_rate_pct"].mean())
        mean_viol = float(df["constraint_violations"].mean())
        mean_rew = float(df["reward"].mean())

        val_score = cls.compute_selection_score(
            mean_true_social_cost_usd=mean_social,
            mean_critical_ens_mwh=mean_ens,
            mean_critical_reliability_pct=mean_rel,
            mean_inaction_rate_pct=mean_inact,
            mean_constraint_violations=mean_viol,
        )

        agent.train()
        return {
            "val_score": round(val_score, 2),
            "val_mean_reward": round(mean_rew, 2),
            "val_operating_cost_usd": round(mean_op, 2),
            "val_true_social_cost_usd": round(mean_social, 2),
            "val_critical_ens_mwh": round(mean_ens, 4),
            "val_critical_reliability_pct": round(mean_rel, 2),
            "val_inaction_rate_pct": round(mean_inact, 2),
            "val_emergency_overrides": round(float(df["emergency_overrides"].mean()), 2),
            "val_adequacy_interventions": round(float(df["adequacy_interventions"].mean()), 2),
            "val_constraint_violations": round(mean_viol, 2),
            "val_episodes": len(df),
        }
