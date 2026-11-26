"""
mappo/evaluator.py
============================================================
Deterministic evaluation of trained MAPPO policies across
the 6 benchmark disturbance scenarios (A through F).
============================================================
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd

from rl_environment.microgrid_parallel_env import MicrogridParallelEnv
from mappo.mappo_agent import MAPPOAgent


class MAPPOEvaluator:
    """Evaluates MAPPO agents deterministically across benchmark scenarios."""

    def __init__(
        self,
        agent: MAPPOAgent,
        reward_config_path: Optional[str] = None,
        forecast_mode: Optional[str] = None,
        forecast_models_dir: Optional[str] = None,
    ):
        self.agent = agent
        self.reward_config_path = reward_config_path
        self.forecast_mode = forecast_mode
        self.forecast_models_dir = forecast_models_dir

    def evaluate_scenario(
        self,
        scenario_id: str,
        num_episodes: int = 3,
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        """Run multiple evaluation episodes for a given scenario."""
        env = MicrogridParallelEnv(
            scenario_id=scenario_id,
            max_steps=96,
            dt_minutes=15.0,
            enable_safety_shield=True,
            reward_config_path=self.reward_config_path,
            forecast_mode=self.forecast_mode,
            forecast_models_dir=self.forecast_models_dir,
        )

        episode_rewards = []
        episode_reliabilities = []
        episode_crit_ens = []
        episode_flex_ens = []
        episode_costs = []
        episode_fuels = []
        episode_curts = []
        episode_interventions = []

        last_df = None

        for ep in range(num_episodes):
            obs, infos = env.reset(seed=42 + ep)
            ep_reward = 0.0
            step_records = []

            while env.agents:
                actions, _ = self.agent.select_actions(obs, deterministic=deterministic)
                next_obs, rewards, terminations, truncations, next_infos = env.step(actions)

                last_step = env.episode_history[-1]
                ep_reward += rewards.get("battery", 0.0)

                rec = {
                    "step": last_step["step"],
                    "hour": last_step["hour"],
                    "scenario": scenario_id,
                    "reward_step": rewards.get("battery", 0.0),
                    "battery_soc": last_step["battery_soc"],
                    "battery_mw": last_step["battery_mw"],
                    "generator_mw": last_step["generator_mw"],
                    "grid_import_mw": last_step["grid_import_mw"],
                    "solar_mw": last_step["solar_mw"],
                    "wind_mw": last_step["wind_mw"],
                    "nominal_total_mw": last_step["nominal_total_mw"],
                    "served_total_mw": last_step["served_total_mw"],
                    "nominal_critical_mw": last_step["nominal_critical_mw"],
                    "served_critical_mw": last_step["served_critical_mw"],
                    "shed_critical_mw": last_step["shed_critical_mw"],
                    "nominal_flexible_mw": last_step["nominal_flexible_mw"],
                    "served_flexible_mw": last_step["served_flexible_mw"],
                    "shed_flexible_mw": last_step["shed_flexible_mw"],
                    "step_fuel_l": last_step["step_fuel_l"],
                    "solar_curtail_mw": max(0.0, last_step["solar_available_mw"] - last_step["solar_mw"]),
                    "wind_curtail_mw": max(0.0, last_step["wind_available_mw"] - last_step["wind_mw"]),
                    "shield_intervention": last_step["any_intervention"],
                }
                step_records.append(rec)

                obs = next_obs
                infos = next_infos

            df = pd.DataFrame(step_records)
            last_df = df
            dt_h = env.dt_hours

            tot_crit_mwh = df["nominal_critical_mw"].sum() * dt_h
            served_crit_mwh = df["served_critical_mw"].sum() * dt_h
            shed_crit_mwh = df["shed_critical_mw"].sum() * dt_h
            shed_flex_mwh = df["shed_flexible_mw"].sum() * dt_h
            tot_fuel_l = df["step_fuel_l"].sum()
            curt_mwh = (df["solar_curtail_mw"].sum() + df["wind_curtail_mw"].sum()) * dt_h
            interventions = int(df["shield_intervention"].sum())

            import_mwh = df["grid_import_mw"].clip(lower=0.0).sum() * dt_h
            export_mwh = (-df["grid_import_mw"].clip(upper=0.0)).sum() * dt_h
            operating_cost = (import_mwh * 120.0) - (export_mwh * 40.0) + (tot_fuel_l * 1.50)

            rel_pct = (served_crit_mwh / tot_crit_mwh * 100.0) if tot_crit_mwh > 0 else 100.0

            episode_rewards.append(ep_reward)
            episode_reliabilities.append(rel_pct)
            episode_crit_ens.append(shed_crit_mwh)
            episode_flex_ens.append(shed_flex_mwh)
            episode_costs.append(operating_cost)
            episode_fuels.append(tot_fuel_l)
            episode_curts.append(curt_mwh)
            episode_interventions.append(interventions)

        return {
            "scenario": scenario_id,
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "critical_reliability_pct": float(np.mean(episode_reliabilities)),
            "critical_ens_mwh": float(np.mean(episode_crit_ens)),
            "flexible_ens_mwh": float(np.mean(episode_flex_ens)),
            "operating_cost_usd": float(np.mean(episode_costs)),
            "fuel_consumed_l": float(np.mean(episode_fuels)),
            "renewable_curtailment_mwh": float(np.mean(episode_curts)),
            "shield_interventions": int(np.mean(episode_interventions)),
            "trajectory_df": last_df,
        }

    def evaluate_all_scenarios(
        self,
        num_episodes_per_scenario: int = 3,
    ) -> List[Dict[str, Any]]:
        """Evaluate across Scenarios A through F."""
        scenarios = [
            "scenario_a",
            "scenario_b",
            "scenario_c",
            "scenario_d",
            "scenario_e",
            "scenario_f",
        ]
        results = []
        for sc in scenarios:
            res = self.evaluate_scenario(sc, num_episodes=num_episodes_per_scenario)
            results.append(res)
        return results
