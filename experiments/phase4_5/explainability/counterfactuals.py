"""
experiments/phase4_5/explainability/counterfactuals.py
====================================================
GRIDGUARD AI — Phase 4.5 Controlled Counterfactual Simulator Experiments

Evaluates causal system impacts via controlled re-simulation against identical
paired weather/stochastic episode seeds (never inventing causal claims from
local policy gradients alone):
  1. no_battery_discharge : Disables battery discharge capability (charging allowed).
  2. grid_loss_full       : Forces full 24h utility grid outage & disables resync.
  3. solar_drop_30pct     : Reduces solar irradiance cloud factor by 30%.
  4. demand_surge_20pct   : Scales critical, flexible, and other demand by +20%.
  5. no_generator         : Locks backup diesel generator permanently OFF (0 MW).

Runs strictly across BOTH Intrinsic (adequacy_fallback=False) and Hybrid
(+Fallback, adequacy_fallback=True) modes so safety-layer reliance is explicit.
"""

import random
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional
import numpy as np
import pandas as pd

from rl_environment.microgrid_parallel_env import MicrogridParallelEnv
from rl_environment.economic_accounting import EconomicConfig, compute_episode_economics
from experiments.phase4.robustness.scenarios.robust_scenario_suite import HeldOutScenarios


COUNTERFACTUAL_SPECS: Dict[str, str] = {
    "no_battery_discharge": "Disable battery discharging (available_discharge_mw -> 0 MW; charging still allowed).",
    "grid_loss_full": "Force complete 24-hour utility grid outage and block synchronization.",
    "solar_drop_30pct": "Reduce solar generation cloud factor by 30% across all hours.",
    "demand_surge_20pct": "Increase nominal critical, flexible, and other demand by +20% across all hours.",
    "no_generator": "Lock backup diesel generator permanently OFF (capacity 0 MW, startup disabled).",
}


class CounterfactualScenarioProxy:
    """
    Non-mutating wrapper around a ScenarioManager / RobustScenarioManager that
    applies a named counterfactual perturbation at query time.
    """

    def __init__(self, base_scenario: Any, cf_name: Optional[str] = None):
        self._base = base_scenario
        self.cf_name = cf_name
        self.name = f"{getattr(base_scenario, 'name', 'Scenario')}[{cf_name or 'factual'}]"

    def __getattr__(self, item: str) -> Any:
        return getattr(self._base, item)

    def get_cloud_factor(self, hour: float) -> float:
        val = float(self._base.get_cloud_factor(hour))
        if self.cf_name == "solar_drop_30pct":
            return float(np.clip(val * 0.70, 0.0, 1.0))
        return val

    def is_grid_available(self, hour: float) -> bool:
        if self.cf_name == "grid_loss_full":
            return False
        return bool(self._base.is_grid_available(hour))

    def is_sync_verified(self, hour: float) -> bool:
        if self.cf_name == "grid_loss_full":
            return False
        return bool(self._base.is_sync_verified(hour))

    def get_demand_multipliers(self, hour: float) -> Dict[str, float]:
        if hasattr(self._base, "get_demand_multipliers"):
            base_m = dict(self._base.get_demand_multipliers(hour))
        else:
            base_m = {"critical": 1.0, "flexible": 1.0, "other": 1.0}
        if self.cf_name == "demand_surge_20pct":
            return {k: float(v) * 1.20 for k, v in base_m.items()}
        return base_m

    def get_generator_derating_factor(self, hour: float) -> float:
        if self.cf_name == "no_generator":
            return 0.0
        if hasattr(self._base, "get_generator_derating_factor"):
            return float(self._base.get_generator_derating_factor(hour))
        return 1.0


def _apply_post_reset_asset_locks(env: MicrogridParallelEnv, cf_name: Optional[str]) -> None:
    """
    Apply physical asset locks right after env.reset().
    Keep generator.capacity_mw > 0 so OperationalAdequacyChecker never divides by zero,
    while forcing state=OFF, can_start_now=False, and update()=0.0 MW.
    """
    if cf_name == "no_generator":
        env.generator.min_down_steps = 100000
        env.generator.request_start = lambda: False
        env.generator.update = lambda target_mw, dt_hours: 0.0
    elif cf_name == "no_battery_discharge":
        env.battery.available_discharge_mw = lambda dt_hours, emergency=False: 0.0


def _run_single_cf_episode(
    mappo_agent: Any,
    scenario_id: str,
    cf_name: Optional[str],
    adequacy_fallback: bool,
    eval_seed: int,
    forecast_models_dir: str,
    reward_cfg_path: str,
    econ_cfg: EconomicConfig,
    env: Optional[MicrogridParallelEnv] = None,
) -> Dict[str, Any]:
    heldouts = HeldOutScenarios.get_all_heldout_scenarios()
    if scenario_id not in heldouts:
        raise ValueError(f"Unknown held-out scenario for counterfactual: {scenario_id}")
    proxy_scen = CounterfactualScenarioProxy(heldouts[scenario_id], cf_name=cf_name)

    if env is None:
        env = MicrogridParallelEnv(
            scenario_id=scenario_id,
            forecast_mode="realistic",
            forecast_models_dir=forecast_models_dir,
            reward_config_path=reward_cfg_path,
            enable_adequacy_check=True,
            adequacy_fallback=bool(adequacy_fallback),
            enable_constraint_costs=True,
        )
    else:
        # Ensure clean physical models when reusing a cached environment across asset-lock counterfactuals
        env._init_physical_models()

    # MicrogridParallelEnv.reset does not reseed global np.random/random before generating
    # daily wind/cloud noise, so paired counterfactual comparisons must reseed explicitly.
    random.seed(int(eval_seed))
    np.random.seed(int(eval_seed))
    obs, _ = env.reset(seed=int(eval_seed), options={"scenario_obj": proxy_scen})
    _apply_post_reset_asset_locks(env, cf_name)

    total_reward = 0.0
    emergency_overrides = 0
    adequacy_interventions = 0
    done = False

    while env.agents and not done:
        actions, _ = mappo_agent.select_actions(obs, deterministic=True)
        obs, rewards, terms, truncs, infos = env.step(actions)
        total_reward += float(rewards.get("battery", 0.0))
        done = any(terms.values()) or any(truncs.values())

        bat_info = infos.get("battery", {})
        if bat_info.get("shield_diagnostics", {}).get("any_emergency", False):
            emergency_overrides += 1
        if bat_info.get("adequacy_diagnostics", {}).get("adequacy_intervention", False):
            adequacy_interventions += 1

    econ = compute_episode_economics(env.episode_history, dt_hours=env.dt_hours, config=econ_cfg)

    return {
        "total_reward": round(float(total_reward), 4),
        "critical_reliability_pct": round(float(econ["critical_reliability_pct"]), 4),
        "critical_ens_mwh": round(float(econ["critical_ens_mwh"]), 6),
        "total_ens_mwh": round(float(econ["total_ens_mwh"]), 6),
        "operating_cost_usd": round(float(econ["operating_cost_usd"]), 4),
        "true_social_cost_usd": round(float(econ["true_social_cost_usd"]), 4),
        "fuel_liters": round(float(econ["fuel_consumed_l"]), 4),
        "battery_throughput_mwh": round(float(econ["battery_throughput_mwh"]), 6),
        "emergency_overrides": int(emergency_overrides),
        "adequacy_interventions": int(adequacy_interventions),
    }


def run_counterfactual_suite(
    mappo_agent: Any,
    controller_name: str = "Bat-Only (Seed 202)",
    scenarios: Iterable[str] = ("scenario_h2", "scenario_h6"),
    eval_seed: int = 42,
    forecast_models_dir: str = "experiments/phase4/forecasting/models",
    reward_cfg_path: str = "experiments/phase4_4_4/configs/reward_v3_social.yaml",
    econ_cfg_path: Optional[str] = "experiments/phase4_4_4/configs/economics_v1.yaml",
    cf_names: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Execute factual + controlled counterfactual suite across all requested scenarios
    and both modes (Intrinsic vs Hybrid +Fallback) using paired seeds.
    """
    econ_cfg = EconomicConfig.from_yaml(econ_cfg_path)
    selected_cfs = list(cf_names) if cf_names is not None else list(COUNTERFACTUAL_SPECS.keys())
    for c in selected_cfs:
        if c not in COUNTERFACTUAL_SPECS:
            raise ValueError(f"Unsupported counterfactual '{c}'. Valid options: {list(COUNTERFACTUAL_SPECS)}")

    rows: List[Dict[str, Any]] = []
    modes = [("Intrinsic", False), ("Hybrid (+Fallback)", True)]
    env_cache: Dict[bool, MicrogridParallelEnv] = {}

    for scen in scenarios:
        for mode_label, fb_enabled in modes:
            if fb_enabled not in env_cache:
                env_cache[fb_enabled] = MicrogridParallelEnv(
                    scenario_id=scen,
                    forecast_mode="realistic",
                    forecast_models_dir=forecast_models_dir,
                    reward_config_path=reward_cfg_path,
                    enable_adequacy_check=True,
                    adequacy_fallback=bool(fb_enabled),
                    enable_constraint_costs=True,
                )
            cached_env = env_cache[fb_enabled]
            factual = _run_single_cf_episode(
                mappo_agent=mappo_agent,
                scenario_id=scen,
                cf_name=None,
                adequacy_fallback=fb_enabled,
                eval_seed=eval_seed,
                forecast_models_dir=forecast_models_dir,
                reward_cfg_path=reward_cfg_path,
                econ_cfg=econ_cfg,
                env=cached_env,
            )
            rows.append({
                "simulated_counterfactual": False,
                "counterfactual": "factual",
                "description": "Unperturbed held-out scenario baseline",
                "mode": mode_label,
                "controller": f"{controller_name} [{mode_label}]",
                "scenario": scen,
                "eval_seed": int(eval_seed),
                **factual,
                "delta_reward": 0.0,
                "delta_critical_reliability_pct": 0.0,
                "delta_critical_ens_mwh": 0.0,
                "delta_operating_cost_usd": 0.0,
                "delta_social_cost_usd": 0.0,
                "delta_fuel_liters": 0.0,
            })

            for cf_key in selected_cfs:
                cf_res = _run_single_cf_episode(
                    mappo_agent=mappo_agent,
                    scenario_id=scen,
                    cf_name=cf_key,
                    adequacy_fallback=fb_enabled,
                    eval_seed=eval_seed,
                    forecast_models_dir=forecast_models_dir,
                    reward_cfg_path=reward_cfg_path,
                    econ_cfg=econ_cfg,
                    env=cached_env,
                )
                rows.append({
                    "simulated_counterfactual": True,
                    "counterfactual": cf_key,
                    "description": COUNTERFACTUAL_SPECS[cf_key],
                    "mode": mode_label,
                    "controller": f"{controller_name} [{mode_label}]",
                    "scenario": scen,
                    "eval_seed": int(eval_seed),
                    **cf_res,
                    "delta_reward": round(cf_res["total_reward"] - factual["total_reward"], 4),
                    "delta_critical_reliability_pct": round(
                        cf_res["critical_reliability_pct"] - factual["critical_reliability_pct"], 4
                    ),
                    "delta_critical_ens_mwh": round(cf_res["critical_ens_mwh"] - factual["critical_ens_mwh"], 6),
                    "delta_operating_cost_usd": round(
                        cf_res["operating_cost_usd"] - factual["operating_cost_usd"], 4
                    ),
                    "delta_social_cost_usd": round(
                        cf_res["true_social_cost_usd"] - factual["true_social_cost_usd"], 4
                    ),
                    "delta_fuel_liters": round(cf_res["fuel_liters"] - factual["fuel_liters"], 4),
                })

    return pd.DataFrame(rows)
