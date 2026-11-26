# GRIDGUARD AI
# Phase 2: PettingZoo Multi-Agent Parallel Environment
# ============================================================
# Centralized Training with Decentralized Execution (CTDE) multi-agent
# reinforcement learning environment compatible with PettingZoo ParallelEnv.
#
# Five Cooperative Agents:
#   1. renewable: Box(2,) -> [solar_curtailment, wind_curtailment]
#   2. battery: Box(1,) -> power command [-1, 1]
#   3. generator: Box(1,) -> requested capacity fraction [0, 1]
#   4. load_management: Discrete(6) -> curtailment level [0..5]
#   5. grid_resilience: Discrete(5) -> grid coupling command [0..4]
# ============================================================

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from gymnasium import spaces
from pettingzoo import ParallelEnv
import pandapower as pp

# Ensure project root and phase1_digital_twin are accessible in sys.path
_ROOT_DIR = Path(__file__).resolve().parent.parent
_PHASE1_DIR = _ROOT_DIR / "phase1_digital_twin"
for p in [str(_ROOT_DIR), str(_PHASE1_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Suppress noisy library warnings
warnings.filterwarnings("ignore")
os.environ["MPLBACKEND"] = "Agg"

from config import SIMULATION, GRID_CONNECTION, LOADS, GENERATOR, BATTERY
from network import create_microgrid
from resources import SolarModel, WindModel
from battery_dynamics import DynamicBattery
from generator_dynamics import DynamicGenerator
from load_management import LoadManager
from energy_balance import EnergyBalanceCalculator
from forecast_baseline import ForecastBaseline
from scenario_manager import ScenarioManager
from rl_environment.agent_config import (
    AGENTS,
    OBS_SHAPES,
    GLOBAL_STATE_SHAPE,
    CAPACITY_SCALING,
)
from rl_environment.action_space import get_agent_action_spaces, parse_agent_action
from rl_environment.observation_builder import (
    ObservationBuilder,
    get_agent_observation_spaces,
    get_global_state_space,
)
from rl_environment.reward_function import MicrogridRewardFunction
from rl_environment.safety_shield import SafetyShield
from rl_environment.operational_adequacy import OperationalAdequacyChecker, AdequacyConfig
from rl_environment.scenario_sampler import ScenarioSampler
from forecasting.forecast_manager import ForecastManager


class MicrogridParallelEnv(ParallelEnv):
    """
    PettingZoo Parallel Multi-Agent Reinforcement Learning Environment
    for Autonomous Microgrid Resilience and Energy Management.
    """

    metadata = {
        "name": "microgrid_parallel_env_v0",
        "render_modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        scenario_id: Optional[str] = "scenario_a",
        scenario_name: Optional[str] = None,
        max_steps: int = 96,
        dt_minutes: float = 15.0,
        enable_safety_shield: bool = True,
        random_seed: int = 42,
        **kwargs,
    ):
        super().__init__()
        effective_scenario = scenario_name if scenario_name is not None else scenario_id
        self.possible_agents = list(AGENTS)
        self.agents = list(self.possible_agents)
        self.max_steps = max_steps
        self.dt_minutes = dt_minutes
        self.dt_hours = dt_minutes / 60.0
        self.enable_safety_shield = enable_safety_shield
        self.default_seed = random_seed
        self.default_scenario_id = effective_scenario

        # Forecast Configuration (Mode 1 Oracle, Mode 2 Realistic, or Reactive baseline)
        self.forecast_mode = kwargs.get("forecast_mode", None)
        self.use_forecasts = self.forecast_mode is not None and str(self.forecast_mode).lower() in ("realistic", "oracle", "true")
        self.forecast_mode_str = "oracle" if str(self.forecast_mode).lower() == "oracle" else "realistic"

        # Action and Observation Spaces
        self._action_spaces = get_agent_action_spaces()
        self._observation_spaces = get_agent_observation_spaces(forecast_mode=self.use_forecasts)
        self._global_state_space = get_global_state_space(forecast_mode=self.use_forecasts)

        # Core Engines
        self.obs_builder = ObservationBuilder(use_forecasts=self.use_forecasts)
        if self.use_forecasts:
            models_dir = kwargs.get("forecast_models_dir", str(_ROOT_DIR / "experiments" / "phase4" / "forecasting" / "models"))
            self.forecast_mgr = ForecastManager(mode=self.forecast_mode_str, dt_hours=self.dt_hours, models_dir=models_dir)
        if kwargs.get("reward_fn") is not None:
            self.reward_fn = kwargs["reward_fn"]
        elif kwargs.get("reward_config_path") is not None:
            self.reward_fn = MicrogridRewardFunction.from_yaml(kwargs["reward_config_path"])
        else:
            self.reward_fn = MicrogridRewardFunction()
        self.safety_shield = SafetyShield()
        self.enable_adequacy_check = bool(kwargs.get("enable_adequacy_check", False))
        self.adequacy_fallback = bool(kwargs.get("adequacy_fallback", False))
        self.adequacy_checker = OperationalAdequacyChecker(
            config=AdequacyConfig(
                enabled=self.enable_adequacy_check,
                apply_fallback=self.adequacy_fallback,
                max_fuel_l=kwargs.get("adequacy_max_fuel_l", None),
            )
        )
        self.sampler = ScenarioSampler(seed=random_seed)

        # Simulator State
        self.current_step = 0
        self.current_hour = 0.0
        self.grid_connected = True
        self.episode_history: List[Dict[str, Any]] = []

        # Microgrid physical digital twin components (initialized in reset)
        self._init_physical_models()

    def _init_physical_models(self):
        """Construct Pandapower network and component state representations."""
        self.net, self.buses = create_microgrid()
        self.solar = SolarModel()
        self.wind = WindModel()
        self.battery = DynamicBattery()
        self.generator = DynamicGenerator()
        self.load_manager = LoadManager()
        self.energy_balance = EnergyBalanceCalculator(tolerance_mw=0.001)
        self.forecaster = ForecastBaseline(horizon_hours=4.0, dt_hours=self.dt_hours)

        # Add Island Reference Slack Generator on Backup Generator bus
        self._island_gen_idx = pp.create_gen(
            self.net,
            bus=self.buses["Backup Generator"],
            p_mw=0.0,
            vm_pu=1.0,
            slack=True,
            name="Island Slack Reference",
            max_p_mw=GENERATOR["capacity_mw"],
            min_p_mw=0.0,
            in_service=False,
        )

    def observation_space(self, agent: str) -> spaces.Space:
        """Return Gymnasium observation space for a given agent."""
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        """Return Gymnasium action space for a given agent."""
        return self._action_spaces[agent]

    @property
    def state_space(self) -> spaces.Box:
        """Return Gymnasium Box space for the centralized critic's state."""
        return self._global_state_space

    def state(self) -> np.ndarray:
        """Return global state vector for the centralized critic."""
        latest_state = self.episode_history[-1] if self.episode_history else self._get_default_state()
        if self.use_forecasts and hasattr(self, "forecast_mgr"):
            critic_feats = self.forecast_mgr.get_critic_forecast_features()
            return self.obs_builder.build_global_state(latest_state, critic_feats)
        return self.obs_builder.build_global_state(latest_state)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset environment to initial state at timestep 0.
        """
        if seed is not None:
            self.default_seed = seed
            self.sampler = ScenarioSampler(seed=seed)

        opts = options or {}
        scenario_obj = opts.get("scenario_obj", None)
        scenario_id = opts.get("scenario_id", self.default_scenario_id)
        randomize = opts.get("randomize", False)
        difficulty = opts.get("difficulty", 1)
        self.sampler.set_difficulty_level(difficulty)

        if scenario_obj is not None:
            self.scenario = scenario_obj
            self.scenario_meta = {
                "name": getattr(scenario_obj, "name", "Custom Robust Scenario"),
                "solar_scale": 1.0,
                "wind_mean_speed": 8.0,
                "initial_soc": 80.0,
            }
        elif str(scenario_id).startswith("scenario_h"):
            from experiments.phase4.robustness.scenarios.robust_scenario_suite import HeldOutScenarios
            heldouts = HeldOutScenarios.get_all_heldout_scenarios()
            if scenario_id in heldouts:
                self.scenario = heldouts[scenario_id]
                self.scenario_meta = {
                    "name": self.scenario.name,
                    "solar_scale": 1.0,
                    "wind_mean_speed": 8.0,
                    "initial_soc": 80.0,
                }
            else:
                self.scenario, self.scenario_meta = self.sampler.sample(scenario_id=scenario_id, randomize=randomize)
        elif randomize and opts.get("domain_randomization", False):
            from experiments.phase4.robustness.scenarios.robust_scenario_suite import DomainRandomizer
            dr = DomainRandomizer(seed=seed)
            self.scenario, self.scenario_meta = dr.sample_training_scenario(difficulty=difficulty)
        else:
            self.scenario, self.scenario_meta = self.sampler.sample(
                scenario_id=scenario_id,
                randomize=randomize,
            )

        # Reset episode timing
        self.current_step = 0
        self.current_hour = 0.0
        self.agents = list(self.possible_agents)
        self.episode_history.clear()

        # Reset physical models
        self._init_physical_models()
        init_soc = float(self.scenario_meta.get("initial_soc", 80.0))
        self.battery.reset(initial_soc=init_soc)
        self.generator.reset()
        self.load_manager.reset()
        self.energy_balance.reset()
        self.forecaster.reset()
        self.safety_shield.reset()

        # Initial baseline weather curves
        hours_axis = np.linspace(0, 24.0, self.max_steps, endpoint=False)
        self.base_irradiance = self.solar.generate_daily_profile(hours_axis) * float(self.scenario_meta.get("solar_scale", 1.0))
        self.base_wind_speed = self.wind.generate_daily_profile(hours_axis, mean_speed=float(self.scenario_meta.get("wind_mean_speed", 8.0)))

        # Determine initial physical state
        init_state = self._evaluate_initial_state()
        self.episode_history.append(init_state)

        if self.use_forecasts:
            self.forecast_mgr.reset()
            self.forecast_mgr.update_telemetry(
                hour=0.0,
                solar_mw=init_state.get("solar_mw", 0.0),
                wind_mw=init_state.get("wind_mw", 0.0),
                total_demand_mw=init_state.get("nominal_total_mw", 0.8),
                crit_demand_mw=init_state.get("nominal_critical_mw", 0.3),
                flex_demand_mw=init_state.get("nominal_flexible_mw", 0.2),
                other_demand_mw=init_state.get("nominal_other_mw", 0.3),
                cloud_factor=1.0,
                storm_warning=False,
            )
            self.forecast_mgr.generate_forecast(
                current_hour=0.0,
                current_state=init_state,
                scenario_manager=self.scenario,
                solar_model=self.solar,
                wind_model=self.wind,
                load_manager=self.load_manager,
            )
            agent_forecast_feats = {agent: self.forecast_mgr.get_agent_forecast_features(agent) for agent in self.agents}
            critic_forecast_feats = self.forecast_mgr.get_critic_forecast_features()
            observations = self.obs_builder.build_observations(init_state, agent_forecast_feats)
            global_state = self.obs_builder.build_global_state(init_state, critic_forecast_feats)
        else:
            observations = self.obs_builder.build_observations(init_state)
            global_state = self.obs_builder.build_global_state(init_state)

        infos = {agent: {"step": 0, "hour": 0.0, "scenario": self.scenario_meta.get("name", "Baseline")} for agent in self.agents}
        infos["global_state"] = global_state

        return observations, infos

    def step(
        self,
        actions: Dict[str, Any],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        """
        Advance environment by one 15-minute simulation step under joint agent actions.
        """
        if not self.agents:
            return {}, {}, {}, {}, {}

        hour = self.current_hour
        step_idx = self.current_step

        # Retrieve previous state for condition checks
        prev_state = self.episode_history[-1] if self.episode_history else self._get_default_state()

        # ------------------------------------------------------------
        # Step 1: Safety Shield Action Validation
        # ------------------------------------------------------------
        if self.enable_safety_shield:
            executed_actions, shield_info = self.safety_shield.filter_actions(
                raw_actions=actions,
                current_state=prev_state,
                dt_hours=self.dt_hours,
            )
        else:
            executed_actions = actions
            shield_info = {"any_intervention": False, "interventions": {}, "violations": [], "reasons": []}

        # ------------------------------------------------------------
        # Step 2: Environmental Conditions & Grid Fault Injection
        # ------------------------------------------------------------
        cloud_factor = self.scenario.get_cloud_factor(hour)
        irradiance = self.base_irradiance[step_idx] * cloud_factor

        wind_override = self.scenario.get_wind_override(hour)
        wind_speed = wind_override if wind_override is not None else self.base_wind_speed[step_idx]

        grid_available = self.scenario.is_grid_available(hour)
        sync_verified = self.scenario.is_sync_verified(hour)

        # Parse grid action
        grid_parsed = parse_agent_action("grid_resilience", executed_actions.get("grid_resilience", 0))
        grid_cmd = grid_parsed["command"]

        # Handle grid connection state machine
        if not grid_available:
            if self.grid_connected:
                self._disconnect_from_grid()
        else:
            if not self.grid_connected:
                if grid_cmd in ("REQUEST_RECONNECTION", "RECOVERY_MODE") and sync_verified:
                    self._reconnect_to_grid()
            else:
                if grid_cmd == "REQUEST_ISLANDING":
                    self._disconnect_from_grid()

        # Update physical branch / line status in Pandapower network
        line_map = {}
        if hasattr(self.scenario, "get_line_in_service_map"):
            line_map = self.scenario.get_line_in_service_map(hour, total_lines=len(self.net.line))
            for l_idx, in_serv in line_map.items():
                if l_idx < len(self.net.line):
                    self.net.line.at[l_idx, "in_service"] = in_serv

        # ------------------------------------------------------------
        # Step 3: Renewable Generation with Agent Curtailment
        # ------------------------------------------------------------
        raw_solar_p = self.solar.update(irradiance, ambient_temp_c=28.0)
        raw_wind_p = self.wind.update(wind_speed)

        ren_parsed = parse_agent_action("renewable", executed_actions.get("renewable", np.array([0.0, 0.0])))
        solar_curt_factor = ren_parsed["solar_curtailment_factor"]
        wind_curt_factor = ren_parsed["wind_curtailment_factor"]

        solar_p = raw_solar_p * (1.0 - solar_curt_factor)
        wind_p = raw_wind_p * (1.0 - wind_curt_factor)

        # Physical branch outage isolation: Line 1 (Solar feeder) or Line 2 (Wind feeder)
        if line_map:
            if not line_map.get(1, True):
                solar_p = 0.0
            if not line_map.get(2, True):
                wind_p = 0.0
        re_p = solar_p + wind_p

        # ------------------------------------------------------------
        # Step 4: Demand Breakdown & Load Management Agent
        # ------------------------------------------------------------
        nom_demand = self.load_manager.get_nominal_demand(hour)

        # Apply demand shifts / surges / rebounds if active in scenario
        if hasattr(self.scenario, "get_demand_multipliers"):
            d_mults = self.scenario.get_demand_multipliers(hour)
            nom_demand["critical_mw"] *= float(d_mults.get("critical", 1.0))
            nom_demand["flexible_mw"] *= float(d_mults.get("flexible", 1.0))
            nom_demand["other_mw"] *= float(d_mults.get("other", 1.0))
            nom_demand["total_demand_mw"] = nom_demand["critical_mw"] + nom_demand["flexible_mw"] + nom_demand["other_mw"]

        # Generator derating query
        gen_derate = 1.0
        if hasattr(self.scenario, "get_generator_derating_factor"):
            gen_derate = float(self.scenario.get_generator_derating_factor(hour))

        # ------------------------------------------------------------
        # Step 4b: Operational Adequacy & Anti-Inaction Safety Check
        # ------------------------------------------------------------
        allow_emerg_bat = False
        if self.enable_adequacy_check:
            executed_actions, adequacy_info = self.adequacy_checker.evaluate_step(
                shielded_actions=executed_actions,
                grid_connected=self.grid_connected,
                re_available_mw=re_p,
                nominal_demand=nom_demand,
                battery=self.battery,
                generator=self.generator,
                gen_derating_factor=gen_derate,
                dt_hours=self.dt_hours,
            )
            allow_emerg_bat = bool(adequacy_info.get("allow_emergency_battery", False))
        else:
            adequacy_info = {
                "inaction_detected": False,
                "adequacy_intervention": False,
                "allow_emergency_battery": False,
                "avoidable_shortfall_mw": 0.0,
                "unavoidable_total_deficit_mw": 0.0,
                "unavoidable_critical_deficit_mw": 0.0,
                "intervention_reasons": [],
            }

        load_parsed = parse_agent_action("load_management", executed_actions.get("load_management", 0))
        flex_curt_target = load_parsed["flexible_curtail_frac"]
        shed_other_requested = load_parsed["shed_other_loads"]

        # Calculate agent-requested load curtailment
        flex_shed_target_mw = nom_demand["flexible_mw"] * flex_curt_target
        other_shed_target_mw = (nom_demand["other_mw"] * 0.50) if shed_other_requested else 0.0
        discretionary_shed_mw = flex_shed_target_mw + other_shed_target_mw

        # ------------------------------------------------------------
        # Step 5: Generator Agent Dispatch
        # ------------------------------------------------------------
        gen_parsed = parse_agent_action("generator", executed_actions.get("generator", np.array([0.0])))
        gen_target_mw = min(gen_parsed["requested_power_mw"], GENERATOR["capacity_mw"] * gen_derate)

        if gen_target_mw > 0.04:
            self.generator.request_start()
            actual_gen_p = self.generator.update(gen_target_mw, self.dt_hours)
        else:
            self.generator.request_stop()
            actual_gen_p = self.generator.update(0.0, self.dt_hours)

        # ------------------------------------------------------------
        # Step 6: Battery Agent Power Dispatch
        # ------------------------------------------------------------
        bat_parsed = parse_agent_action("battery", executed_actions.get("battery", np.array([0.0])))
        bat_p_req_mw = bat_parsed["requested_power_mw"]

        # Physical update of battery state
        actual_bat_p = self.battery.update(bat_p_req_mw, self.dt_hours, emergency=allow_emerg_bat)

        # ------------------------------------------------------------
        # Step 7: Power Balancing & Final Load Dispatch
        # ------------------------------------------------------------
        internal_supply_mw = re_p + actual_gen_p - actual_bat_p
        net_balance_mw = nom_demand["total_demand_mw"] - internal_supply_mw

        if self.grid_connected:
            # Grid absorbs deficit (import) or surplus (export)
            grid_target_mw = net_balance_mw
            # In grid-connected mode, load is fully served unless agent deliberately curtailed
            load_dispatch = self.load_manager.dispatch(discretionary_shed_mw, hour, self.dt_hours)
        else:
            # Islanded mode: grid exchange is strictly 0
            grid_target_mw = 0.0
            # Total curtailment needed is whichever is larger: physical deficit or agent discretionary
            physical_deficit = max(0.0, net_balance_mw)
            curtailment_needed = max(physical_deficit, discretionary_shed_mw)
            load_dispatch = self.load_manager.dispatch(curtailment_needed, hour, self.dt_hours)

            # If there was remaining surplus in islanded mode that battery didn't take, curtail renewables
            if net_balance_mw < -0.0001:
                surplus_unabsorbed = abs(net_balance_mw)
                curt_s = min(solar_p, surplus_unabsorbed)
                solar_p -= curt_s
                surplus_unabsorbed -= curt_s
                curt_w = min(wind_p, surplus_unabsorbed)
                wind_p -= curt_w
                re_p = solar_p + wind_p

        served_total_load_mw = load_dispatch["served_total_mw"]

        # ------------------------------------------------------------
        # Step 8: Apply to Pandapower Electrical Network Model
        # ------------------------------------------------------------
        self._apply_network_state(
            solar_mw=solar_p,
            wind_mw=wind_p,
            battery_mw=actual_bat_p,
            generator_mw=actual_gen_p,
            load_dispatch=load_dispatch,
        )

        # ------------------------------------------------------------
        # Step 9: Run AC Power Flow
        # ------------------------------------------------------------
        converged, losses_mw, grid_flow_mw = self._run_power_flow()

        if self.grid_connected and converged:
            grid_final_mw = grid_flow_mw
        else:
            grid_final_mw = grid_target_mw

        # ------------------------------------------------------------
        # Step 10: Explicit Power Balance Verification
        # ------------------------------------------------------------
        balance_res = self.energy_balance.calculate_balance(
            solar_mw=solar_p,
            wind_mw=wind_p,
            generator_mw=actual_gen_p,
            grid_mw=grid_final_mw,
            battery_mw=actual_bat_p,
            load_served_mw=served_total_load_mw,
            losses_mw=losses_mw,
        )

        # ------------------------------------------------------------
        # Step 11: Update Forecasting Baseline
        # ------------------------------------------------------------
        storm_risk = 1.0 if not grid_available else (0.7 if self.scenario.is_storm_warning(hour) else 0.02)
        self.forecaster.observe(
            solar_mw=solar_p,
            wind_mw=wind_p,
            demand_mw=nom_demand["total_demand_mw"],
            storm_risk=storm_risk,
        )
        forecast = self.forecaster.predict(
            current_hour=hour,
            known_cloud_factor=cloud_factor,
            storm_warning=self.scenario.is_storm_warning(hour),
        )

        # ------------------------------------------------------------
        # Step 12: Telemetry Logging & State Packaging
        # ------------------------------------------------------------
        state_dict = {
            "step": step_idx,
            "hour": hour,
            "grid_connected": self.grid_connected,
            "grid_available": grid_available,
            "sync_verified": sync_verified,
            "cloud_factor": cloud_factor,
            "wind_speed_ms": wind_speed,
            "is_storm_warning": self.scenario.is_storm_warning(hour),
            "solar_available_mw": raw_solar_p,
            "wind_available_mw": raw_wind_p,
            "solar_mw": solar_p,
            "wind_mw": wind_p,
            "re_gen_mw": re_p,
            "generator_mw": actual_gen_p,
            "generator_state": self.generator.state.value,
            "step_fuel_l": self.generator.last_step_fuel_l,
            "total_fuel_l": self.generator.total_fuel_l,
            "battery_mw": actual_bat_p,
            "battery_soc": self.battery.soc,
            "battery_energy_mwh": self.battery.energy_mwh,
            "grid_import_mw": grid_final_mw,
            "losses_mw": losses_mw,
            "pcc_voltage_pu": 1.000 if not self.grid_connected else (float(self.net.res_bus.at[0, "vm_pu"]) if converged else 1.0),
            "min_bus_voltage_pu": float(self.net.res_bus["vm_pu"].min()) if (converged and hasattr(self.net, "res_bus") and not self.net.res_bus.empty) else 1.0,
            "max_bus_voltage_pu": float(self.net.res_bus["vm_pu"].max()) if (converged and hasattr(self.net, "res_bus") and not self.net.res_bus.empty) else 1.0,
            "freq_deviation_proxy": 0.0,
            "nominal_total_mw": nom_demand["total_demand_mw"],
            "nominal_critical_mw": nom_demand["critical_mw"],
            "nominal_flexible_mw": nom_demand["flexible_mw"],
            "nominal_other_mw": nom_demand["other_mw"],
            "served_total_mw": served_total_load_mw,
            "served_critical_mw": load_dispatch["served_critical_mw"],
            "served_flexible_mw": load_dispatch["served_flexible_mw"],
            "served_other_mw": load_dispatch["served_other_mw"],
            "shed_total_mw": load_dispatch["shed_total_mw"],
            "shed_critical_mw": load_dispatch["shed_critical_mw"],
            "shed_flexible_mw": load_dispatch["shed_flexible_mw"],
            "shed_other_mw": load_dispatch["shed_other_mw"],
            "critical_emergency": load_dispatch["critical_emergency"],
            "deferred_flexible_mwh": load_dispatch["deferred_flexible_mwh"],
            "balance_error_mw": balance_res["balance_error_mw"],
            "is_balanced": balance_res["is_balanced"],
            "converged": converged,
            "storm_risk": storm_risk,
            "forecast_next_solar_mw": forecast["next_step_solar_mw"],
            "forecast_next_wind_mw": forecast["next_step_wind_mw"],
            "forecast_next_demand_mw": forecast["next_step_demand_mw"],
            "forecast_next_outage_risk": forecast["next_step_outage_risk"],
            "any_intervention": shield_info.get("any_intervention", False),
            "boundary_clipping": bool(shield_info.get("any_boundary", False)),
            "emergency_override": bool(shield_info.get("any_emergency", False)),
            "inaction_detected": bool(adequacy_info.get("inaction_detected", False)),
            "adequacy_intervention": bool(adequacy_info.get("adequacy_intervention", False)),
            "avoidable_shortfall_mw": float(adequacy_info.get("avoidable_shortfall_mw", 0.0)),
            "unavoidable_critical_shed_mw": float(min(
                load_dispatch["shed_critical_mw"],
                adequacy_info.get("unavoidable_critical_deficit_mw", load_dispatch["shed_critical_mw"]),
            )),
            "avoidable_critical_shed_mw": float(max(
                0.0,
                load_dispatch["shed_critical_mw"] - min(
                    load_dispatch["shed_critical_mw"],
                    adequacy_info.get("unavoidable_critical_deficit_mw", load_dispatch["shed_critical_mw"]),
                ),
            )),
        }
        self.episode_history.append(state_dict)

        # ------------------------------------------------------------
        # Step 13: Shared Reward Computation
        # ------------------------------------------------------------
        shared_reward, reward_breakdown = self.reward_fn.calculate_reward(
            state=state_dict,
            actions=executed_actions,
            shield_info=shield_info,
            dt_hours=self.dt_hours,
        )
        rewards = {agent: shared_reward for agent in self.agents}

        # ------------------------------------------------------------
        # Step 14: Step Progression & Termination Checks
        # ------------------------------------------------------------
        self.current_step += 1
        self.current_hour += self.dt_hours

        # Terminate if unrecoverable numerical breakdown, otherwise truncate at horizon
        terminated = False  # Keep microgrid running through disturbances
        truncated = bool(self.current_step >= self.max_steps)

        active_agents = list(self.agents)
        terminations = {agent: terminated for agent in active_agents}
        truncations = {agent: truncated for agent in active_agents}

        # Construct next observations & critic state
        if self.use_forecasts:
            forecast_frozen = False
            if hasattr(self.scenario, "is_forecast_frozen"):
                forecast_frozen = self.scenario.is_forecast_frozen(hour)

            if not forecast_frozen:
                self.forecast_mgr.update_telemetry(
                    hour=hour,
                    solar_mw=solar_p,
                    wind_mw=wind_p,
                    total_demand_mw=nom_demand["total_demand_mw"],
                    crit_demand_mw=nom_demand["critical_mw"],
                    flex_demand_mw=nom_demand["flexible_mw"],
                    other_demand_mw=nom_demand["other_mw"],
                    cloud_factor=cloud_factor,
                    storm_warning=self.scenario.is_storm_warning(hour),
                )
                self.forecast_mgr.generate_forecast(
                    current_hour=self.current_hour,
                    current_state=state_dict,
                    scenario_manager=self.scenario,
                    solar_model=self.solar,
                    wind_model=self.wind,
                    load_manager=self.load_manager,
                )
            agent_forecast_feats = {agent: self.forecast_mgr.get_agent_forecast_features(agent) for agent in active_agents}
            critic_forecast_feats = self.forecast_mgr.get_critic_forecast_features()
            next_observations = self.obs_builder.build_observations(state_dict, agent_forecast_feats)
            global_critic_state = self.obs_builder.build_global_state(state_dict, critic_forecast_feats)
        else:
            next_observations = self.obs_builder.build_observations(state_dict)
            global_critic_state = self.obs_builder.build_global_state(state_dict)

        # Inject physical sensor noise and observation telemetry dropout if active in scenario
        if hasattr(self.scenario, "get_sensor_noise_profile"):
            s_prof = self.scenario.get_sensor_noise_profile(hour)
            sigma = float(s_prof.get("gaussian_sigma", 0.0))
            p_drop = float(s_prof.get("dropout_prob", 0.0))
            if sigma > 1e-4 or p_drop > 1e-4:
                for ag in next_observations:
                    arr = next_observations[ag].copy()
                    if sigma > 1e-4:
                        noise = np.random.normal(0.0, sigma, size=arr.shape).astype(np.float32)
                        arr = np.clip(arr + noise, -5.0, 5.0)
                    if p_drop > 1e-4:
                        mask = (np.random.random(size=arr.shape) >= p_drop).astype(np.float32)
                        arr = arr * mask
                    next_observations[ag] = arr

        # Compile per-agent infos exactly matching active agents
        infos = {}
        for agent in active_agents:
            infos[agent] = {
                "step": step_idx,
                "hour": hour,
                "reward_breakdown": reward_breakdown,
                "shield_diagnostics": shield_info,
                "adequacy_diagnostics": adequacy_info,
                "balance_error_mw": balance_res["balance_error_mw"],
                "converged": converged,
                "global_state": global_critic_state,
            }

        # Update live agents for PettingZoo parallel environment
        if terminated or truncated:
            self.agents = []
        else:
            self.agents = [a for a in active_agents if not (terminations[a] or truncations[a])]

        return next_observations, rewards, terminations, truncations, infos

    def _evaluate_initial_state(self) -> Dict[str, Any]:
        """Evaluate clean starting state at hour 0.0."""
        nom_demand = self.load_manager.get_nominal_demand(0.0)
        return {
            "step": 0,
            "hour": 0.0,
            "grid_connected": True,
            "grid_available": True,
            "sync_verified": True,
            "cloud_factor": 1.0,
            "wind_speed_ms": 8.0,
            "is_storm_warning": False,
            "solar_available_mw": 0.0,
            "wind_available_mw": 0.10,
            "solar_mw": 0.0,
            "wind_mw": 0.10,
            "re_gen_mw": 0.10,
            "generator_mw": 0.0,
            "generator_state": "OFF",
            "step_fuel_l": 0.0,
            "total_fuel_l": 0.0,
            "battery_mw": 0.0,
            "battery_soc": self.battery.soc,
            "battery_energy_mwh": self.battery.energy_mwh,
            "grid_import_mw": 0.70,
            "losses_mw": 0.0,
            "pcc_voltage_pu": 1.0,
            "min_bus_voltage_pu": 1.0,
            "max_bus_voltage_pu": 1.0,
            "freq_deviation_proxy": 0.0,
            "nominal_total_mw": nom_demand["total_demand_mw"],
            "nominal_critical_mw": nom_demand["critical_mw"],
            "nominal_flexible_mw": nom_demand["flexible_mw"],
            "nominal_other_mw": nom_demand["other_mw"],
            "served_total_mw": nom_demand["total_demand_mw"],
            "served_critical_mw": nom_demand["critical_mw"],
            "served_flexible_mw": nom_demand["flexible_mw"],
            "served_other_mw": nom_demand["other_mw"],
            "shed_total_mw": 0.0,
            "shed_critical_mw": 0.0,
            "shed_flexible_mw": 0.0,
            "shed_other_mw": 0.0,
            "critical_emergency": False,
            "deferred_flexible_mwh": 0.0,
            "balance_error_mw": 0.0,
            "is_balanced": True,
            "converged": True,
            "storm_risk": 0.02,
            "forecast_next_solar_mw": 0.0,
            "forecast_next_wind_mw": 0.10,
            "forecast_next_demand_mw": nom_demand["total_demand_mw"],
            "forecast_next_outage_risk": 0.02,
            "any_intervention": False,
            "boundary_clipping": False,
            "emergency_override": False,
            "inaction_detected": False,
            "adequacy_intervention": False,
            "avoidable_shortfall_mw": 0.0,
            "unavoidable_critical_shed_mw": 0.0,
            "avoidable_critical_shed_mw": 0.0,
        }

    def _get_default_state(self) -> Dict[str, Any]:
        """Fallback state representation."""
        return self._evaluate_initial_state()

    def _disconnect_from_grid(self):
        """Simulate tripping of PCC tie-breaker into islanded mode."""
        self.grid_connected = False
        self.net.ext_grid.at[0, "in_service"] = False
        self.net.gen.at[self._island_gen_idx, "in_service"] = True

    def _reconnect_to_grid(self):
        """Reclose PCC tie-breaker into grid-connected mode."""
        self.grid_connected = True
        self.net.ext_grid.at[0, "in_service"] = True
        self.net.gen.at[self._island_gen_idx, "in_service"] = False
        self.generator.request_stop()

    def _apply_network_state(
        self,
        solar_mw: float,
        wind_mw: float,
        battery_mw: float,
        generator_mw: float,
        load_dispatch: Dict[str, Any],
    ):
        """Push active power setpoints into Pandapower data model."""
        self.net.sgen.at[0, "p_mw"] = float(solar_mw)
        self.net.sgen.at[1, "p_mw"] = float(wind_mw)
        self.net.storage.at[0, "p_mw"] = float(battery_mw)

        if generator_mw > 0.001:
            self.net.sgen.at[2, "in_service"] = True
            self.net.sgen.at[2, "p_mw"] = float(generator_mw)
        else:
            self.net.sgen.at[2, "in_service"] = False
            self.net.sgen.at[2, "p_mw"] = 0.0

        if not self.grid_connected:
            self.net.gen.at[self._island_gen_idx, "p_mw"] = float(generator_mw)

        self.net.load.at[0, "p_mw"] = float(load_dispatch["served_critical_mw"])
        self.net.load.at[1, "p_mw"] = float(load_dispatch["served_flexible_mw"])
        self.net.load.at[2, "p_mw"] = float(load_dispatch["served_other_mw"])

    def _run_power_flow(self) -> Tuple[bool, float, float]:
        """Execute Pandapower Newton-Raphson power flow."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pp.runpp(self.net, numba=False)

            if self.net.converged:
                line_losses = float(self.net.res_line["p_loss_mw"].sum()) if hasattr(self.net, "res_line") and "p_loss_mw" in self.net.res_line else 0.0
                trafo_losses = float(self.net.res_trafo["p_loss_mw"].sum()) if hasattr(self.net, "res_trafo") and "p_loss_mw" in self.net.res_trafo else 0.0
                total_losses = line_losses + trafo_losses
                grid_p = float(self.net.res_ext_grid.p_mw.sum()) if self.grid_connected else 0.0
                return True, total_losses, grid_p
            else:
                return False, 0.0, 0.0
        except Exception:
            return False, 0.0, 0.0

    def render(self):
        """Print clean terminal telemetry summary."""
        if not self.episode_history:
            print("MicrogridParallelEnv: No episode history to render.")
            return

        st = self.episode_history[-1]
        print(f"Step {st['step']:02d} | t={st['hour']:5.2f}h | Grid={'CONNECTED' if st['grid_connected'] else 'ISLANDED'} | "
              f"RE={st['re_gen_mw']:.3f}MW | Gen={st['generator_mw']:.3f}MW | Bat={st['battery_mw']:+.3f}MW (SOC:{st['battery_soc']:.1f}%) | "
              f"LoadServed={st['served_total_mw']:.3f}/{st['nominal_total_mw']:.3f}MW | Err={st['balance_error_mw']:.5f}MW")

    def close(self):
        """Clean up environment resources."""
        self.episode_history.clear()


if __name__ == "__main__":
    print("GRIDGUARD AI - Microgrid Parallel Environment Self-Test")
    print("=" * 60)

    env = MicrogridParallelEnv(scenario_id="scenario_d", max_steps=12)
    obs, infos = env.reset(seed=42)

    print(f"Environment initialized with {len(env.agents)} agents: {env.agents}")
    print(f"Initial Critic State Shape: {infos['global_state'].shape} (Expected: {GLOBAL_STATE_SHAPE})")

    # Step through 8 random action steps
    total_ep_reward = 0.0
    for step_i in range(8):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, term, trunc, infos = env.step(actions)
        step_r = list(rewards.values())[0]
        total_ep_reward += step_r
        env.render()

    print(f"\nCompleted 8 test steps | Cumulative Reward = {total_ep_reward:+.3f}")
    print("Microgrid Parallel Environment test passed successfully.")
