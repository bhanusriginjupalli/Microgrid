# GRIDGUARD AI
# Phase 2: Observation Space & Observation Builder
# ============================================================
# Constructs standardized, normalized float32 observations for:
#   - 5 Decentralized Actors: Local observations + coordination features
#   - Centralized Critic: Comprehensive global microgrid state vector
# Features cyclical time encoding and strict range clamping.
# ============================================================

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from gymnasium import spaces

from rl_environment.agent_config import (
    AGENTS,
    LOCAL_OBS_SCHEMA,
    GLOBAL_STATE_SCHEMA,
    OBS_SHAPES,
    GLOBAL_STATE_SHAPE,
    FORECAST_OBS_SHAPES,
    FORECAST_GLOBAL_STATE_SHAPE,
    CAPACITY_SCALING,
)


def get_agent_observation_spaces(forecast_mode: bool = False) -> Dict[str, spaces.Box]:
    """Return dictionary of Gymnasium Box observation spaces for all agents."""
    shapes = FORECAST_OBS_SHAPES if forecast_mode else OBS_SHAPES
    spaces_dict = {}
    for agent_id, shape in shapes.items():
        spaces_dict[agent_id] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(shape,),
            dtype=np.float32,
        )
    return spaces_dict


def get_global_state_space(forecast_mode: bool = False) -> spaces.Box:
    """Return Gymnasium Box space for the centralized critic's global state."""
    shape = FORECAST_GLOBAL_STATE_SHAPE if forecast_mode else GLOBAL_STATE_SHAPE
    return spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(shape,),
        dtype=np.float32,
    )


class ObservationBuilder:
    """
    Transforms raw physical simulation states and forecasts into
    normalized observation vectors for decentralized actors and the critic.
    Supports both reactive mode and forecast-aware multi-horizon mode.
    """

    def __init__(self, use_forecasts: bool = False):
        self.use_forecasts = use_forecasts
        self.obs_shapes = FORECAST_OBS_SHAPES if use_forecasts else OBS_SHAPES
        self.global_shape = FORECAST_GLOBAL_STATE_SHAPE if use_forecasts else GLOBAL_STATE_SHAPE

    def build_observations(
        self,
        state: Dict[str, Any],
        forecast_features: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Build normalized local observation vectors for all 5 agents.
        """
        """
        Build normalized local observation vectors for all 5 agents.
        """
        hour = float(state.get("hour", 0.0))
        time_sin = float(np.sin(2.0 * np.pi * (hour % 24.0) / 24.0))
        time_cos = float(np.cos(2.0 * np.pi * (hour % 24.0) / 24.0))

        # Solar & Wind availability vs output
        solar_avail = float(state.get("solar_available_mw", state.get("solar_mw", 0.0)))
        wind_avail = float(state.get("wind_available_mw", state.get("wind_mw", 0.0)))
        solar_out = float(state.get("solar_mw", 0.0))
        wind_out = float(state.get("wind_mw", 0.0))
        re_out = solar_out + wind_out

        # Demand
        total_demand = float(state.get("nominal_total_mw", state.get("total_load_mw", 0.8)))
        crit_demand = float(state.get("nominal_critical_mw", 0.3))
        flex_demand = float(state.get("nominal_flexible_mw", 0.2))
        other_demand = float(state.get("nominal_other_mw", 0.3))

        # Storage
        soc = float(state.get("battery_soc", 80.0))
        bat_p = float(state.get("battery_mw", 0.0))
        usable_soc = float(np.clip((soc - 20.0) / 75.0, 0.0, 1.0))
        reserve_margin = float((soc - 30.0) / 100.0)

        # Generator
        gen_out = float(state.get("generator_mw", 0.0))
        gen_state_str = state.get("generator_state", "OFF")
        state_onehot = [
            1.0 if gen_state_str == "OFF" else 0.0,
            1.0 if gen_state_str == "STARTING" else 0.0,
            1.0 if gen_state_str == "RUNNING" else 0.0,
            1.0 if gen_state_str == "SHUTTING_DOWN" else 0.0,
        ]

        # Grid
        grid_conn = 1.0 if state.get("grid_connected", True) else 0.0
        grid_avail = 1.0 if state.get("grid_available", True) else 0.0
        sync_ver = 1.0 if state.get("sync_verified", True) else 0.0
        pcc_voltage = float(state.get("pcc_voltage_pu", 1.0))
        freq_dev = float(state.get("freq_deviation_proxy", 0.0))

        # Weather & Disturbance
        cloud_factor = float(state.get("cloud_factor", 1.0))
        wind_speed = float(state.get("wind_speed_ms", 8.0))
        outage_risk = float(state.get("storm_risk", state.get("forecast_next_outage_risk", 0.02)))
        storm_warning = 1.0 if state.get("is_storm_warning", False) else 0.0

        # Forecasts
        fc_solar = float(state.get("forecast_next_solar_mw", solar_avail))
        fc_wind = float(state.get("forecast_next_wind_mw", wind_avail))
        fc_demand = float(state.get("forecast_next_demand_mw", total_demand))

        # Deficit / Imbalance
        power_deficit = max(0.0, total_demand - re_out)
        imbalance = float(state.get("balance_error_mw", 0.0))
        deferred_load = float(state.get("deferred_flexible_mwh", 0.0))
        fuel_cum = float(state.get("total_fuel_l", 0.0))

        # ----------------------------------------------------
        # 1. Renewable Agent Observation (shape: 12)
        # ----------------------------------------------------
        obs_renewable = np.array([
            solar_avail / CAPACITY_SCALING["solar_mw"],
            wind_avail / CAPACITY_SCALING["wind_mw"],
            solar_out / CAPACITY_SCALING["solar_mw"],
            wind_out / CAPACITY_SCALING["wind_mw"],
            total_demand / CAPACITY_SCALING["total_load_mw"],
            grid_conn,
            cloud_factor,
            wind_speed / 25.0,
            fc_solar / CAPACITY_SCALING["solar_mw"],
            fc_wind / CAPACITY_SCALING["wind_mw"],
            time_sin,
            time_cos,
        ], dtype=np.float32)

        # ----------------------------------------------------
        # 2. Battery Agent Observation (shape: 10)
        # ----------------------------------------------------
        obs_battery = np.array([
            soc / 100.0,
            bat_p / CAPACITY_SCALING["battery_power_mw"],
            usable_soc,
            power_deficit / CAPACITY_SCALING["total_load_mw"],
            (fc_solar + fc_wind) / CAPACITY_SCALING["total_re_mw"],
            fc_demand / CAPACITY_SCALING["total_load_mw"],
            grid_conn,
            reserve_margin,
            time_sin,
            time_cos,
        ], dtype=np.float32)

        # ----------------------------------------------------
        # 3. Generator Agent Observation (shape: 13)
        # ----------------------------------------------------
        obs_generator = np.array([
            state_onehot[0],
            state_onehot[1],
            state_onehot[2],
            state_onehot[3],
            gen_out / CAPACITY_SCALING["generator_capacity_mw"],
            power_deficit / CAPACITY_SCALING["total_load_mw"],
            soc / 100.0,
            re_out / CAPACITY_SCALING["total_re_mw"],
            grid_avail,
            fc_demand / CAPACITY_SCALING["total_load_mw"],
            fuel_cum / 500.0,
            time_sin,
            time_cos,
        ], dtype=np.float32)

        # ----------------------------------------------------
        # 4. Load Management Agent Observation (shape: 11)
        # ----------------------------------------------------
        obs_load = np.array([
            total_demand / CAPACITY_SCALING["total_load_mw"],
            crit_demand / CAPACITY_SCALING["critical_load_mw"],
            flex_demand / CAPACITY_SCALING["flexible_load_mw"],
            other_demand / CAPACITY_SCALING["other_load_mw"],
            power_deficit / CAPACITY_SCALING["total_load_mw"],
            soc / 100.0,
            grid_conn,
            1.0 if gen_state_str == "RUNNING" else 0.0,
            deferred_load / 1.0,
            time_sin,
            time_cos,
        ], dtype=np.float32)

        # ----------------------------------------------------
        # 5. Grid Resilience Agent Observation (shape: 12)
        # ----------------------------------------------------
        total_avail_internal_gen = solar_avail + wind_avail + CAPACITY_SCALING["generator_capacity_mw"]
        obs_grid = np.array([
            grid_conn,
            grid_avail,
            sync_ver,
            (pcc_voltage - 0.90) / 0.20,
            freq_dev,
            total_avail_internal_gen / 1.15,
            soc / 100.0,
            imbalance / CAPACITY_SCALING["total_load_mw"],
            outage_risk,
            storm_warning,
            time_sin,
            time_cos,
        ], dtype=np.float32)

        # Ensure all values are finite float32
        observations = {
            "renewable": np.nan_to_num(obs_renewable, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32),
            "battery": np.nan_to_num(obs_battery, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32),
            "generator": np.nan_to_num(obs_generator, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32),
            "load_management": np.nan_to_num(obs_load, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32),
            "grid_resilience": np.nan_to_num(obs_grid, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32),
        }

        # Augment with multi-horizon forecast features if enabled
        if self.use_forecasts:
            for agent in AGENTS:
                if forecast_features is not None and agent in forecast_features:
                    f_vec = np.asarray(forecast_features[agent], dtype=np.float32)
                else:
                    target_dim = self.obs_shapes[agent] - len(observations[agent])
                    f_vec = np.zeros(target_dim, dtype=np.float32)
                observations[agent] = np.concatenate([observations[agent], f_vec]).astype(np.float32)

        return observations

    def build_global_state(
        self,
        state: Dict[str, Any],
        critic_forecast_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Construct global state vector for the centralized critic.
        Shape: (26,) reactive, or (45,) forecast-aware.
        """
        hour = float(state.get("hour", 0.0))
        time_sin = float(np.sin(2.0 * np.pi * (hour % 24.0) / 24.0))
        time_cos = float(np.cos(2.0 * np.pi * (hour % 24.0) / 24.0))

        solar_mw = float(state.get("solar_mw", 0.0))
        wind_mw = float(state.get("wind_mw", 0.0))
        gen_mw = float(state.get("generator_mw", 0.0))
        bat_mw = float(state.get("battery_mw", 0.0))
        grid_mw = float(state.get("grid_import_mw", 0.0))

        served_tot = float(state.get("served_total_mw", 0.8))
        nom_tot = float(state.get("nominal_total_mw", 0.8))
        crit_serv = float(state.get("served_critical_mw", 0.3))
        flex_serv = float(state.get("served_flexible_mw", 0.2))
        other_serv = float(state.get("served_other_mw", 0.3))

        crit_shed = float(state.get("shed_critical_mw", 0.0))
        flex_shed = float(state.get("shed_flexible_mw", 0.0))
        other_shed = float(state.get("shed_other_mw", 0.0))

        soc = float(state.get("battery_soc", 80.0))
        state_map = {"OFF": 0.0, "STARTING": 0.33, "RUNNING": 0.66, "SHUTTING_DOWN": 1.0}
        gen_state_val = state_map.get(state.get("generator_state", "OFF"), 0.0)

        grid_conn = 1.0 if state.get("grid_connected", True) else 0.0
        grid_avail = 1.0 if state.get("grid_available", True) else 0.0
        sync_ver = 1.0 if state.get("sync_verified", True) else 0.0
        pcc_v = float(state.get("pcc_voltage_pu", 1.0))
        freq_dev = float(state.get("freq_deviation_proxy", 0.0))
        storm_risk = float(state.get("storm_risk", 0.02))

        fc_solar = float(state.get("forecast_next_solar_mw", solar_mw))
        fc_wind = float(state.get("forecast_next_wind_mw", wind_mw))
        fc_demand = float(state.get("forecast_next_demand_mw", nom_tot))

        global_vec = np.array([
            solar_mw / CAPACITY_SCALING["solar_mw"],
            wind_mw / CAPACITY_SCALING["wind_mw"],
            gen_mw / CAPACITY_SCALING["generator_capacity_mw"],
            bat_mw / CAPACITY_SCALING["battery_power_mw"],
            grid_mw / CAPACITY_SCALING["grid_import_max_mw"],
            served_tot / CAPACITY_SCALING["total_load_mw"],
            nom_tot / CAPACITY_SCALING["total_load_mw"],
            crit_serv / CAPACITY_SCALING["critical_load_mw"],
            flex_serv / CAPACITY_SCALING["flexible_load_mw"],
            other_serv / CAPACITY_SCALING["other_load_mw"],
            crit_shed / CAPACITY_SCALING["critical_load_mw"],
            flex_shed / CAPACITY_SCALING["flexible_load_mw"],
            other_shed / CAPACITY_SCALING["other_load_mw"],
            soc / 100.0,
            gen_state_val,
            grid_conn,
            grid_avail,
            sync_ver,
            pcc_v,
            freq_dev,
            storm_risk,
            fc_solar / CAPACITY_SCALING["solar_mw"],
            fc_wind / CAPACITY_SCALING["wind_mw"],
            fc_demand / CAPACITY_SCALING["total_load_mw"],
            time_sin,
            time_cos,
        ], dtype=np.float32)

        base_global = np.nan_to_num(global_vec, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

        if self.use_forecasts:
            if critic_forecast_features is not None:
                c_vec = np.asarray(critic_forecast_features, dtype=np.float32)
            else:
                target_dim = self.global_shape - len(base_global)
                c_vec = np.zeros(target_dim, dtype=np.float32)
            return np.concatenate([base_global, c_vec]).astype(np.float32)

        return base_global


if __name__ == "__main__":
    builder = ObservationBuilder()
    dummy_state = {
        "hour": 14.5,
        "solar_mw": 0.42,
        "wind_mw": 0.18,
        "battery_soc": 76.5,
        "battery_mw": -0.15,
        "generator_mw": 0.0,
        "generator_state": "OFF",
        "grid_connected": True,
        "grid_available": True,
        "sync_verified": True,
        "pcc_voltage_pu": 0.999,
        "nominal_total_mw": 0.85,
        "served_total_mw": 0.85,
        "nominal_critical_mw": 0.30,
        "served_critical_mw": 0.30,
        "nominal_flexible_mw": 0.22,
        "served_flexible_mw": 0.22,
        "nominal_other_mw": 0.33,
        "served_other_mw": 0.33,
    }
    obs = builder.build_observations(dummy_state)
    g_state = builder.build_global_state(dummy_state)

    print("GRIDGUARD AI - Observation Builder Self-Test")
    print("=" * 50)
    for agent_id, vec in obs.items():
        print(f"Agent: {agent_id:16s} | Shape: {vec.shape} | Mean: {np.mean(vec):.3f} | Min: {np.min(vec):.3f} | Max: {np.max(vec):.3f}")
        assert vec.shape[0] == OBS_SHAPES[agent_id], f"Shape mismatch for {agent_id}"
        assert np.all(np.isfinite(vec)), f"Non-finite values in {agent_id}"

    print(f"Global State Shape: {g_state.shape} (Expected: {GLOBAL_STATE_SHAPE})")
    assert g_state.shape[0] == GLOBAL_STATE_SHAPE
    assert np.all(np.isfinite(g_state))
    print("Observation Builder self-test passed successfully.")
