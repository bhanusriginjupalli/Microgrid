# GRIDGUARD AI
# Phase 2: Agent Configuration & Space Metadata
# ============================================================
# Centralized definitions for the 5 cooperative microgrid agents:
#   1. renewable: Box(2,) in [0, 1] (Solar & Wind curtailment)
#   2. battery: Box(1,) in [-1, 1] (Continuous power command)
#   3. generator: Box(1,) in [0, 1] (Requested dispatch fraction)
#   4. load_management: Discrete(6) (Priority load curtailment tiers)
#   5. grid_resilience: Discrete(5) (Grid coupling & transition modes)
# ============================================================

from typing import List, Dict, Any

AGENTS: List[str] = [
    "renewable",
    "battery",
    "generator",
    "load_management",
    "grid_resilience",
]

# Physical capacity scaling constants for normalization
CAPACITY_SCALING: Dict[str, float] = {
    "solar_mw": 0.50,
    "wind_mw": 0.25,
    "total_re_mw": 0.75,
    "battery_power_mw": 0.25,
    "battery_capacity_mwh": 1.00,
    "generator_capacity_mw": 0.40,
    "grid_import_max_mw": 1.00,
    "critical_load_mw": 0.30,
    "flexible_load_mw": 0.20,
    "other_load_mw": 0.30,
    "total_load_mw": 0.80,
    "max_voltage_pu": 1.10,
    "min_voltage_pu": 0.90,
    "frequency_nominal_hz": 50.0,
}

# Feature names and dimensions for observations
LOCAL_OBS_SCHEMA: Dict[str, List[str]] = {
    "renewable": [
        "solar_available_norm",       # 0: Solar potential / 0.5 MW
        "wind_available_norm",        # 1: Wind potential / 0.25 MW
        "solar_output_norm",           # 2: Actual solar output / 0.5 MW
        "wind_output_norm",            # 3: Actual wind output / 0.25 MW
        "demand_total_norm",           # 4: Total demand / 0.8 MW
        "grid_status_binary",          # 5: 1.0 if grid connected, 0.0 if islanded
        "cloud_factor_norm",           # 6: Solar cloud attenuation factor [0, 1]
        "wind_speed_norm",             # 7: Wind speed / 25 m/s
        "forecast_solar_t1_norm",      # 8: Next-step solar forecast / 0.5 MW
        "forecast_wind_t1_norm",       # 9: Next-step wind forecast / 0.25 MW
        "time_sin",                    # 10: sin(2pi * hour / 24)
        "time_cos",                    # 11: cos(2pi * hour / 24)
    ],
    "battery": [
        "battery_soc_norm",            # 0: SOC / 100 [0, 1]
        "battery_power_norm",          # 1: Power / P_max in [-1, 1] (+ charge, - discharge)
        "usable_capacity_norm",        # 2: (SOC - min_soc) / (max_soc - min_soc)
        "power_deficit_norm",          # 3: (Demand - RE) / 0.8 MW
        "forecast_re_t1_norm",         # 4: Next-step (solar + wind) / 0.75 MW
        "forecast_demand_t1_norm",     # 5: Next-step demand / 0.8 MW
        "grid_status_binary",          # 6: 1.0 if connected, 0.0 if islanded
        "emergency_reserve_margin",    # 7: (SOC - emergency_reserve) / 100
        "time_sin",                    # 8: sin(2pi * hour / 24)
        "time_cos",                    # 9: cos(2pi * hour / 24)
    ],
    "generator": [
        "generator_state_onehot_0",    # 0: OFF
        "generator_state_onehot_1",    # 1: STARTING
        "generator_state_onehot_2",    # 2: RUNNING
        "generator_state_onehot_3",    # 3: SHUTTING_DOWN
        "generator_output_norm",       # 4: Output / capacity (0.4 MW)
        "power_deficit_norm",          # 5: (Demand - RE) / 0.8 MW
        "battery_soc_norm",            # 6: SOC / 100
        "re_output_norm",              # 7: RE output / 0.75 MW
        "grid_available_binary",       # 8: 1.0 if grid available, 0.0 if outage
        "forecast_demand_t1_norm",     # 9: Next-step demand / 0.8 MW
        "fuel_consumed_norm",          # 10: Cumulative fuel / 500 L
        "time_sin",                    # 11: sin(2pi * hour / 24)
        "time_cos",                    # 12: cos(2pi * hour / 24)
    ],
    "load_management": [
        "demand_total_norm",           # 0: Total demand / 0.8 MW
        "demand_critical_norm",        # 1: Critical load / 0.3 MW
        "demand_flexible_norm",        # 2: Flexible load / 0.2 MW
        "demand_other_norm",           # 3: Other load / 0.3 MW
        "power_deficit_norm",          # 4: Power deficit / 0.8 MW
        "battery_soc_norm",            # 5: SOC / 100
        "grid_status_binary",          # 6: 1.0 if connected, 0.0 if islanded
        "generator_running_binary",    # 7: 1.0 if running, 0.0 otherwise
        "deferred_load_norm",          # 8: Deferred flexible load queue / 1.0 MWh
        "time_sin",                    # 9: sin(2pi * hour / 24)
        "time_cos",                    # 10: cos(2pi * hour / 24)
    ],
    "grid_resilience": [
        "grid_connected_binary",       # 0: 1.0 if PCC breaker closed
        "grid_available_binary",       # 1: 1.0 if utility grid powered, 0.0 if grid outage
        "sync_verified_binary",        # 2: 1.0 if phase & voltage sync verified
        "voltage_pcc_norm",            # 3: (V_pcc - 0.9) / 0.2
        "freq_deviation_norm",         # 4: Modeled frequency deviation proxy
        "available_generation_norm",   # 5: Total available internal gen / 1.15 MW
        "battery_soc_norm",            # 6: SOC / 100
        "power_imbalance_norm",        # 7: Bus power imbalance / 0.8 MW
        "outage_risk_norm",            # 8: Storm/outage risk [0, 1]
        "is_storm_warning_binary",     # 9: 1.0 if storm warning active
        "time_sin",                    # 10: sin(2pi * hour / 24)
        "time_cos",                    # 11: cos(2pi * hour / 24)
    ],
}

GLOBAL_STATE_SCHEMA: List[str] = [
    "solar_mw_norm",
    "wind_mw_norm",
    "generator_mw_norm",
    "battery_mw_norm",
    "grid_import_mw_norm",
    "load_served_mw_norm",
    "load_nominal_mw_norm",
    "critical_served_norm",
    "flexible_served_norm",
    "other_served_norm",
    "critical_shed_norm",
    "flexible_shed_norm",
    "other_shed_norm",
    "battery_soc_norm",
    "generator_state_val",
    "grid_connected_binary",
    "grid_available_binary",
    "sync_verified_binary",
    "pcc_voltage_pu",
    "freq_deviation_proxy",
    "storm_risk",
    "forecast_solar_norm",
    "forecast_wind_norm",
    "forecast_demand_norm",
    "time_sin",
    "time_cos",
]

OBS_SHAPES: Dict[str, int] = {agent: len(features) for agent, features in LOCAL_OBS_SCHEMA.items()}
GLOBAL_STATE_SHAPE: int = len(GLOBAL_STATE_SCHEMA)

# ------------------------------------------------------------
# Phase 4.2: Forecast-Aware Augmented Observation Schemas
# ------------------------------------------------------------
FORECAST_OBS_AUGMENTATIONS: Dict[str, List[str]] = {
    "renewable": [
        "fc_solar_30m_norm",
        "fc_solar_1h_norm",
        "fc_solar_4h_norm",
        "fc_wind_30m_norm",
        "fc_wind_1h_norm",
        "fc_wind_4h_norm",
        "fc_solar_unc_1h_norm",
        "fc_wind_unc_1h_norm",
    ],
    "battery": [
        "fc_net_load_15m_norm",
        "fc_net_load_30m_norm",
        "fc_net_load_1h_norm",
        "fc_net_load_2h_norm",
        "fc_net_load_4h_norm",
        "fc_outage_prob_15m",
        "fc_outage_prob_1h",
        "fc_outage_prob_4h",
        "fc_net_load_unc_1h_norm",
    ],
    "generator": [
        "fc_net_load_15m_norm",
        "fc_net_load_30m_norm",
        "fc_net_load_1h_norm",
        "fc_net_load_2h_norm",
        "fc_outage_prob_15m",
        "fc_outage_prob_30m",
        "fc_outage_prob_1h",
        "fc_outage_prob_2h",
    ],
    "load_management": [
        "fc_net_load_15m_norm",
        "fc_net_load_1h_norm",
        "fc_net_load_4h_norm",
        "fc_outage_prob_15m",
        "fc_outage_prob_1h",
        "fc_outage_prob_4h",
    ],
    "grid_resilience": [
        "fc_outage_prob_15m",
        "fc_outage_prob_30m",
        "fc_outage_prob_1h",
        "fc_outage_prob_2h",
        "fc_outage_prob_4h",
        "fc_storm_risk_1h",
        "fc_storm_risk_4h",
    ],
}

FORECAST_GLOBAL_AUGMENTATIONS: List[str] = [
    "fc_solar_30m_norm",
    "fc_solar_1h_norm",
    "fc_solar_4h_norm",
    "fc_wind_30m_norm",
    "fc_wind_1h_norm",
    "fc_wind_4h_norm",
    "fc_demand_30m_norm",
    "fc_demand_1h_norm",
    "fc_demand_4h_norm",
    "fc_net_load_15m_norm",
    "fc_net_load_30m_norm",
    "fc_net_load_1h_norm",
    "fc_net_load_2h_norm",
    "fc_net_load_4h_norm",
    "fc_outage_prob_15m",
    "fc_outage_prob_30m",
    "fc_outage_prob_1h",
    "fc_outage_prob_2h",
    "fc_outage_prob_4h",
]

FORECAST_OBS_SHAPES: Dict[str, int] = {
    agent: len(LOCAL_OBS_SCHEMA[agent]) + len(FORECAST_OBS_AUGMENTATIONS[agent])
    for agent in AGENTS
}
FORECAST_GLOBAL_STATE_SHAPE: int = GLOBAL_STATE_SHAPE + len(FORECAST_GLOBAL_AUGMENTATIONS)

