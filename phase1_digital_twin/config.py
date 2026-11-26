# GRIDGUARD AI
# Phase 1: Microgrid Configuration
# ============================================================
# All power values are in MW, energy in MWh.
# For example, 500 kW = 0.5 MW, 1000 kWh = 1.0 MWh.
# ============================================================

GRID = {
    "frequency_hz": 50,
    "voltage_kv": 11,
    "base_mva": 10,
}

SOLAR = {
    "capacity_mw": 0.5,
    "initial_output_mw": 0.35,
}

WIND = {
    "capacity_mw": 0.25,
    "initial_output_mw": 0.15,
}

BATTERY = {
    "capacity_mwh": 1.0,
    "max_power_mw": 0.25,
    "initial_soc": 80,       # percent
    "min_soc": 20,            # percent
    "max_soc": 95,            # percent
    "charge_efficiency": 0.95,
    "discharge_efficiency": 0.95,
    "emergency_reserve_soc": 30,  # percent (Phase 1.5 reserve threshold)
}

GENERATOR = {
    "capacity_mw": 0.4,
    "initial_output_mw": 0.0,
    "min_output_mw": 0.08,             # 80 kW minimum stable output
    "ramp_up_rate_mw_per_min": 0.02,    # 20 kW/min ramp rate
    "ramp_down_rate_mw_per_min": 0.02,  # 20 kW/min ramp down rate
    "startup_delay_steps": 2,           # 2 steps (30 mins at 15-min step)
    "fuel_rate_base_l_per_hr": 20.0,    # Liters/hour idling
    "fuel_rate_full_l_per_hr": 80.0,    # Liters/hour full output
    "startup_fuel_l": 5.0,              # Liters consumed on cold startup
    "min_up_time_steps": 4,             # 4 steps (1 hour)
    "min_down_time_steps": 2,           # 2 steps (30 mins)
}

GRID_CONNECTION = {
    "max_import_mw": 1.0,
    "connected": True,
}

LOADS = {
    "critical_mw": 0.3,
    "flexible_mw": 0.2,
    "other_mw": 0.3,
}

SIMULATION = {
    "time_step_minutes": 15,
    "duration_hours": 24,
}

COSTS = {
    "grid_import_per_mwh": 120.0,
    "grid_export_per_mwh": 40.0,
    "fuel_per_liter": 1.5,
    "critical_unserved_penalty_per_mwh": 5000.0,
    "flexible_shed_penalty_per_mwh": 200.0,
    "other_shed_penalty_per_mwh": 500.0,
}

