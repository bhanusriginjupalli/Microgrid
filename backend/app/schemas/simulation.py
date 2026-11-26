"""Pydantic schemas for scenario catalog and live simulation lifecycle."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from backend.app.schemas.common import DISCLAIMER_TEXT


class DisturbanceEventSchema(BaseModel):
    type: str
    start_hour: float
    end_hour: float
    details: Dict[str, Any] = Field(default_factory=dict)


class ScenarioDescriptor(BaseModel):
    scenario_id: str
    category: str  # "Benchmark A-F" | "Held-Out H1-H6"
    name: str
    description: str
    events: List[DisturbanceEventSchema] = Field(default_factory=list)
    has_outage: bool = False
    has_storm: bool = False
    has_topology_shift: bool = False
    has_sensor_degradation: bool = False
    has_stale_forecast: bool = False


class ControllerOption(BaseModel):
    controller_id: str
    label: str
    controller_type: str  # "mappo" | "greedy" | "rule_based" | "mpc"
    mode: str  # "Intrinsic" | "Hybrid (+Fallback)" | "Heuristic"
    adequacy_fallback: bool
    checkpoint_path: Optional[str] = None
    forecast_mode: Optional[str] = None
    description: str
    is_primary_candidate: bool = False


class ScenariosResponse(BaseModel):
    disclaimer: str = DISCLAIMER_TEXT
    scenarios: List[ScenarioDescriptor] = Field(default_factory=list)
    controllers: List[ControllerOption] = Field(default_factory=list)


class CustomScenarioConditions(BaseModel):
    solar_cloud_severity: float = Field(default=0.0, ge=0.0, le=0.95)
    wind_speed_ms: Optional[float] = Field(default=None, ge=0.0, le=30.0)
    critical_load_multiplier: float = Field(default=1.0, ge=0.5, le=2.5)
    flexible_load_multiplier: float = Field(default=1.0, ge=0.5, le=2.5)
    grid_outage_enabled: bool = False
    outage_start_hour: float = Field(default=11.0, ge=0.0, le=23.5)
    outage_end_hour: float = Field(default=14.0, ge=0.5, le=24.0)


class SimulationStartRequest(BaseModel):
    controller_id: str = Field(default="bat_only_seed202_intrinsic")
    scenario_id: str = Field(default="scenario_h2")
    seed: int = Field(default=42, ge=0, le=999999)
    step_delay_ms: int = Field(default=250, ge=0, le=5000)
    forecast_mode: Optional[str] = Field(default=None)
    custom_conditions: Optional[CustomScenarioConditions] = Field(default=None)


class SimulationSpeedRequest(BaseModel):
    step_delay_ms: int = Field(..., ge=0, le=5000)


class ForecastHorizonPoint(BaseModel):
    horizon_step: int
    horizon_minutes: int
    target_hour: float
    solar_q10: Optional[float] = None
    solar_q50: Optional[float] = None
    solar_q90: Optional[float] = None
    wind_q10: Optional[float] = None
    wind_q50: Optional[float] = None
    wind_q90: Optional[float] = None
    demand_q10: Optional[float] = None
    demand_q50: Optional[float] = None
    demand_q90: Optional[float] = None
    net_load_q10: Optional[float] = None
    net_load_q50: Optional[float] = None
    net_load_q90: Optional[float] = None
    outage_prob: Optional[float] = None
    storm_risk: Optional[float] = None


class ForecastSnapshot(BaseModel):
    mode: str  # "realistic" | "oracle" | "disabled"
    current_hour: float
    actual_solar_mw: float
    actual_wind_mw: float
    actual_demand_mw: float
    actual_net_load_mw: float
    horizons: List[ForecastHorizonPoint] = Field(default_factory=list)


class SimulationStepSnapshot(BaseModel):
    step: int
    hour: float
    timestamp: str
    solar_mw: float
    wind_mw: float
    solar_available_mw: float
    wind_available_mw: float
    total_generation_mw: float
    nominal_critical_mw: float
    nominal_flexible_mw: float
    nominal_other_mw: float
    nominal_total_mw: float
    served_critical_mw: float
    served_flexible_mw: float
    served_other_mw: float
    shed_critical_mw: float
    shed_flexible_mw: float
    shed_other_mw: float
    battery_mw: float
    battery_soc: float
    generator_mw: float
    generator_state: str
    step_fuel_l: float
    total_fuel_l: float
    grid_connected: bool
    grid_available: bool
    sync_verified: bool
    grid_import_mw: float
    grid_export_mw: float
    pcc_voltage_pu: float
    converged: bool
    balance_error_mw: float
    is_storm_warning: bool
    shared_reward: float
    cumulative_reward: float
    reward_breakdown: Dict[str, float] = Field(default_factory=dict)
    operating_cost_usd_cum: float
    true_social_cost_usd_cum: float
    critical_reliability_pct_cum: float
    critical_ens_mwh_cum: float
    shield_any_intervention: bool = False
    shield_any_emergency: bool = False
    shield_any_boundary: bool = False
    adequacy_intervention: bool = False
    inaction_detected: bool = False
    agent_decisions: List[Dict[str, Any]] = Field(default_factory=list)
    interventions: List[Dict[str, Any]] = Field(default_factory=list)
    forecast: Optional[ForecastSnapshot] = None
    network: Optional[Dict[str, Any]] = None
    alerts: List[str] = Field(default_factory=list)


class SimulationStateResponse(BaseModel):
    disclaimer: str = DISCLAIMER_TEXT
    status: str  # "idle" | "running" | "paused" | "completed" | "error"
    error_message: Optional[str] = None
    controller_id: Optional[str] = None
    controller_label: Optional[str] = None
    controller_mode: Optional[str] = None
    scenario_id: Optional[str] = None
    scenario_name: Optional[str] = None
    seed: int = 42
    step_delay_ms: int = 250
    forecast_mode: Optional[str] = None
    current_step: int = 0
    max_steps: int = 96
    latest_step: Optional[SimulationStepSnapshot] = None
    history: List[SimulationStepSnapshot] = Field(default_factory=list)
    summary_economics: Optional[Dict[str, Any]] = None
