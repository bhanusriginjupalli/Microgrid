"""Pydantic schemas for system status and 9-bus pandapower digital twin telemetry."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from backend.app.schemas.common import DISCLAIMER_TEXT, ProvenanceInfo


class BusTelemetry(BaseModel):
    bus_idx: int
    name: str
    role: str
    vn_kv: float
    vm_pu: Optional[float] = None
    va_degree: Optional[float] = None
    p_mw: Optional[float] = None
    q_mvar: Optional[float] = None
    in_service: bool = True
    is_simulated: bool = True


class LineTelemetry(BaseModel):
    line_idx: int
    name: str
    from_bus: int
    to_bus: int
    from_bus_name: str
    to_bus_name: str
    p_from_mw: Optional[float] = None
    p_to_mw: Optional[float] = None
    loading_percent: Optional[float] = None
    in_service: bool = True
    power_direction: str = "idle"  # "forward", "reverse", "idle", "disconnected"


class NetworkTopologyResponse(BaseModel):
    disclaimer: str = DISCLAIMER_TEXT
    is_simulated: bool = True
    data_source: str  # "live_simulation" | "pandapower_nominal_topology"
    step: int = 0
    hour: float = 0.0
    converged: Optional[bool] = None
    grid_connected: Optional[bool] = None
    grid_available: Optional[bool] = None
    sync_verified: Optional[bool] = None
    buses: List[BusTelemetry] = Field(default_factory=list)
    lines: List[LineTelemetry] = Field(default_factory=list)
    asset_summary: Dict[str, Any] = Field(default_factory=dict)


class SystemStatusResponse(BaseModel):
    disclaimer: str = DISCLAIMER_TEXT
    simulation_status: str  # "idle" | "running" | "paused" | "completed" | "error"
    active_scenario: Optional[str] = None
    active_scenario_name: Optional[str] = None
    active_controller: Optional[str] = None
    controller_mode: Optional[str] = None  # "Intrinsic" | "Hybrid (+Fallback)" | "Heuristic"
    eval_seed: int = 42
    current_step: int = 0
    max_steps: int = 96
    current_hour: float = 0.0
    solar_mw: Optional[float] = None
    wind_mw: Optional[float] = None
    total_generation_mw: Optional[float] = None
    total_demand_mw: Optional[float] = None
    battery_soc_pct: Optional[float] = None
    battery_power_mw: Optional[float] = None
    grid_connected: Optional[bool] = None
    grid_available: Optional[bool] = None
    grid_import_mw: Optional[float] = None
    grid_export_mw: Optional[float] = None
    generator_mw: Optional[float] = None
    generator_state: Optional[str] = None
    critical_served_mw: Optional[float] = None
    critical_reliability_pct: Optional[float] = None
    critical_ens_mwh: Optional[float] = None
    operating_cost_usd: Optional[float] = None
    true_social_cost_usd: Optional[float] = None
    alerts: List[str] = Field(default_factory=list)
    provenance: Optional[ProvenanceInfo] = None
