"""Pydantic schemas for Phase 4 Saved Episode Library read-only API endpoints."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from backend.app.schemas.common import DISCLAIMER_TEXT


class EpisodeIndexItem(BaseModel):
    episode_id: str
    schema_version: str = "1.0.0"
    experiment_id: str = "phase6_model_comparison"
    algorithm: str
    controller_name: str
    controller_group: str
    track: str
    mode: str
    scenario_id: str
    training_seed: Optional[int] = None
    evaluation_seed: int
    selected_checkpoint_episode: Optional[int] = None
    episode_length: int
    timestep_minutes: float
    start_timestamp: Optional[str] = None
    completion_status: str
    enable_safety_shield: bool
    adequacy_fallback: bool
    checkpoint_path: Optional[str] = None
    trace_path: str
    summary_path: str
    reward: float
    true_social_cost_usd: float
    operating_cost_usd: float
    grid_import_cost_usd: Optional[float] = None
    grid_export_revenue_usd: Optional[float] = None
    fuel_consumed_l: Optional[float] = None
    fuel_cost_usd: Optional[float] = None
    generator_operating_hours: Optional[float] = None
    generator_startup_count: Optional[int] = None
    startup_wear_cost_usd: Optional[float] = None
    critical_ens_mwh: float
    flexible_ens_mwh: Optional[float] = None
    other_ens_mwh: Optional[float] = None
    total_ens_mwh: Optional[float] = None
    critical_reliability_pct: float
    total_reliability_pct: Optional[float] = None
    renewable_curtailment_mwh: Optional[float] = None
    renewable_utilization_pct: Optional[float] = None
    battery_throughput_mwh: Optional[float] = None
    battery_degradation_cost_usd: Optional[float] = None
    terminal_battery_soc_pct: Optional[float] = None
    voltage_violation_steps: Optional[int] = None
    power_flow_failures: Optional[int] = None
    power_flow_converged_pct: Optional[float] = None
    unshielded_feasible_pct: Optional[float] = None
    boundary_clipping_count: Optional[int] = None
    emergency_override_count: Optional[int] = None
    supply_adequacy_intervention_count: Optional[int] = None
    fallback_intervention_steps: Optional[int] = None


class EpisodeListResponse(BaseModel):
    disclaimer: str = DISCLAIMER_TEXT
    schema_version: str = "1.0.0"
    total: int
    offset: int
    limit: int
    episodes: List[EpisodeIndexItem] = Field(default_factory=list)


class EpisodeDetailResponse(BaseModel):
    disclaimer: str = DISCLAIMER_TEXT
    episode_id: str
    schema_version: str
    metadata: Dict[str, Any]
    metrics: Dict[str, Any]


class EpisodeTraceResponse(BaseModel):
    disclaimer: str = DISCLAIMER_TEXT
    episode_id: str
    schema_version: str
    metadata: Dict[str, Any]
    metrics: Dict[str, Any]
    steps: List[Dict[str, Any]] = Field(default_factory=list)


class EpisodeCompareResponse(BaseModel):
    disclaimer: str = DISCLAIMER_TEXT
    reference_episode_id: str
    scenario_id: str
    evaluation_seed: int
    aligned_step_count: int
    episodes_compared: List[Dict[str, Any]] = Field(default_factory=list)
    aligned_timesteps: List[Dict[str, Any]] = Field(default_factory=list)


class ScenarioEpisodesResponse(BaseModel):
    disclaimer: str = DISCLAIMER_TEXT
    scenario_id: str
    total: int
    episodes: List[EpisodeIndexItem] = Field(default_factory=list)
    h2_comparison: Optional[Dict[str, Any]] = None


class EpisodeLibrarySummaryResponse(BaseModel):
    disclaimer: str = DISCLAIMER_TEXT
    schema_version: str
    total_episodes: int
    tracks: List[str] = Field(default_factory=list)
    algorithms: List[str] = Field(default_factory=list)
    scenarios: List[str] = Field(default_factory=list)
    evaluation_seeds: List[int] = Field(default_factory=list)
    group_summaries: List[Dict[str, Any]] = Field(default_factory=list)
    manifest_integrity: Dict[str, Any] = Field(default_factory=dict)
