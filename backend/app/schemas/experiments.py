"""Pydantic schemas for experiments catalog, benchmark comparison, and analytics."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from backend.app.schemas.common import DISCLAIMER_TEXT, ProvenanceInfo


class ExperimentSummary(BaseModel):
    experiment_id: str
    phase: str
    title: str
    description: str
    available: bool
    source_path: str
    row_count: int = 0
    controllers_count: int = 0
    separates_intrinsic_and_hybrid: bool = True
    warnings: List[str] = Field(default_factory=list)


class ExperimentsListResponse(BaseModel):
    disclaimer: str = DISCLAIMER_TEXT
    experiments: List[ExperimentSummary] = Field(default_factory=list)


class ExperimentDetailResponse(BaseModel):
    disclaimer: str = DISCLAIMER_TEXT
    experiment: ExperimentSummary
    provenance: Optional[ProvenanceInfo] = None
    columns: List[str] = Field(default_factory=list)
    records: List[Dict[str, Any]] = Field(default_factory=list)
    auxiliary_tables: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)


class ControllerComparisonRow(BaseModel):
    controller: str
    controller_group: Optional[str] = None
    ablation: Optional[str] = None
    train_seed: Optional[str] = None
    adequacy_fallback: Optional[bool] = None
    mode: str  # "Intrinsic" | "Hybrid (+Fallback)" | "Heuristic"
    episodes_n: Optional[int] = None
    true_social_cost_mean: Optional[float] = None
    true_social_cost_std: Optional[float] = None
    true_social_cost_cvar10: Optional[float] = None
    operating_cost_mean: Optional[float] = None
    critical_ens_mean: Optional[float] = None
    critical_reliability_mean: Optional[float] = None
    flexible_ens_mean: Optional[float] = None
    other_ens_mean: Optional[float] = None
    fuel_consumed_l_mean: Optional[float] = None
    battery_throughput_mean: Optional[float] = None
    re_curtailment_mwh_mean: Optional[float] = None
    unshielded_feasible_pct: Optional[float] = None
    boundary_clipping_mean: Optional[float] = None
    emergency_override_mean: Optional[float] = None
    adequacy_intervention_mean: Optional[float] = None
    power_flow_converged_pct: Optional[float] = None


class Phase45ScenarioComparisonRow(BaseModel):
    controller: str
    mode: str
    scenario: str
    eval_seed: int
    total_reward: Optional[float] = None
    critical_reliability_pct: Optional[float] = None
    total_reliability_pct: Optional[float] = None
    critical_ens_mwh: Optional[float] = None
    flexible_ens_mwh: Optional[float] = None
    other_ens_mwh: Optional[float] = None
    total_ens_mwh: Optional[float] = None
    operating_cost_usd: Optional[float] = None
    true_social_cost_usd: Optional[float] = None
    fuel_liters: Optional[float] = None
    battery_throughput_mwh: Optional[float] = None
    renewable_curtailment_mwh: Optional[float] = None
    boundary_clips: Optional[int] = None
    emergency_overrides: Optional[int] = None
    adequacy_interventions: Optional[int] = None


class AnalyticsSummaryResponse(BaseModel):
    disclaimer: str = DISCLAIMER_TEXT
    available: bool = True
    primary_candidate: ProvenanceInfo
    key_conclusions: List[str] = Field(default_factory=list)
    phase45_means_by_controller: List[Dict[str, Any]] = Field(default_factory=list)
    phase444_highlights: List[ControllerComparisonRow] = Field(default_factory=list)
    fallback_attribution_rows: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class AnalyticsComparisonResponse(BaseModel):
    disclaimer: str = DISCLAIMER_TEXT
    available: bool = True
    source_phase444: str
    source_phase45: str
    phase444_controllers: List[ControllerComparisonRow] = Field(default_factory=list)
    phase45_traces: List[Phase45ScenarioComparisonRow] = Field(default_factory=list)
    autonomy_metrics: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
