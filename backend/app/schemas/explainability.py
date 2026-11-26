"""Pydantic schemas for Phase 4.5 Explainable AI (X-MARL) endpoints."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from backend.app.schemas.common import DISCLAIMER_TEXT, ProvenanceInfo


class SafetyInterventionRow(BaseModel):
    controller: str
    mode: str
    scenario: str
    eval_seed: int
    step: int
    hour: float
    timestamp: str
    agent: str
    intervention_stage: str  # BOUNDARY_CLIP | EMERGENCY_OVERRIDE | ADEQUACY_FALLBACK | BOTH_SHIELD_AND_FALLBACK
    triggering_condition: str
    reason: str
    prevented_violation: bool
    operational_effect: str


class AgentDecisionRow(BaseModel):
    controller: str
    mode: str
    scenario: str
    eval_seed: int
    step: int
    hour: float
    timestamp: str
    agent: str
    raw_policy_output: Any = None
    decoded_action: Dict[str, Any] = Field(default_factory=dict)
    shielded_action: Dict[str, Any] = Field(default_factory=dict)
    executed_action: Dict[str, Any] = Field(default_factory=dict)
    shield_modified: bool = False
    fallback_modified: bool = False
    intervention_stage: str = "NONE"
    operational_constraints: Dict[str, Any] = Field(default_factory=dict)
    reward_total: Optional[float] = None
    agent_reward_components: Dict[str, Any] = Field(default_factory=dict)
    relevant_observations: Dict[str, float] = Field(default_factory=dict)
    explanation: str


class ExplainabilityDecisionsResponse(BaseModel):
    disclaimer: str = DISCLAIMER_TEXT
    available: bool = True
    provenance: Optional[ProvenanceInfo] = None
    scenario: str = "scenario_h2"
    controllers: List[str] = Field(default_factory=list)
    decisions: List[AgentDecisionRow] = Field(default_factory=list)
    h2_timeline: List[Dict[str, Any]] = Field(default_factory=list)
    reward_decomposition_summary: List[Dict[str, Any]] = Field(default_factory=list)
    safety_interventions: List[SafetyInterventionRow] = Field(default_factory=list)
    intervention_counts_summary: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ExplainabilityAttributionsResponse(BaseModel):
    disclaimer: str = DISCLAIMER_TEXT
    available: bool = True
    sensitivity_vs_causality_note: str = (
        "Integrated Gradients and perturbation sensitivity quantify local policy responsiveness to "
        "observation inputs, NOT physical causal outcomes on grid reliability or cost."
    )
    agents: List[str] = Field(default_factory=list)
    aggregated_by_agent: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    controlled_perturbation_validation: List[Dict[str, Any]] = Field(default_factory=list)
    max_ig_completeness_error: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)


class ExplainabilityCounterfactualsResponse(BaseModel):
    disclaimer: str = DISCLAIMER_TEXT
    available: bool = True
    scenarios: List[str] = Field(default_factory=list)
    modes: List[str] = Field(default_factory=list)
    records: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
