"""Shared Pydantic models for health, provenance, and simulation disclaimers."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


DISCLAIMER_TEXT = "SIMULATION ENVIRONMENT — NOT CONNECTED TO PHYSICAL GRID"


class ProvenanceInfo(BaseModel):
    checkpoint_path: Optional[str] = None
    experiment: Optional[str] = None
    seed: Optional[int] = None
    episode: Optional[int] = None
    val_score: Optional[float] = None
    forecast_mode: Optional[str] = None
    verified: bool = False
    source_file: Optional[str] = None
    notes: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    app_name: str = "GRIDGUARD AI Control Center API"
    version: str = "5.0.0"
    disclaimer: str = DISCLAIMER_TEXT
    artifacts_available: Dict[str, bool] = Field(default_factory=dict)
    missing_artifacts: List[str] = Field(default_factory=list)
    simulation_status: str = "idle"
    active_scenario: Optional[str] = None
    active_controller: Optional[str] = None
    provenance: Optional[ProvenanceInfo] = None


class ActionStatusResponse(BaseModel):
    ok: bool = True
    message: str
    status: str
    data: Optional[Dict[str, Any]] = None
