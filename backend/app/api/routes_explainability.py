"""Phase 4.5 X-MARL explainability endpoints: decisions, attributions, and counterfactuals."""

from typing import Optional
from fastapi import APIRouter, Query
from backend.app.api.deps import get_loader
from backend.app.schemas.explainability import (
    ExplainabilityAttributionsResponse,
    ExplainabilityCounterfactualsResponse,
    ExplainabilityDecisionsResponse,
)

router = APIRouter(prefix="/api/explainability", tags=["Explainability"])


@router.get("/decisions", response_model=ExplainabilityDecisionsResponse)
def explainability_decisions(
    controller: Optional[str] = Query(default=None),
    agent: Optional[str] = Query(default=None),
    step_min: Optional[int] = Query(default=None, ge=0, le=96),
    step_max: Optional[int] = Query(default=None, ge=0, le=96),
    limit: int = Query(default=500, ge=1, le=2500),
):
    return get_loader().get_explainability_decisions(
        controller=controller,
        agent=agent,
        step_min=step_min,
        step_max=step_max,
        limit=limit,
    )


@router.get("/attributions", response_model=ExplainabilityAttributionsResponse)
def explainability_attributions():
    return get_loader().get_explainability_attributions()


@router.get("/counterfactuals", response_model=ExplainabilityCounterfactualsResponse)
def explainability_counterfactuals(
    scenario: Optional[str] = Query(default=None),
    mode: Optional[str] = Query(default=None),
):
    return get_loader().get_explainability_counterfactuals(scenario=scenario, mode=mode)
