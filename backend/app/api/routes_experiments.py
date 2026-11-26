"""Experiments catalog and detail endpoints backed by verified disk artifacts."""

from fastapi import APIRouter, HTTPException
from backend.app.api.deps import get_loader
from backend.app.schemas.experiments import ExperimentDetailResponse, ExperimentsListResponse

router = APIRouter(prefix="/api", tags=["Experiments"])


@router.get("/experiments", response_model=ExperimentsListResponse)
def list_experiments():
    loader = get_loader()
    return ExperimentsListResponse(experiments=loader.list_experiments())


@router.get("/experiments/{experiment_id}", response_model=ExperimentDetailResponse)
def get_experiment(experiment_id: str):
    loader = get_loader()
    detail = loader.get_experiment_detail(experiment_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")
    return detail
