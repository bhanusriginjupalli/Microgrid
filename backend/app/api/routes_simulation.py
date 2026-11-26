"""Live simulation lifecycle control endpoints."""

from fastapi import APIRouter, HTTPException, Query
from backend.app.api.deps import get_sim_service
from backend.app.schemas.simulation import (
    SimulationSpeedRequest,
    SimulationStartRequest,
    SimulationStateResponse,
)

router = APIRouter(prefix="/api/simulation", tags=["Simulation"])


@router.post("/start", response_model=SimulationStateResponse)
def start_simulation(req: SimulationStartRequest):
    sim = get_sim_service()
    try:
        return sim.start(req)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to start simulation: {exc}")


@router.post("/pause", response_model=SimulationStateResponse)
def pause_simulation():
    return get_sim_service().pause()


@router.post("/resume", response_model=SimulationStateResponse)
def resume_simulation():
    return get_sim_service().resume()


@router.post("/reset", response_model=SimulationStateResponse)
def reset_simulation():
    return get_sim_service().reset()


@router.post("/speed", response_model=SimulationStateResponse)
def set_simulation_speed(req: SimulationSpeedRequest):
    return get_sim_service().set_speed(req.step_delay_ms)


@router.get("/state", response_model=SimulationStateResponse)
def get_simulation_state(include_history: bool = Query(default=True)):
    return get_sim_service().get_state(include_history=include_history)
