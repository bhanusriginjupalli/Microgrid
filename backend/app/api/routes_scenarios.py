"""Scenario A-F and Held-Out H1-H6 catalog + controller options."""

from fastapi import APIRouter
from backend.app.api.deps import get_sim_service
from backend.app.schemas.simulation import ScenariosResponse

router = APIRouter(prefix="/api", tags=["Scenarios"])


@router.get("/scenarios", response_model=ScenariosResponse)
def get_scenarios():
    sim = get_sim_service()
    return ScenariosResponse(
        scenarios=sim.list_scenarios(),
        controllers=sim.list_controllers(),
    )
