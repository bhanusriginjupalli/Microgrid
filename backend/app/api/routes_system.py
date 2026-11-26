"""System health, operational status, and 9-bus pandapower digital twin endpoints."""

from fastapi import APIRouter
from backend.app.api.deps import get_loader, get_sim_service
from backend.app.schemas.common import HealthResponse
from backend.app.schemas.system import NetworkTopologyResponse, SystemStatusResponse

router = APIRouter(prefix="/api", tags=["System"])


@router.get("/health", response_model=HealthResponse)
def health_check():
    loader = get_loader()
    sim = get_sim_service()
    checks, missing = loader.check_artifact_availability()
    state = sim.get_state(include_history=False)
    return HealthResponse(
        status="ok",
        artifacts_available=checks,
        missing_artifacts=missing,
        simulation_status=state.status,
        active_scenario=state.scenario_id,
        active_controller=state.controller_label,
        provenance=loader.get_primary_provenance(),
    )


@router.get("/system/status", response_model=SystemStatusResponse)
def system_status():
    loader = get_loader()
    sim = get_sim_service()
    state = sim.get_state(include_history=False)
    latest = state.latest_step

    if latest is None:
        return SystemStatusResponse(
            simulation_status=state.status,
            active_scenario=state.scenario_id,
            active_scenario_name=state.scenario_name,
            active_controller=state.controller_label,
            controller_mode=state.controller_mode,
            eval_seed=state.seed,
            current_step=0,
            max_steps=state.max_steps,
            current_hour=0.0,
            provenance=loader.get_primary_provenance(),
            alerts=["Simulation idle. Start a scenario from the Simulation Control bar to stream live steps."],
        )

    return SystemStatusResponse(
        simulation_status=state.status,
        active_scenario=state.scenario_id,
        active_scenario_name=state.scenario_name,
        active_controller=state.controller_label,
        controller_mode=state.controller_mode,
        eval_seed=state.seed,
        current_step=state.current_step,
        max_steps=state.max_steps,
        current_hour=latest.hour,
        solar_mw=latest.solar_mw,
        wind_mw=latest.wind_mw,
        total_generation_mw=latest.total_generation_mw,
        total_demand_mw=latest.nominal_total_mw,
        battery_soc_pct=latest.battery_soc,
        battery_power_mw=latest.battery_mw,
        grid_connected=latest.grid_connected,
        grid_available=latest.grid_available,
        grid_import_mw=latest.grid_import_mw,
        grid_export_mw=latest.grid_export_mw,
        generator_mw=latest.generator_mw,
        generator_state=latest.generator_state,
        critical_served_mw=latest.served_critical_mw,
        critical_reliability_pct=latest.critical_reliability_pct_cum,
        critical_ens_mwh=latest.critical_ens_mwh_cum,
        operating_cost_usd=latest.operating_cost_usd_cum,
        true_social_cost_usd=latest.true_social_cost_usd_cum,
        alerts=latest.alerts,
        provenance=loader.get_primary_provenance(),
    )


@router.get("/system/network", response_model=NetworkTopologyResponse)
def system_network():
    sim = get_sim_service()
    return sim.get_network()
