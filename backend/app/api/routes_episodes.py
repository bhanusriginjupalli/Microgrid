"""Read-only API endpoints for the Phase 4 Saved Episode Library."""

import json
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query

from backend.app.api.deps import get_episode_library
from backend.app.schemas.episode_library import (
    EpisodeCompareResponse,
    EpisodeDetailResponse,
    EpisodeIndexItem,
    EpisodeLibrarySummaryResponse,
    EpisodeListResponse,
    EpisodeTraceResponse,
    ScenarioEpisodesResponse,
)

router = APIRouter(prefix="/api", tags=["Episode Library"])

VALID_SORT_FIELDS = {
    "episode_id",
    "algorithm",
    "track",
    "mode",
    "scenario_id",
    "training_seed",
    "evaluation_seed",
    "true_social_cost_usd",
    "operating_cost_usd",
    "critical_ens_mwh",
    "critical_reliability_pct",
    "fallback_intervention_steps",
}


@router.get("/episodes", response_model=EpisodeListResponse)
def list_saved_episodes(
    algorithm: Optional[str] = Query(default=None),
    track: Optional[str] = Query(default=None),
    mode: Optional[str] = Query(default=None),
    scenario_id: Optional[str] = Query(default=None),
    training_seed: Optional[int] = Query(default=None),
    evaluation_seed: Optional[int] = Query(default=None),
    social_cost_min: Optional[float] = Query(default=None),
    social_cost_max: Optional[float] = Query(default=None),
    critical_ens_min: Optional[float] = Query(default=None),
    critical_ens_max: Optional[float] = Query(default=None),
    critical_reliability_min: Optional[float] = Query(default=None),
    critical_reliability_max: Optional[float] = Query(default=None),
    completion_status: Optional[str] = Query(default=None),
    search: Optional[str] = Query(default=None),
    sort_by: str = Query(default="episode_id"),
    descending: bool = Query(default=False),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=900),
):
    if sort_by not in VALID_SORT_FIELDS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid sort_by '{sort_by}'. Allowed: {sorted(VALID_SORT_FIELDS)}",
        )
    lib = get_episode_library()
    if not lib.is_available():
        raise HTTPException(status_code=503, detail="Saved episode library index not built yet.")

    rows, total = lib.list_episodes(
        algorithm=algorithm,
        track=track,
        mode=mode,
        scenario_id=scenario_id,
        training_seed=training_seed,
        evaluation_seed=evaluation_seed,
        social_cost_min=social_cost_min,
        social_cost_max=social_cost_max,
        critical_ens_min=critical_ens_min,
        critical_ens_max=critical_ens_max,
        critical_reliability_min=critical_reliability_min,
        critical_reliability_max=critical_reliability_max,
        completion_status=completion_status if isinstance(completion_status, str) else None,
        search=search if isinstance(search, str) else None,
        sort_by=sort_by,
        descending=descending,
        offset=offset,
        limit=limit,
    )
    return EpisodeListResponse(
        total=total,
        offset=offset,
        limit=limit,
        episodes=[EpisodeIndexItem(**r) for r in rows],
    )


@router.get("/episodes/summary", response_model=EpisodeLibrarySummaryResponse)
def episode_library_summary():
    lib = get_episode_library()
    if not lib.is_available():
        raise HTTPException(status_code=503, detail="Saved episode library index not built yet.")
    return EpisodeLibrarySummaryResponse(**lib.get_library_summary())


@router.get("/episodes/compare", response_model=EpisodeCompareResponse)
def compare_saved_episodes(
    episode_ids: List[str] = Query(..., description="Two or more episode IDs (repeat param or comma-separated)"),
    require_same_scenario: bool = Query(default=True),
    require_same_eval_seed: bool = Query(default=True),
):
    from experiments.phase6_model_comparison.episode_library.comparison import compare_episodes

    lib = get_episode_library()
    if not lib.is_available():
        raise HTTPException(status_code=503, detail="Saved episode library index not built yet.")

    expanded_ids: List[str] = []
    for item in episode_ids:
        for part in item.split(","):
            p = part.strip()
            if p:
                expanded_ids.append(p)

    try:
        res = compare_episodes(
            library=lib,
            episode_ids=expanded_ids,
            require_same_scenario=require_same_scenario,
            require_same_eval_seed=require_same_eval_seed,
        )
        return EpisodeCompareResponse(**res)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/episodes/scenarios/{scenario_id}", response_model=ScenarioEpisodesResponse)
def scenario_episodes(
    scenario_id: str,
    track: Optional[str] = Query(default=None),
    algorithm: Optional[str] = Query(default=None),
):
    lib = get_episode_library()
    if not lib.is_available():
        raise HTTPException(status_code=503, detail="Saved episode library index not built yet.")

    all_scenarios = {r["scenario_id"] for r in lib.load_index()}
    if scenario_id not in all_scenarios:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found in episode library.")

    items = lib.get_scenario_episodes(scenario_id=scenario_id, track=track, algorithm=algorithm)
    h2_comp = None
    if scenario_id == "scenario_h2" and lib.h2_dataset_path.exists():
        with open(lib.h2_dataset_path, "r", encoding="utf-8") as f:
            h2_comp = json.load(f)

    return ScenarioEpisodesResponse(
        scenario_id=scenario_id,
        total=len(items),
        episodes=[EpisodeIndexItem(**r) for r in items],
        h2_comparison=h2_comp,
    )


@router.get("/episodes/{episode_id}", response_model=EpisodeDetailResponse)
def get_saved_episode_summary(episode_id: str):
    lib = get_episode_library()
    if not lib.is_available():
        raise HTTPException(status_code=503, detail="Saved episode library index not built yet.")
    pkg = lib.get_episode_summary(episode_id)
    if pkg is None:
        raise HTTPException(status_code=404, detail=f"Episode '{episode_id}' not found.")
    return EpisodeDetailResponse(**pkg)


@router.get("/episodes/{episode_id}/trace", response_model=EpisodeTraceResponse)
def get_saved_episode_trace(episode_id: str):
    lib = get_episode_library()
    if not lib.is_available():
        raise HTTPException(status_code=503, detail="Saved episode library index not built yet.")
    try:
        pkg = lib.get_episode_trace(episode_id)
        return EpisodeTraceResponse(**pkg)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Episode '{episode_id}' not found.") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
