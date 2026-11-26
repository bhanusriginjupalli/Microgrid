"""Analytics summary and multi-controller comparison endpoints."""

from typing import Optional
from fastapi import APIRouter, Query
from backend.app.api.deps import get_loader
from backend.app.schemas.experiments import AnalyticsComparisonResponse, AnalyticsSummaryResponse

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])


@router.get("/summary", response_model=AnalyticsSummaryResponse)
def analytics_summary():
    return get_loader().get_analytics_summary()


@router.get("/comparison", response_model=AnalyticsComparisonResponse)
def analytics_comparison(
    ablation: Optional[str] = Query(default=None, description="Optional Phase 4.4.4 ablation filter"),
    mode: Optional[str] = Query(default=None, description="Optional mode filter: Intrinsic | Hybrid (+Fallback) | Heuristic"),
):
    return get_loader().get_analytics_comparison(ablation=ablation, mode=mode)
