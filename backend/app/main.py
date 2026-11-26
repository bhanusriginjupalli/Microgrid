"""
backend/app/main.py
===================
GRIDGUARD AI — Phase 5 FastAPI Application Entrypoint

Run locally from project root:
  .venv\\Scripts\\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
"""

from contextlib import asynccontextmanager
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "phase1_digital_twin") not in sys.path:
    sys.path.insert(0, str(ROOT / "phase1_digital_twin"))

from backend.app.api import (
    routes_analytics,
    routes_episodes,
    routes_experiments,
    routes_explainability,
    routes_scenarios,
    routes_simulation,
    routes_system,
    routes_ws,
)
from backend.app.simulation.runner import simulation_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    simulation_service.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(
        title="GRIDGUARD AI — Research Control & Explainability API",
        description=(
            "FastAPI backend integrating GRIDGUARD AI Phase 1-4.5 digital twin, "
            "forecasting, safety-constrained MAPPO, and X-MARL explainability. "
            "SIMULATION ENVIRONMENT — NOT CONNECTED TO PHYSICAL GRID."
        ),
        version="5.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(routes_system.router)
    app.include_router(routes_experiments.router)
    app.include_router(routes_analytics.router)
    app.include_router(routes_episodes.router)
    app.include_router(routes_explainability.router)
    app.include_router(routes_scenarios.router)
    app.include_router(routes_simulation.router)
    app.include_router(routes_ws.router)

    # Optionally serve production frontend build if frontend/dist exists (with SPA client-route fallback)
    dist_dir = ROOT / "frontend" / "dist"
    if dist_dir.exists() and (dist_dir / "index.html").exists():
        assets_dir = dist_dir / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="frontend-assets")

        @app.get("/", include_in_schema=False)
        def serve_index():
            return FileResponse(str(dist_dir / "index.html"))

        @app.get("/{full_path:path}", include_in_schema=False)
        def serve_spa(full_path: str):
            if full_path.startswith("api/") or full_path.startswith("ws/"):
                raise HTTPException(status_code=404, detail="Not Found")
            target = dist_dir / full_path
            if full_path and target.exists() and target.is_file():
                return FileResponse(str(target))
            return FileResponse(str(dist_dir / "index.html"))

    return app


app = create_app()
