"""
backend/app/simulation/runner.py
================================
Controlled singleton simulation lifecycle manager for GRIDGUARD AI Phase 5.

Reuses existing root simulation, forecasting, safety, economics, and X-MARL
modules without duplicating simulator physics:
  - phase1_digital_twin.network.create_microgrid
  - phase1_digital_twin.scenario_manager.ScenarioManager
  - experiments.phase4.robustness.scenarios.robust_scenario_suite.HeldOutScenarios
  - rl_environment.microgrid_parallel_env.MicrogridParallelEnv
  - rl_environment.economic_accounting.EconomicConfig, compute_episode_economics
  - controllers.{GreedyController, RuleBasedEMS, MPCController}
  - experiments.phase4_4_3.benchmark_phase4_4_3.load_mappo_checkpoint
  - experiments.phase4_5.explainability.decision_explainer.DecisionExplainer
"""

import asyncio
import random
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np

from phase1_digital_twin.network import create_microgrid
from phase1_digital_twin.scenario_manager import ScenarioManager
from rl_environment.agent_config import AGENTS
from rl_environment.economic_accounting import EconomicConfig, compute_episode_economics
from rl_environment.microgrid_parallel_env import MicrogridParallelEnv
from controllers.greedy_controller import GreedyController
from controllers.rule_based_ems import RuleBasedEMS
from controllers.mpc_controller import MPCController
from experiments.phase4.robustness.scenarios.robust_scenario_suite import HeldOutScenarios
from experiments.phase4_5.explainability.decision_explainer import DecisionExplainer
from backend.app.experiment_loader.loader import sanitize_value
from backend.app.schemas.simulation import (
    ControllerOption,
    CustomScenarioConditions,
    DisturbanceEventSchema,
    ForecastHorizonPoint,
    ForecastSnapshot,
    ScenarioDescriptor,
    SimulationStartRequest,
    SimulationStateResponse,
    SimulationStepSnapshot,
)
from backend.app.schemas.system import (
    BusTelemetry,
    LineTelemetry,
    NetworkTopologyResponse,
)

ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = str(ROOT / "experiments" / "phase4" / "forecasting" / "models")
REWARD_CFG = str(ROOT / "experiments" / "phase4_4_4" / "configs" / "reward_v3_social.yaml")
ECON_CFG_PATH = ROOT / "experiments" / "phase4_4_4" / "configs" / "economics_v1.yaml"


BUS_ROLES = {
    0: "utility_pcc",
    1: "main_bus",
    2: "solar_pv",
    3: "wind_turbine",
    4: "battery_storage",
    5: "backup_generator",
    6: "critical_load",
    7: "flexible_load",
    8: "other_load",
}


class _CustomScenarioManager(ScenarioManager):
    """Wraps user-supplied interactive what-if conditions safely onto ScenarioManager."""

    def __init__(self, cond: CustomScenarioConditions):
        super().__init__(
            name="Custom What-If Scenario",
            description=(
                f"Cloud={cond.solar_cloud_severity:.0%}, "
                f"CritLoad={cond.critical_load_multiplier:.2f}x, "
                f"FlexLoad={cond.flexible_load_multiplier:.2f}x"
            ),
        )
        self._cond = cond
        if cond.solar_cloud_severity > 0.0:
            self.add_event("cloud", 0.0, 24.0, severity=float(cond.solar_cloud_severity))
        if cond.wind_speed_ms is not None:
            self.add_event("wind_lull", 0.0, 24.0, wind_speed_ms=float(cond.wind_speed_ms))
        if cond.grid_outage_enabled and cond.outage_end_hour > cond.outage_start_hour:
            self.add_event(
                "grid_outage",
                float(cond.outage_start_hour),
                float(cond.outage_end_hour),
                reason="Interactive user outage",
            )

    def get_demand_multipliers(self, hour: float) -> Dict[str, float]:
        return {
            "critical": float(self._cond.critical_load_multiplier),
            "flexible": float(self._cond.flexible_load_multiplier),
            "other": 1.0,
        }

    def is_sync_verified(self, hour: float) -> bool:
        return self.is_grid_available(hour)


def _extract_network_snapshot(
    net: Any,
    step: int,
    hour: float,
    converged: Optional[bool],
    grid_connected: Optional[bool],
    grid_available: Optional[bool],
    sync_verified: Optional[bool],
    data_source: str,
    state_dict: Optional[Dict[str, Any]] = None,
) -> NetworkTopologyResponse:
    buses: List[BusTelemetry] = []
    has_res_bus = hasattr(net, "res_bus") and net.res_bus is not None and not net.res_bus.empty
    for idx in net.bus.index:
        row = net.bus.loc[idx]
        vm_pu = None
        va_deg = None
        p_mw = None
        q_mvar = None
        if has_res_bus and idx in net.res_bus.index:
            r = net.res_bus.loc[idx]
            vm_pu = sanitize_value(r.get("vm_pu"))
            va_deg = sanitize_value(r.get("va_degree"))
            p_mw = sanitize_value(r.get("p_mw"))
            q_mvar = sanitize_value(r.get("q_mvar"))
        buses.append(
            BusTelemetry(
                bus_idx=int(idx),
                name=str(row.get("name", f"Bus {idx}")),
                role=BUS_ROLES.get(int(idx), "bus"),
                vn_kv=float(row.get("vn_kv", 11.0)),
                vm_pu=vm_pu,
                va_degree=va_deg,
                p_mw=p_mw,
                q_mvar=q_mvar,
                in_service=bool(row.get("in_service", True)),
                is_simulated=True,
            )
        )

    lines: List[LineTelemetry] = []
    has_res_line = hasattr(net, "res_line") and net.res_line is not None and not net.res_line.empty
    for l_idx in net.line.index:
        lrow = net.line.loc[l_idx]
        fb = int(lrow["from_bus"])
        tb = int(lrow["to_bus"])
        in_srv = bool(lrow.get("in_service", True))
        p_from = None
        p_to = None
        loading = None
        direction = "disconnected" if not in_srv else "idle"
        if has_res_line and l_idx in net.res_line.index and in_srv:
            rl = net.res_line.loc[l_idx]
            p_from = sanitize_value(rl.get("p_from_mw"))
            p_to = sanitize_value(rl.get("p_to_mw"))
            loading = sanitize_value(rl.get("loading_percent"))
            if p_from is not None:
                if abs(p_from) < 1e-4:
                    direction = "idle"
                elif p_from > 0:
                    direction = "forward"
                else:
                    direction = "reverse"
        lines.append(
            LineTelemetry(
                line_idx=int(l_idx),
                name=str(lrow.get("name", f"Line {l_idx}")),
                from_bus=fb,
                to_bus=tb,
                from_bus_name=str(net.bus.loc[fb, "name"]),
                to_bus_name=str(net.bus.loc[tb, "name"]),
                p_from_mw=p_from,
                p_to_mw=p_to,
                loading_percent=loading,
                in_service=in_srv,
                power_direction=direction,
            )
        )

    asset_summary: Dict[str, Any] = {}
    if state_dict:
        asset_summary = {
            "solar_mw": round(float(state_dict.get("solar_mw", 0.0)), 4),
            "wind_mw": round(float(state_dict.get("wind_mw", 0.0)), 4),
            "battery_mw": round(float(state_dict.get("battery_mw", 0.0)), 4),
            "battery_soc": round(float(state_dict.get("battery_soc", 0.0)), 2),
            "generator_mw": round(float(state_dict.get("generator_mw", 0.0)), 4),
            "generator_state": str(state_dict.get("generator_state", "OFF")),
            "grid_import_mw": round(float(state_dict.get("grid_import_mw", 0.0)), 4),
            "grid_export_mw": round(float(state_dict.get("grid_export_mw", 0.0)), 4),
            "served_critical_mw": round(float(state_dict.get("served_critical_mw", 0.0)), 4),
            "served_flexible_mw": round(float(state_dict.get("served_flexible_mw", 0.0)), 4),
            "served_other_mw": round(float(state_dict.get("served_other_mw", 0.0)), 4),
            "shed_critical_mw": round(float(state_dict.get("shed_critical_mw", 0.0)), 4),
        }

    return NetworkTopologyResponse(
        is_simulated=True,
        data_source=data_source,
        step=int(step),
        hour=round(float(hour), 2),
        converged=converged,
        grid_connected=grid_connected,
        grid_available=grid_available,
        sync_verified=sync_verified,
        buses=buses,
        lines=lines,
        asset_summary=asset_summary,
    )


def _extract_forecast_snapshot(
    env: MicrogridParallelEnv,
    current_hour: float,
    state_dict: Dict[str, Any],
) -> Optional[ForecastSnapshot]:
    s_act = float(state_dict.get("solar_mw", 0.0))
    w_act = float(state_dict.get("wind_mw", 0.0))
    d_act = float(state_dict.get("nominal_total_mw", 0.0))
    nl_act = d_act - (s_act + w_act)

    if not getattr(env, "use_forecasts", False) or not hasattr(env, "forecast_mgr"):
        return ForecastSnapshot(
            mode="disabled",
            current_hour=round(current_hour, 2),
            actual_solar_mw=round(s_act, 4),
            actual_wind_mw=round(w_act, 4),
            actual_demand_mw=round(d_act, 4),
            actual_net_load_mw=round(nl_act, 4),
            horizons=[],
        )

    lf = env.forecast_mgr.latest_forecast or {}
    h_steps = list(lf.get("horizons_steps", []))
    h_mins = list(lf.get("horizons_minutes", []))
    sol = lf.get("solar", {})
    wnd = lf.get("wind", {})
    dem = lf.get("demand", {})
    nl = lf.get("net_load", {})
    out_p = lf.get("outage_prob", [])
    stm_r = lf.get("storm_risk", [])

    points: List[ForecastHorizonPoint] = []
    for i, hs in enumerate(h_steps):
        hm = int(h_mins[i]) if i < len(h_mins) else int(hs) * 15
        points.append(
            ForecastHorizonPoint(
                horizon_step=int(hs),
                horizon_minutes=hm,
                target_hour=round((current_hour + hm / 60.0) % 24.0, 2),
                solar_q10=sanitize_value(sol.get("q10", [])[i]) if i < len(sol.get("q10", [])) else None,
                solar_q50=sanitize_value(sol.get("point", [])[i]) if i < len(sol.get("point", [])) else None,
                solar_q90=sanitize_value(sol.get("q90", [])[i]) if i < len(sol.get("q90", [])) else None,
                wind_q10=sanitize_value(wnd.get("q10", [])[i]) if i < len(wnd.get("q10", [])) else None,
                wind_q50=sanitize_value(wnd.get("point", [])[i]) if i < len(wnd.get("point", [])) else None,
                wind_q90=sanitize_value(wnd.get("q90", [])[i]) if i < len(wnd.get("q90", [])) else None,
                demand_q10=sanitize_value(dem.get("q10", [])[i]) if i < len(dem.get("q10", [])) else None,
                demand_q50=sanitize_value(dem.get("point", [])[i]) if i < len(dem.get("point", [])) else None,
                demand_q90=sanitize_value(dem.get("q90", [])[i]) if i < len(dem.get("q90", [])) else None,
                net_load_q10=sanitize_value(nl.get("q10", [])[i]) if i < len(nl.get("q10", [])) else None,
                net_load_q50=sanitize_value(nl.get("point", [])[i]) if i < len(nl.get("point", [])) else None,
                net_load_q90=sanitize_value(nl.get("q90", [])[i]) if i < len(nl.get("q90", [])) else None,
                outage_prob=sanitize_value(out_p[i]) if i < len(out_p) else None,
                storm_risk=sanitize_value(stm_r[i]) if i < len(stm_r) else None,
            )
        )

    return ForecastSnapshot(
        mode=str(lf.get("mode", env.forecast_mode_str)),
        current_hour=round(current_hour, 2),
        actual_solar_mw=round(s_act, 4),
        actual_wind_mw=round(w_act, 4),
        actual_demand_mw=round(d_act, 4),
        actual_net_load_mw=round(nl_act, 4),
        horizons=points,
    )


class SimulationService:
    """Thread-safe controller for interactive live simulation + WebSocket streaming."""

    def __init__(self):
        self._lock = threading.RLock()
        self._status: str = "idle"  # idle | running | paused | completed | error
        self._error_message: Optional[str] = None
        self._run_id: int = 0
        self._pause_event = threading.Event()
        self._pause_event.set()  # set = not paused
        self._stop_requested = False
        self._worker: Optional[threading.Thread] = None

        self._controller_id: Optional[str] = None
        self._controller_label: Optional[str] = None
        self._controller_mode: Optional[str] = None
        self._scenario_id: Optional[str] = None
        self._scenario_name: Optional[str] = None
        self._seed: int = 42
        self._step_delay_ms: int = 250
        self._forecast_mode: Optional[str] = "realistic"
        self._max_steps: int = 96

        self._history: List[SimulationStepSnapshot] = []
        self._latest_network: Optional[NetworkTopologyResponse] = None
        self._summary_economics: Optional[Dict[str, Any]] = None

        self._econ_cfg = EconomicConfig.from_yaml(ECON_CFG_PATH)
        self._mappo_cache: Dict[Tuple[str, str], Any] = {}

        # WebSocket subscribers: set of (loop, asyncio.Queue)
        self._subscribers: Set[Tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = set()
        self._sub_lock = threading.Lock()

        # Nominal pandapower network baseline for Digital Twin view prior to simulation start
        net, _ = create_microgrid()
        self._nominal_network = _extract_network_snapshot(
            net=net,
            step=0,
            hour=0.0,
            converged=None,
            grid_connected=True,
            grid_available=True,
            sync_verified=True,
            data_source="pandapower_nominal_topology",
        )

    def register_ws_subscriber(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> None:
        with self._sub_lock:
            self._subscribers.add((loop, queue))

    def unregister_ws_subscriber(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> None:
        with self._sub_lock:
            self._subscribers.discard((loop, queue))

    def _broadcast_event(self, payload: Dict[str, Any]) -> None:
        def _safe_enqueue(q: asyncio.Queue, item: Dict[str, Any]) -> None:
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(item)
            except asyncio.QueueFull:
                pass

        with self._sub_lock:
            subs = list(self._subscribers)
        for loop, q in subs:
            if not loop.is_closed():
                loop.call_soon_threadsafe(_safe_enqueue, q, payload)

    def list_controllers(self) -> List[ControllerOption]:
        ckpt_202 = ROOT / "experiments" / "phase4_4_4" / "checkpoints" / "bat_only_seed202" / "best_model.pt"
        ckpt_101 = ROOT / "experiments" / "phase4_4_4" / "checkpoints" / "bat_only_seed101" / "best_model.pt"
        ckpt_42 = ROOT / "experiments" / "phase4_4_4" / "checkpoints" / "bat_only_seed42" / "best_model.pt"
        comp_root = ROOT / "experiments" / "phase6_model_comparison" / "results"
        ckpt_mappo_matched_42 = comp_root / "mappo_matched" / "seed_42" / "checkpoints" / "best_model.pt"
        ckpt_ippo_42 = comp_root / "ippo" / "seed_42" / "checkpoints" / "best_model.pt"
        ckpt_happo_42 = comp_root / "happo" / "seed_42" / "checkpoints" / "best_model.pt"

        options: List[ControllerOption] = [
            ControllerOption(
                controller_id="bat_only_seed202_intrinsic",
                label="MAPPO — Battery-Shaped (Seed 202) [Intrinsic]",
                controller_type="mappo",
                mode="Intrinsic",
                adequacy_fallback=False,
                checkpoint_path=str(ckpt_202),
                forecast_mode="realistic",
                description="Primary Battery-Shaped MAPPO candidate (Seed 202) evaluated WITHOUT Adequacy Fallback (Safety Shield only).",
                is_primary_candidate=True,
            ),
            ControllerOption(
                controller_id="bat_only_seed202_hybrid",
                label="MAPPO — Battery-Shaped (Seed 202) [Hybrid +Fallback]",
                controller_type="mappo",
                mode="Hybrid (+Fallback)",
                adequacy_fallback=True,
                checkpoint_path=str(ckpt_202),
                forecast_mode="realistic",
                description="Hybrid system pairing MAPPO Battery-Shaped (Seed 202) with deterministic Adequacy Fallback overrides during deficits.",
                is_primary_candidate=False,
            ),
        ]

        if ckpt_mappo_matched_42.exists():
            options.extend([
                ControllerOption(
                    controller_id="mappo_matched_seed42_intrinsic",
                    label="MAPPO — Matched Multi-Seed (Seed 42) [Intrinsic]",
                    controller_type="mappo_matched",
                    mode="Intrinsic",
                    adequacy_fallback=False,
                    checkpoint_path=str(ckpt_mappo_matched_42),
                    forecast_mode="realistic",
                    description="Centralized-critic MAPPO (Matched comparative benchmark, Seed 42) evaluated WITHOUT Adequacy Fallback.",
                ),
                ControllerOption(
                    controller_id="mappo_matched_seed42_hybrid",
                    label="MAPPO — Matched Multi-Seed (Seed 42) [Hybrid +Fallback]",
                    controller_type="mappo_matched",
                    mode="Hybrid (+Fallback)",
                    adequacy_fallback=True,
                    checkpoint_path=str(ckpt_mappo_matched_42),
                    forecast_mode="realistic",
                    description="Hybrid system pairing Centralized-critic MAPPO (Seed 42) with Adequacy Fallback deficit protection.",
                ),
            ])

        if ckpt_ippo_42.exists():
            options.extend([
                ControllerOption(
                    controller_id="ippo_seed42_intrinsic",
                    label="IPPO — Independent Critics (Seed 42) [Intrinsic]",
                    controller_type="ippo",
                    mode="Intrinsic",
                    adequacy_fallback=False,
                    checkpoint_path=str(ckpt_ippo_42),
                    forecast_mode="realistic",
                    description="Decentralized Independent PPO (IPPO, Seed 42) evaluated WITHOUT Adequacy Fallback.",
                ),
                ControllerOption(
                    controller_id="ippo_seed42_hybrid",
                    label="IPPO — Independent Critics (Seed 42) [Hybrid +Fallback]",
                    controller_type="ippo",
                    mode="Hybrid (+Fallback)",
                    adequacy_fallback=True,
                    checkpoint_path=str(ckpt_ippo_42),
                    forecast_mode="realistic",
                    description="Hybrid system pairing IPPO (Seed 42) with deterministic Adequacy Fallback overrides.",
                ),
            ])

        if ckpt_happo_42.exists():
            options.extend([
                ControllerOption(
                    controller_id="happo_seed42_intrinsic",
                    label="HAPPO — Sequential Update (Seed 42) [Intrinsic]",
                    controller_type="happo",
                    mode="Intrinsic",
                    adequacy_fallback=False,
                    checkpoint_path=str(ckpt_happo_42),
                    forecast_mode="realistic",
                    description="Heterogeneous-Agent PPO (HAPPO, Seed 42) sequential update policy evaluated WITHOUT Adequacy Fallback.",
                ),
                ControllerOption(
                    controller_id="happo_seed42_hybrid",
                    label="HAPPO — Sequential Update (Seed 42) [Hybrid +Fallback]",
                    controller_type="happo",
                    mode="Hybrid (+Fallback)",
                    adequacy_fallback=True,
                    checkpoint_path=str(ckpt_happo_42),
                    forecast_mode="realistic",
                    description="Hybrid system pairing HAPPO (Seed 42) with deterministic Adequacy Fallback overrides.",
                ),
            ])

        options.extend([
            ControllerOption(
                controller_id="greedy",
                label="Greedy Controller (Rule Heuristic)",
                controller_type="greedy",
                mode="Heuristic",
                adequacy_fallback=False,
                forecast_mode="realistic",
                description="Instantaneous myopic heuristic baseline serving net demand greedily.",
            ),
            ControllerOption(
                controller_id="rule_based",
                label="Rule-Based EMS (Priority Heuristic)",
                controller_type="rule_based",
                mode="Heuristic",
                adequacy_fallback=False,
                forecast_mode="realistic",
                description="Priority-based deterministic microgrid energy management controller.",
            ),
            ControllerOption(
                controller_id="mpc",
                label="Lookahead Heuristic EMS (MPC-8)",
                controller_type="mpc",
                mode="Heuristic",
                adequacy_fallback=False,
                forecast_mode="realistic",
                description="Receding-horizon lookahead heuristic EMS baseline.",
            ),
        ])

        for seed_num, ckpt in [(101, ckpt_101), (42, ckpt_42)]:
            if ckpt.exists():
                options.append(
                    ControllerOption(
                        controller_id=f"bat_only_seed{seed_num}_intrinsic",
                        label=f"MAPPO — Battery-Shaped (Seed {seed_num}) [Intrinsic]",
                        controller_type="mappo",
                        mode="Intrinsic",
                        adequacy_fallback=False,
                        checkpoint_path=str(ckpt),
                        forecast_mode="realistic",
                        description=f"Retained Battery-Shaped MAPPO variant (Seed {seed_num}) in intrinsic mode.",
                    )
                )
        return options

    def list_scenarios(self) -> List[ScenarioDescriptor]:
        items: List[ScenarioDescriptor] = []

        # Benchmark A-F
        bench_methods = [
            ("scenario_a", ScenarioManager.scenario_a_normal),
            ("scenario_b", ScenarioManager.scenario_b_solar_collapse),
            ("scenario_c", ScenarioManager.scenario_c_grid_outage),
            ("scenario_d", ScenarioManager.scenario_d_severe_storm),
            ("scenario_e", ScenarioManager.scenario_e_prolonged_outage),
            ("scenario_f", ScenarioManager.scenario_f_grid_restoration),
        ]
        for sid, fn in bench_methods:
            sm = fn()
            evs = [
                DisturbanceEventSchema(
                    type=str(e["type"]),
                    start_hour=float(e["start_hour"]),
                    end_hour=float(e["end_hour"]),
                    details={k: sanitize_value(v) for k, v in e.items() if k not in ("type", "start_hour", "end_hour")},
                )
                for e in getattr(sm, "events", [])
            ]
            items.append(
                ScenarioDescriptor(
                    scenario_id=sid,
                    category="Benchmark A-F",
                    name=sm.name,
                    description=sm.description,
                    events=evs,
                    has_outage=any("outage" in e.type for e in evs),
                    has_storm=any("storm" in e.type for e in evs),
                )
            )

        # Held-Out H1-H6
        heldouts = HeldOutScenarios.get_all_heldout_scenarios()
        for sid, sm in heldouts.items():
            evs: List[DisturbanceEventSchema] = []
            for e in getattr(sm, "events", []):
                evs.append(
                    DisturbanceEventSchema(
                        type=str(e["type"]),
                        start_hour=float(e["start_hour"]),
                        end_hour=float(e["end_hour"]),
                        details={k: sanitize_value(v) for k, v in e.items() if k not in ("type", "start_hour", "end_hour")},
                    )
                )
            for lo in getattr(sm, "line_outages", []):
                evs.append(
                    DisturbanceEventSchema(
                        type="line_outage",
                        start_hour=float(lo["start_hour"]),
                        end_hour=float(lo["end_hour"]),
                        details={"line_idx": lo["line_idx"], "reason": lo.get("reason", "")},
                    )
                )
            for ds in getattr(sm, "demand_shifts", []):
                evs.append(
                    DisturbanceEventSchema(
                        type="demand_surge",
                        start_hour=float(ds["start_hour"]),
                        end_hour=float(ds["end_hour"]),
                        details={
                            "critical_mult": ds["critical_mult"],
                            "flexible_mult": ds["flexible_mult"],
                            "other_mult": ds["other_mult"],
                            "reason": ds.get("reason", ""),
                        },
                    )
                )
            for gd in getattr(sm, "generator_deratings", []):
                evs.append(
                    DisturbanceEventSchema(
                        type="generator_derating",
                        start_hour=float(gd["start_hour"]),
                        end_hour=float(gd["end_hour"]),
                        details={"capacity_factor": gd["capacity_factor"], "reason": gd.get("reason", "")},
                    )
                )
            for sn in getattr(sm, "sensor_noise_events", []):
                evs.append(
                    DisturbanceEventSchema(
                        type="sensor_degradation",
                        start_hour=float(sn["start_hour"]),
                        end_hour=float(sn["end_hour"]),
                        details={"gaussian_sigma": sn["gaussian_sigma"], "dropout_prob": sn["dropout_prob"]},
                    )
                )
            for ff in getattr(sm, "forecast_freeze_events", []):
                evs.append(
                    DisturbanceEventSchema(
                        type="forecast_freeze",
                        start_hour=float(ff["start_hour"]),
                        end_hour=float(ff["end_hour"]),
                        details={},
                    )
                )

            items.append(
                ScenarioDescriptor(
                    scenario_id=sid,
                    category="Held-Out H1-H6",
                    name=sm.name,
                    description=sm.description,
                    events=evs,
                    has_outage=any("outage" in e.type for e in evs),
                    has_storm=any("storm" in e.type for e in evs),
                    has_topology_shift=bool(getattr(sm, "line_outages", [])),
                    has_sensor_degradation=bool(getattr(sm, "sensor_noise_events", [])),
                    has_stale_forecast=bool(getattr(sm, "forecast_freeze_events", [])),
                )
            )

        return items

    def _get_rl_agent(self, controller_type: str, checkpoint_path: Path, forecast_mode: str) -> Any:
        key = (controller_type, str(checkpoint_path), forecast_mode)
        if key in self._mappo_cache:
            return self._mappo_cache[key]
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
        if controller_type in ("mappo", "mappo_matched"):
            # Import the training loader only when an RL policy is selected.
            from experiments.phase4_4_3.benchmark_phase4_4_3 import load_mappo_checkpoint

            expected_experiment = "bat_only" if controller_type == "mappo" else None
            agent = load_mappo_checkpoint(
                checkpoint_path,
                forecast_mode=forecast_mode,
                expected_experiment=expected_experiment,
            )
        elif controller_type == "ippo":
            from experiments.phase6_model_comparison.evaluation.controller_registry import load_ippo_checkpoint
            agent = load_ippo_checkpoint(checkpoint_path, forecast_mode=forecast_mode)
        elif controller_type == "happo":
            from happo import load_happo_checkpoint
            agent = load_happo_checkpoint(checkpoint_path, forecast_mode=forecast_mode)
        else:
            raise ValueError(f"Unsupported RL controller_type: {controller_type}")
        self._mappo_cache[key] = agent
        return agent

    def start(self, req: SimulationStartRequest) -> SimulationStateResponse:
        ctrl_map = {c.controller_id: c for c in self.list_controllers()}
        if req.controller_id not in ctrl_map:
            raise ValueError(f"Unknown controller_id: {req.controller_id}")
        scen_map = {s.scenario_id: s for s in self.list_scenarios()}
        if req.scenario_id not in scen_map:
            raise ValueError(f"Unknown scenario_id: {req.scenario_id}")

        ctrl_opt = ctrl_map[req.controller_id]
        scen_desc = scen_map[req.scenario_id]
        fmode = req.forecast_mode or ctrl_opt.forecast_mode or "realistic"
        if fmode not in ("realistic", "oracle"):
            raise ValueError(f"Invalid forecast_mode '{fmode}'. Must be 'realistic' or 'oracle'.")

        # Pre-validate checkpoint before launching background loop
        if ctrl_opt.controller_type in ("mappo", "mappo_matched", "ippo", "happo"):
            if not ctrl_opt.checkpoint_path:
                raise ValueError(f"No checkpoint path configured for {ctrl_opt.controller_id}")
            self._get_rl_agent(ctrl_opt.controller_type, Path(ctrl_opt.checkpoint_path), fmode)

        with self._lock:
            self._stop_existing_worker_locked()
            self._run_id += 1
            active_run_id = self._run_id
            self._stop_requested = False
            self._pause_event.set()
            self._error_message = None
            self._status = "running"
            self._controller_id = ctrl_opt.controller_id
            self._controller_label = ctrl_opt.label
            self._controller_mode = ctrl_opt.mode
            self._scenario_id = "custom_whatif" if req.custom_conditions is not None else scen_desc.scenario_id
            self._scenario_name = (
                "Custom Interactive Conditions" if req.custom_conditions is not None else scen_desc.name
            )
            self._seed = int(req.seed)
            self._step_delay_ms = int(req.step_delay_ms)
            self._forecast_mode = fmode
            self._history = []
            self._latest_network = None
            self._summary_economics = None

            self._worker = threading.Thread(
                target=self._run_loop,
                args=(
                    active_run_id,
                    ctrl_opt,
                    scen_desc.scenario_id,
                    int(req.seed),
                    fmode,
                    req.custom_conditions,
                ),
                daemon=True,
                name=f"gridguard-sim-{active_run_id}",
            )
            self._worker.start()

        return self.get_state()

    def _stop_existing_worker_locked(self) -> None:
        self._stop_requested = True
        self._pause_event.set()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=3.0)
        self._worker = None

    def pause(self) -> SimulationStateResponse:
        with self._lock:
            if self._status == "running":
                self._pause_event.clear()
                self._status = "paused"
                self._broadcast_event({"type": "status", "status": "paused", "current_step": len(self._history)})
        return self.get_state()

    def resume(self) -> SimulationStateResponse:
        with self._lock:
            if self._status == "paused":
                self._pause_event.set()
                self._status = "running"
                self._broadcast_event({"type": "status", "status": "running", "current_step": len(self._history)})
        return self.get_state()

    def reset(self) -> SimulationStateResponse:
        with self._lock:
            self._stop_existing_worker_locked()
            self._run_id += 1
            self._status = "idle"
            self._error_message = None
            self._history = []
            self._latest_network = None
            self._summary_economics = None
            self._broadcast_event({"type": "reset", "status": "idle"})
        return self.get_state()

    def set_speed(self, step_delay_ms: int) -> SimulationStateResponse:
        with self._lock:
            self._step_delay_ms = int(np.clip(step_delay_ms, 0, 5000))
        return self.get_state()

    def shutdown(self) -> None:
        with self._lock:
            self._stop_existing_worker_locked()

    def _run_loop(
        self,
        run_id: int,
        ctrl_opt: ControllerOption,
        scenario_id: str,
        seed: int,
        forecast_mode: str,
        custom_conditions: Optional[CustomScenarioConditions] = None,
    ) -> None:
        try:
            env = MicrogridParallelEnv(
                scenario_id=scenario_id,
                forecast_mode=forecast_mode,
                forecast_models_dir=MODELS_DIR,
                reward_config_path=REWARD_CFG,
                enable_adequacy_check=True,
                adequacy_fallback=bool(ctrl_opt.adequacy_fallback),
                enable_constraint_costs=True,
            )

            if ctrl_opt.controller_type in ("mappo", "mappo_matched", "ippo", "happo"):
                ctrl_obj = self._get_rl_agent(ctrl_opt.controller_type, Path(ctrl_opt.checkpoint_path), forecast_mode)
            elif ctrl_opt.controller_type == "greedy":
                ctrl_obj = GreedyController()
            elif ctrl_opt.controller_type == "rule_based":
                ctrl_obj = RuleBasedEMS()
            elif ctrl_opt.controller_type == "mpc":
                ctrl_obj = MPCController(horizon_steps=8, dt_hours=env.dt_hours, name=ctrl_opt.label)
            else:
                raise ValueError(f"Unsupported controller type: {ctrl_opt.controller_type}")

            random.seed(seed)
            np.random.seed(seed)
            reset_opts: Dict[str, Any] = {"scenario_id": scenario_id}
            if custom_conditions is not None:
                reset_opts["scenario_obj"] = _CustomScenarioManager(custom_conditions)
            obs, _ = env.reset(seed=seed, options=reset_opts)
            if hasattr(ctrl_obj, "reset"):
                ctrl_obj.reset()

            explainer = DecisionExplainer(controller_name=ctrl_opt.label, mode=ctrl_opt.mode)
            cum_reward = 0.0
            done = False

            while env.agents and not done:
                with self._lock:
                    if self._stop_requested or self._run_id != run_id:
                        return
                self._pause_event.wait()
                with self._lock:
                    if self._stop_requested or self._run_id != run_id:
                        return
                    delay_sec = self._step_delay_ms / 1000.0

                step_idx = int(env.current_step)
                pre_state = dict(env.episode_history[-1])
                obs_snap = {ag: np.asarray(obs[ag], dtype=np.float32).copy() for ag in AGENTS if ag in obs}

                if ctrl_opt.controller_type in ("mappo", "mappo_matched", "ippo", "happo"):
                    raw_actions, _ = ctrl_obj.select_actions(obs, deterministic=True)
                else:
                    raw_actions = ctrl_obj.select_actions(obs, pre_state)

                obs, rewards, terms, truncs, infos = env.step(raw_actions)
                shared_rew = float(rewards.get("battery", 0.0))
                cum_reward += shared_rew
                done = any(terms.values()) or any(truncs.values())

                post_state = dict(env.episode_history[-1])
                bat_info = infos.get("battery", {})
                sh_diag = bat_info.get("shield_diagnostics", {}) or {}
                ad_diag = bat_info.get("adequacy_diagnostics", {}) or {}
                rb = bat_info.get("reward_breakdown", {}) or {}

                prev_ag_len = len(explainer.agent_records)
                prev_int_len = len(explainer.intervention_records)
                explainer.record_step(
                    scenario_id=scenario_id,
                    eval_seed=seed,
                    step_idx=step_idx,
                    pre_state=pre_state,
                    obs_dict=obs_snap,
                    raw_actions=raw_actions,
                    post_state=post_state,
                    shield_diag=sh_diag,
                    adeq_diag=ad_diag,
                    reward_breakdown=rb,
                    shared_reward=shared_rew,
                )
                step_agents = explainer.agent_records[prev_ag_len:]
                step_ints = explainer.intervention_records[prev_int_len:]

                cum_econ = compute_episode_economics(env.episode_history, dt_hours=env.dt_hours, config=self._econ_cfg)
                net_snap = _extract_network_snapshot(
                    net=env.net,
                    step=step_idx,
                    hour=float(pre_state.get("hour", step_idx * 0.25)),
                    converged=bool(post_state.get("converged", True)),
                    grid_connected=bool(post_state.get("grid_connected", True)),
                    grid_available=bool(post_state.get("grid_available", True)),
                    sync_verified=bool(post_state.get("sync_verified", True)),
                    data_source="live_simulation",
                    state_dict=post_state,
                )
                fc_snap = _extract_forecast_snapshot(
                    env=env,
                    current_hour=float(post_state.get("hour", (step_idx + 1) * 0.25)),
                    state_dict=post_state,
                )

                alerts: List[str] = []
                if not bool(post_state.get("grid_connected", True)):
                    alerts.append("ISLANDED_MODE: Microgrid disconnected from utility PCC")
                if bool(post_state.get("is_storm_warning", False)):
                    alerts.append("STORM_WARNING: Severe weather advisory active")
                if float(post_state.get("shed_critical_mw", 0.0)) > 1e-4:
                    alerts.append(f"CRITICAL_SHED: {float(post_state['shed_critical_mw']):.3f} MW unserved critical load")
                if bool(ad_diag.get("adequacy_intervention", False)):
                    alerts.append("ADEQUACY_FALLBACK: Deterministic fallback intervened on supply shortfall")
                if bool(sh_diag.get("any_emergency", False)):
                    alerts.append("EMERGENCY_SHIELD_OVERRIDE: SafetyShield executed emergency override")

                sol = float(post_state.get("solar_mw", 0.0))
                wnd = float(post_state.get("wind_mw", 0.0))
                gen = float(post_state.get("generator_mw", 0.0))

                step_snap = SimulationStepSnapshot(
                    step=step_idx,
                    hour=round(float(pre_state.get("hour", step_idx * 0.25)), 2),
                    timestamp=step_agents[0]["timestamp"] if step_agents else f"{int(step_idx * 0.25):02d}:00",
                    solar_mw=round(sol, 4),
                    wind_mw=round(wnd, 4),
                    solar_available_mw=round(float(post_state.get("solar_available_mw", sol)), 4),
                    wind_available_mw=round(float(post_state.get("wind_available_mw", wnd)), 4),
                    total_generation_mw=round(sol + wnd + gen, 4),
                    nominal_critical_mw=round(float(post_state.get("nominal_critical_mw", 0.0)), 4),
                    nominal_flexible_mw=round(float(post_state.get("nominal_flexible_mw", 0.0)), 4),
                    nominal_other_mw=round(float(post_state.get("nominal_other_mw", 0.0)), 4),
                    nominal_total_mw=round(float(post_state.get("nominal_total_mw", 0.0)), 4),
                    served_critical_mw=round(float(post_state.get("served_critical_mw", 0.0)), 4),
                    served_flexible_mw=round(float(post_state.get("served_flexible_mw", 0.0)), 4),
                    served_other_mw=round(float(post_state.get("served_other_mw", 0.0)), 4),
                    shed_critical_mw=round(float(post_state.get("shed_critical_mw", 0.0)), 4),
                    shed_flexible_mw=round(float(post_state.get("shed_flexible_mw", 0.0)), 4),
                    shed_other_mw=round(float(post_state.get("shed_other_mw", 0.0)), 4),
                    battery_mw=round(float(post_state.get("battery_mw", 0.0)), 4),
                    battery_soc=round(float(post_state.get("battery_soc", 0.0)), 2),
                    generator_mw=round(gen, 4),
                    generator_state=str(post_state.get("generator_state", "OFF")),
                    step_fuel_l=round(float(post_state.get("step_fuel_l", 0.0)), 4),
                    total_fuel_l=round(float(post_state.get("total_fuel_l", 0.0)), 4),
                    grid_connected=bool(post_state.get("grid_connected", True)),
                    grid_available=bool(post_state.get("grid_available", True)),
                    sync_verified=bool(post_state.get("sync_verified", True)),
                    grid_import_mw=round(float(post_state.get("grid_import_mw", 0.0)), 4),
                    grid_export_mw=round(float(post_state.get("grid_export_mw", 0.0)), 4),
                    pcc_voltage_pu=round(float(post_state.get("pcc_voltage_pu", 1.0)), 4),
                    converged=bool(post_state.get("converged", True)),
                    balance_error_mw=round(float(post_state.get("balance_error_mw", 0.0)), 6),
                    is_storm_warning=bool(post_state.get("is_storm_warning", False)),
                    shared_reward=round(shared_rew, 4),
                    cumulative_reward=round(cum_reward, 4),
                    reward_breakdown={k: round(float(v), 5) for k, v in rb.items() if isinstance(v, (int, float, np.number))},
                    operating_cost_usd_cum=round(float(cum_econ["operating_cost_usd"]), 2),
                    true_social_cost_usd_cum=round(float(cum_econ["true_social_cost_usd"]), 2),
                    critical_reliability_pct_cum=round(float(cum_econ["critical_reliability_pct"]), 3),
                    critical_ens_mwh_cum=round(float(cum_econ["critical_ens_mwh"]), 5),
                    shield_any_intervention=bool(sh_diag.get("any_intervention", False)),
                    shield_any_emergency=bool(sh_diag.get("any_emergency", False)),
                    shield_any_boundary=bool(sh_diag.get("any_boundary", False)),
                    adequacy_intervention=bool(ad_diag.get("adequacy_intervention", False)),
                    inaction_detected=bool(ad_diag.get("inaction_detected", False)),
                    agent_decisions=sanitize_value(step_agents),
                    interventions=sanitize_value(step_ints),
                    forecast=fc_snap,
                    network=net_snap.model_dump(),
                    alerts=alerts,
                )

                with self._lock:
                    if self._stop_requested or self._run_id != run_id:
                        return
                    self._history.append(step_snap)
                    self._latest_network = net_snap
                    self._summary_economics = sanitize_value(cum_econ)

                self._broadcast_event({
                    "type": "step",
                    "status": "running",
                    "current_step": step_idx + 1,
                    "max_steps": self._max_steps,
                    "step_snapshot": step_snap.model_dump(),
                    "summary_economics": sanitize_value(cum_econ),
                })

                if delay_sec > 0:
                    time.sleep(delay_sec)

            with self._lock:
                if not self._stop_requested and self._run_id == run_id:
                    self._status = "completed"
                    self._broadcast_event({
                        "type": "completed",
                        "status": "completed",
                        "current_step": len(self._history),
                        "summary_economics": self._summary_economics,
                    })

        except Exception as exc:
            with self._lock:
                if self._run_id == run_id:
                    self._status = "error"
                    self._error_message = str(exc)
            self._broadcast_event({"type": "error", "status": "error", "error_message": str(exc)})

    def get_state(self, include_history: bool = True) -> SimulationStateResponse:
        with self._lock:
            hist = list(self._history) if include_history else []
            latest = self._history[-1] if self._history else None
            return SimulationStateResponse(
                status=self._status,
                error_message=self._error_message,
                controller_id=self._controller_id,
                controller_label=self._controller_label,
                controller_mode=self._controller_mode,
                scenario_id=self._scenario_id,
                scenario_name=self._scenario_name,
                seed=self._seed,
                step_delay_ms=self._step_delay_ms,
                forecast_mode=self._forecast_mode,
                current_step=len(self._history),
                max_steps=self._max_steps,
                latest_step=latest,
                history=hist,
                summary_economics=self._summary_economics,
            )

    def get_network(self) -> NetworkTopologyResponse:
        with self._lock:
            if self._latest_network is not None:
                return self._latest_network
            return self._nominal_network


simulation_service = SimulationService()
