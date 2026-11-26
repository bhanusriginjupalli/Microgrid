import React, { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import {
  Activity,
  BarChart3,
  Bot,
  Brain,
  CloudSun,
  Menu,
  Network,
  Pause,
  Play,
  RotateCcw,
  ShieldAlert,
  Sliders,
  Database,
  X,
  Zap,
} from "lucide-react";
import { useSimulation } from "../context/SimulationContext";
import { StatusBadge, modeBadgeTone } from "../components/StatusBadge";
import { RescueEventOverlay } from "../components/RescueEventOverlay";
import { formatModelLabel } from "../components/episodes/episodeHelpers";

const CLIENT_OPERATIONAL_NAV = [
  { to: "/", label: "1. System Overview", icon: Activity },
  { to: "/digital-twin", label: "2. Digital Twin", icon: Network },
  { to: "/energy-management", label: "3. Energy Management", icon: Zap },
  { to: "/multi-agent-control", label: "4. Multi-Agent Control", icon: Bot },
  { to: "/forecasting", label: "5. Forecasting", icon: CloudSun },
  { to: "/resilience-simulation", label: "6. Resilience Simulation", icon: ShieldAlert },
];

const TECHNICAL_RESEARCH_NAV = [
  { to: "/explainable-ai", label: "7. Explainable AI", icon: Brain },
  { to: "/performance-analytics", label: "8. Performance Analytics", icon: BarChart3 },
  { to: "/saved-episodes", label: "9. Saved Episode Library", icon: Database },
];

const CLIENT_CONTROLLER_OPTIONS = [
  { controller_id: "bat_only_seed202_hybrid", label: "Recommended AI Hybrid Controller" },
  { controller_id: "rule_based", label: "Priority Rule Energy Manager" },
  { controller_id: "greedy", label: "Fast Greedy Controller" },
];

export const DashboardShell: React.FC = () => {
  const {
    audienceMode,
    setAudienceMode,
    scenarios,
    controllers,
    simState,
    activeSnapshot,
    selectedStepIndex,
    setSelectedStepIndex,
    wsConnected,
    actionError,
    clearError,
    startSim,
    pauseSim,
    resumeSim,
    resetSim,
    updateSpeed,
  } = useSimulation();

  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [controllerId, setControllerId] = useState("bat_only_seed202_intrinsic");
  const [scenarioId, setScenarioId] = useState("scenario_h2");
  const [seed, setSeed] = useState(42);
  const [delayMs, setDelayMs] = useState(120);
  const [forecastMode, setForecastMode] = useState("realistic");

  const status = simState?.status ?? "idle";
  const histLen = simState?.history.length ?? 0;

  return (
    <div className="min-h-screen flex flex-col bg-gridbg text-slate-100">
      {/* Persistent Top Identity Header */}
      <header className="sticky top-0 z-30 border-b border-gridborder bg-[#09111f]/95 backdrop-blur">
        <div className="px-4 py-2.5 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => setSidebarOpen((v) => !v)}
              className="lg:hidden p-1.5 rounded border border-gridborder text-slate-300 hover:bg-gridcard"
              aria-label="Toggle Navigation"
            >
              {sidebarOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
            </button>
            <div className="w-8 h-8 rounded bg-cyan-500/15 border border-cyan-500/40 flex items-center justify-center text-cyan-300 font-bold text-sm">
              GG
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-sm md:text-base font-bold tracking-wider text-slate-100">
                  GRIDGUARD AI
                </h1>
                <StatusBadge
                  label={audienceMode === "client" ? "Client Operations View" : "Microgrid Research Platform"}
                  tone="indigo"
                />
              </div>
              <p className="text-[11px] text-slate-400">
                Autonomous Multi-Agent Intelligence for Self-Healing, Resilient Microgrids
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <div className="px-2.5 py-1 rounded border border-amber-500/50 bg-amber-950/35 text-amber-300 text-[11px] font-mono font-semibold tracking-wide">
              SIMULATION ENVIRONMENT — NOT CONNECTED TO PHYSICAL GRID
            </div>
            <StatusBadge
              label={wsConnected ? "WS LIVE" : "WS STANDBY"}
              tone={wsConnected ? "emerald" : "slate"}
            />
            {simState?.controller_mode && (
              <StatusBadge
                label={simState.controller_mode}
                tone={modeBadgeTone(simState.controller_mode)}
              />
            )}
          </div>
        </div>

        {/* Simulation Control Toolbar */}
        <div className="px-4 py-2 border-t border-gridborder/80 bg-[#0b1526] flex flex-wrap items-center gap-2.5 text-xs">
          <div className="flex items-center gap-1.5 text-slate-400 font-medium">
            <Sliders className="w-3.5 h-3.5 text-cyan-400" />
            <span>Sim Control:</span>
          </div>

          <select
            aria-label="Controller Selection"
            value={
              audienceMode === "client" &&
              !CLIENT_CONTROLLER_OPTIONS.some((c) => c.controller_id === controllerId)
                ? "bat_only_seed202_hybrid"
                : controllerId
            }
            onChange={(e) => setControllerId(e.target.value)}
            className="bg-gridcard border border-gridborder rounded px-2 py-1 text-xs text-slate-100 focus:outline-none focus:border-cyan-500"
          >
            {(audienceMode === "client"
              ? CLIENT_CONTROLLER_OPTIONS
              : controllers.length > 0
              ? controllers
              : [
                  { controller_id: "bat_only_seed202_intrinsic", label: "MAPPO — Battery-Shaped (Seed 202) [Intrinsic]" },
                  { controller_id: "bat_only_seed202_hybrid", label: "MAPPO — Battery-Shaped (Seed 202) [Hybrid +Fallback]" },
                  { controller_id: "mappo_matched_seed42_intrinsic", label: "MAPPO — Matched Multi-Seed (Seed 42) [Intrinsic]" },
                  { controller_id: "ippo_seed42_intrinsic", label: "IPPO — Independent Critics (Seed 42) [Intrinsic]" },
                  { controller_id: "happo_seed42_intrinsic", label: "HAPPO — Sequential Update (Seed 42) [Intrinsic]" },
                  { controller_id: "greedy", label: "Greedy Controller (Rule Heuristic)" },
                  { controller_id: "rule_based", label: "Rule-Based EMS (Priority Heuristic)" },
                ]
            ).map((c) => (
              <option key={c.controller_id} value={c.controller_id}>
                {audienceMode === "client" ? c.label : formatModelLabel(c.label)}
              </option>
            ))}
          </select>

          <select
            aria-label="Scenario Selection"
            value={scenarioId}
            onChange={(e) => setScenarioId(e.target.value)}
            className="bg-gridcard border border-gridborder rounded px-2 py-1 text-xs text-slate-100 focus:outline-none focus:border-cyan-500"
          >
            {(scenarios.length > 0
              ? scenarios
              : [
                  { scenario_id: "scenario_h2", name: "Scenario H2: Renewable Drought", category: "Held-Out H1-H6" },
                  { scenario_id: "scenario_a", name: "Scenario A: Normal Operation", category: "Benchmark A-F" },
                ]
            ).map((s) => (
              <option key={s.scenario_id} value={s.scenario_id}>
                [{s.category}] {s.name}
              </option>
            ))}
          </select>

          <label className="flex items-center gap-1 text-slate-400">
            <span>Seed</span>
            <input
              type="number"
              min={0}
              max={99999}
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value) || 0)}
              className="w-16 bg-gridcard border border-gridborder rounded px-2 py-1 font-mono text-xs text-slate-100"
            />
          </label>

          <select
            aria-label="Forecast Mode"
            value={forecastMode}
            onChange={(e) => setForecastMode(e.target.value)}
            className="bg-gridcard border border-gridborder rounded px-2 py-1 text-xs text-slate-100"
          >
            <option value="realistic">Forecast: Realistic</option>
            <option value="oracle">Forecast: Oracle</option>
          </select>

          <select
            aria-label="Step Delay"
            value={delayMs}
            onChange={(e) => {
              const v = Number(e.target.value);
              setDelayMs(v);
              void updateSpeed(v);
            }}
            className="bg-gridcard border border-gridborder rounded px-2 py-1 text-xs text-slate-100"
          >
            <option value={0}>Speed: Instant (0ms)</option>
            <option value={80}>Speed: Fast (80ms)</option>
            <option value={120}>Speed: Normal (120ms)</option>
            <option value={300}>Speed: Slow (300ms)</option>
            <option value={750}>Speed: Step Inspection (750ms)</option>
          </select>

          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={() =>
                void startSim({
                  controller_id: controllerId,
                  scenario_id: scenarioId,
                  seed,
                  step_delay_ms: delayMs,
                  forecast_mode: forecastMode,
                })
              }
              className="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-cyan-600 hover:bg-cyan-500 text-white font-medium"
            >
              <Play className="w-3 h-3" /> Start
            </button>

            {status === "running" && (
              <button
                type="button"
                onClick={() => void pauseSim()}
                className="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-amber-600 hover:bg-amber-500 text-white font-medium"
              >
                <Pause className="w-3 h-3" /> Pause
              </button>
            )}

            {status === "paused" && (
              <button
                type="button"
                onClick={() => void resumeSim()}
                className="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-emerald-600 hover:bg-emerald-500 text-white font-medium"
              >
                <Play className="w-3 h-3" /> Resume
              </button>
            )}

            <button
              type="button"
              onClick={() => void resetSim()}
              className="inline-flex items-center gap-1 px-2.5 py-1 rounded border border-gridborder bg-gridcard hover:bg-slate-800 text-slate-200"
            >
              <RotateCcw className="w-3 h-3" /> Reset
            </button>
          </div>

          <div className="ml-auto flex items-center gap-3 font-mono text-[11px] text-slate-300">
            <span>
              Status:{" "}
              <strong
                className={
                  status === "running"
                    ? "text-emerald-400"
                    : status === "error"
                    ? "text-red-400"
                    : status === "completed"
                    ? "text-cyan-400"
                    : "text-amber-300"
                }
              >
                {status.toUpperCase()}
              </strong>
            </span>
            <span>
              Step: {activeSnapshot ? activeSnapshot.step + 1 : 0}/{simState?.max_steps ?? 96} (
              {activeSnapshot?.timestamp ?? "00:00"})
            </span>
          </div>
        </div>

        {/* Historical Step Scrubber when history exists */}
        {histLen > 1 && (
          <div className="px-4 py-1.5 border-t border-gridborder/60 bg-[#080f1c] flex items-center gap-3 text-[11px] text-slate-400">
            <span className="font-mono whitespace-nowrap">Timestep Scrubber:</span>
            <input
              type="range"
              min={0}
              max={histLen - 1}
              value={selectedStepIndex ?? histLen - 1}
              onChange={(e) => setSelectedStepIndex(Number(e.target.value))}
              className="flex-1 accent-cyan-400"
            />
            {selectedStepIndex !== null ? (
              <button
                type="button"
                onClick={() => setSelectedStepIndex(null)}
                className="px-2 py-0.5 rounded bg-cyan-500/20 border border-cyan-500/40 text-cyan-300 font-mono"
              >
                Pin to Live ({histLen})
              </button>
            ) : (
              <span className="font-mono text-emerald-400">LIVE TAIL</span>
            )}
          </div>
        )}

        {(actionError || simState?.error_message) && (
          <div className="px-4 py-2 bg-red-950/60 border-t border-red-500/40 text-xs text-red-200 flex items-center justify-between">
            <span>{actionError ?? simState?.error_message}</span>
            {actionError && (
              <button type="button" onClick={clearError} className="underline ml-3">
                Dismiss
              </button>
            )}
          </div>
        )}
      </header>

      {/* Main Content Area with Responsive Sidebar */}
      <div className="flex flex-1 min-h-0">
        <aside
          className={`${
            sidebarOpen ? "fixed inset-y-0 left-0 z-40 w-64 pt-16" : "hidden"
          } lg:static lg:block lg:w-60 shrink-0 border-r border-gridborder bg-gridpanel`}
        >
          <div className="px-3 pt-3 pb-1">
            <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-1.5">
              Presentation Mode
            </div>
            <div className="grid grid-cols-2 gap-1 p-1 rounded-md bg-gridbg border border-gridborder text-[11px]">
              <button
                type="button"
                onClick={() => setAudienceMode("client")}
                className={`py-1 rounded font-medium transition-colors ${
                  audienceMode === "client"
                    ? "bg-cyan-500/20 text-cyan-200 border border-cyan-500/40"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                Client Demo
              </button>
              <button
                type="button"
                onClick={() => setAudienceMode("technical")}
                className={`py-1 rounded font-medium transition-colors ${
                  audienceMode === "technical"
                    ? "bg-indigo-500/25 text-indigo-200 border border-indigo-400/50"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                Technical
              </button>
            </div>
          </div>

          <nav className="p-3 space-y-1">
            <div className="px-2 pt-1 pb-1 text-[10px] uppercase tracking-wider text-slate-500 font-semibold">
              Operations &amp; Live Twin
            </div>
            {CLIENT_OPERATIONAL_NAV.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                end={to === "/"}
                onClick={() => setSidebarOpen(false)}
                className={({ isActive }) =>
                  `flex items-center gap-2.5 px-3 py-2 rounded-md text-xs font-medium transition-colors ${
                    isActive
                      ? "bg-cyan-500/15 text-cyan-300 border border-cyan-500/40"
                      : "text-slate-300 hover:bg-gridcard hover:text-white border border-transparent"
                  }`
                }
              >
                <Icon className="w-4 h-4 shrink-0" />
                <span>{label}</span>
              </NavLink>
            ))}

            {audienceMode === "technical" && (
              <>
                <div className="px-2 pt-3 pb-1 text-[10px] uppercase tracking-wider text-slate-500 font-semibold border-t border-gridborder/60 mt-2">
                  Models &amp; Research Audit
                </div>
                {TECHNICAL_RESEARCH_NAV.map(({ to, label, icon: Icon }) => (
                  <NavLink
                    key={to}
                    to={to}
                    onClick={() => setSidebarOpen(false)}
                    className={({ isActive }) =>
                      `flex items-center gap-2.5 px-3 py-2 rounded-md text-xs font-medium transition-colors ${
                        isActive
                          ? "bg-cyan-500/15 text-cyan-300 border border-cyan-500/40"
                          : "text-slate-300 hover:bg-gridcard hover:text-white border border-transparent"
                      }`
                    }
                  >
                    <Icon className="w-4 h-4 shrink-0" />
                    <span>{label}</span>
                  </NavLink>
                ))}
              </>
            )}
          </nav>

          {audienceMode === "technical" && (
            <div className="m-3 p-3 rounded-lg bg-gridbg/70 border border-gridborder text-[11px] space-y-1.5 text-slate-400">
              <div className="font-semibold text-slate-200">Research Attribution Rule</div>
              <p className="leading-relaxed">
                <strong className="text-amber-300">Intrinsic</strong> evaluates the learned MARL
                policy (MAPPO / IPPO / HAPPO) without adequacy overrides.{" "}
                <strong className="text-cyan-300">Hybrid (+Fallback)</strong> pairs RL with
                deterministic adequacy recovery.
              </p>
            </div>
          )}
        </aside>

        <main className="flex-1 p-4 md:p-6 overflow-y-auto min-w-0">
          <Outlet />
        </main>
      </div>
      <RescueEventOverlay />
    </div>
  );
};
