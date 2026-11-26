import React, { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  FastForward,
  Pause,
  Play,
  RotateCcw,
  ShieldAlert,
  SkipBack,
  SkipForward,
} from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../../services/api";
import { EmptyState } from "../EmptyState";
import { StatusBadge, modeBadgeTone } from "../StatusBadge";
import {
  GRID_RESILIENCE_LABELS,
  LOAD_MGMT_LABELS,
  TRACK_LABELS,
  fmtNum,
  formatActionSummary,
  formatModelLabel,
  reliabilityTone,
} from "./episodeHelpers";
import type { EpisodeTraceResponse, EpisodeTraceStep } from "../../types/api";

interface EpisodeReplayViewTabProps {
  episodeId: string | null;
  onSelectEpisodeId: (id: string) => void;
}

const DEFAULT_REPLAY_CHOICES = [
  {
    label: "H2 · MAPPO (Matched Multi-Seed) Track A (Seed 42, Eval 42)",
    id: "track_a_intrinsic__mappo_matched__seed42__scenario_h2__eval42",
  },
  {
    label: "H2 · MAPPO (Matched Multi-Seed) Track C Hybrid (Seed 42, Eval 42)",
    id: "track_c_hybrid__mappo_matched__seed42__scenario_h2__eval42",
  },
  {
    label: "H2 · IPPO (Independent Critics) Track A (Seed 42, Eval 42)",
    id: "track_a_intrinsic__ippo__seed42__scenario_h2__eval42",
  },
  {
    label: "H2 · HAPPO (Sequential Update) Track A (Seed 42, Eval 42)",
    id: "track_a_intrinsic__happo__seed42__scenario_h2__eval42",
  },
  {
    label: "H2 · Rule-Based EMS Track A (eval42)",
    id: "track_a_intrinsic__rule_based_ems__seeddeterministic__scenario_h2__eval42",
  },
];

const MILESTONE_JUMPS = [
  { step: 0, label: "0 · 00:00" },
  { step: 24, label: "24 · 06:00" },
  { step: 48, label: "48 · 12:00" },
  { step: 72, label: "72 · 18:00" },
  { step: 95, label: "95 · 23:45" },
];

export const EpisodeReplayViewTab: React.FC<EpisodeReplayViewTabProps> = ({
  episodeId,
  onSelectEpisodeId,
}) => {
  const activeId = episodeId || DEFAULT_REPLAY_CHOICES[0].id;
  const [customInput, setCustomInput] = useState(activeId);
  const [trace, setTrace] = useState<EpisodeTraceResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speedMultiplier, setSpeedMultiplier] = useState<1 | 2 | 4>(1);

  useEffect(() => {
    setCustomInput(activeId);
    setPlaying(false);
    setStepIdx(0);
    const controller = new AbortController();
    setLoading(true);
    setError(null);

    api
      .getSavedEpisodeTrace(activeId, controller.signal)
      .then((res) => {
        setTrace(res);
        setStepIdx(0);
      })
      .catch((err: unknown) => {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setError(err instanceof Error ? err.message : "Failed to load episode trace.");
        setTrace(null);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });

    return () => controller.abort();
  }, [activeId]);

  const maxStep = Math.max(0, (trace?.steps.length ?? 96) - 1);

  useEffect(() => {
    if (!playing || !trace) return;
    const delayMs = speedMultiplier === 4 ? 180 : speedMultiplier === 2 ? 400 : 800;
    const timer = window.setInterval(() => {
      setStepIdx((prev) => {
        if (prev >= maxStep) {
          setPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, delayMs);
    return () => window.clearInterval(timer);
  }, [playing, speedMultiplier, maxStep, trace]);

  const currentStep: EpisodeTraceStep | null = trace?.steps?.[stepIdx] ?? null;
  const meta = (trace?.metadata ?? {}) as Record<string, unknown>;
  const metrics = (trace?.metrics ?? {}) as Record<string, number | string | null>;

  const timelineData = useMemo(() => {
    return (trace?.steps ?? []).map((s) => ({
      step_idx: s.step_idx,
      timestamp_label: s.timestamp_label,
      soc: Number(s.battery_soc_pct.toFixed(2)),
      re_mw: Number((s.solar_mw + s.wind_mw).toFixed(3)),
      gen_mw: Number(s.generator_mw.toFixed(3)),
      shed_crit: Number(s.shed_critical_mw.toFixed(3)),
    }));
  }, [trace]);

  const firstOutageStep = useMemo(() => {
    return trace?.steps.findIndex((s) => !s.grid_available || s.scenario_events.length > 0) ?? -1;
  }, [trace]);

  const firstFallbackStep = useMemo(() => {
    return trace?.steps.findIndex((s) => s.fallback_events?.adequacy_intervention) ?? -1;
  }, [trace]);

  const loadCustomId = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = customInput.trim();
    if (trimmed) onSelectEpisodeId(trimmed);
  };

  const agentRows = useMemo(() => {
    if (!currentStep) return [];
    const prop = currentStep.proposed_actions ?? {};
    const exec = currentStep.executed_actions ?? {};
    const loadProp = Number(prop.load_management ?? 0);
    const loadExec = Number(exec.load_management ?? 0);
    const gridProp = Number(prop.grid_resilience ?? 0);
    const gridExec = Number(exec.grid_resilience ?? 0);

    return [
      {
        agent: "Renewable Curtailment",
        proposed: formatActionSummary(prop.renewable),
        executed: formatActionSummary(exec.renewable),
        physical: `Solar ${fmtNum(currentStep.solar_mw, 3)} / Wind ${fmtNum(currentStep.wind_mw, 3)} MW`,
        altered:
          formatActionSummary(prop.renewable) !== formatActionSummary(exec.renewable),
      },
      {
        agent: "Battery Storage",
        proposed: formatActionSummary(prop.battery),
        executed: formatActionSummary(exec.battery),
        physical: `Applied ${fmtNum(
          exec.applied_battery_mw ?? currentStep.battery_mw,
          3
        )} MW (SOC ${fmtNum(currentStep.battery_soc_pct, 1)}%)`,
        altered: formatActionSummary(prop.battery) !== formatActionSummary(exec.battery),
      },
      {
        agent: "Diesel Generator",
        proposed: formatActionSummary(prop.generator),
        executed: formatActionSummary(exec.generator),
        physical: `Applied ${fmtNum(
          exec.applied_generator_mw ?? currentStep.generator_mw,
          3
        )} MW (${currentStep.generator_state})`,
        altered:
          formatActionSummary(prop.generator) !== formatActionSummary(exec.generator),
      },
      {
        agent: "Load Management",
        proposed: LOAD_MGMT_LABELS[loadProp] ?? String(prop.load_management ?? "N/A"),
        executed: LOAD_MGMT_LABELS[loadExec] ?? String(exec.load_management ?? "N/A"),
        physical: `Shed Crit ${fmtNum(currentStep.shed_critical_mw, 3)} · Flex ${fmtNum(
          currentStep.shed_flexible_mw,
          3
        )} MW`,
        altered: loadProp !== loadExec,
      },
      {
        agent: "Grid Resilience",
        proposed:
          GRID_RESILIENCE_LABELS[gridProp] ?? String(prop.grid_resilience ?? "N/A"),
        executed:
          GRID_RESILIENCE_LABELS[gridExec] ?? String(exec.grid_resilience ?? "N/A"),
        physical: `Grid ${currentStep.grid_connected ? "Connected" : "Islanded"} (${fmtNum(
          currentStep.grid_import_mw,
          3
        )} MW)`,
        altered: gridProp !== gridExec,
      },
    ];
  }, [currentStep]);

  return (
    <div className="space-y-4">
      {/* Episode Switcher */}
      <div className="bg-gridcard border border-gridborder rounded-lg p-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2 flex-1 min-w-[260px]">
          <span className="text-xs font-semibold text-slate-300">Quick Select:</span>
          <select
            aria-label="Replay preset episode"
            value={
              DEFAULT_REPLAY_CHOICES.some((c) => c.id === activeId) ? activeId : ""
            }
            onChange={(e) => {
              if (e.target.value) onSelectEpisodeId(e.target.value);
            }}
            className="px-2.5 py-1.5 rounded-md bg-gridpanel border border-gridborder text-xs text-slate-100 max-w-md"
          >
            <option value="" disabled>
              Custom Episode Selected
            </option>
            {DEFAULT_REPLAY_CHOICES.map((c) => (
              <option key={c.id} value={c.id}>
                {c.label}
              </option>
            ))}
          </select>
        </div>

        <form onSubmit={loadCustomId} className="flex items-center gap-2 flex-1 min-w-[280px]">
          <input
            type="text"
            value={customInput}
            onChange={(e) => setCustomInput(e.target.value)}
            placeholder="Paste exact episode_id..."
            aria-label="Episode ID for replay"
            className="flex-1 px-2.5 py-1.5 rounded-md bg-gridpanel border border-gridborder text-xs font-mono text-slate-100"
          />
          <button
            type="submit"
            className="px-3 py-1.5 rounded-md bg-cyan-500/20 border border-cyan-400/50 text-cyan-200 text-xs font-medium hover:bg-cyan-500/30"
          >
            Load Trace
          </button>
        </form>
      </div>

      {error ? (
        <EmptyState title="Unable to Load Episode Trace" description={error} tone="error" />
      ) : loading || !trace || !currentStep ? (
        <EmptyState
          title="Loading 96-Step Canonical Episode Trace"
          description={`Fetching ${activeId} lazily from backend trace endpoint...`}
          tone="info"
        />
      ) : (
        <>
          {/* Metadata Header */}
          <div className="bg-gridcard border border-gridborder rounded-lg p-4 space-y-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <div className="font-mono text-xs text-cyan-300 break-all">
                  {trace.episode_id}
                </div>
                <div className="text-xs text-slate-300 mt-1 flex flex-wrap items-center gap-2">
                  <span className="font-semibold text-white">
                    {formatModelLabel(String(meta.controller_name ?? meta.algorithm ?? "Unknown"))}
                  </span>
                  <StatusBadge
                    label={String(meta.mode ?? "Unknown")}
                    tone={modeBadgeTone(String(meta.mode ?? ""))}
                  />
                  <span className="text-slate-400">
                    {TRACK_LABELS[String(meta.track ?? "")] ?? String(meta.track ?? "")}
                  </span>
                  <StatusBadge
                    label={`Crit Rel ${fmtNum(
                      Number(metrics.critical_reliability_pct ?? 0),
                      2
                    )}%`}
                    tone={reliabilityTone(Number(metrics.critical_reliability_pct ?? 0))}
                  />
                </div>
              </div>
              <div className="text-right font-mono text-[11px] text-slate-400 space-y-0.5">
                <div>
                  Scenario: {String(meta.scenario_id ?? "").toUpperCase()} · train=
                  {String(meta.training_seed ?? "N/A")} · eval={String(meta.evaluation_seed ?? "")} ·
                  Reward: reward_v3_social.yaml
                </div>
                <div>
                  Checkpoint:{" "}
                  {meta.checkpoint_path
                    ? `ep=${String(meta.selected_checkpoint_episode ?? "?")} (${String(
                        meta.checkpoint_path
                      )})`
                    : "Deterministic Baseline (no checkpoint)"}
                </div>
                <div>
                  Social Cost: ${fmtNum(Number(metrics.true_social_cost_usd ?? 0), 2)} · Op Cost: $
                  {fmtNum(Number(metrics.operating_cost_usd ?? 0), 2)} · Status:{" "}
                  {String(meta.completion_status ?? "completed")}
                </div>
                <div>
                  Start Timestamp:{" "}
                  {meta.start_timestamp
                    ? String(meta.start_timestamp)
                    : "Unavailable (not recorded in offline trace)"}
                </div>
              </div>
            </div>
          </div>

          {/* Playback Controls */}
          <div className="bg-gridcard border border-gridborder rounded-lg p-4 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap items-center gap-1.5">
                <button
                  type="button"
                  onClick={() => {
                    setPlaying(false);
                    setStepIdx(0);
                  }}
                  aria-label="Reset to step 0"
                  className="p-2 rounded bg-gridpanel border border-gridborder text-slate-300 hover:text-white"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setPlaying(false);
                    setStepIdx((s) => Math.max(0, s - 1));
                  }}
                  aria-label="Previous step"
                  className="p-2 rounded bg-gridpanel border border-gridborder text-slate-300 hover:text-white"
                >
                  <SkipBack className="w-3.5 h-3.5" />
                </button>
                <button
                  type="button"
                  onClick={() => {
                    if (stepIdx >= maxStep) setStepIdx(0);
                    setPlaying((p) => !p);
                  }}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded bg-cyan-500/20 border border-cyan-400/50 text-cyan-200 text-xs font-semibold hover:bg-cyan-500/30"
                >
                  {playing ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                  {playing ? "Pause" : "Play"}
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setPlaying(false);
                    setStepIdx((s) => Math.min(maxStep, s + 1));
                  }}
                  aria-label="Next step"
                  className="p-2 rounded bg-gridpanel border border-gridborder text-slate-300 hover:text-white"
                >
                  <SkipForward className="w-3.5 h-3.5" />
                </button>

                <div className="flex items-center gap-1 ml-2">
                  <FastForward className="w-3.5 h-3.5 text-slate-400" />
                  {([1, 2, 4] as const).map((sp) => (
                    <button
                      key={sp}
                      type="button"
                      onClick={() => setSpeedMultiplier(sp)}
                      className={`px-2 py-1 rounded text-[11px] font-mono border ${
                        speedMultiplier === sp
                          ? "bg-cyan-500/20 border-cyan-400/50 text-cyan-200"
                          : "bg-gridpanel border-gridborder text-slate-400"
                      }`}
                    >
                      {sp}x
                    </button>
                  ))}
                </div>
              </div>

              {/* Jump Milestones */}
              <div className="flex flex-wrap items-center gap-1.5">
                {MILESTONE_JUMPS.map((m) => (
                  <button
                    key={m.step}
                    type="button"
                    onClick={() => {
                      setPlaying(false);
                      setStepIdx(Math.min(maxStep, m.step));
                    }}
                    className={`px-2 py-1 rounded text-[11px] font-mono border ${
                      stepIdx === m.step
                        ? "bg-indigo-500/25 border-indigo-400/60 text-indigo-200"
                        : "bg-gridpanel border-gridborder text-slate-300 hover:text-white"
                    }`}
                  >
                    {m.label}
                  </button>
                ))}
                {firstOutageStep >= 0 && (
                  <button
                    type="button"
                    onClick={() => {
                      setPlaying(false);
                      setStepIdx(firstOutageStep);
                    }}
                    className="px-2 py-1 rounded text-[11px] font-mono border border-amber-500/40 bg-amber-950/30 text-amber-200"
                  >
                    Outage @ {firstOutageStep}
                  </button>
                )}
                {firstFallbackStep >= 0 && (
                  <button
                    type="button"
                    onClick={() => {
                      setPlaying(false);
                      setStepIdx(firstFallbackStep);
                    }}
                    className="px-2 py-1 rounded text-[11px] font-mono border border-cyan-500/40 bg-cyan-950/30 text-cyan-200"
                  >
                    Fallback @ {firstFallbackStep}
                  </button>
                )}
              </div>
            </div>

            <div className="space-y-1">
              <div className="flex items-center justify-between text-xs font-mono text-slate-300">
                <span>
                  Step {currentStep.step_idx} / {maxStep} ({currentStep.timestamp_label})
                </span>
                <span className="flex items-center gap-2">
                  <StatusBadge
                    label={currentStep.grid_available ? "Grid Available" : "GRID OUTAGE"}
                    tone={currentStep.grid_available ? "emerald" : "red"}
                  />
                  <StatusBadge
                    label={currentStep.converged ? "PF Converged" : "PF Failed"}
                    tone={currentStep.converged ? "emerald" : "red"}
                  />
                </span>
              </div>
              <input
                type="range"
                aria-label="Episode step slider"
                min={0}
                max={maxStep}
                value={stepIdx}
                onChange={(e) => {
                  setPlaying(false);
                  setStepIdx(Number(e.target.value));
                }}
                className="w-full accent-cyan-400 cursor-pointer"
              />
            </div>
          </div>

          {/* 4 Synchronized Step Inspector Panels */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* 1. Microgrid Physical State */}
            <div className="bg-gridcard border border-gridborder rounded-lg p-4 space-y-3">
              <div className="text-xs font-semibold uppercase tracking-wider text-slate-200">
                1. Microgrid Energy &amp; Reliability State (Step {currentStep.step_idx})
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
                <div className="bg-gridpanel p-2 rounded border border-gridborder/70">
                  <div className="text-[10px] text-slate-400">Solar Dispatched / Avail</div>
                  <div className="font-mono text-slate-100 mt-0.5">
                    {fmtNum(currentStep.solar_mw, 3)} / {fmtNum(currentStep.solar_available_mw, 3)} MW
                  </div>
                </div>
                <div className="bg-gridpanel p-2 rounded border border-gridborder/70">
                  <div className="text-[10px] text-slate-400">Wind Dispatched / Avail</div>
                  <div className="font-mono text-slate-100 mt-0.5">
                    {fmtNum(currentStep.wind_mw, 3)} / {fmtNum(currentStep.wind_available_mw, 3)} MW
                  </div>
                </div>
                <div className="bg-gridpanel p-2 rounded border border-gridborder/70">
                  <div className="text-[10px] text-slate-400">RE Curtailment</div>
                  <div className="font-mono text-amber-300 mt-0.5">
                    {fmtNum(currentStep.renewable_curtailment_mw, 3)} MW
                  </div>
                </div>
                <div className="bg-gridpanel p-2 rounded border border-gridborder/70">
                  <div className="text-[10px] text-slate-400">Battery SOC / Power</div>
                  <div className="font-mono text-cyan-300 mt-0.5">
                    {fmtNum(currentStep.battery_soc_pct, 1)}% · {fmtNum(currentStep.battery_mw, 3)} MW
                  </div>
                </div>
                <div className="bg-gridpanel p-2 rounded border border-gridborder/70">
                  <div className="text-[10px] text-slate-400">Diesel Gen / Step Fuel</div>
                  <div className="font-mono text-slate-100 mt-0.5">
                    {currentStep.generator_state} · {fmtNum(currentStep.generator_mw, 3)} MW (
                    {fmtNum(currentStep.step_fuel_l, 2)} L)
                  </div>
                </div>
                <div className="bg-gridpanel p-2 rounded border border-gridborder/70">
                  <div className="text-[10px] text-slate-400">Voltages (min / PCC / max)</div>
                  <div className="font-mono text-slate-100 mt-0.5">
                    {fmtNum(currentStep.voltages_pu.min_bus, 3)} /{" "}
                    {fmtNum(currentStep.voltages_pu.pcc, 3)} /{" "}
                    {fmtNum(currentStep.voltages_pu.max_bus, 3)} pu
                  </div>
                </div>
              </div>

              <div className="border-t border-gridborder/60 pt-2.5 grid grid-cols-3 gap-2 text-xs">
                <div>
                  <div className="text-[10px] text-slate-400">Critical Served / Nominal</div>
                  <div className="font-mono text-slate-100">
                    {fmtNum(currentStep.served_critical_mw, 3)} /{" "}
                    {fmtNum(currentStep.nominal_critical_mw, 3)} MW
                  </div>
                  <div className="text-[11px] font-mono text-red-300">
                    Shed: {fmtNum(currentStep.shed_critical_mw, 3)} MW
                  </div>
                </div>
                <div>
                  <div className="text-[10px] text-slate-400">Flexible Served / Nominal</div>
                  <div className="font-mono text-slate-100">
                    {fmtNum(currentStep.served_flexible_mw, 3)} /{" "}
                    {fmtNum(currentStep.nominal_flexible_mw, 3)} MW
                  </div>
                  <div className="text-[11px] font-mono text-amber-300">
                    Shed: {fmtNum(currentStep.shed_flexible_mw, 3)} MW
                  </div>
                </div>
                <div>
                  <div className="text-[10px] text-slate-400">Other Served / Nominal</div>
                  <div className="font-mono text-slate-100">
                    {fmtNum(currentStep.served_other_mw, 3)} /{" "}
                    {fmtNum(currentStep.nominal_other_mw, 3)} MW
                  </div>
                  <div className="text-[11px] font-mono text-slate-400">
                    Shed: {fmtNum(currentStep.shed_other_mw, 3)} MW
                  </div>
                </div>
              </div>

              {currentStep.scenario_events.length > 0 && (
                <div className="bg-amber-950/25 border border-amber-500/40 rounded px-2.5 py-1.5 text-[11px] text-amber-200 flex items-center gap-1.5">
                  <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                  <span>Scenario Events: {currentStep.scenario_events.join(", ")}</span>
                </div>
              )}
            </div>

            {/* 2. 5-Agent Proposed vs Executed Actions */}
            <div className="bg-gridcard border border-gridborder rounded-lg p-4 space-y-2.5">
              <div className="text-xs font-semibold uppercase tracking-wider text-slate-200">
                2. Five-Agent Decisions (Proposed vs Executed)
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-gridborder text-[10px] uppercase text-slate-400">
                      <th className="py-1.5 px-2">Agent</th>
                      <th className="py-1.5 px-2">Proposed</th>
                      <th className="py-1.5 px-2">Executed</th>
                      <th className="py-1.5 px-2">Physical Output</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gridborder/60">
                    {agentRows.map((row) => (
                      <tr key={row.agent}>
                        <td className="py-1.5 px-2 font-medium text-slate-200">
                          {row.agent}
                        </td>
                        <td className="py-1.5 px-2 font-mono text-[11px] text-slate-300">
                          {row.proposed}
                        </td>
                        <td className="py-1.5 px-2 font-mono text-[11px]">
                          <span className={row.altered ? "text-amber-300 font-semibold" : "text-cyan-300"}>
                            {row.executed}
                          </span>
                        </td>
                        <td className="py-1.5 px-2 font-mono text-[11px] text-slate-400">
                          {row.physical}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* 3. Safety Shield & Adequacy Fallback */}
            <div className="bg-gridcard border border-gridborder rounded-lg p-4 space-y-2.5">
              <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-200">
                <ShieldAlert className="w-4 h-4 text-amber-400" />
                3. Safety Shield &amp; Adequacy Fallback Interventions
              </div>
              <div className="flex flex-wrap gap-2">
                <StatusBadge
                  label={
                    currentStep.shield_events.any_intervention
                      ? "Shield Intervened"
                      : "Shield Idle"
                  }
                  tone={currentStep.shield_events.any_intervention ? "amber" : "slate"}
                />
                <StatusBadge
                  label={
                    currentStep.shield_events.any_boundary
                      ? "Boundary Clipping"
                      : "No Boundary Clip"
                  }
                  tone={currentStep.shield_events.any_boundary ? "amber" : "slate"}
                />
                <StatusBadge
                  label={
                    currentStep.shield_events.any_emergency
                      ? "Emergency Override"
                      : "No Emergency Override"
                  }
                  tone={currentStep.shield_events.any_emergency ? "red" : "slate"}
                />
                <StatusBadge
                  label={
                    currentStep.fallback_events.adequacy_intervention
                      ? "Fallback Intervened (Track C)"
                      : currentStep.fallback_events.adequacy_fallback_enabled
                      ? "Fallback Enabled (Idle)"
                      : "Fallback Disabled"
                  }
                  tone={
                    currentStep.fallback_events.adequacy_intervention
                      ? "cyan"
                      : "slate"
                  }
                />
              </div>

              <div className="text-xs text-slate-300 space-y-1.5 pt-1">
                <div>
                  <span className="text-slate-400">Shield Reasons:</span>{" "}
                  {currentStep.shield_events.reasons.length > 0
                    ? currentStep.shield_events.reasons.join("; ")
                    : "None on this step"}
                </div>
                <div>
                  <span className="text-slate-400">Fallback Reasons:</span>{" "}
                  {currentStep.fallback_events.reasons.length > 0
                    ? `${currentStep.fallback_events.reasons.join("; ")} (avoidable shortfall: ${fmtNum(
                        currentStep.fallback_events.avoidable_shortfall_mw,
                        3
                      )} MW)`
                    : "None on this step"}
                </div>
              </div>
            </div>

            {/* 4. Financial & Reward Breakdown */}
            <div className="bg-gridcard border border-gridborder rounded-lg p-4 space-y-2.5">
              <div className="text-xs font-semibold uppercase tracking-wider text-slate-200">
                4. Step Financials &amp; Canonical Reward Decomposition
              </div>
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div className="bg-gridpanel p-2 rounded border border-gridborder/70">
                  <div className="text-[10px] text-slate-400">Step Operating Cost</div>
                  <div className="font-mono text-slate-100 mt-0.5">
                    ${fmtNum(currentStep.step_operating_cost_usd, 2)}
                  </div>
                </div>
                <div className="bg-gridpanel p-2 rounded border border-gridborder/70">
                  <div className="text-[10px] text-slate-400">Step Social Cost</div>
                  <div className="font-mono text-amber-300 mt-0.5">
                    ${fmtNum(currentStep.step_social_cost_usd, 2)}
                  </div>
                </div>
                <div className="bg-gridpanel p-2 rounded border border-gridborder/70">
                  <div className="text-[10px] text-slate-400">Step Reward</div>
                  <div className="font-mono text-cyan-300 mt-0.5">
                    {fmtNum(currentStep.reward, 4)}
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5 text-[11px] font-mono pt-1">
                {Object.entries(currentStep.reward_breakdown ?? {}).map(([k, v]) => (
                  <div
                    key={k}
                    className="flex items-center justify-between gap-2 min-w-0 px-2.5 py-1 rounded bg-gridpanel/60 border border-gridborder/50"
                  >
                    <span className="text-slate-400 truncate" title={k}>
                      {k}
                    </span>
                    <span className="text-slate-200 shrink-0">{Number(v).toFixed(3)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Interactive 96-Step Timeline Cursor Chart */}
          <div className="bg-gridcard border border-gridborder rounded-lg p-4">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
              <div className="text-xs font-semibold uppercase tracking-wider text-slate-200">
                96-Step Replay Timeline Cursor (Click Any Point to Jump)
              </div>
              <span className="text-[11px] font-mono text-cyan-300">
                Active Step: {stepIdx} ({currentStep.timestamp_label})
              </span>
            </div>
            <div className="h-56 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={timelineData}
                  onClick={(state) => {
                    if (state && typeof state.activeTooltipIndex === "number") {
                      setPlaying(false);
                      setStepIdx(state.activeTooltipIndex);
                    }
                  }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="step_idx" tick={{ fill: "#94a3b8", fontSize: 10 }} />
                  <YAxis yAxisId="left" tick={{ fill: "#94a3b8", fontSize: 10 }} />
                  <YAxis
                    yAxisId="right"
                    orientation="right"
                    domain={[0, 100]}
                    tick={{ fill: "#22d3ee", fontSize: 10 }}
                    tickFormatter={(v) => `${v}%`}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#0f172a",
                      borderColor: "#334155",
                      fontSize: "11px",
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: "10px" }} />
                  <ReferenceLine
                    x={stepIdx}
                    yAxisId="left"
                    stroke="#38bdf8"
                    strokeWidth={2}
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="soc"
                    name="Battery SOC (%)"
                    stroke="#22d3ee"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="re_mw"
                    name="Renewable MW"
                    stroke="#10b981"
                    strokeWidth={1.6}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="gen_mw"
                    name="Generator MW"
                    stroke="#f59e0b"
                    strokeWidth={1.6}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="shed_crit"
                    name="Critical Shed MW"
                    stroke="#ef4444"
                    strokeWidth={1.8}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
