import React, { useEffect, useState } from "react";
import { Sliders, Play, ShieldAlert } from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../services/api";
import { useSimulation } from "../context/SimulationContext";
import { StatusBadge, modeBadgeTone } from "../components/StatusBadge";
import { formatModelLabel } from "../components/episodes/episodeHelpers";
import type { CustomScenarioConditions, ScenarioDescriptor } from "../types/api";

export const ResilienceSimulationPage: React.FC = () => {
  const { audienceMode, scenarios, controllers, simState, startSim } = useSimulation();
  const [selectedScenario, setSelectedScenario] = useState<string>("scenario_h2");
  const [selectedCtrl, setSelectedCtrl] = useState<string>("bat_only_seed202_hybrid");
  const [seed, setSeed] = useState<number>(42);
  const [useCustom, setUseCustom] = useState<boolean>(false);
  const [solarCloud, setSolarCloud] = useState<number>(40);
  const [windOverride, setWindOverride] = useState<string>("8.0");
  const [critLoadMult, setCritLoadMult] = useState<number>(1.3);
  const [flexLoadMult, setFlexLoadMult] = useState<number>(1.0);
  const [outageEnabled, setOutageEnabled] = useState<boolean>(true);
  const [outageStart, setOutageStart] = useState<number>(11);
  const [outageEnd, setOutageEnd] = useState<number>(14);
  const [p45Traces, setP45Traces] = useState<Record<string, unknown>[]>([]);

  useEffect(() => {
    api
      .getAnalyticsComparison()
      .then((res) => setP45Traces(res.phase45_traces ?? []))
      .catch(() => {});
  }, []);

  const currentDesc: ScenarioDescriptor | undefined = scenarios.find(
    (s) => s.scenario_id === selectedScenario
  );

  const hist = simState?.history ?? [];

  const handleLaunch = () => {
    const custom_conditions: CustomScenarioConditions | undefined = useCustom
      ? {
          solar_cloud_severity: Math.min(0.95, Math.max(0, solarCloud / 100)),
          wind_speed_ms:
            windOverride.trim() !== "" && Number.isFinite(Number(windOverride))
              ? Number(windOverride)
              : null,
          critical_load_multiplier: critLoadMult,
          flexible_load_multiplier: flexLoadMult,
          grid_outage_enabled: outageEnabled,
          outage_start_hour: outageStart,
          outage_end_hour: Math.max(outageStart + 0.5, outageEnd),
        }
      : undefined;

    void startSim({
      controller_id:
        audienceMode === "client" &&
        !["bat_only_seed202_hybrid", "rule_based", "greedy"].includes(selectedCtrl)
          ? "bat_only_seed202_hybrid"
          : selectedCtrl,
      scenario_id: selectedScenario,
      seed,
      step_delay_ms: 40,
      forecast_mode: "realistic",
      custom_conditions,
    });
  };

  const applyPreset = (preset: "clear" | "edge_surge" | "blackout") => {
    setUseCustom(true);
    if (preset === "clear") {
      setSolarCloud(0);
      setWindOverride("9.5");
      setCritLoadMult(1.0);
      setFlexLoadMult(1.0);
      setOutageEnabled(false);
    } else if (preset === "edge_surge") {
      setSolarCloud(65);
      setWindOverride("4.5");
      setCritLoadMult(1.5);
      setFlexLoadMult(1.2);
      setOutageEnabled(false);
    } else {
      setSolarCloud(75);
      setWindOverride("2.0");
      setCritLoadMult(1.4);
      setFlexLoadMult(1.0);
      setOutageEnabled(true);
      setOutageStart(11);
      setOutageEnd(15);
    }
  };

  const ctrlChoices =
    audienceMode === "client"
      ? [
          { controller_id: "bat_only_seed202_hybrid", label: "Recommended AI Hybrid Controller" },
          { controller_id: "rule_based", label: "Priority Rule Energy Manager" },
          { controller_id: "greedy", label: "Fast Greedy Controller" },
        ]
      : controllers.map((c) => ({
          controller_id: c.controller_id,
          label: formatModelLabel(c.label),
        }));

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-slate-100">Resilience Simulation Runner (A–F &amp; Held-Out H1–H6)</h2>
          <p className="text-xs text-slate-400">
            Select benchmark Scenarios A–F or held-out disturbances H1–H6, configure controller &amp; repeatable random seed, and inspect system survival curves.
          </p>
        </div>
        <StatusBadge label="Exact Repeatable Seeding Preserved" tone="cyan" />
      </div>

      {/* Scenario Launcher Card */}
      <div className="bg-gridpanel border border-gridborder rounded-lg p-4 grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="space-y-3">
          <div className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Configure Experiment Run
          </div>

          <label className="block text-xs space-y-1">
            <span className="text-slate-300">Scenario (A–F / H1–H6)</span>
            <select
              value={selectedScenario}
              onChange={(e) => setSelectedScenario(e.target.value)}
              className="w-full bg-gridcard border border-gridborder rounded px-2.5 py-1.5 text-xs text-slate-100"
            >
              {scenarios.map((s) => (
                <option key={s.scenario_id} value={s.scenario_id}>
                  [{s.category}] {s.name}
                </option>
              ))}
            </select>
          </label>

          <label className="block text-xs space-y-1">
            <span className="text-slate-300">Controller</span>
            <select
              value={
                audienceMode === "client" &&
                !["bat_only_seed202_hybrid", "rule_based", "greedy"].includes(selectedCtrl)
                  ? "bat_only_seed202_hybrid"
                  : selectedCtrl
              }
              onChange={(e) => setSelectedCtrl(e.target.value)}
              className="w-full bg-gridcard border border-gridborder rounded px-2.5 py-1.5 text-xs text-slate-100"
            >
              {ctrlChoices.map((c) => (
                <option key={c.controller_id} value={c.controller_id}>
                  {c.label}
                </option>
              ))}
            </select>
          </label>

          <label className="block text-xs space-y-1">
            <span className="text-slate-300">Repeatable Random Seed</span>
            <input
              type="number"
              min={0}
              max={999999}
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value) || 0)}
              className="w-full bg-gridcard border border-gridborder rounded px-2.5 py-1.5 font-mono text-xs text-slate-100"
            />
          </label>

          <div className="pt-2 border-t border-gridborder/70 space-y-2">
            <label className="flex items-center justify-between text-xs text-cyan-200 font-medium cursor-pointer">
              <span className="inline-flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5" />
                Custom What-If Conditions
              </span>
              <input
                type="checkbox"
                checked={useCustom}
                onChange={(e) => setUseCustom(e.target.checked)}
                className="accent-cyan-400"
              />
            </label>

            {useCustom && (
              <div className="bg-gridcard/80 border border-gridborder rounded p-2.5 space-y-2 text-[11px]">
                <div className="flex flex-wrap gap-1">
                  <button
                    type="button"
                    onClick={() => applyPreset("clear")}
                    className="px-2 py-1 rounded bg-gridpanel border border-gridborder text-slate-200 hover:border-cyan-400"
                  >
                    Clear Day
                  </button>
                  <button
                    type="button"
                    onClick={() => applyPreset("edge_surge")}
                    className="px-2 py-1 rounded bg-gridpanel border border-gridborder text-slate-200 hover:border-cyan-400"
                  >
                    Edge Surge (1.5x)
                  </button>
                  <button
                    type="button"
                    onClick={() => applyPreset("blackout")}
                    className="px-2 py-1 rounded bg-gridpanel border border-gridborder text-slate-200 hover:border-cyan-400"
                  >
                    4h Blackout + Peak
                  </button>
                </div>

                <label className="block space-y-0.5">
                  <span className="text-slate-300 flex justify-between">
                    <span>Solar Cloud Attenuation</span>
                    <span className="font-mono text-cyan-300">{solarCloud}%</span>
                  </span>
                  <input
                    type="range"
                    min={0}
                    max={95}
                    value={solarCloud}
                    onChange={(e) => setSolarCloud(Number(e.target.value))}
                    className="w-full accent-cyan-400"
                  />
                </label>

                <div className="grid grid-cols-2 gap-2">
                  <label className="block space-y-0.5">
                    <span className="text-slate-300">Wind Speed (m/s)</span>
                    <input
                      type="number"
                      step="0.5"
                      min={0}
                      max={30}
                      value={windOverride}
                      onChange={(e) => setWindOverride(e.target.value)}
                      className="w-full bg-gridpanel border border-gridborder rounded px-2 py-1 font-mono text-slate-100"
                    />
                  </label>
                  <label className="block space-y-0.5">
                    <span className="text-slate-300">Crit/Edge Load (x)</span>
                    <input
                      type="number"
                      step="0.1"
                      min={0.5}
                      max={2.5}
                      value={critLoadMult}
                      onChange={(e) => setCritLoadMult(Number(e.target.value) || 1.0)}
                      className="w-full bg-gridpanel border border-gridborder rounded px-2 py-1 font-mono text-slate-100"
                    />
                  </label>
                </div>

                <div className="grid grid-cols-3 gap-2 items-end">
                  <label className="block space-y-0.5">
                    <span className="text-slate-300">Flex Load (x)</span>
                    <input
                      type="number"
                      step="0.1"
                      min={0.5}
                      max={2.5}
                      value={flexLoadMult}
                      onChange={(e) => setFlexLoadMult(Number(e.target.value) || 1.0)}
                      className="w-full bg-gridpanel border border-gridborder rounded px-2 py-1 font-mono text-slate-100"
                    />
                  </label>
                  <label className="block space-y-0.5">
                    <span className="text-slate-300">Outage Start h</span>
                    <input
                      type="number"
                      step="0.5"
                      min={0}
                      max={23}
                      disabled={!outageEnabled}
                      value={outageStart}
                      onChange={(e) => setOutageStart(Number(e.target.value) || 0)}
                      className="w-full bg-gridpanel border border-gridborder rounded px-2 py-1 font-mono text-slate-100 disabled:opacity-40"
                    />
                  </label>
                  <label className="block space-y-0.5">
                    <span className="text-slate-300">End h</span>
                    <input
                      type="number"
                      step="0.5"
                      min={0.5}
                      max={24}
                      disabled={!outageEnabled}
                      value={outageEnd}
                      onChange={(e) => setOutageEnd(Number(e.target.value) || 24)}
                      className="w-full bg-gridpanel border border-gridborder rounded px-2 py-1 font-mono text-slate-100 disabled:opacity-40"
                    />
                  </label>
                </div>

                <label className="flex items-center gap-2 text-slate-300 cursor-pointer pt-0.5">
                  <input
                    type="checkbox"
                    checked={outageEnabled}
                    onChange={(e) => setOutageEnabled(e.target.checked)}
                    className="accent-cyan-400"
                  />
                  Inject Utility Grid Outage
                </label>

                <p className="text-[10px] text-slate-400 leading-snug pt-0.5">
                  Controls electrical/weather inputs in the simulator (HVAC/room cooling temperature is not separately simulated).
                </p>
              </div>
            )}
          </div>

          <button
            type="button"
            onClick={handleLaunch}
            className="w-full inline-flex items-center justify-center gap-2 px-3 py-2 rounded bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold"
          >
            <Play className="w-3.5 h-3.5" />{" "}
            {useCustom ? "Run Custom What-If Scenario (96 Steps)" : "Run Full 24h Scenario (96 Steps)"}
          </button>
        </div>

        {/* Disturbance Timeline Inspection */}
        <div className="lg:col-span-2 bg-gridcard/60 border border-gridborder rounded-lg p-4 space-y-3">
          {currentDesc ? (
            <>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <h3 className="text-sm font-semibold text-slate-100">{currentDesc.name}</h3>
                  <p className="text-xs text-slate-400 mt-0.5">{currentDesc.description}</p>
                </div>
                <StatusBadge label={currentDesc.category} tone="indigo" />
              </div>

              <div className="flex flex-wrap gap-1.5">
                {currentDesc.has_outage && <StatusBadge label="Grid Outage" tone="red" />}
                {currentDesc.has_storm && <StatusBadge label="Severe Storm" tone="amber" />}
                {currentDesc.has_topology_shift && <StatusBadge label="Line Trip" tone="red" />}
                {currentDesc.has_sensor_degradation && <StatusBadge label="Sensor Noise/Dropout" tone="amber" />}
                {currentDesc.has_stale_forecast && <StatusBadge label="Frozen Forecast" tone="indigo" />}
                {currentDesc.events.length === 0 && <StatusBadge label="Clear Nominal Baseline" tone="emerald" />}
              </div>

              <div className="space-y-1.5 pt-1">
                <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                  Registered Disturbance Events Timeline
                </div>
                {currentDesc.events.length === 0 ? (
                  <div className="text-xs text-slate-400">No fault injections in this baseline scenario.</div>
                ) : (
                  currentDesc.events.map((ev, i) => (
                    <div
                      key={i}
                      className="flex flex-wrap items-center justify-between gap-2 px-3 py-1.5 rounded bg-gridbg/80 border border-gridborder text-xs font-mono"
                    >
                      <span className="text-amber-300 flex items-center gap-1.5">
                        <ShieldAlert className="w-3.5 h-3.5" /> {ev.type}
                      </span>
                      <span className="text-slate-300">
                        {ev.start_hour.toFixed(1)}h → {ev.end_hour.toFixed(1)}h
                      </span>
                      <span className="text-slate-400 text-[11px]">{JSON.stringify(ev.details)}</span>
                    </div>
                  ))
                )}
              </div>
            </>
          ) : (
            <div className="text-xs text-slate-400">Select a scenario to inspect disturbance timeline.</div>
          )}
        </div>
      </div>

      {/* Live Response Charts */}
      {hist.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="bg-gridpanel border border-gridborder rounded-lg p-4">
            <h3 className="text-sm font-semibold text-slate-100 mb-3">
              Live Resilience Response: Reliability (%) &amp; Critical ENS (MWh)
            </h3>
            <div className="h-60">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={hist}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="hour" stroke="#94a3b8" fontSize={11} />
                  <YAxis yAxisId="rel" domain={[0, 100]} stroke="#10b981" fontSize={11} />
                  <YAxis yAxisId="ens" orientation="right" stroke="#ef4444" fontSize={11} />
                  <Tooltip contentStyle={{ backgroundColor: "#0d1626", borderColor: "#1f304d", fontSize: 12 }} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line yAxisId="rel" type="monotone" dataKey="critical_reliability_pct_cum" name="Critical Reliability (%)" stroke="#10b981" dot={false} strokeWidth={2} />
                  <Line yAxisId="ens" type="monotone" dataKey="critical_ens_mwh_cum" name="Cum Critical ENS (MWh)" stroke="#ef4444" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-gridpanel border border-gridborder rounded-lg p-4">
            <h3 className="text-sm font-semibold text-slate-100 mb-3">
              Battery &amp; Generator Emergency Survival Trajectory
            </h3>
            <div className="h-60">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={hist}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="hour" stroke="#94a3b8" fontSize={11} />
                  <YAxis yAxisId="soc" domain={[0, 100]} stroke="#22d3ee" fontSize={11} />
                  <YAxis yAxisId="gen" orientation="right" stroke="#a855f7" fontSize={11} />
                  <Tooltip contentStyle={{ backgroundColor: "#0d1626", borderColor: "#1f304d", fontSize: 12 }} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line yAxisId="soc" type="monotone" dataKey="battery_soc" name="Battery SOC (%)" stroke="#22d3ee" dot={false} strokeWidth={2} />
                  <Line yAxisId="gen" type="monotone" dataKey="generator_mw" name="Generator (MW)" stroke="#a855f7" dot={false} strokeWidth={2} />
                  <Line yAxisId="gen" type="monotone" dataKey="battery_mw" name="Battery (MW)" stroke="#f59e0b" dot={false} strokeWidth={1.6} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* Held-Out Scenario Benchmark Matrix (Technical View) */}
      {audienceMode === "technical" && (
      <div className="bg-gridpanel border border-gridborder rounded-lg p-4">
        <h3 className="text-sm font-semibold text-slate-100 mb-3">
          Held-Out Scenarios (H1–H6) Benchmark Comparison (Eval Seed 42)
        </h3>
        <div className="overflow-x-auto max-h-80">
          <table className="w-full text-xs text-left border-collapse">
            <thead>
              <tr className="border-b border-gridborder text-slate-400 uppercase text-[11px]">
                <th className="py-2 pr-3">Scenario</th>
                <th className="py-2 px-3">Controller</th>
                <th className="py-2 px-3">Mode</th>
                <th className="py-2 px-3 text-right">Crit Rel (%)</th>
                <th className="py-2 px-3 text-right">Crit ENS (MWh)</th>
                <th className="py-2 px-3 text-right">Op Cost ($)</th>
                <th className="py-2 px-3 text-right">Social Cost ($)</th>
                <th className="py-2 pl-3 text-right">Adequacy Interv.</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gridborder/60 font-mono">
              {p45Traces.map((r, i) => (
                <tr key={i} className="hover:bg-gridcard/50">
                  <td className="py-1.5 pr-3 text-cyan-300">{String(r.scenario)}</td>
                  <td className="py-1.5 px-3 font-sans text-slate-100">{formatModelLabel(String(r.controller))}</td>
                  <td className="py-1.5 px-3">
                    <StatusBadge label={String(r.mode)} tone={modeBadgeTone(String(r.mode))} />
                  </td>
                  <td className="py-1.5 px-3 text-right">{Number(r.critical_reliability_pct ?? 0).toFixed(2)}</td>
                  <td className="py-1.5 px-3 text-right">{Number(r.critical_ens_mwh ?? 0).toFixed(4)}</td>
                  <td className="py-1.5 px-3 text-right">${Number(r.operating_cost_usd ?? 0).toFixed(2)}</td>
                  <td className="py-1.5 px-3 text-right">${Number(r.true_social_cost_usd ?? 0).toFixed(2)}</td>
                  <td className="py-1.5 pl-3 text-right">{String(r.adequacy_interventions ?? "N/A")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      )}
    </div>
  );
};
