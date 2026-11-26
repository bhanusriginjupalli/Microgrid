import React, { useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BatteryCharging,
  CheckCircle2,
  Cpu,
  Flame,
  ShieldCheck,
  Zap,
} from "lucide-react";
import { api } from "../services/api";
import { useSimulation } from "../context/SimulationContext";
import { KpiCard } from "../components/KpiCard";
import { StatusBadge, modeBadgeTone } from "../components/StatusBadge";
import { formatModelLabel } from "../components/episodes/episodeHelpers";
import type { AnalyticsSummaryResponse } from "../types/api";

export const SystemOverviewPage: React.FC = () => {
  const { audienceMode, health, simState, activeSnapshot } = useSimulation();
  const [summary, setSummary] = useState<AnalyticsSummaryResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api
      .getAnalyticsSummary()
      .then(setSummary)
      .catch((e) => setErr(e instanceof Error ? e.message : "Failed to load summary"));
  }, []);

  const snap = activeSnapshot;
  const prov = health?.provenance ?? summary?.primary_candidate ?? null;
  const alerts = snap?.alerts ?? [];
  const p45Means = summary?.phase45_means_by_controller ?? [];

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-slate-100">System Overview &amp; Operational Status</h2>
          <p className="text-xs text-slate-400">
            Live microgrid simulation KPIs, grid &amp; asset status indicators, alerts, and verified model checkpoint provenance.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <StatusBadge
            label={snap ? (snap.grid_connected ? "GRID CONNECTED" : "ISLANDED") : "SIMULATION IDLE"}
            tone={snap ? (snap.grid_connected ? "emerald" : "red") : "slate"}
          />
          {simState?.scenario_id && <StatusBadge label={simState.scenario_id} tone="indigo" />}
          {simState?.controller_mode && (
            <StatusBadge label={simState.controller_mode} tone={modeBadgeTone(simState.controller_mode)} />
          )}
        </div>
      </div>

      {/* Live Operational KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-6 gap-3">
        <KpiCard
          title="Total Generation"
          value={snap ? snap.total_generation_mw.toFixed(3) : null}
          unit="MW"
          subtitle={
            snap
              ? `PV ${snap.solar_mw.toFixed(3)} · Wind ${snap.wind_mw.toFixed(3)} · Gen ${snap.generator_mw.toFixed(3)}`
              : "Start simulation for live step telemetry"
          }
          accent="green"
        />
        <KpiCard
          title="Total Demand"
          value={snap ? snap.nominal_total_mw.toFixed(3) : null}
          unit="MW"
          subtitle={snap ? `Crit served ${snap.served_critical_mw.toFixed(3)} MW` : "Critical + Flexible + Other"}
          accent="blue"
        />
        <KpiCard
          title="Battery SOC"
          value={snap ? snap.battery_soc.toFixed(1) : null}
          unit="%"
          subtitle={
            snap
              ? `${snap.battery_mw >= 0 ? "Charging" : "Discharging"} ${Math.abs(snap.battery_mw).toFixed(3)} MW`
              : "20% hard min / 30% reserve / 95% max"
          }
          accent="cyan"
        />
        <KpiCard
          title="Grid Exchange"
          value={
            snap
              ? snap.grid_import_mw > 0
                ? `+${snap.grid_import_mw.toFixed(3)}`
                : `-${snap.grid_export_mw.toFixed(3)}`
              : null
          }
          unit="MW"
          subtitle={
            snap
              ? `Grid ${snap.grid_available ? "Available" : "Outage"} · V_pcc ${snap.pcc_voltage_pu.toFixed(4)} pu`
              : "+Import / -Export at PCC"
          }
          accent="amber"
        />
        <KpiCard
          title="Critical Reliability"
          value={snap ? snap.critical_reliability_pct_cum.toFixed(2) : null}
          unit="%"
          subtitle={snap ? `Crit ENS ${snap.critical_ens_mwh_cum.toFixed(4)} MWh` : "Cumulative episode metric"}
          accent={snap && snap.critical_reliability_pct_cum < 95 ? "amber" : "green"}
        />
        <KpiCard
          title="Cumulative Costs"
          value={snap ? `$${snap.true_social_cost_usd_cum.toFixed(0)}` : null}
          subtitle={snap ? `Op Cost $${snap.operating_cost_usd_cum.toFixed(0)}` : "True Social Cost (incl. VoLL)"}
          accent="red"
        />
      </div>

      {/* Live Asset & Recent Events Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Asset & Scenario Status Panel */}
        <div className="bg-gridpanel border border-gridborder rounded-lg p-4 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
              <Cpu className="w-4 h-4 text-cyan-400" />
              Active Simulation Telemetry
            </h3>
            <span className="text-[11px] font-mono text-slate-400">
              {snap ? `Step ${snap.step + 1}/${simState?.max_steps ?? 96} (${snap.timestamp})` : "Idle"}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="bg-gridcard border border-gridborder rounded p-2.5">
              <div className="text-slate-400 text-[11px]">Controller</div>
              <div className="font-medium text-slate-100 mt-0.5 truncate">
                {simState?.controller_label ? formatModelLabel(simState.controller_label) : "None Selected"}
              </div>
            </div>
            <div className="bg-gridcard border border-gridborder rounded p-2.5">
              <div className="text-slate-400 text-[11px]">Scenario &amp; Seed</div>
              <div className="font-medium font-mono text-slate-100 mt-0.5 truncate">
                {simState?.scenario_id ?? "—"} (seed {simState?.seed ?? 42})
              </div>
            </div>
            <div className="bg-gridcard border border-gridborder rounded p-2.5">
              <div className="text-slate-400 text-[11px] flex items-center gap-1">
                <Flame className="w-3 h-3 text-amber-400" /> Generator State
              </div>
              <div className="font-mono text-slate-100 mt-0.5">
                {snap ? `${snap.generator_state} (${snap.generator_mw.toFixed(3)} MW)` : "Unavailable"}
              </div>
            </div>
            <div className="bg-gridcard border border-gridborder rounded p-2.5">
              <div className="text-slate-400 text-[11px] flex items-center gap-1">
                <BatteryCharging className="w-3 h-3 text-cyan-400" /> AC Power Flow
              </div>
              <div className="font-mono text-slate-100 mt-0.5">
                {snap ? (snap.converged ? "CONVERGED" : "DIVERGED") : "Unavailable"}
              </div>
            </div>
          </div>

          {audienceMode === "technical" && prov && (
            <div className="border-t border-gridborder pt-3 text-[11px] space-y-1 text-slate-400">
              <div className="flex items-center justify-between text-slate-200 font-semibold">
                <span className="flex items-center gap-1.5">
                  <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                  Primary Checkpoint Provenance
                </span>
                <StatusBadge
                  label={prov.verified ? "VERIFIED" : "UNVERIFIED"}
                  tone={prov.verified ? "emerald" : "red"}
                />
              </div>
              <div className="font-mono">
                model={prov.experiment === "bat_only" ? "MAPPO Battery-Shaped (bat_only)" : (prov.experiment ?? "N/A")} · seed={prov.seed ?? "N/A"} · ep={prov.episode ?? "N/A"} · val_score=
                {prov.val_score ?? "N/A"}
              </div>
              <div className="truncate text-slate-500" title={prov.checkpoint_path ?? ""}>
                {prov.checkpoint_path ?? "Unavailable"}
              </div>
            </div>
          )}
        </div>

        {/* Alerts & Recent Simulation Events */}
        <div className="bg-gridpanel border border-gridborder rounded-lg p-4 space-y-3 lg:col-span-2">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-400" />
              System Alerts &amp; Recent Step Events
            </h3>
            <span className="text-[11px] font-mono text-slate-400">
              {simState?.history.length ?? 0} timesteps recorded
            </span>
          </div>

          {alerts.length > 0 ? (
            <div className="space-y-1.5">
              {alerts.map((al, idx) => (
                <div
                  key={idx}
                  className="px-3 py-1.5 rounded border border-amber-500/40 bg-amber-950/30 text-xs font-mono text-amber-200"
                >
                  {al}
                </div>
              ))}
            </div>
          ) : (
            <div className="px-3 py-2 rounded border border-gridborder bg-gridcard/60 text-xs text-slate-400 flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              {snap ? "No active physical fault or fallback alerts at current timestep." : "Start a simulation above to stream live step events."}
            </div>
          )}

          <div className="space-y-1 max-h-44 overflow-y-auto pr-1">
            {(simState?.history ?? [])
              .slice(-6)
              .reverse()
              .map((h) => (
                <div
                  key={h.step}
                  className="flex flex-wrap items-center justify-between gap-2 px-2.5 py-1.5 rounded bg-gridcard/70 border border-gridborder/70 text-[11px] font-mono"
                >
                  <span className="text-cyan-300">
                    Step {h.step + 1} ({h.timestamp})
                  </span>
                  <span className="text-slate-300">
                    Grid={h.grid_connected ? "ON" : "ISLAND"} · Bat={h.battery_mw >= 0 ? "+" : ""}
                    {h.battery_mw.toFixed(3)}MW ({h.battery_soc.toFixed(1)}%) · Gen={h.generator_mw.toFixed(3)}MW
                  </span>
                  <span className="flex items-center gap-1">
                    {h.shield_any_boundary && <StatusBadge label="SHIELD CLIP" tone="amber" />}
                    {h.adequacy_intervention && <StatusBadge label="FALLBACK" tone="cyan" />}
                    {!h.shield_any_boundary && !h.adequacy_intervention && (
                      <StatusBadge label="NOMINAL" tone="emerald" />
                    )}
                  </span>
                </div>
              ))}
          </div>
        </div>
      </div>

      {/* Held-Out Key Research Conclusions & Mean Comparison (Technical View Only) */}
      {audienceMode === "technical" && (
      <div className="bg-gridpanel border border-gridborder rounded-lg p-4 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400" />
            Held-Out Benchmark Evaluation Summary (Scenarios H1–H6, Eval Seed 42)
          </h3>
          <span className="text-[11px] text-slate-400">
            Intrinsic policy performance is strictly separated from Hybrid (+Fallback)
          </span>
        </div>

        {err && <div className="text-xs text-red-400">{err}</div>}

        {summary && (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
              {summary.key_conclusions.map((note, idx) => (
                <div
                  key={idx}
                  className="p-2.5 rounded bg-gridcard/70 border border-gridborder text-xs text-slate-300 flex items-start gap-2"
                >
                  <Zap className="w-3.5 h-3.5 text-cyan-400 shrink-0 mt-0.5" />
                  <span>{formatModelLabel(note)}</span>
                </div>
              ))}
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left border-collapse">
                <thead>
                  <tr className="border-b border-gridborder text-slate-400 uppercase text-[11px]">
                    <th className="py-2 pr-3">Controller</th>
                    <th className="py-2 px-3">Mode</th>
                    <th className="py-2 px-3 text-right">Critical Rel. (%)</th>
                    <th className="py-2 px-3 text-right">Critical ENS (MWh)</th>
                    <th className="py-2 px-3 text-right">Operating Cost ($)</th>
                    <th className="py-2 px-3 text-right">True Social Cost ($)</th>
                    <th className="py-2 px-3 text-right">Boundary Clips / Ep</th>
                    <th className="py-2 pl-3 text-right">Adequacy Fallbacks / Ep</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gridborder/60 font-mono">
                  {p45Means.map((row, i) => (
                    <tr key={i} className="hover:bg-gridcard/50">
                      <td className="py-2 pr-3 font-sans font-medium text-slate-100">
                        {formatModelLabel(String(row.controller))}
                      </td>
                      <td className="py-2 px-3">
                        <StatusBadge
                          label={String(row.mode)}
                          tone={modeBadgeTone(String(row.mode))}
                        />
                      </td>
                      <td className="py-2 px-3 text-right">
                        {typeof row.critical_reliability_pct === "number"
                          ? row.critical_reliability_pct.toFixed(2)
                          : "N/A"}
                      </td>
                      <td className="py-2 px-3 text-right">
                        {typeof row.critical_ens_mwh === "number"
                          ? row.critical_ens_mwh.toFixed(4)
                          : "N/A"}
                      </td>
                      <td className="py-2 px-3 text-right">
                        {typeof row.operating_cost_usd === "number"
                          ? `$${row.operating_cost_usd.toFixed(2)}`
                          : "N/A"}
                      </td>
                      <td className="py-2 px-3 text-right font-semibold text-cyan-300">
                        {typeof row.true_social_cost_usd === "number"
                          ? `$${row.true_social_cost_usd.toFixed(2)}`
                          : "N/A"}
                      </td>
                      <td className="py-2 px-3 text-right">
                        {typeof row.boundary_clips === "number"
                          ? row.boundary_clips.toFixed(1)
                          : "N/A"}
                      </td>
                      <td className="py-2 pl-3 text-right">
                        {typeof row.adequacy_interventions === "number"
                          ? row.adequacy_interventions.toFixed(1)
                          : "N/A"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
      )}
    </div>
  );
};
