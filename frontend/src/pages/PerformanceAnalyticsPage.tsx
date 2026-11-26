import React, { useEffect, useMemo, useState } from "react";
import { BarChart3 } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../services/api";
import { StatusBadge, modeBadgeTone } from "../components/StatusBadge";
import { formatModelLabel } from "../components/episodes/episodeHelpers";
import type {
  AnalyticsComparisonResponse,
  ControllerComparisonRow,
  ExperimentSummary,
} from "../types/api";

function fmt(val: number | null | undefined, digits = 2, prefix = ""): string {
  if (val === null || val === undefined || !Number.isFinite(val)) return "Unavailable";
  return `${prefix}${val.toFixed(digits)}`;
}

export const PerformanceAnalyticsPage: React.FC = () => {
  const [comp, setComp] = useState<AnalyticsComparisonResponse | null>(null);
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [modeFilter, setModeFilter] = useState<string>("ALL");
  const [ablationFilter, setAblationFilter] = useState<string>("FOCUS");

  useEffect(() => {
    Promise.all([api.getAnalyticsComparison(), api.getExperiments()])
      .then(([c, e]) => {
        setComp(c);
        setExperiments(e.experiments);
      })
      .catch(() => {});
  }, []);

  const rows: ControllerComparisonRow[] = useMemo(() => {
    const raw = comp?.phase444_controllers ?? [];
    return raw.filter((r) => {
      if (modeFilter !== "ALL" && r.mode !== modeFilter) return false;
      if (ablationFilter === "FOCUS") {
        return (
          r.ablation === "Heuristic" ||
          r.ablation === "4.4.4 Bat-Only" ||
          r.controller.includes("4.4.4 Baseline (Seed 202)")
        );
      }
      if (ablationFilter !== "ALL" && r.ablation !== ablationFilter) return false;
      return true;
    });
  }, [comp, modeFilter, ablationFilter]);

  const displayRows = useMemo(
    () => rows.map((r) => ({ ...r, display_controller: formatModelLabel(r.controller) })),
    [rows]
  );

  const ablations = useMemo(() => {
    const s = new Set<string>();
    (comp?.phase444_controllers ?? []).forEach((r) => {
      if (r.ablation) s.add(r.ablation);
    });
    return Array.from(s);
  }, [comp]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-slate-100">Performance Analytics &amp; Benchmark Comparison</h2>
          <p className="text-xs text-slate-400">
            Multi-seed (30 episodes/controller) comparison of Greedy, Rule-Based EMS, Lookahead Heuristic EMS, MAPPO Battery-Shaped Intrinsic, Hybrid (+Fallback), and shaping ablations.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2 text-xs">
          <select
            value={ablationFilter}
            onChange={(e) => setAblationFilter(e.target.value)}
            className="bg-gridcard border border-gridborder rounded px-2 py-1 text-slate-100"
          >
            <option value="FOCUS">Focus: Heuristics + MAPPO (Battery-Shaped)</option>
            <option value="ALL">All Evaluated MAPPO Variants &amp; Baselines</option>
            {ablations.map((ab) => (
              <option key={ab} value={ab}>
                {formatModelLabel(ab)}
              </option>
            ))}
          </select>

          <select
            value={modeFilter}
            onChange={(e) => setModeFilter(e.target.value)}
            className="bg-gridcard border border-gridborder rounded px-2 py-1 text-slate-100"
          >
            <option value="ALL">Modes: All (Separated)</option>
            <option value="Intrinsic">Mode: Intrinsic Only</option>
            <option value="Hybrid (+Fallback)">Mode: Hybrid (+Fallback) Only</option>
            <option value="Heuristic">Mode: Heuristic Baselines</option>
          </select>
        </div>
      </div>

      {/* True Social Cost & Critical Reliability Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-gridpanel border border-gridborder rounded-lg p-4">
          <h3 className="text-sm font-semibold text-slate-100 mb-3 flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-cyan-400" />
            True Social Cost vs. Operating Cost ($ Mean, 30 Eps)
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={displayRows}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="display_controller" stroke="#94a3b8" fontSize={9.5} interval={0} angle={-15} textAnchor="end" height={68} />
                <YAxis stroke="#94a3b8" fontSize={11} />
                <Tooltip contentStyle={{ backgroundColor: "#0d1626", borderColor: "#1f304d", fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="true_social_cost_mean" name="True Social Cost ($)" fill="#f87171" />
                <Bar dataKey="operating_cost_mean" name="Operating Cost ($)" fill="#38bdf8" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-gridpanel border border-gridborder rounded-lg p-4">
          <h3 className="text-sm font-semibold text-slate-100 mb-3">
            Unshielded Proposal Feasibility (%) vs. Critical Reliability (%)
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={displayRows}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="display_controller" stroke="#94a3b8" fontSize={9.5} interval={0} angle={-15} textAnchor="end" height={68} />
                <YAxis domain={[60, 100]} stroke="#94a3b8" fontSize={11} />
                <Tooltip contentStyle={{ backgroundColor: "#0d1626", borderColor: "#1f304d", fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="critical_reliability_mean" name="Critical Reliability (%)" fill="#10b981" />
                <Bar dataKey="unshielded_feasible_pct" name="Unshielded Feasibility (%)" fill="#f59e0b" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Full Metrics Table */}
      <div className="bg-gridpanel border border-gridborder rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-slate-100">
            Consistent Accounting &amp; Autonomy Comparison Table
          </h3>
          <span className="text-[11px] text-slate-400">
            Fallback interventions attributed separately — never credited to Intrinsic policy
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border-collapse font-mono">
            <thead>
              <tr className="border-b border-gridborder text-slate-400 uppercase text-[10.5px] font-sans">
                <th className="py-2 pr-3">Controller</th>
                <th className="py-2 px-2">Mode</th>
                <th className="py-2 px-2 text-right">Social Cost</th>
                <th className="py-2 px-2 text-right">Op Cost</th>
                <th className="py-2 px-2 text-right">Crit ENS</th>
                <th className="py-2 px-2 text-right">Crit Rel%</th>
                <th className="py-2 px-2 text-right">Fuel (L)</th>
                <th className="py-2 px-2 text-right">Bat (MWh)</th>
                <th className="py-2 px-2 text-right">Feas%</th>
                <th className="py-2 px-2 text-right">Emerg</th>
                <th className="py-2 px-2 text-right">Fallback</th>
                <th className="py-2 pl-2 text-right">AC PF%</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gridborder/60">
              {displayRows.map((r, i) => (
                <tr key={i} className="hover:bg-gridcard/50">
                  <td className="py-1.5 pr-3 font-sans font-medium text-slate-100">{r.display_controller}</td>
                  <td className="py-1.5 px-2">
                    <StatusBadge label={r.mode} tone={modeBadgeTone(r.mode)} />
                  </td>
                  <td className="py-1.5 px-2 text-right text-cyan-300">{fmt(r.true_social_cost_mean, 2, "$")}</td>
                  <td className="py-1.5 px-2 text-right">{fmt(r.operating_cost_mean, 2, "$")}</td>
                  <td className="py-1.5 px-2 text-right">{fmt(r.critical_ens_mean, 4)}</td>
                  <td className="py-1.5 px-2 text-right">{fmt(r.critical_reliability_mean, 2, "")}</td>
                  <td className="py-1.5 px-2 text-right">{fmt(r.fuel_consumed_l_mean, 1)}</td>
                  <td className="py-1.5 px-2 text-right">{fmt(r.battery_throughput_mean, 3)}</td>
                  <td className="py-1.5 px-2 text-right">{fmt(r.unshielded_feasible_pct, 1)}</td>
                  <td className="py-1.5 px-2 text-right">{fmt(r.emergency_override_mean, 2)}</td>
                  <td className="py-1.5 px-2 text-right text-amber-300">{fmt(r.adequacy_intervention_mean, 2)}</td>
                  <td className="py-1.5 pl-2 text-right">{fmt(r.power_flow_converged_pct, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Retained Experiment Artifacts Catalog */}
      <div className="bg-gridpanel border border-gridborder rounded-lg p-4">
        <h3 className="text-sm font-semibold text-slate-100 mb-3">
          Indexed Research Experiment Datasets
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
          {experiments.map((ex) => (
            <div key={ex.experiment_id} className="bg-gridcard/70 border border-gridborder rounded-lg p-3 space-y-1.5">
              <div className="flex items-center justify-between gap-2">
                <StatusBadge label={ex.phase} tone="indigo" />
                <StatusBadge label={ex.available ? "AVAILABLE" : "MISSING"} tone={ex.available ? "emerald" : "red"} />
              </div>
              <div className="text-xs font-semibold text-slate-100">{ex.title}</div>
              <div className="text-[11px] text-slate-400">{ex.description}</div>
              <div className="text-[11px] font-mono text-slate-300 pt-1">
                Rows: {ex.row_count} · Controllers: {ex.controllers_count} · Mode Separated:{" "}
                {ex.separates_intrinsic_and_hybrid ? "Yes" : "Legacy"}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
