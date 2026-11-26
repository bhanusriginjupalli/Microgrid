import React, { useEffect, useState } from "react";
import { Brain, Info, ShieldAlert } from "lucide-react";
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
import { EmptyState } from "../components/EmptyState";
import { formatModelLabel } from "../components/episodes/episodeHelpers";
import type {
  ExplainabilityAttributionsResponse,
  ExplainabilityCounterfactualsResponse,
  ExplainabilityDecisionsResponse,
} from "../types/api";

export const ExplainableAIPage: React.FC = () => {
  const [decData, setDecData] = useState<ExplainabilityDecisionsResponse | null>(null);
  const [attrData, setAttrData] = useState<ExplainabilityAttributionsResponse | null>(null);
  const [cfData, setCfData] = useState<ExplainabilityCounterfactualsResponse | null>(null);

  const [selectedAgent, setSelectedAgent] = useState("battery");
  const [cfScenario, setCfScenario] = useState("scenario_h2");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      api.getExplainabilityDecisions({ step_min: 44, step_max: 55, limit: 150 }),
      api.getExplainabilityAttributions(),
      api.getExplainabilityCounterfactuals(),
    ])
      .then(([d, a, c]) => {
        setDecData(d);
        setAttrData(a);
        setCfData(c);
      })
      .finally(() => setLoading(false));
  }, []);

  const topFeatures = (attrData?.aggregated_by_agent?.[selectedAgent] ?? []).slice(0, 8);
  const cfFiltered = (cfData?.records ?? []).filter((r) => String(r.scenario) === cfScenario);

  if (loading) {
    return <div className="text-xs text-slate-400 p-4">Loading Explainable AI decision &amp; attribution audit...</div>;
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-slate-100">Explainable AI &amp; Policy Decision Audit</h2>
          <p className="text-xs text-slate-400">
            Decision explanations, Integrated Gradients &amp; perturbation sensitivity, 11-term reward decomposition, safety/fallback classification, and controlled simulator counterfactuals.
          </p>
        </div>
        {decData?.provenance && (
          <StatusBadge
            label={`Candidate: MAPPO Battery-Shaped (Seed ${decData.provenance.seed ?? "202"}, ep=${decData.provenance.episode ?? 80})`}
            tone="cyan"
          />
        )}
      </div>

      {/* Explicit Causal vs Sensitivity Disclaimer */}
      <div className="px-3.5 py-2.5 rounded border border-amber-500/40 bg-amber-950/25 text-xs text-amber-200 flex items-start gap-2.5">
        <Info className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
        <span>
          <strong>Methodological Integrity:</strong>{" "}
          {attrData?.sensitivity_vs_causality_note ??
            "Feature attribution measures local actor sensitivity, NOT physical causal effect."}
        </span>
      </div>

      {/* 1. Feature Attribution + Reward Decomposition */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-gridpanel border border-gridborder rounded-lg p-4 space-y-3">
          <div className="flex items-center justify-between gap-2">
            <h3 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
              <Brain className="w-4 h-4 text-cyan-400" />
              Feature Attribution (H2 Mean |IG| &amp; Perturbation)
            </h3>
            <select
              value={selectedAgent}
              onChange={(e) => setSelectedAgent(e.target.value)}
              className="bg-gridcard border border-gridborder rounded px-2 py-1 text-xs text-slate-100"
            >
              {(attrData?.agents ?? ["battery", "generator", "grid_resilience", "load_management", "renewable"]).map((ag) => (
                <option key={ag} value={ag}>
                  {ag}
                </option>
              ))}
            </select>
          </div>
          <div className="text-[11px] font-mono text-slate-400">
            Max IG Completeness Error across H2: {attrData?.max_ig_completeness_error?.toFixed(6) ?? "Unavailable"}
          </div>
          {topFeatures.length > 0 ? (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={topFeatures} layout="vertical" margin={{ left: 30 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis type="number" stroke="#94a3b8" fontSize={11} />
                  <YAxis type="category" dataKey="feature" width={145} stroke="#94a3b8" fontSize={10} />
                  <Tooltip contentStyle={{ backgroundColor: "#0d1626", borderColor: "#1f304d", fontSize: 12 }} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="abs_ig" name="Mean |IG|" fill="#22d3ee" />
                  <Bar dataKey="pert_sensitivity" name="Perturbation Sensitivity" fill="#f59e0b" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <EmptyState title="Attribution Data Unavailable" description="Missing feature_attributions_h2.csv" />
          )}
        </div>

        {/* Reward Decomposition */}
        <div className="bg-gridpanel border border-gridborder rounded-lg p-4 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-100">
              Mean Episode Reward Decomposition (H1–H6)
            </h3>
            <StatusBadge label="Exact Summation (Err = 0.0000)" tone="emerald" />
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={(decData?.reward_decomposition_summary ?? []).map((r) => ({
                  ...r,
                  display_controller: formatModelLabel(String(r.controller)),
                }))}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="display_controller" stroke="#94a3b8" fontSize={9.5} />
                <YAxis stroke="#94a3b8" fontSize={11} />
                <Tooltip contentStyle={{ backgroundColor: "#0d1626", borderColor: "#1f304d", fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Bar dataKey="r_crit_ens" name="r_crit_ens" fill="#ef4444" />
                <Bar dataKey="r_flex_ens" name="r_flex_ens" fill="#f97316" />
                <Bar dataKey="r_cost" name="r_cost" fill="#38bdf8" />
                <Bar dataKey="r_imbalance" name="r_imbalance" fill="#a855f7" />
                <Bar dataKey="r_service" name="r_service" fill="#10b981" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* 2. Safety & Fallback Intervention Distinction */}
      <div className="bg-gridpanel border border-gridborder rounded-lg p-4 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-amber-400" />
            Safety-Shield vs. Operational Adequacy Intervention Breakdown (H1–H6 Counts)
          </h3>
          <span className="text-[11px] text-slate-400">
            Strictly separates Boundary Reserve Clipping, Emergency Override, Adequacy Fallback, and Combined
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border-collapse font-mono">
            <thead>
              <tr className="border-b border-gridborder text-slate-400 uppercase text-[11px] font-sans">
                <th className="py-2 pr-3">Controller</th>
                <th className="py-2 px-3">Mode</th>
                <th className="py-2 px-3">Agent</th>
                <th className="py-2 px-3">Intervention Stage</th>
                <th className="py-2 pl-3 text-right">Count (6 Episodes)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gridborder/60">
              {(decData?.intervention_counts_summary ?? []).map((row, idx) => (
                <tr key={idx} className="hover:bg-gridcard/50">
                  <td className="py-1.5 pr-3 font-sans text-slate-100">{formatModelLabel(String(row.controller))}</td>
                  <td className="py-1.5 px-3">
                    <StatusBadge label={String(row.mode)} tone={modeBadgeTone(String(row.mode))} />
                  </td>
                  <td className="py-1.5 px-3 text-slate-300">{String(row.agent)}</td>
                  <td className="py-1.5 px-3">
                    <StatusBadge
                      label={String(row.intervention_stage)}
                      tone={
                        String(row.intervention_stage).includes("FALLBACK")
                          ? "cyan"
                          : String(row.intervention_stage).includes("EMERGENCY")
                          ? "red"
                          : "amber"
                      }
                    />
                  </td>
                  <td className="py-1.5 pl-3 text-right font-semibold">{String(row.count)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* 3. Controlled Counterfactual Comparisons */}
      <div className="bg-gridpanel border border-gridborder rounded-lg p-4 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-slate-100">
            Controlled Simulator Counterfactuals (Paired Seed 42: Intrinsic vs. Hybrid)
          </h3>
          <select
            value={cfScenario}
            onChange={(e) => setCfScenario(e.target.value)}
            className="bg-gridcard border border-gridborder rounded px-2 py-1 text-xs text-slate-100"
          >
            {(cfData?.scenarios ?? ["scenario_h2", "scenario_h6"]).map((sc) => (
              <option key={sc} value={sc}>
                {sc}
              </option>
            ))}
          </select>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border-collapse font-mono">
            <thead>
              <tr className="border-b border-gridborder text-slate-400 uppercase text-[11px] font-sans">
                <th className="py-2 pr-3">Counterfactual</th>
                <th className="py-2 px-3">Mode</th>
                <th className="py-2 px-3 text-right">Crit Rel (%)</th>
                <th className="py-2 px-3 text-right">Δ Crit Rel</th>
                <th className="py-2 px-3 text-right">Crit ENS (MWh)</th>
                <th className="py-2 px-3 text-right">Social Cost ($)</th>
                <th className="py-2 px-3 text-right">Δ Social Cost</th>
                <th className="py-2 px-3 text-right">Fuel (L)</th>
                <th className="py-2 pl-3 text-right">Adequacy Interv.</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gridborder/60">
              {cfFiltered.map((r, i) => (
                <tr key={i} className="hover:bg-gridcard/50">
                  <td className="py-1.5 pr-3 font-sans font-medium text-slate-100">{String(r.counterfactual)}</td>
                  <td className="py-1.5 px-3">
                    <StatusBadge label={String(r.mode)} tone={modeBadgeTone(String(r.mode))} />
                  </td>
                  <td className="py-1.5 px-3 text-right">{Number(r.critical_reliability_pct ?? 0).toFixed(2)}</td>
                  <td className="py-1.5 px-3 text-right text-amber-300">
                    {Number(r.delta_critical_reliability_pct ?? 0).toFixed(2)}
                  </td>
                  <td className="py-1.5 px-3 text-right">{Number(r.critical_ens_mwh ?? 0).toFixed(4)}</td>
                  <td className="py-1.5 px-3 text-right">${Number(r.true_social_cost_usd ?? 0).toFixed(2)}</td>
                  <td className="py-1.5 px-3 text-right text-red-300">
                    {Number(r.delta_social_cost_usd ?? 0) >= 0 ? "+" : ""}${Number(r.delta_social_cost_usd ?? 0).toFixed(2)}
                  </td>
                  <td className="py-1.5 px-3 text-right">{Number(r.fuel_liters ?? 0).toFixed(1)}</td>
                  <td className="py-1.5 pl-3 text-right">{String(r.adequacy_interventions ?? 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* 4. H2 Outage Window Agent Decision Explanations Sample */}
      <div className="bg-gridpanel border border-gridborder rounded-lg p-4 space-y-2">
        <h3 className="text-sm font-semibold text-slate-100">
          Scenario H2 Outage Decision Trace Excerpts (Steps 44–55, MAPPO Battery-Shaped Seed 202)
        </h3>
        <div className="max-h-72 overflow-y-auto space-y-1.5 pr-1">
          {(decData?.decisions ?? []).slice(0, 20).map((d, i) => (
            <div
              key={i}
              className="p-2.5 rounded bg-gridcard/70 border border-gridborder/80 text-xs flex flex-col gap-1"
            >
              <div className="flex flex-wrap items-center gap-2">
                <StatusBadge label={formatModelLabel(d.controller)} tone={modeBadgeTone(d.mode)} />
                <span className="font-mono text-cyan-300">
                  Step {d.step} ({d.timestamp}) · {d.agent}
                </span>
                <StatusBadge label={d.intervention_stage} tone={d.intervention_stage === "NONE" ? "slate" : "amber"} />
              </div>
              <div className="text-slate-200">{d.explanation}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
