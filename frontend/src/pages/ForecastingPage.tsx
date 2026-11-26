import React, { useMemo, useState } from "react";
import { CloudSun, Play } from "lucide-react";
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useSimulation } from "../context/SimulationContext";
import { StatusBadge } from "../components/StatusBadge";
import { EmptyState } from "../components/EmptyState";
import { KpiCard } from "../components/KpiCard";

type ForecastTarget = "solar" | "wind" | "demand" | "net_load";

export const ForecastingPage: React.FC = () => {
  const { activeSnapshot, startSim } = useSimulation();
  const [target, setTarget] = useState<ForecastTarget>("net_load");

  const fc = activeSnapshot?.forecast ?? null;

  const horizonData = useMemo(() => {
    if (!fc) return [];
    const actualNow =
      target === "solar"
        ? fc.actual_solar_mw
        : target === "wind"
        ? fc.actual_wind_mw
        : target === "demand"
        ? fc.actual_demand_mw
        : fc.actual_net_load_mw;

    return fc.horizons.map((h) => {
      const q10 = h[`${target}_q10`];
      const q50 = h[`${target}_q50`];
      const q90 = h[`${target}_q90`];
      const bandBase = q10 ?? 0;
      const bandSpan = q10 !== null && q90 !== null ? Math.max(0, q90 - q10) : 0;
      return {
        horizon: `+${h.horizon_minutes}m (${h.target_hour.toFixed(2)}h)`,
        q10,
        q50,
        q90,
        bandBase,
        bandSpan,
        actual_current: actualNow,
        outage_prob_pct: h.outage_prob !== null ? Number((h.outage_prob * 100).toFixed(1)) : null,
        storm_risk_pct: h.storm_risk !== null ? Number((h.storm_risk * 100).toFixed(1)) : null,
      };
    });
  }, [fc, target]);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-slate-100">Multi-Horizon Probabilistic Forecasting</h2>
          <p className="text-xs text-slate-400">
            Causal LightGBM quantile forecasts (q10 / q50 / q90) + outage/storm risk across 15m, 30m, 1h, 2h, and 4h horizons compared against current actual simulated telemetry.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <StatusBadge
            label={`EVALUATION MODE: ${fc ? fc.mode.toUpperCase() : "IDLE"}`}
            tone={fc?.mode === "oracle" ? "amber" : fc?.mode === "realistic" ? "cyan" : "slate"}
          />
          <select
            value={target}
            onChange={(e) => setTarget(e.target.value as ForecastTarget)}
            className="bg-gridcard border border-gridborder rounded px-2 py-1 text-xs text-slate-100"
          >
            <option value="net_load">Target: Net Load (MW)</option>
            <option value="solar">Target: Solar PV (MW)</option>
            <option value="wind">Target: Wind Turbine (MW)</option>
            <option value="demand">Target: Total Demand (MW)</option>
          </select>
        </div>
      </div>

      {!fc || fc.horizons.length === 0 ? (
        <EmptyState
          title="No Active Forecast Stream"
          description="Run any scenario with Forecast Mode set to Realistic (LightGBM quantile intervals) or Oracle to visualize live 15m–240m predictions alongside actual simulated measurements."
          action={
            <button
              type="button"
              onClick={() =>
                void startSim({
                  controller_id: "bat_only_seed202_intrinsic",
                  scenario_id: "scenario_d",
                  seed: 42,
                  step_delay_ms: 150,
                  forecast_mode: "realistic",
                })
              }
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-medium"
            >
              <Play className="w-3.5 h-3.5" /> Launch Scenario D (Storm) Realistic Forecast Stream
            </button>
          }
        />
      ) : (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <KpiCard
              title="Actual Solar (Now)"
              value={fc.actual_solar_mw.toFixed(3)}
              unit="MW"
              subtitle={`Current Hour ${fc.current_hour.toFixed(2)}h`}
              accent="amber"
            />
            <KpiCard
              title="Actual Wind (Now)"
              value={fc.actual_wind_mw.toFixed(3)}
              unit="MW"
              subtitle="Simulated turbine telemetry"
              accent="green"
            />
            <KpiCard
              title="Actual Demand (Now)"
              value={fc.actual_demand_mw.toFixed(3)}
              unit="MW"
              subtitle="Nominal total load"
              accent="blue"
            />
            <KpiCard
              title="Actual Net Load (Now)"
              value={fc.actual_net_load_mw.toFixed(3)}
              unit="MW"
              subtitle="Demand - (Solar + Wind)"
              accent="cyan"
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="bg-gridpanel border border-gridborder rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
                  <CloudSun className="w-4 h-4 text-cyan-400" />
                  Quantile Prediction Intervals (q10 / q50 / q90) vs. Actual Current
                </h3>
                <span className="text-[11px] font-mono text-slate-400">80% Interval</span>
              </div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={horizonData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="horizon" stroke="#94a3b8" fontSize={10.5} />
                    <YAxis stroke="#94a3b8" fontSize={11} />
                    <Tooltip contentStyle={{ backgroundColor: "#0d1626", borderColor: "#1f304d", fontSize: 12 }} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Area type="monotone" dataKey="q90" name="Predicted q90 (MW)" stroke="#38bdf8" strokeDasharray="4 4" fill="#38bdf8" fillOpacity={0.12} />
                    <Line type="monotone" dataKey="q50" name="Predicted q50 (MW)" stroke="#22d3ee" strokeWidth={2.4} dot />
                    <Line type="monotone" dataKey="q10" name="Predicted q10 (MW)" stroke="#818cf8" strokeDasharray="4 4" dot={false} />
                    <Line type="monotone" dataKey="actual_current" name="Actual Simulated (Now)" stroke="#34d399" strokeDasharray="2 2" dot={false} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-gridpanel border border-gridborder rounded-lg p-4">
              <h3 className="text-sm font-semibold text-slate-100 mb-3">
                Disturbance Risk Horizons: Outage Probability &amp; Storm Risk (%)
              </h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={horizonData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="horizon" stroke="#94a3b8" fontSize={10.5} />
                    <YAxis domain={[0, 100]} stroke="#94a3b8" fontSize={11} />
                    <Tooltip contentStyle={{ backgroundColor: "#0d1626", borderColor: "#1f304d", fontSize: 12 }} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Bar dataKey="outage_prob_pct" name="Outage Prob (%)" fill="#ef4444" />
                    <Bar dataKey="storm_risk_pct" name="Storm Risk (%)" fill="#f59e0b" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="bg-gridpanel border border-gridborder rounded-lg p-4 overflow-x-auto">
            <h3 className="text-sm font-semibold text-slate-100 mb-2">Forecast Horizon Table</h3>
            <table className="w-full text-xs text-left border-collapse font-mono">
              <thead>
                <tr className="border-b border-gridborder text-slate-400 uppercase text-[11px] font-sans">
                  <th className="py-2 pr-3">Horizon</th>
                  <th className="py-2 px-3 text-right">Solar [q10, q50, q90]</th>
                  <th className="py-2 px-3 text-right">Wind [q10, q50, q90]</th>
                  <th className="py-2 px-3 text-right">Demand [q10, q50, q90]</th>
                  <th className="py-2 px-3 text-right">Net Load q50</th>
                  <th className="py-2 px-3 text-right">Outage Prob</th>
                  <th className="py-2 pl-3 text-right">Storm Risk</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gridborder/60">
                {fc.horizons.map((h) => (
                  <tr key={h.horizon_step} className="hover:bg-gridcard/50">
                    <td className="py-2 pr-3 text-cyan-300">+{h.horizon_minutes} min ({h.target_hour.toFixed(2)}h)</td>
                    <td className="py-2 px-3 text-right">
                      [{h.solar_q10?.toFixed(3) ?? "N/A"}, {h.solar_q50?.toFixed(3) ?? "N/A"}, {h.solar_q90?.toFixed(3) ?? "N/A"}]
                    </td>
                    <td className="py-2 px-3 text-right">
                      [{h.wind_q10?.toFixed(3) ?? "N/A"}, {h.wind_q50?.toFixed(3) ?? "N/A"}, {h.wind_q90?.toFixed(3) ?? "N/A"}]
                    </td>
                    <td className="py-2 px-3 text-right">
                      [{h.demand_q10?.toFixed(3) ?? "N/A"}, {h.demand_q50?.toFixed(3) ?? "N/A"}, {h.demand_q90?.toFixed(3) ?? "N/A"}]
                    </td>
                    <td className="py-2 px-3 text-right">{h.net_load_q50?.toFixed(3) ?? "N/A"} MW</td>
                    <td className="py-2 px-3 text-right text-red-300">
                      {h.outage_prob !== null ? `${(h.outage_prob * 100).toFixed(1)}%` : "N/A"}
                    </td>
                    <td className="py-2 pl-3 text-right text-amber-300">
                      {h.storm_risk !== null ? `${(h.storm_risk * 100).toFixed(1)}%` : "N/A"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
};
