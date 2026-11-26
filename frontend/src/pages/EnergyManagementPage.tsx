import React, { useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
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
import { StatusBadge } from "../components/StatusBadge";
import { formatModelLabel } from "../components/episodes/episodeHelpers";
import { EmptyState } from "../components/EmptyState";

export const EnergyManagementPage: React.FC = () => {
  const { simState, activeSnapshot } = useSimulation();
  const [h2Timeline, setH2Timeline] = useState<Record<string, unknown>[]>([]);
  const [fallbackCtrl, setFallbackCtrl] = useState("Bat-Only (Seed 202) [Intrinsic]");

  useEffect(() => {
    api
      .getExplainabilityDecisions({ limit: 5 })
      .then((res) => setH2Timeline(res.h2_timeline ?? []))
      .catch(() => {});
  }, []);

  const hasLiveHistory = (simState?.history.length ?? 0) > 0;

  const series = useMemo(() => {
    if (hasLiveHistory && simState) {
      return simState.history.map((s) => ({
        hour: s.hour,
        timestamp: s.timestamp,
        battery_soc: s.battery_soc,
        battery_mw: s.battery_mw,
        generator_mw: s.generator_mw,
        solar_mw: s.solar_mw,
        wind_mw: s.wind_mw,
        re_avail_mw: Number((s.solar_available_mw + s.wind_available_mw).toFixed(4)),
        re_curtail_mw: Number(
          Math.max(0, s.solar_available_mw + s.wind_available_mw - (s.solar_mw + s.wind_mw)).toFixed(4)
        ),
        grid_import_mw: s.grid_import_mw,
        grid_export_mw: s.grid_export_mw,
        served_critical_mw: s.served_critical_mw,
        served_flexible_mw: s.served_flexible_mw,
        shed_critical_mw: s.shed_critical_mw,
        shed_flexible_mw: s.shed_flexible_mw,
        shed_other_mw: s.shed_other_mw,
        operating_cost_cum: s.operating_cost_usd_cum,
        social_cost_cum: s.true_social_cost_usd_cum,
      }));
    }

    return h2Timeline
      .filter((r) => r.controller === fallbackCtrl)
      .map((r) => {
        const sAvail = Number(r.solar_available_mw ?? 0) + Number(r.wind_available_mw ?? 0);
        const sDeliv = Number(r.solar_mw ?? 0) + Number(r.wind_mw ?? 0);
        return {
          hour: Number(r.hour ?? 0),
          timestamp: String(r.hour ?? 0),
          battery_soc: Number(r.battery_soc ?? 0),
          battery_mw: Number(r.battery_mw ?? 0),
          generator_mw: Number(r.generator_mw ?? 0),
          solar_mw: Number(r.solar_mw ?? 0),
          wind_mw: Number(r.wind_mw ?? 0),
          re_avail_mw: Number(sAvail.toFixed(4)),
          re_curtail_mw: Number(Math.max(0, sAvail - sDeliv).toFixed(4)),
          grid_import_mw: Number(r.grid_import_mw ?? 0),
          grid_export_mw: Number(r.grid_export_mw ?? 0),
          served_critical_mw: Number(r.served_critical_mw ?? 0),
          served_flexible_mw: Number(r.served_flexible_mw ?? 0),
          shed_critical_mw: Number(r.shed_critical_mw ?? 0),
          shed_flexible_mw: Number(r.shed_flexible_mw ?? 0),
          shed_other_mw: Number(r.shed_other_mw ?? 0),
          operating_cost_cum: null,
          social_cost_cum: null,
        };
      });
  }, [hasLiveHistory, simState, h2Timeline, fallbackCtrl]);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-slate-100">Energy Management &amp; Dispatch</h2>
          <p className="text-xs text-slate-400">
            Battery SOC &amp; dispatch, generator output, renewable curtailment, priority load service/shedding, grid exchange, and cumulative costs.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <StatusBadge
            label={hasLiveHistory ? "SOURCE: LIVE SIMULATION" : "SOURCE: SAVED BENCHMARK ARCHIVE (H2)"}
            tone={hasLiveHistory ? "emerald" : "indigo"}
          />
          {!hasLiveHistory && (
            <select
              value={fallbackCtrl}
              onChange={(e) => setFallbackCtrl(e.target.value)}
              className="bg-gridcard border border-gridborder rounded px-2 py-1 text-xs text-slate-100"
            >
              <option value="Bat-Only (Seed 202) [Intrinsic]">{formatModelLabel("Bat-Only (Seed 202) [Intrinsic]")}</option>
              <option value="Bat-Only (Seed 202) [Hybrid +Fallback]">{formatModelLabel("Bat-Only (Seed 202) [Hybrid +Fallback]")}</option>
              <option value="Greedy Controller">Greedy Controller</option>
              <option value="Rule-Based EMS">Rule-Based EMS</option>
            </select>
          )}
        </div>
      </div>

      {series.length === 0 ? (
        <EmptyState
          title="No Dispatch Timeline Available"
          description="Start a simulation from the top bar or verify saved benchmark archive timelines are available."
        />
      ) : (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Battery SOC & Dispatch */}
            <div className="bg-gridpanel border border-gridborder rounded-lg p-4">
              <h3 className="text-sm font-semibold text-slate-100 mb-3">
                Battery SOC (%) &amp; Power Dispatch (+Charge / -Discharge MW)
              </h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={series}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="hour" stroke="#94a3b8" fontSize={11} />
                    <YAxis yAxisId="soc" domain={[0, 100]} stroke="#22d3ee" fontSize={11} />
                    <YAxis yAxisId="mw" orientation="right" domain={[-0.3, 0.3]} stroke="#f59e0b" fontSize={11} />
                    <Tooltip contentStyle={{ backgroundColor: "#0d1626", borderColor: "#1f304d", fontSize: 12 }} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Line yAxisId="soc" type="monotone" dataKey="battery_soc" name="Battery SOC (%)" stroke="#22d3ee" dot={false} strokeWidth={2} />
                    <Line yAxisId="mw" type="monotone" dataKey="battery_mw" name="Battery Power (MW)" stroke="#f59e0b" dot={false} strokeWidth={1.8} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Supply Mix & Grid Exchange */}
            <div className="bg-gridpanel border border-gridborder rounded-lg p-4">
              <h3 className="text-sm font-semibold text-slate-100 mb-3">
                Generation Dispatch, Grid Imports &amp; Renewable Curtailment (MW)
              </h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={series}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="hour" stroke="#94a3b8" fontSize={11} />
                    <YAxis stroke="#94a3b8" fontSize={11} />
                    <Tooltip contentStyle={{ backgroundColor: "#0d1626", borderColor: "#1f304d", fontSize: 12 }} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Area type="monotone" dataKey="solar_mw" stackId="1" name="Solar (MW)" stroke="#eab308" fill="#eab308" fillOpacity={0.35} />
                    <Area type="monotone" dataKey="wind_mw" stackId="1" name="Wind (MW)" stroke="#34d399" fill="#34d399" fillOpacity={0.35} />
                    <Area type="monotone" dataKey="generator_mw" stackId="1" name="Diesel Gen (MW)" stroke="#a855f7" fill="#a855f7" fillOpacity={0.35} />
                    <Area type="monotone" dataKey="grid_import_mw" stackId="1" name="Grid Import (MW)" stroke="#38bdf8" fill="#38bdf8" fillOpacity={0.25} />
                    <Line type="monotone" dataKey="re_curtail_mw" name="RE Curtailed (MW)" stroke="#f87171" dot={false} strokeWidth={1.5} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Load Served & Shed by Priority */}
            <div className="bg-gridpanel border border-gridborder rounded-lg p-4">
              <h3 className="text-sm font-semibold text-slate-100 mb-3">
                Load Service &amp; Curtailment by Priority Tier (MW)
              </h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={series}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="hour" stroke="#94a3b8" fontSize={11} />
                    <YAxis stroke="#94a3b8" fontSize={11} />
                    <Tooltip contentStyle={{ backgroundColor: "#0d1626", borderColor: "#1f304d", fontSize: 12 }} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Bar dataKey="served_critical_mw" stackId="a" name="Critical Served" fill="#10b981" />
                    <Bar dataKey="served_flexible_mw" stackId="a" name="Flexible Served" fill="#3b82f6" />
                    <Bar dataKey="shed_flexible_mw" stackId="a" name="Flexible Shed" fill="#f59e0b" />
                    <Bar dataKey="shed_critical_mw" stackId="a" name="Critical Shed" fill="#ef4444" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Operating Cost & True Social Cost */}
            <div className="bg-gridpanel border border-gridborder rounded-lg p-4">
              <h3 className="text-sm font-semibold text-slate-100 mb-3">
                Cumulative Operating Cost vs. True Social Cost ($)
              </h3>
              {hasLiveHistory ? (
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={series}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="hour" stroke="#94a3b8" fontSize={11} />
                      <YAxis stroke="#94a3b8" fontSize={11} />
                      <Tooltip contentStyle={{ backgroundColor: "#0d1626", borderColor: "#1f304d", fontSize: 12 }} />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      <Line type="monotone" dataKey="operating_cost_cum" name="Operating Cost ($)" stroke="#38bdf8" dot={false} strokeWidth={2} />
                      <Line type="monotone" dataKey="social_cost_cum" name="True Social Cost ($)" stroke="#f87171" dot={false} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <EmptyState
                  title="Live Cumulative Cost Curve Unavailable in Static Timeline"
                  description="Start any scenario simulation from the top bar to stream step-by-step compute_episode_economics() Operating Cost and True Social Cost curves."
                />
              )}
            </div>
          </div>

          {activeSnapshot && (
            <div className="bg-gridpanel border border-gridborder rounded-lg p-4 text-xs font-mono flex flex-wrap gap-6">
              <span>Step {activeSnapshot.step + 1} ({activeSnapshot.timestamp})</span>
              <span>Grid Export: {activeSnapshot.grid_export_mw.toFixed(4)} MW</span>
              <span>Step Fuel: {activeSnapshot.step_fuel_l.toFixed(3)} L (Total {activeSnapshot.total_fuel_l.toFixed(1)} L)</span>
              <span>Balance Error: {activeSnapshot.balance_error_mw.toFixed(6)} MW</span>
            </div>
          )}
        </>
      )}
    </div>
  );
};
