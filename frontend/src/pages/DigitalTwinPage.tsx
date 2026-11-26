import React, { useEffect, useState } from "react";
import { Network } from "lucide-react";
import { api } from "../services/api";
import { useSimulation } from "../context/SimulationContext";
import { StatusBadge } from "../components/StatusBadge";
import type { NetworkTopologyResponse } from "../types/api";

const BUS_COORDS: Record<number, { x: number; y: number; short: string }> = {
  0: { x: 110, y: 185, short: "Bus 0: Utility PCC" },
  1: { x: 350, y: 185, short: "Bus 1: Main 11kV Bus" },
  2: { x: 250, y: 55, short: "Bus 2: Solar PV" },
  3: { x: 460, y: 55, short: "Bus 3: Wind Turbine" },
  4: { x: 250, y: 315, short: "Bus 4: Battery Storage" },
  5: { x: 460, y: 315, short: "Bus 5: Backup Diesel" },
  6: { x: 680, y: 70, short: "Bus 6: Critical Loads" },
  7: { x: 700, y: 185, short: "Bus 7: Flexible Loads" },
  8: { x: 680, y: 300, short: "Bus 8: Other Loads" },
};

export const DigitalTwinPage: React.FC = () => {
  const { activeSnapshot } = useSimulation();
  const [fallbackNet, setFallbackNet] = useState<NetworkTopologyResponse | null>(null);
  const [selectedBusIdx, setSelectedBusIdx] = useState<number>(1);

  useEffect(() => {
    api.getNetwork().then(setFallbackNet).catch(() => {});
  }, []);

  const net = activeSnapshot?.network ?? fallbackNet;
  const isLive = net?.data_source === "live_simulation";
  const selectedBus = net?.buses.find((b) => b.bus_idx === selectedBusIdx) ?? null;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-slate-100">9-Bus Pandapower Digital Twin</h2>
          <p className="text-xs text-slate-400">
            Interactive 11 kV campus microgrid single-line diagram backed directly by Pandapower AC power-flow telemetry.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <StatusBadge
            label={isLive ? "SIMULATED LIVE AC POWER FLOW" : "NOMINAL TOPOLOGY (SIM IDLE)"}
            tone={isLive ? "cyan" : "slate"}
          />
          {net && (
            <StatusBadge
              label={
                net.converged === null
                  ? "AC PF: NOT RUN"
                  : net.converged
                  ? "AC PF: CONVERGED"
                  : "AC PF: DIVERGED"
              }
              tone={net.converged === null ? "slate" : net.converged ? "emerald" : "red"}
            />
          )}
        </div>
      </div>

      {/* Explicit Disclaimer Distinguishing Simulated Flow from Physical Grid */}
      <div className="px-3.5 py-2 rounded border border-cyan-500/30 bg-cyan-950/20 text-xs text-cyan-200 flex items-center gap-2">
        <Network className="w-4 h-4 text-cyan-400 shrink-0" />
        <span>
          <strong>Simulated Power-Flow Representation:</strong> All bus voltages (p.u.), branch flows (MW), and breaker states come from the in-memory Pandapower digital twin — not physical SCADA telemetry.
        </span>
      </div>

      {/* Interactive 9-Bus SVG Schematic */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <div className="xl:col-span-2 bg-gridpanel border border-gridborder rounded-lg p-4">
          <div className="flex items-center justify-between mb-2 text-xs text-slate-400">
            <span>Click any bus node to inspect Pandapower bus &amp; branch metrics</span>
            <span className="font-mono">
              PCC Grid: {net?.grid_connected ? "CONNECTED" : "ISLANDED"} · Sync:{" "}
              {net?.sync_verified ? "VERIFIED" : "HOLD"}
            </span>
          </div>

          <div className="w-full overflow-x-auto">
            <svg viewBox="0 0 820 370" className="w-full min-w-[660px] h-auto select-none">
              <defs>
                <marker id="arrow-fwd" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                  <path d="M 0 1 L 10 5 L 0 9 z" fill="#22d3ee" />
                </marker>
                <marker id="arrow-rev" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                  <path d="M 0 1 L 10 5 L 0 9 z" fill="#34d399" />
                </marker>
              </defs>

              {/* Lines */}
              {(net?.lines ?? []).map((ln) => {
                const cFrom = BUS_COORDS[ln.from_bus];
                const cTo = BUS_COORDS[ln.to_bus];
                if (!cFrom || !cTo) return null;
                const pFrom = ln.p_from_mw ?? 0;
                const active = ln.in_service && Math.abs(pFrom) > 1e-4;
                const reverse = ln.power_direction === "reverse";
                const stroke = !ln.in_service
                  ? "#ef4444"
                  : active
                  ? reverse
                    ? "#34d399"
                    : "#22d3ee"
                  : "#475569";

                const mx = (cFrom.x + cTo.x) / 2;
                const my = (cFrom.y + cTo.y) / 2;
                const x1 = reverse ? cTo.x : cFrom.x;
                const y1 = reverse ? cTo.y : cFrom.y;
                const x2 = reverse ? cFrom.x : cTo.x;
                const y2 = reverse ? cFrom.y : cTo.y;

                return (
                  <g key={ln.line_idx}>
                    <line
                      x1={x1}
                      y1={y1}
                      x2={x2}
                      y2={y2}
                      stroke={stroke}
                      strokeWidth={active ? 2.8 : 1.8}
                      strokeDasharray={!ln.in_service ? "6 4" : undefined}
                      markerEnd={active ? (reverse ? "url(#arrow-rev)" : "url(#arrow-fwd)") : undefined}
                    />
                    <rect x={mx - 32} y={my - 10} width={64} height={18} rx={4} fill="#0b1526" stroke="#1f304d" />
                    <text x={mx} y={my + 3} textAnchor="middle" fill="#cbd5e1" fontSize="9.5" fontFamily="monospace">
                      {ln.in_service
                        ? ln.p_from_mw !== null
                          ? `${Math.abs(ln.p_from_mw).toFixed(3)} MW`
                          : "IDLE"
                        : "TRIPPED"}
                    </text>
                  </g>
                );
              })}

              {/* Bus Nodes */}
              {(net?.buses ?? []).map((b) => {
                const coord = BUS_COORDS[b.bus_idx];
                if (!coord) return null;
                const isSel = selectedBusIdx === b.bus_idx;
                const isIsoPcc = b.bus_idx === 0 && net?.grid_connected === false;
                const fill = isIsoPcc ? "#450a0a" : isSel ? "#0e2a47" : "#111d33";
                const border = isIsoPcc ? "#ef4444" : isSel ? "#22d3ee" : "#334155";
                const vmStr = b.vm_pu !== null ? `${b.vm_pu.toFixed(4)} pu` : "N/A";

                return (
                  <g
                    key={b.bus_idx}
                    onClick={() => setSelectedBusIdx(b.bus_idx)}
                    className="cursor-pointer"
                  >
                    <rect
                      x={coord.x - 74}
                      y={coord.y - 24}
                      width={148}
                      height={48}
                      rx={8}
                      fill={fill}
                      stroke={border}
                      strokeWidth={isSel ? 2.2 : 1.4}
                    />
                    <text x={coord.x} y={coord.y - 5} textAnchor="middle" fill="#f1f5f9" fontSize="11" fontWeight="600">
                      {coord.short}
                    </text>
                    <text x={coord.x} y={coord.y + 13} textAnchor="middle" fill="#94a3b8" fontSize="10" fontFamily="monospace">
                      {b.vn_kv.toFixed(1)}kV · {vmStr}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
        </div>

        {/* Selected Bus + Asset Telemetry */}
        <div className="bg-gridpanel border border-gridborder rounded-lg p-4 space-y-4">
          <h3 className="text-sm font-semibold text-slate-100">Selected Bus &amp; Asset Readout</h3>
          {selectedBus ? (
            <div className="bg-gridcard border border-gridborder rounded-lg p-3 space-y-2 text-xs font-mono">
              <div className="font-sans font-semibold text-cyan-300 text-sm">
                Bus {selectedBus.bus_idx}: {selectedBus.name}
              </div>
              <div className="grid grid-cols-2 gap-2 text-slate-300">
                <div>Nominal kV: {selectedBus.vn_kv.toFixed(1)} kV</div>
                <div>Voltage: {selectedBus.vm_pu !== null ? `${selectedBus.vm_pu.toFixed(4)} pu` : "Unavailable"}</div>
                <div>Angle: {selectedBus.va_degree !== null ? `${selectedBus.va_degree.toFixed(3)}°` : "Unavailable"}</div>
                <div>Net P: {selectedBus.p_mw !== null ? `${selectedBus.p_mw.toFixed(4)} MW` : "Unavailable"}</div>
              </div>
            </div>
          ) : (
            <div className="text-xs text-slate-400">Select a bus in the schematic.</div>
          )}

          <div className="space-y-2">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Asset &amp; Load Distribution
            </h4>
            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              {[
                ["Solar PV", net?.asset_summary?.solar_mw, "MW"],
                ["Wind Turbine", net?.asset_summary?.wind_mw, "MW"],
                ["Battery Power", net?.asset_summary?.battery_mw, "MW"],
                ["Battery SOC", net?.asset_summary?.battery_soc, "%"],
                ["Backup Gen", net?.asset_summary?.generator_mw, "MW"],
                ["Grid Import", net?.asset_summary?.grid_import_mw, "MW"],
                ["Critical Served", net?.asset_summary?.served_critical_mw, "MW"],
                ["Critical Shed", net?.asset_summary?.shed_critical_mw, "MW"],
              ].map(([lbl, val, unit]) => (
                <div key={String(lbl)} className="bg-gridcard/70 border border-gridborder rounded p-2">
                  <div className="font-sans text-[11px] text-slate-400">{lbl}</div>
                  <div className="text-slate-100 mt-0.5">
                    {val !== undefined && val !== null ? `${val} ${unit}` : "Unavailable"}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Pandapower Lines Table */}
      <div className="bg-gridpanel border border-gridborder rounded-lg p-4">
        <h3 className="text-sm font-semibold text-slate-100 mb-3">11 kV Distribution Branch Power Flow</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border-collapse">
            <thead>
              <tr className="border-b border-gridborder text-slate-400 uppercase text-[11px]">
                <th className="py-2 pr-3">Line</th>
                <th className="py-2 px-3">From → To</th>
                <th className="py-2 px-3">Status</th>
                <th className="py-2 px-3">Direction</th>
                <th className="py-2 px-3 text-right">P_from (MW)</th>
                <th className="py-2 px-3 text-right">Loading (%)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gridborder/60 font-mono">
              {(net?.lines ?? []).map((ln) => (
                <tr key={ln.line_idx} className="hover:bg-gridcard/50">
                  <td className="py-2 pr-3 font-sans font-medium text-slate-200">{ln.name}</td>
                  <td className="py-2 px-3 text-slate-300">
                    {ln.from_bus_name} → {ln.to_bus_name}
                  </td>
                  <td className="py-2 px-3">
                    <StatusBadge
                      label={ln.in_service ? "IN SERVICE" : "OUTAGE"}
                      tone={ln.in_service ? "emerald" : "red"}
                    />
                  </td>
                  <td className="py-2 px-3 text-slate-300">{ln.power_direction}</td>
                  <td className="py-2 px-3 text-right">
                    {ln.p_from_mw !== null ? ln.p_from_mw.toFixed(4) : "Unavailable"}
                  </td>
                  <td className="py-2 px-3 text-right">
                    {ln.loading_percent !== null ? `${ln.loading_percent.toFixed(2)}%` : "Unavailable"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
