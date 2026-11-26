import React, { useEffect, useState } from "react";
import { Bot } from "lucide-react";
import { api } from "../services/api";
import { useSimulation } from "../context/SimulationContext";
import { StatusBadge } from "../components/StatusBadge";
import { EmptyState } from "../components/EmptyState";
import { formatModelLabel } from "../components/episodes/episodeHelpers";
import type { AgentDecisionRecord } from "../types/api";

export const MultiAgentControlPage: React.FC = () => {
  const { activeSnapshot } = useSimulation();
  const [archiveCtrl, setArchiveCtrl] = useState("Bat-Only (Seed 202) [Intrinsic]");
  const [archiveStep, setArchiveStep] = useState(45);
  const [archiveRecords, setArchiveRecords] = useState<AgentDecisionRecord[]>([]);
  const [controllers, setControllers] = useState<string[]>([]);

  const isLive = Boolean(activeSnapshot && activeSnapshot.agent_decisions.length > 0);

  useEffect(() => {
    if (isLive) return;
    api
      .getExplainabilityDecisions({
        controller: archiveCtrl,
        step_min: archiveStep,
        step_max: archiveStep,
        limit: 10,
      })
      .then((res) => {
        setControllers(res.controllers);
        setArchiveRecords(res.decisions);
      })
      .catch(() => {});
  }, [isLive, archiveCtrl, archiveStep]);

  const records = isLive ? activeSnapshot!.agent_decisions : archiveRecords;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-slate-100">Multi-Agent Control &amp; Action Pipeline</h2>
          <p className="text-xs text-slate-400">
            Inspect each of the 5 decentralized CTDE agents: observation vector summary, raw proposal, decoded physical command, SafetyShield / Fallback modifications, and reward alignment.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <StatusBadge
            label={isLive ? `LIVE STEP ${activeSnapshot!.step + 1} (${activeSnapshot!.timestamp})` : `ARCHIVE H2 STEP ${archiveStep}`}
            tone={isLive ? "emerald" : "indigo"}
          />
          {!isLive && (
            <>
              <select
                value={archiveCtrl}
                onChange={(e) => setArchiveCtrl(e.target.value)}
                className="bg-gridcard border border-gridborder rounded px-2 py-1 text-xs text-slate-100"
              >
                {(controllers.length > 0
                  ? controllers
                  : [
                      "Bat-Only (Seed 202) [Intrinsic]",
                      "Bat-Only (Seed 202) [Hybrid +Fallback]",
                      "Greedy Controller",
                      "Rule-Based EMS",
                    ]
                ).map((c) => (
                  <option key={c} value={c}>
                    {formatModelLabel(c)}
                  </option>
                ))}
              </select>
              <label className="flex items-center gap-1.5 text-xs text-slate-300 font-mono">
                Step
                <input
                  type="number"
                  min={0}
                  max={95}
                  value={archiveStep}
                  onChange={(e) => setArchiveStep(Math.max(0, Math.min(95, Number(e.target.value) || 0)))}
                  className="w-16 bg-gridcard border border-gridborder rounded px-2 py-1 text-xs"
                />
              </label>
            </>
          )}
        </div>
      </div>

      {records.length === 0 ? (
        <EmptyState
          title="No Agent Decision Records Found"
          description="Verify saved decision explanation records or start a live simulation."
        />
      ) : (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
          {records.map((rec) => {
            const obsEntries = Object.entries(rec.relevant_observations ?? {}).slice(0, 8);
            const stageTone =
              rec.intervention_stage === "NONE"
                ? "emerald"
                : rec.intervention_stage === "ADEQUACY_FALLBACK" || rec.intervention_stage === "BOTH_SHIELD_AND_FALLBACK"
                ? "cyan"
                : rec.intervention_stage === "EMERGENCY_OVERRIDE"
                ? "red"
                : "amber";

            return (
              <div
                key={rec.agent}
                className="bg-gridpanel border border-gridborder rounded-lg p-4 space-y-3"
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <Bot className="w-4 h-4 text-cyan-400" />
                    <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-100">
                      {rec.agent.replace("_", " ")} Agent
                    </h3>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <StatusBadge label={rec.intervention_stage} tone={stageTone} />
                    {rec.shield_modified && <StatusBadge label="Shield Modified" tone="amber" />}
                    {rec.fallback_modified && <StatusBadge label="Fallback Override" tone="cyan" />}
                  </div>
                </div>

                <p className="text-xs text-slate-200 bg-gridcard/80 border border-gridborder rounded p-2.5 leading-relaxed">
                  {rec.explanation}
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-[11px] font-mono">
                  <div className="bg-gridbg/70 border border-gridborder/80 rounded p-2">
                    <div className="text-slate-400 font-sans mb-1">Raw Output &amp; Decoded Action</div>
                    <div className="text-slate-300 break-all">raw: {JSON.stringify(rec.raw_policy_output)}</div>
                    <div className="text-slate-200 break-all mt-1">{JSON.stringify(rec.decoded_action)}</div>
                  </div>
                  <div className="bg-gridbg/70 border border-gridborder/80 rounded p-2">
                    <div className="text-slate-400 font-sans mb-1">Executed Action (Post-Shield/Fallback)</div>
                    <div className="text-cyan-300 break-all">{JSON.stringify(rec.executed_action)}</div>
                    <div className="text-slate-400 mt-1 break-all">
                      constraints: {JSON.stringify(rec.operational_constraints)}
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-1.5 text-[11px] font-mono">
                  {obsEntries.map(([k, v]) => (
                    <div key={k} className="px-2 py-1 rounded bg-gridcard/50 border border-gridborder/60 truncate" title={`${k}: ${v}`}>
                      <span className="text-slate-400">{k}: </span>
                      <span className="text-slate-200">{typeof v === "number" ? v.toFixed(3) : String(v)}</span>
                    </div>
                  ))}
                </div>

                <div className="text-[11px] font-mono text-slate-400 flex flex-wrap gap-3 border-t border-gridborder/70 pt-2">
                  <span>Step Reward: {rec.reward_total !== null ? rec.reward_total.toFixed(3) : "N/A"}</span>
                  {Object.entries(rec.agent_reward_components ?? {}).map(([rk, rv]) => (
                    <span key={rk}>
                      {rk}: <strong className="text-slate-200">{Number(rv).toFixed(3)}</strong>
                    </span>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
