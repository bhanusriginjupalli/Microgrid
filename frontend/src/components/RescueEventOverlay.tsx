import React, { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  BatteryCharging,
  Bell,
  BellOff,
  CheckCircle2,
  Flame,
  Pin,
  ShieldAlert,
  ShieldCheck,
  ThermometerSnowflake,
  X,
} from "lucide-react";
import { useSimulation } from "../context/SimulationContext";
import {
  deriveRescueEvents,
  type RescueEvent,
  type RescueSeverity,
} from "../utils/rescueEvents";

const SEVERITY_STYLES: Record<
  RescueSeverity,
  {
    border: string;
    bg: string;
    badgeBg: string;
    iconColor: string;
  }
> = {
  warning: {
    border: "border-amber-500/60 border-l-amber-400",
    bg: "bg-[#1a160b]/95",
    badgeBg: "bg-amber-500/20 text-amber-200 border-amber-500/40",
    iconColor: "text-amber-400",
  },
  critical: {
    border: "border-red-500/60 border-l-red-400",
    bg: "bg-[#1e0d12]/95",
    badgeBg: "bg-red-500/20 text-red-200 border-red-500/40",
    iconColor: "text-red-400",
  },
  action: {
    border: "border-cyan-500/60 border-l-cyan-400",
    bg: "bg-[#091828]/95",
    badgeBg: "bg-cyan-500/20 text-cyan-200 border-cyan-500/40",
    iconColor: "text-cyan-400",
  },
  rescue: {
    border: "border-emerald-500/60 border-l-emerald-400",
    bg: "bg-[#091c18]/95",
    badgeBg: "bg-emerald-500/20 text-emerald-200 border-emerald-500/40",
    iconColor: "text-emerald-400",
  },
  success: {
    border: "border-emerald-400/70 border-l-emerald-300",
    bg: "bg-[#081f17]/95",
    badgeBg: "bg-emerald-500/25 text-emerald-100 border-emerald-400/50",
    iconColor: "text-emerald-300",
  },
  danger: {
    border: "border-rose-500/70 border-l-rose-400",
    bg: "bg-[#240a10]/95",
    badgeBg: "bg-rose-500/25 text-rose-200 border-rose-500/50",
    iconColor: "text-rose-400",
  },
};

function renderEventIcon(kind: RescueEvent["kind"], className: string) {
  switch (kind) {
    case "upcoming_criticality":
      return <AlertTriangle className={className} />;
    case "grid_islanded":
    case "critical_shortfall":
      return <ShieldAlert className={className} />;
    case "cooling_throttled":
      return <ThermometerSnowflake className={className} />;
    case "battery_discharge":
      return <BatteryCharging className={className} />;
    case "generator_starting":
    case "generator_online":
      return <Flame className={className} />;
    case "safety_intervention":
      return <ShieldCheck className={className} />;
    case "grid_restored":
      return <CheckCircle2 className={className} />;
  }
}

export const RescueEventOverlay: React.FC = () => {
  const { simState, activeSnapshot } = useSimulation();
  const [dismissedIds, setDismissedIds] = useState<Set<string>>(new Set());
  const [pinnedIds, setPinnedIds] = useState<Set<string>>(new Set());
  const [muted, setMuted] = useState(false);

  const history = simState?.history ?? [];
  const currentStep = activeSnapshot?.step ?? -1;

  // Reset dismissals whenever a new simulation starts or resets
  useEffect(() => {
    if (history.length <= 1) {
      setDismissedIds(new Set());
      setPinnedIds(new Set());
    }
  }, [history.length, simState?.scenario_id, simState?.controller_id]);

  const allEvents = useMemo(
    () => deriveRescueEvents(history, currentStep >= 0 ? currentStep : undefined),
    [history, currentStep]
  );

  // Show recent events around the active step (or at completion keep the latest rescue milestones visible)
  const visibleEvents = useMemo(() => {
    if (muted || allEvents.length === 0 || currentStep < 0) return [];
    const windowSteps = simState?.status === "completed" ? 96 : 12;
    return allEvents
      .filter((ev) => {
        if (dismissedIds.has(ev.id)) return false;
        if (pinnedIds.has(ev.id)) return true;
        return currentStep - ev.step <= windowSteps;
      })
      .slice(-2)
      .reverse();
  }, [allEvents, currentStep, dismissedIds, pinnedIds, muted, simState?.status]);

  if (allEvents.length === 0) return null;

  return (
    <div className="fixed bottom-4 right-4 z-50 w-80 sm:w-[22rem] max-w-[calc(100vw-2rem)] flex flex-col items-end gap-2 pointer-events-none">
      {/* Compact Popups Control Pill */}
      <div className="pointer-events-auto flex items-center gap-1.5">
        <button
          type="button"
          onClick={() => setMuted((m) => !m)}
          className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border border-gridborder bg-[#0b1526]/95 backdrop-blur text-[11px] font-medium text-slate-200 shadow-lg hover:border-cyan-400/60"
          title={muted ? "Show Live AI Rescue Popups" : "Minimize Live AI Rescue Popups"}
        >
          {muted ? (
            <BellOff className="w-3 h-3 text-slate-400" />
          ) : (
            <Bell className="w-3 h-3 text-cyan-400 animate-pulse" />
          )}
          <span>
            {muted
              ? `Rescue Alerts (${allEvents.length})`
              : `Live AI Rescue Popups (${allEvents.length})`}
          </span>
        </button>
      </div>

      {/* Stacked Live Toast Popups */}
      {visibleEvents.map((ev) => {
        const st = SEVERITY_STYLES[ev.severity];
        const isPinned = pinnedIds.has(ev.id);
        return (
          <div
            key={ev.id}
            className={`pointer-events-auto w-full rounded-lg border border-l-4 ${st.border} ${st.bg} backdrop-blur-md shadow-2xl p-3 space-y-1.5 animate-rescue-in`}
          >
            <div className="flex items-start justify-between gap-2">
              <div className="flex items-center gap-2 min-w-0">
                {renderEventIcon(ev.kind, `w-4 h-4 shrink-0 ${st.iconColor}`)}
                <span
                  className={`px-1.5 py-0.5 rounded border text-[10px] font-bold tracking-wider truncate ${st.badgeBg}`}
                >
                  {ev.badge}
                </span>
              </div>
              <div className="flex items-center gap-1 shrink-0">
                <span className="text-[10px] font-mono text-slate-400">
                  {ev.timestamp} · S{ev.step + 1}
                </span>
                <button
                  type="button"
                  onClick={() =>
                    setPinnedIds((prev) => {
                      const next = new Set(prev);
                      if (next.has(ev.id)) next.delete(ev.id);
                      else next.add(ev.id);
                      return next;
                    })
                  }
                  className={`p-0.5 rounded hover:bg-slate-800 ${
                    isPinned ? "text-cyan-300" : "text-slate-500 hover:text-slate-300"
                  }`}
                  title={isPinned ? "Unpin alert" : "Pin alert on screen"}
                >
                  <Pin className="w-3 h-3" />
                </button>
                <button
                  type="button"
                  onClick={() =>
                    setDismissedIds((prev) => {
                      const next = new Set(prev);
                      next.add(ev.id);
                      return next;
                    })
                  }
                  className="p-0.5 rounded text-slate-500 hover:text-slate-200 hover:bg-slate-800"
                  title="Dismiss"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            <div className="text-xs font-semibold text-slate-100 leading-snug">
              {ev.title}
            </div>
            <p className="text-[11px] text-slate-300 leading-relaxed">{ev.description}</p>

            {ev.chips.length > 0 && (
              <div className="flex flex-wrap gap-1.5 pt-0.5">
                {ev.chips.map((ch, idx) => (
                  <span
                    key={idx}
                    className="px-1.5 py-0.5 rounded bg-black/30 border border-slate-700/80 text-[10px] font-mono text-slate-200"
                  >
                    {ch}
                  </span>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};
