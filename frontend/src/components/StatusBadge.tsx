import React from "react";

interface StatusBadgeProps {
  label: string;
  tone?: "cyan" | "emerald" | "amber" | "red" | "slate" | "indigo";
}

const TONE_MAP: Record<string, string> = {
  cyan: "bg-cyan-500/15 text-cyan-300 border-cyan-500/40",
  emerald: "bg-emerald-500/15 text-emerald-300 border-emerald-500/40",
  amber: "bg-amber-500/15 text-amber-300 border-amber-500/40",
  red: "bg-red-500/15 text-red-300 border-red-500/40",
  slate: "bg-slate-700/40 text-slate-300 border-slate-600/60",
  indigo: "bg-indigo-500/15 text-indigo-300 border-indigo-500/40",
};

export const StatusBadge: React.FC<StatusBadgeProps> = ({ label, tone = "slate" }) => {
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded border text-[11px] font-medium tracking-wide whitespace-nowrap ${
        TONE_MAP[tone] ?? TONE_MAP.slate
      }`}
    >
      {label}
    </span>
  );
};

export function modeBadgeTone(mode: string | null | undefined): "amber" | "cyan" | "emerald" | "indigo" | "slate" {
  if (!mode) return "slate";
  const m = mode.toLowerCase();
  if (m.includes("intrinsic")) return "amber";
  if (m.includes("shielded")) return "indigo";
  if (m.includes("hybrid") || m.includes("fallback")) return "cyan";
  if (m.includes("heuristic")) return "emerald";
  return "slate";
}
