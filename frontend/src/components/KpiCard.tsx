import React from "react";

interface KpiCardProps {
  title: string;
  value: string | number | null | undefined;
  unit?: string;
  subtitle?: string;
  accent?: "cyan" | "green" | "amber" | "red" | "blue" | "slate";
  badge?: React.ReactNode;
}

const ACCENT_BORDER: Record<string, string> = {
  cyan: "border-l-cyan-400",
  green: "border-l-emerald-400",
  amber: "border-l-amber-400",
  red: "border-l-red-400",
  blue: "border-l-blue-400",
  slate: "border-l-slate-500",
};

export const KpiCard: React.FC<KpiCardProps> = ({
  title,
  value,
  unit,
  subtitle,
  accent = "blue",
  badge,
}) => {
  const display =
    value === null || value === undefined || value === ""
      ? "Unavailable"
      : typeof value === "number" && !Number.isFinite(value)
      ? "Unavailable"
      : value;

  const isUnavailable = display === "Unavailable";

  return (
    <div
      className={`bg-gridcard border border-gridborder border-l-4 ${
        ACCENT_BORDER[accent] ?? ACCENT_BORDER.blue
      } rounded-lg px-4 py-3 flex flex-col justify-between shadow-sm`}
    >
      <div className="flex items-start justify-between gap-2">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
          {title}
        </span>
        {badge}
      </div>
      <div className="mt-1.5 flex items-baseline gap-1.5">
        <span
          className={`text-xl font-semibold font-mono ${
            isUnavailable ? "text-slate-500 text-sm" : "text-slate-100"
          }`}
        >
          {display}
        </span>
        {!isUnavailable && unit && (
          <span className="text-xs font-mono text-slate-400">{unit}</span>
        )}
      </div>
      {subtitle && (
        <div className="mt-1 text-[11px] text-slate-400 leading-tight">{subtitle}</div>
      )}
    </div>
  );
};
