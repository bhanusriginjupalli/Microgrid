import React from "react";
import { AlertTriangle, Info } from "lucide-react";

interface EmptyStateProps {
  title: string;
  description: string;
  tone?: "info" | "warning" | "error";
  action?: React.ReactNode;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  title,
  description,
  tone = "info",
  action,
}) => {
  const colors =
    tone === "error"
      ? "border-red-500/40 bg-red-950/20 text-red-200"
      : tone === "warning"
      ? "border-amber-500/40 bg-amber-950/20 text-amber-200"
      : "border-gridborder bg-gridpanel text-slate-300";

  return (
    <div className={`border rounded-lg p-6 flex flex-col items-start gap-2 ${colors}`}>
      <div className="flex items-center gap-2 font-semibold text-sm">
        {tone === "info" ? (
          <Info className="w-4 h-4 text-cyan-400" />
        ) : (
          <AlertTriangle className="w-4 h-4 text-amber-400" />
        )}
        <span>{title}</span>
      </div>
      <p className="text-xs text-slate-400 max-w-2xl leading-relaxed">{description}</p>
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
};
