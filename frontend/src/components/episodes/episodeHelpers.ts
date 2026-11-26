export const TRACK_LABELS: Record<string, string> = {
  track_a_intrinsic: "Track A · Intrinsic / Heuristic",
  track_b_shielded: "Track B · Shielded",
  track_c_hybrid: "Track C · Hybrid (+Fallback)",
};

export const COMPARE_COLORS = ["#22d3ee", "#f59e0b", "#a855f7", "#10b981"];

export function reliabilityTone(pct: number): "emerald" | "amber" | "red" {
  if (pct >= 98) return "emerald";
  if (pct >= 90) return "amber";
  return "red";
}

export function fmtNum(
  val: number | null | undefined,
  digits = 2,
  prefix = ""
): string {
  if (val === null || val === undefined || !Number.isFinite(val)) {
    return "Unavailable";
  }
  return `${prefix}${val.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}`;
}

export function extractArrayVal(
  raw: unknown,
  index = 0
): number | null {
  if (typeof raw === "number" && Number.isFinite(raw)) return raw;
  if (Array.isArray(raw) && typeof raw[index] === "number" && Number.isFinite(raw[index])) {
    return raw[index];
  }
  return null;
}

export function formatActionSummary(raw: unknown, decimals = 3): string {
  if (raw === null || raw === undefined) return "N/A";
  if (typeof raw === "number") return Number.isFinite(raw) ? raw.toFixed(decimals) : "N/A";
  if (Array.isArray(raw)) {
    return `[${raw
      .map((v) => (typeof v === "number" && Number.isFinite(v) ? v.toFixed(decimals) : String(v)))
      .join(", ")}]`;
  }
  return String(raw);
}

export const LOAD_MGMT_LABELS: Record<number, string> = {
  0: "0 · Hold",
  1: "1 · Shed Other",
  2: "2 · Shed Flex+Other",
  3: "3 · Critical Cut Risk",
};

export const GRID_RESILIENCE_LABELS: Record<number, string> = {
  0: "0 · Hold",
  1: "1 · Connect Grid",
  2: "2 · Island Grid",
  3: "3 · Emergency Shed",
  4: "4 · Standby / Auto",
};

/**
 * Format internal/experiment controller identifiers into clear human-readable names
 * that always state the underlying algorithm family (MAPPO, IPPO, HAPPO, or Heuristic)
 * without internal development phase numbering.
 */
export function formatModelLabel(raw: string | null | undefined): string {
  if (!raw) return "Unknown Controller";
  const s = String(raw).trim();
  if (s === "MAPPO_MATCHED") return "MAPPO (Matched Multi-Seed)";
  if (s === "IPPO") return "IPPO (Independent Critics)";
  if (s === "HAPPO") return "HAPPO (Sequential Update)";

  let out = s;
  out = out.replace(/^4\.4\.4\s+/i, "MAPPO — ");
  out = out.replace(/^P4\.4\.3\s+/i, "MAPPO — ");
  out = out.replace(/\bBat-Only\b/g, "Battery-Shaped");
  out = out.replace(/\bGen-Only\b/g, "Generator-Shaped");
  out = out.replace(/\bCombined\b/g, "Combined-Shaped");
  out = out.replace(/\[\+Fallback\]/g, "[Hybrid +Fallback]");
  out = out.replace(/Phase\s*4\.5/gi, "Explainability Benchmark");
  out = out.replace(/Phase\s*4\.4\.4/gi, "Policy Refinement");
  if (
    (out.includes("Battery-Shaped") ||
      out.includes("Generator-Shaped") ||
      out.includes("Combined-Shaped")) &&
    !out.includes("MAPPO")
  ) {
    out = out.replace(/(Battery-Shaped|Generator-Shaped|Combined-Shaped)/g, "MAPPO — $1");
  }
  return out;
}
