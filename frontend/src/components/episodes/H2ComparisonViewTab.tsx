import React, { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  GitCompare,
  Play,
  Sparkles,
  X,
} from "lucide-react";
import {
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
import { api } from "../../services/api";
import { EmptyState } from "../EmptyState";
import { StatusBadge, modeBadgeTone } from "../StatusBadge";
import {
  COMPARE_COLORS,
  TRACK_LABELS,
  extractArrayVal,
  fmtNum,
  formatModelLabel,
  reliabilityTone,
} from "./episodeHelpers";
import type {
  EpisodeCompareResponse,
  EpisodeCompareSummaryRow,
  EpisodeTraceResponse,
  H2ComparisonDataset,
  H2MatrixRepresentative,
  ScenarioEpisodesResponse,
} from "../../types/api";

interface H2ComparisonViewTabProps {
  externalCompareIds: string[];
  onClearExternalCompareIds: () => void;
  onInspectEpisode: (episodeId: string) => void;
}

type ChartCategoryTab = "energy" | "reliability" | "behavior" | "economic";

const ALGORITHMS_ORDER = ["IPPO", "MAPPO_MATCHED", "HAPPO", "Rule-Based EMS"];
const TRACKS_ORDER = ["track_a_intrinsic", "track_b_shielded", "track_c_hybrid"];

const METRIC_GLOSSARY: Array<{ label: string; help: string }> = [
  {
    label: "True Social Cost ($)",
    help: "Operating cost + unserved energy valuation (critical $15,000/MWh, flexible $1,500/MWh, other $500/MWh) + battery degradation.",
  },
  {
    label: "Critical ENS (MWh)",
    help: "Unserved critical healthcare/essential load integrated across all 96 15-minute steps.",
  },
  {
    label: "Critical Reliability (%)",
    help: "Percentage of nominal critical load successfully served across the 24-hour episode.",
  },
  {
    label: "Fallback Steps",
    help: "Steps where the Track C operational adequacy fallback intervened to prevent avoidable critical shortfalls. Do not attribute fallback improvements to intrinsic RL alone.",
  },
];

interface MultiTraceChartProps {
  title: string;
  subtitle?: string;
  data: Array<Record<string, number | string | null>>;
  series: Array<{ key: string; label: string; color: string }>;
  unit?: string;
  unavailableReason?: string;
}

const MultiTraceChart: React.FC<MultiTraceChartProps> = ({
  title,
  subtitle,
  data,
  series,
  unit = "",
  unavailableReason,
}) => {
  return (
    <div className="bg-gridcard border border-gridborder rounded-lg p-3.5 flex flex-col">
      <div className="mb-2">
        <h4 className="text-xs font-semibold text-slate-100">{title}</h4>
        {subtitle && <p className="text-[11px] text-slate-400 mt-0.5">{subtitle}</p>}
      </div>
      {unavailableReason ? (
        <div className="h-52 flex items-center justify-center text-xs text-slate-400 border border-dashed border-gridborder rounded">
          {unavailableReason}
        </div>
      ) : (
        <div className="h-52 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 6, right: 10, left: -8, bottom: 2 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis
                dataKey="step_idx"
                tick={{ fill: "#94a3b8", fontSize: 10 }}
                tickFormatter={(v) => `${v}`}
              />
              <YAxis
                tick={{ fill: "#94a3b8", fontSize: 10 }}
                tickFormatter={(v) => (unit ? `${v}${unit}` : `${v}`)}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0f172a",
                  borderColor: "#334155",
                  fontSize: "11px",
                }}
                labelFormatter={(label, payload) => {
                  const row = payload?.[0]?.payload as Record<string, unknown> | undefined;
                  const ts = row?.timestamp_label ? ` (${String(row.timestamp_label)})` : "";
                  return `Step ${label}${ts}`;
                }}
              />
              <Legend wrapperStyle={{ fontSize: "10px" }} />
              {series.map((s) => (
                <Line
                  key={s.key}
                  type="monotone"
                  dataKey={s.key}
                  name={s.label}
                  stroke={s.color}
                  strokeWidth={1.8}
                  dot={false}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
};

export const H2ComparisonViewTab: React.FC<H2ComparisonViewTabProps> = ({
  externalCompareIds,
  onClearExternalCompareIds,
  onInspectEpisode,
}) => {
  const [h2Payload, setH2Payload] = useState<ScenarioEpisodesResponse | null>(null);
  const [h2Error, setH2Error] = useState<string | null>(null);

  // H2 Matched Quartet Controls
  const [selectedTrack, setSelectedTrack] = useState<string>("track_a_intrinsic");
  const [selectedTrainSeed, setSelectedTrainSeed] = useState<number>(42);
  const [selectedEvalSeed, setSelectedEvalSeed] = useState<number | null>(null);
  const [includeRuleBased, setIncludeRuleBased] = useState<boolean>(true);
  const [useCustomSelection, setUseCustomSelection] = useState<boolean>(false);

  // Validation switches matching /api/episodes/compare
  const [requireSameScenario, setRequireSameScenario] = useState<boolean>(true);
  const [requireSameEvalSeed, setRequireSameEvalSeed] = useState<boolean>(true);

  // Comparison results + lazy full traces for 18 charts
  const [compareResult, setCompareResult] = useState<EpisodeCompareResponse | null>(null);
  const [tracesById, setTracesById] = useState<Record<string, EpisodeTraceResponse>>({});
  const [compareLoading, setCompareLoading] = useState<boolean>(false);
  const [compareError, setCompareError] = useState<string | null>(null);
  const [chartTab, setChartTab] = useState<ChartCategoryTab>("energy");

  useEffect(() => {
    const controller = new AbortController();
    api
      .getScenarioEpisodes("scenario_h2", undefined, controller.signal)
      .then((res) => {
        setH2Payload(res);
        const firstSeed = res.h2_comparison?.matched_seed_pairs?.[0]?.evaluation_seed ?? null;
        if (firstSeed !== null) {
          setSelectedEvalSeed((prev) => prev ?? firstSeed);
        }
      })
      .catch((err: unknown) => {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setH2Error(err instanceof Error ? err.message : "Failed to load H2 comparison dataset.");
      });
    return () => controller.abort();
  }, []);

  // Automatically switch to custom mode if user selects >=2 episodes in the Browser tab
  useEffect(() => {
    if (externalCompareIds.length >= 2) {
      setUseCustomSelection(true);
    }
  }, [externalCompareIds]);

  const h2Data: H2ComparisonDataset | null = h2Payload?.h2_comparison ?? null;

  const activeEpisodeIds: string[] = useMemo(() => {
    if (useCustomSelection && externalCompareIds.length >= 2) {
      return externalCompareIds.slice(0, 4);
    }
    if (!h2Data || selectedEvalSeed === null) return [];
    const pair = h2Data.matched_seed_pairs.find(
      (p) => p.evaluation_seed === selectedEvalSeed
    );
    if (!pair) return [];
    const ids: string[] = [];
    for (const alg of ["IPPO", "MAPPO_MATCHED", "HAPPO"]) {
      const candidates = pair.episodes_by_track_algorithm[`${selectedTrack}::${alg}`] ?? [];
      const match =
        candidates.find((id) => id.includes(`__seed${selectedTrainSeed}__`)) ??
        candidates[0];
      if (match) ids.push(match);
    }
    if (includeRuleBased) {
      const rb =
        pair.episodes_by_track_algorithm["track_a_intrinsic::Rule-Based EMS"]?.[0];
      if (rb) ids.push(rb);
    }
    return ids;
  }, [
    useCustomSelection,
    externalCompareIds,
    h2Data,
    selectedEvalSeed,
    selectedTrack,
    selectedTrainSeed,
    includeRuleBased,
  ]);

  const seriesMeta = useMemo(() => {
    if (!compareResult) return [];
    return compareResult.episodes_compared.map((row, idx) => ({
      id: row.episode_id,
      label: `${row.algorithm} (${TRACK_LABELS[row.track]?.split("·")[0]?.trim() ?? row.track}, ts=${
        row.training_seed ?? "det"
      })`,
      color: COMPARE_COLORS[idx % COMPARE_COLORS.length],
    }));
  }, [compareResult]);

  // Precompute all 18 comparative 96-step time series cleanly from aligned traces
  const chartData = useMemo(() => {
    if (!compareResult || seriesMeta.length === 0) return null;
    const stepCount = compareResult.aligned_step_count;
    const rows: Array<Record<string, number | string | null>> = [];
    const cumEns: Record<string, number> = {};
    const cumBatThroughput: Record<string, number> = {};
    const cumOpCost: Record<string, number> = {};
    const cumSocCost: Record<string, number> = {};
    seriesMeta.forEach((s) => {
      cumEns[s.id] = 0;
      cumBatThroughput[s.id] = 0;
      cumOpCost[s.id] = 0;
      cumSocCost[s.id] = 0;
    });

    for (let t = 0; t < stepCount; t++) {
      const aligned = compareResult.aligned_timesteps[t];
      const point: Record<string, number | string | null> = {
        step_idx: aligned?.step_idx ?? t,
        hour: aligned?.hour ?? t * 0.25,
        timestamp_label: aligned?.timestamp_label ?? "",
      };

      const firstStep = seriesMeta
        .map((s) => tracesById[s.id]?.steps?.[t])
        .find((st) => st !== undefined);
      const refNominalDemand: number | null = firstStep
        ? firstStep.nominal_critical_mw +
          firstStep.nominal_flexible_mw +
          firstStep.nominal_other_mw
        : null;

      seriesMeta.forEach((s) => {
        const trace = tracesById[s.id];
        const step = trace?.steps?.[t];
        const prefix = s.id;
        if (!step) return;

        const totalRe = step.solar_mw + step.wind_mw;
        const servedTotal =
          step.served_critical_mw + step.served_flexible_mw + step.served_other_mw;

        cumEns[s.id] += step.shed_critical_mw * 0.25;
        cumBatThroughput[s.id] += Math.abs(step.battery_mw) * 0.25;
        cumOpCost[s.id] += step.step_operating_cost_usd;
        cumSocCost[s.id] += step.step_social_cost_usd;

        point[`${prefix}__re`] = Number(totalRe.toFixed(4));
        point[`${prefix}__soc`] = Number(step.battery_soc_pct.toFixed(2));
        point[`${prefix}__bat_mw`] = Number(step.battery_mw.toFixed(4));
        point[`${prefix}__gen_mw`] = Number(step.generator_mw.toFixed(4));
        point[`${prefix}__grid_mw`] = Number(step.grid_import_mw.toFixed(4));
        point[`${prefix}__served`] = Number(servedTotal.toFixed(4));
        point[`${prefix}__shed_crit`] = Number(step.shed_critical_mw.toFixed(4));
        point[`${prefix}__cum_ens`] = Number(cumEns[s.id].toFixed(4));
        point[`${prefix}__shed_other`] = Number(
          (step.shed_flexible_mw + step.shed_other_mw).toFixed(4)
        );
        point[`${prefix}__vmin`] = Number(step.voltages_pu.min_bus.toFixed(4));
        point[`${prefix}__prop_bat`] = extractArrayVal(step.proposed_actions?.battery, 0);
        point[`${prefix}__exec_bat`] =
          step.executed_actions?.applied_battery_mw ??
          extractArrayVal(step.executed_actions?.battery, 0);
        point[`${prefix}__shield`] = step.shield_events?.any_intervention ? 1 : 0;
        point[`${prefix}__fallback`] = step.fallback_events?.adequacy_intervention ? 1 : 0;
        point[`${prefix}__gen_on`] = step.generator_state === "OFF" ? 0 : 1;
        point[`${prefix}__cum_bat_mwh`] = Number(cumBatThroughput[s.id].toFixed(4));
        point[`${prefix}__cum_op`] = Number(cumOpCost[s.id].toFixed(2));
        point[`${prefix}__cum_soc`] = Number(cumSocCost[s.id].toFixed(2));
        point[`${prefix}__reward`] = Number(step.reward.toFixed(4));
      });
      point["__nominal_demand"] =
        refNominalDemand !== null ? Number(refNominalDemand.toFixed(4)) : null;
      rows.push(point);
    }

    // Aggregate reward component breakdown totals for chart 18 companion
    const rewardComponents = seriesMeta.map((s) => {
      const trace = tracesById[s.id];
      const sums = { r_cost: 0, r_crit_ens: 0, r_flex_ens: 0, r_service: 0, r_other: 0 };
      (trace?.steps ?? []).forEach((st) => {
        const rb = st.reward_breakdown ?? {};
        sums.r_cost += Number(rb.r_cost ?? 0);
        sums.r_crit_ens += Number(rb.r_crit_ens ?? 0);
        sums.r_flex_ens += Number(rb.r_flex_ens ?? 0);
        sums.r_service += Number(rb.r_service ?? 0);
        sums.r_other +=
          Number(rb.r_emissions ?? 0) +
          Number(rb.r_degrad ?? 0) +
          Number(rb.r_voltage ?? 0) +
          Number(rb.r_imbalance ?? 0) +
          Number(rb.r_curt ?? 0) +
          Number(rb.r_reserve ?? 0) +
          Number(rb.r_other_ens ?? 0);
      });
      return {
        controller: s.label,
        r_cost: Number(sums.r_cost.toFixed(2)),
        r_crit_ens: Number(sums.r_crit_ens.toFixed(2)),
        r_flex_ens: Number(sums.r_flex_ens.toFixed(2)),
        r_service: Number(sums.r_service.toFixed(2)),
        r_other: Number(sums.r_other.toFixed(2)),
      };
    });

    return { rows, rewardComponents };
  }, [compareResult, seriesMeta, tracesById]);

  const insights = useMemo(() => {
    if (!compareResult || compareResult.episodes_compared.length === 0) return null;
    const rows = compareResult.episodes_compared;
    const sortedByCost = [...rows].sort(
      (a, b) => a.true_social_cost_usd - b.true_social_cost_usd
    );
    const sortedByRel = [...rows].sort(
      (a, b) => b.critical_reliability_pct - a.critical_reliability_pct
    );
    const maxFallback = [...rows].sort(
      (a, b) => b.fallback_intervention_steps - a.fallback_intervention_steps
    )[0];

    const refTrace = tracesById[compareResult.reference_episode_id];
    const outageSteps = (refTrace?.steps ?? [])
      .filter((s) => !s.grid_available || s.scenario_events.length > 0)
      .map((s) => s.step_idx);

    const hasTrackC = rows.some((r) => r.track === "track_c_hybrid");
    const mappoIntrinsic = h2Data?.matrix?.["MAPPO_MATCHED"]?.["track_a_intrinsic"];
    const mappoHybrid = h2Data?.matrix?.["MAPPO_MATCHED"]?.["track_c_hybrid"];

    return {
      bestCost: sortedByCost[0],
      worstCost: sortedByCost[sortedByCost.length - 1],
      bestRel: sortedByRel[0],
      maxFallback,
      outageRange:
        outageSteps.length > 0
          ? `Steps ${outageSteps[0]}–${outageSteps[outageSteps.length - 1]} (${outageSteps.length} steps)`
          : "None detected in aligned trace",
      hasTrackC,
      mappoIntrinsic,
      mappoHybrid,
    };
  }, [compareResult, tracesById, h2Data]);

  const loadRepresentativeSet = (
    repKey: string,
    repLabel: string
  ) => {
    if (!h2Data) return;
    // Open representative episode for selected track across RL algorithms + Rule-Based EMS if available
    const picked: string[] = [];
    for (const alg of ALGORITHMS_ORDER) {
      const tr = alg === "Rule-Based EMS" ? "track_a_intrinsic" : selectedTrack;
      const rep = h2Data.matrix?.[alg]?.[tr]?.representatives?.[repKey] as
        | H2MatrixRepresentative
        | undefined;
      if (rep?.episode_id) picked.push(rep.episode_id);
    }
    if (picked.length >= 2) {
      setRequireSameEvalSeed(false);
      setUseCustomSelection(true);
      // Compare via local inspect helper or custom URL: wait, externalCompareIds is controlled by parent, so let's handle preset custom ids cleanly in parent or local override!
      setLocalPresetIds({ label: repLabel, ids: picked });
    }
  };

  const [localPresetIds, setLocalPresetIds] = useState<{
    label: string;
    ids: string[];
  } | null>(null);

  const effectiveEpisodeIds =
    localPresetIds && localPresetIds.ids.length >= 2
      ? localPresetIds.ids
      : activeEpisodeIds;

  // Keep effect synced with effectiveEpisodeIds instead of activeEpisodeIds
  useEffect(() => {
    if (effectiveEpisodeIds.length < 2) return;
    const controller = new AbortController();
    setCompareLoading(true);
    setCompareError(null);

    Promise.all([
      api.compareSavedEpisodes(
        effectiveEpisodeIds,
        {
          require_same_scenario: requireSameScenario,
          require_same_eval_seed: requireSameEvalSeed,
        },
        controller.signal
      ),
      ...effectiveEpisodeIds.map((id) => api.getSavedEpisodeTrace(id, controller.signal)),
    ])
      .then(([cmp, ...loadedTraces]) => {
        setCompareResult(cmp);
        const map: Record<string, EpisodeTraceResponse> = {};
        loadedTraces.forEach((tr) => {
          map[tr.episode_id] = tr;
        });
        setTracesById(map);
      })
      .catch((err: unknown) => {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setCompareError(
          err instanceof Error ? err.message : "Failed to compare selected episodes."
        );
        setCompareResult(null);
      })
      .finally(() => {
        if (!controller.signal.aborted) setCompareLoading(false);
      });

    return () => controller.abort();
  }, [effectiveEpisodeIds, requireSameScenario, requireSameEvalSeed]);

  if (h2Error) {
    return <EmptyState title="Unable to Load H2 Comparison" description={h2Error} tone="error" />;
  }

  return (
    <div className="space-y-4">
      {/* H2 4x3 Matrix */}
      <div className="bg-gridcard border border-gridborder rounded-lg p-4 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-semibold text-slate-100">
              Scenario H2 Controller Matrix ({h2Data?.scenario_title ?? "Scenario H2"})
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Mean metrics across packaged H2 episodes ({h2Data?.total_h2_episodes ?? 140} H2 comparison
              episodes across 4 target controllers; Greedy &amp; Lookahead Heuristic remain in the 150-record H2 library).
            </p>
          </div>
          <span className="text-[11px] font-mono text-slate-400">
            Track A vs Track B vs Track C kept distinct
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-gridborder bg-gridpanel/70 text-[11px] uppercase tracking-wider text-slate-400">
                <th className="py-2 px-3">Track</th>
                {ALGORITHMS_ORDER.map((alg) => (
                  <th key={alg} className="py-2 px-3">
                    {formatModelLabel(alg)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gridborder/60">
              {TRACKS_ORDER.map((tr) => (
                <tr key={tr}>
                  <td className="py-2.5 px-3 font-medium text-slate-200 whitespace-nowrap">
                    {TRACK_LABELS[tr]}
                  </td>
                  {ALGORITHMS_ORDER.map((alg) => {
                    const cell = h2Data?.matrix?.[alg]?.[tr];
                    if (!cell || !cell.available) {
                      return (
                        <td key={alg} className="py-2.5 px-3 align-top">
                          <div className="inline-flex flex-col gap-1">
                            <StatusBadge label="Unavailable" tone="slate" />
                            <span className="text-[11px] text-slate-400 leading-snug max-w-[200px]">
                              {cell?.reason ?? "Not applicable for heuristic controller."}
                            </span>
                          </div>
                        </td>
                      );
                    }
                    return (
                      <td key={alg} className="py-2.5 px-3 align-top">
                        <div className="space-y-1">
                          <div className="flex items-center gap-1.5">
                            <StatusBadge
                              label={`n=${cell.episodes_n}`}
                              tone={modeBadgeTone(cell.mode)}
                            />
                            <StatusBadge
                              label={`${fmtNum(cell.mean_critical_reliability_pct, 2)}%`}
                              tone={reliabilityTone(cell.mean_critical_reliability_pct ?? 0)}
                            />
                          </div>
                          <div className="font-mono text-[11px] text-slate-100">
                            ${fmtNum(cell.mean_true_social_cost_usd, 1)}
                          </div>
                          <div className="text-[11px] text-slate-400 font-mono">
                            ENS {fmtNum(cell.mean_critical_ens_mwh, 4)} MWh · fb{" "}
                            {fmtNum(cell.mean_fallback_interventions, 1)}
                          </div>
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Selector & Quick-Picks */}
      <div className="bg-gridcard border border-gridborder rounded-lg p-4 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-200">
            <GitCompare className="w-4 h-4 text-cyan-400" />
            Paired Multi-Controller Trace Comparator
          </div>
          {localPresetIds && (
            <button
              type="button"
              onClick={() => {
                setLocalPresetIds(null);
                setRequireSameEvalSeed(true);
              }}
              className="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-gridpanel border border-gridborder text-xs text-slate-300 hover:text-white"
            >
              <X className="w-3.5 h-3.5" />
              Exit Quick-Pick ({localPresetIds.label})
            </button>
          )}
          {useCustomSelection && externalCompareIds.length >= 2 && !localPresetIds && (
            <button
              type="button"
              onClick={() => {
                onClearExternalCompareIds();
                setUseCustomSelection(false);
              }}
              className="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-gridpanel border border-gridborder text-xs text-slate-300 hover:text-white"
            >
              <X className="w-3.5 h-3.5" />
              Clear Browser Picks &amp; Use H2 Matched Seeds
            </button>
          )}
        </div>

        {(!useCustomSelection || externalCompareIds.length < 2) && !localPresetIds && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
            <select
              aria-label="H2 comparison track"
              value={selectedTrack}
              onChange={(e) => setSelectedTrack(e.target.value)}
              className="px-2.5 py-1.5 rounded-md bg-gridpanel border border-gridborder text-xs text-slate-100"
            >
              {TRACKS_ORDER.map((tr) => (
                <option key={tr} value={tr}>
                  {TRACK_LABELS[tr]}
                </option>
              ))}
            </select>

            <select
              aria-label="H2 comparison training seed"
              value={selectedTrainSeed}
              onChange={(e) => setSelectedTrainSeed(Number(e.target.value))}
              className="px-2.5 py-1.5 rounded-md bg-gridpanel border border-gridborder text-xs text-slate-100"
            >
              <option value={42}>RL Train Seed 42</option>
              <option value={101}>RL Train Seed 101</option>
              <option value={202}>RL Train Seed 202</option>
            </select>

            <select
              aria-label="H2 comparison evaluation seed"
              value={selectedEvalSeed ?? ""}
              onChange={(e) => setSelectedEvalSeed(Number(e.target.value))}
              className="px-2.5 py-1.5 rounded-md bg-gridpanel border border-gridborder text-xs text-slate-100"
            >
              {(h2Data?.matched_seed_pairs ?? []).map((p) => (
                <option key={p.evaluation_seed} value={p.evaluation_seed}>
                  Eval Seed {p.evaluation_seed} (Matched)
                </option>
              ))}
            </select>

            <label className="flex items-center gap-2 px-2.5 py-1.5 rounded-md bg-gridpanel border border-gridborder text-xs text-slate-200 cursor-pointer">
              <input
                type="checkbox"
                checked={includeRuleBased}
                onChange={(e) => setIncludeRuleBased(e.target.checked)}
                className="accent-cyan-400"
              />
              Include Rule-Based EMS (Track A)
            </label>
          </div>
        )}

        {/* Representative Quick-Picks + Compatibility Guards */}
        <div className="flex flex-wrap items-center justify-between gap-2 pt-1 border-t border-gridborder/60">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-[11px] text-slate-400 mr-1">
              H2 Representative Quick-Picks ({TRACK_LABELS[selectedTrack]?.split("·")[0]}):
            </span>
            {[
              { key: "best_social_cost", label: "Lowest Social Cost" },
              { key: "median_social_cost", label: "Median" },
              { key: "worst_social_cost", label: "Worst Cost" },
              { key: "most_reliable", label: "Most Reliable" },
              { key: "highest_critical_ens", label: "Highest Critical ENS" },
            ].map((item) => (
              <button
                key={item.key}
                type="button"
                onClick={() => loadRepresentativeSet(item.key, item.label)}
                className="px-2 py-1 rounded bg-gridpanel border border-gridborder text-[11px] text-slate-200 hover:border-cyan-400/60"
              >
                {item.label}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-3 text-[11px] text-slate-300">
            <label className="inline-flex items-center gap-1.5 cursor-pointer">
              <input
                type="checkbox"
                checked={requireSameScenario}
                onChange={(e) => setRequireSameScenario(e.target.checked)}
                className="accent-cyan-400"
              />
              Require Same Scenario
            </label>
            <label className="inline-flex items-center gap-1.5 cursor-pointer">
              <input
                type="checkbox"
                checked={requireSameEvalSeed}
                onChange={(e) => setRequireSameEvalSeed(e.target.checked)}
                className="accent-cyan-400"
              />
              Require Same Eval Seed
            </label>
          </div>
        </div>

        {localPresetIds && (
          <div className="bg-amber-950/20 border border-amber-500/30 rounded px-3 py-2 text-[11px] text-amber-200 flex items-start gap-2">
            <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
            <span>
              <strong>Representative Case Warning ({localPresetIds.label}):</strong> Selected using{" "}
              <em>{localPresetIds.label}</em> within each algorithm&apos;s {selectedTrack} pool. Single
              representative episodes are NOT overall controller averages and may belong to different seeds
              (Eval-Seed guard relaxed for cross-controller quick-pick).
            </span>
          </div>
        )}
      </div>

      {/* Comparison Results & Deltas */}
      {compareError ? (
        <EmptyState
          title="Incompatible or Failed Comparison"
          description={compareError}
          tone="warning"
        />
      ) : compareLoading || !compareResult ? (
        <EmptyState
          title="Loading Aligned Comparison"
          description="Fetching aligned summary metrics and 96-step traces for selected controllers..."
          tone="info"
        />
      ) : (
        <>
          {/* Summary Table */}
          <div className="bg-gridcard border border-gridborder rounded-lg overflow-hidden">
            <div className="px-4 py-2.5 border-b border-gridborder flex flex-wrap items-center justify-between gap-2 text-xs">
              <div className="font-semibold text-slate-200">
                Aligned Episode Summary ({compareResult.scenario_id.toUpperCase()} ·{" "}
                {compareResult.aligned_step_count} steps · Reference:{" "}
                <span className="font-mono text-cyan-300">
                  {compareResult.reference_episode_id}
                </span>
                ) · <span className="text-slate-400 font-normal">Reward: <code className="text-slate-200">reward_v3_social.yaml</code></span>
              </div>
              {insights?.hasTrackC && (
                <StatusBadge
                  label="Includes Track C Hybrid (+Fallback)"
                  tone="cyan"
                />
              )}
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-gridborder bg-gridpanel/70 text-[11px] uppercase tracking-wider text-slate-400">
                    <th className="py-2 px-3">Controller &amp; Track</th>
                    <th className="py-2 px-3">Seeds</th>
                    <th className="py-2 px-3 text-right">Social Cost</th>
                    <th className="py-2 px-3 text-right">Delta vs Ref</th>
                    <th className="py-2 px-3 text-right">Op. Cost</th>
                    <th className="py-2 px-3 text-right">Crit. ENS</th>
                    <th className="py-2 px-3 text-right">Crit. Rel.</th>
                    <th className="py-2 px-3 text-right">Curtail / Fuel</th>
                    <th className="py-2 px-3 text-right">Shield / FB</th>
                    <th className="py-2 px-3 text-right">Inspect</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gridborder/60">
                  {compareResult.episodes_compared.map((row: EpisodeCompareSummaryRow, idx: number) => (
                    <tr key={row.episode_id} className="hover:bg-gridpanel/40">
                      <td className="py-2.5 px-3">
                        <div className="flex items-center gap-2">
                          <span
                            className="w-2.5 h-2.5 rounded-full shrink-0"
                            style={{ backgroundColor: COMPARE_COLORS[idx % COMPARE_COLORS.length] }}
                          />
                          <span className="font-semibold text-slate-100">{formatModelLabel(row.algorithm)}</span>
                          <StatusBadge label={row.mode} tone={modeBadgeTone(row.mode)} />
                        </div>
                        <div className="font-mono text-[10px] text-slate-400 mt-0.5 pl-4">
                          {row.episode_id}
                        </div>
                      </td>
                      <td className="py-2.5 px-3 font-mono text-[11px] text-slate-300 whitespace-nowrap">
                        <div>ts={row.training_seed ?? "N/A"} / es={row.evaluation_seed}</div>
                        <div className="text-[10px] text-slate-400">
                          {(() => {
                            const meta = tracesById[row.episode_id]?.metadata as
                              | Record<string, unknown>
                              | undefined;
                            const ep = meta?.selected_checkpoint_episode;
                            const ckpt = meta?.checkpoint_path ? String(meta.checkpoint_path) : null;
                            if (!ckpt) return "ckpt: deterministic baseline";
                            const shortName = ckpt.split("/").slice(-3).join("/");
                            return `ckpt: ep=${String(ep ?? "?")} (${shortName})`;
                          })()}
                        </div>
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono text-slate-100 whitespace-nowrap">
                        ${fmtNum(row.true_social_cost_usd, 2)}
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono whitespace-nowrap">
                        <span
                          className={
                            row.delta_social_cost_vs_ref_usd < 0
                              ? "text-emerald-300"
                              : row.delta_social_cost_vs_ref_usd > 0
                              ? "text-amber-300"
                              : "text-slate-400"
                          }
                        >
                          {row.delta_social_cost_vs_ref_usd >= 0 ? "+" : ""}
                          ${fmtNum(row.delta_social_cost_vs_ref_usd, 2)}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono text-slate-300 whitespace-nowrap">
                        ${fmtNum(row.operating_cost_usd, 2)}
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono text-slate-200 whitespace-nowrap">
                        {fmtNum(row.critical_ens_mwh, 4)}
                      </td>
                      <td className="py-2.5 px-3 text-right whitespace-nowrap">
                        <StatusBadge
                          label={`${fmtNum(row.critical_reliability_pct, 2)}%`}
                          tone={reliabilityTone(row.critical_reliability_pct)}
                        />
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono text-[11px] text-slate-300 whitespace-nowrap">
                        {fmtNum(row.renewable_curtailment_mwh, 3)} MWh / {fmtNum(row.fuel_consumed_l, 1)} L
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono text-[11px] text-slate-300 whitespace-nowrap">
                        clip={row.boundary_clipping_count} · emg={row.emergency_override_count} · fb=
                        {row.fallback_intervention_steps}
                      </td>
                      <td className="py-2.5 px-3 text-right whitespace-nowrap">
                        <button
                          type="button"
                          onClick={() => onInspectEpisode(row.episode_id)}
                          className="inline-flex items-center gap-1 px-2 py-1 rounded bg-cyan-500/15 border border-cyan-500/40 text-cyan-300 text-[11px] hover:bg-cyan-500/25"
                        >
                          <Play className="w-3 h-3" />
                          Replay
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Concise Metric Glossary */}
            <div className="px-4 py-2.5 bg-gridpanel/50 border-t border-gridborder grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-2 text-[11px] text-slate-400">
              {METRIC_GLOSSARY.map((g) => (
                <div key={g.label}>
                  <span className="font-semibold text-slate-200">{g.label}:</span> {g.help}
                </div>
              ))}
            </div>
          </div>

          {/* 18 Comparative Trace Charts across 4 clean sub-tabs */}
          {chartData && (
            <div className="bg-gridcard border border-gridborder rounded-lg p-4 space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-gridborder pb-3">
                <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-200">
                  <BarChart3 className="w-4 h-4 text-cyan-400" />
                  18 Aligned 96-Step Comparative Visualizations
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {[
                    { id: "energy" as const, label: "1. Energy & Grid (5)" },
                    { id: "reliability" as const, label: "2. Reliability & Voltage (5)" },
                    { id: "behavior" as const, label: "3. Controller Behavior (5)" },
                    { id: "economic" as const, label: "4. Economics & Reward (3)" },
                  ].map((tab) => (
                    <button
                      key={tab.id}
                      type="button"
                      onClick={() => setChartTab(tab.id)}
                      className={`px-3 py-1.5 rounded-md text-xs font-medium border transition-colors ${
                        chartTab === tab.id
                          ? "bg-cyan-500/20 border-cyan-400/50 text-cyan-200"
                          : "bg-gridpanel border-gridborder text-slate-300 hover:text-white"
                      }`}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>
              </div>

              {chartTab === "energy" && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                  <MultiTraceChart
                    title="1. Solar & Wind Dispatched Output"
                    subtitle="Combined solar_mw + wind_mw across 96 steps"
                    data={chartData.rows}
                    unit=" MW"
                    series={seriesMeta.map((s) => ({
                      key: `${s.id}__re`,
                      label: s.label,
                      color: s.color,
                    }))}
                  />
                  <MultiTraceChart
                    title="2. Battery State of Charge (SOC)"
                    subtitle="battery_soc_pct (20–95% operational window)"
                    data={chartData.rows}
                    unit="%"
                    series={seriesMeta.map((s) => ({
                      key: `${s.id}__soc`,
                      label: s.label,
                      color: s.color,
                    }))}
                  />
                  <MultiTraceChart
                    title="3. Battery Power Dispatch"
                    subtitle="Positive = discharge to microgrid, Negative = charge"
                    data={chartData.rows}
                    unit=" MW"
                    series={seriesMeta.map((s) => ({
                      key: `${s.id}__bat_mw`,
                      label: s.label,
                      color: s.color,
                    }))}
                  />
                  <MultiTraceChart
                    title="4. Diesel Generator Output"
                    subtitle="generator_mw active thermal dispatch"
                    data={chartData.rows}
                    unit=" MW"
                    series={seriesMeta.map((s) => ({
                      key: `${s.id}__gen_mw`,
                      label: s.label,
                      color: s.color,
                    }))}
                  />
                  <MultiTraceChart
                    title="5. Utility Grid Import"
                    subtitle="grid_import_mw (zero during islanded / outage steps)"
                    data={chartData.rows}
                    unit=" MW"
                    series={seriesMeta.map((s) => ({
                      key: `${s.id}__grid_mw`,
                      label: s.label,
                      color: s.color,
                    }))}
                  />
                </div>
              )}

              {chartTab === "reliability" && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                  <MultiTraceChart
                    title="6. Nominal Demand vs Total Served Load"
                    subtitle="Dashed slate line = reference nominal total load"
                    data={chartData.rows}
                    unit=" MW"
                    series={[
                      { key: "__nominal_demand", label: "Nominal Demand", color: "#94a3b8" },
                      ...seriesMeta.map((s) => ({
                        key: `${s.id}__served`,
                        label: `${s.label} Served`,
                        color: s.color,
                      })),
                    ]}
                  />
                  <MultiTraceChart
                    title="7. Critical Load Shedding per Step"
                    subtitle="shed_critical_mw (lower is better; 0 = zero outage loss)"
                    data={chartData.rows}
                    unit=" MW"
                    series={seriesMeta.map((s) => ({
                      key: `${s.id}__shed_crit`,
                      label: s.label,
                      color: s.color,
                    }))}
                  />
                  <MultiTraceChart
                    title="8. Cumulative Critical ENS Trajectory"
                    subtitle="Running total MWh of unserved critical load"
                    data={chartData.rows}
                    unit=" MWh"
                    series={seriesMeta.map((s) => ({
                      key: `${s.id}__cum_ens`,
                      label: s.label,
                      color: s.color,
                    }))}
                  />
                  <MultiTraceChart
                    title="9. Flexible & Other Load Shedding"
                    subtitle="shed_flexible_mw + shed_other_mw"
                    data={chartData.rows}
                    unit=" MW"
                    series={seriesMeta.map((s) => ({
                      key: `${s.id}__shed_other`,
                      label: s.label,
                      color: s.color,
                    }))}
                  />
                  <MultiTraceChart
                    title="10. Minimum Bus Voltage (p.u.)"
                    subtitle="voltages_pu.min_bus across all buses"
                    data={chartData.rows}
                    unit=" pu"
                    series={seriesMeta.map((s) => ({
                      key: `${s.id}__vmin`,
                      label: s.label,
                      color: s.color,
                    }))}
                  />
                </div>
              )}

              {chartTab === "behavior" && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                  <MultiTraceChart
                    title="11. Proposed vs Executed Battery Action (Reference & Peers)"
                    subtitle="Solid = applied_battery_mw; first peer shown for executed delta"
                    data={chartData.rows}
                    series={[
                      {
                        key: `${seriesMeta[0].id}__prop_bat`,
                        label: `${seriesMeta[0].label} Proposed (norm)`,
                        color: "#e2e8f0",
                      },
                      ...seriesMeta.map((s) => ({
                        key: `${s.id}__exec_bat`,
                        label: `${s.label} Applied MW`,
                        color: s.color,
                      })),
                    ]}
                  />
                  <MultiTraceChart
                    title="12. Safety Shield Interventions per Step (0/1)"
                    subtitle="shield_events.any_intervention flag by step"
                    data={chartData.rows}
                    series={seriesMeta.map((s) => ({
                      key: `${s.id}__shield`,
                      label: s.label,
                      color: s.color,
                    }))}
                  />
                  <MultiTraceChart
                    title="13. Operational Adequacy Fallback Interventions (0/1)"
                    subtitle="fallback_events.adequacy_intervention (Track C only)"
                    data={chartData.rows}
                    series={seriesMeta.map((s) => ({
                      key: `${s.id}__fallback`,
                      label: s.label,
                      color: s.color,
                    }))}
                  />
                  <MultiTraceChart
                    title="14. Generator Online Status (0 = OFF / 1 = ON or WARM)"
                    subtitle="generator_state active commitment timeline"
                    data={chartData.rows}
                    series={seriesMeta.map((s) => ({
                      key: `${s.id}__gen_on`,
                      label: s.label,
                      color: s.color,
                    }))}
                  />
                  <MultiTraceChart
                    title="15. Cumulative Battery Energy Throughput"
                    subtitle="Integrated |battery_mw| * 0.25 MWh"
                    data={chartData.rows}
                    unit=" MWh"
                    series={seriesMeta.map((s) => ({
                      key: `${s.id}__cum_bat_mwh`,
                      label: s.label,
                      color: s.color,
                    }))}
                  />
                </div>
              )}

              {chartTab === "economic" && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                  <MultiTraceChart
                    title="16. Cumulative Operating Cost ($)"
                    subtitle="Running sum of step_operating_cost_usd"
                    data={chartData.rows}
                    unit="$"
                    series={seriesMeta.map((s) => ({
                      key: `${s.id}__cum_op`,
                      label: s.label,
                      color: s.color,
                    }))}
                  />
                  <MultiTraceChart
                    title="17. Cumulative True Social Cost ($)"
                    subtitle="Operating cost + ENS social penalty + battery degradation"
                    data={chartData.rows}
                    unit="$"
                    series={seriesMeta.map((s) => ({
                      key: `${s.id}__cum_soc`,
                      label: s.label,
                      color: s.color,
                    }))}
                  />
                  <MultiTraceChart
                    title="18a. Step Reward Trajectory"
                    subtitle="Per-step canonical environment reward"
                    data={chartData.rows}
                    series={seriesMeta.map((s) => ({
                      key: `${s.id}__reward`,
                      label: s.label,
                      color: s.color,
                    }))}
                  />
                  <div className="bg-gridcard border border-gridborder rounded-lg p-3.5 flex flex-col">
                    <div className="mb-2">
                      <h4 className="text-xs font-semibold text-slate-100">
                        18b. Episode Reward Component Totals
                      </h4>
                      <p className="text-[11px] text-slate-400 mt-0.5">
                        Summed across 96 steps from canonical step reward_breakdown
                      </p>
                    </div>
                    <div className="h-52 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={chartData.rewardComponents}
                          margin={{ top: 6, right: 10, left: -8, bottom: 2 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                          <XAxis
                            dataKey="controller"
                            tick={{ fill: "#94a3b8", fontSize: 10 }}
                          />
                          <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "#0f172a",
                              borderColor: "#334155",
                              fontSize: "11px",
                            }}
                          />
                          <Legend wrapperStyle={{ fontSize: "10px" }} />
                          <Bar dataKey="r_cost" name="r_cost" fill="#f59e0b" />
                          <Bar dataKey="r_crit_ens" name="r_crit_ens" fill="#ef4444" />
                          <Bar dataKey="r_service" name="r_service" fill="#10b981" />
                          <Bar dataKey="r_other" name="r_other" fill="#64748b" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Data-Driven Comparison Insights Panel */}
          {insights && (
            <div className="bg-gridcard border border-gridborder rounded-lg p-4 space-y-2">
              <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-200">
                <Sparkles className="w-4 h-4 text-cyan-400" />
                Observed Comparison Insights (Computed from Loaded Traces)
              </div>
              <ul className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs text-slate-300">
                <li className="bg-gridpanel/60 border border-gridborder/70 rounded p-2.5 flex items-start gap-2">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 mt-0.5 shrink-0" />
                  <span>
                    <strong>Lowest Social Cost in Selection:</strong> {insights.bestCost.algorithm} (
                    {insights.bestCost.mode}) at ${fmtNum(insights.bestCost.true_social_cost_usd, 2)} vs{" "}
                    {insights.worstCost.algorithm} at $
                    {fmtNum(insights.worstCost.true_social_cost_usd, 2)}.
                  </span>
                </li>
                <li className="bg-gridpanel/60 border border-gridborder/70 rounded p-2.5 flex items-start gap-2">
                  <CheckCircle2 className="w-3.5 h-3.5 text-cyan-400 mt-0.5 shrink-0" />
                  <span>
                    <strong>Critical Reliability Leader:</strong> {insights.bestRel.algorithm} (
                    {fmtNum(insights.bestRel.critical_reliability_pct, 2)}%, ENS{" "}
                    {fmtNum(insights.bestRel.critical_ens_mwh, 4)} MWh). Outage/event window:{" "}
                    {insights.outageRange}.
                  </span>
                </li>
                <li className="bg-gridpanel/60 border border-gridborder/70 rounded p-2.5 flex items-start gap-2">
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-400 mt-0.5 shrink-0" />
                  <span>
                    <strong>Fallback Attribution:</strong> Highest fallback count in selection:{" "}
                    {insights.maxFallback.algorithm} ({insights.maxFallback.fallback_intervention_steps}{" "}
                    steps).{" "}
                    {insights.mappoIntrinsic?.available && insights.mappoHybrid?.available
                      ? `H2 MAPPO_MATCHED Track A mean is $${fmtNum(
                          insights.mappoIntrinsic.mean_true_social_cost_usd,
                          1
                        )} vs Track C Hybrid $${fmtNum(
                          insights.mappoHybrid.mean_true_social_cost_usd,
                          1
                        )} (policy + fallback system).`
                      : "Fallback steps reflect operational adequacy override, not intrinsic RL alone."}
                  </span>
                </li>
                <li className="bg-gridpanel/60 border border-gridborder/70 rounded p-2.5 flex items-start gap-2">
                  <CheckCircle2 className="w-3.5 h-3.5 text-slate-400 mt-0.5 shrink-0" />
                  <span>
                    <strong>Accounting Reconciliation Note:</strong> Summary{" "}
                    <code className="text-cyan-300">fuel_consumed_l</code> includes the 10.0 L cold-start
                    ignition fuel per generator startup, whereas step traces report running burn per step.
                  </span>
                </li>
              </ul>
            </div>
          )}
        </>
      )}
    </div>
  );
};
