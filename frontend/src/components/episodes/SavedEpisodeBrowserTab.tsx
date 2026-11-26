import React, { useEffect, useState } from "react";
import {
  ArrowDownUp,
  ChevronLeft,
  ChevronRight,
  GitCompare,
  Play,
  RotateCcw,
  Search,
} from "lucide-react";
import { api } from "../../services/api";
import { KpiCard } from "../KpiCard";
import { EmptyState } from "../EmptyState";
import { StatusBadge, modeBadgeTone } from "../StatusBadge";
import { formatModelLabel } from "./episodeHelpers";
import type {
  EpisodeIndexItem,
  EpisodeLibrarySummaryResponse,
} from "../../types/api";

interface SavedEpisodeBrowserTabProps {
  summary: EpisodeLibrarySummaryResponse | null;
  selectedCompareIds: string[];
  onToggleCompareId: (episode: EpisodeIndexItem) => void;
  onInspectEpisode: (episodeId: string) => void;
  onOpenComparatorWith: (episodeIds: string[]) => void;
}

const TRACK_LABELS: Record<string, string> = {
  track_a_intrinsic: "Track A · Intrinsic / Heuristic",
  track_b_shielded: "Track B · Shielded",
  track_c_hybrid: "Track C · Hybrid (+Fallback)",
};

function reliabilityTone(pct: number): "emerald" | "amber" | "red" {
  if (pct >= 98) return "emerald";
  if (pct >= 90) return "amber";
  return "red";
}

function fmtNum(val: number | null | undefined, digits = 2, prefix = ""): string {
  if (val === null || val === undefined || !Number.isFinite(val)) return "Unavailable";
  return `${prefix}${val.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}`;
}

export const SavedEpisodeBrowserTab: React.FC<SavedEpisodeBrowserTabProps> = ({
  summary,
  selectedCompareIds,
  onToggleCompareId,
  onInspectEpisode,
  onOpenComparatorWith,
}) => {
  const [searchInput, setSearchInput] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [algorithm, setAlgorithm] = useState("");
  const [track, setTrack] = useState("");
  const [scenarioId, setScenarioId] = useState("");
  const [evaluationSeed, setEvaluationSeed] = useState("");
  const [trainingSeed, setTrainingSeed] = useState("");
  const [socialCostMax, setSocialCostMax] = useState("");
  const [reliabilityMin, setReliabilityMin] = useState("");
  const [sortBy, setSortBy] = useState("episode_id");
  const [descending, setDescending] = useState(false);
  const [offset, setOffset] = useState(0);
  const [limit, setLimit] = useState(25);

  const [episodes, setEpisodes] = useState<EpisodeIndexItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setDebouncedSearch(searchInput.trim());
      setOffset(0);
    }, 250);
    return () => window.clearTimeout(timer);
  }, [searchInput]);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);

    api
      .getSavedEpisodes(
        {
          search: debouncedSearch || undefined,
          algorithm: algorithm || undefined,
          track: track || undefined,
          scenario_id: scenarioId || undefined,
          evaluation_seed: evaluationSeed ? Number(evaluationSeed) : undefined,
          training_seed: trainingSeed ? Number(trainingSeed) : undefined,
          social_cost_max: socialCostMax ? Number(socialCostMax) : undefined,
          critical_reliability_min: reliabilityMin ? Number(reliabilityMin) : undefined,
          sort_by: sortBy,
          descending,
          offset,
          limit,
        },
        controller.signal
      )
      .then((res) => {
        setEpisodes(res.episodes);
        setTotal(res.total);
      })
      .catch((err: unknown) => {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setError(err instanceof Error ? err.message : "Failed to load saved episodes.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });

    return () => controller.abort();
  }, [
    debouncedSearch,
    algorithm,
    track,
    scenarioId,
    evaluationSeed,
    trainingSeed,
    socialCostMax,
    reliabilityMin,
    sortBy,
    descending,
    offset,
    limit,
  ]);

  const resetFilters = () => {
    setSearchInput("");
    setDebouncedSearch("");
    setAlgorithm("");
    setTrack("");
    setScenarioId("");
    setEvaluationSeed("");
    setTrainingSeed("");
    setSocialCostMax("");
    setReliabilityMin("");
    setSortBy("episode_id");
    setDescending(false);
    setOffset(0);
  };

  const totalPages = Math.max(1, Math.ceil(total / limit));
  const currentPage = Math.floor(offset / limit) + 1;
  const manifestIntegrity = summary?.manifest_integrity ?? {};
  const missingFiles = Number(manifestIntegrity["missing_trace_files"] ?? 0);

  return (
    <div className="space-y-4">
      {/* Library Summary KPI Strip */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        <KpiCard
          title="Indexed Episodes"
          value={summary?.total_episodes ?? null}
          subtitle="Lazy-loaded canonical index (no bulk JSON preload)"
          accent="cyan"
        />
        <KpiCard
          title="Evaluation Scenarios"
          value={summary?.scenarios.length ?? null}
          subtitle={summary?.scenarios.join(", ") || "Scenarios"}
          accent="blue"
        />
        <KpiCard
          title="Control Tracks"
          value={summary?.tracks.length ?? null}
          subtitle="Intrinsic / Shielded / Hybrid kept strictly separated"
          accent="amber"
        />
        <KpiCard
          title="Trace Integrity"
          value={
            summary
              ? missingFiles === 0
                ? "Verified"
                : `${missingFiles} Missing`
              : null
          }
          subtitle={
            summary
              ? `${summary.algorithms.length} controllers · ${summary.evaluation_seeds.length} eval seeds`
              : "Checking index manifest..."
          }
          accent={missingFiles === 0 ? "green" : "red"}
        />
      </div>

      {/* Track Separation Notice */}
      {summary && (
        <div className="bg-amber-950/20 border border-amber-500/30 rounded-lg px-4 py-2.5 text-xs text-amber-200 flex flex-wrap items-center justify-between gap-2">
          <span>
            <strong>Scientific Track Separation:</strong> Track A (Intrinsic RL &amp; Baselines),
            Track B (Shielded RL), and Track C (Hybrid + Adequacy Fallback) remain labeled and
            separated. Fallback gains are attributed to the hybrid system, not intrinsic RL alone.
          </span>
          <span className="font-mono text-[11px] text-amber-300/90">
            Schema {summary.schema_version}
          </span>
        </div>
      )}

      {/* Filter Toolbar */}
      <div className="bg-gridcard border border-gridborder rounded-lg p-4 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Filter &amp; Search Saved Episode Library
          </div>
          <div className="flex items-center gap-2">
            {selectedCompareIds.length >= 2 && (
              <button
                type="button"
                onClick={() => onOpenComparatorWith(selectedCompareIds)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-cyan-500/20 border border-cyan-400/50 text-cyan-200 text-xs font-medium hover:bg-cyan-500/30 transition-colors"
              >
                <GitCompare className="w-3.5 h-3.5" />
                Compare Selected ({selectedCompareIds.length})
              </button>
            )}
            <button
              type="button"
              onClick={resetFilters}
              className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md border border-gridborder bg-gridpanel text-slate-300 text-xs hover:text-white transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Reset Filters
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
          <div className="relative">
            <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-2.5" />
            <input
              type="text"
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              placeholder="Search episode ID or controller..."
              aria-label="Search episode ID"
              className="w-full pl-8 pr-2.5 py-1.5 rounded-md bg-gridpanel border border-gridborder text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-400"
            />
          </div>

          <select
            aria-label="Filter by scenario"
            value={scenarioId}
            onChange={(e) => {
              setScenarioId(e.target.value);
              setOffset(0);
            }}
            className="px-2.5 py-1.5 rounded-md bg-gridpanel border border-gridborder text-xs text-slate-100"
          >
            <option value="">All Scenarios (H1–H6)</option>
            {(summary?.scenarios ?? ["scenario_h1", "scenario_h2", "scenario_h3", "scenario_h4", "scenario_h5", "scenario_h6"]).map((sc) => (
              <option key={sc} value={sc}>
                {sc.toUpperCase()}
              </option>
            ))}
          </select>

          <select
            aria-label="Filter by algorithm"
            value={algorithm}
            onChange={(e) => {
              setAlgorithm(e.target.value);
              setOffset(0);
            }}
            className="px-2.5 py-1.5 rounded-md bg-gridpanel border border-gridborder text-xs text-slate-100"
          >
            <option value="">All Controllers / Algorithms</option>
            {(summary?.algorithms ?? []).map((alg) => (
              <option key={alg} value={alg}>
                {formatModelLabel(alg)}
              </option>
            ))}
          </select>

          <select
            aria-label="Filter by track"
            value={track}
            onChange={(e) => {
              setTrack(e.target.value);
              setOffset(0);
            }}
            className="px-2.5 py-1.5 rounded-md bg-gridpanel border border-gridborder text-xs text-slate-100"
          >
            <option value="">All Tracks (A / B / C)</option>
            {(summary?.tracks ?? ["track_a_intrinsic", "track_b_shielded", "track_c_hybrid"]).map((tr) => (
              <option key={tr} value={tr}>
                {TRACK_LABELS[tr] ?? tr}
              </option>
            ))}
          </select>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-2.5">
          <select
            aria-label="Filter by training seed"
            value={trainingSeed}
            onChange={(e) => {
              setTrainingSeed(e.target.value);
              setOffset(0);
            }}
            className="px-2.5 py-1.5 rounded-md bg-gridpanel border border-gridborder text-xs text-slate-100"
          >
            <option value="">All Training Seeds</option>
            <option value="42">Train Seed 42</option>
            <option value="101">Train Seed 101</option>
            <option value="202">Train Seed 202</option>
          </select>

          <select
            aria-label="Filter by evaluation seed"
            value={evaluationSeed}
            onChange={(e) => {
              setEvaluationSeed(e.target.value);
              setOffset(0);
            }}
            className="px-2.5 py-1.5 rounded-md bg-gridpanel border border-gridborder text-xs text-slate-100"
          >
            <option value="">All Eval Seeds</option>
            {(summary?.evaluation_seeds ?? [42, 43, 44, 45, 46]).map((seed) => (
              <option key={seed} value={String(seed)}>
                Eval Seed {seed}
              </option>
            ))}
          </select>

          <input
            type="number"
            step="100"
            value={socialCostMax}
            onChange={(e) => {
              setSocialCostMax(e.target.value);
              setOffset(0);
            }}
            placeholder="Max Social Cost ($)"
            aria-label="Maximum social cost USD"
            className="px-2.5 py-1.5 rounded-md bg-gridpanel border border-gridborder text-xs text-slate-100 placeholder-slate-500"
          />

          <input
            type="number"
            step="0.5"
            min="0"
            max="100"
            value={reliabilityMin}
            onChange={(e) => {
              setReliabilityMin(e.target.value);
              setOffset(0);
            }}
            placeholder="Min Critical Rel (%)"
            aria-label="Minimum critical reliability percent"
            className="px-2.5 py-1.5 rounded-md bg-gridpanel border border-gridborder text-xs text-slate-100 placeholder-slate-500"
          />

          <div className="flex items-center gap-1.5">
            <select
              aria-label="Sort episodes by"
              value={sortBy}
              onChange={(e) => {
                setSortBy(e.target.value);
                setOffset(0);
              }}
              className="flex-1 px-2.5 py-1.5 rounded-md bg-gridpanel border border-gridborder text-xs text-slate-100"
            >
              <option value="episode_id">Sort: Episode ID</option>
              <option value="true_social_cost_usd">Sort: True Social Cost</option>
              <option value="operating_cost_usd">Sort: Operating Cost</option>
              <option value="critical_ens_mwh">Sort: Critical ENS</option>
              <option value="critical_reliability_pct">Sort: Critical Reliability</option>
              <option value="fallback_intervention_steps">Sort: Fallback Steps</option>
              <option value="evaluation_seed">Sort: Eval Seed</option>
            </select>
            <button
              type="button"
              onClick={() => setDescending((d) => !d)}
              title={descending ? "Descending" : "Ascending"}
              aria-label="Toggle sort order"
              className="px-2.5 py-1.5 rounded-md bg-gridpanel border border-gridborder text-xs text-slate-200 hover:text-white inline-flex items-center gap-1"
            >
              <ArrowDownUp className="w-3.5 h-3.5" />
              {descending ? "Desc" : "Asc"}
            </button>
          </div>
        </div>
      </div>

      {/* Results Table */}
      {error ? (
        <EmptyState title="Unable to Load Saved Episodes" description={error} tone="error" />
      ) : !loading && episodes.length === 0 ? (
        <EmptyState
          title="No Saved Episodes Match Current Filters"
          description="Try clearing one or more filters (scenario, track, seed, or cost/reliability thresholds)."
          tone="warning"
          action={
            <button
              type="button"
              onClick={resetFilters}
              className="px-3 py-1.5 rounded-md bg-gridcard border border-gridborder text-xs text-slate-200 hover:text-white"
            >
              Reset Filters
            </button>
          }
        />
      ) : (
        <div className="bg-gridcard border border-gridborder rounded-lg overflow-hidden">
          <div className="px-4 py-2.5 border-b border-gridborder flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400">
            <span>
              {loading
                ? "Loading matching episodes..."
                : `Showing ${offset + 1}–${Math.min(offset + episodes.length, total)} of ${total} matching episodes`}
            </span>
            <div className="flex items-center gap-2">
              <span>Per page:</span>
              <select
                aria-label="Episodes per page"
                value={limit}
                onChange={(e) => {
                  setLimit(Number(e.target.value));
                  setOffset(0);
                }}
                className="px-2 py-1 rounded bg-gridpanel border border-gridborder text-slate-200 text-xs"
              >
                <option value={15}>15</option>
                <option value={25}>25</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
              </select>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-gridborder bg-gridpanel/70 text-[11px] uppercase tracking-wider text-slate-400">
                  <th className="py-2.5 px-3">Episode ID &amp; Controller</th>
                  <th className="py-2.5 px-3">Track / Mode</th>
                  <th className="py-2.5 px-3">Scenario &amp; Seeds</th>
                  <th className="py-2.5 px-3 text-right">Social Cost</th>
                  <th className="py-2.5 px-3 text-right">Op. Cost</th>
                  <th className="py-2.5 px-3 text-right">Critical ENS</th>
                  <th className="py-2.5 px-3 text-right">Critical Rel.</th>
                  <th className="py-2.5 px-3 text-right">Fallback</th>
                  <th className="py-2.5 px-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gridborder/60">
                {episodes.map((ep) => {
                  const isPicked = selectedCompareIds.includes(ep.episode_id);
                  return (
                    <tr
                      key={ep.episode_id}
                      className="hover:bg-gridpanel/50 transition-colors"
                    >
                      <td className="py-2.5 px-3">
                        <div className="font-mono text-[11px] text-cyan-300 break-all">
                          {ep.episode_id}
                        </div>
                        <div className="text-[11px] text-slate-400 mt-0.5">
                          {ep.controller_name} ({formatModelLabel(ep.algorithm)})
                        </div>
                      </td>
                      <td className="py-2.5 px-3 whitespace-nowrap">
                        <StatusBadge
                          label={ep.mode}
                          tone={modeBadgeTone(ep.mode)}
                        />
                        <div className="text-[10px] text-slate-500 mt-0.5 font-mono">
                          {ep.track}
                        </div>
                      </td>
                      <td className="py-2.5 px-3 whitespace-nowrap font-mono text-[11px] text-slate-300">
                        <div>{ep.scenario_id.toUpperCase()}</div>
                        <div className="text-[10px] text-slate-400">
                          train={ep.training_seed ?? "N/A"} · eval={ep.evaluation_seed}
                        </div>
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono text-slate-100 whitespace-nowrap">
                        ${fmtNum(ep.true_social_cost_usd, 2)}
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono text-slate-300 whitespace-nowrap">
                        ${fmtNum(ep.operating_cost_usd, 2)}
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono text-slate-200 whitespace-nowrap">
                        {fmtNum(ep.critical_ens_mwh, 4)} MWh
                      </td>
                      <td className="py-2.5 px-3 text-right whitespace-nowrap">
                        <StatusBadge
                          label={`${fmtNum(ep.critical_reliability_pct, 2)}%`}
                          tone={reliabilityTone(ep.critical_reliability_pct)}
                        />
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono text-slate-300 whitespace-nowrap">
                        {ep.fallback_intervention_steps} / {ep.episode_length}
                      </td>
                      <td className="py-2.5 px-3 text-right whitespace-nowrap">
                        <div className="inline-flex items-center gap-1.5">
                          <button
                            type="button"
                            onClick={() => onInspectEpisode(ep.episode_id)}
                            className="inline-flex items-center gap-1 px-2 py-1 rounded bg-cyan-500/15 border border-cyan-500/40 text-cyan-300 hover:bg-cyan-500/25 text-[11px] font-medium"
                          >
                            <Play className="w-3 h-3" />
                            Replay
                          </button>
                          <button
                            type="button"
                            onClick={() => onToggleCompareId(ep)}
                            className={`inline-flex items-center gap-1 px-2 py-1 rounded border text-[11px] font-medium transition-colors ${
                              isPicked
                                ? "bg-indigo-500/25 border-indigo-400/60 text-indigo-200"
                                : "bg-gridpanel border-gridborder text-slate-300 hover:text-white"
                            }`}
                          >
                            <GitCompare className="w-3 h-3" />
                            {isPicked ? "Picked" : "Compare"}
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Pagination Footer */}
          <div className="px-4 py-2.5 border-t border-gridborder flex items-center justify-between text-xs text-slate-400">
            <span>
              Page {currentPage} of {totalPages}
            </span>
            <div className="flex items-center gap-2">
              <button
                type="button"
                disabled={offset === 0}
                onClick={() => setOffset((o) => Math.max(0, o - limit))}
                className="inline-flex items-center gap-1 px-2.5 py-1 rounded border border-gridborder bg-gridpanel text-slate-200 disabled:opacity-40"
              >
                <ChevronLeft className="w-3.5 h-3.5" />
                Previous
              </button>
              <button
                type="button"
                disabled={offset + limit >= total}
                onClick={() => setOffset((o) => o + limit)}
                className="inline-flex items-center gap-1 px-2.5 py-1 rounded border border-gridborder bg-gridpanel text-slate-200 disabled:opacity-40"
              >
                Next
                <ChevronRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
