import React, { useEffect, useState } from "react";
import { Database, GitCompare, Play, Table2 } from "lucide-react";
import { api } from "../services/api";
import { SavedEpisodeBrowserTab } from "../components/episodes/SavedEpisodeBrowserTab";
import { H2ComparisonViewTab } from "../components/episodes/H2ComparisonViewTab";
import { EpisodeReplayViewTab } from "../components/episodes/EpisodeReplayViewTab";
import type {
  EpisodeIndexItem,
  EpisodeLibrarySummaryResponse,
} from "../types/api";

type ActiveTab = "browser" | "h2_comparison" | "replay";

export const SavedEpisodeLibraryPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<ActiveTab>("browser");
  const [summary, setSummary] = useState<EpisodeLibrarySummaryResponse | null>(null);
  const [selectedCompareIds, setSelectedCompareIds] = useState<string[]>([]);
  const [replayEpisodeId, setReplayEpisodeId] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    api
      .getEpisodeLibrarySummary(controller.signal)
      .then((res) => setSummary(res))
      .catch(() => {});
    return () => controller.abort();
  }, []);

  const handleToggleCompare = (ep: EpisodeIndexItem) => {
    setSelectedCompareIds((prev) => {
      if (prev.includes(ep.episode_id)) {
        return prev.filter((id) => id !== ep.episode_id);
      }
      if (prev.length >= 4) {
        return [...prev.slice(1), ep.episode_id];
      }
      return [...prev, ep.episode_id];
    });
  };

  const handleInspectEpisode = (episodeId: string) => {
    setReplayEpisodeId(episodeId);
    setActiveTab("replay");
  };

  const handleOpenComparator = (ids: string[]) => {
    setSelectedCompareIds(ids.slice(0, 4));
    setActiveTab("h2_comparison");
  };

  return (
    <div className="space-y-4">
      {/* Header Banner */}
      <div className="bg-gridcard border border-gridborder rounded-lg p-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <Database className="w-5 h-5 text-cyan-400" />
            <h2 className="text-lg font-bold text-white">
              9. Saved Episode Library &amp; H2 Comparative Study
            </h2>
          </div>
          <p className="text-xs text-slate-400 mt-1 max-w-3xl">
            Read-only research browser and 96-step replay inspector over all 900 saved benchmark
            evaluation episodes (IPPO, MAPPO Matched Multi-Seed, HAPPO, Rule-Based EMS, Greedy EMS,
            and Lookahead Heuristic EMS across Scenarios H1–H6). Live Simulation Mode on the top
            toolbar remains independent.
          </p>
        </div>

        <div className="flex flex-wrap gap-1.5">
          <button
            type="button"
            onClick={() => setActiveTab("browser")}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium border transition-colors ${
              activeTab === "browser"
                ? "bg-cyan-500/20 border-cyan-400/50 text-cyan-200"
                : "bg-gridpanel border-gridborder text-slate-300 hover:text-white"
            }`}
          >
            <Table2 className="w-3.5 h-3.5" />
            Episode Library Browser
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("h2_comparison")}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium border transition-colors ${
              activeTab === "h2_comparison"
                ? "bg-cyan-500/20 border-cyan-400/50 text-cyan-200"
                : "bg-gridpanel border-gridborder text-slate-300 hover:text-white"
            }`}
          >
            <GitCompare className="w-3.5 h-3.5" />
            H2 Comparison &amp; 18 Charts
            {selectedCompareIds.length > 0 && (
              <span className="ml-1 px-1.5 py-0.5 rounded-full bg-indigo-500/30 text-indigo-200 text-[10px]">
                {selectedCompareIds.length}
              </span>
            )}
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("replay")}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium border transition-colors ${
              activeTab === "replay"
                ? "bg-cyan-500/20 border-cyan-400/50 text-cyan-200"
                : "bg-gridpanel border-gridborder text-slate-300 hover:text-white"
            }`}
          >
            <Play className="w-3.5 h-3.5" />
            96-Step Episode Replay
          </button>
        </div>
      </div>

      {/* Tab Content */}
      {activeTab === "browser" && (
        <SavedEpisodeBrowserTab
          summary={summary}
          selectedCompareIds={selectedCompareIds}
          onToggleCompareId={handleToggleCompare}
          onInspectEpisode={handleInspectEpisode}
          onOpenComparatorWith={handleOpenComparator}
        />
      )}

      {activeTab === "h2_comparison" && (
        <H2ComparisonViewTab
          externalCompareIds={selectedCompareIds}
          onClearExternalCompareIds={() => setSelectedCompareIds([])}
          onInspectEpisode={handleInspectEpisode}
        />
      )}

      {activeTab === "replay" && (
        <EpisodeReplayViewTab
          episodeId={replayEpisodeId}
          onSelectEpisodeId={(id) => setReplayEpisodeId(id)}
        />
      )}
    </div>
  );
};
