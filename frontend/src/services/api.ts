import type {
  AnalyticsComparisonResponse,
  AnalyticsSummaryResponse,
  CustomScenarioConditions,
  EpisodeCompareResponse,
  EpisodeDetailResponse,
  EpisodeLibrarySummaryResponse,
  EpisodeListResponse,
  EpisodeTraceResponse,
  ExperimentSummary,
  ExplainabilityAttributionsResponse,
  ExplainabilityCounterfactualsResponse,
  ExplainabilityDecisionsResponse,
  HealthResponse,
  NetworkTopologyResponse,
  ScenarioEpisodesResponse,
  ScenariosResponse,
  SimulationStateResponse,
  SystemStatusResponse,
} from "../types/api";

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) msg = String(body.detail);
    } catch {
      // ignore non-json body
    }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export const api = {
  getHealth: () => fetchJson<HealthResponse>("/api/health"),
  getSystemStatus: () => fetchJson<SystemStatusResponse>("/api/system/status"),
  getNetwork: () => fetchJson<NetworkTopologyResponse>("/api/system/network"),
  getScenarios: () => fetchJson<ScenariosResponse>("/api/scenarios"),
  getExperiments: () =>
    fetchJson<{ disclaimer: string; experiments: ExperimentSummary[] }>("/api/experiments"),
  getAnalyticsSummary: () => fetchJson<AnalyticsSummaryResponse>("/api/analytics/summary"),
  getAnalyticsComparison: (params?: { ablation?: string; mode?: string }) => {
    const qs = new URLSearchParams();
    if (params?.ablation) qs.set("ablation", params.ablation);
    if (params?.mode) qs.set("mode", params.mode);
    const suffix = qs.toString() ? `?${qs.toString()}` : "";
    return fetchJson<AnalyticsComparisonResponse>(`/api/analytics/comparison${suffix}`);
  },
  getExplainabilityDecisions: (params?: {
    controller?: string;
    agent?: string;
    step_min?: number;
    step_max?: number;
    limit?: number;
  }) => {
    const qs = new URLSearchParams();
    if (params?.controller) qs.set("controller", params.controller);
    if (params?.agent) qs.set("agent", params.agent);
    if (params?.step_min !== undefined) qs.set("step_min", String(params.step_min));
    if (params?.step_max !== undefined) qs.set("step_max", String(params.step_max));
    if (params?.limit !== undefined) qs.set("limit", String(params.limit));
    const suffix = qs.toString() ? `?${qs.toString()}` : "";
    return fetchJson<ExplainabilityDecisionsResponse>(`/api/explainability/decisions${suffix}`);
  },
  getExplainabilityAttributions: () =>
    fetchJson<ExplainabilityAttributionsResponse>("/api/explainability/attributions"),
  getExplainabilityCounterfactuals: (params?: { scenario?: string; mode?: string }) => {
    const qs = new URLSearchParams();
    if (params?.scenario) qs.set("scenario", params.scenario);
    if (params?.mode) qs.set("mode", params.mode);
    const suffix = qs.toString() ? `?${qs.toString()}` : "";
    return fetchJson<ExplainabilityCounterfactualsResponse>(`/api/explainability/counterfactuals${suffix}`);
  },
  getSimulationState: (includeHistory = true) =>
    fetchJson<SimulationStateResponse>(`/api/simulation/state?include_history=${includeHistory}`),
  startSimulation: (payload: {
    controller_id: string;
    scenario_id: string;
    seed: number;
    step_delay_ms: number;
    forecast_mode?: string;
    custom_conditions?: CustomScenarioConditions;
  }) =>
    fetchJson<SimulationStateResponse>("/api/simulation/start", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  pauseSimulation: () =>
    fetchJson<SimulationStateResponse>("/api/simulation/pause", { method: "POST" }),
  resumeSimulation: () =>
    fetchJson<SimulationStateResponse>("/api/simulation/resume", { method: "POST" }),
  resetSimulation: () =>
    fetchJson<SimulationStateResponse>("/api/simulation/reset", { method: "POST" }),
  setSimulationSpeed: (step_delay_ms: number) =>
    fetchJson<SimulationStateResponse>("/api/simulation/speed", {
      method: "POST",
      body: JSON.stringify({ step_delay_ms }),
    }),
  getEpisodeLibrarySummary: (signal?: AbortSignal) =>
    fetchJson<EpisodeLibrarySummaryResponse>("/api/episodes/summary", { signal }),
  getSavedEpisodes: (
    params: {
      algorithm?: string;
      track?: string;
      mode?: string;
      scenario_id?: string;
      training_seed?: number;
      evaluation_seed?: number;
      social_cost_min?: number;
      social_cost_max?: number;
      critical_ens_min?: number;
      critical_ens_max?: number;
      critical_reliability_min?: number;
      critical_reliability_max?: number;
      completion_status?: string;
      search?: string;
      sort_by?: string;
      descending?: boolean;
      offset?: number;
      limit?: number;
    },
    signal?: AbortSignal
  ) => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== "") {
        qs.set(key, String(value));
      }
    });
    const suffix = qs.toString() ? `?${qs.toString()}` : "";
    return fetchJson<EpisodeListResponse>(`/api/episodes${suffix}`, { signal });
  },
  getSavedEpisodeDetail: (episodeId: string, signal?: AbortSignal) =>
    fetchJson<EpisodeDetailResponse>(
      `/api/episodes/${encodeURIComponent(episodeId)}`,
      { signal }
    ),
  getSavedEpisodeTrace: (episodeId: string, signal?: AbortSignal) =>
    fetchJson<EpisodeTraceResponse>(
      `/api/episodes/${encodeURIComponent(episodeId)}/trace`,
      { signal }
    ),
  getScenarioEpisodes: (
    scenarioId: string,
    params?: { track?: string; algorithm?: string },
    signal?: AbortSignal
  ) => {
    const qs = new URLSearchParams();
    if (params?.track) qs.set("track", params.track);
    if (params?.algorithm) qs.set("algorithm", params.algorithm);
    const suffix = qs.toString() ? `?${qs.toString()}` : "";
    return fetchJson<ScenarioEpisodesResponse>(
      `/api/episodes/scenarios/${encodeURIComponent(scenarioId)}${suffix}`,
      { signal }
    );
  },
  compareSavedEpisodes: (
    episodeIds: string[],
    options?: {
      require_same_scenario?: boolean;
      require_same_eval_seed?: boolean;
    },
    signal?: AbortSignal
  ) => {
    const qs = new URLSearchParams();
    qs.set("episode_ids", episodeIds.join(","));
    if (options?.require_same_scenario !== undefined) {
      qs.set("require_same_scenario", String(options.require_same_scenario));
    }
    if (options?.require_same_eval_seed !== undefined) {
      qs.set("require_same_eval_seed", String(options.require_same_eval_seed));
    }
    return fetchJson<EpisodeCompareResponse>(
      `/api/episodes/compare?${qs.toString()}`,
      { signal }
    );
  },
};

export function getSimulationWsUrl(): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/simulation`;
}
