import React, { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import { api, getSimulationWsUrl } from "../services/api";
import type {
  ControllerOption,
  CustomScenarioConditions,
  HealthResponse,
  ScenarioDescriptor,
  SimulationStateResponse,
  SimulationStepSnapshot,
} from "../types/api";

export type AudienceMode = "technical" | "client";

interface SimulationContextValue {
  audienceMode: AudienceMode;
  setAudienceMode: (mode: AudienceMode) => void;
  health: HealthResponse | null;
  scenarios: ScenarioDescriptor[];
  controllers: ControllerOption[];
  simState: SimulationStateResponse | null;
  activeSnapshot: SimulationStepSnapshot | null;
  selectedStepIndex: number | null; // null = follow latest step
  setSelectedStepIndex: (idx: number | null) => void;
  wsConnected: boolean;
  loading: boolean;
  actionError: string | null;
  clearError: () => void;
  startSim: (params: {
    controller_id: string;
    scenario_id: string;
    seed: number;
    step_delay_ms: number;
    forecast_mode?: string;
    custom_conditions?: CustomScenarioConditions;
  }) => Promise<void>;
  pauseSim: () => Promise<void>;
  resumeSim: () => Promise<void>;
  resetSim: () => Promise<void>;
  updateSpeed: (ms: number) => Promise<void>;
  refreshState: () => Promise<void>;
}

const SimulationContext = createContext<SimulationContextValue | undefined>(undefined);

export const SimulationProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [audienceMode, setAudienceMode] = useState<AudienceMode>("technical");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [scenarios, setScenarios] = useState<ScenarioDescriptor[]>([]);
  const [controllers, setControllers] = useState<ControllerOption[]>([]);
  const [simState, setSimState] = useState<SimulationStateResponse | null>(null);
  const [selectedStepIndex, setSelectedStepIndex] = useState<number | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [loading, setLoading] = useState(true);
  const [actionError, setActionError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);

  const refreshState = useCallback(async () => {
    try {
      const [h, sc, st] = await Promise.all([
        api.getHealth(),
        api.getScenarios(),
        api.getSimulationState(true),
      ]);
      setHealth(h);
      setScenarios(sc.scenarios);
      setControllers(sc.controllers);
      setSimState(st);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Failed to connect to backend API.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshState();
  }, [refreshState]);

  useEffect(() => {
    let closed = false;
    let reconnectTimer: number | undefined;

    const connect = () => {
      if (closed) return;
      try {
        const ws = new WebSocket(getSimulationWsUrl());
        wsRef.current = ws;

        ws.onopen = () => {
          setWsConnected(true);
        };

        ws.onmessage = (evt) => {
          try {
            const msg = JSON.parse(evt.data);
            if (msg.type === "init" && msg.state) {
              setSimState(msg.state as SimulationStateResponse);
            } else if (msg.type === "step" && msg.step_snapshot) {
              const snap = msg.step_snapshot as SimulationStepSnapshot;
              setSimState((prev) => {
                if (!prev) return prev;
                const existing = prev.history.some((h) => h.step === snap.step);
                const nextHist = existing ? prev.history : [...prev.history, snap];
                return {
                  ...prev,
                  status: "running",
                  current_step: msg.current_step ?? nextHist.length,
                  latest_step: snap,
                  history: nextHist,
                  summary_economics: msg.summary_economics ?? prev.summary_economics,
                };
              });
            } else if (msg.type === "status") {
              setSimState((prev) => (prev ? { ...prev, status: msg.status } : prev));
            } else if (msg.type === "completed") {
              setSimState((prev) =>
                prev
                  ? {
                      ...prev,
                      status: "completed",
                      current_step: msg.current_step ?? prev.current_step,
                      summary_economics: msg.summary_economics ?? prev.summary_economics,
                    }
                  : prev
              );
            } else if (msg.type === "reset") {
              setSelectedStepIndex(null);
              setSimState((prev) =>
                prev
                  ? {
                      ...prev,
                      status: "idle",
                      error_message: null,
                      current_step: 0,
                      latest_step: null,
                      history: [],
                      summary_economics: null,
                    }
                  : prev
              );
            } else if (msg.type === "error") {
              setSimState((prev) =>
                prev
                  ? { ...prev, status: "error", error_message: String(msg.error_message ?? "Error") }
                  : prev
              );
            }
          } catch {
            // ignore malformed frame
          }
        };

        ws.onclose = () => {
          setWsConnected(false);
          if (!closed) {
            reconnectTimer = window.setTimeout(connect, 2500);
          }
        };

        ws.onerror = () => {
          setWsConnected(false);
        };
      } catch {
        setWsConnected(false);
      }
    };

    connect();
    return () => {
      closed = true;
      if (reconnectTimer) window.clearTimeout(reconnectTimer);
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.close();
      }
    };
  }, []);

  const startSim = useCallback(
    async (params: {
      controller_id: string;
      scenario_id: string;
      seed: number;
      step_delay_ms: number;
      forecast_mode?: string;
      custom_conditions?: CustomScenarioConditions;
    }) => {
      setActionError(null);
      setSelectedStepIndex(null);
      try {
        const st = await api.startSimulation(params);
        setSimState(st);
      } catch (err) {
        setActionError(err instanceof Error ? err.message : "Failed to start simulation.");
      }
    },
    []
  );

  const pauseSim = useCallback(async () => {
    setActionError(null);
    try {
      setSimState(await api.pauseSimulation());
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Failed to pause simulation.");
    }
  }, []);

  const resumeSim = useCallback(async () => {
    setActionError(null);
    try {
      setSimState(await api.resumeSimulation());
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Failed to resume simulation.");
    }
  }, []);

  const resetSim = useCallback(async () => {
    setActionError(null);
    setSelectedStepIndex(null);
    try {
      setSimState(await api.resetSimulation());
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Failed to reset simulation.");
    }
  }, []);

  const updateSpeed = useCallback(async (ms: number) => {
    try {
      setSimState(await api.setSimulationSpeed(ms));
    } catch {
      // ignore speed update error
    }
  }, []);

  const history = simState?.history ?? [];
  const activeSnapshot =
    selectedStepIndex !== null && history[selectedStepIndex]
      ? history[selectedStepIndex]
      : simState?.latest_step ?? null;

  return (
    <SimulationContext.Provider
      value={{
        audienceMode,
        setAudienceMode,
        health,
        scenarios,
        controllers,
        simState,
        activeSnapshot,
        selectedStepIndex,
        setSelectedStepIndex,
        wsConnected,
        loading,
        actionError,
        clearError: () => setActionError(null),
        startSim,
        pauseSim,
        resumeSim,
        resetSim,
        updateSpeed,
        refreshState,
      }}
    >
      {children}
    </SimulationContext.Provider>
  );
};

export function useSimulation(): SimulationContextValue {
  const ctx = useContext(SimulationContext);
  if (!ctx) throw new Error("useSimulation must be used inside SimulationProvider");
  return ctx;
}
