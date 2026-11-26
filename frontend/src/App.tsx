import React from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { SimulationProvider } from "./context/SimulationContext";
import { DashboardShell } from "./layouts/DashboardShell";
import { SystemOverviewPage } from "./pages/SystemOverviewPage";
import { DigitalTwinPage } from "./pages/DigitalTwinPage";
import { EnergyManagementPage } from "./pages/EnergyManagementPage";
import { MultiAgentControlPage } from "./pages/MultiAgentControlPage";
import { ForecastingPage } from "./pages/ForecastingPage";
import { ResilienceSimulationPage } from "./pages/ResilienceSimulationPage";
import { ExplainableAIPage } from "./pages/ExplainableAIPage";
import { PerformanceAnalyticsPage } from "./pages/PerformanceAnalyticsPage";
import { SavedEpisodeLibraryPage } from "./pages/SavedEpisodeLibraryPage";

export const App: React.FC = () => {
  return (
    <SimulationProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<DashboardShell />}>
            <Route path="/" element={<SystemOverviewPage />} />
            <Route path="/digital-twin" element={<DigitalTwinPage />} />
            <Route path="/energy-management" element={<EnergyManagementPage />} />
            <Route path="/multi-agent-control" element={<MultiAgentControlPage />} />
            <Route path="/forecasting" element={<ForecastingPage />} />
            <Route path="/resilience-simulation" element={<ResilienceSimulationPage />} />
            <Route path="/explainable-ai" element={<ExplainableAIPage />} />
            <Route path="/performance-analytics" element={<PerformanceAnalyticsPage />} />
            <Route path="/saved-episodes" element={<SavedEpisodeLibraryPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </SimulationProvider>
  );
};

export default App;
