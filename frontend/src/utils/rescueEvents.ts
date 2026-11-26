import type { SimulationStepSnapshot } from "../types/api";

export type RescueSeverity =
  | "warning"
  | "critical"
  | "action"
  | "rescue"
  | "success"
  | "danger";

export interface RescueEvent {
  id: string;
  step: number;
  timestamp: string;
  hour: number;
  kind:
    | "upcoming_criticality"
    | "grid_islanded"
    | "cooling_throttled"
    | "battery_discharge"
    | "generator_starting"
    | "generator_online"
    | "safety_intervention"
    | "critical_shortfall"
    | "grid_restored";
  severity: RescueSeverity;
  badge: string;
  title: string;
  description: string;
  chips: string[];
}

export interface EdgeThermalSummary {
  criticalServedPct: number;
  criticalServedMw: number;
  criticalNominalMw: number;
  coolingThrottlePct: number;
  flexShedMw: number;
  otherShedMw: number;
  totalDeferrableShedMw: number;
  thermalModeLabel: string;
  powerSourceSummary: string;
}

function getMaxForecastOutageRisk(snap: SimulationStepSnapshot): number {
  const horizons = snap.forecast?.horizons ?? [];
  let maxRisk = 0;
  for (const h of horizons) {
    const op = h.outage_prob ?? 0;
    const sr = h.storm_risk ?? 0;
    if (op > maxRisk) maxRisk = op;
    if (sr > maxRisk) maxRisk = sr;
  }
  return maxRisk;
}

export function deriveEdgeThermalStatus(
  snap: SimulationStepSnapshot | null
): EdgeThermalSummary | null {
  if (!snap) return null;
  const critNom = Math.max(0, snap.nominal_critical_mw);
  const critServed = Math.max(0, snap.served_critical_mw);
  const criticalServedPct = critNom > 1e-5 ? Math.min(100, (critServed / critNom) * 100) : 100;

  const flexNom = Math.max(0, snap.nominal_flexible_mw);
  const flexShed = Math.max(0, snap.shed_flexible_mw);
  const otherShed = Math.max(0, snap.shed_other_mw);
  const coolingThrottlePct =
    flexNom > 1e-5 ? Math.min(100, Math.round((flexShed / flexNom) * 100)) : 0;

  let thermalModeLabel = "Nominal Cooling (100% Flex)";
  if (coolingThrottlePct >= 75) {
    thermalModeLabel = `Deep Cooling Shed (-${coolingThrottlePct}% HVAC/Flex)`;
  } else if (coolingThrottlePct >= 20) {
    thermalModeLabel = `Eco Cooling Throttle (-${coolingThrottlePct}% HVAC/Flex)`;
  } else if (otherShed > 0.02) {
    thermalModeLabel = "Auxiliary Load Curtailment Active";
  }

  const activeSources: string[] = [];
  if (snap.grid_connected && snap.grid_import_mw > 0.01) {
    activeSources.push(`Grid +${snap.grid_import_mw.toFixed(2)} MW`);
  } else if (!snap.grid_connected) {
    activeSources.push("Islanded Microgrid");
  }
  if (snap.solar_mw + snap.wind_mw > 0.02) {
    activeSources.push(`Solar+Wind ${(snap.solar_mw + snap.wind_mw).toFixed(2)} MW`);
  }
  if (snap.battery_mw < -0.02) {
    activeSources.push(`Battery +${Math.abs(snap.battery_mw).toFixed(2)} MW`);
  }
  if (snap.generator_mw > 0.02) {
    activeSources.push(`Backup Diesel +${snap.generator_mw.toFixed(2)} MW`);
  } else if (snap.generator_state === "STARTING") {
    activeSources.push("Backup Diesel Warming Up");
  }

  return {
    criticalServedPct,
    criticalServedMw: critServed,
    criticalNominalMw: critNom,
    coolingThrottlePct,
    flexShedMw: flexShed,
    otherShedMw: otherShed,
    totalDeferrableShedMw: flexShed + otherShed,
    thermalModeLabel,
    powerSourceSummary: activeSources.length > 0 ? activeSources.join(" · ") : "Local Standby",
  };
}

export function deriveRescueEvents(
  history: SimulationStepSnapshot[],
  upToStep?: number
): RescueEvent[] {
  if (!history || history.length === 0) return [];
  const events: RescueEvent[] = [];

  let lastWarningStep = -99;
  let lastCoolingStep = -99;
  let lastBatteryStep = -99;
  let lastSafetyStep = -99;
  let lastCriticalRiskStep = -99;

  for (let i = 0; i < history.length; i++) {
    const s = history[i];
    if (upToStep !== undefined && s.step > upToStep) break;
    const prev = i > 0 ? history[i - 1] : undefined;

    const critPct =
      s.nominal_critical_mw > 1e-5
        ? Math.min(100, (s.served_critical_mw / s.nominal_critical_mw) * 100)
        : 100;
    const flexShedTotal = s.shed_flexible_mw + s.shed_other_mw;
    const prevFlexShedTotal = prev ? prev.shed_flexible_mw + prev.shed_other_mw : 0;
    const flexPct =
      s.nominal_flexible_mw > 1e-5
        ? Math.min(100, Math.round((s.shed_flexible_mw / s.nominal_flexible_mw) * 100))
        : 0;
    const forecastRisk = getMaxForecastOutageRisk(s);
    const prevForecastRisk = prev ? getMaxForecastOutageRisk(prev) : 0;

    // 1. Upcoming Criticality Warning (Storm, Forecast Outage Risk, or Edge Surge)
    const stormStarted = s.is_storm_warning && !(prev?.is_storm_warning ?? false);
    const forecastSpiked = forecastRisk >= 0.45 && prevForecastRisk < 0.45;
    const surgeStarted = s.nominal_total_mw >= 0.90 && (prev?.nominal_total_mw ?? 0) < 0.88;

    if ((stormStarted || forecastSpiked || surgeStarted) && s.step - lastWarningStep >= 8) {
      lastWarningStep = s.step;
      const reasonTitle = surgeStarted
        ? `Critical Edge Demand Surge (${s.nominal_total_mw.toFixed(2)} MW)`
        : stormStarted
        ? "Severe Weather / Grid Instability Warning"
        : `Upcoming Outage Risk Detected (${Math.round(forecastRisk * 100)}%)`;

      events.push({
        id: `step-${s.step}-upcoming_criticality`,
        step: s.step,
        timestamp: s.timestamp,
        hour: s.hour,
        kind: "upcoming_criticality",
        severity: "warning",
        badge: "UPCOMING CRITICALITY",
        title: reasonTitle,
        description:
          "AI Forecaster flagged incoming grid stress. Pre-staging battery reserve (>30% SOC) and preparing local backup sources to shield Critical / Edge buses.",
        chips: [
          `Demand ${s.nominal_total_mw.toFixed(2)} MW`,
          `Battery ${s.battery_soc.toFixed(0)}%`,
          `Crit Served ${critPct.toFixed(1)}%`,
        ],
      });
    }

    // 2. Utility Grid Lost -> Switching to Autonomous Islanded Microgrid
    if (!s.grid_connected && (prev?.grid_connected ?? true)) {
      events.push({
        id: `step-${s.step}-grid_islanded`,
        step: s.step,
        timestamp: s.timestamp,
        hour: s.hour,
        kind: "grid_islanded",
        severity: "critical",
        badge: "GRID LOST · RL ISLANDING",
        title: "Grid Outage — Switching to Local Islanded Microgrid",
        description:
          "Main utility feeder tripped offline. GRIDGUARD AI isolated the campus PCC breaker safely so solar, wind, battery & backup diesel carry the critical load.",
        chips: [
          `PV+Wind ${(s.solar_mw + s.wind_mw).toFixed(2)} MW`,
          `Battery ${s.battery_soc.toFixed(0)}%`,
          `Crit Protected ${critPct.toFixed(1)}%`,
        ],
      });
    }

    // 3. Reducing HVAC / Flexible Cooling Load to Protect Critical Bus
    const coolingActivated = flexShedTotal > 0.02 && prevFlexShedTotal <= 0.015;
    const coolingSteppedUp = flexShedTotal - prevFlexShedTotal >= 0.06 && s.step - lastCoolingStep >= 4;
    if ((coolingActivated || coolingSteppedUp) && s.shed_critical_mw <= 0.01) {
      lastCoolingStep = s.step;
      events.push({
        id: `step-${s.step}-cooling_throttled`,
        step: s.step,
        timestamp: s.timestamp,
        hour: s.hour,
        kind: "cooling_throttled",
        severity: "action",
        badge: "RL RESCUE · COOLING REDUCED",
        title: `Stepping Down Flexible HVAC & Cooling (-${flexShedTotal.toFixed(2)} MW)`,
        description: `Load-Management RL agent curtailed ${flexPct}% of flexible chiller/deferrable demand (HVAC equivalent) so 100% power stays locked on Critical / Edge servers.`,
        chips: [
          `Flex Shed -${flexShedTotal.toFixed(2)} MW`,
          `Crit Served ${critPct.toFixed(1)}%`,
          `V_pcc ${s.pcc_voltage_pu.toFixed(3)} pu`,
        ],
      });
    }

    // 4. Discharging Battery Reserve
    const batStartedRescue =
      s.battery_mw <= -0.05 &&
      (prev?.battery_mw ?? 0) > -0.04 &&
      (!s.grid_connected ||
        s.is_storm_warning ||
        s.nominal_total_mw > 0.78 ||
        s.solar_mw + s.wind_mw < 0.12);
    if (batStartedRescue && s.step - lastBatteryStep >= 5) {
      lastBatteryStep = s.step;
      events.push({
        id: `step-${s.step}-battery_discharge`,
        step: s.step,
        timestamp: s.timestamp,
        hour: s.hour,
        kind: "battery_discharge",
        severity: "action",
        badge: "RL RESCUE · BATTERY DISCHARGE",
        title: `Discharging Battery Storage (+${Math.abs(s.battery_mw).toFixed(2)} MW)`,
        description:
          "Battery RL agent instantaneously injects stored energy to bridge the supply gap and prevent critical frequency/voltage sag.",
        chips: [
          `Output +${Math.abs(s.battery_mw).toFixed(2)} MW`,
          `SOC ${s.battery_soc.toFixed(1)}%`,
          `Crit ${critPct.toFixed(1)}%`,
        ],
      });
    }

    // 5. Bringing Backup Diesel Generator Online (Warm-Up -> Running)
    if (s.generator_state === "STARTING" && prev?.generator_state !== "STARTING") {
      events.push({
        id: `step-${s.step}-generator_starting`,
        step: s.step,
        timestamp: s.timestamp,
        hour: s.hour,
        kind: "generator_starting",
        severity: "action",
        badge: "RL RESCUE · GENERATOR SPIN-UP",
        title: "Bringing Backup Diesel Generator Online (Warm-Up)",
        description:
          "Generator RL agent initiated cold-start spin-up (30-min physical sync delay) to supply 0.40 MW backup power; BESS covers critical loads in the meantime.",
        chips: [
          "State: STARTING",
          `Battery ${s.battery_mw < 0 ? `+${Math.abs(s.battery_mw).toFixed(2)}` : "0.00"} MW`,
          `SOC ${s.battery_soc.toFixed(0)}%`,
        ],
      });
    }

    const genBecameActive =
      s.generator_state === "RUNNING" &&
      s.generator_mw > 0.04 &&
      (prev?.generator_state !== "RUNNING" || (prev?.generator_mw ?? 0) <= 0.04);
    if (genBecameActive) {
      events.push({
        id: `step-${s.step}-generator_online`,
        step: s.step,
        timestamp: s.timestamp,
        hour: s.hour,
        kind: "generator_online",
        severity: "rescue",
        badge: "BACKUP GENERATOR ONLINE",
        title: `Backup Generator Synchronized (+${s.generator_mw.toFixed(2)} MW)`,
        description:
          "Backup diesel unit is now delivering steady power across the islanded bus, protecting Critical / Edge loads without interruption.",
        chips: [
          `Diesel +${s.generator_mw.toFixed(2)} MW`,
          `Fuel ${s.step_fuel_l.toFixed(1)} L/step`,
          `Crit ${critPct.toFixed(1)}%`,
        ],
      });
    }

    // 6. Safety Shield / Hybrid Adequacy Fallback Stepped In
    const guardNow = s.adequacy_intervention || s.shield_any_emergency;
    const guardPrev = prev ? prev.adequacy_intervention || prev.shield_any_emergency : false;
    if (guardNow && !guardPrev && !genBecameActive && s.step - lastSafetyStep >= 6) {
      lastSafetyStep = s.step;
      events.push({
        id: `step-${s.step}-safety_intervention`,
        step: s.step,
        timestamp: s.timestamp,
        hour: s.hour,
        kind: "safety_intervention",
        severity: "rescue",
        badge: s.adequacy_intervention ? "AI HYBRID GUARD" : "SAFETY SHIELD OVERRIDE",
        title: s.adequacy_intervention
          ? "Hybrid Adequacy Layer Secured Supply Margin"
          : "Safety Shield Enforced Physical Protection",
        description: s.adequacy_intervention
          ? "Supervisory adequacy layer committed emergency battery/generator headroom automatically to keep critical reliability near 100%."
          : "Physical SafetyShield blocked an unsafe breaker or deep-discharge action before any grid fault could escalate.",
        chips: [
          `Crit Rel ${s.critical_reliability_pct_cum.toFixed(1)}%`,
          `V_pcc ${s.pcc_voltage_pu.toFixed(3)} pu`,
        ],
      });
    }

    // 7. Honest Critical Unserved Alert (if critical load actually shed)
    if (
      s.shed_critical_mw > 0.005 &&
      (prev?.shed_critical_mw ?? 0) <= 0.005 &&
      s.step - lastCriticalRiskStep >= 6
    ) {
      lastCriticalRiskStep = s.step;
      events.push({
        id: `step-${s.step}-critical_shortfall`,
        step: s.step,
        timestamp: s.timestamp,
        hour: s.hour,
        kind: "critical_shortfall",
        severity: "danger",
        badge: "CRITICAL LOAD SHORTFALL",
        title: `Critical Bus Under-Served (-${s.shed_critical_mw.toFixed(3)} MW)`,
        description:
          "Active controller could not cover full critical demand at this step — compare against the Recommended AI Hybrid Controller to see autonomous rescue.",
        chips: [
          `Shed -${s.shed_critical_mw.toFixed(3)} MW`,
          `Rel ${s.critical_reliability_pct_cum.toFixed(1)}%`,
        ],
      });
    }

    // 8. Utility Grid Restored -> Day Saved
    if (s.grid_connected && prev !== undefined && !prev.grid_connected) {
      events.push({
        id: `step-${s.step}-grid_restored`,
        step: s.step,
        timestamp: s.timestamp,
        hour: s.hour,
        kind: "grid_restored",
        severity: "success",
        badge: "GRID RESTORED · RESCUE COMPLETE",
        title: `Utility Feeder Reconnected (${s.critical_reliability_pct_cum.toFixed(1)}% Critical Served)`,
        description:
          "Phase angle synchronization verified. GRIDGUARD AI smoothly re-tied to the main utility grid and begins ramping backup diesel down.",
        chips: [
          `Crit Rel ${s.critical_reliability_pct_cum.toFixed(1)}%`,
          `Grid ${s.grid_import_mw.toFixed(2)} MW`,
          `V_pcc ${s.pcc_voltage_pu.toFixed(3)} pu`,
        ],
      });
    }
  }

  return events;
}
