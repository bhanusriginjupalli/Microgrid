export interface ProvenanceInfo {
  checkpoint_path: string | null;
  experiment: string | null;
  seed: number | null;
  episode: number | null;
  val_score: number | null;
  forecast_mode: string | null;
  verified: boolean;
  source_file: string | null;
  notes: string | null;
}

export interface HealthResponse {
  status: string;
  app_name: string;
  version: string;
  disclaimer: string;
  artifacts_available: Record<string, boolean>;
  missing_artifacts: string[];
  simulation_status: string;
  active_scenario: string | null;
  active_controller: string | null;
  provenance: ProvenanceInfo | null;
}

export interface CustomScenarioConditions {
  solar_cloud_severity: number;
  wind_speed_ms?: number | null;
  critical_load_multiplier: number;
  flexible_load_multiplier: number;
  grid_outage_enabled: boolean;
  outage_start_hour: number;
  outage_end_hour: number;
}

export interface SystemStatusResponse {
  disclaimer: string;
  simulation_status: "idle" | "running" | "paused" | "completed" | "error";
  active_scenario: string | null;
  active_scenario_name: string | null;
  active_controller: string | null;
  controller_mode: string | null;
  eval_seed: number;
  current_step: number;
  max_steps: number;
  current_hour: number;
  solar_mw: number | null;
  wind_mw: number | null;
  total_generation_mw: number | null;
  total_demand_mw: number | null;
  battery_soc_pct: number | null;
  battery_power_mw: number | null;
  grid_connected: boolean | null;
  grid_available: boolean | null;
  grid_import_mw: number | null;
  grid_export_mw: number | null;
  generator_mw: number | null;
  generator_state: string | null;
  critical_served_mw: number | null;
  critical_reliability_pct: number | null;
  critical_ens_mwh: number | null;
  operating_cost_usd: number | null;
  true_social_cost_usd: number | null;
  alerts: string[];
  provenance: ProvenanceInfo | null;
}

export interface BusTelemetry {
  bus_idx: number;
  name: string;
  role: string;
  vn_kv: number;
  vm_pu: number | null;
  va_degree: number | null;
  p_mw: number | null;
  q_mvar: number | null;
  in_service: boolean;
  is_simulated: boolean;
}

export interface LineTelemetry {
  line_idx: number;
  name: string;
  from_bus: number;
  to_bus: number;
  from_bus_name: string;
  to_bus_name: string;
  p_from_mw: number | null;
  p_to_mw: number | null;
  loading_percent: number | null;
  in_service: boolean;
  power_direction: "forward" | "reverse" | "idle" | "disconnected";
}

export interface NetworkTopologyResponse {
  disclaimer: string;
  is_simulated: boolean;
  data_source: string;
  step: number;
  hour: number;
  converged: boolean | null;
  grid_connected: boolean | null;
  grid_available: boolean | null;
  sync_verified: boolean | null;
  buses: BusTelemetry[];
  lines: LineTelemetry[];
  asset_summary: Record<string, number | string>;
}

export interface DisturbanceEvent {
  type: string;
  start_hour: number;
  end_hour: number;
  details: Record<string, unknown>;
}

export interface ScenarioDescriptor {
  scenario_id: string;
  category: string;
  name: string;
  description: string;
  events: DisturbanceEvent[];
  has_outage: boolean;
  has_storm: boolean;
  has_topology_shift: boolean;
  has_sensor_degradation: boolean;
  has_stale_forecast: boolean;
}

export interface ControllerOption {
  controller_id: string;
  label: string;
  controller_type: string;
  mode: string;
  adequacy_fallback: boolean;
  checkpoint_path: string | null;
  forecast_mode: string | null;
  description: string;
  is_primary_candidate: boolean;
}

export interface ScenariosResponse {
  disclaimer: string;
  scenarios: ScenarioDescriptor[];
  controllers: ControllerOption[];
}

export interface ForecastHorizonPoint {
  horizon_step: number;
  horizon_minutes: number;
  target_hour: number;
  solar_q10: number | null;
  solar_q50: number | null;
  solar_q90: number | null;
  wind_q10: number | null;
  wind_q50: number | null;
  wind_q90: number | null;
  demand_q10: number | null;
  demand_q50: number | null;
  demand_q90: number | null;
  net_load_q10: number | null;
  net_load_q50: number | null;
  net_load_q90: number | null;
  outage_prob: number | null;
  storm_risk: number | null;
}

export interface ForecastSnapshot {
  mode: string;
  current_hour: number;
  actual_solar_mw: number;
  actual_wind_mw: number;
  actual_demand_mw: number;
  actual_net_load_mw: number;
  horizons: ForecastHorizonPoint[];
}

export interface AgentDecisionRecord {
  controller: string;
  mode: string;
  scenario: string;
  eval_seed: number;
  step: number;
  hour: number;
  timestamp: string;
  agent: string;
  relevant_observations: Record<string, number>;
  raw_policy_output: unknown;
  decoded_action: Record<string, unknown>;
  shielded_action: Record<string, unknown>;
  executed_action: Record<string, unknown>;
  shield_modified: boolean;
  fallback_modified: boolean;
  intervention_stage: string;
  operational_constraints: Record<string, unknown>;
  reward_total: number | null;
  agent_reward_components: Record<string, number>;
  explanation: string;
}

export interface SimulationStepSnapshot {
  step: number;
  hour: number;
  timestamp: string;
  solar_mw: number;
  wind_mw: number;
  solar_available_mw: number;
  wind_available_mw: number;
  total_generation_mw: number;
  nominal_critical_mw: number;
  nominal_flexible_mw: number;
  nominal_other_mw: number;
  nominal_total_mw: number;
  served_critical_mw: number;
  served_flexible_mw: number;
  served_other_mw: number;
  shed_critical_mw: number;
  shed_flexible_mw: number;
  shed_other_mw: number;
  battery_mw: number;
  battery_soc: number;
  generator_mw: number;
  generator_state: string;
  step_fuel_l: number;
  total_fuel_l: number;
  grid_connected: boolean;
  grid_available: boolean;
  sync_verified: boolean;
  grid_import_mw: number;
  grid_export_mw: number;
  pcc_voltage_pu: number;
  converged: boolean;
  balance_error_mw: number;
  is_storm_warning: boolean;
  shared_reward: number;
  cumulative_reward: number;
  reward_breakdown: Record<string, number>;
  operating_cost_usd_cum: number;
  true_social_cost_usd_cum: number;
  critical_reliability_pct_cum: number;
  critical_ens_mwh_cum: number;
  shield_any_intervention: boolean;
  shield_any_emergency: boolean;
  shield_any_boundary: boolean;
  adequacy_intervention: boolean;
  inaction_detected: boolean;
  agent_decisions: AgentDecisionRecord[];
  interventions: Record<string, unknown>[];
  forecast: ForecastSnapshot | null;
  network: NetworkTopologyResponse | null;
  alerts: string[];
}

export interface SimulationStateResponse {
  disclaimer: string;
  status: "idle" | "running" | "paused" | "completed" | "error";
  error_message: string | null;
  controller_id: string | null;
  controller_label: string | null;
  controller_mode: string | null;
  scenario_id: string | null;
  scenario_name: string | null;
  seed: number;
  step_delay_ms: number;
  forecast_mode: string | null;
  current_step: number;
  max_steps: number;
  latest_step: SimulationStepSnapshot | null;
  history: SimulationStepSnapshot[];
  summary_economics: Record<string, unknown> | null;
}

export interface ControllerComparisonRow {
  controller: string;
  controller_group: string | null;
  ablation: string | null;
  train_seed: string | null;
  adequacy_fallback: boolean | null;
  mode: string;
  episodes_n: number | null;
  true_social_cost_mean: number | null;
  true_social_cost_std: number | null;
  true_social_cost_cvar10: number | null;
  operating_cost_mean: number | null;
  critical_ens_mean: number | null;
  critical_reliability_mean: number | null;
  flexible_ens_mean: number | null;
  other_ens_mean: number | null;
  fuel_consumed_l_mean: number | null;
  battery_throughput_mean: number | null;
  re_curtailment_mwh_mean: number | null;
  unshielded_feasible_pct: number | null;
  boundary_clipping_mean: number | null;
  emergency_override_mean: number | null;
  adequacy_intervention_mean: number | null;
  power_flow_converged_pct: number | null;
}

export interface AnalyticsSummaryResponse {
  disclaimer: string;
  available: boolean;
  primary_candidate: ProvenanceInfo;
  key_conclusions: string[];
  phase45_means_by_controller: Record<string, unknown>[];
  phase444_highlights: ControllerComparisonRow[];
  fallback_attribution_rows: Record<string, unknown>[];
  warnings: string[];
}

export interface AnalyticsComparisonResponse {
  disclaimer: string;
  available: boolean;
  source_phase444: string;
  source_phase45: string;
  phase444_controllers: ControllerComparisonRow[];
  phase45_traces: Record<string, unknown>[];
  autonomy_metrics: Record<string, unknown>[];
  warnings: string[];
}

export interface SafetyInterventionRow {
  controller: string;
  mode: string;
  scenario: string;
  eval_seed: number;
  step: number;
  hour: number;
  timestamp: string;
  agent: string;
  intervention_stage: string;
  triggering_condition: string;
  reason: string;
  prevented_violation: boolean;
  operational_effect: string;
}

export interface ExplainabilityDecisionsResponse {
  disclaimer: string;
  available: boolean;
  provenance: ProvenanceInfo | null;
  scenario: string;
  controllers: string[];
  decisions: AgentDecisionRecord[];
  h2_timeline: Record<string, unknown>[];
  reward_decomposition_summary: Record<string, unknown>[];
  safety_interventions: SafetyInterventionRow[];
  intervention_counts_summary: Record<string, unknown>[];
  warnings: string[];
}

export interface ExplainabilityAttributionsResponse {
  disclaimer: string;
  available: boolean;
  sensitivity_vs_causality_note: string;
  agents: string[];
  aggregated_by_agent: Record<string, Record<string, unknown>[]>;
  controlled_perturbation_validation: Record<string, unknown>[];
  max_ig_completeness_error: number | null;
  warnings: string[];
}

export interface ExplainabilityCounterfactualsResponse {
  disclaimer: string;
  available: boolean;
  scenarios: string[];
  modes: string[];
  records: Record<string, unknown>[];
  warnings: string[];
}

export interface ExperimentSummary {
  experiment_id: string;
  phase: string;
  title: string;
  description: string;
  available: boolean;
  source_path: string;
  row_count: number;
  controllers_count: number;
  separates_intrinsic_and_hybrid: boolean;
  warnings: string[];
}

export interface EpisodeIndexItem {
  episode_id: string;
  schema_version: string;
  experiment_id: string;
  algorithm: string;
  controller_name: string;
  controller_group: string;
  track: string;
  mode: string;
  scenario_id: string;
  training_seed: number | null;
  evaluation_seed: number;
  selected_checkpoint_episode: number | null;
  episode_length: number;
  timestep_minutes: number;
  start_timestamp: string | null;
  completion_status: string;
  enable_safety_shield: boolean;
  adequacy_fallback: boolean;
  checkpoint_path: string | null;
  trace_path: string;
  summary_path: string;
  reward: number;
  true_social_cost_usd: number;
  operating_cost_usd: number;
  grid_import_cost_usd: number;
  grid_export_revenue_usd: number;
  fuel_consumed_l: number;
  fuel_cost_usd: number;
  generator_operating_hours: number;
  generator_startup_count: number;
  startup_wear_cost_usd: number;
  critical_ens_mwh: number;
  flexible_ens_mwh: number;
  other_ens_mwh: number;
  total_ens_mwh: number;
  critical_reliability_pct: number;
  total_reliability_pct: number;
  renewable_curtailment_mwh: number;
  renewable_utilization_pct: number;
  battery_throughput_mwh: number;
  battery_degradation_cost_usd: number;
  terminal_battery_soc_pct: number;
  voltage_violation_steps: number;
  power_flow_failures: number;
  power_flow_converged_pct: number;
  unshielded_feasible_pct: number;
  boundary_clipping_count: number;
  emergency_override_count: number;
  supply_adequacy_intervention_count: number;
  fallback_intervention_steps: number;
}

export interface EpisodeListResponse {
  disclaimer: string;
  schema_version: string;
  total: number;
  offset: number;
  limit: number;
  episodes: EpisodeIndexItem[];
}

export interface EpisodeDetailResponse {
  disclaimer: string;
  episode_id: string;
  schema_version: string;
  metadata: Record<string, unknown>;
  metrics: Record<string, number | string | null>;
}

export interface EpisodeTraceStep {
  step_idx: number;
  hour: number;
  timestamp_label: string;
  observations: Record<string, number[]>;
  proposed_actions: Record<string, unknown>;
  executed_actions: Record<string, unknown> & {
    applied_battery_mw?: number;
    applied_generator_mw?: number;
  };
  solar_available_mw: number;
  solar_mw: number;
  wind_available_mw: number;
  wind_mw: number;
  renewable_curtailment_mw: number;
  battery_soc_pct: number;
  battery_mw: number;
  generator_state: string;
  generator_mw: number;
  step_fuel_l: number;
  grid_connected: boolean;
  grid_available: boolean;
  grid_import_mw: number;
  nominal_critical_mw: number;
  nominal_flexible_mw: number;
  nominal_other_mw: number;
  served_critical_mw: number;
  served_flexible_mw: number;
  served_other_mw: number;
  shed_critical_mw: number;
  shed_flexible_mw: number;
  shed_other_mw: number;
  voltages_pu: {
    pcc: number;
    min_bus: number;
    max_bus: number;
  };
  reward: number;
  reward_breakdown: Record<string, number>;
  step_operating_cost_usd: number;
  step_social_cost_usd: number;
  shield_events: {
    any_intervention: boolean;
    any_boundary: boolean;
    any_emergency: boolean;
    violations: string[];
    reasons: string[];
    interventions_by_agent: Record<string, boolean>;
  };
  fallback_events: {
    adequacy_fallback_enabled: boolean;
    inaction_detected: boolean;
    adequacy_intervention: boolean;
    avoidable_shortfall_mw: number;
    reasons: string[];
  };
  scenario_events: string[];
  converged: boolean;
}

export interface EpisodeTraceResponse {
  disclaimer: string;
  episode_id: string;
  schema_version: string;
  metadata: Record<string, unknown>;
  metrics: Record<string, number | string | null>;
  steps: EpisodeTraceStep[];
}

export interface EpisodeCompareSummaryRow {
  episode_id: string;
  algorithm: string;
  controller_name: string;
  track: string;
  mode: string;
  training_seed: number | null;
  evaluation_seed: number;
  scenario_id: string;
  true_social_cost_usd: number;
  operating_cost_usd: number;
  critical_ens_mwh: number;
  critical_reliability_pct: number;
  renewable_curtailment_mwh: number;
  fuel_consumed_l: number;
  battery_throughput_mwh: number;
  terminal_battery_soc_pct: number;
  boundary_clipping_count: number;
  emergency_override_count: number;
  fallback_intervention_steps: number;
  delta_social_cost_vs_ref_usd: number;
  delta_critical_ens_vs_ref_mwh: number;
  delta_critical_rel_vs_ref_pct: number;
}

export interface EpisodeCompareTimestep {
  step_idx: number;
  hour: number;
  timestamp_label: string;
  scenario_events: string[];
  values: Record<string, Record<string, number | string | boolean>>;
  deltas_vs_reference: Record<string, Record<string, number>>;
}

export interface EpisodeCompareResponse {
  disclaimer: string;
  reference_episode_id: string;
  scenario_id: string;
  evaluation_seed: number;
  aligned_step_count: number;
  episodes_compared: EpisodeCompareSummaryRow[];
  aligned_timesteps: EpisodeCompareTimestep[];
}

export interface H2MatrixRepresentative {
  episode_id: string;
  controller_name: string;
  training_seed: number | null;
  evaluation_seed: number;
  true_social_cost_usd: number;
  critical_ens_mwh: number;
  critical_reliability_pct: number;
  fallback_intervention_steps: number;
  criterion: string;
  warning: string;
}

export interface H2MatrixCell {
  available: boolean;
  reason?: string;
  mode?: string;
  episodes_n?: number;
  mean_true_social_cost_usd?: number;
  mean_operating_cost_usd?: number;
  mean_critical_ens_mwh?: number;
  mean_critical_reliability_pct?: number;
  mean_fallback_interventions?: number;
  representatives: Record<string, H2MatrixRepresentative>;
  episodes: EpisodeIndexItem[];
}

export interface H2ComparisonDataset {
  scenario_id: string;
  scenario_title: string;
  algorithms_included: string[];
  tracks_included: string[];
  total_h2_episodes: number;
  matrix: Record<string, Record<string, H2MatrixCell>>;
  matched_seed_pairs: Array<{
    evaluation_seed: number;
    episodes_by_track_algorithm: Record<string, string[] | null>;
  }>;
}

export interface ScenarioEpisodesResponse {
  disclaimer: string;
  scenario_id: string;
  total: number;
  episodes: EpisodeIndexItem[];
  h2_comparison: H2ComparisonDataset | null;
}

export interface EpisodeLibrarySummaryResponse {
  disclaimer: string;
  schema_version: string;
  total_episodes: number;
  tracks: string[];
  algorithms: string[];
  scenarios: string[];
  evaluation_seeds: number[];
  group_summaries: Array<{
    track: string;
    mode: string;
    algorithm: string;
    episodes_n: number;
    true_social_cost_mean: number;
    operating_cost_mean: number;
    critical_ens_mean: number;
    critical_reliability_mean: number;
    fallback_interventions_mean: number;
  }>;
  manifest_integrity: Record<string, unknown>;
}
