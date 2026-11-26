"""
experiments/phase4/robustness/generate_robustness_plots.py
============================================================
GRIDGUARD AI - Phase 4.3: Diagnostic & Benchmark Visualization Engine

Generates 17 publication-grade figures evaluating controller robustness
under unseen domain shifts, syncing all figures to the Brain Artifacts directory:
  Plot 1:  Reward Distribution Under Unseen Scenarios (Box/Violin)
  Plot 2:  Operating Cost Distribution Under Domain Shifts
  Plot 3:  Critical Energy Not Served (ENS) Across Controllers
  Plot 4:  Critical Power Reliability (%) Comparison
  Plot 5:  Worst-Case Controller Performance (Min Reward)
  Plot 6:  Conditional Value at Risk (CVaR 5%) Resilience Ranking
  Plot 7:  Performance Degradation vs. Disturbance Severity
  Plot 8:  Network Topology Fault Recovery Trajectories (Scenario H1)
  Plot 9:  Battery SOC Trajectory Under Extended Renewable Drought (Scenario H2)
  Plot 10: Generator Dispatch Under Prolonged Grid Outage (Scenario H6)
  Plot 11: Load Shedding & Demand Breakdown Under Surges (Scenario H3)
  Plot 12: Robustness Under Sensor Noise & Dropout (Scenario H4)
  Plot 13: Robustness Under Stale Forecast Delays (Scenario H5)
  Plot 14: Safety Shield Interventions: Emergency vs Boundary Reserve Clipping
  Plot 15: Robustness Ablation Study (DR vs No-Noise vs No-Uncertainty)
  Plot 16: Multi-Seed Training Convergence Curves (Seeds 42, 101, 202)
  Plot 17: Post-Disturbance Microgrid Recovery Time (Minutes)
============================================================
"""

import sys
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

ARTIFACT_DIR = Path("C:/Users/Akshita/.gemini/antigravity/brain/c7894dc8-eb09-414e-a3d6-aa6f8e97ba7f")
PLOTS_DIR = ROOT / "experiments" / "phase4" / "robustness" / "plots"
RESULTS_DIR = ROOT / "experiments" / "phase4" / "robustness" / "results"
TRAJ_DIR = RESULTS_DIR / "trajectories"
LOGS_DIR = ROOT / "experiments" / "phase4" / "robustness" / "logs"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Set high-quality styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
    "figure.dpi": 200,
})

def sync_artifact(plot_name: str):
    src = PLOTS_DIR / plot_name
    dst = ARTIFACT_DIR / plot_name
    shutil.copy2(src, dst)
    print(f"  [+] Synced: {plot_name} -> Brain Artifacts")


def compute_cvar(values: np.ndarray, alpha: float = 0.05, higher_is_better: bool = True) -> float:
    """Compute Conditional Value at Risk for the worst alpha percentile."""
    arr = np.sort(values)
    if higher_is_better:
        cutoff_idx = max(1, int(np.ceil(alpha * len(arr))))
        return float(np.mean(arr[:cutoff_idx]))
    else:
        cutoff_idx = max(1, int(np.ceil((1.0 - alpha) * len(arr))))
        return float(np.mean(arr[cutoff_idx:]))


def generate_all_plots():
    print("=" * 70)
    print("Generating GRIDGUARD AI Phase 4.3 Robustness Diagnostic Figures...")
    print("=" * 70)

    raw_df = pd.read_csv(RESULTS_DIR / "robustness_raw_benchmark_records.csv")
    summary_df = pd.read_csv(RESULTS_DIR / "robustness_benchmark_summary.csv")

    # Filter main controllers for cleaner comparative charts
    main_controllers = [
        "Rule-Based EMS",
        "Greedy Controller",
        "Lookahead MPC",
        "Reactive MAPPO (Phase 4.1)",
        "Forecast-Aware MAPPO (Phase 4.2)",
        "Robust Forecast MAPPO (Seed 42)",
    ]

    palette = {
        "Rule-Based EMS": "#2ca02c",
        "Greedy Controller": "#d62728",
        "Lookahead MPC": "#9467bd",
        "Reactive MAPPO (Phase 4.1)": "#ff7f0e",
        "Forecast-Aware MAPPO (Phase 4.2)": "#1f77b4",
        "Robust Forecast MAPPO (Seed 42)": "#008080",
        "Oracle Forecast MAPPO (Ref)": "#7f7f7f",
    }

    scen_labels = {
        "scenario_h1": "H1: Topology Fault",
        "scenario_h2": "H2: RE Drought (18h)",
        "scenario_h3": "H3: Demand Surge (+70%)",
        "scenario_h4": "H4: Sensor Noise",
        "scenario_h5": "H5: Stale Forecast",
        "scenario_h6": "H6: Compound Fault",
    }

    # -------------------------------------------------------------------------
    # Plot 1: Reward Distribution Under Unseen Scenarios
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.5))
    plot_df = raw_df[raw_df["controller"].isin(main_controllers)].copy()
    plot_df["scenario_label"] = plot_df["scenario"].map(scen_labels)

    scens = list(scen_labels.values())
    x = np.arange(len(scens))
    width = 0.14
    for i, ctrl in enumerate(main_controllers):
        means = []
        stds = []
        for s_code, s_name in scen_labels.items():
            sub = raw_df[(raw_df["controller"] == ctrl) & (raw_df["scenario"] == s_code)]
            means.append(sub["reward"].mean())
            stds.append(sub["reward"].std())
        ax.bar(x + (i - 2.5) * width, means, width, yerr=stds, capsize=3,
               label=ctrl, color=palette.get(ctrl, "#333333"), alpha=0.9, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(scens, rotation=15, ha="right")
    ax.set_ylabel("Episode Reward (Higher is Better)")
    ax.set_title("Plot 1: Episode Reward Distribution Across Unseen Held-Out Scenarios")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot1_reward_distribution_unseen_scenarios.png")
    plt.close()
    sync_artifact("plot1_reward_distribution_unseen_scenarios.png")

    # -------------------------------------------------------------------------
    # Plot 2: Operating Cost Distribution
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, ctrl in enumerate(main_controllers):
        means = []
        for s_code in scen_labels:
            sub = raw_df[(raw_df["controller"] == ctrl) & (raw_df["scenario"] == s_code)]
            means.append(sub["operating_cost_usd"].mean())
        ax.bar(x + (i - 2.5) * width, means, width, label=ctrl,
               color=palette.get(ctrl, "#333333"), alpha=0.9, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(scens, rotation=15, ha="right")
    ax.set_ylabel("Operating Cost [USD] (Lower is Better)")
    ax.set_title("Plot 2: Daily Microgrid Operating Cost Under Unseen Domain Shifts")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot2_operating_cost_distribution.png")
    plt.close()
    sync_artifact("plot2_operating_cost_distribution.png")

    # -------------------------------------------------------------------------
    # Plot 3: Critical Energy Not Served (ENS)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, ctrl in enumerate(main_controllers):
        means = []
        for s_code in scen_labels:
            sub = raw_df[(raw_df["controller"] == ctrl) & (raw_df["scenario"] == s_code)]
            means.append(sub["critical_ens_mwh"].mean())
        ax.bar(x + (i - 2.5) * width, means, width, label=ctrl,
               color=palette.get(ctrl, "#333333"), alpha=0.9, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(scens, rotation=15, ha="right")
    ax.set_ylabel("Critical Energy Not Served [MWh] (Zero is Ideal)")
    ax.set_title("Plot 3: Critical Load Shedding (Critical ENS) Under Unseen Disturbances")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot3_critical_ens_comparison.png")
    plt.close()
    sync_artifact("plot3_critical_ens_comparison.png")

    # -------------------------------------------------------------------------
    # Plot 4: Critical Reliability (%) Comparison
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, ctrl in enumerate(main_controllers):
        means = []
        for s_code in scen_labels:
            sub = raw_df[(raw_df["controller"] == ctrl) & (raw_df["scenario"] == s_code)]
            means.append(sub["critical_reliability_pct"].mean())
        ax.bar(x + (i - 2.5) * width, means, width, label=ctrl,
               color=palette.get(ctrl, "#333333"), alpha=0.9, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(scens, rotation=15, ha="right")
    ax.set_ylabel("Critical Reliability [%] (100% Target)")
    ax.set_ylim(70, 103)
    ax.axhline(100.0, color="red", linestyle=":", label="100% Reliability Target")
    ax.set_title("Plot 4: Critical Power Reliability (%) Across Held-Out Scenarios")
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot4_critical_reliability_comparison.png")
    plt.close()
    sync_artifact("plot4_critical_reliability_comparison.png")

    # -------------------------------------------------------------------------
    # Plot 5: Worst-Case Controller Performance
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ctrl_worst = []
    ctrl_names = []
    for ctrl in main_controllers:
        sub = raw_df[raw_df["controller"] == ctrl]
        ctrl_worst.append(sub["reward"].min())
        ctrl_names.append(ctrl)

    bars = ax.barh(ctrl_names, ctrl_worst, color=[palette.get(c, "#555") for c in ctrl_names], edgecolor="black")
    ax.set_xlabel("Worst-Case Single-Episode Reward (Higher is Better)")
    ax.set_title("Plot 5: Worst-Case Resilience Floor Across All 30 Evaluated Episodes")
    for bar in bars:
        w = bar.get_width()
        ax.text(w - 100, bar.get_y() + bar.get_height()/2, f"{w:.1f}", va="center", ha="right", color="white", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot5_worst_case_controller_performance.png")
    plt.close()
    sync_artifact("plot5_worst_case_controller_performance.png")

    # -------------------------------------------------------------------------
    # Plot 6: Conditional Value at Risk (CVaR 5%)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    cvar_vals = []
    for ctrl in main_controllers:
        sub = raw_df[raw_df["controller"] == ctrl]
        cvar_vals.append(compute_cvar(sub["reward"].values, alpha=0.05, higher_is_better=True))

    bars = ax.bar(ctrl_names, cvar_vals, color=[palette.get(c, "#555") for c in ctrl_names], edgecolor="black", width=0.55)
    ax.set_ylabel("CVaR 5% Reward (Higher is Better)")
    ax.set_xticklabels(ctrl_names, rotation=20, ha="right")
    ax.set_title("Plot 6: Tail Risk Analysis — Conditional Value at Risk (CVaR 5%)")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot6_cvar_comparison.png")
    plt.close()
    sync_artifact("plot6_cvar_comparison.png")

    # -------------------------------------------------------------------------
    # Plot 7: Performance Degradation vs Disturbance Severity
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # Order scenarios by severity level: H4 (mild) < H5 < H1 < H2 < H3 < H6 (extreme)
    severity_order = ["scenario_h4", "scenario_h5", "scenario_h1", "scenario_h2", "scenario_h3", "scenario_h6"]
    sev_labels = ["H4 (Sensor)", "H5 (Stale)", "H1 (Topology)", "H2 (Drought)", "H3 (Surge)", "H6 (Compound)"]
    sev_idx = np.arange(len(severity_order))

    for ctrl in main_controllers:
        means = [raw_df[(raw_df["controller"] == ctrl) & (raw_df["scenario"] == s)]["reward"].mean() for s in severity_order]
        ax.plot(sev_idx, means, marker="o", linewidth=2, label=ctrl, color=palette.get(ctrl, "#555"))

    ax.set_xticks(sev_idx)
    ax.set_xticklabels(sev_labels, rotation=15)
    ax.set_ylabel("Episode Reward")
    ax.set_xlabel("Disturbance Severity Profile (Mild -> Extreme Stress)")
    ax.set_title("Plot 7: Policy Performance Degradation Profile Across Severity Spectrum")
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot7_performance_degradation_vs_severity.png")
    plt.close()
    sync_artifact("plot7_performance_degradation_vs_severity.png")

    # -------------------------------------------------------------------------
    # Plot 8: Network Topology Fault Recovery (Scenario H1 Trajectory)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    traj_h1_rob = pd.read_csv(TRAJ_DIR / "scenario_h1_Robust_Forecast_MAPPO_Seed_42.csv")
    traj_h1_rb = pd.read_csv(TRAJ_DIR / "scenario_h1_Rule_Based_EMS.csv")

    hours = traj_h1_rob["hour"]
    ax1.plot(hours, traj_h1_rob["solar_available_mw"], "k--", label="Potential Solar", alpha=0.5)
    ax1.plot(hours, traj_h1_rob["solar_mw"], color="goldenrod", linewidth=2, label="Dispatched Solar (Branch Trip at 10-13h)")
    ax1.plot(hours, traj_h1_rob["wind_mw"], color="teal", linewidth=2, label="Dispatched Wind (Branch Trip at 14-16h)")
    ax1.axvspan(10, 13, color="orange", alpha=0.15, label="Solar Feeder Trip (Line 1)")
    ax1.axvspan(14, 16, color="cyan", alpha=0.15, label="Wind Feeder Trip (Line 2)")
    ax1.axvspan(12, 13, color="red", alpha=0.15, label="Concurrent Grid Outage")
    ax1.set_ylabel("Power [MW]")
    ax1.set_title("Plot 8: Network Topology Fault Response — Physical Feeder Disconnections (H1)")
    ax1.legend(loc="upper right", frameon=True, ncol=2)

    ax2.plot(hours, traj_h1_rob["battery_soc"], color="#008080", linewidth=2, label="Robust MAPPO Battery SOC")
    ax2.plot(hours, traj_h1_rb["battery_soc"], color="#2ca02c", linestyle="--", linewidth=2, label="Rule-Based Battery SOC")
    ax2.axhline(30.0, color="red", linestyle=":", label="30% Emergency Reserve Floor")
    ax2.set_xlabel("Simulation Timeline [Hours]")
    ax2.set_ylabel("Battery SOC [%]")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot8_network_fault_recovery_trajectories.png")
    plt.close()
    sync_artifact("plot8_network_fault_recovery_trajectories.png")

    # -------------------------------------------------------------------------
    # Plot 9: Battery SOC Trajectory Under Extended Renewable Drought (Scenario H2)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    traj_h2_rob = pd.read_csv(TRAJ_DIR / "scenario_h2_Robust_Forecast_MAPPO_Seed_42.csv")
    traj_h2_fc = pd.read_csv(TRAJ_DIR / "scenario_h2_Forecast_Aware_MAPPO_Phase_4.2.csv")
    traj_h2_rb = pd.read_csv(TRAJ_DIR / "scenario_h2_Rule_Based_EMS.csv")

    ax.plot(hours, traj_h2_rob["battery_soc"], color="#008080", linewidth=2.5, label="Robust Forecast MAPPO (Seed 42)")
    ax.plot(hours, traj_h2_fc["battery_soc"], color="#1f77b4", linestyle="-.", linewidth=2, label="Standard Forecast MAPPO (Phase 4.2)")
    ax.plot(hours, traj_h2_rb["battery_soc"], color="#2ca02c", linestyle="--", linewidth=2, label="Rule-Based EMS")
    ax.axvspan(11, 14, color="red", alpha=0.15, label="Grid Outage During Drought")
    ax.axhline(30.0, color="red", linestyle=":", label="30% Reserve Floor")
    ax.axhline(20.0, color="black", linestyle="--", label="20% Hard Minimum Limit")

    ax.set_xlabel("Simulation Timeline [Hours]")
    ax.set_ylabel("Battery State of Charge [%]")
    ax.set_title("Plot 9: Battery Reserve Preservation During 18h Correlated Renewable Drought (H2)")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot9_battery_soc_under_renewable_drought.png")
    plt.close()
    sync_artifact("plot9_battery_soc_under_renewable_drought.png")

    # -------------------------------------------------------------------------
    # Plot 10: Generator Dispatch Under Prolonged Grid Outage (Scenario H6)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    traj_h6_rob = pd.read_csv(TRAJ_DIR / "scenario_h6_Robust_Forecast_MAPPO_Seed_42.csv")
    traj_h6_rb = pd.read_csv(TRAJ_DIR / "scenario_h6_Rule_Based_EMS.csv")

    ax1.plot(hours, traj_h6_rob["generator_mw"], color="#008080", linewidth=2.5, label="Robust MAPPO Generator Dispatch")
    ax1.plot(hours, traj_h6_rb["generator_mw"], color="#2ca02c", linestyle="--", linewidth=2, label="Rule-Based Generator Dispatch")
    ax1.axhline(0.20, color="red", linestyle=":", label="Derated 50% Generator Limit (0.20 MW)")
    ax1.axvspan(10, 15, color="red", alpha=0.15, label="5-Hour Grid Outage Window")
    ax1.set_ylabel("Generator Power [MW]")
    ax1.set_title("Plot 10: Generator Dispatch Under 5h Outage & 50% Mechanical Derating (H6)")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(hours, traj_h6_rob["served_critical_mw"], color="#008080", linewidth=2, label="Robust MAPPO Critical Served")
    ax2.plot(hours, traj_h6_rob["nominal_critical_mw"], color="black", linestyle=":", label="Surged Critical Demand (+50%)")
    ax2.plot(hours, traj_h6_rb["served_critical_mw"], color="#2ca02c", linestyle="--", linewidth=2, label="Rule-Based Critical Served")
    ax2.set_xlabel("Simulation Timeline [Hours]")
    ax2.set_ylabel("Critical Load [MW]")
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot10_generator_dispatch_prolonged_outage.png")
    plt.close()
    sync_artifact("plot10_generator_dispatch_prolonged_outage.png")

    # -------------------------------------------------------------------------
    # Plot 11: Load Shedding & Demand Surges (Scenario H3)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    traj_h3 = pd.read_csv(TRAJ_DIR / "scenario_h3_Robust_Forecast_MAPPO_Seed_42.csv")

    ax.plot(hours, traj_h3["nominal_total_mw"], "k-", linewidth=2, label="Total Surged Demand (+70% Crit, +80% Flex)")
    ax.plot(hours, traj_h3["served_total_mw"], color="#008080", linewidth=2.5, label="Total Served Load (Robust MAPPO)")
    ax.fill_between(hours, traj_h3["served_total_mw"], traj_h3["nominal_total_mw"], color="red", alpha=0.2, label="Unserved / Shed Load")
    ax.axvspan(12, 15, color="red", alpha=0.15, label="Midday Grid Outage (3 Hours)")
    ax.set_xlabel("Simulation Timeline [Hours]")
    ax.set_ylabel("Demand & Served Power [MW]")
    ax.set_title("Plot 11: Dynamic Load Shedding Response Under Extreme Demand Surges (H3)")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot11_load_shedding_under_demand_surges.png")
    plt.close()
    sync_artifact("plot11_load_shedding_under_demand_surges.png")

    # -------------------------------------------------------------------------
    # Plot 12: Robustness Under Sensor Noise & Dropout (Scenario H4)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    h4_controllers = [
        "Rule-Based EMS",
        "Reactive MAPPO (Phase 4.1)",
        "Forecast-Aware MAPPO (Phase 4.2)",
        "Robust Forecast MAPPO (Seed 42)",
        "Ablation: No Sensor Noise",
    ]
    h4_rewards = [raw_df[(raw_df["controller"] == c) & (raw_df["scenario"] == "scenario_h4")]["reward"].mean() for c in h4_controllers]
    h4_ens = [raw_df[(raw_df["controller"] == c) & (raw_df["scenario"] == "scenario_h4")]["critical_ens_mwh"].mean() * 1000 for c in h4_controllers]

    x_h4 = np.arange(len(h4_controllers))
    bars = ax.bar(x_h4, h4_rewards, color=["#2ca02c", "#ff7f0e", "#1f77b4", "#008080", "#8c564b"], width=0.55, edgecolor="black")
    ax.set_ylabel("Episode Reward Under Sensor Corruption")
    ax.set_xticks(x_h4)
    ax.set_xticklabels(h4_controllers, rotation=20, ha="right")
    ax.set_title("Plot 12: Telemetry Noise Resilience (Gaussian sigma=0.12 + 15% Dropout) (H4)")
    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h - 20, f"ENS: {h4_ens[i]:.1f} kWh", ha="center", va="top", fontsize=9, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot12_robustness_under_sensor_noise.png")
    plt.close()
    sync_artifact("plot12_robustness_under_sensor_noise.png")

    # -------------------------------------------------------------------------
    # Plot 13: Robustness Under Stale Forecast Delays (Scenario H5)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    h5_controllers = [
        "Rule-Based EMS",
        "Greedy Controller",
        "Forecast-Aware MAPPO (Phase 4.2)",
        "Robust Forecast MAPPO (Seed 42)",
        "Ablation: No Uncertainty",
    ]
    h5_rewards = [raw_df[(raw_df["controller"] == c) & (raw_df["scenario"] == "scenario_h5")]["reward"].mean() for c in h5_controllers]
    bars = ax.bar(h5_controllers, h5_rewards, color=["#2ca02c", "#d62728", "#1f77b4", "#008080", "#e377c2"], width=0.55, edgecolor="black")
    ax.set_ylabel("Episode Reward")
    ax.set_xticklabels(h5_controllers, rotation=20, ha="right")
    ax.set_title("Plot 13: Stale / Frozen Forecast Resilience (4h Freeze Under Storm Arrival) (H5)")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot13_robustness_under_stale_forecasts.png")
    plt.close()
    sync_artifact("plot13_robustness_under_stale_forecasts.png")

    # -------------------------------------------------------------------------
    # Plot 14: Safety Shield Interventions (Emergency vs Boundary)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ctrls_shield = [
        "Reactive MAPPO (Phase 4.1)",
        "Forecast-Aware MAPPO (Phase 4.2)",
        "Robust Forecast MAPPO (Seed 42)",
        "Ablation: No Uncertainty",
    ]
    emerg_means = [summary_df[summary_df["controller"] == c]["emergency_interventions_mean"].mean() for c in ctrls_shield]
    bound_means = [summary_df[summary_df["controller"] == c]["boundary_interventions_mean"].mean() for c in ctrls_shield]

    x_sh = np.arange(len(ctrls_shield))
    ax.bar(x_sh, emerg_means, 0.45, label="Emergency Interventions (Critical Shed / Undercharge)", color="#d62728", edgecolor="black")
    ax.bar(x_sh, bound_means, 0.45, bottom=emerg_means, label="Boundary Clipping (Reserve Floor / Overcharge)", color="#1f77b4", edgecolor="black")
    ax.set_xticks(x_sh)
    ax.set_xticklabels(ctrls_shield, rotation=15, ha="right")
    ax.set_ylabel("Average Interventions Per Episode")
    ax.set_title("Plot 14: Safety Shield Intervention Taxonomy: Emergency vs Boundary Reserve Clipping")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot14_safety_shield_interventions_taxonomy.png")
    plt.close()
    sync_artifact("plot14_safety_shield_interventions_taxonomy.png")

    # -------------------------------------------------------------------------
    # Plot 15: Robustness Ablation Study
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    abl_ctrls = [
        "Forecast-Aware MAPPO (Phase 4.2)",
        "Robust Forecast MAPPO (Seed 42)",
        "Ablation: No Sensor Noise",
        "Ablation: No Uncertainty",
    ]
    abl_labels = [
        "Standard Forecast MAPPO",
        "Robust MAPPO (Full DR)",
        "Ablation: No Sensor Noise DR",
        "Ablation: No Uncertainty Intervals",
    ]
    # Average across all 6 scenarios
    abl_mean_rew = [raw_df[raw_df["controller"] == c]["reward"].mean() for c in abl_ctrls]
    abl_cvar5 = [compute_cvar(raw_df[raw_df["controller"] == c]["reward"].values, alpha=0.05, higher_is_better=True) for c in abl_ctrls]

    x_ab = np.arange(len(abl_ctrls))
    ax.bar(x_ab - 0.18, abl_mean_rew, 0.35, label="Mean Reward (All Held-Out Scenarios)", color="#008080", edgecolor="black")
    ax.bar(x_ab + 0.18, abl_cvar5, 0.35, label="CVaR 5% (Worst 5% Stress Episodes)", color="#ff7f0e", edgecolor="black")
    ax.set_xticks(x_ab)
    ax.set_xticklabels(abl_labels, rotation=15, ha="right")
    ax.set_ylabel("Reward Metric (Higher is Better)")
    ax.set_title("Plot 15: Robustness Component Ablation Analysis Across 30 Held-Out Episodes")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot15_robustness_ablation_study.png")
    plt.close()
    sync_artifact("plot15_robustness_ablation_study.png")

    # -------------------------------------------------------------------------
    # Plot 16: Multi-Seed Training Convergence Curves
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    seeds = [42, 101, 202]
    colors = ["#008080", "#2ca02c", "#9467bd"]

    for s, col in zip(seeds, colors):
        log_file = LOGS_DIR / f"robust_seed{s}" / "mappo_training_log.csv"
        if log_file.exists():
            df_log = pd.read_csv(log_file)
            col_name = "rolling_reward_20" if "rolling_reward_20" in df_log.columns else "episode_reward"
            ax.plot(df_log["episode"], df_log[col_name], color=col, linewidth=2, label=f"Robust MAPPO (Seed {s})")

    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Rolling Average Reward")
    ax.set_title("Plot 16: Independent Multi-Seed Training Convergence Curves (Seeds 42, 101, 202)")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot16_multiseed_training_curves.png")
    plt.close()
    sync_artifact("plot16_multiseed_training_curves.png")

    # -------------------------------------------------------------------------
    # Plot 17: Post-Disturbance Recovery Time
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    recov_times = [summary_df[summary_df["controller"] == c]["recovery_time_min_mean"].mean() for c in main_controllers]
    bars = ax.bar(main_controllers, recov_times, color=[palette.get(c, "#555") for c in main_controllers], width=0.55, edgecolor="black")
    ax.set_ylabel("Mean Recovery Time [Minutes] (Lower is Better)")
    ax.set_xticklabels(main_controllers, rotation=20, ha="right")
    ax.set_title("Plot 17: Average Post-Disturbance Microgrid Normalization Time")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 10, f"{h:.0f}m", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot17_recovery_time_comparison.png")
    plt.close()
    sync_artifact("plot17_recovery_time_comparison.png")

    print("=" * 70)
    print("All 17 Phase 4.3 Plots Successfully Generated & Synced to Artifacts!")
    print("=" * 70)


if __name__ == "__main__":
    generate_all_plots()
