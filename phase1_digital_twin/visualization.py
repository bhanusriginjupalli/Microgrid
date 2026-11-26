# GRIDGUARD AI
# Phase 1: Visualization & Reporting
# ============================================================
# Generates plots and summary reports from simulation results.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Default output directory (relative to this file)
_DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "data", "simulation_results")


def plot_power_balance(df: pd.DataFrame, save_dir: str = None):
    """Stacked area chart of generation vs demand."""
    fig, ax = plt.subplots(figsize=(14, 6))

    hours = df["hour"]
    ax.fill_between(hours, 0, df["solar_mw"], alpha=0.6, label="Solar", color="#FFD700")
    ax.fill_between(hours, df["solar_mw"],
                    df["solar_mw"] + df["wind_mw"],
                    alpha=0.6, label="Wind", color="#87CEEB")

    # Battery discharge shown as positive contribution
    batt_discharge = df["battery_mw"].clip(upper=0).abs()
    ax.fill_between(hours,
                    df["solar_mw"] + df["wind_mw"],
                    df["solar_mw"] + df["wind_mw"] + batt_discharge,
                    alpha=0.6, label="Battery Discharge", color="#32CD32")

    ax.fill_between(hours,
                    df["solar_mw"] + df["wind_mw"] + batt_discharge,
                    df["solar_mw"] + df["wind_mw"] + batt_discharge + df["generator_mw"],
                    alpha=0.6, label="Generator", color="#FF6347")

    # Grid import
    grid_pos = df["grid_import_mw"].clip(lower=0)
    ax.fill_between(hours,
                    df["solar_mw"] + df["wind_mw"] + batt_discharge + df["generator_mw"],
                    df["solar_mw"] + df["wind_mw"] + batt_discharge + df["generator_mw"] + grid_pos,
                    alpha=0.4, label="Grid Import", color="#808080")

    ax.plot(hours, df["total_load_mw"], "k--", linewidth=2, label="Total Load")

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Power (MW)")
    ax.set_title("GRIDGUARD AI — Power Balance")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 24)
    ax.grid(True, alpha=0.3)

    # Shade disturbance periods
    _shade_disturbances(ax, df)

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, "power_balance.png"), dpi=150)
    plt.show()


def plot_battery_soc(df: pd.DataFrame, save_dir: str = None):
    """Battery state-of-charge over time."""
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(df["hour"], df["battery_soc"], color="#32CD32", linewidth=2)
    ax.axhline(20, color="red", linestyle=":", label="Min SOC (20%)")
    ax.axhline(95, color="orange", linestyle=":", label="Max SOC (95%)")
    ax.fill_between(df["hour"], 20, df["battery_soc"],
                    where=df["battery_soc"] >= 20, alpha=0.2, color="#32CD32")

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("SOC (%)")
    ax.set_title("GRIDGUARD AI — Battery State of Charge")
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, "battery_soc.png"), dpi=150)
    plt.show()


def plot_grid_and_weather(df: pd.DataFrame, save_dir: str = None):
    """Two-panel plot: grid import/export and weather conditions."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Panel 1: Grid power
    ax1.plot(df["hour"], df["grid_import_mw"], color="#808080", linewidth=2)
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.fill_between(df["hour"], 0, df["grid_import_mw"],
                     where=df["grid_import_mw"] >= 0, alpha=0.3,
                     color="red", label="Import")
    ax1.fill_between(df["hour"], 0, df["grid_import_mw"],
                     where=df["grid_import_mw"] < 0, alpha=0.3,
                     color="green", label="Export")
    ax1.set_ylabel("Grid Power (MW)")
    ax1.set_title("GRIDGUARD AI — Grid Import / Export")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    _shade_disturbances(ax1, df)

    # Panel 2: Irradiance & wind speed
    ax2.plot(df["hour"], df["irradiance"], color="#FFD700", linewidth=2, label="Irradiance")
    ax2b = ax2.twinx()
    ax2b.plot(df["hour"], df["wind_speed_ms"], color="#87CEEB", linewidth=1.5,
              linestyle="--", label="Wind Speed")
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Irradiance (norm)")
    ax2b.set_ylabel("Wind Speed (m/s)")
    ax2.set_title("Weather Conditions")
    ax2.legend(loc="upper left")
    ax2b.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, "grid_and_weather.png"), dpi=150)
    plt.show()


def plot_load_shedding(df: pd.DataFrame, save_dir: str = None):
    """Highlight periods of load shedding."""
    if df["load_shed_mw"].max() < 0.001:
        print("No load shedding occurred — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(df["hour"], df["load_shed_mw"], width=0.2, color="red", alpha=0.7)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Load Shed (MW)")
    ax.set_title("GRIDGUARD AI — Load Shedding Events")
    ax.set_xlim(0, 24)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, "load_shedding.png"), dpi=150)
    plt.show()


def generate_report(df: pd.DataFrame):
    """Print a text-based performance report."""
    dt_h = df["hour"].diff().median()  # time step in hours

    print("\n" + "=" * 60)
    print("GRIDGUARD AI — PERFORMANCE REPORT")
    print("=" * 60)

    total_energy_gen = (df["solar_mw"] + df["wind_mw"] + df["generator_mw"]).sum() * dt_h
    total_load_energy = df["total_load_mw"].sum() * dt_h
    total_import = df["grid_import_mw"].clip(lower=0).sum() * dt_h
    total_export = df["grid_import_mw"].clip(upper=0).abs().sum() * dt_h
    total_shed = df["load_shed_mw"].sum() * dt_h

    print(f"\n  Total renewable generation:  {total_energy_gen:.3f} MWh")
    print(f"  Total load served:           {total_load_energy:.3f} MWh")
    print(f"  Total grid import:           {total_import:.3f} MWh")
    print(f"  Total grid export:           {total_export:.3f} MWh")
    print(f"  Total load shed:             {total_shed:.3f} MWh")

    print(f"\n  Battery SOC range:           "
          f"{df['battery_soc'].min():.1f}% – {df['battery_soc'].max():.1f}%")
    print(f"  Power flow convergence:      "
          f"{df['converged'].sum()}/{len(df)} steps")

    # Renewable fraction (excluding grid import)
    if total_load_energy > 0:
        ren_frac = (total_energy_gen / total_load_energy) * 100
        print(f"  Renewable fraction:          {ren_frac:.1f}%")

    print("=" * 60)


def _shade_disturbances(ax, df: pd.DataFrame):
    """Add light shading for grid-disconnected periods."""
    grid_off = ~df["grid_connected"]
    if grid_off.any():
        starts = df.loc[grid_off, "hour"]
        for h in starts:
            ax.axvspan(h, h + 0.25, alpha=0.15, color="red")
        patch = mpatches.Patch(color="red", alpha=0.15, label="Grid Outage")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles + [patch], labels=labels + ["Grid Outage"],
                  loc="upper left")



# ============================================================
# Phase 1.5: Ten Dedicated Resilience & Dynamic Plots
# ============================================================

def generate_phase15_plots(df: pd.DataFrame, scenario_name: str = "Scenario", save_dir: str = None) -> list:
    """
    Generate all 10 required Phase 1.5 verification plots and save to disk.
    Returns list of saved image paths.
    """
    out_dir = save_dir or _DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)
    clean_name = scenario_name.lower().replace(" ", "_").replace(":", "").replace("-", "_")
    saved_files = []

    hours = df["hour"]

    # --- Plot 1: Total demand vs. total generation ---
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    total_gen = df["solar_mw"] + df["wind_mw"] + df["generator_mw"]
    ax1.plot(hours, df["nominal_total_mw"], "k--", label="Nominal Demand", linewidth=1.8)
    ax1.plot(hours, df["served_total_mw"], "b-", label="Served Demand", linewidth=2.0)
    ax1.plot(hours, total_gen, "g-", label="Internal Generation", linewidth=1.5, alpha=0.85)
    _shade_disturbances(ax1, df)
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Power (MW)")
    ax1.set_title(f"1. Total Demand vs Total Generation - {scenario_name}")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    p1 = os.path.join(out_dir, f"{clean_name}_01_demand_vs_generation.png")
    fig1.tight_layout()
    fig1.savefig(p1, dpi=120)
    plt.close(fig1)
    saved_files.append(p1)

    # --- Plot 2: Solar and wind generation ---
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(hours, df["solar_mw"], color="#FFA500", label="Solar PV Output", linewidth=2.0)
    ax2.plot(hours, df["wind_mw"], color="#1E90FF", label="Wind Turbine Output", linewidth=2.0)
    _shade_disturbances(ax2, df)
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Active Power (MW)")
    ax2.set_title(f"2. Solar PV and Wind Turbine Generation - {scenario_name}")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    p2 = os.path.join(out_dir, f"{clean_name}_02_solar_and_wind.png")
    fig2.tight_layout()
    fig2.savefig(p2, dpi=120)
    plt.close(fig2)
    saved_files.append(p2)

    # --- Plot 3: Battery SOC throughout simulation ---
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    ax3.plot(hours, df["battery_soc"], color="#2E8B57", label="Battery SOC (%)", linewidth=2.2)
    ax3.axhline(20, color="red", linestyle="--", label="Min SOC (20%)", linewidth=1.2)
    ax3.axhline(30, color="orange", linestyle=":", label="Emergency Reserve (30%)", linewidth=1.5)
    ax3.axhline(95, color="darkgreen", linestyle="--", label="Max SOC (95%)", linewidth=1.2)
    ax3.fill_between(hours, 20, 30, color="orange", alpha=0.15)
    _shade_disturbances(ax3, df)
    ax3.set_ylim(10, 105)
    ax3.set_xlabel("Hour of Day")
    ax3.set_ylabel("State of Charge (%)")
    ax3.set_title(f"3. Battery State of Charge (SOC) - {scenario_name}")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)
    p3 = os.path.join(out_dir, f"{clean_name}_03_battery_soc.png")
    fig3.tight_layout()
    fig3.savefig(p3, dpi=120)
    plt.close(fig3)
    saved_files.append(p3)

    # --- Plot 4: Battery charging/discharging power ---
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    chg = df["battery_mw"].clip(lower=0)
    dis = df["battery_mw"].clip(upper=0)
    ax4.bar(hours, chg, width=0.22, color="green", alpha=0.7, label="Charging (P > 0)")
    ax4.bar(hours, dis, width=0.22, color="blue", alpha=0.7, label="Discharging (P < 0)")
    ax4.axhline(0, color="black", linewidth=0.8)
    _shade_disturbances(ax4, df)
    ax4.set_xlabel("Hour of Day")
    ax4.set_ylabel("Battery Power (MW)")
    ax4.set_title(f"4. Battery Active Power (Charge / Discharge) - {scenario_name}")
    ax4.legend(loc="upper left")
    ax4.grid(True, alpha=0.3)
    p4 = os.path.join(out_dir, f"{clean_name}_04_battery_power.png")
    fig4.tight_layout()
    fig4.savefig(p4, dpi=120)
    plt.close(fig4)
    saved_files.append(p4)

    # --- Plot 5: Generator power and operating state ---
    fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax5a.plot(hours, df["generator_mw"], color="#B22222", label="Generator Output", linewidth=2.0)
    ax5a.axhline(0.08, color="grey", linestyle=":", label="Min Stable (0.08 MW)")
    ax5a.axhline(0.40, color="black", linestyle="--", label="Rated Capacity (0.40 MW)")
    _shade_disturbances(ax5a, df)
    ax5a.set_ylabel("Power (MW)")
    ax5a.set_title(f"5. Backup Generator Dispatch & State Machine - {scenario_name}")
    ax5a.legend(loc="upper left")
    ax5a.grid(True, alpha=0.3)

    states = df["generator_state"].tolist()
    state_map = {"OFF": 0, "STARTING": 1, "RUNNING": 2, "SHUTTING_DOWN": 3}
    state_vals = [state_map.get(s, 0) for s in states]
    ax5b.step(hours, state_vals, where="post", color="#8B0000", linewidth=1.8)
    ax5b.set_yticks([0, 1, 2, 3])
    ax5b.set_yticklabels(["OFF", "STARTING", "RUNNING", "SHUTDOWN"])
    ax5b.set_xlabel("Hour of Day")
    ax5b.set_ylabel("State")
    ax5b.grid(True, alpha=0.3)
    p5 = os.path.join(out_dir, f"{clean_name}_05_generator_dispatch.png")
    fig5.tight_layout()
    fig5.savefig(p5, dpi=120)
    plt.close(fig5)
    saved_files.append(p5)

    # --- Plot 6: Grid import/export ---
    fig6, ax6 = plt.subplots(figsize=(12, 5))
    grid_imp = df["grid_import_mw"].clip(lower=0)
    grid_exp = df["grid_import_mw"].clip(upper=0).abs()
    ax6.fill_between(hours, 0, grid_imp, color="#DC143C", alpha=0.5, label="Grid Import (MW)")
    ax6.fill_between(hours, 0, -grid_exp, color="#2E8B57", alpha=0.5, label="Grid Export (MW)")
    ax6.axhline(0, color="black", linewidth=0.8)
    _shade_disturbances(ax6, df)
    ax6.set_xlabel("Hour of Day")
    ax6.set_ylabel("Grid Power (MW)")
    ax6.set_title(f"6. Utility Grid Exchange (Import / Export) - {scenario_name}")
    ax6.legend(loc="upper right")
    ax6.grid(True, alpha=0.3)
    p6 = os.path.join(out_dir, f"{clean_name}_06_grid_import_export.png")
    fig6.tight_layout()
    fig6.savefig(p6, dpi=120)
    plt.close(fig6)
    saved_files.append(p6)

    # --- Plot 7: Critical load served ---
    fig7, ax7 = plt.subplots(figsize=(12, 5))
    ax7.plot(hours, df["nominal_critical_mw"], "k--", label="Nominal Critical Demand", linewidth=1.5)
    ax7.plot(hours, df["served_critical_mw"], "g-", label="Served Critical Load", linewidth=2.0)
    if "shed_critical_mw" in df and df["shed_critical_mw"].max() > 0.001:
        ax7.fill_between(hours, df["served_critical_mw"], df["nominal_critical_mw"], color="red", alpha=0.4, label="Critical ENS")
    _shade_disturbances(ax7, df)
    ax7.set_ylim(0, max(df["nominal_critical_mw"]) * 1.3)
    ax7.set_xlabel("Hour of Day")
    ax7.set_ylabel("Critical Load (MW)")
    ax7.set_title(f"7. Critical Load Security - {scenario_name}")
    ax7.legend(loc="lower right")
    ax7.grid(True, alpha=0.3)
    p7 = os.path.join(out_dir, f"{clean_name}_07_critical_load_security.png")
    fig7.tight_layout()
    fig7.savefig(p7, dpi=120)
    plt.close(fig7)
    saved_files.append(p7)

    # --- Plot 8: Flexible load curtailment ---
    fig8, ax8 = plt.subplots(figsize=(12, 5))
    ax8.plot(hours, df["nominal_flexible_mw"], "k--", label="Nominal Flexible Demand", linewidth=1.5)
    ax8.plot(hours, df["served_flexible_mw"], "b-", label="Served Flexible Load", linewidth=2.0)
    shed_flex = df.get("shed_flexible_mw", pd.Series(0.0, index=df.index))
    ax8.fill_between(hours, 0, shed_flex, color="crimson", alpha=0.4, label="Curtailed Flexible Load")
    _shade_disturbances(ax8, df)
    ax8.set_xlabel("Hour of Day")
    ax8.set_ylabel("Flexible Load (MW)")
    ax8.set_title(f"8. Flexible Demand Response & Curtailment - {scenario_name}")
    ax8.legend(loc="upper left")
    ax8.grid(True, alpha=0.3)
    p8 = os.path.join(out_dir, f"{clean_name}_08_flexible_load_response.png")
    fig8.tight_layout()
    fig8.savefig(p8, dpi=120)
    plt.close(fig8)
    saved_files.append(p8)

    # --- Plot 9: Unserved energy by tier ---
    fig9, ax9 = plt.subplots(figsize=(12, 5))
    shed_f = df.get("shed_flexible_mw", pd.Series(0.0, index=df.index))
    shed_o = df.get("shed_other_mw", pd.Series(0.0, index=df.index))
    shed_c = df.get("shed_critical_mw", pd.Series(0.0, index=df.index))
    ax9.bar(hours, shed_f, width=0.22, color="gold", label="Flexible Load Curtailment", alpha=0.8)
    ax9.bar(hours, shed_o, bottom=shed_f, width=0.22, color="darkorange", label="Other Load Shedding", alpha=0.8)
    ax9.bar(hours, shed_c, bottom=shed_f + shed_o, width=0.22, color="red", label="Critical Load Shedding", alpha=0.8)
    _shade_disturbances(ax9, df)
    ax9.set_xlabel("Hour of Day")
    ax9.set_ylabel("Unserved Power (MW)")
    ax9.set_title(f"9. Load Curtailment by Tier (Unserved Energy) - {scenario_name}")
    ax9.legend(loc="upper left")
    ax9.grid(True, alpha=0.3)
    p9 = os.path.join(out_dir, f"{clean_name}_09_unserved_energy_tiers.png")
    fig9.tight_layout()
    fig9.savefig(p9, dpi=120)
    plt.close(fig9)
    saved_files.append(p9)

    # --- Plot 10: Event timeline of storm/outage/recovery ---
    fig10, ax10 = plt.subplots(figsize=(12, 4))
    timeline_y = []
    labels = []

    # Outage periods
    outage_mask = ~df["grid_connected"]
    if outage_mask.any():
        outage_hrs = df.loc[outage_mask, "hour"]
        ax10.broken_barh([(h, 0.25) for h in outage_hrs], (3, 0.8), facecolors="red", alpha=0.8)
        timeline_y.append(3.4)
        labels.append("Islanded Outage")

    # Generator running
    gen_mask = df["generator_mw"] > 0.01
    if gen_mask.any():
        gen_hrs = df.loc[gen_mask, "hour"]
        ax10.broken_barh([(h, 0.25) for h in gen_hrs], (2, 0.8), facecolors="brown", alpha=0.8)
        timeline_y.append(2.4)
        labels.append("Generator Running")

    # Cloud front / Solar reduction
    cloud_mask = df["cloud_factor"] < 0.9
    if cloud_mask.any():
        cloud_hrs = df.loc[cloud_mask, "hour"]
        ax10.broken_barh([(h, 0.25) for h in cloud_hrs], (1, 0.8), facecolors="grey", alpha=0.8)
        timeline_y.append(1.4)
        labels.append("Storm Cloud Front")

    # Curtailment active
    shed_mask = df.get("shed_total_mw", pd.Series(0.0, index=df.index)) > 0.001
    if shed_mask.any():
        shed_hrs = df.loc[shed_mask, "hour"]
        ax10.broken_barh([(h, 0.25) for h in shed_hrs], (0, 0.8), facecolors="gold", alpha=0.8)
        timeline_y.append(0.4)
        labels.append("Demand Curtailment")

    ax10.set_xlim(0, 24)
    ax10.set_ylim(-0.5, 4.5)
    ax10.set_yticks([0.4, 1.4, 2.4, 3.4])
    ax10.set_yticklabels(["Load Curtailment", "Solar Dimming", "Generator Active", "Grid Outage"])
    ax10.set_xlabel("Hour of Day")
    ax10.set_title(f"10. Disturbance, Dispatch & Recovery Timeline - {scenario_name}")
    ax10.grid(True, alpha=0.3)
    p10 = os.path.join(out_dir, f"{clean_name}_10_event_timeline.png")
    fig10.tight_layout()
    fig10.savefig(p10, dpi=120)
    plt.close(fig10)
    saved_files.append(p10)

    return saved_files


# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load saved results and generate all plots
    csv_path = os.path.join(_DEFAULT_OUT, "phase1_results.csv")
    if not os.path.exists(csv_path):
        print(f"No results found at {csv_path}")
        print("Run simulation.py first to generate data.")
    else:
        df = pd.read_csv(csv_path)
        out = _DEFAULT_OUT
        plot_power_balance(df, save_dir=out)
        plot_battery_soc(df, save_dir=out)
        plot_grid_and_weather(df, save_dir=out)
        plot_load_shedding(df, save_dir=out)
        generate_report(df)

