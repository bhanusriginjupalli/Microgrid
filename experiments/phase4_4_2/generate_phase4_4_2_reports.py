"""
experiments/phase4_4_2/generate_phase4_4_2_reports.py
=====================================================
Generates publication-grade diagnostic & comparison plots for Phase 4.4.2.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
BASE_442 = ROOT / "experiments" / "phase4_4_2"
RES_DIR = BASE_442 / "results"
PLOT_DIR = BASE_442 / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def generate_plots():
    comp_df = pd.read_csv(RES_DIR / "controller_comparison.csv")
    ep_df = pd.read_csv(RES_DIR / "episode_level_metrics.csv")
    val_df = pd.read_csv(RES_DIR / "training_validation_history.csv")
    mult_df = pd.read_csv(RES_DIR / "constraint_multiplier_history.csv")

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

    # Filter representative primary controllers for crisp comparisons
    keep_ctrls = [
        "Rule-Based EMS",
        "Greedy Controller",
        "Lookahead Heuristic EMS",
        "Robust Forecast MAPPO (Seed 42)",
        "Legacy Constrained MAPPO (Phase 4.4 Seed 42)",
        "Legacy CVaR MAPPO (Phase 4.4 Seed 42)",
        "Corrected Constrained MAPPO (Seed 42)",
        "Corrected Constrained MAPPO (Seed 42) + Fallback",
        "Tail-Risk Constrained MAPPO (Seed 42)",
        "Tail-Risk Constrained MAPPO (Seed 42) + Fallback",
    ]
    sub_ep = ep_df[ep_df["controller"].isin(keep_ctrls)]
    sub_comp = comp_df[comp_df["controller"].isin(keep_ctrls)].copy()
    order_map = {name: i for i, name in enumerate(keep_ctrls)}
    sub_comp["ord"] = sub_comp["controller"].map(order_map)
    sub_comp = sub_comp.sort_values("ord")

    # ------------------------------------------------------------
    # Plot 1: True Social Cost Breakdown (Operating + VoLL + Storage)
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    means_by_ctrl = sub_ep.groupby("controller")[
        ["operating_cost_usd", "unserved_energy_cost_usd", "battery_degradation_cost_usd", "terminal_soc_adjustment_usd"]
    ].mean().reindex(keep_ctrls)

    x = np.arange(len(keep_ctrls))
    op_vals = means_by_ctrl["operating_cost_usd"].values
    voll_vals = means_by_ctrl["unserved_energy_cost_usd"].values
    stor_vals = (means_by_ctrl["battery_degradation_cost_usd"] + means_by_ctrl["terminal_soc_adjustment_usd"]).values

    ax.bar(x, op_vals, label="Operating Cost (Grid + Fuel)", color="#2b6cb0")
    ax.bar(x, voll_vals, bottom=op_vals, label="Unserved Energy Penalty (VoLL)", color="#e53e3e")
    ax.bar(x, np.maximum(0.0, stor_vals), bottom=op_vals + voll_vals, label="Battery Deg. + Terminal SOC", color="#dd6b20")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" (Phase 4.4 Seed 42)", " (4.4)").replace(" (Seed 42)", " (S42)") for c in keep_ctrls], rotation=28, ha="right")
    ax.set_ylabel("Mean Episode Cost ($ USD)")
    ax.set_title("Phase 4.4.2 Authoritative Economic Breakdown: Operating Cost vs True Social Cost (H1–H6)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot1_true_social_cost_comparison.png", dpi=180)
    plt.close()

    # ------------------------------------------------------------
    # Plot 2: Critical Reliability & Avoidable vs Unavoidable ENS
    # ------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    ax1.bar(x, sub_comp["critical_reliability_mean"], yerr=sub_comp["critical_reliability_std"], capsize=4, color="#38a169")
    ax1.set_ylim(80, 101)
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace(" (Phase 4.4 Seed 42)", " (4.4)").replace(" (Seed 42)", " (S42)") for c in keep_ctrls], rotation=35, ha="right")
    ax1.set_ylabel("Critical Load Serving Reliability (%)")
    ax1.set_title("Critical Load Reliability (Mean ± Std)")

    av_ens = sub_comp["avoidable_critical_ens_mean"].values
    unav_ens = sub_comp["unavoidable_critical_ens_mean"].values
    ax2.bar(x, unav_ens, label="Physically Unavoidable Critical ENS", color="#718096")
    ax2.bar(x, av_ens, bottom=unav_ens, label="Avoidable Critical ENS (Inaction/Policy Deficit)", color="#c53030")
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.replace(" (Phase 4.4 Seed 42)", " (4.4)").replace(" (Seed 42)", " (S42)") for c in keep_ctrls], rotation=35, ha="right")
    ax2.set_ylabel("Mean Critical ENS (MWh / episode)")
    ax2.set_title("Critical ENS Decomposition: Avoidable vs Physically Unavoidable")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot2_critical_reliability_and_ens.png", dpi=180)
    plt.close()

    # ------------------------------------------------------------
    # Plot 3: Unshielded Feasible Rate vs Passive Inaction Steps
    # ------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    ax1.bar(x, sub_comp["unshielded_feasible_pct"], color="#3182ce")
    ax1.set_ylim(70, 102)
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace(" (Phase 4.4 Seed 42)", " (4.4)").replace(" (Seed 42)", " (S42)") for c in keep_ctrls], rotation=35, ha="right")
    ax1.set_ylabel("Unshielded Feasible Proposals (%)")
    ax1.set_title("Actuator Boundary Compliance (Pre-Shield)")

    ax2.bar(x, sub_comp["inaction_steps_mean"], color="#d69e2e")
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.replace(" (Phase 4.4 Seed 42)", " (4.4)").replace(" (Seed 42)", " (S42)") for c in keep_ctrls], rotation=35, ha="right")
    ax2.set_ylabel("Mean Inaction / Starvation Steps per Episode")
    ax2.set_title("Operational Inaction Rate (Available Assets Unused During Deficit)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot3_inaction_and_adequacy_audit.png", dpi=180)
    plt.close()

    # ------------------------------------------------------------
    # Plot 4: Validation Selection & Lagrange Multiplier Dynamics
    # ------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for vname, grp in val_df.groupby("variant_name"):
        if "corrected_constrained" in vname:
            ax1.plot(grp["episode"], grp["val_true_social_cost_usd"], marker="o", label=vname)
    ax1.set_xlabel("Training Episode")
    ax1.set_ylabel("Fixed Validation True Social Cost ($)")
    ax1.set_title("Fixed Validation Suite Trajectory (Multi-Seed)")
    ax1.legend()

    m_s42 = mult_df[mult_df["variant_name"] == "corrected_constrained_seed42"]
    for col in ["lambda_crit", "lambda_inaction", "lambda_bat", "lambda_gen"]:
        if col in m_s42:
            ax2.plot(m_s42["episode"], m_s42[col], marker=".", label=col)
    ax2.set_xlabel("Training Episode")
    ax2.set_ylabel("Lagrange Multiplier Value")
    ax2.set_title("Adaptive Dual Multiplier Trajectories (Corrected Seed 42)")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot4_lagrange_multipliers_and_validation.png", dpi=180)
    plt.close()

    # ------------------------------------------------------------
    # Plot 5: Tail Risk Comparison (Mean vs CVaR 10% & 5% Social Cost)
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5.5))
    w = 0.26
    ax.bar(x - w, sub_comp["true_social_cost_mean"], width=w, label="Mean Social Cost", color="#4a5568")
    ax.bar(x, sub_comp["true_social_cost_cvar10"], width=w, label="CVaR 10% (Worst 10%)", color="#dd6b20")
    ax.bar(x + w, sub_comp["true_social_cost_cvar05"], width=w, label="CVaR 5% (Worst 5%)", color="#c53030")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" (Phase 4.4 Seed 42)", " (4.4)").replace(" (Seed 42)", " (S42)") for c in keep_ctrls], rotation=28, ha="right")
    ax.set_ylabel("True Social Cost ($ USD)")
    ax.set_title("Tail-Risk Evaluation: Mean vs CVaR 10% & CVaR 5% True Social Cost")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot5_cvar_tail_risk_distribution.png", dpi=180)
    plt.close()

    print(f"[+] Generated 5 Phase 4.4.2 plots in {PLOT_DIR}")


if __name__ == "__main__":
    generate_plots()
