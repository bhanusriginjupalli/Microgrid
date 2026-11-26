"""
experiments/phase4_4_3/generate_phase4_4_3_plots.py
===================================================
Generates Phase 4.4.3 extended training convergence, intrinsic vs fallback
attribution, true social cost, and policy autonomy plots.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
BASE_443 = ROOT / "experiments" / "phase4_4_3"
RES_DIR = BASE_443 / "results"
PLOT_DIR = BASE_443 / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def generate_plots():
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

    comp = pd.read_csv(RES_DIR / "controller_comparison.csv")
    fb = pd.read_csv(RES_DIR / "fallback_attribution.csv")
    val_h = pd.read_csv(RES_DIR / "extended_validation_history.csv")
    mult_h = pd.read_csv(RES_DIR / "extended_multiplier_history.csv")

    # ------------------------------------------------------------------
    # Plot 1: Extended 100-Episode Validation Curves & Multiplier Dynamics
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for vname, grp in val_h.groupby("variant_name"):
        style = "-" if "corrected" in vname else "--"
        axes[0].plot(grp["episode"], grp["val_true_social_cost_usd"], marker="o", linestyle=style, label=vname)
        axes[1].plot(grp["episode"], grp["val_critical_reliability_pct"], marker="o", linestyle=style, label=vname)

    axes[0].set_title("Extended Validation: True Social Cost (100 Ep)")
    axes[0].set_xlabel("Training Episode")
    axes[0].set_ylabel("Validation True Social Cost ($)")
    axes[0].legend(fontsize=7.5)

    axes[1].set_title("Extended Validation: Critical Reliability (100 Ep)")
    axes[1].set_xlabel("Training Episode")
    axes[1].set_ylabel("Critical Reliability (%)")

    m42 = mult_h[mult_h["variant_name"] == "ext_corrected_seed42"]
    for col in ["lambda_crit", "lambda_inaction", "lambda_gen", "lambda_bat"]:
        if col in m42:
            axes[2].plot(m42["episode"], m42[col], marker=".", label=col)
    axes[2].set_title("Dual Multiplier Evolution (Ext Corrected S42)")
    axes[2].set_xlabel("Training Episode")
    axes[2].set_ylabel("Multiplier Value")
    axes[2].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot1_extended_training_convergence.png", dpi=180)
    plt.close()

    # ------------------------------------------------------------------
    # Plot 2: Intrinsic Policy vs +Fallback Attribution
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    labels = [f"{r['policy_family']}\n(Seed {r['train_seed']})" for _, r in fb.iterrows()]
    x = np.arange(len(labels))
    w = 0.36

    ax1.bar(x - w / 2, fb["intrinsic_critical_rel_pct"], width=w, label="Intrinsic Policy (No Fallback)", color="#3182ce")
    ax1.bar(x + w / 2, fb["fallback_critical_rel_pct"], width=w, label="Policy + Adequacy Fallback", color="#38a169")
    ax1.axhline(93.98, color="#e53e3e", linestyle="--", linewidth=1.2, label="Rule-Based EMS (93.98%)")
    ax1.set_ylim(85, 97)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylabel("Critical Reliability (%)")
    ax1.set_title("Fallback Attribution: Critical Reliability (Intrinsic vs +Fallback)")
    ax1.legend(fontsize=8)

    ax2.bar(x - w / 2, fb["intrinsic_social_cost_usd"], width=w, label="Intrinsic Policy", color="#dd6b20")
    ax2.bar(x + w / 2, fb["fallback_social_cost_usd"], width=w, label="Policy + Fallback", color="#805ad5")
    ax2.axhline(4195.06, color="#e53e3e", linestyle="--", linewidth=1.2, label="Rule-Based EMS ($4,195)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylabel("Mean True Social Cost ($)")
    ax2.set_title("Fallback Attribution: Marginal True Social Cost Change")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot2_intrinsic_vs_fallback_attribution.png", dpi=180)
    plt.close()

    # ------------------------------------------------------------------
    # Plot 3: Final Controlled Comparison (Mean & CVaR 10% Social Cost)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5.8))
    x_c = np.arange(len(comp))
    w_c = 0.38
    ax.bar(x_c - w_c / 2, comp["true_social_cost_mean"], width=w_c, label="Mean True Social Cost ($)", color="#2b6cb0")
    ax.bar(x_c + w_c / 2, comp["true_social_cost_cvar10"], width=w_c, label="CVaR 10% Worst-Case Cost ($)", color="#c53030")
    ax.set_xticks(x_c)
    ax.set_xticklabels(comp["controller"], rotation=38, ha="right", fontsize=8)
    ax.set_ylabel("True Social Cost ($ USD)")
    ax.set_title("Phase 4.4.3 Controlled Benchmark: Mean vs CVaR 10% True Social Cost (H1–H6)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot3_true_social_cost_and_heuristics.png", dpi=180)
    plt.close()

    # ------------------------------------------------------------------
    # Plot 4: Policy Autonomy & Intervention Attribution
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    ax1.bar(x_c - w_c / 2, comp["raw_outage_gen_prop_pct"], width=w_c, label="Outage Gen Proposal (%)", color="#319795")
    ax1.bar(x_c + w_c / 2, comp["unneeded_gen_prop_pct"], width=w_c, label="Unneeded Healthy-Grid Gen (%)", color="#e53e3e")
    ax1.set_xticks(x_c)
    ax1.set_xticklabels(comp["controller"], rotation=42, ha="right", fontsize=7.5)
    ax1.set_ylabel("Proportion of Steps (%)")
    ax1.set_title("Generator Dispatch Autonomy vs Healthy-Grid Decision Errors")
    ax1.legend(fontsize=8)

    ax2.bar(x_c, comp["interv_physical_mean"], label="Physical Constraint Clips", color="#718096")
    ax2.bar(x_c, comp["interv_policy_err_mean"], bottom=comp["interv_physical_mean"], label="Policy Decision Error Clips", color="#dd6b20")
    ax2.bar(
        x_c,
        comp["adequacy_intervention_mean"],
        bottom=comp["interv_physical_mean"] + comp["interv_policy_err_mean"],
        label="Adequacy Fallback Interventions",
        color="#3182ce",
    )
    ax2.set_xticks(x_c)
    ax2.set_xticklabels(comp["controller"], rotation=42, ha="right", fontsize=7.5)
    ax2.set_ylabel("Mean Interventions / Episode")
    ax2.set_title("Causal Attribution of Safety & Adequacy Interventions")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot4_policy_autonomy_and_interventions.png", dpi=180)
    plt.close()

    print(f"[+] Generated 4 Phase 4.4.3 plots in {PLOT_DIR}", flush=True)


if __name__ == "__main__":
    generate_plots()
