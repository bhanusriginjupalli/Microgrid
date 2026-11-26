"""
experiments/phase4_4_4/generate_phase4_4_4_plots.py
===================================================
Generates Phase 4.4.4 publication figures:
  1. plot1_ablation_training_convergence.png
  2. plot2_generator_autonomy_remediation.png
  3. plot3_battery_deficit_and_reserve.png
  4. plot4_intrinsic_vs_hybrid_attribution.png
  5. plot5_social_cost_and_heuristics.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
BASE_444 = ROOT / "experiments" / "phase4_4_4"
RES_DIR = BASE_444 / "results"
PLOT_DIR = BASE_444 / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def generate_plots():
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

    comp = pd.read_csv(RES_DIR / "controller_comparison.csv")
    fb = pd.read_csv(RES_DIR / "fallback_attribution.csv")
    val_h = pd.read_csv(RES_DIR / "validation_histories.csv")
    train_h = pd.read_csv(RES_DIR / "training_histories.csv")

    # ------------------------------------------------------------------
    # Plot 1: Ablation Training & Validation Convergence
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    colors = {"baseline": "#7f8c8d", "gen_only": "#2980b9", "bat_only": "#27ae60", "combined": "#8e44ad"}

    for exp, c in colors.items():
        v_sub = val_h[val_h["experiment"] == exp].groupby("episode", as_index=False)[
            ["val_true_social_cost_usd", "val_critical_reliability_pct"]
        ].agg(["mean", "std"])
        eps = v_sub["episode"]
        m_sc = v_sub[("val_true_social_cost_usd", "mean")]
        s_sc = v_sub[("val_true_social_cost_usd", "std")].fillna(0)
        axes[0, 0].plot(eps, m_sc, marker="o", color=c, label=exp)
        axes[0, 0].fill_between(eps, m_sc - s_sc, m_sc + s_sc, color=c, alpha=0.15)

        m_rel = v_sub[("val_critical_reliability_pct", "mean")]
        s_rel = v_sub[("val_critical_reliability_pct", "std")].fillna(0)
        axes[0, 1].plot(eps, m_rel, marker="o", color=c, label=exp)
        axes[0, 1].fill_between(eps, m_rel - s_rel, m_rel + s_rel, color=c, alpha=0.15)

        t_sub = train_h[train_h["experiment"] == exp].copy()
        t_sub["bin"] = ((t_sub["episode"] - 1) // 10) * 10 + 5
        t_g = t_sub.groupby("bin", as_index=False)[["unneeded_gen_steps", "deficit_bat_dis_pct"]].mean()
        axes[1, 0].plot(t_g["bin"], t_g["unneeded_gen_steps"], marker="s", color=c, label=exp)
        axes[1, 1].plot(t_g["bin"], t_g["deficit_bat_dis_pct"], marker="s", color=c, label=exp)

    axes[0, 0].set_title("Fixed Validation: True Social Cost (Mean ± SD)")
    axes[0, 0].set_ylabel("Social Cost ($)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_title("Fixed Validation: Critical Reliability (%)")
    axes[0, 1].set_ylabel("Reliability (%)")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].set_title("Training: Unnecessary Healthy-Grid Gen Steps/Ep")
    axes[1, 0].set_xlabel("Training Episode")
    axes[1, 0].set_ylabel("Steps / Episode")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].set_title("Training: Genuine Deficit Battery Discharge (%)")
    axes[1, 1].set_xlabel("Training Episode")
    axes[1, 1].set_ylabel("Deficit Discharge Rate (%)")
    axes[1, 1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot1_ablation_training_convergence.png", dpi=180)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 2: Generator Behavior Across Ablations (Intrinsic Mode)
    # ------------------------------------------------------------------
    intr = comp[comp["adequacy_fallback"] == False].copy()
    abls = ["Heuristic", "P4.4.3 Corrected", "4.4.4 Baseline", "4.4.4 Gen-Only", "4.4.4 Bat-Only", "4.4.4 Combined"]
    g_agg = intr[intr["ablation"].isin(abls)].groupby("ablation", as_index=False).agg(
        unneeded=("gen_unnecessary_steps_mean", "mean"),
        nec_rate=("gen_necessary_rate_pct", "mean"),
        feas=("unshielded_feasible_pct", "mean"),
    )
    g_agg["order"] = g_agg["ablation"].apply(lambda x: abls.index(x))
    g_agg = g_agg.sort_values("order")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    x = np.arange(len(g_agg))
    axes[0].bar(x, g_agg["unneeded"], color="#e74c3c", alpha=0.85)
    axes[0].set_title("Unnecessary Healthy-Grid Gen Proposals/Ep")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(g_agg["ablation"], rotation=28, ha="right", fontsize=8)
    axes[0].set_ylabel("Steps / Ep")

    axes[1].bar(x, g_agg["nec_rate"], color="#2980b9", alpha=0.85)
    axes[1].set_title("Necessary Outage Gen Dispatch Rate (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(g_agg["ablation"], rotation=28, ha="right", fontsize=8)
    axes[1].set_ylim(0, 108)

    axes[2].bar(x, g_agg["feas"], color="#27ae60", alpha=0.85)
    axes[2].set_title("Raw Unshielded Proposal Feasibility (%)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(g_agg["ablation"], rotation=28, ha="right", fontsize=8)
    axes[2].set_ylim(50, 105)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot2_generator_autonomy_remediation.png", dpi=180)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 3: Battery Deficit Discharge Per Seed & Healthy-Grid Behavior
    # ------------------------------------------------------------------
    sub_bat = intr[intr["ablation"].isin(["P4.4.3 Corrected", "4.4.4 Baseline", "4.4.4 Gen-Only", "4.4.4 Bat-Only", "4.4.4 Combined"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    pivot_def = sub_bat.pivot(index="ablation", columns="train_seed", values="raw_deficit_bat_dis_pct").reindex(
        ["P4.4.3 Corrected", "4.4.4 Baseline", "4.4.4 Gen-Only", "4.4.4 Bat-Only", "4.4.4 Combined"]
    )
    pivot_def.plot(kind="bar", ax=axes[0], width=0.72)
    axes[0].set_title("Intrinsic Deficit Battery Discharge (%) by Train Seed")
    axes[0].set_ylabel("Deficit Discharge Response (%)")
    axes[0].set_ylim(0, 108)
    axes[0].tick_params(axis="x", rotation=20)

    pivot_h = sub_bat.pivot(index="ablation", columns="train_seed", values="healthy_bat_dis_pct").reindex(
        ["P4.4.3 Corrected", "4.4.4 Baseline", "4.4.4 Gen-Only", "4.4.4 Bat-Only", "4.4.4 Combined"]
    )
    pivot_h.plot(kind="bar", ax=axes[1], width=0.72)
    axes[1].set_title("Intrinsic Healthy-Grid Battery Discharge Frequency (%)")
    axes[1].set_ylabel("Healthy-Grid Discharge Steps (%)")
    axes[1].set_ylim(0, 108)
    axes[1].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot3_battery_deficit_and_reserve.png", dpi=180)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 4: Intrinsic vs Hybrid Fallback Attribution
    # ------------------------------------------------------------------
    fb_agg = fb.groupby("ablation", as_index=False)[
        ["intrinsic_critical_ens_mwh", "hybrid_critical_ens_mwh", "intrinsic_social_cost_usd", "hybrid_social_cost_usd", "adequacy_interventions_per_ep"]
    ].mean()
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    x2 = np.arange(len(fb_agg))
    w = 0.36
    axes[0].bar(x2 - w / 2, fb_agg["intrinsic_critical_ens_mwh"], w, label="Intrinsic", color="#e67e22")
    axes[0].bar(x2 + w / 2, fb_agg["hybrid_critical_ens_mwh"], w, label="Hybrid (+Fallback)", color="#2ecc71")
    axes[0].set_title("Critical ENS: Intrinsic vs Hybrid")
    axes[0].set_xticks(x2)
    axes[0].set_xticklabels(fb_agg["ablation"], rotation=25, ha="right", fontsize=8)
    axes[0].set_ylabel("MWh / Ep")
    axes[0].legend(fontsize=8)

    axes[1].bar(x2 - w / 2, fb_agg["intrinsic_social_cost_usd"], w, label="Intrinsic", color="#e67e22")
    axes[1].bar(x2 + w / 2, fb_agg["hybrid_social_cost_usd"], w, label="Hybrid (+Fallback)", color="#2ecc71")
    axes[1].set_title("True Social Cost: Intrinsic vs Hybrid")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(fb_agg["ablation"], rotation=25, ha="right", fontsize=8)
    axes[1].set_ylabel("USD / Ep")
    axes[1].legend(fontsize=8)

    axes[2].bar(x2, fb_agg["adequacy_interventions_per_ep"], color="#8e44ad", alpha=0.85)
    axes[2].set_title("Adequacy Fallback Interventions / Ep")
    axes[2].set_xticks(x2)
    axes[2].set_xticklabels(fb_agg["ablation"], rotation=25, ha="right", fontsize=8)
    axes[2].set_ylabel("Steps / Ep")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot4_intrinsic_vs_hybrid_attribution.png", dpi=180)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 5: Family Social Cost & Reliability vs Heuristics
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.0))
    fam_order = [
        ("Greedy Controller", False),
        ("Rule-Based EMS", False),
        ("Lookahead Heuristic EMS", False),
        ("Robust Forecast MAPPO", False),
        ("P4.4.3 Corrected", False),
        ("P4.4.3 Corrected", True),
        ("4.4.4 Baseline", False),
        ("4.4.4 Gen-Only", False),
        ("4.4.4 Bat-Only", False),
        ("4.4.4 Combined", False),
        ("4.4.4 Combined", True),
    ]
    labels, sc_means, sc_stds, rel_means, bar_cols = [], [], [], [], []
    for abl, fb_flag in fam_order:
        if abl in ("Greedy Controller", "Rule-Based EMS", "Lookahead Heuristic EMS"):
            sub = comp[comp["controller"] == abl]
            lbl = abl
            col = "#2ecc71"
        elif abl == "Robust Forecast MAPPO":
            sub = comp[comp["controller"].str.startswith("Robust Forecast MAPPO")]
            lbl = "Robust 4.3 (Intr)"
            col = "#95a5a6"
        else:
            sub = comp[(comp["ablation"] == abl) & (comp["adequacy_fallback"] == fb_flag)]
            lbl = f"{abl} ({'Hyb' if fb_flag else 'Intr'})"
            col = "#3498db" if not fb_flag else "#9b59b6"
        labels.append(lbl)
        sc_means.append(float(sub["true_social_cost_mean"].mean()))
        sc_stds.append(float(sub["true_social_cost_mean"].std()) if len(sub) > 1 else 0.0)
        rel_means.append(float(sub["critical_reliability_mean"].mean()))
        bar_cols.append(col)

    y = np.arange(len(labels))
    axes[0].barh(y, sc_means, xerr=sc_stds, color=bar_cols, alpha=0.88, capsize=3)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("True Social Cost Mean ± Seed SD ($/ep)")
    axes[0].set_title("Held-Out True Social Cost vs. Heuristics")

    axes[1].barh(y, rel_means, color=bar_cols, alpha=0.88)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(["" for _ in labels])
    axes[1].invert_yaxis()
    axes[1].set_xlim(88, 96)
    axes[1].set_xlabel("Critical-Load Reliability Mean (%)")
    axes[1].set_title("Held-Out Critical Reliability (%)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot5_social_cost_and_heuristics.png", dpi=180)
    plt.close(fig)

    print(f"[+] Generated 5 Phase 4.4.4 plots in {PLOT_DIR}", flush=True)


if __name__ == "__main__":
    generate_plots()
