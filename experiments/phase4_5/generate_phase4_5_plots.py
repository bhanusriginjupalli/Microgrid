"""
experiments/phase4_5/generate_phase4_5_plots.py
================================================
Generates all 10 publication-quality Phase 4.5 X-MARL visualizations:
  1.  plot01_agent_action_timeline.png
  2.  plot02_agent_feature_importance.png
  3.  plot03_reward_component_breakdown.png
  4.  plot04_battery_soc_and_dispatch.png
  5.  plot05_generator_dispatch_and_startup.png
  6.  plot06_safety_intervention_timeline.png
  7.  plot07_intrinsic_vs_hybrid_behavior.png
  8.  plot08_counterfactual_scenario_comparison.png
  9.  plot09_critical_reliability_and_ens.png
  10. plot10_system_level_decision_explanation.png
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
BASE_45 = ROOT / "experiments" / "phase4_5"
RES_DIR = BASE_45 / "results"
PLOT_DIR = BASE_45 / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def generate_plots():
    plt.rcParams.update({"font.size": 9.5, "axes.grid": True, "grid.alpha": 0.3})

    scen_df = pd.read_csv(RES_DIR / "scenario_comparison_traces.csv")
    rew_df = pd.read_csv(RES_DIR / "reward_decomposition_all.csv")
    int_df = pd.read_csv(RES_DIR / "safety_intervention_trace.csv")
    tl_df = pd.read_csv(RES_DIR / "h2_system_timeline.csv")
    attr_df = pd.read_csv(RES_DIR / "feature_attributions_h2.csv")
    cf_df = pd.read_csv(RES_DIR / "counterfactual_results.csv")
    with open(RES_DIR / "h2_agent_decision_explanations.json", "r", encoding="utf-8") as f:
        h2_agents = json.load(f)

    ag_df = pd.DataFrame([
        {
            "controller": r["controller"],
            "step": r["step"],
            "hour": r["hour"],
            "agent": r["agent"],
            "raw": r["raw_policy_output"][0] if isinstance(r["raw_policy_output"], list) else r["raw_policy_output"],
            "exec_bat_mw": r["executed_action"].get("requested_power_mw", 0.0) if r["agent"] == "battery" else 0.0,
            "raw_bat_mw": r["decoded_action"].get("requested_power_mw", 0.0) if r["agent"] == "battery" else 0.0,
            "exec_gen_mw": r["executed_action"].get("requested_power_mw", 0.0) if r["agent"] == "generator" else 0.0,
            "raw_gen_mw": r["decoded_action"].get("requested_power_mw", 0.0) if r["agent"] == "generator" else 0.0,
            "load_tier": r["executed_action"].get("action_idx", 0) if r["agent"] == "load_management" else 0,
            "grid_cmd": r["executed_action"].get("action_idx", 0) if r["agent"] == "grid_resilience" else 0,
            "stage": r["intervention_stage"],
        }
        for r in h2_agents
    ])

    intr_name = "Bat-Only (Seed 202) [Intrinsic]"
    hyb_name = "Bat-Only (Seed 202) [Hybrid +Fallback]"

    # 1. Agent Action Timeline (H2)
    fig, axes = plt.subplots(5, 1, figsize=(13, 10.5), sharex=True)
    tl_i = tl_df[tl_df["controller"] == intr_name].sort_values("step")
    tl_h = tl_df[tl_df["controller"] == hyb_name].sort_values("step")
    axes[0].plot(tl_i["hour"], tl_i["solar_mw"] + tl_i["wind_mw"], color="#27ae60", label="RE Delivered (MW)")
    axes[0].plot(tl_i["hour"], tl_i["nominal_total_mw"], color="#2c3e50", ls="--", label="Total Load (MW)")
    axes[0].set_ylabel("MW")
    axes[0].set_title("1. Multi-Agent Action Timeline on Held-Out H2 (Bat-Only Seed 202: Intrinsic vs Hybrid)")
    axes[0].legend(loc="upper right", fontsize=8)

    ag_bi = ag_df[(ag_df["controller"] == intr_name) & (ag_df["agent"] == "battery")].sort_values("step")
    ag_bh = ag_df[(ag_df["controller"] == hyb_name) & (ag_df["agent"] == "battery")].sort_values("step")
    axes[1].plot(ag_bi["hour"], ag_bi["raw_bat_mw"], color="#95a5a6", ls=":", label="Raw Proposal (MW)")
    axes[1].plot(ag_bi["hour"], tl_i["battery_mw"], color="#e67e22", label="Intrinsic Executed (MW)")
    axes[1].plot(ag_bh["hour"], tl_h["battery_mw"], color="#2980b9", alpha=0.85, label="Hybrid Executed (MW)")
    axes[1].set_ylabel("Battery (MW)")
    axes[1].legend(loc="upper right", fontsize=8)

    ag_gi = ag_df[(ag_df["controller"] == intr_name) & (ag_df["agent"] == "generator")].sort_values("step")
    axes[2].plot(ag_gi["hour"], ag_gi["raw_gen_mw"], color="#95a5a6", ls=":", label="Raw Gen Proposal (MW)")
    axes[2].plot(tl_i["hour"], tl_i["generator_mw"], color="#c0392b", label="Intrinsic Gen Output (MW)")
    axes[2].plot(tl_h["hour"], tl_h["generator_mw"], color="#8e44ad", ls="--", label="Hybrid Gen Output (MW)")
    axes[2].set_ylabel("Generator (MW)")
    axes[2].legend(loc="upper right", fontsize=8)

    ag_li = ag_df[(ag_df["controller"] == intr_name) & (ag_df["agent"] == "load_management")].sort_values("step")
    axes[3].step(ag_li["hour"], ag_li["load_tier"], where="post", color="#d35400", label="Load Curtailment Tier [0..5]")
    axes[3].plot(tl_i["hour"], tl_i["shed_critical_mw"] * 10, color="#c0392b", ls="--", label="Crit Shed x10 (MW)")
    axes[3].set_ylabel("Load Tier")
    axes[3].legend(loc="upper right", fontsize=8)

    axes[4].step(tl_i["hour"], tl_i["grid_connected"].astype(int), where="post", color="#16a085", label="Grid Connected (1/0)")
    axes[4].step(tl_i["hour"], tl_i["grid_available"].astype(int), where="post", color="#7f8c8d", ls=":", label="Grid Available (1/0)")
    axes[4].set_ylabel("Grid Status")
    axes[4].set_xlabel("Hour of Day (H2)")
    axes[4].legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot01_agent_action_timeline.png", dpi=180)
    plt.close(fig)

    # 2. Agent Observation Feature Importance (Top 8 IG per Agent)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8))
    axes_flat = axes.flatten()
    agents_list = ["renewable", "battery", "generator", "load_management", "grid_resilience"]
    for idx, ag in enumerate(agents_list):
        sub = attr_df[attr_df["agent"] == ag].groupby("feature", as_index=False)[["abs_ig", "pert_sensitivity"]].mean()
        top = sub.sort_values("abs_ig", ascending=False).head(8).iloc[::-1]
        y = np.arange(len(top))
        axes_flat[idx].barh(y, top["abs_ig"], color="#2980b9", alpha=0.88, label="Mean |IG|")
        axes_flat[idx].set_yticks(y)
        axes_flat[idx].set_yticklabels(top["feature"], fontsize=8)
        axes_flat[idx].set_title(f"{ag.replace('_', ' ').title()} Top Features")
        axes_flat[idx].set_xlabel("Mean |Integrated Gradient|")
    axes_flat[5].axis("off")
    axes_flat[5].text(0.05, 0.5, "Sensitivity vs. Causality Note:\nIntegrated Gradients measure local\npolicy sensitivity wrt observation\nfeatures on Scenario H2; physical\ncausal outcomes are tested separately\nvia controlled counterfactuals.", fontsize=10, va="center")
    fig.suptitle("2. Agent Observation Feature Importance (Integrated Gradients, Scenario H2)", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot02_agent_feature_importance.png", dpi=180)
    plt.close(fig)

    # 3. Reward Component Breakdown
    r_keys = ["r_cost", "r_crit_ens", "r_flex_ens", "r_other_ens", "r_degrad", "r_curt", "r_reserve", "r_service"]
    r_agg = rew_df.groupby("controller", as_index=False)[r_keys].sum()
    r_agg[r_keys] = r_agg[r_keys] / 6.0  # Mean episode sum across H1-H6
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(r_agg))
    w = 0.10
    for j, k in enumerate(r_keys):
        ax.bar(x + (j - len(r_keys) / 2) * w, r_agg[k], w, label=k)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(r_agg["controller"], rotation=15, ha="right")
    ax.set_ylabel("Mean Episode Reward Contribution (pts)")
    ax.set_title("3. Reward Component Decomposition Across H1–H6 (Mean Episode Sum)")
    ax.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot03_reward_component_breakdown.png", dpi=180)
    plt.close(fig)

    # 4. Battery SOC and Dispatch Decisions (H2)
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 6.8), sharex=True)
    for cname, col, ls in [(intr_name, "#e67e22", "-"), (hyb_name, "#2980b9", "-"), ("Rule-Based EMS", "#27ae60", "--")]:
        sub = tl_df[tl_df["controller"] == cname].sort_values("step")
        axes[0].plot(sub["hour"], sub["battery_soc"], color=col, ls=ls, label=cname)
        axes[1].plot(sub["hour"], sub["battery_mw"], color=col, ls=ls, label=cname)
    axes[0].axhline(30.0, color="#c0392b", ls=":", label="Emergency Reserve (30%)")
    axes[0].axhline(20.0, color="#7f8c8d", ls=":", label="Hard Min SOC (20%)")
    axes[0].set_ylabel("Battery SOC (%)")
    axes[0].set_title("4. Battery SOC Trajectory & Dispatch Decisions on Scenario H2 (+Charge / -Discharge)")
    axes[0].legend(fontsize=8)
    axes[1].axhline(0.0, color="black", lw=0.7)
    axes[1].set_ylabel("Battery Power (MW)")
    axes[1].set_xlabel("Hour of Day")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot04_battery_soc_and_dispatch.png", dpi=180)
    plt.close(fig)

    # 5. Generator Dispatch and Startup Decisions (H2)
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 6.5), sharex=True)
    ag_gh = ag_df[(ag_df["controller"] == hyb_name) & (ag_df["agent"] == "generator")].sort_values("step")
    axes[0].plot(ag_gi["hour"], ag_gi["raw_gen_mw"], color="#95a5a6", ls=":", label="Raw Gen Proposal (MW)")
    axes[0].plot(tl_i["hour"], tl_i["generator_mw"], color="#e67e22", label="Intrinsic Executed (MW)")
    axes[0].plot(tl_h["hour"], tl_h["generator_mw"], color="#8e44ad", ls="--", label="Hybrid Executed (MW)")
    axes[0].axhline(0.05, color="#c0392b", ls=":", label="Shield Healthy-Grid Threshold (0.05 MW)")
    axes[0].set_ylabel("Generator Power (MW)")
    axes[0].set_title("5. Generator Raw Proposals, Shield Clipping, and Fallback Startups on H2")
    axes[0].legend(fontsize=8)

    state_map = {"OFF": 0, "STARTING": 1, "RUNNING": 2, "SHUTTING_DOWN": 3}
    axes[1].step(tl_i["hour"], tl_i["generator_state"].map(state_map), where="post", color="#e67e22", label="Intrinsic State")
    axes[1].step(tl_h["hour"], tl_h["generator_state"].map(state_map), where="post", color="#8e44ad", ls="--", label="Hybrid State")
    axes[1].set_yticks([0, 1, 2, 3])
    axes[1].set_yticklabels(["OFF", "STARTING", "RUNNING", "SHUTTING_DOWN"])
    axes[1].set_xlabel("Hour of Day")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot05_generator_dispatch_and_startup.png", dpi=180)
    plt.close(fig)

    # 6. Safety-Shield & Fallback Intervention Timeline (Counts by Controller & Stage)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    if not int_df.empty:
        piv = int_df.pivot_table(index="controller", columns="intervention_stage", values="step", aggfunc="count", fill_value=0)
        piv.plot(kind="bar", stacked=True, ax=axes[0], colormap="Set2")
        axes[0].set_title("6a. Intervention Counts Across H1–H6 by Stage")
        axes[0].set_ylabel("Total Agent-Step Interventions (6 eps)")
        axes[0].tick_params(axis="x", rotation=18)

    h2_int = int_df[int_df["scenario"] == "scenario_h2"]
    if not h2_int.empty:
        stage_codes = {s: i + 1 for i, s in enumerate(sorted(h2_int["intervention_stage"].unique()))}
        for cname, col in [(intr_name, "#e67e22"), (hyb_name, "#2980b9")]:
            sub = h2_int[h2_int["controller"] == cname]
            axes[1].scatter(sub["hour"], sub["intervention_stage"].map(stage_codes), label=cname, alpha=0.8, s=35)
        axes[1].set_yticks(list(stage_codes.values()))
        axes[1].set_yticklabels(list(stage_codes.keys()), fontsize=8)
        axes[1].set_xlabel("Hour of Day (H2)")
        axes[1].set_title("6b. Safety & Fallback Intervention Timeline on H2")
        axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot06_safety_intervention_timeline.png", dpi=180)
    plt.close(fig)

    # 7. Intrinsic vs Hybrid Policy Behavior Across H1-H6
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    sub_ih = scen_df[scen_df["controller"].isin([intr_name, hyb_name])]
    p_sc = sub_ih.pivot(index="scenario", columns="controller", values="true_social_cost_usd")
    p_sc.plot(kind="bar", ax=axes[0], color=["#2980b9", "#e67e22"])
    axes[0].set_title("7a. True Social Cost: Intrinsic vs Hybrid (H1–H6)")
    axes[0].set_ylabel("True Social Cost ($)")
    axes[0].tick_params(axis="x", rotation=20)

    p_ens = sub_ih.pivot(index="scenario", columns="controller", values="critical_ens_mwh")
    p_ens.plot(kind="bar", ax=axes[1], color=["#2980b9", "#e67e22"])
    axes[1].set_title("7b. Critical ENS: Intrinsic vs Hybrid (H1–H6)")
    axes[1].set_ylabel("Critical ENS (MWh)")
    axes[1].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot07_intrinsic_vs_hybrid_behavior.png", dpi=180)
    plt.close(fig)

    # 8. Counterfactual Scenario Comparison (H2 & H6)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    cf_only = cf_df[cf_df["simulated_counterfactual"] == True]
    p_cf_sc = cf_only.pivot_table(index="counterfactual", columns="mode", values="delta_social_cost_usd", aggfunc="mean")
    p_cf_sc.plot(kind="bar", ax=axes[0], color=["#2980b9", "#e67e22"])
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_title("8a. Simulated Counterfactual Δ Social Cost (vs Factual)")
    axes[0].set_ylabel("Δ Social Cost ($)")
    axes[0].tick_params(axis="x", rotation=25)

    p_cf_ens = cf_only.pivot_table(index="counterfactual", columns="mode", values="delta_critical_ens_mwh", aggfunc="mean")
    p_cf_ens.plot(kind="bar", ax=axes[1], color=["#2980b9", "#e67e22"])
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("8b. Simulated Counterfactual Δ Critical ENS (vs Factual)")
    axes[1].set_ylabel("Δ Critical ENS (MWh)")
    axes[1].tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot08_counterfactual_scenario_comparison.png", dpi=180)
    plt.close(fig)

    # 9. Critical-Load Reliability and ENS Across All Comparison Controllers (H1-H6)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))
    c_agg = scen_df.groupby("controller", as_index=False)[["critical_reliability_pct", "critical_ens_mwh"]].mean()
    y = np.arange(len(c_agg))
    axes[0].barh(y, c_agg["critical_reliability_pct"], color="#27ae60", alpha=0.85)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(c_agg["controller"], fontsize=8.5)
    axes[0].set_xlim(88, 96)
    axes[0].set_xlabel("Mean Critical Reliability (%)")
    axes[0].set_title("9a. Mean Critical Reliability Across H1–H6")

    axes[1].barh(y, c_agg["critical_ens_mwh"], color="#c0392b", alpha=0.85)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(["" for _ in y])
    axes[1].set_xlabel("Mean Critical ENS (MWh)")
    axes[1].set_title("9b. Mean Critical ENS Across H1–H6")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot09_critical_reliability_and_ens.png", dpi=180)
    plt.close(fig)

    # 10. System-Level Decision Explanation Narrative (H2)
    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    ax.plot(tl_h["hour"], tl_h["nominal_total_mw"], color="#2c3e50", lw=1.8, label="Nominal Demand (MW)")
    ax.plot(tl_h["hour"], tl_h["solar_mw"] + tl_h["wind_mw"], color="#27ae60", lw=1.5, label="Renewable Gen (MW)")
    ax.plot(tl_h["hour"], tl_h["generator_mw"], color="#8e44ad", lw=1.5, label="Hybrid Generator (MW)")
    ax.plot(tl_h["hour"], -tl_h["battery_mw"], color="#2980b9", lw=1.5, label="Hybrid Battery Disch. (MW)")
    outage_mask = ~tl_h["grid_connected"]
    ax.fill_between(tl_h["hour"], 0, tl_h["nominal_total_mw"].max() * 1.15, where=outage_mask, color="#e74c3c", alpha=0.14, label="Grid Outage Window")

    # Annotate key explainable events
    out_steps = tl_h[outage_mask]
    if not out_steps.empty:
        h_mid = float(out_steps["hour"].iloc[len(out_steps) // 2])
        ax.annotate("Outage & Fallback:\nGen + Bat Disch.\nprotect critical load", xy=(h_mid, 0.55), xytext=(h_mid - 3.5, 0.92),
                    arrowprops=dict(arrowstyle="->", color="#c0392b"), fontsize=8.5, bbox=dict(boxstyle="round", fc="#fff3cd", ec="#c0392b"))
    healthy_clip = ag_df[(ag_df["controller"] == intr_name) & (ag_df["agent"] == "generator") & (ag_df["stage"] == "BOUNDARY_CLIP")]
    if not healthy_clip.empty:
        hc_hr = float(healthy_clip["hour"].iloc[0])
        ax.annotate("Healthy Grid:\nSafetyShield clips\nunneeded Gen >0.05MW", xy=(hc_hr, 0.05), xytext=(hc_hr + 0.8, 0.42),
                    arrowprops=dict(arrowstyle="->", color="#7f8c8d"), fontsize=8.5, bbox=dict(boxstyle="round", fc="#ecf0f1", ec="#7f8c8d"))

    ax.set_xlabel("Hour of Day (Scenario H2)")
    ax.set_ylabel("Power (MW)")
    ax.set_title("10. System-Level Decision Explanation Diagram (Scenario H2, Bat-Only Seed 202)")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot10_system_level_decision_explanation.png", dpi=180)
    plt.close(fig)

    print(f"[+] Generated 10 Phase 4.5 X-MARL plots in {PLOT_DIR}", flush=True)


if __name__ == "__main__":
    generate_plots()
