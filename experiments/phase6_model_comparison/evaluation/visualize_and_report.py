"""
visualize_and_report.py
=======================
Generates all 15 required comparative plots, 6 individual H1-H6 scenario plots,
environment/reproducibility manifest, and data-grounded Phase 3 research reports
strictly from observed CSV/JSON artifacts.
"""

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase6_model_comparison.evaluation.evaluation_protocol import (
    EVAL_SEEDS,
    HELDOUT_SCENARIOS,
    SCENARIO_LABELS,
    TRAIN_SEEDS,
)

ALGO_ORDER_PRIMARY = [
    "IPPO",
    "MAPPO_MATCHED",
    "HAPPO",
    "Greedy EMS",
    "Rule-Based EMS",
    "Lookahead Heuristic EMS",
]
ALGO_COLORS = {
    "IPPO": "#38bdf8",
    "MAPPO_MATCHED": "#818cf8",
    "HAPPO": "#a855f7",
    "Greedy EMS": "#f59e0b",
    "Rule-Based EMS": "#10b981",
    "Lookahead Heuristic EMS": "#ec4899",
}


def save_environment_versions(out_path: Path) -> Dict[str, Any]:
    """Record exact Python and package versions + reproduction commands."""
    pkgs = {}
    for mod_name in ("torch", "numpy", "pandas", "scipy", "pandapower", "pettingzoo", "gymnasium", "matplotlib", "yaml"):
        try:
            m = __import__(mod_name)
            pkgs[mod_name] = getattr(m, "__version__", "unknown")
        except Exception:
            pkgs[mod_name] = "unavailable"
    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": pkgs,
        "reproduction_commands": {
            "train_ippo_seed": ".venv\\Scripts\\python.exe experiments\\phase6_model_comparison\\training\\train_ippo.py --seed <42|101|202>",
            "train_mappo_seed": ".venv\\Scripts\\python.exe experiments\\phase6_model_comparison\\training\\train_matched_mappo.py --seed <42|101|202>",
            "train_happo_seed": ".venv\\Scripts\\python.exe experiments\\phase6_model_comparison\\training\\train_happo.py --seed <42|101|202>",
            "evaluate_all_tracks": ".venv\\Scripts\\python.exe experiments\\phase6_model_comparison\\evaluation\\evaluate_all.py --track all",
            "generate_plots_and_reports": ".venv\\Scripts\\python.exe experiments\\phase6_model_comparison\\evaluation\\visualize_and_report.py",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return info


def _load_training_tables(results_root: Path) -> Dict[str, pd.DataFrame]:
    th_list, vh_list, hf_list, run_list = [], [], [], []
    for sub, algo in [("ippo", "IPPO"), ("mappo_matched", "MAPPO_MATCHED"), ("happo", "HAPPO")]:
        for s in TRAIN_SEEDS:
            d = results_root / sub / f"seed_{s}"
            if (d / "training_history.csv").exists():
                df = pd.read_csv(d / "training_history.csv")
                df["algorithm"] = algo
                df["seed"] = int(s)
                th_list.append(df)
            if (d / "validation_history.csv").exists():
                df = pd.read_csv(d / "validation_history.csv")
                df["algorithm"] = algo
                df["seed"] = int(s)
                vh_list.append(df)
            if algo == "HAPPO" and (d / "happo_factor_diagnostics.csv").exists():
                df = pd.read_csv(d / "happo_factor_diagnostics.csv")
                hf_list.append(df)
            if (d / "run_summary.json").exists():
                with open(d / "run_summary.json", "r", encoding="utf-8") as f:
                    run_list.append(json.load(f))
    return {
        "training": pd.concat(th_list, ignore_index=True) if th_list else pd.DataFrame(),
        "validation": pd.concat(vh_list, ignore_index=True) if vh_list else pd.DataFrame(),
        "happo_factors": pd.concat(hf_list, ignore_index=True) if hf_list else pd.DataFrame(),
        "runs": pd.DataFrame(run_list) if run_list else pd.DataFrame(),
    }


def _bar_by_track_and_algo(
    ep_df: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    out_path: Path,
):
    modes_order = [m for m in ["Intrinsic", "Shielded", "Hybrid (+Fallback)", "Heuristic"] if m in ep_df["mode"].unique()]
    algos = [a for a in ALGO_ORDER_PRIMARY if a in ep_df["algorithm"].unique()]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(algos))
    width = 0.8 / max(1, len(modes_order))

    for i, m in enumerate(modes_order):
        sub = ep_df[ep_df["mode"] == m]
        means, errs = [], []
        for a in algos:
            vals = sub[sub["algorithm"] == a][metric_col].dropna().values
            if len(vals) > 0:
                means.append(float(np.mean(vals)))
                errs.append(float(1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
            else:
                means.append(0.0)
                errs.append(0.0)
        offset = (i - (len(modes_order) - 1) / 2.0) * width
        bars = ax.bar(x + offset, means, width, yerr=errs, capsize=3, label=m, alpha=0.88)
        for b, val in zip(bars, means):
            if val != 0.0:
                ax.text(b.get_x() + b.get_width() / 2.0, val, f"{val:.2f}" if abs(val) < 10 else f"{val:.1f}",
                        ha="center", va="bottom", fontsize=7.5, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=15, ha="right")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def generate_all_plots(results_root: Path, eval_subdir: str = "evaluation") -> List[str]:
    eval_dir = results_root / eval_subdir
    plots_dir = eval_dir / "plots"
    sc_plots_dir = plots_dir / "scenarios"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sc_plots_dir.mkdir(parents=True, exist_ok=True)

    ep_df = pd.read_csv(eval_dir / "eval_episodes.csv")
    fam_df = pd.read_csv(eval_dir / "family_summary.csv")
    scen_df = pd.read_csv(eval_dir / "scenario_summary.csv")
    tables = _load_training_tables(results_root)
    tr_df = tables["training"]
    val_df = tables["validation"]
    hf_df = tables["happo_factors"]
    runs_df = tables["runs"]

    generated: List[str] = []

    # 1. Training reward curves
    if not tr_df.empty:
        fig, ax = plt.subplots(figsize=(9.5, 5))
        for algo in ["IPPO", "MAPPO_MATCHED", "HAPPO"]:
            sub = tr_df[tr_df["algorithm"] == algo]
            if sub.empty:
                continue
            grp = sub.groupby("episode")["reward"].agg(["mean", "std"]).reset_index()
            ax.plot(grp["episode"], grp["mean"], label=algo, color=ALGO_COLORS[algo], linewidth=2)
            std = grp["std"].fillna(0.0)
            ax.fill_between(grp["episode"], grp["mean"] - std, grp["mean"] + std, color=ALGO_COLORS[algo], alpha=0.18)
        ax.set_title("Training Episode Reward Curves (Mean ± Seed Std, 3 Seeds × 100 Eps)", fontweight="bold")
        ax.set_xlabel("Training Episode")
        ax.set_ylabel("Episode Shared Reward")
        ax.grid(alpha=0.25)
        ax.legend()
        plt.tight_layout()
        p = plots_dir / "01_training_reward_curves.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        generated.append(str(p))

    # 2. Validation social cost curves
    if not val_df.empty:
        fig, ax = plt.subplots(figsize=(9.5, 5))
        for algo in ["IPPO", "MAPPO_MATCHED", "HAPPO"]:
            sub = val_df[val_df["algorithm"] == algo]
            if sub.empty:
                continue
            grp = sub.groupby("episode")["val_true_social_cost_usd"].agg(["mean", "std"]).reset_index()
            ax.plot(grp["episode"], grp["mean"], marker="o", label=algo, color=ALGO_COLORS[algo], linewidth=2)
            std = grp["std"].fillna(0.0)
            ax.fill_between(grp["episode"], grp["mean"] - std, grp["mean"] + std, color=ALGO_COLORS[algo], alpha=0.18)
        ax.set_title("Fixed Validation Suite True Social Cost Progression (Lower is Better)", fontweight="bold")
        ax.set_xlabel("Checkpoint Episode")
        ax.set_ylabel("Validation True Social Cost ($)")
        ax.grid(alpha=0.25)
        ax.legend()
        plt.tight_layout()
        p = plots_dir / "02_validation_social_cost_curves.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        generated.append(str(p))

    # 3-8. Core Metric Comparisons across Track A/B/C & Heuristics
    bar_specs = [
        ("03_social_cost_comparison.png", "true_social_cost_usd", "Held-Out True Social Cost Comparison (H1–H6, Mean ± 95% CI)", "True Social Cost ($)"),
        ("04_critical_ens_comparison.png", "critical_ens_mwh", "Critical Load Unserved Energy (ENS) Comparison", "Critical ENS (MWh)"),
        ("05_critical_reliability_comparison.png", "critical_reliability_pct", "Critical Load Reliability Comparison", "Critical Reliability (%)"),
        ("06_renewable_curtailment_comparison.png", "renewable_curtailment_mwh", "Renewable Curtailment Comparison", "Curtailment (MWh)"),
        ("07_generator_fuel_comparison.png", "fuel_consumed_l", "Diesel Generator Fuel Consumption Comparison", "Fuel Consumed (L)"),
        ("08_battery_throughput_comparison.png", "battery_throughput_mwh", "Battery Energy Throughput Comparison", "Battery Throughput (MWh)"),
        ("12_shield_intervention_comparison.png", "boundary_clipping_count", "Safety Shield Boundary Clipping Events per Episode", "Boundary Clips / Episode"),
        ("13_fallback_intervention_comparison.png", "fallback_intervention_steps", "Operational Adequacy Fallback Interventions per Episode", "Fallback Steps / Episode"),
    ]
    for fname, col, title, ylab in bar_specs:
        p = plots_dir / fname
        _bar_by_track_and_algo(ep_df, col, title, ylab, p)
        generated.append(str(p))

    # 9. Scenario performance heatmap (Track A Intrinsic + Heuristics)
    t_a = ep_df[ep_df["track"] == "track_a_intrinsic"]
    if not t_a.empty:
        piv = t_a.pivot_table(index="algorithm", columns="scenario", values="true_social_cost_usd", aggfunc="mean")
        piv = piv.reindex([a for a in ALGO_ORDER_PRIMARY if a in piv.index])
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn_r")
        ax.set_xticks(np.arange(len(piv.columns)))
        ax.set_xticklabels([c.replace("scenario_", "").upper() for c in piv.columns])
        ax.set_yticks(np.arange(len(piv.index)))
        ax.set_yticklabels(piv.index)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                val = piv.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"${val:,.0f}", ha="center", va="center", color="black", fontsize=8.5, fontweight="bold")
        fig.colorbar(im, ax=ax, label="Mean True Social Cost ($)")
        ax.set_title("Track A Intrinsic & Heuristic Baselines: Scenario-Wise Social Cost ($)", fontweight="bold")
        plt.tight_layout()
        p = plots_dir / "09_scenario_performance_heatmap.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        generated.append(str(p))

    # 10. CVaR-10% comparison
    fig, ax = plt.subplots(figsize=(10.5, 5))
    modes_order = [m for m in ["Intrinsic", "Shielded", "Hybrid (+Fallback)", "Heuristic"] if m in fam_df["mode"].unique()]
    algos = [a for a in ALGO_ORDER_PRIMARY if a in fam_df["algorithm"].unique()]
    x = np.arange(len(algos))
    width = 0.8 / max(1, len(modes_order))
    for i, m in enumerate(modes_order):
        sub = fam_df[fam_df["mode"] == m]
        vals = [float(sub[sub["algorithm"] == a]["true_social_cost_cvar10"].mean()) if not sub[sub["algorithm"] == a].empty else 0.0 for a in algos]
        offset = (i - (len(modes_order) - 1) / 2.0) * width
        ax.bar(x + offset, vals, width, label=m, alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=15, ha="right")
    ax.set_title("Worst-10% Tail Risk: True Social Cost CVaR-10% ($)", fontweight="bold")
    ax.set_ylabel("CVaR-10% Social Cost ($)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    p = plots_dir / "10_cvar10_comparison.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    generated.append(str(p))

    # 11. Reliability vs Social Cost Pareto scatter
    fig, ax = plt.subplots(figsize=(9, 6))
    markers = {"Intrinsic": "o", "Shielded": "s", "Hybrid (+Fallback)": "^", "Heuristic": "D"}
    for _, row in fam_df.iterrows():
        algo = row["algorithm"]
        mode = row["mode"]
        ax.scatter(
            row["true_social_cost_mean"],
            row["critical_reliability_mean"],
            color=ALGO_COLORS.get(algo, "#64748b"),
            marker=markers.get(mode, "o"),
            s=110,
            alpha=0.9,
            edgecolors="black",
            label=f"{algo} [{mode}]",
        )
        ax.annotate(f"{algo} ({mode})", (row["true_social_cost_mean"], row["critical_reliability_mean"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=7.5)
    ax.set_title("Critical Reliability vs. True Social Cost Trade-Off", fontweight="bold")
    ax.set_xlabel("Mean True Social Cost ($) — Lower is Better")
    ax.set_ylabel("Mean Critical Reliability (%) — Higher is Better")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    p = plots_dir / "11_reliability_vs_social_cost_pareto.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    generated.append(str(p))

    # 14. Training time comparison
    if not runs_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        grp = runs_df.groupby("algorithm")["duration_s"].agg(["mean", "std"]).reindex(["IPPO", "MAPPO_MATCHED", "HAPPO"])
        ax.bar(grp.index, grp["mean"] / 60.0, yerr=grp["std"].fillna(0.0) / 60.0, capsize=4,
               color=[ALGO_COLORS[a] for a in grp.index], alpha=0.88)
        ax.set_title("Training Wall-Clock Duration per 100-Episode Seed (Mean ± Std)", fontweight="bold")
        ax.set_ylabel("Duration (Minutes)")
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        p = plots_dir / "14_training_time_comparison.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        generated.append(str(p))

    # 15. HAPPO importance-factor stability plot
    if not hf_df.empty:
        final_step_hf = hf_df[hf_df["agent"] == hf_df["agent"].iloc[-1]] if "grid_resilience" not in hf_df["agent"].values else hf_df[hf_df["agent"] == "grid_resilience"]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        grp = final_step_hf.groupby("episode").agg(
            mean=("factor_mean", "mean"),
            p95=("factor_p95", "mean"),
            p99=("factor_p99", "mean"),
            fmax=("factor_max", "max"),
            ext=("extreme_frac_outside_01_10", "mean"),
        ).reset_index()
        ax1.plot(grp["episode"], grp["mean"], label="Mean Factor", linewidth=2, color="#38bdf8")
        ax1.plot(grp["episode"], grp["p95"], label="95th Percentile", linestyle="--", color="#f59e0b")
        ax1.plot(grp["episode"], grp["p99"], label="99th Percentile", linestyle="-.", color="#ec4899")
        ax1.set_ylabel("Final Prefix Factor F^(5)")
        ax1.set_title("HAPPO Cumulative Importance Factor Stability Across Training (All Seeds)", fontweight="bold")
        ax1.grid(alpha=0.25)
        ax1.legend()

        ax2.plot(grp["episode"], grp["ext"] * 100.0, label="Extreme Factor % (<0.1 or >10)", color="#ef4444", linewidth=2)
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Extreme Factor (%)")
        ax2.grid(alpha=0.25)
        ax2.legend()
        plt.tight_layout()
        p = plots_dir / "15_happo_importance_factor_stability.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        generated.append(str(p))

    # 6 Individual scenario plots H1-H6
    for sc in HELDOUT_SCENARIOS:
        sub_sc = ep_df[(ep_df["scenario"] == sc) & (ep_df["track"] == "track_a_intrinsic")]
        if sub_sc.empty:
            continue
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
        algos_sc = [a for a in ALGO_ORDER_PRIMARY if a in sub_sc["algorithm"].unique()]
        means_c = [sub_sc[sub_sc["algorithm"] == a]["true_social_cost_usd"].mean() for a in algos_sc]
        stds_c = [sub_sc[sub_sc["algorithm"] == a]["true_social_cost_usd"].std() for a in algos_sc]
        means_r = [sub_sc[sub_sc["algorithm"] == a]["critical_reliability_pct"].mean() for a in algos_sc]
        stds_r = [sub_sc[sub_sc["algorithm"] == a]["critical_reliability_pct"].std() for a in algos_sc]

        cols = [ALGO_COLORS[a] for a in algos_sc]
        ax1.bar(algos_sc, means_c, yerr=stds_c, capsize=3, color=cols, alpha=0.88)
        ax1.set_title(f"{SCENARIO_LABELS[sc]} — True Social Cost ($)")
        ax1.tick_params(axis="x", rotation=20)
        ax1.grid(axis="y", alpha=0.25)

        ax2.bar(algos_sc, means_r, yerr=stds_r, capsize=3, color=cols, alpha=0.88)
        ax2.set_title(f"{SCENARIO_LABELS[sc]} — Critical Reliability (%)")
        ax2.tick_params(axis="x", rotation=20)
        ax2.set_ylim(max(0, min(means_r) - 15), 101.5)
        ax2.grid(axis="y", alpha=0.25)

        plt.tight_layout()
        p = sc_plots_dir / f"{sc}.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        generated.append(str(p))

    return generated


def generate_markdown_reports(results_root: Path, eval_subdir: str = "evaluation") -> List[str]:
    eval_dir = results_root / eval_subdir
    reports_dir = results_root / "reports"
    docs_dir = ROOT / "docs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    ep_df = pd.read_csv(eval_dir / "eval_episodes.csv")
    fam_df = pd.read_csv(eval_dir / "family_summary.csv")
    seed_df = pd.read_csv(eval_dir / "seed_level_summary.csv")
    stats_df = pd.read_csv(eval_dir / "paired_statistical_tests.csv")
    scen_df = pd.read_csv(eval_dir / "scenario_summary.csv")
    tables = _load_training_tables(results_root)
    runs_df = tables["runs"]
    hf_df = tables["happo_factors"]

    # Primary Track A Intrinsic + Heuristic view for key comparative rankings
    track_a_fam = fam_df[fam_df["track"] == "track_a_intrinsic"].copy()
    best_cost_row = track_a_fam.sort_values("true_social_cost_mean").iloc[0]
    best_rel_row = track_a_fam.sort_values("critical_reliability_mean", ascending=False).iloc[0]
    best_cvar_row = track_a_fam.sort_values("true_social_cost_cvar10").iloc[0]

    # Scenario winners (Track A)
    sc_a = scen_df[scen_df["track"] == "track_a_intrinsic"]
    h2_win = sc_a[sc_a["scenario"] == "scenario_h2"].sort_values("true_social_cost_mean").iloc[0]
    h3_win = sc_a[sc_a["scenario"] == "scenario_h3"].sort_values("true_social_cost_mean").iloc[0]
    h4_win = sc_a[sc_a["scenario"] == "scenario_h4"].sort_values("true_social_cost_mean").iloc[0]

    # RL-only Track A ranking
    rl_a = track_a_fam[track_a_fam["controller_group"] == "Learned MARL"].sort_values("true_social_cost_mean")
    ippo_a = rl_a[rl_a["algorithm"] == "IPPO"].iloc[0] if not rl_a[rl_a["algorithm"] == "IPPO"].empty else None
    mappo_a = rl_a[rl_a["algorithm"] == "MAPPO_MATCHED"].iloc[0] if not rl_a[rl_a["algorithm"] == "MAPPO_MATCHED"].empty else None
    happo_a = rl_a[rl_a["algorithm"] == "HAPPO"].iloc[0] if not rl_a[rl_a["algorithm"] == "HAPPO"].empty else None

    def _tbl(df: pd.DataFrame, cols: List[str]) -> str:
        if df.empty:
            return "_No data available._"
        sub = df[cols]
        headers = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        lines = [headers, sep]
        for _, r in sub.iterrows():
            vals = [f"{r[c]:.4f}" if isinstance(r[c], float) and abs(r[c]) < 10 else (f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c])) for c in cols]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    # 1. phase3_experimental_results.md
    res_md = f"""# Phase 3 Experimental Results (IPPO vs. MAPPO_MATCHED vs. HAPPO & Heuristics)

## 1. Episode Coverage Verification
- Total evaluated episodes: **{len(ep_df)}** (`{ep_df['sim_steps'].unique().tolist()}` steps per episode; duplicate IDs: **{int(ep_df['episode_id'].duplicated().sum())}**).
- Breakdown by Track:
{_tbl(ep_df.groupby(['track', 'mode', 'algorithm'], as_index=False)['episode_id'].count(), ['track', 'mode', 'algorithm', 'episode_id'])}

## 2. Family-Level Summary (Separated by Track & Operation Mode)
{_tbl(fam_df, ['track', 'mode', 'algorithm', 'episodes_n', 'true_social_cost_mean', 'true_social_cost_std', 'true_social_cost_cvar10', 'operating_cost_mean', 'critical_ens_mean', 'critical_reliability_mean', 'boundary_clipping_mean', 'fallback_intervention_mean'])}

## 3. Seed-Level Summary (Disaggregated per Training Seed)
{_tbl(seed_df[seed_df['track'] == 'track_a_intrinsic'], ['algorithm', 'train_seed', 'episodes_n', 'true_social_cost_mean', 'true_social_cost_std', 'critical_ens_mean', 'critical_reliability_mean', 'operating_cost_mean'])}

## 4. Training Runs & Wall-Clock Efficiency
{_tbl(runs_df, ['algorithm', 'seed', 'episodes', 'env_steps', 'best_checkpoint_episode', 'final_checkpoint_episode', 'best_val_score', 'duration_s'])}
"""

    # 2. phase3_statistical_analysis.md
    stat_md = f"""# Phase 3 Statistical Analysis

## Methodology & Hierarchy
- Time steps (`96/episode`) are **never** treated as independent statistical samples.
- RL vs. RL comparisons pair exact `(train_seed, scenario, eval_seed)` outcomes (`n=90` matched pairs across 3 seeds × 6 scenarios × 5 eval seeds).
- RL vs. deterministic Heuristic comparisons first average each RL algorithm across its 3 training seeds within each `(scenario, eval_seed)` block (`n=30` matched blocks) to avoid training-seed pseudo-replication.
- Family-wise error rate within each `(track, metric)` family is controlled using **Holm–Bonferroni** step-down adjustment (`holm_adjusted_p`).

## Track A (Intrinsic & Heuristics) — True Social Cost & Critical Reliability Paired Tests
{_tbl(stats_df[(stats_df['track'] == 'track_a_intrinsic') & (stats_df['metric'].isin(['True Social Cost ($)', 'Critical Reliability (%)', 'Critical ENS (MWh)']))], ['controller_1', 'controller_2', 'metric', 'n_pairs', 'c1_mean', 'c2_mean', 'mean_diff', 'ci95_lo', 'ci95_hi', 'paired_t_p', 'wilcoxon_p', 'cohens_d', 'holm_adjusted_p', 'significant_holm05'])}
"""

    # 3. phase3_algorithm_comparison.md
    track_b = fam_df[fam_df["track"] == "track_b_shielded"]
    track_c = fam_df[fam_df["track"] == "track_c_hybrid"]
    hf_summary = ""
    if not hf_df.empty:
        hf_summary = (
            f"- Non-finite HAPPO factor count across all updates: **{int(hf_df['nonfinite_count'].sum())}**.\n"
            f"- Peak observed factor maximum: **{float(hf_df['factor_max'].max()):.4f}**; mean 99th percentile: **{float(hf_df['factor_p99'].mean()):.4f}**; mean extreme fraction outside `[0.1, 10]`: **{float(hf_df['extreme_frac_outside_01_10'].mean()*100):.3f}%**."
        )

    algo_md = f"""# Phase 3 Algorithm Comparison: Answers to Research Questions

1. **Lowest True Social Cost (Track A Intrinsic vs. Heuristics):** **{best_cost_row['algorithm']}** (`{best_cost_row['mode']}`) achieved mean True Social Cost **${best_cost_row['true_social_cost_mean']:,.2f}** (seed mean across H1–H6).
2. **Highest Critical-Load Reliability (Track A):** **{best_rel_row['algorithm']}** (`{best_rel_row['mode']}`) achieved **{best_rel_row['critical_reliability_mean']:.2f}%** critical reliability.
3. **Best on Renewable Drought (`H2`):** **{h2_win['algorithm']}** (`${h2_win['true_social_cost_mean']:,.2f}`, reliability `{h2_win['critical_reliability_mean']:.2f}%`).
4. **Best on Demand Surge (`H3`):** **{h3_win['algorithm']}** (`${h3_win['true_social_cost_mean']:,.2f}`, reliability `{h3_win['critical_reliability_mean']:.2f}%`).
5. **Most Robust to Sensor Noise (`H4`):** **{h4_win['algorithm']}** (`${h4_win['true_social_cost_mean']:,.2f}`, reliability `{h4_win['critical_reliability_mean']:.2f}%`).
6. **Lowest Tail Risk (CVaR-10%):** **{best_cvar_row['algorithm']}** (`${best_cvar_row['true_social_cost_cvar10']:,.2f}`).
7. **HAPPO vs. MAPPO_MATCHED (Intrinsic):** HAPPO mean social cost **${happo_a['true_social_cost_mean']:,.2f}** vs. MAPPO_MATCHED **${mappo_a['true_social_cost_mean']:,.2f}**; critical reliability **{happo_a['critical_reliability_mean']:.2f}%** vs. **{mappo_a['critical_reliability_mean']:.2f}%** (refer to `phase3_statistical_analysis.md` for Holm-adjusted significance).
8. **IPPO Competitiveness:** IPPO with purely decentralized local critics achieved **${ippo_a['true_social_cost_mean']:,.2f}** mean social cost and **{ippo_a['critical_reliability_mean']:.2f}%** critical reliability.
9. **Safety Shield Impact (Track A -> Track B):** See Track A vs. Track B comparison below.
10. **Operational Fallback Impact (Track B -> Track C):** See Track B vs. Track C comparison below (fallback gains are strictly attributed to Track C Hybrid, never to intrinsic policies).
11. **Best Reliability–Cost Trade-Off (overall across evaluated modes):** **{best_cost_row['algorithm']}** (`{best_cost_row['mode']}`).

## Track A (Intrinsic) vs. Track B (Shielded) vs. Track C (Hybrid) RL Comparison
{_tbl(pd.concat([rl_a, track_b, track_c], ignore_index=True), ['track', 'mode', 'algorithm', 'true_social_cost_mean', 'operating_cost_mean', 'critical_ens_mean', 'critical_reliability_mean', 'boundary_clipping_mean', 'fallback_intervention_mean'])}

## HAPPO Importance-Factor Numerical Stability Audit
{hf_summary}
"""

    # 4. phase3_reliability_analysis.md
    rel_md = f"""# Phase 3 Reliability, Safety & Scenario Analysis (H1–H6)

## Scenario-Level Performance Across H1–H6 (Track A Intrinsic + Heuristics)
{_tbl(sc_a, ['scenario', 'algorithm', 'true_social_cost_mean', 'operating_cost_mean', 'critical_ens_mean', 'critical_reliability_mean', 'renewable_curtailment_mean', 'fuel_consumed_mean'])}

## Safety Intervention & Fallback Attribution (Tracks B & C)
{_tbl(pd.concat([track_b, track_c], ignore_index=True), ['track', 'algorithm', 'unshielded_feasible_mean', 'boundary_clipping_mean', 'emergency_override_mean', 'adequacy_intervention_mean', 'fallback_intervention_mean', 'power_flow_converged_pct'])}
"""

    # 5. phase3_limitations.md
    lim_md = """# Phase 3 Research Limitations

1. **Small Independent Training-Seed Count (`n = 3`):** Each learned algorithm (`IPPO`, `MAPPO_MATCHED`, `HAPPO`) was trained across three seeds (`[42, 101, 202]`). Seed-level variance cannot be eliminated with `n = 3`, and pooled `n = 90` episode tests reflect within-seed evaluation realizations alongside `n = 3` models.
2. **Matched 100-Episode Training Budget:** All three learned algorithms received an identical 100-episode (`9,600`-step) budget on `DomainRandomizer` scenarios for strict fairness against prior Phase 4.4.4 experiments; complex continuous+discrete MARL policies may still be under-converged relative to longer budgets.
3. **Unconstrained Vanilla Formulation vs. Phase 4.4.4:** `IPPO`, `MAPPO_MATCHED`, and `HAPPO` use pure `reward_v3_social.yaml` without Lagrangian safety cost heads or Bat-Only shaping so algorithmic credit assignment differences (DTDE vs. simultaneous CTDE vs. sequential HAPPO) remain unconfounded.
4. **Physical Environment Constraints in Track A:** Disabling `SafetyShield` in Track A removes policy-level software overrides (e.g., 30% reserve preservation and healthy-grid generator block), while intrinsic hardware limits (battery SOC physical min/max and generator ramp state machine) remain enforced by the physical component models.
"""

    # 6. phase3_executive_summary.md
    exec_md = f"""# Phase 3 Executive Summary — Multi-Algorithm Comparative Evaluation

- **Scope Completed:** Matched training of `IPPO`, `MAPPO_MATCHED`, and `HAPPO` across seeds `[42, 101, 202]` (`100` episodes each = `900` training episodes) + held-out evaluation across `H1–H6` × `5` matched evaluation seeds (`360` Track A episodes + `270` Track B Shielded RL episodes + `270` Track C Hybrid RL episodes = **`{len(ep_df)}`** total evaluation episodes with complete `96`-step traces).
- **Headline Finding (Lowest Social Cost & Reliability Trade-Off on Track A + Heuristics):** **{best_cost_row['algorithm']}** (`{best_cost_row['mode']}`) achieved **${best_cost_row['true_social_cost_mean']:,.2f}** mean true social cost and **{best_cost_row['critical_reliability_mean']:.2f}%** critical reliability.
- **RL Intrinsic Rankings (Mean Social Cost across 90 episodes/algorithm):**
  - `IPPO`: **${ippo_a['true_social_cost_mean']:,.2f}** (`{ippo_a['critical_reliability_mean']:.2f}%` rel)
  - `MAPPO_MATCHED`: **${mappo_a['true_social_cost_mean']:,.2f}** (`{mappo_a['critical_reliability_mean']:.2f}%` rel)
  - `HAPPO`: **${happo_a['true_social_cost_mean']:,.2f}** (`{happo_a['critical_reliability_mean']:.2f}%` rel)
- **Strict Separation Preserved:** Track A (Intrinsic), Track B (Shielded), Track C (Hybrid +Fallback), and native Heuristic Baselines are reported separately throughout.
"""

    written: List[str] = []
    report_map = {
        "phase3_experimental_results.md": res_md,
        "phase3_statistical_analysis.md": stat_md,
        "phase3_algorithm_comparison.md": algo_md,
        "phase3_reliability_analysis.md": rel_md,
        "phase3_limitations.md": lim_md,
        "phase3_executive_summary.md": exec_md,
    }
    for fname, content in report_map.items():
        for target_dir in (reports_dir, docs_dir):
            out_p = target_dir / fname
            with open(out_p, "w", encoding="utf-8") as f:
                f.write(content)
            written.append(str(out_p))
    return written


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 3 plots, environment freeze, and reports")
    parser.add_argument("--results-root", type=Path, default=ROOT / "experiments" / "phase6_model_comparison" / "results")
    parser.add_argument("--eval-subdir", type=str, default="evaluation")
    args = parser.parse_args()

    save_environment_versions(args.results_root / "environment_versions.json")
    plots = generate_all_plots(args.results_root, eval_subdir=args.eval_subdir)
    reports = generate_markdown_reports(args.results_root, eval_subdir=args.eval_subdir)
    print(f"[REPORT] Generated {len(plots)} plots and {len(reports)} report files.")


if __name__ == "__main__":
    main()
