"""
statistical_analysis.py
=======================
Hierarchical and paired statistical comparison utilities for Phase 3.
Reuses scipy.stats paired t-test, Wilcoxon signed-rank, compute_cvar, and
Holm-Bonferroni step-down family-wise correction without treating episode
timesteps as independent samples.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from experiments.phase4_4_2.benchmark_phase4_4_2 import compute_cvar

STAT_METRICS: List[Tuple[str, str, bool]] = [
    ("true_social_cost_usd", "True Social Cost ($)", False),
    ("operating_cost_usd", "Operating Cost ($)", False),
    ("critical_ens_mwh", "Critical ENS (MWh)", False),
    ("critical_reliability_pct", "Critical Reliability (%)", True),
    ("renewable_curtailment_mwh", "Renewable Curtailment (MWh)", False),
    ("fuel_consumed_l", "Fuel Consumed (L)", False),
    ("battery_throughput_mwh", "Battery Throughput (MWh)", False),
    ("unshielded_feasible_pct", "Unshielded Feasible (%)", True),
]


def _dist_stats(vals: np.ndarray, higher_is_better: bool = False) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return {"mean": np.nan, "std": np.nan, "median": np.nan, "iqr": np.nan, "ci95_lo": np.nan, "ci95_hi": np.nan, "cvar10": np.nan}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    med = float(np.median(arr))
    q75, q25 = np.percentile(arr, [75, 25])
    iqr = float(q75 - q25)
    se = std / np.sqrt(n) if n > 1 else 0.0
    half_ci = float(stats.t.ppf(0.975, df=n - 1) * se) if n > 1 and std > 0 else 0.0
    cvar10 = float(compute_cvar(arr, alpha=0.10, higher_is_better=higher_is_better))
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "median": round(med, 4),
        "iqr": round(iqr, 4),
        "ci95_lo": round(mean - half_ci, 4),
        "ci95_hi": round(mean + half_ci, 4),
        "cvar10": round(cvar10, 4),
    }


def compute_seed_and_family_summaries(ep_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produce BOTH:
      1. Seed-level summaries (one row per track x algorithm x train_seed, n=30 eps for RL/heuristics)
      2. Family-level summaries (preserving seed-level mean/std over n=3 seeds separately from pooled n=90 eps)
    """
    seed_rows: List[Dict[str, Any]] = []
    for (track, mode, algo, cgroup, tseed), grp in ep_df.groupby(
        ["track", "mode", "algorithm", "controller_group", "train_seed"], sort=False
    ):
        row: Dict[str, Any] = {
            "track": track,
            "mode": mode,
            "algorithm": algo,
            "controller_group": cgroup,
            "train_seed": str(tseed),
            "episodes_n": int(len(grp)),
            "power_flow_converged_pct": round(float(grp["power_flow_converged_pct"].mean()), 2),
            "boundary_clipping_mean": round(float(grp["boundary_clipping_count"].mean()), 3),
            "emergency_override_mean": round(float(grp["emergency_override_count"].mean()), 3),
            "adequacy_intervention_mean": round(float(grp["supply_adequacy_intervention_count"].mean()), 3),
            "fallback_intervention_mean": round(float(grp["fallback_intervention_steps"].mean()), 3),
            "inference_latency_ms_mean": round(float(grp["inference_latency_ms_mean"].mean()), 4),
        }
        for col, _, higher_is_better in STAT_METRICS:
            ds = _dist_stats(grp[col].values, higher_is_better=higher_is_better)
            prefix = col.replace("_usd", "").replace("_mwh", "").replace("_pct", "").replace("_l", "")
            for k, v in ds.items():
                row[f"{prefix}_{k}"] = v
        seed_rows.append(row)

    seed_df = pd.DataFrame(seed_rows)

    family_rows: List[Dict[str, Any]] = []
    for (track, mode, algo, cgroup), ep_grp in ep_df.groupby(
        ["track", "mode", "algorithm", "controller_group"], sort=False
    ):
        s_grp = seed_df[(seed_df["track"] == track) & (seed_df["algorithm"] == algo)]
        n_seeds = int(s_grp["train_seed"].nunique())
        row: Dict[str, Any] = {
            "track": track,
            "mode": mode,
            "algorithm": algo,
            "controller_group": cgroup,
            "train_seeds_n": n_seeds,
            "episodes_n": int(len(ep_grp)),
            "power_flow_converged_pct": round(float(ep_grp["power_flow_converged_pct"].mean()), 2),
            "boundary_clipping_mean": round(float(ep_grp["boundary_clipping_count"].mean()), 3),
            "emergency_override_mean": round(float(ep_grp["emergency_override_count"].mean()), 3),
            "adequacy_intervention_mean": round(float(ep_grp["supply_adequacy_intervention_count"].mean()), 3),
            "fallback_intervention_mean": round(float(ep_grp["fallback_intervention_steps"].mean()), 3),
            "inference_latency_ms_mean": round(float(ep_grp["inference_latency_ms_mean"].mean()), 4),
        }
        for col, _, higher_is_better in STAT_METRICS:
            ep_ds = _dist_stats(ep_grp[col].values, higher_is_better=higher_is_better)
            prefix = col.replace("_usd", "").replace("_mwh", "").replace("_pct", "").replace("_l", "")
            for k, v in ep_ds.items():
                row[f"{prefix}_{k}"] = v
            seed_means = s_grp[f"{prefix}_mean"].dropna().values
            row[f"{prefix}_seed_mean"] = round(float(np.mean(seed_means)), 4) if len(seed_means) else np.nan
            row[f"{prefix}_seed_std"] = round(float(np.std(seed_means, ddof=1)), 4) if len(seed_means) > 1 else 0.0
        family_rows.append(row)

    return seed_df, pd.DataFrame(family_rows)


def _run_paired_vector_test(v1: np.ndarray, v2: np.ndarray) -> Dict[str, float]:
    diff = v1 - v2
    n = len(diff)
    md = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    ci_h = float(stats.t.ppf(0.975, df=n - 1) * se) if n > 1 and sd > 0 else 0.0
    d_cohen = float(md / sd) if sd > 1e-12 else 0.0
    if np.allclose(diff, 0.0):
        t_p, w_p = 1.0, 1.0
    else:
        _, t_p = stats.ttest_rel(v1, v2)
        try:
            _, w_p = stats.wilcoxon(v1, v2)
        except ValueError:
            w_p = 1.0
    return {
        "n_pairs": int(n),
        "c1_mean": round(float(np.mean(v1)), 4),
        "c2_mean": round(float(np.mean(v2)), 4),
        "mean_diff": round(md, 4),
        "std_diff": round(sd, 4),
        "ci95_lo": round(md - ci_h, 4),
        "ci95_hi": round(md + ci_h, 4),
        "paired_t_p": float(t_p),
        "wilcoxon_p": float(w_p),
        "cohens_d": round(d_cohen, 4),
    }


def compute_paired_comparisons(ep_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute hierarchical matched pairwise statistical tests with Holm-Bonferroni correction:
      - Within each track (comparing Track RL models vs each other on exact (train_seed, scenario, eval_seed),
        n=90 matched pairs; AND comparing each RL family's 3-seed mean block against each Heuristic Baseline
        on exact (scenario, eval_seed), n=30 independent scenario-seed blocks so seeds are NOT pseudo-replicated).
    """
    rows: List[Dict[str, Any]] = []
    heur_df = ep_df[ep_df["controller_group"] == "Heuristic Baseline"]
    learned_tracks = [t for t in ep_df["track"].unique() if not ep_df[(ep_df["track"] == t) & (ep_df["controller_group"] == "Learned MARL")].empty]

    for track in learned_tracks:
        t_learned = ep_df[(ep_df["track"] == track) & (ep_df["controller_group"] == "Learned MARL")]
        mode_lbl = str(t_learned["mode"].iloc[0])

        # 1. RL vs RL matched on exact (train_seed, scenario, eval_seed)
        rl_pairs = [("HAPPO", "MAPPO_MATCHED"), ("HAPPO", "IPPO"), ("MAPPO_MATCHED", "IPPO")]
        for a1, a2 in rl_pairs:
            d1 = t_learned[t_learned["algorithm"] == a1]
            d2 = t_learned[t_learned["algorithm"] == a2]
            if d1.empty or d2.empty:
                continue
            for m_col, m_label, _ in STAT_METRICS[:6]:
                merged = pd.merge(
                    d1[["train_seed", "scenario", "eval_seed", m_col]],
                    d2[["train_seed", "scenario", "eval_seed", m_col]],
                    on=["train_seed", "scenario", "eval_seed"],
                    suffixes=("_1", "_2"),
                ).dropna()
                if len(merged) < 2:
                    continue
                res = _run_paired_vector_test(merged[f"{m_col}_1"].values.astype(float), merged[f"{m_col}_2"].values.astype(float))
                rows.append({
                    "track": track,
                    "mode": mode_lbl,
                    "comparison_level": "matched_seed_scenario_evalseed",
                    "controller_1": a1,
                    "controller_2": a2,
                    "metric": m_label,
                    **res,
                })

        # 2. RL family (averaged across its 3 independent seeds per (scenario, eval_seed)) vs Heuristics (n=30 blocks)
        rl_block = (
            t_learned.groupby(["algorithm", "scenario", "eval_seed"], as_index=False)[[m[0] for m in STAT_METRICS[:6]]]
            .mean()
        )
        for rl_algo in ["IPPO", "MAPPO_MATCHED", "HAPPO"]:
            d1 = rl_block[rl_block["algorithm"] == rl_algo]
            if d1.empty:
                continue
            for heur_algo in heur_df["algorithm"].unique():
                d2 = heur_df[heur_df["algorithm"] == heur_algo]
                for m_col, m_label, _ in STAT_METRICS[:6]:
                    merged = pd.merge(
                        d1[["scenario", "eval_seed", m_col]],
                        d2[["scenario", "eval_seed", m_col]],
                        on=["scenario", "eval_seed"],
                        suffixes=("_1", "_2"),
                    ).dropna()
                    if len(merged) < 2:
                        continue
                    res = _run_paired_vector_test(merged[f"{m_col}_1"].values.astype(float), merged[f"{m_col}_2"].values.astype(float))
                    rows.append({
                        "track": track,
                        "mode": mode_lbl,
                        "comparison_level": "seed_mean_scenario_evalseed",
                        "controller_1": rl_algo,
                        "controller_2": heur_algo,
                        "metric": m_label,
                        **res,
                    })

    df_stats = pd.DataFrame(rows)
    if not df_stats.empty:
        df_stats["holm_adjusted_p"] = 1.0
        for (_, _), grp in df_stats.groupby(["track", "metric"]):
            idx_sorted = grp.sort_values("paired_t_p").index
            k_m = len(idx_sorted)
            running_max = 0.0
            for rank, idx in enumerate(idx_sorted):
                adj = min(1.0, float(df_stats.loc[idx, "paired_t_p"]) * (k_m - rank))
                running_max = max(running_max, adj)
                df_stats.loc[idx, "holm_adjusted_p"] = running_max
        df_stats["significant_p05"] = df_stats["paired_t_p"] < 0.05
        df_stats["significant_holm05"] = df_stats["holm_adjusted_p"] < 0.05
    return df_stats
