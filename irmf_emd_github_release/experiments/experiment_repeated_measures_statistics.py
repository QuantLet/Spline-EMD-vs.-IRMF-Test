#!/usr/bin/python
# coding: UTF-8

"""Repeated-measures statistics for the four-method EMD-family benchmark."""

from collections import defaultdict
import math
import numpy as np

from project_config import (
    CONDITIONAL_ENDPOINTS_BY_DIMENSION,
    CONDITIONAL_PRIMARY_ENDPOINTS_BY_DIMENSION,
    LOWER_IS_BETTER_METRICS,
    PRIMARY_ENDPOINTS_BY_DIMENSION,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


METHODS = ("IRMF", "EMD", "EEMD", "CEEMDAN")
BASELINES = ("EMD", "EEMD", "CEEMDAN")
PRIMARY_METRICS = tuple(
    metric
    for dcfg in PRIMARY_ENDPOINTS_BY_DIMENSION.values()
    for metrics in dcfg["subdimensions"].values()
    for metric in metrics
)
CONDITIONAL_PRIMARY_METRICS = tuple(
    metric
    for dcfg in CONDITIONAL_PRIMARY_ENDPOINTS_BY_DIMENSION.values()
    for metrics in dcfg["subdimensions"].values()
    for metric in metrics
)
CONDITIONAL_METRICS = tuple(
    metric
    for dcfg in CONDITIONAL_ENDPOINTS_BY_DIMENSION.values()
    for metrics in dcfg["subdimensions"].values()
    for metric in metrics
)
METRICS = (
    "denoise_psnr",
    "case_score",
    "reconstruction_score",
    "structural_fidelity_score",
    "contamination_resistance_score",
    "imf_recovery_score",
    "decomposition_adequacy_score",
)
METRICS = tuple(dict.fromkeys(PRIMARY_METRICS + CONDITIONAL_PRIMARY_METRICS + CONDITIONAL_METRICS + METRICS))
LOWER_IS_BETTER = set(LOWER_IS_BETTER_METRICS)
PRACTICAL_THRESHOLDS = {
    "case_score": 0.01,
    "reconstruction_score": 0.01,
    "structural_fidelity_score": 0.01,
    "contamination_resistance_score": 0.01,
    "denoise_nmse": 0.005,
    "denoise_corr": 0.01,
    "denoise_psnr": 0.25,
    "imf_recovery_corr": 0.01,
    "imf_recovery_nrmse": 0.01,
    "component_splitting_index": 0.01,
    "component_merging_index": 0.01,
    "decomposition_count_error": 0.05,
    "clean_region_nmse": 0.005,
    "contaminated_region_nmse": 0.005,
    "contamination_spillover_error": 0.005,
    "imf_recovery_score": 0.01,
    "mode_mixing_index": 0.01,
    "inter_imf_entanglement_index": 0.01,
    "noise_capture_corr": 0.01,
    "outlier_resistance_index": 0.01,
}
PRIMARY_RANKING_METRICS = tuple(dict.fromkeys(PRIMARY_METRICS + CONDITIONAL_PRIMARY_METRICS))
PRIMARY_METRIC_APPLICABILITY = {
    metric: tuple(dcfg.get("conditional_applicability", {}).get(metric, ()))
    for dcfg in PRIMARY_ENDPOINTS_BY_DIMENSION.values()
    for metrics in dcfg["subdimensions"].values()
    for metric in metrics
}
PRIMARY_METRIC_APPLICABILITY.update({
    metric: tuple(dcfg.get("applicable_noises", ()))
    for dcfg in CONDITIONAL_PRIMARY_ENDPOINTS_BY_DIMENSION.values()
    for metrics in dcfg["subdimensions"].values()
    for metric in metrics
})


def _finite(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return None


def _normal_p_from_z(z):
    return float(math.erfc(abs(float(z)) / math.sqrt(2.0)))


def _chi_square_sf_approx(x, df):
    """Wilson-Hilferty chi-square survival approximation."""
    if x is None or df <= 0:
        return None
    x = max(float(x), 0.0)
    if x == 0:
        return 1.0
    z = ((x / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _wilcoxon_approx(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values[np.abs(values) > 1e-12]
    n = values.size
    if n < 2:
        return {"n": int(n), "w_plus": None, "z": None, "p_approx": None}
    order = np.argsort(np.abs(values))
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    w_plus = float(np.sum(ranks[values > 0]))
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    z = (w_plus - mean) / math.sqrt(var + 1e-12)
    return {"n": int(n), "w_plus": w_plus, "z": float(z), "p_approx": _normal_p_from_z(z)}


def _holm_adjust(rows, p_key="p_approx", q_key="p_holm"):
    valid = [(idx, row[p_key]) for idx, row in enumerate(rows) if row.get(p_key) is not None]
    if not valid:
        return rows
    valid = sorted(valid, key=lambda item: item[1])
    m = len(valid)
    adjusted = [None] * len(rows)
    running = 0.0
    for rank, (idx, p) in enumerate(valid, start=1):
        val = min(1.0, p * (m - rank + 1))
        running = max(running, val)
        adjusted[idx] = running
    for idx, val in enumerate(adjusted):
        if val is not None:
            rows[idx][q_key] = float(val)
    return rows


def _bh_adjust(rows, p_key="p_approx", q_key="p_bh"):
    valid = [(idx, row[p_key]) for idx, row in enumerate(rows) if row.get(p_key) is not None]
    if not valid:
        return rows
    valid = sorted(valid, key=lambda item: item[1])
    m = len(valid)
    adjusted = [None] * len(rows)
    running = 1.0
    for reverse_rank, (idx, p) in enumerate(reversed(valid), start=1):
        rank = m - reverse_rank + 1
        val = min(1.0, p * m / rank)
        running = min(running, val)
        adjusted[idx] = running
    for idx, val in enumerate(adjusted):
        if val is not None:
            rows[idx][q_key] = float(val)
    return rows


def _bootstrap_ci(values, rng, n_boot=2000, alpha=0.05):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, None
    if values.size == 1:
        return float(values[0]), float(values[0])
    idx = rng.integers(0, values.size, size=(int(n_boot), values.size))
    means = np.mean(values[idx], axis=1)
    return float(np.quantile(means, alpha / 2.0)), float(np.quantile(means, 1.0 - alpha / 2.0))


def _trajectory_id(row):
    return f"{row.get('signal')}|{row.get('noise')}"


def _metric_value(row, method, metric):
    return _finite(row.get(method, {}).get(metric))


def _benefit_difference(row, baseline, metric):
    if not _metric_applicable(row, metric):
        return None
    irmf = _metric_value(row, "IRMF", metric)
    base = _metric_value(row, baseline, metric)
    if irmf is None or base is None:
        return None
    raw = irmf - base
    return -raw if metric in LOWER_IS_BETTER else raw


def _metric_applicable(row, metric):
    applicable_noises = set(PRIMARY_METRIC_APPLICABILITY.get(metric, ()))
    if applicable_noises and row.get("noise") not in applicable_noises:
        return False
    if metric in {"clean_region_nmse", "contaminated_region_nmse"}:
        return any(
            bool(row.get(method, {}).get("contamination_region_applicable"))
            for method in METHODS
        )
    return True


def _long_rows(family_rows):
    out = []
    for idx, row in enumerate(family_rows):
        block_id = f"{row.get('signal')}|{row.get('noise')}|{row.get('sigma')}"
        for method in METHODS:
            method_data = row.get(method, {})
            if not isinstance(method_data, dict) or method_data.get("error"):
                continue
            for metric in METRICS:
                if not _metric_applicable(row, metric):
                    continue
                value = _finite(method_data.get(metric))
                if value is None:
                    continue
                out.append({
                    "block_id": block_id,
                    "case_index": int(idx),
                    "signal": row.get("signal"),
                    "noise": row.get("noise"),
                    "sigma": row.get("sigma"),
                    "data_seed": row.get("data_seed", row.get("seed")),
                    "algorithm_seed": row.get("algorithm_seed"),
                    "method": method,
                    "metric": metric,
                    "value": value,
                })
    return out


def _paired_effect_rows(family_rows, n_boot=2000, seed=20260715):
    rng = np.random.default_rng(seed)
    out = []
    for metric in METRICS:
        for baseline in BASELINES:
            values = []
            for row in family_rows:
                delta = _benefit_difference(row, baseline, metric)
                if delta is not None and np.isfinite(delta):
                    values.append(float(delta))
            arr = np.asarray(values, dtype=float)
            if arr.size == 0:
                continue
            ci_low, ci_high = _bootstrap_ci(arr, rng=rng, n_boot=n_boot)
            sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            dz = float(np.mean(arr) / (sd + 1e-12)) if arr.size > 1 else None
            wil = _wilcoxon_approx(arr)
            out.append({
                "metric": metric,
                "comparison": f"IRMF vs {baseline}",
                "baseline": baseline,
                "hypothesis_family": "primary_method_comparison" if metric in PRIMARY_RANKING_METRICS else "secondary_metric_exploratory",
                "smallest_effect_size_of_interest": PRACTICAL_THRESHOLDS.get(metric),
                "direction": "positive gain means IRMF better",
                "n_paired_blocks": int(arr.size),
                "mean_paired_gain": float(np.mean(arr)),
                "median_paired_gain": float(np.median(arr)),
                "paired_cohens_dz": dz,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "practical_interpretation": _practical_interpretation(
                    ci_low, ci_high, PRACTICAL_THRESHOLDS.get(metric)
                ),
                "win_rate": float(np.mean(arr > 0)),
                "tie_rate_practical": float(np.mean(np.abs(arr) <= PRACTICAL_THRESHOLDS.get(metric, 0.0))),
                "wilcoxon_w_plus": wil["w_plus"],
                "wilcoxon_z": wil["z"],
                "wilcoxon_p_approx": wil["p_approx"],
            })
    primary = _holm_adjust(
        [row for row in out if row["hypothesis_family"] == "primary_method_comparison"],
        p_key="wilcoxon_p_approx",
        q_key="wilcoxon_p_holm_primary",
    )
    secondary = _bh_adjust(
        [row for row in out if row["hypothesis_family"] == "secondary_metric_exploratory"],
        p_key="wilcoxon_p_approx",
        q_key="wilcoxon_p_bh_secondary",
    )
    return primary + secondary


def _practical_interpretation(ci_low, ci_high, threshold):
    if ci_low is None or ci_high is None or threshold is None:
        return None
    threshold = float(threshold)
    if ci_low > threshold:
        return "meaningful_irmf_superiority"
    if ci_high < -threshold:
        return "meaningful_baseline_superiority"
    if ci_low >= -threshold and ci_high <= threshold:
        return "practical_equivalence"
    return "inconclusive"


def _hierarchical_bootstrap_ci(deltas_by_trajectory, rng, n_boot=2000, alpha=0.05):
    trajectories = sorted(deltas_by_trajectory)
    if not trajectories:
        return None, None
    if len(trajectories) == 1:
        vals = np.asarray(deltas_by_trajectory[trajectories[0]], dtype=float)
        val = float(np.mean(vals)) if vals.size else None
        return val, val
    means = []
    for _ in range(int(n_boot)):
        sampled = rng.choice(trajectories, size=len(trajectories), replace=True)
        vals = []
        for tid in sampled:
            arr = np.asarray(deltas_by_trajectory[tid], dtype=float)
            if arr.size:
                vals.append(float(np.mean(arr)))
        if vals:
            means.append(float(np.mean(vals)))
    if not means:
        return None, None
    return float(np.quantile(means, alpha / 2.0)), float(np.quantile(means, 1.0 - alpha / 2.0))


def _trajectory_paired_effect_rows(family_rows, n_boot=2000, seed=20260715):
    rng = np.random.default_rng(seed + 17)
    out = []
    for metric in METRICS:
        for baseline in BASELINES:
            by_trajectory = defaultdict(list)
            by_cell = []
            for row in family_rows:
                delta = _benefit_difference(row, baseline, metric)
                if delta is not None and np.isfinite(delta):
                    by_trajectory[_trajectory_id(row)].append(float(delta))
                    by_cell.append(float(delta))
            if not by_cell:
                continue
            trajectory_means = np.asarray([np.mean(v) for v in by_trajectory.values() if v], dtype=float)
            ci_low, ci_high = _hierarchical_bootstrap_ci(by_trajectory, rng=rng, n_boot=n_boot)
            threshold = PRACTICAL_THRESHOLDS.get(metric)
            out.append({
                "metric": metric,
                "comparison": f"IRMF vs {baseline}",
                "baseline": baseline,
                "hypothesis_family": "primary_method_comparison" if metric in PRIMARY_RANKING_METRICS else "secondary_metric_exploratory",
                "benchmark_unit": "signal_noise_trajectory",
                "n_trajectories": int(len(by_trajectory)),
                "n_cells": int(len(by_cell)),
                "mean_trajectory_gain": float(np.mean(trajectory_means)) if trajectory_means.size else None,
                "median_trajectory_gain": float(np.median(trajectory_means)) if trajectory_means.size else None,
                "hierarchical_bootstrap_ci_low": ci_low,
                "hierarchical_bootstrap_ci_high": ci_high,
                "smallest_effect_size_of_interest": threshold,
                "practical_interpretation": _practical_interpretation(ci_low, ci_high, threshold),
                "trajectory_win_rate": float(np.mean(trajectory_means > 0.0)) if trajectory_means.size else None,
                "trajectory_tie_rate_practical": (
                    float(np.mean(np.abs(trajectory_means) <= threshold))
                    if trajectory_means.size and threshold is not None else None
                ),
            })
    primary = _holm_adjust(
        [row for row in out if row["hypothesis_family"] == "primary_method_comparison"],
        p_key="p_approx",
        q_key="p_holm_primary",
    )
    secondary = [row for row in out if row["hypothesis_family"] == "secondary_metric_exploratory"]
    return primary + secondary


def _dummy_matrix(records, terms):
    cols = [("intercept", None)]
    levels = {}
    for term in terms:
        if ":" in term:
            a, b = term.split(":", 1)
            combos = sorted({(r[a], r[b]) for r in records})
            levels[term] = combos[1:]
            cols.extend((term, combo) for combo in combos[1:])
        else:
            vals = sorted({r[term] for r in records})
            levels[term] = vals[1:]
            cols.extend((term, val) for val in vals[1:])
    X = np.ones((len(records), len(cols)), dtype=float)
    for j, (term, val) in enumerate(cols[1:], start=1):
        if ":" in term:
            a, b = term.split(":", 1)
            X[:, j] = [1.0 if (r[a], r[b]) == val else 0.0 for r in records]
        else:
            X[:, j] = [1.0 if r[term] == val else 0.0 for r in records]
    return X, cols


def _sse(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(np.sum(resid ** 2)), int(np.linalg.matrix_rank(X))


def _variance_decomposition(long_rows):
    effects = ("method", "signal", "noise", "sigma", "method:noise", "method:sigma", "method:signal")
    out = []
    for metric in METRICS:
        records = [r for r in long_rows if r["metric"] == metric]
        if len(records) < 8:
            continue
        y = np.asarray([r["value"] for r in records], dtype=float)
        X_full, _ = _dummy_matrix(records, effects)
        sse_full, rank_full = _sse(y, X_full)
        sst = float(np.sum((y - np.mean(y)) ** 2))
        df_error = max(len(y) - rank_full, 1)
        for effect in effects:
            reduced_effects = tuple(e for e in effects if e != effect)
            X_reduced, _ = _dummy_matrix(records, reduced_effects)
            sse_reduced, rank_reduced = _sse(y, X_reduced)
            ss_effect = max(0.0, sse_reduced - sse_full)
            df_effect = max(rank_full - rank_reduced, 1)
            ms_effect = ss_effect / df_effect
            ms_error = sse_full / df_error
            f_approx = ms_effect / (ms_error + 1e-12)
            out.append({
                "metric": metric,
                "effect": effect,
                "n_observations": int(len(y)),
                "ss_effect_partial": float(ss_effect),
                "partial_eta_squared": float(ss_effect / (ss_effect + sse_full + 1e-12)),
                "variance_explained_against_total_sst": float(ss_effect / (sst + 1e-12)),
                "df_effect": int(df_effect),
                "df_error": int(df_error),
                "f_approx": float(f_approx),
            })
    return out


def _rank_values(values, higher_is_better=True):
    items = [(m, v) for m, v in values.items() if v is not None and np.isfinite(v)]
    if len(items) < 2:
        return {}
    sorted_items = sorted(items, key=lambda kv: kv[1], reverse=higher_is_better)
    ranks = {}
    i = 0
    while i < len(sorted_items):
        j = i
        while j + 1 < len(sorted_items) and abs(sorted_items[j + 1][1] - sorted_items[i][1]) < 1e-12:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_items[k][0]] = avg_rank
        i = j + 1
    return ranks


def _aggregate_blocks(family_rows, metric, aggregation):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in family_rows:
        if not _metric_applicable(row, metric):
            continue
        if aggregation == "signal_noise":
            block_id = f"{row.get('signal')}|{row.get('noise')}"
        elif aggregation == "sigma":
            block_id = f"sigma={row.get('sigma')}|{row.get('signal')}|{row.get('noise')}"
        else:
            block_id = f"{row.get('signal')}|{row.get('noise')}|{row.get('sigma')}"
        for method in METHODS:
            value = _metric_value(row, method, metric)
            if value is not None:
                grouped[block_id][method].append(value)
    blocks = []
    for block_id, by_method in grouped.items():
        values = {method: float(np.mean(vals)) for method, vals in by_method.items() if vals}
        if len(values) >= 2:
            blocks.append((block_id, values))
    return blocks


def _ranking_rows(family_rows, aggregation="signal_noise_sigma"):
    ranking_rows = []
    pairwise_rows = []
    for metric in PRIMARY_RANKING_METRICS:
        blocks = _aggregate_blocks(family_rows, metric, aggregation=aggregation)
        higher = metric not in LOWER_IS_BETTER
        rank_by_method = defaultdict(list)
        value_by_method = defaultdict(list)
        for block_id, values in blocks:
            ranks = _rank_values(values, higher_is_better=higher)
            for method, rank in ranks.items():
                rank_by_method[method].append(rank)
                value_by_method[method].append(values[method])
        methods = [m for m in METHODS if rank_by_method.get(m)]
        k = len(methods)
        n_blocks = min((len(rank_by_method[m]) for m in methods), default=0)
        if k < 2 or n_blocks < 2:
            continue
        avg_ranks = {m: float(np.mean(rank_by_method[m])) for m in methods}
        q = 12.0 * n_blocks / (k * (k + 1.0)) * sum(
            (avg_ranks[m] - (k + 1.0) / 2.0) ** 2 for m in methods
        )
        p = _chi_square_sf_approx(q, k - 1)
        nemenyi_q_alpha_05 = 2.569 if k == 4 else 1.96
        cd = float(nemenyi_q_alpha_05 * math.sqrt(k * (k + 1.0) / (6.0 * n_blocks)))
        for method in methods:
            ranking_rows.append({
                "aggregation": aggregation,
                "metric": metric,
                "method": method,
                "n_blocks": int(n_blocks),
                "average_rank": avg_ranks[method],
                "friedman_chi_square_approx": float(q),
                "friedman_df": int(k - 1),
                "friedman_p_approx": p,
                "critical_difference_alpha_0p05": cd,
            })

        for i, m1 in enumerate(methods):
            for m2 in methods[i + 1:]:
                diffs = []
                for row in family_rows:
                    v1 = _metric_value(row, m1, metric)
                    v2 = _metric_value(row, m2, metric)
                    if v1 is None or v2 is None:
                        continue
                    raw = v1 - v2
                    diffs.append(-raw if metric in LOWER_IS_BETTER else raw)
                wil = _wilcoxon_approx(diffs)
                pairwise_rows.append({
                    "aggregation": aggregation,
                    "metric": metric,
                    "comparison": f"{m1} vs {m2}",
                    "n_paired": wil["n"],
                    "mean_benefit_first_minus_second": float(np.mean(diffs)) if diffs else None,
                    "wilcoxon_z": wil["z"],
                    "p_approx": wil["p_approx"],
                })
    return ranking_rows, _holm_adjust(pairwise_rows)


def _runtime_rows(family_rows):
    out = []
    by_method = defaultdict(list)
    for row in family_rows:
        for method in METHODS:
            runtime = _finite(row.get(method, {}).get("runtime_seconds"))
            if runtime is not None:
                by_method[method].append(runtime)
    baseline = np.median(by_method.get("EMD", [np.nan]))
    for method, vals in by_method.items():
        arr = np.asarray(vals, dtype=float)
        out.append({
            "method": method,
            "n": int(arr.size),
            "median_runtime_seconds": float(np.median(arr)),
            "mean_runtime_seconds": float(np.mean(arr)),
            "relative_median_runtime_vs_emd": float(np.median(arr) / (baseline + 1e-12)) if np.isfinite(baseline) else None,
        })
    return sorted(out, key=lambda r: r["method"])


def run_repeated_measures_statistics(output_root, family_rows, n_boot=2000, seed=20260715):
    output_root = ensure_dir(output_root)
    long = _long_rows(family_rows)
    paired = _paired_effect_rows(family_rows, n_boot=n_boot, seed=seed)
    hierarchical = _trajectory_paired_effect_rows(family_rows, n_boot=n_boot, seed=seed)
    variance = _variance_decomposition(long)
    ranking_default, pairwise_default = _ranking_rows(family_rows, aggregation="signal_noise_sigma")
    ranking_signal_noise, pairwise_signal_noise = _ranking_rows(family_rows, aggregation="signal_noise")
    ranking_sigma, pairwise_sigma = _ranking_rows(family_rows, aggregation="sigma")
    runtime = _runtime_rows(family_rows)

    write_csv(long, output_root / "long_repeated_measures_table.csv")
    write_json(long, output_root / "long_repeated_measures_table.json")
    write_csv(paired, output_root / "paired_effect_sizes_vs_baselines.csv")
    write_json(paired, output_root / "paired_effect_sizes_vs_baselines.json")
    write_csv(hierarchical, output_root / "hierarchical_trajectory_paired_effects.csv")
    write_json(hierarchical, output_root / "hierarchical_trajectory_paired_effects.json")
    write_csv(variance, output_root / "variance_decomposition_by_metric.csv")
    write_json(variance, output_root / "variance_decomposition_by_metric.json")
    write_csv(ranking_default, output_root / "friedman_cd_ranking_signal_noise_sigma.csv")
    write_json(ranking_default, output_root / "friedman_cd_ranking_signal_noise_sigma.json")
    write_csv(pairwise_default, output_root / "holm_pairwise_wilcoxon_signal_noise_sigma.csv")
    write_json(pairwise_default, output_root / "holm_pairwise_wilcoxon_signal_noise_sigma.json")
    write_csv(ranking_signal_noise, output_root / "friedman_cd_ranking_signal_noise_aggregated.csv")
    write_json(ranking_signal_noise, output_root / "friedman_cd_ranking_signal_noise_aggregated.json")
    write_csv(pairwise_signal_noise, output_root / "holm_pairwise_wilcoxon_signal_noise_aggregated.csv")
    write_json(pairwise_signal_noise, output_root / "holm_pairwise_wilcoxon_signal_noise_aggregated.json")
    write_csv(ranking_sigma, output_root / "friedman_cd_ranking_sigma_stratified.csv")
    write_json(ranking_sigma, output_root / "friedman_cd_ranking_sigma_stratified.json")
    write_csv(pairwise_sigma, output_root / "holm_pairwise_wilcoxon_sigma_stratified.csv")
    write_json(pairwise_sigma, output_root / "holm_pairwise_wilcoxon_sigma_stratified.json")
    write_csv(runtime, output_root / "runtime_cost_summary.csv")
    write_json(runtime, output_root / "runtime_cost_summary.json")

    protocol = {
        "section": "7 Statistical Inference and Cross-Method Ranking",
        "estimand": "Repeated-measures comparison across fixed benchmark blocks.",
        "benchmark_unit": "Primary block is signal_class x noise_model x sigma. Methods are repeated within each block.",
        "methods": list(METHODS),
        "paired_effect_size": "paired Cohen's dz computed on direction-aligned IRMF-minus-baseline gains.",
        "hierarchical_bootstrap": (
            "Trajectory-level inference resamples signal-noise trajectories and "
            "averages over sigma cells within each trajectory.  This avoids "
            "treating repeated sigma levels from the same signal-noise pair as "
            "fully independent datasets."
        ),
        "universal_primary_metrics": list(PRIMARY_METRICS),
        "conditional_primary_metrics": list(CONDITIONAL_PRIMARY_METRICS),
        "primary_metrics": list(PRIMARY_RANKING_METRICS),
        "secondary_metrics": [m for m in METRICS if m not in PRIMARY_RANKING_METRICS],
        "multiplicity_policy": (
            "Primary IRMF-vs-baseline comparisons are marked as a Holm-controlled "
            "family. Secondary metrics are exploratory and should be interpreted "
            "with FDR/BH-style caution."
        ),
        "practical_significance": {
            "thresholds": PRACTICAL_THRESHOLDS,
            "interpretation": (
                "CI above +delta = meaningful IRMF superiority; CI within +/-delta "
                "= practical equivalence; CI below -delta = meaningful baseline superiority."
            ),
        },
        "variance_decomposition": "No-dependency fixed-effect OLS partial contribution with method, signal, noise, sigma, method:noise, method:sigma, and method:signal.",
        "ranking": "Friedman average ranks and approximate Nemenyi critical difference; pairwise Wilcoxon p-values are Holm adjusted.",
        "aggregation_sensitivity": [
            "signal_noise_sigma primary blocks",
            "signal_noise blocks aggregated over sigma",
            "sigma-stratified blocks",
        ],
        "note": "When outer Monte Carlo seeds are added, aggregate seeds within each benchmark cell before treating the cell as a CD-ranking block.",
    }
    write_json(protocol, output_root / "benchmark_unit_protocol.json")
    return {
        "n_long_rows": len(long),
        "n_paired_rows": len(paired),
        "n_hierarchical_rows": len(hierarchical),
        "n_variance_rows": len(variance),
        "n_ranking_rows": len(ranking_default),
        "n_runtime_rows": len(runtime),
    }
