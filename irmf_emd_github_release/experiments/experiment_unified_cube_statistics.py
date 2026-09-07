#!/usr/bin/python
# coding: UTF-8

"""Post-process V5.18 unified benchmark cube rows.

This module treats the unified cube as the single source of truth for paper
benchmark statistics.  It does not rerun IRMF/EMD-family algorithms.
"""

from collections import defaultdict
import csv
import math
from pathlib import Path

import numpy as np

from project_config import (
    CASE_SCORE_ROLE,
    COMPOSITE_SENSITIVITY_ENDPOINTS,
    COMPONENT_ALIGNMENT_POLICY,
    CONDITIONAL_ENDPOINTS_BY_DIMENSION,
    CONDITIONAL_PRIMARY_ENDPOINTS_BY_DIMENSION,
    DECOMPOSITION_SEMANTICS_BY_METHOD,
    DIAGNOSTIC_METRICS_BY_DIMENSION,
    EVALUATION_DIMENSION_ESTIMANDS,
    EVALUATION_FRAMEWORK_VERSION,
    EVALUATION_METHODS,
    LOWER_IS_BETTER_METRICS,
    PRIMARY_ENDPOINTS_BY_DIMENSION,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


METHODS = tuple(EVALUATION_METHODS)
BASELINES = tuple(m for m in METHODS if m != "IRMF")
LOWER_IS_BETTER = set(LOWER_IS_BETTER_METRICS)
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
SUPPLEMENTARY_METRICS = (
    "case_score_final",
    "reconstruction_score",
    "structural_fidelity_score",
    "contamination_resistance_score",
    "runtime_seconds",
    "any_failure",
    "computational_failure",
    "structural_failure",
    "denoising_failure",
)
PAIRWISE_METRICS = tuple(dict.fromkeys(PRIMARY_METRICS + CONDITIONAL_PRIMARY_METRICS + SUPPLEMENTARY_METRICS))
CONDITIONAL_PAIRWISE_METRICS = tuple(dict.fromkeys(CONDITIONAL_METRICS))
NEMENYI_Q_ALPHA_05_K4 = 2.569


def _finite(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return None


def _boolish(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _read_csv_rows(path):
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _metric_value(row, method, metric):
    return _finite(row.get(f"{method}_{metric}"))


def _higher_is_better(metric):
    if metric in {
        "any_failure",
        "computational_failure",
        "structural_failure",
        "denoising_failure",
        "runtime_seconds",
    }:
        return False
    return metric not in LOWER_IS_BETTER


def _metric_applicable(row, dimension, metric):
    dcfg = CONDITIONAL_PRIMARY_ENDPOINTS_BY_DIMENSION.get(dimension, {})
    conditional_metrics = {
        item
        for metrics in dcfg.get("subdimensions", {}).values()
        for item in metrics
    }
    applicable_noises = set(dcfg.get("applicable_noises", ())) if metric in conditional_metrics else set()
    if applicable_noises and row.get("noise") not in applicable_noises:
        return False
    if metric in {"clean_region_nmse", "contaminated_region_nmse"}:
        return any(
            _boolish(row.get(f"{method}_contamination_region_applicable")) is True
            for method in METHODS
        )
    return True


def _benefit_difference(row, baseline, metric):
    irmf = _metric_value(row, "IRMF", metric)
    base = _metric_value(row, baseline, metric)
    if irmf is None or base is None:
        return None
    raw = irmf - base
    return raw if _higher_is_better(metric) else -raw


def _rank_values(values, higher_is_better=True):
    items = [(k, v) for k, v in values.items() if v is not None and np.isfinite(v)]
    if len(items) < 2:
        return {}
    items = sorted(items, key=lambda kv: kv[1], reverse=higher_is_better)
    ranks = {}
    i = 0
    while i < len(items):
        j = i
        while j + 1 < len(items) and abs(items[j + 1][1] - items[i][1]) < 1e-12:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[items[k][0]] = float(avg)
        i = j + 1
    return ranks


def _cell_id(row):
    if "__cell_id" in row:
        return row["__cell_id"]
    return "|".join(str(row.get(k)) for k in ("signal_regime", "signal", "noise", "sigma", "seed"))


def _trajectory_id(row):
    if "__trajectory_id" in row:
        return row["__trajectory_id"]
    return "|".join(str(row.get(k)) for k in ("signal_regime", "signal", "noise"))


def _sigma_float(row):
    value = _finite(row.get("sigma"))
    return value if value is not None else np.nan


def _bootstrap_cluster_ci(items, rng, n_boot=1000):
    """Cluster bootstrap over signal-regime/signal/noise trajectories."""
    clusters = defaultdict(list)
    for item in items:
        clusters[item["trajectory_id"]].append(float(item["benefit"]))
    keys = list(clusters)
    if not keys:
        return None, None
    if len(keys) == 1:
        vals = np.asarray(clusters[keys[0]], dtype=float)
        mean = float(np.mean(vals)) if vals.size else None
        return mean, mean
    boot = []
    for _ in range(int(n_boot)):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        vals = []
        for key in sampled:
            vals.extend(clusters[key])
        if vals:
            boot.append(float(np.mean(vals)))
    if not boot:
        return None, None
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _paired_effect_rows(rows, rng):
    out = []
    group_defs = [("all", None)]
    group_defs.extend((f"regime:{regime}", {"signal_regime": regime})
                      for regime in sorted({r.get("signal_regime") for r in rows}))
    for group_name, filt in group_defs:
        group_rows = rows
        if filt:
            group_rows = [
                r for r in rows
                if all(str(r.get(k)) == str(v) for k, v in filt.items())
            ]
        for metric in PAIRWISE_METRICS:
            for baseline in BASELINES:
                items = []
                for row in group_rows:
                    benefit = _benefit_difference(row, baseline, metric)
                    if benefit is None:
                        continue
                    items.append({
                        "benefit": float(benefit),
                        "trajectory_id": _trajectory_id(row),
                    })
                vals = np.asarray([item["benefit"] for item in items], dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                ci_low, ci_high = _bootstrap_cluster_ci(items, rng)
                sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
                dz = float(np.mean(vals) / sd) if sd > 1e-12 else None
                delta = 1e-12
                out.append({
                    "group": group_name,
                    "comparison": f"IRMF_vs_{baseline}",
                    "metric": metric,
                    "higher_is_better": _higher_is_better(metric),
                    "n_paired_units": int(vals.size),
                    "n_trajectories": int(len({item["trajectory_id"] for item in items})),
                    "mean_paired_gain": float(np.mean(vals)),
                    "median_paired_gain": float(np.median(vals)),
                    "paired_cohens_dz": dz,
                    "cluster_bootstrap_ci_low": ci_low,
                    "cluster_bootstrap_ci_high": ci_high,
                    "win_rate": float(np.mean(vals > delta)),
                    "tie_rate": float(np.mean(np.abs(vals) <= delta)),
                    "loss_rate": float(np.mean(vals < -delta)),
                    "inference_unit_note": (
                        "paired cell-seed differences; uncertainty uses cluster "
                        "bootstrap over signal-regime/signal/noise trajectories"
                    ),
                })
    return out


def _conditional_endpoint_applicable(row, dimension):
    dcfg = CONDITIONAL_ENDPOINTS_BY_DIMENSION.get(dimension, {})
    noises = set(dcfg.get("applicable_noises", ()))
    return bool(noises) and row.get("noise") in noises


def _conditional_endpoint_rows(rows):
    out = []
    for row in rows:
        for dimension, dcfg in CONDITIONAL_ENDPOINTS_BY_DIMENSION.items():
            if not _conditional_endpoint_applicable(row, dimension):
                continue
            for subdimension, metrics in dcfg["subdimensions"].items():
                for metric in metrics:
                    for method in METHODS:
                        value = _metric_value(row, method, metric)
                        if value is None:
                            continue
                        out.append({
                            "block_id": _cell_id(row),
                            "signal_regime": row.get("signal_regime"),
                            "signal": row.get("signal"),
                            "noise": row.get("noise"),
                            "sigma": row.get("sigma"),
                            "seed": row.get("seed"),
                            "dimension": dimension,
                            "subdimension": subdimension,
                            "metric": metric,
                            "method": method,
                            "value": value,
                            "higher_is_better": _higher_is_better(metric),
                            "endpoint_role": "conditional_endpoint_not_universal_primary_rank",
                        })
    return out


def _conditional_endpoint_summary_rows(conditional_rows):
    grouped = defaultdict(list)
    for row in conditional_rows:
        key = (
            "all",
            row.get("dimension"),
            row.get("subdimension"),
            row.get("metric"),
            row.get("method"),
        )
        grouped[key].append(float(row["value"]))
        regime_key = (
            f"regime:{row.get('signal_regime')}",
            row.get("dimension"),
            row.get("subdimension"),
            row.get("metric"),
            row.get("method"),
        )
        grouped[regime_key].append(float(row["value"]))
    out = []
    for key, vals in sorted(grouped.items()):
        arr = np.asarray(vals, dtype=float)
        out.append({
            "group": key[0],
            "dimension": key[1],
            "subdimension": key[2],
            "metric": key[3],
            "method": key[4],
            "higher_is_better": _higher_is_better(key[3]),
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "q25": float(np.quantile(arr, 0.25)),
            "q75": float(np.quantile(arr, 0.75)),
            "endpoint_role": "conditional_endpoint_not_universal_primary_rank",
        })
    return out


def _conditional_paired_effect_rows(rows, rng):
    out = []
    for dimension, dcfg in CONDITIONAL_ENDPOINTS_BY_DIMENSION.items():
        applicable_rows = [r for r in rows if _conditional_endpoint_applicable(r, dimension)]
        group_defs = [("all", None)]
        group_defs.extend(
            (f"regime:{regime}", {"signal_regime": regime})
            for regime in sorted({r.get("signal_regime") for r in applicable_rows})
        )
        for group_name, filt in group_defs:
            group_rows = applicable_rows
            if filt:
                group_rows = [
                    r for r in applicable_rows
                    if all(str(r.get(k)) == str(v) for k, v in filt.items())
                ]
            for subdimension, metrics in dcfg["subdimensions"].items():
                for metric in metrics:
                    for baseline in BASELINES:
                        items = []
                        for row in group_rows:
                            benefit = _benefit_difference(row, baseline, metric)
                            if benefit is None:
                                continue
                            items.append({
                                "benefit": float(benefit),
                                "trajectory_id": _trajectory_id(row),
                            })
                        vals = np.asarray([item["benefit"] for item in items], dtype=float)
                        vals = vals[np.isfinite(vals)]
                        if vals.size == 0:
                            continue
                        ci_low, ci_high = _bootstrap_cluster_ci(items, rng)
                        sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
                        dz = float(np.mean(vals) / sd) if sd > 1e-12 else None
                        out.append({
                            "group": group_name,
                            "dimension": dimension,
                            "subdimension": subdimension,
                            "comparison": f"IRMF_vs_{baseline}",
                            "metric": metric,
                            "higher_is_better": _higher_is_better(metric),
                            "n_paired_units": int(vals.size),
                            "n_trajectories": int(len({item["trajectory_id"] for item in items})),
                            "mean_paired_gain": float(np.mean(vals)),
                            "median_paired_gain": float(np.median(vals)),
                            "paired_cohens_dz": dz,
                            "cluster_bootstrap_ci_low": ci_low,
                            "cluster_bootstrap_ci_high": ci_high,
                            "win_rate": float(np.mean(vals > 1e-12)),
                            "tie_rate": float(np.mean(np.abs(vals) <= 1e-12)),
                            "loss_rate": float(np.mean(vals < -1e-12)),
                            "endpoint_role": "conditional_endpoint_not_universal_primary_rank",
                        })
    return out


def _metric_summary_rows(rows):
    long = defaultdict(list)
    for row in rows:
        for method in METHODS:
            for metric in PAIRWISE_METRICS:
                value = _metric_value(row, method, metric)
                if value is None:
                    continue
                for group in ("all", f"regime:{row.get('signal_regime')}"):
                    long[(group, method, metric)].append(value)
    out = []
    for (group, method, metric), vals in sorted(long.items()):
        arr = np.asarray(vals, dtype=float)
        out.append({
            "group": group,
            "method": method,
            "metric": metric,
            "higher_is_better": _higher_is_better(metric),
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "q25": float(np.quantile(arr, 0.25)),
            "q75": float(np.quantile(arr, 0.75)),
        })
    return out


def _primary_rank_rows(rows):
    metric_rank_rows = []
    subdimension_rank_rows = []
    dimension_rank_rows = []
    overall_rank_rows = []
    primary_dimension_configs = list(PRIMARY_ENDPOINTS_BY_DIMENSION.items()) + list(CONDITIONAL_PRIMARY_ENDPOINTS_BY_DIMENSION.items())
    for row in rows:
        block = _cell_id(row)
        per_method_dimension_values = defaultdict(dict)
        for dimension, dcfg in primary_dimension_configs:
            for subdimension, metrics in dcfg["subdimensions"].items():
                metric_rank_maps = []
                for metric in metrics:
                    if not _metric_applicable(row, dimension, metric):
                        continue
                    values = {method: _metric_value(row, method, metric) for method in METHODS}
                    ranks = _rank_values(values, higher_is_better=_higher_is_better(metric))
                    if not ranks:
                        continue
                    metric_rank_maps.append(ranks)
                    for method, rank in ranks.items():
                        metric_rank_rows.append({
                            "block_id": block,
                            "signal_regime": row.get("signal_regime"),
                            "signal": row.get("signal"),
                            "noise": row.get("noise"),
                            "sigma": row.get("sigma"),
                            "seed": row.get("seed"),
                            "dimension": dimension,
                            "subdimension": subdimension,
                            "metric": metric,
                            "method": method,
                            "value": values.get(method),
                            "within_block_metric_rank": rank,
                        })
                if not metric_rank_maps:
                    continue
                sub_values = {}
                for method in METHODS:
                    vals = [r[method] for r in metric_rank_maps if method in r]
                    if vals:
                        sub_values[method] = float(np.mean(vals))
                sub_ranks = _rank_values(sub_values, higher_is_better=False)
                for method, avg in sub_values.items():
                    subdimension_rank_rows.append({
                        "block_id": block,
                        "signal_regime": row.get("signal_regime"),
                        "signal": row.get("signal"),
                        "noise": row.get("noise"),
                        "sigma": row.get("sigma"),
                        "seed": row.get("seed"),
                        "dimension": dimension,
                        "subdimension": subdimension,
                        "method": method,
                        "average_metric_rank": avg,
                        "within_block_subdimension_rank": sub_ranks.get(method),
                    })
                    if sub_ranks.get(method) is not None:
                        per_method_dimension_values[method].setdefault(dimension, []).append(sub_ranks[method])

        block_dimension_ranks_by_method = {method: [] for method in METHODS}
        for dimension, _ in primary_dimension_configs:
            values = {}
            for method in METHODS:
                vals = per_method_dimension_values.get(method, {}).get(dimension, [])
                if vals:
                    values[method] = float(np.mean(vals))
            dim_ranks = _rank_values(values, higher_is_better=False)
            for method, rank in dim_ranks.items():
                block_dimension_ranks_by_method.setdefault(method, []).append(rank)
                dimension_rank_rows.append({
                    "block_id": block,
                    "signal_regime": row.get("signal_regime"),
                    "signal": row.get("signal"),
                    "noise": row.get("noise"),
                    "sigma": row.get("sigma"),
                    "seed": row.get("seed"),
                    "dimension": dimension,
                    "method": method,
                    "average_subdimension_rank": values.get(method),
                    "within_block_dimension_rank": rank,
                })

        overall_values = {}
        for method in METHODS:
            vals = block_dimension_ranks_by_method.get(method, [])
            if vals:
                overall_values[method] = float(np.mean(vals))
        overall_ranks = _rank_values(overall_values, higher_is_better=False)
        for method, rank in overall_ranks.items():
            overall_rank_rows.append({
                "block_id": block,
                "signal_regime": row.get("signal_regime"),
                "signal": row.get("signal"),
                "noise": row.get("noise"),
                "sigma": row.get("sigma"),
                "seed": row.get("seed"),
                "method": method,
                "average_dimension_rank": overall_values.get(method),
                "within_block_overall_rank": rank,
                "interpretation": "summary_only_not_primary_hypothesis",
            })
    return metric_rank_rows, subdimension_rank_rows, dimension_rank_rows, overall_rank_rows


def _aggregate_rank_rows(rank_rows, rank_key, group_keys):
    grouped = defaultdict(list)
    for row in rank_rows:
        key = tuple(row.get(k) for k in group_keys)
        value = _finite(row.get(rank_key))
        if value is not None:
            grouped[key].append(value)
    out = []
    for key, vals in sorted(grouped.items()):
        item = {name: value for name, value in zip(group_keys, key)}
        arr = np.asarray(vals, dtype=float)
        item.update({
            "n_blocks": int(arr.size),
            "mean_rank": float(np.mean(arr)),
            "median_rank": float(np.median(arr)),
            "std_rank": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        })
        out.append(item)
    return out


def _friedman_rows(dimension_rank_rows, overall_rank_rows):
    sources = [("dimension", dimension_rank_rows, "within_block_dimension_rank"),
               ("overall", overall_rank_rows, "within_block_overall_rank")]
    out = []
    for source_name, rank_rows, rank_key in sources:
        group_dims = ["all"]
        if source_name == "dimension":
            group_dims.extend(sorted({r.get("dimension") for r in rank_rows}))
        for group in group_dims:
            rows = rank_rows
            if group != "all":
                rows = [r for r in rank_rows if r.get("dimension") == group]
            by_block = defaultdict(dict)
            for row in rows:
                rank = _finite(row.get(rank_key))
                if rank is not None:
                    by_block[row["block_id"]][row["method"]] = rank
            complete = [v for v in by_block.values() if all(m in v for m in METHODS)]
            n = len(complete)
            k = len(METHODS)
            if n == 0:
                continue
            avg_ranks = {m: float(np.mean([b[m] for b in complete])) for m in METHODS}
            friedman = (12.0 * n / (k * (k + 1.0))) * sum((avg_ranks[m] - (k + 1.0) / 2.0) ** 2 for m in METHODS)
            cd = NEMENYI_Q_ALPHA_05_K4 * math.sqrt(k * (k + 1.0) / (6.0 * n))
            for method in METHODS:
                out.append({
                    "rank_source": source_name,
                    "rank_group": group,
                    "method": method,
                    "n_complete_blocks": int(n),
                    "average_rank": avg_ranks[method],
                    "friedman_chi_square": float(friedman),
                    "friedman_df": int(k - 1),
                    "nemenyi_cd_alpha_0_05": float(cd),
                    "note": (
                        "Friedman/CD is a rank-based descriptive omnibus "
                        "summary over complete cell-seed blocks; paired gains "
                        "and bootstrap intervals remain the main effect-size evidence."
                    ),
                })
    return out


def _sigma_degradation_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        for method in METHODS:
            for metric in PRIMARY_METRICS + CONDITIONAL_PRIMARY_METRICS + ("case_score_final", "runtime_seconds"):
                dimension = "contamination_resistance" if metric in CONDITIONAL_PRIMARY_METRICS else ""
                if dimension and not _metric_applicable(row, dimension, metric):
                    continue
                value = _metric_value(row, method, metric)
                if value is None:
                    continue
                key = (
                    row.get("signal_regime"), row.get("signal"), row.get("noise"),
                    row.get("seed"), method, metric
                )
                grouped[key].append((_sigma_float(row), value))
    out = []
    for key, vals in sorted(grouped.items()):
        vals = sorted((s, v) for s, v in vals if np.isfinite(s) and np.isfinite(v))
        if len(vals) < 2:
            continue
        x = np.asarray([s for s, _ in vals], dtype=float)
        y = np.asarray([v for _, v in vals], dtype=float)
        if len(set(x)) < 2:
            continue
        slope = float(np.polyfit(x, y, deg=1)[0])
        auc = float(np.trapz(y, x) / (np.max(x) - np.min(x)))
        direction = 1.0 if _higher_is_better(key[-1]) else -1.0
        out.append({
            "signal_regime": key[0],
            "signal": key[1],
            "noise": key[2],
            "seed": key[3],
            "method": key[4],
            "metric": key[5],
            "sigma_min": float(np.min(x)),
            "sigma_max": float(np.max(x)),
            "normalized_robustness_auc": auc,
            "linear_slope_per_sigma": slope,
            "beneficial_degradation_slope": direction * slope,
            "absolute_change": float(y[-1] - y[0]),
            "beneficial_absolute_change": float(direction * (y[-1] - y[0])),
        })
    return out


def run_unified_cube_statistics(output_root, cube_root):
    """Generate V5.19 paper statistics from V5.18 unified cube outputs."""
    output_root = ensure_dir(output_root)
    cube_root = Path(cube_root)
    rows_path = cube_root / "unified_benchmark_cube_rows.csv"
    if not rows_path.exists():
        raise FileNotFoundError(f"Unified cube rows not found: {rows_path}")
    rows = _read_csv_rows(rows_path)
    rng = np.random.default_rng(20260721)

    metric_summary = _metric_summary_rows(rows)
    paired = _paired_effect_rows(rows, rng)
    conditional_rows = _conditional_endpoint_rows(rows)
    conditional_summary = _conditional_endpoint_summary_rows(conditional_rows)
    conditional_paired = _conditional_paired_effect_rows(rows, rng)
    metric_ranks, subdim_ranks, dim_ranks, overall_ranks = _primary_rank_rows(rows)
    dim_rank_summary = _aggregate_rank_rows(
        dim_ranks, "within_block_dimension_rank",
        ["signal_regime", "dimension", "method"],
    )
    overall_rank_summary = _aggregate_rank_rows(
        overall_ranks, "within_block_overall_rank",
        ["signal_regime", "method"],
    )
    friedman = _friedman_rows(dim_ranks, overall_ranks)
    degradation = _sigma_degradation_rows(rows)
    degradation_summary = _aggregate_rank_rows(
        [
            {
                **row,
                "rank_value": row["normalized_robustness_auc"]
                if _higher_is_better(row["metric"]) else -row["normalized_robustness_auc"],
            }
            for row in degradation
        ],
        "rank_value",
        ["signal_regime", "method", "metric"],
    )

    write_csv(metric_summary, output_root / "method_metric_summary.csv")
    write_json(metric_summary, output_root / "method_metric_summary.json")
    write_csv(paired, output_root / "paired_effect_sizes.csv")
    write_json(paired, output_root / "paired_effect_sizes.json")
    write_csv(conditional_rows, output_root / "conditional_endpoint_values.csv")
    write_json(conditional_rows, output_root / "conditional_endpoint_values.json")
    write_csv(conditional_summary, output_root / "conditional_endpoint_summary.csv")
    write_json(conditional_summary, output_root / "conditional_endpoint_summary.json")
    write_csv(conditional_paired, output_root / "conditional_endpoint_paired_effects.csv")
    write_json(conditional_paired, output_root / "conditional_endpoint_paired_effects.json")
    write_csv(metric_ranks, output_root / "primary_metric_ranks.csv")
    write_csv(subdim_ranks, output_root / "primary_subdimension_ranks.csv")
    write_csv(dim_ranks, output_root / "dimension_ranks_by_block.csv")
    write_csv(overall_ranks, output_root / "overall_summary_ranks_by_block.csv")
    write_csv(dim_rank_summary, output_root / "dimension_average_ranks.csv")
    write_json(dim_rank_summary, output_root / "dimension_average_ranks.json")
    write_csv(overall_rank_summary, output_root / "overall_summary_average_ranks.csv")
    write_json(overall_rank_summary, output_root / "overall_summary_average_ranks.json")
    write_csv(friedman, output_root / "friedman_cd_rank_summary.csv")
    write_json(friedman, output_root / "friedman_cd_rank_summary.json")
    write_csv(degradation, output_root / "sigma_degradation_trajectories.csv")
    write_json(degradation, output_root / "sigma_degradation_trajectories.json")
    write_csv(degradation_summary, output_root / "sigma_degradation_summary.csv")
    write_json(degradation_summary, output_root / "sigma_degradation_summary.json")

    dashboard = {
        "section": "V5.20 unified-cube statistical inference and evaluation framework",
        "source_cube": str(cube_root),
        "n_cube_rows": int(len(rows)),
        "methods": list(METHODS),
        "evaluation_framework_version": EVALUATION_FRAMEWORK_VERSION,
        "dimension_estimands": EVALUATION_DIMENSION_ESTIMANDS,
        "decomposition_semantics_by_method": DECOMPOSITION_SEMANTICS_BY_METHOD,
        "component_alignment_policy": COMPONENT_ALIGNMENT_POLICY,
        "primary_endpoints_by_dimension": PRIMARY_ENDPOINTS_BY_DIMENSION,
        "conditional_primary_endpoints_by_dimension": CONDITIONAL_PRIMARY_ENDPOINTS_BY_DIMENSION,
        "primary_metrics": list(PRIMARY_METRICS),
        "conditional_primary_metrics": list(CONDITIONAL_PRIMARY_METRICS),
        "conditional_endpoints_by_dimension": CONDITIONAL_ENDPOINTS_BY_DIMENSION,
        "conditional_metrics": list(CONDITIONAL_METRICS),
        "diagnostic_metrics_by_dimension": DIAGNOSTIC_METRICS_BY_DIMENSION,
        "composite_sensitivity_endpoint_policy": {
            "role": CASE_SCORE_ROLE,
            "endpoints": list(COMPOSITE_SENSITIVITY_ENDPOINTS),
            "not_used_for_primary_ranking": True,
        },
        "outputs": {
            "method_metric_summary": "method_metric_summary.csv",
            "paired_effect_sizes": "paired_effect_sizes.csv",
            "conditional_endpoint_values": "conditional_endpoint_values.csv",
            "conditional_endpoint_summary": "conditional_endpoint_summary.csv",
            "conditional_endpoint_paired_effects": "conditional_endpoint_paired_effects.csv",
            "dimension_average_ranks": "dimension_average_ranks.csv",
            "overall_summary_average_ranks": "overall_summary_average_ranks.csv",
            "friedman_cd_rank_summary": "friedman_cd_rank_summary.csv",
            "sigma_degradation_trajectories": "sigma_degradation_trajectories.csv",
            "sigma_degradation_summary": "sigma_degradation_summary.csv",
        },
        "statistical_protocol": {
            "paired_effects": (
                "IRMF-vs-baseline paired differences are formed within the same "
                "signal-regime/signal/noise/sigma/seed cell. Positive values "
                "always favor IRMF."
            ),
            "uncertainty": "95% cluster bootstrap CI over signal-regime/signal/noise trajectories.",
            "ranking": (
                "Primary metrics are ranked within each cell-seed block, then "
                "aggregated by subdimension, dimension, and summary rank. Overall "
                "summary rank is descriptive, not a primary hypothesis test."
            ),
            "conditional_endpoints": (
                "Outlier-region endpoints are computed only for explicit "
                "contamination-label noise models and are reported as conditional "
                "mechanism evidence; they are not mixed into the universal primary "
                "rank or overall summary rank."
            ),
            "friedman_cd": "Rank-based omnibus summary over complete cell-seed blocks.",
        },
    }
    write_json(dashboard, output_root / "unified_cube_statistics_dashboard.json")
    return dashboard
