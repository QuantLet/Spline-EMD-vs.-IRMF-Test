#!/usr/bin/python
# coding: UTF-8

"""V5.24 structural-metric schema qualification run.

This stage audits the behavior of true-component splitting/merging metrics.
It is not a paper-level method ranking stage.
"""

from collections import defaultdict
import numpy as np

from project_config import (
    CONDITIONAL_PRIMARY_ENDPOINTS_BY_DIMENSION,
    DEFAULT_FS,
    DEFAULT_METHOD_TIMEOUT_SECONDS,
    DEFAULT_N,
    EVALUATION_METHODS,
    EVALUATION_FRAMEWORK_FROZEN_DATE,
    EVALUATION_FRAMEWORK_STATUS,
    EVALUATION_FRAMEWORK_VERSION,
    GLOBAL_CEEMDAN_PARAMS,
    GLOBAL_EEMD_PARAMS,
    GLOBAL_EMD_PARAMS,
    GLOBAL_IRMF_PARAMS,
    PRIMARY_ENDPOINTS_BY_DIMENSION,
)
from diagnostics.shared_physical_diagnostics import (
    STRUCTURAL_METRIC_SCHEMA_VERSION,
    true_component_mixing_diagnostics,
)
from experiments.experiment_utils import run_fixed_method_family_case
from experiments.experiment_unified_benchmark_cube import _algorithm_seed_for, _finite
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


VALIDATION_SIGNALS = (
    "stationary_multi_sine",
    "close_frequencies",
    "chirp",
    "frequency_jump",
    "intermittent_oscillation",
)
VALIDATION_NOISES = ("gaussian", "impulsive", "ar1_colored", "burst")
VALIDATION_SIGMAS = (0.05, 0.20)
VALIDATION_SEEDS = tuple(range(10))
ASSOCIATION_THRESHOLDS = (0.02, 0.05, 0.10)
DEFAULT_ASSOCIATION_THRESHOLD = 0.05
THRESHOLD_SIGN_STABILITY_MIN_RATE = 0.80
ALL_GATED_MAX_RATE = 0.05
QUALIFICATION_AUDIT_VERSION = "V1.2"
IMF_COUNT_POOLED_SCREENING_THRESHOLD = 0.90
IMF_COUNT_RESIDUALIZED_MAX_ABS_CORR = 0.50
IMF_COUNT_STRATIFIED_EXTREME_ABS_CORR = 0.80
IMF_COUNT_STRATIFIED_SYSTEMATIC_MEDIAN_ABS_CORR = 0.70
IMF_COUNT_STRATIFIED_SYSTEMATIC_P90_ABS_CORR = 0.95
IMF_COUNT_STRATIFIED_MIN_ELIGIBLE_STRATA = 5
IMF_COUNT_STRATIFIED_MIN_N = 8
METHODS = tuple(EVALUATION_METHODS)
STRUCTURAL_LEGACY_FIELDS = {
    "mode_mixing_index",
    "mode_mixing_index_legacy",
    "decomposition_count_error",
}
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
PRIMARY_ANALYSIS_METRICS = tuple(dict.fromkeys(PRIMARY_METRICS + CONDITIONAL_PRIMARY_METRICS))


def _method_rows(rows):
    for row in rows:
        for method in METHODS:
            summary = row.get(method)
            if isinstance(summary, dict):
                yield row, method, summary


def _range_audit(rows):
    metrics = (
        "component_splitting_index",
        "component_merging_index",
        "spurious_mode_energy_ratio",
        "missing_true_component_fraction",
        "unmatched_estimated_component_fraction",
    )
    grouped = defaultdict(list)
    for _, method, summary in _method_rows(rows):
        for metric in metrics:
            val = _finite(summary.get(metric))
            grouped[(method, metric)].append(val)
    out = []
    for (method, metric), vals in sorted(grouped.items()):
        finite = np.asarray([v for v in vals if v is not None], dtype=float)
        nonfinite = len(vals) - len(finite)
        out_of_range = int(np.sum((finite < -1e-12) | (finite > 1.0 + 1e-12))) if len(finite) else 0
        out.append({
            "method": method,
            "metric": metric,
            "n": int(len(vals)),
            "finite_count": int(len(finite)),
            "nonfinite_count": int(nonfinite),
            "out_of_range_count": out_of_range,
            "min": float(np.min(finite)) if len(finite) else np.nan,
            "max": float(np.max(finite)) if len(finite) else np.nan,
            "mean": float(np.mean(finite)) if len(finite) else np.nan,
            "range_audit_status": "passed" if nonfinite == 0 and out_of_range == 0 else "failed",
        })
    return out


def _gating_audit(rows):
    out = []
    grouped = defaultdict(list)
    for row, method, summary in _method_rows(rows):
        grouped[(
            method,
            row.get("signal"),
            row.get("noise"),
            row.get("sigma"),
        )].append(summary)
    for (method, signal, noise, sigma), vals in sorted(grouped.items(), key=lambda item: str(item[0])):
        missing = np.asarray([
            _finite(v.get("missing_true_component_fraction")) or 0.0 for v in vals
        ], dtype=float)
        unmatched = np.asarray([
            _finite(v.get("unmatched_estimated_component_fraction")) or 0.0 for v in vals
        ], dtype=float)
        spurious = np.asarray([
            _finite(v.get("spurious_mode_energy_ratio")) or 0.0 for v in vals
        ], dtype=float)
        available = [bool(v.get("true_component_metrics_available", False)) for v in vals]
        out.append({
            "method": method,
            "signal": signal,
            "noise": noise,
            "sigma": sigma,
            "n_seeds": int(len(vals)),
            "true_component_metrics_available_rate": float(np.mean(available)) if vals else np.nan,
            "missing_true_component_rate": float(np.mean(missing > 0.0)) if len(missing) else np.nan,
            "mean_missing_true_component_fraction": float(np.mean(missing)) if len(missing) else np.nan,
            "unmatched_estimated_component_rate": float(np.mean(unmatched > 0.0)) if len(unmatched) else np.nan,
            "mean_unmatched_estimated_component_fraction": float(np.mean(unmatched)) if len(unmatched) else np.nan,
            "mean_spurious_mode_energy_ratio": float(np.mean(spurious)) if len(spurious) else np.nan,
        })
    return out


def _all_gated_audit(rows):
    grouped = defaultdict(list)
    for _, method, summary in _method_rows(rows):
        grouped[method].append(summary)
    out = []
    for method, vals in sorted(grouped.items()):
        total = len(vals)
        available = [
            v for v in vals
            if bool(v.get("true_component_metrics_available", False))
        ]
        all_gated = [
            v for v in available
            if _finite(v.get("component_splitting_index")) is None
            and _finite(v.get("component_merging_index")) is None
        ]
        out.append({
            "method": method,
            "n_method_runs": int(total),
            "n_true_component_available_runs": int(len(available)),
            "all_gated_count": int(len(all_gated)),
            "all_gated_rate": float(len(all_gated) / max(len(available), 1)),
            "all_gated_max_rate": float(ALL_GATED_MAX_RATE),
            "all_gated_audit_status": (
                "passed"
                if float(len(all_gated) / max(len(available), 1)) <= ALL_GATED_MAX_RATE
                else "requires_review"
            ),
        })
    return out


def _pearson_abs_with_n(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan, int(len(x))
    return float(abs(np.corrcoef(x, y)[0, 1])), int(len(x))


def _rankdata_average(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_vals = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_vals[end] == sorted_vals[start]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _spearman_abs_with_n(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan, int(len(x))
    return _pearson_abs_with_n(_rankdata_average(x), _rankdata_average(y))


def _leave_one_out_abs_correlations(x, y, seeds=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if seeds is None:
        seeds = list(range(len(x)))
    out = []
    for idx in range(len(x)):
        keep = np.ones(len(x), dtype=bool)
        keep[idx] = False
        corr, n_pairs = _pearson_abs_with_n(x[keep], y[keep])
        out.append({
            "left_out_seed": seeds[idx],
            "abs_pearson_correlation": corr,
            "n_valid_pairs": n_pairs,
        })
    return out


def _residualize_by_groups(values, group_keys):
    values = np.asarray(values, dtype=float)
    residuals = np.full_like(values, np.nan, dtype=float)
    grouped = defaultdict(list)
    for idx, key in enumerate(group_keys):
        if np.isfinite(values[idx]):
            grouped[key].append(idx)
    for indices in grouped.values():
        vals = values[indices]
        residuals[indices] = vals - float(np.mean(vals))
    return residuals


def _metric_dependency_arrays(rows, method, metric):
    x_vals = []
    y_vals = []
    group_keys = []
    stratum_keys = []
    seeds = []
    for row in rows:
        summary = row.get(method, {})
        if not isinstance(summary, dict):
            continue
        imf_count = _finite(summary.get("effective_imf_count"))
        if imf_count is None:
            imf_count = _finite(summary.get("imf_count"))
        metric_val = _finite(summary.get(metric))
        x_vals.append(np.nan if imf_count is None else imf_count)
        y_vals.append(np.nan if metric_val is None else metric_val)
        key = (row.get("signal"), row.get("noise"), row.get("sigma"))
        group_keys.append(key)
        stratum_keys.append(key)
        seeds.append(row.get("seed"))
    return (
        np.asarray(x_vals, dtype=float),
        np.asarray(y_vals, dtype=float),
        group_keys,
        stratum_keys,
        seeds,
    )


def _stratified_dependency_summary(x_vals, y_vals, stratum_keys, seeds):
    grouped = defaultdict(list)
    for idx, key in enumerate(stratum_keys):
        grouped[key].append(idx)
    rows = []
    for key, indices in sorted(grouped.items(), key=lambda item: str(item[0])):
        x = x_vals[indices]
        y = y_vals[indices]
        corr, n_pairs = _pearson_abs_with_n(x, y)
        spear, _ = _spearman_abs_with_n(x, y)
        x_keep = x[np.isfinite(x)]
        y_keep = y[np.isfinite(y)]
        x_levels = int(len(np.unique(x_keep))) if len(x_keep) else 0
        y_levels = int(len(np.unique(y_keep))) if len(y_keep) else 0
        level_counts = {}
        for val in x_keep:
            label = str(int(val)) if float(val).is_integer() else str(float(val))
            level_counts[label] = int(level_counts.get(label, 0) + 1)
        min_level_n = int(min(level_counts.values())) if level_counts else 0
        identifiable = bool(
            n_pairs >= IMF_COUNT_STRATIFIED_MIN_N
            and x_levels >= 2
            and y_levels >= 2
            and np.isfinite(corr)
        )
        loo = _leave_one_out_abs_correlations(
            x,
            y,
            seeds=[seeds[i] for i in indices],
        ) if identifiable else []
        finite_loo = [
            item["abs_pearson_correlation"]
            for item in loo
            if np.isfinite(item["abs_pearson_correlation"])
        ]
        loo_min = float(np.min(finite_loo)) if finite_loo else np.nan
        loo_max = float(np.max(finite_loo)) if finite_loo else np.nan
        high_leverage = bool(
            identifiable
            and np.isfinite(corr)
            and np.isfinite(loo_min)
            and corr >= IMF_COUNT_STRATIFIED_EXTREME_ABS_CORR
            and loo_min < 0.50
        )
        rows.append({
            "signal": key[0],
            "noise": key[1],
            "sigma": key[2],
            "n_valid_pairs": int(n_pairs),
            "imf_count_levels": x_levels,
            "effective_imf_count_unique_values": sorted(level_counts.keys()),
            "effective_imf_count_level_counts": level_counts,
            "minimum_observations_per_imf_count_level": min_level_n,
            "metric_value_levels": y_levels,
            "identifiable": identifiable,
            "abs_pearson_correlation": corr,
            "abs_spearman_correlation": spear,
            "leave_one_seed_out_min_abs_correlation": loo_min,
            "leave_one_seed_out_max_abs_correlation": loo_max,
            "high_leverage_seed_pattern": high_leverage,
            "extreme_stratum": bool(identifiable and np.isfinite(corr) and corr >= IMF_COUNT_STRATIFIED_EXTREME_ABS_CORR),
            "stratum_status": (
                "computed" if identifiable else "not_identifiable_within_stratum"
            ),
        })
    identifiable_corrs = np.asarray([
        row["abs_pearson_correlation"]
        for row in rows
        if row["identifiable"] and np.isfinite(row["abs_pearson_correlation"])
    ], dtype=float)
    return rows, {
        "n_total_strata": int(len(rows)),
        "n_eligible_strata": int(len(identifiable_corrs)),
        "n_non_identifiable_strata": int(len(rows) - len(identifiable_corrs)),
        "median_within_stratum_abs_correlation": (
            float(np.median(identifiable_corrs)) if len(identifiable_corrs) else np.nan
        ),
        "p90_within_stratum_abs_correlation": (
            float(np.percentile(identifiable_corrs, 90)) if len(identifiable_corrs) else np.nan
        ),
        "max_within_stratum_abs_correlation": (
            float(np.max(identifiable_corrs)) if len(identifiable_corrs) else np.nan
        ),
        "n_extreme_strata": int(np.sum(identifiable_corrs >= IMF_COUNT_STRATIFIED_EXTREME_ABS_CORR)) if len(identifiable_corrs) else 0,
    }


def _imf_count_dependency_audit(rows):
    out = []
    strata_out = []
    for method in sorted(METHODS):
        for metric in ("component_splitting_index", "component_merging_index"):
            imf_count, metric_vals, group_keys, stratum_keys, seeds = _metric_dependency_arrays(
                rows, method, metric
            )
            pooled_corr, n_pairs = _pearson_abs_with_n(imf_count, metric_vals)
            rx = _residualize_by_groups(imf_count, group_keys)
            ry = _residualize_by_groups(metric_vals, group_keys)
            residual_corr, residual_n = _pearson_abs_with_n(rx, ry)
            stratum_rows, stratum_summary = _stratified_dependency_summary(
                imf_count, metric_vals, stratum_keys, seeds
            )
            for srow in stratum_rows:
                strata_out.append({
                    "method": method,
                    "metric": metric,
                    **srow,
                })
            pooled_screening_warning = bool(
                n_pairs >= 10
                and np.isfinite(pooled_corr)
                and pooled_corr >= IMF_COUNT_POOLED_SCREENING_THRESHOLD
            )
            residualized_requires_review = bool(
                residual_n >= 10
                and np.isfinite(residual_corr)
                and residual_corr >= IMF_COUNT_RESIDUALIZED_MAX_ABS_CORR
            )
            median_stratum_corr = stratum_summary["median_within_stratum_abs_correlation"]
            p90_stratum_corr = stratum_summary["p90_within_stratum_abs_correlation"]
            stratified_requires_review = bool(
                stratum_summary["n_eligible_strata"] >= IMF_COUNT_STRATIFIED_MIN_ELIGIBLE_STRATA
                and (
                    (
                        np.isfinite(median_stratum_corr)
                        and median_stratum_corr >= IMF_COUNT_STRATIFIED_SYSTEMATIC_MEDIAN_ABS_CORR
                    )
                    or (
                        np.isfinite(p90_stratum_corr)
                        and p90_stratum_corr >= IMF_COUNT_STRATIFIED_SYSTEMATIC_P90_ABS_CORR
                    )
                )
            )
            if residualized_requires_review or stratified_requires_review:
                status = "requires_review"
            elif pooled_screening_warning:
                status = "passed_with_pooled_confounding_note"
            else:
                status = "passed"
            out.append({
                "qualification_audit_version": QUALIFICATION_AUDIT_VERSION,
                "method": method,
                "metric": metric,
                "dependency_metric": "effective_imf_count_or_imf_count",
                "pooled_n_valid_pairs": n_pairs,
                "pooled_abs_pearson_correlation": pooled_corr,
                "pooled_screening_threshold": IMF_COUNT_POOLED_SCREENING_THRESHOLD,
                "pooled_screening_warning": pooled_screening_warning,
                "residualized_control": "signal_x_noise_x_sigma_fixed_effects",
                "residualized_n_valid_pairs": residual_n,
                "residualized_abs_pearson_correlation": residual_corr,
                "residualized_max_abs_correlation": IMF_COUNT_RESIDUALIZED_MAX_ABS_CORR,
                "residualized_requires_review": residualized_requires_review,
                "stratified_min_n": IMF_COUNT_STRATIFIED_MIN_N,
                "stratified_extreme_abs_correlation": IMF_COUNT_STRATIFIED_EXTREME_ABS_CORR,
                "stratified_systematic_median_abs_correlation": IMF_COUNT_STRATIFIED_SYSTEMATIC_MEDIAN_ABS_CORR,
                "stratified_systematic_p90_abs_correlation": IMF_COUNT_STRATIFIED_SYSTEMATIC_P90_ABS_CORR,
                "stratified_min_eligible_strata": IMF_COUNT_STRATIFIED_MIN_ELIGIBLE_STRATA,
                "stratified_requires_review": stratified_requires_review,
                **stratum_summary,
                "dependency_audit_status": status,
                "near_equivalence_warning": status == "requires_review",
            })
    return out, strata_out


def _threshold_sensitivity_rows(full_case_results, thresholds=ASSOCIATION_THRESHOLDS):
    rows = []
    for meta, case, results in full_case_results:
        true_components = case.get("true_components")
        for method, result in results.items():
            imfs = result.get("imfs") if isinstance(result, dict) else None
            for threshold in thresholds:
                diag = true_component_mixing_diagnostics(
                    imfs,
                    true_components=true_components,
                    association_threshold=threshold,
                )
                rows.append({
                    **meta,
                    "method": method,
                    "association_threshold": float(threshold),
                    "component_splitting_index": diag.get("component_splitting_index"),
                    "component_merging_index": diag.get("component_merging_index"),
                    "missing_true_component_fraction": diag.get("missing_true_component_fraction"),
                    "unmatched_estimated_component_fraction": diag.get("unmatched_estimated_component_fraction"),
                    "spurious_mode_energy_ratio": diag.get("spurious_mode_energy_ratio"),
                    "true_component_metrics_available": diag.get("true_component_metrics_available"),
                })
    return rows


def _threshold_sensitivity_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["association_threshold"])].append(row)
    out = []
    for (method, threshold), vals in sorted(grouped.items()):
        for metric in (
            "component_splitting_index",
            "component_merging_index",
            "missing_true_component_fraction",
            "unmatched_estimated_component_fraction",
            "spurious_mode_energy_ratio",
        ):
            data = np.asarray([
                v for v in (_finite(row.get(metric)) for row in vals) if v is not None
            ], dtype=float)
            out.append({
                "method": method,
                "association_threshold": threshold,
                "metric": metric,
                "n": int(len(data)),
                "mean": float(np.mean(data)) if len(data) else np.nan,
                "median": float(np.median(data)) if len(data) else np.nan,
                "sd": float(np.std(data, ddof=1)) if len(data) > 1 else 0.0,
            })
    return out


def _sign(value, tol=1e-12):
    value = _finite(value)
    if value is None:
        return None
    if value > tol:
        return 1
    if value < -tol:
        return -1
    return 0


def _value_or_nan(row, metric):
    value = _finite(row.get(metric)) if row else None
    return value if value is not None else np.nan


def _threshold_sign_stability_audit(rows):
    by_key = {}
    for row in rows:
        key = (
            row.get("signal"),
            row.get("noise"),
            row.get("sigma"),
            row.get("seed"),
            row.get("method"),
            row.get("association_threshold"),
        )
        by_key[key] = row
    out = []
    for metric in ("component_splitting_index", "component_merging_index"):
        for baseline in ("EMD", "EEMD", "CEEMDAN"):
            stable = 0
            total = 0
            changed_examples = []
            strata = sorted({
                (r.get("signal"), r.get("noise"), r.get("sigma"), r.get("seed"))
                for r in rows
            })
            for stratum in strata:
                default_irmf = by_key.get((*stratum, "IRMF", DEFAULT_ASSOCIATION_THRESHOLD))
                default_base = by_key.get((*stratum, baseline, DEFAULT_ASSOCIATION_THRESHOLD))
                if not default_irmf or not default_base:
                    continue
                # Lower is better for both metrics, so positive benefit means IRMF lower.
                default_sign = _sign(
                    _value_or_nan(default_base, metric) - _value_or_nan(default_irmf, metric)
                )
                if default_sign is None:
                    continue
                for threshold in ASSOCIATION_THRESHOLDS:
                    if abs(float(threshold) - DEFAULT_ASSOCIATION_THRESHOLD) < 1e-12:
                        continue
                    irmf = by_key.get((*stratum, "IRMF", float(threshold)))
                    base = by_key.get((*stratum, baseline, float(threshold)))
                    if not irmf or not base:
                        continue
                    sign = _sign(
                        _value_or_nan(base, metric) - _value_or_nan(irmf, metric)
                    )
                    if sign is None:
                        continue
                    total += 1
                    if sign == default_sign:
                        stable += 1
                    elif len(changed_examples) < 10:
                        changed_examples.append({
                            "signal": stratum[0],
                            "noise": stratum[1],
                            "sigma": stratum[2],
                            "seed": stratum[3],
                            "threshold": float(threshold),
                            "default_sign": default_sign,
                            "threshold_sign": sign,
                        })
            rate = float(stable / total) if total else np.nan
            out.append({
                "metric": metric,
                "comparison": f"IRMF_vs_{baseline}",
                "default_association_threshold": float(DEFAULT_ASSOCIATION_THRESHOLD),
                "thresholds_compared": [
                    float(t) for t in ASSOCIATION_THRESHOLDS
                    if abs(float(t) - DEFAULT_ASSOCIATION_THRESHOLD) >= 1e-12
                ],
                "n_sign_comparisons": int(total),
                "stable_sign_count": int(stable),
                "stable_sign_rate": rate,
                "minimum_acceptable_stable_sign_rate": float(THRESHOLD_SIGN_STABILITY_MIN_RATE),
                "threshold_sensitivity_status": (
                    "insufficient_sample"
                    if total < 10 else
                    "passed"
                    if rate >= THRESHOLD_SIGN_STABILITY_MIN_RATE else
                    "requires_review"
                ),
                "changed_sign_examples": changed_examples,
            })
    return out


def _controlled_construct_dependency_audit():
    n = 512
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    true_components = np.vstack([
        np.sin(2.0 * np.pi * 5.0 * t),
        np.sin(2.0 * np.pi * 11.0 * t),
    ])
    rng = np.random.default_rng(20260726)

    def diag(imfs):
        return true_component_mixing_diagnostics(
            np.asarray(imfs, dtype=float),
            true_components=true_components,
        )

    cases = {
        "perfect_one_to_one": diag([true_components[0], true_components[1]]),
        "added_gated_out_mode": diag([
            true_components[0],
            true_components[1],
            1e-4 * rng.normal(size=n),
        ]),
        "added_active_spurious_mode": diag([
            true_components[0],
            true_components[1],
            0.5 * rng.normal(size=n),
        ]),
        "dominant_component_split": diag([
            0.5 * true_components[0],
            0.5 * true_components[0],
            true_components[1],
        ]),
        "same_cardinality_concentrated": diag([
            0.9 * true_components[0],
            0.05 * true_components[0],
            0.05 * true_components[0],
            true_components[1],
        ]),
        "same_cardinality_diffuse": diag([
            0.4 * true_components[0],
            0.35 * true_components[0],
            0.25 * true_components[0],
            true_components[1],
        ]),
    }

    def val(case_name, metric):
        value = cases[case_name].get(metric)
        return float(value) if np.isfinite(value) else np.nan

    checks = [
        {
            "check": "gated_out_mode_does_not_change_splitting",
            "passed": abs(
                val("added_gated_out_mode", "component_splitting_index")
                - val("perfect_one_to_one", "component_splitting_index")
            ) <= 1e-9,
        },
        {
            "check": "active_spurious_mode_detected_as_spurious",
            "passed": (
                val("added_active_spurious_mode", "spurious_mode_energy_ratio") > 0.01
                and val("added_active_spurious_mode", "unmatched_estimated_component_fraction") > 0.0
            ),
        },
        {
            "check": "dominant_component_split_increases_splitting",
            "passed": (
                val("dominant_component_split", "component_splitting_index")
                > val("perfect_one_to_one", "component_splitting_index") + 0.10
            ),
        },
        {
            "check": "same_cardinality_distinguishes_concentration",
            "passed": (
                val("same_cardinality_diffuse", "component_splitting_index")
                > val("same_cardinality_concentrated", "component_splitting_index") + 0.10
            ),
        },
    ]

    rows = []
    for case_name, result in sorted(cases.items()):
        rows.append({
            "case": case_name,
            "component_splitting_index": result.get("component_splitting_index"),
            "component_merging_index": result.get("component_merging_index"),
            "missing_true_component_fraction": result.get("missing_true_component_fraction"),
            "unmatched_estimated_component_fraction": result.get("unmatched_estimated_component_fraction"),
            "spurious_mode_energy_ratio": result.get("spurious_mode_energy_ratio"),
        })
    status = "passed" if all(check["passed"] for check in checks) else "requires_review"
    return rows, [{
        "audit": "controlled_construct_dependency",
        "qualification_audit_version": QUALIFICATION_AUDIT_VERSION,
        "status": status,
        "checks": checks,
    }]


def _safe_abs_corr_sq(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    c = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(c):
        return 0.0
    return float(abs(c) ** 2)


def _eemd_association_profile(imfs, true_components, energy_threshold=0.01,
                              association_threshold=0.05, allocation_floor=0.01):
    if imfs is None or true_components is None:
        return None
    imfs = np.asarray(imfs, dtype=float)
    true_components = np.asarray(true_components, dtype=float)
    if imfs.ndim == 1:
        imfs = imfs[None, :]
    if true_components.ndim == 1:
        true_components = true_components[None, :]
    if imfs.size == 0 or true_components.size == 0 or imfs.shape[1] != true_components.shape[1]:
        return None

    k_est, k_true = imfs.shape[0], true_components.shape[0]
    assoc = np.zeros((k_est, k_true), dtype=float)
    for i in range(k_est):
        for j in range(k_true):
            assoc[i, j] = _safe_abs_corr_sq(imfs[i], true_components[j])

    energy = np.sum(imfs ** 2, axis=1)
    energy_ratio = energy / (np.sum(energy) + 1e-12)
    active = energy_ratio >= float(energy_threshold)
    gated = assoc.copy()
    gated[~active, :] = 0.0
    true_strength = np.sum(gated, axis=0)
    estimated_strength = np.sum(assoc, axis=1)
    associated_true = true_strength > float(association_threshold)
    active_unmatched = active & (estimated_strength <= float(association_threshold))

    component_rows = []
    dominant = []
    second = []
    effective_numbers = []
    associated_counts = []
    for j in range(k_true):
        col = gated[:, j]
        total = float(np.sum(col))
        if total <= 1e-12:
            weights = np.zeros(k_est, dtype=float)
        else:
            weights = col / total
        positive = np.sort(weights[weights > 0.0])[::-1]
        dom = float(positive[0]) if len(positive) else 0.0
        sec = float(positive[1]) if len(positive) > 1 else 0.0
        eff = float(1.0 / (np.sum(weights ** 2) + 1e-12)) if np.sum(weights) > 0 else 0.0
        count = int(np.sum(weights >= float(allocation_floor)))
        dominant.append(dom)
        second.append(sec)
        effective_numbers.append(eff)
        associated_counts.append(count)
        component_rows.append({
            "true_component_index": int(j),
            "true_component_associated": bool(associated_true[j]),
            "dominant_association_share": dom,
            "second_largest_association_share": sec,
            "effective_number_of_associations": eff,
            "associated_estimated_component_count": count,
            "association_vector": [float(v) for v in weights],
        })

    return {
        "component_rows": component_rows,
        "dominant_association_share_mean": float(np.mean(dominant)) if dominant else np.nan,
        "dominant_association_share_min": float(np.min(dominant)) if dominant else np.nan,
        "second_largest_association_share_mean": float(np.mean(second)) if second else np.nan,
        "effective_number_of_associations_mean": float(np.mean(effective_numbers)) if effective_numbers else np.nan,
        "associated_estimated_component_count_mean": float(np.mean(associated_counts)) if associated_counts else np.nan,
        "unmatched_estimated_component_count": int(np.sum(active_unmatched)),
        "unmatched_estimated_component_fraction": float(np.sum(active_unmatched) / max(int(np.sum(active)), 1)),
        "active_estimated_component_count": int(np.sum(active)),
    }


def _ols_fit_predict(x_train, y_train, x_test):
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    keep = np.isfinite(y_train) & np.all(np.isfinite(x_train), axis=1)
    if np.sum(keep) < x_train.shape[1]:
        return np.full(x_test.shape[0], np.nan)
    beta = np.linalg.pinv(x_train[keep]) @ y_train[keep]
    return x_test @ beta


def _cv_regression(rows, feature_names, fold_key="seed"):
    if not rows:
        return {
            "n": 0,
            "cv_r2": np.nan,
            "cv_mae": np.nan,
            "cv_residual_variance": np.nan,
        }
    strata = sorted({row["stratum"] for row in rows})
    stratum_index = {key: idx for idx, key in enumerate(strata)}
    folds = sorted({row[fold_key] for row in rows})
    y = np.asarray([row["component_splitting_index"] for row in rows], dtype=float)

    def design(sub_rows):
        mat = []
        for row in sub_rows:
            feats = [1.0]
            # Drop the first stratum to avoid a saturated intercept duplicate.
            for stratum in strata[1:]:
                feats.append(1.0 if row["stratum"] == stratum else 0.0)
            for name in feature_names:
                feats.append(float(row.get(name, np.nan)))
            mat.append(feats)
        return np.asarray(mat, dtype=float)

    pred = np.full(len(rows), np.nan, dtype=float)
    for fold in folds:
        train_idx = [idx for idx, row in enumerate(rows) if row[fold_key] != fold]
        test_idx = [idx for idx, row in enumerate(rows) if row[fold_key] == fold]
        if not train_idx or not test_idx:
            continue
        x_train = design([rows[idx] for idx in train_idx])
        y_train = y[train_idx]
        x_test = design([rows[idx] for idx in test_idx])
        pred[test_idx] = _ols_fit_predict(x_train, y_train, x_test)

    keep = np.isfinite(y) & np.isfinite(pred)
    if np.sum(keep) < 3:
        return {
            "n": int(np.sum(keep)),
            "cv_r2": np.nan,
            "cv_mae": np.nan,
            "cv_residual_variance": np.nan,
        }
    sse = float(np.sum((y[keep] - pred[keep]) ** 2))
    sst = float(np.sum((y[keep] - np.mean(y[keep])) ** 2))
    return {
        "n": int(np.sum(keep)),
        "cv_r2": float(1.0 - sse / (sst + 1e-12)),
        "cv_mae": float(np.mean(np.abs(y[keep] - pred[keep]))),
        "cv_residual_variance": float(np.var(y[keep] - pred[keep])),
    }


def _targeted_eemd_splitting_incremental_validity_review(full_case_results):
    case_rows = []
    component_rows = []
    for meta, case, results in full_case_results:
        result = results.get("EEMD", {}) if isinstance(results, dict) else {}
        if not isinstance(result, dict):
            continue
        profile = _eemd_association_profile(result.get("imfs"), case.get("true_components"))
        splitting = _finite(result.get("component_splitting_index"))
        eff_count = _finite(result.get("effective_imf_count"))
        if profile is None or splitting is None or eff_count is None:
            continue
        stratum = f"{meta.get('signal')}|{meta.get('noise')}|{meta.get('sigma')}"
        row = {
            **meta,
            "method": "EEMD",
            "metric": "component_splitting_index",
            "stratum": stratum,
            "effective_imf_count": int(eff_count),
            "component_splitting_index": float(splitting),
            "dominant_association_share_mean": profile["dominant_association_share_mean"],
            "dominant_association_share_min": profile["dominant_association_share_min"],
            "second_largest_association_share_mean": profile["second_largest_association_share_mean"],
            "effective_number_of_associations_mean": profile["effective_number_of_associations_mean"],
            "associated_estimated_component_count_mean": profile["associated_estimated_component_count_mean"],
            "unmatched_estimated_component_count": profile["unmatched_estimated_component_count"],
            "unmatched_estimated_component_fraction": profile["unmatched_estimated_component_fraction"],
            "active_estimated_component_count": profile["active_estimated_component_count"],
        }
        case_rows.append(row)
        for crow in profile["component_rows"]:
            component_rows.append({
                **meta,
                "method": "EEMD",
                "effective_imf_count": int(eff_count),
                "component_splitting_index": float(splitting),
                **crow,
            })

    fixed_count_rows = []
    grouped = defaultdict(list)
    for row in case_rows:
        grouped[(row["signal"], row["noise"], row["sigma"], row["effective_imf_count"])].append(row)
    for key, vals in sorted(grouped.items(), key=lambda item: str(item[0])):
        splits = np.asarray([v["component_splitting_index"] for v in vals], dtype=float)
        if len(splits) < 2:
            continue
        q75, q25 = np.percentile(splits, [75, 25])
        fixed_count_rows.append({
            "signal": key[0],
            "noise": key[1],
            "sigma": key[2],
            "effective_imf_count": key[3],
            "n": int(len(splits)),
            "splitting_min": float(np.min(splits)),
            "splitting_max": float(np.max(splits)),
            "splitting_range": float(np.max(splits) - np.min(splits)),
            "splitting_iqr": float(q75 - q25),
            "splitting_sd": float(np.std(splits, ddof=1)) if len(splits) > 1 else 0.0,
            "dominant_share_range": float(
                np.max([v["dominant_association_share_mean"] for v in vals])
                - np.min([v["dominant_association_share_mean"] for v in vals])
            ),
        })

    model_specs = {
        "fixed_effects_only": [],
        "count_only_plus_fixed_effects": ["effective_imf_count"],
        "count_plus_association_features": [
            "effective_imf_count",
            "dominant_association_share_mean",
            "second_largest_association_share_mean",
            "effective_number_of_associations_mean",
            "unmatched_estimated_component_fraction",
        ],
    }
    model_rows = []
    for name, features in model_specs.items():
        model_rows.append({
            "model": name,
            "features": features,
            **_cv_regression(case_rows, features),
        })
    model_map = {row["model"]: row for row in model_rows}
    delta_rows = []
    if "count_only_plus_fixed_effects" in model_map and "count_plus_association_features" in model_map:
        a = model_map["count_only_plus_fixed_effects"]
        b = model_map["count_plus_association_features"]
        delta_rows.append({
            "comparison": "association_features_increment_over_count_only",
            "delta_cv_r2": (
                float(b["cv_r2"] - a["cv_r2"])
                if np.isfinite(a["cv_r2"]) and np.isfinite(b["cv_r2"]) else np.nan
            ),
            "delta_cv_mae": (
                float(b["cv_mae"] - a["cv_mae"])
                if np.isfinite(a["cv_mae"]) and np.isfinite(b["cv_mae"]) else np.nan
            ),
            "delta_cv_residual_variance": (
                float(b["cv_residual_variance"] - a["cv_residual_variance"])
                if np.isfinite(a["cv_residual_variance"]) and np.isfinite(b["cv_residual_variance"]) else np.nan
            ),
        })

    matched_examples = []
    for key, vals in grouped.items():
        if len(vals) < 2:
            continue
        vals = sorted(vals, key=lambda row: row["component_splitting_index"])
        low = vals[0]
        high = vals[-1]
        diff = high["component_splitting_index"] - low["component_splitting_index"]
        if diff <= 1e-6:
            continue
        matched_examples.append({
            "signal": key[0],
            "noise": key[1],
            "sigma": key[2],
            "effective_imf_count": key[3],
            "seed_low": low["seed"],
            "seed_high": high["seed"],
            "splitting_low": low["component_splitting_index"],
            "splitting_high": high["component_splitting_index"],
            "splitting_difference": float(diff),
            "dominant_share_low": low["dominant_association_share_mean"],
            "dominant_share_high": high["dominant_association_share_mean"],
            "second_share_low": low["second_largest_association_share_mean"],
            "second_share_high": high["second_largest_association_share_mean"],
            "effective_association_number_low": low["effective_number_of_associations_mean"],
            "effective_association_number_high": high["effective_number_of_associations_mean"],
        })
    matched_examples = sorted(
        matched_examples,
        key=lambda row: row["splitting_difference"],
        reverse=True,
    )[:30]

    count_model = model_map.get("count_only_plus_fixed_effects", {})
    assoc_model = model_map.get("count_plus_association_features", {})
    fixed_model = model_map.get("fixed_effects_only", {})
    delta = delta_rows[0] if delta_rows else {}
    summary = [{
        "review": "targeted_eemd_splitting_incremental_validity",
        "qualification_audit_version": QUALIFICATION_AUDIT_VERSION,
        "n_cases": int(len(case_rows)),
        "n_fixed_count_groups_with_variation": int(len(fixed_count_rows)),
        "median_fixed_count_splitting_range": (
            float(np.median([r["splitting_range"] for r in fixed_count_rows]))
            if fixed_count_rows else np.nan
        ),
        "max_fixed_count_splitting_range": (
            float(np.max([r["splitting_range"] for r in fixed_count_rows]))
            if fixed_count_rows else np.nan
        ),
        "count_only_cv_r2": count_model.get("cv_r2", np.nan),
        "count_only_cv_mae": count_model.get("cv_mae", np.nan),
        "fixed_effects_only_cv_r2": fixed_model.get("cv_r2", np.nan),
        "fixed_effects_only_cv_mae": fixed_model.get("cv_mae", np.nan),
        "delta_cv_r2_count_over_fixed_effects": (
            float(count_model.get("cv_r2", np.nan) - fixed_model.get("cv_r2", np.nan))
            if np.isfinite(count_model.get("cv_r2", np.nan))
            and np.isfinite(fixed_model.get("cv_r2", np.nan)) else np.nan
        ),
        "association_model_cv_r2": assoc_model.get("cv_r2", np.nan),
        "association_model_cv_mae": assoc_model.get("cv_mae", np.nan),
        "delta_cv_r2_association_over_count": delta.get("delta_cv_r2", np.nan),
        "delta_cv_mae_association_over_count": delta.get("delta_cv_mae", np.nan),
        "matched_count_example_count": int(len(matched_examples)),
        "interpretation_guardrail": (
            "review tests incremental validity of EEMD splitting; it is not a method ranking"
        ),
    }]
    return {
        "case_rows": case_rows,
        "component_rows": component_rows,
        "fixed_count_variation_rows": fixed_count_rows,
        "model_rows": model_rows,
        "model_delta_rows": delta_rows,
        "matched_examples": matched_examples,
        "summary": summary,
    }


def _adjudicate_imf_count_dependency(high_dependency, eemd_incremental_review):
    automated_status = "requires_review" if high_dependency else "passed"
    summary = (
        eemd_incremental_review.get("summary", [{}])[0]
        if isinstance(eemd_incremental_review, dict)
        else {}
    )
    only_eemd_splitting = bool(high_dependency) and all(
        row.get("method") == "EEMD"
        and row.get("metric") == "component_splitting_index"
        for row in high_dependency
    )
    count_increment = _finite(summary.get("delta_cv_r2_count_over_fixed_effects"))
    assoc_increment = _finite(summary.get("delta_cv_r2_association_over_count"))
    delta_mae = _finite(summary.get("delta_cv_mae_association_over_count"))
    n_fixed = int(summary.get("n_fixed_count_groups_with_variation") or 0)
    n_matched = int(summary.get("matched_count_example_count") or 0)
    n_cases = int(summary.get("n_cases") or 0)
    incremental_passed = bool(
        n_cases >= 100
        and n_fixed >= 10
        and n_matched >= 10
        and assoc_increment is not None
        and assoc_increment >= 0.05
        and delta_mae is not None
        and delta_mae < 0.0
        and count_increment is not None
        and count_increment <= 0.02
    )
    if not high_dependency:
        final_status = "passed"
    elif only_eemd_splitting and incremental_passed:
        final_status = "passed_with_method_behavior_note"
    else:
        final_status = "requires_review"
    return {
        "automated_dependency_gate_status": automated_status,
        "targeted_incremental_validity_review_status": (
            "passed" if incremental_passed else "not_passed_or_not_applicable"
        ),
        "final_dependency_qualification_status": final_status,
        "adjudicated_rows": high_dependency if only_eemd_splitting else [],
        "adjudication_reason": (
            "EEMD splitting/count dependency is documented as coupled method behavior; "
            "targeted incremental-validity review shows component_splitting_index "
            "retains association-structure information beyond effective IMF count"
            if final_status == "passed_with_method_behavior_note"
            else ""
        ),
        "interpretation_caveat": (
            "For EEMD, component_splitting_index should not be interpreted as "
            "statistically independent of effective IMF count; both may respond "
            "jointly to the same decomposition behavior. The metric is retained "
            "because matched-count and predictive incremental-validity analyses "
            "show information beyond component cardinality alone."
            if final_status == "passed_with_method_behavior_note"
            else ""
        ),
        "metric_definition_changed": False,
        "primary_endpoint_changed": False,
        "applicability_policy_changed": False,
    }


def _legacy_isolation_audit(rows):
    primary_legacy = sorted(set(PRIMARY_ANALYSIS_METRICS) & STRUCTURAL_LEGACY_FIELDS)
    mismatches = []
    usage_flags = []
    for row, method, summary in _method_rows(rows):
        legacy = _finite(summary.get("mode_mixing_index_legacy"))
        entangle = _finite(summary.get("inter_imf_entanglement_index"))
        if legacy is not None and entangle is not None and abs(legacy - entangle) > 1e-12:
            mismatches.append({
                "signal": row.get("signal"),
                "noise": row.get("noise"),
                "sigma": row.get("sigma"),
                "seed": row.get("seed"),
                "method": method,
                "mode_mixing_index_legacy": legacy,
                "inter_imf_entanglement_index": entangle,
            })
        flag = summary.get("legacy_metric_used_in_primary_analysis")
        if str(flag).strip().lower() in {"true", "1", "yes"}:
            usage_flags.append({
                "signal": row.get("signal"),
                "noise": row.get("noise"),
                "sigma": row.get("sigma"),
                "seed": row.get("seed"),
                "method": method,
            })
    status = (
        "passed"
        if not primary_legacy and not mismatches and not usage_flags
        else "requires_review"
    )
    return [{
        "audit": "legacy_metric_isolation",
        "status": status,
        "primary_legacy_metric_count": int(len(primary_legacy)),
        "primary_legacy_metrics": primary_legacy,
        "legacy_alias_mismatch_count": int(len(mismatches)),
        "legacy_used_in_primary_flag_count": int(len(usage_flags)),
        "mismatch_examples": mismatches[:10],
        "usage_flag_examples": usage_flags[:10],
    }]


def _method_failure_audit(rows):
    grouped = defaultdict(list)
    for _, method, summary in _method_rows(rows):
        grouped[method].append(summary)
    out = []
    for method, vals in sorted(grouped.items()):
        failures = [
            v for v in vals
            if bool(v.get("any_failure", False))
            or bool(v.get("timeout_flag", False))
            or bool(v.get("exception_flag", False))
        ]
        out.append({
            "method": method,
            "n_method_runs": int(len(vals)),
            "failure_count": int(len(failures)),
            "timeout_count": int(sum(bool(v.get("timeout_flag", False)) for v in vals)),
            "exception_count": int(sum(bool(v.get("exception_flag", False)) for v in vals)),
            "nonfinite_output_count": int(sum(bool(v.get("nonfinite_output_flag", False)) for v in vals)),
            "failures_documented": True,
            "failure_rate": float(len(failures) / max(len(vals), 1)),
        })
    return out


def _write_freeze_report(output_root, dashboard):
    status = dashboard.get("benchmark_validation_status")
    ready = status == "structural_metric_schema_validated_and_frozen"
    lines = [
        "# V5.24 Structural Metric Qualification Report",
        "",
        f"- Evaluation framework version: `{EVALUATION_FRAMEWORK_VERSION}`",
        f"- Framework status: `{EVALUATION_FRAMEWORK_STATUS}`",
        f"- Framework frozen date: `{EVALUATION_FRAMEWORK_FROZEN_DATE}`",
        f"- Structural metric schema: `{STRUCTURAL_METRIC_SCHEMA_VERSION}`",
        f"- Qualification audit version: `{dashboard.get('qualification_audit_version', QUALIFICATION_AUDIT_VERSION)}`",
        f"- Qualification status: `{status}`",
        "",
        "## Qualification Scope",
        "",
        "The present qualification run assesses the `TRUE_COMPONENT_MIXING_V1.0` structural metric schema, with primary focus on `component_splitting_index` and `component_merging_index`.",
        "It does not constitute equal-depth qualification of all V5.24 primary endpoints, such as standard reconstruction metrics, component-recovery matching metrics, or contamination-specific endpoints.",
        "No method ranking, performance comparison, or scientific conclusion regarding IRMF, EMD, EEMD, or CEEMDAN is made in this stage.",
        "A successful qualification decision authorizes subsequent benchmark execution; it does not itself constitute benchmark evidence.",
        "",
        "## Qualification Pipeline",
        "",
        "1. Metric definition",
        "2. Numerical validation",
        "3. Applicability validation",
        "4. Threshold robustness",
        "5. Controlled construct validation",
        "6. Dependency audit",
        "7. Targeted scientific review",
        "8. Qualification adjudication",
        "9. Schema freeze",
        "10. Benchmark authorization",
        "",
        "## Primary Evaluation Dimensions",
        "",
        "- [x] Signal Recovery",
        "- [x] Component Recovery",
        "- [x] Component Allocation Fidelity",
        "- [x] Noise Separation",
        "- [x] Contamination-specific conditional primary endpoints with applicability gate",
        "",
        "## Validation Gate",
        "",
    ]
    for key, value in dashboard.get("exit_criteria", {}).items():
        marker = "x" if (
            value == "passed"
            or str(value).startswith("passed_with")
            or str(value).startswith("external_required")
        ) else " "
        lines.append(f"- [{marker}] `{key}`: `{value}`")
    adjudication = dashboard.get("imf_count_dependency_final_adjudication", {})
    lines.extend([
        "",
        "## Legacy Metrics",
        "",
        "- `mode_mixing_index`: legacy compatibility field only.",
        "- `mode_mixing_index_legacy`: explicit legacy alias.",
        "- `inter_imf_entanglement_index`: preferred diagnostic name for no-truth internal IMF entanglement.",
        "- Legacy metrics are excluded from primary structural endpoint analysis.",
        "",
        "## IMF-Count Dependency Rule",
        "",
        "- Pooled IMF-count correlation is treated as a screening warning, not as a standalone failure criterion.",
        "- Single extreme within-stratum correlations are written to `imf_count_dependency_extreme_strata_review.csv`.",
        "- Schema qualification is based on residualized comparable-case dependency, systematic stratified dependency evidence, and controlled construct behavior tests.",
        "- Metric values and metric definitions are unchanged by qualification audit V1.2.",
        "",
        "## Post-Audit Adjudication",
        "",
        f"- Automated dependency gate status: `{adjudication.get('automated_dependency_gate_status', '')}`",
        f"- Targeted incremental-validity review status: `{adjudication.get('targeted_incremental_validity_review_status', '')}`",
        f"- Final dependency qualification status: `{adjudication.get('final_dependency_qualification_status', '')}`",
        f"- Interpretation caveat: {adjudication.get('interpretation_caveat', '')}",
        "",
        "## Freeze Decision",
        "",
    ])
    if ready:
        lines.extend([
            "- [x] TRUE_COMPONENT_MIXING_V1.0 has completed empirical qualification under the predefined metric qualification protocol and is frozen for benchmark use.",
            "- [x] Full V5.24 structural benchmark rerun authorized.",
            "- [ ] Full V5.24 structural benchmark rerun completed.",
            "- [ ] Statistical benchmark analysis completed.",
            "- [ ] Scientific method conclusions authorized.",
        ])
    else:
        lines.extend([
            "- [ ] TRUE_COMPONENT_MIXING_V1.0 not yet authorized for full structural conclusions.",
            "- [ ] Review dashboard items marked `requires_review` before full rerun/backfill.",
        ])
    lines.extend([
        "",
        "## Project Stage Boundary",
        "",
        "- [x] Scientific unit tests completed.",
        "- [x] Evaluation framework design completed.",
        "- [x] Structural metric qualification completed." if ready else "- [ ] Structural metric qualification completed.",
        "- [x] Schema freeze completed." if ready else "- [ ] Schema freeze completed.",
        "- [x] Full benchmark execution authorized." if ready else "- [ ] Full benchmark execution authorized.",
        "- [ ] Full benchmark execution completed.",
        "- [ ] Statistical benchmark analysis completed.",
        "- [ ] Scientific conclusions authorized.",
        "",
        "## Protocol Reusability",
        "",
        "The present qualification protocol is implemented here for the `TRUE_COMPONENT_MIXING_V1.0` structural metric schema.",
        "Future evaluation metrics may undergo analogous predefined qualification procedures before being incorporated into subsequent benchmark framework revisions.",
    ])
    lines.append("")
    (output_root / "evaluation_framework_freeze_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run_structural_metric_validation_slice(
        output_root,
        irmf_params=GLOBAL_IRMF_PARAMS,
        emd_params=GLOBAL_EMD_PARAMS,
        eemd_params=GLOBAL_EEMD_PARAMS,
        ceemdan_params=GLOBAL_CEEMDAN_PARAMS,
        signals=VALIDATION_SIGNALS,
        noises=VALIDATION_NOISES,
        sigmas=VALIDATION_SIGMAS,
        seeds=VALIDATION_SEEDS,
        timeout_seconds=DEFAULT_METHOD_TIMEOUT_SECONDS,
        quick=False,
):
    output_root = ensure_dir(output_root)
    if quick:
        signals = tuple(signals[:2])
        noises = tuple(noises[:2])
        sigmas = tuple(sigmas[:1])
        seeds = tuple(seeds[:1])

    rows = []
    full_case_results = []
    for signal_name in signals:
        for noise_name in noises:
            for sigma in sigmas:
                for seed in seeds:
                    algorithm_seed = _algorithm_seed_for(int(seed))
                    case, results, row = run_fixed_method_family_case(
                        signal_name=signal_name,
                        noise_name=noise_name,
                        sigma=float(sigma),
                        irmf_params=irmf_params,
                        emd_params=emd_params,
                        eemd_params=eemd_params,
                        ceemdan_params=ceemdan_params,
                        n=DEFAULT_N,
                        fs=DEFAULT_FS,
                        seed=int(seed),
                        algorithm_seed=algorithm_seed,
                        run_id_prefix=f"v524_metric_validation_{signal_name}_{noise_name}_{sigma}_seed{seed}",
                        timeout_seconds=timeout_seconds,
                    )
                    row["metric_validation_slice"] = True
                    row["structural_metric_schema_version"] = STRUCTURAL_METRIC_SCHEMA_VERSION
                    rows.append(row)
                    full_case_results.append((
                        {
                            "signal": signal_name,
                            "noise": noise_name,
                            "sigma": float(sigma),
                            "seed": int(seed),
                        },
                        case,
                        results,
                    ))

    range_rows = _range_audit(rows)
    gating_rows = _gating_audit(rows)
    imf_dependency_rows, imf_dependency_strata_rows = _imf_count_dependency_audit(rows)
    threshold_rows = _threshold_sensitivity_rows(full_case_results)
    threshold_summary = _threshold_sensitivity_summary(threshold_rows)
    threshold_stability = _threshold_sign_stability_audit(threshold_rows)
    all_gated_rows = _all_gated_audit(rows)
    legacy_rows = _legacy_isolation_audit(rows)
    failure_rows = _method_failure_audit(rows)
    controlled_rows, controlled_summary = _controlled_construct_dependency_audit()
    eemd_incremental_review = _targeted_eemd_splitting_incremental_validity_review(full_case_results)

    write_json(rows, output_root / "structural_metric_validation_rows.json")
    write_csv(rows, output_root / "structural_metric_validation_rows.csv")
    write_json(range_rows, output_root / "range_audit.json")
    write_csv(range_rows, output_root / "range_audit.csv")
    write_json(gating_rows, output_root / "gating_audit.json")
    write_csv(gating_rows, output_root / "gating_audit.csv")
    write_json(imf_dependency_rows, output_root / "imf_count_dependency_audit.json")
    write_csv(imf_dependency_rows, output_root / "imf_count_dependency_audit.csv")
    write_json(imf_dependency_strata_rows, output_root / "imf_count_dependency_stratified_audit.json")
    write_csv(imf_dependency_strata_rows, output_root / "imf_count_dependency_stratified_audit.csv")
    extreme_strata_rows = [
        row for row in imf_dependency_strata_rows
        if row.get("extreme_stratum")
    ]
    write_json(extreme_strata_rows, output_root / "imf_count_dependency_extreme_strata_review.json")
    write_csv(extreme_strata_rows, output_root / "imf_count_dependency_extreme_strata_review.csv")
    write_json(threshold_rows, output_root / "association_threshold_sensitivity_rows.json")
    write_csv(threshold_rows, output_root / "association_threshold_sensitivity_rows.csv")
    write_json(threshold_summary, output_root / "association_threshold_sensitivity_summary.json")
    write_csv(threshold_summary, output_root / "association_threshold_sensitivity_summary.csv")
    write_json(threshold_stability, output_root / "association_threshold_sign_stability_audit.json")
    write_csv(threshold_stability, output_root / "association_threshold_sign_stability_audit.csv")
    write_json(all_gated_rows, output_root / "all_gated_case_audit.json")
    write_csv(all_gated_rows, output_root / "all_gated_case_audit.csv")
    write_json(legacy_rows, output_root / "legacy_metric_isolation_audit.json")
    write_csv(legacy_rows, output_root / "legacy_metric_isolation_audit.csv")
    write_json(failure_rows, output_root / "method_failure_audit.json")
    write_csv(failure_rows, output_root / "method_failure_audit.csv")
    write_json(controlled_rows, output_root / "controlled_construct_dependency_rows.json")
    write_csv(controlled_rows, output_root / "controlled_construct_dependency_rows.csv")
    write_json(controlled_summary, output_root / "controlled_construct_dependency_audit.json")
    write_csv(controlled_summary, output_root / "controlled_construct_dependency_audit.csv")
    write_json(
        eemd_incremental_review["summary"],
        output_root / "eemd_splitting_incremental_validity_summary.json",
    )
    write_csv(
        eemd_incremental_review["summary"],
        output_root / "eemd_splitting_incremental_validity_summary.csv",
    )
    write_json(
        eemd_incremental_review["fixed_count_variation_rows"],
        output_root / "eemd_splitting_fixed_count_variation.json",
    )
    write_csv(
        eemd_incremental_review["fixed_count_variation_rows"],
        output_root / "eemd_splitting_fixed_count_variation.csv",
    )
    write_json(
        eemd_incremental_review["model_rows"],
        output_root / "eemd_splitting_count_prediction_models.json",
    )
    write_csv(
        eemd_incremental_review["model_rows"],
        output_root / "eemd_splitting_count_prediction_models.csv",
    )
    write_json(
        eemd_incremental_review["model_delta_rows"],
        output_root / "eemd_splitting_model_delta.json",
    )
    write_csv(
        eemd_incremental_review["model_delta_rows"],
        output_root / "eemd_splitting_model_delta.csv",
    )
    write_json(
        eemd_incremental_review["matched_examples"],
        output_root / "eemd_splitting_matched_count_examples.json",
    )
    write_csv(
        eemd_incremental_review["matched_examples"],
        output_root / "eemd_splitting_matched_count_examples.csv",
    )
    write_json(
        eemd_incremental_review["component_rows"],
        output_root / "eemd_splitting_component_source_rows.json",
    )
    write_csv(
        eemd_incremental_review["component_rows"],
        output_root / "eemd_splitting_component_source_rows.csv",
    )

    failed_range = sum(1 for row in range_rows if row.get("range_audit_status") != "passed")
    pooled_dependency_screening = [
        row for row in imf_dependency_rows
        if row.get("pooled_screening_warning")
    ]
    high_dependency = [
        row for row in imf_dependency_rows
        if row.get("dependency_audit_status") == "requires_review"
    ]
    imf_dependency_adjudication = _adjudicate_imf_count_dependency(
        high_dependency,
        eemd_incremental_review,
    )
    write_json(
        imf_dependency_adjudication,
        output_root / "imf_count_dependency_final_adjudication.json",
    )
    write_csv(
        [imf_dependency_adjudication],
        output_root / "imf_count_dependency_final_adjudication.csv",
    )
    all_gated_failures = [
        row for row in all_gated_rows
        if row.get("all_gated_audit_status") != "passed"
    ]
    threshold_reviews = [
        row for row in threshold_stability
        if row.get("threshold_sensitivity_status") == "requires_review"
    ]
    legacy_status = legacy_rows[0]["status"] if legacy_rows else "requires_review"
    controlled_status = controlled_summary[0]["status"] if controlled_summary else "requires_review"
    exit_criteria = {
        "scientific_unit_tests": "external_required_passed_before_this_stage",
        "range_and_finite_values": "passed" if failed_range == 0 else "failed",
        "legacy_metric_isolation": legacy_status,
        "gating_coverage_report_generated": "passed" if gating_rows else "failed",
        "all_gated_case_rate": "passed" if not all_gated_failures else "requires_review",
        "imf_count_dependency": imf_dependency_adjudication[
            "final_dependency_qualification_status"
        ],
        "controlled_construct_dependency": controlled_status,
        "threshold_sensitivity": "passed" if not threshold_reviews else "requires_review",
        "method_failures_documented": "passed" if failure_rows else "failed",
        "validation_artifacts_archived": "passed",
    }
    blocking_statuses = {"failed", "requires_review"}
    full_exit_ready = not any(value in blocking_statuses for value in exit_criteria.values())
    dashboard = {
        "stage": "V5.24 structural metric schema qualification",
        "role": (
            "qualification of the TRUE_COMPONENT_MIXING_V1.0 structural metric schema only; "
            "not a paper-level method ranking"
        ),
        "qualification_scope_level": "structural_metric_schema_only",
        "qualification_scope": (
            "assesses TRUE_COMPONENT_MIXING_V1.0 structural metrics, especially "
            "component_splitting_index and component_merging_index; does not provide "
            "equal-depth qualification for all V5.24 primary endpoints and does not "
            "make method rankings, performance comparisons, or scientific conclusions "
            "about IRMF, EMD, EEMD, or CEEMDAN"
        ),
        "non_structural_primary_endpoint_qualification_status": (
            "outside_scope_of_this_structural_metric_schema_qualification"
        ),
        "evaluation_framework_version": EVALUATION_FRAMEWORK_VERSION,
        "evaluation_framework_status": EVALUATION_FRAMEWORK_STATUS,
        "evaluation_framework_frozen_date": EVALUATION_FRAMEWORK_FROZEN_DATE,
        "qualification_audit_version": QUALIFICATION_AUDIT_VERSION,
        "qualification_audit_change_reason": (
            "pooled IMF-count correlation treated as screening; schema qualification "
            "uses residualized comparable-case dependency, stratified evidence, and "
            "controlled construct behavior tests"
        ),
        "supersedes_qualification_audit_version": "V1.0",
        "metric_values_and_definitions_changed": False,
        "structural_metric_schema_version": STRUCTURAL_METRIC_SCHEMA_VERSION,
        "signals": list(signals),
        "noises": list(noises),
        "sigmas": list(sigmas),
        "seeds": list(seeds),
        "methods": list(METHODS),
        "n_signal_noise_sigma_seed_cells": int(len(rows)),
        "n_method_evaluations": int(len(rows) * len(METHODS)),
        "range_audit_failed_rows": int(failed_range),
        "imf_count_pooled_screening_warnings": pooled_dependency_screening,
        "imf_count_dependency_reviews": high_dependency,
        "imf_count_dependency_final_adjudication": imf_dependency_adjudication,
        "imf_count_extreme_strata_review_count": int(len(extreme_strata_rows)),
        "all_gated_case_warnings": all_gated_failures,
        "threshold_sign_stability_reviews": threshold_reviews,
        "legacy_metric_isolation_status": legacy_status,
        "controlled_construct_dependency_status": controlled_status,
        "targeted_eemd_splitting_incremental_validity_review": (
            eemd_incremental_review["summary"][0] if eemd_incremental_review["summary"] else {}
        ),
        "method_failure_audit": failure_rows,
        "exit_criteria": exit_criteria,
        "benchmark_validation_status": (
            "passed_initial_audit" if quick and failed_range == 0 and legacy_status == "passed"
            else "structural_metric_schema_validated_and_frozen"
            if full_exit_ready else "requires_review"
        ),
        "outputs": {
            "rows": "structural_metric_validation_rows.csv",
            "range_audit": "range_audit.csv",
            "gating_audit": "gating_audit.csv",
            "all_gated_case_audit": "all_gated_case_audit.csv",
            "imf_count_dependency_audit": "imf_count_dependency_audit.csv",
            "imf_count_dependency_stratified_audit": "imf_count_dependency_stratified_audit.csv",
            "imf_count_dependency_extreme_strata_review": "imf_count_dependency_extreme_strata_review.csv",
            "imf_count_dependency_final_adjudication": "imf_count_dependency_final_adjudication.csv",
            "controlled_construct_dependency_audit": "controlled_construct_dependency_audit.csv",
            "eemd_splitting_incremental_validity_summary": "eemd_splitting_incremental_validity_summary.csv",
            "eemd_splitting_fixed_count_variation": "eemd_splitting_fixed_count_variation.csv",
            "eemd_splitting_count_prediction_models": "eemd_splitting_count_prediction_models.csv",
            "eemd_splitting_matched_count_examples": "eemd_splitting_matched_count_examples.csv",
            "eemd_splitting_component_source_rows": "eemd_splitting_component_source_rows.csv",
            "association_threshold_sensitivity": "association_threshold_sensitivity_summary.csv",
            "association_threshold_sign_stability_audit": "association_threshold_sign_stability_audit.csv",
            "legacy_metric_isolation_audit": "legacy_metric_isolation_audit.csv",
            "method_failure_audit": "method_failure_audit.csv",
        },
    }
    write_json(dashboard, output_root / "structural_metric_validation_dashboard.json")
    _write_freeze_report(output_root, dashboard)
    return dashboard
