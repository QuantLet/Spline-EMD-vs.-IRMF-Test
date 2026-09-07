#!/usr/bin/python
# coding: UTF-8

"""Directionality audit for regenerated V5.27 statistics."""

import csv
from pathlib import Path

from project_config import (
    LOWER_IS_BETTER_METRICS,
    V527_PRIMARY_ENDPOINTS_BY_DIMENSION,
    V527_PRIMARY_METRIC_CODE_FIELD_MAP,
    V527_PRIMARY_SCHEMA_VERSION,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


EXPECTED_LOWER_IS_BETTER = {
    "denoise_nmse",
    "matched_component_nrmse",
    "relative_decomposition_count_error",
    "missing_true_component_energy_ratio",
    "spurious_mode_energy_ratio",
    "noise_energy_log_error",
    "signal_leakage_into_noise",
    "contaminated_region_nmse",
    "clean_region_nmse",
    "contamination_spillover_error",
}
EXPECTED_HIGHER_IS_BETTER = {
    "denoise_corr",
    "matched_component_corr",
    "noise_capture_corr",
}


def _all_metrics():
    out = []
    for metrics in V527_PRIMARY_ENDPOINTS_BY_DIMENSION.values():
        out.extend(metrics)
    return tuple(dict.fromkeys(out))


def _read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _config_rows():
    lower = set(LOWER_IS_BETTER_METRICS)
    rows = []
    for dimension, metrics in V527_PRIMARY_ENDPOINTS_BY_DIMENSION.items():
        for metric in metrics:
            code_field = V527_PRIMARY_METRIC_CODE_FIELD_MAP[metric]
            actual_lower = metric in lower or code_field in lower
            expected_lower = metric in EXPECTED_LOWER_IS_BETTER
            rows.append({
                "audit_layer": "config",
                "dimension": dimension,
                "metric": metric,
                "code_field": code_field,
                "expected_higher_is_better": metric in EXPECTED_HIGHER_IS_BETTER,
                "expected_lower_is_better": expected_lower,
                "actual_lower_is_better": actual_lower,
                "actual_higher_is_better": not actual_lower,
                "directionality_status": "passed" if actual_lower == expected_lower else "failed",
            })
    return rows


def _paired_rows(stats_root):
    path = Path(stats_root) / "paired_effect_sizes.csv"
    rows = _read_csv(path)
    lookup = {}
    for row in rows:
        if row.get("group") == "all" and row.get("comparison") == "IRMF_vs_EMD":
            lookup[row.get("metric")] = row
    out = []
    for metric in _all_metrics():
        row = lookup.get(metric, {})
        expected_higher = metric in EXPECTED_HIGHER_IS_BETTER
        actual = str(row.get("higher_is_better", "")).strip().lower()
        actual_higher = actual == "true"
        out.append({
            "audit_layer": "paired_effect_sizes",
            "metric": metric,
            "expected_higher_is_better": expected_higher,
            "actual_higher_is_better": actual_higher,
            "mean_paired_gain_irmf_vs_emd": row.get("mean_paired_gain"),
            "win_rate_irmf_vs_emd": row.get("win_rate"),
            "directionality_status": "passed" if actual_higher == expected_higher else "failed",
        })
    return out


def _summary_rows(stats_root):
    path = Path(stats_root) / "method_metric_summary.csv"
    rows = _read_csv(path)
    lookup = {}
    for row in rows:
        if row.get("group") == "all" and row.get("method") == "IRMF":
            lookup[row.get("metric")] = row
    out = []
    for metric in _all_metrics():
        row = lookup.get(metric, {})
        expected_higher = metric in EXPECTED_HIGHER_IS_BETTER
        actual = str(row.get("higher_is_better", "")).strip().lower()
        actual_higher = actual == "true"
        out.append({
            "audit_layer": "method_metric_summary",
            "metric": metric,
            "expected_higher_is_better": expected_higher,
            "actual_higher_is_better": actual_higher,
            "directionality_status": "passed" if actual_higher == expected_higher else "failed",
        })
    return out


def _rank_coverage_rows(stats_root):
    rows = _read_csv(Path(stats_root) / "primary_metric_ranks.csv")
    metrics = _all_metrics()
    out = []
    for metric in metrics:
        metric_rows = [r for r in rows if r.get("metric") == metric]
        out.append({
            "audit_layer": "rank_coverage",
            "metric": metric,
            "n_rank_rows": len(metric_rows),
            "directionality_status": "passed" if metric_rows else "failed",
            "note": "Rank direction is inherited from config lower/higher-is-better policy.",
        })
    return out


def run_v527_directionality_audit(output_root, stats_root):
    output_root = ensure_dir(output_root)
    stats_root = Path(stats_root)
    rows = _config_rows() + _paired_rows(stats_root) + _summary_rows(stats_root) + _rank_coverage_rows(stats_root)
    failures = [r for r in rows if r.get("directionality_status") != "passed"]
    dashboard = {
        "stage": "v527_primary_directionality_audit",
        "schema_version": V527_PRIMARY_SCHEMA_VERSION,
        "schema_status": "frozen",
        "statistics_root": str(stats_root),
        "audit_status": "passed" if not failures else "requires_review",
        "n_checks": len(rows),
        "n_failures": len(failures),
        "expected_lower_is_better": sorted(EXPECTED_LOWER_IS_BETTER),
        "expected_higher_is_better": sorted(EXPECTED_HIGHER_IS_BETTER),
        "scope": (
            "Audits directionality in config, paired effect labels, summary "
            "labels, and primary rank coverage. Ranking and Friedman/CD inherit "
            "direction from the same config policy."
        ),
    }
    write_csv(rows, output_root / "v527_primary_directionality_audit.csv")
    write_json(dashboard, output_root / "v527_primary_directionality_audit_dashboard.json")
    return dashboard
