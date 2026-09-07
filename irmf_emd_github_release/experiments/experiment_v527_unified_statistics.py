#!/usr/bin/python
# coding: UTF-8

"""Regenerate statistics under the frozen V5.27 primary schema.

This stage reuses the existing protocol-controlled benchmark cube.  It does
not rerun decomposition methods or recompute metric values; it remaps existing
code fields to the V5.27 publication-layer endpoint names and regenerates the
statistical layer under the frozen 13-endpoint schema.
"""

from contextlib import contextmanager
from pathlib import Path
import json

import numpy as np

from project_config import (
    CASE_SCORE_ROLE,
    COMPOSITE_SENSITIVITY_ENDPOINTS,
    DIAGNOSTIC_METRICS_BY_DIMENSION,
    EVALUATION_METHODS,
    LOWER_IS_BETTER_METRICS,
    V527_CONTAMINATION_RESISTANCE_CONSTRUCTS,
    V527_DIAGNOSTIC_DEMOTIONS,
    V527_NOISE_SEPARATION_CONSTRUCTS,
    V527_PRIMARY_ENDPOINTS_BY_DIMENSION,
    V527_PRIMARY_METRIC_CODE_FIELD_MAP,
    V527_PRIMARY_SCHEMA_VERSION,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json
from experiments import experiment_unified_cube_statistics as base_stats


METHODS = tuple(EVALUATION_METHODS)
BASELINES = tuple(m for m in METHODS if m != "IRMF")
CONTAMINATION_NOISES = ("impulsive", "burst", "huber_contamination")
FROZEN_SCHEMA_FILE = "v527_primary_schema_frozen.json"
FREEZE_MANIFEST_FILE = "v527_primary_schema_freeze_manifest.json"


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _all_v527_metrics():
    out = []
    for metrics in V527_PRIMARY_ENDPOINTS_BY_DIMENSION.values():
        out.extend(metrics)
    return tuple(dict.fromkeys(out))


def _subdimension_config(metrics, construct_map=None):
    construct_map = construct_map or {}
    subdimensions = {}
    for metric in metrics:
        subdimensions[construct_map.get(metric, metric)] = (metric,)
    return {"subdimensions": subdimensions}


def _v527_universal_primary_config():
    return {
        dimension: _subdimension_config(
            metrics,
            {
                **V527_NOISE_SEPARATION_CONSTRUCTS,
            },
        )
        for dimension, metrics in V527_PRIMARY_ENDPOINTS_BY_DIMENSION.items()
        if dimension != "contamination_resistance"
    }


def _v527_conditional_primary_config():
    metrics = V527_PRIMARY_ENDPOINTS_BY_DIMENSION["contamination_resistance"]
    return {
        "contamination_resistance": {
            **_subdimension_config(metrics, V527_CONTAMINATION_RESISTANCE_CONSTRUCTS),
            "applicable_noises": CONTAMINATION_NOISES,
        }
    }


def _copy_publication_metric_fields(rows):
    copied = []
    for row in rows:
        new_row = dict(row)
        for method in METHODS:
            for publication_metric, code_field in V527_PRIMARY_METRIC_CODE_FIELD_MAP.items():
                src = f"{method}_{code_field}"
                dst = f"{method}_{publication_metric}"
                if src in new_row and dst not in new_row:
                    new_row[dst] = new_row[src]
        copied.append(new_row)
    return copied


def _v527_primary_metrics_by_domain():
    conditional = tuple(V527_PRIMARY_ENDPOINTS_BY_DIMENSION["contamination_resistance"])
    universal = tuple(m for m in _all_v527_metrics() if m not in conditional)
    return universal, conditional


def _v527_lower_is_better_metrics():
    lower = set(LOWER_IS_BETTER_METRICS)
    for publication_metric, code_field in V527_PRIMARY_METRIC_CODE_FIELD_MAP.items():
        if code_field in lower:
            lower.add(publication_metric)
    return lower


@contextmanager
def _patched_base_stats_globals():
    universal, conditional = _v527_primary_metrics_by_domain()
    saved = {
        "PRIMARY_ENDPOINTS_BY_DIMENSION": base_stats.PRIMARY_ENDPOINTS_BY_DIMENSION,
        "CONDITIONAL_PRIMARY_ENDPOINTS_BY_DIMENSION": base_stats.CONDITIONAL_PRIMARY_ENDPOINTS_BY_DIMENSION,
        "CONDITIONAL_ENDPOINTS_BY_DIMENSION": base_stats.CONDITIONAL_ENDPOINTS_BY_DIMENSION,
        "PRIMARY_METRICS": base_stats.PRIMARY_METRICS,
        "CONDITIONAL_PRIMARY_METRICS": base_stats.CONDITIONAL_PRIMARY_METRICS,
        "CONDITIONAL_METRICS": base_stats.CONDITIONAL_METRICS,
        "PAIRWISE_METRICS": base_stats.PAIRWISE_METRICS,
        "CONDITIONAL_PAIRWISE_METRICS": base_stats.CONDITIONAL_PAIRWISE_METRICS,
        "LOWER_IS_BETTER": base_stats.LOWER_IS_BETTER,
    }
    base_stats.PRIMARY_ENDPOINTS_BY_DIMENSION = _v527_universal_primary_config()
    base_stats.CONDITIONAL_PRIMARY_ENDPOINTS_BY_DIMENSION = _v527_conditional_primary_config()
    base_stats.CONDITIONAL_ENDPOINTS_BY_DIMENSION = {}
    base_stats.PRIMARY_METRICS = universal
    base_stats.CONDITIONAL_PRIMARY_METRICS = conditional
    base_stats.CONDITIONAL_METRICS = ()
    base_stats.PAIRWISE_METRICS = tuple(
        dict.fromkeys(universal + conditional + base_stats.SUPPLEMENTARY_METRICS)
    )
    base_stats.CONDITIONAL_PAIRWISE_METRICS = ()
    base_stats.LOWER_IS_BETTER = _v527_lower_is_better_metrics()
    try:
        yield universal, conditional
    finally:
        for key, value in saved.items():
            setattr(base_stats, key, value)


def _required_source_columns():
    rows = []
    for publication_metric, code_field in V527_PRIMARY_METRIC_CODE_FIELD_MAP.items():
        for method in METHODS:
            rows.append({
                "publication_metric": publication_metric,
                "code_field": code_field,
                "method": method,
                "required_source_column": f"{method}_{code_field}",
                "publication_column": f"{method}_{publication_metric}",
            })
    return rows


def _validate_source_columns(rows):
    if not rows:
        raise ValueError("V5.27 statistics requires non-empty cube rows.")
    columns = set(rows[0].keys())
    missing = [
        item for item in _required_source_columns()
        if item["required_source_column"] not in columns
    ]
    return missing


def _write_v527_statistics_outputs(output_root, cube_root, schema_root):
    rows_path = Path(cube_root) / "unified_benchmark_cube_rows.csv"
    if not rows_path.exists():
        raise FileNotFoundError(f"Unified cube rows not found: {rows_path}")

    frozen_schema_path = Path(schema_root) / FROZEN_SCHEMA_FILE
    freeze_manifest_path = Path(schema_root) / FREEZE_MANIFEST_FILE
    if not frozen_schema_path.exists() or not freeze_manifest_path.exists():
        raise FileNotFoundError(
            "V5.27 statistics requires frozen schema artifacts. Run "
            "`paper_pipeline.py v527-freeze-primary-schema` first."
        )
    frozen_schema = _read_json(frozen_schema_path)
    if frozen_schema.get("schema_status") != "frozen":
        raise RuntimeError("V5.27 statistics denied: schema_status is not frozen.")
    if frozen_schema.get("qualification_status") != "passed":
        raise RuntimeError("V5.27 statistics denied: qualification_status is not passed.")

    raw_rows = base_stats._read_csv_rows(rows_path)
    missing_sources = _validate_source_columns(raw_rows)
    if missing_sources:
        write_csv(missing_sources, output_root / "missing_v527_source_columns.csv")
        raise RuntimeError(
            "V5.27 statistics denied: required source metric columns are missing. "
            f"See {output_root / 'missing_v527_source_columns.csv'}"
        )

    rows = _copy_publication_metric_fields(raw_rows)
    rng = np.random.default_rng(20260721)
    with _patched_base_stats_globals() as (universal_metrics, conditional_metrics):
        metric_summary = base_stats._metric_summary_rows(rows)
        paired = base_stats._paired_effect_rows(rows, rng)
        metric_ranks, subdim_ranks, dim_ranks, overall_ranks = base_stats._primary_rank_rows(rows)
        dim_rank_summary = base_stats._aggregate_rank_rows(
            dim_ranks,
            "within_block_dimension_rank",
            ["signal_regime", "dimension", "method"],
        )
        overall_rank_summary = base_stats._aggregate_rank_rows(
            overall_ranks,
            "within_block_overall_rank",
            ["signal_regime", "method"],
        )
        friedman = base_stats._friedman_rows(dim_ranks, overall_ranks)
        degradation = base_stats._sigma_degradation_rows(rows)
        degradation_summary = base_stats._aggregate_rank_rows(
            [
                {
                    **row,
                    "rank_value": row["normalized_robustness_auc"]
                    if base_stats._higher_is_better(row["metric"])
                    else -row["normalized_robustness_auc"],
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
    write_csv(_required_source_columns(), output_root / "v527_publication_field_mapping.csv")

    dashboard = {
        "stage": "v527_unified_statistics_regeneration",
        "schema_version": V527_PRIMARY_SCHEMA_VERSION,
        "schema_status": "frozen",
        "qualification_status": "passed",
        "benchmark_results_under_v5_27": "statistics_regenerated",
        "source_cube": str(cube_root),
        "source_frozen_schema": str(frozen_schema_path),
        "benchmark_results_recomputed": False,
        "algorithm_cube_rerun_required": False,
        "n_cube_rows": int(len(rows)),
        "n_method_evaluations": int(len(rows) * len(METHODS)),
        "methods": list(METHODS),
        "primary_endpoint_count": int(len(universal_metrics) + len(conditional_metrics)),
        "universal_primary_metrics": list(universal_metrics),
        "contamination_domain_primary_metrics": list(conditional_metrics),
        "primary_endpoints_by_dimension": V527_PRIMARY_ENDPOINTS_BY_DIMENSION,
        "primary_metric_code_field_map": V527_PRIMARY_METRIC_CODE_FIELD_MAP,
        "noise_separation_constructs": V527_NOISE_SEPARATION_CONSTRUCTS,
        "contamination_resistance_constructs": V527_CONTAMINATION_RESISTANCE_CONSTRUCTS,
        "diagnostic_demotions": V527_DIAGNOSTIC_DEMOTIONS,
        "composite_sensitivity_endpoint_policy": {
            "role": CASE_SCORE_ROLE,
            "endpoints": list(COMPOSITE_SENSITIVITY_ENDPOINTS),
            "not_used_for_primary_ranking": True,
        },
        "outputs": {
            "method_metric_summary": "method_metric_summary.csv",
            "paired_effect_sizes": "paired_effect_sizes.csv",
            "primary_metric_ranks": "primary_metric_ranks.csv",
            "dimension_average_ranks": "dimension_average_ranks.csv",
            "overall_summary_average_ranks": "overall_summary_average_ranks.csv",
            "friedman_cd_rank_summary": "friedman_cd_rank_summary.csv",
            "sigma_degradation_trajectories": "sigma_degradation_trajectories.csv",
            "sigma_degradation_summary": "sigma_degradation_summary.csv",
            "v527_publication_field_mapping": "v527_publication_field_mapping.csv",
        },
        "statistical_protocol": {
            "algorithm_execution": (
                "No methods were rerun. V5.27 regenerates statistics from the "
                "existing protocol-controlled full benchmark cube."
            ),
            "publication_field_mapping": (
                "Publication-layer endpoint names are copied from frozen code "
                "fields before statistics are generated; e.g. matched_component "
                "metrics map to imf_recovery fields."
            ),
            "ranking": (
                "Primary metrics are ranked within each cell-seed block, then "
                "aggregated by construct subdimension, dimension, and descriptive "
                "overall rank under the frozen V5.27 schema."
            ),
            "conditional_contamination_domain": (
                "Contamination Resistance primary endpoints are evaluated only "
                "on contamination-regime rows."
            ),
        },
    }
    write_json(dashboard, output_root / "v527_unified_statistics_dashboard.json")
    return dashboard


def run_v527_unified_statistics(output_root, cube_root, schema_root):
    output_root = ensure_dir(output_root)
    return _write_v527_statistics_outputs(
        output_root=output_root,
        cube_root=Path(cube_root),
        schema_root=Path(schema_root),
    )
