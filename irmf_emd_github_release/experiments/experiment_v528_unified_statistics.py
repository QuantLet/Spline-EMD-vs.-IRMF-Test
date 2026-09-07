#!/usr/bin/python
# coding: UTF-8

"""Regenerate statistics under the V5.28 normalized-spillover schema."""

from contextlib import contextmanager
from pathlib import Path
import json

import numpy as np

from project_config import (
    CASE_SCORE_ROLE,
    COMPOSITE_SENSITIVITY_ENDPOINTS,
    EVALUATION_METHODS,
    LOWER_IS_BETTER_METRICS,
    V527_NOISE_SEPARATION_CONSTRUCTS,
    V528_CONTAMINATION_RESISTANCE_CONSTRUCTS,
    V528_DIAGNOSTIC_DEMOTIONS,
    V528_PRIMARY_ENDPOINTS_BY_DIMENSION,
    V528_PRIMARY_METRIC_CODE_FIELD_MAP,
    V528_PRIMARY_SCHEMA_VERSION,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json
from experiments import experiment_unified_cube_statistics as base_stats
from experiments.experiment_utils import paper_json_safe


METHODS = tuple(EVALUATION_METHODS)
CONTAMINATION_NOISES = ("impulsive", "burst", "huber_contamination")


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(paper_json_safe(row), separators=(",", ":")))
            f.write("\n")


def _all_v528_metrics():
    out = []
    for metrics in V528_PRIMARY_ENDPOINTS_BY_DIMENSION.values():
        out.extend(metrics)
    return tuple(dict.fromkeys(out))


def _subdimension_config(metrics, construct_map=None):
    construct_map = construct_map or {}
    return {"subdimensions": {construct_map.get(m, m): (m,) for m in metrics}}


def _universal_primary_config():
    return {
        dimension: _subdimension_config(metrics, V527_NOISE_SEPARATION_CONSTRUCTS)
        for dimension, metrics in V528_PRIMARY_ENDPOINTS_BY_DIMENSION.items()
        if dimension != "contamination_resistance"
    }


def _conditional_primary_config():
    metrics = V528_PRIMARY_ENDPOINTS_BY_DIMENSION["contamination_resistance"]
    return {
        "contamination_resistance": {
            **_subdimension_config(metrics, V528_CONTAMINATION_RESISTANCE_CONSTRUCTS),
            "applicable_noises": CONTAMINATION_NOISES,
        }
    }


def _derive_normalized_spillover_loss(score):
    try:
        score = float(score)
        if not np.isfinite(score) or score <= 0.0:
            return ""
        return float(-np.log(score))
    except Exception:
        return ""


def _copy_publication_metric_fields(rows):
    copied = []
    for row in rows:
        new_row = dict(row)
        for method in METHODS:
            score_col = f"{method}_contamination_spillover_score"
            dst = f"{method}_normalized_contamination_spillover_loss"
            if dst not in new_row and score_col in new_row:
                new_row[dst] = _derive_normalized_spillover_loss(new_row.get(score_col))
            for publication_metric, code_field in V528_PRIMARY_METRIC_CODE_FIELD_MAP.items():
                if publication_metric == "normalized_contamination_spillover_loss":
                    continue
                src = f"{method}_{code_field}"
                dst = f"{method}_{publication_metric}"
                if src in new_row and dst not in new_row:
                    new_row[dst] = new_row[src]
        copied.append(new_row)
    return copied


def _cache_group_ids(rows):
    for row in rows:
        row["__cell_id"] = "|".join(str(row.get(k)) for k in (
            "signal_regime", "signal", "noise", "sigma", "seed"
        ))
        row["__trajectory_id"] = "|".join(str(row.get(k)) for k in (
            "signal_regime", "signal", "noise"
        ))


def _primary_metrics_by_domain():
    conditional = tuple(V528_PRIMARY_ENDPOINTS_BY_DIMENSION["contamination_resistance"])
    universal = tuple(m for m in _all_v528_metrics() if m not in conditional)
    return universal, conditional


def _lower_is_better_metrics():
    lower = set(LOWER_IS_BETTER_METRICS)
    for publication_metric, code_field in V528_PRIMARY_METRIC_CODE_FIELD_MAP.items():
        if code_field in lower:
            lower.add(publication_metric)
    lower.add("normalized_contamination_spillover_loss")
    return lower


@contextmanager
def _patched_base_stats_globals():
    universal, conditional = _primary_metrics_by_domain()
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
    base_stats.PRIMARY_ENDPOINTS_BY_DIMENSION = _universal_primary_config()
    base_stats.CONDITIONAL_PRIMARY_ENDPOINTS_BY_DIMENSION = _conditional_primary_config()
    base_stats.CONDITIONAL_ENDPOINTS_BY_DIMENSION = {}
    base_stats.PRIMARY_METRICS = universal
    base_stats.CONDITIONAL_PRIMARY_METRICS = conditional
    base_stats.CONDITIONAL_METRICS = ()
    base_stats.PAIRWISE_METRICS = tuple(
        dict.fromkeys(universal + conditional + base_stats.SUPPLEMENTARY_METRICS)
    )
    base_stats.CONDITIONAL_PAIRWISE_METRICS = ()
    base_stats.LOWER_IS_BETTER = _lower_is_better_metrics()
    try:
        yield universal, conditional
    finally:
        for key, value in saved.items():
            setattr(base_stats, key, value)


def _required_source_columns():
    rows = []
    for publication_metric, code_field in V528_PRIMARY_METRIC_CODE_FIELD_MAP.items():
        for method in METHODS:
            rows.append({
                "publication_metric": publication_metric,
                "code_field": code_field,
                "method": method,
                "required_source_column": f"{method}_{code_field}",
                "publication_column": f"{method}_{publication_metric}",
                "derived_formula": (
                    "-log(contamination_spillover_score)"
                    if publication_metric == "normalized_contamination_spillover_loss"
                    else ""
                ),
            })
    return rows


def _validate_source_columns(rows):
    if not rows:
        raise ValueError("V5.28 statistics requires non-empty cube rows.")
    columns = set(rows[0].keys())
    return [
        item for item in _required_source_columns()
        if item["required_source_column"] not in columns
    ]


def run_v528_unified_statistics(output_root, cube_root, schema_root=None):
    output_root = ensure_dir(output_root)
    rows_path = Path(cube_root) / "unified_benchmark_cube_rows.csv"
    if not rows_path.exists():
        raise FileNotFoundError(f"Unified cube rows not found: {rows_path}")
    if schema_root is not None:
        frozen_schema = Path(schema_root) / "v528_primary_schema_frozen.json"
        freeze_manifest = Path(schema_root) / "v528_primary_schema_freeze_manifest.json"
        if not frozen_schema.exists() or not freeze_manifest.exists():
            raise FileNotFoundError(
                "V5.28 statistics requires frozen schema artifacts. Run "
                "`paper_pipeline.py v528-freeze-primary-schema` first."
            )
        frozen_doc = _read_json(frozen_schema)
        if frozen_doc.get("schema_status") != "frozen":
            raise RuntimeError("V5.28 statistics denied: schema_status is not frozen.")
        if frozen_doc.get("qualification_status") != "passed":
            raise RuntimeError("V5.28 statistics denied: qualification_status is not passed.")

    raw_rows = base_stats._read_csv_rows(rows_path)
    missing_sources = _validate_source_columns(raw_rows)
    if missing_sources:
        write_csv(missing_sources, output_root / "missing_v528_source_columns.csv")
        raise RuntimeError(
            "V5.28 statistics denied: required source metric columns are missing. "
            f"See {output_root / 'missing_v528_source_columns.csv'}"
        )

    rows = _copy_publication_metric_fields(raw_rows)
    _cache_group_ids(rows)
    rng = np.random.default_rng(20260721)
    with _patched_base_stats_globals() as (universal_metrics, conditional_metrics):
        metric_summary = base_stats._metric_summary_rows(rows)
        paired = base_stats._paired_effect_rows(rows, rng)
        metric_ranks, subdim_ranks, dim_ranks, overall_ranks = base_stats._primary_rank_rows(rows)
        dim_rank_summary = base_stats._aggregate_rank_rows(
            dim_ranks, "within_block_dimension_rank",
            ["signal_regime", "dimension", "method"],
        )
        overall_rank_summary = base_stats._aggregate_rank_rows(
            overall_ranks, "within_block_overall_rank",
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
    _write_jsonl(degradation, output_root / "sigma_degradation_trajectories.jsonl")
    write_csv(degradation_summary, output_root / "sigma_degradation_summary.csv")
    write_json(degradation_summary, output_root / "sigma_degradation_summary.json")
    write_csv(_required_source_columns(), output_root / "v528_publication_field_mapping.csv")

    dashboard = {
        "stage": "v528_unified_statistics_regeneration",
        "schema_version": V528_PRIMARY_SCHEMA_VERSION,
        "schema_status": "frozen",
        "qualification_status": "passed",
        "benchmark_results_under_v5_28": "statistics_regenerated",
        "source_cube": str(cube_root),
        "benchmark_results_recomputed": False,
        "algorithm_cube_rerun_required": False,
        "n_cube_rows": int(len(rows)),
        "n_method_evaluations": int(len(rows) * len(METHODS)),
        "methods": list(METHODS),
        "primary_endpoint_count": int(len(universal_metrics) + len(conditional_metrics)),
        "large_output_policy": {
            "sigma_degradation_trajectories": (
                "CSV plus JSONL; pretty JSON array intentionally omitted to avoid "
                "slow serialization of large trajectory tables."
            )
        },
        "universal_primary_metrics": list(universal_metrics),
        "contamination_domain_primary_metrics": list(conditional_metrics),
        "primary_endpoints_by_dimension": V528_PRIMARY_ENDPOINTS_BY_DIMENSION,
        "primary_metric_code_field_map": V528_PRIMARY_METRIC_CODE_FIELD_MAP,
        "contamination_resistance_constructs": V528_CONTAMINATION_RESISTANCE_CONSTRUCTS,
        "diagnostic_demotions": V528_DIAGNOSTIC_DEMOTIONS,
        "composite_sensitivity_endpoint_policy": {
            "role": CASE_SCORE_ROLE,
            "endpoints": list(COMPOSITE_SENSITIVITY_ENDPOINTS),
            "not_used_for_primary_ranking": True,
        },
        "normalized_spillover_definition": {
            "metric": "normalized_contamination_spillover_loss",
            "formula": "-log(contamination_spillover_score)",
            "equivalent_formula": "raw_spillover_mse / mean_clean_signal_energy",
            "source_metric": "contamination_spillover_score",
            "algorithm_rerun_required": False,
        },
        "statistical_protocol": {
            "algorithm_execution": "No methods were rerun.",
            "schema_change": (
                "V5.28 replaces raw contamination_spillover_error primary "
                "with normalized_contamination_spillover_loss."
            ),
        },
    }
    write_json(dashboard, output_root / "v528_unified_statistics_dashboard.json")
    return dashboard
