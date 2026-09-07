#!/usr/bin/python
# coding: UTF-8

"""V5.28 normalized-spillover primary schema qualification/freeze."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from project_config import (
    V528_CONTAMINATION_RESISTANCE_CONSTRUCTS,
    V528_DIAGNOSTIC_DEMOTIONS,
    V528_PRIMARY_ENDPOINTS_BY_DIMENSION,
    V528_PRIMARY_METRIC_CODE_FIELD_MAP,
    V528_PRIMARY_SCHEMA_VERSION,
    V528_VERSION_BOUNDARY,
)
from experiments.paper_pipeline_utils import ensure_dir, write_json


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_v528_primary_schema_freeze(output_root, v527_scale_audit_root):
    output_root = ensure_dir(output_root)
    scale_dashboard_path = Path(v527_scale_audit_root) / "v527_spillover_scale_audit_dashboard.json"
    if not scale_dashboard_path.exists():
        raise FileNotFoundError(
            "V5.28 freeze requires v527_spillover_scale_audit_dashboard.json."
        )
    scale_dashboard = _read_json(scale_dashboard_path)
    requirements = {
        "v527_spillover_scale_audit_passed": scale_dashboard.get("audit_status") == "passed",
        "raw_vs_normalized_rank_equal": scale_dashboard.get("rank_equal_rate_raw_vs_normalized") == 1.0,
        "raw_vs_normalized_pairwise_win_equal": scale_dashboard.get("win_equal_rate_raw_vs_normalized") == 1.0,
    }
    if not all(requirements.values()):
        failed = [k for k, v in requirements.items() if not v]
        raise RuntimeError("V5.28 freeze denied; unmet requirements: " + ", ".join(failed))

    frozen = {
        "schema_version": V528_PRIMARY_SCHEMA_VERSION,
        "schema_status": "frozen",
        "qualification_status": "passed",
        "benchmark_results_under_v5_28": "pending_statistics_regeneration",
        "benchmark_results_recomputed": False,
        "algorithm_cube_rerun_required": False,
        "statistics_regeneration_required": True,
        "version_boundary": V528_VERSION_BOUNDARY,
        "primary_endpoints_by_dimension": V528_PRIMARY_ENDPOINTS_BY_DIMENSION,
        "primary_metric_code_field_map": V528_PRIMARY_METRIC_CODE_FIELD_MAP,
        "contamination_resistance_constructs": V528_CONTAMINATION_RESISTANCE_CONSTRUCTS,
        "diagnostic_demotions": V528_DIAGNOSTIC_DEMOTIONS,
        "schema_revision": {
            "from": "V5.27_noise_contamination_construct_revision",
            "change": (
                "Replace raw contamination_spillover_error primary endpoint "
                "with normalized_contamination_spillover_loss."
            ),
            "unchanged_primary_endpoint_count": 13,
            "algorithm_rerun_required": False,
        },
        "normalized_spillover_definition": {
            "metric": "normalized_contamination_spillover_loss",
            "formula": "-log(contamination_spillover_score)",
            "equivalent_formula": "raw_spillover_mse / mean_clean_signal_energy",
            "source_metric": "contamination_spillover_score",
            "lower_is_better": True,
        },
        "source_scale_audit": str(scale_dashboard_path),
        "source_scale_audit_sha256": _sha256(scale_dashboard_path),
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_requirements": requirements,
    }
    manifest = {
        "stage": "v528_primary_schema_freeze",
        "schema_version": V528_PRIMARY_SCHEMA_VERSION,
        "schema_status": "frozen",
        "qualification_status": "passed",
        "benchmark_results_under_v5_28": "pending_statistics_regeneration",
        "benchmark_results_recomputed": False,
        "algorithm_cube_rerun_required": False,
        "statistics_regeneration_required": True,
        "source_scale_audit": str(scale_dashboard_path),
        "source_scale_audit_sha256": _sha256(scale_dashboard_path),
        "freeze_requirements": requirements,
        "next_required_stage": "v528_statistics_regeneration",
    }
    write_json(frozen, output_root / "v528_primary_schema_frozen.json")
    write_json(manifest, output_root / "v528_primary_schema_freeze_manifest.json")
    return manifest


def run_v528_final_statistics_validation(output_root, schema_root, stats_root, scale_audit_root):
    output_root = ensure_dir(output_root)
    schema_path = Path(schema_root) / "v528_primary_schema_frozen.json"
    freeze_manifest_path = Path(schema_root) / "v528_primary_schema_freeze_manifest.json"
    stats_dashboard_path = Path(stats_root) / "v528_unified_statistics_dashboard.json"
    scale_dashboard_path = Path(scale_audit_root) / "v527_spillover_scale_audit_dashboard.json"
    required_paths = {
        "schema": schema_path,
        "freeze_manifest": freeze_manifest_path,
        "statistics_dashboard": stats_dashboard_path,
        "scale_audit": scale_dashboard_path,
    }
    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "V5.28 final statistics validation missing artifacts: "
            + ", ".join(missing)
        )
    schema = _read_json(schema_path)
    stats = _read_json(stats_dashboard_path)
    scale = _read_json(scale_dashboard_path)
    checks = {
        "schema_frozen": schema.get("schema_status") == "frozen",
        "schema_qualification_passed": schema.get("qualification_status") == "passed",
        "statistics_regenerated": stats.get("benchmark_results_under_v5_28") == "statistics_regenerated",
        "statistics_schema_frozen": stats.get("schema_status") == "frozen",
        "statistics_qualification_passed": stats.get("qualification_status") == "passed",
        "algorithm_cube_not_rerun": stats.get("algorithm_cube_rerun_required") is False,
        "benchmark_results_not_recomputed": stats.get("benchmark_results_recomputed") is False,
        "primary_endpoint_count_13": stats.get("primary_endpoint_count") == 13,
        "scale_audit_passed": scale.get("audit_status") == "passed",
    }
    status = "validated" if all(checks.values()) else "requires_review"
    manifest = {
        "stage": "v528_final_statistics_validation",
        "schema_version": V528_PRIMARY_SCHEMA_VERSION,
        "schema_status": "frozen",
        "qualification_status": "passed",
        "benchmark_results_under_v5_28": "statistics_regenerated",
        "statistics_regeneration_completed": status == "validated",
        "final_statistics_status": status,
        "benchmark_results_recomputed": False,
        "algorithm_cube_rerun_required": False,
        "validation_checks": checks,
        "source_artifacts": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for name, path in required_paths.items()
        },
        "final_statistics_root": str(stats_root),
        "recommended_publication_schema": (
            "V5.28" if status == "validated" else "not_authorized"
        ),
        "interpretation_boundary": (
            "V5.28 final statistics validate the statistical layer under the "
            "frozen normalized-spillover schema. The algorithm cube was not "
            "rerun; only schema-level field derivation and statistical "
            "aggregation were regenerated."
        ),
    }
    write_json(manifest, output_root / "v528_final_statistics_validation_manifest.json")
    return manifest
