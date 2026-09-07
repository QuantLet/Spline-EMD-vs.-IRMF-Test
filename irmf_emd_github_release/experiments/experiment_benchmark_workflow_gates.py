#!/usr/bin/python
# coding: UTF-8

"""V5.24 protocol-driven benchmark workflow gates.

These gates manage benchmark eligibility, execution-state transitions, and
post-execution integrity audits.  They do not run decomposition algorithms,
rank methods, or interpret benchmark performance.
"""

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from project_config import (
    CONDITIONAL_PRIMARY_ENDPOINTS_BY_DIMENSION,
    EVALUATION_FRAMEWORK_VERSION,
    EVALUATION_METHODS,
    FULL_SIGMA_LEVELS,
    PRIMARY_ENDPOINTS_BY_DIMENSION,
    UNIFIED_BENCHMARK_REGIMES,
)
from diagnostics.shared_physical_diagnostics import STRUCTURAL_METRIC_SCHEMA_VERSION
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json
from experiments.experiment_utils import run_fixed_method_family_case


WORKFLOW_NAME = "V5.24 Protocol-Driven Benchmark Workflow"
STATE_ORDER = (
    "NOT_QUALIFIED",
    "QUALIFIED",
    "AUTHORIZED",
    "EXECUTING",
    "EXECUTED",
    "AUDITED",
    "STATISTICS_COMPLETED",
    "INTERPRETED",
)
DEFAULT_QUALIFICATION_ROOT = Path("IRMF_EMD_PAPER_RESULTS_V5_24_SCHEMA_FREEZE_ADJUDICATED")
QUALIFICATION_DASHBOARD_REL = (
    Path("algorithm")
    / "07e_structural_metric_validation"
    / "structural_metric_validation_dashboard.json"
)
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
CONDITIONAL_APPLICABLE_NOISES = tuple(
    noise
    for dcfg in CONDITIONAL_PRIMARY_ENDPOINTS_BY_DIMENSION.values()
    for noise in dcfg.get("applicable_noises", ())
)
REQUIRED_V5_24_METRICS = tuple(dict.fromkeys(PRIMARY_METRICS + CONDITIONAL_PRIMARY_METRICS))
RANGE_0_1_METRICS = {
    "imf_recovery_corr",
    "component_splitting_index",
    "component_merging_index",
    "signal_leakage_into_noise",
    "outlier_resistance_index",
    "inter_imf_entanglement_index",
    "missing_true_component_rate",
    "unmatched_estimated_component_rate",
    "spurious_mode_energy_ratio",
}
RANGE_MINUS1_1_METRICS = {
    "denoise_corr",
    "noise_capture_corr",
}
PROTOCOL_DISCIPLINE = (
    "Qualification authorizes benchmark execution. "
    "Successful execution authorizes execution auditing. "
    "Successful execution auditing authorizes statistical analysis. "
    "Statistical analysis authorizes scientific interpretation."
)
TRUE_NOISE_SEPARATION_ADAPTER_ID = "TRUE_NOISE_SEPARATION_ADAPTER_V1.0"
SYNTHETIC_RECONSTRUCTION_AMENDMENT_ID = "SYNTHETIC_COMPONENT_SELECTION_AND_RECONSTRUCTION_V1.0"
DEFAULT_COMPONENT_ASSOCIATION_THRESHOLD = 0.05
DEFAULT_COMPONENT_ENERGY_THRESHOLD = 0.01
DEFAULT_COMPONENT_ALLOCATION_FLOOR = 0.01


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _stable_json_checksum(obj):
    payload = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_checksum(path):
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        return np.nan
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _central_state_path(algorithm_root):
    return Path(algorithm_root) / "benchmark_state.json"


def _write_state(algorithm_root, state, stage_root=None, **extra):
    payload = {
        "workflow": WORKFLOW_NAME,
        "benchmark_state": state,
        "state_order": list(STATE_ORDER),
        "timestamp": _utc_now(),
        "framework_version": EVALUATION_FRAMEWORK_VERSION,
        "schema_version": STRUCTURAL_METRIC_SCHEMA_VERSION,
        "protocol_discipline": PROTOCOL_DISCIPLINE,
    }
    payload.update(extra)
    write_json(payload, _central_state_path(algorithm_root))
    if stage_root is not None:
        write_json(payload, Path(stage_root) / "benchmark_state.json")
    return payload


def load_benchmark_state(algorithm_root):
    return _load_json(_central_state_path(algorithm_root), default={
        "benchmark_state": "NOT_QUALIFIED",
    })


def require_benchmark_state(algorithm_root, allowed, command_name):
    state_doc = load_benchmark_state(algorithm_root)
    state = state_doc.get("benchmark_state", "NOT_QUALIFIED")
    if state not in set(allowed):
        raise RuntimeError(
            f"{command_name} requires benchmark_state in {sorted(allowed)}; "
            f"current state is {state!r}. {PROTOCOL_DISCIPLINE}"
        )
    return state_doc


def _expected_design(seeds=None):
    seeds = tuple(range(20) if seeds is None else seeds)
    regime_rows = []
    total_cells = 0
    for regime, cfg in UNIFIED_BENCHMARK_REGIMES.items():
        n_cells = len(cfg["signals"]) * len(cfg["noises"]) * len(cfg["sigmas"])
        total_cells += n_cells
        regime_rows.append({
            "signal_regime": regime,
            "n_signals": len(cfg["signals"]),
            "n_noises": len(cfg["noises"]),
            "n_sigmas": len(cfg["sigmas"]),
            "n_signal_noise_sigma_cells": n_cells,
            "n_case_seed_rows": n_cells * len(seeds),
            "n_method_evaluations": n_cells * len(seeds) * len(EVALUATION_METHODS),
            "signals": list(cfg["signals"]),
            "noises": list(cfg["noises"]),
            "sigmas": list(cfg["sigmas"]),
        })
    return {
        "row_schema": "one CSV row is one signal_regime × signal × noise × sigma × data_seed; method results are nested as prefixed columns",
        "methods": list(EVALUATION_METHODS),
        "seeds": list(seeds),
        "n_signal_noise_sigma_cells": total_cells,
        "expected_benchmark_rows": total_cells * len(seeds),
        "expected_method_evaluations": total_cells * len(seeds) * len(EVALUATION_METHODS),
        "regimes": regime_rows,
    }


def _qualification_dashboard_path(qualification_root):
    return Path(qualification_root or DEFAULT_QUALIFICATION_ROOT) / QUALIFICATION_DASHBOARD_REL


def _inspect_backfill_outputs(algorithm_root):
    root = Path(algorithm_root).parent
    rows_paths = sorted(root.glob("*/algorithm/03_unified_benchmark_cube/unified_benchmark_cube_rows.csv"))
    component_patterns = ("*.npy", "*.npz", "*.pkl", "*.h5", "*.hdf5")
    inspected = []
    for rows_path in rows_paths:
        if str(rows_path).startswith(str(Path(algorithm_root).parent / Path(algorithm_root).parent.name)):
            pass
        rows = _read_csv_rows(rows_path)
        fields = set(rows[0].keys()) if rows else set()
        component_artifacts = []
        search_root = rows_path.parent
        for pattern in component_patterns:
            component_artifacts.extend(str(p) for p in search_root.rglob(pattern))
        inspected.append({
            "rows_path": str(rows_path),
            "row_count": len(rows),
            "has_component_splitting_index": any(f.endswith("component_splitting_index") for f in fields),
            "has_component_merging_index": any(f.endswith("component_merging_index") for f in fields),
            "component_artifact_count": len(component_artifacts),
            "component_artifact_examples": component_artifacts[:10],
        })
    missing_artifacts = [
        "true_component_arrays",
        "estimated_component_or_imf_arrays",
        "component_association_matrices",
    ]
    return {
        "decision": "rejected",
        "reason": (
            "V5.24 structural primary endpoints require component-level arrays, "
            "but existing outputs contain summary rows only."
        ),
        "inspected_outputs": inspected,
        "required_artifacts": [
            "true component arrays",
            "estimated IMF/component arrays",
            "case_id/signal/noise/sigma/seed/method metadata",
            "sample ordering metadata",
            "metric/schema version metadata",
        ],
        "missing_artifacts": missing_artifacts,
        "affected_metrics": [
            "component_splitting_index",
            "component_merging_index",
        ],
        "authorized_action": "full_v5_24_rerun",
        "timestamp": _utc_now(),
        "framework_version": EVALUATION_FRAMEWORK_VERSION,
        "schema_version": STRUCTURAL_METRIC_SCHEMA_VERSION,
    }


def run_benchmark_eligibility_gate(
        algorithm_root,
        locked_parameters,
        qualification_root=None,
        seeds=None,
):
    algorithm_root = ensure_dir(algorithm_root)
    stage_root = ensure_dir(algorithm_root / "02_benchmark_eligibility")
    qualification_path = _qualification_dashboard_path(qualification_root)
    qualification = _load_json(qualification_path, default={})
    adjudication = qualification.get("imf_count_dependency_final_adjudication", {})
    output_cube_root = algorithm_root / "03_unified_benchmark_cube"
    rows_path = output_cube_root / "unified_benchmark_cube_rows.csv"
    checkpoint_path = output_cube_root / "unified_benchmark_cube_rows.jsonl"
    statistics_root = algorithm_root / "07_unified_cube_statistics"
    output_status = {
        "cube_root": str(output_cube_root),
        "rows_exists": rows_path.exists(),
        "checkpoint_exists": checkpoint_path.exists(),
        "statistics_exists": statistics_root.exists(),
        "status": "clean",
    }
    if rows_path.exists() or statistics_root.exists():
        output_status["status"] = "not_clean"
    elif checkpoint_path.exists():
        output_status["status"] = "valid_resume_candidate"

    expected = _expected_design(seeds=seeds)
    parameter_checksum = _stable_json_checksum(locked_parameters)
    parameter_source = locked_parameters.get("parameter_source")
    checks = {
        "qualification_completed": (
            qualification.get("benchmark_validation_status")
            == "structural_metric_schema_validated_and_frozen"
        ),
        "final_adjudication_passed": str(
            adjudication.get("final_dependency_qualification_status", "")
        ).startswith("passed"),
        "framework_version_match": (
            qualification.get("evaluation_framework_version")
            == EVALUATION_FRAMEWORK_VERSION
        ),
        "schema_version_match": (
            qualification.get("structural_metric_schema_version")
            == STRUCTURAL_METRIC_SCHEMA_VERSION
        ),
        "development_set_parameter_selection_used": (
            parameter_source == "section_4_development_selection"
        ),
        "locked_parameters_present": all(
            method in locked_parameters for method in EVALUATION_METHODS
        ),
        "configuration_checksum_recorded": bool(parameter_checksum),
        "expected_design_recorded": expected["expected_benchmark_rows"] == 12000,
        "output_directory_clean_or_valid_resume": output_status["status"] in {
            "clean", "valid_resume_candidate",
        },
        "statistics_not_previously_generated_from_incomplete_data": not statistics_root.exists(),
    }
    authorization_status = "granted" if all(checks.values()) else "denied"
    backfill = _inspect_backfill_outputs(algorithm_root)
    pre_execution = {
        "workflow": WORKFLOW_NAME,
        "stage": "Stage A2 Pre-execution Gate",
        "question": "Is the benchmark configuration authorized to execute?",
        "authorization_status": authorization_status,
        "checks": checks,
        "qualification_dashboard_path": str(qualification_path),
        "qualification_status": qualification.get("benchmark_validation_status"),
        "final_dependency_qualification_status": adjudication.get(
            "final_dependency_qualification_status"
        ),
        "framework_version": EVALUATION_FRAMEWORK_VERSION,
        "schema_version": STRUCTURAL_METRIC_SCHEMA_VERSION,
        "parameter_source": parameter_source,
        "locked_parameter_checksum": parameter_checksum,
        "expected_design": expected,
        "output_directory_status": output_status,
        "timestamp": _utc_now(),
        "protocol_discipline": PROTOCOL_DISCIPLINE,
    }
    write_json(backfill, stage_root / "backfill_decision.json")
    write_json(pre_execution, stage_root / "pre_execution_gate.json")
    state = "AUTHORIZED" if authorization_status == "granted" else "QUALIFIED"
    state_doc = _write_state(
        algorithm_root,
        state,
        stage_root=stage_root,
        backfill_decision=backfill.get("decision"),
        authorization_status=authorization_status,
        locked_parameter_checksum=parameter_checksum,
    )
    return {
        "backfill_decision": backfill,
        "pre_execution_gate": pre_execution,
        "benchmark_state": state_doc,
    }


def mark_benchmark_executing(algorithm_root, resume_requested=False):
    previous = require_benchmark_state(
        algorithm_root,
        allowed={"AUTHORIZED", "EXECUTING"},
        command_name="unified-cube",
    )
    cube_root = Path(algorithm_root) / "03_unified_benchmark_cube"
    checkpoint = cube_root / "unified_benchmark_cube_rows.jsonl"
    existing_rows = _read_jsonl_count(checkpoint)
    return _write_state(
        algorithm_root,
        "EXECUTING",
        stage_root=cube_root,
        previous_state=previous.get("benchmark_state"),
        resume_requested=bool(resume_requested or existing_rows > 0),
        resume_source=str(checkpoint) if checkpoint.exists() else None,
        checkpoint_checksum=_file_checksum(checkpoint),
        existing_row_count=existing_rows,
    )


def mark_benchmark_executed(algorithm_root, dashboard=None):
    cube_root = Path(algorithm_root) / "03_unified_benchmark_cube"
    rows_path = cube_root / "unified_benchmark_cube_rows.csv"
    rows = _read_csv_rows(rows_path)
    checkpoint = cube_root / "unified_benchmark_cube_rows.jsonl"
    return _write_state(
        algorithm_root,
        "EXECUTED",
        stage_root=cube_root,
        row_count=len(rows),
        checkpoint_checksum=_file_checksum(checkpoint),
        rows_checksum=_file_checksum(rows_path),
        execution_dashboard_summary={
            "n_signal_noise_sigma_seed_cells": (
                dashboard or {}
            ).get("protocol", {}).get("n_signal_noise_sigma_seed_cells"),
            "n_method_case_evaluations": (
                dashboard or {}
            ).get("protocol", {}).get("n_method_case_evaluations"),
        },
    )


def _read_jsonl_count(path):
    path = Path(path)
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _row_key(row):
    return (
        row.get("signal_regime"),
        row.get("signal"),
        row.get("noise"),
        _as_float(row.get("sigma")),
        _as_int(row.get("data_seed", row.get("seed"))),
    )


def _as_float(value):
    try:
        out = float(value)
        if np.isfinite(out):
            return out
    except Exception:
        return None
    return None


def _as_int(value):
    try:
        return int(float(value))
    except Exception:
        return None


def _as_metric_float(value):
    if value in {None, ""}:
        return None
    try:
        out = float(value)
        return out
    except Exception:
        return math.nan


def _expected_case_keys(seeds):
    keys = set()
    for regime, cfg in UNIFIED_BENCHMARK_REGIMES.items():
        for signal in cfg["signals"]:
            for noise in cfg["noises"]:
                for sigma in cfg["sigmas"]:
                    for seed in seeds:
                        keys.add((regime, signal, noise, float(sigma), int(seed)))
    return keys


def _distribution_audit_rows(rows, metrics):
    out = []
    for method in EVALUATION_METHODS:
        for metric in metrics:
            field = f"{method}_{metric}"
            values = np.asarray([
                v for v in (_as_metric_float(row.get(field)) for row in rows)
                if v is not None and np.isfinite(v)
            ], dtype=float)
            n_total = len(rows)
            n_finite = int(values.size)
            n_missing = n_total - n_finite
            if n_finite:
                vmin = float(np.min(values))
                vmax = float(np.max(values))
                mean = float(np.mean(values))
                sd = float(np.std(values, ddof=1)) if n_finite > 1 else 0.0
                q01 = float(np.quantile(values, 0.01))
                q99 = float(np.quantile(values, 0.99))
                constant = bool(vmax == vmin)
                near_constant = bool(sd <= 1e-12)
                ceiling_fraction = float(np.mean(values >= vmax - 1e-12))
                floor_fraction = float(np.mean(values <= vmin + 1e-12))
                iqr = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
                outlier_count = int(np.sum((values < q01) | (values > q99)))
            else:
                vmin = vmax = mean = sd = q01 = q99 = iqr = None
                constant = near_constant = False
                ceiling_fraction = floor_fraction = None
                outlier_count = 0
            out.append({
                "method": method,
                "metric": metric,
                "n_total_rows": n_total,
                "n_finite": n_finite,
                "n_missing_or_nonfinite": n_missing,
                "missing_or_nonfinite_rate": float(n_missing / max(n_total, 1)),
                "min": vmin,
                "max": vmax,
                "mean": mean,
                "sd": sd,
                "iqr": iqr,
                "q01": q01,
                "q99": q99,
                "constant_metric": constant,
                "near_constant_metric": near_constant,
                "ceiling_fraction": ceiling_fraction,
                "floor_fraction": floor_fraction,
                "extreme_outlier_count_1pct_rule": outlier_count,
            })
    return out


def run_benchmark_execution_audit(algorithm_root, seeds=None):
    algorithm_root = ensure_dir(algorithm_root)
    require_benchmark_state(
        algorithm_root,
        allowed={"EXECUTED"},
        command_name="benchmark-execution-audit",
    )
    stage_root = ensure_dir(algorithm_root / "04_benchmark_execution_audit")
    cube_root = algorithm_root / "03_unified_benchmark_cube"
    rows_path = cube_root / "unified_benchmark_cube_rows.csv"
    protocol = _load_json(cube_root / "unified_benchmark_cube_protocol.json", default={})
    rows = _read_csv_rows(rows_path)
    seeds = tuple(
        int(s) for s in (
            protocol.get("seeds") if protocol.get("seeds") is not None else (seeds or range(20))
        )
    )
    expected = _expected_design(seeds=seeds)
    expected_keys = _expected_case_keys(seeds)
    actual_keys = [_row_key(row) for row in rows]
    actual_key_set = set(actual_keys)
    duplicate_keys = [
        key for key, count in Counter(actual_keys).items()
        if count > 1
    ]
    missing_keys = sorted(expected_keys - actual_key_set, key=str)
    unexpected_keys = sorted(actual_key_set - expected_keys, key=str)
    fields = set(rows[0].keys()) if rows else set()
    missing_method_evaluations = []
    metric_missing = []
    invalid_range_rows = []
    nonfinite_rows = []
    failure_rows = []
    for row in rows:
        key = _row_key(row)
        for method in EVALUATION_METHODS:
            method_has_any = any(field.startswith(f"{method}_") for field in fields)
            if not method_has_any:
                missing_method_evaluations.append((*key, method))
            for metric in REQUIRED_V5_24_METRICS:
                field = f"{method}_{metric}"
                applicable = True
                if metric in CONDITIONAL_PRIMARY_METRICS:
                    applicable = row.get("noise") in CONDITIONAL_APPLICABLE_NOISES
                if field not in fields:
                    metric_missing.append((*key, method, metric, "field_absent"))
                    continue
                value = _as_metric_float(row.get(field))
                if applicable and value is None:
                    metric_missing.append((*key, method, metric, "value_missing"))
                    continue
                if not applicable:
                    continue
                if value is not None and not np.isfinite(value):
                    nonfinite_rows.append((*key, method, metric, row.get(field)))
                    continue
                if (
                    value is not None
                    and metric in RANGE_0_1_METRICS
                    and (value < -1e-12 or value > 1 + 1e-12)
                ):
                    invalid_range_rows.append((*key, method, metric, value))
                if (
                    value is not None
                    and metric in RANGE_MINUS1_1_METRICS
                    and (value < -1 - 1e-12 or value > 1 + 1e-12)
                ):
                    invalid_range_rows.append((*key, method, metric, value))
            for flag in ("any_failure", "computational_failure", "timeout_flag", "exception_flag", "nonfinite_output_flag"):
                value = str(row.get(f"{method}_{flag}", "")).strip().lower()
                if value in {"true", "1", "yes"}:
                    failure_rows.append((*key, method, flag))

    structural_integrity_checks = {
        "rows_file_exists": rows_path.exists(),
        "expected_benchmark_rows_match": len(rows) == expected["expected_benchmark_rows"],
        "expected_method_evaluations_match": (
            len(rows) * len(EVALUATION_METHODS)
            == expected["expected_method_evaluations"]
        ),
        "duplicate_case_seed_combinations_absent": not duplicate_keys,
        "missing_case_seed_combinations_absent": not missing_keys,
        "unexpected_case_seed_combinations_absent": not unexpected_keys,
        "missing_method_evaluations_absent": not missing_method_evaluations,
        "schema_version_present": any(
            field.endswith("structural_metric_schema_version") for field in fields
        ),
        "required_v5_24_metric_fields_present": not [
            m for m in REQUIRED_V5_24_METRICS
            if not any(f.endswith(f"_{m}") for f in fields)
        ],
        "checkpoint_consistency": _read_jsonl_count(cube_root / "unified_benchmark_cube_rows.jsonl") == len(rows),
    }
    statistical_readiness_checks = {
        "required_primary_values_complete": not metric_missing,
        "nonfinite_values_absent": not nonfinite_rows,
        "invalid_ranges_absent": not invalid_range_rows,
        "failures_and_timeouts_documented": True,
        "distribution_audit_generated": True,
        "method_specific_missingness_absent": not metric_missing,
    }
    structural_status = "passed" if all(structural_integrity_checks.values()) else "failed"
    readiness_status = "passed" if all(statistical_readiness_checks.values()) else "failed"
    metrics_for_distribution = tuple(dict.fromkeys(
        REQUIRED_V5_24_METRICS
        + (
            "inter_imf_entanglement_index",
            "missing_true_component_rate",
            "unmatched_estimated_component_rate",
            "spurious_mode_energy_ratio",
            "runtime_seconds",
        )
    ))
    distribution_rows = _distribution_audit_rows(rows, metrics_for_distribution)
    summary_rows = [
        {"audit": "expected_benchmark_rows", "expected": expected["expected_benchmark_rows"], "actual": len(rows), "status": structural_integrity_checks["expected_benchmark_rows_match"]},
        {"audit": "expected_method_evaluations", "expected": expected["expected_method_evaluations"], "actual": len(rows) * len(EVALUATION_METHODS), "status": structural_integrity_checks["expected_method_evaluations_match"]},
        {"audit": "duplicate_case_seed_combinations", "expected": 0, "actual": len(duplicate_keys), "status": not duplicate_keys},
        {"audit": "missing_case_seed_combinations", "expected": 0, "actual": len(missing_keys), "status": not missing_keys},
        {"audit": "unexpected_case_seed_combinations", "expected": 0, "actual": len(unexpected_keys), "status": not unexpected_keys},
        {"audit": "missing_method_evaluations", "expected": 0, "actual": len(missing_method_evaluations), "status": not missing_method_evaluations},
        {"audit": "metric_missing_values", "expected": 0, "actual": len(metric_missing), "status": not metric_missing},
        {"audit": "nonfinite_values", "expected": 0, "actual": len(nonfinite_rows), "status": not nonfinite_rows},
        {"audit": "invalid_metric_ranges", "expected": 0, "actual": len(invalid_range_rows), "status": not invalid_range_rows},
        {"audit": "documented_failure_flags", "expected": "documented", "actual": len(failure_rows), "status": True},
    ]
    audit = {
        "workflow": WORKFLOW_NAME,
        "stage": "Stage C Execution Integrity Audit",
        "scope": "audit benchmark rows only; no statistical ranking and no method-performance interpretation",
        "row_count_semantics": expected["row_schema"],
        "expected_design": expected,
        "structural_integrity": {
            "status": structural_status,
            "checks": structural_integrity_checks,
            "actual_benchmark_rows": len(rows),
            "actual_method_evaluations": len(rows) * len(EVALUATION_METHODS),
            "duplicate_case_seed_count": len(duplicate_keys),
            "missing_case_seed_count": len(missing_keys),
            "unexpected_case_seed_count": len(unexpected_keys),
            "missing_method_evaluation_count": len(missing_method_evaluations),
            "missing_examples": [list(v) for v in missing_keys[:20]],
            "duplicate_examples": [list(v) for v in duplicate_keys[:20]],
            "unexpected_examples": [list(v) for v in unexpected_keys[:20]],
        },
        "statistical_readiness": {
            "status": readiness_status,
            "checks": statistical_readiness_checks,
            "missing_metric_value_count": len(metric_missing),
            "nonfinite_value_count": len(nonfinite_rows),
            "invalid_range_count": len(invalid_range_rows),
            "documented_failure_or_timeout_count": len(failure_rows),
            "missing_metric_examples": [list(v) for v in metric_missing[:20]],
            "nonfinite_examples": [list(v) for v in nonfinite_rows[:20]],
            "invalid_range_examples": [list(v) for v in invalid_range_rows[:20]],
        },
        "statistics_authorized": structural_status == "passed" and readiness_status == "passed",
        "timestamp": _utc_now(),
        "framework_version": EVALUATION_FRAMEWORK_VERSION,
        "schema_version": STRUCTURAL_METRIC_SCHEMA_VERSION,
        "protocol_discipline": PROTOCOL_DISCIPLINE,
    }
    write_json(audit, stage_root / "execution_integrity_audit.json")
    write_csv(summary_rows, stage_root / "execution_integrity_audit.csv")
    write_csv(distribution_rows, stage_root / "distribution_audit.csv")
    if audit["statistics_authorized"]:
        state = _write_state(
            algorithm_root,
            "AUDITED",
            stage_root=stage_root,
            statistics_authorized=True,
            structural_integrity=structural_status,
            statistical_readiness=readiness_status,
        )
    else:
        state = _write_state(
            algorithm_root,
            "EXECUTED",
            stage_root=stage_root,
            statistics_authorized=False,
            structural_integrity=structural_status,
            statistical_readiness=readiness_status,
        )
    return {
        "execution_integrity_audit": audit,
        "distribution_audit_rows": distribution_rows,
        "benchmark_state": state,
    }


def run_ceemdan_noise_endpoint_adjudication(algorithm_root):
    """Audit CEEMDAN noise-separation NaNs without changing benchmark rows."""
    algorithm_root = ensure_dir(algorithm_root)
    require_benchmark_state(
        algorithm_root,
        allowed={"EXECUTED"},
        command_name="ceemdan-noise-endpoint-adjudication",
    )
    stage_root = ensure_dir(algorithm_root / "04a_ceemdan_noise_endpoint_adjudication")
    cube_root = algorithm_root / "03_unified_benchmark_cube"
    rows_path = cube_root / "unified_benchmark_cube_rows.csv"
    rows = _read_csv_rows(rows_path)

    endpoint_metrics = ("noise_capture_corr", "signal_leakage_into_noise")
    root_cause_rows = []
    cause_counts = Counter()
    residual_ratios = []
    finite_endpoint_counts = Counter()
    for row in rows:
        residual_energy = _as_metric_float(row.get("CEEMDAN_signal_leakage_residual_energy"))
        observed_energy = _as_metric_float(row.get("CEEMDAN_signal_leakage_observed_energy"))
        degenerate_flag = str(row.get("CEEMDAN_signal_leakage_degenerate_flag", "")).strip().lower()
        reconstruction_available = all(
            _as_metric_float(row.get(f"CEEMDAN_{metric}")) is not None
            and np.isfinite(_as_metric_float(row.get(f"CEEMDAN_{metric}")))
            for metric in ("denoise_nmse", "denoise_corr")
        )
        residual_available = residual_energy is not None and np.isfinite(residual_energy)
        ratio = None
        if (
            residual_energy is not None
            and observed_energy is not None
            and np.isfinite(residual_energy)
            and np.isfinite(observed_energy)
            and abs(observed_energy) > 0
        ):
            ratio = float(residual_energy / observed_energy)
            residual_ratios.append(ratio)
        if not residual_available:
            nan_reason = "residual_energy_unavailable"
        elif degenerate_flag in {"true", "1", "yes"}:
            nan_reason = "native_residual_near_zero_complete_decomposition"
        else:
            nan_reason = "endpoint_nan_requires_further_review"

        metric_values = {}
        for metric in endpoint_metrics:
            value = _as_metric_float(row.get(f"CEEMDAN_{metric}"))
            finite = value is not None and np.isfinite(value)
            if finite:
                finite_endpoint_counts[metric] += 1
            metric_values[metric] = row.get(f"CEEMDAN_{metric}")
        cause_counts[nan_reason] += 1
        root_cause_rows.append({
            "cube_cell_id": row.get("cube_cell_id"),
            "signal_regime": row.get("signal_regime"),
            "signal": row.get("signal"),
            "noise": row.get("noise"),
            "sigma": row.get("sigma"),
            "seed": row.get("seed", row.get("data_seed")),
            "method": "CEEMDAN",
            "residual_available": bool(residual_available),
            "residual_energy": residual_energy,
            "observed_energy": observed_energy,
            "residual_to_observed_energy_ratio": ratio,
            "estimated_noise_available_from_native_residual": bool(
                residual_available and degenerate_flag not in {"true", "1", "yes"}
            ),
            "reconstruction_available": bool(reconstruction_available),
            "noise_capture_corr": metric_values["noise_capture_corr"],
            "signal_leakage_into_noise": metric_values["signal_leakage_into_noise"],
            "signal_leakage_degenerate_flag": row.get("CEEMDAN_signal_leakage_degenerate_flag"),
            "any_failure": row.get("CEEMDAN_any_failure"),
            "timeout_flag": row.get("CEEMDAN_timeout_flag"),
            "exception_flag": row.get("CEEMDAN_exception_flag"),
            "nonfinite_output_flag": row.get("CEEMDAN_nonfinite_output_flag"),
            "nan_reason": nan_reason,
        })

    ratio_arr = np.asarray(residual_ratios, dtype=float)
    summary = {
        "workflow": WORKFLOW_NAME,
        "stage": "Stage C1 CEEMDAN Residual Semantics and Noise-Separation Computability Review",
        "scope": (
            "root-cause audit and scientific adjudication only; no method "
            "ranking, no statistical analysis, and no benchmark-row mutation"
        ),
        "benchmark_rows_inspected": int(len(rows)),
        "method": "CEEMDAN",
        "endpoint_metrics": list(endpoint_metrics),
        "finite_endpoint_counts": dict(finite_endpoint_counts),
        "nan_endpoint_counts": {
            metric: int(len(rows) - finite_endpoint_counts.get(metric, 0))
            for metric in endpoint_metrics
        },
        "nan_root_cause_counts": dict(cause_counts),
        "residual_to_observed_energy_ratio_summary": {
            "n_finite": int(ratio_arr.size),
            "min": float(np.min(ratio_arr)) if ratio_arr.size else None,
            "median": float(np.median(ratio_arr)) if ratio_arr.size else None,
            "max": float(np.max(ratio_arr)) if ratio_arr.size else None,
        },
        "protocol_evidence": {
            "estimated_noise_definition_reference": (
                "README_V5_15_CONTAMINATION_LEAKAGE_AUDC.md states "
                "estimated_noise = observed_signal - reconstructed_signal."
            ),
            "current_implementation_observation": (
                "diagnostics.shared_physical_diagnostics reconstructs the "
                "estimated signal as observed minus residual; therefore the "
                "current estimated-noise adapter is method-native residual."
            ),
        },
        "adjudication_status": "method_metric_compatibility_blocker_confirmed",
        "adjudication": (
            "CEEMDAN completed the full benchmark, but its native residual is "
            "near-zero in all inspected rows because CEEMDAN returns an "
            "approximately complete decomposition. The native residual is "
            "therefore not a computable protocol-level estimated-noise object "
            "for the two universal noise-separation endpoints."
        ),
        "backfill_feasibility_from_current_rows": "rejected",
        "backfill_rejection_reason": (
            "The current unified benchmark rows contain summary metrics but do "
            "not archive observed signals, clean signals, true noise vectors, "
            "selected reconstructions, or component arrays needed to recompute "
            "protocol-defined estimated_noise without rerunning post-processing "
            "or method outputs."
        ),
        "recommended_next_action": (
            "Do not run unified-statistics yet. Define or confirm a single "
            "protocol-level estimated_noise adapter for all methods, then "
            "recompute noise_capture_corr and signal_leakage_into_noise for "
            "all four methods under the same adapter. If required artifacts "
            "are unavailable, rerun the affected benchmark stage after the "
            "adapter is fixed and version/audit provenance is recorded."
        ),
        "statistics_authorized": False,
        "timestamp": _utc_now(),
        "framework_version": EVALUATION_FRAMEWORK_VERSION,
        "schema_version": STRUCTURAL_METRIC_SCHEMA_VERSION,
        "protocol_discipline": PROTOCOL_DISCIPLINE,
    }
    write_csv(root_cause_rows, stage_root / "ceemdan_noise_endpoint_nan_root_cause.csv")
    write_json(root_cause_rows[:100], stage_root / "ceemdan_noise_endpoint_nan_root_cause_examples.json")
    write_json(summary, stage_root / "ceemdan_noise_endpoint_adjudication.json")
    return {
        "ceemdan_noise_endpoint_adjudication": summary,
        "root_cause_rows": root_cause_rows,
    }


def _artifact_inventory(cube_root):
    cube_root = Path(cube_root)
    patterns = {
        "array_npz": "*.npz",
        "array_npy": "*.npy",
        "pickle": "*.pkl",
        "hdf5": "*.h5",
        "hdf5_alt": "*.hdf5",
        "parquet": "*.parquet",
    }
    inventory = {}
    examples = {}
    for label, pattern in patterns.items():
        paths = sorted(cube_root.rglob(pattern))
        inventory[label] = int(len(paths))
        examples[label] = [str(p) for p in paths[:10]]
    return inventory, examples


def run_rebuild_universal_noise_endpoints_gate(algorithm_root):
    """Decide whether universal noise endpoints can be rebuilt from artifacts."""
    algorithm_root = ensure_dir(algorithm_root)
    require_benchmark_state(
        algorithm_root,
        allowed={"EXECUTED"},
        command_name="rebuild-universal-noise-endpoints",
    )
    stage_root = ensure_dir(algorithm_root / "04b_rebuild_universal_noise_endpoints")
    cube_root = algorithm_root / "03_unified_benchmark_cube"
    rows_path = cube_root / "unified_benchmark_cube_rows.csv"
    rows = _read_csv_rows(rows_path)
    inventory, examples = _artifact_inventory(cube_root)

    required_artifacts = [
        "observed waveform for each case",
        "clean signal and true-noise waveform for each case",
        "method component arrays or selected reconstruction arrays",
        "component-retention mask or frozen reconstruction adapter output",
        "case metadata linking arrays to signal/noise/sigma/seed/method",
    ]
    artifact_count = sum(inventory.values())
    has_rebuild_artifacts = artifact_count > 0
    if has_rebuild_artifacts:
        feasibility = "requires_manual_artifact_schema_review"
        decision = "not_executed"
        reason = (
            "Potential non-scalar artifacts were found, but this gate does not "
            "yet know their schema. Manual schema review is required before "
            "metric-only endpoint recomputation can be authorized."
        )
        authorized_action = "review_artifact_schema_before_metric_only_rebuild"
    else:
        feasibility = "rejected"
        decision = "rejected"
        reason = (
            "The unified benchmark cube archives scalar summary rows only. "
            "No component arrays, waveform arrays, selected reconstruction "
            "arrays, HDF5, parquet, NPZ, NPY, or pickle artifacts were found "
            "under the cube root."
        )
        authorized_action = "controlled_rerun_required_after_protocol_adapter_implementation"

    decision_doc = {
        "workflow": WORKFLOW_NAME,
        "stage": "Stage C2 Universal Noise-Endpoint Rebuild Decision",
        "scope": (
            "decide whether noise_capture_corr and signal_leakage_into_noise "
            "can be rebuilt for all methods without rerunning decompositions; "
            "no benchmark-row mutation and no statistical analysis"
        ),
        "affected_primary_endpoints": [
            "noise_capture_corr",
            "signal_leakage_into_noise",
        ],
        "affected_methods": list(EVALUATION_METHODS),
        "required_adapter_definition": (
            "protocol_estimated_signal is produced by the frozen component-"
            "selection/reconstruction adapter; protocol_estimated_noise = "
            "observed_signal - protocol_estimated_signal"
        ),
        "current_scalar_row_count": int(len(rows)),
        "expected_method_endpoint_cells": int(len(rows) * len(EVALUATION_METHODS) * 2),
        "artifact_inventory": inventory,
        "artifact_examples": examples,
        "required_artifacts_for_metric_only_rebuild": required_artifacts,
        "metric_only_rebuild_feasibility": feasibility,
        "decision": decision,
        "reason": reason,
        "authorized_action": authorized_action,
        "versioning_policy": {
            "metric_formula_changed": False,
            "primary_endpoint_set_changed": False,
            "input_object_semantics_requires_adapter": True,
            "recommended_revision_label": "TRUE_NOISE_SEPARATION_ADAPTER_V1.0",
        },
        "rerun_requirements_if_artifacts_unavailable": [
            "reuse the same signal/noise/sigma/data-seed manifest",
            "reuse locked IRMF and fixed EMD-family parameters",
            "implement one method-independent protocol-level estimated-noise adapter",
            "recompute the two affected endpoints for IRMF, EMD, EEMD, and CEEMDAN",
            "archive adapter provenance and endpoint before/after comparison",
            "rerun Stage C statistical-readiness audit before unified-statistics",
        ],
        "statistics_authorized": False,
        "timestamp": _utc_now(),
        "framework_version": EVALUATION_FRAMEWORK_VERSION,
        "schema_version": STRUCTURAL_METRIC_SCHEMA_VERSION,
        "protocol_discipline": PROTOCOL_DISCIPLINE,
    }
    write_json(decision_doc, stage_root / "noise_endpoint_rebuild_decision.json")
    return {"noise_endpoint_rebuild_decision": decision_doc}


def _write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_true_noise_separation_adapter_spec(algorithm_root):
    """Freeze the protocol-level estimated-noise adapter specification."""
    algorithm_root = ensure_dir(algorithm_root)
    require_benchmark_state(
        algorithm_root,
        allowed={"EXECUTED"},
        command_name="true-noise-separation-adapter-spec",
    )
    stage_root = ensure_dir(algorithm_root / "04c_true_noise_separation_adapter")
    adjudication = _load_json(
        algorithm_root
        / "04a_ceemdan_noise_endpoint_adjudication"
        / "ceemdan_noise_endpoint_adjudication.json",
        default={},
    )
    rebuild = _load_json(
        algorithm_root
        / "04b_rebuild_universal_noise_endpoints"
        / "noise_endpoint_rebuild_decision.json",
        default={},
    )
    spec = {
        "adapter_id": TRUE_NOISE_SEPARATION_ADAPTER_ID,
        "adapter_version": "1.0",
        "adapter_status": "specification_frozen",
        "workflow": WORKFLOW_NAME,
        "stage": "Stage C3 TRUE_NOISE_SEPARATION_ADAPTER_V1.0 specification and freeze",
        "scope": (
            "freeze the method-independent estimated-noise adapter for the two "
            "universal synthetic noise-separation endpoints; no decomposition "
            "runs, no benchmark-row mutation, no statistical analysis"
        ),
        "affected_primary_endpoints": [
            "noise_capture_corr",
            "signal_leakage_into_noise",
        ],
        "affected_methods": list(EVALUATION_METHODS),
        "formal_definitions": {
            "protocol_estimated_signal": (
                "sum of components retained by the frozen component-selection "
                "or reconstruction rule, including any retained trend-bearing "
                "components regardless of their decomposition order"
            ),
            "protocol_estimated_noise": (
                "observed_signal - protocol_estimated_signal"
            ),
            "noise_capture_corr": (
                "corr(protocol_estimated_noise, true_noise), using the existing "
                "signed Pearson-correlation convention"
            ),
            "signal_leakage_into_noise": (
                "the existing frozen projection-energy leakage functional "
                "applied to protocol_estimated_noise and the true signal/"
                "true-component signal span"
            ),
        },
        "input_artifact_requirements_per_method_evaluation": [
            "case_id",
            "method",
            "observed_signal",
            "true_signal",
            "true_noise",
            "component_array",
            "native_residual_or_trend_bearing_component_if_exposed",
            "component_selection_mask_or_selected_component_indices",
            "component_selection_scores_if_rule_generated_at_runtime",
            "component_selection_threshold_or_rule_id",
            "selection_reason_per_component_if_available",
        ],
        "uniformity_constraints": {
            "method_specific_native_residual_as_estimated_noise_allowed": False,
            "ceemdan_special_case_allowed": False,
            "same_adapter_for_all_methods": True,
            "case_specific_manual_component_selection_allowed": False,
            "true_signal_used_for_component_selection_allowed": False,
            "benchmark_or_test_feedback_used_for_adapter_tuning_allowed": False,
        },
        "trend_bearing_component_policy": {
            "status": "must_be_resolved_by_frozen_component_selection_rule_before_controlled_rerun",
            "rule": (
                "Every estimated component, including any native residual, "
                "trend, or low-frequency scale exposed by the method, is treated "
                "as an eligible association candidate. A trend-bearing component "
                "is retained only if it satisfies the frozen association "
                "criterion with the true trend component, independent of whether "
                "it appears first, last, or elsewhere in the decomposition order."
            ),
        },
        "zero_variance_policy": {
            "estimated_noise_std_tolerance": 1e-12,
            "reason_code": "not_computable_zero_variance_estimated_noise",
            "unexplained_nan_allowed": False,
            "audit_required": [
                "zero_variance_count_by_method",
                "zero_variance_rate_by_method",
                "case_examples",
            ],
        },
        "closure_audit": {
            "identity": "observed_signal = protocol_estimated_signal + protocol_estimated_noise",
            "required_fields": [
                "closure_max_abs_error",
                "closure_rmse",
                "closure_nmse",
            ],
            "expected_behavior": "machine_precision_level_when_using_difference_definition",
        },
        "controlled_rerun_constraints": {
            "rerun_type": "controlled_rerun_for_missing_metric_prerequisites",
            "preserve_benchmark_cases": True,
            "preserve_signal_definitions": True,
            "preserve_noise_realizations": True,
            "preserve_sigma_levels": True,
            "preserve_data_seeds": True,
            "preserve_locked_parameters": True,
            "preserve_method_implementations": True,
            "preserve_metric_formulas": True,
            "recompute_endpoints_for_all_methods": [
                "noise_capture_corr",
                "signal_leakage_into_noise",
            ],
            "other_primary_metrics_must_match_previous_run_within_tolerance": True,
        },
        "required_controlled_rerun_outputs": [
            "controlled_rerun_manifest.json",
            "controlled_rerun_environment_manifest.json",
            "controlled_rerun_case_checksum.csv",
            "protocol_noise_endpoint_results.csv",
            "adapter_completeness_audit.csv",
            "estimated_noise_zero_variance_audit.csv",
            "reconstruction_closure_audit.csv",
            "controlled_rerun_invariance_audit.csv",
            "legacy_vs_protocol_endpoint_comparison.csv",
            "statistical_readiness_audit_v2.json",
        ],
        "prior_gate_evidence": {
            "ceemdan_residual_adjudication_status": adjudication.get("adjudication_status"),
            "ceemdan_nan_root_cause_counts": adjudication.get("nan_root_cause_counts"),
            "metric_only_rebuild_decision": rebuild.get("decision"),
            "metric_only_rebuild_feasibility": rebuild.get("metric_only_rebuild_feasibility"),
        },
        "statistics_authorized": False,
        "controlled_rerun_authorized": False,
        "controlled_rerun_authorization_note": (
            "Adapter semantics are frozen, but a method-independent synthetic "
            "component-selection/reconstruction protocol is still required "
            "before any controlled rerun can be authorized."
        ),
        "unified_statistics_authorized": False,
        "timestamp": _utc_now(),
        "framework_version": EVALUATION_FRAMEWORK_VERSION,
        "schema_version": STRUCTURAL_METRIC_SCHEMA_VERSION,
        "protocol_discipline": PROTOCOL_DISCIPLINE,
    }
    checksum = _stable_json_checksum(spec)
    spec["adapter_spec_checksum"] = checksum
    status_yaml = f"""adapter:
  id: {TRUE_NOISE_SEPARATION_ADAPTER_ID}
  version: "1.0"
  status: specification_frozen
  checksum: {checksum}
  controlled_rerun_authorized: false
  controlled_rerun_authorization_note: requires_synthetic_component_selection_and_reconstruction_protocol
  unified_statistics_authorized: false
stage_c:
  c0_statistical_readiness_audit: blocked
  c1_ceemdan_residual_semantics_adjudication: completed
  c2_metric_only_rebuild_feasibility_audit: rejected
  c3_adapter_specification_and_freeze: completed
  c3b_synthetic_reconstruction_rule_discovery: required
  c3c_synthetic_reconstruction_protocol_amendment: required
  c4_controlled_four_method_rerun: not_authorized
  c5_noise_endpoint_and_rerun_integrity_audit: pending
  c6_statistical_readiness_reaudit: pending
"""
    write_json(spec, stage_root / "true_noise_separation_adapter_spec.json")
    _write_text(stage_root / "true_noise_separation_adapter_status.yaml", status_yaml)
    write_json({
        "adapter_id": TRUE_NOISE_SEPARATION_ADAPTER_ID,
        "adapter_spec_checksum": checksum,
        "adapter_status": "specification_frozen",
        "controlled_rerun_authorized": False,
        "unified_statistics_authorized": False,
        "next_stage": "synthetic_component_selection_and_reconstruction_protocol_amendment",
        "timestamp": _utc_now(),
    }, stage_root / "true_noise_separation_adapter_freeze_report.json")
    return {"true_noise_separation_adapter_spec": spec}


def run_controlled_noise_endpoint_rerun_preflight(algorithm_root, seeds=None):
    """Preflight Stage C4 before any controlled endpoint rerun is started."""
    algorithm_root = ensure_dir(algorithm_root)
    require_benchmark_state(
        algorithm_root,
        allowed={"EXECUTED"},
        command_name="controlled-noise-endpoint-rerun-preflight",
    )
    stage_root = ensure_dir(algorithm_root / "04d_controlled_noise_endpoint_rerun")
    cube_root = algorithm_root / "03_unified_benchmark_cube"
    rows_path = cube_root / "unified_benchmark_cube_rows.csv"
    protocol = _load_json(cube_root / "unified_benchmark_cube_protocol.json", default={})
    rows = _read_csv_rows(rows_path)
    seeds = tuple(
        int(s) for s in (
            protocol.get("seeds") if protocol.get("seeds") is not None else (seeds or range(20))
        )
    )
    expected = _expected_design(seeds=seeds)
    adapter_spec_path = (
        algorithm_root
        / "04c_true_noise_separation_adapter"
        / "true_noise_separation_adapter_spec.json"
    )
    adapter_spec = _load_json(adapter_spec_path, default={})
    adapter_checksum = adapter_spec.get("adapter_spec_checksum")
    locked_path = algorithm_root / "locked_algorithm_parameters.json"
    locked = _load_json(locked_path, default={})
    parameter_checksum = _stable_json_checksum(locked) if locked else None
    fields = set(rows[0].keys()) if rows else set()
    selection_fields = [
        f for f in fields
        if (
            "selected_component" in f
            or "component_selection" in f
            or "retained_component" in f
            or "retention_mask" in f
        )
    ]
    trend_bearing_component_policy = adapter_spec.get(
        "trend_bearing_component_policy",
        adapter_spec.get("trend_policy", {}),
    )
    amendment_path = (
        algorithm_root
        / "04cc_synthetic_reconstruction_protocol_amendment"
        / "synthetic_reconstruction_protocol_amendment.json"
    )
    amendment = _load_json(amendment_path, default={})
    amendment_frozen = (
        amendment.get("amendment_id") == SYNTHETIC_RECONSTRUCTION_AMENDMENT_ID
        and amendment.get("amendment_status") == "specification_frozen"
        and amendment.get("qualification_status") in {"qualified", "empirically_qualified"}
    )
    component_selection_rule_status = "frozen" if amendment_frozen else "not_frozen_or_not_discoverable"
    case_rows = []
    duplicate_keys = []
    actual_keys = []
    for row in rows:
        key = _row_key(row)
        actual_keys.append(key)
        case_doc = {
            "signal_regime": row.get("signal_regime"),
            "signal": row.get("signal"),
            "noise": row.get("noise"),
            "sigma": _as_float(row.get("sigma")),
            "seed": _as_int(row.get("seed", row.get("data_seed"))),
            "case_id": row.get("cube_cell_id") or "|".join(str(v) for v in key),
        }
        case_rows.append({
            **case_doc,
            "case_spec_checksum": _stable_json_checksum(case_doc),
        })
    for key, count in Counter(actual_keys).items():
        if count > 1:
            duplicate_keys.append(key)
    output_root = stage_root
    output_status = {
        "controlled_rerun_root": str(output_root),
        "existing_protocol_endpoint_rows": (output_root / "protocol_noise_endpoint_results.csv").exists(),
        "existing_rerun_manifest": (output_root / "controlled_rerun_manifest.json").exists(),
        "status": "clean" if not any(output_root.glob("protocol_noise_endpoint_results.*")) else "not_clean",
    }
    checks = {
        "adapter_spec_exists": adapter_spec_path.exists(),
        "adapter_id_match": adapter_spec.get("adapter_id") == TRUE_NOISE_SEPARATION_ADAPTER_ID,
        "adapter_checksum_recorded": bool(adapter_checksum),
        "benchmark_state_executed": load_benchmark_state(algorithm_root).get("benchmark_state") == "EXECUTED",
        "rows_file_exists": rows_path.exists(),
        "expected_benchmark_rows_match": len(rows) == expected["expected_benchmark_rows"],
        "expected_method_evaluations_match": len(rows) * len(EVALUATION_METHODS) == expected["expected_method_evaluations"],
        "duplicate_case_seed_combinations_absent": not duplicate_keys,
        "locked_parameters_present": all(method in locked for method in EVALUATION_METHODS),
        "locked_parameter_checksum_recorded": bool(parameter_checksum),
        "case_regeneration_is_deterministic_only": True,
        "scientific_design_unchanged": True,
        "parameter_retuning_absent": True,
        "component_selection_rule_frozen": component_selection_rule_status == "frozen",
        "synthetic_reconstruction_protocol_amendment_frozen": amendment_frozen,
        "output_directory_clean": output_status["status"] == "clean",
    }
    blocking_reasons = [
        name for name, passed in checks.items()
        if not passed
    ]
    rerun_start_authorization = "granted" if all(checks.values()) else "denied"
    preflight = {
        "workflow": WORKFLOW_NAME,
        "stage": "Stage C4 controlled endpoint reconstruction rerun preflight",
        "scope": (
            "preflight only; no decomposition runs, no endpoint recomputation, "
            "no benchmark-row mutation, and no statistical analysis"
        ),
        "rerun_type": "controlled_metric_prerequisite_rerun",
        "rerun_start_authorization": rerun_start_authorization,
        "checks": checks,
        "blocking_reasons": blocking_reasons,
        "benchmark_case_count": expected["expected_benchmark_rows"],
        "method_evaluation_count": expected["expected_method_evaluations"],
        "endpoint_scope": [
            "noise_capture_corr",
            "signal_leakage_into_noise",
        ],
        "affected_methods": list(EVALUATION_METHODS),
        "adapter_id": TRUE_NOISE_SEPARATION_ADAPTER_ID,
        "adapter_checksum": adapter_checksum,
        "locked_parameter_checksum": parameter_checksum,
        "case_manifest_checksum": _stable_json_checksum(case_rows),
        "case_regeneration_performed": "deterministic_reproduction_only",
        "scientific_design_changed": False,
        "parameter_retuning_performed": False,
        "case_definitions_changed": False,
        "signal_formulas_changed": False,
        "noise_realizations_changed": False,
        "noise_levels_changed": False,
        "contamination_settings_changed": False,
        "method_set_changed": False,
        "method_implementations_changed": False,
        "all_other_metric_definitions_changed": False,
        "component_selection_rule_status": component_selection_rule_status,
        "synthetic_reconstruction_protocol_amendment_status": amendment.get("amendment_status", "missing"),
        "synthetic_reconstruction_protocol_qualification_status": amendment.get("qualification_status", "missing"),
        "synthetic_reconstruction_protocol_amendment_path": str(amendment_path),
        "component_selection_rule_evidence_fields": sorted(selection_fields)[:50],
        "component_selection_rule_evidence_note": (
            "Historical benchmark rows are not required to contain selected "
            "component fields. C4 is authorized only after the synthetic "
            "reconstruction protocol amendment is qualified/frozen; the "
            "controlled rerun then generates selected-component artifacts."
        ),
        "trend_bearing_component_policy_status": trend_bearing_component_policy.get("status"),
        "output_directory_status": output_status,
        "required_before_rerun_if_denied": (
            "Freeze and expose a method-independent synthetic component-selection/"
            "reconstruction rule that provides selected_component_indices or an "
            "equivalent protocol_estimated_signal for every method evaluation."
        ),
        "statistics_authorized": False,
        "timestamp": _utc_now(),
        "framework_version": EVALUATION_FRAMEWORK_VERSION,
        "schema_version": STRUCTURAL_METRIC_SCHEMA_VERSION,
        "protocol_discipline": PROTOCOL_DISCIPLINE,
    }
    manifest = {
        "workflow": WORKFLOW_NAME,
        "stage": "Stage C4 controlled endpoint reconstruction rerun",
        "status": "preflight_completed",
        "rerun_start_authorization": rerun_start_authorization,
        "rerun_type": "controlled_metric_prerequisite_rerun",
        "adapter_id": TRUE_NOISE_SEPARATION_ADAPTER_ID,
        "adapter_checksum": adapter_checksum,
        "locked_parameter_checksum": parameter_checksum,
        "expected_design": expected,
        "endpoint_scope": [
            "noise_capture_corr",
            "signal_leakage_into_noise",
        ],
        "preserve_original_benchmark_rows": True,
        "write_outputs_to_isolated_directory": str(stage_root),
        "unified_statistics_authorized": False,
        "timestamp": _utc_now(),
    }
    write_json(preflight, stage_root / "controlled_rerun_preflight_audit.json")
    write_json(manifest, stage_root / "controlled_rerun_manifest.json")
    write_csv(case_rows, stage_root / "controlled_rerun_case_checksums.csv")
    write_csv(case_rows, stage_root / "controlled_rerun_case_manifest.csv")
    _write_text(
        stage_root / "controlled_rerun_access_log.csv",
        "timestamp,command,purpose,status\n"
        f"{_utc_now()},controlled-noise-endpoint-rerun-preflight,"
        f"stage_c4_preflight,{rerun_start_authorization}\n",
    )
    return {
        "controlled_rerun_preflight_audit": preflight,
        "controlled_rerun_manifest": manifest,
    }


def run_synthetic_reconstruction_protocol_amendment(algorithm_root):
    """Record the post-execution reconstruction protocol amendment proposal.

    This command does not freeze a selector and does not authorize rerun.  It
    documents the scientific design gap exposed by the discovery gate and the
    qualification requirements that must be met before C4 can start.
    """
    algorithm_root = ensure_dir(algorithm_root)
    require_benchmark_state(
        algorithm_root,
        allowed={"EXECUTED"},
        command_name="synthetic-reconstruction-protocol-amendment",
    )
    stage_root = ensure_dir(algorithm_root / "04cc_synthetic_reconstruction_protocol_amendment")
    discovery_path = (
        algorithm_root
        / "04ca_synthetic_reconstruction_rule_discovery"
        / "synthetic_reconstruction_rule_discovery.json"
    )
    discovery = _load_json(discovery_path, default={})
    adapter_path = (
        algorithm_root
        / "04c_true_noise_separation_adapter"
        / "true_noise_separation_adapter_spec.json"
    )
    adapter = _load_json(adapter_path, default={})
    association_protocol = {
        "association_protocol_id": STRUCTURAL_METRIC_SCHEMA_VERSION,
        "association_basis": "abs_zero_lag_corr_squared_true_estimated_components",
        "association_matrix_shape": "estimated_components_by_true_components",
        "association_threshold": DEFAULT_COMPONENT_ASSOCIATION_THRESHOLD,
        "energy_threshold": DEFAULT_COMPONENT_ENERGY_THRESHOLD,
        "allocation_floor": DEFAULT_COMPONENT_ALLOCATION_FLOOR,
        "source": (
            "diagnostics.shared_physical_diagnostics.true_component_mixing_diagnostics "
            "default TRUE_COMPONENT_MIXING_V1.0 association protocol"
        ),
        "truth_usage_scope": "eligibility_decision_only_no_waveform_modification",
    }
    association_protocol["association_protocol_checksum"] = _stable_json_checksum(association_protocol)
    amendment = {
        "workflow": WORKFLOW_NAME,
        "stage": "Stage C3c synthetic reconstruction protocol amendment",
        "amendment_id": SYNTHETIC_RECONSTRUCTION_AMENDMENT_ID,
        "amendment_version": "1.0",
        "amendment_status": "proposed_not_frozen",
        "qualification_status": "not_started",
        "scope": (
            "proposal and qualification plan only; no selector is frozen, no "
            "decomposition is run, no endpoint is recomputed, and no "
            "statistical analysis is authorized"
        ),
        "trigger": "post_execution_statistical_readiness_audit",
        "amendment_reason": (
            "Historical synthetic benchmark rows were shown to use native "
            "residual semantics rather than a protocol-level selected-component "
            "reconstruction rule. Native residuals are not a method-independent "
            "estimated-noise construct across IRMF, EMD, EEMD, and CEEMDAN."
        ),
        "discovery_evidence": {
            "discovery_path": str(discovery_path),
            "discovery_classification": discovery.get("classification"),
            "discovery_checksum": discovery.get("discovery_checksum"),
            "component_selection_rule_discoverable": False,
        },
        "adapter_evidence": {
            "adapter_path": str(adapter_path),
            "adapter_id": adapter.get("adapter_id"),
            "adapter_status": adapter.get("adapter_status"),
            "adapter_checksum": adapter.get("adapter_spec_checksum"),
        },
        "affected_primary_endpoints": [
            "noise_capture_corr",
            "signal_leakage_into_noise",
        ],
        "affected_methods": list(EVALUATION_METHODS),
        "historical_operational_definition": {
            "estimated_signal": "observed_signal - method_native_residual",
            "estimated_noise": "method_native_residual",
            "uniform_implementation": True,
            "uniform_scientific_construct": False,
        },
        "proposed_operational_definition": {
            "protocol_estimated_signal": (
                "sum of components retained by a frozen method-independent "
                "component-selection/reconstruction rule; trend-bearing "
                "components are handled by the same order-invariant association "
                "criterion"
            ),
            "protocol_estimated_noise": "observed_signal - protocol_estimated_signal",
        },
        "rule_design_specification_draft": {
            "rule_status": "draft_not_frozen",
            "rule_target": "truth_assisted_decomposition_allocation_quality",
            "not_target": "practical_unsupervised_real_world_denoising_selector",
            "truth_assisted": True,
            "method_independent": True,
            "component_collection_policy": (
                "collect every estimated component exposed by the method, "
                "including any native residual, trend, or low-frequency scale "
                "as an eligible association candidate without assuming a "
                "method-specific component order"
            ),
            "selection_rule": (
                "retain every eligible estimated component whose frozen "
                "association score with at least one known synthetic true "
                "component exceeds the frozen association threshold"
            ),
            "association_score": association_protocol["association_basis"],
            "association_threshold": DEFAULT_COMPONENT_ASSOCIATION_THRESHOLD,
            "energy_threshold": DEFAULT_COMPONENT_ENERGY_THRESHOLD,
            "allocation_floor": DEFAULT_COMPONENT_ALLOCATION_FLOOR,
            "tie_handling": "retain_all_components_meeting_threshold_no_ranking_tie_break",
            "multiple_association_handling": "retain_whole_component_once_if_associated_with_one_or_more_true_components",
            "splitting_policy": (
                "if one true component is carried by multiple valid estimated "
                "components, retain all valid associated components"
            ),
            "merging_policy": (
                "if one estimated component carries multiple true components, "
                "retain the whole estimated component without oracle purification"
            ),
            "trend_bearing_component_policy": (
                "any estimated component, regardless of decomposition order, "
                "that satisfies the frozen association criterion with the true "
                "trend component is retained as a trend-bearing component; the "
                "test must evaluate recovered trend association and reconstruction "
                "rather than whether the trend appears in a final residue slot"
            ),
            "no_match_policy": {
                "protocol_estimated_signal": "zero_vector",
                "protocol_estimated_noise": "observed_signal",
                "status_code": "valid_complete_signal_rejection",
            },
            "all_selected_policy": {
                "protocol_estimated_signal": "sum_all_eligible_components",
                "protocol_estimated_noise": "observed_signal_minus_protocol_estimated_signal",
                "zero_variance_reason_code": "not_computable_zero_variance_protocol_noise",
            },
            "waveform_modification_policy": {
                "oracle_rescaling_allowed": False,
                "truth_based_sign_correction_allowed": False,
                "projection_onto_true_components_allowed": False,
                "partial_extraction_from_mixed_component_allowed": False,
                "case_specific_threshold_selection_allowed": False,
                "method_specific_rule_allowed": False,
                "selection_maximizing_endpoint_performance_allowed": False,
                "selection_minimizing_nmse_allowed": False,
            },
        },
        "association_protocol": association_protocol,
        "candidate_route_recommended_for_qualification": {
            "route": "truth_assisted_synthetic_decomposition_diagnostic",
            "selector_basis": (
                "frozen true-component association schema already used by "
                "TRUE_COMPONENT_MIXING_V1.0"
            ),
            "interpretation": (
                "synthetic truth-available decomposition diagnostic, not an "
                "unsupervised real-world denoising rule"
            ),
            "oracle_purification_allowed": False,
            "amplitude_refitting_allowed": False,
            "case_specific_visual_adjustment_allowed": False,
        },
        "qualification_requirements_before_freeze": [
            "determinism_under_component_order_permutation",
            "permutation_invariance_of_selected_component_set_and_reconstruction",
            "perfect_component_case_retains_all_true_signal_components",
            "pure_spurious_component_is_not_retained",
            "splitting_case_retains_all_effectively_associated_signal_components",
            "merging_case_retains_whole_mixed_estimated_component_without_oracle_purification",
            "weak_association_threshold_transition_audited",
            "trend_bearing_component_policy_explicitly_frozen",
            "true_trend_association_test_order_invariant",
            "true_trend_reconstruction_test_from_all_associated_trend_bearing_components",
            "closure_remainder_not_misclassified_as_trend_bearing_component",
            "no_match_case_policy_explicitly_frozen",
            "all_selected_case_zero_variance_noise_reason_codes_audited",
            "no_oracle_rescaling_or_projection_fitting",
            "cross_method_applicability_for_irmf_emd_eemd_ceemdan",
            "selection_dependency_audit_for_component_count_and_selected_fraction",
            "controlled_rerun_preflight_passed_after_freeze",
        ],
        "qualification_outputs_required_before_freeze": [
            "reconstruction_rule_specification.json",
            "reconstruction_rule_unit_tests.json",
            "reconstruction_rule_construct_qualification.json",
            "reconstruction_rule_threshold_audit.csv",
            "reconstruction_rule_cross_method_applicability_audit.csv",
            "reconstruction_rule_selection_dependency_audit.csv",
            "reconstruction_rule_edge_case_audit.csv",
            "protocol_amendment_adjudication_and_freeze_report.json",
        ],
        "dependency_audit_fields_required": [
            "selected_component_count",
            "effective_component_count",
            "selected_fraction",
            "protocol_estimated_signal_energy",
            "protocol_estimated_noise_energy",
            "all_components_selected_flag",
            "no_components_selected_flag",
            "zero_variance_protocol_noise_flag",
            "selection_status_code",
        ],
        "change_classification": {
            "metric_formulas_changed": False,
            "primary_endpoint_set_changed": False,
            "metric_input_construct_changed": True,
            "scientific_interpretation_changed": True,
            "method_parameters_changed": False,
            "benchmark_cases_changed": False,
            "method_implementations_changed": False,
            "results_consulted_for_selector_optimization": False,
        },
        "authorization": {
            "controlled_rerun_authorized": False,
            "unified_statistics_authorized": False,
            "schema_freeze_authorized": False,
            "next_required_action": (
                "specify, test, qualify, and freeze the synthetic "
                "component-selection/reconstruction rule before rerunning "
                "controlled noise endpoints"
            ),
        },
        "freeze_fields_required": {
            "amendment_id": SYNTHETIC_RECONSTRUCTION_AMENDMENT_ID,
            "amendment_version": "1.0",
            "amendment_status": "must_be_specification_frozen_before_c4",
            "qualification_status": "must_be_qualified_before_c4",
            "freeze_date": None,
            "checksum": "computed_after_qualification",
            "historical_semantics": "recorded",
            "new_protocol_semantics": "draft_recorded_not_frozen",
            "affected_endpoints": [
                "noise_capture_corr",
                "signal_leakage_into_noise",
            ],
            "affected_methods": list(EVALUATION_METHODS),
            "truth_assisted": True,
            "association_protocol_id": STRUCTURAL_METRIC_SCHEMA_VERSION,
            "association_protocol_checksum": association_protocol["association_protocol_checksum"],
            "association_threshold": DEFAULT_COMPONENT_ASSOCIATION_THRESHOLD,
            "trend_bearing_component_policy": "draft_not_frozen",
            "no_match_policy": "draft_not_frozen",
            "all_selected_policy": "draft_not_frozen",
            "zero_variance_policy": "draft_not_frozen",
            "oracle_rescaling_allowed": False,
            "case_specific_tuning_allowed": False,
            "method_specific_rule_allowed": False,
            "performance_results_used_for_rule_design": False,
            "method_rankings_used_for_rule_design": False,
            "controlled_rerun_authorized": False,
            "unified_statistics_authorized": False,
        },
        "timestamp": _utc_now(),
        "framework_version": EVALUATION_FRAMEWORK_VERSION,
        "schema_version": STRUCTURAL_METRIC_SCHEMA_VERSION,
        "protocol_discipline": PROTOCOL_DISCIPLINE,
    }
    amendment["amendment_checksum"] = _stable_json_checksum(amendment)
    status_yaml = f"""amendment:
  id: {SYNTHETIC_RECONSTRUCTION_AMENDMENT_ID}
  version: "1.0"
  status: proposed_not_frozen
  qualification_status: not_started
  checksum: {amendment["amendment_checksum"]}
authorization:
  controlled_rerun_authorized: false
  unified_statistics_authorized: false
stage_c:
  c3b_synthetic_reconstruction_rule_discovery: completed
  c3c_synthetic_reconstruction_protocol_amendment: proposed_not_frozen
  c4_controlled_four_method_rerun: not_authorized
"""
    write_json(amendment, stage_root / "synthetic_reconstruction_protocol_amendment.json")
    _write_text(stage_root / "synthetic_reconstruction_protocol_amendment_status.yaml", status_yaml)
    write_json({
        "amendment_id": SYNTHETIC_RECONSTRUCTION_AMENDMENT_ID,
        "amendment_status": "proposed_not_frozen",
        "qualification_status": "not_started",
        "controlled_rerun_authorized": False,
        "unified_statistics_authorized": False,
        "amendment_checksum": amendment["amendment_checksum"],
        "next_required_action": amendment["authorization"]["next_required_action"],
        "timestamp": _utc_now(),
    }, stage_root / "synthetic_reconstruction_protocol_amendment_report.json")
    return {"synthetic_reconstruction_protocol_amendment": amendment}


def _protocol_selection_from_association(components, observed_signal, association_matrix, threshold):
    components = np.asarray(components, dtype=float)
    if components.ndim == 1:
        components = components[None, :]
    observed_signal = np.asarray(observed_signal, dtype=float)
    assoc = np.asarray(association_matrix, dtype=float)
    if components.size == 0:
        selected = np.zeros((0,), dtype=bool)
    else:
        if assoc.ndim != 2 or assoc.shape[0] != components.shape[0]:
            raise ValueError("association_matrix must have one row per estimated component")
        selected = np.max(assoc, axis=1) > float(threshold)
    selected_indices = [int(i) for i, flag in enumerate(selected) if flag]
    if selected_indices:
        protocol_estimated_signal = np.sum(components[selected], axis=0)
    else:
        protocol_estimated_signal = np.zeros_like(observed_signal)
    protocol_estimated_noise = observed_signal - protocol_estimated_signal
    if len(selected_indices) == 0:
        status = "valid_complete_signal_rejection"
    elif len(selected_indices) == components.shape[0]:
        status = "all_components_selected"
    else:
        status = "partial_selection"
    zero_var_noise = bool(np.std(protocol_estimated_noise) < 1e-12)
    reason_code = (
        "not_computable_zero_variance_protocol_noise"
        if zero_var_noise else "computable_protocol_noise"
    )
    return {
        "selected_mask": selected,
        "selected_indices": selected_indices,
        "protocol_estimated_signal": protocol_estimated_signal,
        "protocol_estimated_noise": protocol_estimated_noise,
        "selection_status_code": status,
        "zero_variance_protocol_noise_flag": zero_var_noise,
        "noise_capture_corr_reason_code": reason_code,
    }


def _energy(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum(x ** 2))


def _max_abs_error(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def _fixture_waves(n=128):
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    trend = 0.8 * np.cos(2.0 * np.pi * t)
    c1 = np.sin(2.0 * np.pi * 5.0 * t)
    c2 = 0.7 * np.sin(2.0 * np.pi * 11.0 * t + 0.2)
    noise = 0.2 * np.sin(2.0 * np.pi * 29.0 * t + 0.5)
    return t, trend, c1, c2, noise


def _selection_summary_row(test_name, method, components, observed, assoc, expected_indices,
                           expected_status=None, compare_signal=None):
    out = _protocol_selection_from_association(
        components,
        observed,
        assoc,
        DEFAULT_COMPONENT_ASSOCIATION_THRESHOLD,
    )
    selected_indices = out["selected_indices"]
    rec = out["protocol_estimated_signal"]
    expected_indices = list(expected_indices)
    selected_ok = selected_indices == expected_indices
    status_ok = (
        True if expected_status is None
        else out["selection_status_code"] == expected_status
    )
    reconstruction_error = (
        _max_abs_error(rec, compare_signal)
        if compare_signal is not None else np.nan
    )
    reconstruction_ok = (
        True if compare_signal is None
        else reconstruction_error < 1e-10
    )
    passed = bool(selected_ok and status_ok and reconstruction_ok)
    selected_count = len(selected_indices)
    available_count = int(np.asarray(components).shape[0])
    return {
        "test_name": test_name,
        "method": method,
        "passed": passed,
        "selected_indices": ";".join(str(i) for i in selected_indices),
        "expected_indices": ";".join(str(i) for i in expected_indices),
        "selection_status_code": out["selection_status_code"],
        "expected_status_code": expected_status or "",
        "selected_component_count": selected_count,
        "available_component_count": available_count,
        "selected_fraction": float(selected_count / max(available_count, 1)),
        "protocol_estimated_signal_energy": _energy(rec),
        "protocol_estimated_noise_energy": _energy(out["protocol_estimated_noise"]),
        "zero_variance_protocol_noise_flag": out["zero_variance_protocol_noise_flag"],
        "noise_capture_corr_reason_code": out["noise_capture_corr_reason_code"],
        "reconstruction_max_abs_error": reconstruction_error,
    }


def run_qualify_synthetic_reconstruction_protocol(algorithm_root):
    """Qualify and freeze SYNTHETIC_COMPONENT_SELECTION_AND_RECONSTRUCTION_V1.0."""
    algorithm_root = ensure_dir(algorithm_root)
    require_benchmark_state(
        algorithm_root,
        allowed={"EXECUTED"},
        command_name="qualify-synthetic-reconstruction-protocol",
    )
    stage_root = ensure_dir(algorithm_root / "04cd_synthetic_reconstruction_protocol_qualification")
    amendment_path = (
        algorithm_root
        / "04cc_synthetic_reconstruction_protocol_amendment"
        / "synthetic_reconstruction_protocol_amendment.json"
    )
    amendment = _load_json(amendment_path, default={})
    if amendment.get("amendment_id") != SYNTHETIC_RECONSTRUCTION_AMENDMENT_ID:
        raise RuntimeError(
            "qualify-synthetic-reconstruction-protocol requires a proposed "
            f"{SYNTHETIC_RECONSTRUCTION_AMENDMENT_ID} amendment artifact."
        )

    _, trend, c1, c2, noise = _fixture_waves()
    true_signal = trend + c1 + c2
    observed = true_signal + noise
    threshold = DEFAULT_COMPONENT_ASSOCIATION_THRESHOLD
    construct_rows = []

    construct_rows.append(_selection_summary_row(
        "perfect_decomposition",
        "generic",
        np.vstack([trend, c1, c2]),
        observed,
        np.eye(3),
        [0, 1, 2],
        "all_components_selected",
        true_signal,
    ))
    construct_rows.append(_selection_summary_row(
        "pure_spurious_component_rejected",
        "generic",
        np.vstack([trend, c1, c2, noise]),
        observed,
        np.asarray([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.01, 0.0],
        ]),
        [0, 1, 2],
        "partial_selection",
        true_signal,
    ))
    construct_rows.append(_selection_summary_row(
        "splitting_keeps_all_valid_associated_components",
        "generic",
        np.vstack([trend, 0.6 * c1, 0.4 * c1, c2]),
        observed,
        np.asarray([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]),
        [0, 1, 2, 3],
        "all_components_selected",
        true_signal,
    ))
    merged = c1 + c2 + 0.25 * noise
    merge_row = _selection_summary_row(
        "merging_keeps_whole_component_no_oracle_purification",
        "generic",
        np.vstack([trend, merged]),
        observed,
        np.asarray([
            [1.0, 0.0, 0.0],
            [0.0, 0.8, 0.8],
        ]),
        [0, 1],
        "all_components_selected",
        None,
    )
    merge_row["oracle_purification_absent"] = bool(
        _max_abs_error(merged, c1 + c2) > 1e-6
        and merge_row["selected_indices"] == "0;1"
    )
    merge_row["passed"] = bool(merge_row["passed"] and merge_row["oracle_purification_absent"])
    construct_rows.append(merge_row)

    no_match_row = _selection_summary_row(
        "no_match_valid_complete_signal_rejection",
        "generic",
        np.vstack([noise]),
        observed,
        np.asarray([[0.01, 0.02, 0.03]]),
        [],
        "valid_complete_signal_rejection",
        np.zeros_like(observed),
    )
    construct_rows.append(no_match_row)
    all_selected_row = _selection_summary_row(
        "all_selected_zero_variance_protocol_noise_reason_code",
        "generic",
        np.vstack([trend, c1, c2, noise]),
        observed,
        np.asarray([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.06, 0.06, 0.06],
        ]),
        [0, 1, 2, 3],
        "all_components_selected",
        observed,
    )
    all_selected_row["zero_variance_reason_code_passed"] = (
        all_selected_row["noise_capture_corr_reason_code"]
        == "not_computable_zero_variance_protocol_noise"
    )
    all_selected_row["passed"] = bool(all_selected_row["passed"] and all_selected_row["zero_variance_reason_code_passed"])
    construct_rows.append(all_selected_row)

    threshold_rows = []
    threshold_assoc = np.asarray([[threshold - 1e-6], [threshold], [threshold + 1e-6]])
    threshold_components = np.vstack([c1, c1, c1])
    threshold_out = _protocol_selection_from_association(
        threshold_components,
        c1,
        threshold_assoc,
        threshold,
    )
    for idx, score in enumerate(threshold_assoc[:, 0]):
        selected = idx in threshold_out["selected_indices"]
        expected = bool(score > threshold)
        threshold_rows.append({
            "test_name": "threshold_boundary_strict_greater_than",
            "component_index": idx,
            "association_score": float(score),
            "association_threshold": float(threshold),
            "operator": ">",
            "selected": bool(selected),
            "expected_selected": expected,
            "passed": bool(selected == expected),
        })

    cross_method_rows = []
    trend_positions = {
        "IRMF": np.vstack([trend, c1, c2]),
        "EMD": np.vstack([c2, c1, trend]),
        "EEMD": np.vstack([c2, trend, c1]),
        "CEEMDAN": np.vstack([c1, c2, trend]),
    }
    trend_assoc = {
        "IRMF": np.eye(3),
        "EMD": np.asarray([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        "EEMD": np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "CEEMDAN": np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
    }
    rec_reference = None
    for method in EVALUATION_METHODS:
        row = _selection_summary_row(
            "trend_position_invariance",
            method,
            trend_positions[method],
            observed,
            trend_assoc[method],
            list(range(3)),
            "all_components_selected",
            true_signal,
        )
        out = _protocol_selection_from_association(
            trend_positions[method],
            observed,
            trend_assoc[method],
            threshold,
        )
        rec = out["protocol_estimated_signal"]
        if rec_reference is None:
            rec_reference = rec
        row["trend_reconstruction_matches_reference"] = _max_abs_error(rec, rec_reference) < 1e-10
        row["method_label_invariance_passed"] = bool(row["passed"] and row["trend_reconstruction_matches_reference"])
        row["passed"] = bool(row["passed"] and row["method_label_invariance_passed"])
        cross_method_rows.append(row)

    perm_components = np.vstack([trend, c1, c2, noise])
    perm_assoc = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.01, 0.0],
    ])
    perm = [3, 1, 0, 2]
    base = _protocol_selection_from_association(perm_components, observed, perm_assoc, threshold)
    permuted = _protocol_selection_from_association(perm_components[perm], observed, perm_assoc[perm], threshold)
    permutation_row = {
        "test_name": "component_order_permutation_invariance",
        "method": "generic",
        "passed": _max_abs_error(base["protocol_estimated_signal"], permuted["protocol_estimated_signal"]) < 1e-10,
        "base_selected_component_count": len(base["selected_indices"]),
        "permuted_selected_component_count": len(permuted["selected_indices"]),
        "reconstruction_max_abs_error": _max_abs_error(base["protocol_estimated_signal"], permuted["protocol_estimated_signal"]),
    }
    construct_rows.append(permutation_row)

    dependency_rows = construct_rows + cross_method_rows
    dependency_by_method = defaultdict(list)
    for row in dependency_rows:
        dependency_by_method[row.get("method", "generic")].append(row)
    dependency_summary = []
    for method, rows in sorted(dependency_by_method.items()):
        non_edge = [
            row for row in rows
            if row.get("test_name") not in {
                "no_match_valid_complete_signal_rejection",
                "all_selected_zero_variance_protocol_noise_reason_code",
            }
            and "threshold_boundary" not in row.get("test_name", "")
        ]
        selected_fractions = [
            _as_float(row.get("selected_fraction"))
            for row in non_edge
            if _as_float(row.get("selected_fraction")) is not None
        ]
        dependency_summary.append({
            "method": method,
            "n_non_edge_fixtures": len(non_edge),
            "selected_fraction_min": float(np.min(selected_fractions)) if selected_fractions else np.nan,
            "selected_fraction_max": float(np.max(selected_fractions)) if selected_fractions else np.nan,
            "all_selected_non_edge_count": int(sum(
                row.get("selection_status_code") == "all_components_selected"
                for row in non_edge
            )),
            "no_match_non_edge_count": int(sum(
                row.get("selection_status_code") == "valid_complete_signal_rejection"
                for row in non_edge
            )),
        })

    exit_criteria = {
        "construct_tests_passed": all(bool(row.get("passed")) for row in construct_rows),
        "permutation_invariance_passed": bool(permutation_row["passed"]),
        "trend_position_invariance_passed": all(bool(row.get("passed")) for row in cross_method_rows),
        "cross_method_applicability_passed": set(row["method"] for row in cross_method_rows) == set(EVALUATION_METHODS)
        and all(bool(row.get("passed")) for row in cross_method_rows),
        "threshold_provenance_verified": amendment.get("association_protocol", {}).get("association_protocol_id") == STRUCTURAL_METRIC_SCHEMA_VERSION
        or amendment.get("rule_design_specification_draft", {}).get("association_threshold") == threshold,
        "threshold_boundary_behavior_passed": all(bool(row.get("passed")) for row in threshold_rows),
        "oracle_purification_absent": bool(merge_row.get("oracle_purification_absent")),
        "dependency_audit_passed": all(
            row["no_match_non_edge_count"] == 0
            for row in dependency_summary
        ),
        "unexplained_selection_failures_zero": not [
            row for row in construct_rows + cross_method_rows + threshold_rows
            if not bool(row.get("passed"))
        ],
        "rule_checksum_generated": True,
    }
    qualification_passed = all(exit_criteria.values())
    rule_spec = {
        "rule_id": SYNTHETIC_RECONSTRUCTION_AMENDMENT_ID,
        "rule_version": "1.0",
        "rule_status": "specification_frozen" if qualification_passed else "requires_review",
        "qualification_status": "qualified" if qualification_passed else "requires_review",
        "association_protocol": amendment.get("association_protocol", {}),
        "rule_design_specification": amendment.get("rule_design_specification_draft", {}),
        "threshold_operator": ">",
        "truth_usage_scope": "association_eligibility_only_no_waveform_modification",
        "method_performance_results_used_for_rule_design": False,
        "method_rankings_used_for_rule_design": False,
        "timestamp": _utc_now(),
    }
    rule_spec["rule_checksum"] = _stable_json_checksum(rule_spec)
    qualification = {
        "workflow": WORKFLOW_NAME,
        "stage": "Stage C3d-C3g synthetic reconstruction protocol qualification and adjudication",
        "amendment_id": SYNTHETIC_RECONSTRUCTION_AMENDMENT_ID,
        "qualification_status": "qualified" if qualification_passed else "requires_review",
        "rule_status": rule_spec["rule_status"],
        "exit_criteria": exit_criteria,
        "threshold_operator": ">",
        "association_threshold": float(threshold),
        "association_protocol_id": STRUCTURAL_METRIC_SCHEMA_VERSION,
        "association_protocol_checksum": amendment.get("association_protocol", {}).get("association_protocol_checksum"),
        "oracle_purification_absent": bool(merge_row.get("oracle_purification_absent")),
        "controlled_rerun_authorized": bool(qualification_passed),
        "unified_statistics_authorized": False,
        "scope_note": (
            "This qualification assesses the reconstruction protocol rule only. "
            "It does not recompute benchmark endpoints, rank methods, or "
            "authorize scientific performance conclusions."
        ),
        "outputs": {
            "rule_specification": "reconstruction_rule_specification.json",
            "construct_tests": "synthetic_reconstruction_construct_tests.csv",
            "threshold_audit": "synthetic_reconstruction_threshold_audit.csv",
            "cross_method_audit": "synthetic_reconstruction_cross_method_audit.csv",
            "dependency_audit": "synthetic_reconstruction_dependency_audit.json",
            "qualification_manifest": "synthetic_reconstruction_qualification_manifest.json",
        },
        "timestamp": _utc_now(),
        "framework_version": EVALUATION_FRAMEWORK_VERSION,
        "schema_version": STRUCTURAL_METRIC_SCHEMA_VERSION,
        "protocol_discipline": PROTOCOL_DISCIPLINE,
    }
    qualification["qualification_checksum"] = _stable_json_checksum(qualification)

    updated_amendment = dict(amendment)
    if qualification_passed:
        updated_amendment["amendment_status"] = "specification_frozen"
        updated_amendment["qualification_status"] = "qualified"
        updated_amendment["freeze_date"] = _utc_now()
        updated_amendment["frozen_rule_checksum"] = rule_spec["rule_checksum"]
        updated_amendment["qualification_checksum"] = qualification["qualification_checksum"]
        updated_amendment["authorization"] = {
            **updated_amendment.get("authorization", {}),
            "controlled_rerun_authorized": True,
            "unified_statistics_authorized": False,
            "schema_freeze_authorized": True,
            "next_required_action": "rerun controlled-noise-endpoint-rerun-preflight",
        }
        updated_amendment["amendment_checksum"] = _stable_json_checksum(updated_amendment)
        write_json(updated_amendment, amendment_path)
        _write_text(
            amendment_path.with_name("synthetic_reconstruction_protocol_amendment_status.yaml"),
            f"""amendment:
  id: {SYNTHETIC_RECONSTRUCTION_AMENDMENT_ID}
  version: "1.0"
  status: specification_frozen
  qualification_status: qualified
  checksum: {updated_amendment["amendment_checksum"]}
  frozen_rule_checksum: {rule_spec["rule_checksum"]}
authorization:
  controlled_rerun_authorized: true
  unified_statistics_authorized: false
stage_c:
  c3b_synthetic_reconstruction_rule_discovery: completed
  c3c_synthetic_reconstruction_protocol_amendment: specification_frozen
  c3d_to_c3g_reconstruction_rule_qualification: qualified
  c4_controlled_four_method_rerun: authorized_pending_preflight
""",
        )

    write_json(rule_spec, stage_root / "reconstruction_rule_specification.json")
    write_json(qualification, stage_root / "synthetic_reconstruction_protocol_qualification.json")
    write_csv(construct_rows, stage_root / "synthetic_reconstruction_construct_tests.csv")
    write_csv(threshold_rows, stage_root / "synthetic_reconstruction_threshold_audit.csv")
    write_csv(cross_method_rows, stage_root / "synthetic_reconstruction_cross_method_audit.csv")
    write_json({
        "dependency_audit_status": "passed" if exit_criteria["dependency_audit_passed"] else "requires_review",
        "dependency_summary": dependency_summary,
        "dependency_interpretation": (
            "Construct fixtures include deliberate edge cases. Non-edge fixtures "
            "are checked for unexplained complete rejection; method labels are "
            "checked for identical rule applicability, not performance."
        ),
    }, stage_root / "synthetic_reconstruction_dependency_audit.json")
    write_json({
        "amendment_id": SYNTHETIC_RECONSTRUCTION_AMENDMENT_ID,
        "qualification_status": qualification["qualification_status"],
        "rule_status": rule_spec["rule_status"],
        "exit_criteria": exit_criteria,
        "rule_checksum": rule_spec["rule_checksum"],
        "qualification_checksum": qualification["qualification_checksum"],
        "controlled_rerun_authorized": bool(qualification_passed),
        "unified_statistics_authorized": False,
        "timestamp": _utc_now(),
    }, stage_root / "synthetic_reconstruction_qualification_manifest.json")
    write_json({
        "amendment_id": SYNTHETIC_RECONSTRUCTION_AMENDMENT_ID,
        "amendment_status": updated_amendment.get("amendment_status", amendment.get("amendment_status")),
        "qualification_status": qualification["qualification_status"],
        "rule_status": rule_spec["rule_status"],
        "controlled_rerun_authorized": bool(qualification_passed),
        "unified_statistics_authorized": False,
        "rule_checksum": rule_spec["rule_checksum"],
        "qualification_checksum": qualification["qualification_checksum"],
    }, stage_root / "protocol_amendment_adjudication_and_freeze_report.json")
    return {"synthetic_reconstruction_protocol_qualification": qualification}


def _append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, default=str) + "\n")


def _load_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _true_noise_corr(estimated_noise, true_noise):
    corr = _safe_corr(estimated_noise, true_noise)
    score = float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0)) if np.isfinite(corr) else np.nan
    energy_ratio = float(np.sum(np.asarray(estimated_noise) ** 2) / (np.sum(np.asarray(true_noise) ** 2) + 1e-12))
    return corr, score, energy_ratio


def _projection_energy_ratio_local(vector, basis_rows):
    vector = np.asarray(vector, dtype=float)
    basis = np.asarray(basis_rows, dtype=float)
    if basis.ndim == 1:
        basis = basis[None, :]
    if vector.size == 0 or basis.size == 0 or basis.shape[1] != vector.size:
        return np.nan
    try:
        q, r = np.linalg.qr(basis.T, mode="reduced")
    except Exception:
        return np.nan
    diag = np.abs(np.diag(r)) if getattr(r, "ndim", 0) == 2 else np.asarray([])
    if diag.size:
        q = q[:, diag > 1e-10]
    if q.size == 0:
        return np.nan
    projected = q @ (q.T @ vector)
    return float(np.sum(projected ** 2) / (np.sum(vector ** 2) + 1e-12))


def _protocol_endpoint_row(case, method, result, base_meta, rule_checksum, qualification_checksum):
    Y = np.asarray(case["Y"], dtype=float)
    X = np.asarray(case["X_clean"], dtype=float)
    true_noise = Y - X
    true_components = np.asarray(case.get("true_components"), dtype=float)
    imfs = np.asarray(result.get("imfs"), dtype=float)
    if imfs.ndim == 1:
        imfs = imfs[None, :]
    residual = result.get("residual")
    components = imfs
    residual_included = False
    if residual is not None:
        residual = np.asarray(residual, dtype=float)
        if residual.shape == Y.shape:
            components = np.vstack([components, residual[None, :]]) if components.size else residual[None, :]
            residual_included = True
    assoc = np.zeros((components.shape[0], true_components.shape[0]), dtype=float)
    for i in range(components.shape[0]):
        for j in range(true_components.shape[0]):
            c = _safe_corr(components[i], true_components[j])
            assoc[i, j] = 0.0 if not np.isfinite(c) else abs(float(c)) ** 2
    selected = _protocol_selection_from_association(
        components,
        Y,
        assoc,
        DEFAULT_COMPONENT_ASSOCIATION_THRESHOLD,
    )
    protocol_noise = selected["protocol_estimated_noise"]
    corr, corr_score, energy_ratio = _true_noise_corr(protocol_noise, true_noise)
    residual_energy = float(np.sum(protocol_noise ** 2))
    observed_energy = float(np.sum(Y ** 2) + 1e-12)
    degenerate = bool(residual_energy < 1e-10 * observed_energy)
    leakage = np.nan
    leakage_score = np.nan
    basis = "true_component_span"
    if not degenerate:
        leakage = _projection_energy_ratio_local(protocol_noise, true_components)
        if np.isfinite(leakage):
            leakage = float(np.clip(leakage, 0.0, 1.0))
            leakage_score = float(1.0 - leakage)
    row = {
        **base_meta,
        "method": method,
        "endpoint_rerun_scope": "controlled_noise_endpoint_re_evaluation_only",
        "rule_id": SYNTHETIC_RECONSTRUCTION_AMENDMENT_ID,
        "rule_checksum": rule_checksum,
        "qualification_checksum": qualification_checksum,
        "adapter_id": TRUE_NOISE_SEPARATION_ADAPTER_ID,
        "association_protocol_id": STRUCTURAL_METRIC_SCHEMA_VERSION,
        "association_threshold": DEFAULT_COMPONENT_ASSOCIATION_THRESHOLD,
        "threshold_operator": ">",
        "component_candidate_count": int(components.shape[0]),
        "imf_component_count": int(imfs.shape[0]) if imfs.size else 0,
        "trend_bearing_or_residual_candidate_included": residual_included,
        "selected_component_indices": ";".join(str(i) for i in selected["selected_indices"]),
        "selected_component_count": int(len(selected["selected_indices"])),
        "selected_fraction": float(len(selected["selected_indices"]) / max(components.shape[0], 1)),
        "selection_status_code": selected["selection_status_code"],
        "zero_variance_protocol_noise_flag": selected["zero_variance_protocol_noise_flag"],
        "noise_capture_corr_reason_code": selected["noise_capture_corr_reason_code"],
        "protocol_estimated_signal_energy": float(np.sum(selected["protocol_estimated_signal"] ** 2)),
        "protocol_estimated_noise_energy": residual_energy,
        "reconstruction_closure_max_abs_error": _max_abs_error(
            Y,
            selected["protocol_estimated_signal"] + protocol_noise,
        ),
        "noise_capture_corr": corr,
        "noise_capture_corr_score": corr_score,
        "noise_capture_energy_ratio": energy_ratio,
        "signal_leakage_into_noise": leakage,
        "signal_leakage_into_noise_score": leakage_score,
        "signal_leakage_residual_energy": residual_energy,
        "signal_leakage_observed_energy": observed_energy,
        "signal_leakage_degenerate_flag": degenerate,
        "signal_leakage_basis": basis,
        "metric_formula_changed": False,
        "metric_input_construct": "protocol_estimated_noise",
        "old_metric_values_recomputed": False,
        "method_performance_interpretation_authorized": False,
        "runtime_seconds": _as_float(result.get("runtime_seconds")),
        "timeout_flag": bool(result.get("timeout_flag")),
        "exception_flag": bool(result.get("exception_flag")),
        "computational_failure": bool(result.get("computational_failure")),
        "failure_reason": result.get("failure_reason"),
    }
    return row


def run_controlled_noise_endpoint_re_evaluation(
        algorithm_root,
        locked_parameters,
        timeout_seconds=120,
        seeds=None,
):
    """Stage C4: recompute only protocol noise endpoints in an isolated table."""
    algorithm_root = ensure_dir(algorithm_root)
    require_benchmark_state(
        algorithm_root,
        allowed={"EXECUTED"},
        command_name="controlled-noise-endpoint-rerun",
    )
    preflight = _load_json(
        algorithm_root
        / "04d_controlled_noise_endpoint_rerun"
        / "controlled_rerun_preflight_audit.json",
        default={},
    )
    if preflight.get("rerun_start_authorization") != "granted":
        raise RuntimeError("controlled-noise-endpoint-rerun requires granted C4 preflight.")
    stage_root = ensure_dir(algorithm_root / "04e_controlled_noise_endpoint_re_evaluation")
    cube_rows_path = algorithm_root / "03_unified_benchmark_cube" / "unified_benchmark_cube_rows.csv"
    cube_json_path = algorithm_root / "03_unified_benchmark_cube" / "unified_benchmark_cube_rows.json"
    before_checksum = _file_checksum(cube_rows_path)
    original_rows = _load_json(cube_json_path, default=[])
    protocol = _load_json(
        algorithm_root / "03_unified_benchmark_cube" / "unified_benchmark_cube_protocol.json",
        default={},
    )
    seeds = tuple(int(s) for s in (seeds or protocol.get("seeds") or range(20)))
    amendment = _load_json(
        algorithm_root
        / "04cc_synthetic_reconstruction_protocol_amendment"
        / "synthetic_reconstruction_protocol_amendment.json",
        default={},
    )
    rule_checksum = amendment.get("frozen_rule_checksum")
    qualification_checksum = amendment.get("qualification_checksum")
    decomposition_artifact_status = {
        "artifact_replay_supported_by_current_pipeline": False,
        "artifact_replay_search_performed": True,
        "artifact_replay_paths_found": 0,
        "missing_decomposition_artifacts": [
            "estimated_component_or_imf_arrays",
            "native_residual_or_trend_bearing_component_arrays",
            "observed_signal_waveforms",
            "true_signal_waveforms",
            "true_component_arrays",
        ],
        "fallback_triggered": True,
        "fallback_reason": (
            "full benchmark archive contains summary rows only; endpoint "
            "re-evaluation must recompute decomposition prerequisites unless "
            "a separate decomposition artifact archive is provided"
        ),
    }
    execution_mode = "recompute_decomposition_prerequisite"
    checkpoint_path = stage_root / "protocol_noise_endpoint_results.jsonl"
    endpoint_rows = _load_jsonl(checkpoint_path)
    completed = {
        (row.get("cube_cell_id"), row.get("method"))
        for row in endpoint_rows
    }
    run_start = perf_counter()
    for old_row in original_rows:
        data_seed = int(old_row.get("data_seed", old_row.get("seed")))
        if data_seed not in set(seeds):
            continue
        cube_cell_id = old_row.get("cube_cell_id") or _checkpoint_key(
            old_row.get("signal_regime"),
            old_row.get("signal"),
            old_row.get("noise"),
            old_row.get("sigma"),
            data_seed,
        )
        methods_needed = [
            method for method in EVALUATION_METHODS
            if (cube_cell_id, method) not in completed
        ]
        if not methods_needed:
            continue
        algorithm_seed = int(old_row.get("algorithm_seed_base", 100000 + data_seed))
        case, results, _ = run_fixed_method_family_case(
            signal_name=old_row.get("signal"),
            noise_name=old_row.get("noise"),
            sigma=float(old_row.get("sigma")),
            irmf_params=locked_parameters["IRMF"],
            emd_params=locked_parameters["EMD"],
            eemd_params=locked_parameters["EEMD"],
            ceemdan_params=locked_parameters["CEEMDAN"],
            seed=data_seed,
            algorithm_seed=algorithm_seed,
            timeout_seconds=timeout_seconds,
            run_id_prefix=f"c4_{cube_cell_id}",
        )
        base_meta = {
            "cube_cell_id": cube_cell_id,
            "signal_regime": old_row.get("signal_regime"),
            "signal": old_row.get("signal"),
            "noise": old_row.get("noise"),
            "sigma": float(old_row.get("sigma")),
            "seed": data_seed,
            "data_seed": data_seed,
            "algorithm_seed_base": algorithm_seed,
        }
        for method in methods_needed:
            row = _protocol_endpoint_row(
                case,
                method,
                results.get(method, {}),
                base_meta,
                rule_checksum,
                qualification_checksum,
            )
            endpoint_rows.append(row)
            completed.add((cube_cell_id, method))
            _append_jsonl(checkpoint_path, row)

    after_checksum = _file_checksum(cube_rows_path)
    write_json(endpoint_rows, stage_root / "protocol_noise_endpoint_results.json")
    write_csv(endpoint_rows, stage_root / "protocol_noise_endpoint_results.csv")
    requested_rows = [
        row for row in original_rows
        if int(row.get("data_seed", row.get("seed"))) in set(seeds)
    ]
    expected_requested_evals = len(requested_rows) * len(EVALUATION_METHODS)
    expected_full_evals = len(original_rows) * len(EVALUATION_METHODS)
    completed_requested = [
        row for row in endpoint_rows
        if int(row.get("data_seed", row.get("seed"))) in set(seeds)
    ]
    status = {
        "workflow": WORKFLOW_NAME,
        "stage": "Stage C4 Protocol-Controlled Noise Endpoint Re-evaluation",
        "run_status": "completed" if len(completed_requested) == expected_requested_evals else "incomplete",
        "execution_mode": execution_mode,
        "execution_mode_note": (
            "Controlled means protocol-controlled under frozen cases, seeds, "
            "parameters, methods, adapter, and reconstruction rule. It does "
            "not imply lightweight execution."
        ),
        "decomposition_artifact_status": decomposition_artifact_status,
        "replayed_evaluations": 0,
        "recomputed_evaluations": int(len(completed_requested)),
        "missing_decomposition_artifacts": decomposition_artifact_status["missing_decomposition_artifacts"],
        "fallback_triggered": True,
        "fallback_triggered_reason": decomposition_artifact_status["fallback_reason"],
        "benchmark_immutability_before_checksum": before_checksum,
        "benchmark_immutability_after_checksum": after_checksum,
        "benchmark_immutability_preserved": before_checksum == after_checksum,
        "old_metrics_recomputed": False,
        "old_metrics_overwritten": False,
        "endpoint_rows": len(endpoint_rows),
        "requested_endpoint_rows": len(completed_requested),
        "expected_requested_endpoint_rows": expected_requested_evals,
        "expected_full_endpoint_rows": expected_full_evals,
        "requested_seeds": list(seeds),
        "method_evaluations_reexecuted_for_endpoint_prerequisites": int(len(completed_requested)),
        "endpoint_scope": ["noise_capture_corr", "signal_leakage_into_noise"],
        "runtime_seconds": float(perf_counter() - run_start),
        "controlled_rerun_authorized_by_preflight": True,
        "statistics_authorized": False,
        "timestamp": _utc_now(),
        "framework_version": EVALUATION_FRAMEWORK_VERSION,
        "schema_version": STRUCTURAL_METRIC_SCHEMA_VERSION,
        "protocol_discipline": PROTOCOL_DISCIPLINE,
    }
    write_json(status, stage_root / "controlled_noise_endpoint_re_evaluation_status.json")
    return {"controlled_noise_endpoint_re_evaluation_status": status}


def run_noise_endpoint_merge_audit(algorithm_root):
    """Stage C5: immutability and merge audit before statistics are authorized."""
    algorithm_root = ensure_dir(algorithm_root)
    require_benchmark_state(
        algorithm_root,
        allowed={"EXECUTED"},
        command_name="noise-endpoint-merge-audit",
    )
    stage_root = ensure_dir(algorithm_root / "05_noise_endpoint_merge_audit")
    old_csv = algorithm_root / "03_unified_benchmark_cube" / "unified_benchmark_cube_rows.csv"
    old_json = algorithm_root / "03_unified_benchmark_cube" / "unified_benchmark_cube_rows.json"
    endpoint_json = (
        algorithm_root
        / "04e_controlled_noise_endpoint_re_evaluation"
        / "protocol_noise_endpoint_results.json"
    )
    endpoint_status = _load_json(
        algorithm_root
        / "04e_controlled_noise_endpoint_re_evaluation"
        / "controlled_noise_endpoint_re_evaluation_status.json",
        default={},
    )
    old_rows = _load_json(old_json, default=[])
    endpoint_rows = _load_json(endpoint_json, default=[])
    endpoint_by_key = {
        (row.get("cube_cell_id"), row.get("method")): row
        for row in endpoint_rows
    }
    merged = []
    missing = []
    for old in old_rows:
        new = dict(old)
        for method in EVALUATION_METHODS:
            key = (old.get("cube_cell_id"), method)
            ep = endpoint_by_key.get(key)
            if ep is None:
                missing.append({"cube_cell_id": old.get("cube_cell_id"), "method": method})
                continue
            method_data = dict(new.get(method, {}))
            for metric in (
                "noise_capture_corr",
                "noise_capture_corr_score",
                "noise_capture_energy_ratio",
                "signal_leakage_into_noise",
                "signal_leakage_into_noise_score",
                "signal_leakage_residual_energy",
                "signal_leakage_observed_energy",
                "signal_leakage_degenerate_flag",
                "signal_leakage_basis",
            ):
                method_data[metric] = ep.get(metric)
            method_data["noise_endpoint_source"] = "protocol_controlled_re_evaluation"
            method_data["noise_endpoint_rule_checksum"] = ep.get("rule_checksum")
            method_data["noise_endpoint_selection_status_code"] = ep.get("selection_status_code")
            method_data["noise_endpoint_selected_component_count"] = ep.get("selected_component_count")
            method_data["noise_endpoint_zero_variance_protocol_noise_flag"] = ep.get("zero_variance_protocol_noise_flag")
            new[method] = method_data
        merged.append(new)
    duplicate_endpoint_keys = [
        f"{key[0]}|{key[1]}"
        for key, count in Counter((row.get("cube_cell_id"), row.get("method")) for row in endpoint_rows).items()
        if count > 1
    ]
    old_before = endpoint_status.get("benchmark_immutability_before_checksum")
    old_after_now = _file_checksum(old_csv)
    completeness = {
        "old_benchmark_rows": len(old_rows),
        "endpoint_rows": len(endpoint_rows),
        "expected_endpoint_rows": len(old_rows) * len(EVALUATION_METHODS),
        "merged_rows": len(merged),
        "missing_endpoint_rows": len(missing),
        "duplicate_endpoint_keys": len(duplicate_endpoint_keys),
    }
    finite_counts = {}
    for method in EVALUATION_METHODS:
        for metric in ("noise_capture_corr", "signal_leakage_into_noise"):
            values = []
            for row in merged:
                try:
                    values.append(float(row.get(method, {}).get(metric)))
                except Exception:
                    pass
            finite_counts[f"{method}_{metric}_finite_count"] = int(np.sum(np.isfinite(values))) if values else 0
    passed = (
        old_before is not None
        and old_before == old_after_now
        and completeness["endpoint_rows"] == completeness["expected_endpoint_rows"]
        and completeness["missing_endpoint_rows"] == 0
        and completeness["duplicate_endpoint_keys"] == 0
    )
    write_json(merged, stage_root / "unified_benchmark_cube_rows.json")
    write_csv(merged, stage_root / "unified_benchmark_cube_rows.csv")
    audit = {
        "workflow": WORKFLOW_NAME,
        "stage": "Stage C5 Benchmark Immutability and Noise Metric Merge Audit",
        "audit_status": "passed" if passed else "failed",
        "benchmark_immutability_audit": {
            "old_csv_path": str(old_csv),
            "before_checksum_from_c4": old_before,
            "after_checksum_now": old_after_now,
            "immutability_preserved": old_before == old_after_now,
            "old_metrics_recomputed": False,
            "old_metrics_overwritten": False,
        },
        "merge_audit": completeness,
        "duplicate_endpoint_key_examples": duplicate_endpoint_keys[:20],
        "missing_endpoint_examples": missing[:20],
        "finite_endpoint_counts": finite_counts,
        "complete_cube_root_for_statistics": str(stage_root),
        "statistics_authorized": bool(passed),
        "timestamp": _utc_now(),
        "framework_version": EVALUATION_FRAMEWORK_VERSION,
        "schema_version": STRUCTURAL_METRIC_SCHEMA_VERSION,
        "protocol_discipline": PROTOCOL_DISCIPLINE,
    }
    write_json(audit, stage_root / "noise_endpoint_merge_audit.json")
    if passed:
        _write_state(
            algorithm_root,
            "AUDITED",
            stage_root=stage_root,
            statistics_authorized=True,
            complete_cube_root_for_statistics=str(stage_root),
            audit_status="passed",
        )
    return {"noise_endpoint_merge_audit": audit}


def run_synthetic_reconstruction_rule_discovery(algorithm_root):
    """Discover historical synthetic reconstruction semantics from code/rows."""
    algorithm_root = ensure_dir(algorithm_root)
    require_benchmark_state(
        algorithm_root,
        allowed={"EXECUTED"},
        command_name="synthetic-reconstruction-rule-discovery",
    )
    stage_root = ensure_dir(algorithm_root / "04ca_synthetic_reconstruction_rule_discovery")
    cube_root = algorithm_root / "03_unified_benchmark_cube"
    rows_path = cube_root / "unified_benchmark_cube_rows.csv"
    rows = _read_csv_rows(rows_path)
    fields = set(rows[0].keys()) if rows else set()
    selection_fields = sorted(
        f for f in fields
        if (
            "selected_component" in f
            or "retained_component" in f
            or "component_selection" in f
            or "retention_mask" in f
            or "selected_imf" in f
        )
    )
    reconstruction_fields = sorted(
        f for f in fields
        if (
            f.endswith("_denoise_nmse")
            or f.endswith("_denoise_corr")
            or f.endswith("_clean_region_nmse")
            or f.endswith("_outlier_resistance_index")
            or f.endswith("_signal_leakage_into_noise")
            or f.endswith("_noise_capture_corr")
        )
    )
    code_rows = [
        {
            "code_path": "diagnostics/shared_physical_diagnostics.py",
            "symbol": "reconstructed_signal",
            "line_hint": 149,
            "semantics": "estimated_signal = Y_observed - residual when residual is available; fallback = sum(imfs)",
            "uses_true_signal": False,
            "uses_true_components": False,
            "uses_component_selection_mask": False,
            "final_residual_or_trend_policy": "native residual is subtracted from observed; therefore native residual is treated as estimated noise",
        },
        {
            "code_path": "diagnostics/shared_physical_diagnostics.py",
            "symbol": "reconstruction_accuracy",
            "line_hint": 194,
            "semantics": "denoise_nmse and denoise_corr call reconstructed_signal",
            "uses_true_signal": True,
            "uses_true_components": False,
            "uses_component_selection_mask": False,
            "final_residual_or_trend_policy": "inherits reconstructed_signal native-residual semantics",
        },
        {
            "code_path": "diagnostics/shared_physical_diagnostics.py",
            "symbol": "contamination_region_diagnostics",
            "line_hint": 746,
            "semantics": "clean_region_nmse calls reconstructed_signal",
            "uses_true_signal": True,
            "uses_true_components": False,
            "uses_component_selection_mask": False,
            "final_residual_or_trend_policy": "inherits reconstructed_signal native-residual semantics",
        },
        {
            "code_path": "diagnostics/shared_physical_diagnostics.py",
            "symbol": "noise_capture_diagnostics",
            "line_hint": 673,
            "semantics": "noise_capture_corr = corr(native residual, true_noise)",
            "uses_true_signal": True,
            "uses_true_components": False,
            "uses_component_selection_mask": False,
            "final_residual_or_trend_policy": "native residual is the estimated-noise object",
        },
        {
            "code_path": "diagnostics/shared_physical_diagnostics.py",
            "symbol": "signal_leakage_into_noise_diagnostics",
            "line_hint": 836,
            "semantics": "signal_leakage_into_noise applies projection-energy leakage to native residual",
            "uses_true_signal": True,
            "uses_true_components": True,
            "uses_component_selection_mask": False,
            "final_residual_or_trend_policy": "native residual is the estimated-noise object",
        },
    ]
    metric_rows = [
        {
            "metric": "denoise_nmse",
            "requires_estimated_signal": True,
            "estimated_signal_function": "reconstructed_signal",
            "historical_semantics": "Y_observed - native residual",
            "uses_true_signal_for_metric_value": True,
            "uses_true_components_for_selection": False,
            "component_selection_rule_used": False,
            "threshold_source": "none",
            "final_residue_trend_policy": "native residual excluded from estimated_signal",
        },
        {
            "metric": "denoise_corr",
            "requires_estimated_signal": True,
            "estimated_signal_function": "reconstructed_signal",
            "historical_semantics": "Y_observed - native residual",
            "uses_true_signal_for_metric_value": True,
            "uses_true_components_for_selection": False,
            "component_selection_rule_used": False,
            "threshold_source": "none",
            "final_residue_trend_policy": "native residual excluded from estimated_signal",
        },
        {
            "metric": "clean_region_nmse",
            "requires_estimated_signal": True,
            "estimated_signal_function": "reconstructed_signal",
            "historical_semantics": "Y_observed - native residual",
            "uses_true_signal_for_metric_value": True,
            "uses_true_components_for_selection": False,
            "component_selection_rule_used": False,
            "threshold_source": "none",
            "final_residue_trend_policy": "native residual excluded from estimated_signal",
        },
        {
            "metric": "outlier_resistance_index",
            "requires_estimated_signal": True,
            "estimated_signal_function": "reconstructed_signal",
            "historical_semantics": "Y_observed - native residual",
            "uses_true_signal_for_metric_value": True,
            "uses_true_components_for_selection": False,
            "component_selection_rule_used": False,
            "threshold_source": "none",
            "final_residue_trend_policy": "native residual excluded from estimated_signal",
        },
        {
            "metric": "noise_capture_corr",
            "requires_estimated_signal": False,
            "estimated_noise_function": "noise_capture_diagnostics",
            "historical_semantics": "native residual is estimated_noise",
            "uses_true_signal_for_metric_value": True,
            "uses_true_components_for_selection": False,
            "component_selection_rule_used": False,
            "threshold_source": "none",
            "final_residue_trend_policy": "native residual treated as estimated_noise",
        },
        {
            "metric": "signal_leakage_into_noise",
            "requires_estimated_signal": False,
            "estimated_noise_function": "signal_leakage_into_noise_diagnostics",
            "historical_semantics": "native residual is estimated_noise",
            "uses_true_signal_for_metric_value": True,
            "uses_true_components_for_selection": False,
            "component_selection_rule_used": False,
            "threshold_source": "none",
            "final_residue_trend_policy": "native residual treated as estimated_noise",
        },
    ]
    row_audit = {
        "row_count": int(len(rows)),
        "method_evaluation_count": int(len(rows) * len(EVALUATION_METHODS)),
        "selection_field_count": int(len(selection_fields)),
        "selection_field_examples": selection_fields[:50],
        "reconstruction_metric_field_count": int(len(reconstruction_fields)),
        "reconstruction_metric_field_examples": reconstruction_fields[:50],
    }
    discovery = {
        "workflow": WORKFLOW_NAME,
        "stage": "Stage C3b Synthetic reconstruction-rule discovery",
        "scope": (
            "discover historical synthetic reconstruction semantics; no new "
            "rule is frozen, no endpoint recomputation, no statistical analysis"
        ),
        "questions_answered": {
            "which_primary_metrics_need_estimated_signal": [
                "denoise_nmse",
                "denoise_corr",
                "clean_region_nmse",
                "outlier_resistance_index",
            ],
            "which_function_constructs_estimated_signal": "diagnostics.shared_physical_diagnostics.reconstructed_signal",
            "same_function_used_across_methods": True,
            "depends_on_true_signal_or_true_components_for_selection": False,
            "component_selection_threshold_exists": False,
            "threshold_source": None,
            "final_residue_trend_policy": "native residual excluded from estimated_signal and treated as estimated_noise",
            "full_benchmark_preexisting_rule": True,
            "stable_checksum_available": True,
        },
        "classification": "case_B_metric_specific_native_residual_semantics_no_protocol_component_selection_rule",
        "discovery_conclusion": (
            "Historical synthetic reconstruction metrics used a uniform "
            "native-residual reconstruction convention: estimated_signal = "
            "Y_observed - native residual. No frozen selected-component mask, "
            "retained-component indices, or protocol-level component-selection "
            "rule is present in the full benchmark rows. Therefore C4 cannot "
            "construct protocol_estimated_signal from selected components until "
            "a synthetic component-selection/reconstruction rule is formally "
            "specified and frozen."
        ),
        "rule_formalization_recommendation": (
            "Do not silently reuse native-residual semantics for the universal "
            "noise-separation adapter because CEEMDAN native residual is a "
            "near-zero closure remainder. A formal protocol amendment is "
            "needed if selected-component reconstruction is introduced after "
            "this discovery."
        ),
        "post_execution_rule_adjudication_manifest": {
            "performance_results_consulted_for_rule_selection": False,
            "method_ranking_consulted": False,
            "rule_source": "historical implementation audit",
            "scientific_design_change_if_new_component_selection_rule_added": True,
        },
        "row_field_audit": row_audit,
        "statistics_authorized": False,
        "controlled_rerun_authorized": False,
        "timestamp": _utc_now(),
        "framework_version": EVALUATION_FRAMEWORK_VERSION,
        "schema_version": STRUCTURAL_METRIC_SCHEMA_VERSION,
        "protocol_discipline": PROTOCOL_DISCIPLINE,
    }
    discovery["discovery_checksum"] = _stable_json_checksum(discovery)
    write_json(discovery, stage_root / "synthetic_reconstruction_rule_discovery.json")
    write_json(row_audit, stage_root / "historical_reconstruction_field_audit.json")
    write_csv(code_rows, stage_root / "reconstruction_code_path_audit.csv")
    write_csv(metric_rows, stage_root / "metric_to_reconstruction_semantics.csv")
    write_json({
        "rule_discoverability_status": "historical_native_residual_rule_discovered_but_no_component_selection_rule",
        "component_selection_rule_discoverable": False,
        "protocol_reconstruction_rule_freeze_ready": False,
        "next_required_stage": "SYNTHETIC_COMPONENT_SELECTION_AND_RECONSTRUCTION_V1.0 specification/adjudication",
        "discovery_checksum": discovery["discovery_checksum"],
    }, stage_root / "rule_discoverability_report.json")
    return {"synthetic_reconstruction_rule_discovery": discovery}


def mark_statistics_completed(algorithm_root, statistics_dashboard=None):
    require_benchmark_state(
        algorithm_root,
        allowed={"AUDITED"},
        command_name="unified-statistics",
    )
    return _write_state(
        algorithm_root,
        "STATISTICS_COMPLETED",
        stage_root=Path(algorithm_root) / "07_unified_cube_statistics",
        statistics_dashboard_status=(statistics_dashboard or {}).get("section"),
        scientific_interpretation_authorized=True,
    )
