#!/usr/bin/python
# coding: UTF-8

"""V5.37 smoke execution for Section 6.3 comparator and signal robustness."""

from datetime import datetime, timezone
from time import perf_counter

import numpy as np

from project_config import (
    DEFAULT_FS,
    DEFAULT_METHOD_TIMEOUT_SECONDS,
    DEFAULT_N,
    GLOBAL_CEEMDAN_PARAMS,
    GLOBAL_EEMD_PARAMS,
    GLOBAL_EMD_PARAMS,
    GLOBAL_IRMF_PARAMS,
    LOWER_IS_BETTER_METRICS,
    V528_PRIMARY_ENDPOINTS_BY_DIMENSION,
    V528_PRIMARY_METRIC_CODE_FIELD_MAP,
)
from experiments.experiment_emd_family_benchmark import _run_locked_emd_family_method
from experiments.experiment_signal_variant_robustness import _variant_family
from experiments.experiment_utils import (
    _failure_summary_for_exception,
    _run_with_timeout,
    attach_failure_flags,
    make_signal_noise_case,
    method_result_summary,
    run_fixed_emd_case,
    run_fixed_irmf_case,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


V537_SECTION6_3_SMOKE_VERSION = "V5.37_section6_3_comparator_signal_smoke"
METHODS = ("IRMF", "EMD", "EEMD", "CEEMDAN")


def _primary_metrics():
    out = []
    for metrics in V528_PRIMARY_ENDPOINTS_BY_DIMENSION.values():
        out.extend(metrics)
    return tuple(out)


PRIMARY_METRICS = _primary_metrics()
LOWER_IS_BETTER = set(LOWER_IS_BETTER_METRICS)


def _metric_code_field(metric):
    return V528_PRIMARY_METRIC_CODE_FIELD_MAP.get(metric, metric)


def _finite(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return None


def _case(signal_name, noise_name, target_snr_db, seed, n, fs):
    sigma = 10.0 ** (-float(target_snr_db) / 20.0)
    case = make_signal_noise_case(
        signal_name=signal_name,
        noise_name=noise_name,
        sigma=sigma,
        n=n,
        fs=fs,
        seed=seed,
        target_snr_db=target_snr_db,
    )
    case["seed"] = int(seed)
    return case


def _run_locked_method(method, case, algorithm_seed, timeout_seconds):
    start = perf_counter()
    try:
        if method == "IRMF":
            result = _run_with_timeout(
                lambda: run_fixed_irmf_case(
                    Y=case["Y"],
                    X_clean=case["X_clean"],
                    t=case["t"],
                    fs=DEFAULT_FS,
                    irmf_params=GLOBAL_IRMF_PARAMS,
                    expected_noise_ratio=case.get("expected_noise_ratio"),
                    true_components=case.get("true_components"),
                    contamination_mask=case.get("contamination_mask"),
                    run_id=f"v537_signal_variant_{method.lower()}",
                ),
                timeout_seconds=timeout_seconds,
            )
        elif method == "EMD":
            result = _run_with_timeout(
                lambda: run_fixed_emd_case(
                    Y=case["Y"],
                    X_clean=case["X_clean"],
                    t=case["t"],
                    fs=DEFAULT_FS,
                    emd_params=GLOBAL_EMD_PARAMS,
                    true_components=case.get("true_components"),
                    contamination_mask=case.get("contamination_mask"),
                    run_id=f"v537_signal_variant_{method.lower()}",
                ),
                timeout_seconds=timeout_seconds,
            )
        else:
            seed = int(algorithm_seed) if method == "EEMD" else int(algorithm_seed) + 50000
            result = _run_with_timeout(
                lambda: _run_locked_emd_family_method(
                    method_name=method,
                    case=case,
                    fs=DEFAULT_FS,
                    emd_params=GLOBAL_EMD_PARAMS,
                    eemd_params=GLOBAL_EEMD_PARAMS,
                    ceemdan_params=GLOBAL_CEEMDAN_PARAMS,
                    algorithm_seed=seed,
                ),
                timeout_seconds=timeout_seconds,
            )
        runtime = float(perf_counter() - start)
        return attach_failure_flags(result, case["Y"], case["X_clean"], timeout_seconds, runtime), runtime, None
    except Exception as exc:
        runtime = float(perf_counter() - start)
        return _failure_summary_for_exception(method, exc, runtime, timeout_seconds), runtime, exc


def _run_baseline_alternative(method, case, algorithm_seed, timeout_seconds):
    start = perf_counter()
    try:
        if method == "EMD":
            alt_params = {**GLOBAL_EMD_PARAMS, "nbsym": 4}
            result = _run_with_timeout(
                lambda: run_fixed_emd_case(
                    Y=case["Y"],
                    X_clean=case["X_clean"],
                    t=case["t"],
                    fs=DEFAULT_FS,
                    emd_params=alt_params,
                    true_components=case.get("true_components"),
                    contamination_mask=case.get("contamination_mask"),
                    run_id="v537_baseline_alt_emd_nbsym4",
                ),
                timeout_seconds=timeout_seconds,
            )
            config = {"nbsym": 4}
        elif method == "EEMD":
            alt_params = {**GLOBAL_EEMD_PARAMS, "noise_width": 0.03}
            result = _run_with_timeout(
                lambda: _run_locked_emd_family_method(
                    method_name="EEMD",
                    case=case,
                    fs=DEFAULT_FS,
                    emd_params=GLOBAL_EMD_PARAMS,
                    eemd_params=alt_params,
                    ceemdan_params=GLOBAL_CEEMDAN_PARAMS,
                    algorithm_seed=int(algorithm_seed),
                ),
                timeout_seconds=timeout_seconds,
            )
            config = {"noise_width": 0.03}
        elif method == "CEEMDAN":
            alt_params = {**GLOBAL_CEEMDAN_PARAMS, "epsilon": 0.003}
            result = _run_with_timeout(
                lambda: _run_locked_emd_family_method(
                    method_name="CEEMDAN",
                    case=case,
                    fs=DEFAULT_FS,
                    emd_params=GLOBAL_EMD_PARAMS,
                    eemd_params=GLOBAL_EEMD_PARAMS,
                    ceemdan_params=alt_params,
                    algorithm_seed=int(algorithm_seed) + 50000,
                ),
                timeout_seconds=timeout_seconds,
            )
            config = {"epsilon": 0.003}
        else:
            raise ValueError(f"unsupported baseline alternative: {method}")
        runtime = float(perf_counter() - start)
        return attach_failure_flags(result, case["Y"], case["X_clean"], timeout_seconds, runtime), runtime, config, None
    except Exception as exc:
        runtime = float(perf_counter() - start)
        return _failure_summary_for_exception(method, exc, runtime, timeout_seconds), runtime, {}, exc


def _result_row(scope, signal_name, case, method, result, runtime, error, **extra):
    row = {
        "scope": scope,
        "signal": signal_name,
        "variant_family": _variant_family(signal_name),
        "noise": case["noise_name"],
        "target_snr_db": case["target_snr_db"],
        "seed": case["seed"],
        "method": method,
        "runtime_seconds": runtime,
        "status": "failed" if error else "completed",
        "error": str(error) if error else None,
        "noise_design": case.get("noise_design"),
        "realized_input_snr_db": case.get("realized_input_snr_db"),
    }
    row.update(extra)
    row.update(method_result_summary(result))
    for metric in PRIMARY_METRICS:
        row[metric] = result.get(_metric_code_field(metric))
    return row


def _metric_completeness(rows):
    out = []
    completed = [r for r in rows if r.get("status") == "completed"]
    for metric in PRIMARY_METRICS:
        finite = 0
        nonfinite = 0
        missing = 0
        for row in completed:
            if metric not in row:
                missing += 1
            elif _finite(row.get(metric)) is None:
                nonfinite += 1
            else:
                finite += 1
        out.append({
            "metric": metric,
            "code_field": _metric_code_field(metric),
            "higher_is_better": metric not in LOWER_IS_BETTER,
            "n_completed_rows": len(completed),
            "n_finite": finite,
            "n_nonfinite": nonfinite,
            "n_missing_field": missing,
        })
    return out


def _status(rows, expected, started, finished):
    n_completed = sum(1 for r in rows if r.get("status") == "completed")
    n_failed = sum(1 for r in rows if r.get("status") == "failed")
    metric_rows = _metric_completeness(rows)
    missing = [r for r in metric_rows if int(r["n_missing_field"]) > 0]
    variant_rows = [r for r in rows if r.get("scope") == "signal_variant_locked_methods"]
    alt_rows = [r for r in rows if r.get("scope") == "baseline_alternative"]
    baseline_violation = [
        r for r in alt_rows
        if r.get("alternative_predeclared") != True or r.get("retuning_policy") != "predeclared_no_oracle_retuning"
    ]
    passed = (
        expected == n_completed + n_failed
        and n_failed == 0
        and not missing
        and len(variant_rows) > 0
        and len(alt_rows) > 0
        and not baseline_violation
    )
    return {
        "module_status": "smoke_passed_execution_integrity" if passed else "smoke_failed_or_requires_attention",
        "protocol_id": V537_SECTION6_3_SMOKE_VERSION,
        "upstream_protocol": "V5.34_section6_executable_protocols",
        "protocol_frozen": True,
        "execution_scope": "comparator_signal_targeted_smoke_execution_integrity_only",
        "execution_complete": bool(expected == n_completed + n_failed),
        "n_expected_cells": int(expected),
        "n_completed_cells": int(n_completed),
        "n_failed_cells": int(n_failed),
        "statistics_complete": False,
        "claim_authorized": False,
        "claim_boundary": (
            "This smoke validates signal-variant and comparator-alternative "
            "execution integrity only. It does not authorize robustness claims."
        ),
        "blocking_issues_present": not passed,
        "blocking_issues": {
            "failed_method_evaluations": int(n_failed),
            "missing_primary_metric_fields": int(len(missing)),
            "baseline_retuning_policy_violations": int(len(baseline_violation)),
            "signal_variant_rows": int(len(variant_rows)),
            "baseline_alternative_rows": int(len(alt_rows)),
        },
        "started_timestamp_utc": started,
        "finished_timestamp_utc": finished,
    }


def run_v537_section6_3_comparator_signal_smoke(
        output_root,
        variant_signals=("stationary_multi_sine_wide", "chirp_fast"),
        noise_name="gaussian",
        target_snr_db=15.0,
        seed=0,
        n=DEFAULT_N,
        fs=DEFAULT_FS,
        timeout_seconds=DEFAULT_METHOD_TIMEOUT_SECONDS,
):
    output_root = ensure_dir(output_root)
    started = datetime.now(timezone.utc).isoformat()
    rows = []
    algorithm_seed = 100000 + int(seed)
    for signal_name in variant_signals:
        case = _case(signal_name, noise_name, target_snr_db, seed, n, fs)
        for method in METHODS:
            result, runtime, error = _run_locked_method(method, case, algorithm_seed, timeout_seconds)
            rows.append(_result_row("signal_variant_locked_methods", signal_name, case, method, result, runtime, error))

    alt_case_signal = variant_signals[0]
    alt_case = _case(alt_case_signal, noise_name, target_snr_db, seed, n, fs)
    for method in ("EMD", "EEMD", "CEEMDAN"):
        result, runtime, config, error = _run_baseline_alternative(method, alt_case, algorithm_seed, timeout_seconds)
        rows.append(_result_row(
            "baseline_alternative",
            alt_case_signal,
            alt_case,
            method,
            result,
            runtime,
            error,
            alternative_config=config,
            alternative_predeclared=True,
            retuning_policy="predeclared_no_oracle_retuning",
        ))
    finished = datetime.now(timezone.utc).isoformat()
    expected = int(len(variant_signals) * len(METHODS) + 3)
    status = _status(rows, expected, started, finished)
    metric_rows = _metric_completeness(rows)
    protocol = {
        "schema_version": V537_SECTION6_3_SMOKE_VERSION,
        "protocol_status": "smoke_execution_completed",
        "purpose": "Validate execution integrity for Section 6.3 before full execution.",
        "not_a_scientific_claim": True,
        "variant_signals": list(variant_signals),
        "noise": noise_name,
        "target_snr_db": float(target_snr_db),
        "seed": int(seed),
        "n": int(n),
        "fs": float(fs),
        "methods": list(METHODS),
        "baseline_alternatives": {
            "EMD": {"nbsym": 4},
            "EEMD": {"noise_width": 0.03},
            "CEEMDAN": {"epsilon": 0.003},
        },
        "expected_cell_formula": "n_variant_signals x n_methods + n_baseline_alternatives",
        "n_expected_cells": expected,
        "status_artifact": status,
    }
    write_json(protocol, output_root / "v537_section6_3_smoke_protocol.json")
    write_json(status, output_root / "v537_section6_3_smoke_status.json")
    write_json(rows, output_root / "v537_section6_3_smoke_rows.json")
    write_csv(rows, output_root / "v537_section6_3_smoke_rows.csv")
    write_csv(metric_rows, output_root / "v537_section6_3_smoke_metric_completeness.csv")
    return {"protocol": protocol, "status": status, "rows": rows, "metric_completeness": metric_rows}
