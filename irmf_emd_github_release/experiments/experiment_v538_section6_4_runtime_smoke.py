#!/usr/bin/python
# coding: UTF-8

"""V5.38 lightweight smoke for Section 6.4 runtime/scaling instrumentation."""

import platform
import sys
from datetime import datetime, timezone
from time import perf_counter

import numpy as np

from project_config import (
    DEFAULT_FS,
    DEFAULT_METHOD_TIMEOUT_SECONDS,
    GLOBAL_CEEMDAN_PARAMS,
    GLOBAL_EEMD_PARAMS,
    GLOBAL_EMD_PARAMS,
    GLOBAL_IRMF_PARAMS,
)
from experiments.experiment_emd_family_benchmark import _run_locked_emd_family_method
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


V538_SECTION6_4_SMOKE_VERSION = "V5.38_section6_4_runtime_instrumentation_smoke"
METHODS = ("IRMF", "EMD", "EEMD", "CEEMDAN")


def _environment_manifest():
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": None,
        "numpy_version": np.__version__,
        "parallelization_policy": "method calls executed serially in smoke instrumentation",
    }


def _run_method(method, case, fs, algorithm_seed, timeout_seconds):
    start = perf_counter()
    try:
        if method == "IRMF":
            result = _run_with_timeout(
                lambda: run_fixed_irmf_case(
                    Y=case["Y"],
                    X_clean=case["X_clean"],
                    t=case["t"],
                    fs=fs,
                    irmf_params=GLOBAL_IRMF_PARAMS,
                    expected_noise_ratio=case.get("expected_noise_ratio"),
                    true_components=case.get("true_components"),
                    contamination_mask=case.get("contamination_mask"),
                    run_id=f"v538_runtime_{method.lower()}",
                ),
                timeout_seconds=timeout_seconds,
            )
        elif method == "EMD":
            result = _run_with_timeout(
                lambda: run_fixed_emd_case(
                    Y=case["Y"],
                    X_clean=case["X_clean"],
                    t=case["t"],
                    fs=fs,
                    emd_params=GLOBAL_EMD_PARAMS,
                    true_components=case.get("true_components"),
                    contamination_mask=case.get("contamination_mask"),
                    run_id=f"v538_runtime_{method.lower()}",
                ),
                timeout_seconds=timeout_seconds,
            )
        else:
            seed = int(algorithm_seed) if method == "EEMD" else int(algorithm_seed) + 50000
            result = _run_with_timeout(
                lambda: _run_locked_emd_family_method(
                    method_name=method,
                    case=case,
                    fs=fs,
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


def _status(rows, expected, environment, started, finished):
    n_completed = sum(1 for row in rows if row.get("status") == "completed")
    n_failed = sum(1 for row in rows if row.get("status") == "failed")
    required_env = ("python_version", "platform", "numpy_version", "parallelization_policy")
    missing_env = [k for k in required_env if not environment.get(k)]
    runtime_missing = [
        row for row in rows
        if row.get("runtime_seconds") is None or not np.isfinite(float(row.get("runtime_seconds")))
    ]
    passed = (
        expected == n_completed + n_failed
        and n_failed == 0
        and not missing_env
        and not runtime_missing
    )
    return {
        "module_status": "smoke_passed_instrumentation_integrity" if passed else "smoke_failed_or_requires_attention",
        "protocol_id": V538_SECTION6_4_SMOKE_VERSION,
        "upstream_protocol": "V5.34_section6_executable_protocols",
        "protocol_frozen": True,
        "execution_scope": "runtime_scaling_lightweight_instrumentation_smoke_only",
        "execution_complete": bool(expected == n_completed + n_failed),
        "n_expected_cells": int(expected),
        "n_completed_cells": int(n_completed),
        "n_failed_cells": int(n_failed),
        "statistics_complete": False,
        "claim_authorized": False,
        "claim_boundary": (
            "This smoke validates runtime instrumentation and environment "
            "recording only. It does not authorize computational-scaling claims."
        ),
        "blocking_issues_present": not passed,
        "blocking_issues": {
            "failed_method_evaluations": int(n_failed),
            "missing_environment_fields": missing_env,
            "missing_runtime_rows": int(len(runtime_missing)),
        },
        "started_timestamp_utc": started,
        "finished_timestamp_utc": finished,
    }


def run_v538_section6_4_runtime_instrumentation_smoke(
        output_root,
        n_grid=(250, 500),
        methods=METHODS,
        signal_name="stationary_multi_sine",
        noise_name="gaussian",
        target_snr_db=15.0,
        seed=0,
        fs=DEFAULT_FS,
        timeout_seconds=DEFAULT_METHOD_TIMEOUT_SECONDS,
):
    output_root = ensure_dir(output_root)
    started = datetime.now(timezone.utc).isoformat()
    environment = _environment_manifest()
    rows = []
    for n in n_grid:
        sigma = 10.0 ** (-float(target_snr_db) / 20.0)
        case = make_signal_noise_case(
            signal_name=signal_name,
            noise_name=noise_name,
            sigma=sigma,
            n=int(n),
            fs=fs,
            seed=seed,
            target_snr_db=target_snr_db,
        )
        case["seed"] = int(seed)
        for method in methods:
            result, runtime, error = _run_method(
                method=method,
                case=case,
                fs=fs,
                algorithm_seed=100000 + int(seed),
                timeout_seconds=timeout_seconds,
            )
            row = {
                "n": int(n),
                "fs": float(fs),
                "signal": signal_name,
                "noise": noise_name,
                "target_snr_db": float(target_snr_db),
                "seed": int(seed),
                "method": method,
                "runtime_seconds": runtime,
                "timeout_seconds": float(timeout_seconds),
                "status": "failed" if error else "completed",
                "error": str(error) if error else None,
                "timing_policy": "decomposition_plus_metric_computation",
                "method_evaluation_unit": "single synthetic signal/noise/SNR/seed/method run",
            }
            row.update(method_result_summary(result))
            rows.append(row)
    finished = datetime.now(timezone.utc).isoformat()
    expected = int(len(tuple(n_grid)) * len(tuple(methods)))
    status = _status(rows, expected, environment, started, finished)
    runtime_summary = []
    for method in methods:
        vals = [float(r["runtime_seconds"]) for r in rows if r["method"] == method and r["status"] == "completed"]
        runtime_summary.append({
            "method": method,
            "n_completed": int(len(vals)),
            "median_runtime_seconds": float(np.median(vals)) if vals else None,
            "max_runtime_seconds": float(np.max(vals)) if vals else None,
        })
    protocol = {
        "schema_version": V538_SECTION6_4_SMOKE_VERSION,
        "protocol_status": "smoke_execution_completed",
        "purpose": "Validate runtime/scaling instrumentation before full Section 6.4 execution.",
        "not_a_scientific_claim": True,
        "n_grid": list(n_grid),
        "methods": list(methods),
        "signal": signal_name,
        "noise": noise_name,
        "target_snr_db": float(target_snr_db),
        "seed": int(seed),
        "expected_cell_formula": "n_grid x methods",
        "n_expected_cells": expected,
        "status_artifact": status,
    }
    write_json(protocol, output_root / "v538_section6_4_smoke_protocol.json")
    write_json(status, output_root / "v538_section6_4_smoke_status.json")
    write_json(environment, output_root / "v538_section6_4_environment_manifest.json")
    write_json(rows, output_root / "v538_section6_4_smoke_rows.json")
    write_csv(rows, output_root / "v538_section6_4_smoke_rows.csv")
    write_csv(runtime_summary, output_root / "v538_section6_4_smoke_runtime_summary.csv")
    return {
        "protocol": protocol,
        "status": status,
        "environment": environment,
        "rows": rows,
        "runtime_summary": runtime_summary,
    }
