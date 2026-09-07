#!/usr/bin/python
# coding: UTF-8

"""V5.52 full Section 6.4 computational efficiency and scaling."""

import platform
import sys
from collections import defaultdict
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
    V557_SECTION6_4_SCALING_SIGNALS,
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


V552_SECTION6_4_VERSION = "V5.52_section6_4_computational_efficiency_scaling"
METHODS = ("IRMF", "EMD", "EEMD", "CEEMDAN")
ACCURACY_CONTEXT_METRICS = (
    "denoise_nmse",
    "denoise_corr",
    "matched_component_corr",
    "matched_component_nrmse",
    "relative_decomposition_count_error",
    "imf_count",
    "effective_imf_count",
    "computational_failure",
)
ACCURACY_CONTEXT_ALIASES = {
    "matched_component_corr": "imf_recovery_corr",
    "matched_component_nrmse": "imf_recovery_nrmse",
}


def _environment_manifest():
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy_version": np.__version__,
        "parallelization_policy": "serial outer-loop timing",
        "timing_policy": "outer perf_counter around decomposition plus metric computation",
    }


def _run_method(method, case, algorithm_seed, timeout_seconds):
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
                    run_id=f"v552_runtime_{method.lower()}",
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
                    run_id=f"v552_runtime_{method.lower()}",
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


def _runtime_summaries(rows):
    by_method_n = defaultdict(list)
    by_method = defaultdict(list)
    failures = defaultdict(int)
    for row in rows:
        if row.get("status") != "completed":
            failures[(row["method"], row["n"])] += 1
            continue
        val = float(row["runtime_seconds"])
        by_method_n[(row["method"], row["n"])].append(val)
        by_method[row["method"]].append(val)
    by_n_rows = []
    for (method, n), vals in sorted(by_method_n.items(), key=lambda item: str(item[0])):
        arr = np.asarray(vals, dtype=float)
        by_n_rows.append({
            "method": method,
            "n": int(n),
            "n_completed": int(arr.size),
            "n_failed": int(failures.get((method, n), 0)),
            "median_runtime_seconds": float(np.median(arr)),
            "iqr_runtime_seconds": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            "p90_runtime_seconds": float(np.percentile(arr, 90)),
            "p95_runtime_seconds": float(np.percentile(arr, 95)),
        })
    by_method_rows = []
    for method, vals in sorted(by_method.items()):
        arr = np.asarray(vals, dtype=float)
        by_method_rows.append({
            "method": method,
            "n_completed": int(arr.size),
            "median_runtime_seconds": float(np.median(arr)),
            "iqr_runtime_seconds": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            "p90_runtime_seconds": float(np.percentile(arr, 90)),
            "p95_runtime_seconds": float(np.percentile(arr, 95)),
        })
    scaling = []
    for method in METHODS:
        points = [
            row for row in by_n_rows
            if row["method"] == method and row["median_runtime_seconds"] > 0
        ]
        if len(points) >= 2:
            x = np.log(np.asarray([p["n"] for p in points], dtype=float))
            y = np.log(np.asarray([p["median_runtime_seconds"] for p in points], dtype=float))
            beta, alpha = np.polyfit(x, y, deg=1)
            pred = alpha + beta * x
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
            scaling.append({
                "method": method,
                "empirical_scaling_exponent_beta": float(beta),
                "log_runtime_intercept_alpha": float(alpha),
                "r_squared_log_log_fit": float(r2) if np.isfinite(r2) else None,
                "n_points": int(len(points)),
                "claim_boundary": "empirical scaling exponent, not theoretical complexity proof",
            })
    return by_method_rows, by_n_rows, scaling


def _finite(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return None


def _metric_value(row, metric):
    source = ACCURACY_CONTEXT_ALIASES.get(metric, metric)
    return _finite(row.get(source))


def _accuracy_context_summaries(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("status") != "completed":
            continue
        for metric in ACCURACY_CONTEXT_METRICS:
            val = _metric_value(row, metric)
            if val is not None:
                grouped[(row["method"], int(row["n"]), metric)].append(val)
    out = []
    for (method, n, metric), vals in sorted(grouped.items(), key=lambda item: str(item[0])):
        arr = np.asarray(vals, dtype=float)
        out.append({
            "method": method,
            "n": int(n),
            "metric": metric,
            "source_field": ACCURACY_CONTEXT_ALIASES.get(metric, metric),
            "n_values": int(arr.size),
            "median": float(np.median(arr)),
            "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            "mean": float(np.mean(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
            "role": "accuracy_context_for_runtime_scaling_not_primary_section6_performance_claim",
        })
    return out


def _accuracy_runtime_tradeoff_rows(runtime_by_n, accuracy_context):
    runtime_lookup = {
        (row["method"], int(row["n"])): float(row["median_runtime_seconds"])
        for row in runtime_by_n
    }
    out = []
    for row in accuracy_context:
        key = (row["method"], int(row["n"]))
        runtime = runtime_lookup.get(key)
        if runtime is None:
            continue
        out.append({
            "method": row["method"],
            "n": int(row["n"]),
            "metric": row["metric"],
            "median_metric": row["median"],
            "median_runtime_seconds": runtime,
            "role": "paired_accuracy_runtime_context",
        })
    return out


def run_v552_section6_4_computational_scaling(
        output_root,
        n_grid=(250, 500, 1000, 2000, 4000),
        signals=V557_SECTION6_4_SCALING_SIGNALS,
        noise_name="gaussian",
        target_snr_db=15.0,
        seeds=(0, 1),
        timing_repeats=3,
        methods=METHODS,
        fs=DEFAULT_FS,
        timeout_seconds=DEFAULT_METHOD_TIMEOUT_SECONDS,
):
    output_root = ensure_dir(output_root)
    started = datetime.now(timezone.utc).isoformat()
    environment = _environment_manifest()
    rows = []
    for n in n_grid:
        for signal_name in signals:
            for seed in seeds:
                sigma = 10.0 ** (-float(target_snr_db) / 20.0)
                case = make_signal_noise_case(
                    signal_name=signal_name,
                    noise_name=noise_name,
                    sigma=sigma,
                    n=int(n),
                    fs=fs,
                    seed=int(seed),
                    target_snr_db=target_snr_db,
                )
                for method in methods:
                    for repeat_idx in range(int(timing_repeats)):
                        result, runtime, error = _run_method(
                            method,
                            case,
                            algorithm_seed=100000 + int(seed),
                            timeout_seconds=timeout_seconds,
                        )
                        row = {
                            "method": method,
                            "n": int(n),
                            "signal": signal_name,
                            "noise": noise_name,
                            "target_snr_db": float(target_snr_db),
                            "seed": int(seed),
                            "timing_repeat": int(repeat_idx),
                            "runtime_seconds": runtime,
                            "timeout_seconds": float(timeout_seconds),
                            "status": "failed" if error else "completed",
                            "error": str(error) if error else None,
                            "timing_policy": environment["timing_policy"],
                        }
                        row.update(method_result_summary(result))
                        row["result_method_label"] = row.get("method")
                        row["method"] = method
                        rows.append(row)
    by_method, by_n, scaling = _runtime_summaries(rows)
    accuracy_context = _accuracy_context_summaries(rows)
    accuracy_runtime = _accuracy_runtime_tradeoff_rows(by_n, accuracy_context)
    finished = datetime.now(timezone.utc).isoformat()
    n_failed = int(sum(1 for r in rows if r.get("status") == "failed"))
    dashboard = {
        "schema_version": V552_SECTION6_4_VERSION,
        "section": "6.4 Computational Efficiency and Scaling",
        "module_status": "execution_complete_pending_claim_qualification" if n_failed == 0 else "execution_complete_with_failures",
        "execution_complete": True,
        "statistics_complete": True,
        "claim_authorized": False,
        "n_method_evaluations": int(len(rows)),
        "n_failed": n_failed,
        "n_grid": list(n_grid),
        "signals": list(signals),
        "seeds": list(seeds),
        "timing_repeats": int(timing_repeats),
        "methods": list(methods),
        "target_snr_db": float(target_snr_db),
        "claim_boundary": (
            "Runtime results describe empirical wall-clock cost under the "
            "recorded environment and do not prove theoretical complexity. "
            "Accuracy metrics recorded across n are context diagnostics for "
            "sample-resolution performance preservation, not new primary "
            "performance endpoints."
        ),
        "started_timestamp_utc": started,
        "finished_timestamp_utc": finished,
    }
    protocol = {
        "schema_version": V552_SECTION6_4_VERSION,
        "runtime_design": "fixed_target_snr_outer_wrapper_timing",
        "n_grid": list(n_grid),
        "signals": list(signals),
        "noise": noise_name,
        "target_snr_db": float(target_snr_db),
        "seeds": list(seeds),
        "timing_repeats": int(timing_repeats),
        "methods": list(methods),
        "empirical_scaling_model": "log(runtime) = alpha + beta log(n) + epsilon",
        "accuracy_context_metrics": list(ACCURACY_CONTEXT_METRICS),
        "accuracy_context_aliases": dict(ACCURACY_CONTEXT_ALIASES),
    }
    write_json(protocol, output_root / "v552_section6_4_runtime_scaling_protocol.json")
    write_json(dashboard, output_root / "v552_section6_4_runtime_scaling_dashboard.json")
    write_json(environment, output_root / "v552_section6_4_environment_manifest.json")
    write_csv(rows, output_root / "v552_section6_4_runtime_rows.csv")
    write_csv(by_method, output_root / "v552_runtime_by_method.csv")
    write_csv(by_n, output_root / "v552_runtime_by_method_and_n.csv")
    write_csv(scaling, output_root / "v552_runtime_scaling_exponents.csv")
    write_csv(accuracy_context, output_root / "v552_accuracy_context_by_method_and_n.csv")
    write_csv(accuracy_runtime, output_root / "v552_accuracy_runtime_tradeoff_context.csv")
    return {
        "protocol": protocol,
        "dashboard": dashboard,
        "environment": environment,
        "rows": rows,
        "runtime_by_method": by_method,
        "runtime_by_method_and_n": by_n,
        "scaling": scaling,
        "accuracy_context": accuracy_context,
        "accuracy_runtime_tradeoff": accuracy_runtime,
    }
