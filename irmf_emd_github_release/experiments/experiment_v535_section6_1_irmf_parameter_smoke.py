#!/usr/bin/python
# coding: UTF-8

"""V5.35 smoke execution for Section 6.1 IRMF parameter sensitivity."""

from datetime import datetime, timezone
from time import perf_counter

import numpy as np

from project_config import (
    DEFAULT_FS,
    DEFAULT_N,
    GLOBAL_IRMF_PARAMS,
    IRMF_PARAMETER_SENSITIVITY_OAT_FACTORS,
    IRMF_PARAMETER_SENSITIVITY_PARAMETERS,
    IRMF_PARAMETER_SENSITIVITY_JOINT_INTERACTIONS,
    LOWER_IS_BETTER_METRICS,
    V528_PRIMARY_ENDPOINTS_BY_DIMENSION,
    V528_PRIMARY_METRIC_CODE_FIELD_MAP,
)
from experiments.experiment_utils import (
    make_signal_noise_case,
    method_result_summary,
    run_fixed_irmf_case,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


V535_SECTION6_1_SMOKE_VERSION = "V5.35_section6_1_irmf_parameter_smoke"


def _primary_metrics():
    out = []
    for metrics in V528_PRIMARY_ENDPOINTS_BY_DIMENSION.values():
        out.extend(metrics)
    return tuple(out)


PRIMARY_METRICS = _primary_metrics()
LOWER_IS_BETTER = set(LOWER_IS_BETTER_METRICS)


def _metric_code_field(metric):
    return V528_PRIMARY_METRIC_CODE_FIELD_MAP.get(metric, metric)


def _value(result, metric):
    return result.get(_metric_code_field(metric))


def _finite(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return None


def _scaled_params(base, updates):
    params = dict(base)
    parameter_names = tuple(IRMF_PARAMETER_SENSITIVITY_PARAMETERS)
    factors = {name: 1.0 for name in parameter_names}
    factors.update(updates)
    for name, factor in factors.items():
        params[name] = float(base[name]) * float(factor)
    return params, factors


def _parameter_configurations(irmf_params):
    configs = []
    parameter_names = tuple(IRMF_PARAMETER_SENSITIVITY_PARAMETERS)
    base = dict(irmf_params)
    for name in parameter_names:
        base[name] = float(base[name])
    seen = set()

    def add_config(design_type, label, updates):
        params, factors = _scaled_params(base, updates)
        key = (
            design_type,
            label,
            tuple((name, round(factors[name], 12)) for name in parameter_names),
        )
        if key in seen:
            return
        seen.add(key)
        configs.append({
            "config_id": len(configs) + 1,
            "design_type": design_type,
            "config_label": label,
            "factors": factors,
            "params": params,
        })

    add_config("local_oat", "locked_center", {})
    for name in parameter_names:
        for factor in IRMF_PARAMETER_SENSITIVITY_OAT_FACTORS:
            if abs(float(factor) - 1.0) <= 1e-12:
                continue
            add_config("local_oat", f"{name}_x_{float(factor):.2f}", {name: factor})

    for left, right in IRMF_PARAMETER_SENSITIVITY_JOINT_INTERACTIONS:
        for left_factor in (0.8, 1.2):
            for right_factor in (0.8, 1.2):
                add_config(
                    "targeted_interaction",
                    f"{left}_x_{left_factor:.2f}__{right}_x_{right_factor:.2f}",
                    {left: left_factor, right: right_factor},
                )
    return configs


def _smoke_cases(signals, noises, target_snr_db_levels, seeds, n, fs):
    cases = []
    for signal_name in signals:
        for noise_name in noises:
            for target_snr_db in target_snr_db_levels:
                sigma_label = 10.0 ** (-float(target_snr_db) / 20.0)
                for seed in seeds:
                    case = make_signal_noise_case(
                        signal_name=signal_name,
                        noise_name=noise_name,
                        sigma=sigma_label,
                        n=n,
                        fs=fs,
                        seed=seed,
                        target_snr_db=target_snr_db,
                    )
                    case["seed"] = int(seed)
                    cases.append(case)
    return cases


def _flat_row(config, case, result, runtime_seconds, error=None):
    row = {
        "config_id": config["config_id"],
        "design_type": config["design_type"],
        "config_label": config["config_label"],
        "signal": case["signal_name"],
        "noise": case["noise_name"],
        "target_snr_db": case["target_snr_db"],
        "sigma_label": case["sigma"],
        "seed": case.get("seed"),
        "noise_design": case.get("noise_design"),
        "realized_input_snr_db": case.get("realized_input_snr_db"),
        "runtime_seconds": runtime_seconds,
        "status": "failed" if error else "completed",
        "error": str(error) if error else None,
    }
    for name in IRMF_PARAMETER_SENSITIVITY_PARAMETERS:
        row[f"{name}_factor"] = config["factors"][name]
        row[f"{name}_value"] = config["params"][name]
    if result is not None:
        summary = method_result_summary(result)
        for key, value in summary.items():
            if key not in row:
                row[key] = value
        for metric in PRIMARY_METRICS:
            row[metric] = _value(result, metric)
    return row


def _metric_completeness(rows):
    out = []
    completed = [row for row in rows if row.get("status") == "completed"]
    for metric in PRIMARY_METRICS:
        finite = 0
        nonfinite = 0
        missing = 0
        for row in completed:
            if metric not in row:
                missing += 1
                continue
            value = _finite(row.get(metric))
            if value is None:
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


def _status_artifact(rows, configs, cases, started_utc, finished_utc):
    n_expected = int(len(configs) * len(cases))
    n_completed = sum(1 for row in rows if row.get("status") == "completed")
    n_failed = sum(1 for row in rows if row.get("status") == "failed")
    metric_rows = _metric_completeness(rows)
    unexpected_missing_metrics = [
        row for row in metric_rows
        if row["n_missing_field"] > 0
    ]
    return {
        "module_status": (
            "smoke_passed_execution_integrity"
            if n_expected == n_completed + n_failed and n_failed == 0 and not unexpected_missing_metrics
            else "smoke_failed_or_requires_attention"
        ),
        "protocol_id": V535_SECTION6_1_SMOKE_VERSION,
        "upstream_protocol": "V5.34_section6_executable_protocols",
        "protocol_frozen": True,
        "execution_scope": "small_grid_smoke_execution_integrity_only",
        "execution_complete": bool(n_expected == n_completed + n_failed),
        "n_expected_cells": n_expected,
        "n_completed_cells": int(n_completed),
        "n_failed_cells": int(n_failed),
        "statistics_complete": False,
        "claim_authorized": False,
        "claim_boundary": (
            "This smoke authorizes only execution-integrity confidence for 6.1. "
            "It does not authorize scientific parameter-robustness claims."
        ),
        "blocking_issues": [
            "failed method evaluations" if n_failed else None,
            "missing primary metric fields" if unexpected_missing_metrics else None,
        ],
        "blocking_issues_present": bool(n_failed or unexpected_missing_metrics),
        "started_timestamp_utc": started_utc,
        "finished_timestamp_utc": finished_utc,
    }


def run_v535_section6_1_irmf_parameter_smoke(
        output_root,
        irmf_params=None,
        signals=("stationary_multi_sine",),
        noises=("gaussian", "huber_contamination"),
        target_snr_db_levels=(15.0,),
        seeds=(0,),
        n=DEFAULT_N,
        fs=DEFAULT_FS,
):
    output_root = ensure_dir(output_root)
    base_irmf_params = dict(irmf_params or GLOBAL_IRMF_PARAMS)
    started_utc = datetime.now(timezone.utc).isoformat()
    configs = _parameter_configurations(base_irmf_params)
    cases = _smoke_cases(
        signals=signals,
        noises=noises,
        target_snr_db_levels=target_snr_db_levels,
        seeds=seeds,
        n=n,
        fs=fs,
    )
    rows = []
    for config in configs:
        for case in cases:
            start = perf_counter()
            result = None
            error = None
            try:
                result = run_fixed_irmf_case(
                    Y=case["Y"],
                    X_clean=case["X_clean"],
                    t=case["t"],
                    fs=fs,
                    irmf_params=config["params"],
                    expected_noise_ratio=case.get("expected_noise_ratio"),
                    true_components=case.get("true_components"),
                    contamination_mask=case.get("contamination_mask"),
                    run_id=(
                        "v535_section6_1_smoke_"
                        f"{config['config_label']}_{case['signal_name']}_"
                        f"{case['noise_name']}_snr{float(case['target_snr_db']):g}_"
                        f"seed{case.get('seed')}"
                    ),
                )
            except Exception as exc:
                error = exc
            runtime = float(perf_counter() - start)
            rows.append(_flat_row(config, case, result, runtime, error=error))

    finished_utc = datetime.now(timezone.utc).isoformat()
    status = _status_artifact(rows, configs, cases, started_utc, finished_utc)
    protocol = {
        "schema_version": V535_SECTION6_1_SMOKE_VERSION,
        "protocol_status": "smoke_execution_completed",
        "purpose": "Validate execution integrity for Section 6.1 before full execution.",
        "not_a_scientific_claim": True,
        "base_irmf_params": base_irmf_params,
        "H_parameterization": base_irmf_params.get("H_parameterization", "absolute"),
        "relative_H_rule": (
            "Perturb c_H, not absolute H; run_fixed_irmf_case derives "
            "H_case = c_H * sigma_hat(Y) for each smoke case."
            if base_irmf_params.get("H_parameterization") == "relative_noise_scale"
            else None
        ),
        "parameter_configurations": configs,
        "smoke_case_design": {
            "signals": list(signals),
            "noises": list(noises),
            "target_snr_db_levels": list(target_snr_db_levels),
            "seeds": list(seeds),
            "n": int(n),
            "fs": float(fs),
        },
        "expected_cell_formula": (
            "n_parameter_configurations x n_signals x n_noises x "
            "n_target_snr_levels x n_seeds"
        ),
        "n_parameter_configurations": int(len(configs)),
        "n_cases": int(len(cases)),
        "n_expected_cells": int(len(configs) * len(cases)),
        "status_artifact": status,
    }
    metric_rows = _metric_completeness(rows)
    write_json(protocol, output_root / "v535_section6_1_smoke_protocol.json")
    write_json(status, output_root / "v535_section6_1_smoke_status.json")
    write_json(rows, output_root / "v535_section6_1_smoke_rows.json")
    write_csv(rows, output_root / "v535_section6_1_smoke_rows.csv")
    write_csv(metric_rows, output_root / "v535_section6_1_smoke_metric_completeness.csv")
    write_json({
        "runtime_seconds_total": float(sum(row.get("runtime_seconds") or 0.0 for row in rows)),
        "runtime_seconds_median": (
            float(np.median([row["runtime_seconds"] for row in rows]))
            if rows else None
        ),
        "runtime_seconds_max": (
            float(np.max([row["runtime_seconds"] for row in rows]))
            if rows else None
        ),
        "rows": int(len(rows)),
    }, output_root / "v535_section6_1_smoke_runtime_summary.json")
    return {
        "protocol": protocol,
        "status": status,
        "rows": rows,
        "metric_completeness": metric_rows,
    }
