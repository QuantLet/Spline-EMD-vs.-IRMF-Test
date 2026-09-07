#!/usr/bin/python
# coding: UTF-8

"""V5.36 smoke execution for Section 6.2 contamination-design sensitivity."""

from datetime import datetime, timezone
from time import perf_counter

import numpy as np

from noise_bank.noise_models import generate_noise
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
from signal_bank.synthetic_signals import get_signal, get_true_components
from experiments.experiment_utils import (
    _failure_summary_for_exception,
    _run_with_timeout,
    attach_failure_flags,
    method_result_summary,
    realized_snr_db,
    run_fixed_emd_case,
    run_fixed_irmf_case,
    target_snr_noise_scale,
)
from experiments.experiment_emd_family_benchmark import _run_locked_emd_family_method
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


V536_SECTION6_2_SMOKE_VERSION = "V5.36_section6_2_contamination_design_smoke"
METHODS = ("IRMF", "EMD", "EEMD", "CEEMDAN")


def _primary_metrics():
    out = []
    for metrics in V528_PRIMARY_ENDPOINTS_BY_DIMENSION.values():
        out.extend(metrics)
    return tuple(out)


PRIMARY_METRICS = _primary_metrics()
LOWER_IS_BETTER = set(LOWER_IS_BETTER_METRICS)
CONTAMINATION_PRIMARY = (
    "contaminated_region_nmse",
    "clean_region_nmse",
    "normalized_contamination_spillover_loss",
)


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


def _mask_for_geometry(n, lam, geometry, seed, x_clean):
    rng = np.random.default_rng(seed + 19017)
    count = max(1, int(round(float(lam) * int(n))))
    mask = np.zeros(n, dtype=bool)
    geometry = str(geometry)
    if geometry == "random_dispersed":
        idx = rng.choice(n, size=min(count, n), replace=False)
        mask[idx] = True
        return mask
    if geometry == "clustered_block":
        start = int(rng.integers(0, max(1, n - count + 1)))
        mask[start:start + count] = True
        return mask
    if geometry == "boundary":
        half = max(1, count // 2)
        mask[:half] = True
        mask[max(half, n - (count - half)):] = True
        return mask
    if geometry == "high_curvature_or_transient_aligned":
        x = np.asarray(x_clean, dtype=float)
        curvature = np.abs(np.gradient(np.gradient(x)))
        idx = np.argsort(curvature)[-min(count, n):]
        mask[idx] = True
        return mask
    if geometry == "smooth_region_aligned":
        x = np.asarray(x_clean, dtype=float)
        curvature = np.abs(np.gradient(np.gradient(x)))
        idx = np.argsort(curvature)[:min(count, n)]
        mask[idx] = True
        return mask
    if geometry == "peak_or_extremum_aligned":
        x = np.asarray(x_clean, dtype=float)
        idx = np.argsort(np.abs(x))[-min(count, n):]
        mask[idx] = True
        return mask
    raise ValueError(f"unknown contamination geometry: {geometry}")


def _standardize(x, eps=1e-12):
    x = np.asarray(x, dtype=float)
    x = x - float(np.mean(x))
    sd = float(np.std(x))
    if sd <= eps:
        return x
    return x / sd


def _make_geometry_case(signal_name, geometry, lam, outlier_magnitude, target_snr_db, seed, n, fs):
    t_phys = np.arange(n, dtype=float) / float(fs)
    t_unit = np.linspace(0.0, 1.0, n, endpoint=False) if n > 1 else np.array([0.0])
    x_clean = np.asarray(get_signal(signal_name, t_unit), dtype=float)
    true_components = get_true_components(signal_name, t_unit)
    mask = _mask_for_geometry(n, lam, geometry, seed, x_clean)
    rng = np.random.default_rng(seed + 271828)
    base = rng.standard_normal(n)
    if np.any(mask):
        signs = rng.choice([-1.0, 1.0], size=int(np.sum(mask)))
        magnitudes = np.abs(
            rng.normal(float(outlier_magnitude), 0.2 * float(outlier_magnitude), size=int(np.sum(mask)))
        )
        base[mask] = signs * magnitudes
    base = _standardize(base)
    alpha = target_snr_noise_scale(x_clean, base, target_snr_db)
    noise = alpha * base
    return {
        "signal_name": signal_name,
        "noise_name": "huber_contamination",
        "sigma": 10.0 ** (-float(target_snr_db) / 20.0),
        "target_snr_db": float(target_snr_db),
        "noise_design": "target_snr_energy_ratio_geometry_contamination",
        "noise_scale_alpha": float(alpha),
        "realized_input_snr_db": realized_snr_db(x_clean, noise),
        "clean_signal_rms": float(np.sqrt(np.mean(x_clean ** 2))),
        "noise_rms": float(np.sqrt(np.mean(noise ** 2))),
        "t": t_unit,
        "T": t_phys,
        "X_clean": x_clean,
        "Y": x_clean + noise,
        "noise": noise,
        "contamination_mask": mask,
        "contamination_mask_applicable": True,
        "expected_noise_ratio": float(np.sum(noise ** 2) / (np.sum((x_clean + noise) ** 2) + 1e-10)),
        "true_components": true_components,
        "seed": int(seed),
    }


def _make_generator_case(signal_name, lam, outlier_magnitude, target_snr_db, seed, n, fs):
    t_phys = np.arange(n, dtype=float) / float(fs)
    t_unit = np.linspace(0.0, 1.0, n, endpoint=False) if n > 1 else np.array([0.0])
    x_clean = np.asarray(get_signal(signal_name, t_unit), dtype=float)
    true_components = get_true_components(signal_name, t_unit)
    kwargs = {"lam": float(lam), "outlier_scale": float(outlier_magnitude)}
    base = generate_noise("huber_contamination", n=n, sigma=1.0, seed=seed, **kwargs)
    rng = np.random.default_rng(seed)
    _ = rng.standard_normal(n)
    mask = rng.random(n) < float(lam)
    alpha = target_snr_noise_scale(x_clean, base, target_snr_db)
    noise = alpha * base
    return {
        "signal_name": signal_name,
        "noise_name": "huber_contamination",
        "sigma": 10.0 ** (-float(target_snr_db) / 20.0),
        "target_snr_db": float(target_snr_db),
        "noise_design": "target_snr_energy_ratio_huber_generator",
        "noise_scale_alpha": float(alpha),
        "realized_input_snr_db": realized_snr_db(x_clean, noise),
        "clean_signal_rms": float(np.sqrt(np.mean(x_clean ** 2))),
        "noise_rms": float(np.sqrt(np.mean(noise ** 2))),
        "t": t_unit,
        "T": t_phys,
        "X_clean": x_clean,
        "Y": x_clean + noise,
        "noise": noise,
        "contamination_mask": mask,
        "contamination_mask_applicable": True,
        "expected_noise_ratio": float(np.sum(noise ** 2) / (np.sum((x_clean + noise) ** 2) + 1e-10)),
        "true_components": true_components,
        "seed": int(seed),
    }


def _case_designs():
    return [
        {
            "design_id": 1,
            "axis": "lambda",
            "signal": "stationary_multi_sine",
            "lambda": 0.03,
            "outlier_magnitude": 10.0,
            "geometry": "random_dispersed",
            "case_builder": "huber_generator",
        },
        {
            "design_id": 2,
            "axis": "lambda",
            "signal": "stationary_multi_sine",
            "lambda": 0.10,
            "outlier_magnitude": 10.0,
            "geometry": "random_dispersed",
            "case_builder": "huber_generator",
        },
        {
            "design_id": 3,
            "axis": "magnitude",
            "signal": "stationary_multi_sine",
            "lambda": 0.05,
            "outlier_magnitude": 5.0,
            "geometry": "random_dispersed",
            "case_builder": "huber_generator",
        },
        {
            "design_id": 4,
            "axis": "geometry",
            "signal": "chirp",
            "lambda": 0.05,
            "outlier_magnitude": 10.0,
            "geometry": "clustered_block",
            "case_builder": "geometry_mask",
        },
        {
            "design_id": 5,
            "axis": "geometry",
            "signal": "chirp",
            "lambda": 0.05,
            "outlier_magnitude": 10.0,
            "geometry": "boundary",
            "case_builder": "geometry_mask",
        },
        {
            "design_id": 6,
            "axis": "magnitude_x_position",
            "signal": "chirp",
            "lambda": 0.05,
            "outlier_magnitude": 15.0,
            "geometry": "boundary",
            "case_builder": "geometry_mask",
        },
    ]


def _build_case(design, target_snr_db, seed, n, fs):
    if design["case_builder"] == "huber_generator":
        return _make_generator_case(
            signal_name=design["signal"],
            lam=design["lambda"],
            outlier_magnitude=design["outlier_magnitude"],
            target_snr_db=target_snr_db,
            seed=seed,
            n=n,
            fs=fs,
        )
    return _make_geometry_case(
        signal_name=design["signal"],
        geometry=design["geometry"],
        lam=design["lambda"],
        outlier_magnitude=design["outlier_magnitude"],
        target_snr_db=target_snr_db,
        seed=seed,
        n=n,
        fs=fs,
    )


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
                    run_id=f"v536_section6_2_smoke_{method.lower()}",
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
                    run_id=f"v536_section6_2_smoke_{method.lower()}",
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
        return attach_failure_flags(
            result,
            Y_observed=case["Y"],
            X_clean=case["X_clean"],
            timeout_seconds=timeout_seconds,
            runtime_seconds=runtime,
        ), runtime, None
    except Exception as exc:
        runtime = float(perf_counter() - start)
        return _failure_summary_for_exception(method, exc, runtime, timeout_seconds), runtime, exc


def _method_row(design, case, method, result, runtime, error):
    summary = method_result_summary(result)
    row = {
        "design_id": design["design_id"],
        "axis": design["axis"],
        "signal": design["signal"],
        "noise": "huber_contamination",
        "target_snr_db": case["target_snr_db"],
        "seed": case["seed"],
        "method": method,
        "lambda_target": design["lambda"],
        "outlier_magnitude_target": design["outlier_magnitude"],
        "position_geometry": design["geometry"],
        "case_builder": design["case_builder"],
        "noise_design": case["noise_design"],
        "realized_input_snr_db": case["realized_input_snr_db"],
        "runtime_seconds": runtime,
        "status": "failed" if error else "completed",
        "error": str(error) if error else None,
        "contaminated_point_count": int(np.sum(case["contamination_mask"])),
        "contaminated_fraction_realized": float(np.mean(case["contamination_mask"])),
        "lambda_abs_error": abs(float(np.mean(case["contamination_mask"])) - float(design["lambda"])),
    }
    row.update(summary)
    for metric in PRIMARY_METRICS:
        row[metric] = result.get(_metric_code_field(metric))
    return row


def _mask_audit(design, case):
    mask = np.asarray(case["contamination_mask"], dtype=bool)
    n = len(mask)
    idx = np.flatnonzero(mask)
    clean = ~mask
    radius = 5
    expanded = mask.copy()
    for i in idx:
        expanded[max(0, i - radius):min(n, i + radius + 1)] = True
    spill = expanded & clean
    return {
        "design_id": design["design_id"],
        "axis": design["axis"],
        "case_builder": design["case_builder"],
        "position_geometry": design["geometry"],
        "lambda_target": design["lambda"],
        "outlier_magnitude_target": design["outlier_magnitude"],
        "n": int(n),
        "contaminated_count": int(np.sum(mask)),
        "contaminated_fraction_realized": float(np.mean(mask)),
        "lambda_abs_error": abs(float(np.mean(mask)) - float(design["lambda"])),
        "mask_min_index": int(idx.min()) if idx.size else None,
        "mask_max_index": int(idx.max()) if idx.size else None,
        "mask_mean_index_fraction": float(np.mean(idx) / max(n - 1, 1)) if idx.size else None,
        "clean_count": int(np.sum(clean)),
        "spillover_count": int(np.sum(spill)),
        "mask_clean_overlap_count": int(np.sum(mask & clean)),
        "mask_spillover_overlap_count": int(np.sum(mask & spill)),
        "clean_spillover_overlap_count": int(np.sum(clean & spill)),
        "mask_has_expected_nonzero_count": bool(np.sum(mask) > 0),
        "clean_region_available": bool(np.any(clean)),
        "spillover_region_available": bool(np.any(spill)),
    }


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


def _status(rows, mask_rows, designs, started_utc, finished_utc):
    n_expected = int(len(designs) * len(METHODS))
    n_completed = sum(1 for r in rows if r.get("status") == "completed")
    n_failed = sum(1 for r in rows if r.get("status") == "failed")
    metric_rows = _metric_completeness(rows)
    missing = [r for r in metric_rows if int(r["n_missing_field"]) > 0]
    contamination_metric_bad = [
        r for r in metric_rows
        if r["metric"] in CONTAMINATION_PRIMARY and (
            int(r["n_missing_field"]) > 0 or int(r["n_finite"]) < int(r["n_completed_rows"])
        )
    ]
    mask_bad = [
        r for r in mask_rows
        if (
            not r["mask_has_expected_nonzero_count"]
            or not r["clean_region_available"]
            or not r["spillover_region_available"]
            or r["mask_clean_overlap_count"] != 0
            or r["mask_spillover_overlap_count"] != 0
        )
    ]
    passed = (
        n_expected == n_completed + n_failed
        and n_failed == 0
        and not missing
        and not contamination_metric_bad
        and not mask_bad
    )
    return {
        "module_status": (
            "smoke_passed_execution_integrity"
            if passed else "smoke_failed_or_requires_attention"
        ),
        "protocol_id": V536_SECTION6_2_SMOKE_VERSION,
        "upstream_protocol": "V5.34_section6_executable_protocols",
        "protocol_frozen": True,
        "execution_scope": "contamination_design_smoke_execution_integrity_only",
        "execution_complete": bool(n_expected == n_completed + n_failed),
        "n_expected_cells": n_expected,
        "n_completed_cells": int(n_completed),
        "n_failed_cells": int(n_failed),
        "statistics_complete": False,
        "claim_authorized": False,
        "claim_boundary": (
            "This smoke validates contamination-design execution integrity only. "
            "It does not authorize contamination-robustness scientific claims."
        ),
        "blocking_issues_present": not passed,
        "blocking_issues": {
            "failed_method_evaluations": int(n_failed),
            "missing_primary_metric_fields": int(len(missing)),
            "bad_contamination_primary_completeness": int(len(contamination_metric_bad)),
            "bad_mask_audit_rows": int(len(mask_bad)),
        },
        "started_timestamp_utc": started_utc,
        "finished_timestamp_utc": finished_utc,
    }


def run_v536_section6_2_contamination_design_smoke(
        output_root,
        target_snr_db=15.0,
        seed=0,
        n=DEFAULT_N,
        fs=DEFAULT_FS,
        timeout_seconds=DEFAULT_METHOD_TIMEOUT_SECONDS,
):
    output_root = ensure_dir(output_root)
    started = datetime.now(timezone.utc).isoformat()
    designs = _case_designs()
    rows = []
    mask_rows = []
    case_manifest = []
    for design in designs:
        case = _build_case(design, target_snr_db=target_snr_db, seed=seed, n=n, fs=fs)
        mask_rows.append(_mask_audit(design, case))
        case_manifest.append({
            "design_id": design["design_id"],
            "axis": design["axis"],
            "signal": design["signal"],
            "lambda": design["lambda"],
            "outlier_magnitude": design["outlier_magnitude"],
            "geometry": design["geometry"],
            "case_builder": design["case_builder"],
            "target_snr_db": target_snr_db,
            "seed": seed,
            "realized_input_snr_db": case["realized_input_snr_db"],
        })
        for method in METHODS:
            result, runtime, error = _run_method(
                method=method,
                case=case,
                algorithm_seed=100000 + int(seed),
                timeout_seconds=timeout_seconds,
            )
            rows.append(_method_row(design, case, method, result, runtime, error))
    finished = datetime.now(timezone.utc).isoformat()
    status = _status(rows, mask_rows, designs, started, finished)
    metric_rows = _metric_completeness(rows)
    protocol = {
        "schema_version": V536_SECTION6_2_SMOKE_VERSION,
        "protocol_status": "smoke_execution_completed",
        "purpose": "Validate execution integrity for Section 6.2 before full execution.",
        "not_a_scientific_claim": True,
        "methods": list(METHODS),
        "case_designs": designs,
        "case_manifest": case_manifest,
        "target_snr_db": float(target_snr_db),
        "seed": int(seed),
        "n": int(n),
        "fs": float(fs),
        "n_expected_cells": int(len(designs) * len(METHODS)),
        "expected_cell_formula": "n_contamination_designs x n_methods",
        "status_artifact": status,
    }
    write_json(protocol, output_root / "v536_section6_2_smoke_protocol.json")
    write_json(status, output_root / "v536_section6_2_smoke_status.json")
    write_json(rows, output_root / "v536_section6_2_smoke_rows.json")
    write_csv(rows, output_root / "v536_section6_2_smoke_rows.csv")
    write_csv(mask_rows, output_root / "v536_section6_2_smoke_mask_audit.csv")
    write_csv(metric_rows, output_root / "v536_section6_2_smoke_metric_completeness.csv")
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
    }, output_root / "v536_section6_2_smoke_runtime_summary.json")
    return {
        "protocol": protocol,
        "status": status,
        "rows": rows,
        "mask_audit": mask_rows,
        "metric_completeness": metric_rows,
    }
