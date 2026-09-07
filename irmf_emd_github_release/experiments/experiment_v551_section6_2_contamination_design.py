#!/usr/bin/python
# coding: UTF-8

"""V5.51 full Section 6.2 contamination-design sensitivity.

The design is intentionally controlled rather than factorial: one-axis-at-a-time
rate, magnitude, geometry, energy-matched shape comparison, and a small
lambda-by-magnitude interaction.  All cases use target-SNR scaling.
"""

from collections import defaultdict
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
    V557_SECTION6_2_CONTAMINATION_SIGNALS,
    V528_PRIMARY_METRIC_CODE_FIELD_MAP,
)
from experiments.experiment_emd_family_benchmark import _run_locked_emd_family_method
from experiments.experiment_utils import (
    _failure_summary_for_exception,
    _run_with_timeout,
    attach_failure_flags,
    method_result_summary,
    run_fixed_emd_case,
    run_fixed_irmf_case,
    realized_snr_db,
    target_snr_noise_scale,
)
from experiments.experiment_v536_section6_2_contamination_smoke import (
    CONTAMINATION_PRIMARY,
    METHODS,
    _build_case,
    _make_geometry_case,
    _mask_audit,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


V551_SECTION6_2_VERSION = "V5.51_section6_2_contamination_design_sensitivity"
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


def _derive_normalized_spillover_loss(score):
    score = _finite(score)
    if score is None or score <= 0.0:
        return np.nan
    return float(-np.log(score))


def _audit_normalized_spillover_loss(rows, tolerance=1e-10):
    audit_rows = []
    diffs = []
    n_checked = 0
    n_passed = 0
    n_failed = 0
    for row in rows:
        if row.get("status") != "completed":
            continue
        score = _finite(row.get("contamination_spillover_score"))
        loss = _finite(row.get("normalized_contamination_spillover_loss"))
        expected = _derive_normalized_spillover_loss(score)
        if score is None or loss is None or not np.isfinite(expected):
            continue
        diff = abs(float(loss) - float(expected))
        passed = bool(diff <= tolerance)
        n_checked += 1
        n_passed += int(passed)
        n_failed += int(not passed)
        diffs.append(diff)
        audit_rows.append({
            "design_id": row.get("design_id"),
            "axis": row.get("axis"),
            "signal": row.get("signal"),
            "method": row.get("method"),
            "seed": row.get("seed"),
            "contamination_spillover_score": score,
            "normalized_contamination_spillover_loss": loss,
            "expected_normalized_contamination_spillover_loss": expected,
            "abs_error": diff,
            "passed": passed,
            "formula": "-log(contamination_spillover_score)",
        })
    return {
        "audit_rows": audit_rows,
        "summary": {
            "formula": "-log(contamination_spillover_score)",
            "tolerance": float(tolerance),
            "n_checked": int(n_checked),
            "n_passed": int(n_passed),
            "n_failed": int(n_failed),
            "max_abs_error": float(max(diffs)) if diffs else None,
            "mean_abs_error": float(np.mean(diffs)) if diffs else None,
            "passed": bool(n_checked > 0 and n_failed == 0),
        },
    }


def _noise_energy(case):
    val = _finite(case.get("noise_energy"))
    if val is not None:
        return val
    return float(np.sum(np.asarray(case["noise"], dtype=float) ** 2))


def _noise_rms(case):
    val = _finite(case.get("noise_rms"))
    if val is not None:
        return val
    return float(np.sqrt(np.mean(np.asarray(case["noise"], dtype=float) ** 2)))


def _standardize(x, eps=1e-12):
    x = np.asarray(x, dtype=float)
    x = x - float(np.mean(x))
    sd = float(np.std(x))
    if sd <= eps:
        return x
    return x / sd


def _make_variance_matched_gaussian_reference(contam_case, seed):
    rng = np.random.default_rng(int(seed) + 818191)
    base = _standardize(rng.standard_normal(len(contam_case["X_clean"])))
    alpha = target_snr_noise_scale(
        contam_case["X_clean"],
        base,
        contam_case["target_snr_db"],
    )
    noise = alpha * base
    case = dict(contam_case)
    case["Y"] = np.asarray(case["X_clean"], dtype=float) + noise
    case["noise"] = noise
    case["noise_name"] = "gaussian_variance_matched_reference"
    case["noise_design"] = "target_snr_energy_ratio_variance_matched_gaussian_reference"
    case["noise_scale_alpha"] = float(alpha)
    case["noise_energy"] = float(np.sum(noise ** 2))
    case["noise_rms"] = float(np.sqrt(np.mean(noise ** 2)))
    case["realized_input_snr_db"] = realized_snr_db(case["X_clean"], noise)
    case["expected_noise_ratio"] = float(np.sum(noise ** 2) / (np.sum(case["Y"] ** 2) + 1e-10))
    return case


def _designs_for_signal(signal_name):
    designs = []
    did = 0

    def add(axis, lam, kappa, geometry, comparison="contamination", note=None):
        nonlocal did
        did += 1
        designs.append({
            "design_id": f"{signal_name}_{did:03d}",
            "axis": axis,
            "signal": signal_name,
            "lambda": float(lam),
            "outlier_magnitude": float(kappa),
            "kappa": float(kappa),
            "geometry": geometry,
            "case_builder": "geometry_mask",
            "variance_comparison": comparison,
            "design_note": note,
        })

    for lam in (0.01, 0.03, 0.05, 0.10, 0.15, 0.20):
        add("rate", lam, 5.0, "random_dispersed")
    for kappa in (3.0, 5.0, 8.0):
        add("magnitude", 0.05, kappa, "random_dispersed")
    for geometry in ("random_dispersed", "clustered_block", "boundary", "high_curvature_or_transient_aligned"):
        add("geometry", 0.05, 5.0, geometry)
    add("variance_matched_shape", 0.05, 5.0, "random_dispersed", "contamination")
    add("variance_matched_shape", 0.05, 5.0, "random_dispersed", "gaussian_reference")
    for lam in (0.03, 0.05, 0.10):
        for kappa in (3.0, 5.0, 8.0):
            add("lambda_x_magnitude", lam, kappa, "random_dispersed")
    return designs


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
                    run_id=f"v551_section6_2_{method.lower()}",
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
                    run_id=f"v551_section6_2_{method.lower()}",
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


def _row(design, case, method, result, runtime, error):
    summary = method_result_summary(result)
    out = {
        "design_id": design["design_id"],
        "axis": design["axis"],
        "signal": design["signal"],
        "noise": case.get("noise_name", "huber_contamination"),
        "target_snr_db": float(case["target_snr_db"]),
        "seed": int(case["seed"]),
        "method": method,
        "lambda_target": float(design["lambda"]),
        "kappa": float(design["kappa"]),
        "outlier_magnitude_target": float(design["outlier_magnitude"]),
        "position_geometry": design["geometry"],
        "variance_comparison": design["variance_comparison"],
        "noise_design": case["noise_design"],
        "realized_input_snr_db": _finite(case.get("realized_input_snr_db")),
        "noise_energy": _noise_energy(case),
        "noise_rms": _noise_rms(case),
        "runtime_seconds": runtime,
        "status": "failed" if error else "completed",
        "error": str(error) if error else None,
        "contaminated_point_count": int(np.sum(case["contamination_mask"])),
        "contaminated_fraction_realized": float(np.mean(case["contamination_mask"])),
    }
    out.update(summary)
    out["result_method_label"] = out.get("method")
    out["method"] = method
    for metric in CONTAMINATION_PRIMARY:
        if metric == "normalized_contamination_spillover_loss":
            out[metric] = _derive_normalized_spillover_loss(
                result.get("contamination_spillover_score")
            )
        else:
            out[metric] = result.get(_metric_code_field(metric))
    return out


def _summaries(rows):
    completed = [r for r in rows if r.get("status") == "completed"]
    metric_summary = []
    grouped = defaultdict(list)
    for row in completed:
        for metric in CONTAMINATION_PRIMARY:
            val = _finite(row.get(metric))
            if val is not None:
                key = (
                    row["axis"], row["method"], metric,
                    row.get("lambda_target"), row.get("kappa"),
                    row.get("position_geometry"), row.get("variance_comparison"),
                )
                grouped[key].append(val)
    for key, vals in sorted(grouped.items(), key=lambda item: str(item[0])):
        arr = np.asarray(vals, dtype=float)
        metric_summary.append({
            "axis": key[0],
            "method": key[1],
            "metric": key[2],
            "lambda_target": key[3],
            "kappa": key[4],
            "position_geometry": key[5],
            "variance_comparison": key[6],
            "n": int(arr.size),
            "median": float(np.median(arr)),
            "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            "mean": float(np.mean(arr)),
        })

    paired = []
    by_cell_metric = defaultdict(dict)
    for row in completed:
        cell = (
            row["axis"], row["design_id"], row["signal"], row["target_snr_db"],
            row["seed"], row["lambda_target"], row["kappa"],
            row["position_geometry"], row["variance_comparison"],
        )
        for metric in CONTAMINATION_PRIMARY:
            val = _finite(row.get(metric))
            if val is not None:
                by_cell_metric[(cell, metric)][row["method"]] = val
    for (cell, metric), vals in sorted(by_cell_metric.items(), key=lambda item: str(item[0])):
        if "IRMF" not in vals:
            continue
        higher = metric not in LOWER_IS_BETTER
        for baseline in ("EMD", "EEMD", "CEEMDAN"):
            if baseline not in vals:
                continue
            delta = vals["IRMF"] - vals[baseline]
            win = vals["IRMF"] > vals[baseline] if higher else vals["IRMF"] < vals[baseline]
            paired.append({
                "axis": cell[0],
                "design_id": cell[1],
                "signal": cell[2],
                "target_snr_db": cell[3],
                "seed": cell[4],
                "lambda_target": cell[5],
                "kappa": cell[6],
                "position_geometry": cell[7],
                "variance_comparison": cell[8],
                "metric": metric,
                "baseline": baseline,
                "irmf_minus_baseline": float(delta),
                "irmf_win": bool(win),
                "higher_is_better": bool(higher),
            })
    win_summary = []
    grouped_paired = defaultdict(list)
    for row in paired:
        grouped_paired[(row["axis"], row["baseline"], row["metric"])].append(row)
    for key, vals in sorted(grouped_paired.items(), key=lambda item: str(item[0])):
        deltas = np.asarray([v["irmf_minus_baseline"] for v in vals], dtype=float)
        win_summary.append({
            "axis": key[0],
            "baseline": key[1],
            "metric": key[2],
            "n": len(vals),
            "win_rate": float(np.mean([bool(v["irmf_win"]) for v in vals])),
            "median_irmf_minus_baseline": float(np.median(deltas)),
            "iqr_irmf_minus_baseline": float(np.percentile(deltas, 75) - np.percentile(deltas, 25)),
        })
    return metric_summary, paired, win_summary


def run_v551_section6_2_contamination_design_sensitivity(
        output_root,
        signals=V557_SECTION6_2_CONTAMINATION_SIGNALS,
        seeds=(0, 1, 2),
        target_snr_db=15.0,
        n=DEFAULT_N,
        fs=DEFAULT_FS,
        timeout_seconds=DEFAULT_METHOD_TIMEOUT_SECONDS,
):
    output_root = ensure_dir(output_root)
    started = datetime.now(timezone.utc).isoformat()
    rows = []
    mask_rows = []
    design_manifest = []
    for signal_name in signals:
        for design in _designs_for_signal(signal_name):
            for seed in seeds:
                case = _build_case(design, target_snr_db=target_snr_db, seed=seed, n=n, fs=fs)
                if design["variance_comparison"] == "gaussian_reference":
                    case = _make_variance_matched_gaussian_reference(case, seed)
                mask_rows.append(_mask_audit(design, case))
                design_manifest.append({
                    "design_id": design["design_id"],
                    "axis": design["axis"],
                    "signal": signal_name,
                    "seed": int(seed),
                    "lambda": float(design["lambda"]),
                    "kappa": float(design["kappa"]),
                    "geometry": design["geometry"],
                    "variance_comparison": design["variance_comparison"],
                    "target_snr_db": float(target_snr_db),
                    "realized_input_snr_db": _finite(case.get("realized_input_snr_db")),
                    "noise_energy": _noise_energy(case),
                    "noise_rms": _noise_rms(case),
                })
                for method in METHODS:
                    result, runtime, error = _run_method(
                        method,
                        case,
                        algorithm_seed=100000 + int(seed),
                        timeout_seconds=timeout_seconds,
                    )
                    rows.append(_row(design, case, method, result, runtime, error))
    metric_summary, paired, win_summary = _summaries(rows)
    spillover_formula_audit = _audit_normalized_spillover_loss(rows)
    finished = datetime.now(timezone.utc).isoformat()
    n_expected = int(len(rows))
    n_failed = int(sum(1 for r in rows if r.get("status") == "failed"))
    status = {
        "schema_version": V551_SECTION6_2_VERSION,
        "module_status": "execution_complete_pending_claim_qualification" if n_failed == 0 else "execution_complete_with_failures",
        "protocol_status": "full_section6_2_execution",
        "target_snr_design": "target_snr_energy_ratio",
        "section": "6.2 Contamination Design Sensitivity",
        "execution_complete": True,
        "statistics_complete": True,
        "claim_authorized": False,
        "n_method_evaluations": n_expected,
        "n_failed": n_failed,
        "normalized_spillover_formula_audit": spillover_formula_audit["summary"],
        "signals": list(signals),
        "seeds": list(seeds),
        "target_snr_db": float(target_snr_db),
        "axes": ["rate", "magnitude", "geometry", "variance_matched_shape", "lambda_x_magnitude"],
        "claim_boundary": (
            "This module supports contamination-design robustness only after "
            "V5.53 claim qualification. It does not retune methods."
        ),
        "started_timestamp_utc": started,
        "finished_timestamp_utc": finished,
    }
    protocol = {
        "schema_version": V551_SECTION6_2_VERSION,
        "design": "controlled_one_axis_at_a_time_plus_selected_interaction",
        "not_full_factorial": True,
        "methods": list(METHODS),
        "signals": list(signals),
        "seeds": list(seeds),
        "target_snr_db": float(target_snr_db),
        "primary_contamination_endpoints": list(CONTAMINATION_PRIMARY),
        "normalized_spillover_definition": {
            "metric": "normalized_contamination_spillover_loss",
            "formula": "-log(contamination_spillover_score)",
            "equivalent_formula": "raw_spillover_mse / mean_clean_signal_energy",
            "source_metric": "contamination_spillover_score",
            "direction": "lower_is_better",
        },
        "variance_matched_definition": (
            "All rows use target-SNR scaling. The variance-matched shape axis "
            "compares contamination-shaped noise with a Gaussian reference under "
            "the same target-SNR energy and same evaluation support mask."
        ),
    }
    write_json(protocol, output_root / "v551_section6_2_contamination_design_protocol.json")
    write_json(status, output_root / "v551_section6_2_contamination_design_dashboard.json")
    write_json(rows, output_root / "v551_section6_2_contamination_design_rows.json")
    write_csv(rows, output_root / "v551_section6_2_contamination_design_rows.csv")
    write_csv(mask_rows, output_root / "v551_section6_2_mask_audit.csv")
    write_csv(
        spillover_formula_audit["audit_rows"],
        output_root / "v551_section6_2_normalized_spillover_formula_audit.csv",
    )
    write_csv(design_manifest, output_root / "v551_section6_2_design_manifest.csv")
    write_csv(metric_summary, output_root / "v551_section6_2_metric_summary.csv")
    write_csv(paired, output_root / "v551_section6_2_paired_effects.csv")
    write_csv(win_summary, output_root / "v551_section6_2_win_rate_summary.csv")
    return {
        "protocol": protocol,
        "status": status,
        "rows": rows,
        "metric_summary": metric_summary,
        "paired_effects": paired,
        "win_summary": win_summary,
    }
