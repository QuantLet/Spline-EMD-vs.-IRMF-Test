#!/usr/bin/python
# coding: UTF-8

"""Single-window smoke for the real-world V1 validation endpoint schema.

This is an operational smoke only.  It exercises the nine real-world validation
endpoints on one MIT-BIH development-record window using temporary,
development-only reconstruction and detector rules.  It does not lock real-data
thresholds, access held-out records, or authorize real-world performance claims.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
import json

import numpy as np

from experiments.experiment_real_data_development_run import (
    METHODS,
    _load_mitbih_record,
    _median_mad_normalize,
    _run_method,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


SCHEMA_SMOKE_ID = "REAL_WORLD_V1_SCHEMA_SINGLE_WINDOW_SMOKE"
DEFAULT_RECORD_ID = "100"
DEFAULT_WINDOW_INDEX = 0
WINDOW_LENGTH_SEC = 10.0
FS_EXPECTED = 360.0

TEMP_RECONSTRUCTION_RULE = {
    "rule_id": "TEMP_REAL_WORLD_FREQ_BAND_COMPONENT_RECONSTRUCTION_SMOKE_V0",
    "status": "temporary_smoke_not_locked",
    "method_neutral": True,
    "candidate_collection": (
        "all exposed components plus residual/trend-bearing object when available"
    ),
    "component_selection_feature": "physiological_band_energy_ratio_0p5_40Hz",
    "component_selection_threshold": 0.20,
    "fallback": "select highest physiological-band-ratio component if none pass",
    "waveform_modification": "none",
}

TEMP_RPEAK_DETECTOR = {
    "detector_id": "TEMP_FIND_PEAKS_ON_RECONSTRUCTION_SMOKE_V0",
    "status": "temporary_smoke_not_locked",
    "height_threshold_normalized": 1.0,
    "minimum_peak_distance_ms": 200.0,
    "matching_tolerance_ms": 100.0,
    "qrs_window_before_ms": 80.0,
    "qrs_window_after_ms": 120.0,
}

TEMP_COMPONENT_STABILITY = {
    "perturbation_type": "additive_micro_noise",
    "scale": "0.01 * robust_MAD(Y)",
    "seed": 17001,
    "hungarian_cost": "1 - abs(corr(component, perturbed_component))",
    "matched_threshold_abs_corr": 0.50,
}

TEMP_CONTAMINATION = {
    "artifact_type": "short_burst",
    "seed": 18001,
    "start_sec": 5.0,
    "duration_ms": 80.0,
    "amplitude": "3.0 * robust_MAD(Y)",
    "spillover_radius_ms": 100.0,
}

EPS_ABSOLUTE = 1e-12
RELATIVE_FLOOR = 1e-8
LOW_ENERGY_FLAG_RELATIVE_THRESHOLD = 1e-8


def _read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size or a.size < 3:
        return np.nan
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return np.nan
    out = float(np.corrcoef(a, b)[0, 1])
    return out if np.isfinite(out) else np.nan


def _candidate_components(result, y):
    y = np.asarray(y, dtype=float)
    imfs = np.asarray(result.get("imfs"), dtype=float)
    if imfs.ndim == 1:
        imfs = imfs[None, :]
    components = imfs if imfs.size else np.zeros((0, y.size), dtype=float)
    residual = result.get("residual")
    residual_included = False
    if residual is not None:
        residual = np.asarray(residual, dtype=float)
        if residual.shape == y.shape:
            components = np.vstack([components, residual[None, :]])
            residual_included = True
    return components, residual_included


def _band_energy_ratio(x, fs, low=0.5, high=40.0):
    x = np.asarray(x, dtype=float)
    if x.size < 3 or not np.all(np.isfinite(x)):
        return 0.0
    centered = x - np.mean(x)
    freqs = np.fft.rfftfreq(centered.size, d=1.0 / float(fs))
    power = np.abs(np.fft.rfft(centered)) ** 2
    total = float(np.sum(power) + EPS_ABSOLUTE)
    mask = (freqs >= float(low)) & (freqs <= float(high))
    return float(np.sum(power[mask]) / total)


def _protocol_reconstruction(y, result, fs):
    y = np.asarray(y, dtype=float)
    components, residual_included = _candidate_components(result, y)
    if components.size == 0:
        return {
            "xhat": np.zeros_like(y),
            "residual": y.copy(),
            "component_candidate_count": 0,
            "selected_component_count": 0,
            "selected_component_indices": "",
            "residual_or_trend_candidate_included": residual_included,
            "physiology_band_ratios": [],
            "selection_status_code": "no_components_available",
        }
    ratios = np.asarray([_band_energy_ratio(c, fs) for c in components], dtype=float)
    selected = ratios >= float(TEMP_RECONSTRUCTION_RULE["component_selection_threshold"])
    if not np.any(selected):
        selected[int(np.argmax(ratios))] = True
        status = "fallback_highest_band_ratio_selected"
    elif np.all(selected):
        status = "all_components_selected"
    else:
        status = "partial_selection"
    xhat = np.sum(components[selected], axis=0)
    return {
        "xhat": xhat,
        "residual": y - xhat,
        "component_candidate_count": int(components.shape[0]),
        "selected_component_count": int(np.sum(selected)),
        "selected_component_indices": ";".join(str(i) for i in np.flatnonzero(selected)),
        "residual_or_trend_candidate_included": residual_included,
        "physiology_band_ratios": [float(x) for x in ratios],
        "selection_status_code": status,
    }


def _detect_peaks(x, fs):
    try:
        from scipy.signal import find_peaks
    except Exception:
        return np.asarray([], dtype=int), "scipy_find_peaks_unavailable"
    distance = max(1, int(round(TEMP_RPEAK_DETECTOR["minimum_peak_distance_ms"] * fs / 1000.0)))
    peaks, _ = find_peaks(
        np.asarray(x, dtype=float),
        height=float(TEMP_RPEAK_DETECTOR["height_threshold_normalized"]),
        distance=distance,
    )
    return np.asarray(peaks, dtype=int), ""


def _match_peaks(reference, detected, tolerance_samples):
    reference = [int(x) for x in reference]
    detected = [int(x) for x in detected]
    used = set()
    matches = []
    for ref in reference:
        best_idx = None
        best_dist = None
        for idx, det in enumerate(detected):
            if idx in used:
                continue
            dist = abs(det - ref)
            if dist <= tolerance_samples and (best_dist is None or dist < best_dist):
                best_idx = idx
                best_dist = dist
        if best_idx is not None:
            used.add(best_idx)
            matches.append((ref, detected[best_idx], best_dist))
    return matches


def _rpeak_metrics(xhat, ann_rel, fs):
    peaks, warning = _detect_peaks(xhat, fs)
    tolerance = int(round(TEMP_RPEAK_DETECTOR["matching_tolerance_ms"] * fs / 1000.0))
    matches = _match_peaks(ann_rel, peaks, tolerance)
    n_ref = int(len(ann_rel))
    n_det = int(len(peaks))
    n_match = int(len(matches))
    precision = float(n_match / n_det) if n_det else np.nan
    recall = float(n_match / n_ref) if n_ref else np.nan
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0
        else np.nan
    )
    timing = [1000.0 * dist / float(fs) for _ref, _det, dist in matches]
    return {
        "detected_peak_count": n_det,
        "reference_annotation_count": n_ref,
        "matched_peak_count": n_match,
        "r_peak_precision": precision,
        "r_peak_recall": recall,
        "r_peak_f1": f1,
        "r_peak_timing_error_ms": float(np.median(timing)) if timing else np.nan,
        "r_peak_detector_warning": warning,
    }


def _qrs_locked_residual_leakage(y, residual, ann_rel, fs):
    before = int(round(TEMP_RPEAK_DETECTOR["qrs_window_before_ms"] * fs / 1000.0))
    after = int(round(TEMP_RPEAK_DETECTOR["qrs_window_after_ms"] * fs / 1000.0))
    mask = np.zeros(len(y), dtype=bool)
    for ref in ann_rel:
        lo = max(0, int(ref) - before)
        hi = min(len(y), int(ref) + after + 1)
        mask[lo:hi] = True
    if not np.any(mask):
        return np.nan, True, "not_computable_no_qrs_annotations"
    numerator = float(np.sum(np.asarray(residual)[mask] ** 2))
    local = float(np.sum(np.asarray(y)[mask] ** 2))
    full = float(np.sum(np.asarray(y) ** 2))
    floor = max(EPS_ABSOLUTE, RELATIVE_FLOOR * full)
    denom = max(local, floor)
    low_flag = bool(local <= LOW_ENERGY_FLAG_RELATIVE_THRESHOLD * full)
    return float(numerator / denom), low_flag, ""


def _effective_count(components):
    if components.size == 0:
        return 0
    energy = np.sum(components ** 2, axis=1)
    total = float(np.sum(energy))
    if total <= EPS_ABSOLUTE:
        return 0
    return int(np.sum(energy / total >= 0.01))


def _hungarian_matching(a_components, b_components):
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:
        return [], "scipy_linear_sum_assignment_unavailable"
    a = np.asarray(a_components, dtype=float)
    b = np.asarray(b_components, dtype=float)
    if a.size == 0 or b.size == 0:
        return [], ""
    cost = np.ones((a.shape[0], b.shape[0]), dtype=float)
    corr = np.zeros_like(cost)
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            c = _safe_corr(a[i], b[j])
            sim = 0.0 if not np.isfinite(c) else abs(c)
            corr[i, j] = sim
            cost[i, j] = 1.0 - sim
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j), float(corr[i, j])) for i, j in zip(rows, cols)], ""


def _component_stability(base_components, pert_components):
    matches, warning = _hungarian_matching(base_components, pert_components)
    if not matches:
        return np.nan, np.nan, warning or "not_computable_empty_component_set"
    stability = float(np.mean([m[2] for m in matches]))
    threshold = float(TEMP_COMPONENT_STABILITY["matched_threshold_abs_corr"])
    matched_base = {i for i, _j, c in matches if c >= threshold}
    matched_pert = {j for _i, j, c in matches if c >= threshold}
    e_base = np.sum(base_components ** 2, axis=1) if base_components.size else np.asarray([])
    e_pert = np.sum(pert_components ** 2, axis=1) if pert_components.size else np.asarray([])
    total_base = float(np.sum(e_base))
    total_pert = float(np.sum(e_pert))
    if total_base <= EPS_ABSOLUTE and total_pert <= EPS_ABSOLUTE:
        return stability, np.nan, "not_computable_zero_component_energy"
    unmatched_base = float(np.sum([e_base[i] for i in range(len(e_base)) if i not in matched_base]))
    unmatched_pert = float(np.sum([e_pert[j] for j in range(len(e_pert)) if j not in matched_pert]))
    ratio_base = unmatched_base / max(total_base, EPS_ABSOLUTE)
    ratio_pert = unmatched_pert / max(total_pert, EPS_ABSOLUTE)
    return stability, float(0.5 * (ratio_base + ratio_pert)), warning


def _denom_ratio(num, local_ref, full_ref):
    floor = max(EPS_ABSOLUTE, RELATIVE_FLOOR * float(full_ref))
    denom = max(float(local_ref), floor)
    low_flag = bool(float(local_ref) <= LOW_ENERGY_FLAG_RELATIVE_THRESHOLD * float(full_ref))
    return float(num / denom), low_flag


def _contamination_mask(n, fs):
    start = int(round(TEMP_CONTAMINATION["start_sec"] * fs))
    dur = int(round(TEMP_CONTAMINATION["duration_ms"] * fs / 1000.0))
    start = min(max(0, start), max(0, n - 1))
    end = min(n, start + max(1, dur))
    mask = np.zeros(n, dtype=bool)
    mask[start:end] = True
    radius = int(round(TEMP_CONTAMINATION["spillover_radius_ms"] * fs / 1000.0))
    neigh = np.zeros(n, dtype=bool)
    lo = max(0, start - radius)
    hi = min(n, end + radius)
    neigh[lo:hi] = True
    spill = neigh & ~mask
    return mask, spill


def _inject_contamination(y, fs):
    rng = np.random.default_rng(int(TEMP_CONTAMINATION["seed"]))
    y = np.asarray(y, dtype=float)
    mask, spill = _contamination_mask(len(y), fs)
    mad = float(np.median(np.abs(y - np.median(y))))
    scale = 3.0 * (mad if mad > 1e-12 else float(np.std(y) + EPS_ABSOLUTE))
    artifact = np.zeros_like(y)
    artifact[mask] = scale * rng.normal(size=int(np.sum(mask)))
    return y + artifact, mask, spill


def _contamination_metrics(x0, xc, mask, spill):
    diff = np.asarray(xc, dtype=float) - np.asarray(x0, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    full_ref = float(np.sum(x0 ** 2) + EPS_ABSOLUTE)
    clean = ~mask
    num_c = float(np.sum(diff[mask] ** 2)) if np.any(mask) else np.nan
    den_c = float(np.sum(x0[mask] ** 2)) if np.any(mask) else np.nan
    num_clean = float(np.sum(diff[clean] ** 2)) if np.any(clean) else np.nan
    den_clean = float(np.sum(x0[clean] ** 2)) if np.any(clean) else np.nan
    num_spill = float(np.sum(diff[spill] ** 2)) if np.any(spill) else np.nan
    den_spill = float(np.mean(x0[spill] ** 2) * max(1, int(np.sum(spill)))) if np.any(spill) else np.nan
    c_val, c_low = _denom_ratio(num_c, den_c, full_ref) if np.isfinite(num_c) else (np.nan, True)
    clean_val, clean_low = _denom_ratio(num_clean, den_clean, full_ref) if np.isfinite(num_clean) else (np.nan, True)
    spill_val, spill_low = _denom_ratio(num_spill, den_spill, full_ref) if np.isfinite(num_spill) else (np.nan, True)
    return {
        "paired_contaminated_region_reconstruction_deviation": c_val,
        "paired_contaminated_region_low_energy_flag": c_low,
        "paired_clean_region_reconstruction_deviation": clean_val,
        "paired_clean_region_low_energy_flag": clean_low,
        "normalized_paired_spillover_loss": spill_val,
        "normalized_paired_spillover_low_energy_flag": spill_low,
    }


def _run_method_bundle(method, y, fs, params, seed, timeout_seconds):
    t_unit = np.linspace(0.0, 1.0, len(y), endpoint=False)
    result, runtime = _run_method(method, y, t_unit, fs, params, seed, timeout_seconds)
    recon = _protocol_reconstruction(y, result, fs)
    components, _ = _candidate_components(result, y)
    return result, recon, components, float(runtime)


def run_real_world_v1_schema_single_window_smoke(
        output_root,
        data_root,
        params,
        record_id=DEFAULT_RECORD_ID,
        window_index=DEFAULT_WINDOW_INDEX,
        timeout_seconds=120,
):
    output_root = ensure_dir(output_root)
    y_raw, ann_samples, fs, lead_name, lead_warning = _load_mitbih_record(
        data_root,
        str(record_id),
        "MLII_if_available",
    )
    win = int(round(WINDOW_LENGTH_SEC * fs))
    start = int(window_index) * win
    end = min(len(y_raw), start + win)
    if end - start < win:
        raise ValueError("Requested ECG smoke window is shorter than the frozen 10-second smoke window.")
    y = _median_mad_normalize(y_raw[start:end])
    ann_rel = ann_samples[(ann_samples >= start) & (ann_samples < end)] - start
    micro_rng = np.random.default_rng(int(TEMP_COMPONENT_STABILITY["seed"]))
    mad = float(np.median(np.abs(y - np.median(y))))
    perturb_scale = 0.01 * (mad if mad > 1e-12 else float(np.std(y) + EPS_ABSOLUTE))
    y_delta = y + perturb_scale * micro_rng.normal(size=len(y))
    y_contam, contam_mask, spill_mask = _inject_contamination(y, fs)

    rows = []
    bundles = {}
    for method in METHODS:
        row = {
            "schema_smoke_id": SCHEMA_SMOKE_ID,
            "record_id": str(record_id),
            "window_index": int(window_index),
            "window_start_sample": int(start),
            "window_end_sample": int(end),
            "window_start_sec": float(start / fs),
            "window_end_sec": float(end / fs),
            "lead_name": lead_name,
            "lead_warning": lead_warning,
            "sampling_rate_hz": float(fs),
            "method": method,
            "smoke_only": True,
            "thresholds_locked": False,
            "performance_claims_authorized": False,
            "held_out_accessed": False,
            "reconstruction_rule_id": TEMP_RECONSTRUCTION_RULE["rule_id"],
            "rpeak_detector_id": TEMP_RPEAK_DETECTOR["detector_id"],
        }
        try:
            t0 = perf_counter()
            base_result, base_recon, base_components, base_runtime = _run_method_bundle(
                method, y, fs, params, 20260811, timeout_seconds
            )
            pert_result, pert_recon, pert_components, pert_runtime = _run_method_bundle(
                method, y_delta, fs, params, 20260812, timeout_seconds
            )
            contam_result, contam_recon, _contam_components, contam_runtime = _run_method_bundle(
                method, y_contam, fs, params, 20260813, timeout_seconds
            )
            stability, unmatched_ratio, stability_warning = _component_stability(
                base_components,
                pert_components,
            )
            count_base = _effective_count(base_components)
            count_pert = _effective_count(pert_components)
            count_instability = abs(count_pert - count_base) / max(count_base, 1)
            rpeak = _rpeak_metrics(base_recon["xhat"], ann_rel, fs)
            qrs_leak, qrs_low, qrs_reason = _qrs_locked_residual_leakage(
                y,
                base_recon["residual"],
                ann_rel,
                fs,
            )
            contam_metrics = _contamination_metrics(
                base_recon["xhat"],
                contam_recon["xhat"],
                contam_mask,
                spill_mask,
            )
            row.update(rpeak)
            row.update({
                "perturbation_matched_component_stability": stability,
                "perturbation_component_stability_warning": stability_warning,
                "effective_component_count_baseline": int(count_base),
                "effective_component_count_perturbed": int(count_pert),
                "effective_component_count_instability": float(count_instability),
                "symmetric_unmatched_component_energy_ratio": unmatched_ratio,
                "qrs_locked_residual_leakage": qrs_leak,
                "qrs_locked_residual_leakage_low_energy_flag": qrs_low,
                "qrs_locked_residual_leakage_reason_code": qrs_reason,
                "component_candidate_count": base_recon["component_candidate_count"],
                "selected_component_count": base_recon["selected_component_count"],
                "selected_component_indices": base_recon["selected_component_indices"],
                "selection_status_code": base_recon["selection_status_code"],
                "residual_or_trend_candidate_included": base_recon["residual_or_trend_candidate_included"],
                "baseline_runtime_sec": base_runtime,
                "perturbed_runtime_sec": pert_runtime,
                "contaminated_runtime_sec": contam_runtime,
                "total_runtime_sec": float(perf_counter() - t0),
                "run_status": "passed",
                "failure_reason": "",
            })
            row.update(contam_metrics)
            bundles[method] = {
                "xhat": base_recon["xhat"],
                "residual": base_recon["residual"],
                "y": y,
            }
        except Exception as exc:
            row.update({
                "run_status": "failed",
                "failure_reason": str(exc),
            })
        rows.append(row)

    status = {
        "schema_smoke_id": SCHEMA_SMOKE_ID,
        "record_id": str(record_id),
        "window_index": int(window_index),
        "development_record": str(record_id) == DEFAULT_RECORD_ID,
        "smoke_only": True,
        "thresholds_locked": False,
        "performance_claims_authorized": False,
        "held_out_accessed": False,
        "n_methods": len(METHODS),
        "passed_method_runs": int(sum(1 for row in rows if row.get("run_status") == "passed")),
        "failed_method_runs": int(sum(1 for row in rows if row.get("run_status") != "passed")),
        "reference_annotation_count": int(len(ann_rel)),
        "reconstruction_rule": TEMP_RECONSTRUCTION_RULE,
        "rpeak_detector": TEMP_RPEAK_DETECTOR,
        "component_stability_perturbation": TEMP_COMPONENT_STABILITY,
        "controlled_contamination": TEMP_CONTAMINATION,
        "interpretation_boundary": (
            "Single development-window engineering smoke for endpoint "
            "computability.  Uses temporary thresholds and does not authorize "
            "real-world performance claims."
        ),
    }
    status["smoke_status"] = "passed" if status["failed_method_runs"] == 0 else "failed"
    write_csv(rows, output_root / "real_world_v1_schema_single_window_smoke.csv")
    write_json(rows, output_root / "real_world_v1_schema_single_window_smoke.json")
    write_json(status, output_root / "real_world_v1_schema_single_window_smoke_status.json")
    return status

