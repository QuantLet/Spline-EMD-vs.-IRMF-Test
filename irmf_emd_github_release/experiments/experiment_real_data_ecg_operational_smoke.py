#!/usr/bin/python
# coding: UTF-8

"""Non-locking ECG operational smoke test on development records only.

This smoke test exercises the ECG downstream code path on a tiny development
slice.  It does not lock detector thresholds, does not access held-out records,
and does not authorize real-data performance claims.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np

from experiments.experiment_real_data_development_run import (
    METHODS,
    _load_split,
    _load_split_audit,
    _load_record_subject_map,
    _load_mitbih_record,
    _make_windows,
    _median_mad_normalize,
    _run_method,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


SMOKE_DETECTOR = {
    "detector_name": "temporary_find_peaks_on_proxy_reconstruction",
    "detector_threshold_normalized": 1.0,
    "minimum_peak_distance_ms": 200.0,
    "matching_tolerance_ms": 100.0,
    "qrs_window_before_ms": 80.0,
    "qrs_window_after_ms": 120.0,
    "threshold_status": "temporary_smoke_not_locked",
}


def _load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _proxy_reconstruction(y, result):
    residual = None if result is None else result.get("residual")
    if residual is None:
        return None
    residual = np.asarray(residual, dtype=float)
    y = np.asarray(y, dtype=float)
    if residual.shape != y.shape:
        return None
    return y - residual


def _detect_peaks(x, fs):
    try:
        from scipy.signal import find_peaks
    except Exception:
        return np.asarray([], dtype=int), "scipy_find_peaks_unavailable"
    x = np.asarray(x, dtype=float)
    if x.size == 0 or not np.all(np.isfinite(x)):
        return np.asarray([], dtype=int), "invalid_signal"
    distance = max(1, int(round(SMOKE_DETECTOR["minimum_peak_distance_ms"] * float(fs) / 1000.0)))
    peaks, _ = find_peaks(x, height=float(SMOKE_DETECTOR["detector_threshold_normalized"]), distance=distance)
    return np.asarray(peaks, dtype=int), ""


def _match_peaks(reference, detected, tolerance_samples):
    reference = [int(x) for x in reference]
    detected = [int(x) for x in detected]
    used = set()
    matches = []
    for ref in reference:
        best = None
        best_dist = None
        for idx, det in enumerate(detected):
            if idx in used:
                continue
            dist = abs(det - ref)
            if dist <= tolerance_samples and (best_dist is None or dist < best_dist):
                best = idx
                best_dist = dist
        if best is not None:
            used.add(best)
            matches.append((ref, detected[best], best_dist))
    return matches


def _qrs_corr(y, rec, matches, fs):
    before = int(round(SMOKE_DETECTOR["qrs_window_before_ms"] * float(fs) / 1000.0))
    after = int(round(SMOKE_DETECTOR["qrs_window_after_ms"] * float(fs) / 1000.0))
    corrs = []
    ratios = []
    y = np.asarray(y, dtype=float)
    rec = np.asarray(rec, dtype=float)
    for ref, _det, _dist in matches:
        lo = max(0, int(ref) - before)
        hi = min(len(y), int(ref) + after + 1)
        if hi - lo < 3:
            continue
        a = y[lo:hi]
        b = rec[lo:hi]
        if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
            continue
        c = float(np.corrcoef(a, b)[0, 1])
        if np.isfinite(c):
            corrs.append(c)
        ratios.append(float((np.max(np.abs(b)) + 1e-12) / (np.max(np.abs(a)) + 1e-12)))
    return (
        float(np.mean(corrs)) if corrs else np.nan,
        float(np.mean(ratios)) if ratios else np.nan,
    )


def _event_metrics(y, rec, ann_rel_samples, fs):
    peaks, detector_warning = _detect_peaks(rec, fs)
    tolerance = int(round(SMOKE_DETECTOR["matching_tolerance_ms"] * float(fs) / 1000.0))
    matches = _match_peaks(ann_rel_samples, peaks, tolerance)
    n_ref = len(ann_rel_samples)
    n_det = len(peaks)
    n_match = len(matches)
    sensitivity = float(n_match / n_ref) if n_ref else np.nan
    ppv = float(n_match / n_det) if n_det else np.nan
    f1 = float(2.0 * sensitivity * ppv / (sensitivity + ppv)) if np.isfinite(sensitivity) and np.isfinite(ppv) and (sensitivity + ppv) > 0 else np.nan
    timing = [1000.0 * dist / float(fs) for _ref, _det, dist in matches]
    qrs_corr, qrs_amp_ratio = _qrs_corr(y, rec, matches, fs)
    return {
        "detected_peak_count": int(n_det),
        "reference_annotation_count": int(n_ref),
        "matched_peak_count": int(n_match),
        "r_peak_sensitivity_smoke": sensitivity,
        "r_peak_ppv_smoke": ppv,
        "r_peak_f1_smoke": f1,
        "r_peak_timing_error_ms_mean_smoke": float(np.mean(timing)) if timing else np.nan,
        "qrs_morphology_corr_smoke": qrs_corr,
        "qrs_amplitude_ratio_smoke": qrs_amp_ratio,
        "detector_warning": detector_warning,
    }


def run_real_data_ecg_operational_smoke(
        output_root,
        protocol_root,
        data_root,
        params,
        algorithm_seed=20260724,
        timeout_seconds=120,
        max_windows_per_record=1,
):
    output_root = ensure_dir(output_root)
    protocol_stage_dir, dev_records, held_out_records = _load_split(protocol_root)
    record_subject_map = _load_record_subject_map(protocol_stage_dir)
    split_audit, split_audit_path = _load_split_audit(protocol_root)
    locked_rule = _load_json(protocol_stage_dir / "locked_reconstruction_rule.json", default={})
    protocol = locked_rule.get("protocol", {})
    protocol_id = protocol.get("id", "SECTION8_REALDATA_V1.0")
    protocol_version = protocol.get("version", "1.0")

    rows = []
    status = {
        "protocol_id": protocol_id,
        "protocol_version": protocol_version,
        "smoke_only": True,
        "thresholds_locked": False,
        "performance_claims_authorized": False,
        "held_out_accessed": False,
        "held_out_records_accessed": 0,
        "detector_policy": dict(SMOKE_DETECTOR),
        "split_audit_status": None if split_audit is None else split_audit.get("validation_status"),
        "split_audit_path": None if split_audit_path is None else str(split_audit_path),
        "development_records_expected": len(dev_records),
        "max_windows_per_record": max_windows_per_record,
    }
    if split_audit is None or split_audit.get("validation_status") != "passed":
        status.update({"smoke_status": "failed", "failure_reason": "split_audit_not_passed"})
        write_json(status, output_root / "ecg_operational_smoke_status.json")
        write_csv(rows, output_root / "ecg_operational_smoke_audit.csv")
        return status
    data_root = Path(data_root)
    if not data_root.exists():
        status.update({"smoke_status": "failed", "failure_reason": "data_root_missing"})
        write_json(status, output_root / "ecg_operational_smoke_status.json")
        write_csv(rows, output_root / "ecg_operational_smoke_audit.csv")
        return status
    if set(dev_records).intersection(held_out_records):
        raise PermissionError("held-out record access is prohibited in ECG operational smoke")

    for record_id in dev_records:
        y_raw, ann_samples, fs, lead_name, lead_warning = _load_mitbih_record(
            data_root,
            record_id,
            "MLII_if_available",
        )
        windows = _make_windows(len(y_raw), fs, 10.0)
        if max_windows_per_record is not None:
            windows = windows[:int(max_windows_per_record)]
        for window_idx, start, end in windows:
            y = y_raw[start:end]
            ann_rel = ann_samples[(ann_samples >= start) & (ann_samples < end)] - start
            y_norm = _median_mad_normalize(y)
            t_unit = np.linspace(0.0, 1.0, len(y_norm), endpoint=False)
            base = {
                "protocol_id": protocol_id,
                "protocol_version": protocol_version,
                "record_id": str(record_id),
                "subject_id": record_subject_map.get(str(record_id)),
                "window_id": f"{record_id}_{window_idx:04d}",
                "window_start_sample": int(start),
                "window_end_sample": int(end),
                "window_start_sec": float(start / fs),
                "window_end_sec": float(end / fs),
                "lead_name": lead_name,
                "sampling_rate_hz": float(fs),
                "reference_annotation_count": int(len(ann_rel)),
                "smoke_only": True,
                "thresholds_locked": False,
                "performance_claims_authorized": False,
                "reconstruction_policy": "temporary_proxy_reconstruction_y_minus_residual",
            }
            for method in METHODS:
                row = dict(base)
                row["method"] = method
                row["random_seed"] = int(algorithm_seed)
                try:
                    result, runtime = _run_method(method, y_norm, t_unit, fs, params, algorithm_seed, timeout_seconds)
                    rec = _proxy_reconstruction(y_norm, result)
                    if rec is None:
                        raise RuntimeError("proxy_reconstruction_unavailable")
                    metrics = _event_metrics(y_norm, rec, ann_rel, fs)
                    row.update(metrics)
                    row["component_count"] = int(np.asarray(result.get("imfs")).shape[0])
                    row["residual_available"] = result.get("residual") is not None
                    row["runtime_sec"] = float(runtime)
                    row["run_status"] = "passed"
                    row["failure_reason"] = ""
                except Exception as exc:
                    row["run_status"] = "failed"
                    row["failure_reason"] = str(exc)
                    row["runtime_sec"] = np.nan
                rows.append(row)

    failed = [row for row in rows if row.get("run_status") != "passed"]
    status.update({
        "smoke_status": "passed" if not failed else "failed",
        "development_records_accessed": int(len(dev_records)),
        "total_windows": int(len({row["window_id"] for row in rows})),
        "method_runs": int(len(rows)),
        "passed_method_runs": int(sum(1 for row in rows if row.get("run_status") == "passed")),
        "failed_method_runs": int(len(failed)),
        "held_out_accessed": False,
        "held_out_records_accessed": 0,
        "statistics_authorized": False,
    })
    write_json(status, output_root / "ecg_operational_smoke_status.json")
    write_csv(rows, output_root / "ecg_operational_smoke_audit.csv")
    write_json(rows, output_root / "ecg_operational_smoke_audit.json")
    return status
