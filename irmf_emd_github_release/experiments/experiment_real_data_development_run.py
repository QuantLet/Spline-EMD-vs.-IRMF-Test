#!/usr/bin/python
# coding: UTF-8

"""Stage 3B development-record-only ECG execution audit.

This command accesses development records only. It validates loading,
annotation availability, preprocessing, windowing, and four-method numerical
execution. It does not lock thresholds, does not access held-out waveforms, and
does not produce method-superiority claims.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter
import importlib.metadata
import json
import platform
import subprocess
import sys
import numpy as np

from project_config import DEFAULT_METHOD_TIMEOUT_SECONDS
from core_algorithms.eemd_wrapper import run_eemd
from core_algorithms.ceemdan_wrapper import run_ceemdan
from experiments.experiment_real_data_split_audit import _read_list_after
from experiments.experiment_utils import (
    _run_with_timeout,
    run_fixed_irmf_case,
    run_fixed_emd_case,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


RUN_STATUS_VALUES = ("passed", "passed_with_warning", "failed", "skipped")
METHODS = ("IRMF", "EMD", "EEMD", "CEEMDAN")
PIPELINE_VERSION = "stage3b_development_execution_audit_v1"


def _protocol_stage_dir(protocol_root):
    protocol_root = Path(protocol_root)
    candidates = [
        protocol_root / "algorithm" / "15a_real_data_protocol",
        protocol_root / "15a_real_data_protocol",
        protocol_root,
    ]
    for path in candidates:
        if (path / "split_registry.yaml").exists():
            return path
    return candidates[0]


def _load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _load_split(protocol_root):
    stage_dir = _protocol_stage_dir(protocol_root)
    lines = (stage_dir / "split_registry.yaml").read_text(encoding="utf-8").splitlines()
    dev = _read_list_after(lines, "development_records") or []
    held = _read_list_after(lines, "held_out_test_records") or []
    return stage_dir, [str(x) for x in dev], [str(x) for x in held]


def _load_split_audit(protocol_root):
    protocol_root = Path(protocol_root)
    candidates = [
        protocol_root / "algorithm" / "15b_real_data_split_audit" / "split_audit_report.json",
        protocol_root / "15b_real_data_split_audit" / "split_audit_report.json",
        protocol_root / "algorithm" / "15a_real_data_protocol" / "split_audit_report.json",
        protocol_root / "15a_real_data_protocol" / "split_audit_report.json",
    ]
    for path in candidates:
        data = _load_json(path)
        if data is not None:
            return data, path
    return None, None


def _load_record_subject_map(protocol_stage_dir):
    path = Path(protocol_stage_dir) / "mitbih_record_subject_map.yaml"
    mapping = {}
    if not path.exists():
        return mapping
    in_record_map = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "record_to_subject:":
            in_record_map = True
            continue
        if not in_record_map or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip().strip('"').strip("'")
        value = value.strip().strip('"').strip("'")
        if key:
            mapping[key] = value
    return mapping


def _module_version(package_name, import_name=None):
    candidates = [package_name]
    if import_name and import_name != package_name:
        candidates.append(import_name)
    for name in candidates:
        try:
            return importlib.metadata.version(name)
        except Exception:
            continue
    try:
        module = __import__(import_name or package_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return "not_installed"


def _git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return "unavailable"


def _write_environment_manifest(output_root, protocol_id, protocol_version):
    manifest = {
        "protocol_id": protocol_id,
        "protocol_version": protocol_version,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy_version": _module_version("numpy"),
        "scipy_version": _module_version("scipy"),
        "wfdb_version": _module_version("wfdb"),
        "pyemd_version": _module_version("EMD-signal", "PyEMD"),
        "git_commit": _git_commit(),
    }
    write_json(manifest, Path(output_root) / "environment_manifest.json")
    return manifest


def _read_csv_rows(path):
    import csv
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_yaml_status(path, status):
    def yval(value):
        if isinstance(value, bool):
            return str(value).lower()
        return value
    status_yaml = "\n".join([
        "development_run_status:",
        f"  protocol_id: {status['protocol_id']}",
        f"  protocol_version: \"{status['protocol_version']}\"",
        "  split_audit_required: true",
        f"  split_audit_status: {status.get('split_audit_status')}",
        f"  development_records_expected: {status.get('development_records_expected', 0)}",
        f"  development_records_accessed: {yval(status.get('development_records_accessed', 0))}",
        f"  held_out_records_accessed: {yval(status.get('held_out_records_accessed', 0))}",
        f"  total_windows: {status.get('total_windows', 0)}",
        f"  completed_method_runs: {status.get('completed_method_runs', 0)}",
        f"  failed_method_runs: {status.get('failed_method_runs', 0)}",
        f"  warning_method_runs: {status.get('warning_method_runs', 0)}",
        f"  skipped_windows: {status.get('skipped_windows', 0)}",
        f"  development_run_failure_count: {status.get('development_run_failure_count', 0)}",
        f"  development_run_status: {status.get('development_run_status')}",
    ])
    Path(path).write_text(status_yaml + "\n", encoding="utf-8")


def _yaml_scalar(value):
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace('"', '\\"')
    return f'"{text}"'


def _to_yaml_lines(obj, indent=0):
    pad = " " * indent
    if isinstance(obj, dict):
        lines = []
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.extend(_to_yaml_lines(value, indent + 2))
            else:
                lines.append(f"{pad}{key}: {_yaml_scalar(value)}")
        return lines
    if isinstance(obj, list):
        if not obj:
            return [f"{pad}[]"]
        lines = []
        for value in obj:
            if isinstance(value, (dict, list)):
                lines.append(f"{pad}-")
                lines.extend(_to_yaml_lines(value, indent + 2))
            else:
                lines.append(f"{pad}- {_yaml_scalar(value)}")
        return lines
    return [f"{pad}{_yaml_scalar(obj)}"]


def _write_yaml(obj, path):
    Path(path).write_text("\n".join(_to_yaml_lines(obj)) + "\n", encoding="utf-8")


def _update_protocol_files(protocol_stage_dir, status, status_rel):
    protocol_stage_dir = Path(protocol_stage_dir)
    protocol_status_path = protocol_stage_dir / "protocol_status.yaml"
    run_completed = status.get("development_run_status") in {"passed", "passed_with_warning"}
    if protocol_status_path.exists():
        text = protocol_status_path.read_text(encoding="utf-8")
        if "development_record_execution_status:" in text:
            text = text.replace(
                "development_record_execution_status: not_started",
                f"development_record_execution_status: {status.get('development_run_status')}",
            )
        else:
            text = text.replace(
                "operational_protocol:\n    status: pending_development_lock",
                (
                    "operational_protocol:\n"
                    "    status: pending_development_lock\n"
                    f"    development_record_execution_status: {status.get('development_run_status')}"
                ),
            )
        if "development_record_execution_completed:" in text:
            text = text.replace(
                "development_record_execution_completed: false",
                f"development_record_execution_completed: {str(run_completed).lower()}",
            )
        else:
            text = text.replace(
                f"development_record_execution_status: {status.get('development_run_status')}",
                (
                    f"development_record_execution_status: {status.get('development_run_status')}\n"
                    f"    development_record_execution_completed: {str(run_completed).lower()}"
                ),
            )
        protocol_status_path.write_text(text, encoding="utf-8")

    checklist_path = protocol_stage_dir / "section_8_audit_checklist.csv"
    rows = _read_csv_rows(checklist_path)
    if rows:
        updated = []
        found = False
        for row in rows:
            row = dict(row)
            if row.get("check_item") == "Development-record execution completed":
                found = True
                row["status"] = "complete" if run_completed else "pending"
                row["evidence_file"] = status_rel
                row["notes"] = (
                    "Stage 3B development-record execution audit completed."
                    if run_completed else
                    "Stage 3B attempted but did not pass; see development-run status."
                )
            updated.append(row)
        if not found:
            updated.append({
                "audit_category": "Execution Audit",
                "check_item": "Development-record execution completed",
                "status": "complete" if run_completed else "pending",
                "evidence_file": status_rel,
                "notes": (
                    "Stage 3B development-record execution audit completed."
                    if run_completed else
                    "Stage 3B attempted but did not pass; see development-run status."
                ),
            })
        write_csv(updated, checklist_path)
        write_json(updated, protocol_stage_dir / "section_8_audit_checklist.json")


def _median_mad_normalize(y):
    y = np.asarray(y, dtype=float)
    med = float(np.nanmedian(y))
    mad = float(np.nanmedian(np.abs(y - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.nanstd(y))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return (y - med) / scale


def _identity_error(y, imfs, residual):
    if imfs is None or residual is None:
        return np.nan
    y = np.asarray(y, dtype=float)
    imfs = np.asarray(imfs, dtype=float)
    residual = np.asarray(residual, dtype=float)
    recon = np.sum(imfs, axis=0) + residual if imfs.size else residual
    if recon.shape != y.shape:
        return np.nan
    return float(np.sum((recon - y) ** 2) / (np.sum(y ** 2) + 1e-12))


def _component_count(imfs):
    try:
        return int(np.asarray(imfs).shape[0])
    except Exception:
        return 0


def _all_finite(*arrays):
    for array in arrays:
        try:
            arr = np.asarray(array, dtype=float)
            if arr.size == 0 or not np.all(np.isfinite(arr)):
                return False
        except Exception:
            return False
    return True


def _load_mitbih_record(data_root, record_id, lead_rule):
    try:
        import wfdb
    except Exception as exc:
        raise ImportError("wfdb is required to read MIT-BIH .dat/.hea/.atr records") from exc
    base = Path(data_root) / str(record_id)
    record = wfdb.rdrecord(str(base))
    ann = wfdb.rdann(str(base), "atr")
    fs = float(record.fs)
    names = list(getattr(record, "sig_name", []) or [])
    if lead_rule == "MLII_if_available" and "MLII" in names:
        lead_index = names.index("MLII")
        lead_name = "MLII"
        lead_warning = None
    else:
        lead_index = 0
        lead_name = names[0] if names else "channel_0"
        lead_warning = "target_lead_unavailable_fallback_to_first_channel"
    y = np.asarray(record.p_signal[:, lead_index], dtype=float)
    samples = np.asarray(ann.sample, dtype=int)
    return y, samples, fs, lead_name, lead_warning


def _record_file_check(data_root, record_id):
    data_root = Path(data_root)
    return {
        "hea": (data_root / f"{record_id}.hea").exists(),
        "dat": (data_root / f"{record_id}.dat").exists(),
        "atr": (data_root / f"{record_id}.atr").exists(),
    }


def _dataset_integrity_check(
        data_root,
        all_records,
        development_records,
        held_out_records,
        fs_expected,
):
    """Check dataset readiness without reading held-out waveform contents."""
    data_root = Path(data_root)
    missing_records = []
    missing_files = []
    unreadable_records = []
    unreadable_annotations = []
    sampling_rate_mismatch_records = []
    readable_development_records = []
    for record_id in all_records:
        files = _record_file_check(data_root, record_id)
        missing = [ext for ext, exists in files.items() if not exists]
        if missing:
            missing_records.append(str(record_id))
            missing_files.append({
                "record_id": str(record_id),
                "missing_extensions": missing,
            })
    for record_id in development_records:
        if str(record_id) in set(missing_records):
            continue
        try:
            y_raw, ann_samples, fs, _lead_name, _lead_warning = _load_mitbih_record(
                data_root,
                record_id,
                "MLII_if_available",
            )
            if len(y_raw) <= 0:
                unreadable_records.append(str(record_id))
            if ann_samples is None or len(ann_samples) <= 0:
                unreadable_annotations.append(str(record_id))
            if abs(float(fs) - float(fs_expected)) > 1e-9:
                sampling_rate_mismatch_records.append(str(record_id))
            readable_development_records.append(str(record_id))
        except ImportError as exc:
            unreadable_records.append(f"{record_id}:{exc}")
            break
        except Exception as exc:
            unreadable_records.append(f"{record_id}:{exc}")
    status = "passed"
    failure_reasons = []
    if missing_records:
        status = "failed"
        failure_reasons.append("missing_record_files")
    if unreadable_records:
        status = "failed"
        failure_reasons.append("development_waveform_unreadable")
    if unreadable_annotations:
        status = "failed"
        failure_reasons.append("development_annotation_unreadable")
    if sampling_rate_mismatch_records:
        status = "failed"
        failure_reasons.append("sampling_rate_mismatch")
    report = {
        "status": status,
        "scope": (
            "file_presence_checked_for_all_preregistered_records; "
            "waveform_annotation_readability_checked_for_development_records_only"
        ),
        "held_out_content_read": False,
        "held_out_readability_check_deferred_until_stage_4": True,
        "total_records_expected": int(len(all_records)),
        "total_records_found": int(len(all_records) - len(set(missing_records))),
        "development_records_expected": int(len(development_records)),
        "development_records_readable": int(len(set(readable_development_records))),
        "held_out_records_expected": int(len(held_out_records)),
        "missing_records": sorted(set(missing_records)),
        "missing_files": missing_files,
        "unreadable_records": unreadable_records,
        "unreadable_annotations": unreadable_annotations,
        "sampling_rate_expected_hz": float(fs_expected),
        "sampling_rate_mismatch_records": sampling_rate_mismatch_records,
        "failure_reasons": failure_reasons,
    }
    return report


def _write_development_access_log(output_root, rows):
    write_csv(rows, Path(output_root) / "development_access_log.csv")
    write_json(rows, Path(output_root) / "development_access_log.json")


def _make_windows(n_samples, fs, window_length_sec):
    win = int(round(float(window_length_sec) * float(fs)))
    if win <= 0:
        return []
    starts = list(range(0, n_samples - win + 1, win))
    return [(idx, start, start + win) for idx, start in enumerate(starts)]


def _run_method(method, y, t_unit, fs, params, seed, timeout_seconds):
    start = perf_counter()
    if method == "IRMF":
        result = _run_with_timeout(
            lambda: run_fixed_irmf_case(
                Y=y,
                X_clean=None,
                t=t_unit,
                fs=fs,
                irmf_params=params["IRMF"],
                true_components=None,
                run_id="real_data_development",
            ),
            timeout_seconds=timeout_seconds,
        )
    elif method == "EMD":
        result = _run_with_timeout(
            lambda: run_fixed_emd_case(
                Y=y,
                X_clean=None,
                t=t_unit,
                fs=fs,
                emd_params=params["EMD"],
                true_components=None,
                run_id="real_data_development",
            ),
            timeout_seconds=timeout_seconds,
        )
    elif method == "EEMD":
        eemd = params["EEMD"]
        result = _run_with_timeout(
            lambda: run_eemd(
                y,
                max_imf=eemd.get("max_imf", -1),
                trials=eemd.get("trials", 100),
                noise_width=eemd.get("noise_width", 0.05),
                parallel=eemd.get("parallel", False),
                random_seed=seed,
            ),
            timeout_seconds=timeout_seconds,
        )
    elif method == "CEEMDAN":
        ceemdan = params["CEEMDAN"]
        result = _run_with_timeout(
            lambda: run_ceemdan(
                y,
                max_imf=ceemdan.get("max_imf", -1),
                trials=ceemdan.get("trials", 100),
                epsilon=ceemdan.get("epsilon", 0.005),
                parallel=ceemdan.get("parallel", False),
                random_seed=seed,
            ),
            timeout_seconds=timeout_seconds,
        )
    else:
        raise ValueError(method)
    runtime = perf_counter() - start
    return result, runtime


def _method_audit_row(base, method, result=None, runtime_sec=None, error=None, seed=None, params=None):
    row = dict(base)
    row["method"] = method
    row["random_seed"] = None if seed is None else int(seed)
    row["effective_seed"] = None if seed is None else int(seed)
    row["run_status"] = "failed" if error else "passed"
    row["failure_reason"] = None if error is None else str(error)
    row["warning_codes"] = ""
    row["runtime_sec"] = None if runtime_sec is None else float(runtime_sec)
    row["residual_available"] = False
    row["component_count"] = 0
    row["reconstruction_identity_error"] = np.nan
    row["boundary_status"] = "not_evaluated"
    row["ensemble_size"] = None
    row["noise_amplitude"] = None
    if result is not None and not error:
        imfs = result.get("imfs")
        residual = result.get("residual")
        row["residual_available"] = residual is not None
        row["component_count"] = _component_count(imfs)
        row["reconstruction_identity_error"] = _identity_error(base["preprocessed_signal"], imfs, residual)
        finite_ok = _all_finite(imfs, residual)
        warnings = []
        if row["component_count"] <= 0:
            warnings.append("zero_components")
        if not finite_ok:
            warnings.append("nonfinite_output")
        if np.isfinite(row["reconstruction_identity_error"]) and row["reconstruction_identity_error"] > 1e-8:
            warnings.append("identity_error_above_tolerance")
        if method == "EEMD":
            row["ensemble_size"] = params["EEMD"].get("trials")
            row["noise_amplitude"] = params["EEMD"].get("noise_width")
        if method == "CEEMDAN":
            row["ensemble_size"] = params["CEEMDAN"].get("trials")
            row["noise_amplitude"] = params["CEEMDAN"].get("epsilon")
        if warnings:
            row["run_status"] = "passed_with_warning"
            row["warning_codes"] = ";".join(warnings)
        if not finite_ok:
            row["run_status"] = "failed"
            row["failure_reason"] = "nonfinite_output"
    row.pop("preprocessed_signal", None)
    return row


def _preflight_failed_report(
        output_root,
        protocol_id,
        protocol_version,
        reason,
        split_audit=None,
        protocol_stage_dir=None,
        development_records_expected=0,
        dataset_integrity=None,
        access_rows=None,
):
    _write_environment_manifest(output_root, protocol_id, protocol_version)
    _write_development_access_log(output_root, [] if access_rows is None else access_rows)
    if dataset_integrity is not None:
        write_json(dataset_integrity, output_root / "dataset_integrity_report.json")
        _write_yaml({"dataset_integrity": dataset_integrity}, output_root / "dataset_integrity_report.yaml")
    report = {
        "protocol_id": protocol_id,
        "protocol_version": protocol_version,
        "development_records_expected": int(development_records_expected),
        "development_records_accessed": 0,
        "held_out_records_accessed": 0,
        "environment_manifest_archived": True,
        "dataset_integrity_status": (
            None if dataset_integrity is None else dataset_integrity.get("status")
        ),
        "development_run_status": "failed",
        "failure_reason": reason,
        "split_audit_required": True,
        "split_audit_status": None if split_audit is None else split_audit.get("validation_status"),
        "total_windows": 0,
        "completed_method_runs": 0,
        "failed_method_runs": 0,
        "warning_method_runs": 0,
        "skipped_windows": 0,
        "stage_3b_exit_criterion": {
            "split_audit_passed": bool(split_audit and split_audit.get("validation_status") == "passed"),
            "dataset_integrity_passed": bool(dataset_integrity and dataset_integrity.get("status") == "passed"),
            "loader_passed": False,
            "annotation_parser_passed": False,
            "four_methods_executed": False,
            "no_held_out_access": True,
            "numerical_validity_checks_passed": False,
            "environment_manifest_archived": True,
            "stage_3b_complete": False,
        },
    }
    write_json(report, output_root / "development_run_status.json")
    _write_yaml_status(output_root / "development_run_status.yaml", report)
    write_csv([], output_root / "development_run_audit.csv")
    write_json([], output_root / "development_run_audit.json")
    if protocol_stage_dir is not None:
        _update_protocol_files(
            protocol_stage_dir,
            report,
            "development_run_status.json",
        )
    return report


def run_real_data_development_run(
        output_root,
        protocol_root,
        data_root,
        params,
        algorithm_seed=20260724,
        timeout_seconds=DEFAULT_METHOD_TIMEOUT_SECONDS,
        max_windows_per_record=None,
):
    output_root = ensure_dir(output_root)
    protocol_stage_dir, dev_records, held_out_records = _load_split(protocol_root)
    record_subject_map = _load_record_subject_map(protocol_stage_dir)
    split_audit, split_audit_path = _load_split_audit(protocol_root)
    locked_rule = _load_json(protocol_stage_dir / "locked_reconstruction_rule.json", default={})
    protocol = locked_rule.get("protocol", {})
    protocol_id = protocol.get("id", "SECTION8_REALDATA_V1.0")
    protocol_version = protocol.get("version", "1.0")
    _write_environment_manifest(output_root, protocol_id, protocol_version)
    if split_audit is None:
        return _preflight_failed_report(
            output_root,
            protocol_id,
            protocol_version,
            "split_audit_missing",
            protocol_stage_dir=protocol_stage_dir,
            development_records_expected=len(dev_records),
        )
    if split_audit.get("validation_status") != "passed":
        return _preflight_failed_report(
            output_root,
            protocol_id,
            protocol_version,
            "split_audit_not_passed",
            split_audit=split_audit,
            protocol_stage_dir=protocol_stage_dir,
            development_records_expected=len(dev_records),
        )
    data_root = Path(data_root) if data_root is not None else None
    if data_root is None or not data_root.exists():
        return _preflight_failed_report(
            output_root,
            protocol_id,
            protocol_version,
            "data_root_missing",
            split_audit=split_audit,
            protocol_stage_dir=protocol_stage_dir,
            development_records_expected=len(dev_records),
        )
    requested = set(dev_records)
    held_out_violation = sorted(requested.intersection(held_out_records))
    if held_out_violation:
        raise PermissionError(
            "held-out record access is prohibited before Stage 3 operational locking; "
            f"violating IDs: {held_out_violation}"
        )
    fs_expected = 360.0
    all_records = sorted(set(dev_records + held_out_records))
    dataset_integrity = _dataset_integrity_check(
        data_root,
        all_records,
        dev_records,
        held_out_records,
        fs_expected,
    )
    write_json(dataset_integrity, output_root / "dataset_integrity_report.json")
    _write_yaml({"dataset_integrity": dataset_integrity}, output_root / "dataset_integrity_report.yaml")
    if dataset_integrity.get("status") != "passed":
        denied_access_rows = [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "protocol_id": protocol_id,
                "protocol_version": protocol_version,
                "record_id": record_id,
                "subject_id": record_subject_map.get(str(record_id)),
                "split": "development",
                "user_command": "real-data-development-run",
                "purpose": "stage_3b_development_execution_audit",
                "status": "denied",
                "reason": "dataset_integrity_failed",
            }
            for record_id in dev_records
        ]
        return _preflight_failed_report(
            output_root,
            protocol_id,
            protocol_version,
            "dataset_integrity_failed",
            split_audit=split_audit,
            protocol_stage_dir=protocol_stage_dir,
            development_records_expected=len(dev_records),
            dataset_integrity=dataset_integrity,
            access_rows=denied_access_rows,
        )
    rows = []
    access_rows = []
    record_failures = []
    window_length_sec = 10.0
    lead_rule = "MLII_if_available"
    held_out_accessed = False
    for record_id in dev_records:
        access_base = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "protocol_id": protocol_id,
            "protocol_version": protocol_version,
            "record_id": record_id,
            "subject_id": record_subject_map.get(str(record_id)),
            "split": "development",
            "user_command": "real-data-development-run",
            "purpose": "stage_3b_development_execution_audit",
        }
        files = _record_file_check(data_root, record_id)
        if not all(files.values()):
            record_failures.append(f"missing_files_for_record_{record_id}")
            item = dict(access_base)
            item["status"] = "failed"
            item["reason"] = "missing_files"
            access_rows.append(item)
            continue
        try:
            y_raw, ann_samples, fs, lead_name, lead_warning = _load_mitbih_record(
                data_root, record_id, lead_rule
            )
        except Exception as exc:
            record_failures.append(f"loader_error_{record_id}:{exc}")
            item = dict(access_base)
            item["status"] = "failed"
            item["reason"] = f"loader_error:{exc}"
            access_rows.append(item)
            continue
        item = dict(access_base)
        item["status"] = "granted"
        item["reason"] = "development_record_access"
        access_rows.append(item)
        fs_warning = None if abs(float(fs) - fs_expected) < 1e-9 else "unexpected_sampling_rate"
        windows = _make_windows(len(y_raw), fs, window_length_sec)
        if max_windows_per_record is not None:
            windows = windows[:int(max_windows_per_record)]
        for window_idx, start, end in windows:
            y = y_raw[start:end]
            subject_id = record_subject_map.get(str(record_id))
            valid_fraction = float(np.mean(np.isfinite(y))) if len(y) else 0.0
            ann_count = int(np.sum((ann_samples >= start) & (ann_samples < end)))
            if len(y) == 0 or valid_fraction < 1.0:
                base = {
                    "protocol_id": protocol_id,
                    "protocol_version": protocol_version,
                    "record_id": record_id,
                    "subject_id": subject_id,
                    "window_id": f"{record_id}_{window_idx:04d}",
                    "window_start_sample": int(start),
                    "window_end_sample": int(end),
                    "window_start_sec": float(start / fs),
                    "window_end_sec": float(end / fs),
                    "lead_name": lead_name,
                    "sampling_rate_hz": float(fs),
                    "annotation_count": ann_count,
                    "valid_sample_fraction": valid_fraction,
                    "preprocessing_status": "skipped_invalid_window",
                    "preprocessed_signal": np.asarray([], dtype=float),
                }
                for method in METHODS:
                    row = _method_audit_row(base, method, error="invalid_window", seed=algorithm_seed, params=params)
                    row["run_status"] = "skipped"
                    rows.append(row)
                continue
            y_norm = _median_mad_normalize(y)
            t_unit = np.linspace(0.0, 1.0, len(y_norm), endpoint=False)
            warning_codes = [w for w in (lead_warning, fs_warning) if w]
            base = {
                "protocol_id": protocol_id,
                "protocol_version": protocol_version,
                "record_id": record_id,
                "subject_id": subject_id,
                "window_id": f"{record_id}_{window_idx:04d}",
                "window_start_sample": int(start),
                "window_end_sample": int(end),
                "window_start_sec": float(start / fs),
                "window_end_sec": float(end / fs),
                "lead_name": lead_name,
                "sampling_rate_hz": float(fs),
                "annotation_count": ann_count,
                "valid_sample_fraction": valid_fraction,
                "preprocessing_status": "passed",
                "preprocessing_warning_codes": ";".join(warning_codes),
                "preprocessed_signal": y_norm,
            }
            for method in METHODS:
                try:
                    result, runtime = _run_method(
                        method, y_norm, t_unit, fs, params, algorithm_seed, timeout_seconds
                    )
                    rows.append(_method_audit_row(
                        base, method, result=result, runtime_sec=runtime,
                        seed=algorithm_seed, params=params,
                    ))
                except Exception as exc:
                    rows.append(_method_audit_row(
                        base, method, error=exc, seed=algorithm_seed, params=params,
                    ))
    _write_development_access_log(output_root, access_rows)
    write_csv(rows, output_root / "development_run_audit.csv")
    write_json(rows, output_root / "development_run_audit.json")
    total_windows = len({row["window_id"] for row in rows}) if rows else 0
    failed = [row for row in rows if row.get("run_status") == "failed"]
    warnings = [row for row in rows if row.get("run_status") == "passed_with_warning"]
    skipped = [row for row in rows if row.get("run_status") == "skipped"]
    completed = [row for row in rows if row.get("run_status") in {"passed", "passed_with_warning"}]
    critical_failure = bool(record_failures)
    development_run_status = "passed"
    if critical_failure or failed:
        development_run_status = "failed"
    elif warnings or skipped:
        development_run_status = "passed_with_warning"
    loader_passed = len(record_failures) == 0 and len(dev_records) > 0
    annotation_parser_passed = bool(rows) and all(
        row.get("annotation_count") is not None for row in rows
    )
    four_methods_executed = bool(rows) and set(row.get("method") for row in rows) == set(METHODS)
    numerical_validity_checks_passed = not failed
    stage_3b_exit_criterion = {
        "split_audit_passed": split_audit.get("validation_status") == "passed",
        "dataset_integrity_passed": dataset_integrity.get("status") == "passed",
        "loader_passed": loader_passed,
        "annotation_parser_passed": annotation_parser_passed,
        "four_methods_executed": four_methods_executed,
        "no_held_out_access": held_out_accessed is False,
        "numerical_validity_checks_passed": numerical_validity_checks_passed,
        "environment_manifest_archived": (output_root / "environment_manifest.json").exists(),
    }
    stage_3b_exit_criterion["stage_3b_complete"] = all(stage_3b_exit_criterion.values())
    status = {
        "protocol_id": protocol_id,
        "protocol_version": protocol_version,
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "split_audit_required": True,
        "split_audit_status": split_audit.get("validation_status"),
        "split_audit_path": str(split_audit_path),
        "environment_manifest_path": str(output_root / "environment_manifest.json"),
        "environment_manifest_archived": True,
        "dataset_integrity_path": str(output_root / "dataset_integrity_report.json"),
        "dataset_integrity_status": dataset_integrity.get("status"),
        "development_access_log_path": str(output_root / "development_access_log.csv"),
        "development_records_expected": int(len(dev_records)),
        "development_records_accessed": int(len(dev_records) - len(record_failures)),
        "development_record_ids": dev_records,
        "development_subject_ids": sorted({
            record_subject_map.get(str(record_id))
            for record_id in dev_records
            if record_subject_map.get(str(record_id)) is not None
        }),
        "held_out_records_accessed": int(1 if held_out_accessed else 0),
        "held_out_access_violation_attempted": False,
        "total_windows": int(total_windows),
        "completed_method_runs": int(len(completed)),
        "failed_method_runs": int(len(failed)),
        "warning_method_runs": int(len(warnings)),
        "skipped_windows": int(len({row["window_id"] for row in skipped})),
        "development_run_failure_count": int(len(failed) + len(record_failures)),
        "record_loader_failures": record_failures,
        "run_status_values": list(RUN_STATUS_VALUES),
        "stage_3b_exit_criterion": stage_3b_exit_criterion,
        "development_run_status": development_run_status,
        "status_rule": (
            "passed only if held_out_records_accessed == 0, all expected "
            "development records are processed, no critical loader/annotation "
            "error occurs, and all successful outputs pass finite-value checks"
        ),
        "claim_boundary": (
            "Development-run audit checks execution feasibility only. It does "
            "not tune thresholds, rank methods, access held-out records, or "
            "support real-data performance claims."
        ),
    }
    write_json(status, output_root / "development_run_status.json")
    _write_yaml_status(output_root / "development_run_status.yaml", status)
    _update_protocol_files(
        protocol_stage_dir,
        status,
        "../15c_real_data_development_run/development_run_status.json",
    )
    return status
