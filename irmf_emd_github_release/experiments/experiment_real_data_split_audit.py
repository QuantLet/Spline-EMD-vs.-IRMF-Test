#!/usr/bin/python
# coding: UTF-8

"""Section 8 split audit.

This audit reads the pre-registered split registry only.  It does not read
held-out waveform contents, does not run any decomposition algorithm, and does
not produce performance metrics.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json

from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


VALIDATION_STATUS_VALUES = ("passed", "failed", "not_run")


def _read_list_after(lines, key):
    """Extract a simple YAML list under a key from the scaffold registry."""
    prefix = f"{key}:"
    for idx, line in enumerate(lines):
        if line.strip() != prefix:
            continue
        out = []
        base_indent = len(line) - len(line.lstrip(" "))
        for follow in lines[idx + 1:]:
            stripped = follow.strip()
            if not stripped:
                continue
            indent = len(follow) - len(follow.lstrip(" "))
            if indent <= base_indent:
                break
            if stripped.startswith("- "):
                out.append(stripped[2:].strip().strip('"').strip("'"))
        return out
    return None


def _read_scalar_after(lines, key):
    prefix = f"{key}:"
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(prefix):
            value = stripped[len(prefix):].strip()
            return value.strip('"').strip("'")
    return None


def _default_subject_map(records):
    """Conservative default: each record is treated as its own subject.

    MIT-BIH subject identity is not fully derivable from record IDs alone.
    Therefore this map is sufficient to compute record overlap, but not
    sufficient to pass subject-overlap verification.
    """
    return {str(record): f"UNKNOWN_SUBJECT_FOR_RECORD_{record}" for record in records}


def _load_subject_map(path, records):
    if path is None:
        return _default_subject_map(records), False, "subject_mapping_missing"
    path = Path(path)
    if not path.exists():
        return _default_subject_map(records), False, "subject_mapping_file_not_found"
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            mapping = {str(k): str(v) for k, v in data.items()}
        elif path.suffix.lower() in {".yaml", ".yml"}:
            mapping = {}
            in_record_map = False
            for raw in path.read_text(encoding="utf-8").splitlines():
                if not raw.strip() or raw.lstrip().startswith("#"):
                    continue
                stripped = raw.strip()
                if stripped == "record_to_subject:":
                    in_record_map = True
                    continue
                if in_record_map:
                    indent = len(raw) - len(raw.lstrip(" "))
                    if indent == 0 and not stripped.startswith(('"', "'")):
                        break
                    if ":" not in stripped:
                        continue
                    key, value = stripped.split(":", 1)
                    key = key.strip().strip('"').strip("'")
                    value = value.strip().strip('"').strip("'")
                    if key:
                        if key in mapping and mapping[key] != value:
                            return _default_subject_map(records), False, "record_maps_to_multiple_subjects"
                        mapping[key] = value
        else:
            mapping = {}
            for raw in path.read_text(encoding="utf-8").splitlines():
                if not raw.strip() or raw.lstrip().startswith("#"):
                    continue
                parts = [p.strip() for p in raw.replace("\t", ",").split(",")]
                if len(parts) >= 2 and parts[0].lower() not in {"record", "record_id"}:
                    mapping[str(parts[0])] = str(parts[1])
    except Exception as exc:
        return _default_subject_map(records), False, f"subject_mapping_read_error:{exc}"
    missing = [str(record) for record in records if str(record) not in mapping]
    if missing:
        fallback = _default_subject_map(records)
        fallback.update(mapping)
        return fallback, False, "subject_mapping_incomplete"
    return mapping, True, None


def _duplicate_count(items):
    seen = set()
    dup = set()
    for item in items:
        if item in seen:
            dup.add(item)
        seen.add(item)
    return len(dup), sorted(dup)


def _update_protocol_status_text(text, report, audit_report_rel):
    """Patch the Stage-2 protocol_status.yaml split-validation audit block."""
    replacements = {
        "record_overlap_count: null": f"record_overlap_count: {report['record_overlap_count']}",
        "duplicate_window_count: null": f"duplicate_window_count: {report['duplicate_window_count']}",
        "validation_status: not_run": f"validation_status: {report['validation_status']}",
        'reason: "Local MIT-BIH/CWRU data audit has not been executed at Stage 2."':
            f'reason: "Split audit executed; see {audit_report_rel}."',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Subject overlap cannot be passed without a complete record->subject map.
    text = text.replace("subject_overlap_count: null", "subject_overlap_count: null")
    return text


def _checklist_update_rows(checklist_rows, report, report_rel):
    out = []
    for row in checklist_rows:
        row = dict(row)
        item = row.get("check_item")
        if item == "Record overlap verified":
            row["status"] = "complete" if report["record_overlap_count"] == 0 else "pending"
            row["evidence_file"] = report_rel
            row["notes"] = (
                "Record overlap audit passed."
                if report["record_overlap_count"] == 0
                else "Record overlap detected; split registry must be corrected."
            )
        elif item == "Subject overlap verified":
            subject_passed = (
                report["subject_mapping_complete"]
                and report["subject_overlap_count"] == 0
            )
            row["status"] = "complete" if subject_passed else "pending"
            row["evidence_file"] = report_rel
            row["notes"] = (
                "Subject overlap audit passed."
                if subject_passed
                else "Pending: complete record-to-subject mapping is required."
            )
        out.append(row)
    return out


def _read_csv_rows(path):
    import csv
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def run_real_data_split_audit(
        output_root,
        protocol_root,
        subject_map_path=None,
):
    output_root = ensure_dir(output_root)
    protocol_root = Path(protocol_root)
    registry_path = protocol_root / "algorithm" / "15a_real_data_protocol" / "split_registry.yaml"
    if not registry_path.exists():
        registry_path = protocol_root / "15a_real_data_protocol" / "split_registry.yaml"
    if not registry_path.exists():
        registry_path = protocol_root / "split_registry.yaml"
    audit_timestamp = datetime.now(timezone.utc).isoformat()
    failure_reasons = []
    dev = test = None
    protocol_id = protocol_version = None
    if registry_path.exists():
        lines = registry_path.read_text(encoding="utf-8").splitlines()
        dev = _read_list_after(lines, "development_records")
        test = _read_list_after(lines, "held_out_test_records")
        protocol_id = _read_scalar_after(lines, "id")
        protocol_version = _read_scalar_after(lines, "version") or _read_scalar_after(lines, "protocol_version")
    else:
        lines = []
        failure_reasons.append("registry_missing")
    if dev is None:
        failure_reasons.append("development_records_missing")
        dev = []
    if test is None:
        failure_reasons.append("held_out_test_records_missing")
        test = []
    dev = [str(x) for x in dev]
    test = [str(x) for x in test]
    record_overlap_ids = sorted(set(dev).intersection(test))
    if record_overlap_ids:
        failure_reasons.append("record_overlap_detected")
    dev_dups_count, dev_dups = _duplicate_count(dev)
    test_dups_count, test_dups = _duplicate_count(test)
    duplicate_window_ids = sorted(set(dev_dups + test_dups))
    duplicate_window_count = int(dev_dups_count + test_dups_count)
    if duplicate_window_count > 0:
        failure_reasons.append("duplicate_record_ids_detected")
    all_records = sorted(set(dev + test))
    subject_map, subject_mapping_complete, subject_map_reason = _load_subject_map(subject_map_path, all_records)
    dev_subject_ids = sorted({subject_map.get(record) for record in dev if subject_map.get(record) is not None})
    test_subject_ids = sorted({subject_map.get(record) for record in test if subject_map.get(record) is not None})
    subject_overlap_ids = sorted(set(dev_subject_ids).intersection(test_subject_ids))
    if not subject_mapping_complete:
        failure_reasons.append(subject_map_reason or "subject_mapping_incomplete")
    elif subject_overlap_ids:
        failure_reasons.append("subject_overlap_detected")
    validation_status = "passed" if not failure_reasons else "failed"
    report = {
        "protocol_id": protocol_id or "UNKNOWN_PROTOCOL_ID",
        "protocol_version": protocol_version or "UNKNOWN_PROTOCOL_VERSION",
        "registry_path": str(registry_path),
        "audit_timestamp": audit_timestamp,
        "audit_scope": "split_registry_only_no_waveform_access",
        "development_records": dev,
        "held_out_records": test,
        "record_overlap_ids": record_overlap_ids,
        "record_overlap_count": int(len(record_overlap_ids)),
        "development_subject_ids": dev_subject_ids,
        "held_out_subject_ids": test_subject_ids,
        "subject_overlap_ids": subject_overlap_ids if subject_mapping_complete else [],
        "subject_overlap_count": int(len(subject_overlap_ids)) if subject_mapping_complete else None,
        "subject_mapping_complete": bool(subject_mapping_complete),
        "subject_mapping_source": str(subject_map_path) if subject_map_path else None,
        "duplicate_window_ids": duplicate_window_ids,
        "duplicate_window_count": int(duplicate_window_count),
        "validation_status": validation_status,
        "allowed_validation_status_values": list(VALIDATION_STATUS_VALUES),
        "failure_reasons": failure_reasons,
    }
    write_json(report, output_root / "split_audit_report.json")
    write_csv([report], output_root / "split_audit_report.csv")

    # Mirror a copy beside the protocol scaffold for paper-section indexing.
    protocol_stage_dir = registry_path.parent if registry_path.exists() else output_root
    write_json(report, protocol_stage_dir / "split_audit_report.json")
    write_csv([report], protocol_stage_dir / "split_audit_report.csv")

    status_path = protocol_stage_dir / "protocol_status.yaml"
    if status_path.exists():
        text = status_path.read_text(encoding="utf-8")
        rel = "split_audit_report.json"
        text = _update_protocol_status_text(text, report, rel)
        if subject_mapping_complete:
            text = text.replace("subject_overlap_count: null", f"subject_overlap_count: {len(subject_overlap_ids)}")
            text = text.replace("subject_overlap_verified: false", f"subject_overlap_verified: {str(len(subject_overlap_ids) == 0).lower()}")
        text = text.replace("record_overlap_verified: false", f"record_overlap_verified: {str(len(record_overlap_ids) == 0).lower()}")
        status_path.write_text(text, encoding="utf-8")
    checklist_path = protocol_stage_dir / "section_8_audit_checklist.csv"
    rows = _read_csv_rows(checklist_path)
    if rows:
        updated = _checklist_update_rows(
            rows,
            report,
            "split_audit_report.json",
        )
        write_csv(updated, checklist_path)
        write_json(updated, protocol_stage_dir / "section_8_audit_checklist.json")
    return report
