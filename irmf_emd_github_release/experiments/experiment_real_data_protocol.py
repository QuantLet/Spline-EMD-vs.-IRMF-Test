#!/usr/bin/python
# coding: UTF-8

"""Sections 7-8 real-data locked-protocol scaffold.

This stage does not download MIT-BIH, NSTDB, or CWRU data and does not report
real-data performance.  It materializes the frozen real-data continuum protocol files
that must exist before development-record locking and held-out evaluation.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

from project_config import (
    DEFAULT_METHOD_TIMEOUT_SECONDS,
    GLOBAL_CEEMDAN_PARAMS,
    GLOBAL_EEMD_PARAMS,
    GLOBAL_EMD_PARAMS,
    GLOBAL_IRMF_PARAMS,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


PROTOCOL_ID = "SECTION8_REALDATA_V1.0"
PROTOCOL_VERSION = "1.0"
FROZEN_DATE = "2026-07-24"
ECG_PROPOSED_DEVELOPMENT_RECORDS = (
    "100", "101", "106", "108", "114", "118",
    "200", "203", "207", "213", "222", "233",
)
ECG_PROPOSED_HELD_OUT_RECORDS = (
    "102", "103", "104", "105", "107", "109", "111", "112", "113",
    "115", "116", "117", "119", "121", "122", "123", "124", "201",
    "202", "205", "208", "209", "210", "212", "214", "215", "217",
    "219", "220", "221", "223", "228", "230", "231", "232", "234",
)
MITBIH_RECORDS = (
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119",
    "121", "122", "123", "124", "200", "201", "202", "203", "205",
    "207", "208", "209", "210", "212", "213", "214", "215", "217",
    "219", "220", "221", "222", "223", "228", "230", "231", "232",
    "233", "234",
)


def _stable_hash(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _yaml_list(items, indent=4) -> str:
    pad = " " * indent
    return "\n".join(f"{pad}- {item}" for item in items)


def _write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _real_data_protocol_yaml() -> str:
    return f"""protocol:
  id: {PROTOCOL_ID}
  version: "{PROTOCOL_VERSION}"
  frozen_date: {FROZEN_DATE}
  maturity_stage: 2
  maturity_stage_total: 4

protocol_version: "{PROTOCOL_VERSION}"
protocol_status: DESIGN_LOCKED
paper_section: "7 Semi-synthetic Bridge Validation and 8 Real-world Validation"

common:
  normalization: median_mad
  parameter_source: synthetic_development_set
  per_record_parameter_tuning: false
  manual_component_selection: false
  pure_real_oracle: false
  statistical_unit: record
  method_timeout_seconds: {DEFAULT_METHOD_TIMEOUT_SECONDS}
  interpretation_boundary: "semi-synthetic recovery is approximate-reference recovery; pure real-world evidence is structure, stability, utility, and leakage diagnostics"

ecg:
  dataset: MIT-BIH Arrhythmia Database
  noise_dataset: MIT-BIH Noise Stress Test Database
  sampling_rate_hz: 360
  lead_rule: MLII_if_available
  window_length_sec: 10
  overlap_fraction: 0.0
  split_unit: record_or_subject

  development_stage:
    selects:
      - preprocessing_compatibility_rules
      - component_selection_thresholds
      - r_peak_detector_threshold
      - r_peak_matching_tolerance
      - qrs_comparison_window
    prohibited:
      - held_out_record_parameter_tuning
      - manual_test_record_imf_selection
      - test_record_detector_threshold_tuning

  semi_synthetic:
    evidence_role: approximate_reference_validation
    reference_name: approximate_reference
    noise_types:
      - gaussian
      - impulsive
      - nstdb_real_noise
    noise_levels:
      - low
      - medium
      - high
    contamination_seeds: 10
    oracle_name: approximate_reference_oracle_upper_bound
    primary_outputs:
      - denoise_nmse
      - denoise_corr
      - r_peak_f1
      - r_peak_timing_error
      - qrs_morphology_correlation
      - outlier_resistance_index
      - clean_region_nmse
      - contamination_spillover_error

  pure_real:
    evidence_role: truth_free_validation
    r_peak_detector: locked_on_development_records
    qrs_window_ms: locked_on_development_records
    primary_outputs:
      - r_peak_sensitivity
      - r_peak_ppv
      - r_peak_f1
      - r_peak_timing_error
      - reconstruction_stability
      - r_peak_stability
      - seed_stability
    leakage_diagnostics:
      - r_peak_locked_residual_average
      - residual_qrs_correlation
      - residual_event_energy
    secondary_diagnostics:
      - residual_whiteness
      - residual_entropy
      - residual_energy_ratio
      - component_count
      - reconstruction_identity_error
      - runtime
      - failure_rate

bearing:
  dataset: CWRU Bearing Dataset
  sampling_rate_hz: 12000
  sensor_rule: drive_end
  window_length_sec: 1
  overlap_fraction: 0.0
  split_unit: source_file_or_condition
  leakage_warning: "do not randomly split adjacent or overlapping windows across train/test"
  status: planned_after_ecg_minimal_closed_loop
"""


def _split_registry_yaml() -> str:
    dev = _yaml_list(ECG_PROPOSED_DEVELOPMENT_RECORDS, indent=4)
    test = _yaml_list(ECG_PROPOSED_HELD_OUT_RECORDS, indent=4)
    return f"""protocol:
  id: {PROTOCOL_ID}
  version: "{PROTOCOL_VERSION}"
  frozen_date: {FROZEN_DATE}

protocol_version: "{PROTOCOL_VERSION}"
registry_status: PRE_REGISTERED_DRAFT_PENDING_DATA_AUDIT
freeze_rule: "After raw data audit, record identities may only be changed for documented data unavailability or corruption, never based on method performance."

split_validation:
  record_overlap_count: null
  subject_overlap_count: null
  duplicate_window_count: null
  validation_status: not_run
  reason: "Local data audit has not been executed at Stage 2."

ecg:
  dataset: MIT-BIH Arrhythmia Database
  split_unit: record_or_subject
  same_record_or_subject_may_cross_development_and_test: false
  development_records:
{dev}
  held_out_test_records:
{test}
  audit_required_before_execution:
    - record_exists_locally
    - target_lead_available_or_fallback_documented
    - annotations_available
    - usable_duration_sufficient_for_10s_windows

bearing:
  dataset: CWRU Bearing Dataset
  split_unit: source_file_or_condition
  same_source_file_or_adjacent_windows_may_cross_development_and_test: false
  development_conditions:
    - TO_BE_FILLED_AFTER_CWRU_DATA_AUDIT
  held_out_test_conditions:
    - TO_BE_FILLED_AFTER_CWRU_DATA_AUDIT
  split_policy:
    - prefer_condition_level_split_by_load_rpm_fault_size_or_source_file
    - no_random_shuffle_of_overlapping_windows
"""


def _mitbih_record_subject_map_yaml() -> str:
    rows = []
    for record in MITBIH_RECORDS:
        subject = "subject_201_202" if record in {"201", "202"} else f"subject_{record}"
        rows.append(f'  "{record}": "{subject}"')
    mapping = "\n".join(rows)
    return f"""dataset: MIT-BIH Arrhythmia Database
mapping_version: "1.0"
mapping_source: "PhysioNet MIT-BIH documentation: 48 records from 47 subjects; records 201 and 202 treated as the shared-subject pair."
mapping_policy: "All records are treated as independent subjects except 201 and 202, which are mapped to subject_201_202 unless more detailed source metadata supersedes this file."
record_to_subject:
{mapping}
"""


def _locked_reconstruction_rule(irmf_params, emd_params, eemd_params, ceemdan_params):
    method_params = {
        "irmf": dict(irmf_params),
        "emd": dict(emd_params),
        "eemd": dict(eemd_params),
        "ceemdan": dict(ceemdan_params),
    }
    return {
        "protocol": {
            "id": PROTOCOL_ID,
            "version": PROTOCOL_VERSION,
            "frozen_date": FROZEN_DATE,
            "maturity_stage": 2,
        },
        "protocol_version": PROTOCOL_VERSION,
        "rule_status": "PENDING_DEVELOPMENT_RECORD_LOCK",
        "parameter_selection_source": "synthetic_development_set",
        "manual_override_allowed": False,
        "pure_real_oracle_allowed": False,
        "method_parameter_hash": _stable_hash(method_params),
        "methods": method_params,
        "ecg_component_selection": {
            "rule_type": "frequency_aware_locked_rule",
            "selection_scope": "development_records_only",
            "test_record_manual_selection_allowed": False,
            "features": [
                "psd_center_frequency",
                "band_energy_ratio",
                "component_signal_correlation",
            ],
            "physiology_band_hz": {"low": 0.5, "high": 40.0},
            "thresholds": {
                "status": "TO_BE_SELECTED_ON_DEVELOPMENT_RECORDS",
                "band_energy_ratio_min": None,
                "component_signal_correlation_min": None,
            },
        },
        "ecg_downstream": {
            "r_peak_detector": "TO_BE_LOCKED_ON_DEVELOPMENT_RECORDS",
            "detector_threshold": None,
            "matching_tolerance_ms": None,
            "qrs_window_before_ms": None,
            "qrs_window_after_ms": None,
            "test_record_threshold_tuning_allowed": False,
        },
        "bearing_component_selection": {
            "status": "PLANNED_AFTER_ECG_MINIMAL_CLOSED_LOOP",
            "rule_type": "fault_frequency_aware_locked_rule",
            "selection_scope": "development_conditions_only",
        },
    }


def _evidence_matrix_rows():
    return [
        {
            "evidence_setting": "semi_synthetic_ecg",
            "ground_truth_level": "approximate_reference",
            "primary_claim_supported": "recovery with respect to the approximate reference",
            "primary_outputs": "denoise_nmse; denoise_corr; R-peak F1; timing error; QRS morphology correlation",
            "oracle_policy": "approximate-reference oracle upper bound only",
        },
        {
            "evidence_setting": "pure_real_ecg",
            "ground_truth_level": "truth_free_with_annotations",
            "primary_claim_supported": "application-relevant structure preservation, stability, and utility",
            "primary_outputs": "R-peak sensitivity; PPV; F1; timing error; reconstruction and peak stability",
            "oracle_policy": "not defined",
        },
        {
            "evidence_setting": "pure_real_ecg_leakage",
            "ground_truth_level": "truth_free_with_event_annotations",
            "primary_claim_supported": "whether meaningful QRS structure is transferred into residual",
            "primary_outputs": "R-peak-locked residual average; residual QRS correlation; residual event energy",
            "oracle_policy": "not defined",
        },
        {
            "evidence_setting": "cwru_bearing",
            "ground_truth_level": "condition_labels_and_fault_metadata",
            "primary_claim_supported": "mechanical structure preservation and fault-discrimination utility",
            "primary_outputs": "fault-frequency contrast; harmonic preservation; fault-discrimination utility",
            "oracle_policy": "semi-synthetic approximate-reference oracle only when known contamination is added",
        },
    ]


def _protocol_status_yaml() -> str:
    return f"""protocol:
  id: {PROTOCOL_ID}
  version: "{PROTOCOL_VERSION}"
  frozen_date: {FROZEN_DATE}
  checksum_scope: "protocol_status + protocol scaffold files"

protocol_version: "{PROTOCOL_VERSION}"
section_8:
  legacy_status_key: true
  manuscript_sections:
    - "7 Semi-synthetic Bridge Validation"
    - "8 Real-world Validation"
  protocol_maturity:
    stage: 2
    stage_total: 4
    stage_name: Frozen scientific protocol
    description: "Evidence hierarchy, claim boundaries, data split scaffold, and base method parameters are preregistered; ECG operational thresholds remain pending development-set locking."

  stage_definitions:
    stage_1: Scientific design drafted
    stage_2: Frozen scientific protocol
    stage_3: Locked operational protocol
    stage_4: Held-out empirical evaluation completed

  scientific_protocol:
    status: frozen
    evidence_hierarchy_frozen: true
    semi_synthetic_pure_real_boundary_frozen: true
    pure_real_oracle_disabled: true

  data_governance:
    split_registry_status: preregistered_draft_pending_data_audit
    development_test_split_preregistered: true
    record_overlap_verified: false
    subject_overlap_verified: false
    held_out_access_initiated: false
    split_validation:
      record_overlap_count: null
      subject_overlap_count: null
      duplicate_window_count: null
      validation_status: not_run
      reason: "Local MIT-BIH/CWRU data audit has not been executed at Stage 2."

  base_method_protocol:
    status: locked
    parameter_source: synthetic_development_set_or_project_config_fixed_params
    per_record_parameter_tuning_allowed: false
    manual_component_selection_on_test_allowed: false

  operational_protocol:
    status: pending_development_lock
    development_record_execution_status: not_started
    development_record_execution_completed: false
    component_selection_rule_locked: false
    r_peak_detector_locked: false
    matching_tolerance_locked: false
    qrs_window_locked: false

  held_out_evaluation:
    status: not_started
    semi_synthetic_held_out_ecg_completed: false
    pure_real_held_out_ecg_completed: false
    cwru_bearing_completed: false

  empirical_results:
    status: unavailable
    performance_claims_allowed: false

  stage_completion_criteria:
    stage_3_locked_operational_protocol:
      complete_only_if:
        - component_selection_rule_locked
        - detector_thresholds_locked
        - matching_tolerance_locked
        - qrs_analysis_window_locked
        - record_overlap_verification_passed
        - subject_overlap_verification_passed
        - frozen_protocol_files_regenerated
        - no_held_out_record_accessed_before_locking
    stage_4_held_out_empirical_evaluation:
      complete_only_if:
        - held_out_semi_synthetic_evaluation_completed
        - held_out_pure_real_evaluation_completed
        - record_level_statistics_completed
        - statistical_comparisons_completed
        - all_outputs_archived
"""


def _audit_checklist_rows():
    return [
        {
            "audit_category": "Design Audit",
            "check_item": "Evidence hierarchy frozen",
            "status": "complete",
            "evidence_file": "real_data_protocol.yaml; section_8_real_data_design_spec.json",
            "notes": "Semi-synthetic and pure-real evidence roles are separated.",
        },
        {
            "audit_category": "Design Audit",
            "check_item": "Semi-synthetic / pure-real boundary frozen",
            "status": "complete",
            "evidence_file": "section_8_evidence_matrix.csv",
            "notes": "Approximate-reference recovery is separated from truth-free evidence.",
        },
        {
            "audit_category": "Design Audit",
            "check_item": "Pure-real oracle disabled",
            "status": "complete",
            "evidence_file": "real_data_protocol.yaml; locked_reconstruction_rule.json",
            "notes": "Pure real-world oracle claims are not allowed.",
        },
        {
            "audit_category": "Design Audit",
            "check_item": "Development/test split preregistered",
            "status": "complete",
            "evidence_file": "split_registry.yaml",
            "notes": "Record lists are preregistered pending raw data audit.",
        },
        {
            "audit_category": "Design Audit",
            "check_item": "Base algorithm parameters locked",
            "status": "complete",
            "evidence_file": "locked_reconstruction_rule.json",
            "notes": "IRMF, EMD, EEMD, and CEEMDAN base configurations are recorded.",
        },
        {
            "audit_category": "Execution Audit",
            "check_item": "Record overlap verified",
            "status": "pending",
            "evidence_file": "TO_BE_GENERATED_BY_DATA_AUDIT",
            "notes": "Requires local MIT-BIH record audit.",
        },
        {
            "audit_category": "Execution Audit",
            "check_item": "Subject overlap verified",
            "status": "pending",
            "evidence_file": "TO_BE_GENERATED_BY_DATA_AUDIT",
            "notes": "Requires local subject/record metadata audit.",
        },
        {
            "audit_category": "Execution Audit",
            "check_item": "Development-record execution completed",
            "status": "pending",
            "evidence_file": "TO_BE_GENERATED_BY_DEVELOPMENT_RUN",
            "notes": "Stage 3B must run development records only and produce a numerical/stability audit.",
        },
        {
            "audit_category": "Execution Audit",
            "check_item": "Component-selection rule locked",
            "status": "pending",
            "evidence_file": "locked_reconstruction_rule.json",
            "notes": "Thresholds must be selected on development records only.",
        },
        {
            "audit_category": "Execution Audit",
            "check_item": "R-peak detector locked",
            "status": "pending",
            "evidence_file": "locked_reconstruction_rule.json",
            "notes": "Detector threshold must be selected on development records only.",
        },
        {
            "audit_category": "Execution Audit",
            "check_item": "Matching tolerance locked",
            "status": "pending",
            "evidence_file": "locked_reconstruction_rule.json",
            "notes": "Peak-matching tolerance remains pending development-set locking.",
        },
        {
            "audit_category": "Execution Audit",
            "check_item": "QRS window locked",
            "status": "pending",
            "evidence_file": "locked_reconstruction_rule.json",
            "notes": "QRS analysis window remains pending development-set locking.",
        },
        {
            "audit_category": "Execution Audit",
            "check_item": "Held-out access initiated",
            "status": "pending",
            "evidence_file": "TO_BE_GENERATED_BY_HELD_OUT_RUN",
            "notes": "Held-out records must not be accessed before operational locking.",
        },
        {
            "audit_category": "Execution Audit",
            "check_item": "Held-out evaluation completed",
            "status": "pending",
            "evidence_file": "TO_BE_GENERATED_BY_HELD_OUT_RUN",
            "notes": "No real-data performance result is available at Stage 2.",
        },
    ]


def run_real_data_protocol_scaffold(
        output_root,
        irmf_params=GLOBAL_IRMF_PARAMS,
        emd_params=GLOBAL_EMD_PARAMS,
        eemd_params=GLOBAL_EEMD_PARAMS,
        ceemdan_params=GLOBAL_CEEMDAN_PARAMS,
):
    output_root = ensure_dir(output_root)
    _write_text(output_root / "real_data_protocol.yaml", _real_data_protocol_yaml())
    _write_text(output_root / "split_registry.yaml", _split_registry_yaml())
    _write_text(output_root / "mitbih_record_subject_map.yaml", _mitbih_record_subject_map_yaml())
    _write_text(output_root / "protocol_status.yaml", _protocol_status_yaml())
    locked_rule = _locked_reconstruction_rule(irmf_params, emd_params, eemd_params, ceemdan_params)
    write_json(locked_rule, output_root / "locked_reconstruction_rule.json")
    rows = _evidence_matrix_rows()
    write_csv(rows, output_root / "section_8_evidence_matrix.csv")
    write_json(rows, output_root / "section_8_evidence_matrix.json")
    audit_rows = _audit_checklist_rows()
    write_csv(audit_rows, output_root / "section_8_audit_checklist.csv")
    write_json(audit_rows, output_root / "section_8_audit_checklist.json")
    component_hashes = {
        "locked_reconstruction_rule_sha256": _stable_hash(locked_rule),
            "split_registry_yaml_sha256": hashlib.sha256(
                _split_registry_yaml().encode("utf-8")
            ).hexdigest(),
            "mitbih_record_subject_map_yaml_sha256": hashlib.sha256(
                _mitbih_record_subject_map_yaml().encode("utf-8")
            ).hexdigest(),
            "real_data_protocol_yaml_sha256": hashlib.sha256(
                _real_data_protocol_yaml().encode("utf-8")
            ).hexdigest(),
        "protocol_status_yaml_sha256": hashlib.sha256(
            _protocol_status_yaml().encode("utf-8")
        ).hexdigest(),
    }
    protocol_checksum = _stable_hash({
        "protocol_id": PROTOCOL_ID,
        "protocol_version": PROTOCOL_VERSION,
        "component_hashes": component_hashes,
    })
    by_category = {}
    for row in audit_rows:
        category = row["audit_category"]
        by_category.setdefault(category, {"complete": 0, "pending": 0})
        by_category[category][row["status"]] += 1
    design = {
        "protocol": {
            "id": PROTOCOL_ID,
            "version": PROTOCOL_VERSION,
            "frozen_date": FROZEN_DATE,
            "checksum": protocol_checksum,
        },
        "protocol_version": PROTOCOL_VERSION,
        "section": "7 Semi-synthetic Bridge Validation and 8 Real-world Validation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "design_status": "frozen_sections_7_8_protocol_scaffold",
        "protocol_maturity": {
            "stage": 2,
            "stage_total": 4,
            "stage_name": "Frozen scientific protocol",
            "next_stage": "Locked operational protocol after development-set locking",
        },
        "execution_status": "no_real_data_results_claimed_by_this_stage",
        "three_required_files": {
            "real_data_protocol": "real_data_protocol.yaml",
            "split_registry": "split_registry.yaml",
            "mitbih_record_subject_map": "mitbih_record_subject_map.yaml",
            "locked_reconstruction_rule": "locked_reconstruction_rule.json",
            "protocol_status": "protocol_status.yaml",
        },
        "split_validation": {
            "record_overlap_count": None,
            "subject_overlap_count": None,
            "duplicate_window_count": None,
            "validation_status": "not_run",
            "reason": "Local MIT-BIH/CWRU data audit has not been executed at Stage 2.",
        },
        "evidence_chain": {
            "semi_synthetic": [
                "approximate reference",
                "recovery",
                "structure preservation",
                "contamination resistance",
                "approximate-reference oracle upper bound",
            ],
            "pure_real_world": [
                "no clean truth",
                "application-relevant structure preservation",
                "stability and reproducibility",
                "downstream utility",
                "event-locked residual leakage diagnostics",
            ],
        },
        "claim_boundary": (
            "Semi-synthetic evidence quantifies recovery with respect to an "
            "approximate reference. Pure real-world evidence evaluates whether "
            "the locked decomposition preserves application-relevant structures, "
            "remains stable under minor perturbations, and supports downstream "
            "analysis without systematically transferring meaningful signal "
            "structure into the residual."
        ),
        "hashes": component_hashes,
        "audit_checklist_summary": {
            "complete": sum(1 for row in audit_rows if row["status"] == "complete"),
            "pending": sum(1 for row in audit_rows if row["status"] == "pending"),
            "by_category": by_category,
        },
        "stage_completion_criteria": {
            "stage_3_locked_operational_protocol": [
                "component-selection rule locked",
                "detector thresholds locked",
                "matching tolerance locked",
                "QRS analysis window locked",
                "record overlap verification passed",
                "subject overlap verification passed",
                "frozen protocol files regenerated",
                "no held-out record accessed before locking",
            ],
            "stage_4_held_out_empirical_evaluation": [
                "held-out semi-synthetic evaluation completed",
                "held-out pure-real evaluation completed",
                "record-level statistics completed",
                "statistical comparisons completed",
                "all outputs archived",
            ],
        },
    }
    write_json(design, output_root / "section_8_real_data_design_spec.json")
    readme = """# Sections 7-8 Real-Data Locked Protocol

This directory materializes the frozen real-data evidence-continuum design.  It is not a
performance result folder.

Required protocol files:

- Protocol ID: `SECTION8_REALDATA_V1.0`
- Protocol checksum: recorded in `section_8_real_data_design_spec.json`
- `real_data_protocol.yaml`: rules that may not change by held-out test record.
- `split_registry.yaml`: pre-registered development/test identities.
- `mitbih_record_subject_map.yaml`: auditable MIT-BIH record-to-subject map
  used for subject-overlap verification; records 201 and 202 share one subject.
- `protocol_status.yaml`: maturity state and lifecycle status for Sections 7-8.
- `locked_reconstruction_rule.json`: method parameters, component-selection
  rules, and downstream thresholds.  Threshold values remain marked
  `TO_BE_SELECTED_ON_DEVELOPMENT_RECORDS` until the ECG development-record
  locking pass is run.
- `section_8_audit_checklist.csv`: checklist distinguishing completed protocol
  items from pending operational-locking and held-out-evaluation items.

Current maturity:

- Stage 2 / 4: Frozen scientific protocol.
- Stage 3 requires ECG development-record locking of component-selection,
  R-peak detection, matching tolerance, and QRS-window thresholds.
- Stage 4 requires held-out empirical evaluation.

Stage 3 completion criteria:

1. Component-selection rule locked.
2. Detector thresholds locked.
3. Matching tolerance locked.
4. QRS analysis window locked.
5. Record overlap verification passed.
6. Subject overlap verification passed.
7. Frozen protocol files regenerated.
8. No held-out record accessed before locking.

Stage 4 completion criteria:

1. Held-out semi-synthetic evaluation completed.
2. Held-out pure-real evaluation completed.
3. Record-level statistics completed.
4. Statistical comparisons completed.
5. All outputs archived.

Evidence boundary:

- Section 7 semi-synthetic real signals support approximate-reference recovery claims.
- Section 8 pure real-world signals support structure-preservation, stability,
  reproducibility, utility, and event-locked residual leakage claims.
- Pure real-world data do not support true clean-signal recovery, true IMF
  recovery, or pure-real oracle claims.
"""
    _write_text(output_root / "README_section_8_real_data_protocol.md", readme)
    return design
