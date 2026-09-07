#!/usr/bin/python
# coding: UTF-8

"""Real-world validation endpoint schema gates.

This module materializes the truth-free real-world endpoint schema that is
construct-matched to the frozen V5.28 synthetic primary schema.  The commands in
this module do not access MIT-BIH records, run decompositions, tune thresholds,
or report method performance.  They only freeze endpoint definitions and audit
the protocol boundaries needed before development-record locking.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


REAL_WORLD_SCHEMA_ID = "REAL_WORLD_VALIDATION_ENDPOINT_SCHEMA_V1.0"
REAL_WORLD_SCHEMA_VERSION = "1.0"
SOURCE_SYNTHETIC_SCHEMA = "V5.28_normalized_spillover_primary_revision"

EPS_ABSOLUTE = 1e-12
DENOMINATOR_RELATIVE_FLOOR = 1e-8
LOW_ENERGY_FLAG_RELATIVE_THRESHOLD = 1e-8
EXTREME_RATIO_REPORT_QUANTILES = (0.95, 0.99, 0.999)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _stable_hash(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _schema_root(output_root):
    return Path(output_root)


def _endpoint_rows():
    return [
        {
            "endpoint": "r_peak_f1",
            "construct": "signal_recovery",
            "direction": "higher_is_better",
            "role": "primary",
            "estimand": "annotation_level_event_preservation",
            "formula": "2 * precision * recall / (precision + recall + eps)",
            "requires": "expert_R_peak_annotations; frozen_R_peak_detector; frozen_matching_tolerance_ms",
            "degenerate_case_handling": "if TP+FP=0 and TP+FN=0 mark no_reference_events; otherwise use eps-stabilized precision/recall",
            "synthetic_construct_alignment": "denoise_corr / denoise_nmse as signal recovery, but operationalized through annotated event preservation",
        },
        {
            "endpoint": "r_peak_timing_error_ms",
            "construct": "signal_recovery",
            "direction": "lower_is_better",
            "role": "primary",
            "estimand": "annotation_level_temporal_fidelity",
            "formula": "median_{matched peaks} |t_hat_r - t_ref_r| in milliseconds",
            "requires": "expert_R_peak_annotations; frozen_R_peak_detector; frozen_matching_tolerance_ms",
            "degenerate_case_handling": "if no matched peaks mark not_computable_no_matched_peaks",
            "synthetic_construct_alignment": "temporal-fidelity counterpart to truth-aware signal recovery",
        },
        {
            "endpoint": "perturbation_matched_component_stability",
            "construct": "component_recovery_quality",
            "direction": "higher_is_better",
            "role": "primary",
            "estimand": "decomposition_component_reproducibility_under_predefined_micro_perturbation",
            "formula": "mean_{(i,j) in Hungarian(1-|corr(c_i,c'_j)|)} |corr(c_i,c'_j)|",
            "requires": "baseline_components; perturbed_components; frozen_micro_perturbation_protocol",
            "degenerate_case_handling": "if either component set is empty mark not_computable_empty_component_set",
            "synthetic_construct_alignment": "matched_component_corr tool reused for stability, not truth recovery",
        },
        {
            "endpoint": "effective_component_count_instability",
            "construct": "component_set_fidelity",
            "direction": "lower_is_better",
            "role": "primary",
            "estimand": "component_set_cardinality_stability",
            "formula": "|K_eff_delta - K_eff| / max(K_eff, 1)",
            "requires": "baseline_effective_component_count; perturbed_effective_component_count; frozen_effective_count_rule",
            "degenerate_case_handling": "denominator uses max(K_eff,1); no zero denominator",
            "synthetic_construct_alignment": "real-data stability analogue of RCCE",
        },
        {
            "endpoint": "symmetric_unmatched_component_energy_ratio",
            "construct": "component_set_fidelity",
            "direction": "lower_is_better",
            "role": "primary",
            "estimand": "component_set_energy_stability",
            "formula": "0.5 * (E_unmatched_baseline/E_baseline + E_unmatched_perturbed/E_perturbed)",
            "requires": "Hungarian matched components; frozen matching threshold; component energies",
            "degenerate_case_handling": (
                "use denominator floor for each side; flag low component energy "
                "when total component energy is below the frozen relative floor; "
                "if both sides have zero effective component energy mark "
                "not_computable_zero_component_energy"
            ),
            "synthetic_construct_alignment": "truth-free stability analogue of missing/spurious component energy",
        },
        {
            "endpoint": "qrs_locked_residual_leakage",
            "construct": "noise_separation",
            "direction": "lower_is_better",
            "role": "primary",
            "estimand": "annotation_anchored_physiological_signal_leakage_into_estimated_residual",
            "formula": "sum_r sum_{t in W_r} eps_hat_real[t]^2 / (sum_r sum_{t in W_r} Y[t]^2 + eps)",
            "requires": "expert_R_peak_annotations; frozen_QRS_window; protocol_reconstructed_signal",
            "degenerate_case_handling": (
                "use denominator floor; low QRS energy denominator flagged; "
                "if no annotated QRS windows are available mark "
                "not_computable_no_qrs_annotations; primary summary uses "
                "paired ranks/medians plus denominator flags"
            ),
            "synthetic_construct_alignment": "annotation-anchored analogue of signal_leakage_into_noise",
        },
        {
            "endpoint": "paired_contaminated_region_reconstruction_deviation",
            "construct": "contamination_resistance",
            "direction": "lower_is_better",
            "role": "primary",
            "estimand": "controlled_perturbation_local_reconstruction_stability_inside_injected_artifact_mask",
            "formula": "|| (Xhat_c - Xhat_0)_C ||_2^2 / (|| Xhat_0_C ||_2^2 + eps_floor)",
            "requires": "baseline_reconstruction; contaminated_reconstruction; frozen_artifact_mask",
            "degenerate_case_handling": "denominator floor and low-energy-mask flag required; report medians/ranks and extreme-ratio summary",
            "synthetic_construct_alignment": "paired-reference analogue of contaminated_region_nmse",
        },
        {
            "endpoint": "paired_clean_region_reconstruction_deviation",
            "construct": "contamination_resistance",
            "direction": "lower_is_better",
            "role": "primary",
            "estimand": "controlled_perturbation_reconstruction_stability_outside_injected_artifact_mask",
            "formula": "|| (Xhat_c - Xhat_0)_notC ||_2^2 / (|| Xhat_0_notC ||_2^2 + eps_floor)",
            "requires": "baseline_reconstruction; contaminated_reconstruction; frozen_artifact_mask",
            "degenerate_case_handling": "denominator floor and low-energy-mask flag required; report medians/ranks and extreme-ratio summary",
            "synthetic_construct_alignment": "paired-reference analogue of clean_region_nmse",
        },
        {
            "endpoint": "normalized_paired_spillover_loss",
            "construct": "contamination_resistance",
            "direction": "lower_is_better",
            "role": "primary",
            "estimand": "controlled_perturbation_local_spillover_stability_in_neighboring_clean_samples",
            "formula": "|| (Xhat_c - Xhat_0)_{N(C)\\C} ||_2^2 / (mean_{t in N(C)\\C} Xhat_0[t]^2 + eps_floor)",
            "requires": "baseline_reconstruction; contaminated_reconstruction; frozen_artifact_mask; frozen_spillover_radius",
            "degenerate_case_handling": "denominator floor and low-energy-neighborhood flag required; report medians/ranks and extreme-ratio summary",
            "synthetic_construct_alignment": "paired-reference analogue of normalized_contamination_spillover_loss",
        },
    ]


def _secondary_endpoint_rows():
    return [
        {
            "endpoint": "residual_structured_dependence_index",
            "construct": "noise_separation",
            "role": "secondary",
            "reason_not_primary": "real ECG noise may be colored or structured; whiteness is not a truth condition",
        },
        {
            "endpoint": "residual_psd_summary",
            "construct": "noise_separation",
            "role": "diagnostic",
            "reason_not_primary": "mechanism interpretation only",
        },
        {
            "endpoint": "component_count_distribution",
            "construct": "component_set_fidelity",
            "role": "diagnostic",
            "reason_not_primary": "descriptive distribution, not a validation endpoint",
        },
        {
            "endpoint": "runtime_seconds",
            "construct": "operational_feasibility",
            "role": "diagnostic",
            "reason_not_primary": "computational reporting, not scientific performance construct",
        },
    ]


def _method_neutrality_rule():
    return {
        "rule": (
            "Real-world reconstruction must depend only on observable component "
            "waveforms, frozen component-level features, annotation-derived "
            "windows where applicable, and development-locked thresholds."
        ),
        "allowed_inputs": [
            "estimated_component_waveform",
            "estimated_residual_or_trend_bearing_candidate_waveform",
            "component_frequency_features",
            "component_energy_features",
            "development_locked_thresholds",
            "expert_R_peak_annotations_for_validation_windows",
        ],
        "prohibited_inputs": [
            "method_label",
            "first_component_identity",
            "last_component_identity",
            "native_residual_semantics_as_noise_truth",
            "component_extraction_order_as_physical_label",
            "held_out_record_threshold_tuning",
            "manual_component_selection",
            "method_specific_selection_rule",
        ],
        "candidate_collection_policy": (
            "Every exposed estimated component, including any residual, trend, "
            "or low-frequency/high-frequency remainder object, must be treated "
            "as an eligible candidate without assigning it a universal physical "
            "meaning from its position or name."
        ),
    }


def _perturbation_protocol():
    return {
        "protocol_id": "REAL_WORLD_MICRO_PERTURBATION_PROTOCOL_V1.0",
        "protocol_status": "frozen_for_schema_qualification",
        "primary_component_stability_perturbation": {
            "type": "additive_micro_noise",
            "distribution": "Gaussian",
            "scale": "0.01 * robust_MAD(Y)",
            "n_replicates": 5,
            "random_seeds": [17001, 17002, 17003, 17004, 17005],
            "physiology_preservation_claim": (
                "The perturbation is intended to test algorithmic stability "
                "without changing underlying ECG event timing or morphology."
            ),
        },
        "not_primary_perturbations": {
            "small_amplitude_scaling": "reserved_for_sensitivity_diagnostic_only",
            "small_temporal_jitter": "reserved_for_sensitivity_diagnostic_only",
        },
        "controlled_contamination_perturbation": {
            "artifact_types": ["spike", "short_burst", "baseline_step"],
            "artifact_region_policy": "masks selected from development-locked random seeds and kept fixed for held-out evaluation",
            "artifact_amplitude_grid": ["low", "medium", "high"],
            "random_seeds": [18001, 18002, 18003, 18004, 18005],
            "spillover_radius_ms": "locked_on_development_records_before_held_out_evaluation",
        },
        "held_out_boundary": {
            "development_records_may_select": [
                "R_peak_detector_threshold",
                "R_peak_matching_tolerance_ms",
                "QRS_window_ms",
                "effective_component_energy_threshold",
                "real_world_reconstruction_thresholds",
                "spillover_radius_ms",
            ],
            "held_out_records_may_not_select": [
                "any_method_parameter",
                "any_component_selection_threshold",
                "any_detector_threshold",
                "any_metric_denominator_policy",
                "any_artifact_mask_policy",
            ],
        },
    }


def _denominator_policy():
    return {
        "eps_absolute": EPS_ABSOLUTE,
        "relative_floor": DENOMINATOR_RELATIVE_FLOOR,
        "low_energy_flag_relative_threshold": LOW_ENERGY_FLAG_RELATIVE_THRESHOLD,
        "floor_rule": (
            "For local ratio endpoints, denominator_eff = max(local_reference_energy, "
            "eps_absolute, relative_floor * full_window_reference_energy)."
        ),
        "low_energy_flag_rule": (
            "Flag denominator_low_energy=true when local_reference_energy <= "
            "low_energy_flag_relative_threshold * full_window_reference_energy."
        ),
        "summary_policy": {
            "primary_comparison": "paired ranks and paired effects within record/window/perturbation blocks",
            "absolute_summary": "median, IQR, and quantiles; arithmetic means are supplementary",
            "extreme_ratio_reporting": [float(q) for q in EXTREME_RATIO_REPORT_QUANTILES],
            "low_energy_rows": "reported separately and retained with flags unless non-finite",
        },
        "qrs_leakage_denominator": {
            "primary": "observed_signal_energy_inside_frozen_QRS_windows",
            "caveat": "annotation-anchored physiological leakage proxy, not true signal leakage",
            "sensitivity": "baseline_reconstruction_QRS_energy_or_QRS_template_energy",
        },
    }


def run_real_world_validation_schema_draft(output_root):
    output_root = ensure_dir(output_root)
    endpoints = _endpoint_rows()
    schema = {
        "schema_id": REAL_WORLD_SCHEMA_ID,
        "schema_version": REAL_WORLD_SCHEMA_VERSION,
        "schema_status": "draft_requires_audit",
        "source_synthetic_schema": SOURCE_SYNTHETIC_SCHEMA,
        "design_principle": "same_construct_framework_layer_appropriate_endpoints",
        "truth_available_primary_endpoint_count": 13,
        "real_world_primary_validation_endpoint_count": len(endpoints),
        "constructs": [
            "signal_recovery",
            "component_recovery_quality",
            "component_set_fidelity",
            "noise_separation",
            "contamination_resistance",
        ],
        "primary_endpoints": endpoints,
        "secondary_and_diagnostic_endpoints": _secondary_endpoint_rows(),
        "method_neutrality_rule": _method_neutrality_rule(),
        "denominator_policy": _denominator_policy(),
        "perturbation_protocol_reference": "REAL_WORLD_MICRO_PERTURBATION_PROTOCOL_V1.0",
        "statistical_boundary": {
            "synthetic_and_real_metrics_combined_into_single_rank": False,
            "real_world_endpoints_interpreted_as_synthetic_metric_substitutes": False,
            "real_world_validation_role": (
                "external validity, reproducibility, structural plausibility, "
                "annotation-level preservation, and controlled perturbation robustness"
            ),
        },
        "timestamp_utc": _utc_now(),
    }
    schema["schema_checksum"] = _stable_hash(schema)
    write_json(schema, output_root / "real_world_validation_endpoint_schema.json")
    write_csv(endpoints, output_root / "real_world_primary_endpoint_registry.csv")
    write_csv(_secondary_endpoint_rows(), output_root / "real_world_secondary_endpoint_registry.csv")
    write_json({
        "stage": "real_world_validation_schema_draft",
        "schema_id": REAL_WORLD_SCHEMA_ID,
        "schema_status": "draft_requires_audit",
        "primary_endpoint_count": len(endpoints),
        "next_required_gates": [
            "real-world-endpoint-formula-audit",
            "real-world-perturbation-protocol-freeze",
            "real-world-denominator-degeneracy-audit",
            "real-world-method-neutrality-audit",
            "real-world-validation-preflight",
        ],
        "schema_checksum": schema["schema_checksum"],
    }, output_root / "real_world_validation_schema_draft_dashboard.json")
    return schema


def run_real_world_endpoint_formula_audit(output_root, schema_root):
    output_root = ensure_dir(output_root)
    schema_path = Path(schema_root) / "real_world_validation_endpoint_schema.json"
    schema = _read_json(schema_path)
    if schema is None:
        raise FileNotFoundError("real_world_validation_endpoint_schema.json is required.")
    rows = []
    for endpoint in schema["primary_endpoints"]:
        formula = endpoint.get("formula", "")
        requires = endpoint.get("requires", "")
        degenerate = endpoint.get("degenerate_case_handling", "")
        row = {
            "endpoint": endpoint["endpoint"],
            "construct": endpoint["construct"],
            "has_formula": bool(formula),
            "has_direction": endpoint.get("direction") in {"higher_is_better", "lower_is_better"},
            "has_estimand": bool(endpoint.get("estimand")),
            "has_requirements": bool(requires),
            "has_degenerate_case_handling": bool(degenerate),
            "truth_free_boundary_clear": "true" not in endpoint.get("estimand", "").lower(),
            "formula_uses_paired_reference_where_required": (
                endpoint["construct"] != "contamination_resistance"
                or "Xhat_c - Xhat_0" in formula
            ),
            "passed": False,
        }
        row["passed"] = all(
            bool(row[k])
            for k in [
                "has_formula",
                "has_direction",
                "has_estimand",
                "has_requirements",
                "has_degenerate_case_handling",
                "formula_uses_paired_reference_where_required",
            ]
        )
        rows.append(row)
    passed = all(row["passed"] for row in rows)
    dashboard = {
        "stage": "real_world_endpoint_formula_audit",
        "schema_id": schema.get("schema_id"),
        "audit_status": "passed" if passed else "requires_review",
        "n_endpoints": len(rows),
        "n_failed": sum(not row["passed"] for row in rows),
        "source_schema": str(schema_path),
        "source_schema_checksum": schema.get("schema_checksum"),
        "timestamp_utc": _utc_now(),
    }
    write_csv(rows, output_root / "real_world_endpoint_formula_audit.csv")
    write_json(dashboard, output_root / "real_world_endpoint_formula_audit_dashboard.json")
    return dashboard


def run_real_world_perturbation_protocol_freeze(output_root, schema_root):
    output_root = ensure_dir(output_root)
    schema_path = Path(schema_root) / "real_world_validation_endpoint_schema.json"
    schema = _read_json(schema_path)
    if schema is None:
        raise FileNotFoundError("real_world_validation_endpoint_schema.json is required.")
    protocol = _perturbation_protocol()
    protocol["schema_id"] = schema.get("schema_id")
    protocol["schema_checksum"] = schema.get("schema_checksum")
    protocol["freeze_timestamp_utc"] = _utc_now()
    protocol["protocol_checksum"] = _stable_hash(protocol)
    checks = {
        "primary_perturbation_type_frozen": bool(protocol["primary_component_stability_perturbation"]["type"]),
        "primary_perturbation_scale_frozen": bool(protocol["primary_component_stability_perturbation"]["scale"]),
        "n_replicates_frozen": protocol["primary_component_stability_perturbation"]["n_replicates"] == 5,
        "random_seeds_frozen": len(protocol["primary_component_stability_perturbation"]["random_seeds"]) == 5,
        "amplitude_scaling_not_primary": protocol["not_primary_perturbations"]["small_amplitude_scaling"].endswith("only"),
        "temporal_jitter_not_primary": protocol["not_primary_perturbations"]["small_temporal_jitter"].endswith("only"),
        "held_out_boundary_declared": bool(protocol["held_out_boundary"]),
    }
    dashboard = {
        "stage": "real_world_perturbation_protocol_freeze",
        "protocol_id": protocol["protocol_id"],
        "protocol_status": "frozen" if all(checks.values()) else "requires_review",
        "checks": checks,
        "schema_id": schema.get("schema_id"),
        "protocol_checksum": protocol["protocol_checksum"],
        "timestamp_utc": _utc_now(),
    }
    write_json(protocol, output_root / "real_world_perturbation_protocol_frozen.json")
    write_json(dashboard, output_root / "real_world_perturbation_protocol_freeze_dashboard.json")
    return dashboard


def run_real_world_denominator_degeneracy_audit(output_root, schema_root):
    output_root = ensure_dir(output_root)
    schema_path = Path(schema_root) / "real_world_validation_endpoint_schema.json"
    schema = _read_json(schema_path)
    if schema is None:
        raise FileNotFoundError("real_world_validation_endpoint_schema.json is required.")
    policy = schema.get("denominator_policy", {})
    ratio_endpoints = [
        ep for ep in schema["primary_endpoints"]
        if "/" in ep.get("formula", "") or "ratio" in ep["endpoint"] or "loss" in ep["endpoint"]
    ]
    rows = []
    for ep in ratio_endpoints:
        row = {
            "endpoint": ep["endpoint"],
            "construct": ep["construct"],
            "has_denominator_floor": "floor" in ep.get("degenerate_case_handling", "").lower()
            or ep["endpoint"] in {"effective_component_count_instability", "r_peak_f1"},
            "has_low_energy_flag": "low" in ep.get("degenerate_case_handling", "").lower()
            or ep["endpoint"] in {"effective_component_count_instability", "r_peak_f1"},
            "has_nonfinite_reason_policy": "mark" in ep.get("degenerate_case_handling", "").lower()
            or ep["endpoint"] in {"effective_component_count_instability", "r_peak_f1"},
            "summary_not_mean_only": policy.get("summary_policy", {}).get("absolute_summary") is not None,
            "passed": False,
        }
        row["passed"] = all(bool(row[k]) for k in [
            "has_denominator_floor",
            "has_low_energy_flag",
            "summary_not_mean_only",
        ])
        rows.append(row)
    checks = {
        "eps_absolute_declared": policy.get("eps_absolute") == EPS_ABSOLUTE,
        "relative_floor_declared": policy.get("relative_floor") == DENOMINATOR_RELATIVE_FLOOR,
        "low_energy_flag_declared": policy.get("low_energy_flag_relative_threshold") == LOW_ENERGY_FLAG_RELATIVE_THRESHOLD,
        "extreme_quantile_reporting_declared": bool(policy.get("summary_policy", {}).get("extreme_ratio_reporting")),
        "qrs_denominator_caveat_declared": bool(policy.get("qrs_leakage_denominator", {}).get("caveat")),
        "all_ratio_endpoints_pass": all(row["passed"] for row in rows),
    }
    dashboard = {
        "stage": "real_world_denominator_degeneracy_audit",
        "audit_status": "passed" if all(checks.values()) else "requires_review",
        "checks": checks,
        "n_ratio_like_endpoints": len(rows),
        "n_failed_endpoint_checks": sum(not row["passed"] for row in rows),
        "timestamp_utc": _utc_now(),
    }
    write_csv(rows, output_root / "real_world_denominator_degeneracy_audit.csv")
    write_json(dashboard, output_root / "real_world_denominator_degeneracy_audit_dashboard.json")
    return dashboard


def run_real_world_method_neutrality_audit(output_root, schema_root):
    output_root = ensure_dir(output_root)
    schema_path = Path(schema_root) / "real_world_validation_endpoint_schema.json"
    schema = _read_json(schema_path)
    if schema is None:
        raise FileNotFoundError("real_world_validation_endpoint_schema.json is required.")
    rule = schema.get("method_neutrality_rule", {})
    prohibited = set(rule.get("prohibited_inputs", []))
    required_prohibitions = {
        "method_label",
        "first_component_identity",
        "last_component_identity",
        "native_residual_semantics_as_noise_truth",
        "component_extraction_order_as_physical_label",
        "held_out_record_threshold_tuning",
        "manual_component_selection",
        "method_specific_selection_rule",
    }
    checks = {
        "candidate_collection_policy_declared": bool(rule.get("candidate_collection_policy")),
        "method_label_prohibited": "method_label" in prohibited,
        "first_last_identity_prohibited": {
            "first_component_identity",
            "last_component_identity",
        }.issubset(prohibited),
        "native_residual_semantics_prohibited": "native_residual_semantics_as_noise_truth" in prohibited,
        "extraction_order_semantics_prohibited": "component_extraction_order_as_physical_label" in prohibited,
        "held_out_tuning_prohibited": "held_out_record_threshold_tuning" in prohibited,
        "manual_selection_prohibited": "manual_component_selection" in prohibited,
        "method_specific_rule_prohibited": "method_specific_selection_rule" in prohibited,
        "all_required_prohibitions_present": required_prohibitions.issubset(prohibited),
    }
    dashboard = {
        "stage": "real_world_method_neutrality_audit",
        "audit_status": "passed" if all(checks.values()) else "requires_review",
        "checks": checks,
        "method_neutrality_rule": rule,
        "timestamp_utc": _utc_now(),
    }
    rows = [{"check": k, "passed": v} for k, v in checks.items()]
    write_csv(rows, output_root / "real_world_method_neutrality_audit.csv")
    write_json(dashboard, output_root / "real_world_method_neutrality_audit_dashboard.json")
    return dashboard


def run_real_world_validation_preflight(output_root, schema_root, formula_root,
                                        perturbation_root, denominator_root,
                                        neutrality_root):
    output_root = ensure_dir(output_root)
    paths = {
        "schema": Path(schema_root) / "real_world_validation_endpoint_schema.json",
        "formula_audit": Path(formula_root) / "real_world_endpoint_formula_audit_dashboard.json",
        "perturbation_protocol": Path(perturbation_root) / "real_world_perturbation_protocol_freeze_dashboard.json",
        "denominator_audit": Path(denominator_root) / "real_world_denominator_degeneracy_audit_dashboard.json",
        "method_neutrality_audit": Path(neutrality_root) / "real_world_method_neutrality_audit_dashboard.json",
    }
    loaded = {}
    missing = []
    for key, path in paths.items():
        obj = _read_json(path)
        if obj is None:
            missing.append(key)
        else:
            loaded[key] = obj
    if missing:
        raise FileNotFoundError("Missing real-world validation preflight artifacts: " + ", ".join(missing))
    checks = {
        "schema_draft_exists": True,
        "formula_audit_passed": loaded["formula_audit"].get("audit_status") == "passed",
        "perturbation_protocol_frozen": loaded["perturbation_protocol"].get("protocol_status") == "frozen",
        "denominator_audit_passed": loaded["denominator_audit"].get("audit_status") == "passed",
        "method_neutrality_audit_passed": loaded["method_neutrality_audit"].get("audit_status") == "passed",
    }
    status = "passed" if all(checks.values()) else "requires_review"
    manifest = {
        "stage": "real_world_validation_preflight",
        "schema_id": REAL_WORLD_SCHEMA_ID,
        "preflight_status": status,
        "schema_ready_for_development_record_locking": status == "passed",
        "held_out_real_world_validation_authorized": False,
        "held_out_authorization_blocker": (
            "development-record operational thresholds, split audit, local data "
            "availability audit, and no-held-out-access audit remain required"
        ),
        "checks": checks,
        "source_artifacts": {k: str(v) for k, v in paths.items()},
        "timestamp_utc": _utc_now(),
    }
    write_json(manifest, output_root / "real_world_validation_preflight_manifest.json")
    return manifest
