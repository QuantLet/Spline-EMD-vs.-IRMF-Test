#!/usr/bin/python
# coding: UTF-8
"""Independent algorithm-validation pipeline.

The methodology layer must finish before this layer is interpreted scientifically.
This module enforces the paper-locked IRMF loss family:
    loss_name = gaussian_smoothed_median
The loss width H is read from the locked development-set IRMF configuration.
No alternative rho function is permitted in the algorithm benchmark.
"""
from pathlib import Path
import json

import numpy as np

from project_config import (
    GLOBAL_CEEMDAN_PARAMS, GLOBAL_EEMD_PARAMS, GLOBAL_EMD_PARAMS,
    GLOBAL_IRMF_PARAMS, PARAMETER_SELECTION_NOISES, PARAMETER_SELECTION_SIGMAS,
    PARAMETER_SELECTION_SEVERITY_DESIGN_VERSION,
    PARAMETER_SELECTION_SIGNALS, PARAMETER_SELECTION_TARGET_SNR_DB_LEVELS,
    TARGET_SNR_DB_LEVELS, V530_BENCHMARK_SEEDS,
    V530_CONVERGENCE_AUDIT_SEEDS, V530_SEED_CONVERGENCE_AUDIT_VERSION,
    V530_SEED_CONVERGENCE_PREFIXES,
    V530_SEED_POLICY_VERSION,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json, write_manifest
from experiments.experiment_global_parameter_selection import run_global_parameter_selection
from experiments.experiment_emd_family_benchmark import (
    run_emd_family_parameter_calibration, run_fixed_emd_family_benchmark,
    run_emd_family_baseline_sensitivity,
)
from experiments.experiment_repeated_measures_statistics import run_repeated_measures_statistics
from experiments.experiment_primary_evaluation_framework import run_primary_evaluation_framework
from experiments.experiment_metric_weighting_sensitivity import run_case_score_weighting_sensitivity
from experiments.experiment_metric_taxonomy import write_metric_taxonomy
from experiments.experiment_robustness_sensitivity import run_all_robustness_sensitivity
from experiments.experiment_signal_variant_robustness import run_signal_variant_robustness
from experiments.experiment_challenging_signal_suite import run_challenging_signal_suite
from experiments.experiment_controlled_challenging_diagnostics import run_controlled_challenging_diagnostics
from experiments.experiment_oracle_adaptivity_analysis import run_irmf_oracle_adaptivity_analysis
from experiments.experiment_result_sensitivity_analysis import run_v5_2_posthoc_result_checks
from experiments.experiment_unified_benchmark_cube import run_unified_benchmark_cube
from experiments.experiment_unified_cube_statistics import run_unified_cube_statistics
from experiments.experiment_v526_primary_schema_qualification import (
    run_v526_primary_schema_qualification,
)
from experiments.experiment_v527_primary_schema_qualification import (
    freeze_v527_primary_schema,
    run_v527_primary_schema_qualification,
)
from experiments.experiment_v527_unified_statistics import run_v527_unified_statistics
from experiments.experiment_v527_formula_validity_audit import (
    run_v527_formula_validity_audit,
)
from experiments.experiment_v527_directionality_audit import (
    run_v527_directionality_audit,
)
from experiments.experiment_v527_spillover_scale_audit import (
    run_v527_spillover_scale_audit,
)
from experiments.experiment_v528_primary_schema import (
    run_v528_final_statistics_validation,
    run_v528_primary_schema_freeze,
)
from experiments.experiment_v528_unified_statistics import run_v528_unified_statistics
from experiments.experiment_v529_snr_design import (
    run_v529_snr_design_draft,
    run_v529_snr_design_freeze,
    run_v529_snr_realization_audit,
)
from experiments.experiment_v530_clean_baseline import run_v530_clean_baseline_schema
from experiments.experiment_v531_time_axis_semantics import run_v531_time_axis_semantics_audit
from experiments.experiment_v532_section6_scope import run_v532_section6_scope
from experiments.experiment_v533_irmf_parameter_sensitivity_protocol import (
    run_v533_irmf_parameter_sensitivity_protocol,
)
from experiments.experiment_v534_section6_executable_protocols import (
    run_v534_section6_executable_protocols,
)
from experiments.experiment_v535_section6_1_irmf_parameter_smoke import (
    run_v535_section6_1_irmf_parameter_smoke,
)
from experiments.experiment_v536_section6_2_contamination_smoke import (
    run_v536_section6_2_contamination_design_smoke,
)
from experiments.experiment_v537_section6_3_comparator_signal_smoke import (
    run_v537_section6_3_comparator_signal_smoke,
)
from experiments.experiment_v538_section6_4_runtime_smoke import (
    run_v538_section6_4_runtime_instrumentation_smoke,
)
from experiments.experiment_v539_irmf_h_tuning_amendment import (
    run_v539_irmf_h_tuning_amendment,
)
from experiments.experiment_v540_time_frequency_secondary_schema import (
    run_v540_time_frequency_secondary_schema,
)
from experiments.experiment_v541_primary_secondary_closure import (
    run_v541_primary_secondary_closure,
)
from experiments.experiment_v542_irmf_relative_h_parameterization import (
    run_v542_irmf_relative_h_parameterization,
)
from experiments.experiment_v543_primary_metric_manual_code_literature_audit import (
    run_v543_primary_metric_manual_code_literature_audit,
)
from experiments.experiment_v544_h_decision_gate import run_v544_h_decision_gate
from experiments.experiment_v545_relative_h_scale_estimator_qualification import (
    run_v545_relative_h_scale_estimator_qualification,
)
from experiments.experiment_v548_parameter_selection_adjudication import (
    run_v548_parameter_selection_adjudication,
)
from experiments.experiment_v550_section6_main_text_assembly import (
    run_v550_section6_main_text_assembly,
)
from experiments.experiment_v551_section6_2_contamination_design import (
    run_v551_section6_2_contamination_design_sensitivity,
)
from experiments.experiment_v552_section6_4_computational_scaling import (
    run_v552_section6_4_computational_scaling,
)
from experiments.experiment_v553_section6_claim_qualification import (
    run_v553_section6_claim_qualification,
)
from experiments.experiment_v554_noise_realization_stability import (
    run_v554_noise_realization_decomposition_stability,
)
from experiments.experiment_v555_noise_realization_stability_qualification import (
    run_v555_noise_realization_stability_qualification,
)
from experiments.experiment_v556_cross_realization_waveform_stability import (
    run_v556_cross_realization_waveform_stability_protocol,
)
from experiments.experiment_v557_section6_challenging_signal_extension import (
    run_v557_section6_challenging_signal_extension_amendment,
)
from experiments.experiment_v558_section6_pre_execution_audit import (
    run_v558_section6_pre_execution_audit,
)
from experiments.experiment_v559_section6_post_execution_qualification import (
    run_v559_section6_post_execution_qualification,
)
from experiments.experiment_v560_matching_sensitivity_protocol import (
    run_v560_matching_sensitivity_protocol,
)
from experiments.experiment_v560_matching_sensitivity_subset import (
    run_v560_matching_sensitivity_subset,
)
from experiments.experiment_v561_section6_5_matching_rule_robustness import (
    run_v561_section6_5_matching_rule_robustness,
)
from experiments.experiment_v562_section6_reviewer_readiness_synthesis import (
    run_v562_section6_reviewer_readiness_synthesis,
)
from experiments.experiment_v563_section6_failure_region_dominance_synthesis import (
    run_v563_section6_failure_region_dominance_synthesis,
)
from experiments.experiment_v564_section6_3a_challenging_family_variant_extension import (
    run_v564_section6_3a_challenging_family_variant_extension,
)
from experiments.experiment_v564a_section6_3a_canonical_anchor_reference import (
    run_v564a_section6_3a_canonical_anchor_reference,
)
from experiments.experiment_v565_section6_3a_canonical_structured_perturbation import (
    run_v565_section6_3a_canonical_structured_perturbation,
)
from experiments.experiment_v566_section6_3a_two_axis_interaction_design import (
    run_v566_section6_3a_two_axis_interaction_design,
)
from experiments.experiment_v567_section6_3a_analysis_qualification_freeze import (
    run_v567_section6_3a_analysis_qualification_freeze,
)
from experiments.experiment_proxy_metric_validation import run_proxy_metric_validation
from experiments.experiment_real_data_protocol import run_real_data_protocol_scaffold
from experiments.experiment_real_data_development_run import run_real_data_development_run
from experiments.experiment_real_data_ecg_operational_smoke import run_real_data_ecg_operational_smoke
from experiments.experiment_real_data_split_audit import run_real_data_split_audit
from experiments.experiment_real_data_semisynthetic import run_real_data_semisynthetic_contamination
from experiments.experiment_real_world_validation import run_real_world_validation
from experiments.experiment_real_world_validation_schema import (
    run_real_world_denominator_degeneracy_audit,
    run_real_world_endpoint_formula_audit,
    run_real_world_method_neutrality_audit,
    run_real_world_perturbation_protocol_freeze,
    run_real_world_validation_preflight,
    run_real_world_validation_schema_draft,
)
from experiments.experiment_real_world_v1_schema_smoke import (
    run_real_world_v1_schema_single_window_smoke,
)
from experiments.experiment_structural_metric_validation import run_structural_metric_validation_slice
from experiments.experiment_parameter_transfer_analysis import run_parameter_transfer_analysis
from experiments.experiment_benchmark_workflow_gates import (
    mark_benchmark_executed,
    mark_benchmark_executing,
    mark_statistics_completed,
    load_benchmark_state,
    require_benchmark_state,
    run_benchmark_eligibility_gate,
    run_benchmark_execution_audit,
    run_ceemdan_noise_endpoint_adjudication,
    run_rebuild_universal_noise_endpoints_gate,
    run_true_noise_separation_adapter_spec,
    run_controlled_noise_endpoint_rerun_preflight,
    run_synthetic_reconstruction_rule_discovery,
    run_synthetic_reconstruction_protocol_amendment,
    run_qualify_synthetic_reconstruction_protocol,
    run_controlled_noise_endpoint_re_evaluation,
    run_noise_endpoint_merge_audit,
    _utc_now,
    _write_state,
)
from diagnostics.shared_physical_diagnostics import SYNTHETIC_RECONSTRUCTION_PROTOCOL_ID
from algorithm.paper_section_map import write_paper_section_map

LOCKED_LOSS_NAME = "gaussian_smoothed_median"
V539_PARAMETER_SELECTION_PREFIX = "V5.39"
V539_H_TUNING_AMENDMENT_REL = (
    Path("16k_v539_irmf_h_tuning_amendment")
    / "v539_irmf_h_tuning_status.json"
)
V545_RELATIVE_H_QUALIFICATION_REL = (
    Path("16q_v545_relative_h_scale_estimator_qualification")
    / "v545_relative_H_scale_estimator_qualification_dashboard.json"
)
V548_PARAMETER_SELECTION_ADJUDICATION_REL = (
    Path("16r_v548_parameter_selection_adjudication")
    / "v548_parameter_selection_adjudication_dashboard.json"
)


def _parameter_lock_audit(parameter_source, n_irmf_candidates=0):
    return {
        "selection_dataset": (
            "canonical_development_v2_60case_target_snr"
            if parameter_source == "section_4_development_selection"
            else None
        ),
        "parameter_source": parameter_source,
        "n_development_cases": int(
            len(PARAMETER_SELECTION_SIGNALS)
            * len(PARAMETER_SELECTION_NOISES)
            * len(PARAMETER_SELECTION_TARGET_SNR_DB_LEVELS)
        ),
        "development_signals": list(PARAMETER_SELECTION_SIGNALS),
        "development_noises": list(PARAMETER_SELECTION_NOISES),
        "development_sigmas_legacy_not_active_for_v547": list(PARAMETER_SELECTION_SIGMAS),
        "development_severity_design_version": PARAMETER_SELECTION_SEVERITY_DESIGN_VERSION,
        "development_severity_axis": "target_snr_db",
        "development_target_snr_db_levels": list(PARAMETER_SELECTION_TARGET_SNR_DB_LEVELS),
        "benchmark_protocol_family": {
            "benchmark_design_version": "V5.30_target_snr_main_benchmark",
            "scale_estimator_qualification_protocol": "V5.45",
            "parameter_selection_protocol": "V5.47",
            "parameter_adjudication_protocol": "V5.48",
            "selection_execution_protocol": "V5.49",
            "versioning_note": (
                "V5.45/V5.47/V5.48/V5.49 are parameter-provenance modules; "
                "they do not supersede the V5.30 scientific benchmark design."
            ),
        },
        "selection_metric": "maximum mean development-set case_score",
        "locked_before_main_benchmark": parameter_source == "section_4_development_selection",
        "challenging_suite_excluded_from_selection": True,
        "canonical_benchmark_excluded_from_selection": True,
        "emd_family_final_rankings_excluded_from_selection": True,
        "formula_instance_policy": (
            "class-overlap allowed between development and canonical benchmark; "
            "formula instances, parameters, transient locations, and random "
            "realizations are separated"
        ),
        "canonical_benchmark_role": "held-out formula-instance evaluation within the canonical signal taxonomy",
        "challenging_benchmark_role": "out-of-development structural generalization benchmark",
        "n_irmf_candidates": int(n_irmf_candidates),
    }


def enforce_locked_irmf_loss(params):
    """Return a copy with the paper loss locked, or fail on conflicting input."""
    out = dict(params or {})
    supplied_name = out.get("loss_name", LOCKED_LOSS_NAME)
    if supplied_name != LOCKED_LOSS_NAME:
        raise ValueError(
            "Algorithm layer requires loss_name='gaussian_smoothed_median'; "
            f"received {supplied_name!r}. Alternative losses belong to methodology/."
        )
    if out.get("H_parameterization") == "relative_noise_scale":
        out["loss_name"] = LOCKED_LOSS_NAME
        out.setdefault("scale_estimator_id", "first_difference_mad_truth_free")
        out.pop("loss_tuning", None)
        out.pop("H", None)
        return out
    supplied_h = float(out.get("H", GLOBAL_IRMF_PARAMS.get("H", 1.0)))
    out["loss_name"] = LOCKED_LOSS_NAME
    out["loss_tuning"] = {"H": supplied_h}
    out["H"] = supplied_h
    return out


def _load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _require_relative_h_parameter_selection(root, locked):
    """Guard formal target-SNR reruns against stale absolute-H IRMF locks."""
    amendment = _load_json(root / V539_H_TUNING_AMENDMENT_REL, default={})
    relative_h_qualification = _load_json(root / V545_RELATIVE_H_QUALIFICATION_REL, default={})
    adjudication = _load_json(root / V548_PARAMETER_SELECTION_ADJUDICATION_REL, default={})
    selection_protocol_path = (
        root / "01_global_parameter_selection" / "section_4_parameter_selection_protocol.json"
    )
    selection_protocol = _load_json(selection_protocol_path, default={})
    selected = selection_protocol.get("selected_global_irmf_params", {})
    protocol_version = str(selection_protocol.get("protocol_version", ""))
    c_h_options = selection_protocol.get("robust_loss", {}).get("c_H_options", [])
    checks = {
        "v539_h_amendment_present": (
            amendment.get("schema_version") == "V5.39_irmf_h_tuning_protocol_amendment"
        ),
        "v545_relative_h_qualification_passed": (
            relative_h_qualification.get("qualification_passed") is True
            and relative_h_qualification.get("activation_status")
            == "qualified_for_relative_H_development_selection"
        ),
        "parameter_selection_protocol_is_v547": protocol_version.startswith(
            "V5.47"
        ),
        "development_case_count_is_60": (
            selection_protocol.get("development_set", {}).get("n_development_cases") == 60
        ),
        "development_severity_design_is_target_snr": (
            selection_protocol.get("development_set", {}).get("severity_axis")
            == "target_snr_db"
        ),
        "relative_H_parameterization_in_selection_protocol": (
            selection_protocol.get("robust_loss", {}).get("H_parameterization")
            == "relative_noise_scale"
        ),
        "c_H_tuned_in_selection_protocol": (
            selection_protocol.get("robust_loss", {}).get("c_H_tuned") is True
        ),
        "c_H_grid_has_multiple_levels": len(c_h_options) > 1,
        "selected_params_include_c_H": selected.get("c_H") is not None,
        "locked_IRMF_includes_selected_c_H": (
            selected.get("c_H") is not None
            and abs(float(locked.get("IRMF", {}).get("c_H", float("nan"))) - float(selected["c_H"])) < 1e-12
        ),
        "locked_IRMF_uses_relative_H": (
            locked.get("IRMF", {}).get("H_parameterization") == "relative_noise_scale"
        ),
        "parameter_source_is_development_selection": (
            locked.get("parameter_source") == "section_4_development_selection"
        ),
        "v548_parameter_selection_adjudication_passed": (
            adjudication.get("parameter_lock_authorized") is True
            and adjudication.get("adjudication_status") == "passed_parameter_lock_authorized"
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(
            "V5.30 target-SNR formal rerun is blocked until V5.47 "
            "relative-H IRMF development selection has completed. "
            f"Failed checks: {failed}. Run "
            "`python paper_pipeline.py parameter-selection --force-parameter-selection "
            "--output-root <root>` after V5.45 relative-H qualification, then retry "
            "the target-SNR rerun."
        )
    return {
        "relative_H_parameter_selection_gate_status": "passed",
        "checks": checks,
        "amendment_path": str(root / V539_H_TUNING_AMENDMENT_REL),
        "relative_h_qualification_path": str(root / V545_RELATIVE_H_QUALIFICATION_REL),
        "parameter_adjudication_path": str(root / V548_PARAMETER_SELECTION_ADJUDICATION_REL),
        "parameter_adjudication_status": adjudication.get("adjudication_status"),
        "selection_protocol_path": str(selection_protocol_path),
        "selection_protocol_version": protocol_version,
        "selected_IRMF_params": {
            key: selected.get(key)
            for key in ("h1", "a", "h_min", "c_H")
        },
    }


def _prepare_locked_parameters(root, calibrate=True):
    root = ensure_dir(root)
    selected, candidate_summary, _ = run_global_parameter_selection(root / "01_global_parameter_selection")
    irmf = dict(GLOBAL_IRMF_PARAMS)
    if selected:
        keys = (
            ("h1", "a", "h_min", "c_H", "boundary_mode", "min_support_points")
            if selected.get("H_parameterization") == "relative_noise_scale"
            else ("h1", "a", "h_min", "H", "boundary_mode", "min_support_points")
        )
        for key in keys:
            if key in selected:
                irmf[key] = selected[key]
        if selected.get("H_parameterization") == "relative_noise_scale":
            irmf["H_parameterization"] = "relative_noise_scale"
            irmf["scale_estimator_id"] = selected.get("scale_estimator_id")
            irmf.pop("H", None)
            irmf.pop("loss_tuning", None)
    irmf = enforce_locked_irmf_loss(irmf)

    emd = dict(GLOBAL_EMD_PARAMS)
    eemd = dict(GLOBAL_EEMD_PARAMS)
    ceemdan = dict(GLOBAL_CEEMDAN_PARAMS)
    calibration = None
    if calibrate:
        calibration = run_emd_family_parameter_calibration(root / "02_emd_family_calibration", emd_params=emd)
        if isinstance(calibration, dict):
            emd.update(calibration.get("EMD", {}))
            eemd.update(calibration.get("EEMD", {}))
            ceemdan.update(calibration.get("CEEMDAN", {}))

    locked = {
        "IRMF": irmf,
        "EMD": emd,
        "EEMD": eemd,
        "CEEMDAN": ceemdan,
        "rho_lock": (
            {
                "loss_name": LOCKED_LOSS_NAME,
                "H_parameterization": "relative_noise_scale",
                "c_H": float(irmf["c_H"]),
                "scale_estimator_id": irmf.get("scale_estimator_id"),
                "H_realized_rule": "H_case = c_H * sigma_hat(Y_case)",
            }
            if irmf.get("H_parameterization") == "relative_noise_scale"
            else {"loss_name": LOCKED_LOSS_NAME, "H": float(irmf["H"])}
        ),
        "n_irmf_candidates": len(candidate_summary),
        "emd_family_calibration": calibration,
        "parameter_source": "section_4_development_selection",
        "parameter_lock_audit": _parameter_lock_audit(
            "section_4_development_selection",
            n_irmf_candidates=len(candidate_summary),
        ),
    }
    write_json(locked, root / "locked_algorithm_parameters.json")
    return locked


def _load_completed_development_selection_locked_parameters(root, calibrate=True):
    """Build locked parameters from completed current-code Section 4 artifacts."""
    root = ensure_dir(root)
    selection_root = root / "01_global_parameter_selection"
    protocol = _load_json(selection_root / "section_4_parameter_selection_protocol.json")
    candidate_summary = _load_json(selection_root / "section_4_candidate_summary.json", default=[])
    if not protocol or not protocol.get("selected_global_irmf_params"):
        return None
    if not str(protocol.get("protocol_version", "")).startswith("V5.47"):
        return None
    selected = protocol["selected_global_irmf_params"]
    if selected.get("H_parameterization") == "relative_noise_scale":
        required = ("h1", "a", "h_min", "c_H", "boundary_mode", "min_support_points")
    else:
        required = ("h1", "a", "h_min", "H", "boundary_mode", "min_support_points")
    if any(key not in selected for key in required):
        return None

    irmf = dict(GLOBAL_IRMF_PARAMS)
    for key in required:
        irmf[key] = selected[key]
    if selected.get("H_parameterization") == "relative_noise_scale":
        irmf["H_parameterization"] = "relative_noise_scale"
        irmf["scale_estimator_id"] = selected.get("scale_estimator_id")
        irmf.pop("H", None)
        irmf.pop("loss_tuning", None)
    irmf = enforce_locked_irmf_loss(irmf)

    emd = dict(GLOBAL_EMD_PARAMS)
    eemd = dict(GLOBAL_EEMD_PARAMS)
    ceemdan = dict(GLOBAL_CEEMDAN_PARAMS)
    calibration = None
    if calibrate:
        calibration = run_emd_family_parameter_calibration(root / "02_emd_family_calibration", emd_params=emd)
        if isinstance(calibration, dict):
            emd.update(calibration.get("EMD", {}))
            eemd.update(calibration.get("EEMD", {}))
            ceemdan.update(calibration.get("CEEMDAN", {}))

    locked = {
        "IRMF": irmf,
        "EMD": emd,
        "EEMD": eemd,
        "CEEMDAN": ceemdan,
        "rho_lock": (
            {
                "loss_name": LOCKED_LOSS_NAME,
                "H_parameterization": "relative_noise_scale",
                "c_H": float(irmf["c_H"]),
                "scale_estimator_id": irmf.get("scale_estimator_id"),
                "H_realized_rule": "H_case = c_H * sigma_hat(Y_case)",
            }
            if irmf.get("H_parameterization") == "relative_noise_scale"
            else {"loss_name": LOCKED_LOSS_NAME, "H": float(irmf["H"])}
        ),
        "n_irmf_candidates": len(candidate_summary),
        "emd_family_calibration": calibration,
        "parameter_source": "section_4_development_selection",
        "parameter_source_detail": "recovered_from_completed_current_code_development_selection_artifacts",
        "development_selection_protocol_path": str(selection_root / "section_4_parameter_selection_protocol.json"),
        "selected_global_irmf_params": selected,
        "parameter_lock_audit": _parameter_lock_audit(
            "section_4_development_selection",
            n_irmf_candidates=len(candidate_summary),
        ),
    }
    write_json(locked, root / "locked_algorithm_parameters.json")
    return locked


def run_metric_taxonomy_stage(root):
    """Write publication-facing metric taxonomy without rerunning experiments."""
    root = ensure_dir(root)
    taxonomy = write_metric_taxonomy(root / "05b_metric_taxonomy")
    write_json({
        "stage": "metric_taxonomy",
        "status": "completed",
        "scope": "publication_taxonomy_only",
        "output": "algorithm/05b_metric_taxonomy",
    }, root / "metric_taxonomy_dashboard.json")
    return taxonomy


def load_or_prepare_locked_parameters(output_root, calibrate=True, force_reselect=False):
    root = ensure_dir(output_root)
    path = root / "locked_algorithm_parameters.json"
    if force_reselect:
        recovered = _load_completed_development_selection_locked_parameters(
            root,
            calibrate=calibrate,
        )
        if recovered is not None:
            return recovered
        return _prepare_locked_parameters(root, calibrate=calibrate)
    locked = _load_json(path)
    if locked is None:
        return _prepare_locked_parameters(root, calibrate=calibrate)
    locked["IRMF"] = enforce_locked_irmf_loss(locked["IRMF"])
    return locked


def current_config_locked_parameters():
    """Use project_config fixed parameters without running Section 4 calibration."""
    irmf = enforce_locked_irmf_loss(dict(GLOBAL_IRMF_PARAMS))
    return {
        "IRMF": irmf,
        "EMD": dict(GLOBAL_EMD_PARAMS),
        "EEMD": dict(GLOBAL_EEMD_PARAMS),
        "CEEMDAN": dict(GLOBAL_CEEMDAN_PARAMS),
        "rho_lock": {"loss_name": LOCKED_LOSS_NAME, "H": float(irmf["H"])},
        "n_irmf_candidates": 0,
        "emd_family_calibration": None,
        "parameter_source": "project_config_current_fixed_params",
        "parameter_lock_audit": _parameter_lock_audit(
            "project_config_current_fixed_params",
            n_irmf_candidates=0,
        ),
    }


def _write_locked_parameter_audit(root, locked):
    """Persist the exact locked parameters used by a stage."""
    root = ensure_dir(root)
    locked = dict(locked)
    locked.setdefault("parameter_lock_audit", _parameter_lock_audit(
        locked.get("parameter_source", "unknown")
    ))
    write_json(locked, root / "locked_algorithm_parameters.json")
    write_json({
        "parameter_source": locked.get("parameter_source"),
        "rho_lock": locked.get("rho_lock"),
        "IRMF": locked.get("IRMF"),
        "EMD": locked.get("EMD"),
        "EEMD": locked.get("EEMD"),
        "CEEMDAN": locked.get("CEEMDAN"),
        "parameter_lock_audit": locked.get("parameter_lock_audit"),
    }, root / "locked_parameter_audit.json")
    return locked


def run_parameter_selection_stage(output_root, calibrate=True):
    root = ensure_dir(output_root)
    result = _prepare_locked_parameters(root, calibrate=calibrate)
    write_paper_section_map(root)
    return result


def run_paper_section_map_stage(output_root):
    """Refresh the paper-facing section/output correspondence map."""
    return write_paper_section_map(ensure_dir(output_root))


def run_v554_noise_realization_stability_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v554_noise_realization_decomposition_stability(root)
    write_json({
        "scope": "supplementary_noise_realization_decomposition_stability",
        "no_fixed_input_stochastic_repeatability": True,
        "no_algorithm_rerun": True,
        "no_primary_endpoint_change": True,
        "no_ranking_bearing_claim": True,
        "result": result,
    }, root / "v554_noise_realization_stability_dashboard.json")
    write_paper_section_map(root)
    return result


def run_v555_noise_realization_stability_qualification_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v555_noise_realization_stability_qualification(root)
    write_json({
        "scope": "supplementary_noise_realization_stability_qualification",
        "no_fixed_input_stochastic_repeatability": True,
        "no_algorithm_rerun": True,
        "no_primary_endpoint_change": True,
        "no_ranking_bearing_claim": True,
        "result": result,
    }, root / "v555_noise_realization_stability_qualification_dashboard.json")
    write_paper_section_map(root)
    return result


def run_v556_cross_realization_waveform_stability_protocol_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v556_cross_realization_waveform_stability_protocol(
        root / "16y_v556_cross_realization_waveform_stability"
    )
    write_json({
        "scope": "supplementary_cross_realization_waveform_stability_protocol",
        "no_algorithm_rerun": True,
        "no_primary_endpoint_change": True,
        "no_ranking_bearing_claim": True,
        "result": result,
    }, root / "v556_cross_realization_waveform_stability_dashboard.json")
    write_paper_section_map(root)
    return result


def run_v557_section6_challenging_signal_extension_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v557_section6_challenging_signal_extension_amendment(
        root / "16z_v557_section6_challenging_signal_extension"
    )
    write_json({
        "scope": "section6_challenging_signal_extension_amendment",
        "no_algorithm_rerun": True,
        "no_primary_endpoint_change": True,
        "result": result,
    }, root / "v557_section6_challenging_signal_extension_dashboard.json")
    write_paper_section_map(root)
    return result


def run_parameter_transfer_stage(output_root, calibrate=True, use_current_config_params=False):
    """Run development-test independence and parameter-selection stability diagnostics."""
    root = ensure_dir(output_root)
    locked = (
        current_config_locked_parameters()
        if use_current_config_params
        else load_or_prepare_locked_parameters(root, calibrate=calibrate)
    )
    locked = _write_locked_parameter_audit(root, locked)
    result = run_parameter_transfer_analysis(
        root / "16_parameter_transfer_analysis",
        reference_params=locked["IRMF"],
    )
    write_json({
        "scope": "parameter_transfer_and_selection_stability",
        "parameter_source": locked.get("parameter_source"),
        "rho_lock": locked.get("rho_lock"),
        "parameter_lock_audit": locked.get("parameter_lock_audit"),
        "result": result,
    }, root / "parameter_transfer_dashboard.json")
    write_paper_section_map(root)
    return result


def run_benchmark_stage(output_root, calibrate=True, full_family=True):
    root = ensure_dir(output_root)
    locked = load_or_prepare_locked_parameters(root, calibrate=calibrate)
    irmf, emd = locked["IRMF"], locked["EMD"]
    family_rows = family_summary = None
    if full_family:
        family_rows, family_summary = run_fixed_emd_family_benchmark(
            root / "03_canonical_emd_family_benchmark",
            irmf_params=irmf,
            emd_params=emd,
            eemd_params=locked["EEMD"],
            ceemdan_params=locked["CEEMDAN"],
            mode="full",
            return_rows=True,
        )
    out = {
        "family_rows": family_rows,
        "family_summary": family_summary,
    }
    write_json({
        "section_5_primary_benchmark": "algorithm/03_canonical_emd_family_benchmark",
        "family_summary": family_summary,
        "rho_lock": locked["rho_lock"],
        "paper_note": (
            "Section 5 is the four-method EMD-family benchmark. Pairwise "
            "comparisons, including IRMF-vs-EMD, are derived from the same "
            "IRMF/EMD/EEMD/CEEMDAN rows; no separate two-method benchmark is run."
        ),
    }, root / "benchmark_dashboard.json")
    write_paper_section_map(root)
    return out


def run_unified_benchmark_cube_stage(
        output_root,
        calibrate=True,
        quick=False,
        seeds=None,
        timeout_seconds=None,
        use_current_config_params=False,
        force_parameter_selection=False,
):
    root = ensure_dir(output_root)
    mark_benchmark_executing(root)
    locked = (
        current_config_locked_parameters()
        if use_current_config_params
        else load_or_prepare_locked_parameters(
            root,
            calibrate=calibrate,
            force_reselect=force_parameter_selection,
        )
    )
    if locked.get("parameter_source") != "section_4_development_selection":
        raise RuntimeError(
            "Formal V5.24 benchmark execution requires parameters locked by "
            "the current code development-set selection protocol "
            "(parameter_source='section_4_development_selection'). "
            f"Current parameter_source={locked.get('parameter_source')!r}. "
            "Run benchmark-eligibility-gate/unified-cube without "
            "--use-current-config-params and with --force-parameter-selection "
            "when a stale locked parameter file exists."
        )
    locked = _write_locked_parameter_audit(root, locked)
    rows, dashboard = run_unified_benchmark_cube(
        root / "03_unified_benchmark_cube",
        irmf_params=locked["IRMF"],
        emd_params=locked["EMD"],
        eemd_params=locked["EEMD"],
        ceemdan_params=locked["CEEMDAN"],
        seeds=(tuple(seeds) if seeds is not None else None) or (0,),
        timeout_seconds=timeout_seconds or 120,
        quick=quick,
        parameter_source=locked.get("parameter_source"),
        parameter_lock_audit=locked.get("parameter_lock_audit"),
    )
    dashboard["rho_lock"] = locked["rho_lock"]
    dashboard["parameter_source"] = locked.get("parameter_source", "locked_algorithm_parameters")
    dashboard["parameter_lock_audit"] = locked.get("parameter_lock_audit")
    write_json(dashboard, root / "unified_benchmark_cube_dashboard.json")
    mark_benchmark_executed(root, dashboard=dashboard)
    write_paper_section_map(root)
    return {"rows": rows, "dashboard": dashboard}


def run_benchmark_eligibility_gate_stage(
        output_root,
        calibrate=True,
        use_current_config_params=False,
        qualification_root=None,
        seeds=None,
        force_parameter_selection=False,
):
    root = ensure_dir(output_root)
    locked = (
        current_config_locked_parameters()
        if use_current_config_params
        else load_or_prepare_locked_parameters(
            root,
            calibrate=calibrate,
            force_reselect=force_parameter_selection,
        )
    )
    locked = _write_locked_parameter_audit(root, locked)
    result = run_benchmark_eligibility_gate(
        root,
        locked_parameters=locked,
        qualification_root=qualification_root,
        seeds=seeds,
    )
    write_json({
        "scope": "stage_A_benchmark_eligibility_gate_only",
        "no_algorithm_runs": True,
        "no_statistical_analysis": True,
        "no_method_performance_interpretation": True,
        "result": result,
    }, root / "benchmark_eligibility_gate_dashboard.json")
    write_paper_section_map(root)
    return result


def run_benchmark_execution_audit_stage(output_root):
    root = ensure_dir(output_root)
    result = run_benchmark_execution_audit(root)
    write_json({
        "scope": "stage_C_execution_integrity_audit_only",
        "no_algorithm_runs": True,
        "no_statistical_ranking": True,
        "no_method_performance_interpretation": True,
        "result": result,
    }, root / "benchmark_execution_audit_dashboard.json")
    write_paper_section_map(root)
    return result


def run_ceemdan_noise_endpoint_adjudication_stage(output_root):
    root = ensure_dir(output_root)
    result = run_ceemdan_noise_endpoint_adjudication(root)
    write_json({
        "scope": "stage_C1_ceemdan_noise_endpoint_adjudication_only",
        "no_algorithm_runs": True,
        "no_statistical_ranking": True,
        "no_method_performance_interpretation": True,
        "result": result["ceemdan_noise_endpoint_adjudication"],
    }, root / "ceemdan_noise_endpoint_adjudication_dashboard.json")
    write_paper_section_map(root)
    return result


def run_rebuild_universal_noise_endpoints_stage(output_root):
    root = ensure_dir(output_root)
    result = run_rebuild_universal_noise_endpoints_gate(root)
    write_json({
        "scope": "stage_C2_universal_noise_endpoint_rebuild_decision_only",
        "no_algorithm_runs": True,
        "no_statistical_ranking": True,
        "no_method_performance_interpretation": True,
        "result": result["noise_endpoint_rebuild_decision"],
    }, root / "noise_endpoint_rebuild_decision_dashboard.json")
    write_paper_section_map(root)
    return result


def run_true_noise_separation_adapter_spec_stage(output_root):
    root = ensure_dir(output_root)
    result = run_true_noise_separation_adapter_spec(root)
    write_json({
        "scope": "stage_C3_true_noise_separation_adapter_specification_only",
        "no_algorithm_runs": True,
        "no_statistical_ranking": True,
        "no_method_performance_interpretation": True,
        "result": result["true_noise_separation_adapter_spec"],
    }, root / "true_noise_separation_adapter_dashboard.json")
    write_paper_section_map(root)
    return result


def run_controlled_noise_endpoint_rerun_preflight_stage(output_root):
    root = ensure_dir(output_root)
    result = run_controlled_noise_endpoint_rerun_preflight(root)
    write_json({
        "scope": "stage_C4_controlled_noise_endpoint_rerun_preflight_only",
        "no_algorithm_runs": True,
        "no_statistical_ranking": True,
        "no_method_performance_interpretation": True,
        "result": result["controlled_rerun_preflight_audit"],
    }, root / "controlled_noise_endpoint_rerun_preflight_dashboard.json")
    write_paper_section_map(root)
    return result


def run_synthetic_reconstruction_rule_discovery_stage(output_root):
    root = ensure_dir(output_root)
    result = run_synthetic_reconstruction_rule_discovery(root)
    write_json({
        "scope": "stage_C3b_synthetic_reconstruction_rule_discovery_only",
        "no_algorithm_runs": True,
        "no_statistical_ranking": True,
        "no_method_performance_interpretation": True,
        "result": result["synthetic_reconstruction_rule_discovery"],
    }, root / "synthetic_reconstruction_rule_discovery_dashboard.json")
    write_paper_section_map(root)
    return result


def run_synthetic_reconstruction_protocol_amendment_stage(output_root):
    root = ensure_dir(output_root)
    result = run_synthetic_reconstruction_protocol_amendment(root)
    write_json({
        "scope": "stage_C3c_synthetic_reconstruction_protocol_amendment_proposal_only",
        "no_algorithm_runs": True,
        "no_endpoint_recomputation": True,
        "no_statistical_ranking": True,
        "no_method_performance_interpretation": True,
        "result": result["synthetic_reconstruction_protocol_amendment"],
    }, root / "synthetic_reconstruction_protocol_amendment_dashboard.json")
    write_paper_section_map(root)
    return result


def run_qualify_synthetic_reconstruction_protocol_stage(output_root):
    root = ensure_dir(output_root)
    result = run_qualify_synthetic_reconstruction_protocol(root)
    write_json({
        "scope": "stage_C3d_to_C3g_synthetic_reconstruction_protocol_qualification_only",
        "no_algorithm_runs": True,
        "no_endpoint_recomputation": True,
        "no_statistical_ranking": True,
        "no_method_performance_interpretation": True,
        "result": result["synthetic_reconstruction_protocol_qualification"],
    }, root / "synthetic_reconstruction_protocol_qualification_dashboard.json")
    write_paper_section_map(root)
    return result


def run_controlled_noise_endpoint_re_evaluation_stage(
        output_root,
        calibrate=True,
        use_current_config_params=False,
        timeout_seconds=120,
        seeds=None,
):
    root = ensure_dir(output_root)
    if use_current_config_params:
        raise RuntimeError(
            "controlled-noise-endpoint-rerun must use the locked parameters "
            "from the executed benchmark root; --use-current-config-params is "
            "not allowed for C4."
        )
    locked = load_or_prepare_locked_parameters(root, calibrate=calibrate)
    result = run_controlled_noise_endpoint_re_evaluation(
        root,
        locked_parameters=locked,
        timeout_seconds=timeout_seconds,
        seeds=seeds,
    )
    write_json({
        "scope": "stage_C4_protocol_controlled_noise_endpoint_re_evaluation_only",
        "old_benchmark_metrics_recomputed": False,
        "old_benchmark_metrics_overwritten": False,
        "no_statistical_ranking": True,
        "no_method_performance_interpretation": True,
        "result": result["controlled_noise_endpoint_re_evaluation_status"],
    }, root / "controlled_noise_endpoint_re_evaluation_dashboard.json")
    write_paper_section_map(root)
    return result


def _protocol_metric_dependency_map():
    protocol_signal = {
        "denoise_nmse", "denoise_mse", "denoise_psnr", "denoise_corr",
        "spectral_corr", "reconstruction_score", "input_snr_db",
        "output_snr_db", "snr_gain_db", "outlier_resistance_index", "ori",
        "clean_region_nmse", "clean_region_preservation_score",
        "contaminated_region_nmse", "contaminated_region_preservation_score",
        "contamination_spillover_error", "contamination_spillover_score",
        "case_score", "case_score_final", "general_physical_score",
        "contamination_resistance_score", "robustness_score",
    }
    protocol_noise = {
        "noise_capture_corr", "noise_capture_corr_score",
        "noise_capture_energy_ratio", "signal_leakage_into_noise",
        "signal_leakage_into_noise_score", "signal_leakage_residual_energy",
        "signal_leakage_observed_energy", "signal_leakage_degenerate_flag",
        "clean_region_signal_leakage", "clean_region_signal_leakage_score",
        "noise_energy_ratio", "noise_energy_log_error",
        "remaining_noise_energy_ratio",
    }
    component_association = {
        "imf_recovery_corr", "imf_recovery_nrmse", "imf_recovery_rmse",
        "imf_recovery_score", "component_splitting_index",
        "component_merging_index", "missing_true_component_rate",
        "spurious_mode_energy_ratio", "unmatched_estimated_component_rate",
    }
    decomposition_only = {
        "imf_count", "effective_imf_count", "true_component_count",
        "decomposition_count_error", "inter_imf_entanglement_index",
        "mode_mixing_index", "mode_mixing_index_legacy", "spectral_leakage",
        "strict_io", "frequency_overlap_max_offdiag",
        "frequency_separation_score", "over_decomposition_penalty",
        "under_decomposition_index",
    }
    rows = []
    for metric in sorted(protocol_signal | protocol_noise | component_association | decomposition_only):
        rows.append({
            "metric_name": metric,
            "depends_on_decomposition": True,
            "depends_on_component_association": metric in component_association,
            "depends_on_protocol_estimated_signal": metric in protocol_signal,
            "depends_on_protocol_estimated_noise": metric in protocol_noise,
            "expected_to_match_original": metric in (component_association | decomposition_only),
            "expected_change_reason": (
                "authorized_reconstruction_protocol_amendment"
                if metric in (protocol_signal | protocol_noise)
                else "none_expected_except_numerical_drift"
            ),
        })
    return rows


def _is_finite_scalar(value):
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def _synthetic_metric_applicable(row, metric):
    contamination_regimes = {"impulsive", "burst", "huber_contamination"}
    conditional_metrics = {"outlier_resistance_index", "clean_region_nmse"}
    return metric not in conditional_metrics or row.get("noise") in contamination_regimes


def _method_metric_missingness_rows(rows, metrics):
    grouped = {}
    affected = []
    for row in rows:
        for method in ("IRMF", "EMD", "EEMD", "CEEMDAN"):
            data = row.get(method, {})
            for metric in metrics:
                if not _synthetic_metric_applicable(row, metric):
                    continue
                reason = data.get("protocol_noise_reason_code", "") or ""
                finite = _is_finite_scalar(data.get(metric))
                if finite:
                    reason = "finite"
                key = (
                    method,
                    row.get("signal"),
                    row.get("noise"),
                    row.get("sigma"),
                    metric,
                    reason,
                )
                item = grouped.setdefault(key, {
                    "method": method,
                    "signal": row.get("signal"),
                    "noise": row.get("noise"),
                    "sigma": row.get("sigma"),
                    "metric": metric,
                    "reason_code": reason,
                    "total_rows": 0,
                    "finite_rows": 0,
                    "nonfinite_rows": 0,
                    "example_case_ids": [],
                })
                item["total_rows"] += 1
                if finite:
                    item["finite_rows"] += 1
                else:
                    item["nonfinite_rows"] += 1
                    if len(item["example_case_ids"]) < 5:
                        item["example_case_ids"].append(row.get("cube_cell_id"))
                    affected.append({
                        "cube_cell_id": row.get("cube_cell_id"),
                        "method": method,
                        "signal": row.get("signal"),
                        "noise": row.get("noise"),
                        "sigma": row.get("sigma"),
                        "data_seed": row.get("data_seed"),
                        "metric": metric,
                        "reason_code": reason,
                    })
    out = []
    for item in grouped.values():
        total = max(int(item["total_rows"]), 1)
        item["finite_rate"] = float(item["finite_rows"] / total)
        item["nonfinite_rate"] = float(item["nonfinite_rows"] / total)
        item["example_case_ids"] = ";".join(str(x) for x in item["example_case_ids"])
        out.append(item)
    return sorted(out, key=lambda r: (
        str(r["metric"]), str(r["method"]), str(r["signal"]),
        str(r["noise"]), float(r["sigma"]) if r["sigma"] is not None else -1.0,
        str(r["reason_code"]),
    )), affected


def _metric_specific_complete_block_rows(rows, metrics):
    conditional_metrics = {"outlier_resistance_index", "clean_region_nmse"}
    out = []
    for metric in metrics:
        eligible = [
            row for row in rows
            if _synthetic_metric_applicable(row, metric)
        ]
        complete = 0
        incomplete_examples = []
        for row in eligible:
            method_finite = {
                method: _is_finite_scalar(row.get(method, {}).get(metric))
                for method in ("IRMF", "EMD", "EEMD", "CEEMDAN")
            }
            if all(method_finite.values()):
                complete += 1
            elif len(incomplete_examples) < 10:
                incomplete_examples.append(row.get("cube_cell_id"))
        total = len(eligible)
        out.append({
            "metric": metric,
            "applicability": (
                "conditional_contamination_regimes"
                if metric in conditional_metrics else
                "universal_synthetic_cases"
            ),
            "eligible_case_seed_blocks": int(total),
            "complete_pair_blocks": int(complete),
            "incomplete_pair_blocks": int(total - complete),
            "complete_pair_rate": float(complete / total) if total else np.nan,
            "metric_specific_complete_blocks_required_for_primary_paired_analysis": True,
            "incomplete_examples": ";".join(str(x) for x in incomplete_examples),
        })
    return out


def _entirely_missing_method_regime_metric_rows(missingness_rows):
    grouped = {}
    for row in missingness_rows:
        key = (
            row["method"],
            row["signal"],
            row["noise"],
            row["sigma"],
            row["metric"],
        )
        item = grouped.setdefault(key, {
            "method": row["method"],
            "signal": row["signal"],
            "noise": row["noise"],
            "sigma": row["sigma"],
            "metric": row["metric"],
            "total_rows": 0,
            "finite_rows": 0,
            "nonfinite_rows": 0,
            "reason_codes": [],
        })
        item["total_rows"] += int(row["total_rows"])
        item["finite_rows"] += int(row["finite_rows"])
        item["nonfinite_rows"] += int(row["nonfinite_rows"])
        if row["reason_code"] not in item["reason_codes"]:
            item["reason_codes"].append(row["reason_code"])
    out = []
    for item in grouped.values():
        if item["total_rows"] > 0 and item["finite_rows"] == 0:
            item["reason_codes"] = ";".join(str(x) for x in item["reason_codes"])
            out.append(item)
    return sorted(out, key=lambda r: (
        str(r["metric"]), str(r["method"]), str(r["signal"]),
        str(r["noise"]), float(r["sigma"]) if r["sigma"] is not None else -1.0,
    ))


def _statistical_readiness_summary(
        rows,
        endpoint_nan_rows,
        complete_block_rows,
        unexplained_nan,
        inf_count,
        statistics_authorized,
):
    endpoint_metrics = ("noise_capture_corr", "signal_leakage_into_noise")
    methods = ("IRMF", "EMD", "EEMD", "CEEMDAN")
    total_endpoint_values = int(len(rows) * len(methods) * len(endpoint_metrics))
    explained_nan_count = sum(1 for row in endpoint_nan_rows if bool(row.get("explained")))
    unexplained_nan_count = int(unexplained_nan)
    finite_endpoint_values = int(total_endpoint_values - explained_nan_count - unexplained_nan_count - int(inf_count))
    complete_block_rate_by_metric = []
    for row in complete_block_rows:
        eligible = int(row["eligible_case_seed_blocks"])
        complete = int(row["complete_pair_blocks"])
        rate = float(complete / eligible) if eligible else np.nan
        complete_block_rate_by_metric.append({
            "metric": row["metric"],
            "eligible_case_seed_blocks": eligible,
            "complete_pair_blocks": complete,
            "complete_pair_block_loss": int(eligible - complete),
            "complete_pair_block_rate": rate,
        })
    largest_losses = sorted(
        complete_block_rate_by_metric,
        key=lambda r: (-int(r["complete_pair_block_loss"]), str(r["metric"])),
    )

    summary = {
        "total_endpoint_values": total_endpoint_values,
        "finite_endpoint_values": finite_endpoint_values,
        "explained_noncomputable_values": int(explained_nan_count),
        "explained_noncomputable_rate": (
            float(explained_nan_count / total_endpoint_values)
            if total_endpoint_values else np.nan
        ),
        "unexplained_nan_values": unexplained_nan_count,
        "infinite_values": int(inf_count),
        "complete_pair_blocks_min": int(min(
            (row["complete_pair_blocks"] for row in complete_block_rows),
            default=0,
        )),
        "complete_pair_block_rate_min": float(min(
            (float(row["complete_pair_rate"]) for row in complete_block_rows if np.isfinite(float(row["complete_pair_rate"]))),
            default=np.nan,
        )),
        "largest_complete_block_loss_by_metric": largest_losses[0] if largest_losses else {},
        "statistics_authorized": bool(statistics_authorized),
    }

    by_method = []
    for method in methods:
        total = int(len(rows) * len(endpoint_metrics))
        affected = [row for row in endpoint_nan_rows if row.get("method") == method]
        explained = sum(1 for row in affected if bool(row.get("explained")))
        unexplained = len(affected) - explained
        finite = total - explained - unexplained
        by_method.append({
            "method": method,
            "total_endpoint_values": total,
            "finite_endpoint_values": int(finite),
            "explained_noncomputable_values": int(explained),
            "unexplained_nan_values": int(unexplained),
            "explained_noncomputable_rate": float(explained / total) if total else np.nan,
        })

    by_reason_map = {}
    for row in endpoint_nan_rows:
        reason = row.get("reason_code") or "missing_reason_code"
        item = by_reason_map.setdefault(reason, {
            "reason_code": reason,
            "values": 0,
            "explained_values": 0,
            "unexplained_values": 0,
        })
        item["values"] += 1
        item["explained_values"] += int(bool(row.get("explained")))
        item["unexplained_values"] += int(not bool(row.get("explained")))
    by_reason = sorted(by_reason_map.values(), key=lambda r: str(r["reason_code"]))

    by_regime_map = {}
    for row in endpoint_nan_rows:
        key = (
            row.get("signal"),
            row.get("noise"),
            row.get("sigma"),
            row.get("reason_code") or "missing_reason_code",
        )
        item = by_regime_map.setdefault(key, {
            "signal": row.get("signal"),
            "noise": row.get("noise"),
            "sigma": row.get("sigma"),
            "reason_code": row.get("reason_code") or "missing_reason_code",
            "values": 0,
            "methods": set(),
            "example_case_ids": [],
        })
        item["values"] += 1
        item["methods"].add(row.get("method"))
        if len(item["example_case_ids"]) < 5:
            item["example_case_ids"].append(row.get("cube_cell_id"))
    by_regime = []
    for item in by_regime_map.values():
        item = dict(item)
        item["methods"] = ";".join(sorted(str(x) for x in item["methods"]))
        item["example_case_ids"] = ";".join(str(x) for x in item["example_case_ids"])
        by_regime.append(item)
    by_regime = sorted(by_regime, key=lambda r: (
        str(r["signal"]), str(r["noise"]),
        float(r["sigma"]) if r["sigma"] is not None else -1.0,
        str(r["reason_code"]),
    ))

    summary_rows = [
        {"category": "total_endpoint_values", "value": summary["total_endpoint_values"]},
        {"category": "finite_endpoint_values", "value": summary["finite_endpoint_values"]},
        {"category": "explained_noncomputable", "value": summary["explained_noncomputable_values"]},
        {"category": "explained_noncomputable_rate", "value": summary["explained_noncomputable_rate"]},
        {"category": "unexplained_nan", "value": summary["unexplained_nan_values"]},
        {"category": "infinite_values", "value": summary["infinite_values"]},
        {"category": "complete_pair_blocks_min", "value": summary["complete_pair_blocks_min"]},
        {"category": "complete_pair_block_rate_min", "value": summary["complete_pair_block_rate_min"]},
        {"category": "statistics_authorized", "value": summary["statistics_authorized"]},
    ]
    return {
        "summary": summary,
        "summary_rows": summary_rows,
        "by_method": by_method,
        "by_reason_code": by_reason,
        "by_regime": by_regime,
        "complete_block_rate_by_metric": complete_block_rate_by_metric,
    }


def _selection_behavior_diagnostics(rows):
    methods = ("IRMF", "EMD", "EEMD", "CEEMDAN")
    by_method = []
    by_regime_map = {}
    for method in methods:
        total = 0
        all_selected = 0
        zero_noise = 0
        for row in rows:
            data = row.get(method, {})
            if not isinstance(data, dict):
                continue
            total += 1
            is_all = data.get("protocol_selection_status_code") == "all_components_selected"
            is_zero = bool(data.get("protocol_zero_variance_noise_flag"))
            all_selected += int(is_all)
            zero_noise += int(is_zero)
            key = (method, row.get("signal"), row.get("noise"), row.get("sigma"))
            item = by_regime_map.setdefault(key, {
                "method": method,
                "signal": row.get("signal"),
                "noise": row.get("noise"),
                "sigma": row.get("sigma"),
                "total_rows": 0,
                "all_components_selected_rows": 0,
                "zero_variance_protocol_noise_rows": 0,
            })
            item["total_rows"] += 1
            item["all_components_selected_rows"] += int(is_all)
            item["zero_variance_protocol_noise_rows"] += int(is_zero)
        by_method.append({
            "method": method,
            "total_method_rows": int(total),
            "all_components_selected_rows": int(all_selected),
            "all_components_selected_rate": float(all_selected / total) if total else np.nan,
            "zero_variance_protocol_noise_rows": int(zero_noise),
            "zero_variance_protocol_noise_rate": float(zero_noise / total) if total else np.nan,
        })
    by_regime = []
    for item in by_regime_map.values():
        total = max(int(item["total_rows"]), 1)
        item["all_components_selected_rate"] = float(item["all_components_selected_rows"] / total)
        item["zero_variance_protocol_noise_rate"] = float(item["zero_variance_protocol_noise_rows"] / total)
        by_regime.append(item)
    return {
        "by_method": by_method,
        "by_regime": sorted(by_regime, key=lambda r: (
            str(r["method"]), str(r["signal"]), str(r["noise"]),
            float(r["sigma"]) if r["sigma"] is not None else -1.0,
        )),
    }


def _complete_block_composition_diagnostics(rows, metrics):
    methods = ("IRMF", "EMD", "EEMD", "CEEMDAN")
    out_map = {}
    for metric in metrics:
        for row in rows:
            if not _synthetic_metric_applicable(row, metric):
                continue
            key = (metric, row.get("signal"), row.get("noise"), row.get("sigma"))
            item = out_map.setdefault(key, {
                "metric": metric,
                "signal": row.get("signal"),
                "noise": row.get("noise"),
                "sigma": row.get("sigma"),
                "total_blocks": 0,
                "complete_blocks": 0,
                "excluded_blocks": 0,
                "excluded_case_examples": [],
            })
            item["total_blocks"] += 1
            complete = all(_is_finite_scalar(row.get(method, {}).get(metric)) for method in methods)
            item["complete_blocks"] += int(complete)
            item["excluded_blocks"] += int(not complete)
            if not complete and len(item["excluded_case_examples"]) < 5:
                item["excluded_case_examples"].append(row.get("cube_cell_id"))
    out = []
    for item in out_map.values():
        total = max(int(item["total_blocks"]), 1)
        item["complete_block_rate"] = float(item["complete_blocks"] / total)
        item["excluded_block_rate"] = float(item["excluded_blocks"] / total)
        item["excluded_case_examples"] = ";".join(str(x) for x in item["excluded_case_examples"])
        out.append(item)
    return sorted(out, key=lambda r: (
        str(r["metric"]), str(r["signal"]), str(r["noise"]),
        float(r["sigma"]) if r["sigma"] is not None else -1.0,
    ))


def _zero_variance_noise_diagnostic_rows(rows):
    out = []
    for row in rows:
        for method in ("IRMF", "EMD", "EEMD", "CEEMDAN"):
            data = row.get(method, {})
            if not isinstance(data, dict):
                continue
            if data.get("protocol_noise_reason_code") != "not_computable_zero_variance_protocol_noise":
                continue
            out.append({
                "cube_cell_id": row.get("cube_cell_id"),
                "method": method,
                "signal": row.get("signal"),
                "noise": row.get("noise"),
                "sigma": row.get("sigma"),
                "data_seed": row.get("data_seed"),
                "protocol_selection_status_code": data.get("protocol_selection_status_code"),
                "all_components_selected_flag": data.get("protocol_selection_status_code") == "all_components_selected",
                "zero_variance_protocol_noise_flag": bool(data.get("protocol_zero_variance_noise_flag")),
                "noise_energy_ratio": data.get("noise_energy_ratio"),
                "noise_energy_log_error": data.get("noise_energy_log_error"),
                "remaining_noise_energy_ratio": data.get("remaining_noise_energy_ratio"),
                "noise_energy_bias_class": data.get("noise_energy_bias_class"),
                "protocol_estimated_noise_energy": data.get("protocol_estimated_noise_energy"),
            })
    return out


def run_protocol_controlled_full_benchmark_rerun_stage(
        output_root,
        calibrate=True,
        use_current_config_params=False,
        timeout_seconds=120,
        seeds=None,
        force_parameter_selection=False,
        use_v530_target_snr_grid=False,
):
    root = ensure_dir(output_root)
    require_benchmark_state(
        root,
        # This command is a controlled rerun under a newly locked protocol.  A
        # historical benchmark root may already have advanced beyond EXECUTED
        # (for example to STATISTICS_COMPLETED); the rerun-specific manifest and
        # checkpoint guards below still prevent mixing fixed-sigma and V5.30
        # target-SNR artifacts.
        allowed={"EXECUTED", "AUDITED", "STATISTICS_COMPLETED"},
        command_name="protocol-controlled-full-benchmark-rerun",
    )
    if use_current_config_params:
        raise RuntimeError(
            "protocol-controlled-full-benchmark-rerun must use benchmark-root "
            "locked development-selected parameters; --use-current-config-params "
            "is not allowed."
        )
    amendment = _load_json(
        root / "04cc_synthetic_reconstruction_protocol_amendment" / "synthetic_reconstruction_protocol_amendment.json",
        default={},
    )
    if amendment.get("amendment_status") != "specification_frozen" or amendment.get("qualification_status") != "qualified":
        raise RuntimeError("Full protocol rerun requires frozen and qualified reconstruction amendment.")
    target_snr_db_levels = None
    benchmark_run_label = "protocol_controlled_full_benchmark_rerun"
    if use_v530_target_snr_grid:
        snr_schema = _load_json(
            root / "16_v530_snr_design" / "v530_snr_design_frozen.json",
            default={},
        )
        if (
            snr_schema.get("schema_status") != "frozen"
            or snr_schema.get("qualification_status") != "passed"
        ):
            raise RuntimeError(
                "V5.30 target-SNR rerun requires frozen "
                "algorithm/16_v530_snr_design/v530_snr_design_frozen.json."
            )
        target_snr_db_levels = tuple(TARGET_SNR_DB_LEVELS)
        if seeds is None:
            seeds = tuple(V530_BENCHMARK_SEEDS)
        benchmark_run_label = "v530_protocol_controlled_target_snr_full_benchmark_rerun"
    locked = load_or_prepare_locked_parameters(
        root,
        calibrate=calibrate,
        force_reselect=force_parameter_selection,
    )
    locked = _write_locked_parameter_audit(root, locked)
    relative_h_parameter_gate = (
        _require_relative_h_parameter_selection(root, locked)
        if use_v530_target_snr_grid
        else None
    )
    stage_root = root / "06_protocol_controlled_full_benchmark_rerun"
    existing_checkpoint = stage_root / "unified_benchmark_cube_rows.jsonl"
    if use_v530_target_snr_grid and existing_checkpoint.exists():
        existing_rows = []
        with open(existing_checkpoint, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    existing_rows.append(json.loads(raw))
                except Exception:
                    continue
                if len(existing_rows) >= 25:
                    break
        conflicting = [
            row for row in existing_rows
            if row.get("noise_severity_design") not in {None, "target_snr_energy_ratio"}
        ]
        fixed_sigma_rows = [
            row for row in existing_rows
            if row.get("target_snr_db") is None
            or row.get("noise_severity_design") == "fixed_sigma"
        ]
        if conflicting or fixed_sigma_rows:
            raise RuntimeError(
                "V5.30 target-SNR rerun refuses to reuse an existing fixed-sigma "
                "06_protocol_controlled_full_benchmark_rerun checkpoint. Use a "
                "clean V5.30 output root or archive the old fixed-sigma rerun "
                "directory before starting."
            )
    metric_map = _protocol_metric_dependency_map()
    write_json(metric_map, stage_root / "metric_dependency_map.json")
    write_csv(metric_map, stage_root / "metric_dependency_map.csv")
    rows, dashboard = run_unified_benchmark_cube(
        stage_root,
        irmf_params=locked["IRMF"],
        emd_params=locked["EMD"],
        eemd_params=locked["EEMD"],
        ceemdan_params=locked["CEEMDAN"],
        seeds=seeds,
        timeout_seconds=timeout_seconds,
        parameter_source=locked.get("parameter_source"),
        parameter_lock_audit=locked.get("parameter_lock_audit"),
        reconstruction_protocol_id=SYNTHETIC_RECONSTRUCTION_PROTOCOL_ID,
        benchmark_run_label=benchmark_run_label,
        target_snr_db_levels=target_snr_db_levels,
    )
    dashboard["protocol_controlled_full_rerun"] = {
        "benchmark_design_version": (
            "V5.30_target_snr_main_benchmark"
            if use_v530_target_snr_grid
            else "V5.28_fixed_sigma_reference_benchmark"
        ),
        "benchmark_protocol_family": (
            {
                "benchmark_design_version": "V5.30_target_snr_main_benchmark",
                "scale_estimator_qualification_protocol": "V5.45",
                "parameter_selection_protocol": "V5.47",
                "parameter_adjudication_protocol": "V5.48",
                "selection_execution_protocol": "V5.49",
                "interpretation": (
                    "V5.45/V5.47/V5.48/V5.49 document the final IRMF "
                    "parameter provenance. They do not redefine the V5.30 "
                    "scientific benchmark design."
                ),
            }
            if use_v530_target_snr_grid
            else None
        ),
        "reconstruction_protocol_id": SYNTHETIC_RECONSTRUCTION_PROTOCOL_ID,
        "noise_severity_design": (
            "v530_target_snr_energy_ratio"
            if use_v530_target_snr_grid
            else "v528_fixed_sigma"
        ),
        "target_snr_db_levels": (
            list(target_snr_db_levels) if target_snr_db_levels is not None else None
        ),
        "seed_policy_version": (
            V530_SEED_POLICY_VERSION if use_v530_target_snr_grid else None
        ),
        "n_primary_benchmark_seeds": (
            len(V530_BENCHMARK_SEEDS) if use_v530_target_snr_grid else None
        ),
        "seed_convergence_audit_plan": (
            {
                "version": V530_SEED_CONVERGENCE_AUDIT_VERSION,
                "candidate_prefixes": list(V530_SEED_CONVERGENCE_PREFIXES),
                "available_convergence_audit_seeds": list(V530_CONVERGENCE_AUDIT_SEEDS),
                "preferred_design": "same full benchmark cells, nested seed prefixes only",
                "fallback_design": "representative subset spanning major signal, noise, and SNR regimes",
                "role": "independent Monte Carlo qualification; not part of primary SNR aggregation unless promoted after audit",
            }
            if use_v530_target_snr_grid
            else None
        ),
        "relative_H_parameter_selection_gate": relative_h_parameter_gate,
        "amendment_checksum": amendment.get("amendment_checksum"),
        "frozen_rule_checksum": amendment.get("frozen_rule_checksum"),
        "qualification_checksum": amendment.get("qualification_checksum"),
        "primary_statistical_input_if_audit_passes": str(stage_root),
        "legacy_original_benchmark_role": "reproducibility_reference_only",
    }
    write_json(dashboard, root / "protocol_controlled_full_benchmark_rerun_dashboard.json")
    write_paper_section_map(root)
    return {"rows": rows, "dashboard": dashboard}


def run_protocol_full_rerun_audit_stage(output_root, tolerance=1e-9):
    root = ensure_dir(output_root)
    require_benchmark_state(
        root,
        allowed={"EXECUTED"},
        command_name="protocol-full-rerun-audit",
    )
    stage_root = ensure_dir(root / "06_protocol_controlled_full_benchmark_rerun")
    old_rows = _load_json(root / "03_unified_benchmark_cube" / "unified_benchmark_cube_rows.json", default=[])
    new_rows = _load_json(stage_root / "unified_benchmark_cube_rows.json", default=[])
    metric_map = _protocol_metric_dependency_map()
    write_json(metric_map, stage_root / "metric_dependency_map.json")
    write_csv(metric_map, stage_root / "metric_dependency_map.csv")
    expected_match = {row["metric_name"] for row in metric_map if row["expected_to_match_original"]}
    old_by_key = {row.get("cube_cell_id"): row for row in old_rows}
    new_by_key = {row.get("cube_cell_id"): row for row in new_rows}
    missing = sorted(set(old_by_key) - set(new_by_key))
    extra = sorted(set(new_by_key) - set(old_by_key))
    repro_rows = []
    exceed = 0
    for key in sorted(set(old_by_key) & set(new_by_key)):
        old = old_by_key[key]
        new = new_by_key[key]
        for method in ("IRMF", "EMD", "EEMD", "CEEMDAN"):
            od = old.get(method, {})
            nd = new.get(method, {})
            for metric in sorted(expected_match):
                try:
                    ov = float(od.get(metric))
                    nv = float(nd.get(metric))
                except Exception:
                    continue
                if not (np.isfinite(ov) and np.isfinite(nv)):
                    continue
                diff = abs(ov - nv)
                bad = diff > float(tolerance)
                exceed += int(bad)
                if bad:
                    repro_rows.append({
                        "cube_cell_id": key,
                        "method": method,
                        "metric": metric,
                        "old_value": ov,
                        "new_value": nv,
                        "abs_difference": diff,
                        "tolerance": float(tolerance),
                    })
    primary_metrics_for_missingness = [
        "denoise_nmse",
        "denoise_corr",
        "imf_recovery_corr",
        "imf_recovery_nrmse",
        "component_splitting_index",
        "component_merging_index",
        "noise_capture_corr",
        "signal_leakage_into_noise",
        "outlier_resistance_index",
        "clean_region_nmse",
    ]
    endpoint_nan_rows = []
    unexplained_nan = 0
    invalid_reason_code = 0
    invalid_reason_condition = 0
    inf_count = 0
    allowed_nan_reasons = {"not_computable_zero_variance_protocol_noise"}
    noise_zero_energy_tolerance = 1e-10
    for row in new_rows:
        for method in ("IRMF", "EMD", "EEMD", "CEEMDAN"):
            data = row.get(method, {})
            for metric in ("noise_capture_corr", "signal_leakage_into_noise"):
                try:
                    v = float(data.get(metric))
                    finite = np.isfinite(v)
                    is_inf = bool(np.isinf(v))
                except Exception:
                    finite = False
                    is_inf = False
                inf_count += int(is_inf)
                if not finite:
                    reason = data.get("protocol_noise_reason_code")
                    reason_allowed = reason in allowed_nan_reasons
                    zero_flag = bool(data.get("protocol_zero_variance_noise_flag"))
                    noise_energy = data.get("protocol_estimated_noise_energy")
                    try:
                        noise_energy_ok = bool(float(noise_energy) <= noise_zero_energy_tolerance)
                    except Exception:
                        noise_energy_ok = False
                    selection_status = data.get("protocol_selection_status_code")
                    condition_valid = bool(
                        reason_allowed
                        and zero_flag
                        and noise_energy_ok
                    )
                    explained = bool(reason_allowed and condition_valid)
                    invalid_reason_code += int(not reason_allowed)
                    invalid_reason_condition += int(reason_allowed and not condition_valid)
                    unexplained_nan += int(not explained)
                    endpoint_nan_rows.append({
                        "cube_cell_id": row.get("cube_cell_id"),
                        "method": method,
                        "signal": row.get("signal"),
                        "noise": row.get("noise"),
                        "sigma": row.get("sigma"),
                        "data_seed": row.get("data_seed"),
                        "metric": metric,
                        "reason_code": reason,
                        "explained": explained,
                        "reason_code_allowed": reason_allowed,
                        "zero_variance_flag": zero_flag,
                        "protocol_estimated_noise_energy": noise_energy,
                        "noise_energy_tolerance": noise_zero_energy_tolerance,
                        "reason_condition_validated": condition_valid,
                        "protocol_selection_status_code": selection_status,
                    })
    missingness_rows, affected_nonfinite_rows = _method_metric_missingness_rows(
        new_rows,
        primary_metrics_for_missingness,
    )
    complete_block_rows = _metric_specific_complete_block_rows(
        new_rows,
        primary_metrics_for_missingness,
    )
    entirely_missing_method_regime_rows = _entirely_missing_method_regime_metric_rows(missingness_rows)
    incomplete_metric_count = sum(
        1 for row in complete_block_rows
        if row["eligible_case_seed_blocks"] > 0 and row["complete_pair_blocks"] == 0
    )
    full_expected = len(old_rows) == len(new_rows) and not missing and not extra
    partial_seed_smoke = bool(new_rows and not extra and not full_expected)
    seeds_completed = sorted({
        int(row.get("data_seed"))
        for row in new_rows
        if row.get("data_seed") is not None
    })
    missingness_requires_adjudication = bool(
        endpoint_nan_rows
        or any(row["incomplete_pair_blocks"] > 0 for row in complete_block_rows)
    )
    gate_conditions_passed = (
        full_expected
        and unexplained_nan == 0
        and invalid_reason_code == 0
        and invalid_reason_condition == 0
        and inf_count == 0
        and incomplete_metric_count == 0
    )
    partial_smoke_passed = bool(
        partial_seed_smoke
        and unexplained_nan == 0
        and invalid_reason_code == 0
        and invalid_reason_condition == 0
        and inf_count == 0
        and incomplete_metric_count == 0
    )
    passed = bool(gate_conditions_passed)
    audit_status = (
        "passed"
        if passed else
        "smoke_passed_not_statistics_ready"
        if partial_smoke_passed else
        "failed"
    )
    audit = {
        "audit_status": audit_status,
        "execution_scope": (
            "full_registered_cube" if full_expected else
            "partial_seed_smoke" if partial_seed_smoke else
            "incomplete_or_invalid_execution"
        ),
        "seeds_completed": seeds_completed,
        "full_seed_coverage_complete": bool(full_expected),
        "primary_statistical_input": str(stage_root),
        "legacy_original_benchmark_role": "reproducibility_reference_only",
        "old_rows": len(old_rows),
        "new_rows": len(new_rows),
        "missing_case_ids": len(missing),
        "extra_case_ids": len(extra),
        "expected_to_match_metric_exceedance_count": exceed,
        "expected_to_match_metric_exceedance_interpretation": (
            "recorded_for_reproducibility_audit_not_blocking_unless_review_decides"
        ),
        "nan_legality_gate": {
            "allowed_nan_reason_codes": sorted(allowed_nan_reasons),
            "noise_zero_energy_tolerance": noise_zero_energy_tolerance,
            "unexplained_nan_count": unexplained_nan,
            "invalid_reason_code_count": invalid_reason_code,
            "invalid_reason_condition_count": invalid_reason_condition,
            "inf_count": inf_count,
            "gate_status": (
                "passed"
                if unexplained_nan == 0 and invalid_reason_code == 0
                and invalid_reason_condition == 0 and inf_count == 0
                else "failed"
            ),
        },
        "missingness_pattern_gate": {
            "audited_grouping": "method × signal × noise × sigma × metric × reason_code",
            "affected_nonfinite_primary_metric_rows": len(affected_nonfinite_rows),
            "method_regime_metric_groups_entirely_missing": len(entirely_missing_method_regime_rows),
            "requires_adjudication": missingness_requires_adjudication,
            "gate_status": (
                "failed"
                if full_expected and entirely_missing_method_regime_rows else
                "requires_adjudication" if missingness_requires_adjudication else "passed"
            ),
        },
        "statistical_readiness_gate": {
            "paired_analysis_policy": "metric_specific_complete_case_blocks",
            "metrics_with_zero_complete_pair_blocks": incomplete_metric_count,
            "complete_pair_audit_path": str(stage_root / "metric_specific_complete_block_audit.csv"),
            "missingness_related_method_behavior_report_required": bool(endpoint_nan_rows),
            "sensitivity_analysis_required": bool(endpoint_nan_rows),
            "gate_status": "passed" if incomplete_metric_count == 0 else "failed",
        },
        "unexplained_noise_endpoint_nan_count": unexplained_nan,
        "explained_noise_endpoint_nan_count": len(endpoint_nan_rows) - unexplained_nan,
        "statistics_authorized": bool(passed),
        "timestamp": _utc_now(),
    }
    readiness = _statistical_readiness_summary(
        new_rows,
        endpoint_nan_rows,
        complete_block_rows,
        unexplained_nan=unexplained_nan,
        inf_count=inf_count,
        statistics_authorized=passed,
    )
    selection_behavior = _selection_behavior_diagnostics(new_rows)
    complete_block_composition = _complete_block_composition_diagnostics(
        new_rows,
        primary_metrics_for_missingness,
    )
    zero_variance_noise_diagnostic = _zero_variance_noise_diagnostic_rows(new_rows)
    audit["statistical_readiness_report"] = readiness["summary"]
    audit["selection_behavior_report"] = {
        "all_components_selected_rate_role": "secondary_missingness_diagnostic",
        "zero_variance_protocol_noise_rate_role": "secondary_missingness_diagnostic",
        "by_method_path": str(stage_root / "selection_behavior_by_method.csv"),
        "by_regime_path": str(stage_root / "selection_behavior_by_regime.csv"),
    }
    audit["complete_block_composition_report"] = {
        "role": "selection_bias_diagnostic_for_metric_specific_complete_blocks",
        "path": str(stage_root / "complete_block_composition_by_regime.csv"),
    }
    audit["zero_variance_noise_diagnostic_report"] = {
        "role": "secondary_energy_diagnostics_for_correlation_noncomputable_cases",
        "path": str(stage_root / "zero_variance_noise_diagnostic_summary.csv"),
        "rows": len(zero_variance_noise_diagnostic),
    }
    write_csv(repro_rows, stage_root / "original_vs_full_rerun_reproducibility_exceedances.csv")
    write_csv(endpoint_nan_rows, stage_root / "full_rerun_endpoint_nan_audit.csv")
    write_csv(missingness_rows, stage_root / "missingness_pattern_audit.csv")
    write_csv(affected_nonfinite_rows, stage_root / "affected_nonfinite_primary_metric_rows.csv")
    write_csv(complete_block_rows, stage_root / "metric_specific_complete_block_audit.csv")
    write_csv(
        entirely_missing_method_regime_rows,
        stage_root / "entirely_missing_method_regime_metric_groups.csv",
    )
    write_json(readiness["summary"], stage_root / "statistical_readiness_report.json")
    write_csv(readiness["summary_rows"], stage_root / "statistical_readiness_report.csv")
    write_csv(readiness["by_method"], stage_root / "statistical_readiness_by_method.csv")
    write_csv(readiness["by_reason_code"], stage_root / "statistical_readiness_by_reason_code.csv")
    write_csv(readiness["by_regime"], stage_root / "statistical_readiness_by_regime.csv")
    write_csv(
        readiness["complete_block_rate_by_metric"],
        stage_root / "statistical_readiness_complete_block_by_metric.csv",
    )
    write_csv(selection_behavior["by_method"], stage_root / "selection_behavior_by_method.csv")
    write_csv(selection_behavior["by_regime"], stage_root / "selection_behavior_by_regime.csv")
    write_csv(complete_block_composition, stage_root / "complete_block_composition_by_regime.csv")
    write_csv(zero_variance_noise_diagnostic, stage_root / "zero_variance_noise_diagnostic_summary.csv")
    write_json(audit, stage_root / "protocol_full_rerun_audit.json")
    if passed:
        _write_state(
            root,
            "AUDITED",
            stage_root=stage_root,
            statistics_authorized=True,
            complete_cube_root_for_statistics=str(stage_root),
            audit_status="passed",
            primary_statistical_input="protocol_controlled_full_benchmark_rerun",
        )
    write_json({"scope": "protocol_full_rerun_audit", "result": audit}, root / "protocol_full_rerun_audit_dashboard.json")
    write_paper_section_map(root)
    return {"protocol_full_rerun_audit": audit}


def run_noise_endpoint_merge_audit_stage(output_root):
    root = ensure_dir(output_root)
    result = run_noise_endpoint_merge_audit(root)
    write_json({
        "scope": "stage_C5_benchmark_immutability_and_noise_metric_merge_audit",
        "old_benchmark_metrics_recomputed": False,
        "old_benchmark_metrics_overwritten": False,
        "no_statistical_ranking": True,
        "no_method_performance_interpretation": True,
        "result": result["noise_endpoint_merge_audit"],
    }, root / "noise_endpoint_merge_audit_dashboard.json")
    write_paper_section_map(root)
    return result


def _unified_cube_root(root):
    return Path(root) / "03_unified_benchmark_cube"


def _has_unified_cube_rows(root):
    return (_unified_cube_root(root) / "unified_benchmark_cube_rows.csv").exists()


def run_unified_cube_statistics_stage(output_root, ensure_cube=False, **cube_kwargs):
    root = ensure_dir(output_root)
    require_benchmark_state(
        root,
        allowed={"AUDITED"},
        command_name="unified-statistics",
    )
    if ensure_cube and not _has_unified_cube_rows(root):
        run_unified_benchmark_cube_stage(root, **cube_kwargs)
    if not _has_unified_cube_rows(root):
        raise FileNotFoundError(
            "Unified cube rows are required for V5.20 statistics. Run "
            "`paper_pipeline.py unified-cube` first, or call with ensure_cube=True."
        )
    state_doc = load_benchmark_state(root)
    complete_cube_root = state_doc.get("complete_cube_root_for_statistics")
    stats_cube_root = Path(complete_cube_root) if complete_cube_root else _unified_cube_root(root)
    result = run_unified_cube_statistics(
        root / "07_unified_cube_statistics",
        cube_root=stats_cube_root,
    )
    mark_statistics_completed(root, statistics_dashboard=result)
    write_paper_section_map(root)
    return result


def run_v526_primary_schema_qualification_stage(output_root):
    root = ensure_dir(output_root)
    state_doc = load_benchmark_state(root)
    complete_cube_root = state_doc.get("complete_cube_root_for_statistics")
    if complete_cube_root:
        cube_root = Path(complete_cube_root)
    elif (root / "06_protocol_controlled_full_benchmark_rerun" / "unified_benchmark_cube_rows.csv").exists():
        cube_root = root / "06_protocol_controlled_full_benchmark_rerun"
    elif _has_unified_cube_rows(root):
        cube_root = _unified_cube_root(root)
    else:
        raise FileNotFoundError(
            "V5.26 primary-schema qualification requires an existing unified "
            "benchmark cube. Run the protocol-controlled full benchmark and "
            "audit first."
        )
    result = run_v526_primary_schema_qualification(
        root / "09_v526_primary_schema_qualification",
        cube_root=cube_root,
    )
    write_paper_section_map(root)
    return result


def run_v527_primary_schema_qualification_stage(output_root):
    root = ensure_dir(output_root)
    state_doc = load_benchmark_state(root)
    complete_cube_root = state_doc.get("complete_cube_root_for_statistics")
    if complete_cube_root:
        cube_root = Path(complete_cube_root)
    elif (root / "06_protocol_controlled_full_benchmark_rerun" / "unified_benchmark_cube_rows.csv").exists():
        cube_root = root / "06_protocol_controlled_full_benchmark_rerun"
    elif _has_unified_cube_rows(root):
        cube_root = _unified_cube_root(root)
    else:
        raise FileNotFoundError(
            "V5.27 primary-schema qualification requires an existing unified "
            "benchmark cube. Run the protocol-controlled full benchmark and "
            "audit first."
        )
    result = run_v527_primary_schema_qualification(
        root / "10_v527_primary_schema_qualification",
        cube_root=cube_root,
    )
    write_paper_section_map(root)
    return result


def run_v527_primary_schema_freeze_stage(output_root):
    root = ensure_dir(output_root)
    result = freeze_v527_primary_schema(
        root / "10_v527_primary_schema_qualification",
    )
    write_paper_section_map(root)
    return result


def run_v527_unified_statistics_stage(output_root):
    root = ensure_dir(output_root)
    state_doc = load_benchmark_state(root)
    complete_cube_root = state_doc.get("complete_cube_root_for_statistics")
    if complete_cube_root:
        cube_root = Path(complete_cube_root)
    elif (root / "06_protocol_controlled_full_benchmark_rerun" / "unified_benchmark_cube_rows.csv").exists():
        cube_root = root / "06_protocol_controlled_full_benchmark_rerun"
    elif _has_unified_cube_rows(root):
        cube_root = _unified_cube_root(root)
    else:
        raise FileNotFoundError(
            "V5.27 statistics requires an existing unified benchmark cube. "
            "Run the protocol-controlled full benchmark and audit first."
        )
    result = run_v527_unified_statistics(
        root / "11_v527_unified_statistics",
        cube_root=cube_root,
        schema_root=root / "10_v527_primary_schema_qualification",
    )
    write_paper_section_map(root)
    return result


def run_v527_formula_validity_audit_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v527_formula_validity_audit(
        root / "10b_v527_formula_validity_audit",
    )
    write_paper_section_map(root)
    return result


def run_v527_directionality_audit_stage(output_root):
    root = ensure_dir(output_root)
    stats_root = root / "11_v527_unified_statistics"
    result = run_v527_directionality_audit(
        root / "11b_v527_directionality_audit",
        stats_root=stats_root,
    )
    write_paper_section_map(root)
    return result


def run_v527_spillover_scale_audit_stage(output_root):
    root = ensure_dir(output_root)
    state_doc = load_benchmark_state(root)
    complete_cube_root = state_doc.get("complete_cube_root_for_statistics")
    if complete_cube_root:
        cube_root = Path(complete_cube_root)
    elif (root / "06_protocol_controlled_full_benchmark_rerun" / "unified_benchmark_cube_rows.csv").exists():
        cube_root = root / "06_protocol_controlled_full_benchmark_rerun"
    elif _has_unified_cube_rows(root):
        cube_root = _unified_cube_root(root)
    else:
        raise FileNotFoundError(
            "V5.27 spillover scale audit requires an existing unified benchmark cube."
        )
    result = run_v527_spillover_scale_audit(
        root / "11c_v527_spillover_scale_audit",
        cube_root=cube_root,
    )
    write_paper_section_map(root)
    return result


def run_v528_primary_schema_freeze_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v528_primary_schema_freeze(
        root / "12_v528_primary_schema",
        v527_scale_audit_root=root / "11c_v527_spillover_scale_audit",
    )
    write_paper_section_map(root)
    return result


def run_v528_unified_statistics_stage(output_root):
    root = ensure_dir(output_root)
    state_doc = load_benchmark_state(root)
    complete_cube_root = state_doc.get("complete_cube_root_for_statistics")
    if complete_cube_root:
        cube_root = Path(complete_cube_root)
    elif (root / "06_protocol_controlled_full_benchmark_rerun" / "unified_benchmark_cube_rows.csv").exists():
        cube_root = root / "06_protocol_controlled_full_benchmark_rerun"
    elif _has_unified_cube_rows(root):
        cube_root = _unified_cube_root(root)
    else:
        raise FileNotFoundError(
            "V5.28 statistics requires an existing unified benchmark cube."
        )
    result = run_v528_unified_statistics(
        root / "13_v528_unified_statistics",
        cube_root=cube_root,
        schema_root=root / "12_v528_primary_schema",
    )
    write_paper_section_map(root)
    return result


def run_v528_final_statistics_validation_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v528_final_statistics_validation(
        root / "14_v528_final_statistics_validation",
        schema_root=root / "12_v528_primary_schema",
        stats_root=root / "13_v528_unified_statistics",
        scale_audit_root=root / "11c_v527_spillover_scale_audit",
    )
    write_paper_section_map(root)
    return result


def run_v529_snr_design_draft_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v529_snr_design_draft(
        root / "16_v529_snr_design",
    )
    write_paper_section_map(root)
    return result


def run_v529_snr_realization_audit_stage(output_root):
    root = ensure_dir(output_root)
    if not (root / "16_v529_snr_design" / "v530_snr_design_draft.json").exists():
        run_v529_snr_design_draft(root / "16_v529_snr_design")
    result = run_v529_snr_realization_audit(
        root / "16_v529_snr_design",
    )
    write_paper_section_map(root)
    return result


def run_v529_snr_design_freeze_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v529_snr_design_freeze(
        root / "16_v529_snr_design",
    )
    write_paper_section_map(root)
    return result


def run_v530_snr_design_draft_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v529_snr_design_draft(
        root / "16_v530_snr_design",
    )
    write_paper_section_map(root)
    return result


def run_v530_snr_realization_audit_stage(output_root):
    root = ensure_dir(output_root)
    if not (root / "16_v530_snr_design" / "v530_snr_design_draft.json").exists():
        run_v529_snr_design_draft(root / "16_v530_snr_design")
    result = run_v529_snr_realization_audit(
        root / "16_v530_snr_design",
    )
    write_paper_section_map(root)
    return result


def run_v530_snr_design_freeze_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v529_snr_design_freeze(
        root / "16_v530_snr_design",
    )
    write_paper_section_map(root)
    return result


def run_v530_clean_baseline_schema_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v530_clean_baseline_schema(
        root / "16b_v530_clean_baseline",
    )
    write_paper_section_map(root)
    return result


def run_v531_time_axis_semantics_audit_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v531_time_axis_semantics_audit(
        root / "16c_v531_time_axis_semantics",
    )
    write_paper_section_map(root)
    return result


def run_v532_section6_scope_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v532_section6_scope(
        root / "16d_v532_section6_scope",
    )
    write_paper_section_map(root)
    return result


def run_v533_irmf_parameter_sensitivity_protocol_stage(output_root):
    root = ensure_dir(output_root)
    locked = load_or_prepare_locked_parameters(root, calibrate=False)
    result = run_v533_irmf_parameter_sensitivity_protocol(
        root / "16e_v533_irmf_parameter_sensitivity",
        irmf_params=locked["IRMF"],
    )
    write_paper_section_map(root)
    return result


def run_v534_section6_executable_protocols_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v534_section6_executable_protocols(
        root / "16f_v534_section6_executable_protocols",
    )
    write_paper_section_map(root)
    return result


def run_v535_section6_1_irmf_parameter_smoke_stage(output_root):
    root = ensure_dir(output_root)
    locked = load_or_prepare_locked_parameters(root, calibrate=False)
    result = run_v535_section6_1_irmf_parameter_smoke(
        root / "16g_v535_section6_1_irmf_parameter_smoke",
        irmf_params=locked["IRMF"],
    )
    write_paper_section_map(root)
    return result


def run_v536_section6_2_contamination_design_smoke_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v536_section6_2_contamination_design_smoke(
        root / "16h_v536_section6_2_contamination_design_smoke",
    )
    write_paper_section_map(root)
    return result


def run_v537_section6_3_comparator_signal_smoke_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v537_section6_3_comparator_signal_smoke(
        root / "16i_v537_section6_3_comparator_signal_smoke",
    )
    write_paper_section_map(root)
    return result


def run_v538_section6_4_runtime_instrumentation_smoke_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v538_section6_4_runtime_instrumentation_smoke(
        root / "16j_v538_section6_4_runtime_instrumentation_smoke",
    )
    write_paper_section_map(root)
    return result


def run_v539_irmf_h_tuning_amendment_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v539_irmf_h_tuning_amendment(
        root / "16k_v539_irmf_h_tuning_amendment",
    )
    write_paper_section_map(root)
    return result


def run_v540_time_frequency_secondary_schema_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v540_time_frequency_secondary_schema(
        root / "16l_v540_time_frequency_secondary_schema",
    )
    write_paper_section_map(root)
    return result


def run_v541_primary_secondary_closure_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v541_primary_secondary_closure(
        root / "16m_v541_primary_secondary_closure",
    )
    write_paper_section_map(root)
    return result


def run_v542_irmf_relative_h_parameterization_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v542_irmf_relative_h_parameterization(
        root / "16n_v542_irmf_relative_h_parameterization",
    )
    write_paper_section_map(root)
    return result


def run_v543_primary_metric_manual_code_literature_audit_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v543_primary_metric_manual_code_literature_audit(
        root / "16o_v543_primary_metric_manual_code_literature_audit",
    )
    write_paper_section_map(root)
    return result


def run_v544_h_decision_gate_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v544_h_decision_gate(
        root / "16p_v544_h_decision_gate",
    )
    write_paper_section_map(root)
    return result


def run_v545_relative_h_scale_estimator_qualification_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v545_relative_h_scale_estimator_qualification(
        root / "16q_v545_relative_h_scale_estimator_qualification",
    )
    write_paper_section_map(root)
    return result


def run_v548_parameter_selection_adjudication_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v548_parameter_selection_adjudication(
        root / "16r_v548_parameter_selection_adjudication",
        algorithm_root=root,
    )
    write_paper_section_map(root)
    return result


def run_v550_section6_main_text_assembly_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v550_section6_main_text_assembly(
        root / "16s_v550_section6_main_text_assembly",
    )
    write_paper_section_map(root)
    return result


def run_v551_section6_2_contamination_design_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v551_section6_2_contamination_design_sensitivity(
        root / "16t_v551_section6_2_contamination_design",
    )
    write_paper_section_map(root)
    return result


def run_v552_section6_4_computational_scaling_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v552_section6_4_computational_scaling(
        root / "16u_v552_section6_4_computational_scaling",
    )
    write_paper_section_map(root)
    return result


def run_v553_section6_claim_qualification_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v553_section6_claim_qualification(
        root / "16v_v553_section6_claim_qualification",
    )
    write_paper_section_map(root)
    return result


def run_v558_section6_pre_execution_audit_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v558_section6_pre_execution_audit(
        root / "16aa_v558_section6_pre_execution_audit",
    )
    write_paper_section_map(root)
    return result


def run_v559_section6_post_execution_qualification_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v559_section6_post_execution_qualification(
        root / "16ab_v559_section6_post_execution_qualification",
    )
    write_paper_section_map(root)
    return result


def run_v560_matching_sensitivity_protocol_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v560_matching_sensitivity_protocol(
        root / "16ac_v560_matching_sensitivity_protocol",
        algorithm_root=root,
    )
    write_json({
        "scope": "supplementary_primary_component_correspondence_sensitivity_protocol",
        "no_algorithm_rerun": True,
        "no_primary_endpoint_change": True,
        "no_primary_matching_change": True,
        "no_ranking_bearing_claim": True,
        "result": result["dashboard"],
    }, root / "v560_matching_sensitivity_protocol_dashboard.json")
    write_paper_section_map(root)
    return result


def run_v560_matching_sensitivity_subset_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v560_matching_sensitivity_subset(
        root / "16ad_v560_matching_sensitivity_subset",
        algorithm_root=root,
    )
    write_json({
        "scope": "supplementary_primary_component_correspondence_sensitivity_small_subset",
        "no_primary_endpoint_change": True,
        "no_primary_matching_change": True,
        "no_ranking_bearing_claim": True,
        "result": result["dashboard"],
    }, root / "v560_matching_sensitivity_subset_dashboard.json")
    write_paper_section_map(root)
    return result


def run_v561_section6_5_matching_rule_robustness_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v561_section6_5_matching_rule_robustness(
        root / "16ae_v561_section6_5_matching_rule_robustness",
        algorithm_root=root,
    )
    write_json({
        "scope": "section6_5_matching_rule_robustness_curve",
        "no_primary_endpoint_change": True,
        "no_primary_matching_change": True,
        "not_retuning": True,
        "not_ranking_bearing_claim": True,
        "result": result["dashboard"],
    }, root / "v561_section6_5_matching_rule_robustness_dashboard.json")
    write_paper_section_map(root)
    return result


def run_v562_section6_reviewer_readiness_synthesis_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v562_section6_reviewer_readiness_synthesis(
        root / "16af_v562_section6_reviewer_readiness_synthesis",
    )
    write_json({
        "scope": "section6_reviewer_readiness_synthesis",
        "no_algorithm_runs_performed": True,
        "no_primary_schema_change": True,
        "not_retuning": True,
        "not_ranking_bearing_claim": True,
        "result": result["dashboard"],
    }, root / "v562_section6_reviewer_readiness_synthesis_dashboard.json")
    write_paper_section_map(root)
    return result


def run_v563_section6_failure_region_dominance_synthesis_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v563_section6_failure_region_dominance_synthesis(
        root / "16ag_v563_section6_failure_region_dominance_synthesis",
    )
    write_json({
        "scope": "section6_failure_region_and_dominance_synthesis",
        "no_algorithm_runs_performed": True,
        "no_primary_schema_change": True,
        "not_retuning": True,
        "not_ranking_bearing_claim": True,
        "result": result["dashboard"],
    }, root / "v563_section6_failure_region_dominance_dashboard.json")
    write_paper_section_map(root)
    return result


def run_v564_section6_3a_challenging_family_variant_extension_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v564_section6_3a_challenging_family_variant_extension(
        root / "16ah_v564_section6_3a_challenging_family_variant_extension",
    )
    write_json({
        "scope": "section6_3a_challenging_family_variant_extension",
        "algorithm_runs_performed": False,
        "no_primary_schema_change": True,
        "not_retuning": True,
        "not_ranking_bearing_claim": True,
        "result": result["dashboard"],
    }, root / "v564_section6_3a_challenging_family_variant_extension_dashboard.json")
    write_paper_section_map(root)
    return result


def run_v564a_section6_3a_canonical_anchor_reference_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v564a_section6_3a_canonical_anchor_reference(
        root / "16ai_v564a_section6_3a_canonical_anchor_reference",
    )
    write_json({
        "scope": "section6_3a_canonical_anchor_reference",
        "algorithm_runs_performed": False,
        "no_primary_schema_change": True,
        "not_retuning": True,
        "not_ranking_bearing_claim": True,
        "result": result["dashboard"],
    }, root / "v564a_section6_3a_canonical_anchor_reference_dashboard.json")
    write_paper_section_map(root)
    return result


def run_v565_section6_3a_canonical_structured_perturbation_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v565_section6_3a_canonical_structured_perturbation(
        root / "16aj_v565_section6_3a_canonical_structured_perturbation",
    )
    write_json({
        "scope": "section6_3a_canonical_structured_perturbation",
        "algorithm_runs_performed": False,
        "no_primary_schema_change": True,
        "not_retuning": True,
        "not_ranking_bearing_claim": True,
        "result": result["dashboard"],
    }, root / "v565_section6_3a_canonical_structured_perturbation_dashboard.json")
    write_paper_section_map(root)
    return result


def run_v566_section6_3a_two_axis_interaction_design_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v566_section6_3a_two_axis_interaction_design(
        root / "16ak_v566_section6_3a_two_axis_interaction_design",
    )
    write_json({
        "scope": "section6_3a_two_axis_interaction_design",
        "algorithm_runs_performed": False,
        "no_primary_schema_change": True,
        "not_retuning": True,
        "not_ranking_bearing_claim": True,
        "result": result["dashboard"],
    }, root / "v566_section6_3a_two_axis_interaction_design_dashboard.json")
    write_paper_section_map(root)
    return result


def run_v567_section6_3a_analysis_qualification_freeze_stage(output_root):
    root = ensure_dir(output_root)
    result = run_v567_section6_3a_analysis_qualification_freeze(
        root / "16al_v567_section6_3a_analysis_qualification_freeze",
    )
    write_json({
        "scope": "section6_3a_analysis_qualification_freeze",
        "algorithm_runs_performed": False,
        "no_signal_registry_change": True,
        "no_primary_schema_change": True,
        "not_retuning": True,
        "not_ranking_bearing_claim": True,
        "result": result["dashboard"],
    }, root / "v567_section6_3a_analysis_qualification_freeze_dashboard.json")
    write_paper_section_map(root)
    return result


def run_real_world_validation_schema_draft_stage(output_root):
    root = ensure_dir(output_root)
    result = run_real_world_validation_schema_draft(
        root / "15_real_world_validation_schema",
    )
    write_paper_section_map(root)
    return result


def run_real_world_endpoint_formula_audit_stage(output_root):
    root = ensure_dir(output_root)
    result = run_real_world_endpoint_formula_audit(
        root / "15b_real_world_endpoint_formula_audit",
        schema_root=root / "15_real_world_validation_schema",
    )
    write_paper_section_map(root)
    return result


def run_real_world_perturbation_protocol_freeze_stage(output_root):
    root = ensure_dir(output_root)
    result = run_real_world_perturbation_protocol_freeze(
        root / "15c_real_world_perturbation_protocol",
        schema_root=root / "15_real_world_validation_schema",
    )
    write_paper_section_map(root)
    return result


def run_real_world_denominator_degeneracy_audit_stage(output_root):
    root = ensure_dir(output_root)
    result = run_real_world_denominator_degeneracy_audit(
        root / "15d_real_world_denominator_degeneracy_audit",
        schema_root=root / "15_real_world_validation_schema",
    )
    write_paper_section_map(root)
    return result


def run_real_world_method_neutrality_audit_stage(output_root):
    root = ensure_dir(output_root)
    result = run_real_world_method_neutrality_audit(
        root / "15e_real_world_method_neutrality_audit",
        schema_root=root / "15_real_world_validation_schema",
    )
    write_paper_section_map(root)
    return result


def run_real_world_validation_preflight_stage(output_root):
    root = ensure_dir(output_root)
    result = run_real_world_validation_preflight(
        root / "15f_real_world_validation_preflight",
        schema_root=root / "15_real_world_validation_schema",
        formula_root=root / "15b_real_world_endpoint_formula_audit",
        perturbation_root=root / "15c_real_world_perturbation_protocol",
        denominator_root=root / "15d_real_world_denominator_degeneracy_audit",
        neutrality_root=root / "15e_real_world_method_neutrality_audit",
    )
    write_paper_section_map(root)
    return result


def run_real_world_v1_schema_smoke_stage(
        output_root,
        data_root=None,
        calibrate=True,
        use_current_config_params=False,
        timeout_seconds=120,
):
    root = ensure_dir(output_root)
    locked = (
        current_config_locked_parameters()
        if use_current_config_params
        else load_or_prepare_locked_parameters(root, calibrate=calibrate)
    )
    locked = _write_locked_parameter_audit(root, locked)
    data_root = (
        Path(data_root)
        if data_root is not None
        else Path(__file__).resolve().parents[1] / "data" / "mit-bih-arrhythmia-database"
    )
    result = run_real_world_v1_schema_single_window_smoke(
        root / "15g_real_world_v1_schema_single_window_smoke",
        data_root=data_root,
        params=locked,
        record_id="100",
        window_index=0,
        timeout_seconds=timeout_seconds,
    )
    write_json({
        "scope": "single_development_window_real_world_v1_schema_smoke_only",
        "no_threshold_locking": True,
        "no_held_out_access": True,
        "no_performance_claims_authorized": True,
        "data_root": str(data_root),
        "result": result,
    }, root / "real_world_v1_schema_smoke_dashboard.json")
    write_paper_section_map(root)
    return result


def run_proxy_metric_validation_stage(output_root):
    root = ensure_dir(output_root)
    if not _has_unified_cube_rows(root):
        raise FileNotFoundError(
            "Unified cube rows are required for proxy validation. Run "
            "`paper_pipeline.py unified-cube` first."
        )
    result = run_proxy_metric_validation(
        root / "07d_proxy_metric_validation",
        cube_root=_unified_cube_root(root),
    )
    write_paper_section_map(root)
    return result


def run_structural_metric_validation_stage(
        output_root,
        calibrate=True,
        quick=False,
        timeout_seconds=None,
        use_current_config_params=False,
):
    """Run a V5.24 metric-behavior validation slice before full reruns."""
    root = ensure_dir(output_root)
    locked = (
        current_config_locked_parameters()
        if use_current_config_params
        else load_or_prepare_locked_parameters(root, calibrate=calibrate)
    )
    locked = _write_locked_parameter_audit(root, locked)
    result = run_structural_metric_validation_slice(
        root / "07e_structural_metric_validation",
        irmf_params=locked["IRMF"],
        emd_params=locked["EMD"],
        eemd_params=locked["EEMD"],
        ceemdan_params=locked["CEEMDAN"],
        timeout_seconds=timeout_seconds or 120,
        quick=quick,
    )
    write_json({
        "structural_metric_validation": result,
        "rho_lock": locked["rho_lock"],
        "parameter_source": locked.get("parameter_source", "locked_algorithm_parameters"),
        "interpretation": (
            "This is a V5.24 metric-behavior audit slice. It validates range, "
            "gating, IMF-count dependence, and association-threshold sensitivity "
            "before authorizing full structural primary conclusions."
        ),
    }, root / "structural_metric_validation_dashboard.json")
    write_paper_section_map(root)
    return result


def run_real_world_validation_stage(
        output_root,
        data_dir=None,
        calibrate=True,
        use_current_config_params=False,
):
    root = ensure_dir(output_root)
    locked = (
        current_config_locked_parameters()
        if use_current_config_params
        else load_or_prepare_locked_parameters(root, calibrate=calibrate)
    )
    locked = _write_locked_parameter_audit(root, locked)
    real_data_protocol = run_real_data_protocol_scaffold(
        root / "15a_real_data_protocol",
        irmf_params=locked["IRMF"],
        emd_params=locked["EMD"],
        eemd_params=locked["EEMD"],
        ceemdan_params=locked["CEEMDAN"],
    )
    semisynthetic = run_real_data_semisynthetic_contamination(
        root / "14_semisynthetic_bridge_validation",
        data_dir=data_dir,
        irmf_params=locked["IRMF"],
        emd_params=locked["EMD"],
        eemd_params=locked["EEMD"],
        ceemdan_params=locked["CEEMDAN"],
    )
    result = run_real_world_validation(
        root / "15_real_world_validation",
        data_dir=data_dir,
        irmf_params=locked["IRMF"],
        emd_params=locked["EMD"],
        eemd_params=locked["EEMD"],
        ceemdan_params=locked["CEEMDAN"],
    )
    dashboard = {
        "real_data_protocol": real_data_protocol,
        "semisynthetic_bridge_validation": semisynthetic,
        "pure_real_world_validation": result,
        "rho_lock": locked["rho_lock"],
        "parameter_source": locked.get("parameter_source", "locked_algorithm_parameters"),
        "interpretation": (
            "Sections 7-8 use a truth-aware locked protocol. Section 7 "
            "semi-synthetic real signals may support approximate-reference "
            "recovery and approximate-reference oracle upper-bound claims. "
            "Section 8 pure real-world signals support application-relevant "
            "structure preservation, stability, downstream utility, and "
            "event-locked residual leakage diagnostics only; ground-truth-"
            "dependent clean-signal or true-IMF claims are not made on pure "
            "real data."
        ),
    }
    write_json(dashboard, root / "real_world_validation_dashboard.json")
    write_paper_section_map(root)
    return dashboard


def run_real_data_protocol_stage(
        output_root,
        calibrate=True,
        use_current_config_params=False,
):
    root = ensure_dir(output_root)
    locked = (
        current_config_locked_parameters()
        if use_current_config_params
        else load_or_prepare_locked_parameters(root, calibrate=calibrate)
    )
    locked = _write_locked_parameter_audit(root, locked)
    result = run_real_data_protocol_scaffold(
        root / "15a_real_data_protocol",
        irmf_params=locked["IRMF"],
        emd_params=locked["EMD"],
        eemd_params=locked["EEMD"],
        ceemdan_params=locked["CEEMDAN"],
    )
    write_json({
        "real_data_protocol": result,
        "rho_lock": locked["rho_lock"],
        "parameter_source": locked.get("parameter_source", "locked_algorithm_parameters"),
        "paper_note": (
            "This stage freezes the Sections 7-8 real-data protocol scaffold "
            "and split registry before any held-out real-data evaluation. It does not "
            "claim MIT-BIH or CWRU performance results."
        ),
    }, root / "real_data_protocol_dashboard.json")
    write_paper_section_map(root)
    return result


def run_real_data_split_audit_stage(
        output_root,
        protocol_root,
        subject_map_path=None,
):
    root = ensure_dir(output_root)
    audit_root = root / "15b_real_data_split_audit"
    result = run_real_data_split_audit(
        audit_root,
        protocol_root=protocol_root,
        subject_map_path=subject_map_path,
    )
    write_json({
        "scope": "section_8_split_registry_audit_only",
        "no_waveform_access": True,
        "no_algorithm_runs": True,
        "no_performance_metrics": True,
        "audit_result": result,
    }, root / "real_data_split_audit_dashboard.json")
    write_paper_section_map(root)
    return result


def run_real_data_development_run_stage(
        output_root,
        protocol_root,
        data_root,
        calibrate=True,
        use_current_config_params=False,
        max_windows_per_record=None,
):
    root = ensure_dir(output_root)
    locked = (
        current_config_locked_parameters()
        if use_current_config_params
        else load_or_prepare_locked_parameters(root, calibrate=calibrate)
    )
    locked = _write_locked_parameter_audit(root, locked)
    result = run_real_data_development_run(
        root / "15c_real_data_development_run",
        protocol_root=protocol_root,
        data_root=data_root,
        params={
            "IRMF": locked["IRMF"],
            "EMD": locked["EMD"],
            "EEMD": locked["EEMD"],
            "CEEMDAN": locked["CEEMDAN"],
        },
        timeout_seconds=120,
        max_windows_per_record=max_windows_per_record,
    )
    write_json({
        "scope": "stage_3b_development_record_execution_audit_only",
        "no_held_out_waveform_access": True,
        "no_threshold_locking": True,
        "no_performance_ranking": True,
        "development_run_status": result,
    }, root / "real_data_development_run_dashboard.json")
    write_paper_section_map(root)
    return result


def run_real_data_ecg_operational_smoke_stage(
        output_root,
        protocol_root,
        data_root,
        calibrate=True,
        use_current_config_params=False,
        max_windows_per_record=1,
):
    root = ensure_dir(output_root)
    locked = (
        current_config_locked_parameters()
        if use_current_config_params
        else load_or_prepare_locked_parameters(root, calibrate=calibrate)
    )
    locked = _write_locked_parameter_audit(root, locked)
    result = run_real_data_ecg_operational_smoke(
        root / "15d_real_data_ecg_operational_smoke",
        protocol_root=protocol_root,
        data_root=data_root,
        params={
            "IRMF": locked["IRMF"],
            "EMD": locked["EMD"],
            "EEMD": locked["EEMD"],
            "CEEMDAN": locked["CEEMDAN"],
        },
        timeout_seconds=120,
        max_windows_per_record=max_windows_per_record,
    )
    write_json({
        "scope": "non_locking_real_data_ecg_operational_smoke_only",
        "development_records_only": True,
        "no_held_out_waveform_access": True,
        "thresholds_locked": False,
        "no_performance_claims_authorized": True,
        "ecg_operational_smoke_status": result,
    }, root / "real_data_ecg_operational_smoke_dashboard.json")
    write_paper_section_map(root)
    return result


def _load_benchmark_rows(root):
    root = Path(root)
    family_candidates = [
        root / "03_canonical_emd_family_benchmark" / "section_5_canonical_emd_family_benchmark.json",
        root / "04_full_emd_family_benchmark" / "section_7b_emd_family_comparison.json",
        root / "04_full_emd_family_benchmark" / "emd_family_benchmark_rows.json",
        root / "04_full_emd_family_benchmark" / "section_5b_emd_family_benchmark.json",
        root / "04_full_emd_family_benchmark" / "fixed_emd_family_benchmark_rows.json",
    ]
    family_rows = None
    for path in family_candidates:
        family_rows = _load_json(path)
        if family_rows is not None:
            break
    return None, family_rows


def run_statistics_stage(output_root, calibrate=True, ensure_benchmark=True, use_current_config_params=False):
    root = ensure_dir(output_root)
    if _has_unified_cube_rows(root):
        result = run_unified_cube_statistics_stage(root)
        proxy = run_proxy_metric_validation_stage(root)
        write_json({
            "statistics_source": "unified_benchmark_cube",
            "unified_cube_statistics": result,
            "proxy_metric_validation": proxy,
            "paper_note": (
                "V5.20 statistics are generated from the unified benchmark cube "
                "rows as the benchmark single source of truth. Legacy family_rows "
                "statistics are bypassed."
            ),
        }, root / "statistics_dashboard.json")
        write_paper_section_map(root)
        return result

    locked = (
        current_config_locked_parameters()
        if use_current_config_params
        else load_or_prepare_locked_parameters(root, calibrate=calibrate)
    )
    if use_current_config_params:
        locked["parameter_source"] = "project_config_fixed_parameters"
        locked["parameter_lock_audit"] = _parameter_lock_audit("project_config_fixed_parameters")
        write_json(locked, root / "locked_algorithm_parameters.json")
    _, family_rows = _load_benchmark_rows(root)
    if ensure_benchmark and family_rows is None:
        if use_current_config_params:
            family_rows, family_summary = run_fixed_emd_family_benchmark(
                root / "03_canonical_emd_family_benchmark",
                irmf_params=locked["IRMF"],
                emd_params=locked["EMD"],
                eemd_params=locked["EEMD"],
                ceemdan_params=locked["CEEMDAN"],
                mode="full",
                return_rows=True,
            )
            write_json({
                "section_5_primary_benchmark": "algorithm/03_canonical_emd_family_benchmark",
                "family_summary": family_summary,
                "rho_lock": locked["rho_lock"],
                "parameter_source": locked["parameter_source"],
                "paper_note": (
                    "Section 5 is the four-method EMD-family benchmark. Pairwise "
                    "comparisons are derived from the same IRMF/EMD/EEMD/CEEMDAN rows."
                ),
            }, root / "benchmark_dashboard.json")
        else:
            result = run_benchmark_stage(root, calibrate=calibrate, full_family=True)
            family_rows = result["family_rows"]
    taxonomy = write_metric_taxonomy(root / "05b_metric_taxonomy")
    repeated = None
    primary_framework = None
    weighting = None
    if family_rows is not None:
        repeated = run_repeated_measures_statistics(root / "07_repeated_measures_statistics", family_rows=family_rows)
        primary_framework = run_primary_evaluation_framework(
            root / "07c_primary_evaluation_framework",
            family_rows=family_rows,
        )
        weighting = run_case_score_weighting_sensitivity(
            root / "07b_case_score_weighting_sensitivity",
            family_rows=family_rows,
        )
    result = {
        "metric_taxonomy": taxonomy,
        "statistical_inference": repeated,
        "mechanism_analysis": primary_framework,
        "repeated_measures": repeated,
        "primary_evaluation_framework": primary_framework,
        "case_score_weighting_sensitivity": weighting,
        "rho_lock": locked["rho_lock"],
        "paper_note": (
            "Four-method benchmark rows are the only benchmark basis. Legacy "
            "two-method IRMF-vs-EMD inference is not run by the default pipeline."
        ),
    }
    write_json(result, root / "statistics_dashboard.json")
    write_paper_section_map(root)
    return result


def run_sensitivity_stage(output_root, calibrate=True, emd_family_sensitivity=True):
    root = ensure_dir(output_root)
    locked = load_or_prepare_locked_parameters(root, calibrate=calibrate)
    irmf, emd = locked["IRMF"], locked["EMD"]
    robustness = run_all_robustness_sensitivity(
        root / "08_robustness_sensitivity_target_snr",
        irmf_params=irmf,
        emd_params=emd,
        eemd_params=locked["EEMD"],
        ceemdan_params=locked["CEEMDAN"],
    )
    variant_rows, variants = run_signal_variant_robustness(
        root / "09_signal_variant_robustness_target_snr",
        irmf_params=irmf,
        emd_params=emd,
        eemd_params=locked["EEMD"],
        ceemdan_params=locked["CEEMDAN"],
    )
    _, family_rows = _load_benchmark_rows(root)
    posthoc = None
    if family_rows is not None:
        posthoc = run_v5_2_posthoc_result_checks(
            root / "10_result_sensitivity", family_rows=family_rows, variant_rows=variant_rows,
        )
    family_sensitivity = None
    if emd_family_sensitivity:
        family_sensitivity = run_emd_family_baseline_sensitivity(
            root / "11_emd_family_sensitivity_target_snr", irmf_params=irmf, emd_params=emd,
            mode="representative",
        )
    result = {
        "robustness": robustness,
        "signal_variants": variants,
        "result_sensitivity": posthoc,
        "emd_family_sensitivity": family_sensitivity,
        "rho_lock": locked["rho_lock"],
    }
    write_json(result, root / "sensitivity_dashboard.json")
    write_paper_section_map(root)
    return result


def run_challenging_signal_stage(output_root, calibrate=True, quick=False, use_current_config_params=False):
    root = ensure_dir(output_root)
    locked = (
        current_config_locked_parameters()
        if use_current_config_params
        else load_or_prepare_locked_parameters(root, calibrate=calibrate)
    )
    rows, summary = run_challenging_signal_suite(
        root / "12_challenging_signal_suite",
        irmf_params=locked["IRMF"],
        emd_params=locked["EMD"],
        eemd_params=locked["EEMD"],
        ceemdan_params=locked["CEEMDAN"],
        quick=quick,
        return_rows=True,
    )
    result = {
        "rows": rows,
        "summary": summary,
        "rho_lock": locked["rho_lock"],
        "quick": bool(quick),
        "parameter_source": locked.get("parameter_source", "locked_algorithm_parameters"),
    }
    write_json(result, root / "challenging_signal_dashboard.json")
    write_paper_section_map(root)
    return result


def run_controlled_challenging_diagnostics_stage(
        output_root,
        calibrate=True,
        quick=False,
        use_current_config_params=False,
):
    root = ensure_dir(output_root)
    locked = (
        current_config_locked_parameters()
        if use_current_config_params
        else load_or_prepare_locked_parameters(root, calibrate=calibrate)
    )
    result = run_controlled_challenging_diagnostics(
        root / "13_controlled_challenging_diagnostics",
        irmf_params=locked["IRMF"],
        emd_params=locked["EMD"],
        eemd_params=locked["EEMD"],
        ceemdan_params=locked["CEEMDAN"],
        quick=quick,
    )
    dashboard = {
        "summary": result["summary"],
        "failure_thresholds": result["failure_thresholds"],
        "rho_lock": locked["rho_lock"],
        "quick": bool(quick),
        "parameter_source": locked.get("parameter_source", "locked_algorithm_parameters"),
        "metric_policy": result["protocol"].get("regime_specific_metric_policy"),
    }
    write_json(dashboard, root / "controlled_challenging_diagnostics_dashboard.json")
    write_paper_section_map(root)
    return dashboard


def run_oracle_adaptivity_stage(
        output_root,
        calibrate=True,
        quick=False,
        use_current_config_params=False,
        run_exploratory_ceemdan_oracle=False,
):
    root = ensure_dir(output_root)
    locked = (
        current_config_locked_parameters()
        if use_current_config_params
        else load_or_prepare_locked_parameters(root, calibrate=calibrate)
    )
    locked = _write_locked_parameter_audit(root, locked)
    result = run_irmf_oracle_adaptivity_analysis(
        root / "14_oracle_adaptivity_analysis",
        irmf_params=locked["IRMF"],
        quick=quick,
        run_exploratory_ceemdan_oracle=run_exploratory_ceemdan_oracle,
    )
    dashboard = {
        "overall": result["overall"],
        "by_signal": result["by_signal"],
        "by_noise": result["by_noise"],
        "by_sigma": result["by_sigma"],
        "candidate_usage": result["candidate_usage"],
        "figures": result["figures"],
        "exploratory_oracle_irmf_vs_ceemdan_summary": result[
            "exploratory_oracle_irmf_vs_ceemdan_summary"
        ],
        "rho_lock": locked["rho_lock"],
        "quick": bool(quick),
        "exploratory_ceemdan_oracle_run": bool(run_exploratory_ceemdan_oracle),
        "parameter_source": locked.get("parameter_source", "locked_algorithm_parameters"),
        "parameter_lock_audit": locked.get("parameter_lock_audit"),
        "interpretation": result["protocol"].get("recommended_paper_language"),
    }
    write_json(dashboard, root / "oracle_adaptivity_dashboard.json")
    write_paper_section_map(root)
    return dashboard


def run_v518_paper_pipeline(
        output_root,
        run_methodology=True,
        quick_regime_map=False,
        calibrate=True,
        quick_unified_cube=False,
        seeds=None,
        timeout_seconds=120,
        use_current_config_params=False,
        run_oracle=False,
        quick_oracle=False,
        run_exploratory_ceemdan_oracle=False,
        run_assets=True,
):
    root = ensure_dir(output_root)
    if run_methodology:
        from methodology.pipeline import run_methodology_pipeline
        run_methodology_pipeline(root.parent / "methodology", quick_regime_map=quick_regime_map)
    cube = run_unified_benchmark_cube_stage(
        root,
        calibrate=calibrate,
        quick=quick_unified_cube,
        seeds=seeds,
        timeout_seconds=timeout_seconds,
        use_current_config_params=use_current_config_params,
    )
    stats = run_unified_cube_statistics_stage(root)
    proxy = run_proxy_metric_validation_stage(root)
    oracle = None
    if run_oracle:
        oracle = run_oracle_adaptivity_stage(
            root,
            calibrate=calibrate,
            quick=quick_oracle,
            use_current_config_params=use_current_config_params,
            run_exploratory_ceemdan_oracle=run_exploratory_ceemdan_oracle,
        )
    assets = None
    if run_assets:
        from paper_assets import run_paper_assets
        assets = run_paper_assets(root.parent)
    dashboard = {
        "pipeline_version": "V5.20",
        "scope": "unified_cube_paper_package",
        "single_source_of_truth": "algorithm/03_unified_benchmark_cube/unified_benchmark_cube_rows.csv",
        "unified_cube": cube["dashboard"],
        "unified_cube_statistics": stats,
        "proxy_metric_validation": proxy,
        "oracle_adaptivity": oracle,
        "paper_assets": assets,
    }
    write_json(dashboard, root / "v520_paper_package_dashboard.json")
    write_paper_section_map(root)
    write_manifest(root, {
        "pipeline_version": "V5.20",
        "scope": "unified_cube_paper_package",
        "completed_stages": [
            *([] if not run_methodology else ["methodology"]),
            "unified_cube",
            "unified_cube_statistics",
            *([] if oracle is None else ["oracle_adaptivity"]),
            *([] if assets is None else ["paper_assets"]),
        ],
    })
    return dashboard


def run_algorithm_pipeline(output_root, calibrate=True, full_family=True, sensitivity=True,
                           challenging=True, quick_challenging=False):
    root = ensure_dir(output_root)
    locked = run_parameter_selection_stage(root, calibrate=calibrate)
    benchmark = run_benchmark_stage(root, calibrate=False, full_family=full_family)
    statistics = run_statistics_stage(root, calibrate=False, ensure_benchmark=full_family)
    sensitivity_results = run_sensitivity_stage(
        root, calibrate=False, emd_family_sensitivity=sensitivity,
    )
    challenging_signals = None
    if challenging:
        challenging_signals = run_challenging_signal_stage(
            root, calibrate=False, quick=quick_challenging,
        )
    dashboard = {
        "scope": "algorithm_validation",
        "rho_function_fixed": True,
        "rho_lock": locked["rho_lock"],
        "selected_irmf_params": locked["IRMF"],
        "selected_emd_params": locked["EMD"],
        "selected_eemd_params": locked["EEMD"],
        "selected_ceemdan_params": locked["CEEMDAN"],
        "benchmark": {
            "section_5_primary_benchmark": "algorithm/03_canonical_emd_family_benchmark",
            "family_summary": benchmark["family_summary"],
        },
        "statistics": statistics,
        "sensitivity": sensitivity_results,
        "challenging_signals": None if challenging_signals is None else {
            "summary": challenging_signals["summary"],
            "quick": challenging_signals["quick"],
        },
    }
    write_json(dashboard, root / "algorithm_dashboard.json")
    write_paper_section_map(root)
    write_manifest(root, {
        "pipeline_version": "V5.6",
        "scope": "algorithm_validation",
        "rho_lock": locked["rho_lock"],
        "completed_stages": [
            "parameter_selection", "benchmark", "statistics", "sensitivity",
            *([] if challenging_signals is None else ["challenging_signal_suite"]),
        ],
    })
    return dashboard
