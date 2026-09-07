#!/usr/bin/python
# coding: UTF-8

"""V5.39 IRMF H-tuning protocol amendment and semantics audit."""

from datetime import datetime, timezone

from project_config import (
    GLOBAL_IRMF_PARAMS,
    PAPER_IRMF_COARSE_PARAMETER_GRID,
    PAPER_IRMF_REFINEMENT_STEPS,
    PARAMETER_SELECTION_NOISES,
    PARAMETER_SELECTION_SIGMAS,
    PARAMETER_SELECTION_SIGNALS,
)
from experiments.paper_pipeline_utils import ensure_dir, write_json


V539_IRMF_H_TUNING_AMENDMENT_VERSION = "V5.39_irmf_h_tuning_protocol_amendment"
EXPECTED_DEVELOPMENT_CASE_COUNT = 40


def run_v539_irmf_h_tuning_amendment(output_root):
    output_root = ensure_dir(output_root)
    h_levels = tuple(float(v) for v in PAPER_IRMF_COARSE_PARAMETER_GRID["H_options"])
    stage1_candidates = (
        len(PAPER_IRMF_COARSE_PARAMETER_GRID["h1_options"])
        * len(PAPER_IRMF_COARSE_PARAMETER_GRID["a_options"])
        * len(PAPER_IRMF_COARSE_PARAMETER_GRID["h_min_options"])
        * len(PAPER_IRMF_COARSE_PARAMETER_GRID["H_options"])
    )
    development_cases = (
        len(PARAMETER_SELECTION_SIGNALS)
        * len(PARAMETER_SELECTION_NOISES)
        * len(PARAMETER_SELECTION_SIGMAS)
    )
    case_count_matches_protocol = development_cases == EXPECTED_DEVELOPMENT_CASE_COUNT
    status = {
        "schema_version": V539_IRMF_H_TUNING_AMENDMENT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "amendment_status": "upstream_parameter_selection_amended_not_run",
        "scope": "IRMF development-set parameter-selection protocol",
        "section6_full_execution_status": "blocked_until_four_parameter_selection_resolved",
        "benchmark_results_recomputed": False,
        "statistics_regenerated": False,
        "claim_authorized": False,
        "current_locked_config_before_reselection": {
            "h1": float(GLOBAL_IRMF_PARAMS["h1"]),
            "a": float(GLOBAL_IRMF_PARAMS["a"]),
            "h_min": float(GLOBAL_IRMF_PARAMS["h_min"]),
            "H": float(GLOBAL_IRMF_PARAMS["H"]),
        },
        "h_semantics_audit": {
            "implementation_path": "core_algorithms/robust_losses.py::gaussian_smoothed_median_loss",
            "local_estimator_path": "core_algorithms/strict_spokoiny_irmf.py::local_m_estimator",
            "formula": (
                "rho_H(x) = sqrt(2/pi) H exp(-x^2/(2H^2)) "
                "+ x {2 Phi(x/H) - 1}"
            ),
            "gradient": "rho'_H(x) = 2 Phi(x/H) - 1",
            "hessian": "rho''_H(x) = (2/H) phi(x/H)",
            "residual_standardization_before_loss": False,
            "scale_semantics": "absolute_residual_scale_parameter",
            "audit_interpretation": (
                "H is passed directly to the robust loss applied to local residuals. "
                "It is therefore a genuine IRMF hyperparameter controlling the robust "
                "contrast scale, not a derived normalized constant."
            ),
        },
        "development_tuning_amendment": {
            "previous_theta": ["h1", "a", "h_min"],
            "revised_theta": ["h1", "a", "h_min", "H"],
            "no_per_case_tuning": True,
            "selection_unit": "common global IRMF configuration selected on the development set",
            "held_out_benchmark_not_used_for_selection": True,
            "H_options": list(h_levels),
            "H_refinement_step": float(PAPER_IRMF_REFINEMENT_STEPS["H"]),
            "stage1_n_candidates": int(stage1_candidates),
            "n_development_cases": int(development_cases),
            "expected_development_cases": int(EXPECTED_DEVELOPMENT_CASE_COUNT),
            "case_count_matches_protocol": bool(case_count_matches_protocol),
            "expected_stage1_method_evaluations": int(stage1_candidates * development_cases),
            "selection_command": (
                "python paper_pipeline.py --output-root <root> "
                "--force-parameter-selection parameter-selection"
            ),
        },
        "selection_rule_freeze": {
            "primary_objective": "maximize mean_case_score across the 40 frozen development cases",
            "objective_reused_from_prior_selection": True,
            "H_added_to_candidate_grid_without_changing_objective": True,
            "case_score_formula": (
                "0.40 * reconstruction_score + 0.35 * structural_fidelity_score "
                "+ 0.25 * contamination_resistance_score"
            ),
            "cross_case_aggregation_rule": (
                "arithmetic mean over the same 40 development cases"
            ),
            "candidate_eligibility_rule": (
                "candidate selectable only if all 40 development cases yield "
                "finite case_score values"
            ),
            "failed_candidate_handling": (
                "non-finite case_score rows are archived and make the candidate "
                "ineligible for selection"
            ),
            "tie_breaking_rule": [
                "larger mean_case_score",
                "larger mean_reconstruction_score",
                "larger mean_structural_fidelity_score",
                "larger mean_contamination_resistance_score",
                "smaller H",
                "smaller h1",
                "smaller a",
                "smaller h_min",
                "smaller candidate_id",
            ],
            "one_global_locked_H": True,
            "case_wise_H_tuning_allowed": False,
        },
        "H_absolute_scale_diagnostic_plan": {
            "diagnostic_only_not_used_for_selection": True,
            "rationale": (
                "Because H is absolute in the current loss implementation, "
                "post-selection diagnostics should inspect performance by signal "
                "RMS/amplitude strata before final benchmark rerun claims."
            ),
            "planned_outputs": [
                "selected_H_performance_by_signal_rms_stratum",
                "candidate_H_score_profiles_by_signal_family",
                "finite_case_score_rate_by_H_level",
            ],
        },
        "downstream_implications": {
            "target_snr_design_rerun_trigger": (
                "The V5.30 fixed-sigma to target-SNR redesign changes the noisy "
                "inputs and therefore requires a full algorithm-cube rerun for "
                "IRMF, EMD, EEMD, and CEEMDAN before final benchmark claims."
            ),
            "execution_order": (
                "Resolve V5.39 four-parameter IRMF development selection before "
                "starting the V5.30 target-SNR full rerun, so the final rerun is "
                "performed once under the final locked configuration."
            ),
            "if_selected_H_equals_current_H": (
                "Existing H=1.0-centered Section 6.1 sensitivity remains centered "
                "correctly, but the development-selection provenance is strengthened."
            ),
            "if_selected_H_differs_from_current_H": (
                "IRMF decomposition outputs change; the formal benchmark cube must "
                "rerun IRMF evaluations under the new locked configuration before "
                "statistics or Section 6 full execution are interpreted."
            ),
            "section6_1_center_point_rule": (
                "Section 6.1 sensitivity must be centered at the final locked "
                "(h1*, a*, h_min*, H*) selected before the main benchmark."
            ),
            "algorithm_cube_rerun_required_if_H_star_changes": True,
            "statistics_only_regeneration_sufficient_if_H_star_changes": False,
            "target_snr_full_rerun_required_regardless_of_H_selection": True,
        },
        "governance": {
            "development_tuning": "selects H* together with h1*, a*, h_min*",
            "main_benchmark": "uses locked H* without per-case adaptation",
            "section6_sensitivity": "perturbs locked H* only to test conclusion robustness",
            "no_retuning_from_section6_results": True,
        },
    }
    write_json(status, output_root / "v539_irmf_h_tuning_amendment.json")
    write_json(status, output_root / "v539_irmf_h_tuning_status.json")
    return status
