#!/usr/bin/python
# coding: UTF-8

"""V5.42 relative noise-scale IRMF H parameterization protocol."""

from datetime import datetime, timezone

from project_config import (
    IRMF_RELATIVE_H_C_OPTIONS,
    IRMF_RELATIVE_H_C_REFINEMENT_STEP,
    IRMF_RELATIVE_H_PARAMETERIZATION_VERSION,
    IRMF_RELATIVE_H_SCALE_ESTIMATOR,
    PARAMETER_SELECTION_NOISES,
    PARAMETER_SELECTION_SIGMAS,
    PARAMETER_SELECTION_SIGNALS,
)
from experiments.paper_pipeline_utils import ensure_dir, write_json


EXPECTED_DEVELOPMENT_CASE_COUNT = 40


def run_v542_irmf_relative_h_parameterization(output_root):
    output_root = ensure_dir(output_root)
    development_cases = (
        len(PARAMETER_SELECTION_SIGNALS)
        * len(PARAMETER_SELECTION_NOISES)
        * len(PARAMETER_SELECTION_SIGMAS)
    )
    status = {
        "schema_version": IRMF_RELATIVE_H_PARAMETERIZATION_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "amendment_status": "relative_H_protocol_frozen_pending_scale_estimator_qualification",
        "scope": "IRMF robust-loss smoothing-scale parameterization",
        "benchmark_results_recomputed": False,
        "statistics_regenerated": False,
        "claim_authorized": False,
        "primary_schema_changed": False,
        "section6_full_execution_status": (
            "blocked_until_final_IRMF_parameterization_and_parameter_selection_resolved"
        ),
        "motivation": {
            "problem_with_absolute_H": (
                "The Gaussian-smoothed median loss width H is applied on the "
                "absolute local residual scale. Under target-SNR inputs, absolute "
                "noise amplitude varies across signals and SNR levels, so a single "
                "global absolute H can imply different smoothing strength across cases."
            ),
            "scientific_resolution": (
                "Tune a dimensionless c_H once on the development set, then derive "
                "H_case mechanically from a frozen truth-free noise-scale estimator."
            ),
        },
        "loss_semantics": {
            "loss": "gaussian_smoothed_median",
            "formula": (
                "rho_H(x) = sqrt(2/pi) H exp(-x^2/(2H^2)) "
                "+ x {2 Phi(x/H) - 1}"
            ),
            "H_interpretation": (
                "Gaussian convolution / smoothing standard deviation of the "
                "absolute-loss contrast; H -> 0 approaches |x|, larger H makes "
                "the contrast smoother near the origin."
            ),
            "not_a_huber_threshold": True,
        },
        "relative_parameterization": {
            "locked_hyperparameter": "c_H",
            "case_specific_derived_quantity": "H_realized",
            "formula": "H_realized_case = c_H_locked * sigma_hat_noise_case",
            "locked_parameter_vector": ["h1", "a", "h_min", "c_H"],
            "absolute_H_no_longer_tuned_directly": True,
            "case_wise_retuning_allowed": False,
            "one_global_locked_c_H": True,
        },
        "truth_free_noise_scale_estimator_protocol": {
            "estimator_id": IRMF_RELATIVE_H_SCALE_ESTIMATOR,
            "input": "observed signal Y only",
            "uses_clean_truth_or_true_noise": False,
            "formula": (
                "sigma_hat = median(|Delta Y - median(Delta Y)|) / "
                "(0.67448975 * sqrt(2)), where Delta Y_t = Y_t - Y_{t-1}"
            ),
            "gaussian_white_noise_calibration": (
                "For iid Gaussian noise, std(Delta epsilon) = sqrt(2) sigma_epsilon; "
                "MAD / 0.67448975 estimates std(Delta epsilon)."
            ),
            "fallback_policy": (
                "If the estimator is non-finite or <= 0, fall back to "
                "median(|Y - median(Y)|) / 0.67448975; if still non-finite or <= 0, "
                "mark the IRMF evaluation ineligible rather than using true noise."
            ),
            "caveat": (
                "First-difference MAD can be influenced by sharp signal transitions, "
                "colored noise, heavy tails, or contamination; therefore it requires "
                "a scale-estimator qualification audit before final adoption."
            ),
        },
        "development_selection_revision": {
            "supersedes_v539_absolute_H_grid_if_qualified": True,
            "selection_unit": "one common global IRMF configuration selected on the development set",
            "development_case_count": int(development_cases),
            "expected_development_case_count": int(EXPECTED_DEVELOPMENT_CASE_COUNT),
            "case_count_matches_protocol": bool(development_cases == EXPECTED_DEVELOPMENT_CASE_COUNT),
            "c_H_options": list(IRMF_RELATIVE_H_C_OPTIONS),
            "c_H_refinement_step": float(IRMF_RELATIVE_H_C_REFINEMENT_STEP),
            "objective_rule": (
                "Reuse the frozen development selection objective; replace H_options "
                "with c_H_options without redesigning the objective."
            ),
            "no_per_case_selection": (
                "Development selects c_H*, not H_case. Main benchmark computes "
                "H_case = c_H* sigma_hat(Y_case) for every case."
            ),
        },
        "required_qualification_before_activation": {
            "scale_estimator_qualification_required": True,
            "scale_estimator_checks": [
                "distribution of sigma_hat by signal family, noise family, and SNR",
                "ratio to realized synthetic noise RMS archived for audit only",
                "behavior on clean baseline where true noise is zero",
                "sensitivity to colored noise, heavy tails, impulsive and Huber contamination",
                "finite positive estimator rate",
            ],
            "method_neutrality_rule": (
                "sigma_hat must depend only on Y and the frozen estimator protocol, "
                "not on method label, extracted components, true clean signal, or true noise."
            ),
            "activation_rule": (
                "Only after qualification may V5.42 replace V5.39 absolute-H "
                "selection in the final IRMF development tuning."
            ),
        },
        "provenance_fields_to_archive_per_case": [
            "c_H_locked",
            "noise_scale_hat",
            "H_realized",
            "noise_scale_estimator_id",
            "noise_scale_estimator_fallback_used",
        ],
        "downstream_implications": {
            "if_activated": (
                "V5.39 absolute-H selection is superseded; rerun development "
                "selection with theta=(h1,a,h_min,c_H), then run V5.30 target-SNR "
                "benchmark once under the selected relative-H configuration."
            ),
            "algorithm_cube_rerun_required": True,
            "statistics_only_regeneration_sufficient": False,
            "section6_1_center_point": (
                "Section 6.1 sensitivity should perturb c_H* around the selected "
                "relative-H configuration, not absolute H, if V5.42 is activated."
            ),
        },
    }
    write_json(status, output_root / "v542_irmf_relative_h_parameterization.json")
    write_json(status, output_root / "v542_irmf_relative_h_parameterization_status.json")
    return status
