#!/usr/bin/python
# coding: UTF-8

"""V5.32 Section 6 sensitivity-scope and reviewer-question protocol."""

from datetime import datetime, timezone

from project_config import (
    CONTAMINATION_LAMBDA_GRID,
    CONTAMINATION_ROBUSTNESS_SIGMA,
    GLOBAL_IRMF_PARAMS,
    PARAMETER_SENSITIVITY_FACTORS,
    PARAMETER_SENSITIVITY_NOISES,
    PARAMETER_SENSITIVITY_SIGMAS,
    PARAMETER_SENSITIVITY_SIGNALS,
)
from experiments.paper_pipeline_utils import ensure_dir, write_json


V532_SECTION6_SCOPE_VERSION = "V5.32_compact_section6_sensitivity_scope"


def run_v532_section6_scope(output_root):
    output_root = ensure_dir(output_root)
    h_sensitivity_factors = {
        **{k: list(v) for k, v in PARAMETER_SENSITIVITY_FACTORS.items()},
    }
    spec = {
        "schema_version": V532_SECTION6_SCOPE_VERSION,
        "scope_status": "design_recorded_requires_targeted_implementation",
        "benchmark_results_recomputed": False,
        "algorithm_cube_rerun_required": False,
        "section_title": "Sensitivity and Robustness Analyses",
        "section_purpose": (
            "Answer the small set of robustness questions most likely to affect "
            "the main synthetic benchmark claims. Detailed protocol checks are "
            "archived in supplementary/protocol audit material rather than "
            "expanded into the main narrative."
        ),
        "recommended_section_structure": [
            {
                "section": "6.1",
                "title": "IRMF Parameter Sensitivity and Robustness",
                "reviewer_question": (
                    "Does the IRMF conclusion depend on one exact locked "
                    "parameter point?"
                ),
                "status": "implemented_with_relative_c_H_and_v530_target_snr_alignment",
                "main_text_scope": [
                    "local sensitivity of h1, a, h_min, and c_H around locked values",
                    "targeted h1-by-a and a-by-h_min interactions",
                    "optional c_H-by-contamination-severity interaction if budget allows",
                    "endpoint-level and construct-level robustness of paired advantages",
                ],
                "deemphasized": [
                    "full factorial all-interaction grids",
                    "Sobol global analysis unless later required",
                    "overall rank as primary evidence",
                ],
            },
            {
                "section": "6.2",
                "title": "Contamination Design Sensitivity",
                "reviewer_question": (
                    "Do conclusions depend on a particular Huber contamination "
                    "rate, outlier magnitude, or outlier geometry?"
                ),
                "status": "partially_existing_rate_only_position_and_magnitude_missing",
                "subsections": [
                    "6.2.1 Contamination rate",
                    "6.2.2 Contamination magnitude",
                    "6.2.3 Contamination position / geometry",
                    "6.2.4 Position-by-magnitude mini-audit",
                ],
            },
            {
                "section": "6.3",
                "title": "Comparator and Signal Robustness",
                "reviewer_question": (
                    "Do conclusions depend on one specific signal construction or "
                    "one reasonable EMD-family baseline configuration?"
                ),
                "status": "existing_signal_variant_module_plus_existing_or_planned_baseline_neighborhood",
                "main_text_scope": [
                    "signal-family variants summarized compactly",
                    "reasonable EMD/EEMD/CEEMDAN baseline alternatives summarized compactly",
                    "full grids and distributions moved to supplement",
                ],
            },
            {
                "section": "6.4",
                "title": "Computational Efficiency and Scaling",
                "reviewer_question": (
                    "What practical computational cost is associated with the "
                    "locked methods and sensitivity analyses?"
                ),
                "status": "planned_runtime_and_scaling_summary_not_yet_run",
            },
        ],
        "main_text_reviewer_question_map": [
            {
                "reviewer_question": "Q1. Does IRMF depend on exact tuning?",
                "main_text_experiment": "6.1 IRMF parameter sensitivity plus targeted interactions",
            },
            {
                "reviewer_question": "Q2. Does robustness depend on one contamination design?",
                "main_text_experiment": "6.2 rate, magnitude, position, and small magnitude-by-position check",
            },
            {
                "reviewer_question": "Q3. Do conclusions depend on one signal family or baseline setting?",
                "main_text_experiment": "6.3 signal-family variants and EMD-family baseline alternatives",
            },
            {
                "reviewer_question": "Q4. What practical cost is associated with the observed gains?",
                "main_text_experiment": "6.4 runtime and runtime-versus-n scaling",
            },
        ],
        "supplementary_or_protocol_audits": [
            {
                "name": "Monte Carlo seed convergence",
                "role": "supplementary robustness audit",
                "reason_moved_out_of_main_text": (
                    "Important for stochastic sufficiency, but not central to the "
                    "paper's main sensitivity narrative."
                ),
                "recommended_design": "nested prefixes R in {10,20,30,50}",
            },
            {
                "name": "Sampling-resolution / discretization sensitivity",
                "role": "supplementary robustness audit",
                "reason_moved_out_of_main_text": (
                    "Checks normalized-time discretization robustness of n, but "
                    "does not alter the main scientific story."
                ),
                "recommended_design": "n in {250,500,1000,2000} on the same t_unit in [0,1)",
            },
            {
                "name": "Association-threshold sensitivity",
                "role": "supplementary reconstruction-protocol sensitivity audit",
                "reason_moved_out_of_main_text": (
                    "The threshold tau controls the common protocol reconstruction, "
                    "not any method's native algorithm. It is therefore best treated "
                    "as a protocol robustness audit."
                ),
                "recommended_design": "tau in {0.025,0.05,0.075,0.10}; no retuning or best-threshold selection",
            },
            {
                "name": "Detailed parameter interaction tables",
                "role": "supplementary detail",
                "reason_moved_out_of_main_text": "Full interaction tables would dilute the main reviewer-question narrative.",
            },
            {
                "name": "Detailed baseline parameter grids",
                "role": "supplementary detail",
                "reason_moved_out_of_main_text": "Main text should report compact stability conclusions only.",
            },
            {
                "name": "Full sensitivity distributions",
                "role": "supplementary detail",
                "reason_moved_out_of_main_text": "Distributional detail is useful for auditability but too bulky for the main text.",
            },
            {
                "name": "Ranking and aggregation robustness",
                "role": "optional supplementary audit",
                "reason_moved_out_of_main_text": (
                    "Overall rank is a secondary descriptive summary; endpoint-level "
                    "and construct-level results remain the primary evidence."
                ),
            },
            {
                "name": "SNR realization audit",
                "role": "protocol/data-generation audit",
                "reason_moved_out_of_main_text": (
                    "SNR is the Section 5 severity axis. The audit validates target "
                    "realization rather than testing a separate performance perturbation."
                ),
            },
            {
                "name": "Standalone boundary sensitivity",
                "role": "deferred or supplementary only",
                "reason_moved_out_of_main_text": (
                    "The compact main Section 6 prioritizes protocol parameters that "
                    "directly control primary endpoint interpretation; boundary tests "
                    "can remain a legacy/deferred supplementary check unless reviewer "
                    "pressure makes them central."
                ),
            },
        ],
        "snr_policy": {
            "main_noise_severity_axis": "Section 5 target-SNR benchmark design",
            "finite_snr_levels_db": [0, 5, 10, 15, 20, 25, 30],
            "clean_reference": "reported separately; not a finite SNR level",
            "section6_snr_sensitivity_status": "not_included_design_absorbed_into_main_benchmark",
            "retained_audit": "SNR realization / calibration audit only",
            "rationale": (
                "Because V5.30 uses target realized SNR as the formal benchmark "
                "severity axis, performance as a function of SNR is analyzed in "
                "Section 5 rather than repeated as a Section 6 sensitivity test."
            ),
        },
        "result_evidence_hierarchy": {
            "level_1_primary_evidence": {
                "name": "endpoint_level_results",
                "description": (
                    "Each of the 13 primary endpoints is reported directly with "
                    "central tendency, distributional spread, paired effects, and "
                    "paired win/loss behavior."
                ),
                "recommended_statistics": [
                    "median",
                    "IQR",
                    "mean_when_interpretable",
                    "paired_effect_size",
                    "bootstrap_or_confidence_interval",
                    "paired_win_rate",
                ],
            },
            "level_2_primary_interpretation": {
                "name": "construct_level_narrative",
                "description": (
                    "The five primary constructs are interpreted separately: signal "
                    "recovery, component recovery quality, component-set fidelity, "
                    "noise separation, and contamination resistance."
                ),
                "policy": (
                    "Do not force conceptually distinct endpoints into a single "
                    "composite score."
                ),
            },
            "level_3_secondary_summary": {
                "name": "descriptive_average_ranks",
                "description": (
                    "Average ranks may be reported as a compact descriptive overview "
                    "but are not the sole basis for scientific claims."
                ),
                "required_wording": (
                    "Overall ranks are descriptive summaries; scientific conclusions "
                    "are based on endpoint-specific and construct-specific results."
                ),
            },
            "prohibited_primary_claim_pattern": (
                "Do not reduce the 13 primary endpoints to one winner-only composite "
                "claim such as 'method A wins overall' without endpoint-level support."
            ),
        },
        "methods_note": {
            "time_axis_semantics": (
                "Time-axis semantics are a Methods/design clarification, not a "
                "standalone main-text Section 6 experiment. Supplementary "
                "discretization sensitivity may cite the normalized-time definition."
            ),
            "synthetic_physical_duration": (
                "Synthetic physical-duration sensitivity is optional secondary "
                "work, not required for the core benchmark."
            ),
            "real_ecg_window_duration": (
                "Physical-window sensitivity belongs to real-world validation, "
                "not Section 6 synthetic sensitivity."
            ),
        },
        "contamination_design_sensitivity": {
            "baseline": {
                "noise_model": "huber_contamination",
                "lambda": float(0.05),
                "outlier_scale": float(10.0),
                "sigma": float(CONTAMINATION_ROBUSTNESS_SIGMA),
            },
            "rate_sensitivity": {
                "status": "existing_legacy_module_available",
                "lambda_grid": list(CONTAMINATION_LAMBDA_GRID),
                "purpose": "stress-test contamination fraction while holding the outlier model fixed",
            },
            "magnitude_sensitivity": {
                "status": "missing_targeted_module_recommended",
                "recommended_parameter": "outlier_scale",
                "recommended_grid": [3.0, 5.0, 8.0, 10.0, 15.0],
                "scale_interpretation_note": (
                    "The current Huber generator creates large outliers before "
                    "standardization/SNR scaling. For publication wording, describe "
                    "this as pre-standardization outlier magnitude and report the "
                    "post-scaling target-SNR design separately."
                ),
            },
            "position_geometry_sensitivity": {
                "status": "missing_targeted_module_recommended",
                "recommended_patterns": [
                    "random_dispersed",
                    "clustered_block",
                    "boundary",
                    "high_curvature_or_transient_aligned",
                    "smooth_region_aligned",
                    "peak_or_extremum_aligned",
                ],
                "purpose": (
                    "Evaluate whether robustness depends on where contamination "
                    "falls relative to local support, boundary asymmetry, curvature, "
                    "transients, or extrema/envelope structure."
                ),
            },
            "position_by_magnitude_mini_audit": {
                "status": "missing_targeted_module_recommended",
                "recommended_positions": ["random_dispersed", "boundary", "clustered_block"],
                "recommended_outlier_scales": [5.0, 10.0, 15.0],
                "purpose": (
                    "Detect interactions where geometry is benign at moderate "
                    "magnitude but harmful at high magnitude."
                ),
            },
            "method_neutrality": (
                "The same contamination masks, magnitudes, target-SNR levels, "
                "signals, and seeds must be used for IRMF, EMD, EEMD, and CEEMDAN."
            ),
        },
        "irmf_parameter_sensitivity_and_robustness": {
            "current_locked_parameters": {
                "h1": float(GLOBAL_IRMF_PARAMS["h1"]),
                "a": float(GLOBAL_IRMF_PARAMS["a"]),
                "h_min": float(GLOBAL_IRMF_PARAMS["h_min"]),
                "H": float(GLOBAL_IRMF_PARAMS["H"]),
            },
            "existing_module_factors": {
                k: list(v) for k, v in PARAMETER_SENSITIVITY_FACTORS.items()
            },
            "required_v532_extension": {
                "include_H": True,
                "factors": h_sensitivity_factors,
                "rationale": (
                    "H is the Gaussian-smoothed-median robust-contrast width and "
                    "a tuning meta-parameter in the IRMF procedure. It should be "
                    "included in neighborhood sensitivity even though it was not "
                    "tuned in the development-set selection."
                ),
            },
            "recommended_design": {
                "one_at_a_time": {
                    "status": "recommended_main_text_local_parameter_sensitivity",
                    "levels": [0.8, 0.9, 1.0, 1.1, 1.2],
                    "parameters": ["h1", "a", "h_min", "H"],
                },
                "targeted_interactions": {
                    "status": "recommended_main_text_targeted_interactions",
                    "pairs": ["h1 x a", "a x h_min"],
                    "optional_if_budget_allows": ["H x contamination_severity"],
                    "rationale": (
                        "h1 and a jointly determine the scale trajectory; a and "
                        "h_min jointly determine how many scales remain available. "
                        "H interactions are only main-text relevant when tied to "
                        "contamination severity."
                    ),
                },
                "larger_joint_neighborhood": {
                    "status": "supplementary_only_if_budget_allows",
                    "avoid_full_factorial_3_to_4_if_costly": True,
                    "suggested_points": [
                        "center",
                        "one_factor_low_high",
                        "few_balanced_corner_combinations",
                    ],
                },
            },
            "scope": {
                "method": "IRMF_only",
                "purpose": (
                    "local sensitivity, parameter interactions, parameter importance, "
                    "and comparative-conclusion robustness around the locked IRMF "
                    "parameters, not retuning"
                ),
                "signals": list(PARAMETER_SENSITIVITY_SIGNALS),
                "noises": list(PARAMETER_SENSITIVITY_NOISES),
                "sigmas": list(PARAMETER_SENSITIVITY_SIGMAS),
            },
        },
        "association_threshold_sensitivity": {
            "status": "recommended_supplementary_protocol_robustness_audit",
            "classification": "reconstruction_protocol_sensitivity_not_algorithm_parameter_sensitivity",
            "frozen_baseline_tau": 0.05,
            "selection_rule": "selected_i = max_j A_ij > tau",
            "protocol_reconstruction": (
                "X_protocol is the sum of selected estimated components; "
                "epsilon_hat = Y - X_protocol."
            ),
            "recommended_tau_grid": [0.025, 0.05, 0.075, 0.10],
            "affected_primary_endpoints": [
                "denoise_nmse",
                "denoise_corr",
                "noise_capture_corr",
                "noise_energy_log_error",
                "signal_leakage_into_noise",
                "contaminated_region_nmse",
                "clean_region_nmse",
                "normalized_contamination_spillover_loss",
            ],
            "not_directly_affected_primary_endpoints": [
                "matched_component_corr",
                "matched_component_nrmse",
                "relative_decomposition_count_error",
                "missing_true_component_energy_ratio",
                "spurious_mode_energy_ratio",
            ],
            "recommended_outputs": [
                "selected_component_count_by_tau",
                "selected_energy_fraction_by_tau",
                "selector_change_rate_vs_tau0",
                "residual_candidate_selection_rate_by_tau",
                "reconstruction_dependent_endpoint_delta_by_tau",
                "endpoint_level_method_ordering_stability_by_tau",
                "construct_level_interpretation_stability_by_tau",
            ],
            "interpretation": (
                "Low tau can falsely include weakly associated components in "
                "X_protocol; high tau can falsely exclude weak but genuine "
                "signal-bearing components. The audit tests stability of this "
                "false-inclusion / false-exclusion trade-off without selecting a "
                "post-hoc best threshold."
            ),
        },
        "boundary_effect_sensitivity": {
            "status": "deferred_or_supplementary_existing_legacy_module_available_requires_v530_alignment",
            "current_legacy_design": {
                "IRMF_variants": ["IRMF-periodic", "IRMF-mirror"],
                "EMD_variants": ["EMD-nbsym2", "EMD-nbsym4"],
                "signals": [
                    "chirp",
                    "frequency_jump",
                    "impulsive_transient",
                    "intermittent_oscillation",
                ],
                "noises": ["gaussian", "impulsive", "huber_contamination"],
                "legacy_sigmas": [0.10, 0.20],
            },
            "purpose": (
                "Evaluate whether endpoint behavior and boundary handling affect "
                "method conclusions, especially for nonstationary, transient, and "
                "contaminated signals."
            ),
            "v530_alignment_required_before_execution": (
                "Replace legacy sigma levels with a frozen representative target-SNR "
                "subset before using this as final Section 6 evidence."
            ),
            "recommended_outputs": [
                "boundary_variant_metric_summary",
                "boundary_region_error_summary",
                "interior_vs_boundary_performance_gap",
                "rank_stability_across_boundary_variants",
            ],
        },
        "ranking_and_aggregation_robustness": {
            "status": "optional_supplementary_secondary_audit",
            "not_core_if": (
                "The manuscript bases primary claims on endpoint-level and "
                "construct-level evidence rather than on a single overall rank."
            ),
            "recommended_if_used": (
                "Run leave-one-metric-out and leave-one-dimension-out checks if "
                "overall average rank is presented as a prominent headline claim."
            ),
            "allowed_role": (
                "Descriptive synthesis and visualization, not primary performance "
                "evidence."
            ),
        },
        "freeze_conditions_before_execution": [
            "contamination mask geometry rules frozen",
            "outlier magnitude grid frozen",
            "target-SNR compatibility policy frozen",
            "IRMF H sensitivity treatment frozen",
            "association-threshold sensitivity tau grid frozen if run",
            "representative signal/noise/SNR subset frozen",
            "main-text versus supplementary audit roles frozen",
            "no benchmark/test outcome inspected for choosing grids",
        ],
        "created_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(spec, output_root / "v532_section6_scope_protocol.json")
    return spec
