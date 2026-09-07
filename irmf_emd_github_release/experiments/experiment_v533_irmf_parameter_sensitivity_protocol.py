#!/usr/bin/python
# coding: UTF-8

"""V5.33 IRMF parameter sensitivity and importance protocol."""

from datetime import datetime, timezone

from project_config import (
    GLOBAL_IRMF_PARAMS,
    IRMF_PARAMETER_SENSITIVITY_IMPORTANCE_DIMENSIONS,
    IRMF_PARAMETER_SENSITIVITY_JOINT_INTERACTIONS,
    IRMF_PARAMETER_SENSITIVITY_OAT_FACTORS,
    IRMF_PARAMETER_SENSITIVITY_PARAMETERS,
    PARAMETER_SENSITIVITY_NOISES,
    PARAMETER_SENSITIVITY_SIGMAS,
    PARAMETER_SENSITIVITY_SIGNALS,
    PARAMETER_SENSITIVITY_TARGET_SNR_DB_LEVELS,
)
from experiments.paper_pipeline_utils import ensure_dir, write_json


V533_IRMF_PARAMETER_SENSITIVITY_VERSION = "V5.33_irmf_parameter_sensitivity_protocol"


def _scaled_values(param_name, base_params):
    base = float(base_params[param_name])
    return [float(base * factor) for factor in IRMF_PARAMETER_SENSITIVITY_OAT_FACTORS]


def run_v533_irmf_parameter_sensitivity_protocol(output_root, irmf_params=None):
    output_root = ensure_dir(output_root)
    base_params = dict(irmf_params or GLOBAL_IRMF_PARAMS)
    relative_h_active = base_params.get("H_parameterization") == "relative_noise_scale"
    spec = {
        "schema_version": V533_IRMF_PARAMETER_SENSITIVITY_VERSION,
        "protocol_status": "design_recorded_not_run",
        "benchmark_results_recomputed": False,
        "algorithm_cube_rerun_required": False,
        "section": "6.4 IRMF Parameter Sensitivity and Robustness",
        "purpose": (
            "Assess local sensitivity, parameter interactions, parameter importance, "
            "and comparative-conclusion robustness around the locked IRMF "
            "configuration without retuning."
        ),
        "locked_parameter_point": {
            name: float(base_params[name])
            for name in IRMF_PARAMETER_SENSITIVITY_PARAMETERS
        },
        "H_parameterization": (
            "relative_noise_scale" if relative_h_active else base_params.get("H_parameterization", "absolute")
        ),
        "relative_H_rule": (
            "Section 6.1 perturbs the locked dimensionless c_H multiplier; "
            "each evaluation case then derives H_case = c_H * sigma_hat(Y) "
            "using the frozen truth-free first-difference MAD scale proxy."
            if relative_h_active else None
        ),
        "spokoiny_mechanism_map": {
            "h1": {
                "construct": "initial_smoothing_scale",
                "meaning": "starting bandwidth for the first robust local fit",
                "increase_effect": (
                    "stronger first-scale smoothing; gross/low-frequency structure "
                    "is emphasized but distinct low-frequency structure may merge"
                ),
                "expected_sensitive_outputs": [
                    "matched_component_corr",
                    "matched_component_nrmse",
                    "relative_decomposition_count_error",
                    "missing_true_component_energy_ratio",
                    "spurious_mode_energy_ratio",
                    "splitting_merging_diagnostics",
                ],
                "theoretical_priority": "high",
            },
            "a": {
                "construct": "scale_progression_rate",
                "meaning": "geometric bandwidth update h_{k+1} = h_k / a",
                "increase_effect": (
                    "coarser scale grid and fewer intermediate scales; may reduce "
                    "redundancy but can skip intrinsic scales"
                ),
                "expected_sensitive_outputs": [
                    "relative_decomposition_count_error",
                    "missing_true_component_energy_ratio",
                    "spurious_mode_energy_ratio",
                    "true_component_splitting_max",
                    "estimated_component_merging_max",
                    "close_frequency_recovery",
                ],
                "theoretical_priority": "high",
            },
            "h_min": {
                "construct": "finest_accessible_scale",
                "meaning": "stopping bandwidth that limits the smallest extracted scale",
                "increase_effect": (
                    "earlier stopping and fewer fine-scale components; may miss "
                    "weak/high-frequency truth but reduce noise-like components"
                ),
                "expected_sensitive_outputs": [
                    "missing_true_component_energy_ratio",
                    "spurious_mode_energy_ratio",
                    "relative_decomposition_count_error",
                    "noise_capture_corr",
                    "noise_energy_log_error",
                    "signal_leakage_into_noise",
                ],
                "theoretical_priority": "medium_high",
            },
            "c_H": {
                "construct": "relative_robust_loss_threshold_multiplier",
                "meaning": (
                    "Dimensionless multiplier for the Gaussian-smoothed-median "
                    "robust contrast width.  The operating loss width is derived "
                    "per case as H_case=c_H*sigma_hat(Y)."
                ),
                "increase_effect": (
                    "larger case-derived H; fit becomes closer to quadratic/mean-like "
                    "behavior at the same observed scale proxy, while smaller c_H "
                    "increases outlier downweighting"
                ),
                "expected_sensitive_outputs": [
                    "contaminated_region_nmse",
                    "clean_region_nmse",
                    "normalized_contamination_spillover_loss",
                    "noise_capture_corr",
                    "signal_leakage_into_noise",
                    "matched_component_recovery_on_transients",
                ],
                "theoretical_priority": "high_especially_contamination",
            },
        },
        "four_layer_question_structure": {
            "local_sensitivity": {
                "question": "How much do metrics change under small one-parameter perturbations?",
                "primary_design": "one-at-a-time response curves around the locked parameter point",
            },
            "parameter_interaction": {
                "question": "Do pairs of parameters have non-additive effects?",
                "primary_design": "small structured joint design focused on mechanistically linked pairs",
            },
            "parameter_importance": {
                "question": "Which parameters explain the most performance variation?",
                "primary_design": "hierarchical/factorial effect decomposition and optional Sobol indices",
            },
            "conclusion_robustness": {
                "question": "Do the qualitative IRMF-vs-baseline conclusions survive parameter perturbation?",
                "primary_design": "rank, win-rate, and paired-effect sign stability across declared configurations",
            },
        },
        "sensitivity_vs_importance_distinction": {
            "sensitivity": (
                "How much does a metric change when one parameter is locally "
                "perturbed around the locked value?"
            ),
            "importance": (
                "How much benchmark variation is attributable to each parameter "
                "over the declared neighborhood?"
            ),
            "robustness": (
                "Whether the qualitative comparative conclusion is preserved "
                "across the declared parameter neighborhood."
            ),
            "required_reporting": [
                "parameter_response_curves",
                "standardized_local_sensitivity_indices",
                "construct_specific_importance_summary",
                "selected_interaction_summary",
                "conclusion_robustness_summary",
            ],
        },
        "one_at_a_time_design": {
            "status": "primary_recommended_design_not_run",
            "factors": list(IRMF_PARAMETER_SENSITIVITY_OAT_FACTORS),
            "parameters": list(IRMF_PARAMETER_SENSITIVITY_PARAMETERS),
            "scaled_values": {
                name: _scaled_values(name, base_params)
                for name in IRMF_PARAMETER_SENSITIVITY_PARAMETERS
            },
            "rule": (
                "Vary one parameter at a time while holding all other locked "
                "IRMF parameters fixed."
            ),
            "reviewer_question": (
                "Are conclusions stable under +/-10-20% local perturbations of "
                "each IRMF parameter?"
            ),
        },
        "joint_sensitivity_design": {
            "status": "secondary_mini_audit_recommended_not_run",
            "interactions": [list(pair) for pair in IRMF_PARAMETER_SENSITIVITY_JOINT_INTERACTIONS],
            "rationale": {
                "h1_x_a": "jointly determines the multiscale path from the first scale",
                "a_x_h_min": "jointly determines total number of accessible scales",
                "c_H_x_h1": (
                    "jointly determines local robustness through the case-derived "
                    "loss width under a smoothing window"
                ),
            },
            "design_policy": (
                "Use a small structured subset rather than a full 5^4 grid; include "
                "center, one-factor low/high points, and a few balanced interaction corners."
            ),
        },
        "importance_analysis_plan": {
            "status": "planned_after_oat_or_joint_results",
            "dimensions": list(IRMF_PARAMETER_SENSITIVITY_IMPORTANCE_DIMENSIONS),
            "recommended_methods": [
                "standardized_effect_size_by_parameter",
                "variance_decomposition_on_declared_design",
                "rank_stability_by_parameter_level",
            ],
            "construct_specific_reporting": (
                "Report global importance and construct-specific importance separately; "
                "do not claim a universal parameter ranking if importance differs by regime."
            ),
        },
        "conclusion_robustness_plan": {
            "status": "planned_after_parameter_runs",
            "purpose": (
                "Assess whether headline IRMF conclusions persist across the "
                "declared parameter neighborhood rather than only at the locked "
                "parameter point."
            ),
            "comparison_targets": [
                "IRMF_vs_EMD",
                "IRMF_vs_EEMD",
                "IRMF_vs_CEEMDAN",
                "IRMF_overall_rank",
                "dimension_specific_ranks",
            ],
            "recommended_outputs": [
                "proportion_of_configurations_preserving_IRMF_best_overall_rank",
                "proportion_of_configurations_preserving_pairwise_effect_direction",
                "win_rate_stability_by_baseline",
                "dimension_rank_stability_by_configuration",
                "worst_case_rank_within_declared_neighborhood",
            ],
            "interpretation_policy": (
                "A parameter setting that performs better than the locked point "
                "must not replace the locked benchmark configuration; this analysis "
                "evaluates robustness, not post-hoc retuning."
            ),
        },
        "statistical_analysis_plan": {
            "status": "design_recorded_not_run",
            "ordinary_one_way_anova_role": (
                "not_primary; may be used only as a simple descriptive supplement "
                "for one-at-a-time curves"
            ),
            "primary_statistical_layers": {
                "level_1_descriptive_local_sensitivity": {
                    "purpose": "show local response of each metric to parameter perturbation",
                    "outputs": [
                        "median_response_curves",
                        "IQR_or_bootstrap_CI_bands",
                        "standardized_local_sensitivity_index",
                        "rank_stability_across_parameter_levels",
                    ],
                    "interpretation": (
                        "Answers whether the locked parameter point lies in a "
                        "stable neighborhood rather than a narrow needle optimum."
                    ),
                },
                "level_1b_conclusion_robustness": {
                    "purpose": (
                        "summarize whether comparative conclusions are preserved "
                        "across parameter configurations"
                    ),
                    "outputs": [
                        "rank_stability",
                        "win_rate_stability",
                        "paired_effect_sign_stability",
                        "configuration_preservation_fraction",
                    ],
                    "interpretation": (
                        "Answers whether IRMF superiority or trade-off conclusions "
                        "depend on a single precise parameter setting."
                    ),
                },
                "level_2_hierarchical_effect_decomposition": {
                    "purpose": "estimate parameter importance and selected interactions",
                    "preferred_model": (
                        "factorial or mixed-effects blocked model with parameter "
                        "effects as fixed effects and benchmark case/seed blocks "
                        "as repeated-structure controls"
                    ),
                    "conceptual_formula": (
                        "M_case,seed,theta = beta0 + f(h1,a,h_min,H) "
                        "+ u_case + u_seed + error"
                    ),
                    "primary_evidence": [
                        "standardized_effect_sizes",
                        "partial_variance_explained_or_partial_eta_squared",
                        "bootstrap_confidence_intervals",
                        "interaction_effect_sizes",
                    ],
                    "p_value_policy": (
                        "p-values are not primary evidence because large benchmark "
                        "sample sizes can make tiny effects statistically significant."
                    ),
                },
                "level_3_optional_global_sobol": {
                    "purpose": (
                        "variance-based global sensitivity over a declared plausible "
                        "four-parameter domain if computational budget allows"
                    ),
                    "outputs": [
                        "first_order_sobol_indices",
                        "total_sobol_indices",
                        "interaction_contribution_summary",
                    ],
                    "status": "optional_not_required_for_section6_core",
                },
            },
            "repeated_structure": {
                "blocking_units": [
                    "signal",
                    "noise",
                    "target_snr_db",
                    "seed",
                    "case_id",
                ],
                "rationale": (
                    "The same synthetic cases are re-evaluated under multiple IRMF "
                    "parameter settings, so observations must not be treated as "
                    "independent iid rows."
                ),
            },
            "metric_distribution_policy": {
                "heavy_tailed_metrics": [
                    "denoise_nmse",
                    "matched_component_nrmse",
                    "contaminated_region_nmse",
                    "clean_region_nmse",
                    "normalized_contamination_spillover_loss",
                ],
                "bounded_metrics": [
                    "denoise_corr",
                    "matched_component_corr",
                    "noise_capture_corr",
                    "signal_leakage_into_noise",
                ],
                "discrete_or_zero_inflated_metrics": [
                    "relative_decomposition_count_error",
                    "component_count_diagnostics",
                ],
                "recommended_handling": [
                    "medians_and_IQRs_for_skewed_metrics",
                    "rank_based_sensitivity_when_distributional_assumptions_fail",
                    "log_or_stabilizing_transform_only_when_formula_meaning_remains_clear",
                    "bootstrap_CI_for_effect_sizes",
                    "avoid_mechanical_Gaussian_ANOVA_for_all_13_primary_metrics",
                ],
            },
            "reporting_sentence": (
                "Parameter sensitivity was evaluated through local response analysis "
                "and hierarchical effect decomposition, with optional variance-based "
                "global sensitivity indices used to quantify parameter importance "
                "and interactions."
            ),
        },
        "scope": {
            "method": "IRMF_only",
            "not_a_recalibration": True,
            "not_a_per_case_tuning_protocol": True,
            "signals": list(PARAMETER_SENSITIVITY_SIGNALS),
            "noises": list(PARAMETER_SENSITIVITY_NOISES),
            "sigma_labels": list(PARAMETER_SENSITIVITY_SIGMAS),
            "target_snr_db_levels": list(PARAMETER_SENSITIVITY_TARGET_SNR_DB_LEVELS),
            "noise_severity_design": "target_snr_energy_ratio",
            "sigma_role": "snr_equivalent_label_not_fixed_noise_amplitude",
            "v530_alignment_status": "aligned_to_representative_target_snr_subset",
        },
        "freeze_conditions_before_execution": [
            "representative signal/noise/SNR subset frozen",
            "OAT grid frozen",
            "joint interaction subset frozen",
            "H included as a sensitivity parameter",
            "outputs grouped by primary evaluation dimension",
            "no final benchmark outcomes used to choose sensitivity grids",
        ],
        "created_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(spec, output_root / "v533_irmf_parameter_sensitivity_protocol.json")
    return spec
