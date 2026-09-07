#!/usr/bin/python
# coding: UTF-8

"""V5.34 executable protocols for compact Section 6 sensitivity analyses."""

from datetime import datetime, timezone

from project_config import (
    CONTAMINATION_LAMBDA_GRID,
    GLOBAL_CEEMDAN_PARAMS,
    GLOBAL_EEMD_PARAMS,
    GLOBAL_EMD_PARAMS,
    GLOBAL_IRMF_PARAMS,
    IRMF_PARAMETER_SENSITIVITY_OAT_FACTORS,
    TARGET_SNR_DB_LEVELS,
)
from experiments.paper_pipeline_utils import ensure_dir, write_json


V534_SECTION6_EXECUTABLE_PROTOCOL_VERSION = (
    "V5.34_section6_executable_protocols"
)

PRIMARY_ENDPOINTS_V528 = [
    "denoise_nmse",
    "denoise_corr",
    "matched_component_corr",
    "matched_component_nrmse",
    "relative_decomposition_count_error",
    "missing_true_component_energy_ratio",
    "spurious_mode_energy_ratio",
    "noise_capture_corr",
    "noise_energy_log_error",
    "signal_leakage_into_noise",
    "contaminated_region_nmse",
    "clean_region_nmse",
    "normalized_contamination_spillover_loss",
]

PRIMARY_CONSTRUCTS_V528 = [
    "Signal Recovery",
    "Component Recovery Quality",
    "Component-Set Fidelity",
    "Noise Separation",
    "Contamination Resistance",
]


def _module_template(
    section,
    title,
    scientific_question,
    factors_grid,
    sampling_unit,
    primary_statistics,
    stability_criterion,
    no_retuning_rule,
    outputs,
    claim_boundary,
):
    return {
        "section": section,
        "title": title,
        "scientific_question": scientific_question,
        "factors_grid": factors_grid,
        "sampling_unit": sampling_unit,
        "primary_statistics": primary_statistics,
        "stability_criterion": stability_criterion,
        "no_retuning_rule": no_retuning_rule,
        "outputs": outputs,
        "claim_boundary": claim_boundary,
    }


def run_v534_section6_executable_protocols(output_root):
    output_root = ensure_dir(output_root)
    locked_irmf = {
        "h1": float(GLOBAL_IRMF_PARAMS["h1"]),
        "a": float(GLOBAL_IRMF_PARAMS["a"]),
        "h_min": float(GLOBAL_IRMF_PARAMS["h_min"]),
        "c_H": float(GLOBAL_IRMF_PARAMS.get("c_H", GLOBAL_IRMF_PARAMS.get("H", 1.0))),
        "H_parameterization": "relative_H_case_equals_c_H_times_truth_free_scale_hat",
    }
    spec = {
        "schema_version": V534_SECTION6_EXECUTABLE_PROTOCOL_VERSION,
        "protocol_status": "executable_protocols_frozen_not_run",
        "benchmark_results_recomputed": False,
        "algorithm_cube_rerun_required": False,
        "governance_principle": {
            "main_text": "robustness of scientific conclusions",
            "supplement_protocol": "credibility of implementation and evaluation choices",
            "scope_status": "closed_no_new_section6_scope_expansion",
        },
        "shared_execution_rules": {
            "primary_endpoint_set": PRIMARY_ENDPOINTS_V528,
            "primary_constructs": PRIMARY_CONSTRUCTS_V528,
            "evidence_hierarchy": [
                "endpoint-level metric results are primary evidence",
                "construct-level interpretation is primary narrative",
                "overall ranks are secondary descriptive summaries",
            ],
            "common_reporting": [
                "median",
                "IQR",
                "paired effect size versus locked/reference setting",
                "bootstrap or paired confidence interval when applicable",
                "paired win/loss rate",
                "endpoint-level method ordering stability",
                "construct-level interpretation stability",
            ],
            "common_no_retuning_rule": (
                "Sensitivity results must not be used to replace locked "
                "parameters, choose new thresholds, select best comparator "
                "settings, or redefine primary claims after outcome inspection."
            ),
            "target_snr_levels_db": list(TARGET_SNR_DB_LEVELS),
            "clean_reference_role": (
                "clean baseline is a separate reference condition and is not "
                "averaged as another finite SNR level"
            ),
        },
        "executable_protocols": [
            _module_template(
                section="6.1",
                title="IRMF Parameter Sensitivity and Robustness",
                scientific_question=(
                    "Are the endpoint-level and construct-level IRMF conclusions "
                    "stable under pre-specified perturbations of the locked IRMF "
                    "parameters?"
                ),
                factors_grid={
                    "locked_parameter_point": locked_irmf,
                    "local_neighborhood": {
                        "parameters": ["h1", "a", "h_min", "c_H"],
                        "multiplicative_factors": list(IRMF_PARAMETER_SENSITIVITY_OAT_FACTORS),
                        "design": (
                            "one-at-a-time around the locked value; perturb c_H, "
                            "not absolute H"
                        ),
                    },
                    "targeted_interactions": {
                        "required_pairs": ["h1 x a", "a x h_min"],
                        "optional_pair": "c_H x contamination_severity",
                        "interaction_design": (
                            "small structured grid using low, locked, and high "
                            "levels; no full factorial all-parameter search"
                        ),
                    },
                    "analysis_layers": [
                        "local sensitivity",
                        "targeted interaction",
                        "endpoint robustness",
                        "construct robustness",
                    ],
                },
                sampling_unit={
                    "unit": "paired benchmark cell under a declared IRMF parameter configuration",
                    "comparison_reference": "locked IRMF parameter point",
                    "methods": ["IRMF sensitivity configurations", "locked EMD/EEMD/CEEMDAN baselines"],
                    "recommended_subset_policy": (
                        "use a frozen representative signal/noise/SNR/seed subset "
                        "for sensitivity cost control; do not choose subset based "
                        "on outcomes"
                    ),
                },
                primary_statistics=[
                    "endpoint median and IQR by parameter configuration",
                    "paired delta versus locked IRMF",
                    "normalized local sensitivity coefficient",
                    "targeted interaction effect size",
                    "paired IRMF-vs-baseline advantage retention rate",
                    "construct-level qualitative conclusion stability",
                ],
                stability_criterion={
                    "stable": [
                        "response curves vary smoothly around the locked point",
                        "endpoint-level best/worse interpretation is not reversed for key claims",
                        "paired IRMF-vs-baseline advantage direction is preserved for declared core constructs",
                    ],
                    "sensitive": [
                        "small local perturbation reverses a core endpoint conclusion",
                        "targeted interaction changes the construct-level interpretation",
                        "locked point appears as a narrow isolated optimum rather than a plateau or stable neighborhood",
                    ],
                    "interpretation_unit": "endpoint and construct first; overall rank secondary only",
                },
                no_retuning_rule=(
                    "The locked IRMF parameters remain fixed regardless of the "
                    "sensitivity outcome. This module cannot select a new h1, a, "
                    "h_min, c_H, or interaction-derived configuration."
                ),
                outputs=[
                    "irmf_parameter_local_response.csv",
                    "irmf_parameter_interaction_summary.csv",
                    "irmf_parameter_construct_robustness.csv",
                    "irmf_parameter_endpoint_delta.csv",
                    "irmf_parameter_sensitivity_dashboard.json",
                    "irmf_parameter_response_curves.png",
                    "irmf_parameter_interaction_heatmaps.png",
                ],
                claim_boundary=(
                    "Supports claims about local and mechanism-driven robustness "
                    "of the locked IRMF conclusions. Does not support claims that "
                    "the locked parameters are globally optimal."
                ),
            ),
            _module_template(
                section="6.2",
                title="Contamination Design Sensitivity",
                scientific_question=(
                    "Are contamination-resistance conclusions stable across "
                    "scientifically meaningful changes in contamination rate, "
                    "outlier magnitude, and contamination geometry?"
                ),
                factors_grid={
                    "contamination_family": "huber_contamination",
                    "rate_lambda_grid": list(CONTAMINATION_LAMBDA_GRID),
                    "outlier_magnitude_grid": [3.0, 5.0, 8.0, 10.0, 15.0],
                    "position_geometry_patterns": [
                        "random_dispersed",
                        "clustered_block",
                        "boundary",
                        "high_curvature_or_transient_aligned",
                        "smooth_region_aligned",
                        "peak_or_extremum_aligned",
                    ],
                    "small_interaction": {
                        "factors": ["outlier_magnitude", "position_geometry"],
                        "magnitudes": [5.0, 10.0, 15.0],
                        "positions": ["random_dispersed", "boundary", "clustered_block"],
                    },
                },
                sampling_unit={
                    "unit": "signal x contamination design x SNR x seed x method evaluation",
                    "pairing": "same signal, contamination mask, magnitude, SNR, and seed across all methods",
                    "methods": ["IRMF", "EMD", "EEMD", "CEEMDAN"],
                },
                primary_statistics=[
                    "contamination endpoint median and IQR by design factor",
                    "paired method effects within identical contamination masks",
                    "win/loss rate by contamination rate, magnitude, and geometry",
                    "small magnitude-by-position interaction summary",
                    "construct-level contamination-resistance stability",
                ],
                stability_criterion={
                    "stable": [
                        "main contamination-resistance ordering is preserved across pre-specified rates",
                        "conclusions do not depend on one outlier magnitude",
                        "no single geometry uniquely drives the main claim",
                    ],
                    "sensitive": [
                        "advantage appears only for one contamination geometry",
                        "high-magnitude or boundary contamination reverses the core conclusion",
                        "position-by-magnitude interaction contradicts the main contamination interpretation",
                    ],
                },
                no_retuning_rule=(
                    "Contamination factors are stress-test conditions, not tuning "
                    "settings. They cannot be used to select a preferred benchmark "
                    "contamination design after inspecting results."
                ),
                outputs=[
                    "contamination_design_metric_summary.csv",
                    "contamination_rate_sensitivity.csv",
                    "contamination_magnitude_sensitivity.csv",
                    "contamination_position_geometry_sensitivity.csv",
                    "contamination_magnitude_position_interaction.csv",
                    "contamination_design_dashboard.json",
                    "contamination_design_curves.png",
                    "contamination_geometry_heatmap.png",
                ],
                claim_boundary=(
                    "Supports claims that contamination-resistance conclusions "
                    "are not an artifact of one Huber rate, magnitude, or geometry. "
                    "Does not claim coverage of every possible artifact process."
                ),
            ),
            _module_template(
                section="6.3",
                title="Comparator and Signal Robustness",
                scientific_question=(
                    "Do the main conclusions persist across pre-specified signal "
                    "family variants and reasonable EMD-family baseline alternatives?"
                ),
                factors_grid={
                    "signal_variant_axis": {
                        "families": "canonical and challenging signal families with frozen within-family variants",
                        "design": "variant identities fixed before analysis",
                    },
                    "comparator_axis": {
                        "EMD": {
                            "baseline": dict(GLOBAL_EMD_PARAMS),
                            "reasonable_alternatives": "pre-specified boundary/sifting alternatives only",
                        },
                        "EEMD": {
                            "baseline": dict(GLOBAL_EEMD_PARAMS),
                            "reasonable_alternatives": "pre-specified trials/noise-width neighborhood only",
                        },
                        "CEEMDAN": {
                            "baseline": dict(GLOBAL_CEEMDAN_PARAMS),
                            "reasonable_alternatives": "pre-specified trials/epsilon neighborhood only",
                        },
                    },
                    "forbidden_design": [
                        "oracle comparator retuning",
                        "outcome-based comparator selection",
                        "dropping signal variants after outcome inspection",
                    ],
                },
                sampling_unit={
                    "unit": "signal-variant cell or comparator-alternative cell paired to the locked benchmark design",
                    "pairing": "same signal/noise/SNR/seed where applicable",
                    "methods": ["IRMF locked configuration", "locked and alternative EMD-family baselines"],
                },
                primary_statistics=[
                    "endpoint median and IQR by signal family/variant",
                    "paired method effects within signal variants",
                    "rank/order stability across variants",
                    "baseline-alternative sensitivity of EMD-family conclusions",
                    "construct-level conclusion stability",
                ],
                stability_criterion={
                    "stable": [
                        "core endpoint/construct conclusions recur across signal-family variants",
                        "reasonable EMD-family alternatives do not erase the declared main pattern",
                        "alternative comparator settings change magnitude more than qualitative interpretation",
                    ],
                    "sensitive": [
                        "main conclusion appears only in one signal variant",
                        "reasonable comparator alternative reverses a declared core claim",
                        "baseline sensitivity suggests unfair comparator under-specification",
                    ],
                },
                no_retuning_rule=(
                    "Comparator alternatives are robustness probes, not a second "
                    "calibration stage. The final comparator settings remain the "
                    "locked baseline settings."
                ),
                outputs=[
                    "signal_variant_endpoint_summary.csv",
                    "signal_variant_construct_stability.csv",
                    "comparator_alternative_endpoint_summary.csv",
                    "comparator_alternative_construct_stability.csv",
                    "comparator_signal_robustness_dashboard.json",
                    "signal_variant_stability_plot.png",
                    "comparator_sensitivity_plot.png",
                ],
                claim_boundary=(
                    "Supports fair-comparator and signal-specification robustness. "
                    "Does not imply that any alternative comparator configuration "
                    "should replace the locked baseline."
                ),
            ),
            _module_template(
                section="6.4",
                title="Computational Efficiency and Scaling",
                scientific_question=(
                    "What computational cost is associated with each locked "
                    "method, and how does runtime scale with sample size n?"
                ),
                factors_grid={
                    "methods": ["IRMF", "EMD", "EEMD", "CEEMDAN"],
                    "runtime_units": [
                        "seconds per method evaluation",
                        "seconds per signal/noise/SNR/seed cell",
                        "total wall-clock estimate for declared benchmark cube",
                    ],
                    "n_grid_for_scaling": [250, 500, 1000, 2000],
                    "hardware_environment_fields": [
                        "cpu_model",
                        "core_count",
                        "memory",
                        "python_version",
                        "package_versions",
                        "parallelization_policy",
                    ],
                },
                sampling_unit={
                    "unit": "method evaluation",
                    "aggregation": "method x n x benchmark-cell subset",
                    "timing_policy": "include decomposition and metric computation; report timeout/failure separately",
                },
                primary_statistics=[
                    "median runtime",
                    "IQR runtime",
                    "p90 and p95 runtime",
                    "timeout/failure rate",
                    "runtime scaling curve by n",
                    "estimated total runtime for the primary cube",
                ],
                stability_criterion={
                    "stable": [
                        "runtime summaries are reported with enough repetitions to avoid single-run timing noise",
                        "timeout/failure rates are explicitly separated from slow finite runs",
                    ],
                    "sensitive": [
                        "runtime estimates dominated by a small number of extreme cells",
                        "method failure/timeout rate changes interpretation of practical feasibility",
                    ],
                },
                no_retuning_rule=(
                    "Runtime results cannot be used to change method parameters "
                    "or omit slow methods from scientific comparisons after "
                    "performance outcomes are known."
                ),
                outputs=[
                    "runtime_by_method.csv",
                    "runtime_by_method_and_n.csv",
                    "runtime_timeout_failure_summary.csv",
                    "computational_environment_manifest.json",
                    "computational_scaling_dashboard.json",
                    "runtime_scaling_curve.png",
                    "runtime_distribution_by_method.png",
                ],
                claim_boundary=(
                    "Supports practical cost and scaling claims under the recorded "
                    "hardware/software environment. Does not establish universal "
                    "asymptotic complexity."
                ),
            ),
        ],
        "supplementary_protocol_audit_contract": {
            "tau_sensitivity": (
                "reconstruction-protocol sensitivity; not IRMF parameter tuning"
            ),
            "seed_convergence": "Monte Carlo sufficiency/provenance audit",
            "n_sensitivity": "normalized-time discretization robustness audit",
            "snr_realization": "data-generation calibration audit",
            "ranking_robustness": (
                "optional secondary audit only if average rank becomes a prominent claim"
            ),
        },
        "freeze_requirements_before_execution": [
            "all grids declared in this artifact",
            "representative subsets declared before outcome inspection",
            "hardware/environment manifest schema declared before runtime analysis",
            "no-retuning rule acknowledged for all modules",
            "main-text versus supplementary audit roles preserved",
        ],
        "created_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(
        spec,
        output_root / "v534_section6_executable_protocols.json",
    )
    return spec
