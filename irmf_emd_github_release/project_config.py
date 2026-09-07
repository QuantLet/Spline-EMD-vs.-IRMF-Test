#!/usr/bin/python
# coding: UTF-8

"""Project-wide benchmark configuration."""

from pathlib import Path

OUTPUT_ROOT_NAME = "IRMF_EMD_BENCHMARK_SUITE_RESULTS"

# Paper-oriented output root.  The original benchmark entry points are kept
# for backward compatibility; the V4E paper pipeline writes under a fresh root
# so earlier V1/V2/V3/V4B/V4C/V4D result folders are not overwritten.
PAPER_OUTPUT_ROOT_NAME = "IRMF_EMD_PAPER_RESULTS_V5_7_LOSS_REGIME_MAP"

SIGNAL_FAMILY = (
    "stationary_multi_sine",
    "chirp",
    "am_fm",
    "frequency_jump",
    "impulsive_transient",
    "intermittent_oscillation",
    "close_frequencies",
)

NOISE_FAMILY = (
    "gaussian",
    "laplace",
    "student_t",
    "impulsive",
    "burst",
    "huber_contamination",
    "ar1_colored",
    "heteroskedastic",
)

# Legacy three-level sigma slice retained for backward-compatible analyses.
# V5.18 main-text benchmark reporting uses FULL_SIGMA_LEVELS:
#     7 canonical signals x 8 noises x 5 sigma levels = 280 cases.
# Section 6.1 reuses the same five-level canonical cube for degradation
# trajectories, normalized robustness AUC, slopes, curvature, and failure rates.
SIGMA_LEVELS = (0.05, 0.10, 0.20)
FULL_SIGMA_LEVELS = (0.05, 0.10, 0.20, 0.30, 0.40)

# V5.30 target-SNR synthetic noise-severity design.  V5.29 briefly used
# legacy-equivalent SNR levels derived from the fixed-sigma grid, but no
# algorithm cube was generated under that design.  V5.30 replaces those inherited
# values with a regular 5-dB grid from extreme to near-clean conditions.
TARGET_SNR_DB_LEVELS = (
    0.0,
    5.0,
    10.0,
    15.0,
    20.0,
    25.0,
    30.0,
)
V529_SYNTHETIC_NOISE_DESIGN_VERSION = "V5.30_regular_5db_target_snr_noise_severity_design"
V529_SUPERSEDED_GRID_NOTE = (
    "The initially frozen V5.29 legacy-equivalent SNR grid was superseded "
    "before algorithm-cube generation; no V5.29 benchmark results were produced."
)
V530_BENCHMARK_SEEDS = tuple(range(20))
V530_CONVERGENCE_AUDIT_SEEDS = tuple(range(50))
V530_SEED_POLICY_VERSION = "V5.30_20_seed_primary_monte_carlo_design"
V530_SEED_CONVERGENCE_PREFIXES = (10, 20, 30, 50)
V530_SEED_CONVERGENCE_AUDIT_VERSION = "V5.30_seed_convergence_audit_10_20_30_50_prefixes"

HUBER_CONTAMINATION_DEFAULT_LAMBDA = 0.05
HUBER_CONTAMINATION_OUTLIER_SCALE = 10.0
HUBER_LAMBDA_GRID = (0.00, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20)

DEFAULT_N = 500
DEFAULT_FS = 500.0
DEFAULT_SEED = 0
DEFAULT_SEARCH_MODE = "quick"
DEFAULT_MONTE_CARLO_TRIALS = 20
DEFAULT_METHOD_TIMEOUT_SECONDS = 120
# V5.23 synthetic benchmark uses the seed factor as Monte Carlo replication.
# These are data-generation seeds; EEMD/CEEMDAN algorithm seeds are derived
# separately inside the unified-cube runner.
UNIFIED_BENCHMARK_SEEDS = tuple(range(DEFAULT_MONTE_CARLO_TRIALS))
UNIFIED_BENCHMARK_QUICK_SEEDS = (0,)

# Global fixed IRMF configuration selected by the development-set protocol.
# Replace these values with the result from Section 4 once parameter selection
# has been run.  They are intentionally centralized so the main paper
# benchmark never performs per-case tuning.
GLOBAL_IRMF_PARAMS = {
    "h1": 0.18,
    "a": 1.4142135623730951,
    "h_min": 0.012,
    "H": 1.00,
    "boundary_mode": "periodic",
    "min_support_points": 3,
}

# Fixed EMD configuration for the main benchmark.
GLOBAL_EMD_PARAMS = {
    "nbsym": 2,
    "spline_kind": "cubic",
    "max_imf": -1,
    "std_thr": None,
    "svar_thr": None,
    "total_power_thr": None,
    "range_thr": None,
}

# Section 4: global parameter selection uses a development set only.  The
# development signals intentionally use separate waveform instances from the
# Section 5 canonical benchmark signals: class-overlap is allowed, but exact
# formula-instance reuse is avoided.  V5.47 aligns the development severity
# semantics with the V5.30 target-SNR benchmark while keeping the same signals,
# noise families, objective, and two-stage search governance:
#     5 development signals × 4 representative noises × 3 SNR levels = 60 cases.
PARAMETER_SELECTION_SIGNALS = (
    "stationary_multi_sine_dev",
    "chirp_dev",
    "am_fm_dev",
    "impulsive_transient_dev",
    "close_frequencies_dev",
)
PARAMETER_SELECTION_NOISES = (
    "gaussian",
    "student_t",
    "impulsive",
    "ar1_colored",
)
PARAMETER_SELECTION_SIGMAS = (0.10, 0.20)
PARAMETER_SELECTION_TARGET_SNR_DB_LEVELS = (5.0, 15.0, 25.0)
PARAMETER_SELECTION_SEVERITY_DESIGN_VERSION = "V5.47_target_snr_development_severity_revision"

# Section 4 two-stage coarse-to-fine global parameter selection.  Stage I
# uses an expanded plausible domain because the balanced-score V3 search selected
# several boundary values, but keeps the geometric scale parameter a in the
# Spokoiny-consistent range (1, 2].  V5.39 promotes the Gaussian-smoothed median
# loss width H from a fixed methodology constant to an explicitly tunable
# development-set hyperparameter because H controls the robust contrast scale.
# Stage II is generated deterministically around the Stage-I best configuration
# using the refinement steps below.
PAPER_IRMF_COARSE_PARAMETER_GRID = {
    "h1_options": (0.08, 0.12, 0.16, 0.20, 0.24, 0.28),
    "a_options": (1.4142135623730951, 1.60, 1.80, 2.00),
    "h_min_options": (0.004, 0.006, 0.010, 0.014, 0.018, 0.022),
    "H_options": (0.50, 0.75, 1.00, 1.25, 1.50),
}
PAPER_IRMF_REFINEMENT_STEPS = {
    "h1": 0.02,
    "a": 0.10,
    "h_min": 0.002,
    "H": 0.125,
    "c_H": 0.125,
}

# V5.42 proposed relative-H parameterization.  These constants do not activate
# the algorithm layer by themselves; they define the candidate scale-equivariant
# replacement for the V5.39 absolute-H grid.
IRMF_RELATIVE_H_PARAMETERIZATION_VERSION = "V5.42_relative_noise_scale_H_parameterization"
IRMF_RELATIVE_H_SCALE_ESTIMATOR = "first_difference_mad_truth_free"
IRMF_RELATIVE_H_C_OPTIONS = (0.50, 0.75, 1.00, 1.25, 1.50)
IRMF_RELATIVE_H_C_REFINEMENT_STEP = 0.125

# Backward-compatible alias for older scripts that expect a single grid name.
PAPER_IRMF_PARAMETER_GRID = PAPER_IRMF_COARSE_PARAMETER_GRID

# 4.2 Comprehensive Signal-Noise Benchmark
COMPREHENSIVE_SIGNAL_FAMILY = SIGNAL_FAMILY
COMPREHENSIVE_NOISE_FAMILY = NOISE_FAMILY
COMPREHENSIVE_SIGMA_LEVELS = SIGMA_LEVELS

# 4.3 Monte Carlo Robustness
MONTE_CARLO_SIGNAL_FAMILY = SIGNAL_FAMILY
MONTE_CARLO_NOISE_FAMILY = NOISE_FAMILY
MONTE_CARLO_SIGMA_LEVELS = SIGMA_LEVELS

# 4.4 EMD Family Benchmark
EMD_FAMILY_METHODS = ("IRMF", "EMD", "EEMD", "CEEMDAN")
EMD_FAMILY_BENCHMARK_MODES = {
    "representative": {
        "signals": SIGNAL_FAMILY,
        "noises": ("gaussian", "laplace", "impulsive", "huber_contamination"),
        "sigmas": (0.20,),
    },
    "full": {
        "signals": SIGNAL_FAMILY,
        "noises": NOISE_FAMILY,
        "sigmas": SIGMA_LEVELS,
    },
}

# V5.18 unified nested factorial benchmark cube.  The cube is generated once
# and then sliced into the known-truth synthetic benchmark section.
UNIFIED_BENCHMARK_REGIMES = {
    "canonical": {
        "signals": SIGNAL_FAMILY,
        "noises": NOISE_FAMILY,
        "sigmas": FULL_SIGMA_LEVELS,
        "paper_slices": {
            "section_5_main": {"sigma_max": 0.40},
            "section_6_1_noise_robustness": {"sigma_max": 0.40},
        },
    },
    "challenging": {
        "signals": (
            "crossing_chirps",
            "time_varying_close_frequencies",
            "piecewise_am_fm_discontinuity",
            "damped_oscillation",
            "trend_plus_oscillation",
            "buried_weak_component",
            "non_sinusoidal_periodic",
            "transient_train",
        ),
        "noises": NOISE_FAMILY,
        "sigmas": FULL_SIGMA_LEVELS,
        "paper_slices": {
            "synthetic_challenging_generalization": {"sigma_max": 0.40},
        },
    },
}
EMD_FAMILY_SENSITIVITY_METHODS = ("EEMD", "CEEMDAN")
EMD_FAMILY_SENSITIVITY_TRIALS = (50, 100)
EMD_FAMILY_SENSITIVITY_NOISE_WIDTHS = (0.03, 0.05, 0.10)
EEMD_PARAMETER_SELECTION_GRID = {
    "trials_options": (50, 100, 200),
    "noise_width_options": (0.02, 0.05, 0.10, 0.20),
}
CEEMDAN_PARAMETER_SELECTION_GRID = {
    "trials_options": (50, 100, 200),
    "epsilon_options": (0.0025, 0.005, 0.010, 0.020),
}
EMD_PARAMETER_SELECTION_GRID = {
    "nbsym_options": (2, 4),
    "spline_kind_options": ("cubic", "akima"),
    "max_imf_options": (-1, 5),
    "std_thr_options": (None, 0.05, 0.10),
    "svar_thr_options": (None,),
    "total_power_thr_options": (None,),
    "range_thr_options": (None,),
}
GLOBAL_EEMD_PARAMS = {
    "trials": 100,
    "noise_width": 0.05,
    "max_imf": -1,
    "parallel": False,
}
GLOBAL_CEEMDAN_PARAMS = {
    "trials": 100,
    "epsilon": 0.005,
    "max_imf": -1,
    "parallel": False,
}

# V5.24 paper-facing evaluation framework.  The legacy case_score remains in
# result files for sensitivity checks, but it is no longer the primary endpoint
# for method ranking or statistical inference.
EVALUATION_METHODS = ("IRMF", "EMD", "EEMD", "CEEMDAN")
EVALUATION_FRAMEWORK_VERSION = "V5.24_true_component_mixing_primary"
EVALUATION_FRAMEWORK_STATUS = "frozen"
EVALUATION_FRAMEWORK_FROZEN_DATE = "2026-07-26"
EVALUATION_DIMENSION_ESTIMANDS = {
    "reconstruction": (
        "Distance between the reconstructed clean signal estimate and the "
        "known clean reference signal."
    ),
    "structural_fidelity": (
        "Recovery of true oscillatory components and true-component mixing "
        "behavior in synthetic and semi-synthetic settings with known "
        "component references."
    ),
    "contamination_resistance": (
        "Ability to isolate stochastic noise or contamination while preserving "
        "informative signal content."
    ),
}
DECOMPOSITION_SEMANTICS_BY_METHOD = {
    "IRMF": {
        "decomposition_direction": "top_down",
        "component_order_semantics": (
            "earlier components tend to represent larger-scale / lower-frequency "
            "structure; later residuals carry finer-scale noise or outliers"
        ),
    },
    "EMD": {
        "decomposition_direction": "bottom_up",
        "component_order_semantics": (
            "earlier IMFs tend to represent higher-frequency oscillations; final "
            "residual often carries low-frequency trend"
        ),
    },
    "EEMD": {
        "decomposition_direction": "bottom_up_noise_assisted",
        "component_order_semantics": (
            "ensemble EMD components retain EMD-family high-to-low extraction "
            "semantics"
        ),
    },
    "CEEMDAN": {
        "decomposition_direction": "bottom_up_adaptive_noise_assisted",
        "component_order_semantics": (
            "complete ensemble EMD components retain EMD-family high-to-low "
            "extraction semantics"
        ),
    },
}
COMPONENT_ALIGNMENT_POLICY = {
    "primary_synthetic_component_recovery": (
        "permutation-invariant Hungarian matching between estimated components "
        "and true components using absolute component correlation; extraction "
        "index is never used as the matching rule"
    ),
    "scale_semantics": (
        "IRMF and EMD-family methods may extract in opposite directions, so "
        "component-level comparisons are interpreted after scale/component "
        "alignment rather than by raw component index"
    ),
    "real_data_policy": (
        "real-data component comparisons should use method-agnostic spectral "
        "scale summaries such as spectral concentration, instantaneous-frequency "
        "continuity, and frequency-band separation; real data should not use "
        "ground-truth component recovery metrics"
    ),
}
PRIMARY_ENDPOINTS_BY_DIMENSION = {
    "reconstruction": {
        "subdimensions": {
            "reconstruction_accuracy": ("denoise_nmse", "denoise_corr"),
        },
    },
    "structural_fidelity": {
        "subdimensions": {
            "component_recovery": ("imf_recovery_corr", "imf_recovery_nrmse"),
            "true_component_mixing": ("component_splitting_index", "component_merging_index"),
        },
    },
    "contamination_resistance": {
        "subdimensions": {
            "noise_capture": ("noise_capture_corr",),
            "signal_preservation": ("signal_leakage_into_noise",),
        },
    },
}
CONDITIONAL_PRIMARY_ENDPOINTS_BY_DIMENSION = {
    "contamination_resistance": {
        "applicable_noises": ("impulsive", "burst", "huber_contamination"),
        "subdimensions": {
            "outlier_resistance": ("outlier_resistance_index",),
            "clean_region_preservation": ("clean_region_nmse",),
        },
        "role": (
            "contamination-specific primary endpoints evaluated only for "
            "contamination-aware benchmark regimes where contamination masks "
            "or clean-region annotations are available; excluded from primary "
            "statistical comparisons for non-contamination cases."
        ),
    },
}
CONDITIONAL_ENDPOINTS_BY_DIMENSION = {
    "contamination_resistance": {
        "applicable_noises": ("impulsive", "burst", "huber_contamination"),
        "subdimensions": {
            "contaminated_region_performance": ("contaminated_region_nmse",),
            "spillover_control": ("contamination_spillover_error",),
        },
        "role": (
            "secondary conditional endpoints for noise models with explicit "
            "contamination or outlier-region labels; outlier_resistance_index "
            "and clean_region_nmse are primary only on their applicable labeled "
            "contamination regimes."
        ),
    },
}
DIAGNOSTIC_METRICS_BY_DIMENSION = {
    "reconstruction": (
        "denoise_psnr",
        "spectral_corr",
        "input_snr_db",
        "output_snr_db",
        "snr_gain_db",
    ),
    "structural_fidelity": (
        "metric_definition_version",
        "structural_metric_schema_version",
        "true_component_metrics_available",
        "inter_imf_entanglement_index",
        "mode_mixing_index",
        "true_component_splitting_max",
        "estimated_component_merging_max",
        "missing_true_component_count",
        "missing_true_component_fraction",
        "missing_true_component_rate",
        "missing_true_component_energy_ratio",
        "unmatched_estimated_component_count",
        "unmatched_estimated_component_fraction",
        "unmatched_estimated_component_rate",
        "spurious_mode_energy_ratio",
        "component_mixing_association_threshold",
        "component_mixing_energy_threshold",
        "component_mixing_allocation_floor",
        "decomposition_count_error",
        "decomposition_adequacy_score",
        "strict_io",
        "spectral_leakage",
        "frequency_overlap_max_offdiag",
        "frequency_separation_score",
        "over_decomposition_penalty",
        "under_decomposition_index",
        "imf_count",
        "effective_imf_count",
        "relative_decomposition_count_error",
        "transient_smearing_index",
        "transient_preservation_score",
    ),
    "contamination_resistance": (
        "contamination_resistance_score",
        "robustness_score",
        "contaminated_region_preservation_score",
        "contamination_spillover_score",
        "signal_leakage_into_noise_score",
        "clean_region_signal_leakage",
        "clean_region_signal_leakage_score",
        "signal_leakage_degenerate_flag",
        "noise_capture_energy_ratio",
    ),
}
LOWER_IS_BETTER_METRICS = (
    "denoise_nmse",
    "imf_recovery_nrmse",
    "matched_component_nrmse",
    "component_splitting_index",
    "component_merging_index",
    "inter_imf_entanglement_index",
    "mode_mixing_index",
    "decomposition_count_error",
    "relative_decomposition_count_error",
    "missing_true_component_energy_ratio",
    "spurious_mode_energy_ratio",
    "noise_energy_log_error",
    "clean_region_nmse",
    "contaminated_region_nmse",
    "contamination_spillover_error",
    "normalized_contamination_spillover_loss",
    "signal_leakage_into_noise",
    "clean_region_signal_leakage",
    "spectral_leakage",
    "strict_io",
    "frequency_overlap_max_offdiag",
    "over_decomposition_penalty",
    "under_decomposition_index",
)

# V5.25 publication-facing taxonomy cleanup.  This does not change the frozen
# V5.24 primary endpoints or benchmark results.  It separates scientific
# diagnostics from metadata, legacy aliases, and derived duplicate fields.
CANONICAL_DIAGNOSTIC_TAXONOMY_VERSION = "V5.25_publication_taxonomy_cleanup"
CANONICAL_DIAGNOSTIC_TAXONOMY_STATUS = "publication_layer_only"
CANONICAL_DIAGNOSTIC_TAXONOMY_PROVENANCE = {
    "benchmark_source_version": "V5.24",
    "taxonomy_layer_version": "V5.25",
    "benchmark_results_recomputed": False,
}
CANONICAL_DIAGNOSTIC_TAXONOMY_BY_GROUP = {
    "cardinality_diagnostics": (
        "imf_count",
        "effective_imf_count",
        "relative_decomposition_count_error",
        "over_decomposition_penalty",
        "under_decomposition_index",
    ),
    "matching_allocation_diagnostics": (
        "true_component_splitting_max",
        "estimated_component_merging_max",
        "missing_true_component_count",
        "missing_true_component_energy_ratio",
        "unmatched_estimated_component_count",
        "spurious_mode_energy_ratio",
    ),
    "inter_component_separation_diagnostics": (
        "inter_imf_entanglement_index",
        "strict_io",
        "spectral_leakage",
        "frequency_overlap_max_offdiag",
        "frequency_separation_score",
    ),
    "signal_specific_diagnostics": (
        "transient_smearing_index",
        "transient_preservation_score",
    ),
    "secondary_noise_separation_diagnostics": (
        "noise_energy_ratio",
        "noise_energy_log_error",
        "remaining_noise_energy_ratio",
        "noise_energy_bias_class",
    ),
}
PROTOCOL_METADATA_FIELDS = (
    "metric_definition_version",
    "evaluation_framework_version",
    "evaluation_framework_status",
    "evaluation_framework_frozen_date",
    "structural_metric_schema_version",
    "true_component_metrics_available",
    "component_mixing_association_threshold",
    "component_mixing_energy_threshold",
    "component_mixing_allocation_floor",
    "component_mixing_basis",
    "component_mixing_index_formula",
)
LEGACY_ALIAS_FIELDS = {
    "mode_mixing_index": "inter_imf_entanglement_index",
    "mode_mixing_index_legacy": "inter_imf_entanglement_index",
}
DERIVED_DUPLICATE_FIELDS = {
    "decomposition_count_error": (
        "relative_decomposition_count_error; V5.24 code already normalizes "
        "abs(effective_imf_count - true_component_count) by true_component_count"
    ),
    "missing_true_component_fraction": "missing_true_component_count / true_component_count",
    "missing_true_component_rate": "missing_true_component_fraction",
    "unmatched_estimated_component_fraction": (
        "unmatched_estimated_component_count / active_estimated_component_count"
    ),
    "unmatched_estimated_component_rate": "unmatched_estimated_component_fraction",
    "signal_leakage_into_noise_score": "1 - signal_leakage_into_noise",
    "clean_region_signal_leakage_score": "1 - clean_region_signal_leakage",
    "contaminated_region_preservation_score": "monotone score transform of contaminated_region_nmse",
    "contamination_spillover_score": "monotone score transform of contamination_spillover_error",
    "clean_region_preservation_score": "monotone score transform of clean_region_nmse",
    "noise_capture_energy_ratio": "legacy noise-energy diagnostic; superseded by noise_energy_ratio",
}
PUBLICATION_COMPONENT_RECOVERY_NAMES = {
    "imf_recovery_corr": "matched_component_corr",
    "imf_recovery_nrmse": "matched_component_nrmse",
}
COMPONENT_RECOVERY_MATCHING_AUDIT = {
    "v5_24_code_uses_hungarian": True,
    "alignment_rule_field": "imf_recovery_alignment_rule",
    "assignment_pairs_field": "imf_recovery_assignment_pairs",
    "publication_layer_interpretation": (
        "V5.25 formalizes the canonical interpretation of the existing "
        "component-recovery metrics as permutation-invariant matched-component "
        "evaluation. It does not silently reprocess V5.24 results."
    ),
}
RELATIVE_DECOMPOSITION_COUNT_ERROR_DEFINITION = {
    "field": "relative_decomposition_count_error",
    "abbreviation": "RCCE",
    "formula": "abs(K_eff_est - K_true) / K_true",
    "truth_available_only": True,
    "na_policy": "NA when true_component_metrics_available is false or K_true is unavailable/nonpositive",
    "count_source": "effective_imf_count",
    "publication_role": "structural cardinality diagnostic, not a primary endpoint",
}

# V5.26 primary-schema revision draft.  This is a metric-architecture revision
# that must be qualified and frozen before confirmatory V5.26 conclusions.
V526_PRIMARY_SCHEMA_VERSION = "V5.26_primary_schema_revision"
V526_PRIMARY_SCHEMA_STATUS = "draft_requires_qualification"
V526_VERSION_BOUNDARY = {
    "V5.24": (
        "Completed benchmark implementation under the previous primary-metric "
        "schema."
    ),
    "V5.25": (
        "Metric-taxonomy cleanup and publication-layer formalization; no "
        "change to benchmark algorithms or numerical results."
    ),
    "V5.26": (
        "Revised primary-metric architecture adding Component-Set Fidelity as "
        "a primary evaluation dimension; requires protocol qualification, "
        "schema freeze, and regeneration of final statistical analysis under "
        "the revised schema."
    ),
}
V526_PRIMARY_ENDPOINTS_BY_DIMENSION = {
    "signal_recovery": ("denoise_nmse", "denoise_corr"),
    "component_recovery_quality": (
        "matched_component_corr",
        "matched_component_nrmse",
    ),
    "component_set_fidelity": (
        "relative_decomposition_count_error",
        "missing_true_component_energy_ratio",
        "spurious_mode_energy_ratio",
    ),
    "noise_separation": ("noise_capture_corr", "signal_leakage_into_noise"),
    "contamination_resistance": ("outlier_resistance_index", "clean_region_nmse"),
}
V526_PRIMARY_METRIC_CODE_FIELD_MAP = {
    "denoise_nmse": "denoise_nmse",
    "denoise_corr": "denoise_corr",
    "matched_component_corr": "imf_recovery_corr",
    "matched_component_nrmse": "imf_recovery_nrmse",
    "relative_decomposition_count_error": "decomposition_count_error",
    "missing_true_component_energy_ratio": "missing_true_component_energy_ratio",
    "spurious_mode_energy_ratio": "spurious_mode_energy_ratio",
    "noise_capture_corr": "noise_capture_corr",
    "signal_leakage_into_noise": "signal_leakage_into_noise",
    "outlier_resistance_index": "outlier_resistance_index",
    "clean_region_nmse": "clean_region_nmse",
}
V526_COMPONENT_SET_FIDELITY_CONSTRUCTS = {
    "relative_decomposition_count_error": "cardinality_consistency",
    "missing_true_component_energy_ratio": "completeness_of_recovered_truth_content",
    "spurious_mode_energy_ratio": "purity_of_estimated_component_set",
}
V526_METHOD_NEUTRALITY_RULE = (
    "Matching and set-fidelity definitions must depend only on observable "
    "component outputs and ground-truth references, not on method-specific "
    "internal labels, extraction order, stopping rules, or IMF semantics."
)

# V5.27 primary-schema draft extends the same construct architecture to noise
# separation and contamination resistance.  It is a separate draft from V5.26
# and requires targeted qualification before any freeze.
V527_PRIMARY_SCHEMA_VERSION = "V5.27_noise_contamination_construct_revision"
V527_PRIMARY_SCHEMA_STATUS = "draft_requires_qualification"
V527_VERSION_BOUNDARY = {
    "V5.26": (
        "Component-Set Fidelity primary schema revision; freeze candidate."
    ),
    "V5.27": (
        "Noise Separation and Contamination Resistance construct-level "
        "redesign; requires targeted construct qualification before freeze."
    ),
}
V527_PRIMARY_ENDPOINTS_BY_DIMENSION = {
    "signal_recovery": ("denoise_nmse", "denoise_corr"),
    "component_recovery_quality": (
        "matched_component_corr",
        "matched_component_nrmse",
    ),
    "component_set_fidelity": (
        "relative_decomposition_count_error",
        "missing_true_component_energy_ratio",
        "spurious_mode_energy_ratio",
    ),
    "noise_separation": (
        "noise_capture_corr",
        "noise_energy_log_error",
        "signal_leakage_into_noise",
    ),
    "contamination_resistance": (
        "contaminated_region_nmse",
        "clean_region_nmse",
        "contamination_spillover_error",
    ),
}
V527_PRIMARY_METRIC_CODE_FIELD_MAP = {
    **V526_PRIMARY_METRIC_CODE_FIELD_MAP,
    "noise_energy_log_error": "noise_energy_log_error",
    "contaminated_region_nmse": "contaminated_region_nmse",
    "contamination_spillover_error": "contamination_spillover_error",
}
V527_NOISE_SEPARATION_CONSTRUCTS = {
    "noise_capture_corr": "noise_identity",
    "noise_energy_log_error": "noise_magnitude_recovery",
    "signal_leakage_into_noise": "signal_purity_of_estimated_noise",
}
V527_CONTAMINATION_RESISTANCE_CONSTRUCTS = {
    "contaminated_region_nmse": "direct_contamination_recovery",
    "clean_region_nmse": "clean_region_preservation",
    "contamination_spillover_error": "spillover_localization",
}
V527_DIAGNOSTIC_DEMOTIONS = {
    "outlier_resistance_index": (
        "demoted_to_diagnostic_or_composite_summary_pending_formula_audit; "
        "not primary in V5.27 draft"
    ),
    "noise_energy_ratio": "directional noise-energy diagnostic",
    "remaining_noise_energy_ratio": "secondary suppression-effectiveness diagnostic",
    "noise_energy_bias_class": "categorical secondary noise-energy diagnostic",
}

# V5.28 is a narrow primary-schema revision of V5.27. It keeps the same
# construct architecture but replaces raw local spillover MSE with a
# dimensionless normalized spillover loss derived from the existing
# contamination_spillover_score. This is a statistical-layer schema revision:
# no algorithm cube rerun is required.
V528_PRIMARY_SCHEMA_VERSION = "V5.28_normalized_spillover_primary_revision"
V528_PRIMARY_SCHEMA_STATUS = "draft_requires_qualification"
V528_VERSION_BOUNDARY = {
    "V5.27": (
        "Frozen construct-level schema using raw contamination_spillover_error; "
        "within-cell ranks are valid but cross-case absolute means are scale-dependent."
    ),
    "V5.28": (
        "Replaces raw spillover primary endpoint with normalized_contamination_spillover_loss; "
        "requires scale audit and statistics regeneration, but no algorithm rerun."
    ),
}
V528_PRIMARY_ENDPOINTS_BY_DIMENSION = {
    **{
        key: value for key, value in V527_PRIMARY_ENDPOINTS_BY_DIMENSION.items()
        if key != "contamination_resistance"
    },
    "contamination_resistance": (
        "contaminated_region_nmse",
        "clean_region_nmse",
        "normalized_contamination_spillover_loss",
    ),
}
V528_PRIMARY_METRIC_CODE_FIELD_MAP = {
    **V527_PRIMARY_METRIC_CODE_FIELD_MAP,
    "normalized_contamination_spillover_loss": "contamination_spillover_score",
}
V528_CONTAMINATION_RESISTANCE_CONSTRUCTS = {
    **{
        key: value for key, value in V527_CONTAMINATION_RESISTANCE_CONSTRUCTS.items()
        if key != "contamination_spillover_error"
    },
    "normalized_contamination_spillover_loss": "spillover_localization",
}
V528_DIAGNOSTIC_DEMOTIONS = {
    **V527_DIAGNOSTIC_DEMOTIONS,
    "contamination_spillover_error": (
        "raw local MSE retained as diagnostic; replaced as primary by "
        "normalized_contamination_spillover_loss in V5.28"
    ),
}
COMPOSITE_SENSITIVITY_ENDPOINTS = ("case_score", "case_score_final")
CASE_SCORE_ROLE = "composite_sensitivity_endpoint_only"

# V5.24 metric-provenance policy.  Qualification effort is allocated by metric
# provenance rather than by applying an identical qualification protocol to
# every endpoint.
METRIC_QUALIFICATION_PRINCIPLE = (
    "The scope of the full evaluation metric qualification is intentionally "
    "limited to newly proposed structural metrics. Established metrics are "
    "adopted directly from prior literature, while derived metrics undergo "
    "only basic construct-oriented verification appropriate to their "
    "adaptation. Accordingly, the qualification effort is allocated according "
    "to metric provenance rather than applying an identical qualification "
    "protocol to every evaluation metric."
)
METRIC_PROVENANCE_QUALIFICATION_APPROACH = (
    {
        "metric_provenance": "Established metrics",
        "examples": ("denoise_nmse", "denoise_corr"),
        "qualification_approach": "Standard definitions; cite prior literature",
        "rationale": (
            "Widely established metrics with well-understood mathematical "
            "properties; no project-specific qualification required."
        ),
    },
    {
        "metric_provenance": "Derived / adapted metrics",
        "examples": (
            "imf_recovery_corr",
            "clean_region_nmse",
            "noise_capture_corr",
            "outlier_resistance_index",
        ),
        "qualification_approach": "Basic construct-oriented verification",
        "rationale": (
            "Derived from established concepts or adapted to the present "
            "benchmark; only basic behavioral verification appropriate to the "
            "metric adaptation is performed."
        ),
    },
    {
        "metric_provenance": "Novel structural metrics",
        "examples": ("component_splitting_index", "component_merging_index"),
        "qualification_approach": "Full evaluation metric qualification",
        "rationale": (
            "Newly proposed structural metrics; therefore subjected to "
            "construct validation, dependency audit, incremental-validity "
            "review, scientific adjudication, and schema freeze before "
            "benchmark use."
        ),
    },
)
PRIMARY_METRIC_PROVENANCE = {
    "denoise_nmse": "Established metrics",
    "denoise_corr": "Established metrics",
    "imf_recovery_corr": "Derived / adapted metrics",
    "imf_recovery_nrmse": "Derived / adapted metrics",
    "noise_capture_corr": "Derived / adapted metrics",
    "signal_leakage_into_noise": "Derived / adapted metrics",
    "outlier_resistance_index": "Derived / adapted metrics",
    "clean_region_nmse": "Derived / adapted metrics",
    "component_splitting_index": "Novel structural metrics",
    "component_merging_index": "Novel structural metrics",
}
PRIMARY_METRIC_QUALIFICATION_SCOPE = {
    "denoise_nmse": "standard_definition",
    "denoise_corr": "standard_definition",
    "imf_recovery_corr": "basic_construct_oriented_verification",
    "imf_recovery_nrmse": "basic_construct_oriented_verification",
    "noise_capture_corr": "basic_construct_oriented_verification",
    "signal_leakage_into_noise": "basic_construct_oriented_verification",
    "outlier_resistance_index": "basic_construct_oriented_verification",
    "clean_region_nmse": "basic_construct_oriented_verification",
    "component_splitting_index": "full_structural_metric_qualification",
    "component_merging_index": "full_structural_metric_qualification",
}

# 4.5 Boundary Effects
BOUNDARY_METHODS = (
    "IRMF-periodic",
    "IRMF-mirror",
    "EMD-nbsym2",
    "EMD-nbsym4",
)
BOUNDARY_BENCHMARK_MODES = {
    "representative": {
        "signals": SIGNAL_FAMILY,
        "noises": ("gaussian", "laplace", "impulsive", "huber_contamination"),
        "sigmas": (0.20,),
    },
    "full": {
        "signals": SIGNAL_FAMILY,
        "noises": NOISE_FAMILY,
        "sigmas": SIGMA_LEVELS,
    },
}

# Section 6 target-SNR severity grids.  These supersede the legacy fixed-sigma
# Section 6 stress-test grids so sensitivity analyses share the V5.30
# target-SNR severity semantics.
SECTION6_NOISE_ROBUSTNESS_TARGET_SNR_DB_LEVELS = TARGET_SNR_DB_LEVELS
SECTION6_REPRESENTATIVE_TARGET_SNR_DB_LEVELS = (5.0, 15.0, 25.0)
SECTION6_SINGLE_TARGET_SNR_DB = 15.0

# Section 6.1: robustness under increasing noise severity.
NOISE_ROBUSTNESS_SIGNALS = SIGNAL_FAMILY
NOISE_ROBUSTNESS_NOISES = NOISE_FAMILY
NOISE_ROBUSTNESS_SIGMAS = tuple(10.0 ** (-snr / 20.0) for snr in SECTION6_NOISE_ROBUSTNESS_TARGET_SNR_DB_LEVELS)
NOISE_ROBUSTNESS_TARGET_SNR_DB_LEVELS = SECTION6_NOISE_ROBUSTNESS_TARGET_SNR_DB_LEVELS

# Section 6.2: Huber contamination-rate stress test.
CONTAMINATION_ROBUSTNESS_SIGNALS = (
    "stationary_multi_sine",
    "frequency_jump",
    "impulsive_transient",
    "intermittent_oscillation",
)
CONTAMINATION_ROBUSTNESS_SIGMA = 10.0 ** (-SECTION6_SINGLE_TARGET_SNR_DB / 20.0)
CONTAMINATION_ROBUSTNESS_TARGET_SNR_DB = SECTION6_SINGLE_TARGET_SNR_DB
CONTAMINATION_LAMBDA_GRID = (0.00, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20)

# Section 6.1 factorial sensitivity constants.  Under the current relative-H
# parameterization, the robust-loss scale is perturbed through the global,
# dimensionless c_H multiplier rather than an absolute H.
PARAMETER_SENSITIVITY_FACTORS = {
    "h1": (0.8, 1.0, 1.2),
    "a": (0.9, 1.0, 1.1),
    "h_min": (0.8, 1.0, 1.2),
    "c_H": (0.8, 1.0, 1.2),
}
PARAMETER_SENSITIVITY_LEVEL_LABELS = ("low", "default", "high")
PARAMETER_SENSITIVITY_SIGNALS = (
    "stationary_multi_sine",
    "chirp",
    "impulsive_transient",
    "close_frequencies",
    "crossing_chirps",
    "time_varying_close_frequencies",
    "buried_weak_component",
)
PARAMETER_SENSITIVITY_NOISES = (
    "gaussian",
    "laplace",
    "impulsive",
    "huber_contamination",
)
PARAMETER_SENSITIVITY_SIGMAS = tuple(10.0 ** (-snr / 20.0) for snr in SECTION6_REPRESENTATIVE_TARGET_SNR_DB_LEVELS)
PARAMETER_SENSITIVITY_TARGET_SNR_DB_LEVELS = SECTION6_REPRESENTATIVE_TARGET_SNR_DB_LEVELS

# V5.33/V5.49: IRMF parameter-neighborhood sensitivity protocol, updated after
# relative-H adoption.  Section 6.1 perturbs the locked c_H multiplier and each
# case derives H_case mechanically as c_H * sigma_hat(Y).
IRMF_PARAMETER_SENSITIVITY_PARAMETERS = ("h1", "a", "h_min", "c_H")
IRMF_PARAMETER_SENSITIVITY_OAT_FACTORS = (0.8, 0.9, 1.0, 1.1, 1.2)
IRMF_PARAMETER_SENSITIVITY_JOINT_INTERACTIONS = (
    ("h1", "a"),
    ("a", "h_min"),
    ("c_H", "h1"),
)
IRMF_PARAMETER_SENSITIVITY_IMPORTANCE_DIMENSIONS = (
    "signal_recovery",
    "component_recovery_quality",
    "component_set_fidelity",
    "noise_separation",
    "contamination_resistance",
)

# V5.40: truth-aware time-frequency / instantaneous-frequency diagnostics.
# These are secondary, signal-specific diagnostics and are not part of the
# 13-endpoint primary schema unless the manuscript claim is later changed before
# a new schema freeze.
TIME_FREQUENCY_SECONDARY_APPLICABLE_SIGNALS = (
    "chirp",
    "am_fm",
    "frequency_jump",
    "chirp_slow",
    "chirp_fast",
    "am_fm_weak",
    "am_fm_strong",
    "frequency_jump_up",
    "frequency_jump_down",
    "frequency_jump_multiple",
    "crossing_chirps",
    "crossing_chirps_shallow",
    "crossing_chirps_steep",
    "time_varying_close_frequencies",
    "time_varying_close_frequencies_moderate_gap",
    "time_varying_close_frequencies_severe_gap",
    "piecewise_am_fm_discontinuity",
    "piecewise_am_fm_discontinuity_mild",
    "piecewise_am_fm_discontinuity_severe",
)
TIME_FREQUENCY_SECONDARY_DIAGNOSTICS = (
    "matched_component_if_absolute_error",
    "matched_component_if_normalized_absolute_error",
    "matched_component_if_valid_fraction",
)

# Section 6.4: boundary sensitivity study.
BOUNDARY_SENSITIVITY_SIGNALS = (
    "chirp",
    "frequency_jump",
    "impulsive_transient",
    "intermittent_oscillation",
)
BOUNDARY_SENSITIVITY_NOISES = ("gaussian", "impulsive", "huber_contamination")
BOUNDARY_SENSITIVITY_SIGMAS = tuple(10.0 ** (-snr / 20.0) for snr in SECTION6_REPRESENTATIVE_TARGET_SNR_DB_LEVELS)
BOUNDARY_SENSITIVITY_TARGET_SNR_DB_LEVELS = SECTION6_REPRESENTATIVE_TARGET_SNR_DB_LEVELS

# V5.57: Section 6 challenging-signal extension amendment.  These constants
# expand signal-difficulty coverage without altering the 13 primary endpoints
# or the four Section 6 scientific questions.
V557_SECTION6_AMENDMENT_VERSION = "V5.57_section6_challenging_signal_extension_amendment"
V557_SECTION6_COMMON_ANCHOR_SIGNALS = (
    "chirp",
    "close_frequencies",
    "impulsive_transient",
)
V557_SECTION6_COMMON_ANCHOR_NOISES = (
    "gaussian",
    "impulsive",
    "huber_contamination",
)
V557_SECTION6_COMMON_ANCHOR_TARGET_SNR_DB_LEVELS = SECTION6_REPRESENTATIVE_TARGET_SNR_DB_LEVELS
V557_SECTION6_2_CONTAMINATION_SIGNALS = (
    "stationary_multi_sine",
    "frequency_jump",
    "impulsive_transient",
    "intermittent_oscillation",
    "crossing_chirps",
    "buried_weak_component",
)
V557_SECTION6_3_CHALLENGING_SIGNAL_EXTENSION = (
    "crossing_chirps",
    "time_varying_close_frequencies",
    "piecewise_am_fm_discontinuity",
    "damped_oscillation",
    "trend_plus_oscillation",
    "buried_weak_component",
    "non_sinusoidal_periodic",
    "transient_train",
)
V564_SECTION6_3A_CHALLENGING_VARIANT_EXTENSION_VERSION = (
    "V5.64_section6_3a_challenging_family_variant_extension"
)
V564_SECTION6_3A_CHALLENGING_FAMILY_VARIANTS = (
    "crossing_chirps_shallow",
    "crossing_chirps_steep",
    "time_varying_close_frequencies_moderate_gap",
    "time_varying_close_frequencies_severe_gap",
    "piecewise_am_fm_discontinuity_mild",
    "piecewise_am_fm_discontinuity_severe",
    "damped_oscillation_slow_decay",
    "damped_oscillation_fast_decay",
    "trend_plus_oscillation_mild_trend",
    "trend_plus_oscillation_strong_trend",
    "buried_weak_component_mild",
    "buried_weak_component_severe",
    "non_sinusoidal_periodic_mild_harmonics",
    "non_sinusoidal_periodic_strong_harmonics",
    "transient_train_sparse",
    "transient_train_dense",
)
V564_SECTION6_3A_CHALLENGING_FAMILY_CONDITIONS = (
    V557_SECTION6_3_CHALLENGING_SIGNAL_EXTENSION
    + V564_SECTION6_3A_CHALLENGING_FAMILY_VARIANTS
)
V557_SECTION6_4_SCALING_SIGNALS = (
    "stationary_multi_sine",
    "chirp",
    "crossing_chirps",
)

# Section 6.5: random-seed stability.
RANDOM_SEED_STABILITY_SIGNALS = (
    "chirp",
    "impulsive_transient",
    "close_frequencies",
)
RANDOM_SEED_STABILITY_NOISES = ("gaussian", "huber_contamination", "ar1_colored")
RANDOM_SEED_STABILITY_SIGMA = 10.0 ** (-SECTION6_SINGLE_TARGET_SNR_DB / 20.0)
RANDOM_SEED_STABILITY_TARGET_SNR_DB = SECTION6_SINGLE_TARGET_SNR_DB
RANDOM_SEED_STABILITY_SEEDS = tuple(range(20))

# Challenging signal-regime benchmark.  These signals are structurally unseen
# during parameter selection and are intentionally kept outside the canonical
# Section 5 slice.  In V5.18 they are generated and reported as a full
# five-level main-text cube:
#     8 challenging signals x 8 noises x 5 sigma levels = 320 cases.
CHALLENGING_SIGNAL_FAMILY = (
    "crossing_chirps",
    "time_varying_close_frequencies",
    "piecewise_am_fm_discontinuity",
    "damped_oscillation",
    "trend_plus_oscillation",
    "buried_weak_component",
    "non_sinusoidal_periodic",
    "transient_train",
)
# Match the canonical benchmark noise design so the challenging benchmark
# isolates the effect of unseen signal structure while holding noise mechanisms
# and noise intensities fixed.
CHALLENGING_SIGNAL_NOISES = NOISE_FAMILY
CHALLENGING_SIGNAL_SIGMAS = (0.05, 0.10, 0.20)

CHALLENGING_SIGNAL_QUICK_FAMILY = (
    "crossing_chirps",
    "time_varying_close_frequencies",
    "transient_train",
)
CHALLENGING_SIGNAL_QUICK_NOISES = ("gaussian", "huber_contamination")
CHALLENGING_SIGNAL_QUICK_SIGMAS = (0.20,)

# Controlled challenging-regime diagnostics.  This is a mechanism-analysis
# layer, not a second scoring system.  The common primary metrics are computed
# for every method/case by the same shared evaluation operator used elsewhere;
# the regime-specific diagnostics below are secondary and are never folded into
# case_score.
CONTROLLED_CHALLENGING_REGIMES = (
    "crossing_chirps",
    "time_varying_close_frequencies",
    "buried_weak_component",
    "trend_plus_oscillation",
    "damped_oscillation",
    "transient_train",
)
CONTROLLED_CHALLENGING_NOISES = (
    "gaussian",
    "student_t",
    "huber_contamination",
    "ar1_colored",
)
CONTROLLED_CHALLENGING_SIGMAS = (0.10, 0.20)
CONTROLLED_CHALLENGING_QUICK_NOISES = ("gaussian",)
CONTROLLED_CHALLENGING_QUICK_SIGMAS = (0.20,)
CONTROLLED_CHALLENGING_DIFFICULTY_LEVELS = {
    "crossing_chirps": {
        "name": "crossing_slope_difference",
        "values": (12.0, 24.0, 36.0),
        "quick_values": (24.0,),
        "direction": "higher_is_harder",
    },
    "time_varying_close_frequencies": {
        "name": "min_frequency_gap",
        "values": (1.00, 0.50, 0.25),
        "quick_values": (0.50,),
        "direction": "lower_is_harder",
    },
    "buried_weak_component": {
        "name": "weak_to_strong_amplitude_ratio",
        "values": (0.50, 0.25, 0.10),
        "quick_values": (0.25,),
        "direction": "lower_is_harder",
    },
    "trend_plus_oscillation": {
        "name": "trend_to_oscillation_energy_ratio",
        "values": (0.50, 1.00, 2.00),
        "quick_values": (1.00,),
        "direction": "higher_is_harder",
    },
    "damped_oscillation": {
        "name": "decay_rate",
        "values": (1.00, 2.50, 4.00),
        "quick_values": (2.50,),
        "direction": "higher_is_harder",
    },
    "transient_train": {
        "name": "minimum_event_spacing",
        "values": (0.20, 0.12, 0.06),
        "quick_values": (0.12,),
        "direction": "lower_is_harder",
    },
}

# Appendix/Section 7: IRMF oracle upper-bound and adaptivity-gap analysis.
# The oracle is not a deployable method and is not used for main claims.  It
# estimates the attainable upper performance bound under ideal per-case
# parameter selection and quantifies the adaptivity gap:
#     gap = oracle performance - fixed global-parameter performance
# with metric direction normalized so positive gap means oracle improvement.
ORACLE_ADAPTIVITY_SIGNALS = SIGNAL_FAMILY
ORACLE_ADAPTIVITY_NOISES = NOISE_FAMILY
ORACLE_ADAPTIVITY_SIGMAS = FULL_SIGMA_LEVELS
ORACLE_ADAPTIVITY_QUICK_SIGNALS = ("chirp", "impulsive_transient")
ORACLE_ADAPTIVITY_QUICK_NOISES = ("gaussian", "huber_contamination")
ORACLE_ADAPTIVITY_QUICK_SIGMAS = (0.20,)
ORACLE_ADAPTIVITY_SEARCH_MODE = "quick"
ORACLE_CEEMDAN_EXPLORATORY_TRIALS = (50, 100, 200)
ORACLE_CEEMDAN_EXPLORATORY_EPSILONS = (0.0025, 0.005, 0.010, 0.020)
ORACLE_CEEMDAN_EXPLORATORY_MAX_IMF = (-1, 5)
ORACLE_CEEMDAN_EXPLORATORY_QUICK_TRIALS = (50,)
ORACLE_CEEMDAN_EXPLORATORY_QUICK_EPSILONS = (0.005, 0.010)
ORACLE_CEEMDAN_EXPLORATORY_QUICK_MAX_IMF = (-1,)

# Appendix A: Spokoiny-inspired theory-diagnostic validation.  These cases are
# representative rather than exhaustive so Appendix A remains a theory-facing
# diagnostic layer instead of another full benchmark.
SPOKOINY_THEORY_VALIDATION_CASES = (
    ("stationary_multi_sine", "gaussian", 0.10),
    ("chirp", "gaussian", 0.10),
    ("frequency_jump", "gaussian", 0.20),
    ("impulsive_transient", "huber_contamination", 0.20),
    ("close_frequencies", "ar1_colored", 0.20),
    ("am_fm", "heteroskedastic", 0.20),
)
SPOKOINY_THEORY_TRACE_PROBE_COUNT = 16

# Section 6.3A: intra-family signal morphology variants.  V5.64A adds the
# seven Section 5 canonical anchors as reference specifications, so 6.3A can
# evaluate anchor-to-variant preservation rather than variants in isolation.
V564A_SECTION6_3A_CANONICAL_ANCHOR_REFERENCE_VERSION = (
    "V5.64A_section6_3a_canonical_anchor_reference_amendment"
)
V564A_SECTION6_3A_CANONICAL_ANCHORS = (
    "stationary_multi_sine",
    "chirp",
    "am_fm",
    "frequency_jump",
    "impulsive_transient",
    "intermittent_oscillation",
    "close_frequencies",
)
V564A_SECTION6_3A_CANONICAL_FAMILY_VARIANTS = (
    "stationary_multi_sine_wide",
    "stationary_multi_sine_medium",
    "stationary_multi_sine_close",
    "chirp_slow",
    "chirp_fast",
    "am_fm_weak",
    "am_fm_strong",
    "frequency_jump_up",
    "frequency_jump_down",
    "frequency_jump_multiple",
    "impulsive_transient_short",
    "impulsive_transient_long",
    "impulsive_transient_strong",
    "intermittent_one_interval",
    "intermittent_two_intervals",
    "intermittent_irregular",
    "close_freq_10_12",
    "close_freq_10_11",
    "close_freq_10_10p5",
)
V565_SECTION6_3A_CANONICAL_STRUCTURED_PERTURBATION_VERSION = (
    "V5.65_section6_3a_canonical_structured_perturbation_extension"
)
V565_SECTION6_3A_CANONICAL_STRUCTURED_VARIANTS = (
    "stationary_multi_sine_balanced_amplitudes",
    "stationary_multi_sine_high_imbalance",
    "chirp_narrow_band",
    "chirp_broad_band",
    "am_fm_am_weak",
    "am_fm_am_strong",
    "am_fm_fm_weak",
    "am_fm_fm_strong",
    "frequency_jump_small_magnitude",
    "frequency_jump_large_magnitude",
    "frequency_jump_early",
    "frequency_jump_late",
    "impulsive_transient_weak",
    "impulsive_transient_early",
    "impulsive_transient_late",
    "intermittent_sparse_duty",
    "intermittent_dense_duty",
    "close_freq_second_weak",
    "close_freq_second_moderate",
)
V566_SECTION6_3A_CANONICAL_TWO_AXIS_INTERACTION_VERSION = (
    "V5.66_section6_3a_canonical_two_axis_interaction_extension"
)
V566_SECTION6_3A_CANONICAL_INTERACTION_VARIANTS = (
    "stationary_multi_sine_wide_balanced",
    "stationary_multi_sine_close_high_imbalance",
    "chirp_slow_narrow_band",
    "chirp_fast_broad_band",
    "frequency_jump_small_early",
    "frequency_jump_large_late",
    "impulsive_transient_weak_early",
    "impulsive_transient_strong_late",
    "intermittent_sparse_one_interval",
    "intermittent_dense_irregular",
    "close_freq_wide_second_moderate",
    "close_freq_veryclose_second_weak",
)
V566_SECTION6_3A_CANONICAL_TWO_AXIS_SIGNAL_SPECS = (
    "stationary_multi_sine",
    "stationary_multi_sine_wide",
    "stationary_multi_sine_close",
    "stationary_multi_sine_balanced_amplitudes",
    "stationary_multi_sine_high_imbalance",
    "stationary_multi_sine_wide_balanced",
    "stationary_multi_sine_close_high_imbalance",
    "chirp",
    "chirp_slow",
    "chirp_fast",
    "chirp_narrow_band",
    "chirp_broad_band",
    "chirp_slow_narrow_band",
    "chirp_fast_broad_band",
    "am_fm",
    "am_fm_am_weak",
    "am_fm_am_strong",
    "am_fm_fm_weak",
    "am_fm_fm_strong",
    "am_fm_weak",
    "am_fm_strong",
    "frequency_jump",
    "frequency_jump_small_magnitude",
    "frequency_jump_large_magnitude",
    "frequency_jump_early",
    "frequency_jump_late",
    "frequency_jump_small_early",
    "frequency_jump_large_late",
    "impulsive_transient",
    "impulsive_transient_weak",
    "impulsive_transient_strong",
    "impulsive_transient_early",
    "impulsive_transient_late",
    "impulsive_transient_weak_early",
    "impulsive_transient_strong_late",
    "intermittent_oscillation",
    "intermittent_sparse_duty",
    "intermittent_dense_duty",
    "intermittent_one_interval",
    "intermittent_irregular",
    "intermittent_sparse_one_interval",
    "intermittent_dense_irregular",
    "close_frequencies",
    "close_freq_10_12",
    "close_freq_10_10p5",
    "close_freq_second_moderate",
    "close_freq_second_weak",
    "close_freq_wide_second_moderate",
    "close_freq_veryclose_second_weak",
)
SIGNAL_VARIANT_FAMILY = (
    *V566_SECTION6_3A_CANONICAL_TWO_AXIS_SIGNAL_SPECS,
    "crossing_chirps",
    "time_varying_close_frequencies",
    "piecewise_am_fm_discontinuity",
    "damped_oscillation",
    "trend_plus_oscillation",
    "buried_weak_component",
    "non_sinusoidal_periodic",
    "transient_train",
    "crossing_chirps_shallow",
    "crossing_chirps_steep",
    "time_varying_close_frequencies_moderate_gap",
    "time_varying_close_frequencies_severe_gap",
    "piecewise_am_fm_discontinuity_mild",
    "piecewise_am_fm_discontinuity_severe",
    "damped_oscillation_slow_decay",
    "damped_oscillation_fast_decay",
    "trend_plus_oscillation_mild_trend",
    "trend_plus_oscillation_strong_trend",
    "buried_weak_component_mild",
    "buried_weak_component_severe",
    "non_sinusoidal_periodic_mild_harmonics",
    "non_sinusoidal_periodic_strong_harmonics",
    "transient_train_sparse",
    "transient_train_dense",
)
SIGNAL_VARIANT_NOISES = (
    "gaussian",
    "laplace",
    "impulsive",
    "huber_contamination",
    "heteroskedastic",
)
SIGNAL_VARIANT_TARGET_SNR_DB_LEVELS = SECTION6_REPRESENTATIVE_TARGET_SNR_DB_LEVELS
SIGNAL_VARIANT_SIGMAS = tuple(10.0 ** (-snr / 20.0) for snr in SIGNAL_VARIANT_TARGET_SNR_DB_LEVELS)
SIGNAL_VARIANT_SEEDS = UNIFIED_BENCHMARK_SEEDS

# V5.56 supplementary waveform-level stability protocol.  This is a
# representative subset rerun specification only; it is not part of the
# primary endpoint schema and is not ranking-bearing.
V556_WAVEFORM_STABILITY_SIGNALS = (
    "chirp",
    "close_frequencies",
    "impulsive_transient",
    "crossing_chirps",
    "time_varying_close_frequencies",
    "piecewise_am_fm_discontinuity",
)
V556_WAVEFORM_STABILITY_NOISES = (
    "gaussian",
    "impulsive",
    "huber_contamination",
    "ar1_colored",
)
V556_WAVEFORM_STABILITY_TARGET_SNR_DB_LEVELS = (5.0, 15.0, 25.0)
V556_WAVEFORM_STABILITY_SEEDS = tuple(range(20))
V556_WAVEFORM_STABILITY_METHODS = EVALUATION_METHODS
V556_WAVEFORM_STABILITY_LOW_SIMILARITY_PRIMARY_THRESHOLD = 0.80
V556_WAVEFORM_STABILITY_LOW_SIMILARITY_THRESHOLD_SENSITIVITY = (0.70, 0.80, 0.90)

# V5.4 loss-methodology upgrade -------------------------------------------------
LOSS_ABLATION_LOSSES = (
    "l2", "l1", "huber", "pseudo_huber", "fair", "tukey",
    "gaussian_smoothed_median",
)
LOSS_ABLATION_SIGNALS = (
    "stationary_multi_sine", "chirp", "impulsive_transient",
    "intermittent_oscillation", "close_frequencies",
)
LOSS_ABLATION_NOISES = (
    "gaussian", "student_t", "impulsive", "burst", "huber_contamination",
)
LOSS_ABLATION_SIGMAS = (0.10, 0.20)
LOSS_ABLATION_SEEDS = (0, 1, 2)
LOSS_TUNING_GRIDS = {
    "l2": ({},),
    "l1": ({"smooth_eps": 1e-6},),
    "huber": ({"delta": 0.75}, {"delta": 1.0}, {"delta": 1.345}, {"delta": 2.0}),
    "pseudo_huber": ({"delta": 0.5}, {"delta": 1.0}, {"delta": 1.5}, {"delta": 2.0}),
    "fair": ({"c": 0.75}, {"c": 1.0}, {"c": 1.4}, {"c": 2.0}),
    "tukey": ({"c": 2.5}, {"c": 3.5}, {"c": 4.685}, {"c": 6.0}),
    "gaussian_smoothed_median": ({"H": 0.5}, {"H": 0.75}, {"H": 1.0}, {"H": 1.5}, {"H": 2.0}),
}
LOSS_CONTAMINATION_LAMBDAS = (0.0, 0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
LOSS_FAILURE_THRESHOLDS = {"denoise_nmse": 1.0, "denoise_corr": 0.5}


# V5.7 methodology loss-regime map --------------------------------------------
LOSS_REGIME_CORE_LOSSES = ("huber", "pseudo_huber", "gaussian_smoothed_median")
LOSS_REGIME_GAUSSIAN_ARE_TARGET = 0.95
LOSS_REGIME_STRUCTURE_SIGNALS = ("frequency_jump", "impulsive_transient", "intermittent_oscillation")
LOSS_REGIME_CONTAMINATION_KINDS = ("isolated_spike", "clustered_burst", "event_overlap", "asymmetric_cluster")
LOSS_REGIME_DISTANCE_POINTS = (-20, -10, -5, 0, 5, 10, 20)
LOSS_REGIME_CONTAMINATION_RATES = (0.02, 0.05, 0.10, 0.20)
LOSS_REGIME_OUTLIER_SCALES = (5.0, 10.0, 20.0)
LOSS_REGIME_QUADRATIC_DELTA_LEVELS = (0.01, 0.02, 0.05, 0.10, 0.20)
LOSS_REGIME_SEEDS = tuple(range(10))
