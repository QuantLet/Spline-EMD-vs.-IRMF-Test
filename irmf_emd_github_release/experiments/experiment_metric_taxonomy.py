#!/usr/bin/python
# coding: UTF-8

"""Write the paper-facing metric taxonomy/protocol."""

from project_config import (
    CANONICAL_DIAGNOSTIC_TAXONOMY_BY_GROUP,
    CANONICAL_DIAGNOSTIC_TAXONOMY_PROVENANCE,
    CANONICAL_DIAGNOSTIC_TAXONOMY_STATUS,
    CANONICAL_DIAGNOSTIC_TAXONOMY_VERSION,
    CASE_SCORE_ROLE,
    COMPOSITE_SENSITIVITY_ENDPOINTS,
    COMPONENT_ALIGNMENT_POLICY,
    COMPONENT_RECOVERY_MATCHING_AUDIT,
    CONDITIONAL_ENDPOINTS_BY_DIMENSION,
    CONDITIONAL_PRIMARY_ENDPOINTS_BY_DIMENSION,
    DERIVED_DUPLICATE_FIELDS,
    DECOMPOSITION_SEMANTICS_BY_METHOD,
    DIAGNOSTIC_METRICS_BY_DIMENSION,
    EVALUATION_DIMENSION_ESTIMANDS,
    EVALUATION_FRAMEWORK_FROZEN_DATE,
    EVALUATION_FRAMEWORK_STATUS,
    EVALUATION_FRAMEWORK_VERSION,
    LEGACY_ALIAS_FIELDS,
    LOWER_IS_BETTER_METRICS,
    METRIC_PROVENANCE_QUALIFICATION_APPROACH,
    METRIC_QUALIFICATION_PRINCIPLE,
    PRIMARY_METRIC_PROVENANCE,
    PRIMARY_METRIC_QUALIFICATION_SCOPE,
    PRIMARY_ENDPOINTS_BY_DIMENSION,
    PROTOCOL_METADATA_FIELDS,
    PUBLICATION_COMPONENT_RECOVERY_NAMES,
    RELATIVE_DECOMPOSITION_COUNT_ERROR_DEFINITION,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


def _metric_provenance_rows():
    rows = []
    for item in METRIC_PROVENANCE_QUALIFICATION_APPROACH:
        rows.append({
            "metric_provenance": item["metric_provenance"],
            "examples": ", ".join(item["examples"]),
            "qualification_approach": item["qualification_approach"],
            "rationale": item["rationale"],
        })
    return rows


def _primary_metric_qualification_rows():
    out = []
    for metric, provenance in sorted(PRIMARY_METRIC_PROVENANCE.items()):
        out.append({
            "metric": metric,
            "metric_provenance": provenance,
            "qualification_scope": PRIMARY_METRIC_QUALIFICATION_SCOPE.get(metric),
            "full_structural_qualification": (
                PRIMARY_METRIC_QUALIFICATION_SCOPE.get(metric)
                == "full_structural_metric_qualification"
            ),
        })
    return out


def _canonical_diagnostic_rows():
    rows = []
    for group, metrics in CANONICAL_DIAGNOSTIC_TAXONOMY_BY_GROUP.items():
        for metric in metrics:
            rows.append({
                "taxonomy_version": CANONICAL_DIAGNOSTIC_TAXONOMY_VERSION,
                "taxonomy_status": CANONICAL_DIAGNOSTIC_TAXONOMY_STATUS,
                "canonical_group": group,
                "metric": metric,
                "scientific_metric_count_role": "canonical_scientific_diagnostic",
            })
    return rows


def _field_deduplication_rows():
    canonical = {
        metric
        for metrics in CANONICAL_DIAGNOSTIC_TAXONOMY_BY_GROUP.values()
        for metric in metrics
    }
    all_diagnostic_fields = {
        metric
        for metrics in DIAGNOSTIC_METRICS_BY_DIMENSION.values()
        for metric in metrics
    }
    all_fields = sorted(
        all_diagnostic_fields
        | canonical
        | set(PROTOCOL_METADATA_FIELDS)
        | set(LEGACY_ALIAS_FIELDS)
        | set(DERIVED_DUPLICATE_FIELDS)
    )
    rows = []
    for field in all_fields:
        if field in canonical:
            field_class = "canonical_scientific_diagnostic"
            canonical_or_source = field
        elif field in PROTOCOL_METADATA_FIELDS:
            field_class = "protocol_metadata_not_scientific_metric"
            canonical_or_source = ""
        elif field in LEGACY_ALIAS_FIELDS:
            field_class = "legacy_alias_not_counted_separately"
            canonical_or_source = LEGACY_ALIAS_FIELDS[field]
        elif field in DERIVED_DUPLICATE_FIELDS:
            field_class = "derived_duplicate_not_counted_separately"
            canonical_or_source = DERIVED_DUPLICATE_FIELDS[field]
        else:
            field_class = "noncanonical_supporting_output"
            canonical_or_source = ""
        rows.append({
            "field": field,
            "field_class": field_class,
            "canonical_or_source": canonical_or_source,
            "include_in_scientific_diagnostic_count": field in canonical,
        })
    return rows


def write_metric_taxonomy(output_root):
    output_root = ensure_dir(output_root)
    taxonomy = {
        "section": "Evaluation Metric Taxonomy",
        "evaluation_framework_version": EVALUATION_FRAMEWORK_VERSION,
        "evaluation_framework_status": EVALUATION_FRAMEWORK_STATUS,
        "evaluation_framework_frozen_date": EVALUATION_FRAMEWORK_FROZEN_DATE,
        "benchmark_source_version": CANONICAL_DIAGNOSTIC_TAXONOMY_PROVENANCE["benchmark_source_version"],
        "taxonomy_layer_version": CANONICAL_DIAGNOSTIC_TAXONOMY_PROVENANCE["taxonomy_layer_version"],
        "benchmark_results_recomputed": CANONICAL_DIAGNOSTIC_TAXONOMY_PROVENANCE["benchmark_results_recomputed"],
        "principle": (
            "Primary paper conclusions are organized by pre-specified "
            "evaluation dimensions rather than by counting individual metrics. "
            "Synthetic benchmark evidence uses eight universal primary "
            "endpoints organized into signal recovery, component recovery, "
            "component allocation fidelity, and noise separation dimensions. "
            "Two contamination-specific conditional primary endpoints are "
            "evaluated only in contamination-aware regimes. The legacy weighted "
            "case_score is retained only as a supplementary sensitivity check."
        ),
        "scientific_questions": {
            "Q1_reconstruction": "Can the clean signal be reconstructed?",
            "Q2_structural_fidelity": (
                "Can true oscillatory components be recovered without splitting "
                "one true component across several modes or merging several true "
                "components into one mode?"
            ),
            "Q3_contamination_resistance": "Can contamination be rejected while preserving clean signal regions?",
        },
        "dimension_estimands": EVALUATION_DIMENSION_ESTIMANDS,
        "decomposition_semantics_by_method": DECOMPOSITION_SEMANTICS_BY_METHOD,
        "component_alignment_policy": COMPONENT_ALIGNMENT_POLICY,
        "primary_endpoints_by_dimension": PRIMARY_ENDPOINTS_BY_DIMENSION,
        "conditional_primary_endpoints_by_dimension": CONDITIONAL_PRIMARY_ENDPOINTS_BY_DIMENSION,
        "conditional_endpoints_by_dimension": CONDITIONAL_ENDPOINTS_BY_DIMENSION,
        "diagnostic_metrics_by_dimension": DIAGNOSTIC_METRICS_BY_DIMENSION,
        "lower_is_better_metrics": list(LOWER_IS_BETTER_METRICS),
        "primary_dimensions": {
            "reconstruction": {
                "primary_endpoints": ["denoise_nmse", "denoise_corr"],
                "legacy_score": "reconstruction_score",
                "role": "overall denoising/reconstruction accuracy",
            },
            "structural_fidelity": {
                "primary_subdimensions": [
                    "component_recovery",
                    "true_component_mixing",
                ],
                "legacy_score": "structural_fidelity_score",
                "role": "truth-referenced component recovery and component-mixing behavior",
            },
            "contamination_resistance": {
                "primary_subdimensions": [
                    "noise_capture",
                    "signal_preservation",
                ],
                "conditional_primary_subdimensions": [
                    "outlier_resistance",
                    "clean_region_preservation",
                ],
                "conditional_subdimensions": [
                    "contaminated_region_performance",
                    "spillover_control",
                ],
                "applicability_gate": {
                    "outlier_resistance_index": ["impulsive", "burst", "huber_contamination"],
                    "clean_region_nmse": ["impulsive", "burst", "huber_contamination"],
                },
                "legacy_score": "robustness_score",
                "legacy_code_field": "contamination_resistance_score",
                "role": (
                    "noise-separation evidence is universal through "
                    "noise_capture_corr and signal_leakage_into_noise. "
                    "outlier_resistance_index and clean_region_nmse are "
                    "contamination-specific conditional primary endpoints and "
                    "enter primary statistics only when the benchmark regime "
                    "provides the corresponding contamination annotations"
                ),
                "trajectory_endpoints": [
                    "normalized_contamination_audc",
                    "absolute_contamination_audc",
                ],
            },
        },
        "case_score_policy": {
            "role": CASE_SCORE_ROLE,
            "composite_sensitivity_endpoints": list(COMPOSITE_SENSITIVITY_ENDPOINTS),
            "current_code_weights": {
                "reconstruction_score": 0.40,
                "structural_fidelity_score": 0.35,
                "robustness_score": 0.25,
            },
            "used_for": [
                "development-set parameter selection",
                "supplementary aggregate ranking",
            ],
            "not_used_as_sole_main_claim": True,
            "not_used_for_primary_friedman_or_cd": True,
            "weighting_sensitivity_output": "../07b_case_score_weighting_sensitivity",
        },
        "hierarchical_ranking_policy": {
            "block": "signal x noise x sigma",
            "seed_policy": "aggregate repeated seeds within each block before case-level ranking",
            "flow": (
                "metric rank -> subdimension rank -> dimension rank -> optional "
                "descriptive hierarchical overall rank"
            ),
            "overall_rank_role": "descriptive_only",
            "output": "../07c_primary_evaluation_framework",
        },
        "metric_redundancy_policy": {
            "purpose": "avoid double-counting highly collinear structural and contamination diagnostics",
            "method": "Spearman correlation on direction-aligned values plus paired IRMF-vs-baseline contrast views",
            "output": "../07c_primary_evaluation_framework",
            "selection_policy": "primary endpoints are pre-specified from scientific roles; redundancy analysis is confirmatory/diagnostic",
        },
        "publication_taxonomy_cleanup": {
            "taxonomy_version": CANONICAL_DIAGNOSTIC_TAXONOMY_VERSION,
            "taxonomy_status": CANONICAL_DIAGNOSTIC_TAXONOMY_STATUS,
            "provenance": CANONICAL_DIAGNOSTIC_TAXONOMY_PROVENANCE,
            "scope": (
                "publication-facing taxonomy layer only; does not change the "
                "frozen V5.24 benchmark results, primary endpoints, or "
                "statistical input cube"
            ),
            "principle": (
                "Primary conclusions are based on pre-specified primary "
                "endpoints. Additional diagnostic fields are organized into a "
                "reduced canonical taxonomy; protocol metadata, legacy aliases, "
                "and derived duplicates are excluded from scientific metric "
                "counts."
            ),
            "canonical_diagnostic_taxonomy_by_group": CANONICAL_DIAGNOSTIC_TAXONOMY_BY_GROUP,
            "protocol_metadata_fields": list(PROTOCOL_METADATA_FIELDS),
            "legacy_alias_fields": LEGACY_ALIAS_FIELDS,
            "derived_duplicate_fields": DERIVED_DUPLICATE_FIELDS,
        },
        "relative_decomposition_count_error": RELATIVE_DECOMPOSITION_COUNT_ERROR_DEFINITION,
        "component_recovery_publication_naming": {
            "code_fields": list(PUBLICATION_COMPONENT_RECOVERY_NAMES.keys()),
            "publication_names": PUBLICATION_COMPONENT_RECOVERY_NAMES,
            "matching_audit": COMPONENT_RECOVERY_MATCHING_AUDIT,
        },
        "contamination_trajectory_policy": {
            "audc_role": "Section 6 contamination-trajectory endpoint, not a case-level case_score component",
            "normalized_contamination_audc": (
                "Area under the relative loss-degradation curve across lambda, "
                "normalized by lambda range and a stable baseline-loss denominator."
            ),
            "absolute_contamination_audc": "Area under the absolute loss-degradation curve; reported as sensitivity check.",
            "trajectory_unit": "signal x base noise x sigma x seed x method x endpoint, varying lambda only",
            "output": "../08_robustness_sensitivity/6_2_contamination_robustness",
        },
        "stability_policy": {
            "stability_not_in_case_score": True,
            "reason": (
                "Stability under parameter perturbation or random seeds is an "
                "algorithm-level property, not a single-case decomposition metric."
            ),
        },
        "primary_inference_metrics": [
            "denoise_nmse",
            "denoise_corr",
            "imf_recovery_corr",
            "imf_recovery_nrmse",
            "component_splitting_index",
            "component_merging_index",
            "noise_capture_corr",
            "signal_leakage_into_noise",
        ],
        "conditional_primary_inference_metrics": [
            "outlier_resistance_index",
            "clean_region_nmse",
        ],
        "secondary_universal_noise_separation_diagnostics": [
            "noise_energy_ratio",
            "noise_energy_log_error",
            "remaining_noise_energy_ratio",
        ],
        "secondary_noise_energy_diagnostic_policy": {
            "primary_endpoint_status": "not_primary",
            "noise_energy_eps": 1e-12,
            "noise_energy_bias_tolerance": 0.05,
            "role": (
                "secondary universal diagnostics for noise-separation quantity "
                "and suppression effectiveness; retained outside the frozen "
                "8 universal primary endpoint set"
            ),
            "preferred_statistical_use": (
                "noise_energy_log_error and remaining_noise_energy_ratio are "
                "suitable for secondary summaries and sensitivity checks; "
                "noise_energy_ratio is primarily directional/interpretable"
            ),
            "zero_variance_protocol_noise_behavior": (
                "Energy diagnostics remain computable when protocol_estimated_noise "
                "has zero variance, unlike correlation-based noise endpoints"
            ),
        },
        "primary_metric_writing_policy": {
            "do_not_describe_as": "10 universal primary endpoints",
            "preferred_description": (
                "eight universal primary endpoints plus two contamination-"
                "specific conditional primary endpoints: Signal Recovery (2), "
                "Component Recovery (2), Component Allocation Fidelity (2), "
                "Noise Separation (2), and contamination-specific conditional "
                "primary endpoints (2, with applicability gate)"
            ),
        },
        "contamination_applicability_policy": (
            "Two contamination-specific primary endpoints "
            "(outlier_resistance_index and clean_region_nmse) are evaluated "
            "only for contamination-aware benchmark regimes where contamination "
            "masks or clean-region annotations are available. They are excluded "
            "from primary statistical comparisons for non-contamination cases."
        ),
        "metric_qualification_principle": METRIC_QUALIFICATION_PRINCIPLE,
        "metric_provenance_qualification_approach": list(
            METRIC_PROVENANCE_QUALIFICATION_APPROACH
        ),
        "primary_metric_provenance": PRIMARY_METRIC_PROVENANCE,
        "primary_metric_qualification_scope": PRIMARY_METRIC_QUALIFICATION_SCOPE,
        "primary_metric_qualification_boundary": (
            "Full structural metric qualification is limited to "
            "component_splitting_index and component_merging_index under "
            "TRUE_COMPONENT_MIXING_V1.0. Other established or derived primary "
            "metrics use the qualification approach appropriate to their "
            "metric provenance."
        ),
        "conditional_inference_metrics": [
            "contaminated_region_nmse",
            "contamination_spillover_error",
        ],
        "secondary_diagnostics": ["case_score", "runtime_seconds"] + [
            metric
            for metrics in DIAGNOSTIC_METRICS_BY_DIMENSION.values()
            for metric in metrics
        ],
        "canonical_scientific_diagnostics": [
            metric
            for metrics in CANONICAL_DIAGNOSTIC_TAXONOMY_BY_GROUP.values()
            for metric in metrics
        ],
        "diagnostic_field_deduplication_policy": (
            "Do not count protocol metadata, legacy aliases, or derived "
            "duplicates as distinct scientific diagnostics."
        ),
        "trajectory_diagnostics": [
            "normalized_contamination_audc",
            "absolute_contamination_audc",
            "complete_lambda_trajectory",
            "has_lambda_zero_baseline",
            "normalized_degradation_slope",
            "absolute_degradation_slope",
        ],
    }
    write_json(taxonomy, output_root / "metric_taxonomy_protocol.json")
    write_csv(
        _metric_provenance_rows(),
        output_root / "metric_provenance_qualification_approach.csv",
    )
    write_csv(
        _primary_metric_qualification_rows(),
        output_root / "primary_metric_qualification_scope.csv",
    )
    write_csv(
        _canonical_diagnostic_rows(),
        output_root / "canonical_diagnostic_taxonomy.csv",
    )
    write_csv(
        _field_deduplication_rows(),
        output_root / "diagnostic_field_deduplication_audit.csv",
    )
    return taxonomy
