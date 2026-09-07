#!/usr/bin/python
# coding: UTF-8

"""V5.41 primary-to-secondary diagnostic closure map."""

from datetime import datetime, timezone

from project_config import (
    CANONICAL_DIAGNOSTIC_TAXONOMY_BY_GROUP,
    TIME_FREQUENCY_SECONDARY_DIAGNOSTICS,
    V528_PRIMARY_ENDPOINTS_BY_DIMENSION,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


V541_CLOSURE_VERSION = "V5.41_primary_to_secondary_diagnostic_closure"


SECONDARY_ROLE_TAXONOMY = {
    "scientific_mechanism_diagnostics": {
        "question": "What scientific mechanism is consistent with the observed primary outcome pattern?",
        "interpretation_rule": (
            "Use as mechanism-consistent evidence, not as causal proof unless a "
            "separate controlled experiment supports causality."
        ),
    },
    "scientific_validity_diagnostics": {
        "question": (
            "Under what case conditions, support conditions, or degeneracy states "
            "is the outcome or diagnostic interpretable?"
        ),
        "interpretation_rule": (
            "Use to qualify computability, applicability, difficulty, support, "
            "or protocol context; do not rank methods on these fields."
        ),
    },
    "computational_reliability_diagnostics": {
        "question": "Was the reported result produced by a valid and stable computation?",
        "interpretation_rule": (
            "Use to separate scientific performance behavior from timeout, "
            "exception, nonfinite output, or other execution failures."
        ),
    },
}


SECONDARY_DIAGNOSTIC_INVENTORY = {
    "reconstruction_context_diagnostics": {
        "role_class": "scientific_validity_diagnostics",
        "diagnostics": (
            "spectral_corr",
            "input_snr_db",
            "output_snr_db",
            "snr_gain_db",
        ),
        "role": (
            "Complementary reconstruction-context diagnostics; useful for "
            "checking case difficulty, spectral agreement, and SNR improvement. "
            "They are not independent primary outcomes."
        ),
    },
    "component_cardinality_diagnostics": {
        "role_class": "scientific_mechanism_diagnostics",
        "diagnostics": (
            "imf_count",
            "effective_imf_count",
            "over_decomposition_penalty",
            "under_decomposition_index",
        ),
        "role": (
            "Explain whether component-set failures are driven by too many, too "
            "few, or energetically negligible components. The V5.28 primary "
            "relative_decomposition_count_error gives count-error magnitude; "
            "over/under diagnostics give direction and mechanism."
        ),
    },
    "matching_allocation_diagnostics": {
        "role_class": "scientific_mechanism_diagnostics",
        "diagnostics": (
            "true_component_splitting_max",
            "estimated_component_merging_max",
            "missing_true_component_count",
            "unmatched_estimated_component_count",
        ),
        "role": (
            "Mechanism diagnostics for allocation failures: fragmentation of a "
            "true component, merging of several true components, missing truth, "
            "or unmatched estimated modes."
        ),
    },
    "inter_component_separation_diagnostics": {
        "role_class": "scientific_mechanism_diagnostics",
        "diagnostics": (
            "inter_imf_entanglement_index",
            "strict_io",
            "spectral_leakage",
            "frequency_overlap_max_offdiag",
            "frequency_separation_score",
        ),
        "role": (
            "Explain whether recovered components are mutually entangled, "
            "spectrally diffuse, or poorly separated. These diagnostics should "
            "be interrogated selectively rather than all reported as another "
            "benchmark table."
        ),
    },
    "signal_specific_diagnostics": {
        "role_class": "scientific_mechanism_diagnostics",
        "diagnostics": (
            "transient_smearing_index",
            "transient_preservation_score",
        ),
        "role": (
            "Signal-specific mechanism diagnostics for transient or localized "
            "structure; not universal primary endpoints."
        ),
    },
    "noise_energy_and_noncomputability_diagnostics": {
        "role_class": "mixed_mechanism_and_validity_diagnostics",
        "diagnostics": (
            "noise_energy_ratio",
            "remaining_noise_energy_ratio",
            "noise_energy_bias_class",
            "noise_energy_not_computable_reason",
            "protocol_zero_variance_noise_flag",
            "protocol_noise_reason_code",
            "signal_leakage_degenerate_flag",
        ),
        "role": (
            "Explain noise-separation failures and zero-variance/non-computable "
            "correlation cases. The V5.28 primary noise_energy_log_error is not "
            "counted again here. Energy ratios/bias classes are scientific "
            "mechanism diagnostics; flags and reason codes are scientific "
            "validity diagnostics."
        ),
    },
    "contamination_mechanism_diagnostics": {
        "role_class": "mixed_mechanism_and_validity_diagnostics",
        "diagnostics": (
            "contamination_spillover_error",
            "contamination_spillover_score",
            "contaminated_region_preservation_score",
            "clean_region_signal_leakage",
            "clean_region_signal_leakage_score",
            "contaminated_fraction",
            "contaminated_point_count",
            "spillover_radius",
        ),
        "role": (
            "Conditional diagnostics for contamination-mask regimes; explain "
            "whether direct contamination, clean-region collateral damage, or "
            "local spillover accompanies robustness failures. Mask size and "
            "radius fields are validity/protocol-context diagnostics."
        ),
    },
    "time_frequency_secondary_diagnostics": {
        "role_class": "mixed_mechanism_and_validity_diagnostics",
        "diagnostics": TIME_FREQUENCY_SECONDARY_DIAGNOSTICS,
        "role": (
            "V5.40 truth-aware IF diagnostics for applicable signal families; "
            "secondary only and not ranking-bearing. IF error fields are "
            "scientific mechanism/fidelity diagnostics; valid_fraction is an "
            "evaluation-support diagnostic."
        ),
    },
    "runtime_and_failure_diagnostics": {
        "role_class": "computational_reliability_diagnostics",
        "diagnostics": (
            "runtime_seconds",
            "timeout_flag",
            "timeout_seconds",
            "exception_flag",
            "failure_level",
            "failure_reason",
            "nonfinite_output_flag",
            "computational_failure",
        ),
        "role": (
            "Execution diagnostics used to distinguish scientific performance "
            "from computational failure or timeout behavior."
        ),
    },
}


METRIC_ROLE_CLASS_OVERRIDES = {
    "spectral_corr": "scientific_validity_diagnostics",
    "input_snr_db": "scientific_validity_diagnostics",
    "output_snr_db": "scientific_validity_diagnostics",
    "snr_gain_db": "scientific_validity_diagnostics",
    "noise_energy_ratio": "scientific_mechanism_diagnostics",
    "remaining_noise_energy_ratio": "scientific_mechanism_diagnostics",
    "noise_energy_bias_class": "scientific_mechanism_diagnostics",
    "noise_energy_not_computable_reason": "scientific_validity_diagnostics",
    "protocol_zero_variance_noise_flag": "scientific_validity_diagnostics",
    "protocol_noise_reason_code": "scientific_validity_diagnostics",
    "signal_leakage_degenerate_flag": "scientific_validity_diagnostics",
    "contamination_spillover_error": "scientific_mechanism_diagnostics",
    "contamination_spillover_score": "scientific_mechanism_diagnostics",
    "contaminated_region_preservation_score": "scientific_mechanism_diagnostics",
    "clean_region_signal_leakage": "scientific_mechanism_diagnostics",
    "clean_region_signal_leakage_score": "scientific_mechanism_diagnostics",
    "contaminated_fraction": "scientific_validity_diagnostics",
    "contaminated_point_count": "scientific_validity_diagnostics",
    "spillover_radius": "scientific_validity_diagnostics",
    "matched_component_if_absolute_error": "scientific_mechanism_diagnostics",
    "matched_component_if_normalized_absolute_error": "scientific_mechanism_diagnostics",
    "matched_component_if_valid_fraction": "scientific_validity_diagnostics",
}


PRIMARY_TO_SECONDARY_CLOSURE = (
    {
        "primary_dimension": "signal_recovery",
        "primary_endpoint": "denoise_nmse",
        "weak_pattern": "high reconstruction error",
        "secondary_diagnostics": (
            "spectral_corr",
            "snr_gain_db",
            "inter_imf_entanglement_index",
            "spectral_leakage",
            "frequency_overlap_max_offdiag",
            "transient_smearing_index",
            "runtime_seconds",
            "failure_reason",
        ),
        "mechanism_question": (
            "Is poor signal recovery accompanied by spectral distortion, insufficient "
            "SNR gain, component entanglement, transient smearing, or execution failure?"
        ),
        "claim_boundary": (
            "Diagnostics indicate mechanisms consistent with the primary pattern; "
            "denoise_nmse remains the outcome endpoint."
        ),
    },
    {
        "primary_dimension": "signal_recovery",
        "primary_endpoint": "denoise_corr",
        "weak_pattern": "low signal shape correlation",
        "secondary_diagnostics": (
            "spectral_corr",
            "snr_gain_db",
            "transient_preservation_score",
            "frequency_separation_score",
        ),
        "mechanism_question": (
            "Is waveform-shape mismatch associated with spectral mismatch, "
            "localized-structure loss, or poor frequency separation?"
        ),
        "claim_boundary": "Do not treat spectral diagnostics as substitutes for signal correlation.",
    },
    {
        "primary_dimension": "component_recovery_quality",
        "primary_endpoint": "matched_component_corr",
        "weak_pattern": "matched components have poor waveform agreement",
        "secondary_diagnostics": (
            "imf_recovery_assignment_pairs",
            "true_component_splitting_max",
            "estimated_component_merging_max",
            "inter_imf_entanglement_index",
            "frequency_overlap_max_offdiag",
            "matched_component_if_absolute_error",
            "matched_component_if_valid_fraction",
        ),
        "mechanism_question": (
            "Are low matched-component correlations accompanied by true-component "
            "splitting, estimated-component merging, component entanglement, or "
            "IF trajectory distortion in applicable regimes?"
        ),
        "claim_boundary": "V5.40 IF diagnostics are secondary and only applicable to IF-defined signals.",
    },
    {
        "primary_dimension": "component_recovery_quality",
        "primary_endpoint": "matched_component_nrmse",
        "weak_pattern": "matched components have large amplitude-sensitive error",
        "secondary_diagnostics": (
            "imf_recovery_assignment_pairs",
            "true_component_splitting_max",
            "estimated_component_merging_max",
            "missing_true_component_count",
            "unmatched_estimated_component_count",
            "spectral_leakage",
        ),
        "mechanism_question": (
            "Is amplitude-sensitive recovery error accompanied by allocation "
            "failures, missing/unmatched components, or diffuse spectral content?"
        ),
        "claim_boundary": "No oracle rescaling is applied; diagnostics do not remove amplitude error.",
    },
    {
        "primary_dimension": "component_set_fidelity",
        "primary_endpoint": "relative_decomposition_count_error",
        "weak_pattern": "wrong effective component count",
        "secondary_diagnostics": (
            "imf_count",
            "effective_imf_count",
            "over_decomposition_penalty",
            "under_decomposition_index",
            "unmatched_estimated_component_count",
            "missing_true_component_count",
        ),
        "mechanism_question": (
            "Does count error reflect raw component proliferation, effective "
            "high-energy over-decomposition, or under-decomposition?"
        ),
        "claim_boundary": "RCCE is primary; raw/effective counts are explanatory diagnostics.",
    },
    {
        "primary_dimension": "component_set_fidelity",
        "primary_endpoint": "missing_true_component_energy_ratio",
        "weak_pattern": "important truth energy is not recovered",
        "secondary_diagnostics": (
            "missing_true_component_count",
            "true_component_splitting_max",
            "estimated_component_merging_max",
            "under_decomposition_index",
            "transient_smearing_index",
        ),
        "mechanism_question": (
            "Is missing truth energy accompanied by under-decomposition, merging, "
            "fragmentation, or loss of localized/transient structure?"
        ),
        "claim_boundary": "Missing-energy is an outcome; split/merge diagnostics explain how it occurs.",
    },
    {
        "primary_dimension": "component_set_fidelity",
        "primary_endpoint": "spurious_mode_energy_ratio",
        "weak_pattern": "substantial unmatched estimated-mode energy",
        "secondary_diagnostics": (
            "unmatched_estimated_component_count",
            "imf_count",
            "effective_imf_count",
            "over_decomposition_penalty",
            "spectral_leakage",
            "frequency_overlap_max_offdiag",
        ),
        "mechanism_question": (
            "Is spurious energy accompanied by many unmatched modes, over-decomposition, "
            "or spectrally diffuse / overlapping components?"
        ),
        "claim_boundary": "Spurious energy is primary; unmatched counts and spectral diagnostics explain severity/source.",
    },
    {
        "primary_dimension": "noise_separation",
        "primary_endpoint": "noise_capture_corr",
        "weak_pattern": "estimated noise has poor structural correlation with true noise or is non-computable",
        "secondary_diagnostics": (
            "protocol_zero_variance_noise_flag",
            "protocol_noise_reason_code",
            "noise_energy_ratio",
            "remaining_noise_energy_ratio",
            "noise_energy_bias_class",
            "signal_leakage_degenerate_flag",
        ),
        "mechanism_question": (
            "Is poor/non-computable noise correlation accompanied by zero estimated noise, "
            "under-recovery, over-recovery, or residual noise left unexplained?"
        ),
        "claim_boundary": "Energy diagnostics explain correlation blind spots but do not replace noise_capture_corr.",
    },
    {
        "primary_dimension": "noise_separation",
        "primary_endpoint": "noise_energy_log_error",
        "weak_pattern": "noise magnitude / energy is poorly calibrated",
        "secondary_diagnostics": (
            "noise_energy_ratio",
            "remaining_noise_energy_ratio",
            "noise_energy_bias_class",
            "protocol_estimated_noise_energy",
            "protocol_zero_variance_noise_flag",
        ),
        "mechanism_question": (
            "Is the method under-recovering or over-recovering noise energy, and "
            "how much true noise remains unexplained?"
        ),
        "claim_boundary": "Directional ratio/bias diagnostics explain the symmetric log-error primary endpoint.",
    },
    {
        "primary_dimension": "noise_separation",
        "primary_endpoint": "signal_leakage_into_noise",
        "weak_pattern": "true signal energy leaks into estimated noise",
        "secondary_diagnostics": (
            "signal_leakage_degenerate_flag",
            "clean_region_signal_leakage",
            "inter_imf_entanglement_index",
            "frequency_overlap_max_offdiag",
            "protocol_selected_fraction",
            "protocol_selected_component_indices",
        ),
        "mechanism_question": (
            "Is signal leakage accompanied by degenerate estimated noise, component "
            "entanglement, frequency overlap, or aggressive component selection?"
        ),
        "claim_boundary": "Selection diagnostics are protocol explanations, not primary outcomes.",
    },
    {
        "primary_dimension": "contamination_resistance",
        "primary_endpoint": "contaminated_region_nmse",
        "weak_pattern": "poor reconstruction at contaminated locations",
        "secondary_diagnostics": (
            "contaminated_fraction",
            "contaminated_point_count",
            "contaminated_region_preservation_score",
            "noise_energy_bias_class",
            "remaining_noise_energy_ratio",
        ),
        "mechanism_secondary_diagnostics": (
            "contaminated_region_preservation_score",
            "noise_energy_bias_class",
            "remaining_noise_energy_ratio",
        ),
        "validity_context_diagnostics": (
            "contaminated_fraction",
            "contaminated_point_count",
        ),
        "mechanism_question": (
            "Is direct contamination failure accompanied by contamination density, "
            "failure to capture contamination energy, or residual contamination?"
        ),
        "claim_boundary": "Applicable only in regimes with explicit contamination masks.",
    },
    {
        "primary_dimension": "contamination_resistance",
        "primary_endpoint": "clean_region_nmse",
        "weak_pattern": "clean observations are damaged while denoising",
        "secondary_diagnostics": (
            "clean_region_signal_leakage",
            "clean_region_signal_leakage_score",
            "signal_leakage_into_noise",
            "transient_smearing_index",
            "inter_imf_entanglement_index",
        ),
        "mechanism_secondary_diagnostics": (
            "clean_region_signal_leakage",
            "clean_region_signal_leakage_score",
            "signal_leakage_into_noise",
            "transient_smearing_index",
            "inter_imf_entanglement_index",
        ),
        "validity_context_diagnostics": (
            "contaminated_fraction",
            "contaminated_point_count",
        ),
        "mechanism_question": (
            "Is clean-region damage accompanied by collateral signal removal, transient "
            "smearing, or component entanglement?"
        ),
        "claim_boundary": "Clean-region preservation is evaluated against clean truth, not observed contaminated values.",
    },
    {
        "primary_dimension": "contamination_resistance",
        "primary_endpoint": "normalized_contamination_spillover_loss",
        "weak_pattern": "contamination influence spreads into neighboring clean samples",
        "secondary_diagnostics": (
            "contamination_spillover_error",
            "contamination_spillover_score",
            "spillover_radius",
            "clean_region_nmse",
            "transient_smearing_index",
        ),
        "mechanism_secondary_diagnostics": (
            "contamination_spillover_error",
            "contamination_spillover_score",
            "transient_smearing_index",
        ),
        "validity_context_diagnostics": (
            "spillover_radius",
        ),
        "mechanism_question": (
            "Is normalized spillover driven by raw local MSE, neighborhood radius, "
            "broader clean-region damage, or localized transient smearing?"
        ),
        "claim_boundary": "Raw spillover error is diagnostic; normalized spillover loss is the primary cross-case endpoint.",
    },
)


def _inventory_rows():
    primary_metrics = {
        metric
        for metrics in V528_PRIMARY_ENDPOINTS_BY_DIMENSION.values()
        for metric in metrics
    }
    canonical = {
        metric
        for metrics in CANONICAL_DIAGNOSTIC_TAXONOMY_BY_GROUP.values()
        for metric in metrics
    }
    rows = []
    for group, spec in SECONDARY_DIAGNOSTIC_INVENTORY.items():
        for metric in spec["diagnostics"]:
            role_class = METRIC_ROLE_CLASS_OVERRIDES.get(metric, spec["role_class"])
            rows.append({
                "closure_version": V541_CLOSURE_VERSION,
                "role_class": role_class,
                "role_class_question": SECONDARY_ROLE_TAXONOMY.get(role_class, {}).get("question", ""),
                "secondary_group": group,
                "metric": metric,
                "role": spec["role"],
                "canonical_v525_diagnostic": metric in canonical,
                "v528_primary_endpoint": metric in primary_metrics,
                "count_as_secondary_scientific_diagnostic": metric not in primary_metrics,
            })
    return rows


def _role_taxonomy_rows():
    rows = []
    for role_class, spec in SECONDARY_ROLE_TAXONOMY.items():
        rows.append({
            "closure_version": V541_CLOSURE_VERSION,
            "role_class": role_class,
            "question": spec["question"],
            "interpretation_rule": spec["interpretation_rule"],
        })
    return rows


def _closure_rows():
    rows = []
    for item in PRIMARY_TO_SECONDARY_CLOSURE:
        rows.append({
            "closure_version": V541_CLOSURE_VERSION,
            "primary_dimension": item["primary_dimension"],
            "primary_endpoint": item["primary_endpoint"],
            "weak_pattern": item["weak_pattern"],
            "secondary_diagnostics": ";".join(item["secondary_diagnostics"]),
            "mechanism_secondary_diagnostics": ";".join(item.get("mechanism_secondary_diagnostics", ())),
            "validity_context_diagnostics": ";".join(item.get("validity_context_diagnostics", ())),
            "mechanism_question": item["mechanism_question"],
            "claim_boundary": item["claim_boundary"],
        })
    return rows


def run_v541_primary_secondary_closure(output_root):
    output_root = ensure_dir(output_root)
    dashboard = {
        "closure_version": V541_CLOSURE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "protocol_mapping_frozen_not_performance_bearing",
        "primary_schema_source": "V5.28_normalized_spillover_primary_revision",
        "primary_endpoint_count": sum(len(v) for v in V528_PRIMARY_ENDPOINTS_BY_DIMENSION.values()),
        "primary_dimensions": V528_PRIMARY_ENDPOINTS_BY_DIMENSION,
        "secondary_inventory_groups": list(SECONDARY_DIAGNOSTIC_INVENTORY),
        "n_secondary_inventory_rows": len(_inventory_rows()),
        "n_primary_to_secondary_routes": len(PRIMARY_TO_SECONDARY_CLOSURE),
        "principle": (
            "Primary endpoints define outcome evidence. Secondary diagnostics "
            "are used only after a primary pattern is observed to indicate "
            "mechanisms consistent with that pattern, qualify scientific validity "
            "or definability, or check computational reliability; they are not "
            "additional ranking-bearing endpoints."
        ),
        "secondary_role_taxonomy": SECONDARY_ROLE_TAXONOMY,
        "no_metric_proliferation_rule": (
            "Do not add a new secondary diagnostic unless it explains a primary "
            "endpoint failure mode not covered by this closure map."
        ),
        "tf_if_boundary": (
            "V5.40 truth-aware IF diagnostics remain secondary, signal-specific, "
            "and non-primary unless a future pre-results schema explicitly adds "
            "a Time-Frequency Fidelity primary construct."
        ),
        "outputs": {
            "role_taxonomy": "secondary_diagnostic_role_taxonomy.csv",
            "inventory": "secondary_diagnostics_inventory.csv",
            "closure": "primary_to_secondary_diagnostic_closure.csv",
            "dashboard": "primary_secondary_diagnostic_closure_dashboard.json",
        },
    }
    write_csv(_role_taxonomy_rows(), output_root / "secondary_diagnostic_role_taxonomy.csv")
    write_csv(_inventory_rows(), output_root / "secondary_diagnostics_inventory.csv")
    write_csv(_closure_rows(), output_root / "primary_to_secondary_diagnostic_closure.csv")
    write_json(dashboard, output_root / "primary_secondary_diagnostic_closure_dashboard.json")
    return dashboard
