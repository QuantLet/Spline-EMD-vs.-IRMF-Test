#!/usr/bin/python
# coding: UTF-8

"""V5.40 time-frequency / instantaneous-frequency secondary schema freeze."""

from datetime import datetime, timezone

from project_config import (
    TIME_FREQUENCY_SECONDARY_APPLICABLE_SIGNALS,
    TIME_FREQUENCY_SECONDARY_DIAGNOSTICS,
)
from experiments.paper_pipeline_utils import ensure_dir, write_json


V540_TIME_FREQUENCY_SCHEMA_VERSION = "V5.40_time_frequency_if_secondary_diagnostics_schema"
V540_ESTIMATOR_PROTOCOL_VERSION = "V5.40_IF_ESTIMATOR_PROTOCOL_V1.0"
V540_IF_EPS = 1e-12
V540_IF_MEDIAN_FILTER_KERNEL = 5
V540_IF_BOUNDARY_EXCLUSION_FRACTION = 0.05
V540_IF_MIN_BOUNDARY_EXCLUSION_SAMPLES = 5
V540_IF_RELATIVE_AMPLITUDE_FLOOR = 0.05
V540_IF_MIN_VALID_FRACTION_FOR_ERROR = 0.30


def run_v540_time_frequency_secondary_schema(output_root):
    output_root = ensure_dir(output_root)
    schema = {
        "schema_version": V540_TIME_FREQUENCY_SCHEMA_VERSION,
        "estimator_protocol_version": V540_ESTIMATOR_PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema_status": "secondary_schema_and_estimator_protocol_frozen_not_executed",
        "estimator_protocol_status": "frozen_not_executed",
        "execution_status": "not_executed",
        "benchmark_results_recomputed": False,
        "tf_if_results_computed": False,
        "primary_schema_changed": False,
        "primary_endpoint_count_unchanged": 13,
        "primary_ranking_bearing": False,
        "primary_claim_authorized": False,
        "primary_endpoint_policy": (
            "Do not add time-frequency / instantaneous-frequency endpoints to "
            "the current primary schema unless the manuscript claim is changed "
            "before a future primary-schema freeze."
        ),
        "scientific_role": (
            "Truth-aware secondary diagnostic for signals with well-defined "
            "instantaneous-frequency reference curves."
        ),
        "construct": {
            "name": "Time-Frequency Fidelity",
            "status": "secondary_signal_specific_diagnostic",
            "question": (
                "After component matching, how accurately is the local "
                "oscillatory frequency law recovered?"
            ),
            "not_equivalent_to": [
                "matched_component_corr",
                "matched_component_nrmse",
                "spectral_leakage",
                "frequency_overlap_max_offdiag",
                "instantaneous_frequency_smoothness",
            ],
        },
        "applicability": {
            "applicable_signals": list(TIME_FREQUENCY_SECONDARY_APPLICABLE_SIGNALS),
            "requires_true_frequency_curves": True,
            "requires_true_components": True,
            "requires_component_matching": True,
            "excluded_or_caution_signals": {
                "stationary_multi_sine": (
                    "constant-frequency references exist, but the primary "
                    "scientific value is lower than nonstationary IF regimes"
                ),
                "close_frequencies": (
                    "two close constant IF curves are better treated through "
                    "component-set and separation diagnostics unless a frequency "
                    "resolution claim is made"
                ),
                "impulsive_transient": (
                    "IF is only defined on very short active intervals; use "
                    "transient-specific diagnostics first"
                ),
                "intermittent_oscillation": (
                    "large inactive intervals make Hilbert IF validity "
                    "amplitude-mask dependent"
                ),
                "trend_plus_oscillation": "trend component has no oscillatory IF",
                "non_sinusoidal_periodic": (
                    "harmonic-rich waveform has several harmonic IF curves; not "
                    "a single carrier IF construct"
                ),
                "transient_train": (
                    "piecewise local IF is useful for mechanism plots but too "
                    "mask-dependent for primary endpoint status"
                ),
            },
        },
        "diagnostics": {
            "matched_component_if_absolute_error": {
                "direction": "lower_is_better",
                "formula": (
                    "sum_{j in matched components} sum_{t in V_j} "
                    "w_j(t) |fhat_j(t) - f_j(t)| / sum_{j,t in V_j} w_j(t)"
                ),
                "unit": "normalized-time frequency units matching synthetic references",
                "meaning": "absolute local IF trajectory error after component matching",
            },
            "matched_component_if_normalized_absolute_error": {
                "direction": "lower_is_better",
                "formula": (
                    "sum w_j(t) |fhat_j(t) - f_j(t)| / "
                    "(sum w_j(t) |f_j(t)| + epsilon)"
                ),
                "meaning": "scale-normalized IF trajectory error",
            },
            "matched_component_if_valid_fraction": {
                "direction": "higher_is_better",
                "formula": "number of valid IF samples / number of candidate IF samples",
                "meaning": "availability / mask adequacy diagnostic, not a performance endpoint",
            },
        },
        "estimation_protocol": {
            "status": "frozen_not_executed",
            "time_axis": {
                "domain": "synthetic_normalized_time",
                "sample_grid": "t_unit = linspace(0, 1, n, endpoint=False)",
                "if_units": "cycles_per_normalized_time_unit",
                "derivative_dt": "median(diff(t_unit))",
                "physical_fs_not_used_for_if_error": True,
            },
            "true_reference_protocol": {
                "source": "signal_bank.synthetic_signals.get_true_frequencies(signal_name, t_unit)",
                "component_indexing": (
                    "true-frequency curve index must match the true-component "
                    "index used by get_true_components for applicable signals"
                ),
                "truth_estimator": "none; use analytic reference IF curves supplied by the generator",
                "interpolation_to_if_grid": "linear interpolation onto t_if = 0.5 * (t[:-1] + t[1:])",
            },
            "component_matching_protocol": {
                "source": "existing primary matched-component assignment",
                "assignment_field": "imf_recovery_assignment_pairs",
                "assignment_rule": "Hungarian assignment with cost C_ij = 1 - |corr(estimated_i, true_j)|",
                "rematch_using_if_error": False,
                "unmatched_components_enter_error_average": False,
            },
            "estimated_if_protocol": {
                "method": "Hilbert analytic signal of each matched estimated component",
                "phase": "unwrap(angle(hilbert(estimated_component)))",
                "derivative": "first difference of unwrapped phase divided by 2*pi*dt",
                "if_grid": "midpoint grid t_if = 0.5 * (t[:-1] + t[1:])",
                "median_filter_kernel_samples": V540_IF_MEDIAN_FILTER_KERNEL,
                "median_filter_scope": "estimated IF only after phase derivative",
                "quantile_clipping": "none for truth-referenced IF error",
                "no_oracle_rescaling": True,
                "no_projection_or_frequency_alignment": True,
            },
            "validity_mask": {
                "boundary_exclusion_fraction_each_side": V540_IF_BOUNDARY_EXCLUSION_FRACTION,
                "minimum_boundary_exclusion_if_samples_each_side": V540_IF_MIN_BOUNDARY_EXCLUSION_SAMPLES,
                "relative_amplitude_floor": V540_IF_RELATIVE_AMPLITUDE_FLOOR,
                "amplitude_floor_rule": (
                    "valid only where both true and estimated analytic amplitudes "
                    "on the IF grid are at least relative_amplitude_floor times "
                    "their component-specific maximum amplitude"
                ),
                "finite_true_if_required": True,
                "finite_estimated_if_required": True,
                "minimum_valid_fraction_for_error": V540_IF_MIN_VALID_FRACTION_FOR_ERROR,
            },
            "aggregation_protocol": {
                "sample_weight": "true analytic amplitude squared on the IF midpoint grid",
                "component_weight": "implicit through pooled sample weights across matched components",
                "absolute_error": "weighted mean |estimated IF - true IF| over valid samples",
                "normalized_error_denominator": "weighted mean |true IF| over valid samples plus epsilon",
                "epsilon": V540_IF_EPS,
                "valid_fraction": (
                    "valid IF samples across matched components divided by candidate "
                    "IF samples after matching and before validity filtering"
                ),
            },
            "noncomputable_reason_codes": [
                "not_applicable_no_true_if_reference",
                "not_computable_no_matched_components",
                "not_computable_low_valid_fraction",
                "not_computable_hilbert_failure",
                "not_computable_nonfinite_reference",
            ],
        },
        "existing_diagnostic_boundary": {
            "spectral_leakage": "global spectral spreading / mechanism diagnostic",
            "frequency_overlap_max_offdiag": "inter-component spectral overlap diagnostic",
            "ifs": "estimated-component IF smoothness only; no truth reference",
            "why_new_secondary_metric_needed": (
                "Existing frequency diagnostics describe separation behavior but "
                "do not measure error against true IF trajectories."
            ),
        },
        "promotion_rule": {
            "may_become_primary_only_if": (
                "The manuscript makes a pre-results primary claim about improved "
                "time-varying oscillatory structure or instantaneous-frequency "
                "recovery, followed by a new schema qualification and freeze."
            ),
            "must_not_promote_because_results_look_good": True,
            "current_recommendation": "keep_secondary",
        },
        "planned_outputs": {
            "schema": "v540_time_frequency_secondary_schema.json",
            "dashboard": "v540_time_frequency_secondary_schema_dashboard.json",
            "future_if_executed": [
                "time_frequency_if_diagnostic_rows.csv",
                "time_frequency_if_summary_by_signal_method.csv",
                "time_frequency_if_validity_audit.json",
            ],
        },
        "execution_gate": {
            "requires_v530_algorithm_cube": True,
            "requires_frozen_irmf_parameters": True,
            "does_not_block_primary_statistics": True,
            "does_not_authorize_primary_claims": True,
        },
        "diagnostic_names": list(TIME_FREQUENCY_SECONDARY_DIAGNOSTICS),
    }
    write_json(schema, output_root / "v540_time_frequency_secondary_schema.json")
    write_json(schema, output_root / "v540_time_frequency_secondary_schema_dashboard.json")
    return schema
