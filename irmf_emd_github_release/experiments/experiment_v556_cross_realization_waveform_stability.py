#!/usr/bin/python
# coding: UTF-8

"""V5.56 supplementary cross-realization waveform-stability protocol.

This module freezes the execution specification for a future representative
subset rerun that persists component waveforms.  It intentionally does not run
the heavy waveform-level experiment by default.
"""

from datetime import datetime, timezone
from math import comb

import numpy as np

from project_config import (
    DEFAULT_N,
    V556_WAVEFORM_STABILITY_LOW_SIMILARITY_PRIMARY_THRESHOLD,
    V556_WAVEFORM_STABILITY_LOW_SIMILARITY_THRESHOLD_SENSITIVITY,
    V556_WAVEFORM_STABILITY_METHODS,
    V556_WAVEFORM_STABILITY_NOISES,
    V556_WAVEFORM_STABILITY_SEEDS,
    V556_WAVEFORM_STABILITY_SIGNALS,
    V556_WAVEFORM_STABILITY_TARGET_SNR_DB_LEVELS,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


V556_CROSS_REALIZATION_WAVEFORM_STABILITY_VERSION = (
    "V5.56_cross_realization_waveform_stability_protocol_cross_realization_hungarian_matching"
)

EPS = 1e-12


def _safe_corr(a, b):
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size != b.size or a.size < 3:
        return np.nan
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return np.nan
    if float(np.std(a)) <= EPS or float(np.std(b)) <= EPS:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _symmetric_nrmse(a, b):
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size != b.size or a.size == 0:
        return np.nan
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return np.nan
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    denom = 0.5 * (
        float(np.sqrt(np.mean(a ** 2)))
        + float(np.sqrt(np.mean(b ** 2)))
    )
    return float(rmse / max(denom, EPS))


def _vector_angle_degrees(a, b):
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size != b.size or a.size == 0:
        return np.nan
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return np.nan
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= EPS:
        return np.nan
    cos = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def match_component_sets(a_components, b_components, objective="abs_corr"):
    """Hungarian one-to-one matching between two estimated component sets.

    This helper is intentionally truth-agnostic.  V5.56 uses it for
    cross-realization matching:

        estimated components from seed r  <->  estimated components from seed r'

    The matching objective maximizes absolute waveform correlation by default,
    because IMF sign flips should not by themselves imply poor waveform
    reproducibility.  Signed correlation is still returned for interpretation.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception as exc:  # pragma: no cover - depends on optional scipy
        return {
            "matched_pairs": [],
            "matching_warning": f"scipy_linear_sum_assignment_unavailable: {exc}",
            "n_components_a": int(len(a_components) if a_components is not None else 0),
            "n_components_b": int(len(b_components) if b_components is not None else 0),
            "n_matched_pairs": 0,
            "matched_coverage_min_fraction": 0.0,
            "component_count_agreement": False,
        }
    a = np.asarray(a_components, dtype=float)
    b = np.asarray(b_components, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        return {
            "matched_pairs": [],
            "matching_warning": "invalid_component_array_shape",
            "n_components_a": int(a.shape[0]) if a.ndim == 2 else 0,
            "n_components_b": int(b.shape[0]) if b.ndim == 2 else 0,
            "n_matched_pairs": 0,
            "matched_coverage_min_fraction": 0.0,
            "component_count_agreement": False,
        }
    n_a, n_b = int(a.shape[0]), int(b.shape[0])
    if n_a == 0 or n_b == 0:
        return {
            "matched_pairs": [],
            "matching_warning": "empty_component_set",
            "n_components_a": n_a,
            "n_components_b": n_b,
            "n_matched_pairs": 0,
            "matched_coverage_min_fraction": 0.0,
            "component_count_agreement": bool(n_a == n_b),
        }
    signed_corr = np.full((n_a, n_b), np.nan, dtype=float)
    abs_corr = np.full((n_a, n_b), np.nan, dtype=float)
    for i in range(n_a):
        for j in range(n_b):
            c = _safe_corr(a[i], b[j])
            signed_corr[i, j] = c
            abs_corr[i, j] = abs(c) if np.isfinite(c) else np.nan
    if objective != "abs_corr":
        raise ValueError("V5.56 currently freezes objective='abs_corr'.")
    score = np.nan_to_num(abs_corr, nan=0.0, posinf=0.0, neginf=0.0)
    rows, cols = linear_sum_assignment(1.0 - score)
    matched_pairs = []
    for i, j in zip(rows, cols):
        matched_pairs.append({
            "component_index_a": int(i),
            "component_index_b": int(j),
            "signed_correlation": float(signed_corr[i, j]) if np.isfinite(signed_corr[i, j]) else np.nan,
            "absolute_correlation": float(abs_corr[i, j]) if np.isfinite(abs_corr[i, j]) else np.nan,
            "symmetric_nrmse": _symmetric_nrmse(a[i], b[j]),
            "vector_angle_degrees": _vector_angle_degrees(a[i], b[j]),
        })
    return {
        "matched_pairs": matched_pairs,
        "matching_warning": "",
        "matching_algorithm": "scipy.optimize.linear_sum_assignment",
        "matching_objective": "maximize_absolute_waveform_correlation",
        "n_components_a": n_a,
        "n_components_b": n_b,
        "n_matched_pairs": int(len(matched_pairs)),
        "matched_coverage_min_fraction": float(len(matched_pairs) / max(min(n_a, n_b), 1)),
        "matched_coverage_union_fraction": float(2 * len(matched_pairs) / max(n_a + n_b, 1)),
        "unmatched_component_count_a": int(max(n_a - len(matched_pairs), 0)),
        "unmatched_component_count_b": int(max(n_b - len(matched_pairs), 0)),
        "component_count_agreement": bool(n_a == n_b),
    }


def _design_rows():
    rows = []
    for signal in V556_WAVEFORM_STABILITY_SIGNALS:
        for noise in V556_WAVEFORM_STABILITY_NOISES:
            for snr in V556_WAVEFORM_STABILITY_TARGET_SNR_DB_LEVELS:
                for method in V556_WAVEFORM_STABILITY_METHODS:
                    rows.append({
                        "signal": signal,
                        "noise": noise,
                        "target_snr_db": float(snr),
                        "method": method,
                        "n_seeds": int(len(V556_WAVEFORM_STABILITY_SEEDS)),
                        "n_seed_pairs": int(comb(len(V556_WAVEFORM_STABILITY_SEEDS), 2)),
                        "pairing_design": "all_seed_pairs_within_signal_noise_snr_method_cell",
                    })
    return rows


def run_v556_cross_realization_waveform_stability_protocol(output_root):
    output_root = ensure_dir(output_root)
    n_signals = len(V556_WAVEFORM_STABILITY_SIGNALS)
    n_noises = len(V556_WAVEFORM_STABILITY_NOISES)
    n_snr = len(V556_WAVEFORM_STABILITY_TARGET_SNR_DB_LEVELS)
    n_seeds = len(V556_WAVEFORM_STABILITY_SEEDS)
    n_methods = len(V556_WAVEFORM_STABILITY_METHODS)
    n_cells = n_signals * n_noises * n_snr * n_methods
    n_method_evaluations = n_signals * n_noises * n_snr * n_seeds * n_methods
    n_seed_pairs_per_cell = comb(n_seeds, 2)
    n_seed_pair_comparisons = n_cells * n_seed_pairs_per_cell

    design_rows = _design_rows()
    protocol = {
        "schema_version": V556_CROSS_REALIZATION_WAVEFORM_STABILITY_VERSION,
        "created_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "module_status": "protocol_frozen_execution_pending",
        "execution_complete": False,
        "statistics_complete": False,
        "claim_authorized": False,
        "paper_role": "Supplementary diagnostic only.",
        "primary_endpoint_schema_changed": False,
        "ranking_bearing": False,
        "fixed_input_stochastic_repeatability": False,
        "estimand": (
            "waveform-level decomposition reproducibility across independent "
            "Monte Carlo noise realizations"
        ),
        "interpretation_boundary": (
            "V5.56 evaluates waveform-level decomposition reproducibility "
            "across Monte Carlo noise realizations on a pre-specified "
            "representative subset. It is supplementary, not primary, not "
            "ranking-bearing, and not a fixed-input stochastic repeatability "
            "test."
        ),
        "subset": {
            "signals": list(V556_WAVEFORM_STABILITY_SIGNALS),
            "noises": list(V556_WAVEFORM_STABILITY_NOISES),
            "target_snr_db_levels": list(V556_WAVEFORM_STABILITY_TARGET_SNR_DB_LEVELS),
            "seeds": list(V556_WAVEFORM_STABILITY_SEEDS),
            "methods": list(V556_WAVEFORM_STABILITY_METHODS),
            "n": int(DEFAULT_N),
        },
        "expected_counts": {
            "n_signal_noise_snr_method_cells": int(n_cells),
            "n_method_evaluations": int(n_method_evaluations),
            "n_seed_pairs_per_cell": int(n_seed_pairs_per_cell),
            "n_seed_pair_comparisons": int(n_seed_pair_comparisons),
        },
        "matching_protocol": {
            "within_cell_pairing": "all_pairs",
            "n_seed_pairs_per_signal_noise_snr_method_cell": int(n_seed_pairs_per_cell),
            "matching_object_kind": (
                "estimated_components_seed_r_vs_estimated_components_seed_r_prime"
            ),
            "not_truth_matching": True,
            "distinction_from_primary_matching": (
                "Primary matched-component recovery matches estimated components "
                "to ground-truth components and evaluates accuracy. V5.56 matches "
                "estimated components from two independent Monte Carlo noise "
                "realizations and evaluates decomposition stability."
            ),
            "component_correspondence": (
                "For each seed pair within the same signal/noise/SNR/method cell, "
                "match estimated component waveforms before computing similarity."
            ),
            "matching_algorithm": "Hungarian one-to-one assignment via scipy.optimize.linear_sum_assignment",
            "similarity_matrix": "S_ij = abs(corr(c_i_from_seed_r, c_j_from_seed_r_prime))",
            "cost_matrix": "C_ij = 1 - S_ij",
            "matching_objective": "maximize absolute waveform correlation",
            "primary_similarity_for_matching": "absolute_correlation",
            "signed_correlation_saved": True,
            "absolute_correlation_saved": True,
            "sign_flip_policy": (
                "A sign flip alone is not treated as waveform-instability for "
                "matching; signed correlation is saved as a secondary polarity/"
                "phase-consistency diagnostic."
            ),
            "component_count_mismatch_policy": (
                "If component counts differ, Hungarian matching returns "
                "min(K_r, K_r_prime) pairs and records unmatched counts, matched "
                "coverage, and component-count agreement."
            ),
            "reference_seed_policy": "none; no single seed is used as reference",
            "waveform_arrays_required": True,
            "compact_rows_sufficient": False,
            "helper_function": (
                "experiments.experiment_v556_cross_realization_waveform_stability."
                "match_component_sets"
            ),
        },
        "outputs_when_executed": {
            "primary_supplementary_diagnostic": (
                "within_cell_median_cross_realization_matched_component_absolute_correlation"
            ),
            "secondary_similarity_diagnostics": [
                "within_cell_median_cross_realization_matched_component_signed_correlation",
                "within_cell_median_cross_realization_symmetric_nrmse",
                "within_cell_median_cross_realization_vector_angle_degrees",
            ],
            "dispersion": "IQR_of_cross_realization_matched_component_absolute_correlation",
            "tail_diagnostic": "low_similarity_failure_rate",
            "consistency_check": "component_count_agreement",
            "coverage_diagnostics": [
                "matched_coverage_min_fraction",
                "matched_coverage_union_fraction",
                "unmatched_component_count_by_side",
            ],
            "threshold_sensitivity": list(
                V556_WAVEFORM_STABILITY_LOW_SIMILARITY_THRESHOLD_SENSITIVITY
            ),
            "primary_low_similarity_threshold": float(
                V556_WAVEFORM_STABILITY_LOW_SIMILARITY_PRIMARY_THRESHOLD
            ),
        },
        "artifact_policy": {
            "do_not_store_waveform_arrays_in_full_benchmark": True,
            "store_waveform_arrays_only_for_v556_subset": True,
            "reason": (
                "Cross-realization waveform matching requires full component "
                "arrays, while V5.54/V5.55 compact stability can be computed "
                "from full benchmark summaries."
            ),
        },
        "relationship_to_v554_v555": {
            "V5.54_V5.55": (
                "full-scope compact stability: cardinality stability and "
                "truth-conditioned recovery persistence"
            ),
            "V5.56": (
                "subset waveform-level stability: matched waveform "
                "reproducibility across noise realizations after cross-realization "
                "Hungarian component correspondence"
            ),
        },
    }
    metric_rows = [
        {
            "metric": "cross_realization_matched_component_absolute_correlation",
            "role": "primary_supplementary_similarity",
            "direction": "higher_is_better",
            "definition": (
                "Absolute waveform correlation after Hungarian matching between "
                "estimated component sets from two Monte Carlo noise realizations."
            ),
        },
        {
            "metric": "cross_realization_matched_component_signed_correlation",
            "role": "secondary_polarity_phase_diagnostic",
            "direction": "higher_is_better",
            "definition": (
                "Signed waveform correlation for the same matched pairs; saved "
                "to reveal polarity/phase inconsistency not used as the matching objective."
            ),
        },
        {
            "metric": "cross_realization_symmetric_nrmse",
            "role": "secondary_amplitude_sensitive_similarity",
            "direction": "lower_is_better",
            "definition": (
                "RMSE between two matched estimated components divided by the "
                "mean RMS of the two components; symmetric because neither "
                "realization is truth/reference."
            ),
        },
        {
            "metric": "cross_realization_vector_angle_degrees",
            "role": "secondary_geometric_similarity",
            "direction": "lower_is_better",
            "definition": "Vector angle between matched estimated component waveforms.",
        },
        {
            "metric": "matched_coverage_min_fraction",
            "role": "coverage",
            "direction": "higher_is_better",
            "definition": "Number of Hungarian matched pairs divided by min(K_r, K_r_prime).",
        },
        {
            "metric": "matched_coverage_union_fraction",
            "role": "coverage",
            "direction": "higher_is_better",
            "definition": "Twice the number of matched pairs divided by K_r + K_r_prime.",
        },
        {
            "metric": "component_count_agreement",
            "role": "consistency_check",
            "direction": "higher_is_better",
            "definition": "Indicator that the two realizations produced the same component count.",
        },
        {
            "metric": "low_similarity_failure_rate",
            "role": "tail_diagnostic",
            "direction": "lower_is_better",
            "definition": (
                "Fraction of matched pairs or seed-pair summaries below the "
                "pre-specified absolute-correlation threshold."
            ),
        },
    ]
    dashboard = {
        "schema_version": V556_CROSS_REALIZATION_WAVEFORM_STABILITY_VERSION,
        "module_status": "protocol_frozen_execution_pending",
        "no_algorithm_runs_performed": True,
        "claim_authorized": False,
        "expected_method_evaluations_when_executed": int(n_method_evaluations),
        "expected_seed_pair_comparisons_when_executed": int(n_seed_pair_comparisons),
        "protocol": protocol,
    }
    write_json(protocol, output_root / "v556_cross_realization_waveform_stability_protocol.json")
    write_json(dashboard, output_root / "v556_cross_realization_waveform_stability_dashboard.json")
    write_csv(design_rows, output_root / "v556_cross_realization_waveform_stability_design_manifest.csv")
    write_csv(metric_rows, output_root / "v556_cross_realization_waveform_stability_metric_definitions.csv")
    return dashboard
