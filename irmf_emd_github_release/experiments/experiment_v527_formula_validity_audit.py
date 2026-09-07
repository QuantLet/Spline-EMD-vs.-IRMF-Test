#!/usr/bin/python
# coding: UTF-8

"""Formula-level validity audit for frozen V5.27 primary endpoints."""

from pathlib import Path

import numpy as np

from diagnostics.shared_physical_diagnostics import (
    EPS,
    NOISE_ENERGY_EPS,
    contamination_region_diagnostics,
    noise_capture_diagnostics,
    signal_leakage_into_noise_diagnostics,
)
from project_config import (
    V527_CONTAMINATION_RESISTANCE_CONSTRUCTS,
    V527_NOISE_SEPARATION_CONSTRUCTS,
    V527_PRIMARY_ENDPOINTS_BY_DIMENSION,
    V527_PRIMARY_METRIC_CODE_FIELD_MAP,
    V527_PRIMARY_SCHEMA_VERSION,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


def _finite(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except Exception:
        return np.nan


def _formula_rows():
    return [
        {
            "metric": "denoise_nmse",
            "construct": "overall_signal_reconstruction_error",
            "formula": "sum((X_hat - X)^2) / (sum(X^2) + eps)",
            "code_field": V527_PRIMARY_METRIC_CODE_FIELD_MAP["denoise_nmse"],
            "formula_validity_status": "established_metric_adopted",
            "notes": "Established normalized reconstruction error.",
        },
        {
            "metric": "denoise_corr",
            "construct": "overall_signal_shape_agreement",
            "formula": "corr(X_hat, X)",
            "code_field": V527_PRIMARY_METRIC_CODE_FIELD_MAP["denoise_corr"],
            "formula_validity_status": "established_metric_adopted",
            "notes": "Established correlation-based reconstruction agreement.",
        },
        {
            "metric": "matched_component_corr",
            "construct": "matched_component_waveform_shape_fidelity",
            "formula": "mean_{(i,j) in M} |corr(c_hat_i, s_j)|, M = Hungarian(1 - |corr|)",
            "code_field": V527_PRIMARY_METRIC_CODE_FIELD_MAP["matched_component_corr"],
            "formula_validity_status": "passed_prior_matching_formula_audit",
            "notes": "Publication-layer name for imf_recovery_corr.",
        },
        {
            "metric": "matched_component_nrmse",
            "construct": "matched_component_waveform_error_including_amplitude",
            "formula": "mean_{(i,j) in M} ||c_hat_i - s_j||_2 / (||s_j||_2 + eps)",
            "code_field": V527_PRIMARY_METRIC_CODE_FIELD_MAP["matched_component_nrmse"],
            "formula_validity_status": "passed_prior_matching_formula_audit",
            "notes": "No post-hoc oracle amplitude rescaling.",
        },
        {
            "metric": "relative_decomposition_count_error",
            "construct": "component_set_cardinality_consistency",
            "formula": "|K_eff_est - K_true| / K_true",
            "code_field": V527_PRIMARY_METRIC_CODE_FIELD_MAP["relative_decomposition_count_error"],
            "formula_validity_status": "passed_prior_component_set_audit",
            "notes": "Backward-compatible source field is decomposition_count_error.",
        },
        {
            "metric": "missing_true_component_energy_ratio",
            "construct": "component_set_completeness",
            "formula": "energy of unmatched true components / total true-component energy",
            "code_field": V527_PRIMARY_METRIC_CODE_FIELD_MAP["missing_true_component_energy_ratio"],
            "formula_validity_status": "passed_prior_component_set_audit",
            "notes": "Truth-aware synthetic/semi-synthetic endpoint.",
        },
        {
            "metric": "spurious_mode_energy_ratio",
            "construct": "component_set_purity",
            "formula": "energy of unmatched estimated components / total estimated-component energy",
            "code_field": V527_PRIMARY_METRIC_CODE_FIELD_MAP["spurious_mode_energy_ratio"],
            "formula_validity_status": "passed_prior_component_set_audit",
            "notes": "Truth-aware synthetic/semi-synthetic endpoint.",
        },
        {
            "metric": "noise_capture_corr",
            "construct": V527_NOISE_SEPARATION_CONSTRUCTS["noise_capture_corr"],
            "formula": "corr(eps_hat, eps), eps = Y - X",
            "code_field": V527_PRIMARY_METRIC_CODE_FIELD_MAP["noise_capture_corr"],
            "formula_validity_status": "passed_formula_audit",
            "notes": "Shape/identity metric; intentionally scale-insensitive.",
        },
        {
            "metric": "noise_energy_log_error",
            "construct": V527_NOISE_SEPARATION_CONSTRUCTS["noise_energy_log_error"],
            "formula": "abs(log((||eps_hat||^2 + eps) / (||eps||^2 + eps)))",
            "code_field": V527_PRIMARY_METRIC_CODE_FIELD_MAP["noise_energy_log_error"],
            "formula_validity_status": "passed_formula_audit",
            "notes": f"Uses frozen NOISE_ENERGY_EPS={NOISE_ENERGY_EPS}.",
        },
        {
            "metric": "signal_leakage_into_noise",
            "construct": V527_NOISE_SEPARATION_CONSTRUCTS["signal_leakage_into_noise"],
            "formula": "||P_S eps_hat||^2 / (||eps_hat||^2 + eps), S = span(true components)",
            "code_field": V527_PRIMARY_METRIC_CODE_FIELD_MAP["signal_leakage_into_noise"],
            "formula_validity_status": "passed_formula_audit",
            "notes": "Near-zero eps_hat is marked non-computable with degenerate flag.",
        },
        {
            "metric": "contaminated_region_nmse",
            "construct": V527_CONTAMINATION_RESISTANCE_CONSTRUCTS["contaminated_region_nmse"],
            "formula": "sum_{t in C}(X_hat_t - X_t)^2 / (sum_{t in C} X_t^2 + eps)",
            "code_field": V527_PRIMARY_METRIC_CODE_FIELD_MAP["contaminated_region_nmse"],
            "formula_validity_status": "passed_formula_audit",
            "notes": "Evaluates clean-truth recovery inside contaminated locations.",
        },
        {
            "metric": "clean_region_nmse",
            "construct": V527_CONTAMINATION_RESISTANCE_CONSTRUCTS["clean_region_nmse"],
            "formula": "sum_{t notin C}(X_hat_t - X_t)^2 / (sum_{t notin C} X_t^2 + eps)",
            "code_field": V527_PRIMARY_METRIC_CODE_FIELD_MAP["clean_region_nmse"],
            "formula_validity_status": "passed_formula_audit",
            "notes": "Evaluates collateral damage outside contaminated locations.",
        },
        {
            "metric": "contamination_spillover_error",
            "construct": V527_CONTAMINATION_RESISTANCE_CONSTRUCTS["contamination_spillover_error"],
            "formula": "mean_{t in expand(C, radius) \\ C}(X_hat_t - X_t)^2",
            "code_field": V527_PRIMARY_METRIC_CODE_FIELD_MAP["contamination_spillover_error"],
            "formula_validity_status": "passed_with_scale_note",
            "notes": (
                "Raw local MSE, not NMSE. Valid for within-cell method ranking; "
                "cross-case mean interpretation should be treated as scale-dependent."
            ),
        },
    ]


def _noise_formula_test_rows():
    n = 256
    t = np.linspace(0, 1, n, endpoint=False)
    x = np.sin(2 * np.pi * 3 * t)
    noise = np.cos(2 * np.pi * 11 * t)
    y = x + noise
    true_components = np.asarray([x])
    cases = [
        ("perfect_noise", noise),
        ("under_scaled_noise", 0.1 * noise),
        ("over_scaled_noise", 2.0 * noise),
        ("signal_leakage_added", noise + 0.5 * x),
        ("zero_estimated_noise", np.zeros_like(noise)),
    ]
    rows = []
    for case_id, eps_hat in cases:
        cap = noise_capture_diagnostics(y, x, residual=eps_hat, estimated_noise=eps_hat)
        leak = signal_leakage_into_noise_diagnostics(
            y,
            x,
            residual=eps_hat,
            true_components=true_components,
            estimated_noise=eps_hat,
        )
        rows.append({
            "case_id": case_id,
            "noise_capture_corr": _finite(cap["noise_capture_corr"]),
            "noise_energy_log_error": _finite(cap["noise_energy_log_error"]),
            "noise_energy_ratio": _finite(cap["noise_energy_ratio"]),
            "signal_leakage_into_noise": _finite(leak["signal_leakage_into_noise"]),
            "signal_leakage_degenerate_flag": bool(leak["signal_leakage_degenerate_flag"]),
            "expected_formula_behavior": {
                "perfect_noise": "identity good, energy good, no signal leakage",
                "under_scaled_noise": "identity good, energy error high, no signal leakage",
                "over_scaled_noise": "identity good, energy error high, no signal leakage",
                "signal_leakage_added": "identity remains high, signal leakage increases",
                "zero_estimated_noise": "correlation/leakage non-computable; energy error finite",
            }[case_id],
        })
    return rows


def _contamination_formula_test_rows():
    n = 160
    t = np.linspace(0, 1, n, endpoint=False)
    x = 1.0 + 0.3 * np.sin(2 * np.pi * 4 * t)
    y = x.copy()
    mask = np.zeros(n, dtype=bool)
    mask[[45, 95]] = True
    y[mask] += 5.0

    spill = np.zeros(n, dtype=bool)
    for i in np.flatnonzero(mask):
        spill[max(0, i - 5):min(n, i + 6)] = True
    spill &= ~mask

    rec_perfect = x.copy()
    rec_contamination_remains = x.copy()
    rec_contamination_remains[mask] += 4.0
    rec_clean_damaged = x.copy()
    rec_clean_damaged[~mask] += 0.35
    rec_spillover = x.copy()
    rec_spillover[spill] += 0.8

    cases = [
        ("perfect_robustness", rec_perfect),
        ("contamination_remains_clean_ok", rec_contamination_remains),
        ("contamination_removed_clean_damaged", rec_clean_damaged),
        ("localized_spillover_only", rec_spillover),
    ]
    rows = []
    for case_id, rec in cases:
        out = contamination_region_diagnostics(
            y,
            x,
            imfs=np.empty((0, n)),
            residual=np.zeros(n),
            contamination_mask=mask,
            spillover_radius=5,
            reconstructed=rec,
        )
        rows.append({
            "case_id": case_id,
            "contaminated_region_nmse": _finite(out["contaminated_region_nmse"]),
            "clean_region_nmse": _finite(out["clean_region_nmse"]),
            "contamination_spillover_error": _finite(out["contamination_spillover_error"]),
            "spillover_radius": int(out["spillover_radius"]),
            "contamination_region_applicable": bool(out["contamination_region_applicable"]),
            "expected_formula_behavior": {
                "perfect_robustness": "all three contamination endpoints near zero",
                "contamination_remains_clean_ok": "contaminated-region error high, clean/spillover near zero",
                "contamination_removed_clean_damaged": "clean-region error high; contaminated-region error near zero",
                "localized_spillover_only": "spillover error high with contaminated-region error near zero",
            }[case_id],
        })
    return rows


def _status(noise_rows, contamination_rows):
    by_noise = {r["case_id"]: r for r in noise_rows}
    by_cont = {r["case_id"]: r for r in contamination_rows}
    checks = {
        "noise_under_scaled_separates_identity_from_magnitude": (
            by_noise["under_scaled_noise"]["noise_capture_corr"] > 0.99
            and by_noise["under_scaled_noise"]["noise_energy_log_error"] > 4.0
        ),
        "noise_signal_leakage_detected": (
            by_noise["signal_leakage_added"]["signal_leakage_into_noise"] > 0.1
        ),
        "zero_estimated_noise_degenerate_flagged": (
            bool(by_noise["zero_estimated_noise"]["signal_leakage_degenerate_flag"])
            and by_noise["zero_estimated_noise"]["noise_energy_log_error"] > 1.0
        ),
        "contaminated_region_error_is_mask_specific": (
            by_cont["contamination_remains_clean_ok"]["contaminated_region_nmse"] > 1.0
            and by_cont["contamination_remains_clean_ok"]["clean_region_nmse"] < 1e-10
        ),
        "clean_region_error_is_mask_complement_specific": (
            by_cont["contamination_removed_clean_damaged"]["clean_region_nmse"] > 0.01
            and by_cont["contamination_removed_clean_damaged"]["contaminated_region_nmse"] < 1e-10
        ),
        "spillover_error_is_neighborhood_specific": (
            by_cont["localized_spillover_only"]["contamination_spillover_error"] > 0.1
            and by_cont["localized_spillover_only"]["contaminated_region_nmse"] < 1e-10
        ),
    }
    return checks, "passed" if all(checks.values()) else "requires_review"


def run_v527_formula_validity_audit(output_root):
    output_root = ensure_dir(output_root)
    formula_rows = _formula_rows()
    noise_rows = _noise_formula_test_rows()
    contamination_rows = _contamination_formula_test_rows()
    checks, status = _status(noise_rows, contamination_rows)
    dashboard = {
        "stage": "v527_formula_level_validity_audit",
        "schema_version": V527_PRIMARY_SCHEMA_VERSION,
        "schema_status": "frozen",
        "audit_status": status,
        "formula_validity_claim": (
            "This audit supports formula-to-construct validity for V5.27 "
            "primary endpoints. It does not claim that the endpoints are "
            "unique or globally optimal."
        ),
        "primary_endpoints_by_dimension": V527_PRIMARY_ENDPOINTS_BY_DIMENSION,
        "formula_level_checks": checks,
        "constants": {
            "EPS": EPS,
            "NOISE_ENERGY_EPS": NOISE_ENERGY_EPS,
            "signal_leakage_residual_energy_eps_multiplier": 1e-10,
            "spillover_radius_default": 5,
        },
        "scale_notes": {
            "contamination_spillover_error": (
                "Raw local MSE on the spillover neighborhood. This is valid for "
                "within-cell method ranking because methods share the same case "
                "scale; cross-case mean interpretation should retain the scale note."
            )
        },
    }
    write_csv(formula_rows, output_root / "v527_primary_formula_validity_audit.csv")
    write_csv(noise_rows, output_root / "v527_noise_formula_sanity_cases.csv")
    write_csv(contamination_rows, output_root / "v527_contamination_formula_sanity_cases.csv")
    write_json(dashboard, output_root / "v527_formula_validity_audit_dashboard.json")
    return dashboard
