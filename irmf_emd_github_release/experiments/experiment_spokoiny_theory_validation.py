#!/usr/bin/python
# coding: UTF-8

"""
Appendix A: Spokoiny-inspired theory-diagnostic validation.

This module does not claim to prove Spokoiny's non-asymptotic theory for the
implemented IRMF algorithm.  It provides a reproducible diagnostic bridge from
the theory flow used in the manuscript to quantities computed by the code:

    Smooth perturbed optimization
        -> Fisher/local estimator expansion
        -> IMF and residual expansion
        -> noise/operator propagation
        -> risk propagation
        -> theory-vs-empirical validation

The output is intentionally appendix-facing: representative cases, scalar
tables, scale-evolution rows, and correlation summaries.
"""

import numpy as np

from project_config import (
    DEFAULT_FS,
    DEFAULT_N,
    DEFAULT_SEED,
    GLOBAL_IRMF_PARAMS,
    SPOKOINY_THEORY_TRACE_PROBE_COUNT,
    SPOKOINY_THEORY_VALIDATION_CASES,
)
from experiments.experiment_utils import make_signal_noise_case, run_fixed_irmf_case
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


LOCAL_OBJECTIVE_DESCRIPTION = {
    "plain_text": (
        "For each location t_i and bandwidth h, IRMF computes a local robust "
        "M-estimator S_h(t_i) by minimizing a kernel-weighted Gaussian-smoothed "
        "median loss over the local neighborhood.  The bandwidth h controls the "
        "local neighborhood.  In the V4E paper pipeline, the smoothed-median "
        "loss width is fixed at H=1.0 following the IRMF methodology slides, "
        "rather than tuned as a free parameter."
    ),
    "latex": (
        r"\\widehat S_h(t_i)=\\arg\\min_{\\theta\\in\\mathbb R}"
        r"\\sum_j K\\left((t_i-t_j)/h\\right)\\rho_H(Y_j-\\theta),"
        r"\\quad "
        r"\\rho_H(u)=\\sqrt{2/\\pi}\\,H\\exp\\{-u^2/(2H^2)\\}"
        r"+u\\{2\\Phi(u/H)-1\\}."
    ),
    "implementation_note": (
        "The implementation solves the one-dimensional local estimating "
        "equation through local_m_estimator and records the empirical gradient "
        "and Hessian proxies at each IRMF scale, using rho'_H(u)=2 Phi(u/H)-1 "
        "and rho''_H(u)=(2/H) phi(u/H).  The reported V4E experiments fix "
        "H=1.0 before global parameter selection."
    ),
}


THEORY_FLOW_MAP = [
    {
        "theory_step": "Smooth perturbed optimization",
        "implementation_object": "local robust M-estimation with Gaussian-smoothed median loss",
        "code_objects": "strict_spokoiny_irmf.local_m_estimator; rho_spline_grad_hess",
        "diagnostic_output": "fixed H=1.0, h, min_support_points, smoothed-median gradient/Hessian summaries",
    },
    {
        "theory_step": "Fisher/local estimator expansion",
        "implementation_object": "local gradient/Hessian perturbation proxies",
        "code_objects": "diagnostics.irmf_local_theory_diagnostics",
        "diagnostic_output": "median_b0, tau3_b0, Wilks remainder proxy, Hessian positivity",
    },
    {
        "theory_step": "IMF expansion",
        "implementation_object": "multiscale residual decomposition induced by robust local smoothing",
        "code_objects": "strict_spokoiny_irmf residual_history, scale_history, imfs",
        "diagnostic_output": "IMF energy ratios and dominant frequencies",
    },
    {
        "theory_step": "Noise propagation",
        "implementation_object": "randomized operator-probe approximation to propagation behavior",
        "code_objects": "diagnostics.irmf_operator_diagnostics",
        "diagnostic_output": "Hutchinson trace proxy, operator norm proxy, contraction ratio",
    },
    {
        "theory_step": "Risk propagation",
        "implementation_object": "empirical clean-signal reconstruction risk by IRMF scale",
        "code_objects": "residual_history plus known synthetic X_clean",
        "diagnostic_output": "risk_by_scale, smooth-region MSE, structure-region MSE",
    },
    {
        "theory_step": "Theory-vs-empirical validation",
        "implementation_object": "case-level association between theory diagnostics and performance",
        "code_objects": "Appendix A correlation table",
        "diagnostic_output": "correlations with NMSE, reconstruction, contamination, and noise capture",
    },
]


def _finite_float(value, default=np.nan):
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except Exception:
        return default


def _safe_corr(x, y):
    x = np.asarray([_finite_float(v) for v in x], dtype=float)
    y = np.asarray([_finite_float(v) for v in y], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        return np.nan
    x = x[mask]
    y = y[mask]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _nanmean(values):
    arr = np.asarray([_finite_float(v) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else np.nan


def _nanstd(values):
    arr = np.asarray([_finite_float(v) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


def _safe_rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) == 0:
        return np.nan
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _dominant_frequency(x, fs):
    x = np.asarray(x, dtype=float)
    if len(x) < 3 or np.allclose(x, x[0]):
        return np.nan
    centered = x - np.mean(x)
    power = np.abs(np.fft.rfft(centered)) ** 2
    freqs = np.fft.rfftfreq(len(centered), d=1.0 / float(fs))
    if len(power) <= 1:
        return np.nan
    idx = int(np.argmax(power[1:]) + 1)
    return float(freqs[idx])


def _moving_variance(x, window):
    x = np.asarray(x, dtype=float)
    window = max(int(window), 3)
    kernel = np.ones(window, dtype=float) / float(window)
    mean = np.convolve(x, kernel, mode="same")
    mean_sq = np.convolve(x ** 2, kernel, mode="same")
    return np.maximum(mean_sq - mean ** 2, 0.0)


def _variance_envelope_metrics(residual, noise, window):
    residual_env = _moving_variance(residual, window)
    noise_env = _moving_variance(noise, window)
    corr = _safe_corr(residual_env, noise_env)
    nmse = float(
        np.sum((residual_env - noise_env) ** 2)
        / (np.sum(noise_env ** 2) + 1e-12)
    )
    return {
        "residual_variance_envelope_corr": corr,
        "residual_variance_envelope_nmse": nmse,
        "residual_variance_envelope_window": int(window),
    }


def _assignment(cost):
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(cost)
        return list(zip(rows, cols))
    except Exception:
        pairs = []
        used_i = set()
        used_j = set()
        while len(used_i) < cost.shape[0] and len(used_j) < cost.shape[1]:
            best = None
            best_val = np.inf
            for i in range(cost.shape[0]):
                if i in used_i:
                    continue
                for j in range(cost.shape[1]):
                    if j in used_j:
                        continue
                    if cost[i, j] < best_val:
                        best_val = cost[i, j]
                        best = (i, j)
            if best is None:
                break
            i, j = best
            pairs.append((i, j))
            used_i.add(i)
            used_j.add(j)
        return pairs


def _dilate_mask(mask, radius):
    mask = np.asarray(mask, dtype=bool)
    radius = int(max(radius, 0))
    if radius == 0 or not np.any(mask):
        return mask
    kernel = np.ones(2 * radius + 1, dtype=int)
    return np.convolve(mask.astype(int), kernel, mode="same") > 0


def _structure_masks(x_clean):
    x = np.asarray(x_clean, dtype=float)
    n = len(x)
    edge = max(int(round(0.10 * n)), 1)
    boundary = np.zeros(n, dtype=bool)
    boundary[:edge] = True
    boundary[-edge:] = True

    if n < 5:
        structure = ~boundary
    else:
        deriv = np.abs(np.gradient(x))
        threshold = np.quantile(deriv, 0.90)
        structure = _dilate_mask(deriv >= threshold, max(int(round(0.025 * n)), 1))
    smooth = ~(boundary | structure)
    if np.sum(smooth) < max(5, int(0.10 * n)):
        smooth = ~boundary
    return {
        "smooth": smooth,
        "structure": structure,
        "boundary": boundary,
    }


def _masked_mse(x_hat, x_true, mask):
    mask = np.asarray(mask, dtype=bool)
    if np.sum(mask) == 0:
        return np.nan
    err = np.asarray(x_hat, dtype=float)[mask] - np.asarray(x_true, dtype=float)[mask]
    return float(np.mean(err ** 2))


def _risk_by_scale(Y, X_clean, noise, residual_history):
    rows = []
    masks = _structure_masks(X_clean)
    total_signal_energy = float(np.mean(np.asarray(X_clean, dtype=float) ** 2) + 1e-12)
    noise_energy = float(np.mean(np.asarray(noise, dtype=float) ** 2) + 1e-12)

    for k, residual_k in enumerate(residual_history, start=1):
        residual_k = np.asarray(residual_k, dtype=float)
        recovered_k = np.asarray(Y, dtype=float) - residual_k
        err = recovered_k - np.asarray(X_clean, dtype=float)
        mse = float(np.mean(err ** 2))
        row = {
            "scale_index": k,
            "risk_mse": mse,
            "risk_nmse": float(mse / total_signal_energy),
            "risk_to_noise_energy_ratio": float(mse / noise_energy),
            "residual_energy_ratio_to_observed": float(
                np.sum(residual_k ** 2) / (np.sum(np.asarray(Y, dtype=float) ** 2) + 1e-12)
            ),
            "smooth_region_mse": _masked_mse(recovered_k, X_clean, masks["smooth"]),
            "structure_region_mse": _masked_mse(recovered_k, X_clean, masks["structure"]),
            "boundary_region_mse": _masked_mse(recovered_k, X_clean, masks["boundary"]),
            "smooth_region_fraction": float(np.mean(masks["smooth"])),
            "structure_region_fraction": float(np.mean(masks["structure"])),
            "boundary_region_fraction": float(np.mean(masks["boundary"])),
        }
        row["structure_to_smooth_mse_ratio"] = float(
            row["structure_region_mse"] / (row["smooth_region_mse"] + 1e-12)
        ) if np.isfinite(row["structure_region_mse"]) and np.isfinite(row["smooth_region_mse"]) else np.nan
        rows.append(row)
    return rows


def _imf_expansion_rows(imfs, fs):
    imfs = np.asarray(imfs, dtype=float)
    if imfs.ndim == 1:
        imfs = imfs[None, :]
    total_energy = float(np.sum(imfs ** 2) + 1e-12)
    rows = []
    for j, imf in enumerate(imfs, start=1):
        energy = float(np.sum(imf ** 2))
        rows.append({
            "imf_index": j,
            "imf_energy": energy,
            "imf_energy_ratio": float(energy / total_energy),
            "imf_dominant_frequency": _dominant_frequency(imf, fs),
        })
    return rows


def _imf_matching_rows(case_id, imfs, true_components, fs):
    if true_components is None:
        return []
    imfs = np.asarray(imfs, dtype=float)
    true_components = np.asarray(true_components, dtype=float)
    if imfs.ndim == 1:
        imfs = imfs[None, :]
    if true_components.ndim == 1:
        true_components = true_components[None, :]
    if imfs.size == 0 or true_components.size == 0 or imfs.shape[1] != true_components.shape[1]:
        return []

    cost = np.ones((imfs.shape[0], true_components.shape[0]), dtype=float)
    for i in range(imfs.shape[0]):
        for j in range(true_components.shape[0]):
            corr = _safe_corr(imfs[i], true_components[j])
            if np.isfinite(corr):
                cost[i, j] = 1.0 - abs(corr)

    imf_total_energy = float(np.sum(imfs ** 2) + 1e-12)
    true_total_energy = float(np.sum(true_components ** 2) + 1e-12)
    rows = []
    for i, j in _assignment(cost):
        corr = _safe_corr(imfs[i], true_components[j])
        rmse = _safe_rmse(imfs[i], true_components[j])
        denom = float(np.sqrt(np.mean(true_components[j] ** 2)) + 1e-12)
        rows.append({
            "case_id": case_id,
            "estimated_imf_index": int(i + 1),
            "true_component_index": int(j + 1),
            "signed_corr": corr,
            "abs_corr": abs(corr) if np.isfinite(corr) else np.nan,
            "rmse": rmse,
            "nrmse": float(rmse / denom) if np.isfinite(rmse) else np.nan,
            "estimated_imf_energy_ratio": float(np.sum(imfs[i] ** 2) / imf_total_energy),
            "true_component_energy_ratio": float(np.sum(true_components[j] ** 2) / true_total_energy),
            "estimated_imf_dominant_frequency": _dominant_frequency(imfs[i], fs),
            "true_component_dominant_frequency": _dominant_frequency(true_components[j], fs),
        })
    return rows


def _scale_evolution_rows(case_id, result, risk_rows):
    local = result.get("local_theory_diagnostics", {})
    operator = result.get("operator_diagnostics", {})
    evolution = result.get("operator_evolution", [])
    n_scales = max(
        len(local.get("bandwidths", [])),
        len(operator.get("bandwidths", [])),
        len(evolution),
        len(risk_rows),
    )
    rows = []
    for i in range(n_scales):
        row = {"case_id": case_id, "scale_index": i + 1}
        for key in [
            "bandwidths",
            "b0_local_median",
            "b0_local_q90",
            "tau3_b0_median",
            "tau3_b0_valid_ratio",
            "wilks_remainder_scale_median",
            "hessian_positive_ratio",
            "hessian_condition_proxy",
            "residual_energy_ratio",
        ]:
            values = local.get(key, [])
            if i < len(values):
                out_key = "h" if key == "bandwidths" else key
                row[out_key] = _finite_float(values[i])
        for key in ["noise_trace_norms", "operator_norms", "contraction_ratios"]:
            values = operator.get(key, [])
            if i < len(values):
                row[key] = _finite_float(values[i])
        if i < len(evolution):
            for key in [
                "trace_norm_proxy",
                "operator_norm_proxy",
                "contraction_ratio",
                "component_energy_ratio",
                "b0_scale_proxy",
            ]:
                row[f"evolution_{key}"] = _finite_float(evolution[i].get(key))
        if i < len(risk_rows):
            row.update({f"risk_{k}": v for k, v in risk_rows[i].items() if k != "scale_index"})
        rows.append(row)
    return rows


def _case_summary(case_id, signal_name, noise_name, sigma, case, result, risk_rows, imf_rows):
    final_risk = risk_rows[-1] if risk_rows else {}
    envelope = _variance_envelope_metrics(
        result.get("residual", np.zeros_like(case["noise"])),
        case["noise"],
        max(len(case["noise"]) // 20, 10),
    )
    return {
        "case_id": case_id,
        "signal": signal_name,
        "noise": noise_name,
        "sigma": sigma,
        "theory_validation_role": "Spokoiny-inspired diagnostic; not a formal theorem check",
        "h1": result.get("h1"),
        "a": result.get("a"),
        "h_min": result.get("h_min"),
        "H": result.get("H"),
        "boundary_mode": result.get("boundary_mode"),
        "imf_count": result.get("imf_count"),
        "median_b0": result.get("median_b0"),
        "max_b0": result.get("max_b0"),
        "final_tau3_b0_median": result.get("final_tau3_b0_median"),
        "final_tau3_b0_max": result.get("final_tau3_b0_max"),
        "final_tau3_valid_ratio": result.get("final_tau3_valid_ratio"),
        "final_wilks_remainder_median": result.get("final_wilks_remainder_median"),
        "final_hessian_positive_ratio": result.get("final_hessian_positive_ratio"),
        "final_hessian_condition": result.get("final_hessian_condition"),
        "local_theory_score": result.get("local_theory_score"),
        "final_trace_norm": result.get("final_trace_norm"),
        "final_operator_norm": result.get("final_operator_norm"),
        "final_contraction": result.get("final_contraction"),
        "operator_score": result.get("operator_score"),
        "operator_evolution_final_trace_norm": result.get("operator_evolution_final_trace_norm"),
        "operator_evolution_mean_contraction": result.get("operator_evolution_mean_contraction"),
        "operator_evolution_final_residual_energy": result.get("operator_evolution_final_residual_energy"),
        "final_risk_nmse": final_risk.get("risk_nmse"),
        "final_smooth_region_mse": final_risk.get("smooth_region_mse"),
        "final_structure_region_mse": final_risk.get("structure_region_mse"),
        "final_boundary_region_mse": final_risk.get("boundary_region_mse"),
        "final_structure_to_smooth_mse_ratio": final_risk.get("structure_to_smooth_mse_ratio"),
        **envelope,
        "imf_energy_entropy": _imf_energy_entropy(imf_rows),
        "dominant_frequency_span": _dominant_frequency_span(imf_rows),
        "case_score": result.get("case_score"),
        "reconstruction_score": result.get("reconstruction_score"),
        "structural_fidelity_score": result.get("structural_fidelity_score"),
        "contamination_resistance_score": result.get("contamination_resistance_score"),
        "denoise_nmse": result.get("denoise_nmse"),
        "noise_capture_corr": result.get("noise_capture_corr"),
        "outlier_resistance_index": result.get("outlier_resistance_index"),
        "imf_recovery_score": result.get("imf_recovery_score"),
        "component_splitting_index": result.get("component_splitting_index"),
        "component_merging_index": result.get("component_merging_index"),
        "inter_imf_entanglement_index": result.get("inter_imf_entanglement_index"),
        "mode_mixing_index": result.get("mode_mixing_index"),
    }


def _imf_energy_entropy(imf_rows):
    p = np.asarray([r["imf_energy_ratio"] for r in imf_rows if np.isfinite(r["imf_energy_ratio"])], dtype=float)
    if len(p) == 0:
        return np.nan
    p = p / (np.sum(p) + 1e-12)
    return float(-np.sum(p * np.log(p + 1e-12)))


def _dominant_frequency_span(imf_rows):
    freqs = np.asarray([r["imf_dominant_frequency"] for r in imf_rows], dtype=float)
    freqs = freqs[np.isfinite(freqs)]
    if len(freqs) == 0:
        return np.nan
    return float(np.max(freqs) - np.min(freqs))


def _theory_empirical_correlations(case_rows):
    theory_keys = [
        "local_theory_score",
        "operator_score",
        "median_b0",
        "final_tau3_b0_median",
        "final_tau3_valid_ratio",
        "final_wilks_remainder_median",
        "final_hessian_positive_ratio",
        "final_trace_norm",
        "final_contraction",
        "operator_evolution_mean_contraction",
        "final_risk_nmse",
        "residual_variance_envelope_corr",
        "residual_variance_envelope_nmse",
    ]
    empirical_keys = [
        "case_score",
        "reconstruction_score",
        "structural_fidelity_score",
        "contamination_resistance_score",
        "denoise_nmse",
        "noise_capture_corr",
        "outlier_resistance_index",
        "imf_recovery_score",
        "component_splitting_index",
        "component_merging_index",
        "inter_imf_entanglement_index",
        "mode_mixing_index",
    ]
    rows = []
    for tkey in theory_keys:
        for ekey in empirical_keys:
            rows.append({
                "theory_diagnostic": tkey,
                "empirical_metric": ekey,
                "pearson_corr": _safe_corr(
                    [r.get(tkey) for r in case_rows],
                    [r.get(ekey) for r in case_rows],
                ),
                "n_cases": len(case_rows),
                "interpretation_note": "exploratory association across representative Appendix A cases",
            })
    return rows


def _trace_variance_rows(case_id, result, case):
    operator = result.get("operator_diagnostics", {})
    traces = operator.get("noise_trace_norms", [])
    contractions = operator.get("contraction_ratios", [])
    residual_history = result.get("residual_history", [])
    noise_var = float(np.var(case["noise"]) + 1e-12)
    rows = []
    for i, trace in enumerate(traces):
        residual_k = np.asarray(residual_history[i], dtype=float) if i < len(residual_history) else None
        residual_var = float(np.var(residual_k)) if residual_k is not None else np.nan
        rows.append({
            "case_id": case_id,
            "scale_index": i + 1,
            "operator_trace_norm_proxy": _finite_float(trace),
            "operator_contraction_ratio": _finite_float(contractions[i]) if i < len(contractions) else np.nan,
            "empirical_residual_variance": residual_var,
            "true_noise_variance": noise_var,
            "residual_to_noise_variance_ratio": float(residual_var / noise_var) if np.isfinite(residual_var) else np.nan,
            "trace_minus_residual_variance_ratio": (
                float(_finite_float(trace) - residual_var / noise_var)
                if np.isfinite(residual_var) and np.isfinite(_finite_float(trace))
                else np.nan
            ),
        })
    return rows


def _failure_case_rows(case_rows, max_cases=6):
    scored = []
    case_scores = np.asarray([_finite_float(r.get("case_score")) for r in case_rows], dtype=float)
    nmse_vals = np.asarray([_finite_float(r.get("denoise_nmse")) for r in case_rows], dtype=float)
    structure_ratios = np.asarray([_finite_float(r.get("final_structure_to_smooth_mse_ratio")) for r in case_rows], dtype=float)

    score_cut = np.nanquantile(case_scores, 0.35) if np.any(np.isfinite(case_scores)) else np.nan
    nmse_cut = np.nanquantile(nmse_vals, 0.65) if np.any(np.isfinite(nmse_vals)) else np.nan
    ratio_cut = np.nanquantile(structure_ratios, 0.65) if np.any(np.isfinite(structure_ratios)) else np.nan

    for r in case_rows:
        reasons = []
        severity = 0.0
        case_score = _finite_float(r.get("case_score"))
        denoise_nmse = _finite_float(r.get("denoise_nmse"))
        structure_ratio = _finite_float(r.get("final_structure_to_smooth_mse_ratio"))
        trace_norm = _finite_float(r.get("final_trace_norm"))
        tau_valid = _finite_float(r.get("final_tau3_valid_ratio"))

        if np.isfinite(score_cut) and np.isfinite(case_score) and case_score <= score_cut:
            reasons.append("low_case_score")
            severity += float(score_cut - case_score)
        if np.isfinite(nmse_cut) and np.isfinite(denoise_nmse) and denoise_nmse >= nmse_cut:
            reasons.append("high_denoise_nmse")
            severity += float(denoise_nmse - nmse_cut)
        if np.isfinite(ratio_cut) and np.isfinite(structure_ratio) and structure_ratio >= ratio_cut:
            reasons.append("high_structure_to_smooth_mse_ratio")
            severity += 0.05 * float(structure_ratio - ratio_cut)
        if np.isfinite(trace_norm) and trace_norm > 1.0:
            reasons.append("operator_trace_norm_above_one")
            severity += 0.05 * float(trace_norm - 1.0)
        if np.isfinite(tau_valid) and tau_valid < 0.95:
            reasons.append("low_tau3_valid_ratio")
            severity += 0.05 * float(0.95 - tau_valid)

        if reasons:
            scored.append({
                "case_id": r.get("case_id"),
                "signal": r.get("signal"),
                "noise": r.get("noise"),
                "sigma": r.get("sigma"),
                "failure_flags": ";".join(reasons),
                "failure_severity_proxy": severity,
                "case_score": case_score,
                "denoise_nmse": denoise_nmse,
                "final_structure_to_smooth_mse_ratio": structure_ratio,
                "final_trace_norm": trace_norm,
                "final_tau3_valid_ratio": tau_valid,
                "interpretation_note": "automatic Appendix A screening; inspect before making manuscript claims",
            })

    return sorted(scored, key=lambda x: x["failure_severity_proxy"], reverse=True)[:max_cases]


def _aggregate_summary(case_rows):
    keys = [
        "median_b0",
        "final_tau3_b0_median",
        "final_tau3_valid_ratio",
        "final_wilks_remainder_median",
        "final_hessian_positive_ratio",
        "final_trace_norm",
        "final_contraction",
        "final_risk_nmse",
        "final_structure_to_smooth_mse_ratio",
        "residual_variance_envelope_corr",
        "residual_variance_envelope_nmse",
        "case_score",
        "reconstruction_score",
        "structural_fidelity_score",
        "contamination_resistance_score",
        "denoise_nmse",
    ]
    return {
        "n_cases": len(case_rows),
        "diagnostic_positioning": (
            "Spokoiny-inspired theory-diagnostic validation; the outputs assess "
            "numerical behavior and empirical associations, not formal "
            "non-asymptotic risk bounds."
        ),
        "trace_probe_count": SPOKOINY_THEORY_TRACE_PROBE_COUNT,
        "means": {key: _nanmean([r.get(key) for r in case_rows]) for key in keys},
    }


def _plot_appendix_figures(output_root, case_rows, scale_rows, risk_rows, corr_rows, trace_rows):
    figure_dir = ensure_dir(output_root / "figures")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {
            "plot_status": "skipped",
            "reason": f"matplotlib unavailable: {exc}",
            "figure_dir": str(figure_dir),
        }

    figures = {}

    def grouped(rows):
        ids = []
        for row in rows:
            cid = row.get("case_id")
            if cid not in ids:
                ids.append(cid)
        return ids

    # Figure A1: scale evolution of key local theory quantities.
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    scale_metrics = [
        ("b0_local_median", "median b0"),
        ("tau3_b0_median", "median tau3*b0"),
        ("risk_risk_nmse", "risk NMSE"),
        ("risk_residual_energy_ratio_to_observed", "residual energy ratio"),
    ]
    for ax, (metric, label) in zip(axes.ravel(), scale_metrics):
        for cid in grouped(scale_rows):
            rows = [r for r in scale_rows if r.get("case_id") == cid]
            x = [r.get("scale_index") for r in rows]
            y = [r.get(metric) for r in rows]
            if any(np.isfinite(_finite_float(v)) for v in y):
                ax.plot(x, y, marker="o", linewidth=1.2, label=cid[:28])
        ax.set_xlabel("IRMF scale")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=6, loc="best")
    path = figure_dir / "appendix_A_fig1_scale_evolution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures["scale_evolution"] = str(path)

    # Figure A2: smooth/structure/boundary risk propagation.
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for cid in grouped(risk_rows):
        rows = [r for r in risk_rows if r.get("case_id") == cid]
        x = [r.get("scale_index") for r in rows]
        y = [r.get("structure_to_smooth_mse_ratio") for r in rows]
        if any(np.isfinite(_finite_float(v)) for v in y):
            ax.plot(x, y, marker="o", linewidth=1.2, label=cid[:32])
    ax.set_xlabel("IRMF scale")
    ax.set_ylabel("structure/smooth MSE ratio")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6, loc="best")
    path = figure_dir / "appendix_A_fig2_risk_propagation_regions.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures["risk_propagation_regions"] = str(path)

    # Figure A3: operator trace proxy against empirical residual variance ratio.
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    x = [_finite_float(r.get("operator_trace_norm_proxy")) for r in trace_rows]
    y = [_finite_float(r.get("residual_to_noise_variance_ratio")) for r in trace_rows]
    ax.scatter(x, y, s=28, alpha=0.75)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.sum(finite) > 1:
        lo = float(min(np.min(np.asarray(x)[finite]), np.min(np.asarray(y)[finite])))
        hi = float(max(np.max(np.asarray(x)[finite]), np.max(np.asarray(y)[finite])))
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.0)
    ax.set_xlabel("operator trace norm proxy")
    ax.set_ylabel("empirical residual/noise variance ratio")
    ax.grid(alpha=0.25)
    path = figure_dir / "appendix_A_fig3_trace_vs_residual_variance.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures["trace_vs_residual_variance"] = str(path)

    # Figure A4: theory diagnostics vs empirical performance.
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    pairs = [
        ("local_theory_score", "denoise_nmse"),
        ("final_trace_norm", "noise_capture_corr"),
        ("final_tau3_valid_ratio", "reconstruction_score"),
        ("final_risk_nmse", "case_score"),
    ]
    for ax, (xkey, ykey) in zip(axes.ravel(), pairs):
        x = [_finite_float(r.get(xkey)) for r in case_rows]
        y = [_finite_float(r.get(ykey)) for r in case_rows]
        ax.scatter(x, y, s=36, alpha=0.85)
        corr = _safe_corr(x, y)
        ax.set_title(f"corr={corr:.3f}" if np.isfinite(corr) else "corr=NA")
        ax.set_xlabel(xkey)
        ax.set_ylabel(ykey)
        ax.grid(alpha=0.25)
    path = figure_dir / "appendix_A_fig4_theory_vs_empirical.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures["theory_vs_empirical"] = str(path)

    return {
        "plot_status": "created",
        "figure_dir": str(figure_dir),
        "figures": figures,
    }


def run_spokoiny_theory_validation(
        output_root,
        irmf_params=GLOBAL_IRMF_PARAMS,
        cases=SPOKOINY_THEORY_VALIDATION_CASES,
        n=DEFAULT_N,
        fs=DEFAULT_FS,
        seed=DEFAULT_SEED,
):
    output_root = ensure_dir(output_root)
    case_rows = []
    scale_rows = []
    risk_rows_all = []
    imf_rows_all = []
    imf_matching_rows_all = []
    trace_variance_rows_all = []

    for idx, (signal_name, noise_name, sigma) in enumerate(cases, start=1):
        case_id = f"A{idx:02d}_{signal_name}_{noise_name}_sigma_{sigma:.2f}"
        case = make_signal_noise_case(
            signal_name=signal_name,
            noise_name=noise_name,
            sigma=sigma,
            n=n,
            fs=fs,
            seed=seed,
        )
        result = run_fixed_irmf_case(
            Y=case["Y"],
            X_clean=case["X_clean"],
            t=case["t"],
            fs=fs,
            irmf_params=irmf_params,
            expected_noise_ratio=case.get("expected_noise_ratio"),
            true_components=case.get("true_components"),
            run_id=f"appendix_A_spokoiny_{idx}",
        )
        risk_rows = _risk_by_scale(
            Y=case["Y"],
            X_clean=case["X_clean"],
            noise=case["noise"],
            residual_history=result.get("residual_history", []),
        )
        imf_rows = _imf_expansion_rows(result.get("imfs", []), fs=fs)
        imf_matching_rows = _imf_matching_rows(
            case_id=case_id,
            imfs=result.get("imfs", []),
            true_components=case.get("true_components"),
            fs=fs,
        )
        trace_variance_rows = _trace_variance_rows(case_id, result, case)
        case_rows.append(_case_summary(case_id, signal_name, noise_name, sigma, case, result, risk_rows, imf_rows))

        for row in _scale_evolution_rows(case_id, result, risk_rows):
            row.update({"signal": signal_name, "noise": noise_name, "sigma": sigma})
            scale_rows.append(row)
        for row in risk_rows:
            row.update({"case_id": case_id, "signal": signal_name, "noise": noise_name, "sigma": sigma})
            risk_rows_all.append(row)
        for row in imf_rows:
            row.update({"case_id": case_id, "signal": signal_name, "noise": noise_name, "sigma": sigma})
            imf_rows_all.append(row)
        for row in imf_matching_rows:
            row.update({"signal": signal_name, "noise": noise_name, "sigma": sigma})
            imf_matching_rows_all.append(row)
        for row in trace_variance_rows:
            row.update({"signal": signal_name, "noise": noise_name, "sigma": sigma})
            trace_variance_rows_all.append(row)

    corr_rows = _theory_empirical_correlations(case_rows)
    failure_rows = _failure_case_rows(case_rows)
    summary = _aggregate_summary(case_rows)
    plot_summary = _plot_appendix_figures(
        output_root=output_root,
        case_rows=case_rows,
        scale_rows=scale_rows,
        risk_rows=risk_rows_all,
        corr_rows=corr_rows,
        trace_rows=trace_variance_rows_all,
    )
    summary["plot_summary"] = plot_summary
    summary["n_failure_cases_flagged"] = len(failure_rows)
    protocol = {
        "section": "Appendix A Spokoiny Implementation Fidelity and Theory Diagnostics",
        "positioning": (
            "Spokoiny-inspired theory-diagnostic validation, not a formal proof "
            "of Spokoiny non-asymptotic Fisher/Wilks/risk bounds."
        ),
        "local_objective_description": LOCAL_OBJECTIVE_DESCRIPTION,
        "theory_flow_map": THEORY_FLOW_MAP,
        "cases": [
            {"signal": s, "noise": z, "sigma": sig}
            for s, z, sig in cases
        ],
        "irmf_params": dict(irmf_params),
        "n": n,
        "fs": fs,
        "seed": seed,
        "outputs": {
            "case_summary": "appendix_A_spokoiny_case_summary.json/csv",
            "scale_evolution": "appendix_A_spokoiny_scale_evolution.json/csv",
            "risk_propagation": "appendix_A_spokoiny_risk_propagation.json/csv",
            "imf_expansion": "appendix_A_spokoiny_imf_expansion.json/csv",
            "imf_matching": "appendix_A_spokoiny_imf_matching.json/csv",
            "trace_vs_variance": "appendix_A_spokoiny_trace_vs_residual_variance.json/csv",
            "theory_vs_empirical": "appendix_A_spokoiny_theory_vs_empirical_correlations.json/csv",
            "failure_cases": "appendix_A_spokoiny_failure_cases.json/csv",
            "figures": "figures/appendix_A_fig*.png",
        },
    }

    write_json(protocol, output_root / "appendix_A_spokoiny_theory_protocol.json")
    with open(output_root / "appendix_A_spokoiny_local_objective.md", "w", encoding="utf-8") as f:
        f.write("# Appendix A Local Robust Objective\n\n")
        f.write(LOCAL_OBJECTIVE_DESCRIPTION["plain_text"] + "\n\n")
        f.write("```latex\n")
        f.write(LOCAL_OBJECTIVE_DESCRIPTION["latex"] + "\n")
        f.write("```\n\n")
        f.write(LOCAL_OBJECTIVE_DESCRIPTION["implementation_note"] + "\n")
    write_json(case_rows, output_root / "appendix_A_spokoiny_case_summary.json")
    write_csv(case_rows, output_root / "appendix_A_spokoiny_case_summary.csv")
    write_json(scale_rows, output_root / "appendix_A_spokoiny_scale_evolution.json")
    write_csv(scale_rows, output_root / "appendix_A_spokoiny_scale_evolution.csv")
    write_json(risk_rows_all, output_root / "appendix_A_spokoiny_risk_propagation.json")
    write_csv(risk_rows_all, output_root / "appendix_A_spokoiny_risk_propagation.csv")
    write_json(imf_rows_all, output_root / "appendix_A_spokoiny_imf_expansion.json")
    write_csv(imf_rows_all, output_root / "appendix_A_spokoiny_imf_expansion.csv")
    write_json(imf_matching_rows_all, output_root / "appendix_A_spokoiny_imf_matching.json")
    write_csv(imf_matching_rows_all, output_root / "appendix_A_spokoiny_imf_matching.csv")
    write_json(trace_variance_rows_all, output_root / "appendix_A_spokoiny_trace_vs_residual_variance.json")
    write_csv(trace_variance_rows_all, output_root / "appendix_A_spokoiny_trace_vs_residual_variance.csv")
    write_json(corr_rows, output_root / "appendix_A_spokoiny_theory_vs_empirical_correlations.json")
    write_csv(corr_rows, output_root / "appendix_A_spokoiny_theory_vs_empirical_correlations.csv")
    write_json(failure_rows, output_root / "appendix_A_spokoiny_failure_cases.json")
    write_csv(failure_rows, output_root / "appendix_A_spokoiny_failure_cases.csv")
    write_json(summary, output_root / "appendix_A_spokoiny_theory_summary.json")
    return summary


if __name__ == "__main__":
    run_spokoiny_theory_validation("IRMF_EMD_PAPER_RESULTS_V4E_SMOOTHED_MEDIAN_H_FIXED_THEORY_CONSTRAINED/appendices/appendix_A_spokoiny_theory_validation")
