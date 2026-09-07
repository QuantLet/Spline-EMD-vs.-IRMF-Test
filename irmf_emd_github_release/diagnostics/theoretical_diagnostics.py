#!/usr/bin/python
# coding: UTF-8

"""
Spokoiny-PDF theoretical diagnostics for IRMF.

This module implements the PDF-facing objects

    W^(k)      local smoothing / robust influence operator
    A^(k)      residual noise propagation operator
    B^(k)      IMF noise propagation operator = W^(k) A^(k-1)
    Sigma      input noise covariance proxy

and the trace-risk quantities

    IMF risk      = tr(B^(k) Sigma B^(k)^T)
    Residual risk = tr(A^(k) Sigma A^(k)^T)

Operator contraction proxies are also reported as Frobenius/energy ratios.

Appendix-A-inspired Fisher / Wilks / Hessian stability proxies are implemented
in experimental/theory_extensions/spokoiny_appendix_diagnostics.py and imported
here only for reporting convenience. They are experimental theory-extension
checks, not part of the core IRMF algorithm or the main physical diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core_algorithms.strict_spokoiny_irmf import spokoiny_kernel, rho_spline_grad_hess
from experimental.theory_extensions.spokoiny_appendix_diagnostics import appendix_a_proxy_summary


Array = np.ndarray
EPS = 1e-12


@dataclass
class TheoreticalRiskRow:
    k: int
    h: float
    imf_trace_risk: float
    imf_trace_risk_norm: float
    residual_trace_risk: float
    residual_trace_risk_norm: float
    imf_operator_frobenius_sq: float
    residual_operator_frobenius_sq: float
    imf_energy_gain: float
    residual_energy_gain: float
    residual_contraction_ratio: float
    imf_trace_to_residual_trace: float
    # Appendix-A-inspired diagnostics
    fisher_step_l2_mean: float
    fisher_step_l2_max: float
    wilks_gap_mean: float
    wilks_gap_sum: float
    hessian_mean: float
    hessian_min: float
    hessian_p05: float
    hessian_p50: float
    hessian_p95: float
    hessian_zero_fraction: float
    operator_stability_score: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _normalize_time_axis(T: Optional[Array], n: int) -> Array:
    if T is None:
        return np.linspace(0.0, 1.0, n, endpoint=False)
    T = np.asarray(T, dtype=float)
    if T.ndim != 1 or len(T) != n:
        raise ValueError("T must be one-dimensional and have the same length as Y.")
    if n < 2:
        return np.zeros(n, dtype=float)
    duration = T[-1] - T[0]
    if duration <= 0:
        raise ValueError("T must be increasing with positive duration.")
    return (T - T[0]) / duration


def bandwidth_sequence(h1: float, a: float, h_min: float) -> List[float]:
    if h1 <= 0 or a <= 1 or h_min <= 0:
        raise ValueError("Require h1 > 0, a > 1, and h_min > 0.")
    out = []
    h = float(h1)
    while h >= h_min:
        out.append(float(h))
        h = h / float(a)
    return out


def indexed_boundary_extension(values: Array, t: Array, boundary_mode: str = "periodic") -> Tuple[Array, Array, Array]:
    values = np.asarray(values)
    n = len(values)
    idx = np.arange(n)
    if boundary_mode == "periodic":
        return (
            np.concatenate([values, values, values]),
            np.concatenate([t - 1.0, t, t + 1.0]),
            np.concatenate([idx, idx, idx]),
        )
    if boundary_mode == "mirror":
        return (
            np.concatenate([values[::-1], values, values[::-1]]),
            np.concatenate([-t[::-1], t, 2.0 - t[::-1]]),
            np.concatenate([idx[::-1], idx, idx[::-1]]),
        )
    raise ValueError("boundary_mode must be 'periodic' or 'mirror'.")


def build_W_matrix(
        n: int,
        h: float,
        T: Optional[Array] = None,
        boundary_mode: str = "periodic",
        operator_mode: str = "linear_mean",
        Y_before: Optional[Array] = None,
        S_k: Optional[Array] = None,
        H: Optional[float] = None,
        min_denominator: float = 1e-12,
) -> Array:
    """Build dense W^(k). Use only for small/medium n or debugging."""
    t = _normalize_time_axis(T, n)
    dummy = np.zeros(n)
    _, t_ext, idx_ext = indexed_boundary_extension(dummy, t, boundary_mode=boundary_mode)

    if operator_mode not in {"linear_mean", "robust_influence"}:
        raise ValueError("operator_mode must be 'linear_mean' or 'robust_influence'.")

    if operator_mode == "robust_influence":
        if Y_before is None or S_k is None or H is None:
            raise ValueError("robust_influence mode requires Y_before, S_k, and H.")
        Y_ext, _, _ = indexed_boundary_extension(np.asarray(Y_before, dtype=float), t, boundary_mode=boundary_mode)
        S_k = np.asarray(S_k, dtype=float)

    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        u = (t[i] - t_ext) / float(h)
        kh = spokoiny_kernel(u)
        valid = kh > 0
        if not np.any(valid):
            continue
        weights = kh[valid].astype(float)
        if operator_mode == "robust_influence":
            residual = Y_ext[valid] - S_k[i]
            _, _, hess_vals = rho_spline_grad_hess(residual, float(H))
            weights = weights * hess_vals
        denom = float(np.sum(weights))
        if denom <= min_denominator:
            continue
        np.add.at(W[i, :], idx_ext[valid], weights / denom)
    return W


def apply_W_to_matrix(
        V: Array,
        h: float,
        T: Optional[Array] = None,
        boundary_mode: str = "periodic",
        operator_mode: str = "linear_mean",
        Y_before: Optional[Array] = None,
        S_k: Optional[Array] = None,
        H: Optional[float] = None,
        min_denominator: float = 1e-12,
) -> Array:
    """Matrix-free W V for Hutchinson trace estimation."""
    V = np.asarray(V, dtype=float)
    one_d = V.ndim == 1
    Vw = V[:, None] if one_d else V
    n = Vw.shape[0]
    t = _normalize_time_axis(T, n)
    dummy = np.zeros(n)
    _, t_ext, idx_ext = indexed_boundary_extension(dummy, t, boundary_mode=boundary_mode)

    if operator_mode not in {"linear_mean", "robust_influence"}:
        raise ValueError("operator_mode must be 'linear_mean' or 'robust_influence'.")

    if operator_mode == "robust_influence":
        if Y_before is None or S_k is None or H is None:
            raise ValueError("robust_influence mode requires Y_before, S_k, and H.")
        Y_ext, _, _ = indexed_boundary_extension(np.asarray(Y_before, dtype=float), t, boundary_mode=boundary_mode)
        S_k = np.asarray(S_k, dtype=float)

    out = np.zeros_like(Vw)
    for i in range(n):
        u = (t[i] - t_ext) / float(h)
        kh = spokoiny_kernel(u)
        valid = kh > 0
        if not np.any(valid):
            continue
        weights = kh[valid].astype(float)
        if operator_mode == "robust_influence":
            residual = Y_ext[valid] - S_k[i]
            _, _, hess_vals = rho_spline_grad_hess(residual, float(H))
            weights = weights * hess_vals
        denom = float(np.sum(weights))
        if denom <= min_denominator:
            continue
        out[i, :] = (weights / denom) @ Vw[idx_ext[valid], :]
    return out[:, 0] if one_d else out


def _sigma2_from_inputs(noise_sigma: Optional[float], sigma_trace: Optional[float], n: int) -> float:
    if sigma_trace is not None:
        return float(sigma_trace) / float(n)
    if noise_sigma is not None:
        return float(noise_sigma) ** 2
    return 1.0



def compute_theoretical_diagnostics_hutchinson(
        n: int,
        h_sequence: Sequence[float],
        T: Optional[Array] = None,
        boundary_mode: str = "periodic",
        operator_mode: str = "linear_mean",
        scale_history: Optional[Sequence[Dict]] = None,
        H: Optional[float] = None,
        noise_sigma: Optional[float] = None,
        sigma_trace: Optional[float] = None,
        probe_count: int = 64,
        random_seed: int = 123,
) -> Dict:
    h_sequence = list(h_sequence)
    rng = np.random.default_rng(random_seed)
    Z0 = rng.choice([-1.0, 1.0], size=(n, int(probe_count)))
    A_prev_Z = Z0.copy()
    sigma2 = _sigma2_from_inputs(noise_sigma, sigma_trace, n)

    rows: List[TheoreticalRiskRow] = []
    prev_residual_trace = sigma2 * float(n)

    for k, h in enumerate(h_sequence, start=1):
        kwargs = {}
        state = None
        if scale_history is not None and k - 1 < len(scale_history):
            state = scale_history[k - 1]
        if operator_mode == "robust_influence":
            if state is None:
                raise ValueError("robust_influence mode requires scale_history.")
            kwargs.update({"Y_before": state["Y_before"], "S_k": state["S_k"], "H": H})

        BZ = apply_W_to_matrix(
            A_prev_Z,
            h=float(h),
            T=T,
            boundary_mode=boundary_mode,
            operator_mode=operator_mode,
            **kwargs,
        )
        A_curr_Z = A_prev_Z - BZ

        imf_frob_sq_est = float(np.mean(np.sum(BZ * BZ, axis=0)))
        resid_frob_sq_est = float(np.mean(np.sum(A_curr_Z * A_curr_Z, axis=0)))
        imf_trace = sigma2 * imf_frob_sq_est
        resid_trace = sigma2 * resid_frob_sq_est

        appendix = appendix_a_proxy_summary(state or {})
        contraction = resid_trace / (prev_residual_trace + EPS)
        stability_score = (
            0.50 * max(0.0, contraction - 1.0)
            + 0.25 * appendix["hessian_zero_fraction"]
            + 0.25 * (appendix["fisher_step_l2_mean"] if np.isfinite(appendix["fisher_step_l2_mean"]) else 0.0)
        )

        rows.append(TheoreticalRiskRow(
            k=k,
            h=float(h),
            imf_trace_risk=imf_trace,
            imf_trace_risk_norm=imf_trace / n,
            residual_trace_risk=resid_trace,
            residual_trace_risk_norm=resid_trace / n,
            imf_operator_frobenius_sq=imf_frob_sq_est,
            residual_operator_frobenius_sq=resid_frob_sq_est,
            imf_energy_gain=imf_frob_sq_est / n,
            residual_energy_gain=resid_frob_sq_est / n,
            residual_contraction_ratio=contraction,
            imf_trace_to_residual_trace=imf_trace / (resid_trace + EPS),
            operator_stability_score=float(stability_score),
            **appendix,
        ))
        A_prev_Z = A_curr_Z
        prev_residual_trace = resid_trace

    return {
        "method": "hutchinson",
        "operator_mode": operator_mode,
        "boundary_mode": boundary_mode,
        "noise_model": "sigma2_I_or_empirical_trace_I",
        "noise_sigma": noise_sigma,
        "sigma_trace": sigma_trace,
        "sigma2_effective": sigma2,
        "probe_count": int(probe_count),
        "random_seed": int(random_seed),
        "rows": [r.to_dict() for r in rows],
        "bandwidths": [r.h for r in rows],
        "imf_trace_risk_norms": [r.imf_trace_risk_norm for r in rows],
        "residual_trace_risk_norms": [r.residual_trace_risk_norm for r in rows],
        "residual_contraction_ratios": [r.residual_contraction_ratio for r in rows],
    }


def compute_theoretical_diagnostics_from_scale_history(
        scale_history: Sequence[Dict],
        T: Optional[Array] = None,
        H: Optional[float] = None,
        noise_sigma: Optional[float] = None,
        sigma_trace: Optional[float] = None,
        method: str = "hutchinson",
        operator_mode: str = "robust_influence",
        probe_count: int = 64,
        random_seed: int = 123,
) -> Dict:
    if not scale_history:
        raise ValueError("scale_history is empty.")
    if method != "hutchinson":
        # Dense exact can be added, but for n=500 and batch runs Hutchinson is safer.
        raise ValueError("FIX9 supports method='hutchinson' in the standalone runners.")
    h_sequence = [float(state["h"]) for state in scale_history]
    n = len(scale_history[0]["Y_before"])
    boundary_mode = scale_history[0].get("boundary_mode", "periodic")
    return compute_theoretical_diagnostics_hutchinson(
        n=n,
        h_sequence=h_sequence,
        T=T,
        boundary_mode=boundary_mode,
        operator_mode=operator_mode,
        scale_history=scale_history,
        H=H,
        noise_sigma=noise_sigma,
        sigma_trace=sigma_trace,
        probe_count=probe_count,
        random_seed=random_seed,
    )


def summarize_theoretical_diagnostics(theoretical_diagnostics: Dict) -> Dict[str, float]:
    rows = theoretical_diagnostics.get("rows", [])
    if not rows:
        return {"theoretical_risk_score": np.nan}
    imf_norms = np.asarray([r["imf_trace_risk_norm"] for r in rows], dtype=float)
    resid_norms = np.asarray([r["residual_trace_risk_norm"] for r in rows], dtype=float)
    contractions = np.asarray([r["residual_contraction_ratio"] for r in rows], dtype=float)
    fisher_steps = np.asarray([r.get("fisher_step_l2_mean", np.nan) for r in rows], dtype=float)
    wilks_gaps = np.asarray([r.get("wilks_gap_mean", np.nan) for r in rows], dtype=float)
    hessian_zero = np.asarray([r.get("hessian_zero_fraction", np.nan) for r in rows], dtype=float)
    stability = np.asarray([r.get("operator_stability_score", np.nan) for r in rows], dtype=float)

    risk_score = (
        0.40 * float(np.nanmean(imf_norms))
        + 0.20 * float(np.nanmax(imf_norms))
        + 0.20 * float(resid_norms[-1])
        + 0.10 * float(np.nanmean(np.maximum(contractions - 1.0, 0.0)))
        + 0.05 * float(np.nanmean(np.nan_to_num(fisher_steps, nan=0.0)))
        + 0.05 * float(np.nanmean(np.nan_to_num(hessian_zero, nan=0.0)))
    )
    return {
        "theoretical_imf_risk_mean": float(np.nanmean(imf_norms)),
        "theoretical_imf_risk_max": float(np.nanmax(imf_norms)),
        "theoretical_imf_risk_final": float(imf_norms[-1]),
        "theoretical_residual_risk_initial": float(resid_norms[0]),
        "theoretical_residual_risk_final": float(resid_norms[-1]),
        "theoretical_residual_contraction_max": float(np.nanmax(contractions)),
        "theoretical_residual_contraction_mean": float(np.nanmean(contractions)),
        "fisher_step_l2_mean_over_scales": float(np.nanmean(fisher_steps)),
        "fisher_step_l2_max_over_scales": float(np.nanmax([r.get("fisher_step_l2_max", np.nan) for r in rows])),
        "wilks_gap_mean_over_scales": float(np.nanmean(wilks_gaps)),
        "hessian_zero_fraction_mean": float(np.nanmean(hessian_zero)),
        "operator_stability_score_mean": float(np.nanmean(stability)),
        "theoretical_risk_score": float(risk_score),
    }
