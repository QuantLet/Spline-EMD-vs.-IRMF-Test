#!/usr/bin/python
# coding: UTF-8

"""
Spokoiny Appendix-A-inspired theory-extension diagnostics.

This module is intentionally placed under:

    experimental/theory_extensions/

rather than the main diagnostics/ package because these quantities are
numerical theory checks inspired by Smooth Perturbed Optimization, Fisher
expansion, Wilks expansion, and Hessian stability. They are not part of the
IRMF decomposition algorithm and should not be treated as mandatory physical
quality diagnostics for EMD-vs-IRMF benchmarking.

The central proxies are:

    Fisher/Newton proxy:      grad / Hessian
    Wilks/quadratic proxy:    0.5 * grad^2 / Hessian
    Hessian stability proxy:  curvature quantiles and zero-curvature fraction

They are useful for reporting whether the robust local optimization behaves in
a locally regular, non-degenerate way along the selected IRMF scale path.
"""

from __future__ import annotations

from typing import Dict
import numpy as np

EPS = 1e-12


def appendix_a_proxy_summary(scale_state: Dict) -> Dict[str, float]:
    """Compute Fisher/Wilks/Hessian proxy diagnostics for one IRMF scale.

    Parameters
    ----------
    scale_state:
        One dictionary from strict_spokoiny_irmf scale_history. It should contain
        `grad_vec` and `hess_vec`. If unavailable, NaNs are returned.
    """
    grad = np.asarray(scale_state.get("grad_vec", []), dtype=float)
    hess = np.asarray(scale_state.get("hess_vec", []), dtype=float)

    if grad.size == 0 or hess.size == 0:
        return {
            "fisher_step_l2_mean": np.nan,
            "fisher_step_l2_max": np.nan,
            "wilks_gap_mean": np.nan,
            "wilks_gap_sum": np.nan,
            "hessian_mean": np.nan,
            "hessian_min": np.nan,
            "hessian_p05": np.nan,
            "hessian_p50": np.nan,
            "hessian_p95": np.nan,
            "hessian_zero_fraction": np.nan,
        }

    safe_h = np.maximum(hess, EPS)
    fisher_step = grad / safe_h
    wilks_gap = 0.5 * (grad ** 2) / safe_h
    finite_h = hess[np.isfinite(hess)]
    if finite_h.size == 0:
        finite_h = np.array([np.nan])

    return {
        "fisher_step_l2_mean": float(np.nanmean(np.abs(fisher_step))),
        "fisher_step_l2_max": float(np.nanmax(np.abs(fisher_step))),
        "wilks_gap_mean": float(np.nanmean(wilks_gap)),
        "wilks_gap_sum": float(np.nansum(wilks_gap)),
        "hessian_mean": float(np.nanmean(finite_h)),
        "hessian_min": float(np.nanmin(finite_h)),
        "hessian_p05": float(np.nanpercentile(finite_h, 5)),
        "hessian_p50": float(np.nanpercentile(finite_h, 50)),
        "hessian_p95": float(np.nanpercentile(finite_h, 95)),
        "hessian_zero_fraction": float(np.mean(hess <= EPS)),
    }


def summarize_appendix_a_rows(rows) -> Dict[str, float]:
    """Aggregate Appendix-A proxy fields over theoretical diagnostic rows."""
    if not rows:
        return {
            "fisher_step_l2_mean_over_scales": np.nan,
            "fisher_step_l2_max_over_scales": np.nan,
            "wilks_gap_mean_over_scales": np.nan,
            "hessian_zero_fraction_mean": np.nan,
        }

    fisher_steps = np.asarray([r.get("fisher_step_l2_mean", np.nan) for r in rows], dtype=float)
    wilks_gaps = np.asarray([r.get("wilks_gap_mean", np.nan) for r in rows], dtype=float)
    hessian_zero = np.asarray([r.get("hessian_zero_fraction", np.nan) for r in rows], dtype=float)
    fisher_max = np.asarray([r.get("fisher_step_l2_max", np.nan) for r in rows], dtype=float)

    return {
        "fisher_step_l2_mean_over_scales": float(np.nanmean(fisher_steps)),
        "fisher_step_l2_max_over_scales": float(np.nanmax(fisher_max)),
        "wilks_gap_mean_over_scales": float(np.nanmean(wilks_gaps)),
        "hessian_zero_fraction_mean": float(np.nanmean(hessian_zero)),
    }
