#!/usr/bin/python
# coding: UTF-8

"""
Layer 1 — IRMF Performance Evaluation.

Public entry point for IRMF-specific internal diagnostics.
"""

import numpy as np

from diagnostics.irmf_local_theory_diagnostics import (
    compute_irmf_local_theory_diagnostics,
    summarize_irmf_local_theory_score,
)
from diagnostics.irmf_operator_diagnostics import (
    compute_irmf_operator_diagnostics,
    summarize_irmf_operator_score,
)
from diagnostics.irmf_operator_evolution import (
    compute_operator_evolution,
    summarize_operator_evolution,
)


def _finite(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def _bounded(x):
    x = _finite(x)
    if not np.isfinite(x):
        return np.nan
    return abs(x) / (1.0 + abs(x))


def _nanmean(values):
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    return float(np.mean(arr)) if len(arr) else np.nan


def _first(result, keys):
    for key in keys:
        if key in result:
            x = _finite(result.get(key))
            if np.isfinite(x):
                return x
    return np.nan


def irmf_performance_score(result):
    """
    Composite score for Layer 1. Smaller is better.

    Level A:
        mean_b0_like, trace_norm, operator_norm, contraction_ratio

    Level B:
        residual_energy_ratio, b0_evolution_mean, b0_evolution_final,
        operator_monotonicity_score

    Level C:
        Hessian diagnostics are reported separately and are not included
        in the composite score.
    """
    mean_b0 = _first(result, ["mean_b0_like", "mean_b0", "operator_evolution_mean_b0_scale"])
    trace_norm = _first(result, ["trace_norm", "operator_evolution_final_trace_norm"])
    operator_norm = _first(result, ["operator_norm"])
    contraction = _first(result, ["contraction_ratio", "operator_evolution_mean_contraction"])
    residual_energy = _first(result, ["residual_energy_ratio", "operator_evolution_final_residual_energy"])
    b0_mean = _first(result, ["b0_evolution_mean", "operator_evolution_mean_b0_scale"])
    b0_final = _first(result, ["b0_evolution_final"])
    mono = _first(result, ["operator_monotonicity_score"])

    vals = [
        _bounded(mean_b0),
        _bounded(trace_norm),
        _bounded(operator_norm),
        _bounded(contraction),
        _bounded(residual_energy),
        _bounded(b0_mean),
        _bounded(b0_final),
        1.0 - mono if np.isfinite(mono) else np.nan,
    ]

    return _nanmean(vals)


def attach_irmf_performance_score(result):
    score = irmf_performance_score(result)
    result["irmf_performance_score"] = score
    # Do not overwrite a theory score already assembled from local-theory and
    # operator diagnostics.  Older code used theory_diagnostic_score as an
    # alias for this Layer-1 performance score; keep that fallback only when no
    # theory score exists.
    if "theory_diagnostic_score" not in result:
        result["theory_diagnostic_score"] = score
    return result
