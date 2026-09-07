#!/usr/bin/python
# coding: UTF-8
"""
Scale-by-scale IRMF operator propagation diagnostics.
Strict-IRMF compatible: reads scale_history only and does not modify recursion.
"""
import numpy as np


def _safe_norm(x):
    return float(np.linalg.norm(np.asarray(x, dtype=float)) + 1e-10)


def compute_operator_evolution(scale_history):
    rows = []
    prev_after_norm = None

    for k, step in enumerate(scale_history, start=1):
        Y_before = np.asarray(step["Y_before"], dtype=float)
        S_k = np.asarray(step["S_k"], dtype=float)
        Y_after = np.asarray(step["Y_after"], dtype=float)
        h = float(step["h"])
        n = len(Y_before)

        before_norm = _safe_norm(Y_before)
        after_norm = _safe_norm(Y_after)
        component_norm = _safe_norm(S_k)

        contraction_ratio = after_norm / before_norm
        residual_energy_ratio = float(np.sum(Y_after ** 2) / (np.sum(Y_before ** 2) + 1e-10))
        component_energy_ratio = float(np.sum(S_k ** 2) / (np.sum(Y_before ** 2) + 1e-10))
        b0_scale_proxy = component_norm / before_norm

        rows.append({
            "k": k,
            "h": h,
            "trace_proxy": float(n * residual_energy_ratio),
            "trace_norm_proxy": float(residual_energy_ratio),
            "operator_norm_proxy": float(contraction_ratio),
            "contraction_ratio": float(contraction_ratio),
            "inter_scale_contraction": float(np.nan if prev_after_norm is None else after_norm / (prev_after_norm + 1e-10)),
            "residual_energy_ratio": residual_energy_ratio,
            "component_energy_ratio": component_energy_ratio,
            "b0_scale_proxy": float(b0_scale_proxy),
            "before_norm": before_norm,
            "after_norm": after_norm,
            "component_norm": component_norm,
        })
        prev_after_norm = after_norm

    return rows


def summarize_operator_evolution(rows):
    if not rows:
        return {
            "operator_evolution_final_trace_norm": np.nan,
            "operator_evolution_mean_contraction": np.nan,
            "operator_evolution_max_contraction": np.nan,
            "operator_evolution_final_residual_energy": np.nan,
            "operator_evolution_mean_b0_scale": np.nan,
        }

    contractions = np.array([r["contraction_ratio"] for r in rows], dtype=float)
    residuals = np.array([r["residual_energy_ratio"] for r in rows], dtype=float)
    b0 = np.array([r["b0_scale_proxy"] for r in rows], dtype=float)
    trace_norm = np.array([r["trace_norm_proxy"] for r in rows], dtype=float)

    return {
        "operator_evolution_final_trace_norm": float(trace_norm[-1]),
        "operator_evolution_mean_contraction": float(np.nanmean(contractions)),
        "operator_evolution_max_contraction": float(np.nanmax(contractions)),
        "operator_evolution_final_residual_energy": float(residuals[-1]),
        "operator_evolution_mean_b0_scale": float(np.nanmean(b0)),
    }
