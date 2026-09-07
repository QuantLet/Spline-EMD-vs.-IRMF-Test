#!/usr/bin/python
# coding: UTF-8

"""
Strict Spokoiny-style IRMF core.

This module contains only the decomposition algorithm:
- kernel
- Gaussian-smoothed median rho / grad / hess
- local Newton M-estimator
- boundary extension
- geometric bandwidth decay
- residual recursion

No scoring, plotting, grid search, or diagnostics are placed here.

Updates in v2:
- Optional external time axis T.
- Optional boundary_mode:
    "periodic" = strict Spokoiny-style reproduction.
    "mirror"   = practical variant for non-periodic real data.
- min_support_points guard for too-small bandwidths.
"""

import numpy as np
from math import erf
from core_algorithms.robust_losses import evaluate_loss, gaussian_smoothed_median_loss

try:
    from scipy.special import ndtr as _normal_cdf
except Exception:
    _normal_cdf = None


def spokoiny_kernel(u):
    abs_u = np.abs(u)
    pos = np.clip(1.0 - abs_u, 0.0, None)
    return 0.75 * (pos ** 2)


def _standard_normal_cdf(z):
    """Vectorized standard normal CDF with a no-scipy fallback."""
    z = np.asarray(z, dtype=float)
    if _normal_cdf is not None:
        return _normal_cdf(z)
    erf_vec = np.vectorize(erf)
    return 0.5 * (1.0 + erf_vec(z / np.sqrt(2.0)))


def rho_spline_grad_hess(x, H):
    """Backward-compatible alias for the Gaussian-smoothed median loss."""
    return gaussian_smoothed_median_loss(x, H=H)


def normalize_time_axis(T):
    """
    Normalize arbitrary increasing time axis to [0, 1).

    This lets IRMF accept external T while preserving the current
    bandwidth convention h in normalized time units.
    """
    T = np.asarray(T, dtype=float)

    if T.ndim != 1:
        raise ValueError("T must be one-dimensional.")

    if len(T) < 2:
        raise ValueError("T must contain at least two points.")

    duration = T[-1] - T[0]

    if duration <= 0:
        raise ValueError("T must be strictly increasing with positive duration.")

    return (T - T[0]) / duration


def make_boundary_extension(Y_k, t, boundary_mode="periodic"):
    """
    Boundary extension for local IRMF estimation.

    boundary_mode:
    - "periodic": strict Spokoiny-style periodic extension.
    - "mirror": practical mirror extension for non-periodic data.

    Synthetic strict-reproduction experiments should use "periodic".
    """
    if boundary_mode == "periodic":
        Y_ext = np.concatenate([Y_k, Y_k, Y_k])
        t_ext = np.concatenate([t - 1.0, t, t + 1.0])
        return Y_ext, t_ext

    if boundary_mode == "mirror":
        Y_left = Y_k[::-1]
        Y_right = Y_k[::-1]

        t_left = -t[::-1]
        t_right = 2.0 - t[::-1]

        Y_ext = np.concatenate([Y_left, Y_k, Y_right])
        t_ext = np.concatenate([t_left, t, t_right])
        return Y_ext, t_ext

    raise ValueError(
        f"Unknown boundary_mode={boundary_mode}. "
        "Use 'periodic' or 'mirror'."
    )


def local_m_estimator(
        Y_ext,
        t_ext,
        t_curr,
        h,
        H,
        max_iter=25,
        tol=1e-6,
        min_support_points=3,
        loss_name="gaussian_smoothed_median",
        loss_tuning=None,
        return_diagnostics=False,
        initialization_offset=0.0,
):
    u = (t_curr - t_ext) / h
    kh = spokoiny_kernel(u)
    valid = kh > 0

    support_count = int(np.sum(valid))

    if support_count < min_support_points:
        diag = {
            "iterations": 0, "converged": False, "max_iter_reached": False,
            "objective_non_descent_count": 0, "min_curvature": None,
            "final_hessian": 0.0, "support_count": int(support_count),
            "insufficient_support": True,
        }
        return (0.0, 0.0, 0.0, diag) if return_diagnostics else (0.0, 0.0, 0.0)

    Y_local = Y_ext[valid]
    kh_local = kh[valid]

    x_est = np.sum(Y_local * kh_local) / (np.sum(kh_local) + 1e-10)
    local_scale = float(np.std(Y_local) + 1e-10)
    x_est = float(x_est + float(initialization_offset) * local_scale)

    converged = False
    objective_non_descent_count = 0
    min_curvature = np.inf
    last_objective = np.inf
    iterations = 0
    line_search_activations = 0
    line_search_steps = 0

    for iteration in range(max_iter):
        iterations = iteration + 1
        residual = Y_local - x_est
        rho_vals, grad_vals, hess_vals = evaluate_loss(residual, loss_name=loss_name, tuning=loss_tuning, H=H)
        objective = float(np.sum(rho_vals * kh_local))
        if objective > last_objective + 1e-10:
            objective_non_descent_count += 1
        last_objective = min(last_objective, objective)
        if hess_vals.size:
            min_curvature = min(min_curvature, float(np.nanmin(hess_vals)))

        G = np.sum(-grad_vals * kh_local)
        F_raw = float(np.sum(hess_vals * kh_local))
        # Convex losses use Newton directly.  Non-convex/redescending losses
        # receive a positive ridge and backtracking so the ablation measures
        # the loss rather than catastrophic solver divergence.
        F = max(abs(F_raw), 1e-8)
        step = float(np.clip(G / F, -5.0 * (np.std(Y_local) + 1e-8), 5.0 * (np.std(Y_local) + 1e-8)))
        step_scale = 1.0
        x_new = x_est - step
        for _ls in range(12):
            trial_residual = Y_local - x_new
            trial_rho, _, _ = evaluate_loss(trial_residual, loss_name=loss_name, tuning=loss_tuning, H=H)
            trial_objective = float(np.sum(trial_rho * kh_local))
            if np.isfinite(trial_objective) and trial_objective <= objective + 1e-12:
                break
            line_search_activations += int(_ls == 0)
            line_search_steps += 1
            step_scale *= 0.5
            x_new = x_est - step_scale * step

        if np.abs(x_new - x_est) < tol:
            x_est = x_new
            converged = True
            break

        x_est = x_new

    residual = Y_local - x_est
    _, final_grad_vals, final_hess_vals = evaluate_loss(residual, loss_name=loss_name, tuning=loss_tuning, H=H)

    G_final = np.sum(-final_grad_vals * kh_local)
    F_final = np.sum(final_hess_vals * kh_local) + 1e-10

    diagnostics = {
        "iterations": int(iterations),
        "converged": bool(converged),
        "max_iter_reached": bool(not converged),
        "objective_non_descent_count": int(objective_non_descent_count),
        "min_curvature": None if not np.isfinite(min_curvature) else float(min_curvature),
        "final_hessian": float(F_final),
        "support_count": int(support_count),
        "line_search_activations": int(line_search_activations),
        "line_search_steps": int(line_search_steps),
        "final_gradient_abs": float(abs(G_final)),
        "initialization_offset": float(initialization_offset),
    }
    if return_diagnostics:
        return x_est, G_final, F_final, diagnostics
    return x_est, G_final, F_final


def strict_spokoiny_irmf(
        Y,
        T=None,
        h1=0.15,
        a=np.sqrt(2),
        h_min=0.005,
        H=0.60,
        verbose=False,
        boundary_mode="periodic",
        min_support_points=3,
        loss_name="gaussian_smoothed_median",
        loss_tuning=None,
        collect_optimization_diagnostics=False,
        initialization_offset=0.0,
):
    """
    Strict IRMF decomposition.

    Parameters
    ----------
    Y : ndarray
        Observed signal.
    T : ndarray or None
        Optional external time axis. If provided, it is normalized internally
        to [0, 1) so h1 and h_min retain the existing normalized convention.
    h1, a, h_min, H : float
        IRMF parameters.
    boundary_mode : str
        "periodic" for strict Spokoiny-style reproduction.
        "mirror" for future practical non-periodic data.
    min_support_points : int
        Minimum number of local points required for a valid local estimator.
    """
    n = len(Y)

    if T is None:
        t = np.linspace(0, 1, n, endpoint=False)
    else:
        if len(T) != n:
            raise ValueError("T and Y must have the same length.")
        t = normalize_time_axis(T)

    Y_k = Y.copy()
    h = h1

    imfs = []
    residual_history = []
    scale_history = []

    while h >= h_min:
        if verbose:
            print(f"Current bandwidth h = {h:.6f}")

        Y_before = Y_k.copy()

        S_k = np.zeros(n)
        grad_vec = np.zeros(n)
        hess_vec = np.zeros(n)
        optimization_diagnostics = []

        Y_ext, t_ext = make_boundary_extension(
            Y_k=Y_k,
            t=t,
            boundary_mode=boundary_mode
        )

        for i in range(n):
            local_out = local_m_estimator(
                Y_ext=Y_ext,
                t_ext=t_ext,
                t_curr=t[i],
                h=h,
                H=H,
                min_support_points=min_support_points,
                loss_name=loss_name,
                loss_tuning=loss_tuning,
                return_diagnostics=collect_optimization_diagnostics,
                initialization_offset=initialization_offset,
            )
            if collect_optimization_diagnostics:
                x_est, G, F, local_diag = local_out
                local_diag["index"] = int(i)
                optimization_diagnostics.append(local_diag)
            else:
                x_est, G, F = local_out
            S_k[i] = x_est
            grad_vec[i] = G
            hess_vec[i] = F

        Y_k = Y_k - S_k

        imfs.append(S_k.copy())
        residual_history.append(Y_k.copy())

        scale_history.append({
            "h": h,
            "Y_before": Y_before.copy(),
            "S_k": S_k.copy(),
            "Y_after": Y_k.copy(),
            "grad_vec": grad_vec.copy(),
            "hess_vec": hess_vec.copy(),
            "boundary_mode": boundary_mode,
            "min_support_points": min_support_points,
            "loss_name": loss_name,
            "loss_tuning": dict(loss_tuning or {}),
            "optimization_diagnostics": optimization_diagnostics,
            "initialization_offset": float(initialization_offset),
        })

        h = h / a

    return imfs, Y_k, residual_history, scale_history
