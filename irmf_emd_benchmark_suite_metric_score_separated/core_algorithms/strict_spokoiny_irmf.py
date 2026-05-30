#!/usr/bin/python
# coding: UTF-8

"""
Strict Spokoiny-style IRMF core.

This module contains only the decomposition algorithm:
- kernel
- robust rho / grad / hess
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


def spokoiny_kernel(u):
    abs_u = np.abs(u)
    pos = np.clip(1.0 - abs_u, 0.0, None)
    return 0.75 * (pos ** 2)


def rho_spline_grad_hess(x, H):
    """
    Analytically consistent third-order spline robust loss.

    Region 2 is scaled by H:
        rho = 0.5*x^2 - (|x|-H)^3/(6H)
        grad = x - sign(x)*(|x|-H)^2/(2H)
        hess = 1 - (|x|-H)/H

    Region 3 constant is 11/6 * H^2, ensuring continuity at |x|=2H.
    """
    abs_x = np.abs(x)
    sgn = np.sign(x)

    rho = np.zeros_like(x)
    grad = np.zeros_like(x)
    hess = np.zeros_like(x)

    idx1 = abs_x <= H
    rho[idx1] = 0.5 * x[idx1] ** 2
    grad[idx1] = x[idx1]
    hess[idx1] = 1.0

    idx2 = (abs_x > H) & (abs_x <= 2 * H)
    ax2 = abs_x[idx2]
    rho[idx2] = 0.5 * ax2 ** 2 - (1.0 / (6.0 * H)) * ((ax2 - H) ** 3)
    grad[idx2] = x[idx2] - (0.5 / H) * sgn[idx2] * ((ax2 - H) ** 2)
    hess[idx2] = 1.0 - (ax2 - H) / H

    idx3 = abs_x > 2 * H
    ax3 = abs_x[idx3]
    rho[idx3] = (11.0 / 6.0) * H ** 2 + 1.5 * H * (ax3 - 2 * H)
    grad[idx3] = sgn[idx3] * 1.5 * H
    hess[idx3] = 0.0

    return rho, grad, hess


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
        min_support_points=3
):
    u = (t_curr - t_ext) / h
    kh = spokoiny_kernel(u)
    valid = kh > 0

    support_count = int(np.sum(valid))

    if support_count < min_support_points:
        return 0.0, 0.0, 0.0

    Y_local = Y_ext[valid]
    kh_local = kh[valid]

    x_est = np.sum(Y_local * kh_local) / (np.sum(kh_local) + 1e-10)

    for _ in range(max_iter):
        residual = Y_local - x_est
        _, grad_vals, hess_vals = rho_spline_grad_hess(residual, H)

        G = np.sum(-grad_vals * kh_local)
        F = np.sum(hess_vals * kh_local) + 1e-10

        step = G / F
        x_new = x_est - step

        if np.abs(x_new - x_est) < tol:
            x_est = x_new
            break

        x_est = x_new

    residual = Y_local - x_est
    _, final_grad_vals, final_hess_vals = rho_spline_grad_hess(residual, H)

    G_final = np.sum(-final_grad_vals * kh_local)
    F_final = np.sum(final_hess_vals * kh_local) + 1e-10

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
        min_support_points=3
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

        Y_ext, t_ext = make_boundary_extension(
            Y_k=Y_k,
            t=t,
            boundary_mode=boundary_mode
        )

        for i in range(n):
            x_est, G, F = local_m_estimator(
                Y_ext=Y_ext,
                t_ext=t_ext,
                t_curr=t[i],
                h=h,
                H=H,
                min_support_points=min_support_points
            )
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
            "min_support_points": min_support_points
        })

        h = h / a

    return imfs, Y_k, residual_history, scale_history
