#!/usr/bin/python
# coding: UTF-8

"""
Mode-mixing diagnostics.

These diagnostics go beyond global spectral leakage by studying:
- IMF time-domain correlation topology
- instantaneous-frequency ridge overlap
- local frequency crossing
"""

import numpy as np
from scipy.signal import hilbert, medfilt


def imf_correlation_matrix(imfs):
    K = len(imfs)
    if K == 0:
        return np.zeros((0, 0))

    M = np.eye(K)
    for i in range(K):
        for j in range(i + 1, K):
            xi = imfs[i]
            xj = imfs[j]
            if np.std(xi) < 1e-10 or np.std(xj) < 1e-10:
                c = 0.0
            else:
                c = abs(np.corrcoef(xi, xj)[0, 1])
            M[i, j] = c
            M[j, i] = c
    return M


def instantaneous_frequency_matrix(imfs, fs, median_kernel=5):
    """
    Returns IF curves with shape (K, n-1).
    """
    curves = []
    for imf in imfs:
        if np.std(imf) < 1e-10:
            curves.append(np.zeros(max(0, len(imf) - 1)))
            continue

        analytic = hilbert(imf)
        phase = np.unwrap(np.angle(analytic))
        omega = np.diff(phase)
        freq = fs * omega / (2.0 * np.pi)

        if len(freq) >= 5:
            k = min(median_kernel, len(freq))
            if k % 2 == 0:
                k -= 1
            if k >= 3:
                freq = medfilt(freq, kernel_size=k)

        # robust clipping against phase artifacts
        if len(freq) > 0:
            cap = np.quantile(np.abs(freq), 0.98)
            freq = np.clip(freq, -cap, cap)

        curves.append(freq)

    if len(curves) == 0:
        return np.zeros((0, 0))

    min_len = min(len(c) for c in curves)
    return np.vstack([c[:min_len] for c in curves])


def ridge_overlap_matrix(imfs, fs, tolerance_hz=2.0):
    """
    Pairwise fraction of time where two instantaneous-frequency ridges
    are closer than tolerance_hz.
    """
    IF = instantaneous_frequency_matrix(imfs, fs)

    K = IF.shape[0]
    if K == 0:
        return np.zeros((0, 0))

    M = np.eye(K)

    for i in range(K):
        for j in range(i + 1, K):
            valid = np.isfinite(IF[i]) & np.isfinite(IF[j])
            if not np.any(valid):
                val = 0.0
            else:
                val = float(np.mean(np.abs(IF[i, valid] - IF[j, valid]) < tolerance_hz))
            M[i, j] = val
            M[j, i] = val

    return M


def local_frequency_crossing_rate(imfs, fs):
    """
    Counts how often adjacent IF ordering changes.
    Higher means stronger ridge crossing / mode mixing.
    """
    IF = instantaneous_frequency_matrix(imfs, fs)
    K, T = IF.shape if IF.ndim == 2 else (0, 0)

    if K <= 1 or T <= 2:
        return 0.0

    crossings = []
    for i in range(K - 1):
        diff = IF[i + 1] - IF[i]
        sign = np.sign(diff)
        changes = np.sum(sign[1:] * sign[:-1] < 0)
        crossings.append(changes / max(1, T - 1))

    return float(np.mean(crossings))


def compute_mode_mixing_diagnostics(imfs, fs):
    corr_M = imf_correlation_matrix(imfs)
    ridge_M = ridge_overlap_matrix(imfs, fs)

    K = len(imfs)
    if K <= 1:
        corr_mean = corr_max = ridge_mean = ridge_max = 0.0
    else:
        off = ~np.eye(K, dtype=bool)
        corr_mean = float(np.mean(corr_M[off]))
        corr_max = float(np.max(corr_M[off]))
        ridge_mean = float(np.mean(ridge_M[off]))
        ridge_max = float(np.max(ridge_M[off]))

    crossing_rate = local_frequency_crossing_rate(imfs, fs)

    mode_mixing_score = (
        0.35 * corr_mean
        + 0.25 * corr_max
        + 0.20 * ridge_mean
        + 0.10 * ridge_max
        + 0.10 * crossing_rate
    )

    return {
        "imf_correlation_matrix": corr_M,
        "imf_corr_mean_offdiag": corr_mean,
        "imf_corr_max_offdiag": corr_max,
        "ridge_overlap_matrix": ridge_M,
        "ridge_overlap_mean_offdiag": ridge_mean,
        "ridge_overlap_max_offdiag": ridge_max,
        "local_frequency_crossing_rate": crossing_rate,
        "mode_mixing_score": float(mode_mixing_score),
    }
