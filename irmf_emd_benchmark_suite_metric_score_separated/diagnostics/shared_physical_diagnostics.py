#!/usr/bin/python
# coding: UTF-8

"""
Shared diagnostics for IRMF and EMD-family decompositions.

Important convention:
- Raw metrics keep their physical meaning and units.
- Composite scores use separate *_score_component fields.
- Smaller composite scores are better.
"""

import numpy as np
from scipy.signal import hilbert, medfilt


# ============================================================
# Basic utilities
# ============================================================

def _as_2d(imfs):
    imfs = np.asarray(imfs, dtype=float)
    if imfs.ndim == 1:
        imfs = imfs[None, :]
    if imfs.size == 0:
        return np.empty((0, 0), dtype=float)
    return imfs


def _safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) != len(b) or len(a) == 0:
        return np.nan

    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def _safe_psnr(x_true, x_hat):
    """
    Peak signal-to-noise ratio in dB.
    This is always returned as a raw dB value and must not be overwritten by score components.
    """
    x_true = np.asarray(x_true, dtype=float)
    x_hat = np.asarray(x_hat, dtype=float)

    mse = np.mean((x_hat - x_true) ** 2)
    if mse <= 1e-20:
        return float("inf")

    peak = np.max(np.abs(x_true)) + 1e-12
    return float(20.0 * np.log10(peak / np.sqrt(mse)))


def _safe_db(x):
    return 10.0 * np.log10(max(float(x), 1e-20))


def _bounded(x):
    """
    Nonnegative bounded transform for score components.
    """
    try:
        x = float(x)
        if not np.isfinite(x):
            return np.nan
        return abs(x) / (1.0 + abs(x))
    except Exception:
        return np.nan


def _nanmean(values):
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    return float(np.mean(arr)) if len(arr) else np.nan


def reconstructed_signal(Y_observed, imfs, residual):
    """
    Clean-signal proxy used for denoising/recovery metrics.

    For Y = signal + noise and a decomposition Y = sum(IMFs) + residual,
    the denoised reconstruction is interpreted as Y - residual.
    """
    try:
        return np.asarray(Y_observed, dtype=float) - np.asarray(residual, dtype=float)
    except Exception:
        imfs = _as_2d(imfs)
        if imfs.size == 0:
            return None
        return np.sum(imfs, axis=0)


# ============================================================
# Layer 2 — Robust estimation
# ============================================================

def reconstruction_accuracy(Y_observed, X_clean, imfs, residual):
    rec = reconstructed_signal(Y_observed, imfs, residual)

    if X_clean is None or rec is None:
        return {
            "denoise_psnr": np.nan,
            "denoise_mse": np.nan,
            "denoise_corr": np.nan,
        }

    X_clean = np.asarray(X_clean, dtype=float)
    rec = np.asarray(rec, dtype=float)

    if len(X_clean) != len(rec):
        return {
            "denoise_psnr": np.nan,
            "denoise_mse": np.nan,
            "denoise_corr": np.nan,
        }

    return {
        "denoise_psnr": _safe_psnr(X_clean, rec),
        "denoise_mse": float(np.mean((rec - X_clean) ** 2)),
        "denoise_corr": _safe_corr(X_clean, rec),
    }


def snr_gain_diagnostics(Y_observed, X_clean, imfs, residual):
    rec = reconstructed_signal(Y_observed, imfs, residual)

    if X_clean is None or rec is None:
        return {
            "input_snr_db": np.nan,
            "output_snr_db": np.nan,
            "snr_gain_db": np.nan,
        }

    Y_observed = np.asarray(Y_observed, dtype=float)
    X_clean = np.asarray(X_clean, dtype=float)
    rec = np.asarray(rec, dtype=float)

    if len(Y_observed) != len(X_clean) or len(rec) != len(X_clean):
        return {
            "input_snr_db": np.nan,
            "output_snr_db": np.nan,
            "snr_gain_db": np.nan,
        }

    signal_power = np.mean(X_clean ** 2) + 1e-20
    input_noise_power = np.mean((Y_observed - X_clean) ** 2) + 1e-20
    output_error_power = np.mean((rec - X_clean) ** 2) + 1e-20

    input_snr = _safe_db(signal_power / input_noise_power)
    output_snr = _safe_db(signal_power / output_error_power)

    return {
        "input_snr_db": float(input_snr),
        "output_snr_db": float(output_snr),
        "snr_gain_db": float(output_snr - input_snr),
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


def imf_recovery_diagnostics(imfs, true_components=None):
    """
    Synthetic-benchmark-only metric.

    Returns N/A-like values when true components are not available.
    """
    if true_components is None:
        return {
            "imf_recovery_rmse": np.nan,
            "imf_recovery_corr": np.nan,
            "imf_recovery_matched_count": 0,
        }

    imfs = _as_2d(imfs)
    true_components = _as_2d(true_components)

    if imfs.size == 0 or true_components.size == 0:
        return {
            "imf_recovery_rmse": np.nan,
            "imf_recovery_corr": np.nan,
            "imf_recovery_matched_count": 0,
        }

    if imfs.shape[1] != true_components.shape[1]:
        return {
            "imf_recovery_rmse": np.nan,
            "imf_recovery_corr": np.nan,
            "imf_recovery_matched_count": 0,
        }

    cost = np.ones((imfs.shape[0], true_components.shape[0]), dtype=float)

    for i in range(imfs.shape[0]):
        for j in range(true_components.shape[0]):
            c = _safe_corr(imfs[i], true_components[j])
            if np.isfinite(c):
                cost[i, j] = 1.0 - abs(c)

    pairs = _assignment(cost)
    rmses = []
    corrs = []

    for i, j in pairs:
        rmses.append(float(np.sqrt(np.mean((imfs[i] - true_components[j]) ** 2))))
        c = _safe_corr(imfs[i], true_components[j])
        if np.isfinite(c):
            corrs.append(abs(c))

    return {
        "imf_recovery_rmse": float(np.nanmean(rmses)) if rmses else np.nan,
        "imf_recovery_corr": float(np.nanmean(corrs)) if corrs else np.nan,
        "imf_recovery_matched_count": int(len(pairs)),
    }


def noise_capture_diagnostics(Y_observed, X_clean, residual):
    """
    Synthetic-benchmark-only residual/noise alignment metric.
    """
    if X_clean is None:
        return {
            "noise_capture_corr": np.nan,
            "noise_capture_energy_ratio": np.nan,
        }

    Y_observed = np.asarray(Y_observed, dtype=float)
    X_clean = np.asarray(X_clean, dtype=float)
    residual = np.asarray(residual, dtype=float)

    if len(Y_observed) != len(X_clean) or len(residual) != len(X_clean):
        return {
            "noise_capture_corr": np.nan,
            "noise_capture_energy_ratio": np.nan,
        }

    true_noise = Y_observed - X_clean

    return {
        "noise_capture_corr": _safe_corr(residual, true_noise),
        "noise_capture_energy_ratio": float(np.sum(residual ** 2) / (np.sum(true_noise ** 2) + 1e-12)),
    }


# ============================================================
# Layer 3 — Decomposition quality
# ============================================================

def strict_io(imfs):
    imfs = _as_2d(imfs)
    K = imfs.shape[0]

    if K <= 1:
        return 0.0

    vals = []

    for i in range(K):
        for j in range(i + 1, K):
            denom = np.linalg.norm(imfs[i]) * np.linalg.norm(imfs[j]) + 1e-12
            vals.append(abs(float(np.dot(imfs[i], imfs[j]) / denom)))

    return float(np.mean(vals)) if vals else 0.0


def classical_io_with_residual(Y_observed, imfs, residual):
    imfs = _as_2d(imfs)

    if imfs.size == 0:
        return 0.0

    components = list(imfs) + [np.asarray(residual, dtype=float)]
    vals = []

    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            denom = np.linalg.norm(components[i]) * np.linalg.norm(components[j]) + 1e-12
            vals.append(abs(float(np.dot(components[i], components[j]) / denom)))

    return float(np.mean(vals)) if vals else 0.0


def spectral_leakage(imfs):
    imfs = _as_2d(imfs)
    K = imfs.shape[0]

    if K <= 1:
        return 0.0

    psds = []

    for imf in imfs:
        p = np.abs(np.fft.rfft(imf)) ** 2
        p = p / (np.sum(p) + 1e-12)
        psds.append(p)

    P = np.vstack(psds)
    M = P @ P.T
    offdiag = ~np.eye(K, dtype=bool)

    return float(np.mean(M[offdiag]))


def frequency_overlap_matrix(imfs):
    imfs = _as_2d(imfs)
    K = imfs.shape[0]

    if K == 0:
        return np.zeros((0, 0))

    psds = []

    for imf in imfs:
        p = np.abs(np.fft.rfft(imf)) ** 2
        p = p / (np.sqrt(np.sum(p ** 2)) + 1e-12)
        psds.append(p)

    P = np.vstack(psds)
    return np.clip(P @ P.T, 0.0, 1.0)


def frequency_overlap_statistics(imfs):
    M = frequency_overlap_matrix(imfs)
    K = M.shape[0]

    if K <= 1:
        return {
            "frequency_overlap_matrix": M,
            "frequency_overlap_mean_offdiag": 0.0,
            "frequency_overlap_max_offdiag": 0.0,
            "frequency_overlap_adjacent_mean": 0.0,
            "frequency_overlap_adjacent_max": 0.0,
        }

    offdiag = ~np.eye(K, dtype=bool)
    adjacent = np.array([M[i, i + 1] for i in range(K - 1)])

    return {
        "frequency_overlap_matrix": M,
        "frequency_overlap_mean_offdiag": float(np.mean(M[offdiag])),
        "frequency_overlap_max_offdiag": float(np.max(M[offdiag])),
        "frequency_overlap_adjacent_mean": float(np.mean(adjacent)),
        "frequency_overlap_adjacent_max": float(np.max(adjacent)),
    }


def center_frequencies(imfs, fs):
    imfs = _as_2d(imfs)
    out = []

    for imf in imfs:
        p = np.abs(np.fft.rfft(imf)) ** 2
        f = np.fft.rfftfreq(len(imf), d=1.0 / fs)
        out.append(float(np.sum(f * p) / (np.sum(p) + 1e-12)))

    return np.asarray(out, dtype=float)


def imf_energy_ratios(imfs):
    imfs = _as_2d(imfs)

    if imfs.size == 0:
        return np.array([])

    e = np.sum(imfs ** 2, axis=1)
    return e / (np.sum(e) + 1e-12)


def dominant_freq_energy_pairs(center_freqs, imfs):
    cf = np.asarray(center_freqs, dtype=float)
    er = imf_energy_ratios(imfs)
    n = min(len(cf), len(er))

    return [(float(cf[i]), float(er[i])) for i in range(n)]


def residual_whiteness_penalty(residual, max_lag=20):
    residual = np.asarray(residual, dtype=float)
    residual = residual - np.mean(residual)

    if len(residual) <= 3:
        return 0.0

    denom = np.sum(residual ** 2) + 1e-12
    vals = []

    for lag in range(1, min(max_lag, len(residual) - 1) + 1):
        vals.append(float(np.sum(residual[:-lag] * residual[lag:]) / denom) ** 2)

    return float(np.mean(vals)) if vals else 0.0


def general_imf_count_penalty(imf_count, target_min=2, target_max=8):
    if imf_count < target_min:
        return float(target_min - imf_count) / max(target_min, 1)
    if imf_count > target_max:
        return float(imf_count - target_max) / max(target_max, 1)
    return 0.0


# ============================================================
# Layer 4 — Supporting diagnostics
# ============================================================

def frequency_spacing_diagnostics(center_freqs):
    cf = np.asarray(center_freqs, dtype=float)
    cf = cf[np.isfinite(cf)]
    cf = cf[cf > 1e-12]

    if len(cf) <= 1:
        return {
            "frequency_spacing_min_ratio": 0.0,
            "frequency_spacing_mean_ratio": 0.0,
            "frequency_spacing_penalty": 0.0,
            "frequency_separation_score": 1.0,
        }

    cf = np.sort(cf)
    ratios = np.diff(cf) / (cf[1:] + 1e-12)

    min_ratio = float(np.min(ratios))
    mean_ratio = float(np.mean(ratios))
    threshold = 0.25
    penalty = float(max(0.0, (threshold - min_ratio) / threshold))

    return {
        "frequency_spacing_min_ratio": min_ratio,
        "frequency_spacing_mean_ratio": mean_ratio,
        "frequency_spacing_penalty": penalty,
        "frequency_separation_score": float(1.0 - penalty),
    }


def instantaneous_frequency_smoothness(imfs):
    imfs = _as_2d(imfs)
    vals = []

    for imf in imfs:
        if np.std(imf) < 1e-12:
            continue

        phase = np.unwrap(np.angle(hilbert(imf)))
        om = np.diff(phase)

        if len(om) >= 5:
            om = medfilt(om, kernel_size=5)

        denom = np.mean(np.abs(om)) ** 2 + 1e-12
        vals.append(float(np.var(om) / denom))

    return float(np.mean(vals)) if vals else 0.0


def energy_ratio(Y_observed, imfs, residual):
    imfs = _as_2d(imfs)
    Y_observed = np.asarray(Y_observed, dtype=float)

    e_imfs = np.sum(imfs ** 2)
    e_y = np.sum(Y_observed ** 2) + 1e-12
    return float(e_imfs / e_y)


def imf_energy_concentration_penalty(imfs):
    r = imf_energy_ratios(imfs)

    if len(r) == 0:
        return 0.0

    entropy = -np.sum(r * np.log(r + 1e-12)) / np.log(len(r) + 1e-12)
    return float(max(0.0, np.max(r) - 0.85) + max(0.0, 0.20 - entropy))


# ============================================================
# Composite scores
# ============================================================

def _inverse_positive_score(x):
    """
    Higher raw x is better -> smaller score component.
    """
    try:
        x = float(x)
        if not np.isfinite(x):
            return np.nan
        return 1.0 / (1.0 + max(x, 0.0))
    except Exception:
        return np.nan


def robust_estimation_score(result):
    """
    Smaller is better.

    Raw metrics are never overwritten.
    """
    psnr = result.get("denoise_psnr", np.nan)
    mse = result.get("denoise_mse", np.nan)
    corr = result.get("denoise_corr", np.nan)
    snr_gain = result.get("snr_gain_db", np.nan)
    imf_rmse = result.get("imf_recovery_rmse", np.nan)
    imf_corr = result.get("imf_recovery_corr", np.nan)
    noise_corr = result.get("noise_capture_corr", np.nan)

    components = {
        "psnr_score_component": _inverse_positive_score(psnr),
        "mse_score_component": _bounded(mse),
        "corr_score_component": 1.0 - corr if np.isfinite(corr) else np.nan,
        "snr_gain_score_component": _inverse_positive_score(snr_gain),
        "imf_recovery_rmse_score_component": _bounded(imf_rmse),
        "imf_recovery_corr_score_component": 1.0 - imf_corr if np.isfinite(imf_corr) else np.nan,
        "noise_capture_corr_score_component": 1.0 - abs(noise_corr) if np.isfinite(noise_corr) else np.nan,
    }

    return _nanmean(list(components.values())), components


def decomposition_quality_score(result):
    imf_count = result.get("imf_count", 0)

    components = {
        "strict_io_score_component": result.get("strict_io", np.nan),
        "spectral_leakage_score_component": result.get("spectral_leakage", np.nan),
        "frequency_overlap_score_component": result.get("frequency_overlap_max_offdiag", np.nan),
        "residual_whiteness_score_component": result.get("residual_whiteness", np.nan),
        "imf_count_score_component": general_imf_count_penalty(imf_count),
    }

    return _nanmean(list(components.values())), components


# ============================================================
# Main entry point
# ============================================================

def evaluate_shared_physical_diagnostics(
        Y_observed,
        X_clean,
        imfs,
        residual,
        fs,
        residual_penalty_mode="whiteness",
        true_components=None
):
    imfs = _as_2d(imfs)

    rec_stats = reconstruction_accuracy(Y_observed, X_clean, imfs, residual)
    snr_stats = snr_gain_diagnostics(Y_observed, X_clean, imfs, residual)
    recovery_stats = imf_recovery_diagnostics(imfs, true_components=true_components)
    noise_stats = noise_capture_diagnostics(Y_observed, X_clean, residual)

    cf = center_frequencies(imfs, fs)
    fover = frequency_overlap_statistics(imfs)
    spacing = frequency_spacing_diagnostics(cf)

    io = strict_io(imfs)
    cio = classical_io_with_residual(Y_observed, imfs, residual)
    leak = spectral_leakage(imfs)
    white = residual_whiteness_penalty(residual)
    ifs = instantaneous_frequency_smoothness(imfs)
    er = energy_ratio(Y_observed, imfs, residual)
    energy_penalty = imf_energy_concentration_penalty(imfs)

    result = {
        **rec_stats,
        **snr_stats,
        **recovery_stats,
        **noise_stats,
        "strict_io": io,
        "io": io,
        "classical_io": cio,
        "spectral_leakage": leak,
        "center_freqs": cf,
        "imf_energy_ratios": imf_energy_ratios(imfs),
        "dominant_freq_energy_pairs": dominant_freq_energy_pairs(cf, imfs),
        "residual_whiteness": white,
        "residual_autocorrelation_score": white,
        "ifs": ifs,
        "instantaneous_frequency_smoothness": ifs,
        "energy_ratio": er,
        "energy_concentration_penalty": energy_penalty,
        "imf_count": int(imfs.shape[0]),
        **fover,
        **spacing,
    }

    result["frequency_separation_score"] = float(
        max(0.0, 1.0 - result.get("frequency_overlap_max_offdiag", 0.0))
    )

    robust_score, robust_components = robust_estimation_score(result)
    decomp_score, decomp_components = decomposition_quality_score(result)

    result.update(robust_components)
    result.update(decomp_components)

    result["robust_estimation_score"] = robust_score
    result["decomposition_quality_score"] = decomp_score

    # Backward-compatible field name.
    result["general_physical_score"] = result["decomposition_quality_score"]

    return result
