#!/usr/bin/python
# coding: UTF-8

"""Ground-truth-free proxy diagnostics for real-world validation.

These diagnostics are intentionally method-agnostic.  They do not assume that
the final residual has the same physical meaning for IRMF and EMD-family
methods.  Component-level summaries are therefore based on spectral scale
properties rather than extraction index.
"""

import numpy as np

EPS = 1e-12


def _as_1d(x):
    if x is None:
        return np.asarray([], dtype=float)
    x = np.asarray(x, dtype=float).ravel()
    return x[np.isfinite(x)]


def _as_2d(x, n=None):
    if x is None:
        return np.zeros((0, int(n or 0)), dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        return np.zeros((0, int(n or 0)), dtype=float)
    return x


def _safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size or a.size < 3:
        return np.nan
    if np.std(a) < EPS or np.std(b) < EPS:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _acf_abs_mean(x, max_lag=40):
    x = _as_1d(x)
    if x.size < 4:
        return np.nan
    x = x - np.mean(x)
    denom = float(np.sum(x ** 2) + EPS)
    vals = []
    for lag in range(1, min(int(max_lag), x.size - 1) + 1):
        vals.append(abs(float(np.sum(x[:-lag] * x[lag:]) / denom)))
    return float(np.mean(vals)) if vals else np.nan


def _kurtosis(x):
    x = _as_1d(x)
    if x.size < 4:
        return np.nan
    z = x - np.mean(x)
    v = float(np.mean(z ** 2) + EPS)
    return float(np.mean(z ** 4) / (v ** 2))


def _hoyer_sparsity(x):
    x = np.abs(_as_1d(x))
    n = x.size
    if n <= 1:
        return np.nan
    l1 = float(np.sum(x))
    l2 = float(np.sqrt(np.sum(x ** 2)) + EPS)
    return float((np.sqrt(n) - l1 / l2) / (np.sqrt(n) - 1.0))


def _spectral_energy_ratio(x, fs, low=None, high=None):
    x = _as_1d(x)
    if x.size < 4:
        return np.nan
    y = x - np.mean(x)
    freqs = np.fft.rfftfreq(y.size, d=1.0 / float(fs))
    power = np.abs(np.fft.rfft(y)) ** 2
    total = float(np.sum(power) + EPS)
    mask = np.ones_like(freqs, dtype=bool)
    if low is not None:
        mask &= freqs >= float(low)
    if high is not None:
        mask &= freqs <= float(high)
    return float(np.sum(power[mask]) / total)


def _spectral_concentration_one(x, fs):
    x = _as_1d(x)
    if x.size < 4 or np.std(x) < EPS:
        return np.nan
    y = x - np.mean(x)
    power = np.abs(np.fft.rfft(y)) ** 2
    total = float(np.sum(power) + EPS)
    p = power / total
    entropy = -float(np.sum(p * np.log(p + EPS)))
    max_entropy = float(np.log(max(2, p.size)))
    return float(1.0 - entropy / (max_entropy + EPS))


def _center_frequency_one(x, fs):
    x = _as_1d(x)
    if x.size < 4 or np.std(x) < EPS:
        return np.nan
    y = x - np.mean(x)
    freqs = np.fft.rfftfreq(y.size, d=1.0 / float(fs))
    power = np.abs(np.fft.rfft(y)) ** 2
    return float(np.sum(freqs * power) / (np.sum(power) + EPS))


def _if_continuity_one(x, fs):
    x = _as_1d(x)
    if x.size < 8 or np.std(x) < EPS:
        return np.nan
    try:
        from scipy.signal import hilbert
        analytic = hilbert(x)
        phase = np.unwrap(np.angle(analytic))
        inst_freq = float(fs) * np.diff(phase) / (2.0 * np.pi)
    except Exception:
        return np.nan
    inst_freq = inst_freq[np.isfinite(inst_freq)]
    if inst_freq.size < 4:
        return np.nan
    roughness = float(np.median(np.abs(np.diff(inst_freq))) / (np.median(np.abs(inst_freq)) + EPS))
    return float(1.0 / (1.0 + roughness))


def _scale_order_consistency(center_freqs):
    cf = np.asarray([v for v in center_freqs if np.isfinite(v)], dtype=float)
    if cf.size <= 2:
        return np.nan
    diffs = np.diff(cf)
    inc = np.mean(diffs >= -EPS)
    dec = np.mean(diffs <= EPS)
    return float(max(inc, dec))


def _frequency_band_separation(center_freqs):
    cf = np.sort(np.asarray([v for v in center_freqs if np.isfinite(v)], dtype=float))
    if cf.size <= 1:
        return np.nan
    diffs = np.diff(cf)
    denom = float(np.max(cf) - np.min(cf) + EPS)
    return float(np.min(diffs) / denom)


def compute_real_proxy_diagnostics(Y_observed, imfs, residual, fs):
    """Return proxy diagnostics usable for synthetic calibration and real data."""
    y = np.asarray(Y_observed, dtype=float).ravel()
    residual = np.asarray(residual if residual is not None else np.zeros_like(y), dtype=float).ravel()
    imfs = _as_2d(imfs, n=y.size)
    if residual.size != y.size:
        residual = np.zeros_like(y)
    fs = float(fs or 1.0)
    nyquist = fs / 2.0

    abs_resid = np.abs(residual - np.median(residual))
    cutoff = np.quantile(abs_resid, 0.95) if abs_resid.size else np.nan
    normal_resid = residual[abs_resid <= cutoff] if np.isfinite(cutoff) else residual

    reconstructed = np.sum(imfs, axis=0) + residual if imfs.size else residual
    reconstruction_identity_error = float(
        np.sum((reconstructed - y) ** 2) / (np.sum(y ** 2) + EPS)
    ) if y.size else np.nan

    concentrations = [_spectral_concentration_one(comp, fs) for comp in imfs]
    continuities = [_if_continuity_one(comp, fs) for comp in imfs]
    center_freqs = [_center_frequency_one(comp, fs) for comp in imfs]

    low_cut = 0.10 * nyquist
    high_cut = 0.50 * nyquist
    residual_high = _spectral_energy_ratio(residual, fs, low=high_cut)
    residual_low = _spectral_energy_ratio(residual, fs, high=low_cut)

    return {
        "proxy_metric_protocol": "V5.21_ground_truth_free_real_proxy_diagnostics",
        "trimmed_residual_whiteness": _acf_abs_mean(normal_resid),
        "residual_autocorrelation_abs_mean": _acf_abs_mean(residual),
        "residual_kurtosis": _kurtosis(residual),
        "residual_sparsity": _hoyer_sparsity(residual),
        "residual_high_frequency_energy_ratio": residual_high,
        "residual_low_frequency_leakage": residual_low,
        "residual_outlier_concentration": float(np.mean(abs_resid > cutoff)) if np.isfinite(cutoff) else np.nan,
        "spectral_concentration_mean": float(np.nanmean(concentrations)) if concentrations else np.nan,
        "spectral_concentration_min": float(np.nanmin(concentrations)) if concentrations else np.nan,
        "instantaneous_frequency_continuity_mean": float(np.nanmean(continuities)) if continuities else np.nan,
        "frequency_band_separation": _frequency_band_separation(center_freqs),
        "scale_order_consistency": _scale_order_consistency(center_freqs),
        "component_center_frequency_min": float(np.nanmin(center_freqs)) if center_freqs else np.nan,
        "component_center_frequency_max": float(np.nanmax(center_freqs)) if center_freqs else np.nan,
        "energy_conservation_error": reconstruction_identity_error,
        "real_proxy_no_ground_truth_required": True,
    }

