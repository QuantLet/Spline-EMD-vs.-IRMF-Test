#!/usr/bin/python
# coding: UTF-8

"""
Hilbert time-frequency diagnostics.

Computes instantaneous amplitude and frequency summaries for IMF sets.
"""

import numpy as np
from scipy.signal import hilbert, medfilt


def hilbert_if_amplitude(imf, fs, median_kernel=5):
    analytic = hilbert(imf)
    amp = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    freq = fs * np.diff(phase) / (2.0 * np.pi)

    if len(freq) >= 5:
        k = min(median_kernel, len(freq))
        if k % 2 == 0:
            k -= 1
        if k >= 3:
            freq = medfilt(freq, kernel_size=k)

    if len(freq) > 0:
        cap = np.quantile(np.abs(freq), 0.98)
        freq = np.clip(freq, -cap, cap)

    return freq, amp[:-1]


def compute_hilbert_tf_diagnostics(imfs, fs):
    weighted_if_means = []
    weighted_if_stds = []
    amplitude_concentration = []

    for imf in imfs:
        if np.std(imf) < 1e-10 or len(imf) < 4:
            continue

        freq, amp = hilbert_if_amplitude(imf, fs)
        if len(freq) == 0:
            continue

        w = amp ** 2
        w = w / (np.sum(w) + 1e-10)

        mu = float(np.sum(w * freq))
        sd = float(np.sqrt(np.sum(w * (freq - mu) ** 2)))

        weighted_if_means.append(mu)
        weighted_if_stds.append(sd)
        amplitude_concentration.append(float(np.max(w)))

    if len(weighted_if_means) == 0:
        return {
            "hilbert_if_mean_avg": 0.0,
            "hilbert_if_std_avg": 0.0,
            "hilbert_if_std_max": 0.0,
            "hilbert_amp_concentration_avg": 0.0,
        }

    return {
        "hilbert_if_mean_avg": float(np.mean(weighted_if_means)),
        "hilbert_if_std_avg": float(np.mean(weighted_if_stds)),
        "hilbert_if_std_max": float(np.max(weighted_if_stds)),
        "hilbert_amp_concentration_avg": float(np.mean(amplitude_concentration)),
    }
