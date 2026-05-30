#!/usr/bin/python
# coding: UTF-8

"""
Noise model bank for V8 robustness benchmark.

Noise types:
- gaussian
- laplace
- student_t
- impulsive
- burst
- colored_ar1
- pink_approx
"""

import numpy as np


def _standardize(x):
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    std = np.std(x) + 1e-10
    return x / std


def gaussian_noise(n, rng):
    return _standardize(rng.standard_normal(n))


def laplace_noise(n, rng):
    return _standardize(rng.laplace(loc=0.0, scale=1.0, size=n))


def student_t_noise(n, rng, df=3):
    return _standardize(rng.standard_t(df=df, size=n))


def impulsive_noise(n, rng, spike_prob=0.03, spike_scale=8.0):
    base = 0.25 * rng.standard_normal(n)
    spikes = rng.random(n) < spike_prob
    spike_values = spike_scale * rng.standard_normal(n) * spikes
    return _standardize(base + spike_values)


def burst_noise(n, rng, burst_fraction=0.08, burst_scale=6.0):
    base = 0.25 * rng.standard_normal(n)
    burst_len = max(3, int(n * burst_fraction))
    start = int(rng.integers(0, max(1, n - burst_len)))
    burst = np.zeros(n)
    burst[start:start + burst_len] = burst_scale * rng.standard_normal(burst_len)
    return _standardize(base + burst)


def colored_ar1_noise(n, rng, phi=0.85):
    eps = rng.standard_normal(n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return _standardize(x)


def pink_approx_noise(n, rng):
    """
    Approximate pink noise using frequency-domain 1/sqrt(f) shaping.
    """
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    shaped = np.fft.irfft(spec * scale, n=n)
    return _standardize(shaped)


NOISE_REGISTRY = {
    "gaussian": gaussian_noise,
    "laplace": laplace_noise,
    "student_t": student_t_noise,
    "impulsive": impulsive_noise,
    "burst": burst_noise,
    "colored_ar1": colored_ar1_noise,
    "pink_approx": pink_approx_noise,
}


def list_noise_models():
    return list(NOISE_REGISTRY.keys())


def get_noise(noise_name, n, sigma, seed=0):
    if noise_name not in NOISE_REGISTRY:
        raise ValueError(f"Unknown noise model {noise_name}. Available: {list_noise_models()}")

    rng = np.random.default_rng(seed)
    unit_noise = NOISE_REGISTRY[noise_name](n, rng)
    return sigma * unit_noise
