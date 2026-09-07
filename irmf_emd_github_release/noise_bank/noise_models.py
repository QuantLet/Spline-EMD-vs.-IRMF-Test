#!/usr/bin/python
# coding: UTF-8

"""
Noise Bank for the IRMF vs EMD benchmark suite.

Updated FIX9+ benchmark design:
    7 signals × 8 noise models = 56 main benchmark cases

Noise Bank:
    1. gaussian              — light-tailed baseline noise
    2. laplace               — heavy-tailed noise
    3. student_t             — fat-tailed financial-style noise
    4. impulsive             — sparse pointwise spikes
    5. burst                 — local short corrupted intervals
    6. huber_contamination   — epsilon-contamination / robust statistics
    7. ar1_colored           — serially dependent colored noise
    8. heteroskedastic       — time-varying volatility noise
"""

from typing import List

import numpy as np

DTYPE = np.float64


# ============================================================
# Helpers
# ============================================================

def _standardize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=DTYPE)
    x = x - np.mean(x)
    s = np.std(x)
    if s < eps:
        return x.astype(DTYPE)
    return (x / s).astype(DTYPE)


# ============================================================
# Raw standardized noise generators
# ============================================================

def gaussian_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Light-tailed Gaussian baseline noise."""
    return _standardize(rng.standard_normal(n))


def laplace_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Heavy-tailed Laplace noise."""
    return _standardize(rng.laplace(0.0, 1.0, size=n))


def student_t_noise(n: int, rng: np.random.Generator, df: float = 3.0) -> np.ndarray:
    """Fat-tailed Student-t noise."""
    return _standardize(rng.standard_t(df=df, size=n))


def impulsive_noise(
    n: int,
    rng: np.random.Generator,
    p: float = 0.03,
    scale: float = 8.0,
) -> np.ndarray:
    """
    Sparse pointwise impulse noise.

    Most observations are approximately Gaussian, while a small fraction p
    receives large random spikes.
    """
    x = rng.standard_normal(n)
    mask = rng.random(n) < p
    if np.any(mask):
        signs = rng.choice([-1.0, 1.0], size=int(np.sum(mask)))
        magnitudes = np.abs(rng.normal(loc=scale, scale=0.25 * scale, size=int(np.sum(mask))))
        x[mask] += signs * magnitudes
    return _standardize(x)


def burst_noise(
    n: int,
    rng: np.random.Generator,
    n_bursts: int = 3,
    burst_len: int = None,
    scale: float = 6.0,
) -> np.ndarray:
    """
    Local burst noise.

    Adds high-variance corruption on a few short contiguous intervals.
    """
    x = rng.standard_normal(n)
    if burst_len is None:
        burst_len = max(3, n // 50)
    burst_len = int(min(max(1, burst_len), n))

    for _ in range(int(n_bursts)):
        start = int(rng.integers(0, max(1, n - burst_len + 1)))
        end = min(n, start + burst_len)
        x[start:end] += rng.normal(0.0, scale, size=end - start)

    return _standardize(x)


def huber_contamination_noise(
    n: int,
    rng: np.random.Generator,
    lam: float = 0.05,
    outlier_scale: float = 10.0,
) -> np.ndarray:
    """
    Huber epsilon-contamination noise model.

        epsilon_i ~ (1 - lambda) N(0, 1)
                    + lambda H_outlier

    H_outlier is implemented as symmetric large outliers.
    The returned vector is standardized so the final sigma control remains
    comparable with the other noise models.
    """
    x = rng.standard_normal(n)
    mask = rng.random(n) < lam
    if np.any(mask):
        signs = rng.choice([-1.0, 1.0], size=int(np.sum(mask)))
        magnitudes = np.abs(rng.normal(outlier_scale, 0.2 * outlier_scale, size=int(np.sum(mask))))
        x[mask] = signs * magnitudes
    return _standardize(x)


def ar1_colored_noise(n: int, rng: np.random.Generator, phi: float = 0.85) -> np.ndarray:
    """
    AR(1) colored noise.

        z_t = phi z_{t-1} + e_t

    Tests robustness under serial dependence.
    """
    eps = rng.standard_normal(n)
    x = np.zeros(n, dtype=DTYPE)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return _standardize(x)


def heteroskedastic_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Heteroskedastic noise with time-varying variance.

    Useful for financial-style volatility clustering / changing noise level.
    """
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    local_sigma = 0.5 + 1.5 * (np.sin(2 * np.pi * 2.0 * t) ** 2)
    x = local_sigma * rng.standard_normal(n)
    return _standardize(x)


# Backward-compatible alias for older code.
colored_ar1_noise = ar1_colored_noise


NOISE_REGISTRY = {
    "gaussian": gaussian_noise,
    "laplace": laplace_noise,
    "student_t": student_t_noise,
    "impulsive": impulsive_noise,
    "burst": burst_noise,
    "huber_contamination": huber_contamination_noise,
    "ar1_colored": ar1_colored_noise,
    "heteroskedastic": heteroskedastic_noise,
}


# ============================================================
# Public API
# ============================================================

def list_noise_models() -> List[str]:
    return list(NOISE_REGISTRY.keys())


def generate_noise(noise_name: str, n: int, sigma: float = 1.0, seed: int = 0, **kwargs) -> np.ndarray:
    """
    Generate additive noise with approximately standard deviation = sigma.

    Parameters
    ----------
    noise_name:
        One of NOISE_REGISTRY.
    n:
        Number of observations.
    sigma:
        Final noise scale after standardization.
    seed:
        Random seed.
    kwargs:
        Noise-specific parameters.
    """
    rng = np.random.default_rng(seed)
    name = str(noise_name)

    if name == "gaussian":
        z = gaussian_noise(n, rng)

    elif name == "laplace":
        z = laplace_noise(n, rng)

    elif name in ("student_t", "student-t", "student"):
        z = student_t_noise(n, rng, df=kwargs.get("df", 3.0))

    elif name == "impulsive":
        z = impulsive_noise(
            n,
            rng,
            p=kwargs.get("p", 0.03),
            scale=kwargs.get("scale", 8.0),
        )

    elif name == "burst":
        z = burst_noise(
            n,
            rng,
            n_bursts=kwargs.get("n_bursts", 3),
            burst_len=kwargs.get("burst_len", None),
            scale=kwargs.get("scale", 6.0),
        )

    elif name in ("huber_contamination", "huber", "epsilon_contamination"):
        z = huber_contamination_noise(
            n,
            rng,
            lam=kwargs.get("lam", kwargs.get("lambda_", 0.05)),
            outlier_scale=kwargs.get("outlier_scale", 10.0),
        )

    elif name in ("ar1_colored", "colored_ar1", "ar1"):
        z = ar1_colored_noise(n, rng, phi=kwargs.get("phi", 0.85))

    elif name in ("heteroskedastic", "heteroscedastic", "volatility"):
        z = heteroskedastic_noise(n, rng)

    else:
        raise ValueError(
            f"Unknown noise model: {noise_name}. Available noise models: {list_noise_models()}"
        )

    return (float(sigma) * z).astype(DTYPE)


# Older experiment_full_signal_noise_robustness.py imported get_noise.
# Keep this alias to avoid breaking legacy scripts.
def get_noise(noise_name: str, n: int, sigma: float = 1.0, seed: int = 0, **kwargs) -> np.ndarray:
    return generate_noise(noise_name=noise_name, n=n, sigma=sigma, seed=seed, **kwargs)
