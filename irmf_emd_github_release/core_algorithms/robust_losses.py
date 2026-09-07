#!/usr/bin/python
# coding: UTF-8
"""Robust loss registry used by IRMF and loss-function ablations.

Every loss returns ``rho, psi, curvature`` for standardized residuals.  The
registry keeps tuning explicit and records whether a loss is convex, smooth,
and redescending.  Gaussian-smoothed median remains the paper-default loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import erf
from typing import Callable, Dict, Mapping, Tuple

import numpy as np

try:
    from scipy.special import ndtr as _normal_cdf
except Exception:  # pragma: no cover
    _normal_cdf = None

ArrayTriple = Tuple[np.ndarray, np.ndarray, np.ndarray]


def _cdf(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if _normal_cdf is not None:
        return _normal_cdf(z)
    return 0.5 * (1.0 + np.vectorize(erf)(z / np.sqrt(2.0)))


def l2_loss(x, scale=1.0, **_) -> ArrayTriple:
    x = np.asarray(x, dtype=float)
    return 0.5 * x**2, x, np.ones_like(x)


def l1_loss(x, scale=1.0, smooth_eps=1e-8, **_) -> ArrayTriple:
    """Numerically regularized L1 for the Newton interface.

    The reported loss is exact |x|.  The score/curvature use a tiny smooth
    approximation solely to avoid a singular Hessian in numerical experiments.
    """
    x = np.asarray(x, dtype=float)
    eps = max(float(smooth_eps), 1e-12)
    denom = np.sqrt(x*x + eps*eps)
    return np.abs(x), x / denom, (eps*eps) / (denom**3)


def huber_loss(x, delta=1.345, **_) -> ArrayTriple:
    x = np.asarray(x, dtype=float)
    d = max(float(delta), 1e-12)
    ax = np.abs(x)
    rho = np.where(ax <= d, 0.5*x*x, d*(ax - 0.5*d))
    psi = np.clip(x, -d, d)
    curvature = (ax < d).astype(float)
    return rho, psi, curvature


def pseudo_huber_loss(x, delta=1.0, **_) -> ArrayTriple:
    x = np.asarray(x, dtype=float)
    d = max(float(delta), 1e-12)
    z = x / d
    root = np.sqrt(1.0 + z*z)
    return d*d*(root - 1.0), x/root, 1.0/(root**3)


def fair_loss(x, c=1.4, **_) -> ArrayTriple:
    x = np.asarray(x, dtype=float)
    c = max(float(c), 1e-12)
    ax = np.abs(x)
    rho = c*c * (ax/c - np.log1p(ax/c))
    psi = x / (1.0 + ax/c)
    curvature = 1.0 / (1.0 + ax/c)**2
    return rho, psi, curvature


def tukey_biweight_loss(x, c=4.685, **_) -> ArrayTriple:
    x = np.asarray(x, dtype=float)
    c = max(float(c), 1e-12)
    z = x/c
    inside = np.abs(z) < 1.0
    one_minus = 1.0 - z*z
    rho = np.full_like(x, c*c/6.0)
    rho[inside] = (c*c/6.0) * (1.0 - one_minus[inside]**3)
    psi = np.zeros_like(x)
    psi[inside] = x[inside] * one_minus[inside]**2
    curvature = np.zeros_like(x)
    curvature[inside] = one_minus[inside] * (1.0 - 5.0*z[inside]**2)
    return rho, psi, curvature


def gaussian_smoothed_median_loss(x, H=1.0, **_) -> ArrayTriple:
    x = np.asarray(x, dtype=float)
    H = max(float(H), 1e-12)
    z = x/H
    phi = np.exp(-0.5*z*z) / np.sqrt(2.0*np.pi)
    Phi = _cdf(z)
    rho = np.sqrt(2.0/np.pi)*H*np.exp(-0.5*z*z) + x*(2.0*Phi - 1.0)
    psi = 2.0*Phi - 1.0
    curvature = (2.0/H)*phi
    return rho, psi, curvature


@dataclass(frozen=True)
class LossSpec:
    name: str
    function: Callable[..., ArrayTriple]
    default_tuning: Mapping[str, float]
    convex: bool
    smooth: bool
    bounded_score: bool
    redescending: bool


LOSS_REGISTRY: Dict[str, LossSpec] = {
    "l2": LossSpec("L2", l2_loss, {}, True, True, False, False),
    "l1": LossSpec("L1", l1_loss, {"smooth_eps": 1e-8}, True, False, True, False),
    "huber": LossSpec("Huber", huber_loss, {"delta": 1.345}, True, False, True, False),
    "pseudo_huber": LossSpec("Pseudo-Huber", pseudo_huber_loss, {"delta": 1.0}, True, True, True, False),
    "fair": LossSpec("Fair", fair_loss, {"c": 1.4}, True, True, True, False),
    "tukey": LossSpec("Tukey biweight", tukey_biweight_loss, {"c": 4.685}, False, True, True, True),
    "gaussian_smoothed_median": LossSpec(
        "Gaussian-smoothed median", gaussian_smoothed_median_loss, {"H": 1.0}, True, True, True, False
    ),
}

ALIASES = {
    "gsm": "gaussian_smoothed_median",
    "smoothed_median": "gaussian_smoothed_median",
    "pseudo-huber": "pseudo_huber",
    "tukey_biweight": "tukey",
}


def canonical_loss_name(name: str) -> str:
    key = str(name).strip().lower().replace(" ", "_")
    key = ALIASES.get(key, key)
    if key not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {sorted(LOSS_REGISTRY)}")
    return key


def evaluate_loss(x, loss_name="gaussian_smoothed_median", tuning=None, H=None) -> ArrayTriple:
    key = canonical_loss_name(loss_name)
    spec = LOSS_REGISTRY[key]
    params = dict(spec.default_tuning)
    if tuning:
        params.update(tuning)
    if key == "gaussian_smoothed_median" and H is not None:
        params["H"] = H
    return spec.function(np.asarray(x, dtype=float), **params)


def loss_metadata(loss_name: str) -> dict:
    key = canonical_loss_name(loss_name)
    spec = LOSS_REGISTRY[key]
    return {
        "loss_key": key,
        "loss_name": spec.name,
        "default_tuning": dict(spec.default_tuning),
        "convex": spec.convex,
        "smooth": spec.smooth,
        "bounded_score": spec.bounded_score,
        "redescending": spec.redescending,
    }
