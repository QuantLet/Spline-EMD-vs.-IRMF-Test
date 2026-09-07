#!/usr/bin/python
# coding: UTF-8

"""
Configuration-level scoring for FIX10 paper-final benchmark.

This module aggregates case-level scores across:
    7 signals × 8 noise models × 3 sigma levels = 168 cases

Configuration-level evaluation is used for:
    - Best IRMF parameter selection across benchmark cases
    - Best EMD parameter selection across benchmark cases
    - Robustness analysis across noise type, sigma level, and Monte Carlo repetitions

All score-like quantities use the convention:
    1 = good
    0 = bad
"""

import numpy as np
from collections import defaultdict


EPS = 1e-12


def _finite_array(values):
    arr = np.asarray([float(v) for v in values if v is not None and np.isfinite(v)], dtype=float)
    return arr


def _stability_from_values(values):
    """
    Convert standard deviation into [0,1] stability score.
        std = 0 -> 1
        larger std -> smaller score
    """
    arr = _finite_array(values)
    if len(arr) == 0:
        return np.nan
    return float(np.exp(-np.std(arr)))


def _mean(values):
    arr = _finite_array(values)
    return float(np.mean(arr)) if len(arr) else np.nan


def _min(values):
    arr = _finite_array(values)
    return float(np.min(arr)) if len(arr) else np.nan


def case_score_from_result(result):
    """
    Return case_score_final/case_score from one run result.
    """
    if result is None:
        return np.nan
    if "case_score_final" in result:
        return result.get("case_score_final", np.nan)
    if "case_score" in result:
        return result.get("case_score", np.nan)
    if "general_physical_score" in result:
        # Backward compatibility: general_physical_score = 1 - case_score_final.
        gps = result.get("general_physical_score", np.nan)
        return 1.0 - gps if np.isfinite(gps) else np.nan
    return np.nan


def aggregate_configuration_scores(rows):
    """
    Aggregate a list of run/case result dictionaries.

    Each row should ideally contain:
        case_score_final
        signal
        noise
        sigma
        mc_trial / trial / seed, optional

    Returns:
        mean_case_score
        worst_case_score
        noise_stability
        sigma_stability
        mc_stability
        configuration_score
    """
    rows = list(rows or [])
    scores = [case_score_from_result(r) for r in rows]

    mean_case_score = _mean(scores)
    worst_case_score = _min(scores)

    # Cross-noise stability.
    by_noise = defaultdict(list)
    for r in rows:
        noise = r.get("noise", r.get("noise_name", None))
        if noise is None:
            continue
        by_noise[noise].append(case_score_from_result(r))
    noise_means = [_mean(v) for v in by_noise.values()]
    noise_stability = _stability_from_values(noise_means)

    # Cross-sigma stability.
    by_sigma = defaultdict(list)
    for r in rows:
        sigma = r.get("sigma", None)
        if sigma is None:
            continue
        by_sigma[float(sigma)].append(case_score_from_result(r))
    sigma_means = [_mean(v) for v in by_sigma.values()]
    sigma_stability = _stability_from_values(sigma_means)

    # Monte Carlo stability, if repetitions exist.
    by_mc = defaultdict(list)
    for r in rows:
        key = (
            r.get("mc_trial", None)
            if "mc_trial" in r else
            r.get("trial", None)
            if "trial" in r else
            r.get("seed", None)
        )
        if key is None:
            continue
        by_mc[key].append(case_score_from_result(r))
    mc_means = [_mean(v) for v in by_mc.values()]
    mc_stability = _stability_from_values(mc_means) if len(mc_means) >= 2 else np.nan

    values = []
    weights = []

    # Recommended final configuration-level score:
    # 0.60 MeanScore + 0.15 WorstCase + 0.10 NoiseStability
    # + 0.10 SigmaStability + 0.05 MCStability
    for value, weight in [
        (mean_case_score, 0.60),
        (worst_case_score, 0.15),
        (noise_stability, 0.10),
        (sigma_stability, 0.10),
        (mc_stability, 0.05),
    ]:
        if value is not None and np.isfinite(value):
            values.append(value)
            weights.append(weight)

    if values:
        weights = np.asarray(weights, dtype=float)
        weights = weights / (np.sum(weights) + EPS)
        configuration_score = float(np.sum(weights * np.asarray(values, dtype=float)))
    else:
        configuration_score = np.nan

    return {
        "mean_case_score": mean_case_score,
        "worst_case_score": worst_case_score,
        "noise_stability": noise_stability,
        "sigma_stability": sigma_stability,
        "mc_stability": mc_stability,
        "configuration_score": configuration_score,
        # Backward-compatible minimization form:
        "configuration_loss": float(1.0 - configuration_score) if np.isfinite(configuration_score) else np.inf,
        "n_cases": int(len(_finite_array(scores))),
        "n_noise_groups": int(len(by_noise)),
        "n_sigma_groups": int(len(by_sigma)),
        "n_mc_groups": int(len(by_mc)),
    }
