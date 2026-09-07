#!/usr/bin/python
# coding: UTF-8

import numpy as np


def compute_frequency_spacing_diagnostics(center_freqs):
    cf = np.asarray(center_freqs, dtype=float)
    cf = cf[np.isfinite(cf)]
    cf = cf[cf > 1e-10]

    if len(cf) <= 1:
        return {
            "frequency_spacing_min_ratio": 0.0,
            "frequency_spacing_mean_ratio": 0.0,
            "frequency_spacing_penalty": 0.0,
        }

    cf = np.sort(cf)
    diffs = np.diff(cf)
    ratios = diffs / (cf[1:] + 1e-10)

    min_ratio = float(np.min(ratios))
    mean_ratio = float(np.mean(ratios))

    threshold = 0.25
    spacing_penalty = float(max(0.0, (threshold - min_ratio) / threshold))

    return {
        "frequency_spacing_min_ratio": min_ratio,
        "frequency_spacing_mean_ratio": mean_ratio,
        "frequency_spacing_penalty": spacing_penalty,
    }
