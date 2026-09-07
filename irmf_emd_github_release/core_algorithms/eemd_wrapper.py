#!/usr/bin/python
# coding: UTF-8

"""EEMD wrapper."""

import numpy as np
from time import perf_counter
from core_algorithms.emd_wrapper import final_residual


def run_eemd(y, max_imf=-1, trials=50, noise_width=0.05, parallel=False, random_seed=0):
    try:
        from PyEMD import EEMD
    except Exception as exc:
        raise ImportError("EEMD requires PyEMD with EEMD support.") from exc

    eemd = EEMD(trials=trials, noise_width=noise_width, parallel=parallel)
    eemd.noise_seed(int(random_seed))
    start = perf_counter()
    imfs = eemd.eemd(np.asarray(y, dtype=float), max_imf=max_imf)
    elapsed = perf_counter() - start
    residual = final_residual(y, imfs)

    return {
        "method": "EEMD",
        "imfs": imfs,
        "residual": residual,
        "trials": trials,
        "noise_width": noise_width,
        "random_seed": random_seed,
        "algorithm_seed": random_seed,
        "max_imf": max_imf,
        "runtime_seconds": elapsed,
    }
