#!/usr/bin/python
# coding: UTF-8

"""CEEMDAN wrapper."""

import numpy as np
from time import perf_counter
from core_algorithms.emd_wrapper import final_residual


def run_ceemdan(y, max_imf=-1, trials=50, epsilon=0.005, parallel=False, random_seed=0):
    try:
        from PyEMD import CEEMDAN
    except Exception as exc:
        raise ImportError("CEEMDAN requires PyEMD with CEEMDAN support.") from exc

    ceemdan = CEEMDAN(trials=trials, epsilon=epsilon, parallel=parallel)
    ceemdan.noise_seed(int(random_seed))
    start = perf_counter()
    imfs = ceemdan.ceemdan(np.asarray(y, dtype=float), max_imf=max_imf)
    elapsed = perf_counter() - start
    residual = final_residual(y, imfs)

    return {
        "method": "CEEMDAN",
        "imfs": imfs,
        "residual": residual,
        "trials": trials,
        "epsilon": epsilon,
        "random_seed": random_seed,
        "algorithm_seed": random_seed,
        "max_imf": max_imf,
        "runtime_seconds": elapsed,
    }
