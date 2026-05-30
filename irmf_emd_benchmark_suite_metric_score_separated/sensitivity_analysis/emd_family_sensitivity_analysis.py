#!/usr/bin/python
# coding: UTF-8

"""
Optional EMD-family sensitivity analysis.

Supports:
- EMD
- EEMD if PyEMD provides it
- CEEMDAN if PyEMD provides it
"""

from core_algorithms.emd_core import run_pyemd_family_decomposition
from diagnostics.shared_physical_diagnostics import evaluate_shared_physical_diagnostics
from diagnostics.emd_specific_diagnostics import compute_emd_specific_diagnostics


def evaluate_emd_family_run(
        run_id,
        method,
        Y,
        T,
        X_clean,
        fs,
        max_imf=-1,
        trials=50,
        noise_width=0.05,
        random_seed=0
):
    imfs, residual, all_components, config = run_pyemd_family_decomposition(
        Y=Y,
        T=T,
        method=method,
        max_imf=max_imf,
        trials=trials,
        noise_width=noise_width,
        random_seed=random_seed
    )

    physical = evaluate_shared_physical_diagnostics(
        Y_observed=Y,
        X_clean=X_clean,
        imfs=imfs,
        residual=residual,
        fs=fs,
        residual_penalty_mode="none"
    )

    emd_specific = compute_emd_specific_diagnostics(residual, fs)

    out = {
        "method": method.upper(),
        "run_id": run_id,
        "max_imf": max_imf,
        "trials": trials,
        "noise_width": noise_width,
        "random_seed": random_seed,
        "config": config,
        "imfs": imfs,
        "residual": residual,
        "all_components": all_components,
        "theory_diagnostic_score": None,
    }

    out.update(physical)
    out.update(emd_specific)

    return out


def try_evaluate_emd_family_run(*args, **kwargs):
    try:
        return evaluate_emd_family_run(*args, **kwargs)
    except ImportError as exc:
        print(f"Skipping optional EMD-family method: {exc}")
        return None
