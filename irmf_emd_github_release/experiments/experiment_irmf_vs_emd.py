#!/usr/bin/python
# coding: UTF-8

"""
Primary IRMF vs EMD comparison experiment.

Purpose:
- same signal
- same noise realization
- same observed Y
- compare best IRMF and best EMD results using shared diagnostics
"""

from pathlib import Path
import numpy as np

from signal_bank.synthetic_signals import get_signal
from experiments.experiment_reporting import print_compact_comparison
from experiments.experiment_utils import (
    ensure_dir,
    make_observed_signal,
    run_single_irmf_case,
    run_single_emd_case,
    serialize_result_summary,
    write_json,
)


def run_irmf_vs_emd_experiment(
        output_root,
        signal_name="stationary_multi_sine",
        sigma=0.05,
        n=500,
        fs=500.0,
        search_mode="quick",
        seed=0
):
    output_root = ensure_dir(output_root)
    t = np.linspace(0, 1, n, endpoint=False)

    X_clean = get_signal(signal_name, t)
    Y, noise, expected_noise_ratio = make_observed_signal(X_clean, sigma, seed=seed)

    case_dir = output_root / "irmf_vs_emd" / signal_name / f"sigma_{sigma:.2f}"
    case_dir.mkdir(parents=True, exist_ok=True)

    irmf_results, irmf_best = run_single_irmf_case(
        Y=Y,
        X_clean=X_clean,
        t=t,
        fs=fs,
        output_dir=case_dir / "irmf",
        search_mode=search_mode,
        expected_noise_ratio=expected_noise_ratio,
        boundary_mode="periodic"
    )

    emd_results, emd_best = run_single_emd_case(
        Y=Y,
        X_clean=X_clean,
        t=t,
        fs=fs,
        output_dir=case_dir / "emd",
        search_mode=search_mode
    )

    irmf_physical = irmf_best["physical_best"]
    emd_physical = emd_best["physical_best"]

    comparison = {
        "signal_name": signal_name,
        "sigma": sigma,
        "expected_noise_ratio": expected_noise_ratio,
        "irmf_physical_best": serialize_result_summary(irmf_physical),
        "emd_physical_best": serialize_result_summary(emd_physical),
        "shared_comparison": {
            "general_physical_score": {
                "IRMF": irmf_physical["general_physical_score"],
                "EMD": emd_physical["general_physical_score"],
            },
            "strict_io": {
                "IRMF": irmf_physical["strict_io"],
                "EMD": emd_physical["strict_io"],
            },
            "spectral_leakage": {
                "IRMF": irmf_physical["spectral_leakage"],
                "EMD": emd_physical["spectral_leakage"],
            },
            "ifs": {
                "IRMF": irmf_physical["ifs"],
                "EMD": emd_physical["ifs"],
            },
            "denoise_psnr": {
                "IRMF": irmf_physical["denoise_psnr"],
                "EMD": emd_physical["denoise_psnr"],
            },
            "denoise_corr": {
                "IRMF": irmf_physical["denoise_corr"],
                "EMD": emd_physical["denoise_corr"],
            },
            "energy_ratio": {
                "IRMF": irmf_physical["energy_ratio"],
                "EMD": emd_physical["energy_ratio"],
            },
        }
    }

    write_json(comparison, case_dir / "irmf_vs_emd_comparison.json")

    print("\n" + "=" * 120)
    print(f"IRMF vs EMD comparison | signal={signal_name} | sigma={sigma}")
    print("=" * 120)
    for metric, vals in comparison["shared_comparison"].items():
        print(f"{metric}: {vals}")

    print_compact_comparison(
        "IRMF vs EMD COMPACT DIAGNOSTIC COMPARISON",
        [("IRMF", irmf_physical), ("EMD", emd_physical)]
    )

    return comparison


def run_irmf_vs_emd_multisigma_experiment(
        output_root,
        signal_name="stationary_multi_sine",
        sigma_levels=(0.05, 0.20),
        n=500,
        fs=500.0,
        search_mode="quick",
        seed=0
):
    """
    Multi-sigma wrapper for IRMF vs EMD comparison.
    """
    summary = {}

    for sigma in sigma_levels:
        print("\n" + "#" * 120)
        print(f"IRMF vs EMD multi-sigma comparison | signal={signal_name} | sigma={sigma}")
        print("#" * 120)

        summary[str(sigma)] = run_irmf_vs_emd_experiment(
            output_root=output_root,
            signal_name=signal_name,
            sigma=sigma,
            n=n,
            fs=fs,
            search_mode=search_mode,
            seed=seed
        )

    return summary
