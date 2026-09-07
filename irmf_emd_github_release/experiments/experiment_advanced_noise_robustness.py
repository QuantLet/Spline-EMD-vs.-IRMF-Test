#!/usr/bin/python
# coding: UTF-8

"""
Advanced noise robustness experiment.

Tests IRMF vs EMD under multiple noise distributions:
Gaussian, Laplace, Student-t, impulsive, burst, colored AR(1), and pink-like noise.
"""

import numpy as np

from signal_bank.synthetic_signals import get_signal
from noise_bank.noise_models import list_noise_models, get_noise
from experiments.experiment_reporting import print_compact_comparison, print_extended_comparison
from experiments.experiment_utils import (
    ensure_dir,
    run_single_irmf_case,
    run_single_emd_case,
    serialize_result_summary,
    write_json,
)


def run_advanced_noise_robustness_experiment(
        output_root,
        signal_name="stationary_multi_sine",
        noise_names=None,
        sigma=0.10,
        n=500,
        fs=500.0,
        search_mode="quick",
        seed=0
):
    output_root = ensure_dir(output_root)

    if noise_names is None:
        noise_names = list_noise_models()

    t = np.linspace(0, 1, n, endpoint=False)
    X_clean = get_signal(signal_name, t)

    case_root = output_root / "advanced_noise_robustness" / signal_name / f"sigma_{sigma:.2f}"
    case_root.mkdir(parents=True, exist_ok=True)

    summary = {}

    for noise_name in noise_names:
        print("\n" + "#" * 120)
        print(f"Advanced noise robustness | signal={signal_name} | noise={noise_name} | sigma={sigma}")
        print("#" * 120)

        noise = get_noise(noise_name, n=n, sigma=sigma, seed=seed)
        Y = X_clean + noise
        expected_noise_ratio = float(np.sum(noise ** 2) / (np.sum(Y ** 2) + 1e-10))

        irmf_results, irmf_best = run_single_irmf_case(
            Y=Y,
            X_clean=X_clean,
            t=t,
            fs=fs,
            output_dir=case_root / noise_name / "irmf",
            search_mode=search_mode,
            expected_noise_ratio=expected_noise_ratio,
            boundary_mode="periodic"
        )

        emd_results, emd_best = run_single_emd_case(
            Y=Y,
            X_clean=X_clean,
            t=t,
            fs=fs,
            output_dir=case_root / noise_name / "emd",
            search_mode=search_mode
        )

        summary[noise_name] = {
            "IRMF": serialize_result_summary(irmf_best["physical_best"]),
            "EMD": serialize_result_summary(emd_best["physical_best"]),
            "expected_noise_ratio": expected_noise_ratio,
        }

        print_compact_comparison(
            f"ADVANCED NOISE COMPACT COMPARISON | noise={noise_name}",
            [
                ("IRMF", irmf_best["physical_best"]),
                ("EMD", emd_best["physical_best"]),
            ]
        )

        print_extended_comparison(
            f"ADVANCED NOISE EXTENDED COMPARISON | noise={noise_name}",
            [
                ("IRMF", irmf_best["physical_best"]),
                ("EMD", emd_best["physical_best"]),
            ]
        )

    write_json(summary, case_root / "advanced_noise_robustness_summary.json")

    return summary
