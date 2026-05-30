#!/usr/bin/python
# coding: UTF-8

"""
V11 full signal × noise × sigma robustness benchmark.

Default design:
    5 signals × 5 noise models × 2 sigma levels = 50 cases

Signals:
    stationary_multi_sine
    chirp
    frequency_jump
    impulsive_transient

Noise models:
    gaussian
    student_t
    laplace
    impulsive
    burst

Sigma:
    0.05
    0.20

This is intentionally strict-IRMF compatible:
- it does not alter strict_spokoiny_irmf.py;
- it uses the existing IRMF parameter search and EMD sensitivity search;
- it produces compact/extended comparison tables and plots.
"""

import numpy as np

from signal_bank.synthetic_signals import get_signal
from noise_bank.noise_models import get_noise
from experiments.experiment_utils import (
    ensure_dir,
    run_single_irmf_case,
    run_single_emd_case,
    serialize_result_summary,
    write_json,
)
from experiments.experiment_reporting import (
    print_compact_comparison,
    print_extended_comparison,
)


def run_full_signal_noise_robustness_experiment(
        output_root,
        signal_names=("stationary_multi_sine", "chirp", "am_fm", "frequency_jump", "impulsive_transient"),
        noise_names=("gaussian", "student_t", "laplace", "impulsive", "burst"),
        sigma_levels=(0.05, 0.20),
        n=500,
        fs=500.0,
        search_mode="quick",
        seed=0
):
    output_root = ensure_dir(output_root)
    case_root = output_root / "full_signal_noise_robustness_40_cases"
    case_root.mkdir(parents=True, exist_ok=True)

    t = np.linspace(0, 1, n, endpoint=False)

    summary = {}

    for signal_name in signal_names:
        summary[signal_name] = {}
        X_clean = get_signal(signal_name, t)

        for noise_name in noise_names:
            summary[signal_name][noise_name] = {}

            for sigma in sigma_levels:
                print("\n" + "#" * 120)
                print(
                    f"FULL ROBUSTNESS CASE | signal={signal_name} | "
                    f"noise={noise_name} | sigma={sigma}"
                )
                print("#" * 120)

                noise = get_noise(noise_name, n=n, sigma=sigma, seed=seed)
                Y = X_clean + noise

                expected_noise_ratio = float(np.sum(noise ** 2) / (np.sum(Y ** 2) + 1e-10))

                output_dir = case_root / signal_name / noise_name / f"sigma_{sigma:.2f}"

                irmf_results, irmf_best = run_single_irmf_case(
                    Y=Y,
                    X_clean=X_clean,
                    t=t,
                    fs=fs,
                    output_dir=output_dir / "irmf",
                    search_mode=search_mode,
                    expected_noise_ratio=expected_noise_ratio,
                    boundary_mode="periodic",
                )

                emd_results, emd_best = run_single_emd_case(
                    Y=Y,
                    X_clean=X_clean,
                    t=t,
                    fs=fs,
                    output_dir=output_dir / "emd",
                    search_mode=search_mode
                )

                irmf_physical = irmf_best["physical_best"]
                emd_physical = emd_best["physical_best"]

                print_compact_comparison(
                    f"FULL ROBUSTNESS COMPACT | {signal_name} | {noise_name} | sigma={sigma:.2f}",
                    [
                        ("IRMF", irmf_physical),
                        ("EMD", emd_physical),
                    ]
                )

                print_extended_comparison(
                    f"FULL ROBUSTNESS EXTENDED | {signal_name} | {noise_name} | sigma={sigma:.2f}",
                    [
                        ("IRMF", irmf_physical),
                        ("EMD", emd_physical),
                    ]
                )

                summary[signal_name][noise_name][str(sigma)] = {
                    "IRMF": serialize_result_summary(irmf_physical),
                    "EMD": serialize_result_summary(emd_physical),
                    "expected_noise_ratio": expected_noise_ratio,
                }

    write_json(summary, case_root / "full_signal_noise_robustness_summary.json")

    return summary
