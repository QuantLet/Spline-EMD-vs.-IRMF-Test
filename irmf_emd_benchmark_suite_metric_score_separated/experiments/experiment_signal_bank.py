#!/usr/bin/python
# coding: UTF-8

"""
Signal bank generalization experiment.

Purpose:
- run IRMF and EMD on all synthetic signal families
- compare generalization across signal morphology
"""

import numpy as np

from signal_bank.synthetic_signals import list_synthetic_signals, get_signal
from experiments.experiment_reporting import print_compact_comparison, print_extended_comparison
from experiments.experiment_utils import (
    ensure_dir,
    make_observed_signal,
    run_single_irmf_case,
    run_single_emd_case,
    serialize_result_summary,
    write_json,
)


def run_signal_bank_experiment(
        output_root,
        signal_names=("stationary_multi_sine", "chirp", "am_fm", "frequency_jump", "impulsive_transient"),
        sigma=0.05,
        n=500,
        fs=500.0,
        search_mode="quick",
        seed=0
):
    output_root = ensure_dir(output_root)

    if signal_names is None:
        signal_names = list_synthetic_signals()

    t = np.linspace(0, 1, n, endpoint=False)
    bank_root = output_root / "signal_bank"
    bank_root.mkdir(parents=True, exist_ok=True)

    summary = {}

    for signal_name in signal_names:
        print("\n" + "#" * 120)
        print(f"Signal bank experiment | signal={signal_name} | sigma={sigma}")
        print("#" * 120)

        X_clean = get_signal(signal_name, t)
        Y, noise, expected_noise_ratio = make_observed_signal(X_clean, sigma, seed=seed)

        case_dir = bank_root / signal_name / f"sigma_{sigma:.2f}"

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

        summary[signal_name] = {
            "IRMF": serialize_result_summary(irmf_best["physical_best"]),
            "EMD": serialize_result_summary(emd_best["physical_best"]),
        }

        print_compact_comparison(
            f"SIGNAL BANK COMPACT COMPARISON | {signal_name}",
            [
                ("IRMF", irmf_best["physical_best"]),
                ("EMD", emd_best["physical_best"]),
            ]
        )

        print_extended_comparison(
            f"SIGNAL BANK EXTENDED COMPARISON | {signal_name}",
            [
                ("IRMF", irmf_best["physical_best"]),
                ("EMD", emd_best["physical_best"]),
            ]
        )

    write_json(summary, bank_root / "signal_bank_summary.json")

    return summary


def run_signal_bank_multisigma_experiment(
        output_root,
        signal_names=("stationary_multi_sine", "chirp", "am_fm", "frequency_jump", "impulsive_transient"),
        sigma_levels=(0.05, 0.20),
        n=500,
        fs=500.0,
        search_mode="quick",
        seed=0
):
    """
    Multi-sigma wrapper for signal-bank benchmark.
    """
    summary = {}

    for sigma in sigma_levels:
        print("\n" + "#" * 120)
        print(f"Signal bank multi-sigma benchmark | sigma={sigma}")
        print("#" * 120)

        summary[str(sigma)] = run_signal_bank_experiment(
            output_root=output_root,
            signal_names=signal_names,
            sigma=sigma,
            n=n,
            fs=fs,
            search_mode=search_mode,
            seed=seed
        )

    return summary
