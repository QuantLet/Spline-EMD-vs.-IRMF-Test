#!/usr/bin/python
# coding: UTF-8

"""
Noise robustness experiment.

Purpose:
- sweep multiple Gaussian noise levels
- compare degradation curves for IRMF and EMD
"""

from pathlib import Path
import numpy as np

from signal_bank.synthetic_signals import get_signal
from experiments.experiment_reporting import print_compact_comparison, print_extended_comparison
from experiments.experiment_utils import (
    ensure_dir,
    make_observed_signal,
    run_single_irmf_case,
    run_single_emd_case,
    serialize_result_summary,
    write_json,
)
from robustness.robustness_analysis import summarize_robustness_by_signal
from visualization.plot_robustness import plot_metric_vs_noise


def run_noise_robustness_experiment(
        output_root,
        signal_name="stationary_multi_sine",
        noise_levels=(0.01, 0.03, 0.05, 0.10, 0.20, 0.30),
        n=500,
        fs=500.0,
        search_mode="quick",
        seed=0
):
    output_root = ensure_dir(output_root)
    t = np.linspace(0, 1, n, endpoint=False)
    X_clean = get_signal(signal_name, t)

    case_root = output_root / "noise_robustness" / signal_name
    case_root.mkdir(parents=True, exist_ok=True)

    irmf_best_by_noise = {}
    emd_best_by_noise = {}

    summary = {}

    for sigma in noise_levels:
        print("\n" + "#" * 120)
        print(f"Noise robustness | signal={signal_name} | sigma={sigma}")
        print("#" * 120)

        Y, noise, expected_noise_ratio = make_observed_signal(X_clean, sigma, seed=seed)

        irmf_results, irmf_best = run_single_irmf_case(
            Y=Y,
            X_clean=X_clean,
            t=t,
            fs=fs,
            output_dir=case_root / f"sigma_{sigma:.2f}" / "irmf",
            search_mode=search_mode,
            expected_noise_ratio=expected_noise_ratio,
            boundary_mode="periodic"
        )

        emd_results, emd_best = run_single_emd_case(
            Y=Y,
            X_clean=X_clean,
            t=t,
            fs=fs,
            output_dir=case_root / f"sigma_{sigma:.2f}" / "emd",
            search_mode=search_mode
        )

        irmf_best_by_noise[sigma] = irmf_best["physical_best"]
        emd_best_by_noise[sigma] = emd_best["physical_best"]

        summary[str(sigma)] = {
            "IRMF": serialize_result_summary(irmf_best_by_noise[sigma]),
            "EMD": serialize_result_summary(emd_best_by_noise[sigma]),
        }

        print_compact_comparison(
            f"NOISE ROBUSTNESS COMPACT COMPARISON | sigma={sigma:.2f}",
            [
                ("IRMF", irmf_best_by_noise[sigma]),
                ("EMD", emd_best_by_noise[sigma]),
            ]
        )

        print_extended_comparison(
            f"NOISE ROBUSTNESS EXTENDED COMPARISON | sigma={sigma:.2f}",
            [
                ("IRMF", irmf_best_by_noise[sigma]),
                ("EMD", emd_best_by_noise[sigma]),
            ]
        )

    irmf_robust = summarize_robustness_by_signal(irmf_best_by_noise)
    emd_robust = summarize_robustness_by_signal(emd_best_by_noise)

    summary["robustness_summary"] = {
        "IRMF": irmf_robust,
        "EMD": emd_robust,
    }

    for metric in ["general_physical_score", "spectral_leakage", "strict_io", "ifs", "denoise_psnr"]:
        plot_metric_vs_noise(irmf_best_by_noise, metric, case_root / "plots", f"irmf_{metric}")
        plot_metric_vs_noise(emd_best_by_noise, metric, case_root / "plots", f"emd_{metric}")

    write_json(summary, case_root / "noise_robustness_summary.json")

    print("\nNoise robustness summary:")
    print(summary["robustness_summary"])

    return summary
