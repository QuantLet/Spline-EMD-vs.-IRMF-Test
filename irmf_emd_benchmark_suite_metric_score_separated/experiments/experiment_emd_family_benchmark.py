#!/usr/bin/python
# coding: UTF-8

"""
Optional EMD-family benchmark.

Compares:
- IRMF
- EMD
- EEMD if available
- CEEMDAN if available
"""

import numpy as np

from signal_bank.synthetic_signals import get_signal
from experiments.experiment_utils import (
    ensure_dir,
    make_observed_signal,
    run_single_irmf_case,
    run_single_emd_case,
    serialize_result_summary,
    write_json,
    save_best_plots,
)
from sensitivity_analysis.emd_family_sensitivity_analysis import try_evaluate_emd_family_run
from experiments.experiment_reporting import print_emd_full_report, print_compact_comparison, print_extended_comparison
from diagnostics.frequency_spacing_diagnostics import compute_frequency_spacing_diagnostics
from diagnostics.snr_gain_diagnostics import compute_snr_gain_diagnostics

def _ensure_v14_metrics(result, Y, X_clean):
    if result is None:
        return result

    if "frequency_spacing_penalty" not in result:
        spacing = compute_frequency_spacing_diagnostics(result.get("center_freqs", []))
        result.update(spacing)

    if "snr_gain_db" not in result:
        try:
            recovered = Y - result["residual"]
        except Exception:
            try:
                recovered = np.sum(result["imfs"], axis=0)
            except Exception:
                recovered = None
        result.update(compute_snr_gain_diagnostics(Y, X_clean, recovered))

    return result




def run_emd_family_benchmark(
        output_root,
        signal_name="stationary_multi_sine",
        sigma=0.05,
        n=500,
        fs=500.0,
        search_mode="quick",
        seed=0,
        optional_methods=("EEMD", "CEEMDAN")
):
    output_root = ensure_dir(output_root)
    t = np.linspace(0, 1, n, endpoint=False)
    X_clean = get_signal(signal_name, t)
    Y, noise, expected_noise_ratio = make_observed_signal(X_clean, sigma, seed=seed)

    case_root = output_root / "emd_family_benchmark" / signal_name / f"sigma_{sigma:.2f}"
    case_root.mkdir(parents=True, exist_ok=True)

    irmf_results, irmf_best = run_single_irmf_case(
        Y=Y,
        X_clean=X_clean,
        t=t,
        fs=fs,
        output_dir=case_root / "irmf",
        search_mode=search_mode,
        expected_noise_ratio=expected_noise_ratio,
        boundary_mode="periodic"
    )

    emd_results, emd_best = run_single_emd_case(
        Y=Y,
        X_clean=X_clean,
        t=t,
        fs=fs,
        output_dir=case_root / "emd",
        search_mode=search_mode
    )

    rows = [
        ("IRMF", irmf_best["physical_best"]),
        ("EMD", emd_best["physical_best"]),
    ]

    summary = {
        "IRMF": serialize_result_summary(irmf_best["physical_best"]),
        "EMD": serialize_result_summary(emd_best["physical_best"]),
    }

    run_id = 1
    for method in optional_methods:
        result = try_evaluate_emd_family_run(
            run_id=run_id,
            method=method,
            Y=Y,
            T=t,
            X_clean=X_clean,
            fs=fs,
            max_imf=-1,
            trials=20,
            noise_width=0.05,
            random_seed=seed
        )
        run_id += 1

        if result is None:
            summary[method] = {"skipped": True}
            continue

        print_emd_full_report(f"{method} FULL DIAGNOSTIC REPORT", result)
        save_best_plots(
            {method.lower(): result},
            t=t,
            fs=fs,
            output_dir=case_root / method.lower(),
            prefix=method.lower()
        )

        rows.append((method, result))
        summary[method] = serialize_result_summary(result)

    print_compact_comparison("IRMF vs EMD-family compact comparison", rows)

    print_extended_comparison("IRMF vs EMD-family extended comparison", rows)

    write_json(summary, case_root / "emd_family_benchmark_summary.json")

    return summary


def run_emd_family_multisigma_benchmark(
        output_root,
        signal_name="stationary_multi_sine",
        sigma_levels=(0.05, 0.20),
        n=500,
        fs=500.0,
        search_mode="quick",
        seed=0,
        optional_methods=("EEMD", "CEEMDAN")
):
    """
    Multi-sigma wrapper for EMD-family benchmark.
    """
    summary = {}

    for sigma in sigma_levels:
        print("\n" + "#" * 120)
        print(f"EMD-family multi-sigma benchmark | signal={signal_name} | sigma={sigma}")
        print("#" * 120)

        summary[str(sigma)] = run_emd_family_benchmark(
            output_root=output_root,
            signal_name=signal_name,
            sigma=sigma,
            n=n,
            fs=fs,
            search_mode=search_mode,
            seed=seed,
            optional_methods=optional_methods
        )

    return summary
