#!/usr/bin/python
# coding: UTF-8
"""Strict IRMF operator propagation experiment."""
import numpy as np
from signal_bank.synthetic_signals import get_signal
from experiments.experiment_utils import ensure_dir, make_observed_signal, run_single_irmf_case, serialize_result_summary, write_json


def run_operator_propagation_experiment(output_root, signal_name='stationary_multi_sine', sigma=0.05, n=500, fs=500.0, search_mode='quick', seed=0):
    output_root = ensure_dir(output_root)
    t = np.linspace(0, 1, n, endpoint=False)
    X_clean = get_signal(signal_name, t)
    Y, noise, expected_noise_ratio = make_observed_signal(X_clean, sigma, seed=seed)
    case_root = output_root / 'operator_propagation' / signal_name / f'sigma_{sigma:.2f}'
    case_root.mkdir(parents=True, exist_ok=True)
    results, best = run_single_irmf_case(
        Y=Y, X_clean=X_clean, t=t, fs=fs, output_dir=case_root,
        search_mode=search_mode, expected_noise_ratio=expected_noise_ratio,
        boundary_mode='periodic'
    )
    summary = {
        'theory_best': serialize_result_summary(best['theory_best']),
        'physical_best': serialize_result_summary(best['physical_best']),
        'theory_best_operator_evolution': best['theory_best'].get('operator_evolution', []),
        'physical_best_operator_evolution': best['physical_best'].get('operator_evolution', []),
    }
    write_json(summary, case_root / 'operator_propagation_summary.json')
    return summary


def run_operator_propagation_multisigma_experiment(
        output_root,
        signal_name="stationary_multi_sine",
        sigma_levels=(0.05, 0.20),
        n=500,
        fs=500.0,
        search_mode="quick",
        seed=0
):
    """
    Multi-sigma wrapper for strict IRMF operator propagation.
    """
    summary = {}

    for sigma in sigma_levels:
        print("\n" + "#" * 120)
        print(f"Operator propagation multi-sigma benchmark | signal={signal_name} | sigma={sigma}")
        print("#" * 120)

        summary[str(sigma)] = run_operator_propagation_experiment(
            output_root=output_root,
            signal_name=signal_name,
            sigma=sigma,
            n=n,
            fs=fs,
            search_mode=search_mode,
            seed=seed
        )

    return summary


def run_operator_propagation_signalbank_experiment(
        output_root,
        signal_names=("stationary_multi_sine", "chirp", "am_fm", "frequency_jump", "impulsive_transient"),
        sigma_levels=(0.05, 0.20),
        n=500,
        fs=500.0,
        search_mode="quick",
        seed=0
):
    """
    Signal-bank wrapper for strict IRMF operator propagation.

    Default:
        5 signals × 2 sigma levels = 8 operator propagation cases

    This keeps operator propagation as a strict IRMF theory diagnostic,
    but makes it comparable across different signal structures.
    """
    summary = {}

    for signal_name in signal_names:
        summary[signal_name] = {}

        for sigma in sigma_levels:
            print("\n" + "#" * 120)
            print(
                f"Operator propagation signal-bank benchmark | "
                f"signal={signal_name} | sigma={sigma}"
            )
            print("#" * 120)

            summary[signal_name][str(sigma)] = run_operator_propagation_experiment(
                output_root=output_root,
                signal_name=signal_name,
                sigma=sigma,
                n=n,
                fs=fs,
                search_mode=search_mode,
                seed=seed
            )

    return summary
