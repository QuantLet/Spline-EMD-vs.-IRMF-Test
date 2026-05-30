#!/usr/bin/python
# coding: UTF-8
"""Monte-Carlo robustness experiment for strict IRMF vs EMD."""
import numpy as np
from signal_bank.synthetic_signals import get_signal
from noise_bank.noise_models import get_noise
from core_algorithms.strict_spokoiny_irmf import strict_spokoiny_irmf
from core_algorithms.emd_core import run_emd_decomposition
from diagnostics.shared_physical_diagnostics import evaluate_shared_physical_diagnostics
from diagnostics.emd_specific_diagnostics import compute_emd_specific_diagnostics
from experiments.experiment_utils import ensure_dir, write_json
from visualization.plot_monte_carlo_summary import plot_metric_bar_with_error
from experiments.experiment_reporting import print_monte_carlo_comparison


def _run_irmf_once(Y, X_clean, t, fs):
    imfs, residual, residual_history, scale_history = strict_spokoiny_irmf(
        Y=Y, T=t, h1=0.14, a=2.0, h_min=0.008, H=0.5,
        boundary_mode='periodic', min_support_points=3, verbose=False
    )
    return evaluate_shared_physical_diagnostics(
        Y_observed=Y, X_clean=X_clean, imfs=imfs, residual=residual,
        fs=fs, residual_penalty_mode='whiteness'
    )


def _run_emd_once(Y, X_clean, t, fs):
    imfs, residual, all_components, config = run_emd_decomposition(
        Y=Y, T=t, nbsym=4, spline_kind='cubic', max_imf=5
    )
    physical = evaluate_shared_physical_diagnostics(
        Y_observed=Y, X_clean=X_clean, imfs=imfs, residual=residual,
        fs=fs, residual_penalty_mode='none'
    )
    physical.update(compute_emd_specific_diagnostics(residual, fs))
    return physical


def _aggregate(records, metrics):
    out = {}
    for metric in metrics:
        vals = np.array([r.get(metric, np.nan) for r in records], dtype=float)
        out[metric] = {
            'mean': float(np.nanmean(vals)),
            'std': float(np.nanstd(vals)),
            'min': float(np.nanmin(vals)),
            'max': float(np.nanmax(vals)),
        }
    return out


def run_monte_carlo_robustness_experiment(output_root, signal_name='stationary_multi_sine', noise_name='student_t', sigma=0.10, n=500, fs=500.0, n_trials=20, seed0=0):
    output_root = ensure_dir(output_root)
    case_root = output_root / 'monte_carlo_robustness' / signal_name / noise_name / f'sigma_{sigma:.2f}'
    case_root.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, 1, n, endpoint=False)
    X_clean = get_signal(signal_name, t)
    irmf_records, emd_records = [], []
    for trial in range(n_trials):
        seed = seed0 + trial
        noise = get_noise(noise_name, n=n, sigma=sigma, seed=seed)
        Y = X_clean + noise
        print(f'Monte-Carlo trial {trial + 1}/{n_trials} | noise={noise_name} | sigma={sigma}')
        irmf_records.append(_run_irmf_once(Y, X_clean, t, fs))
        emd_records.append(_run_emd_once(Y, X_clean, t, fs))
    metrics = ['general_physical_score','denoise_psnr','denoise_corr','spectral_leakage','strict_io','frequency_overlap_max_offdiag','residual_whiteness']
    summary = {
        'IRMF': _aggregate(irmf_records, metrics),
        'EMD': _aggregate(emd_records, metrics),
        'n_trials': n_trials,
        'noise_name': noise_name,
        'sigma': sigma,
    }
    write_json(summary, case_root / 'monte_carlo_summary.json')
    for metric in metrics:
        plot_metric_bar_with_error({'IRMF': summary['IRMF'], 'EMD': summary['EMD']}, metric, case_root / 'plots', name=f'mc_{noise_name}')
    print('\nMonte-Carlo robustness summary')
    print(summary)
    return summary


def run_monte_carlo_robustness_grid_experiment(
        output_root,
        signal_names=("stationary_multi_sine", "chirp", "am_fm", "frequency_jump", "impulsive_transient"),
        noise_names=("gaussian", "student_t", "laplace", "impulsive", "burst"),
        sigma_levels=(0.05, 0.20),
        n=500,
        fs=500.0,
        n_trials=20,
        seed0=0
):
    """
    Grid wrapper for Monte-Carlo robustness.

    Default:
        4 signals × 5 noises × 2 sigma levels.
    """
    summary = {}

    for signal_name in signal_names:
        summary[signal_name] = {}

        for noise_name in noise_names:
            summary[signal_name][noise_name] = {}

            for sigma in sigma_levels:
                print("\n" + "#" * 120)
                print(
                    f"Monte-Carlo grid | signal={signal_name} | "
                    f"noise={noise_name} | sigma={sigma}"
                )
                print("#" * 120)

                summary[signal_name][noise_name][str(sigma)] = run_monte_carlo_robustness_experiment(
                    output_root=output_root,
                    signal_name=signal_name,
                    noise_name=noise_name,
                    sigma=sigma,
                    n=n,
                    fs=fs,
                    n_trials=n_trials,
                    seed0=seed0
                )

    return summary

