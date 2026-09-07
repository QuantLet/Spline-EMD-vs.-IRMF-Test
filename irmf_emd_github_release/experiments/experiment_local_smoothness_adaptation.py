#!/usr/bin/python
# coding: UTF-8
"""Local smoothness adaptation benchmark."""
import numpy as np
from signal_bank.synthetic_signals import get_signal
from experiments.experiment_utils import ensure_dir, make_observed_signal, run_single_irmf_case, run_single_emd_case, serialize_result_summary, write_json
from experiments.experiment_reporting import print_compact_comparison, print_extended_comparison


def run_local_smoothness_adaptation_experiment(output_root, signal_names=('piecewise_holder','heteroscedastic_smoothness'), sigma=0.05, n=500, fs=500.0, search_mode='quick', seed=0):
    output_root = ensure_dir(output_root)
    t = np.linspace(0, 1, n, endpoint=False)
    case_root = output_root / 'local_smoothness_adaptation'
    case_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for signal_name in signal_names:
        print('\n' + '#' * 120)
        print(f'Local smoothness adaptation | signal={signal_name}')
        print('#' * 120)
        X_clean = get_signal(signal_name, t)
        Y, noise, expected_noise_ratio = make_observed_signal(X_clean, sigma, seed=seed)
        signal_dir = case_root / signal_name / f'sigma_{sigma:.2f}'
        irmf_results, irmf_best = run_single_irmf_case(
            Y=Y, X_clean=X_clean, t=t, fs=fs, output_dir=signal_dir / 'irmf',
            search_mode=search_mode, expected_noise_ratio=expected_noise_ratio,
            boundary_mode='periodic'
        )
        emd_results, emd_best = run_single_emd_case(
            Y=Y, X_clean=X_clean, t=t, fs=fs, output_dir=signal_dir / 'emd',
            search_mode=search_mode
        )
        irmf_physical = irmf_best['physical_best']
        emd_physical = emd_best['physical_best']
        print_compact_comparison(f'LOCAL SMOOTHNESS COMPARISON | {signal_name}', [('IRMF', irmf_physical), ('EMD', emd_physical)])
        summary[signal_name] = {'IRMF': serialize_result_summary(irmf_physical), 'EMD': serialize_result_summary(emd_physical)}
    write_json(summary, case_root / 'local_smoothness_adaptation_summary.json')
    return summary


def run_local_smoothness_adaptation_multisigma_experiment(
        output_root,
        signal_names=("piecewise_holder", "heteroscedastic_smoothness"),
        sigma_levels=(0.05, 0.20),
        n=500,
        fs=500.0,
        search_mode="quick",
        seed=0
):
    """
    Multi-sigma wrapper for local smoothness adaptation benchmark.
    """
    summary = {}

    for sigma in sigma_levels:
        print("\n" + "#" * 120)
        print(f"Local smoothness adaptation multi-sigma benchmark | sigma={sigma}")
        print("#" * 120)

        summary[str(sigma)] = run_local_smoothness_adaptation_experiment(
            output_root=output_root,
            signal_names=signal_names,
            sigma=sigma,
            n=n,
            fs=fs,
            search_mode=search_mode,
            seed=seed
        )

    return summary
