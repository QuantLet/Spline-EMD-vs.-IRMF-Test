#!/usr/bin/python
# coding: UTF-8

"""
A priori vs posteriori experiment.

Purpose:
- compute IRMF Fisher/Wilks-inspired theoretical risk curve
- extract theoretical h*
- compare h* with empirical best h1 from parameter search

This experiment is IRMF-specific.
"""

import numpy as np

from signal_bank.synthetic_signals import get_signal
from irmf_prior_theory.apriori_bounds import compute_apriori_risk_bound
from experiments.experiment_utils import (
    ensure_dir,
    make_observed_signal,
    run_single_irmf_case,
    serialize_result_summary,
    write_json,
)
from visualization.plot_irmf_risk_bounds import plot_apriori_risk_bound


def run_apriori_vs_posteriori_experiment(
        output_root,
        signal_name="stationary_multi_sine",
        sigma=0.05,
        n=500,
        fs=500.0,
        search_mode="quick",
        seed=0,
        c3=0.2,
        bias_amplitude=1.5
):
    output_root = ensure_dir(output_root)

    t = np.linspace(0, 1, n, endpoint=False)
    X_clean = get_signal(signal_name, t)
    Y, noise, expected_noise_ratio = make_observed_signal(X_clean, sigma, seed=seed)

    case_root = output_root / "apriori_vs_posteriori" / signal_name / f"sigma_{sigma:.2f}"
    case_root.mkdir(parents=True, exist_ok=True)

    h_grid = np.linspace(0.02, 0.40, 200)

    bounds = compute_apriori_risk_bound(
        h_grid=h_grid,
        n=n,
        sigma=sigma,
        c3=c3,
        bias_amplitude=bias_amplitude
    )

    opt_idx = int(np.argmin(bounds["physical_error_bound"]))
    theoretical_h_star = float(bounds["h"][opt_idx])
    theoretical_min_bound = float(bounds["physical_error_bound"][opt_idx])

    plot_apriori_risk_bound(bounds, case_root, name="apriori_risk_bound")

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

    posterior_physical = irmf_best["physical_best"]
    posterior_theory = irmf_best["theory_best"]

    summary = {
        "signal_name": signal_name,
        "sigma": sigma,
        "theoretical_h_star": theoretical_h_star,
        "theoretical_min_bound": theoretical_min_bound,
        "posterior_physical_best_h1": posterior_physical["h1"],
        "posterior_theory_best_h1": posterior_theory["h1"],
        "absolute_gap_physical": abs(posterior_physical["h1"] - theoretical_h_star),
        "absolute_gap_theory": abs(posterior_theory["h1"] - theoretical_h_star),
        "posterior_physical_best": serialize_result_summary(posterior_physical),
        "posterior_theory_best": serialize_result_summary(posterior_theory),
    }

    write_json(summary, case_root / "apriori_vs_posteriori_summary.json")

    print("\n" + "=" * 120)
    print("A priori vs posteriori summary")
    print("=" * 120)
    print(f"Theoretical h*              : {theoretical_h_star:.6f}")
    print(f"Posterior physical best h1  : {posterior_physical['h1']:.6f}")
    print(f"Posterior theory best h1    : {posterior_theory['h1']:.6f}")
    print(f"Gap physical                : {summary['absolute_gap_physical']:.6f}")
    print(f"Gap theory                  : {summary['absolute_gap_theory']:.6f}")

    return summary
