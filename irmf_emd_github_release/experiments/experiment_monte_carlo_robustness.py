#!/usr/bin/python
# coding: UTF-8

"""
4.3 Monte Carlo Robustness.

4.3.1 Standard Monte Carlo:
    5 signals × 6 noises × 2 sigma × n_trials

4.3.2 Huber contamination breakdown:
    5 signals × lambda grid × 2 sigma × n_trials
"""

from pathlib import Path
import json
import numpy as np

from project_config import (
    MONTE_CARLO_SIGNAL_FAMILY,
    MONTE_CARLO_NOISE_FAMILY,
    MONTE_CARLO_SIGMA_LEVELS,
    DEFAULT_MONTE_CARLO_TRIALS,
    HUBER_LAMBDA_GRID,
    HUBER_CONTAMINATION_OUTLIER_SCALE,
)
from experiments.experiment_utils import make_signal_noise_case, run_single_irmf_case, run_single_emd_case


METRICS = (
    "robust_estimation_score",
    "decomposition_quality_score",
    "denoise_psnr",
    "denoise_corr",
    "snr_gain_db",
    "noise_capture_corr",
    "strict_io",
    "spectral_leakage",
    "frequency_overlap_max_offdiag",
    "residual_whiteness",
)


def _stats(values):
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def run_monte_carlo_robustness_experiment(
        output_root,
        signal_names=MONTE_CARLO_SIGNAL_FAMILY,
        noise_names=MONTE_CARLO_NOISE_FAMILY,
        sigma_levels=MONTE_CARLO_SIGMA_LEVELS,
        n_trials=DEFAULT_MONTE_CARLO_TRIALS,
        search_mode="quick",
        n=500,
        fs=500.0,
        base_seed=1000,
        run_huber_breakdown=True,
):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    standard = run_standard_monte_carlo(
        output_root=output_root / "standard_monte_carlo",
        signal_names=signal_names,
        noise_names=noise_names,
        sigma_levels=sigma_levels,
        n_trials=n_trials,
        search_mode=search_mode,
        n=n,
        fs=fs,
        base_seed=base_seed,
    )

    huber = None
    if run_huber_breakdown:
        huber = run_huber_contamination_monte_carlo_breakdown(
            output_root=output_root / "huber_contamination_breakdown",
            signal_names=signal_names,
            lambda_grid=HUBER_LAMBDA_GRID,
            sigma_levels=sigma_levels,
            n_trials=n_trials,
            search_mode=search_mode,
            n=n,
            fs=fs,
            base_seed=base_seed,
        )

    return {"standard_monte_carlo": standard, "huber_contamination_breakdown": huber}


def _run_trial_case(
        signal_name,
        noise_name,
        sigma,
        seed,
        output_dir,
        search_mode,
        n,
        fs,
        noise_kwargs=None,
):
    case = make_signal_noise_case(
        signal_name=signal_name,
        noise_name=noise_name,
        sigma=sigma,
        n=n,
        fs=fs,
        seed=seed,
        noise_kwargs=noise_kwargs,
    )

    t = case["t"]
    Y = case["Y"]
    X_clean = case["X_clean"]
    true_components = case.get("true_components", None)

    _, irmf_best = run_single_irmf_case(
        Y=Y, X_clean=X_clean, t=t, fs=fs,
        output_dir=output_dir / "irmf",
        search_mode=search_mode,
        true_components=true_components,
    )
    _, emd_best = run_single_emd_case(
        Y=Y, X_clean=X_clean, t=t, fs=fs,
        output_dir=output_dir / "emd",
        true_components=true_components,
    )

    return irmf_best, emd_best


def run_standard_monte_carlo(
        output_root,
        signal_names,
        noise_names,
        sigma_levels,
        n_trials,
        search_mode="quick",
        n=500,
        fs=500.0,
        base_seed=1000,
):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_summary = []

    for signal_name in signal_names:
        for noise_name in noise_names:
            for sigma in sigma_levels:
                print("\n" + "#" * 120)
                print(f"STANDARD MONTE CARLO | signal={signal_name} | noise={noise_name} | sigma={sigma}")
                print("#" * 120)

                irmf_values = {m: [] for m in METRICS}
                emd_values = {m: [] for m in METRICS}

                for trial in range(n_trials):
                    seed = base_seed + 100000 * trial + int(1000 * sigma)
                    trial_dir = output_root / signal_name / noise_name / f"sigma_{sigma}" / f"trial_{trial:03d}"
                    trial_dir.mkdir(parents=True, exist_ok=True)

                    irmf_best, emd_best = _run_trial_case(
                        signal_name=signal_name,
                        noise_name=noise_name,
                        sigma=sigma,
                        seed=seed,
                        output_dir=trial_dir,
                        search_mode=search_mode,
                        n=n,
                        fs=fs,
                    )

                    for m in METRICS:
                        irmf_values[m].append(irmf_best.get(m, np.nan))
                        emd_values[m].append(emd_best.get(m, np.nan))

                all_summary.append({
                    "signal": signal_name,
                    "noise": noise_name,
                    "sigma": sigma,
                    "n_trials": n_trials,
                    "IRMF": {m: _stats(irmf_values[m]) for m in METRICS},
                    "EMD": {m: _stats(emd_values[m]) for m in METRICS},
                })

    with open(output_root / "standard_monte_carlo_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(all_summary), f, indent=2)

    return all_summary


def run_huber_contamination_monte_carlo_breakdown(
        output_root,
        signal_names=MONTE_CARLO_SIGNAL_FAMILY,
        lambda_grid=HUBER_LAMBDA_GRID,
        sigma_levels=MONTE_CARLO_SIGMA_LEVELS,
        n_trials=DEFAULT_MONTE_CARLO_TRIALS,
        search_mode="quick",
        n=500,
        fs=500.0,
        base_seed=1000,
):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_summary = []

    for signal_name in signal_names:
        for sigma in sigma_levels:
            for lam in lambda_grid:
                print("\n" + "#" * 120)
                print(f"HUBER CONTAMINATION MONTE CARLO | signal={signal_name} | sigma={sigma} | lambda={lam}")
                print("#" * 120)

                irmf_values = {m: [] for m in METRICS}
                emd_values = {m: [] for m in METRICS}

                for trial in range(n_trials):
                    seed = base_seed + 100000 * trial + int(1000 * sigma) + int(10000 * lam)
                    trial_dir = output_root / signal_name / f"sigma_{sigma}" / f"lambda_{lam}" / f"trial_{trial:03d}"
                    trial_dir.mkdir(parents=True, exist_ok=True)

                    irmf_best, emd_best = _run_trial_case(
                        signal_name=signal_name,
                        noise_name="huber_contamination",
                        sigma=sigma,
                        seed=seed,
                        output_dir=trial_dir,
                        search_mode=search_mode,
                        n=n,
                        fs=fs,
                        noise_kwargs={
                            "lam": lam,
                            "outlier_scale": HUBER_CONTAMINATION_OUTLIER_SCALE,
                        },
                    )

                    for m in METRICS:
                        irmf_values[m].append(irmf_best.get(m, np.nan))
                        emd_values[m].append(emd_best.get(m, np.nan))

                all_summary.append({
                    "signal": signal_name,
                    "sigma": sigma,
                    "lambda": lam,
                    "n_trials": n_trials,
                    "IRMF": {m: _stats(irmf_values[m]) for m in METRICS},
                    "EMD": {m: _stats(emd_values[m]) for m in METRICS},
                })

    with open(output_root / "huber_contamination_monte_carlo_breakdown_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(all_summary), f, indent=2)

    return all_summary


# Backward-compatible alias
def run_monte_carlo_robustness_grid_experiment(*args, **kwargs):
    return run_monte_carlo_robustness_experiment(*args, **kwargs)
