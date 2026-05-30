#!/usr/bin/python
# coding: UTF-8

"""
Layered reporting utilities.
"""

import numpy as np


def _fmt(v):
    if v is None:
        return "N/A"
    if isinstance(v, str):
        return v
    try:
        if isinstance(v, np.ndarray):
            return np.array2string(v, precision=4, suppress_small=True)
    except Exception:
        pass
    try:
        if not np.isfinite(v):
            return "N/A"
        if abs(float(v)) >= 1000:
            return f"{float(v):.3e}"
        return f"{float(v):.6f}"
    except Exception:
        return str(v)


def _first_available(r, keys, default=None):
    if r is None:
        return default
    for key in keys:
        if key in r:
            value = r.get(key)
            if value is not None:
                try:
                    if isinstance(value, float) and not np.isfinite(value):
                        continue
                except Exception:
                    pass
                return value
    return default


def _get(r, key):
    aliases = {
        "mean_b0_like": ["mean_b0_like", "mean_b0", "operator_evolution_mean_b0_scale"],
        "median_b0_like": ["median_b0_like", "median_b0"],
        "max_b0_like": ["max_b0_like", "max_b0", "b0_evolution_max"],
        "hessian_condition_proxy": ["hessian_condition_proxy", "hessian_condition", "local_condition_number"],
        "hessian_positive_ratio": ["hessian_positive_ratio", "positive_hessian_ratio", "hessian_stability"],
        "trace_norm": ["trace_norm", "operator_evolution_final_trace_norm"],
        "operator_norm": ["operator_norm"],
        "contraction_ratio": ["contraction_ratio", "operator_evolution_mean_contraction"],
        "residual_energy_ratio": ["residual_energy_ratio", "operator_evolution_final_residual_energy"],
        "b0_evolution_mean": ["b0_evolution_mean", "operator_evolution_mean_b0_scale"],
        "b0_evolution_final": ["b0_evolution_final"],
        "operator_monotonicity_score": ["operator_monotonicity_score"],
        "irmf_performance_score": ["irmf_performance_score", "theory_diagnostic_score"],
        "robust_estimation_score": ["robust_estimation_score"],
        "decomposition_quality_score": ["decomposition_quality_score", "general_physical_score"],
        "denoise_psnr": ["denoise_psnr", "psnr"],
        "denoise_mse": ["denoise_mse", "mse"],
        "denoise_corr": ["denoise_corr", "corr"],
        "input_snr_db": ["input_snr_db"],
        "output_snr_db": ["output_snr_db"],
        "snr_gain_db": ["snr_gain_db"],
        "imf_recovery_rmse": ["imf_recovery_rmse"],
        "imf_recovery_corr": ["imf_recovery_corr"],
        "imf_recovery_matched_count": ["imf_recovery_matched_count"],
        "noise_capture_corr": ["noise_capture_corr"],
        "noise_capture_energy_ratio": ["noise_capture_energy_ratio"],
        "strict_io": ["strict_io", "io"],
        "spectral_leakage": ["spectral_leakage"],
        "frequency_overlap_mean_offdiag": ["frequency_overlap_mean_offdiag"],
        "frequency_overlap_max_offdiag": ["frequency_overlap_max_offdiag"],
        "frequency_separation_score": ["frequency_separation_score"],
        "residual_whiteness": ["residual_whiteness"],
        "imf_count": ["imf_count"],
        "frequency_spacing_min_ratio": ["frequency_spacing_min_ratio"],
        "frequency_spacing_mean_ratio": ["frequency_spacing_mean_ratio"],
        "frequency_spacing_penalty": ["frequency_spacing_penalty", "center_frequency_spacing_penalty"],
        "ifs": ["ifs", "instantaneous_frequency_smoothness"],
        "energy_ratio": ["energy_ratio"],
        "energy_concentration_penalty": ["energy_concentration_penalty"],
        "residual_autocorrelation_score": ["residual_autocorrelation_score", "residual_whiteness"],
    }
    return _first_available(r, aliases.get(key, [key]), default=None)


def _print_line(name, value):
    print(f"{name:<42}: {_fmt(value)}")


def _section(title):
    print("\n" + title)
    print("-" * len(title))


def print_irmf_full_report(title, r):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)

    _section("IRMF parameters")
    for key in ["run_id", "h1", "a", "h_min", "H", "boundary_mode", "min_support_points"]:
        if key in r:
            _print_line(key, r.get(key))

    _section("Layer 1 — IRMF Performance Evaluation")

    print("Level A — Core local/operator behaviour")
    for key in ["mean_b0_like", "trace_norm", "operator_norm", "contraction_ratio"]:
        _print_line(key, _get(r, key))

    print("\nLevel B — Multiscale evolution")
    for key in ["residual_energy_ratio", "b0_evolution_mean", "b0_evolution_final", "operator_monotonicity_score"]:
        _print_line(key, _get(r, key))

    print("\nLevel C — Hessian diagnostics")
    for key in ["hessian_condition_proxy", "hessian_positive_ratio"]:
        _print_line(key, _get(r, key))

    _print_line("irmf_performance_score", _get(r, "irmf_performance_score"))

    _section("Layer 2 — Robust Estimation")
    print("Reconstruction Accuracy")
    for key in ["denoise_psnr", "denoise_mse", "denoise_corr", "input_snr_db", "output_snr_db", "snr_gain_db"]:
        _print_line(key, _get(r, key))

    print("\nIMF Recovery")
    for key in ["imf_recovery_rmse", "imf_recovery_corr", "imf_recovery_matched_count"]:
        _print_line(key, _get(r, key))

    print("\nNoise Capture")
    for key in ["noise_capture_corr", "noise_capture_energy_ratio"]:
        _print_line(key, _get(r, key))

    _print_line("robust_estimation_score", _get(r, "robust_estimation_score"))

    _section("Layer 3 — Decomposition Quality Evaluation")
    for key in [
        "strict_io",
        "spectral_leakage",
        "frequency_overlap_mean_offdiag",
        "frequency_overlap_max_offdiag",
        "frequency_separation_score",
        "residual_whiteness",
        "imf_count",
    ]:
        _print_line(key, _get(r, key))

    _print_line("decomposition_quality_score", _get(r, "decomposition_quality_score"))

    _section("Layer 4 — Supporting Diagnostics")
    for key in [
        "frequency_spacing_min_ratio",
        "frequency_spacing_mean_ratio",
        "frequency_spacing_penalty",
        "ifs",
        "energy_ratio",
        "energy_concentration_penalty",
        "residual_autocorrelation_score",
    ]:
        _print_line(key, _get(r, key))


def print_emd_full_report(title, r):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)

    _section("EMD settings")
    for key in ["run_id", "nbsym", "spline_kind", "max_imf"]:
        if key in r:
            _print_line(key, r.get(key))

    _section("Layer 2 — Robust Estimation")
    print("Reconstruction Accuracy")
    for key in ["denoise_psnr", "denoise_mse", "denoise_corr", "input_snr_db", "output_snr_db", "snr_gain_db"]:
        _print_line(key, _get(r, key))

    print("\nIMF Recovery")
    for key in ["imf_recovery_rmse", "imf_recovery_corr", "imf_recovery_matched_count"]:
        _print_line(key, _get(r, key))

    print("\nNoise Capture")
    for key in ["noise_capture_corr", "noise_capture_energy_ratio"]:
        _print_line(key, _get(r, key))

    _print_line("robust_estimation_score", _get(r, "robust_estimation_score"))

    _section("Layer 3 — Decomposition Quality Evaluation")
    for key in [
        "strict_io",
        "spectral_leakage",
        "frequency_overlap_mean_offdiag",
        "frequency_overlap_max_offdiag",
        "frequency_separation_score",
        "residual_whiteness",
        "imf_count",
    ]:
        _print_line(key, _get(r, key))

    _print_line("decomposition_quality_score", _get(r, "decomposition_quality_score"))

    _section("Layer 4 — Supporting Diagnostics")
    for key in [
        "frequency_spacing_min_ratio",
        "frequency_spacing_mean_ratio",
        "frequency_spacing_penalty",
        "ifs",
        "energy_ratio",
        "energy_concentration_penalty",
        "residual_autocorrelation_score",
    ]:
        _print_line(key, _get(r, key))


def print_compact_comparison(title, rows):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)

    metrics = [
        "robust_estimation_score",
        "decomposition_quality_score",
        "denoise_psnr",
        "denoise_mse",
        "denoise_corr",
        "snr_gain_db",
        "noise_capture_corr",
        "strict_io",
        "spectral_leakage",
        "frequency_overlap_max_offdiag",
        "frequency_separation_score",
        "residual_whiteness",
        "imf_count",
    ]

    header = "Method".ljust(16) + "".join(m.ljust(32) for m in metrics)
    print(header)
    print("-" * len(header))

    for name, r in rows:
        line = str(name).ljust(16)
        for m in metrics:
            line += _fmt(_get(r, m)).ljust(32)
        print(line)


def print_extended_comparison(title, rows):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)

    metrics = [
        "robust_estimation_score",
        "decomposition_quality_score",
        "irmf_performance_score",
        "denoise_psnr",
        "denoise_mse",
        "denoise_corr",
        "input_snr_db",
        "output_snr_db",
        "snr_gain_db",
        "imf_recovery_rmse",
        "imf_recovery_corr",
        "noise_capture_corr",
        "noise_capture_energy_ratio",
        "strict_io",
        "spectral_leakage",
        "frequency_overlap_mean_offdiag",
        "frequency_overlap_max_offdiag",
        "frequency_separation_score",
        "residual_whiteness",
        "imf_count",
        "frequency_spacing_min_ratio",
        "frequency_spacing_mean_ratio",
        "frequency_spacing_penalty",
        "ifs",
        "energy_ratio",
        "residual_autocorrelation_score",
    ]

    header = "Method".ljust(16) + "".join(m.ljust(34) for m in metrics)
    print(header)
    print("-" * len(header))

    for name, r in rows:
        line = str(name).ljust(16)
        for m in metrics:
            line += _fmt(_get(r, m)).ljust(34)
        print(line)


def print_monte_carlo_comparison(title, summary):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)

    metrics = [
        "robust_estimation_score",
        "decomposition_quality_score",
        "denoise_psnr",
        "denoise_mse",
        "denoise_corr",
        "snr_gain_db",
        "noise_capture_corr",
        "strict_io",
        "spectral_leakage",
        "frequency_overlap_max_offdiag",
        "residual_whiteness",
    ]

    for method in ["IRMF", "EMD"]:
        if method not in summary:
            continue

        print("\n" + method)
        print("-" * len(method))

        method_summary = summary[method]

        for metric in metrics:
            if metric not in method_summary:
                continue

            stats = method_summary[metric]
            mean = stats.get("mean", None)
            std = stats.get("std", None)
            min_v = stats.get("min", None)
            max_v = stats.get("max", None)

            print(
                f"{metric:<36}: "
                f"mean={_fmt(mean):>12} | "
                f"std={_fmt(std):>12} | "
                f"min={_fmt(min_v):>12} | "
                f"max={_fmt(max_v):>12}"
            )

    if "n_trials" in summary:
        print(f"\nn_trials: {summary['n_trials']}")
    if "noise_name" in summary:
        print(f"noise_name: {summary['noise_name']}")
    if "sigma" in summary:
        print(f"sigma: {summary['sigma']}")
