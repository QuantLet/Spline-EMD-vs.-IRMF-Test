#!/usr/bin/python
# coding: UTF-8

"""Appendix experiments: sanity checks, oracle comparison, reproducibility."""

from pathlib import Path

import numpy as np

from project_config import (
    COMPREHENSIVE_NOISE_FAMILY,
    COMPREHENSIVE_SIGMA_LEVELS,
    COMPREHENSIVE_SIGNAL_FAMILY,
    DEFAULT_FS,
    DEFAULT_N,
    DEFAULT_SEARCH_MODE,
    DEFAULT_SEED,
    GLOBAL_EMD_PARAMS,
    GLOBAL_IRMF_PARAMS,
)
from core_algorithms.strict_spokoiny_irmf import rho_spline_grad_hess, strict_spokoiny_irmf
from experiments.experiment_utils import (
    make_signal_noise_case,
    run_single_emd_case,
    run_single_irmf_case,
    serialize_result_summary,
)
from experiments.paper_pipeline_utils import ensure_dir, write_json, write_manifest


def run_method_faithfulness_checks(
        output_root,
        irmf_params=GLOBAL_IRMF_PARAMS,
        signal_name="stationary_multi_sine",
        noise_name="gaussian",
        sigma=0.10,
        n=DEFAULT_N,
        fs=DEFAULT_FS,
        seed=DEFAULT_SEED,
):
    output_root = ensure_dir(output_root)
    case = make_signal_noise_case(signal_name, noise_name, sigma, n=n, fs=fs, seed=seed)
    imfs, residual, residual_history, scale_history = strict_spokoiny_irmf(
        Y=case["Y"],
        T=case["t"],
        h1=irmf_params["h1"],
        a=irmf_params["a"],
        h_min=irmf_params["h_min"],
        H=irmf_params["H"],
        boundary_mode=irmf_params.get("boundary_mode", "periodic"),
        min_support_points=irmf_params.get("min_support_points", 3),
    )
    reconstruction_error = float(np.max(np.abs(np.sum(np.asarray(imfs), axis=0) + residual - case["Y"])))
    bandwidths = [float(row["h"]) for row in scale_history]
    bandwidth_monotone = all(bandwidths[i + 1] < bandwidths[i] for i in range(len(bandwidths) - 1))

    # Smoothed-median loss sanity checks at the fixed methodology value H=1.
    x_grid = np.linspace(-4.0, 4.0, 401)
    eps = 1e-5
    rho, grad, hess = rho_spline_grad_hess(x_grid, H=1.0)
    rho_plus, _, _ = rho_spline_grad_hess(x_grid + eps, H=1.0)
    rho_minus, _, _ = rho_spline_grad_hess(x_grid - eps, H=1.0)
    _, grad_plus, _ = rho_spline_grad_hess(x_grid + eps, H=1.0)
    _, grad_minus, _ = rho_spline_grad_hess(x_grid - eps, H=1.0)
    finite_diff_grad = (rho_plus - rho_minus) / (2.0 * eps)
    finite_diff_hess = (grad_plus - grad_minus) / (2.0 * eps)
    rho_even_error = float(np.max(np.abs(rho - rho[::-1])))
    grad_odd_error = float(np.max(np.abs(grad + grad[::-1])))
    hessian_min = float(np.min(hess))
    finite_difference_grad_error = float(np.max(np.abs(finite_diff_grad - grad)))
    finite_difference_hessian_error = float(np.max(np.abs(finite_diff_hess - hess)))

    checks = {
        "section": "Appendix A Method-Faithfulness Checks",
        "reconstruction_identity_max_abs_error": reconstruction_error,
        "bandwidths": bandwidths,
        "bandwidth_strictly_decreasing": bandwidth_monotone,
        "fixed_loss_width_H": 1.0,
        "rho_even_error": rho_even_error,
        "rho_prime_odd_error": grad_odd_error,
        "hessian_min": hessian_min,
        "finite_difference_gradient_error": finite_difference_grad_error,
        "finite_difference_hessian_error": finite_difference_hessian_error,
        "passed": bool(
            reconstruction_error < 1e-8
            and bandwidth_monotone
            and rho_even_error < 1e-10
            and grad_odd_error < 1e-10
            and hessian_min >= -1e-12
            and finite_difference_grad_error < 1e-8
            and finite_difference_hessian_error < 1e-8
        ),
    }
    write_json(checks, output_root / "appendix_A_method_faithfulness_checks.json")
    return checks


def run_oracle_comparison_appendix(
        output_root,
        signals=COMPREHENSIVE_SIGNAL_FAMILY,
        noises=COMPREHENSIVE_NOISE_FAMILY,
        sigmas=COMPREHENSIVE_SIGMA_LEVELS,
        search_mode=DEFAULT_SEARCH_MODE,
        n=DEFAULT_N,
        fs=DEFAULT_FS,
        seed=DEFAULT_SEED,
):
    """
    Appendix C: per-case oracle comparison.

    This intentionally uses the legacy grid-search entry points and should not
    be used for main claims.  It is an upper-bound diagnostic only.
    """
    output_root = ensure_dir(output_root)
    rows = []
    for signal_name in signals:
        for noise_name in noises:
            for sigma in sigmas:
                case_dir = output_root / signal_name / noise_name / f"sigma_{sigma}"
                case = make_signal_noise_case(signal_name, noise_name, sigma, n=n, fs=fs, seed=seed)
                _, irmf_best = run_single_irmf_case(
                    Y=case["Y"],
                    X_clean=case["X_clean"],
                    t=case["t"],
                    fs=fs,
                    output_dir=case_dir / "irmf",
                    search_mode=search_mode,
                    expected_noise_ratio=case.get("expected_noise_ratio"),
                    true_components=case.get("true_components"),
                )
                _, emd_best = run_single_emd_case(
                    Y=case["Y"],
                    X_clean=case["X_clean"],
                    t=case["t"],
                    fs=fs,
                    output_dir=case_dir / "emd",
                    search_mode=search_mode,
                    true_components=case.get("true_components"),
                )
                rows.append({
                    "signal": signal_name,
                    "noise": noise_name,
                    "sigma": sigma,
                    "IRMF_oracle_physical_best": serialize_result_summary(irmf_best["physical_best"]),
                    "IRMF_oracle_theory_best": serialize_result_summary(irmf_best["theory_best"]),
                    "EMD_oracle_physical_best": serialize_result_summary(emd_best["physical_best"]),
                    "note": "Oracle upper-bound result; not used for main fixed-parameter claims.",
                })
    write_json(rows, output_root / "appendix_C_oracle_comparison.json")
    return rows


def write_reproducibility_appendix(output_root, extra=None):
    output_root = ensure_dir(output_root)
    return write_manifest(output_root, {
        "section": "Appendix F Reproducibility Details",
        "global_irmf_params": dict(GLOBAL_IRMF_PARAMS),
        "global_emd_params": dict(GLOBAL_EMD_PARAMS),
        **({} if extra is None else dict(extra)),
    })


if __name__ == "__main__":
    root = Path("IRMF_EMD_PAPER_RESULTS_V4E_SMOOTHED_MEDIAN_H_FIXED_THEORY_CONSTRAINED") / "appendices"
    run_method_faithfulness_checks(root / "appendix_A_method_faithfulness_checks")
    write_reproducibility_appendix(root / "appendix_F_reproducibility")
