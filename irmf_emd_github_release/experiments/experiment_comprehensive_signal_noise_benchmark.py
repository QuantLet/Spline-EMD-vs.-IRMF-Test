#!/usr/bin/python
# coding: UTF-8

"""
4.2 Comprehensive Signal-Noise Benchmark.

4.2.1 Main noise benchmark:
    7 signals × 8 noises × 3 sigma levels = 168 cases

4.2.2 Optional Huber contamination lambda sweep:
    7 signals × lambda grid × sigma levels
"""

from pathlib import Path
import json
import numpy as np

from project_config import (
    COMPREHENSIVE_SIGNAL_FAMILY,
    COMPREHENSIVE_NOISE_FAMILY,
    COMPREHENSIVE_SIGMA_LEVELS,
    HUBER_LAMBDA_GRID,
    HUBER_CONTAMINATION_OUTLIER_SCALE,
)

from experiments.experiment_utils import (
    make_signal_noise_case,
    run_single_irmf_case,
    run_single_emd_case,
)

from experiments.experiment_reporting import (
    print_compact_comparison,
    print_extended_comparison,
)


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


def _get_physical_best(result):
    """
    run_single_irmf_case / run_single_emd_case may return:
        {
            "theory_best": {...},
            "physical_best": {...}
        }

    For 4.2 summary we want the best physical run.
    """
    if result is None:
        return {}

    if isinstance(result, dict) and "physical_best" in result:
        physical_best = result.get("physical_best")
        if isinstance(physical_best, dict):
            return physical_best

    if isinstance(result, dict):
        return result

    return {}


def _method_summary(result):
    result = _get_physical_best(result)

    return {
        # run info
        "run_id": result.get("run_id"),

        # FIX10 paper-final Case-Level Scores
        "case_score": result.get("case_score"),
        "case_score_final": result.get("case_score_final"),
        "reconstruction_score": result.get("reconstruction_score"),
        "structural_fidelity_score": result.get("structural_fidelity_score"),
        "structural_fidelity_raw": result.get("structural_fidelity_raw"),
        "contamination_resistance_score": result.get("contamination_resistance_score"),
        "od_ud_penalty_factor": result.get("od_ud_penalty_factor"),

        # Reconstruction score components
        "denoise_corr_score": result.get("denoise_corr_score"),
        "denoise_nmse": result.get("denoise_nmse"),
        "nmse_score": result.get("nmse_score"),
        "spectral_corr": result.get("spectral_corr"),
        "spectral_corr_score": result.get("spectral_corr_score"),

        # Structural Fidelity score components
        "imf_recovery_corr_score": result.get("imf_recovery_corr_score"),
        "imf_recovery_rmse_score": result.get("imf_recovery_rmse_score"),
        "imf_recovery_score": result.get("imf_recovery_score"),
        "orthogonality_leakage_group_score": result.get("orthogonality_leakage_group_score"),
        "true_component_mixing_group_score": result.get("true_component_mixing_group_score"),
        "frequency_separation_group_score": result.get("frequency_separation_group_score"),
        "component_splitting_index": result.get("component_splitting_index"),
        "component_merging_index": result.get("component_merging_index"),
        "inter_imf_entanglement_index": result.get("inter_imf_entanglement_index"),
        "mode_mixing_index": result.get("mode_mixing_index"),
        "transient_smearing_index": result.get("transient_smearing_index"),
        "transient_preservation_score": result.get("transient_preservation_score"),
        "over_decomposition_penalty": result.get("over_decomposition_penalty"),
        "under_decomposition_index": result.get("under_decomposition_index"),
        "effective_imf_count": result.get("effective_imf_count"),
        "true_component_count": result.get("true_component_count"),

        # Contamination Resistance
        "outlier_resistance_index": result.get("outlier_resistance_index"),
        "ori": result.get("ori"),
        "noise_capture_corr_score": result.get("noise_capture_corr_score"),

        # Layer 2 — Robust Estimation
        "robust_estimation_score": result.get("robust_estimation_score"),
        "denoise_mse": result.get("denoise_mse"),
        "denoise_psnr": result.get("denoise_psnr"),
        "denoise_corr": result.get("denoise_corr"),
        "input_snr_db": result.get("input_snr_db"),
        "output_snr_db": result.get("output_snr_db"),
        "snr_gain_db": result.get("snr_gain_db"),
        "noise_capture_corr": result.get("noise_capture_corr"),

        # Layer 3 — Decomposition Quality
        "decomposition_quality_score": result.get("decomposition_quality_score"),
        "strict_io": result.get("strict_io"),
        "spectral_leakage": result.get("spectral_leakage"),
        "frequency_overlap_mean_offdiag": result.get("frequency_overlap_mean_offdiag"),
        "frequency_overlap_max_offdiag": result.get("frequency_overlap_max_offdiag"),
        "frequency_separation_score": result.get("frequency_separation_score"),
        "residual_whiteness": result.get("residual_whiteness"),
        "imf_count": result.get("imf_count"),

        # Layer 4 — Supporting Diagnostics
        "frequency_spacing_min_ratio": result.get("frequency_spacing_min_ratio"),
        "frequency_spacing_mean_ratio": result.get("frequency_spacing_mean_ratio"),
        "frequency_spacing_penalty": result.get("frequency_spacing_penalty"),
        "ifs": result.get("ifs"),
        "energy_ratio": result.get("energy_ratio"),
        "residual_autocorrelation_score": result.get("residual_autocorrelation_score"),

        # IRMF parameters / EMD may be None here
        "h1": result.get("h1"),
        "a": result.get("a"),
        "h_min": result.get("h_min"),
        "H": result.get("H"),
        "boundary_mode": result.get("boundary_mode"),

        # EMD-specific settings
        "nbsym": result.get("nbsym"),
        "spline_kind": result.get("spline_kind"),
        "max_imf": result.get("max_imf"),
    }


def _summarize_pair(
        irmf_best,
        emd_best,
        signal_name,
        noise_name,
        sigma,
        extra=None,
):
    row = {
        "signal": signal_name,
        "noise": noise_name,
        "sigma": sigma,
        "IRMF": _method_summary(irmf_best),
        "EMD": _method_summary(emd_best),
    }

    if extra:
        row.update(extra)

    return row


def run_comprehensive_signal_noise_benchmark(
        output_root,
        signal_names=COMPREHENSIVE_SIGNAL_FAMILY,
        noise_names=COMPREHENSIVE_NOISE_FAMILY,
        sigma_levels=COMPREHENSIVE_SIGMA_LEVELS,
        search_mode="quick",
        n=500,
        fs=500.0,
        seed=0,
        run_huber_lambda_sweep=False,
):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    main_summary = run_main_noise_benchmark(
        output_root=output_root / "main_noise_benchmark",
        signal_names=signal_names,
        noise_names=noise_names,
        sigma_levels=sigma_levels,
        search_mode=search_mode,
        n=n,
        fs=fs,
        seed=seed,
    )

    huber_summary = None

    if run_huber_lambda_sweep:
        huber_summary = run_huber_contamination_lambda_sweep(
            output_root=output_root / "huber_lambda_sweep",
            signal_names=signal_names,
            lambda_grid=HUBER_LAMBDA_GRID,
            sigma_levels=sigma_levels,
            search_mode=search_mode,
            n=n,
            fs=fs,
            seed=seed,
        )

    return {
        "main_noise_benchmark": main_summary,
        "huber_lambda_sweep": huber_summary,
    }


def run_main_noise_benchmark(
        output_root,
        signal_names,
        noise_names,
        sigma_levels,
        search_mode="quick",
        n=500,
        fs=500.0,
        seed=0,
):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary = []

    benchmark_metadata = {
        "design": "7 signals × 8 noise models × 3 sigma levels = 168 main cases",
        "n_signals": len(signal_names),
        "n_noises": len(noise_names),
        "n_sigma_levels": len(sigma_levels),
        "n_cases": len(signal_names) * len(noise_names) * len(sigma_levels),
        "signals": list(signal_names),
        "noises": list(noise_names),
        "sigma_levels": list(sigma_levels),
    }

    with open(output_root / "main_noise_benchmark_design.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(benchmark_metadata), f, indent=2)

    for signal_name in signal_names:
        for noise_name in noise_names:
            for sigma in sigma_levels:

                case_dir = (
                    output_root
                    / signal_name
                    / noise_name
                    / f"sigma_{sigma}"
                )

                case_dir.mkdir(parents=True, exist_ok=True)

                case = make_signal_noise_case(
                    signal_name=signal_name,
                    noise_name=noise_name,
                    sigma=sigma,
                    n=n,
                    fs=fs,
                    seed=seed,
                )

                t = case["t"]
                Y = case["Y"]
                X_clean = case["X_clean"]
                true_components = case.get("true_components", None)

                print("\n" + "#" * 120)
                print(
                    f"COMPREHENSIVE BENCHMARK | "
                    f"signal={signal_name} | "
                    f"noise={noise_name} | "
                    f"sigma={sigma}"
                )
                print("#" * 120)

                _, irmf_best = run_single_irmf_case(
                    Y=Y,
                    X_clean=X_clean,
                    t=t,
                    fs=fs,
                    output_dir=case_dir / "irmf",
                    search_mode=search_mode,
                    true_components=true_components,
                )

                _, emd_best = run_single_emd_case(
                    Y=Y,
                    X_clean=X_clean,
                    t=t,
                    fs=fs,
                    output_dir=case_dir / "emd",
                    true_components=true_components,
                )

                rows = [
                    ("IRMF", _get_physical_best(irmf_best)),
                    ("EMD", _get_physical_best(emd_best)),
                ]

                print_compact_comparison(
                    f"COMPREHENSIVE COMPACT | "
                    f"{signal_name} | {noise_name} | sigma={sigma}",
                    rows,
                )

                print_extended_comparison(
                    f"COMPREHENSIVE EXTENDED | "
                    f"{signal_name} | {noise_name} | sigma={sigma}",
                    rows,
                )

                summary.append(
                    _summarize_pair(
                        irmf_best,
                        emd_best,
                        signal_name,
                        noise_name,
                        sigma,
                    )
                )

    with open(
            output_root / "main_noise_benchmark_summary.json",
            "w",
            encoding="utf-8",
    ) as f:
        json.dump(
            _json_safe(summary),
            f,
            indent=2,
        )

    return summary


def run_huber_contamination_lambda_sweep(
        output_root,
        signal_names=COMPREHENSIVE_SIGNAL_FAMILY,
        lambda_grid=HUBER_LAMBDA_GRID,
        sigma_levels=COMPREHENSIVE_SIGMA_LEVELS,
        search_mode="quick",
        n=500,
        fs=500.0,
        seed=0,
):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary = []

    benchmark_metadata = {
        "design": "7 signals × 8 noise models × 3 sigma levels = 168 main cases",
        "n_signals": len(signal_names),
        "n_noises": len(noise_names),
        "n_sigma_levels": len(sigma_levels),
        "n_cases": len(signal_names) * len(noise_names) * len(sigma_levels),
        "signals": list(signal_names),
        "noises": list(noise_names),
        "sigma_levels": list(sigma_levels),
    }

    with open(output_root / "main_noise_benchmark_design.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(benchmark_metadata), f, indent=2)

    for signal_name in signal_names:
        for sigma in sigma_levels:
            for lam in lambda_grid:

                case_dir = (
                    output_root
                    / signal_name
                    / f"sigma_{sigma}"
                    / f"lambda_{lam}"
                )

                case_dir.mkdir(parents=True, exist_ok=True)

                case = make_signal_noise_case(
                    signal_name=signal_name,
                    noise_name="huber_contamination",
                    sigma=sigma,
                    n=n,
                    fs=fs,
                    seed=seed,
                    noise_kwargs={
                        "lam": lam,
                        "outlier_scale": HUBER_CONTAMINATION_OUTLIER_SCALE,
                    },
                )

                t = case["t"]
                Y = case["Y"]
                X_clean = case["X_clean"]
                true_components = case.get("true_components", None)

                print("\n" + "#" * 120)
                print(
                    f"HUBER LAMBDA SWEEP | "
                    f"signal={signal_name} | "
                    f"sigma={sigma} | "
                    f"lambda={lam}"
                )
                print("#" * 120)

                _, irmf_best = run_single_irmf_case(
                    Y=Y,
                    X_clean=X_clean,
                    t=t,
                    fs=fs,
                    output_dir=case_dir / "irmf",
                    search_mode=search_mode,
                    true_components=true_components,
                )

                _, emd_best = run_single_emd_case(
                    Y=Y,
                    X_clean=X_clean,
                    t=t,
                    fs=fs,
                    output_dir=case_dir / "emd",
                    true_components=true_components,
                )

                rows = [
                    ("IRMF", _get_physical_best(irmf_best)),
                    ("EMD", _get_physical_best(emd_best)),
                ]

                print_compact_comparison(
                    f"HUBER LAMBDA SWEEP COMPACT | "
                    f"{signal_name} | sigma={sigma} | lambda={lam}",
                    rows,
                )

                print_extended_comparison(
                    f"HUBER LAMBDA SWEEP EXTENDED | "
                    f"{signal_name} | sigma={sigma} | lambda={lam}",
                    rows,
                )

                summary.append(
                    _summarize_pair(
                        irmf_best,
                        emd_best,
                        signal_name,
                        "huber_contamination",
                        sigma,
                        extra={"lambda": lam},
                    )
                )

    with open(
            output_root / "huber_lambda_sweep_summary.json",
            "w",
            encoding="utf-8",
    ) as f:
        json.dump(
            _json_safe(summary),
            f,
            indent=2,
        )

    return summary
