#!/usr/bin/python
# coding: UTF-8

"""
4.5 Boundary Effects.

Representative mode:
    5 signals × 4 noises × sigma=0.20 × 4 boundary methods

Full mode:
    5 signals × 6 noises × 2 sigma × 4 boundary methods
"""

from pathlib import Path
import json
import numpy as np

from project_config import BOUNDARY_BENCHMARK_MODES
from experiments.experiment_utils import make_signal_noise_case, run_single_irmf_case, run_single_emd_case
from experiments.experiment_reporting import print_extended_comparison


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


def run_boundary_effects_experiment(
        output_root,
        mode="representative",
        search_mode="quick",
        n=500,
        fs=500.0,
        seed=0,
):
    if mode not in BOUNDARY_BENCHMARK_MODES:
        raise ValueError(f"Unknown boundary benchmark mode: {mode}")

    cfg = BOUNDARY_BENCHMARK_MODES[mode]
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary = []

    for signal_name in cfg["signals"]:
        for noise_name in cfg["noises"]:
            for sigma in cfg["sigmas"]:
                case_dir = output_root / f"boundary_{mode}" / signal_name / noise_name / f"sigma_{sigma}"
                case_dir.mkdir(parents=True, exist_ok=True)

                case = make_signal_noise_case(signal_name, noise_name, sigma, n=n, fs=fs, seed=seed)
                t, Y, X_clean = case["t"], case["Y"], case["X_clean"]
                true_components = case.get("true_components", None)

                print("\n" + "#" * 120)
                print(f"BOUNDARY EFFECTS | mode={mode} | signal={signal_name} | noise={noise_name} | sigma={sigma}")
                print("#" * 120)

                _, irmf_periodic = run_single_irmf_case(
                    Y=Y, X_clean=X_clean, t=t, fs=fs,
                    output_dir=case_dir / "irmf_periodic",
                    search_mode=search_mode,
                    boundary_mode="periodic",
                    true_components=true_components,
                )
                _, irmf_mirror = run_single_irmf_case(
                    Y=Y, X_clean=X_clean, t=t, fs=fs,
                    output_dir=case_dir / "irmf_mirror",
                    search_mode=search_mode,
                    boundary_mode="mirror",
                    true_components=true_components,
                )
                _, emd_nbsym2 = run_single_emd_case(
                    Y=Y, X_clean=X_clean, t=t, fs=fs,
                    output_dir=case_dir / "emd_nbsym2",
                    nbsym=2,
                    true_components=true_components,
                )
                _, emd_nbsym4 = run_single_emd_case(
                    Y=Y, X_clean=X_clean, t=t, fs=fs,
                    output_dir=case_dir / "emd_nbsym4",
                    nbsym=4,
                    true_components=true_components,
                )

                rows = [
                    ("IRMF-periodic", irmf_periodic),
                    ("IRMF-mirror", irmf_mirror),
                    ("EMD-nbsym2", emd_nbsym2),
                    ("EMD-nbsym4", emd_nbsym4),
                ]

                print_extended_comparison(
                    f"BOUNDARY EFFECTS EXTENDED | {signal_name} | {noise_name} | sigma={sigma}",
                    rows,
                )

                summary.append({
                    "mode": mode,
                    "signal": signal_name,
                    "noise": noise_name,
                    "sigma": sigma,
                    "methods": {
                        name: {
                            "robust_estimation_score": result.get("robust_estimation_score"),
                            "decomposition_quality_score": result.get("decomposition_quality_score"),
                            "denoise_psnr": result.get("denoise_psnr"),
                            "denoise_corr": result.get("denoise_corr"),
                            "strict_io": result.get("strict_io"),
                            "spectral_leakage": result.get("spectral_leakage"),
                            "frequency_overlap_max_offdiag": result.get("frequency_overlap_max_offdiag"),
                            "residual_whiteness": result.get("residual_whiteness"),
                        }
                        for name, result in rows
                    },
                })

    with open(output_root / f"boundary_effects_{mode}_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2)

    return summary


# Backward-compatible alias
def run_boundary_effects_multisigma_experiment(*args, **kwargs):
    return run_boundary_effects_experiment(*args, **kwargs)
