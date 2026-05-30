#!/usr/bin/python
# coding: UTF-8

"""
Shared experiment utilities.

This module keeps experiment scripts concise and consistent.
"""

from pathlib import Path
from signal_bank.synthetic_signals import get_true_frequencies, infer_signal_name_from_path, get_true_components
import json
import numpy as np

from parameter_search.irmf_parameter_search import run_irmf_parameter_search
from sensitivity_analysis.emd_sensitivity_analysis import run_emd_sensitivity_analysis
from visualization.plot_imfs import plot_imfs, plot_input_signal
from visualization.plot_filter_bank import plot_filter_bank
from visualization.plot_pareto import plot_pareto_front
from visualization.plot_operator_evolution import plot_operator_evolution
from visualization.plot_mode_mixing import plot_matrix_heatmap
from visualization.plot_hilbert_spectrum import plot_hilbert_ridges
from visualization.plot_frequency_overlap import plot_frequency_overlap_heatmap
from visualization.plot_frequency_energy import plot_frequency_energy_map, plot_center_frequency_energy_table
from visualization.plot_residual_diagnostics import plot_residual_autocorrelation
from visualization.plot_imf_recovery import plot_imf_recovery_pairs
from experiments.experiment_reporting import print_irmf_full_report, print_emd_full_report


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_observed_signal(X_clean, sigma, seed=0):
    rng = np.random.default_rng(seed)
    noise = sigma * rng.standard_normal(len(X_clean))
    Y = X_clean + noise
    expected_noise_ratio = np.sum(noise ** 2) / (np.sum(Y ** 2) + 1e-10)
    return Y, noise, expected_noise_ratio


def default_irmf_grid(search_mode="coarse"):
    if search_mode == "coarse":
        return {
            "h1_options": np.round(np.arange(0.10, 0.251, 0.02), 3),
            "a_options": np.round(np.arange(1.20, 2.051, 0.10), 4),
            "h_min_options": np.round(np.arange(0.005, 0.0201, 0.003), 4),
            "H_options": np.round(np.arange(0.40, 1.201, 0.10), 3),
        }

    if search_mode == "quick":
        return {
            "h1_options": np.array([0.14, 0.18, 0.22]),
            "a_options": np.array([1.4, 1.7, 2.0]),
            "h_min_options": np.array([0.008, 0.014]),
            "H_options": np.array([0.5, 0.8, 1.1]),
        }

    raise ValueError("search_mode must be 'coarse' or 'quick'.")


def default_emd_grid(search_mode="coarse"):
    if search_mode == "coarse":
        return {
            "nbsym_options": [2, 3, 4, 6],
            "spline_kind_options": ["cubic"],
            "max_imf_options": [-1, 4, 5, 6, 7],
            "std_thr_options": [None, 0.05, 0.10],
            "svar_thr_options": [None],
            "total_power_thr_options": [None],
            "range_thr_options": [None],
        }

    if search_mode == "quick":
        return {
            "nbsym_options": [2, 4],
            "spline_kind_options": ["cubic"],
            "max_imf_options": [-1, 5],
            "std_thr_options": [None, 0.10],
            "svar_thr_options": [None],
            "total_power_thr_options": [None],
            "range_thr_options": [None],
        }

    raise ValueError("search_mode must be 'coarse' or 'quick'.")


def select_irmf_best(results):
    theory_best = sorted(results, key=lambda x: x["theory_diagnostic_score"])[0]
    physical_best = sorted(results, key=lambda x: x["general_physical_score"])[0]

    return {
        "theory_best": theory_best,
        "physical_best": physical_best,
    }


def select_emd_best(results):
    physical_best = sorted(results, key=lambda x: x["general_physical_score"])[0]
    return {
        "physical_best": physical_best,
    }


def save_best_plots(
        best_dict,
        t,
        fs,
        output_dir,
        prefix,
        Y_observed=None,
        X_clean=None,
        true_noise=None,
        true_frequencies=None,
        true_components=None
):
    output_dir = ensure_dir(output_dir)

    if true_frequencies is None:
        inferred_signal_name = infer_signal_name_from_path(output_dir)
        if inferred_signal_name is not None:
            true_frequencies = get_true_frequencies(inferred_signal_name, t)

    if true_noise is None and Y_observed is not None and X_clean is not None:
        true_noise = Y_observed - X_clean

    if Y_observed is not None:
        plot_input_signal(
            Y_observed=Y_observed,
            X_clean=X_clean,
            t=t,
            output_dir=output_dir,
            name=f"{prefix}_input_signal"
        )

    for label, r in best_dict.items():
        name = f"{prefix}_{label}_run_{r['run_id']}"

        plot_imfs(
            r["imfs"],
            r["residual"],
            t,
            output_dir,
            name,
            Y_observed=Y_observed,
            X_clean=X_clean,
            true_noise=true_noise
        )

        plot_filter_bank(r["imfs"], fs, output_dir, name)

        plot_frequency_energy_map(r.get("center_freqs", []), r["imfs"], output_dir, name=f"{name}_frequency_energy_map")
        plot_center_frequency_energy_table(r.get("center_freqs", []), r["imfs"], output_dir, name=f"{name}_frequency_energy_table")
        plot_residual_autocorrelation(r["residual"], output_dir, name=f"{name}_residual_autocorrelation")
        if true_components is not None:
            plot_imf_recovery_pairs(r["imfs"], true_components, t, output_dir, name=f"{name}_imf_recovery_pairs")

        if "frequency_overlap_matrix" in r:
            plot_frequency_overlap_heatmap(
                r["frequency_overlap_matrix"],
                output_dir,
                name=name
            )

        if "imf_correlation_matrix" in r:
            plot_matrix_heatmap(
                r["imf_correlation_matrix"],
                output_dir,
                name=f"{name}_imf_correlation_heatmap",
                title="IMF Time-Domain Correlation Matrix",
                colorbar_label="abs correlation"
            )

        if "ridge_overlap_matrix" in r:
            plot_matrix_heatmap(
                r["ridge_overlap_matrix"],
                output_dir,
                name=f"{name}_ridge_overlap_heatmap",
                title="Hilbert Ridge Overlap Matrix",
                colorbar_label="ridge overlap fraction"
            )

        plot_hilbert_ridges(
            r["imfs"],
            t,
            fs,
            output_dir,
            name=name,
            true_frequencies=true_frequencies
        )

        if "operator_evolution" in r:
            plot_operator_evolution(
                r["operator_evolution"],
                output_dir,
                name=name
            )


def serialize_result_summary(r):
    keys = [
        "method", "run_id", "imf_count",
        "general_physical_score",
        "residual_energy_monotonicity_score",
        "trace_monotonicity_score",
        "operator_monotonicity_score",
        "b0_evolution_max",
        "b0_evolution_final",
        "b0_evolution_mean",
        "snr_gain_db",
        "output_snr_db",
        "input_snr_db",
        "frequency_spacing_penalty",
        "frequency_spacing_mean_ratio",
        "frequency_spacing_min_ratio",
        "strict_io", "spectral_leakage", "ifs",
        "frequency_overlap_matrix", "frequency_overlap_mean_offdiag", "frequency_overlap_max_offdiag",
        "frequency_overlap_adjacent_mean", "frequency_overlap_adjacent_max",
        "hilbert_amp_concentration_avg",
        "hilbert_if_std_max",
        "hilbert_if_std_avg",
        "hilbert_if_mean_avg",
        "mode_mixing_score",
        "local_frequency_crossing_rate",
        "ridge_overlap_max_offdiag",
        "ridge_overlap_mean_offdiag",
        "imf_corr_max_offdiag",
        "imf_corr_mean_offdiag",
        "unsupervised_split_penalty", "spacing_penalty",
        "monotonic_penalty", "energy_penalty",
        "denoise_mse", "denoise_psnr",
        "imf_recovery_matched_count",
        "imf_recovery_corr",
        "imf_recovery_rmse", "denoise_corr",
        "spectral_corr",
        "completeness_mse", "completeness_psnr", "completeness_corr",
        "energy_ratio",
        "center_freqs", "imf_energy_ratios",
        "local_theory_score", "operator_score", "theory_diagnostic_score",
        "final_trace_norm", "final_contraction",
        "operator_evolution_mean_b0_scale",
        "operator_evolution_final_residual_energy",
        "operator_evolution_max_contraction",
        "operator_evolution_mean_contraction",
        "operator_evolution_final_trace_norm",
        "median_b0", "max_b0",
        "emd_residual_trend_smoothness",
        "emd_residual_low_frequency_dominance",
        "boundary_signal_energy_ratio", "boundary_imf_energy_ratio", "boundary_residual_energy_ratio",
        "boundary_reconstruction_error_ratio", "endpoint_jump_amplification_observed", "endpoint_jump_amplification_residual",
    ]

    out = {}

    for k in keys:
        if k in r:
            v = r[k]
            if isinstance(v, np.ndarray):
                v = v.tolist()
            elif isinstance(v, (np.float32, np.float64)):
                v = float(v)
            elif isinstance(v, (np.int32, np.int64)):
                v = int(v)
            out[k] = v

    return out


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.int32, np.int64)):
            return int(o)
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=default)




# ============================================================
# V13 parameter-search run plotting
# ============================================================

def _run_label_from_params(r):
    parts = [f"run_{r.get('run_id', 'NA')}"]

    for key in ["h1", "a", "h_min", "H"]:
        if key in r:
            val = str(round(float(r[key]), 4)).replace(".", "p")
            parts.append(f"{key}_{val}")

    return "_".join(parts)


def select_parameter_runs_for_plots(results, mode="top_n", top_n=6):
    """
    Select IRMF parameter-search runs for extra decomposition plots.

    mode:
        none    : no extra plots
        all     : all grid-search runs
        top_n   : top N runs by general_physical_score
        diverse : top N by physical score + top N by theory score
    """
    if results is None:
        return []

    results = list(results)

    if mode == "none":
        return []
    if mode == "all":
        return results

    if mode == "top_n":
        selected = sorted(
            results,
            key=lambda x: x.get("general_physical_score", float("inf"))
        )[:top_n]

    elif mode == "diverse":
        by_physical = sorted(
            results,
            key=lambda x: x.get("general_physical_score", float("inf"))
        )[:top_n]

        by_theory = sorted(
            results,
            key=lambda x: x.get("theory_diagnostic_score", float("inf"))
        )[:top_n]

        selected = by_physical + by_theory

    else:
        raise ValueError("mode must be one of: none, all, top_n, diverse")

    out = []
    seen = set()
    for r in selected:
        rid = r.get("run_id", id(r))
        if rid in seen:
            continue
        seen.add(rid)
        out.append(r)

    return out


def save_parameter_run_plots(
        results,
        t,
        fs,
        output_dir,
        prefix,
        Y_observed=None,
        X_clean=None,
        true_noise=None,
        true_frequencies=None,
        mode="diverse",
        top_n=5,
        true_components=None
):
    if true_frequencies is None:
        inferred_signal_name = infer_signal_name_from_path(output_dir)
        if inferred_signal_name is not None:
            true_frequencies = get_true_frequencies(inferred_signal_name, t)

    """
    Save plots for selected IRMF parameter-search runs.

    Output:
        <output_dir>/parameter_runs/

    This allows inspection of different (h1, a, h_min, H) choices, instead of
    only plotting the best run.
    """
    if mode == "none":
        return

    output_dir = ensure_dir(output_dir) / "parameter_runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if true_noise is None and Y_observed is not None and X_clean is not None:
        true_noise = Y_observed - X_clean

    selected = select_parameter_runs_for_plots(results, mode=mode, top_n=top_n)

    for r in selected:
        name = f"{prefix}_{_run_label_from_params(r)}"

        plot_imfs(
            r["imfs"],
            r["residual"],
            t,
            output_dir,
            name,
            Y_observed=Y_observed,
            X_clean=X_clean,
            true_noise=true_noise
        )

        plot_filter_bank(r["imfs"], fs, output_dir, name)

        plot_frequency_energy_map(r.get("center_freqs", []), r["imfs"], output_dir, name=f"{name}_frequency_energy_map")
        plot_center_frequency_energy_table(r.get("center_freqs", []), r["imfs"], output_dir, name=f"{name}_frequency_energy_table")
        plot_residual_autocorrelation(r["residual"], output_dir, name=f"{name}_residual_autocorrelation")
        if true_components is not None:
            plot_imf_recovery_pairs(r["imfs"], true_components, t, output_dir, name=f"{name}_imf_recovery_pairs")

        if "frequency_overlap_matrix" in r:
            plot_frequency_overlap_heatmap(
                r["frequency_overlap_matrix"],
                output_dir,
                name=name
            )

        if "imf_correlation_matrix" in r:
            plot_matrix_heatmap(
                r["imf_correlation_matrix"],
                output_dir,
                name=f"{name}_imf_correlation_heatmap",
                title="IMF Time-Domain Correlation Matrix",
                colorbar_label="abs correlation"
            )

        if "ridge_overlap_matrix" in r:
            plot_matrix_heatmap(
                r["ridge_overlap_matrix"],
                output_dir,
                name=f"{name}_ridge_overlap_heatmap",
                title="Hilbert Ridge Overlap Matrix",
                colorbar_label="ridge overlap fraction"
            )

        plot_hilbert_ridges(
            r["imfs"],
            t,
            fs,
            output_dir,
            name=name,
            true_frequencies=true_frequencies
        )

        if "operator_evolution" in r:
            plot_operator_evolution(
                r["operator_evolution"],
                output_dir,
                name=name
            )


def run_single_irmf_case(
        Y,
        X_clean,
        t,
        fs,
        output_dir,
        search_mode="quick",
        expected_noise_ratio=None,
        boundary_mode="periodic",
        min_support_points=3,
        plot_parameter_runs_mode="diverse",
        plot_parameter_runs_top_n=5,
        true_frequencies=None,
        true_components=None
):
    if true_components is None:
        inferred_signal_name_for_components = infer_signal_name_from_path(output_dir)
        if inferred_signal_name_for_components is not None:
            true_components = get_true_components(inferred_signal_name_for_components, t)

    if true_frequencies is None:
        inferred_signal_name = infer_signal_name_from_path(output_dir)
        if inferred_signal_name is not None:
            true_frequencies = get_true_frequencies(inferred_signal_name, t)

    grid = default_irmf_grid(search_mode)

    results = run_irmf_parameter_search(
        Y=Y,
        X_clean=X_clean,
        fs=fs,
        T=t,
        expected_noise_ratio=expected_noise_ratio,
        boundary_mode=boundary_mode,
        min_support_points=min_support_points,
        **grid,
        true_components=true_components
    )

    best = select_irmf_best(results)
    print_irmf_full_report("IRMF THEORY BEST FULL DIAGNOSTIC REPORT", best["theory_best"])
    print_irmf_full_report("IRMF PHYSICAL BEST FULL DIAGNOSTIC REPORT", best["physical_best"])
    save_best_plots(best, t, fs, output_dir, "irmf", Y_observed=Y, X_clean=X_clean, true_components=true_components)

    plot_pareto_front(
        results,
        output_dir,
        x_key="general_physical_score",
        y_key="theory_diagnostic_score",
        name="irmf_pareto_physical_theory"
    )

    save_parameter_run_plots(
        results,
        t=t,
        fs=fs,
        output_dir=output_dir,
        prefix="irmf_param",
        Y_observed=Y,
        X_clean=X_clean,
        true_frequencies=true_frequencies,
        mode=plot_parameter_runs_mode,
        top_n=plot_parameter_runs_top_n,
        true_components=true_components
    )

    return results, best


def run_single_emd_case(
        Y,
        X_clean,
        t,
        fs,
        output_dir,
        search_mode="quick",
        true_frequencies=None,
        true_components=None
):
    if true_components is None:
        inferred_signal_name_for_components = infer_signal_name_from_path(output_dir)
        if inferred_signal_name_for_components is not None:
            true_components = get_true_components(inferred_signal_name_for_components, t)

    if true_frequencies is None:
        inferred_signal_name = infer_signal_name_from_path(output_dir)
        if inferred_signal_name is not None:
            true_frequencies = get_true_frequencies(inferred_signal_name, t)

    grid = default_emd_grid(search_mode)

    results = run_emd_sensitivity_analysis(
        Y=Y,
        T=t,
        X_clean=X_clean,
        fs=fs,
        **grid,
        true_components=true_components
    )

    best = select_emd_best(results)
    print_emd_full_report("EMD PHYSICAL BEST FULL DIAGNOSTIC REPORT", best["physical_best"])
    save_best_plots(best, t, fs, output_dir, "emd", Y_observed=Y, X_clean=X_clean, true_components=true_components)

    plot_pareto_front(
        results,
        output_dir,
        x_key="spectral_leakage",
        y_key="strict_io",
        name="emd_pareto_leakage_io"
    )

    return results, best
