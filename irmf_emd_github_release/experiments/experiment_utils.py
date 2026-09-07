#!/usr/bin/python
# coding: UTF-8

"""
Shared experiment utilities.

This module keeps experiment scripts concise and consistent.
"""

from pathlib import Path
from signal_bank.synthetic_signals import get_signal, get_true_frequencies, infer_signal_name_from_path, get_true_components
import json
import numpy as np
import signal
from time import perf_counter

from parameter_search.irmf_parameter_search import run_irmf_parameter_search
from sensitivity_analysis.emd_sensitivity_analysis import run_emd_sensitivity_analysis
from core_algorithms.strict_spokoiny_irmf import strict_spokoiny_irmf
from core_algorithms.emd_core import run_emd_decomposition
from core_algorithms.eemd_wrapper import run_eemd
from core_algorithms.ceemdan_wrapper import run_ceemdan
from diagnostics.shared_physical_diagnostics import evaluate_shared_physical_diagnostics
from diagnostics.irmf_local_theory_diagnostics import (
    compute_irmf_local_theory_diagnostics,
    summarize_irmf_local_theory_score,
)
from diagnostics.irmf_operator_diagnostics import (
    compute_irmf_operator_diagnostics,
    summarize_irmf_operator_score,
)
from diagnostics.irmf_operator_evolution import compute_operator_evolution, summarize_operator_evolution
from diagnostics.layer1_irmf_performance import attach_irmf_performance_score
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
from noise_bank.noise_models import generate_noise


CONTAMINATION_NOISES = {"impulsive", "burst", "huber_contamination", "huber", "epsilon_contamination"}
LOWER_IS_BETTER_SUMMARY_METRICS = {
    "denoise_nmse",
    "imf_recovery_nrmse",
    "component_splitting_index",
    "component_merging_index",
    "inter_imf_entanglement_index",
    "mode_mixing_index",
    "decomposition_count_error",
    "clean_region_nmse",
    "contaminated_region_nmse",
    "contamination_spillover_error",
    "spectral_leakage",
    "strict_io",
    "frequency_overlap_max_offdiag",
    "over_decomposition_penalty",
    "under_decomposition_index",
}

DEFAULT_METHOD_TIMEOUT_SECONDS = 120
STRUCTURAL_FAILURE_IMF_RECOVERY_THRESHOLD = 0.20
STRUCTURAL_FAILURE_MODE_MIXING_THRESHOLD = 0.90
STRUCTURAL_FAILURE_DECOMPOSITION_COUNT_ERROR_THRESHOLD = 2.00
STRUCTURAL_FAILURE_MAX_EFFECTIVE_IMF_COUNT = 12


class MethodTimeoutError(TimeoutError):
    """Raised when one method run exceeds the pre-specified timeout."""


def _timeout_handler(signum, frame):
    raise MethodTimeoutError("method run exceeded pre-specified timeout")


def _run_with_timeout(func, timeout_seconds=None):
    if timeout_seconds is None or timeout_seconds <= 0:
        return func()
    timeout_seconds = int(timeout_seconds)
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        return func()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _contains_nonfinite_output(result):
    if not isinstance(result, dict):
        return True
    for key in ("imfs", "residual"):
        value = result.get(key)
        if value is None:
            return True
        try:
            arr = np.asarray(value, dtype=float)
            if arr.size == 0 or not np.all(np.isfinite(arr)):
                return True
        except Exception:
            return True
    return False


def _noisy_input_nmse(Y, X_clean):
    try:
        y = np.asarray(Y, dtype=float)
        x = np.asarray(X_clean, dtype=float)
        return float(np.sum((y - x) ** 2) / (np.sum(x ** 2) + 1e-12))
    except Exception:
        return np.nan


def _failure_summary_for_exception(method_name, exc, runtime_seconds=None, timeout_seconds=None):
    is_timeout = isinstance(exc, MethodTimeoutError)
    return {
        "method": method_name,
        "error": str(exc),
        "runtime_seconds": runtime_seconds,
        "timeout_seconds": timeout_seconds,
        "timeout_flag": bool(is_timeout),
        "exception_flag": True,
        "nonfinite_output_flag": False,
        "computational_failure": True,
        "structural_failure": False,
        "denoising_failure": False,
        "any_failure": True,
        "failure_level": "computational",
        "failure_reason": "timeout" if is_timeout else "exception",
    }


def attach_failure_flags(
        result,
        Y_observed=None,
        X_clean=None,
        timeout_seconds=None,
        runtime_seconds=None,
):
    """Attach pre-specified computational, structural, and denoising failures."""
    if result is None:
        return _failure_summary_for_exception("unknown", RuntimeError("missing result"), runtime_seconds, timeout_seconds)
    out = dict(result)
    if runtime_seconds is not None:
        out["runtime_seconds"] = float(runtime_seconds)
    out["timeout_seconds"] = None if timeout_seconds is None else float(timeout_seconds)
    out["timeout_flag"] = bool(out.get("timeout_flag", False))
    out["exception_flag"] = bool(out.get("exception_flag", False) or out.get("error"))
    out["nonfinite_output_flag"] = bool(out.get("nonfinite_output_flag", False) or _contains_nonfinite_output(out))
    if Y_observed is not None and X_clean is not None:
        out["noisy_input_nmse"] = _noisy_input_nmse(Y_observed, X_clean)

    computational = bool(out["timeout_flag"] or out["exception_flag"] or out["nonfinite_output_flag"])
    imf_recovery = out.get("imf_recovery_score")
    mode_mixing = out.get("mode_mixing_index")
    count_error = out.get("decomposition_count_error")
    effective_count = out.get("effective_imf_count", out.get("imf_count"))
    structural_reasons = []
    try:
        if imf_recovery is not None and np.isfinite(imf_recovery) and float(imf_recovery) < STRUCTURAL_FAILURE_IMF_RECOVERY_THRESHOLD:
            structural_reasons.append("low_imf_recovery")
    except Exception:
        pass
    try:
        if mode_mixing is not None and np.isfinite(mode_mixing) and float(mode_mixing) > STRUCTURAL_FAILURE_MODE_MIXING_THRESHOLD:
            structural_reasons.append("severe_mode_mixing")
    except Exception:
        pass
    try:
        if count_error is not None and np.isfinite(count_error) and float(count_error) > STRUCTURAL_FAILURE_DECOMPOSITION_COUNT_ERROR_THRESHOLD:
            structural_reasons.append("large_decomposition_count_error")
    except Exception:
        pass
    try:
        if effective_count is not None and np.isfinite(effective_count) and float(effective_count) > STRUCTURAL_FAILURE_MAX_EFFECTIVE_IMF_COUNT:
            structural_reasons.append("imf_count_explosion")
    except Exception:
        pass

    denoising = False
    try:
        denoise_nmse = float(out.get("denoise_nmse"))
        noisy_nmse = float(out.get("noisy_input_nmse"))
        denoising = bool(np.isfinite(denoise_nmse) and np.isfinite(noisy_nmse) and denoise_nmse >= noisy_nmse)
    except Exception:
        denoising = False

    out["computational_failure"] = computational
    out["structural_failure"] = bool((not computational) and structural_reasons)
    out["denoising_failure"] = bool((not computational) and denoising)
    out["failure_reason"] = ";".join(structural_reasons) if structural_reasons else out.get("failure_reason")
    if computational:
        level = "computational"
    elif out["structural_failure"]:
        level = "structural"
    elif out["denoising_failure"]:
        level = "denoising"
    else:
        level = "none"
    out["failure_level"] = level
    out["any_failure"] = level != "none"
    return out


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


def target_snr_noise_scale(X_clean, base_noise, target_snr_db, eps=1e-12):
    """
    Return the scalar alpha such that alpha * base_noise attains target SNR.

    The target follows the energy-ratio definition:

        SNR_db = 10 log10(sum X_clean^2 / sum noise^2).

    This keeps the clean synthetic signal unchanged while making relative noise
    severity comparable across signal families with different RMS amplitudes.
    """
    x = np.asarray(X_clean, dtype=float)
    z = np.asarray(base_noise, dtype=float)
    signal_energy = float(np.sum(x ** 2))
    base_noise_energy = float(np.sum(z ** 2))
    if signal_energy <= eps:
        raise ValueError("target-SNR noise scaling requires nonzero clean signal energy")
    if base_noise_energy <= eps:
        raise ValueError("target-SNR noise scaling requires nonzero base noise energy")
    q = 10.0 ** (float(target_snr_db) / 10.0)
    return float(np.sqrt(signal_energy / (q * base_noise_energy)))


def realized_snr_db(X_clean, noise, eps=1e-12):
    signal_energy = float(np.sum(np.asarray(X_clean, dtype=float) ** 2))
    noise_energy = float(np.sum(np.asarray(noise, dtype=float) ** 2))
    if noise_energy <= eps:
        return np.inf
    return float(10.0 * np.log10((signal_energy + eps) / noise_energy))


def make_signal_noise_case(
        signal_name,
        noise_name,
        sigma,
        n=500,
        fs=500.0,
        seed=0,
        noise_kwargs=None,
        target_snr_db=None,
):
    """
    Build one synthetic signal-noise case.

    Returns a dictionary with:
        t, X_clean, Y, noise, expected_noise_ratio, true_components
    """
    noise_kwargs = {} if noise_kwargs is None else dict(noise_kwargs)

    t = np.arange(n, dtype=float) / float(fs)
    # Normalize to [0, 1] for synthetic signal definitions.
    if n > 1:
        t_unit = np.linspace(0.0, 1.0, n, endpoint=False)
    else:
        t_unit = np.array([0.0])

    X_clean = get_signal(signal_name, t_unit)
    if target_snr_db is None:
        noise = generate_noise(
            noise_name=noise_name,
            n=n,
            sigma=sigma,
            seed=seed,
            **noise_kwargs,
        )
        noise_design = "fixed_sigma"
        noise_scale_alpha = float(sigma)
    else:
        base_noise = generate_noise(
            noise_name=noise_name,
            n=n,
            sigma=1.0,
            seed=seed,
            **noise_kwargs,
        )
        noise_scale_alpha = target_snr_noise_scale(
            X_clean=X_clean,
            base_noise=base_noise,
            target_snr_db=target_snr_db,
        )
        noise = (noise_scale_alpha * base_noise).astype(float)
        noise_design = "target_snr_energy_ratio"
    contamination_mask = infer_contamination_mask(
        noise_name=noise_name,
        n=n,
        seed=seed,
        noise_kwargs=noise_kwargs,
    )
    Y = X_clean + noise
    expected_noise_ratio = float(np.sum(noise ** 2) / (np.sum(Y ** 2) + 1e-10))

    try:
        true_components = get_true_components(signal_name, t_unit)
    except Exception:
        true_components = None

    return {
        "signal_name": signal_name,
        "noise_name": noise_name,
        "sigma": sigma,
        "target_snr_db": target_snr_db,
        "noise_design": noise_design,
        "noise_scale_alpha": noise_scale_alpha,
        "realized_input_snr_db": realized_snr_db(X_clean, noise),
        "clean_signal_energy": float(np.sum(np.asarray(X_clean, dtype=float) ** 2)),
        "clean_signal_rms": float(np.sqrt(np.mean(np.asarray(X_clean, dtype=float) ** 2))),
        "noise_energy": float(np.sum(np.asarray(noise, dtype=float) ** 2)),
        "noise_rms": float(np.sqrt(np.mean(np.asarray(noise, dtype=float) ** 2))),
        "t": t_unit,
        "T": t,
        "X_clean": X_clean,
        "Y": Y,
        "noise": noise,
        "contamination_mask": contamination_mask,
        "contamination_mask_applicable": contamination_mask is not None,
        "expected_noise_ratio": expected_noise_ratio,
        "true_components": true_components,
    }


def infer_contamination_mask(noise_name, n, seed=0, noise_kwargs=None):
    """
    Reconstruct contamination locations for noise models with explicit labels.

    The noise generators currently return only a vector.  This helper mirrors
    their deterministic random draws so region-level contamination diagnostics
    can be computed without changing the generated noise values.
    """
    noise_kwargs = {} if noise_kwargs is None else dict(noise_kwargs)
    name = str(noise_name)
    rng = np.random.default_rng(seed)

    if name == "impulsive":
        _ = rng.standard_normal(n)
        return rng.random(n) < float(noise_kwargs.get("p", 0.03))

    if name == "burst":
        n_bursts = int(noise_kwargs.get("n_bursts", 3))
        burst_len = noise_kwargs.get("burst_len", None)
        if burst_len is None:
            burst_len = max(3, n // 50)
        burst_len = int(min(max(1, burst_len), n))
        mask = np.zeros(n, dtype=bool)
        _ = rng.standard_normal(n)
        for _ in range(n_bursts):
            start = int(rng.integers(0, max(1, n - burst_len + 1)))
            end = min(n, start + burst_len)
            mask[start:end] = True
            _ = rng.normal(0.0, float(noise_kwargs.get("scale", 6.0)), size=end - start)
        return mask

    if name in ("huber_contamination", "huber", "epsilon_contamination"):
        _ = rng.standard_normal(n)
        return rng.random(n) < float(noise_kwargs.get("lam", noise_kwargs.get("lambda_", 0.05)))

    return None



def default_irmf_grid(search_mode="coarse"):
    if search_mode == "coarse":
        return {
            "h1_options": np.round(np.arange(0.10, 0.251, 0.02), 3),
            "a_options": np.round(np.arange(1.20, 2.051, 0.10), 4),
            "h_min_options": np.round(np.arange(0.005, 0.0201, 0.003), 4),
            "H_options": np.array([1.0]),
        }

    if search_mode == "quick":
        return {
            "h1_options": np.array([0.14, 0.18, 0.22]),
            "a_options": np.array([1.4, 1.7, 2.0]),
            "h_min_options": np.array([0.008, 0.014]),
            "H_options": np.array([1.0]),
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
        "case_score", "case_score_final", "case_score_role",
        "reconstruction_score", "structural_fidelity_score", "structural_fidelity_raw",
        "contamination_resistance_score",
        "denoise_corr_score", "nmse_score", "spectral_corr_score",
        "imf_recovery_corr_score", "imf_recovery_rmse_score", "imf_recovery_score",
        "imf_recovery_alignment_rule", "imf_recovery_assignment_pairs",
        "component_splitting_index", "component_merging_index",
        "true_component_splitting_max", "estimated_component_merging_max",
        "true_component_mixing_matrix", "component_mixing_basis",
        "metric_definition_version", "evaluation_framework_version",
        "evaluation_framework_status", "evaluation_framework_frozen_date",
        "structural_metric_schema_version",
        "legacy_metric_alias_status", "true_component_metrics_available",
        "component_mixing_index_formula",
        "component_mixing_association_threshold", "component_mixing_energy_threshold",
        "component_mixing_allocation_floor",
        "true_component_association_strength", "estimated_component_association_strength",
        "missing_true_component_count", "missing_true_component_fraction",
        "missing_true_component_rate", "missing_true_component_energy_ratio",
        "unmatched_estimated_component_count", "unmatched_estimated_component_fraction",
        "unmatched_estimated_component_rate",
        "spurious_mode_energy_ratio",
        "true_component_mixing_group_score",
        "orthogonality_leakage_group_score", "frequency_separation_group_score",
        "local_structure_preservation_group_score", "od_ud_penalty_factor",
        "structural_fidelity_score_policy",
        "outlier_resistance_index", "ori", "noise_capture_corr", "noise_capture_corr_score",
        "noise_capture_energy_ratio",
        "contamination_region_applicable", "contaminated_point_count", "contaminated_fraction",
        "contaminated_region_nmse", "contaminated_region_preservation_score",
        "clean_region_nmse", "clean_region_preservation_score",
        "contamination_spillover_error", "contamination_spillover_score", "spillover_radius",
        "signal_leakage_into_noise", "signal_leakage_into_noise_score",
        "signal_leakage_residual_energy", "signal_leakage_observed_energy",
        "signal_leakage_degenerate_flag", "signal_leakage_basis",
        "clean_region_signal_leakage", "clean_region_signal_leakage_score",
        "mode_mixing_index", "mode_mixing_index_legacy",
        "inter_imf_entanglement_index", "legacy_metric_used_in_primary_analysis",
        "transient_smearing_index", "transient_preservation_score",
        "decomposition_count_error", "decomposition_adequacy_score",
        "over_decomposition_penalty", "under_decomposition_index",
        "effective_imf_count", "true_component_count",
        "relative_decomposition_count_error",
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
        "proxy_metric_protocol", "trimmed_residual_whiteness",
        "residual_autocorrelation_abs_mean", "residual_kurtosis", "residual_sparsity",
        "residual_high_frequency_energy_ratio", "residual_low_frequency_leakage",
        "residual_outlier_concentration", "spectral_concentration_mean",
        "spectral_concentration_min", "instantaneous_frequency_continuity_mean",
        "frequency_band_separation", "scale_order_consistency",
        "component_center_frequency_min", "component_center_frequency_max",
        "energy_conservation_error", "real_proxy_no_ground_truth_required",
        "hilbert_amp_concentration_avg",
        "hilbert_if_std_max",
        "hilbert_if_std_avg",
        "hilbert_if_mean_avg",
        "mode_mixing_score",
        "inter_imf_entanglement_index",
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
        "imf_recovery_rmse", "imf_recovery_nrmse", "denoise_corr",
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


def paper_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: paper_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [paper_json_safe(v) for v in obj]
    return obj


def method_result_summary(result):
    """
    Compact JSON-safe summary for paper tables.

    This deliberately excludes full arrays so section-level summary files stay
    small and easy to read.  Full decompositions are still available from the
    individual run result if a caller wants to save plots.
    """
    if result is None:
        return {}
    keys = [
        "method", "run_id", "signal", "noise", "sigma", "seed",
        "case_score", "case_score_final", "case_score_role", "general_physical_score",
        "reconstruction_score", "structural_fidelity_score", "contamination_resistance_score",
        "robustness_score", "robustness_score_legacy_field",
        "structural_fidelity_raw", "structural_fidelity_score_policy",
        "true_component_mixing_group_score",
        "orthogonality_leakage_group_score", "frequency_separation_group_score",
        "local_structure_preservation_group_score", "od_ud_penalty_factor",
        "denoise_nmse", "denoise_mse", "denoise_corr", "spectral_corr",
        "input_snr_db", "output_snr_db", "snr_gain_db",
        "imf_recovery_score", "imf_recovery_corr", "imf_recovery_rmse", "imf_recovery_nrmse",
        "component_splitting_index", "component_merging_index",
        "imf_recovery_alignment_rule", "imf_recovery_assignment_pairs",
        "metric_definition_version", "evaluation_framework_version",
        "evaluation_framework_status", "evaluation_framework_frozen_date",
        "structural_metric_schema_version",
        "true_component_metrics_available",
        "component_mixing_basis", "component_mixing_index_formula",
        "component_mixing_association_threshold", "component_mixing_energy_threshold",
        "component_mixing_allocation_floor",
        "true_component_splitting_max", "estimated_component_merging_max",
        "true_component_association_strength", "estimated_component_association_strength",
        "missing_true_component_count", "missing_true_component_fraction",
        "missing_true_component_rate", "missing_true_component_energy_ratio",
        "unmatched_estimated_component_count", "unmatched_estimated_component_fraction",
        "unmatched_estimated_component_rate",
        "spurious_mode_energy_ratio",
        "mode_mixing_index", "mode_mixing_index_legacy",
        "inter_imf_entanglement_index", "legacy_metric_used_in_primary_analysis",
        "transient_smearing_index", "transient_preservation_score",
        "strict_io", "spectral_leakage", "frequency_overlap_max_offdiag",
        "frequency_separation_score", "residual_whiteness",
        "proxy_metric_protocol", "trimmed_residual_whiteness",
        "residual_autocorrelation_abs_mean", "residual_kurtosis", "residual_sparsity",
        "residual_high_frequency_energy_ratio", "residual_low_frequency_leakage",
        "residual_outlier_concentration", "spectral_concentration_mean",
        "spectral_concentration_min", "instantaneous_frequency_continuity_mean",
        "frequency_band_separation", "scale_order_consistency",
        "component_center_frequency_min", "component_center_frequency_max",
        "energy_conservation_error", "real_proxy_no_ground_truth_required",
        "noise_capture_corr", "noise_capture_corr_score", "noise_capture_energy_ratio",
        "noise_energy_ratio", "noise_energy_log_error", "remaining_noise_energy_ratio",
        "noise_energy_bias_class", "noise_energy_eps", "noise_energy_bias_tolerance",
        "noise_energy_diagnostic_role", "remaining_noise_energy_ratio_role",
        "noise_energy_not_computable_reason",
        "reconstruction_protocol_id", "reconstruction_protocol_status",
        "reconstruction_protocol_truth_assisted", "reconstruction_protocol_association_threshold",
        "reconstruction_protocol_threshold_operator", "reconstruction_protocol_waveform_modification",
        "protocol_component_candidate_count", "protocol_imf_component_count",
        "protocol_residual_or_trend_bearing_candidate_included",
        "protocol_selected_component_indices", "protocol_selected_component_count",
        "protocol_selected_fraction", "protocol_selection_status_code",
        "protocol_zero_variance_noise_flag", "protocol_noise_reason_code",
        "protocol_estimated_signal_energy", "protocol_estimated_noise_energy",
        "protocol_reconstruction_closure_max_abs_error",
        "outlier_resistance_index", "ori",
        "contamination_region_applicable", "contaminated_point_count", "contaminated_fraction",
        "contaminated_region_nmse", "contaminated_region_preservation_score",
        "clean_region_nmse", "clean_region_preservation_score",
        "contamination_spillover_error", "contamination_spillover_score", "spillover_radius",
        "signal_leakage_into_noise", "signal_leakage_into_noise_score",
        "signal_leakage_residual_energy", "signal_leakage_observed_energy",
        "signal_leakage_degenerate_flag", "signal_leakage_basis",
        "clean_region_signal_leakage", "clean_region_signal_leakage_score",
        "imf_count", "effective_imf_count", "true_component_count",
        "decomposition_count_error", "relative_decomposition_count_error",
        "decomposition_adequacy_score",
        "over_decomposition_penalty", "under_decomposition_index",
        "h1", "a", "h_min", "H", "boundary_mode", "min_support_points",
        "loss_name", "loss_tuning", "collect_optimization_diagnostics",
        "nbsym", "spline_kind", "max_imf", "std_thr",
        "trials", "noise_width", "epsilon", "parallel",
        "random_seed", "algorithm_seed", "data_seed", "runtime_seconds",
        "timeout_seconds", "timeout_flag", "exception_flag", "nonfinite_output_flag",
        "computational_failure", "structural_failure", "denoising_failure",
        "any_failure", "failure_level", "failure_reason", "noisy_input_nmse",
    ]
    out = {}
    for key in keys:
        if key in result:
            out[key] = paper_json_safe(result[key])
    return out


def paired_method_summary(irmf_result, emd_result, metadata=None):
    """Return one paper-row with IRMF, EMD, and paired differences."""
    metadata = {} if metadata is None else dict(metadata)
    row = dict(metadata)
    row["IRMF"] = method_result_summary(irmf_result)
    row["EMD"] = method_result_summary(emd_result)
    for key in [
        "case_score",
        "reconstruction_score",
        "structural_fidelity_score",
        "contamination_resistance_score",
        "denoise_nmse",
        "denoise_corr",
        "imf_recovery_score",
        "imf_recovery_corr",
        "imf_recovery_nrmse",
        "component_splitting_index",
        "component_merging_index",
        "mode_mixing_index",
        "inter_imf_entanglement_index",
        "decomposition_count_error",
        "decomposition_adequacy_score",
        "noise_capture_corr",
        "outlier_resistance_index",
        "clean_region_nmse",
        "clean_region_preservation_score",
        "contaminated_region_nmse",
        "contaminated_region_preservation_score",
        "contamination_spillover_error",
        "contamination_spillover_score",
        "signal_leakage_into_noise",
        "signal_leakage_into_noise_score",
        "clean_region_signal_leakage",
        "clean_region_signal_leakage_score",
    ]:
        iv = irmf_result.get(key) if isinstance(irmf_result, dict) else None
        ev = emd_result.get(key) if isinstance(emd_result, dict) else None
        try:
            if iv is not None and ev is not None and np.isfinite(iv) and np.isfinite(ev):
                row[f"delta_{key}_IRMF_minus_EMD"] = float(iv - ev)
        except Exception:
            pass
    return row


def aggregate_paired_rows(rows):
    """
    Lightweight aggregate statistics for paper section summaries.

    Full inferential testing can be done downstream in R/Python, but this gives
    immediately useful means, standard errors, and win rates for each section.
    """
    out = {"n_cases": int(len(rows)), "metrics": {}}
    metrics = [
        "case_score",
        "reconstruction_score",
        "structural_fidelity_score",
        "contamination_resistance_score",
        "denoise_nmse",
        "denoise_corr",
        "imf_recovery_score",
        "imf_recovery_corr",
        "imf_recovery_nrmse",
        "component_splitting_index",
        "component_merging_index",
        "mode_mixing_index",
        "inter_imf_entanglement_index",
        "decomposition_count_error",
        "decomposition_adequacy_score",
        "noise_capture_corr",
        "outlier_resistance_index",
        "clean_region_nmse",
        "clean_region_preservation_score",
        "contaminated_region_nmse",
        "contaminated_region_preservation_score",
        "contamination_spillover_error",
        "contamination_spillover_score",
        "signal_leakage_into_noise",
        "signal_leakage_into_noise_score",
        "clean_region_signal_leakage",
        "clean_region_signal_leakage_score",
    ]
    for metric in metrics:
        ivals, evals, deltas = [], [], []
        for row in rows:
            i = row.get("IRMF", {}).get(metric)
            e = row.get("EMD", {}).get(metric)
            try:
                if i is not None and np.isfinite(i):
                    ivals.append(float(i))
                if e is not None and np.isfinite(e):
                    evals.append(float(e))
                if i is not None and e is not None and np.isfinite(i) and np.isfinite(e):
                    deltas.append(float(i) - float(e))
            except Exception:
                continue
        if not deltas:
            continue
        arr = np.asarray(deltas, dtype=float)
        higher_is_better = metric not in LOWER_IS_BETTER_SUMMARY_METRICS
        wins = arr > 0 if higher_is_better else arr < 0
        out["metrics"][metric] = {
            "irmf_mean": float(np.mean(ivals)) if ivals else None,
            "emd_mean": float(np.mean(evals)) if evals else None,
            "delta_mean": float(np.mean(arr)),
            "delta_se": float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
            "irmf_win_rate": float(np.mean(wins)),
            "n_paired": int(len(arr)),
        }
    return out


def aggregate_method_family_rows(rows, methods=("IRMF", "EMD", "EEMD", "CEEMDAN")):
    """
    Aggregate rows containing IRMF and one or more EMD-family baselines.

    The output keeps the legacy mean fields for each method and adds paired
    IRMF-vs-baseline quantities.  For lower-is-better metrics, benefit deltas
    are direction-normalized so positive values always favor IRMF.
    """
    metrics = [
        "case_score",
        "reconstruction_score",
        "structural_fidelity_score",
        "contamination_resistance_score",
        "robustness_score",
        "denoise_nmse",
        "imf_recovery_score",
        "component_splitting_index",
        "component_merging_index",
        "mode_mixing_index",
        "inter_imf_entanglement_index",
        "noise_capture_corr",
        "outlier_resistance_index",
    ]
    methods = tuple(method for method in methods if any(isinstance(row.get(method), dict) for row in rows))
    out = {
        "n_cases": int(len(rows)),
        "methods": list(methods),
        "baselines": [method for method in methods if method != "IRMF"],
        "metrics": {},
    }
    for metric in metrics:
        metric_out = {}
        for method in methods:
            vals = []
            for row in rows:
                value = row.get(method, {}).get(metric)
                try:
                    if value is not None and np.isfinite(value):
                        vals.append(float(value))
                except Exception:
                    continue
            metric_out[f"{method}_mean"] = float(np.mean(vals)) if vals else None
            metric_out[f"{method}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else (0.0 if vals else None)
            metric_out[f"{method}_n"] = int(len(vals))

        higher_is_better = metric not in LOWER_IS_BETTER_SUMMARY_METRICS
        for baseline in [method for method in methods if method != "IRMF"]:
            pairs = []
            for row in rows:
                iv = row.get("IRMF", {}).get(metric)
                bv = row.get(baseline, {}).get(metric)
                try:
                    if iv is not None and bv is not None and np.isfinite(iv) and np.isfinite(bv):
                        pairs.append((float(iv), float(bv)))
                except Exception:
                    continue
            if not pairs:
                metric_out[f"IRMF_minus_{baseline}_delta_mean"] = None
                metric_out[f"IRMF_minus_{baseline}_benefit_delta_mean"] = None
                metric_out[f"IRMF_vs_{baseline}_win_rate"] = None
                metric_out[f"IRMF_vs_{baseline}_n_paired"] = 0
                continue
            raw_deltas = np.asarray([iv - bv for iv, bv in pairs], dtype=float)
            benefit_deltas = raw_deltas if higher_is_better else -raw_deltas
            wins = benefit_deltas > 0.0
            metric_out[f"IRMF_minus_{baseline}_delta_mean"] = float(np.mean(raw_deltas))
            metric_out[f"IRMF_minus_{baseline}_benefit_delta_mean"] = float(np.mean(benefit_deltas))
            metric_out[f"IRMF_vs_{baseline}_win_rate"] = float(np.mean(wins))
            metric_out[f"IRMF_vs_{baseline}_n_paired"] = int(len(pairs))
        out["metrics"][metric] = metric_out
    return out


def run_fixed_irmf_case(
        Y,
        X_clean,
        t,
        fs,
        irmf_params,
        expected_noise_ratio=None,
        true_components=None,
        contamination_mask=None,
        run_id="fixed",
        reconstruction_protocol_id=None,
        compute_operator_diagnostics_flag=True,
):
    """Run IRMF once with a pre-selected global parameter set."""
    params = dict(irmf_params)
    if params.get("H_parameterization") == "relative_noise_scale":
        y_arr = np.asarray(Y, dtype=float)
        dy = np.diff(y_arr)
        mad = float(np.median(np.abs(dy - np.median(dy)))) if dy.size else np.nan
        scale_hat = mad / (0.67448975 * np.sqrt(2.0)) if np.isfinite(mad) else np.nan
        fallback_used = False
        if not np.isfinite(scale_hat) or scale_hat <= 1e-12:
            mad_y = float(np.median(np.abs(y_arr - np.median(y_arr)))) if y_arr.size else np.nan
            scale_hat = mad_y / 0.67448975 if np.isfinite(mad_y) else np.nan
            fallback_used = True
        if not np.isfinite(scale_hat) or scale_hat <= 1e-12:
            raise ValueError("relative-H IRMF requires finite positive scale_hat")
        c_h = float(params["c_H"])
        h_realized = float(c_h * scale_hat)
        params["H"] = h_realized
        params["loss_tuning"] = {"H": h_realized}
        params["scale_hat_case"] = float(scale_hat)
        params["H_realized_case"] = h_realized
        params["scale_estimator_fallback_used"] = bool(fallback_used)
    imfs, residual, residual_history, scale_history = strict_spokoiny_irmf(
        Y=Y,
        T=t,
        h1=params.get("h1", 0.18),
        a=params.get("a", np.sqrt(2)),
        h_min=params.get("h_min", 0.012),
        H=params.get("H", 0.80),
        boundary_mode=params.get("boundary_mode", "periodic"),
        min_support_points=params.get("min_support_points", 3),
        loss_name=params.get("loss_name", "gaussian_smoothed_median"),
        loss_tuning=params.get("loss_tuning"),
        collect_optimization_diagnostics=params.get("collect_optimization_diagnostics", False),
        initialization_offset=params.get("initialization_offset", 0.0),
    )
    physical = evaluate_shared_physical_diagnostics(
        Y_observed=Y,
        X_clean=X_clean,
        imfs=imfs,
        residual=residual,
        fs=fs,
        residual_penalty_mode="whiteness",
        true_components=true_components,
        contamination_mask=contamination_mask,
        reconstruction_protocol_id=reconstruction_protocol_id,
    )
    local_theory_diag = compute_irmf_local_theory_diagnostics(Y, scale_history, params.get("H", 0.80))
    local_theory_summary = summarize_irmf_local_theory_score(
        local_theory_diag,
        expected_noise_ratio=expected_noise_ratio,
    )
    if compute_operator_diagnostics_flag:
        operator_diag = compute_irmf_operator_diagnostics(Y, scale_history, params.get("H", 0.80))
        operator_summary = summarize_irmf_operator_score(operator_diag)
    else:
        operator_diag = {}
        operator_summary = {
            "operator_score": np.nan,
            "trace_norm": np.nan,
            "operator_norm": np.nan,
            "contraction_ratio": np.nan,
            "operator_diagnostic_status": "skipped_for_development_selection",
        }
    operator_evolution_rows = compute_operator_evolution(scale_history)
    operator_evolution_summary = summarize_operator_evolution(operator_evolution_rows)
    theory_summary = {}
    theory_summary.update(local_theory_summary)
    theory_summary.update(operator_summary)
    theory_summary["theory_diagnostic_score_raw"] = (
        0.65 * local_theory_summary["local_theory_score"]
        + 0.35 * operator_summary["operator_score"]
    )
    out = {
        "method": "IRMF-fixed",
        "run_id": run_id,
        "imfs": imfs,
        "residual": residual,
        "residual_history": residual_history,
        "scale_history": scale_history,
        "local_theory_diagnostics": local_theory_diag,
        "operator_diagnostics": operator_diag,
        "operator_evolution": operator_evolution_rows,
        **params,
    }
    out.update(physical)
    out.update(theory_summary)
    out.update(operator_evolution_summary)
    out = attach_irmf_performance_score(out)
    return out


def run_fixed_emd_case(
        Y,
        X_clean,
        t,
        fs,
        emd_params,
        true_components=None,
        contamination_mask=None,
        run_id="fixed",
        reconstruction_protocol_id=None,
):
    """Run EMD once with a pre-selected baseline configuration."""
    params = dict(emd_params)
    imfs, residual, all_components, config = run_emd_decomposition(
        Y=Y,
        T=t,
        nbsym=params.get("nbsym", 2),
        spline_kind=params.get("spline_kind", "cubic"),
        max_imf=params.get("max_imf", -1),
        std_thr=params.get("std_thr"),
        svar_thr=params.get("svar_thr"),
        total_power_thr=params.get("total_power_thr"),
        range_thr=params.get("range_thr"),
    )
    physical = evaluate_shared_physical_diagnostics(
        Y_observed=Y,
        X_clean=X_clean,
        imfs=imfs,
        residual=residual,
        fs=fs,
        residual_penalty_mode="none",
        true_components=true_components,
        contamination_mask=contamination_mask,
        reconstruction_protocol_id=reconstruction_protocol_id,
    )
    out = {
        "method": "EMD-fixed",
        "run_id": run_id,
        "imfs": imfs,
        "residual": residual,
        "all_components": all_components,
        "config": config,
        "theory_diagnostic_score": None,
        **params,
    }
    out.update(physical)
    return out


def run_fixed_pair_case(
        signal_name,
        noise_name,
        sigma,
        irmf_params,
        emd_params,
        n=500,
        fs=500.0,
        seed=0,
        noise_kwargs=None,
        target_snr_db=None,
        run_id_prefix="fixed",
):
    """Build one synthetic case and run fixed IRMF plus fixed EMD."""
    case = make_signal_noise_case(
        signal_name=signal_name,
        noise_name=noise_name,
        sigma=sigma,
        n=n,
        fs=fs,
        seed=seed,
        noise_kwargs=noise_kwargs,
        target_snr_db=target_snr_db,
    )
    irmf = run_fixed_irmf_case(
        Y=case["Y"],
        X_clean=case["X_clean"],
        t=case["t"],
        fs=fs,
        irmf_params=irmf_params,
        expected_noise_ratio=case.get("expected_noise_ratio"),
        true_components=case.get("true_components"),
        contamination_mask=case.get("contamination_mask"),
        run_id=f"{run_id_prefix}_irmf",
    )
    emd = run_fixed_emd_case(
        Y=case["Y"],
        X_clean=case["X_clean"],
        t=case["t"],
        fs=fs,
        emd_params=emd_params,
        true_components=case.get("true_components"),
        contamination_mask=case.get("contamination_mask"),
        run_id=f"{run_id_prefix}_emd",
    )
    metadata = {
        "signal": signal_name,
        "noise": noise_name,
        "sigma": sigma,
        "target_snr_db": target_snr_db,
        "noise_design": case.get("noise_design"),
        "noise_scale_alpha": case.get("noise_scale_alpha"),
        "realized_input_snr_db": case.get("realized_input_snr_db"),
        "seed": seed,
    }
    if noise_kwargs:
        metadata.update(noise_kwargs)
    return case, irmf, emd, paired_method_summary(irmf, emd, metadata)


def _run_fixed_eemd_or_ceemdan_case(
        method_name,
        case,
        fs,
        emd_params,
        eemd_params,
        ceemdan_params,
        algorithm_seed,
        run_id,
        reconstruction_protocol_id=None,
):
    method_name = str(method_name).upper()
    if method_name == "EEMD":
        params = dict(eemd_params)
        raw = run_eemd(
            case["Y"],
            max_imf=params.get("max_imf", emd_params.get("max_imf", -1)),
            trials=int(params.get("trials", 100)),
            noise_width=float(params.get("noise_width", 0.05)),
            parallel=bool(params.get("parallel", False)),
            random_seed=int(algorithm_seed),
        )
        method_params = params
    elif method_name == "CEEMDAN":
        params = dict(ceemdan_params)
        raw = run_ceemdan(
            case["Y"],
            max_imf=params.get("max_imf", emd_params.get("max_imf", -1)),
            trials=int(params.get("trials", 100)),
            epsilon=float(params.get("epsilon", 0.005)),
            parallel=bool(params.get("parallel", False)),
            random_seed=int(algorithm_seed),
        )
        method_params = params
    else:
        raise ValueError(f"Unsupported EMD-family method: {method_name}")

    physical = evaluate_shared_physical_diagnostics(
        Y_observed=case["Y"],
        X_clean=case["X_clean"],
        imfs=raw["imfs"],
        residual=raw["residual"],
        fs=fs,
        residual_penalty_mode="none",
        true_components=case.get("true_components"),
        contamination_mask=case.get("contamination_mask"),
        reconstruction_protocol_id=reconstruction_protocol_id,
    )
    return {
        "method": method_name,
        "run_id": run_id,
        "algorithm_seed": int(algorithm_seed),
        **method_params,
        **raw,
        **physical,
    }


def run_fixed_method_family_case(
        signal_name,
        noise_name,
        sigma,
        irmf_params,
        emd_params,
        eemd_params,
        ceemdan_params,
        n=500,
        fs=500.0,
        seed=0,
        algorithm_seed=20260715,
        noise_kwargs=None,
        target_snr_db=None,
        run_id_prefix="fixed_family",
        timeout_seconds=DEFAULT_METHOD_TIMEOUT_SECONDS,
        reconstruction_protocol_id=None,
):
    """Build one synthetic case and run fixed IRMF/EMD/EEMD/CEEMDAN."""
    case = make_signal_noise_case(
        signal_name=signal_name,
        noise_name=noise_name,
        sigma=sigma,
        n=n,
        fs=fs,
        seed=seed,
        noise_kwargs=noise_kwargs,
        target_snr_db=target_snr_db,
    )
    metadata = {
        "signal": signal_name,
        "noise": noise_name,
        "sigma": sigma,
        "target_snr_db": target_snr_db,
        "noise_design": case.get("noise_design"),
        "noise_scale_alpha": case.get("noise_scale_alpha"),
        "realized_input_snr_db": case.get("realized_input_snr_db"),
        "clean_signal_rms": case.get("clean_signal_rms"),
        "noise_rms": case.get("noise_rms"),
        "seed": seed,
        "data_seed": seed,
        "algorithm_seed": algorithm_seed,
        "algorithm_seed_base": algorithm_seed,
        "algorithm_seed_EEMD": int(algorithm_seed),
        "algorithm_seed_CEEMDAN": int(algorithm_seed) + 50000,
    }
    if noise_kwargs:
        metadata.update(noise_kwargs)

    row = dict(metadata)
    results = {}

    start = perf_counter()
    try:
        irmf = _run_with_timeout(
            lambda: run_fixed_irmf_case(
                Y=case["Y"],
                X_clean=case["X_clean"],
                t=case["t"],
                fs=fs,
                irmf_params=irmf_params,
                expected_noise_ratio=case.get("expected_noise_ratio"),
                true_components=case.get("true_components"),
                contamination_mask=case.get("contamination_mask"),
                run_id=f"{run_id_prefix}_irmf",
                reconstruction_protocol_id=reconstruction_protocol_id,
            ),
            timeout_seconds=timeout_seconds,
        )
        irmf = attach_failure_flags(
            irmf,
            Y_observed=case["Y"],
            X_clean=case["X_clean"],
            timeout_seconds=timeout_seconds,
            runtime_seconds=perf_counter() - start,
        )
    except Exception as exc:
        irmf = _failure_summary_for_exception("IRMF", exc, perf_counter() - start, timeout_seconds)
    results["IRMF"] = irmf
    row["IRMF"] = method_result_summary(irmf)

    start = perf_counter()
    try:
        emd = _run_with_timeout(
            lambda: run_fixed_emd_case(
                Y=case["Y"],
                X_clean=case["X_clean"],
                t=case["t"],
                fs=fs,
                emd_params=emd_params,
                true_components=case.get("true_components"),
                contamination_mask=case.get("contamination_mask"),
                run_id=f"{run_id_prefix}_emd",
                reconstruction_protocol_id=reconstruction_protocol_id,
            ),
            timeout_seconds=timeout_seconds,
        )
        emd = attach_failure_flags(
            emd,
            Y_observed=case["Y"],
            X_clean=case["X_clean"],
            timeout_seconds=timeout_seconds,
            runtime_seconds=perf_counter() - start,
        )
    except Exception as exc:
        emd = _failure_summary_for_exception("EMD", exc, perf_counter() - start, timeout_seconds)
    results["EMD"] = emd
    row["EMD"] = method_result_summary(emd)

    for method_name in ("EEMD", "CEEMDAN"):
        method_algorithm_seed = (
            int(algorithm_seed)
            if method_name == "EEMD"
            else int(algorithm_seed) + 50000
        )
        try:
            start = perf_counter()
            result = _run_with_timeout(
                lambda method_name=method_name: _run_fixed_eemd_or_ceemdan_case(
                    method_name=method_name,
                    case=case,
                    fs=fs,
                    emd_params=emd_params,
                    eemd_params=eemd_params,
                    ceemdan_params=ceemdan_params,
                    algorithm_seed=method_algorithm_seed,
                    run_id=f"{run_id_prefix}_{method_name.lower()}",
                    reconstruction_protocol_id=reconstruction_protocol_id,
                ),
                timeout_seconds=timeout_seconds,
            )
            result = attach_failure_flags(
                result,
                Y_observed=case["Y"],
                X_clean=case["X_clean"],
                timeout_seconds=timeout_seconds,
                runtime_seconds=perf_counter() - start,
            )
            results[method_name] = result
            row[method_name] = method_result_summary(result)
        except Exception as exc:
            result = _failure_summary_for_exception(method_name, exc, perf_counter() - start, timeout_seconds)
            results[method_name] = result
            row[method_name] = method_result_summary(result)

    for metric in [
        "case_score",
        "reconstruction_score",
        "structural_fidelity_score",
        "contamination_resistance_score",
        "denoise_nmse",
        "imf_recovery_score",
        "mode_mixing_index",
        "noise_capture_corr",
        "outlier_resistance_index",
    ]:
        iv = row.get("IRMF", {}).get(metric)
        for baseline in ("EMD", "EEMD", "CEEMDAN"):
            bv = row.get(baseline, {}).get(metric)
            try:
                if iv is not None and bv is not None and np.isfinite(iv) and np.isfinite(bv):
                    row[f"delta_{metric}_IRMF_minus_{baseline}"] = float(iv) - float(bv)
            except Exception:
                pass

    return case, results, row




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
,
        nbsym=None
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
    if nbsym is not None:
        grid["nbsym_options"] = [nbsym]

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
