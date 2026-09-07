#!/usr/bin/python
# coding: UTF-8

"""Controlled challenging-regime diagnostics.

This stage complements the full challenging benchmark.  It preserves the common
primary metric backbone for IRMF/EMD/EEMD/CEEMDAN and adds regime-specific
secondary diagnostics for mechanism interpretation only.  Regime-specific
diagnostics are not included in case_score and are not used to define an
alternative overall ranking.
"""

from time import perf_counter

import numpy as np

from project_config import (
    CONTROLLED_CHALLENGING_DIFFICULTY_LEVELS,
    CONTROLLED_CHALLENGING_NOISES,
    CONTROLLED_CHALLENGING_QUICK_NOISES,
    CONTROLLED_CHALLENGING_QUICK_SIGMAS,
    CONTROLLED_CHALLENGING_REGIMES,
    CONTROLLED_CHALLENGING_SIGMAS,
    DEFAULT_FS,
    DEFAULT_N,
    DEFAULT_SEED,
    GLOBAL_CEEMDAN_PARAMS,
    GLOBAL_EEMD_PARAMS,
    GLOBAL_EMD_PARAMS,
    GLOBAL_IRMF_PARAMS,
)
from diagnostics.shared_physical_diagnostics import reconstructed_signal
from experiments.experiment_emd_family_benchmark import _run_locked_emd_family_method
from experiments.experiment_utils import (
    method_result_summary,
    run_fixed_emd_case,
    run_fixed_irmf_case,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json
from noise_bank.noise_models import generate_noise


EPS = 1e-12
PRIMARY_METRICS = (
    "case_score",
    "reconstruction_score",
    "structural_fidelity_score",
    "contamination_resistance_score",
    "denoise_nmse",
    "denoise_corr",
    "imf_recovery_score",
    "component_splitting_index",
    "component_merging_index",
    "outlier_resistance_index",
    "noise_capture_corr",
)

REGIME_PRIMARY_DIAGNOSTICS = {
    "crossing_chirps": ("crossing_local_mixing_error", "identity_consistency"),
    "time_varying_close_frequencies": ("close_window_recovery", "resolution_failure_indicator"),
    "buried_weak_component": ("weak_component_nmse", "weak_component_detection"),
    "trend_plus_oscillation": ("trend_to_imf_leakage", "trend_rmse"),
    "damped_oscillation": ("late_component_recovery", "late_rmse"),
    "transient_train": ("event_recall", "event_timing_error"),
}


def _safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) == 0:
        return np.nan
    if np.std(a) < EPS or np.std(b) < EPS:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _nmse(a, b, mask=None):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        a = a[mask]
        b = b[mask]
    if len(a) == 0 or len(a) != len(b):
        return np.nan
    return float(np.sum((a - b) ** 2) / (np.sum(b ** 2) + EPS))


def _rmse(a, b, mask=None):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        a = a[mask]
        b = b[mask]
    if len(a) == 0 or len(a) != len(b):
        return np.nan
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _as_2d(imfs, n):
    imfs = np.asarray(imfs, dtype=float)
    if imfs.size == 0:
        return np.empty((0, n), dtype=float)
    if imfs.ndim == 1:
        imfs = imfs[None, :]
    return imfs


def _phase_from_frequency(t, freq):
    t = np.asarray(t, dtype=float)
    freq = np.asarray(freq, dtype=float)
    dt = 1.0 / max(len(t), 1)
    return 2.0 * np.pi * np.cumsum(freq) * dt


def _standardize_signal(x):
    x = np.asarray(x, dtype=float)
    s = np.std(x)
    if s < EPS:
        return x - np.mean(x)
    return (x - np.mean(x)) / s


def _best_match(imfs, true_component, mask=None):
    true_component = np.asarray(true_component, dtype=float)
    n = len(true_component)
    imfs = _as_2d(imfs, n)
    if imfs.shape[0] == 0:
        return {
            "index": None,
            "corr": np.nan,
            "abs_corr": np.nan,
            "nmse": np.nan,
            "energy_ratio": np.nan,
            "estimate": np.zeros(n, dtype=float),
        }

    best = None
    for idx, imf in enumerate(imfs):
        a = imf
        b = true_component
        if mask is not None:
            m = np.asarray(mask, dtype=bool)
            a = a[m]
            b = b[m]
        corr = _safe_corr(a, b)
        abs_corr = abs(corr) if np.isfinite(corr) else -np.inf
        sign = 1.0
        if np.isfinite(corr) and corr < 0:
            sign = -1.0
        estimate = sign * imf
        item = {
            "index": int(idx),
            "corr": float(corr) if np.isfinite(corr) else np.nan,
            "abs_corr": float(abs_corr) if np.isfinite(abs_corr) else np.nan,
            "nmse": _nmse(estimate, true_component, mask=mask),
            "energy_ratio": float(np.sum(estimate ** 2) / (np.sum(true_component ** 2) + EPS)),
            "estimate": estimate,
        }
        if best is None or item["abs_corr"] > best["abs_corr"]:
            best = item
    return best


def _component_recovery(imfs, true_components, mask=None):
    true_components = np.asarray(true_components, dtype=float)
    if true_components.ndim == 1:
        true_components = true_components[None, :]
    matches = [_best_match(imfs, comp, mask=mask) for comp in true_components]
    vals = [m["abs_corr"] for m in matches if np.isfinite(m["abs_corr"])]
    return float(np.mean(vals)) if vals else np.nan


def _scale_to_energy_ratio(component, reference, target_ratio):
    component = np.asarray(component, dtype=float)
    reference = np.asarray(reference, dtype=float)
    e_comp = np.sum(component ** 2) + EPS
    e_ref = np.sum(reference ** 2) + EPS
    return component * np.sqrt(float(target_ratio) * e_ref / e_comp)


def _controlled_case(regime, difficulty_value, noise_name, sigma, n, fs, seed):
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    meta = {
        "regime": regime,
        "difficulty_parameter_value": float(difficulty_value),
    }

    if regime == "crossing_chirps":
        slope_diff = float(difficulty_value)
        slope = slope_diff / 2.0
        f1 = 8.0 + slope * t
        f2 = 8.0 + slope * (1.0 - t)
        c1 = np.sin(_phase_from_frequency(t, f1))
        c2 = 0.85 * np.sin(_phase_from_frequency(t, f2) + 0.4)
        components = np.vstack([c1, c2])
        meta.update({
            "crossing_time": 0.5,
            "crossing_slope_difference": slope_diff,
            "crossing_window_width": 0.16,
            "true_frequencies": [f1, f2],
        })

    elif regime == "time_varying_close_frequencies":
        min_gap = float(difficulty_value)
        center = 13.0 + 1.2 * np.sin(2 * np.pi * t)
        gap = min_gap + 5.0 * (t - 0.5) ** 2
        f1 = center - 0.5 * gap
        f2 = center + 0.5 * gap
        c1 = np.sin(_phase_from_frequency(t, f1))
        c2 = 0.85 * np.sin(_phase_from_frequency(t, f2) + 0.7)
        components = np.vstack([c1, c2])
        meta.update({
            "min_frequency_gap": min_gap,
            "duration_below_resolution_threshold": float(np.mean(gap <= max(1.5 * min_gap, 0.5))),
            "close_window_width": 0.22,
            "frequency_gap_curve": gap,
            "true_frequencies": [f1, f2],
        })

    elif regime == "buried_weak_component":
        ratio = float(difficulty_value)
        strong = np.sin(2 * np.pi * 9.0 * t) + 0.35 * np.sin(2 * np.pi * 15.0 * t + 0.2)
        weak = ratio * np.sin(2 * np.pi * 24.0 * t + 1.1)
        components = np.vstack([strong, weak])
        meta.update({
            "weak_to_strong_amplitude_ratio": ratio,
            "weak_component_index": 1,
            "strong_component_index": 0,
            "frequency_separation": 15.0,
        })

    elif regime == "trend_plus_oscillation":
        target_ratio = float(difficulty_value)
        osc1 = np.sin(2 * np.pi * 10.0 * t)
        osc2 = 0.45 * np.sin(2 * np.pi * 18.0 * t + 0.5)
        oscillation = osc1 + osc2
        raw_trend = 0.8 * (t - 0.5) ** 2 + 0.15 * np.sin(2 * np.pi * 1.0 * t)
        trend = _scale_to_energy_ratio(raw_trend - np.mean(raw_trend), oscillation, target_ratio)
        components = np.vstack([trend, osc1, osc2])
        meta.update({
            "trend_to_oscillation_energy_ratio": target_ratio,
            "trend_component_index": 0,
            "trend_curvature_proxy": float(np.mean(np.abs(np.gradient(np.gradient(trend))))),
        })

    elif regime == "damped_oscillation":
        decay = float(difficulty_value)
        carrier = np.sin(2 * np.pi * 12.0 * t)
        damped = np.exp(-decay * t) * np.sin(2 * np.pi * 24.0 * t + 0.3)
        components = np.vstack([carrier, damped])
        meta.update({
            "decay_rate": decay,
            "damped_component_index": 1,
            "terminal_amplitude_ratio": float(np.exp(-decay)),
            "time_below_half_amplitude": float(np.mean(np.exp(-decay * t) < 0.5)),
        })

    elif regime == "transient_train":
        spacing = float(difficulty_value)
        base = 0.55 * np.sin(2 * np.pi * 8.0 * t)
        centers = np.array([0.18, 0.18 + spacing, 0.18 + 2.0 * spacing, 0.18 + 3.2 * spacing])
        centers = centers[centers < 0.88]
        amps = np.array([1.00, -0.75, 0.90, -0.60])[:len(centers)]
        widths = np.array([0.014, 0.020, 0.012, 0.017])[:len(centers)]
        freqs = np.array([26.0, 20.0, 30.0, 22.0])[:len(centers)]
        transient = np.zeros_like(t)
        for amp, center, width, freq in zip(amps, centers, widths, freqs):
            local = np.exp(-0.5 * ((t - center) / width) ** 2)
            transient += amp * local * np.cos(2 * np.pi * freq * (t - center))
        components = np.vstack([base, transient])
        meta.update({
            "minimum_event_spacing": float(np.min(np.diff(centers))) if len(centers) > 1 else np.nan,
            "event_count": int(len(centers)),
            "event_centers": centers.tolist(),
            "event_density": float(len(centers)),
            "event_width_range": [float(np.min(widths)), float(np.max(widths))],
        })

    else:
        raise ValueError(f"Unsupported controlled challenging regime: {regime}")

    x_clean = np.sum(components, axis=0)
    x_clean = _standardize_signal(x_clean)
    components = components / (np.std(np.sum(components, axis=0)) + EPS)
    noise = generate_noise(noise_name=noise_name, n=n, sigma=sigma, seed=seed)
    y = x_clean + noise
    expected_noise_ratio = float(np.sum(noise ** 2) / (np.sum(y ** 2) + EPS))
    return {
        "signal_name": regime,
        "noise_name": noise_name,
        "sigma": float(sigma),
        "t": t,
        "T": np.arange(n, dtype=float) / float(fs),
        "X_clean": x_clean,
        "Y": y,
        "noise": noise,
        "true_components": components,
        "expected_noise_ratio": expected_noise_ratio,
        "metadata": meta,
    }


def _identity_consistency(imfs, true_components, t):
    pre = t < 0.42
    post = t > 0.58
    if np.sum(pre) < 5 or np.sum(post) < 5:
        return np.nan
    pre_idx = [_best_match(imfs, comp, mask=pre)["index"] for comp in true_components]
    post_idx = [_best_match(imfs, comp, mask=post)["index"] for comp in true_components]
    valid = [(a, b) for a, b in zip(pre_idx, post_idx) if a is not None and b is not None]
    if not valid:
        return np.nan
    return float(np.mean([a == b for a, b in valid]))


def _event_diagnostics(rec, base, transient, t, centers, tolerance=0.025):
    estimated_event = np.asarray(rec, dtype=float) - np.asarray(base, dtype=float)
    centers = np.asarray(centers, dtype=float)
    if len(centers) == 0:
        return {
            "event_recall": np.nan,
            "event_timing_error": np.nan,
            "event_amplitude_bias": np.nan,
            "false_event_rate": np.nan,
            "event_energy_outside_neighborhood": np.nan,
        }
    true_amp = []
    est_amp = []
    timing_errors = []
    detected = []
    event_mask = np.zeros(len(t), dtype=bool)
    for center in centers:
        m = np.abs(t - center) <= tolerance
        event_mask |= m
        if not np.any(m):
            continue
        true_local = transient[m]
        est_local = estimated_event[m]
        true_peak = float(np.max(np.abs(true_local)) + EPS)
        local_idx = int(np.argmax(np.abs(est_local)))
        est_peak = float(np.max(np.abs(est_local)))
        detected.append(est_peak >= 0.30 * true_peak)
        true_amp.append(true_peak)
        est_amp.append(est_peak)
        timing_errors.append(float(abs(t[m][local_idx] - center)))
    outside_energy = float(np.sum(estimated_event[~event_mask] ** 2) / (np.sum(estimated_event ** 2) + EPS))
    return {
        "event_recall": float(np.mean(detected)) if detected else np.nan,
        "event_timing_error": float(np.mean(timing_errors)) if timing_errors else np.nan,
        "event_amplitude_bias": float(np.mean(np.asarray(est_amp) - np.asarray(true_amp))) if true_amp else np.nan,
        "false_event_rate": outside_energy,
        "event_energy_outside_neighborhood": outside_energy,
    }


def _regime_diagnostics(regime, case, result):
    t = case["t"]
    x_clean = case["X_clean"]
    components = np.asarray(case["true_components"], dtype=float)
    meta = case.get("metadata", {})
    imfs = result.get("imfs", [])
    residual = result.get("residual", np.zeros_like(x_clean))
    rec = reconstructed_signal(case["Y"], imfs, residual)
    if rec is None:
        rec = np.zeros_like(x_clean)

    out = {}
    if regime == "crossing_chirps":
        width = float(meta.get("crossing_window_width", 0.16))
        center = float(meta.get("crossing_time", 0.5))
        mask = np.abs(t - center) <= 0.5 * width
        out["crossing_window_nmse"] = _nmse(rec, x_clean, mask=mask)
        out["crossing_window_component_recovery"] = _component_recovery(imfs, components, mask=mask)
        out["crossing_local_mixing_error"] = out["crossing_window_nmse"]
        out["identity_consistency"] = _identity_consistency(imfs, components, t)
        out["regime_failure_indicator"] = float(
            (out["crossing_window_component_recovery"] < 0.70)
            or (out["identity_consistency"] < 0.50)
        )

    elif regime == "time_varying_close_frequencies":
        gap = np.asarray(meta.get("frequency_gap_curve", np.ones_like(t)), dtype=float)
        min_gap = float(meta.get("min_frequency_gap", np.min(gap)))
        mask = gap <= max(1.5 * min_gap, min_gap + 0.15)
        recovery = _component_recovery(imfs, components, mask=mask)
        out["close_window_recovery"] = recovery
        out["close_window_nmse"] = _nmse(rec, x_clean, mask=mask)
        out["resolution_failure_indicator"] = float((not np.isfinite(recovery)) or recovery < 0.70)
        out["component_separation_rate"] = float(1.0 - out["resolution_failure_indicator"])
        out["regime_failure_indicator"] = out["resolution_failure_indicator"]

    elif regime == "buried_weak_component":
        weak = components[int(meta.get("weak_component_index", 1))]
        strong = components[int(meta.get("strong_component_index", 0))]
        match = _best_match(imfs, weak)
        out["weak_component_corr"] = match["abs_corr"]
        out["weak_component_nmse"] = match["nmse"]
        out["weak_component_energy_ratio"] = match["energy_ratio"]
        out["weak_component_detection"] = float(
            np.isfinite(match["abs_corr"])
            and match["abs_corr"] >= 0.70
            and 0.25 <= match["energy_ratio"] <= 4.00
        )
        out["missed_weak_component"] = float(1.0 - out["weak_component_detection"])
        out["strong_to_weak_leakage"] = abs(_safe_corr(match["estimate"], strong))
        out["regime_failure_indicator"] = out["missed_weak_component"]

    elif regime == "trend_plus_oscillation":
        trend = components[int(meta.get("trend_component_index", 0))]
        residual = np.asarray(residual, dtype=float)
        if len(residual) != len(trend):
            residual = np.zeros_like(trend)
        imfs_sum = np.sum(_as_2d(imfs, len(trend)), axis=0)
        out["trend_rmse"] = _rmse(residual, trend)
        out["trend_nmse"] = _nmse(residual, trend)
        projection = np.dot(imfs_sum, trend) / (np.sum(trend ** 2) + EPS)
        leaked = projection * trend
        out["trend_to_imf_leakage"] = float(np.sum(leaked ** 2) / (np.sum(trend ** 2) + EPS))
        out["residual_trend_corr"] = abs(_safe_corr(residual, trend))
        out["oscillation_recovery"] = _component_recovery(imfs, components[1:])
        out["regime_failure_indicator"] = float(out["trend_to_imf_leakage"] > 0.50)

    elif regime == "damped_oscillation":
        damped = components[int(meta.get("damped_component_index", 1))]
        early = t < 0.33
        middle = (t >= 0.33) & (t < 0.66)
        late = t >= 0.66
        match_late = _best_match(imfs, damped, mask=late)
        out["early_rmse"] = _rmse(rec, x_clean, mask=early)
        out["middle_rmse"] = _rmse(rec, x_clean, mask=middle)
        out["late_rmse"] = _rmse(rec, x_clean, mask=late)
        out["late_component_recovery"] = match_late["abs_corr"]
        out["amplitude_decay_tracking"] = abs(_safe_corr(np.abs(match_late["estimate"]), np.abs(damped)))
        out["regime_failure_indicator"] = float(
            (not np.isfinite(out["late_component_recovery"]))
            or out["late_component_recovery"] < 0.70
        )

    elif regime == "transient_train":
        base = components[0]
        transient = components[1]
        out.update(_event_diagnostics(rec, base, transient, t, meta.get("event_centers", [])))
        out["transient_smearing_index_secondary"] = out["event_energy_outside_neighborhood"]
        out["regime_failure_indicator"] = float(
            (not np.isfinite(out["event_recall"])) or out["event_recall"] < 0.80
        )

    return out


def _run_method(method, case, fs, irmf_params, emd_params, eemd_params, ceemdan_params, algorithm_seed, run_id):
    start = perf_counter()
    if method == "IRMF":
        result = run_fixed_irmf_case(
            Y=case["Y"],
            X_clean=case["X_clean"],
            t=case["t"],
            fs=fs,
            irmf_params=irmf_params,
            expected_noise_ratio=case.get("expected_noise_ratio"),
            true_components=case.get("true_components"),
            contamination_mask=case.get("contamination_mask"),
            run_id=run_id,
        )
    elif method == "EMD":
        result = run_fixed_emd_case(
            Y=case["Y"],
            X_clean=case["X_clean"],
            t=case["t"],
            fs=fs,
            emd_params=emd_params,
            true_components=case.get("true_components"),
            contamination_mask=case.get("contamination_mask"),
            run_id=run_id,
        )
    else:
        result = _run_locked_emd_family_method(
            method,
            case,
            fs=fs,
            emd_params=emd_params,
            eemd_params=eemd_params,
            ceemdan_params=ceemdan_params,
            algorithm_seed=algorithm_seed,
        )
        result["run_id"] = run_id
    result["runtime_seconds"] = perf_counter() - start
    result["method"] = method
    return result


def _difficulty_specs(quick=False):
    for regime in CONTROLLED_CHALLENGING_REGIMES:
        spec = CONTROLLED_CHALLENGING_DIFFICULTY_LEVELS[regime]
        values = spec["quick_values"] if quick else spec["values"]
        for order, value in enumerate(values):
            yield {
                "regime": regime,
                "difficulty_parameter_name": spec["name"],
                "difficulty_parameter_value": float(value),
                "difficulty_direction": spec["direction"],
                "difficulty_order": int(order),
            }


def _aggregate(rows):
    groups = {}
    metric_keys = set(PRIMARY_METRICS)
    for row in rows:
        metric_keys.update(k for k in row.keys() if k.startswith("diagnostic_"))
    for row in rows:
        key = (
            row["regime"],
            row["difficulty_parameter_name"],
            row["difficulty_parameter_value"],
            row["noise"],
            row["sigma"],
            row["method"],
        )
        groups.setdefault(key, []).append(row)

    summary = []
    for key, subset in sorted(groups.items(), key=lambda item: str(item[0])):
        regime, dname, dvalue, noise, sigma, method = key
        item = {
            "regime": regime,
            "difficulty_parameter_name": dname,
            "difficulty_parameter_value": float(dvalue),
            "noise": noise,
            "sigma": float(sigma),
            "method": method,
            "n_runs": int(len(subset)),
        }
        for metric in sorted(metric_keys):
            vals = []
            for row in subset:
                value = row.get(metric)
                try:
                    if value is not None and np.isfinite(value):
                        vals.append(float(value))
                except Exception:
                    continue
            if vals:
                item[f"{metric}_mean"] = float(np.mean(vals))
                item[f"{metric}_median"] = float(np.median(vals))
        summary.append(item)
    return summary


def _failure_thresholds(summary):
    thresholds = []
    by_group = {}
    for row in summary:
        key = (row["regime"], row["method"], row["noise"], row["sigma"])
        by_group.setdefault(key, []).append(row)

    rules = {
        "crossing_chirps": "diagnostic_regime_failure_indicator_mean >= 0.5",
        "time_varying_close_frequencies": "diagnostic_regime_failure_indicator_mean >= 0.5",
        "buried_weak_component": "diagnostic_weak_component_detection_mean < 0.8",
        "trend_plus_oscillation": "diagnostic_trend_to_imf_leakage_mean > 0.5",
        "damped_oscillation": "diagnostic_late_component_recovery_mean < 0.7",
        "transient_train": "diagnostic_event_recall_mean < 0.8",
    }
    for key, rows in sorted(by_group.items(), key=lambda item: str(item[0])):
        regime, method, noise, sigma = key
        rows = sorted(rows, key=lambda r: r["difficulty_parameter_value"])
        spec = CONTROLLED_CHALLENGING_DIFFICULTY_LEVELS[regime]
        if spec["direction"] == "higher_is_harder":
            rows = sorted(rows, key=lambda r: r["difficulty_parameter_value"])
        else:
            rows = sorted(rows, key=lambda r: r["difficulty_parameter_value"], reverse=True)
        failed = None
        for row in rows:
            if _row_fails(regime, row):
                failed = row
                break
        thresholds.append({
            "regime": regime,
            "method": method,
            "noise": noise,
            "sigma": float(sigma),
            "difficulty_parameter_name": spec["name"],
            "difficulty_direction": spec["direction"],
            "failure_rule": rules.get(regime),
            "estimated_failure_threshold": (
                None if failed is None else float(failed["difficulty_parameter_value"])
            ),
            "failure_observed": failed is not None,
        })
    return thresholds


def _row_fails(regime, row):
    if regime == "buried_weak_component":
        return row.get("diagnostic_weak_component_detection_mean", 1.0) < 0.8
    if regime == "trend_plus_oscillation":
        return row.get("diagnostic_trend_to_imf_leakage_mean", 0.0) > 0.5
    if regime == "damped_oscillation":
        return row.get("diagnostic_late_component_recovery_mean", 1.0) < 0.7
    if regime == "transient_train":
        return row.get("diagnostic_event_recall_mean", 1.0) < 0.8
    return row.get("diagnostic_regime_failure_indicator_mean", 0.0) >= 0.5


def run_controlled_challenging_diagnostics(
        output_root,
        irmf_params=GLOBAL_IRMF_PARAMS,
        emd_params=GLOBAL_EMD_PARAMS,
        eemd_params=GLOBAL_EEMD_PARAMS,
        ceemdan_params=GLOBAL_CEEMDAN_PARAMS,
        noises=CONTROLLED_CHALLENGING_NOISES,
        sigmas=CONTROLLED_CHALLENGING_SIGMAS,
        n=DEFAULT_N,
        fs=DEFAULT_FS,
        data_seed=DEFAULT_SEED,
        algorithm_seed=20260715,
        quick=False,
):
    """Run controlled regime diagnostics with common primary metrics."""
    output_root = ensure_dir(output_root)
    if quick:
        noises = CONTROLLED_CHALLENGING_QUICK_NOISES
        sigmas = CONTROLLED_CHALLENGING_QUICK_SIGMAS

    rows = []
    for spec in _difficulty_specs(quick=quick):
        for noise_name in noises:
            for sigma in sigmas:
                case = _controlled_case(
                    regime=spec["regime"],
                    difficulty_value=spec["difficulty_parameter_value"],
                    noise_name=noise_name,
                    sigma=sigma,
                    n=n,
                    fs=fs,
                    seed=data_seed,
                )
                case.update({"metadata": {**case["metadata"], **spec}})
                for method in ("IRMF", "EMD", "EEMD", "CEEMDAN"):
                    run_id = (
                        f"controlled_{spec['regime']}_"
                        f"{spec['difficulty_parameter_name']}_{spec['difficulty_parameter_value']}_"
                        f"{noise_name}_{sigma}_{method.lower()}"
                    )
                    try:
                        result = _run_method(
                            method,
                            case,
                            fs,
                            irmf_params,
                            emd_params,
                            eemd_params,
                            ceemdan_params,
                            algorithm_seed,
                            run_id,
                        )
                        summary = method_result_summary(result)
                        diagnostics = _regime_diagnostics(spec["regime"], case, result)
                        row = {
                            "section": "controlled_challenging_regime_diagnostics",
                            "benchmark_tier": "Tier 3b",
                            "analysis_role": "secondary_mechanism_diagnostics",
                            "regime": spec["regime"],
                            "method": method,
                            "noise": noise_name,
                            "sigma": float(sigma),
                            "data_seed": int(data_seed),
                            "algorithm_seed": int(algorithm_seed),
                            "quick": bool(quick),
                            "difficulty_parameter_name": spec["difficulty_parameter_name"],
                            "difficulty_parameter_value": spec["difficulty_parameter_value"],
                            "difficulty_direction": spec["difficulty_direction"],
                            "difficulty_order": spec["difficulty_order"],
                            "primary_metrics_use_common_evaluation_operator": True,
                            "regime_specific_metrics_enter_case_score": False,
                        }
                        for metric in PRIMARY_METRICS:
                            if metric in summary:
                                row[metric] = summary[metric]
                        for key, value in case.get("metadata", {}).items():
                            if key in row or key in {"true_frequencies", "frequency_gap_curve"}:
                                continue
                            row[f"difficulty_{key}"] = value
                        for key, value in diagnostics.items():
                            row[f"diagnostic_{key}"] = value
                        row["runtime_seconds"] = summary.get("runtime_seconds")
                    except Exception as exc:
                        row = {
                            "section": "controlled_challenging_regime_diagnostics",
                            "benchmark_tier": "Tier 3b",
                            "analysis_role": "secondary_mechanism_diagnostics",
                            "regime": spec["regime"],
                            "method": method,
                            "noise": noise_name,
                            "sigma": float(sigma),
                            "data_seed": int(data_seed),
                            "algorithm_seed": int(algorithm_seed),
                            "quick": bool(quick),
                            "difficulty_parameter_name": spec["difficulty_parameter_name"],
                            "difficulty_parameter_value": spec["difficulty_parameter_value"],
                            "difficulty_direction": spec["difficulty_direction"],
                            "difficulty_order": spec["difficulty_order"],
                            "error": str(exc),
                        }
                    rows.append(row)
                print(
                    "CONTROLLED CHALLENGING DONE | "
                    f"regime={spec['regime']} | "
                    f"{spec['difficulty_parameter_name']}={spec['difficulty_parameter_value']} | "
                    f"noise={noise_name} | sigma={sigma}",
                    flush=True,
                )

    summary = _aggregate(rows)
    thresholds = _failure_thresholds(summary)
    protocol = {
        "section": "Controlled Challenging-Regime Diagnostics",
        "benchmark_tier": "Tier 3b",
        "purpose": (
            "Estimate degradation patterns and approximate failure thresholds "
            "under controlled structural stress regimes."
        ),
        "primary_metric_policy": {
            "primary_metrics": list(PRIMARY_METRICS),
            "common_evaluation_operator": True,
            "used_for_cross_method_comparison_and_ranking": True,
        },
        "regime_specific_metric_policy": {
            "used_for_mechanism_interpretation_only": True,
            "enter_case_score": False,
            "form_alternative_global_ranking": False,
            "diagnostics_by_regime": REGIME_PRIMARY_DIAGNOSTICS,
        },
        "difficulty_parameter_policy": {
            "used_as_control_variable_or_plot_axis": True,
            "not_a_performance_metric": True,
            "levels": CONTROLLED_CHALLENGING_DIFFICULTY_LEVELS,
        },
        "design": {
            "regimes": list(CONTROLLED_CHALLENGING_REGIMES),
            "noises": list(noises),
            "sigmas": list(sigmas),
            "methods": ["IRMF", "EMD", "EEMD", "CEEMDAN"],
            "n_method_runs": int(len(rows)),
            "quick": bool(quick),
        },
        "interpretation_note": (
            "This stage should be reported as common primary metrics plus "
            "targeted secondary diagnostics.  Principal conclusions and aggregate "
            "rankings should remain based on the common primary metrics."
        ),
    }
    write_json(protocol, output_root / "controlled_challenging_diagnostics_protocol.json")
    write_json(rows, output_root / "controlled_challenging_diagnostic_rows.json")
    write_csv(rows, output_root / "controlled_challenging_diagnostic_rows.csv")
    write_json(summary, output_root / "controlled_challenging_diagnostic_summary.json")
    write_csv(summary, output_root / "controlled_challenging_diagnostic_summary.csv")
    write_json(thresholds, output_root / "controlled_challenging_failure_thresholds.json")
    write_csv(thresholds, output_root / "controlled_challenging_failure_thresholds.csv")
    return {
        "rows": rows,
        "summary": summary,
        "failure_thresholds": thresholds,
        "protocol": protocol,
    }


if __name__ == "__main__":
    run_controlled_challenging_diagnostics(
        "IRMF_EMD_PAPER_RESULTS/controlled_challenging_diagnostics",
        quick=True,
    )
