#!/usr/bin/python
# coding: UTF-8

"""V5.18 unified nested factorial benchmark cube.

The cube is the single source of truth for the canonical and challenging
fixed-parameter benchmark:

    Method x SignalRegime x SignalClass(SignalRegime) x Noise x Sigma x Seed

Paper sections are generated as slices of this cube rather than as independent
experiments.
"""

from collections import defaultdict
import json
import numpy as np

from project_config import (
    DEFAULT_FS,
    DEFAULT_METHOD_TIMEOUT_SECONDS,
    DEFAULT_N,
    EVALUATION_METHODS,
    FULL_SIGMA_LEVELS,
    GLOBAL_CEEMDAN_PARAMS,
    GLOBAL_EEMD_PARAMS,
    GLOBAL_EMD_PARAMS,
    GLOBAL_IRMF_PARAMS,
    LOWER_IS_BETTER_METRICS,
    TARGET_SNR_DB_LEVELS,
    UNIFIED_BENCHMARK_QUICK_SEEDS,
    UNIFIED_BENCHMARK_REGIMES,
    UNIFIED_BENCHMARK_SEEDS,
)
from experiments.experiment_utils import aggregate_method_family_rows, run_fixed_method_family_case
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


METHODS = tuple(EVALUATION_METHODS)
LOWER_IS_BETTER = set(LOWER_IS_BETTER_METRICS)
TRAJECTORY_METRICS = (
    "denoise_nmse",
    "denoise_corr",
    "imf_recovery_corr",
    "imf_recovery_nrmse",
    "component_splitting_index",
    "component_merging_index",
    "inter_imf_entanglement_index",
    "mode_mixing_index",
    "decomposition_count_error",
    "noise_capture_corr",
    "outlier_resistance_index",
    "clean_region_nmse",
    "signal_leakage_into_noise",
)
MONTE_CARLO_SUMMARY_METRICS = (
    "denoise_nmse",
    "denoise_corr",
    "imf_recovery_corr",
    "imf_recovery_nrmse",
    "component_splitting_index",
    "component_merging_index",
    "inter_imf_entanglement_index",
    "mode_mixing_index",
    "decomposition_count_error",
    "noise_capture_corr",
    "outlier_resistance_index",
    "clean_region_nmse",
    "signal_leakage_into_noise",
    "runtime_seconds",
)


def _finite(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return None


def _quick_regimes():
    out = {}
    for regime, cfg in UNIFIED_BENCHMARK_REGIMES.items():
        out[regime] = {
            "signals": tuple(cfg["signals"][:1]),
            "noises": tuple(cfg["noises"][:2]),
            "sigmas": (0.05, 0.20, 0.40),
            "paper_slices": cfg.get("paper_slices", {}),
        }
    return out


def _slice_rows(rows, regime=None, sigma_min=None, sigma_max=None):
    out = []
    for row in rows:
        if regime is not None and row.get("signal_regime") != regime:
            continue
        sigma = _finite(row.get("sigma"))
        if sigma is None:
            continue
        if sigma_min is not None and sigma < float(sigma_min) - 1e-12:
            continue
        if sigma_max is not None and sigma > float(sigma_max) + 1e-12:
            continue
        out.append(row)
    return out


def _write_rows_and_summary(rows, output_root, stem, extra_protocol=None):
    write_json(rows, output_root / f"{stem}.json")
    write_csv(rows, output_root / f"{stem}.csv")
    summary = aggregate_method_family_rows(rows)
    write_json(summary, output_root / f"{stem}_aggregate.json")
    if extra_protocol is not None:
        write_json(extra_protocol, output_root / f"{stem}_protocol.json")
    return summary


def _checkpoint_key(regime_name, signal_name, noise_name, sigma, data_seed):
    return f"{regime_name}|{signal_name}|{noise_name}|{float(sigma):.12g}|{int(data_seed)}"


def _checkpoint_key_with_snr(regime_name, signal_name, noise_name, sigma, target_snr_db, data_seed):
    if target_snr_db is None:
        return _checkpoint_key(regime_name, signal_name, noise_name, sigma, data_seed)
    return (
        f"{regime_name}|{signal_name}|{noise_name}|"
        f"sigma_label_{float(sigma):.12g}|snr_{float(target_snr_db):.12g}|{int(data_seed)}"
    )


def _load_checkpoint_rows(path):
    rows = []
    path = ensure_dir(path.parent) / path.name
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except Exception:
                continue
    return rows


def _append_checkpoint_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _algorithm_seed_for(data_seed):
    return 100000 + int(data_seed)


def _trajectory_groups(rows):
    grouped = defaultdict(list)
    for row in rows:
        sigma = _finite(row.get("sigma"))
        if sigma is None:
            continue
        for method in METHODS:
            data = row.get(method, {})
            if not isinstance(data, dict):
                continue
            for metric in TRAJECTORY_METRICS:
                value = _finite(data.get(metric))
                if value is not None:
                    key = (
                        row.get("signal_regime"),
                        row.get("signal"),
                        row.get("noise"),
                        row.get("seed", row.get("data_seed")),
                        method,
                        metric,
                    )
                    grouped[key].append((sigma, value))
    return grouped


def robustness_auc_rows(rows, expected_sigmas=FULL_SIGMA_LEVELS):
    expected = {round(float(s), 12) for s in expected_sigmas}
    out = []
    for key, points in sorted(_trajectory_groups(rows).items()):
        if len(points) < 2:
            continue
        dedup = {}
        for sigma, value in points:
            dedup[float(sigma)] = float(value)
        x = np.asarray(sorted(dedup), dtype=float)
        y = np.asarray([dedup[v] for v in x], dtype=float)
        if x.size < 2 or float(np.max(x) - np.min(x)) <= 1e-12:
            continue
        metric = key[-1]
        lower = metric in LOWER_IS_BETTER
        auc = float(np.trapz(y, x) / (np.max(x) - np.min(x)))
        baseline = float(y[0])
        terminal = float(y[-1])
        raw_change = terminal - baseline
        absolute_degradation = raw_change if lower else -raw_change
        relative_degradation = absolute_degradation / (abs(baseline) + 1e-12)
        slope = float(np.polyfit(x, y, deg=1)[0])
        degradation_slope = slope if lower else -slope
        curvature = None
        if x.size >= 3:
            curvature = float(np.polyfit(x, y, deg=2)[0])
        observed = {round(float(v), 12) for v in x}
        out.append({
            "signal_regime": key[0],
            "signal": key[1],
            "noise": key[2],
            "seed": key[3],
            "method": key[4],
            "metric": metric,
            "higher_is_better": not lower,
            "n_sigma_levels": int(x.size),
            "sigma_min": float(np.min(x)),
            "sigma_max": float(np.max(x)),
            "complete_sigma_trajectory": bool(observed == expected),
            "normalized_robustness_auc": auc,
            "baseline_value_at_sigma_min": baseline,
            "terminal_value_at_sigma_max": terminal,
            "absolute_degradation": float(absolute_degradation),
            "relative_degradation": float(relative_degradation),
            "linear_slope_per_sigma": slope,
            "degradation_slope_per_sigma": float(degradation_slope),
            "quadratic_curvature": curvature,
            "direction": "AUC follows original metric direction; positive degradation means worse as sigma increases",
        })
    return out


def aggregate_robustness_auc(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["signal_regime"], row["method"], row["metric"])].append(row)
    out = []
    for (regime, method, metric), vals in sorted(grouped.items()):
        auc = np.asarray([v["normalized_robustness_auc"] for v in vals], dtype=float)
        degrade = np.asarray([v["absolute_degradation"] for v in vals], dtype=float)
        slope = np.asarray([v["degradation_slope_per_sigma"] for v in vals], dtype=float)
        complete = np.asarray([bool(v["complete_sigma_trajectory"]) for v in vals], dtype=float)
        out.append({
            "signal_regime": regime,
            "method": method,
            "metric": metric,
            "n_trajectories": int(len(vals)),
            "mean_normalized_robustness_auc": float(np.mean(auc)),
            "median_normalized_robustness_auc": float(np.median(auc)),
            "mean_absolute_degradation": float(np.mean(degrade)),
            "median_absolute_degradation": float(np.median(degrade)),
            "mean_degradation_slope_per_sigma": float(np.mean(slope)),
            "median_degradation_slope_per_sigma": float(np.median(slope)),
            "complete_trajectory_fraction": float(np.mean(complete)),
        })
    return out


def failure_summary_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        for method in METHODS:
            data = row.get(method, {})
            if not isinstance(data, dict):
                continue
            key = (row.get("signal_regime"), row.get("signal"), row.get("noise"), row.get("sigma"), method)
            grouped[key].append(data)
    out = []
    for key, vals in sorted(grouped.items(), key=lambda item: str(item[0])):
        out.append({
            "signal_regime": key[0],
            "signal": key[1],
            "noise": key[2],
            "sigma": key[3],
            "method": key[4],
            "n_runs": int(len(vals)),
            "computational_failure_rate": float(np.mean([bool(v.get("computational_failure")) for v in vals])),
            "structural_failure_rate": float(np.mean([bool(v.get("structural_failure")) for v in vals])),
            "denoising_failure_rate": float(np.mean([bool(v.get("denoising_failure")) for v in vals])),
            "any_failure_rate": float(np.mean([bool(v.get("any_failure")) for v in vals])),
            "timeout_rate": float(np.mean([bool(v.get("timeout_flag")) for v in vals])),
            "nonfinite_output_rate": float(np.mean([bool(v.get("nonfinite_output_flag")) for v in vals])),
            "median_runtime_seconds": float(np.median([
                _finite(v.get("runtime_seconds")) for v in vals if _finite(v.get("runtime_seconds")) is not None
            ])) if any(_finite(v.get("runtime_seconds")) is not None for v in vals) else None,
        })
    return out


def _metric_values_by_method(rows, metrics=MONTE_CARLO_SUMMARY_METRICS):
    grouped = defaultdict(list)
    for row in rows:
        cell = (
            row.get("signal_regime"),
            row.get("signal"),
            row.get("noise"),
            row.get("sigma"),
            row.get("method") if "method" in row else None,
        )
        for method in METHODS:
            data = row.get(method, {})
            if not isinstance(data, dict):
                continue
            key_base = (row.get("signal_regime"), row.get("signal"), row.get("noise"), row.get("sigma"), method)
            for metric in metrics:
                value = _finite(data.get(metric))
                if value is not None:
                    grouped[(*key_base, metric)].append({
                        "value": float(value),
                        "data_seed": row.get("data_seed", row.get("seed")),
                        "numerically_invalid_or_computational_failure": bool(
                            data.get("computational_failure")
                            or data.get("timeout_flag")
                            or data.get("nonfinite_output_flag")
                            or data.get("exception_flag")
                        ),
                    })
    return grouped


def monte_carlo_cell_summary_rows(rows, metrics=MONTE_CARLO_SUMMARY_METRICS):
    grouped = _metric_values_by_method(rows, metrics=metrics)
    out = []
    for key, vals in sorted(grouped.items(), key=lambda item: str(item[0])):
        values = np.asarray([v["value"] for v in vals], dtype=float)
        n = int(values.size)
        sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
        se = float(sd / np.sqrt(n)) if n > 0 else None
        ci = 1.96 * se if se is not None else None
        out.append({
            "signal_regime": key[0],
            "signal": key[1],
            "noise": key[2],
            "sigma": key[3],
            "method": key[4],
            "metric": key[5],
            "n_success": n,
            "mc_mean": float(np.mean(values)) if n else None,
            "mc_sd": sd,
            "mc_se": se,
            "ci95_lower": float(np.mean(values) - ci) if ci is not None else None,
            "ci95_upper": float(np.mean(values) + ci) if ci is not None else None,
            "median": float(np.median(values)) if n else None,
            "q25": float(np.quantile(values, 0.25)) if n else None,
            "q75": float(np.quantile(values, 0.75)) if n else None,
            "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)) if n else None,
            "mad": float(np.median(np.abs(values - np.median(values)))) if n else None,
            "min": float(np.min(values)) if n else None,
            "max": float(np.max(values)) if n else None,
            "n_numerically_invalid_or_computational_failure": int(
                sum(bool(v["numerically_invalid_or_computational_failure"]) for v in vals)
            ),
            "note": (
                "Monte Carlo summary over finite metric values; computational, "
                "timeout, exception, and non-finite failures are reported separately. "
                "Denoising/structural failure criteria are not silently dropped."
            ),
        })
    return out


def paired_method_difference_summary_rows(rows, metrics=MONTE_CARLO_SUMMARY_METRICS):
    pairs = []
    for i, a in enumerate(METHODS):
        for b in METHODS[i + 1:]:
            pairs.append((a, b))
    out = []
    diffs = defaultdict(list)
    for row in rows:
        for a, b in pairs:
            da = row.get(a, {})
            db = row.get(b, {})
            if not isinstance(da, dict) or not isinstance(db, dict):
                continue
            for metric in metrics:
                va = _finite(da.get(metric))
                vb = _finite(db.get(metric))
                if va is None or vb is None:
                    continue
                key = (
                    row.get("signal_regime"), row.get("signal"), row.get("noise"),
                    row.get("sigma"), a, b, metric,
                )
                diffs[key].append(float(va) - float(vb))
    for key, vals in sorted(diffs.items(), key=lambda item: str(item[0])):
        values = np.asarray(vals, dtype=float)
        n = int(values.size)
        sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
        se = float(sd / np.sqrt(n)) if n > 0 else None
        ci = 1.96 * se if se is not None else None
        out.append({
            "signal_regime": key[0],
            "signal": key[1],
            "noise": key[2],
            "sigma": key[3],
            "method_a": key[4],
            "method_b": key[5],
            "metric": key[6],
            "difference_definition": "method_a_minus_method_b",
            "n_paired_seeds": n,
            "mean_paired_difference": float(np.mean(values)) if n else None,
            "sd_paired_difference": sd,
            "se_paired_difference": se,
            "ci95_lower": float(np.mean(values) - ci) if ci is not None else None,
            "ci95_upper": float(np.mean(values) + ci) if ci is not None else None,
            "median_paired_difference": float(np.median(values)) if n else None,
            "q25": float(np.quantile(values, 0.25)) if n else None,
            "q75": float(np.quantile(values, 0.75)) if n else None,
        })
    return out


def method_win_tie_loss_rows(rows, metrics=MONTE_CARLO_SUMMARY_METRICS, tie_tol=1e-12):
    out = []
    grouped = defaultdict(list)
    for row in rows:
        key = (row.get("signal_regime"), row.get("signal"), row.get("noise"), row.get("sigma"))
        grouped[key].append(row)
    for cell, vals in sorted(grouped.items(), key=lambda item: str(item[0])):
        for metric in metrics:
            lower = metric in LOWER_IS_BETTER
            counts = {m: {"wins": 0, "ties": 0, "losses": 0, "n_compared_seeds": 0} for m in METHODS}
            for row in vals:
                method_values = {
                    method: _finite(row.get(method, {}).get(metric))
                    for method in METHODS
                    if isinstance(row.get(method, {}), dict)
                }
                method_values = {m: v for m, v in method_values.items() if v is not None}
                if len(method_values) < 2:
                    continue
                best = min(method_values.values()) if lower else max(method_values.values())
                for method, value in method_values.items():
                    counts[method]["n_compared_seeds"] += 1
                    if abs(value - best) <= tie_tol:
                        counts[method]["ties"] += 1
                    else:
                        counts[method]["losses"] += 1
                winners = [m for m, v in method_values.items() if abs(v - best) <= tie_tol]
                if len(winners) == 1:
                    counts[winners[0]]["wins"] += 1
                    counts[winners[0]]["ties"] -= 1
            for method, c in counts.items():
                n = c["n_compared_seeds"]
                out.append({
                    "signal_regime": cell[0],
                    "signal": cell[1],
                    "noise": cell[2],
                    "sigma": cell[3],
                    "metric": metric,
                    "method": method,
                    "higher_is_better": not lower,
                    "n_compared_seeds": int(n),
                    "wins": int(c["wins"]),
                    "ties": int(c["ties"]),
                    "losses": int(c["losses"]),
                    "win_rate": float(c["wins"] / n) if n else None,
                    "tie_rate": float(c["ties"] / n) if n else None,
                    "loss_rate": float(c["losses"] / n) if n else None,
                })
    return out


def seed_stability_summary_rows(rows, metrics=MONTE_CARLO_SUMMARY_METRICS):
    mc = monte_carlo_cell_summary_rows(rows, metrics=metrics)
    out = []
    grouped = defaultdict(list)
    for row in mc:
        grouped[(row["signal_regime"], row["method"], row["metric"])].append(row)
    for key, vals in sorted(grouped.items(), key=lambda item: str(item[0])):
        sd = np.asarray([v["mc_sd"] for v in vals if _finite(v.get("mc_sd")) is not None], dtype=float)
        iqr = np.asarray([v["iqr"] for v in vals if _finite(v.get("iqr")) is not None], dtype=float)
        n_success = np.asarray([v["n_success"] for v in vals], dtype=float)
        out.append({
            "signal_regime": key[0],
            "method": key[1],
            "metric": key[2],
            "n_cells": int(len(vals)),
            "median_mc_sd_across_cells": float(np.median(sd)) if sd.size else None,
            "mean_mc_sd_across_cells": float(np.mean(sd)) if sd.size else None,
            "median_iqr_across_cells": float(np.median(iqr)) if iqr.size else None,
            "mean_iqr_across_cells": float(np.mean(iqr)) if iqr.size else None,
            "min_successful_seeds_per_cell": int(np.min(n_success)) if n_success.size else 0,
            "median_successful_seeds_per_cell": float(np.median(n_success)) if n_success.size else None,
        })
    return out


def run_unified_benchmark_cube(
        output_root,
        irmf_params=GLOBAL_IRMF_PARAMS,
        emd_params=GLOBAL_EMD_PARAMS,
        eemd_params=GLOBAL_EEMD_PARAMS,
        ceemdan_params=GLOBAL_CEEMDAN_PARAMS,
        regimes=None,
        seeds=UNIFIED_BENCHMARK_SEEDS,
        n=DEFAULT_N,
        fs=DEFAULT_FS,
        timeout_seconds=DEFAULT_METHOD_TIMEOUT_SECONDS,
        quick=False,
        parameter_source=None,
        parameter_lock_audit=None,
        reconstruction_protocol_id=None,
        benchmark_run_label=None,
        target_snr_db_levels=None,
):
    output_root = ensure_dir(output_root)
    regimes = _quick_regimes() if quick else (regimes or UNIFIED_BENCHMARK_REGIMES)
    seeds = tuple(UNIFIED_BENCHMARK_QUICK_SEEDS if quick else seeds)
    checkpoint_path = output_root / "unified_benchmark_cube_rows.jsonl"
    target_snr_db_levels = (
        tuple(float(v) for v in target_snr_db_levels)
        if target_snr_db_levels is not None
        else None
    )
    rows = _load_checkpoint_rows(checkpoint_path)
    completed = {
        row.get("cube_cell_id")
        for row in rows
        if row.get("cube_cell_id")
    }
    for regime_name, cfg in regimes.items():
        for signal_name in cfg["signals"]:
            for noise_name in cfg["noises"]:
                severity_levels = (
                    tuple((10.0 ** (-snr / 20.0), snr) for snr in target_snr_db_levels)
                    if target_snr_db_levels is not None
                    else tuple((float(sigma), None) for sigma in cfg["sigmas"])
                )
                for sigma, target_snr_db in severity_levels:
                    run_id_severity = (
                        f"snr{float(target_snr_db):g}dB"
                        if target_snr_db is not None
                        else f"{float(sigma):g}"
                    )
                    for seed in seeds:
                        data_seed = int(seed)
                        cube_cell_id = _checkpoint_key_with_snr(
                            regime_name,
                            signal_name,
                            noise_name,
                            sigma,
                            target_snr_db,
                            data_seed,
                        )
                        if cube_cell_id in completed:
                            continue
                        algorithm_seed_base = _algorithm_seed_for(data_seed)
                        _, _, row = run_fixed_method_family_case(
                            signal_name=signal_name,
                            noise_name=noise_name,
                            sigma=sigma,
                            irmf_params=irmf_params,
                            emd_params=emd_params,
                            eemd_params=eemd_params,
                            ceemdan_params=ceemdan_params,
                            n=n,
                            fs=fs,
                            seed=data_seed,
                            target_snr_db=target_snr_db,
                            algorithm_seed=algorithm_seed_base,
                            timeout_seconds=timeout_seconds,
                            reconstruction_protocol_id=reconstruction_protocol_id,
                            run_id_prefix=f"cube_{regime_name}_{signal_name}_{noise_name}_{run_id_severity}_dataseed{data_seed}",
                        )
                        row["signal_regime"] = regime_name
                        row["seed"] = data_seed
                        row["data_seed"] = data_seed
                        row["algorithm_seed"] = algorithm_seed_base
                        row["algorithm_seed_base"] = algorithm_seed_base
                        row["algorithm_seed_EEMD"] = algorithm_seed_base
                        row["algorithm_seed_CEEMDAN"] = algorithm_seed_base + 50000
                        row["noise_severity_design"] = (
                            "target_snr_energy_ratio" if target_snr_db is not None else "fixed_sigma"
                        )
                        row["target_snr_db"] = target_snr_db
                        row["sigma_role"] = (
                            "derived_noise_to_signal_rms_ratio_label"
                            if target_snr_db is not None
                            else "absolute_noise_scale"
                        )
                        row["seed_role"] = "data_generation_seed"
                        row["algorithm_seed_policy"] = "EEMD uses 100000 + data_seed; CEEMDAN uses 150000 + data_seed"
                        row["cube_cell_id"] = cube_cell_id
                        rows.append(row)
                        completed.add(cube_cell_id)
                        _append_checkpoint_row(checkpoint_path, row)

    protocol = {
        "section": "V5.23 Unified 20-Replication Monte Carlo Factorial Benchmark Cube",
        "design": (
            "Method x SignalRegime x SignalClass(SignalRegime) x Noise x "
            "Severity x DataSeed"
        ),
        "methods": list(METHODS),
        "regimes": {
            name: {
                "signals": list(cfg["signals"]),
                "noises": list(cfg["noises"]),
                "sigmas": list(cfg["sigmas"]),
                "target_snr_db_levels": (
                    list(target_snr_db_levels) if target_snr_db_levels is not None else None
                ),
            }
            for name, cfg in regimes.items()
        },
        "seeds": list(seeds),
        "n_monte_carlo_replications_per_cell": int(len(seeds)),
        "n_signal_noise_sigma_seed_cells": int(len(rows)),
        "n_method_case_evaluations": int(len(rows) * len(METHODS)),
        "timeout_seconds": timeout_seconds,
        "checkpoint_policy": {
            "enabled": True,
            "checkpoint_file": "unified_benchmark_cube_rows.jsonl",
            "resume_behavior": "completed cube_cell_id rows are skipped on rerun",
            "write_granularity": "one signal-noise-sigma-data_seed cell after all methods finish",
        },
        "seed_policy": {
            "data_seed": "controls signal/noise realization, contamination locations, AR innovations, and heteroskedastic innovations",
            "algorithm_seed": "controls EEMD and CEEMDAN internal ensemble noise only",
            "IRMF_EMD_algorithm_seed": "not_applicable_deterministic_given_input",
            "EEMD_algorithm_seed": "100000 + data_seed",
            "CEEMDAN_algorithm_seed": "150000 + data_seed",
            "paired_design": "all methods process the same observed signal within each data_seed replication",
        },
        "parameter_source": parameter_source,
        "benchmark_run_label": benchmark_run_label,
        "reconstruction_protocol_id": reconstruction_protocol_id,
        "noise_severity_design": (
            "target_snr_energy_ratio" if target_snr_db_levels is not None else "fixed_sigma"
        ),
        "target_snr_db_levels": (
            list(target_snr_db_levels) if target_snr_db_levels is not None else None
        ),
        "target_snr_formula": (
            "N_t = alpha_gamma Z_t, alpha_gamma = sqrt(sum X_t^2 / "
            "(10^(gamma/10) sum Z_t^2))"
            if target_snr_db_levels is not None
            else None
        ),
        "reconstruction_protocol_policy": (
            "native_residual_reconstruction"
            if reconstruction_protocol_id is None else
            "protocol_controlled_synthetic_component_selection_reconstruction"
        ),
        "parameter_lock_audit": parameter_lock_audit,
        "locked_parameter_policy": {
            "per_case_tuning": False,
            "fixed_for_all_cube_cells": True,
            "oracle_results_used_for_main_claims": False,
        },
        "failure_criteria": {
            "computational_failure": "timeout, exception, non-finite output, or missing decomposition output",
            "structural_failure": "imf_recovery_score < 0.20, severe mode mixing, excessive decomposition-count error, or IMF-count explosion",
            "denoising_failure": "denoise_nmse >= noisy_input_nmse",
        },
        "quick": bool(quick),
    }
    write_json(protocol, output_root / "unified_benchmark_cube_protocol.json")
    write_json(rows, output_root / "unified_benchmark_cube_rows.json")
    write_csv(rows, output_root / "unified_benchmark_cube_rows.csv")
    write_json(aggregate_method_family_rows(rows), output_root / "unified_benchmark_cube_aggregate.json")

    section5 = _slice_rows(rows, regime="canonical")
    section6 = _slice_rows(rows, regime="canonical")
    section8 = _slice_rows(rows, regime="challenging")
    summaries = {
        "section_5_canonical_main_slice": _write_rows_and_summary(
            section5,
            output_root / "section_5_canonical_main_slice",
            "section_5_canonical_main_slice",
            {"source": "unified_benchmark_cube", "slice": "canonical regime, all five sigma levels"},
        ),
        "section_6_1_full_noise_robustness_slice": _write_rows_and_summary(
            section6,
            output_root / "section_6_1_full_noise_robustness_slice",
            "section_6_1_full_noise_robustness_slice",
            {"source": "unified_benchmark_cube", "slice": "canonical regime, all sigma levels"},
        ),
        "section_8_challenging_generalization_slice": _write_rows_and_summary(
            section8,
            output_root / "section_8_challenging_generalization_slice",
            "section_8_challenging_generalization_slice",
            {"source": "unified_benchmark_cube", "slice": "challenging regime, all five sigma levels"},
        ),
    }

    auc_rows = robustness_auc_rows(rows)
    auc_aggregate = aggregate_robustness_auc(auc_rows)
    failures = failure_summary_rows(rows)
    mc_summary = monte_carlo_cell_summary_rows(rows)
    paired_mc = paired_method_difference_summary_rows(rows)
    seed_stability = seed_stability_summary_rows(rows)
    win_tie_loss = method_win_tie_loss_rows(rows)
    write_csv(auc_rows, output_root / "robustness_auc_by_trajectory.csv")
    write_json(auc_rows, output_root / "robustness_auc_by_trajectory.json")
    write_csv(auc_aggregate, output_root / "robustness_auc_aggregate.csv")
    write_json(auc_aggregate, output_root / "robustness_auc_aggregate.json")
    write_csv(failures, output_root / "failure_rates_by_cell.csv")
    write_json(failures, output_root / "failure_rates_by_cell.json")
    write_csv(mc_summary, output_root / "monte_carlo_cell_summary.csv")
    write_json(mc_summary, output_root / "monte_carlo_cell_summary.json")
    write_csv(paired_mc, output_root / "paired_method_difference_summary.csv")
    write_json(paired_mc, output_root / "paired_method_difference_summary.json")
    write_csv(seed_stability, output_root / "seed_stability_summary.csv")
    write_json(seed_stability, output_root / "seed_stability_summary.json")
    write_csv(win_tie_loss, output_root / "method_win_tie_loss.csv")
    write_json(win_tie_loss, output_root / "method_win_tie_loss.json")

    dashboard = {
        "protocol": protocol,
        "slice_summaries": summaries,
        "robustness_auc_rows": len(auc_rows),
        "robustness_auc_aggregate_rows": len(auc_aggregate),
        "failure_summary_rows": len(failures),
        "monte_carlo_cell_summary_rows": len(mc_summary),
        "paired_method_difference_summary_rows": len(paired_mc),
        "seed_stability_summary_rows": len(seed_stability),
        "method_win_tie_loss_rows": len(win_tie_loss),
        "outputs": {
            "cube_rows": "unified_benchmark_cube_rows.csv",
            "checkpoint_rows_jsonl": "unified_benchmark_cube_rows.jsonl",
            "section_5": "section_5_canonical_main_slice/",
            "section_6_1": "section_6_1_full_noise_robustness_slice/",
            "section_8": "section_8_challenging_generalization_slice/",
            "robustness_auc": "robustness_auc_by_trajectory.csv",
            "failure_rates": "failure_rates_by_cell.csv",
            "monte_carlo_cell_summary": "monte_carlo_cell_summary.csv",
            "paired_method_difference_summary": "paired_method_difference_summary.csv",
            "seed_stability_summary": "seed_stability_summary.csv",
            "method_win_tie_loss": "method_win_tie_loss.csv",
        },
    }
    write_json(dashboard, output_root / "unified_benchmark_cube_dashboard.json")
    return rows, dashboard


if __name__ == "__main__":
    run_unified_benchmark_cube("IRMF_EMD_PAPER_RESULTS/unified_benchmark_cube", quick=True)
