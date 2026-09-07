#!/usr/bin/python
# coding: UTF-8

"""
Canonical fixed-parameter EMD-family benchmark.

Representative mode:
    5 signals × 4 noises × sigma=0.20 × 4 methods

Full mode:
    5 signals × 6 noises × 2 sigma × 4 methods
"""

from pathlib import Path
import json
import numpy as np
from time import perf_counter

from project_config import (
    CEEMDAN_PARAMETER_SELECTION_GRID,
    EMD_PARAMETER_SELECTION_GRID,
    EEMD_PARAMETER_SELECTION_GRID,
    EMD_FAMILY_BENCHMARK_MODES,
    EMD_FAMILY_SENSITIVITY_METHODS,
    EMD_FAMILY_SENSITIVITY_NOISE_WIDTHS,
    EMD_FAMILY_SENSITIVITY_TRIALS,
    GLOBAL_CEEMDAN_PARAMS,
    GLOBAL_EMD_PARAMS,
    GLOBAL_EEMD_PARAMS,
    GLOBAL_IRMF_PARAMS,
    PARAMETER_SELECTION_NOISES,
    PARAMETER_SELECTION_SIGMAS,
    PARAMETER_SELECTION_SIGNALS,
    SECTION6_REPRESENTATIVE_TARGET_SNR_DB_LEVELS,
)
from experiments.experiment_utils import (
    make_signal_noise_case,
    method_result_summary,
    run_fixed_emd_case,
    run_fixed_irmf_case,
    run_single_emd_case,
    run_single_irmf_case,
)
from experiments.experiment_reporting import print_extended_comparison
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json
from diagnostics.shared_physical_diagnostics import evaluate_shared_physical_diagnostics


def _sigma_label_for_snr(target_snr_db):
    return float(10.0 ** (-float(target_snr_db) / 20.0))
from core_algorithms.eemd_wrapper import run_eemd
from core_algorithms.ceemdan_wrapper import run_ceemdan


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


def _evaluate_family_method(method_name, Y, X_clean, fs, true_components=None, contamination_mask=None):
    method_name = method_name.upper()

    if method_name == "EEMD":
        raw = run_eemd(Y, max_imf=-1, trials=50, noise_width=0.05)
    elif method_name == "CEEMDAN":
        raw = run_ceemdan(Y, max_imf=-1, trials=50, epsilon=0.005)
    else:
        raise ValueError(f"Unsupported EMD-family method: {method_name}")

    physical = evaluate_shared_physical_diagnostics(
        Y_observed=Y,
        X_clean=X_clean,
        imfs=raw["imfs"],
        residual=raw["residual"],
        fs=fs,
        true_components=true_components,
        contamination_mask=contamination_mask,
    )

    return {"method": method_name, "imfs": raw["imfs"], "residual": raw["residual"], **physical}


def _run_locked_emd_family_method(method_name, case, fs, emd_params, eemd_params, ceemdan_params, algorithm_seed):
    method_name = str(method_name).upper()
    true_components = case.get("true_components", None)
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
    else:
        raise ValueError(f"Unsupported EMD-family method: {method_name}")

    physical = evaluate_shared_physical_diagnostics(
        Y_observed=case["Y"],
        X_clean=case["X_clean"],
        imfs=raw["imfs"],
        residual=raw["residual"],
        fs=fs,
        residual_penalty_mode="none",
        true_components=true_components,
        contamination_mask=case.get("contamination_mask"),
    )
    result = {"method": method_name, **raw, **physical}
    return result


def _run_locked_emd_method(case, fs, emd_params):
    """Run classical EMD with a candidate fixed protocol for calibration."""
    return run_fixed_emd_case(
        Y=case["Y"],
        X_clean=case["X_clean"],
        t=case["t"],
        fs=fs,
        emd_params=emd_params,
        true_components=case.get("true_components", None),
        run_id="development_emd_protocol_calibration",
    )


def _family_aggregate(rows):
    metrics = [
        "case_score",
        "reconstruction_score",
        "structural_fidelity_score",
        "contamination_resistance_score",
        "denoise_nmse",
        "imf_recovery_score",
        "component_splitting_index",
        "component_merging_index",
        "inter_imf_entanglement_index",
        "noise_capture_corr",
        "outlier_resistance_index",
    ]
    methods = ["IRMF", "EMD", "EEMD", "CEEMDAN"]
    out = {"n_cases": int(len(rows)), "methods": methods, "metrics": {}}
    for metric in metrics:
        metric_out = {}
        values_by_method = {}
        for method in methods:
            vals = []
            for row in rows:
                value = row.get(method, {}).get(metric)
                try:
                    if value is not None and np.isfinite(value):
                        vals.append(float(value))
                except Exception:
                    continue
            values_by_method[method] = vals
            metric_out[f"{method}_mean"] = float(np.mean(vals)) if vals else None

        irmf_vals = values_by_method.get("IRMF", [])
        higher_is_better = metric not in ("denoise_nmse", "component_splitting_index", "component_merging_index", "inter_imf_entanglement_index", "mode_mixing_index")
        for baseline in ["EMD", "EEMD", "CEEMDAN"]:
            pairs = []
            for row in rows:
                iv = row.get("IRMF", {}).get(metric)
                bv = row.get(baseline, {}).get(metric)
                try:
                    if iv is not None and bv is not None and np.isfinite(iv) and np.isfinite(bv):
                        pairs.append((float(iv), float(bv)))
                except Exception:
                    continue
            if pairs:
                deltas = np.asarray([iv - bv for iv, bv in pairs], dtype=float)
                wins = [(iv > bv) if higher_is_better else (iv < bv) for iv, bv in pairs]
                metric_out[f"IRMF_minus_{baseline}_delta_mean"] = float(np.mean(deltas))
                metric_out[f"IRMF_vs_{baseline}_win_rate"] = float(np.mean(wins))
                metric_out[f"IRMF_vs_{baseline}_n_paired"] = int(len(pairs))
            else:
                metric_out[f"IRMF_minus_{baseline}_delta_mean"] = None
                metric_out[f"IRMF_vs_{baseline}_win_rate"] = None
                metric_out[f"IRMF_vs_{baseline}_n_paired"] = 0
        out["metrics"][metric] = metric_out
    return out


def _configuration_key(row):
    method = row.get("baseline_method")
    if method == "EMD":
        return (
            method,
            row.get("nbsym"),
            row.get("spline_kind"),
            row.get("max_imf"),
            row.get("std_thr"),
            row.get("svar_thr"),
            row.get("total_power_thr"),
            row.get("range_thr"),
            None,
            None,
            None,
        )
    return (
        method,
        None,
        None,
        row.get("max_imf"),
        None,
        None,
        None,
        None,
        row.get("trials"),
        row.get("noise_width"),
        row.get("epsilon"),
    )


def _baseline_sensitivity_aggregate(rows):
    metrics = [
        "case_score",
        "reconstruction_score",
        "structural_fidelity_score",
        "contamination_resistance_score",
        "denoise_nmse",
        "imf_recovery_score",
        "component_splitting_index",
        "component_merging_index",
        "inter_imf_entanglement_index",
        "noise_capture_corr",
        "outlier_resistance_index",
    ]
    out = {
        "n_rows": int(len(rows)),
        "metrics": {},
        "by_configuration": [],
        "best_configuration_by_metric": {},
    }
    config_keys = sorted({_configuration_key(row) for row in rows}, key=lambda x: str(x))

    for (
        method,
        nbsym,
        spline_kind,
        max_imf,
        std_thr,
        svar_thr,
        total_power_thr,
        range_thr,
        trials,
        noise_width,
        epsilon,
    ) in config_keys:
        subset = [
            row for row in rows
            if _configuration_key(row) == (
                method,
                nbsym,
                spline_kind,
                max_imf,
                std_thr,
                svar_thr,
                total_power_thr,
                range_thr,
                trials,
                noise_width,
                epsilon,
            )
        ]
        item = {
            "baseline_method": method,
            "nbsym": nbsym,
            "spline_kind": spline_kind,
            "max_imf": max_imf,
            "std_thr": std_thr,
            "svar_thr": svar_thr,
            "total_power_thr": total_power_thr,
            "range_thr": range_thr,
            "trials": trials,
            "noise_width": noise_width,
            "epsilon": epsilon,
            "n_cases": int(len(subset)),
        }
        for metric in metrics:
            pairs = []
            baseline_vals = []
            irmf_vals = []
            for row in subset:
                iv = row.get("IRMF", {}).get(metric)
                bv = row.get("baseline", {}).get(metric)
                try:
                    if bv is not None and np.isfinite(bv):
                        baseline_vals.append(float(bv))
                    if iv is not None and np.isfinite(iv):
                        irmf_vals.append(float(iv))
                    if iv is not None and bv is not None and np.isfinite(iv) and np.isfinite(bv):
                        pairs.append((float(iv), float(bv)))
                except Exception:
                    continue
            if baseline_vals:
                item[f"baseline_mean_{metric}"] = float(np.mean(baseline_vals))
            if irmf_vals:
                item[f"irmf_mean_{metric}"] = float(np.mean(irmf_vals))
            if pairs:
                higher_is_better = metric not in ("denoise_nmse", "component_splitting_index", "component_merging_index", "inter_imf_entanglement_index", "mode_mixing_index")
                deltas = np.asarray([iv - bv for iv, bv in pairs], dtype=float)
                wins = [(iv > bv) if higher_is_better else (iv < bv) for iv, bv in pairs]
                item[f"irmf_minus_baseline_delta_{metric}"] = float(np.mean(deltas))
                item[f"irmf_vs_baseline_win_rate_{metric}"] = float(np.mean(wins))
        out["by_configuration"].append(item)

    for metric in metrics:
        higher_is_better = metric not in ("denoise_nmse", "component_splitting_index", "component_merging_index", "inter_imf_entanglement_index", "mode_mixing_index")
        candidates = [
            item for item in out["by_configuration"]
            if item.get(f"baseline_mean_{metric}") is not None
        ]
        if not candidates:
            continue
        best = sorted(
            candidates,
            key=lambda item: item[f"baseline_mean_{metric}"],
            reverse=higher_is_better,
        )[0]
        out["best_configuration_by_metric"][metric] = {
            "baseline_method": best["baseline_method"],
            "nbsym": best.get("nbsym"),
            "spline_kind": best.get("spline_kind"),
            "max_imf": best.get("max_imf"),
            "std_thr": best.get("std_thr"),
            "svar_thr": best.get("svar_thr"),
            "total_power_thr": best.get("total_power_thr"),
            "range_thr": best.get("range_thr"),
            "trials": best["trials"],
            "noise_width": best["noise_width"],
            "epsilon": best.get("epsilon"),
            "baseline_mean": best.get(f"baseline_mean_{metric}"),
            "irmf_mean": best.get(f"irmf_mean_{metric}"),
            "irmf_minus_baseline_delta": best.get(f"irmf_minus_baseline_delta_{metric}"),
            "irmf_vs_baseline_win_rate": best.get(f"irmf_vs_baseline_win_rate_{metric}"),
        }
    out["metrics"] = {
        metric: {
            "best_baseline_configuration": out["best_configuration_by_metric"].get(metric),
        }
        for metric in metrics
    }
    return out


def _mean_metric(rows, metric="case_score"):
    vals = []
    for row in rows:
        value = row.get(metric)
        try:
            if value is not None and np.isfinite(value):
                vals.append(float(value))
        except Exception:
            continue
    return float(np.mean(vals)) if vals else None


def _development_cases(n, fs, seed):
    cases = []
    for signal_name in PARAMETER_SELECTION_SIGNALS:
        for noise_name in PARAMETER_SELECTION_NOISES:
            for sigma in PARAMETER_SELECTION_SIGMAS:
                case = make_signal_noise_case(
                    signal_name=signal_name,
                    noise_name=noise_name,
                    sigma=sigma,
                    n=n,
                    fs=fs,
                    seed=seed,
                )
                cases.append({
                    "signal": signal_name,
                    "noise": noise_name,
                    "sigma": sigma,
                    "data_seed": seed,
                    "case": case,
                })
    return cases


def run_emd_family_parameter_calibration(
        output_root,
        emd_params=GLOBAL_EMD_PARAMS,
        emd_grid=EMD_PARAMETER_SELECTION_GRID,
        eemd_grid=EEMD_PARAMETER_SELECTION_GRID,
        ceemdan_grid=CEEMDAN_PARAMETER_SELECTION_GRID,
        n=500,
        fs=500.0,
        data_seed=0,
        algorithm_seed=20260715,
):
    """
    Section 4: development-set calibration for EMD-family baselines.

    This mirrors the global-parameter philosophy used for IRMF: each candidate
    EMD/EEMD/CEEMDAN configuration is evaluated on the same pre-specified
    32-case development set, the best mean case_score is selected once, and the
    selected configuration is locked for the main 168-case benchmark.
    """
    output_root = ensure_dir(output_root)
    cases = _development_cases(n=n, fs=fs, seed=data_seed)
    rows = []
    selected = {}

    method_grids = {
        "EMD": [
            {
                "nbsym": int(nbsym),
                "spline_kind": str(spline_kind),
                "max_imf": int(max_imf),
                "std_thr": std_thr,
                "svar_thr": svar_thr,
                "total_power_thr": total_power_thr,
                "range_thr": range_thr,
            }
            for nbsym in emd_grid["nbsym_options"]
            for spline_kind in emd_grid["spline_kind_options"]
            for max_imf in emd_grid["max_imf_options"]
            for std_thr in emd_grid["std_thr_options"]
            for svar_thr in emd_grid["svar_thr_options"]
            for total_power_thr in emd_grid["total_power_thr_options"]
            for range_thr in emd_grid["range_thr_options"]
        ],
        "EEMD": [
            {"trials": int(trials), "noise_width": float(noise_width)}
            for trials in eemd_grid["trials_options"]
            for noise_width in eemd_grid["noise_width_options"]
        ],
        "CEEMDAN": [
            {"trials": int(trials), "epsilon": float(epsilon)}
            for trials in ceemdan_grid["trials_options"]
            for epsilon in ceemdan_grid["epsilon_options"]
        ],
    }

    for method_name, configs in method_grids.items():
        config_summaries = []
        for config_id, config in enumerate(configs):
            case_scores = []
            config_runtime = []
            for cached in cases:
                case = cached["case"]
                row = {
                    "method": method_name,
                    "candidate_id": int(config_id),
                    "signal": cached["signal"],
                    "noise": cached["noise"],
                    "sigma": cached["sigma"],
                    "data_seed": int(data_seed),
                    "algorithm_seed": int(algorithm_seed),
                    **config,
                }
                try:
                    if method_name == "EMD":
                        result = _run_locked_emd_method(
                            case,
                            fs,
                            emd_params={**GLOBAL_EMD_PARAMS, **config},
                        )
                    elif method_name == "EEMD":
                        result = _run_locked_emd_family_method(
                            method_name,
                            case,
                            fs,
                            emd_params=emd_params,
                            eemd_params={**GLOBAL_EEMD_PARAMS, **config},
                            ceemdan_params=GLOBAL_CEEMDAN_PARAMS,
                            algorithm_seed=algorithm_seed,
                        )
                    else:
                        result = _run_locked_emd_family_method(
                            method_name,
                            case,
                            fs,
                            emd_params=emd_params,
                            eemd_params=GLOBAL_EEMD_PARAMS,
                            ceemdan_params={**GLOBAL_CEEMDAN_PARAMS, **config},
                            algorithm_seed=algorithm_seed,
                        )
                    summary = method_result_summary(result)
                    score = summary.get("case_score")
                    runtime = summary.get("runtime_seconds")
                    if score is not None and np.isfinite(score):
                        case_scores.append(float(score))
                    if runtime is not None and np.isfinite(runtime):
                        config_runtime.append(float(runtime))
                    row.update(summary)
                except Exception as exc:
                    row["error"] = str(exc)
                rows.append(row)

            config_summary = {
                "method": method_name,
                "candidate_id": int(config_id),
                **config,
                "n_development_cases": int(len(cases)),
                "mean_case_score": float(np.mean(case_scores)) if case_scores else None,
                "median_case_score": float(np.median(case_scores)) if case_scores else None,
                "mean_runtime_seconds": float(np.mean(config_runtime)) if config_runtime else None,
                "total_runtime_seconds": float(np.sum(config_runtime)) if config_runtime else None,
            }
            config_summaries.append(config_summary)

        valid = [r for r in config_summaries if r.get("mean_case_score") is not None]
        if valid:
            best = sorted(valid, key=lambda r: r["mean_case_score"], reverse=True)[0]
            if method_name == "EMD":
                selected[method_name] = {
                    **GLOBAL_EMD_PARAMS,
                    "nbsym": best["nbsym"],
                    "spline_kind": best["spline_kind"],
                    "max_imf": best["max_imf"],
                    "std_thr": best.get("std_thr"),
                    "svar_thr": best.get("svar_thr"),
                    "total_power_thr": best.get("total_power_thr"),
                    "range_thr": best.get("range_thr"),
                    "selection_candidate_id": best["candidate_id"],
                }
            elif method_name == "EEMD":
                selected[method_name] = {
                    **GLOBAL_EEMD_PARAMS,
                    "trials": best["trials"],
                    "noise_width": best["noise_width"],
                    "selection_candidate_id": best["candidate_id"],
                }
            else:
                selected[method_name] = {
                    **GLOBAL_CEEMDAN_PARAMS,
                    "trials": best["trials"],
                    "epsilon": best["epsilon"],
                    "selection_candidate_id": best["candidate_id"],
                }
        write_csv(config_summaries, output_root / f"{method_name.lower()}_development_configuration_summary.csv")
        write_json(config_summaries, output_root / f"{method_name.lower()}_development_configuration_summary.json")

    protocol = {
        "section": "4 EMD-family baseline calibration",
        "development_design": {
            "signals": list(PARAMETER_SELECTION_SIGNALS),
            "noises": list(PARAMETER_SELECTION_NOISES),
            "sigmas": list(PARAMETER_SELECTION_SIGMAS),
            "n_cases": int(len(cases)),
        },
        "selection_rule": "For each baseline family, select the candidate with maximum mean development-set case_score.",
        "emd_grid": emd_grid,
        "eemd_grid": eemd_grid,
        "ceemdan_grid": ceemdan_grid,
        "data_seed": int(data_seed),
        "algorithm_seed": int(algorithm_seed),
        "selected_params": selected,
    }
    write_json(protocol, output_root / "emd_family_parameter_calibration_protocol.json")
    write_json(rows, output_root / "emd_family_parameter_calibration_case_results.json")
    write_csv(rows, output_root / "emd_family_parameter_calibration_case_results.csv")
    write_json(selected, output_root / "selected_emd_family_params.json")
    return selected


def run_fixed_emd_family_benchmark(
        output_root,
        irmf_params=GLOBAL_IRMF_PARAMS,
        emd_params=GLOBAL_EMD_PARAMS,
        eemd_params=GLOBAL_EEMD_PARAMS,
        ceemdan_params=GLOBAL_CEEMDAN_PARAMS,
        mode="representative",
        n=500,
        fs=500.0,
        data_seed=0,
        algorithm_seed=20260715,
        seed=None,
        eemd_trials=None,
        ceemdan_trials=None,
        noise_width=None,
        ceemdan_epsilon=None,
        return_rows=False,
):
    """
    Fixed-parameter EMD-family benchmark for the paper pipeline.

    Unlike the legacy function below, this does not run per-case IRMF grid
    search.  It compares the globally locked IRMF configuration with fixed EMD,
    EEMD, and CEEMDAN settings on the same synthetic cases.
    """
    if mode not in EMD_FAMILY_BENCHMARK_MODES:
        raise ValueError(f"Unknown EMD family benchmark mode: {mode}")

    if seed is not None:
        data_seed = seed
    eemd_params = dict(eemd_params)
    ceemdan_params = dict(ceemdan_params)
    if eemd_trials is not None:
        eemd_params["trials"] = int(eemd_trials)
    if ceemdan_trials is not None:
        ceemdan_params["trials"] = int(ceemdan_trials)
    if noise_width is not None:
        eemd_params["noise_width"] = float(noise_width)
    if ceemdan_epsilon is not None:
        ceemdan_params["epsilon"] = float(ceemdan_epsilon)

    cfg = EMD_FAMILY_BENCHMARK_MODES[mode]
    output_root = ensure_dir(output_root)
    rows = []

    for signal_name in cfg["signals"]:
        for noise_name in cfg["noises"]:
            for sigma in cfg["sigmas"]:
                case = make_signal_noise_case(
                    signal_name=signal_name,
                    noise_name=noise_name,
                    sigma=sigma,
                    n=n,
                    fs=fs,
                    seed=data_seed,
                )
                true_components = case.get("true_components", None)
                row = {
                    "mode": mode,
                    "signal": signal_name,
                    "noise": noise_name,
                    "sigma": sigma,
                    "data_seed": data_seed,
                    "algorithm_seed": algorithm_seed,
                }
                start = perf_counter()
                irmf = run_fixed_irmf_case(
                    Y=case["Y"],
                    X_clean=case["X_clean"],
                    t=case["t"],
                    fs=fs,
                    irmf_params=irmf_params,
                    expected_noise_ratio=case.get("expected_noise_ratio"),
                    true_components=true_components,
                    contamination_mask=case.get("contamination_mask"),
                    run_id=f"fixed_family_{signal_name}_{noise_name}_{sigma}_irmf",
                )
                irmf["runtime_seconds"] = perf_counter() - start
                start = perf_counter()
                emd = run_fixed_emd_case(
                    Y=case["Y"],
                    X_clean=case["X_clean"],
                    t=case["t"],
                    fs=fs,
                    emd_params=emd_params,
                    true_components=true_components,
                    contamination_mask=case.get("contamination_mask"),
                    run_id=f"fixed_family_{signal_name}_{noise_name}_{sigma}_emd",
                )
                emd["runtime_seconds"] = perf_counter() - start
                row["IRMF"] = method_result_summary(irmf)
                row["EMD"] = method_result_summary(emd)

                for method_name in ["EEMD", "CEEMDAN"]:
                    try:
                        result = _run_locked_emd_family_method(
                            method_name,
                            case,
                            fs=fs,
                            emd_params=emd_params,
                            eemd_params=eemd_params,
                            ceemdan_params=ceemdan_params,
                            algorithm_seed=algorithm_seed,
                        )
                        result["run_id"] = f"fixed_family_{signal_name}_{noise_name}_{sigma}_{method_name.lower()}"
                        row[method_name] = method_result_summary(result)
                    except Exception as exc:
                        row[method_name] = {"method": method_name, "error": str(exc)}
                rows.append(row)

    summary = _family_aggregate(rows)
    protocol = {
        "section": "5 Canonical Fixed-Parameter EMD-Family Benchmark",
        "mode": mode,
        "design": EMD_FAMILY_BENCHMARK_MODES[mode],
        "n_cases": int(len(rows)),
        "methods": ["IRMF", "EMD", "EEMD", "CEEMDAN"],
        "paper_role": (
            "primary canonical benchmark; pairwise IRMF-vs-EMD summaries are "
            "reported as classical-reference contrasts within this four-method design"
        ),
        "irmf_params": dict(irmf_params),
        "emd_params": dict(emd_params),
        "eemd_params": dict(eemd_params),
        "ceemdan_params": dict(ceemdan_params),
        "data_seed": int(data_seed),
        "algorithm_seed": int(algorithm_seed),
        "note": "IRMF uses the globally locked fixed-parameter configuration; no per-case IRMF grid search is used.",
    }
    write_json(protocol, output_root / "section_5_canonical_emd_family_benchmark_protocol.json")
    write_json(rows, output_root / "section_5_canonical_emd_family_benchmark.json")
    write_csv(rows, output_root / "section_5_canonical_emd_family_benchmark.csv")
    write_json(summary, output_root / "section_5_canonical_emd_family_benchmark_aggregate.json")

    # Backward-compatible aliases for existing analysis scripts and old results.
    write_json(protocol, output_root / "section_7b_emd_family_protocol.json")
    write_json(rows, output_root / "section_7b_emd_family_comparison.json")
    write_csv(rows, output_root / "section_7b_emd_family_comparison.csv")
    write_json(summary, output_root / "section_7b_emd_family_comparison_aggregate.json")
    if return_rows:
        return rows, summary
    return summary


def run_emd_family_baseline_sensitivity(
        output_root,
        irmf_params=GLOBAL_IRMF_PARAMS,
        emd_params=GLOBAL_EMD_PARAMS,
        mode="representative",
        methods=("EMD",) + tuple(EMD_FAMILY_SENSITIVITY_METHODS),
        emd_grid=EMD_PARAMETER_SELECTION_GRID,
        trials_grid=EMD_FAMILY_SENSITIVITY_TRIALS,
        noise_width_grid=EMD_FAMILY_SENSITIVITY_NOISE_WIDTHS,
        epsilon_grid=CEEMDAN_PARAMETER_SELECTION_GRID["epsilon_options"],
        target_snr_db_levels=SECTION6_REPRESENTATIVE_TARGET_SNR_DB_LEVELS,
        n=500,
        fs=500.0,
        data_seed=0,
        algorithm_seed=20260715,
        seed=None,
):
    """
    Appendix D: sensitivity of EMD-family baseline parameters.

    This is not per-case tuning.  Each EMD-family configuration is evaluated
    across the same representative cases, then summarized at the configuration
    level.  Fixed IRMF is included only as a reference comparator.
    """
    if mode not in EMD_FAMILY_BENCHMARK_MODES:
        raise ValueError(f"Unknown EMD family benchmark mode: {mode}")

    if seed is not None:
        data_seed = seed

    cfg = EMD_FAMILY_BENCHMARK_MODES[mode]
    output_root = ensure_dir(output_root)
    rows = []

    case_cache = []
    for signal_name in cfg["signals"]:
        for noise_name in cfg["noises"]:
            for target_snr_db in target_snr_db_levels:
                sigma = _sigma_label_for_snr(target_snr_db)
                case = make_signal_noise_case(
                    signal_name=signal_name,
                    noise_name=noise_name,
                    sigma=sigma,
                    n=n,
                    fs=fs,
                    seed=data_seed,
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
                    run_id=(
                        f"appendix_D_reference_{signal_name}_{noise_name}_"
                        f"snr{float(target_snr_db):g}dB_irmf"
                    ),
                )
                case_cache.append({
                    "signal": signal_name,
                    "noise": noise_name,
                    "sigma": sigma,
                    "target_snr_db": target_snr_db,
                    "noise_severity_design": case.get("noise_design"),
                    "realized_input_snr_db": case.get("realized_input_snr_db"),
                    "data_seed": data_seed,
                    "algorithm_seed": algorithm_seed,
                    "case": case,
                    "irmf_summary": method_result_summary(irmf),
                })

    for method_name in methods:
        method_name = str(method_name).upper()
        if method_name == "EMD":
            param_grid = [
                {
                    "nbsym": int(nbsym),
                    "spline_kind": str(spline_kind),
                    "max_imf": int(max_imf),
                    "std_thr": std_thr,
                    "svar_thr": svar_thr,
                    "total_power_thr": total_power_thr,
                    "range_thr": range_thr,
                    "trials": None,
                    "noise_width": None,
                    "epsilon": None,
                }
                for nbsym in emd_grid["nbsym_options"]
                for spline_kind in emd_grid["spline_kind_options"]
                for max_imf in emd_grid["max_imf_options"]
                for std_thr in emd_grid["std_thr_options"]
                for svar_thr in emd_grid["svar_thr_options"]
                for total_power_thr in emd_grid["total_power_thr_options"]
                for range_thr in emd_grid["range_thr_options"]
            ]
        elif method_name == "EEMD":
            param_grid = [{"trials": int(t), "noise_width": float(w), "epsilon": None}
                          for t in trials_grid for w in noise_width_grid]
        elif method_name == "CEEMDAN":
            param_grid = [{"trials": int(t), "noise_width": None, "epsilon": float(e)}
                          for t in trials_grid for e in epsilon_grid]
        else:
            param_grid = []
        for params in param_grid:
            trials = params["trials"]
            noise_width = params.get("noise_width")
            epsilon = params.get("epsilon")
            for cached in case_cache:
                case = cached["case"]
                true_components = case.get("true_components", None)
                row = {
                    "appendix": "D",
                    "mode": mode,
                    "baseline_method": method_name,
                    "nbsym": params.get("nbsym"),
                    "spline_kind": params.get("spline_kind"),
                    "max_imf": params.get("max_imf"),
                    "std_thr": params.get("std_thr"),
                    "svar_thr": params.get("svar_thr"),
                    "total_power_thr": params.get("total_power_thr"),
                    "range_thr": params.get("range_thr"),
                    "trials": int(trials) if trials is not None else None,
                    "noise_width": float(noise_width) if noise_width is not None else None,
                    "epsilon": float(epsilon) if epsilon is not None else None,
                    "signal": cached["signal"],
                    "noise": cached["noise"],
                    "sigma": cached["sigma"],
                    "target_snr_db": cached.get("target_snr_db"),
                    "noise_severity_design": cached.get("noise_severity_design"),
                    "realized_input_snr_db": cached.get("realized_input_snr_db"),
                    "data_seed": cached["data_seed"],
                    "algorithm_seed": cached["algorithm_seed"],
                    "IRMF": cached["irmf_summary"],
                }
                try:
                    if method_name == "EMD":
                        result = run_fixed_emd_case(
                            Y=case["Y"],
                            X_clean=case["X_clean"],
                            t=case["t"],
                            fs=fs,
                            emd_params={**GLOBAL_EMD_PARAMS, **{
                                "nbsym": params.get("nbsym"),
                                "spline_kind": params.get("spline_kind"),
                                "max_imf": params.get("max_imf"),
                                "std_thr": params.get("std_thr"),
                                "svar_thr": params.get("svar_thr"),
                                "total_power_thr": params.get("total_power_thr"),
                                "range_thr": params.get("range_thr"),
                            }},
                            true_components=true_components,
                            contamination_mask=case.get("contamination_mask"),
                            run_id=(
                                f"appendix_D_emd_nbsym_{params.get('nbsym')}"
                                f"_spline_{params.get('spline_kind')}"
                                f"_maximf_{params.get('max_imf')}"
                                f"_std_{params.get('std_thr')}"
                                f"_{cached['signal']}_{cached['noise']}"
                                f"_snr{float(cached.get('target_snr_db')):g}dB"
                            ),
                        )
                    elif method_name == "EEMD":
                        raw = run_eemd(
                            case["Y"],
                            max_imf=emd_params.get("max_imf", -1),
                            trials=int(trials),
                            noise_width=float(noise_width),
                            random_seed=algorithm_seed,
                        )
                    elif method_name == "CEEMDAN":
                        raw = run_ceemdan(
                            case["Y"],
                            max_imf=emd_params.get("max_imf", -1),
                            trials=int(trials),
                            epsilon=float(epsilon),
                            random_seed=algorithm_seed,
                        )
                    else:
                        raise ValueError(f"Unsupported sensitivity method: {method_name}")

                    if method_name != "EMD":
                        physical = evaluate_shared_physical_diagnostics(
                            Y_observed=case["Y"],
                            X_clean=case["X_clean"],
                            imfs=raw["imfs"],
                            residual=raw["residual"],
                            fs=fs,
                            residual_penalty_mode="none",
                            true_components=true_components,
                            contamination_mask=case.get("contamination_mask"),
                        )
                        result = {
                            "method": method_name,
                            "run_id": (
                                f"appendix_D_{method_name.lower()}_trials_{trials}"
                                f"_scale_{noise_width if noise_width is not None else epsilon}"
                                f"_{cached['signal']}_{cached['noise']}"
                                f"_snr{float(cached.get('target_snr_db')):g}dB"
                            ),
                            **raw,
                            **physical,
                        }
                    row["baseline"] = method_result_summary(result)
                except Exception as exc:
                    row["baseline"] = {"method": method_name, "error": str(exc)}
                rows.append(row)

    aggregate = _baseline_sensitivity_aggregate(rows)
    protocol = {
        "appendix": "D",
        "section": "EMD-family baseline sensitivity",
        "mode": mode,
        "design": cfg,
        "noise_severity_design": "target_snr_energy_ratio",
        "target_snr_db_levels": list(target_snr_db_levels),
        "sigma_role": "snr_equivalent_label_not_fixed_noise_amplitude",
        "methods": list(methods),
        "emd_grid": emd_grid,
        "trials_grid": list(trials_grid),
        "noise_width_grid": list(noise_width_grid),
        "epsilon_grid": list(epsilon_grid),
        "n_cases": int(len(case_cache)),
        "n_baseline_configurations": int(
            (
                len(emd_grid["nbsym_options"])
                * len(emd_grid["spline_kind_options"])
                * len(emd_grid["max_imf_options"])
                * len(emd_grid["std_thr_options"])
                * len(emd_grid["svar_thr_options"])
                * len(emd_grid["total_power_thr_options"])
                * len(emd_grid["range_thr_options"])
            )
            + len(trials_grid) * len(noise_width_grid)
            + len(trials_grid) * len(epsilon_grid)
        ),
        "n_rows": int(len(rows)),
        "irmf_params": dict(irmf_params),
        "emd_params": dict(emd_params),
        "interpretation": (
            "Each EEMD/CEEMDAN configuration is fixed across all representative "
            "cases, and each EMD protocol is fixed across all representative "
            "cases. Results assess baseline-parameter sensitivity and are not "
            "used for per-case tuning."
        ),
    }
    write_json(protocol, output_root / "appendix_D_emd_family_sensitivity_protocol.json")
    write_json(rows, output_root / "appendix_D_emd_family_sensitivity.json")
    write_csv(rows, output_root / "appendix_D_emd_family_sensitivity.csv")
    write_json(aggregate, output_root / "appendix_D_emd_family_sensitivity_aggregate.json")
    write_csv(aggregate["by_configuration"], output_root / "appendix_D_emd_family_sensitivity_by_configuration.csv")
    return aggregate


def run_emd_family_benchmark(
        output_root,
        mode="representative",
        search_mode="quick",
        n=500,
        fs=500.0,
        seed=0,
):
    if mode not in EMD_FAMILY_BENCHMARK_MODES:
        raise ValueError(f"Unknown EMD family benchmark mode: {mode}")

    cfg = EMD_FAMILY_BENCHMARK_MODES[mode]
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary = []

    for signal_name in cfg["signals"]:
        for noise_name in cfg["noises"]:
            for sigma in cfg["sigmas"]:
                case_dir = output_root / f"emd_family_{mode}" / signal_name / noise_name / f"sigma_{sigma}"
                case_dir.mkdir(parents=True, exist_ok=True)

                case = make_signal_noise_case(signal_name, noise_name, sigma, n=n, fs=fs, seed=seed)
                t, Y, X_clean = case["t"], case["Y"], case["X_clean"]
                true_components = case.get("true_components", None)

                print("\n" + "#" * 120)
                print(f"EMD FAMILY BENCHMARK | mode={mode} | signal={signal_name} | noise={noise_name} | sigma={sigma}")
                print("#" * 120)

                _, irmf_best = run_single_irmf_case(
                    Y=Y, X_clean=X_clean, t=t, fs=fs,
                    output_dir=case_dir / "irmf",
                    search_mode=search_mode,
                    true_components=true_components,
                )
                _, emd_best = run_single_emd_case(
                    Y=Y, X_clean=X_clean, t=t, fs=fs,
                    output_dir=case_dir / "emd",
                    true_components=true_components,
                )

                rows = [("IRMF", irmf_best), ("EMD", emd_best)]

                for method in ("EEMD", "CEEMDAN"):
                    try:
                        rows.append((method, _evaluate_family_method(method, Y, X_clean, fs, true_components)))
                    except Exception as exc:
                        print(f"{method} skipped: {exc}")

                print_extended_comparison(
                    f"EMD FAMILY EXTENDED | {signal_name} | {noise_name} | sigma={sigma}",
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

    with open(output_root / f"emd_family_benchmark_{mode}_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2)

    return summary


# Backward-compatible alias
def run_emd_family_multisigma_benchmark(*args, **kwargs):
    return run_emd_family_benchmark(*args, **kwargs)
