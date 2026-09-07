#!/usr/bin/python
# coding: UTF-8

"""Section 6: Robustness and Sensitivity Analyses."""

from pathlib import Path
from collections import defaultdict
import itertools
import json

import numpy as np

from project_config import (
    BOUNDARY_SENSITIVITY_NOISES,
    BOUNDARY_SENSITIVITY_SIGMAS,
    BOUNDARY_SENSITIVITY_SIGNALS,
    BOUNDARY_SENSITIVITY_TARGET_SNR_DB_LEVELS,
    CONTAMINATION_LAMBDA_GRID,
    CONTAMINATION_ROBUSTNESS_SIGMA,
    CONTAMINATION_ROBUSTNESS_SIGNALS,
    CONTAMINATION_ROBUSTNESS_TARGET_SNR_DB,
    DEFAULT_FS,
    DEFAULT_N,
    DEFAULT_SEED,
    GLOBAL_CEEMDAN_PARAMS,
    GLOBAL_EEMD_PARAMS,
    GLOBAL_EMD_PARAMS,
    GLOBAL_IRMF_PARAMS,
    NOISE_ROBUSTNESS_NOISES,
    NOISE_ROBUSTNESS_SIGMAS,
    NOISE_ROBUSTNESS_SIGNALS,
    NOISE_ROBUSTNESS_TARGET_SNR_DB_LEVELS,
    PARAMETER_SENSITIVITY_FACTORS,
    PARAMETER_SENSITIVITY_LEVEL_LABELS,
    PARAMETER_SENSITIVITY_NOISES,
    PARAMETER_SENSITIVITY_SIGMAS,
    PARAMETER_SENSITIVITY_SIGNALS,
    PARAMETER_SENSITIVITY_TARGET_SNR_DB_LEVELS,
    RANDOM_SEED_STABILITY_NOISES,
    RANDOM_SEED_STABILITY_SEEDS,
    RANDOM_SEED_STABILITY_SIGMA,
    RANDOM_SEED_STABILITY_SIGNALS,
    RANDOM_SEED_STABILITY_TARGET_SNR_DB,
    EVALUATION_METHODS,
    LOWER_IS_BETTER_METRICS,
)
from experiments.experiment_utils import (
    make_signal_noise_case,
    method_result_summary,
    run_fixed_method_family_case,
    run_fixed_irmf_case,
    run_fixed_pair_case,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json, write_section_outputs
from diagnostics.shared_physical_diagnostics import SYNTHETIC_RECONSTRUCTION_PROTOCOL_ID


def _finite(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return None


def _contamination_degradation_rows(rows):
    methods = tuple(EVALUATION_METHODS)
    metrics = (
        "denoise_nmse",
        "outlier_resistance_index",
        "clean_region_nmse",
        "contaminated_region_nmse",
        "contamination_spillover_error",
        "noise_capture_corr",
        "contamination_resistance_score",
    )
    lower_is_better = set(LOWER_IS_BETTER_METRICS)
    out = []
    for signal_name in sorted({row.get("signal") for row in rows}):
        signal_rows = [row for row in rows if row.get("signal") == signal_name]
        for method in methods:
            for metric in metrics:
                points = []
                for row in signal_rows:
                    lam = _finite(row.get("lambda"))
                    val = _finite(row.get(method, {}).get(metric))
                    if lam is not None and val is not None:
                        points.append((lam, val))
                if len(points) < 2:
                    continue
                points = sorted(points)
                x = np.asarray([p[0] for p in points], dtype=float)
                y = np.asarray([p[1] for p in points], dtype=float)
                if np.max(x) - np.min(x) <= 1e-12:
                    continue
                beta = np.polyfit(x, y, deg=1)
                baseline = y[0]
                terminal = y[-1]
                raw_change = terminal - baseline
                degradation = raw_change if metric in lower_is_better else -raw_change
                relative_degradation = degradation / (abs(baseline) + 1e-12)
                slope = float(beta[0])
                degradation_slope = slope if metric in lower_is_better else -slope
                out.append({
                    "signal": signal_name,
                    "method": method,
                    "metric": metric,
                    "n_lambda_levels": int(len(points)),
                    "lambda_min": float(np.min(x)),
                    "lambda_max": float(np.max(x)),
                    "baseline_value_at_lambda_min": float(baseline),
                    "terminal_value_at_lambda_max": float(terminal),
                    "raw_change_terminal_minus_baseline": float(raw_change),
                    "degradation_terminal_minus_baseline": float(degradation),
                    "relative_degradation": float(relative_degradation),
                    "linear_slope_per_lambda": slope,
                    "degradation_slope_per_lambda": float(degradation_slope),
                    "direction": "positive degradation means worse as contamination increases",
                })
    return out


def _aggregate_contamination_degradation(rows):
    grouped = {}
    for row in rows:
        key = (row["method"], row["metric"])
        grouped.setdefault(key, []).append(row)
    out = []
    for (method, metric), vals in sorted(grouped.items()):
        slopes = np.asarray([v["degradation_slope_per_lambda"] for v in vals], dtype=float)
        rel = np.asarray([v["relative_degradation"] for v in vals], dtype=float)
        out.append({
            "method": method,
            "metric": metric,
            "n_signals": int(len(vals)),
            "mean_degradation_slope_per_lambda": float(np.mean(slopes)),
            "median_degradation_slope_per_lambda": float(np.median(slopes)),
            "mean_relative_degradation": float(np.mean(rel)),
            "median_relative_degradation": float(np.median(rel)),
        })
    return out


def _contamination_endpoint_specs():
    return {
        "denoise_nmse": {"kind": "loss", "source_metric": "denoise_nmse"},
        "clean_region_nmse": {"kind": "loss", "source_metric": "clean_region_nmse"},
        "contaminated_region_nmse": {"kind": "loss", "source_metric": "contaminated_region_nmse"},
        "contamination_spillover_error": {"kind": "loss", "source_metric": "contamination_spillover_error"},
        "signal_leakage_into_noise": {"kind": "loss", "source_metric": "signal_leakage_into_noise"},
        "clean_region_signal_leakage": {"kind": "loss", "source_metric": "clean_region_signal_leakage"},
        "imf_recovery_nrmse": {"kind": "loss", "source_metric": "imf_recovery_nrmse"},
        "component_splitting_index": {"kind": "loss", "source_metric": "component_splitting_index"},
        "component_merging_index": {"kind": "loss", "source_metric": "component_merging_index"},
        "inter_imf_entanglement_index": {"kind": "loss", "source_metric": "inter_imf_entanglement_index"},
        "outlier_resistance_index": {"kind": "score_complement", "source_metric": "outlier_resistance_index"},
        "noise_capture_corr_score": {"kind": "score_complement", "source_metric": "noise_capture_corr_score"},
    }


def _endpoint_loss_value(value, spec):
    value = _finite(value)
    if value is None:
        return None
    if spec["kind"] == "loss":
        return float(value)
    if spec["kind"] == "score_complement":
        return float(1.0 - np.clip(value, 0.0, 1.0))
    return None


def _contamination_audc_rows(rows, expected_lambdas=CONTAMINATION_LAMBDA_GRID):
    """
    Area under the contamination degradation curve by endpoint.

    The normalized AUDC is based on loss increase relative to lambda=0:
    integral [(L(lambda)-L(0))/max(L(0), eps_L)] d lambda / range(lambda).
    It is endpoint-specific and is not folded into case_score.
    """
    specs = _contamination_endpoint_specs()
    expected_lambdas = tuple(float(x) for x in expected_lambdas)

    baseline_losses = defaultdict(list)
    for row in rows:
        lam = _finite(row.get("lambda"))
        if lam is None or abs(lam - min(expected_lambdas, default=0.0)) > 1e-12:
            continue
        for method in EVALUATION_METHODS:
            data = row.get(method, {})
            if not isinstance(data, dict):
                continue
            for endpoint, spec in specs.items():
                loss = _endpoint_loss_value(data.get(spec["source_metric"]), spec)
                if loss is not None and loss > 0:
                    baseline_losses[endpoint].append(loss)

    denom_floor = {}
    for endpoint in specs:
        vals = np.asarray(baseline_losses.get(endpoint, []), dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size:
            denom_floor[endpoint] = float(max(1e-12, 1e-6 * np.median(vals)))
        else:
            denom_floor[endpoint] = 1e-12

    grouped = defaultdict(list)
    for row in rows:
        lam = _finite(row.get("lambda"))
        if lam is None:
            continue
        group_base = (
            row.get("signal"),
            row.get("noise", "huber_contamination"),
            _finite(row.get("sigma")),
            row.get("seed"),
        )
        for method in EVALUATION_METHODS:
            data = row.get(method, {})
            if not isinstance(data, dict):
                continue
            for endpoint, spec in specs.items():
                loss = _endpoint_loss_value(data.get(spec["source_metric"]), spec)
                if loss is not None:
                    grouped[group_base + (method, endpoint)].append((lam, loss))

    out = []
    expected_set = {round(x, 12) for x in expected_lambdas}
    for key, points in sorted(grouped.items()):
        signal_name, noise_name, sigma, seed, method, endpoint = key
        if len(points) < 2:
            continue
        points = sorted(points)
        dedup = {}
        for lam, loss in points:
            dedup[float(lam)] = float(loss)
        x = np.asarray(sorted(dedup.keys()), dtype=float)
        y = np.asarray([dedup[lam] for lam in x], dtype=float)
        if len(x) < 2 or np.max(x) - np.min(x) <= 1e-12:
            continue

        baseline_idx = int(np.argmin(np.abs(x - np.min(x))))
        baseline = float(y[baseline_idx])
        absolute_degradation = y - baseline
        denominator = max(abs(baseline), denom_floor[endpoint])
        normalized_degradation = absolute_degradation / denominator
        lambda_range = float(np.max(x) - np.min(x))
        absolute_audc = float(np.trapz(absolute_degradation, x) / lambda_range)
        normalized_audc = float(np.trapz(normalized_degradation, x) / lambda_range)
        slope = float(np.polyfit(x, absolute_degradation, deg=1)[0]) if len(x) >= 2 else np.nan
        normalized_slope = float(np.polyfit(x, normalized_degradation, deg=1)[0]) if len(x) >= 2 else np.nan
        observed_set = {round(float(v), 12) for v in x}

        out.append({
            "signal": signal_name,
            "noise": noise_name,
            "sigma": sigma,
            "seed": seed,
            "method": method,
            "endpoint": endpoint,
            "source_metric": specs[endpoint]["source_metric"],
            "endpoint_kind": specs[endpoint]["kind"],
            "n_lambda_levels": int(len(x)),
            "lambda_min": float(np.min(x)),
            "lambda_max": float(np.max(x)),
            "baseline_loss_at_lambda_min": baseline,
            "terminal_loss_at_lambda_max": float(y[-1]),
            "denominator_floor": float(denom_floor[endpoint]),
            "normalized_contamination_audc": normalized_audc,
            "absolute_contamination_audc": absolute_audc,
            "absolute_degradation_slope": slope,
            "normalized_degradation_slope": normalized_slope,
            "complete_lambda_trajectory": bool(observed_set == expected_set),
            "has_lambda_zero_baseline": bool(any(abs(v) <= 1e-12 for v in x)),
            "audc_role": "endpoint_specific_contamination_trajectory_summary",
            "direction": "lower AUDC means less degradation under increasing contamination",
        })
    return out


def _aggregate_contamination_audc(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["endpoint"], row["method"])].append(row)
    out = []
    for (endpoint, method), vals in sorted(grouped.items()):
        norm = np.asarray([v["normalized_contamination_audc"] for v in vals], dtype=float)
        abs_auc = np.asarray([v["absolute_contamination_audc"] for v in vals], dtype=float)
        complete = [bool(v.get("complete_lambda_trajectory")) for v in vals]
        out.append({
            "endpoint": endpoint,
            "method": method,
            "n_trajectories": int(len(vals)),
            "mean_normalized_contamination_audc": float(np.mean(norm)),
            "median_normalized_contamination_audc": float(np.median(norm)),
            "mean_absolute_contamination_audc": float(np.mean(abs_auc)),
            "median_absolute_contamination_audc": float(np.median(abs_auc)),
            "complete_trajectory_fraction": float(np.mean(complete)) if complete else np.nan,
            "direction": "lower is better",
        })
    return out


def _contamination_audc_rank_rows(audc_rows):
    out = []
    grouped = defaultdict(dict)
    for row in audc_rows:
        key = (row["signal"], row["noise"], row["sigma"], row["seed"], row["endpoint"])
        grouped[key][row["method"]] = row["normalized_contamination_audc"]
    for key, values in sorted(grouped.items()):
        items = sorted(
            [(method, value) for method, value in values.items() if np.isfinite(value)],
            key=lambda item: item[1],
        )
        for rank, (method, value) in enumerate(items, start=1):
            out.append({
                "signal": key[0],
                "noise": key[1],
                "sigma": key[2],
                "seed": key[3],
                "endpoint": key[4],
                "method": method,
                "normalized_contamination_audc": float(value),
                "audc_rank": int(rank),
                "direction": "rank 1 has the lowest normalized contamination AUDC",
            })
    return out


def _write_contamination_audc_plot(audc_aggregate, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None


def _sigma_label_for_snr(target_snr_db):
    return float(10.0 ** (-float(target_snr_db) / 20.0))


def _load_jsonl_rows(path):
    path = Path(path)
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
    return rows


def _append_jsonl_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True))
        f.write("\n")


def _read_json_if_exists(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    endpoints = [
        "denoise_nmse",
        "clean_region_nmse",
        "signal_leakage_into_noise",
        "outlier_resistance_index",
        "noise_capture_corr_score",
    ]
    methods = tuple(EVALUATION_METHODS)
    lookup = {
        (row.get("endpoint"), row.get("method")): row.get("median_normalized_contamination_audc")
        for row in audc_aggregate
    }
    x = np.arange(len(endpoints), dtype=float)
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(max(8.0, 1.15 * len(endpoints)), 4.8))
    for idx, method in enumerate(methods):
        vals = [
            _finite(lookup.get((endpoint, method))) if lookup.get((endpoint, method)) is not None else np.nan
            for endpoint in endpoints
        ]
        ax.bar(x + (idx - (len(methods) - 1) / 2.0) * width, vals, width=width, label=method)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(endpoints, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Median normalized contamination AUDC")
    ax.set_title("Endpoint-Specific Contamination Degradation")
    ax.legend(fontsize=8, ncol=min(len(methods), 4))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def run_noise_robustness(
        output_root,
        irmf_params=GLOBAL_IRMF_PARAMS,
        emd_params=GLOBAL_EMD_PARAMS,
        eemd_params=GLOBAL_EEMD_PARAMS,
        ceemdan_params=GLOBAL_CEEMDAN_PARAMS,
        signals=NOISE_ROBUSTNESS_SIGNALS,
        noises=NOISE_ROBUSTNESS_NOISES,
        sigmas=NOISE_ROBUSTNESS_SIGMAS,
        target_snr_db_levels=NOISE_ROBUSTNESS_TARGET_SNR_DB_LEVELS,
        n=DEFAULT_N,
        fs=DEFAULT_FS,
        seed=DEFAULT_SEED,
        algorithm_seed=20260715,
):
    output_root = ensure_dir(output_root)
    rows = []
    for signal_name in signals:
        for noise_name in noises:
            for target_snr_db in target_snr_db_levels:
                sigma = _sigma_label_for_snr(target_snr_db)
                _, _, row = run_fixed_method_family_case(
                    signal_name=signal_name,
                    noise_name=noise_name,
                    sigma=sigma,
                    target_snr_db=target_snr_db,
                    irmf_params=irmf_params,
                    emd_params=emd_params,
                    eemd_params=eemd_params,
                    ceemdan_params=ceemdan_params,
                    n=n,
                    fs=fs,
                    seed=seed,
                    algorithm_seed=algorithm_seed,
                    run_id_prefix=f"noise_robustness_{signal_name}_{noise_name}_snr{float(target_snr_db):g}dB",
                )
                rows.append(row)
    write_json({
        "section": "6.1 Noise Robustness",
        "x_axis": "target_snr_db",
        "noise_severity_design": "target_snr_energy_ratio",
        "target_snr_db_levels": list(target_snr_db_levels),
        "sigma_role": "snr_equivalent_label_not_fixed_noise_amplitude",
        "main_plot": "noise robustness curve",
        "signals": list(signals),
        "noises": list(noises),
        "sigmas": list(sigmas),
        "methods": ["IRMF", "EMD", "EEMD", "CEEMDAN"],
        "comparison_protocol": "All four methods are evaluated on the same signal-noise-sigma cells with globally locked parameters.",
    }, output_root / "section_6_1_protocol.json")
    return rows, write_section_outputs(rows, output_root, "section_6_1_noise_robustness")


def run_contamination_robustness(
        output_root,
        irmf_params=GLOBAL_IRMF_PARAMS,
        emd_params=GLOBAL_EMD_PARAMS,
        eemd_params=GLOBAL_EEMD_PARAMS,
        ceemdan_params=GLOBAL_CEEMDAN_PARAMS,
        signals=CONTAMINATION_ROBUSTNESS_SIGNALS,
        lambdas=CONTAMINATION_LAMBDA_GRID,
        sigma=CONTAMINATION_ROBUSTNESS_SIGMA,
        target_snr_db=CONTAMINATION_ROBUSTNESS_TARGET_SNR_DB,
        n=DEFAULT_N,
        fs=DEFAULT_FS,
        seed=DEFAULT_SEED,
        algorithm_seed=20260715,
):
    output_root = ensure_dir(output_root)
    rows = []
    for signal_name in signals:
        for lam in lambdas:
            _, _, row = run_fixed_method_family_case(
                signal_name=signal_name,
                noise_name="huber_contamination",
                sigma=sigma,
                target_snr_db=target_snr_db,
                irmf_params=irmf_params,
                emd_params=emd_params,
                eemd_params=eemd_params,
                ceemdan_params=ceemdan_params,
                n=n,
                fs=fs,
                seed=seed,
                algorithm_seed=algorithm_seed,
                noise_kwargs={"lam": lam},
                run_id_prefix=f"contamination_{signal_name}_lambda{lam}_snr{float(target_snr_db):g}dB",
            )
            row["lambda"] = lam
            rows.append(row)
    write_json({
        "section": "6.2 Contamination Robustness",
        "x_axis": "Huber contamination rate lambda",
        "noise_severity_design": "target_snr_energy_ratio",
        "target_snr_db": float(target_snr_db),
        "sigma_role": "snr_equivalent_label_not_fixed_noise_amplitude",
        "sigma": sigma,
        "lambdas": list(lambdas),
        "signals": list(signals),
        "methods": ["IRMF", "EMD", "EEMD", "CEEMDAN"],
        "comparison_protocol": "All four methods are evaluated on the same contamination-rate cells with globally locked parameters.",
    }, output_root / "section_6_2_protocol.json")
    degradation_rows = _contamination_degradation_rows(rows)
    degradation_aggregate = _aggregate_contamination_degradation(degradation_rows)
    audc_rows = _contamination_audc_rows(rows, expected_lambdas=lambdas)
    audc_aggregate = _aggregate_contamination_audc(audc_rows)
    audc_rank_rows = _contamination_audc_rank_rows(audc_rows)
    write_csv(degradation_rows, output_root / "section_6_2_contamination_degradation_slopes.csv")
    write_json(degradation_rows, output_root / "section_6_2_contamination_degradation_slopes.json")
    write_csv(degradation_aggregate, output_root / "section_6_2_contamination_degradation_slope_aggregate.csv")
    write_json(degradation_aggregate, output_root / "section_6_2_contamination_degradation_slope_aggregate.json")
    write_csv(audc_rows, output_root / "section_6_2_contamination_audc_by_trajectory.csv")
    write_json(audc_rows, output_root / "section_6_2_contamination_audc_by_trajectory.json")
    write_csv(audc_aggregate, output_root / "section_6_2_contamination_audc_aggregate.csv")
    write_json(audc_aggregate, output_root / "section_6_2_contamination_audc_aggregate.json")
    write_csv(audc_rank_rows, output_root / "section_6_2_contamination_audc_method_ranks.csv")
    write_json(audc_rank_rows, output_root / "section_6_2_contamination_audc_method_ranks.json")
    audc_plot = _write_contamination_audc_plot(
        audc_aggregate,
        output_root / "section_6_2_contamination_audc_curves.png",
    )
    summary = write_section_outputs(rows, output_root, "section_6_2_contamination_robustness")
    summary["contamination_degradation_slope_output"] = {
        "rows": "section_6_2_contamination_degradation_slopes.csv",
        "aggregate": "section_6_2_contamination_degradation_slope_aggregate.csv",
    }
    summary["contamination_audc_output"] = {
        "rows": "section_6_2_contamination_audc_by_trajectory.csv",
        "aggregate": "section_6_2_contamination_audc_aggregate.csv",
        "method_ranks": "section_6_2_contamination_audc_method_ranks.csv",
        "plot": audc_plot,
        "role": "endpoint-specific robustness-curve summary; not used in case_score",
    }
    write_json(summary, output_root / "section_6_2_contamination_robustness_aggregate.json")
    return rows, summary


def _flatten_factorial_rows(rows):
    flat = []
    for row in rows:
        out = {k: v for k, v in row.items() if k != "IRMF"}
        for key, value in row.get("IRMF", {}).items():
            out[f"irmf_{key}"] = value
        flat.append(out)
    return flat


def _aggregate_factorial_sensitivity_rows(rows, parameter_names):
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
        "n_factorial_settings": int(len(set(row["factorial_design_id"] for row in rows))) if rows else 0,
        "parameter_names": list(parameter_names),
        "metrics": {},
        "by_factorial_setting": [],
    }

    for metric in metrics:
        vals = []
        for row in rows:
            value = row.get("IRMF", {}).get(metric)
            if value is not None and np.isfinite(value):
                vals.append(float(value))
        if vals:
            arr = np.asarray(vals, dtype=float)
            out["metrics"][metric] = {
                "mean": float(np.mean(arr)),
                "sd": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "n": int(len(arr)),
            }

    for design_id in sorted(set(row["factorial_design_id"] for row in rows)):
        subset = [row for row in rows if row["factorial_design_id"] == design_id]
        first = subset[0]
        item = {
            "factorial_design_id": design_id,
        }
        for name in parameter_names:
            item[f"{name}_factor"] = first[f"{name}_factor"]
            item[f"{name}_level"] = first[f"{name}_level"]
            item[f"{name}_code"] = first[f"{name}_code"]
            item[f"{name}_value"] = first[f"{name}_value"]
        for metric in metrics:
            vals = [
                float(row["IRMF"][metric])
                for row in subset
                if row.get("IRMF", {}).get(metric) is not None and np.isfinite(row["IRMF"][metric])
            ]
            if vals:
                item[f"mean_{metric}"] = float(np.mean(vals))
        out["by_factorial_setting"].append(item)

    return out


def _design_matrix(rows, parameter_names, terms):
    cols = [np.ones(len(rows), dtype=float)]
    names = ["intercept"]
    for term in terms:
        if ":" in term:
            left, right = term.split(":", 1)
            col = np.asarray([float(row[f"{left}_code"]) * float(row[f"{right}_code"]) for row in rows], dtype=float)
        else:
            col = np.asarray([float(row[f"{term}_code"]) for row in rows], dtype=float)
        cols.append(col)
        names.append(term)
    return np.column_stack(cols), names


def _fit_sse(y, x):
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    return float(np.sum(resid ** 2)), beta


SECTION_6_1_PRIMARY_ENDPOINT_SPECS = [
    {
        "primary_endpoint": "denoise_nmse",
        "source_metric": "denoise_nmse",
        "direction": "lower_is_better",
        "construct": "Signal Recovery",
        "transform": "log1p_nonnegative",
    },
    {
        "primary_endpoint": "denoise_corr",
        "source_metric": "denoise_corr",
        "direction": "higher_is_better",
        "construct": "Signal Recovery",
        "transform": "fisher_z_correlation",
    },
    {
        "primary_endpoint": "matched_component_corr",
        "source_metric": "imf_recovery_corr",
        "direction": "higher_is_better",
        "construct": "Component Recovery Quality",
        "transform": "fisher_z_correlation",
    },
    {
        "primary_endpoint": "matched_component_nrmse",
        "source_metric": "imf_recovery_nrmse",
        "direction": "lower_is_better",
        "construct": "Component Recovery Quality",
        "transform": "log1p_nonnegative",
    },
    {
        "primary_endpoint": "relative_decomposition_count_error",
        "source_metric": "relative_decomposition_count_error",
        "direction": "lower_is_better",
        "construct": "Component-Set Fidelity",
        "transform": "identity",
    },
    {
        "primary_endpoint": "missing_true_component_energy_ratio",
        "source_metric": "missing_true_component_energy_ratio",
        "direction": "lower_is_better",
        "construct": "Component-Set Fidelity",
        "transform": "identity_zero_inflated",
    },
    {
        "primary_endpoint": "spurious_mode_energy_ratio",
        "source_metric": "spurious_mode_energy_ratio",
        "direction": "lower_is_better",
        "construct": "Component-Set Fidelity",
        "transform": "identity_zero_inflated",
    },
    {
        "primary_endpoint": "noise_capture_corr",
        "source_metric": "noise_capture_corr",
        "direction": "higher_is_better",
        "construct": "Noise Separation",
        "transform": "fisher_z_correlation",
    },
    {
        "primary_endpoint": "noise_energy_log_error",
        "source_metric": "noise_energy_log_error",
        "direction": "lower_is_better",
        "construct": "Noise Separation",
        "transform": "log1p_nonnegative",
    },
    {
        "primary_endpoint": "signal_leakage_into_noise",
        "source_metric": "signal_leakage_into_noise",
        "direction": "lower_is_better",
        "construct": "Noise Separation",
        "transform": "log1p_nonnegative",
    },
]


def _section6_case_id(row):
    return (
        f"{row.get('signal')}|{row.get('noise')}|"
        f"{float(row.get('target_snr_db')):.12g}|seed{int(row.get('seed', 0))}"
    )


def _transform_primary_value(value, transform):
    value = _finite(value)
    if value is None:
        return None
    if transform == "fisher_z_correlation":
        return float(np.arctanh(np.clip(value, -0.999999, 0.999999)))
    if transform == "log1p_nonnegative":
        return float(np.log1p(max(0.0, value)))
    return float(value)


def _term_names(parameter_names):
    main_terms = list(parameter_names)
    interaction_terms = [
        f"{a}:{b}" for i, a in enumerate(parameter_names) for b in parameter_names[i + 1:]
    ]
    return main_terms + interaction_terms


def _term_value(row, term):
    if ":" in term:
        left, right = term.split(":", 1)
        return float(row[f"{left}_code"]) * float(row[f"{right}_code"])
    return float(row[f"{term}_code"])


def _within_case_residualize(values, case_ids):
    values = np.asarray(values, dtype=float)
    out = values.copy()
    grouped = defaultdict(list)
    for idx, case_id in enumerate(case_ids):
        grouped[case_id].append(idx)
    for indices in grouped.values():
        out[indices] -= float(np.mean(values[indices]))
    return out


def _fit_case_blocked_factorial(data, parameter_names, response_key="transformed_value"):
    terms = _term_names(parameter_names)
    y = np.asarray([float(row[response_key]) for row in data], dtype=float)
    case_ids = [_section6_case_id(row) for row in data]
    y_blocked = _within_case_residualize(y, case_ids)
    sst_blocked = float(np.sum(y_blocked ** 2))
    if sst_blocked <= 1e-18:
        return {
            "model": "case-blocked fixed-effect factorial regression",
            "r_squared_case_blocked": 0.0,
            "terms": [],
            "note": "within-case response has near-zero variance",
        }

    x_cols = []
    for term in terms:
        raw_col = np.asarray([_term_value(row, term) for row in data], dtype=float)
        x_cols.append(_within_case_residualize(raw_col, case_ids))
    x_full = np.column_stack(x_cols)
    sse_full, beta_full = _fit_sse(y_blocked, x_full)
    r2 = float(max(0.0, 1.0 - sse_full / sst_blocked))
    term_rows = []
    for idx, term in enumerate(terms):
        keep = [j for j in range(len(terms)) if j != idx]
        x_reduced = x_full[:, keep] if keep else np.zeros((len(data), 0), dtype=float)
        if x_reduced.shape[1] == 0:
            sse_reduced = sst_blocked
        else:
            sse_reduced, _ = _fit_sse(y_blocked, x_reduced)
        partial_ss = max(0.0, float(sse_reduced - sse_full))
        term_rows.append({
            "term": term,
            "type": "interaction" if ":" in term else "main_effect",
            "coefficient": float(beta_full[idx]),
            "partial_ss_case_blocked": partial_ss,
            "partial_r2_case_blocked": float(partial_ss / sst_blocked),
        })
    term_rows = sorted(term_rows, key=lambda item: item["partial_r2_case_blocked"], reverse=True)
    return {
        "model": "case-blocked fixed-effect factorial regression",
        "r_squared_case_blocked": r2,
        "sse_full_case_blocked": float(sse_full),
        "sst_case_blocked": sst_blocked,
        "terms": term_rows,
    }


def _raw_setting_medians(data, source_metric):
    grouped = defaultdict(list)
    for row in data:
        value = _finite(row.get("IRMF", {}).get(source_metric))
        if value is not None:
            grouped[int(row["factorial_design_id"])].append(value)
    medians = {}
    for design_id, vals in grouped.items():
        if vals:
            medians[design_id] = float(np.median(np.asarray(vals, dtype=float)))
    return medians


def _default_design_id(rows, parameter_names):
    for row in rows:
        if all(int(row.get(f"{name}_code", 999)) == 0 for name in parameter_names):
            return int(row["factorial_design_id"])
    return None


def _endpoint_stability_summary(data, source_metric, direction, parameter_names):
    vals = [
        _finite(row.get("IRMF", {}).get(source_metric))
        for row in data
    ]
    vals = [v for v in vals if v is not None]
    if not vals:
        return {}
    arr = np.asarray(vals, dtype=float)
    setting_medians = _raw_setting_medians(data, source_metric)
    default_id = _default_design_id(data, parameter_names)
    default_median = setting_medians.get(default_id) if default_id is not None else None
    med_values = list(setting_medians.values())
    if direction == "higher_is_better":
        best = max(med_values) if med_values else None
        oracle_regret = None if best is None or default_median is None else float(best - default_median)
        default_percentile = (
            None if default_median is None or not med_values
            else float(np.mean(np.asarray(med_values) <= default_median))
        )
        tolerance = max(1e-12, 0.05 * abs(best)) if best is not None else None
        plateau_width = (
            None if tolerance is None or best is None or not med_values
            else float(np.mean(np.asarray(med_values) >= best - tolerance))
        )
    else:
        best = min(med_values) if med_values else None
        oracle_regret = None if best is None or default_median is None else float(default_median - best)
        default_percentile = (
            None if default_median is None or not med_values
            else float(np.mean(np.asarray(med_values) >= default_median))
        )
        tolerance = max(1e-12, 0.05 * abs(best)) if best is not None else None
        plateau_width = (
            None if tolerance is None or best is None or not med_values
            else float(np.mean(np.asarray(med_values) <= best + tolerance))
        )
    return {
        "n_finite_raw": int(len(arr)),
        "raw_median": float(np.median(arr)),
        "raw_iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "raw_p90": float(np.percentile(arr, 90)),
        "raw_p95": float(np.percentile(arr, 95)),
        "raw_zero_fraction": float(np.mean(np.abs(arr) <= 1e-12)),
        "raw_nonzero_rate": float(np.mean(np.abs(arr) > 1e-12)),
        "n_factorial_setting_medians": int(len(med_values)),
        "default_factorial_design_id": default_id,
        "default_setting_median": default_median,
        "best_setting_median": best,
        "default_oracle_regret": oracle_regret,
        "default_percentile_directional": default_percentile,
        "plateau_width_fraction_5pct_of_best": plateau_width,
    }


def _main_effect_level_rows(data, spec, parameter_names):
    out = []
    metric = spec["source_metric"]
    for name in parameter_names:
        for level in sorted({row[f"{name}_level"] for row in data}):
            vals = [
                _finite(row.get("IRMF", {}).get(metric))
                for row in data
                if row.get(f"{name}_level") == level
            ]
            vals = [v for v in vals if v is not None]
            if vals:
                arr = np.asarray(vals, dtype=float)
                out.append({
                    "primary_endpoint": spec["primary_endpoint"],
                    "source_metric": metric,
                    "parameter": name,
                    "level": level,
                    "n": int(len(arr)),
                    "median_raw": float(np.median(arr)),
                    "mean_raw": float(np.mean(arr)),
                    "p25_raw": float(np.percentile(arr, 25)),
                    "p75_raw": float(np.percentile(arr, 75)),
                })
    return out


def _interaction_grid_rows(data, spec, parameter_names):
    out = []
    metric = spec["source_metric"]
    for i, left in enumerate(parameter_names):
        for right in parameter_names[i + 1:]:
            for left_level in sorted({row[f"{left}_level"] for row in data}):
                for right_level in sorted({row[f"{right}_level"] for row in data}):
                    vals = [
                        _finite(row.get("IRMF", {}).get(metric))
                        for row in data
                        if row.get(f"{left}_level") == left_level
                        and row.get(f"{right}_level") == right_level
                    ]
                    vals = [v for v in vals if v is not None]
                    if vals:
                        arr = np.asarray(vals, dtype=float)
                        out.append({
                            "primary_endpoint": spec["primary_endpoint"],
                            "source_metric": metric,
                            "parameter_pair": f"{left}:{right}",
                            "left_parameter": left,
                            "right_parameter": right,
                            "left_level": left_level,
                            "right_level": right_level,
                            "n": int(len(arr)),
                            "median_raw": float(np.median(arr)),
                            "mean_raw": float(np.mean(arr)),
                        })
    return out


def _primary_endpoint_case_blocked_factorial_summary(rows, parameter_names):
    out = {
        "analysis_id": "section_6_1_primary_endpoint_case_blocked_factorial_sensitivity",
        "scope": "non-contamination primary endpoints for Section 6.1 IRMF parameter sensitivity",
        "n_rows_total": int(len(rows)),
        "n_factorial_settings": int(len(set(row["factorial_design_id"] for row in rows))) if rows else 0,
        "n_case_blocks": int(len(set(_section6_case_id(row) for row in rows))) if rows else 0,
        "parameter_names": list(parameter_names),
        "primary_endpoint_count": int(len(SECTION_6_1_PRIMARY_ENDPOINT_SPECS)),
        "model": (
            "For each endpoint, response values are transformed when appropriate, "
            "then both the response and coded factorial terms are demeaned within "
            "signal-noise-SNR-seed case blocks before least-squares fitting of "
            "main effects and all pairwise interactions."
        ),
        "claim_boundary": (
            "This is a descriptive sensitivity decomposition for the existing "
            "3^4 factorial rows. It supports endpoint-specific local parameter "
            "sensitivity claims, not universal absence of interactions."
        ),
        "metrics": {},
    }
    summary_rows = []
    term_rows = []
    main_effect_rows = []
    interaction_grid_rows = []
    for spec in SECTION_6_1_PRIMARY_ENDPOINT_SPECS:
        data = []
        for row in rows:
            transformed = _transform_primary_value(
                row.get("IRMF", {}).get(spec["source_metric"]),
                spec["transform"],
            )
            if transformed is not None and np.isfinite(transformed):
                item = dict(row)
                item["transformed_value"] = float(transformed)
                data.append(item)
        stability = _endpoint_stability_summary(data, spec["source_metric"], spec["direction"], parameter_names)
        model = _fit_case_blocked_factorial(data, parameter_names) if len(data) >= 3 else {
            "model": "case-blocked fixed-effect factorial regression",
            "r_squared_case_blocked": None,
            "terms": [],
            "note": "insufficient finite values",
        }
        strongest_main = None
        strongest_interaction = None
        if model.get("terms"):
            mains = [term for term in model["terms"] if term["type"] == "main_effect"]
            interactions = [term for term in model["terms"] if term["type"] == "interaction"]
            strongest_main = mains[0] if mains else None
            strongest_interaction = interactions[0] if interactions else None
        zero_fraction = stability.get("raw_zero_fraction")
        endpoint_record = {
            "primary_endpoint": spec["primary_endpoint"],
            "source_metric": spec["source_metric"],
            "construct": spec["construct"],
            "direction": spec["direction"],
            "transform": spec["transform"],
            "n_finite": int(len(data)),
            "case_blocked_model": model,
            "stability": stability,
            "strongest_main_effect": strongest_main,
            "strongest_2way_interaction": strongest_interaction,
            "interaction_materiality_flag": bool(
                strongest_interaction
                and strongest_interaction.get("partial_r2_case_blocked", 0.0) >= 0.01
            ),
            "zero_inflation_note": (
                "high zero inflation; emphasize nonzero rate and upper-tail summaries"
                if zero_fraction is not None and zero_fraction >= 0.80 else None
            ),
        }
        out["metrics"][spec["primary_endpoint"]] = endpoint_record
        summary_rows.append({
            "primary_endpoint": spec["primary_endpoint"],
            "source_metric": spec["source_metric"],
            "construct": spec["construct"],
            "direction": spec["direction"],
            "transform": spec["transform"],
            "n_finite": int(len(data)),
            "n_case_blocks": int(len(set(_section6_case_id(row) for row in data))) if data else 0,
            "case_blocked_r2": model.get("r_squared_case_blocked"),
            "strongest_main_effect": strongest_main.get("term") if strongest_main else None,
            "strongest_main_partial_r2": strongest_main.get("partial_r2_case_blocked") if strongest_main else None,
            "strongest_2way_interaction": strongest_interaction.get("term") if strongest_interaction else None,
            "strongest_2way_interaction_partial_r2": (
                strongest_interaction.get("partial_r2_case_blocked") if strongest_interaction else None
            ),
            "interaction_materiality_flag": endpoint_record["interaction_materiality_flag"],
            "raw_median": stability.get("raw_median"),
            "raw_iqr": stability.get("raw_iqr"),
            "raw_p90": stability.get("raw_p90"),
            "raw_p95": stability.get("raw_p95"),
            "raw_zero_fraction": stability.get("raw_zero_fraction"),
            "raw_nonzero_rate": stability.get("raw_nonzero_rate"),
            "default_setting_median": stability.get("default_setting_median"),
            "best_setting_median": stability.get("best_setting_median"),
            "default_oracle_regret": stability.get("default_oracle_regret"),
            "default_percentile_directional": stability.get("default_percentile_directional"),
            "plateau_width_fraction_5pct_of_best": stability.get("plateau_width_fraction_5pct_of_best"),
            "interpretation_note": endpoint_record.get("zero_inflation_note"),
        })
        for term in model.get("terms", []):
            term_rows.append({
                "primary_endpoint": spec["primary_endpoint"],
                "source_metric": spec["source_metric"],
                "construct": spec["construct"],
                "transform": spec["transform"],
                "case_blocked_r2": model.get("r_squared_case_blocked"),
                "term": term.get("term"),
                "type": term.get("type"),
                "coefficient": term.get("coefficient"),
                "partial_ss_case_blocked": term.get("partial_ss_case_blocked"),
                "partial_r2_case_blocked": term.get("partial_r2_case_blocked"),
            })
        main_effect_rows.extend(_main_effect_level_rows(data, spec, parameter_names))
        interaction_grid_rows.extend(_interaction_grid_rows(data, spec, parameter_names))
    return out, summary_rows, term_rows, main_effect_rows, interaction_grid_rows


def _factorial_anova_for_metric(rows, parameter_names, metric):
    data = [
        row for row in rows
        if row.get("IRMF", {}).get(metric) is not None and np.isfinite(row["IRMF"][metric])
    ]
    if len(data) < 3:
        return None

    main_terms = list(parameter_names)
    interaction_terms = [f"{a}:{b}" for i, a in enumerate(parameter_names) for b in parameter_names[i + 1:]]
    terms = main_terms + interaction_terms
    y = np.asarray([float(row["IRMF"][metric]) for row in data], dtype=float)
    y_centered = y - np.mean(y)
    sst = float(np.sum(y_centered ** 2))
    if sst <= 1e-18:
        return {
            "metric": metric,
            "n": int(len(data)),
            "r_squared": 0.0,
            "terms": [],
            "note": "response has near-zero variance",
        }

    x_full, names = _design_matrix(data, parameter_names, terms)
    sse_full, beta_full = _fit_sse(y, x_full)
    r2 = float(max(0.0, 1.0 - sse_full / sst))
    term_rows = []
    for term in terms:
        reduced_terms = [t for t in terms if t != term]
        x_reduced, _ = _design_matrix(data, parameter_names, reduced_terms)
        sse_reduced, _ = _fit_sse(y, x_reduced)
        partial_ss = max(0.0, float(sse_reduced - sse_full))
        term_rows.append({
            "term": term,
            "type": "interaction" if ":" in term else "main_effect",
            "coefficient": float(beta_full[names.index(term)]),
            "partial_ss": partial_ss,
            "partial_variance_share": float(partial_ss / sst),
        })

    term_rows = sorted(term_rows, key=lambda item: item["partial_variance_share"], reverse=True)
    return {
        "metric": metric,
        "n": int(len(data)),
        "model": "linear coded full-factorial regression with main effects and pairwise interactions",
        "r_squared": r2,
        "sse_full": sse_full,
        "sst": sst,
        "terms": term_rows,
    }


def _factorial_anova_summary(rows, parameter_names):
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
    ]
    out = {
        "parameter_names": list(parameter_names),
        "metrics": {},
        "interpretation_note": (
            "partial_variance_share is computed by dropping one coded main-effect "
            "or pairwise-interaction term from the full least-squares model. It is "
            "a descriptive factorial sensitivity decomposition, not a randomized "
            "inferential ANOVA with p-values."
        ),
    }
    for metric in metrics:
        result = _factorial_anova_for_metric(rows, parameter_names, metric)
        if result is not None:
            out["metrics"][metric] = result
    return out


def _flatten_anova_rows(anova):
    rows = []
    for metric, result in anova.get("metrics", {}).items():
        for term in result.get("terms", []):
            row = {
                "metric": metric,
                "n": result.get("n"),
                "r_squared": result.get("r_squared"),
                "term": term.get("term"),
                "type": term.get("type"),
                "coefficient": term.get("coefficient"),
                "partial_ss": term.get("partial_ss"),
                "partial_variance_share": term.get("partial_variance_share"),
            }
            rows.append(row)
    return rows


def run_parameter_sensitivity(
        output_root,
        irmf_params=GLOBAL_IRMF_PARAMS,
        emd_params=GLOBAL_EMD_PARAMS,
        factors=PARAMETER_SENSITIVITY_FACTORS,
        level_labels=PARAMETER_SENSITIVITY_LEVEL_LABELS,
        signals=PARAMETER_SENSITIVITY_SIGNALS,
        noises=PARAMETER_SENSITIVITY_NOISES,
        sigmas=PARAMETER_SENSITIVITY_SIGMAS,
        target_snr_db_levels=PARAMETER_SENSITIVITY_TARGET_SNR_DB_LEVELS,
        n=DEFAULT_N,
        fs=DEFAULT_FS,
        seed=DEFAULT_SEED,
        reconstruction_protocol_id=SYNTHETIC_RECONSTRUCTION_PROTOCOL_ID,
):
    """
    Section 6.1 factorial parameter sensitivity and robustness.

    Under the current relative-H protocol, the factorial design perturbs h1, a,
    h_min, and c_H around the selected global IRMF configuration.  c_H is the
    locked dimensionless robust-loss multiplier; every case derives its
    operating H as H_case = c_H * sigma_hat(Y) inside run_fixed_irmf_case.  This
    section intentionally runs IRMF only: it measures stability of the selected
    IRMF parameter neighborhood, rather than another method comparison.
    """
    output_root = ensure_dir(output_root)
    parameter_names = tuple(factors.keys())
    relative_h_active = irmf_params.get("H_parameterization") == "relative_noise_scale"
    if relative_h_active and "H" in parameter_names:
        raise ValueError(
            "Section 6.1 relative-H sensitivity must perturb c_H, not absolute H."
        )
    if relative_h_active and "c_H" not in parameter_names:
        raise ValueError("Section 6.1 relative-H sensitivity requires c_H in factors.")
    level_codes = {label: idx - 1 for idx, label in enumerate(level_labels)}
    parameter_design = []

    for design_id, factor_tuple in enumerate(itertools.product(*[factors[name] for name in parameter_names]), start=1):
        params = dict(irmf_params)
        factor_map = {}
        level_map = {}
        code_map = {}
        for name, factor in zip(parameter_names, factor_tuple):
            idx = list(factors[name]).index(factor)
            label = level_labels[idx]
            params[name] = float(irmf_params[name]) * float(factor)
            factor_map[name] = float(factor)
            level_map[name] = label
            code_map[name] = int(level_codes[label])
        parameter_design.append((design_id, params, factor_map, level_map, code_map))

    checkpoint_path = output_root / "section_6_3_parameter_sensitivity_rows.jsonl"
    status_path = output_root / "section_6_3_parameter_sensitivity_status.json"
    raw_rows = _load_jsonl_rows(checkpoint_path)
    active_checkpoint_ids = set()
    for design_id, _params, _factor_map, _level_map, _code_map in parameter_design:
        for signal_name in signals:
            for noise_name in noises:
                for target_snr_db in target_snr_db_levels:
                    active_checkpoint_ids.add(
                        f"design{int(design_id)}|{signal_name}|{noise_name}|"
                        f"snr{float(target_snr_db):.12g}|seed{int(seed)}"
                    )
    rows = [
        row for row in raw_rows
        if row.get("checkpoint_id") in active_checkpoint_ids
    ]
    completed = {
        row.get("checkpoint_id")
        for row in rows
        if row.get("checkpoint_id")
    }
    expected_rows = int(
        len(parameter_design) * len(signals) * len(noises) * len(target_snr_db_levels)
    )
    write_json({
        "section": "6.1 IRMF Parameter Sensitivity and Robustness",
        "module_status": "running_or_resumable",
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_scope_policy": "active_v557_scope_rows_only",
        "n_raw_checkpoint_rows": int(len(raw_rows)),
        "n_out_of_scope_checkpoint_rows": int(len(raw_rows) - len(rows)),
        "n_expected_rows": expected_rows,
        "n_completed_rows": int(len(completed)),
        "n_remaining_rows": int(expected_rows - len(completed)),
        "noise_severity_design": "target_snr_energy_ratio",
        "target_snr_db_levels": list(target_snr_db_levels),
        "reconstruction_protocol_id": reconstruction_protocol_id,
        "reconstruction_protocol_policy": (
            "protocol_controlled_synthetic_component_selection_reconstruction"
            if reconstruction_protocol_id else
            "native_residual_reconstruction"
        ),
        "H_parameterization": (
            "relative_noise_scale" if relative_h_active else irmf_params.get("H_parameterization", "absolute")
        ),
    }, status_path)

    for design_id, params, factor_map, level_map, code_map in parameter_design:
        for signal_name in signals:
            for noise_name in noises:
                for target_snr_db in target_snr_db_levels:
                    checkpoint_id = (
                        f"design{int(design_id)}|{signal_name}|{noise_name}|"
                        f"snr{float(target_snr_db):.12g}|seed{int(seed)}"
                    )
                    if checkpoint_id in completed:
                        continue
                    sigma = _sigma_label_for_snr(target_snr_db)
                    case = make_signal_noise_case(
                        signal_name=signal_name,
                        noise_name=noise_name,
                        sigma=sigma,
                        n=n,
                        fs=fs,
                        seed=seed,
                        target_snr_db=target_snr_db,
                    )
                    result = run_fixed_irmf_case(
                        Y=case["Y"],
                        X_clean=case["X_clean"],
                        t=case["t"],
                        fs=fs,
                        irmf_params=params,
                        expected_noise_ratio=case.get("expected_noise_ratio"),
                        true_components=case.get("true_components"),
                        reconstruction_protocol_id=reconstruction_protocol_id,
                        run_id=f"factorial_sensitivity_{design_id}_{signal_name}_{noise_name}_{sigma}",
                    )
                    row = {
                        "checkpoint_id": checkpoint_id,
                        "factorial_design_id": design_id,
                        "signal": signal_name,
                        "noise": noise_name,
                        "sigma": sigma,
                        "target_snr_db": target_snr_db,
                        "noise_severity_design": case.get("noise_design"),
                        "realized_input_snr_db": case.get("realized_input_snr_db"),
                        "seed": seed,
                        "IRMF": method_result_summary(result),
                    }
                    for name in parameter_names:
                        row[f"{name}_factor"] = factor_map[name]
                        row[f"{name}_level"] = level_map[name]
                        row[f"{name}_code"] = code_map[name]
                        row[f"{name}_value"] = params[name]
                    rows.append(row)
                    completed.add(checkpoint_id)
                    _append_jsonl_row(checkpoint_path, row)
                    if len(completed) % 25 == 0 or len(completed) == expected_rows:
                        write_json({
                            "section": "6.1 IRMF Parameter Sensitivity and Robustness",
                            "module_status": (
                                "checkpoint_complete_pending_statistics"
                                if len(completed) == expected_rows else
                                "running_or_resumable"
                            ),
                            "checkpoint_file": checkpoint_path.name,
                            "checkpoint_scope_policy": "active_v557_scope_rows_only",
                            "n_raw_checkpoint_rows": int(len(raw_rows) + len(completed) - len(rows)),
                            "n_out_of_scope_checkpoint_rows": int(len(raw_rows) - len(rows)),
                            "n_expected_rows": expected_rows,
                            "n_completed_rows": int(len(completed)),
                            "n_remaining_rows": int(expected_rows - len(completed)),
                            "noise_severity_design": "target_snr_energy_ratio",
                            "target_snr_db_levels": list(target_snr_db_levels),
                            "reconstruction_protocol_id": reconstruction_protocol_id,
                            "reconstruction_protocol_policy": (
                                "protocol_controlled_synthetic_component_selection_reconstruction"
                                if reconstruction_protocol_id else
                                "native_residual_reconstruction"
                            ),
                            "H_parameterization": (
                                "relative_noise_scale"
                                if relative_h_active
                                else irmf_params.get("H_parameterization", "absolute")
                            ),
                        }, status_path)

    aggregate = _aggregate_factorial_sensitivity_rows(rows, parameter_names)
    anova = _factorial_anova_summary(rows, parameter_names)
    (
        primary_case_blocked,
        primary_case_blocked_summary_rows,
        primary_case_blocked_term_rows,
        primary_main_effect_rows,
        primary_interaction_grid_rows,
    ) = _primary_endpoint_case_blocked_factorial_summary(rows, parameter_names)
    model_formula = "response ~ " + " + ".join(parameter_names)
    if len(parameter_names) > 1:
        interaction_terms = [
            f"{left}:{right}"
            for i, left in enumerate(parameter_names)
            for right in parameter_names[i + 1:]
        ]
        model_formula += " + " + " + ".join(interaction_terms)

    write_json({
        "section": "6.1 IRMF Parameter Sensitivity and Robustness",
        "base_irmf_params": dict(irmf_params),
        "factors": factors,
        "level_labels": list(level_labels),
        "H_parameterization": (
            "relative_noise_scale" if relative_h_active else irmf_params.get("H_parameterization", "absolute")
        ),
        "relative_H_rule": (
            "Perturb c_H, not absolute H. For each case, run_fixed_irmf_case "
            "derives H_case = c_H * sigma_hat(Y) using the frozen truth-free "
            "first-difference MAD scale proxy."
            if relative_h_active else None
        ),
        "signals": list(signals),
        "noises": list(noises),
        "sigmas": list(sigmas),
        "target_snr_db_levels": list(target_snr_db_levels),
        "reconstruction_protocol_id": reconstruction_protocol_id,
        "reconstruction_protocol_policy": (
            "protocol_controlled_synthetic_component_selection_reconstruction"
            if reconstruction_protocol_id else
            "native_residual_reconstruction"
        ),
        "noise_severity_design": "target_snr_energy_ratio",
        "sigma_role": "snr_equivalent_label_not_fixed_noise_amplitude",
        "n_factorial_settings": len(parameter_design),
        "n_cases_per_setting": int(len(signals) * len(noises) * len(target_snr_db_levels)),
        "total_irmf_runs": int(len(parameter_design) * len(signals) * len(noises) * len(target_snr_db_levels)),
        "model": model_formula,
        "primary_endpoint_case_blocked_model": (
            "response ~ h1 + a + h_min + c_H + all pairwise interactions "
            "+ signal/noise/SNR/seed case block"
        ),
        "primary_endpoint_case_blocked_scope": (
            "10 non-contamination primary endpoints; contamination primary "
            "endpoints are handled by Section 6.2 contamination design sensitivity."
        ),
        "main_plot": "factorial effect plot, pairwise interaction heatmap, and ANOVA contribution table",
    }, output_root / "section_6_3_protocol.json")
    write_json(rows, output_root / "section_6_3_parameter_sensitivity.json")
    write_csv(_flatten_factorial_rows(rows), output_root / "section_6_3_parameter_sensitivity.csv")
    write_json(aggregate, output_root / "section_6_3_parameter_sensitivity_aggregate.json")
    write_json(anova, output_root / "section_6_3_factorial_anova.json")
    write_csv(_flatten_anova_rows(anova), output_root / "section_6_3_factorial_anova.csv")
    write_json(
        primary_case_blocked,
        output_root / "section_6_3_primary_endpoint_case_blocked_factorial.json",
    )
    write_csv(
        primary_case_blocked_summary_rows,
        output_root / "section_6_3_primary_endpoint_case_blocked_factorial_summary.csv",
    )
    write_csv(
        primary_case_blocked_term_rows,
        output_root / "section_6_3_primary_endpoint_case_blocked_factorial_terms.csv",
    )
    write_csv(
        primary_main_effect_rows,
        output_root / "section_6_3_primary_endpoint_main_effect_level_summary.csv",
    )
    write_csv(
        primary_interaction_grid_rows,
        output_root / "section_6_3_primary_endpoint_interaction_grid_summary.csv",
    )
    write_json({
        "section": "6.1 IRMF Parameter Sensitivity and Robustness",
        "module_status": "statistics_complete",
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_scope_policy": "active_v557_scope_rows_only",
        "n_raw_checkpoint_rows": int(len(_load_jsonl_rows(checkpoint_path))),
        "n_out_of_scope_checkpoint_rows": int(len(_load_jsonl_rows(checkpoint_path)) - len(rows)),
        "n_expected_rows": expected_rows,
        "n_completed_rows": int(len(completed)),
        "n_remaining_rows": int(expected_rows - len(completed)),
        "statistics_complete": True,
        "noise_severity_design": "target_snr_energy_ratio",
        "target_snr_db_levels": list(target_snr_db_levels),
        "reconstruction_protocol_id": reconstruction_protocol_id,
        "reconstruction_protocol_policy": (
            "protocol_controlled_synthetic_component_selection_reconstruction"
            if reconstruction_protocol_id else
            "native_residual_reconstruction"
        ),
        "H_parameterization": (
            "relative_noise_scale"
            if relative_h_active
            else irmf_params.get("H_parameterization", "absolute")
        ),
        "primary_endpoint_case_blocked_factorial_complete": True,
        "primary_endpoint_case_blocked_factorial_outputs": [
            "section_6_3_primary_endpoint_case_blocked_factorial.json",
            "section_6_3_primary_endpoint_case_blocked_factorial_summary.csv",
            "section_6_3_primary_endpoint_case_blocked_factorial_terms.csv",
            "section_6_3_primary_endpoint_main_effect_level_summary.csv",
            "section_6_3_primary_endpoint_interaction_grid_summary.csv",
        ],
    }, status_path)
    return rows, {
        "aggregate": aggregate,
        "anova": anova,
        "primary_endpoint_case_blocked_factorial": primary_case_blocked,
    }


def run_boundary_sensitivity(
        output_root,
        irmf_params=GLOBAL_IRMF_PARAMS,
        emd_params=GLOBAL_EMD_PARAMS,
        signals=BOUNDARY_SENSITIVITY_SIGNALS,
        noises=BOUNDARY_SENSITIVITY_NOISES,
        sigmas=BOUNDARY_SENSITIVITY_SIGMAS,
        target_snr_db_levels=BOUNDARY_SENSITIVITY_TARGET_SNR_DB_LEVELS,
        n=DEFAULT_N,
        fs=DEFAULT_FS,
        seed=DEFAULT_SEED,
):
    output_root = ensure_dir(output_root)
    rows = []
    variants = [
        ("IRMF-periodic", {"boundary_mode": "periodic"}, {"nbsym": emd_params.get("nbsym", 2)}),
        ("IRMF-mirror", {"boundary_mode": "mirror"}, {"nbsym": emd_params.get("nbsym", 2)}),
        ("EMD-nbsym2", {"boundary_mode": irmf_params.get("boundary_mode", "periodic")}, {"nbsym": 2}),
        ("EMD-nbsym4", {"boundary_mode": irmf_params.get("boundary_mode", "periodic")}, {"nbsym": 4}),
    ]
    for label, irmf_update, emd_update in variants:
        local_irmf = dict(irmf_params)
        local_emd = dict(emd_params)
        local_irmf.update(irmf_update)
        local_emd.update(emd_update)
        for signal_name in signals:
            for noise_name in noises:
                for target_snr_db in target_snr_db_levels:
                    sigma = _sigma_label_for_snr(target_snr_db)
                    _, _, _, row = run_fixed_pair_case(
                        signal_name=signal_name,
                        noise_name=noise_name,
                        sigma=sigma,
                        target_snr_db=target_snr_db,
                        irmf_params=local_irmf,
                        emd_params=local_emd,
                        n=n,
                        fs=fs,
                        seed=seed,
                        run_id_prefix=f"boundary_{label}_{signal_name}_{noise_name}_snr{float(target_snr_db):g}dB",
                    )
                    row["boundary_variant"] = label
                    rows.append(row)
    write_json({
        "section": "6.4 Boundary Sensitivity",
        "variants": [v[0] for v in variants],
        "signals": list(signals),
        "noises": list(noises),
        "sigmas": list(sigmas),
        "target_snr_db_levels": list(target_snr_db_levels),
        "noise_severity_design": "target_snr_energy_ratio",
        "sigma_role": "snr_equivalent_label_not_fixed_noise_amplitude",
    }, output_root / "section_6_4_protocol.json")
    return rows, write_section_outputs(rows, output_root, "section_6_4_boundary_sensitivity")


def run_random_seed_stability(
        output_root,
        irmf_params=GLOBAL_IRMF_PARAMS,
        emd_params=GLOBAL_EMD_PARAMS,
        signals=RANDOM_SEED_STABILITY_SIGNALS,
        noises=RANDOM_SEED_STABILITY_NOISES,
        sigma=RANDOM_SEED_STABILITY_SIGMA,
        target_snr_db=RANDOM_SEED_STABILITY_TARGET_SNR_DB,
        seeds=RANDOM_SEED_STABILITY_SEEDS,
        n=DEFAULT_N,
        fs=DEFAULT_FS,
):
    output_root = ensure_dir(output_root)
    rows = []
    for signal_name in signals:
        for noise_name in noises:
            for seed in seeds:
                _, _, _, row = run_fixed_pair_case(
                    signal_name=signal_name,
                    noise_name=noise_name,
                    sigma=sigma,
                    target_snr_db=target_snr_db,
                    irmf_params=irmf_params,
                    emd_params=emd_params,
                    n=n,
                    fs=fs,
                    seed=seed,
                    run_id_prefix=f"seed_{seed}_{signal_name}_{noise_name}_snr{float(target_snr_db):g}dB",
                )
                rows.append(row)
    write_json({
        "section": "6.5 Random Seed Stability",
        "noise_severity_design": "target_snr_energy_ratio",
        "target_snr_db": float(target_snr_db),
        "sigma_role": "snr_equivalent_label_not_fixed_noise_amplitude",
        "sigma": sigma,
        "seeds": list(seeds),
        "signals": list(signals),
        "noises": list(noises),
    }, output_root / "section_6_5_protocol.json")
    return rows, write_section_outputs(rows, output_root, "section_6_5_random_seed_stability")


def run_all_robustness_sensitivity(
        output_root,
        irmf_params=GLOBAL_IRMF_PARAMS,
        emd_params=GLOBAL_EMD_PARAMS,
        eemd_params=GLOBAL_EEMD_PARAMS,
        ceemdan_params=GLOBAL_CEEMDAN_PARAMS,
):
    output_root = ensure_dir(output_root)
    noise_root = output_root / "6_1_noise_robustness"
    contamination_root = output_root / "6_2_contamination_robustness"
    parameter_root = output_root / "6_3_parameter_sensitivity"
    boundary_root = output_root / "6_4_boundary_sensitivity"
    seed_root = output_root / "6_5_random_seed_stability"

    noise = _read_json_if_exists(noise_root / "section_6_1_noise_robustness_aggregate.json")
    if noise is None:
        noise = run_noise_robustness(
            noise_root,
            irmf_params=irmf_params,
            emd_params=emd_params,
            eemd_params=eemd_params,
            ceemdan_params=ceemdan_params,
        )[1]

    contamination = _read_json_if_exists(
        contamination_root / "section_6_2_contamination_robustness_aggregate.json"
    )
    if contamination is None:
        contamination = run_contamination_robustness(
            contamination_root,
            irmf_params=irmf_params,
            emd_params=emd_params,
            eemd_params=eemd_params,
            ceemdan_params=ceemdan_params,
        )[1]

    parameter = _read_json_if_exists(
        parameter_root / "section_6_3_parameter_sensitivity_aggregate.json"
    )
    if parameter is None:
        parameter = run_parameter_sensitivity(parameter_root, irmf_params, emd_params)[1]

    boundary = _read_json_if_exists(boundary_root / "section_6_4_boundary_sensitivity_aggregate.json")
    if boundary is None:
        boundary = run_boundary_sensitivity(boundary_root, irmf_params, emd_params)[1]

    seed = _read_json_if_exists(seed_root / "section_6_5_random_seed_stability_aggregate.json")
    if seed is None:
        seed = run_random_seed_stability(seed_root, irmf_params, emd_params)[1]

    return {
        "noise_robustness": noise,
        "contamination_robustness": contamination,
        "parameter_sensitivity": parameter,
        "boundary_sensitivity": boundary,
        "random_seed_stability": seed,
    }


if __name__ == "__main__":
    run_all_robustness_sensitivity(Path("IRMF_EMD_PAPER_RESULTS") / "section_6_robustness_sensitivity")
