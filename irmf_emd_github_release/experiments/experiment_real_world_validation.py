#!/usr/bin/python
# coding: UTF-8

"""V5.21 real-world validation using calibrated proxy metrics."""

from pathlib import Path
import numpy as np

from project_config import (
    DEFAULT_FS,
    GLOBAL_CEEMDAN_PARAMS,
    GLOBAL_EEMD_PARAMS,
    GLOBAL_EMD_PARAMS,
    GLOBAL_IRMF_PARAMS,
)
from core_algorithms.eemd_wrapper import run_eemd
from core_algorithms.ceemdan_wrapper import run_ceemdan
from diagnostics.shared_physical_diagnostics import evaluate_shared_physical_diagnostics
from experiments.experiment_utils import method_result_summary, run_fixed_emd_case, run_fixed_irmf_case
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json
from signal_bank.real_data_loader import load_real_signal_csv


REAL_PROXY_METRICS = (
    "trimmed_residual_whiteness",
    "residual_autocorrelation_abs_mean",
    "residual_kurtosis",
    "residual_sparsity",
    "residual_high_frequency_energy_ratio",
    "residual_low_frequency_leakage",
    "residual_outlier_concentration",
    "spectral_concentration_mean",
    "spectral_concentration_min",
    "instantaneous_frequency_continuity_mean",
    "frequency_band_separation",
    "scale_order_consistency",
    "energy_conservation_error",
)


def _normalize_time(T):
    T = np.asarray(T, dtype=float)
    if len(T) <= 1:
        return np.asarray([0.0])
    return (T - T[0]) / (T[-1] - T[0])


def _run_eemd_family_real(method, Y, fs, emd_params, eemd_params, ceemdan_params, algorithm_seed):
    method = method.upper()
    if method == "EEMD":
        raw = run_eemd(
            Y,
            max_imf=eemd_params.get("max_imf", emd_params.get("max_imf", -1)),
            trials=eemd_params.get("trials", 100),
            noise_width=eemd_params.get("noise_width", 0.05),
            parallel=eemd_params.get("parallel", False),
            random_seed=algorithm_seed,
        )
    elif method == "CEEMDAN":
        raw = run_ceemdan(
            Y,
            max_imf=ceemdan_params.get("max_imf", emd_params.get("max_imf", -1)),
            trials=ceemdan_params.get("trials", 100),
            epsilon=ceemdan_params.get("epsilon", 0.005),
            parallel=ceemdan_params.get("parallel", False),
            random_seed=algorithm_seed,
        )
    else:
        raise ValueError(method)
    physical = evaluate_shared_physical_diagnostics(
        Y_observed=Y,
        X_clean=None,
        imfs=raw["imfs"],
        residual=raw["residual"],
        fs=fs,
        residual_penalty_mode="none",
        true_components=None,
    )
    return {"method": method, **raw, **physical}


def _run_one_dataset(path, irmf_params, emd_params, eemd_params, ceemdan_params, fs, algorithm_seed):
    T, Y, inferred_fs = load_real_signal_csv(path, fs=fs)
    t_unit = _normalize_time(T)
    Y = np.asarray(Y, dtype=float)
    row = {
        "dataset": Path(path).stem,
        "path": str(path),
        "fs": inferred_fs,
        "n": int(len(Y)),
        "algorithm_seed": int(algorithm_seed),
        "real_data_ground_truth_available": False,
    }
    irmf = run_fixed_irmf_case(
        Y=Y,
        X_clean=None,
        t=t_unit,
        fs=inferred_fs,
        irmf_params=irmf_params,
        expected_noise_ratio=None,
        true_components=None,
        run_id=f"realworld_{Path(path).stem}_IRMF",
    )
    emd = run_fixed_emd_case(
        Y=Y,
        X_clean=None,
        t=t_unit,
        fs=inferred_fs,
        emd_params=emd_params,
        true_components=None,
        run_id=f"realworld_{Path(path).stem}_EMD",
    )
    row["IRMF"] = method_result_summary(irmf)
    row["EMD"] = method_result_summary(emd)
    for method in ("EEMD", "CEEMDAN"):
        try:
            result = _run_eemd_family_real(
                method, Y, inferred_fs, emd_params, eemd_params, ceemdan_params, algorithm_seed
            )
            row[method] = method_result_summary(result)
        except Exception as exc:
            row[method] = {"method": method, "error": str(exc), "exception_flag": True}
    return row


def _long_proxy_rows(rows):
    out = []
    for row in rows:
        for method in ("IRMF", "EMD", "EEMD", "CEEMDAN"):
            data = row.get(method, {})
            if not isinstance(data, dict):
                continue
            for metric in REAL_PROXY_METRICS:
                value = data.get(metric)
                try:
                    value = float(value)
                    if not np.isfinite(value):
                        continue
                except Exception:
                    continue
                out.append({
                    "dataset": row.get("dataset"),
                    "method": method,
                    "metric": metric,
                    "value": value,
                    "real_data_ground_truth_available": False,
                    "metric_role": "ground_truth_free_real_proxy",
                })
    return out


def _summary_rows(long_rows):
    grouped = {}
    for row in long_rows:
        grouped.setdefault((row["method"], row["metric"]), []).append(float(row["value"]))
    out = []
    for (method, metric), vals in sorted(grouped.items()):
        arr = np.asarray(vals, dtype=float)
        out.append({
            "method": method,
            "metric": metric,
            "n_datasets": int(arr.size),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "q25": float(np.quantile(arr, 0.25)),
            "q75": float(np.quantile(arr, 0.75)),
        })
    return out


def run_real_world_validation(
        output_root,
        data_dir=None,
        irmf_params=GLOBAL_IRMF_PARAMS,
        emd_params=GLOBAL_EMD_PARAMS,
        eemd_params=GLOBAL_EEMD_PARAMS,
        ceemdan_params=GLOBAL_CEEMDAN_PARAMS,
        fs=DEFAULT_FS,
        algorithm_seed=20260721,
):
    output_root = ensure_dir(output_root)
    data_dir = Path(data_dir) if data_dir is not None else None
    rows = []
    if data_dir is not None and data_dir.exists():
        for path in sorted(data_dir.glob("*.csv")):
            rows.append(_run_one_dataset(
                path, irmf_params, emd_params, eemd_params, ceemdan_params, fs, algorithm_seed
            ))
    if not rows:
        (output_root / "README_real_world_validation_template.txt").write_text(
            "Place real-signal CSV files in a directory and pass --real-data-dir. "
            "CSV format: one column y with fs supplied, or two columns time,y. "
            "Pure real-data validation reports proxy metrics only; no ground-truth "
            "NMSE, true IMF recovery, noise_capture_corr, or ORI is claimed.\n",
            encoding="utf-8",
        )
    long_rows = _long_proxy_rows(rows)
    summary = _summary_rows(long_rows)
    write_json(rows, output_root / "real_world_validation_rows.json")
    write_csv(rows, output_root / "real_world_validation_rows.csv")
    write_csv(long_rows, output_root / "real_world_proxy_metrics_long.csv")
    write_json(long_rows, output_root / "real_world_proxy_metrics_long.json")
    write_csv(summary, output_root / "real_world_proxy_metric_summary.csv")
    write_json(summary, output_root / "real_world_proxy_metric_summary.json")
    protocol = {
        "section": "8 Real-world Validation",
        "data_dir": str(data_dir) if data_dir is not None else None,
        "methods": ["IRMF", "EMD", "EEMD", "CEEMDAN"],
        "proxy_metrics": list(REAL_PROXY_METRICS),
        "ground_truth_metrics_not_used": [
            "denoise_nmse",
            "imf_recovery_corr",
            "noise_capture_corr",
            "signal_leakage_into_noise",
            "outlier_resistance_index",
        ],
        "interpretation": (
            "Real data have no clean signal, true noise, or true IMFs. This "
            "stage therefore reports ground-truth-free proxy metrics only. "
            "Use experiment_proxy_metric_validation.py to calibrate these proxies "
            "against synthetic ground-truth endpoints before making paper claims."
        ),
        "outputs": {
            "rows": "real_world_validation_rows.json",
            "long_proxy_metrics": "real_world_proxy_metrics_long.csv",
            "summary": "real_world_proxy_metric_summary.csv",
        },
    }
    write_json(protocol, output_root / "real_world_validation_protocol.json")
    return {"n_datasets": len(rows), "n_proxy_rows": len(long_rows), "summary": summary}
