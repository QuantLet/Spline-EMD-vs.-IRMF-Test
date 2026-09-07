#!/usr/bin/python
# coding: UTF-8

"""V5.21 synthetic-to-real proxy metric calibration.

This post-processing stage validates ground-truth-free proxy metrics on the
synthetic unified cube by comparing them with ground-truth endpoints.  Real-data
validation should use only proxy metrics that are interpretable and empirically
calibrated here.
"""

from collections import defaultdict
import csv
from pathlib import Path

import numpy as np

from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


GROUND_TRUTH_METRICS = (
    "denoise_nmse",
    "denoise_corr",
    "imf_recovery_corr",
    "imf_recovery_nrmse",
    "component_splitting_index",
    "component_merging_index",
    "inter_imf_entanglement_index",
    "decomposition_count_error",
    "noise_capture_corr",
    "signal_leakage_into_noise",
)

PROXY_METRICS = (
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

LOWER_IS_BETTER = {
    "denoise_nmse",
    "imf_recovery_nrmse",
    "component_splitting_index",
    "component_merging_index",
    "inter_imf_entanglement_index",
    "decomposition_count_error",
    "signal_leakage_into_noise",
    "trimmed_residual_whiteness",
    "residual_autocorrelation_abs_mean",
    "residual_low_frequency_leakage",
    "energy_conservation_error",
}


def _finite(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return None


def _read_rows(path):
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _metric_value(row, method, metric):
    return _finite(row.get(f"{method}_{metric}"))


def _rank_array(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and abs(x[order[j + 1]] - x[order[i]]) < 1e-12:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def _corr(x, y, spearman=False):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 4:
        return None
    x = x[mask]
    y = y[mask]
    if spearman:
        x = _rank_array(x)
        y = _rank_array(y)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _beneficial(value, metric):
    value = _finite(value)
    if value is None:
        return None
    return -value if metric in LOWER_IS_BETTER else value


def _calibration_rows(rows):
    methods = sorted({
        key.split("_", 1)[0]
        for row in rows
        for key in row
        if "_" in key and key.split("_", 1)[0] in {"IRMF", "EMD", "EEMD", "CEEMDAN"}
    })
    records = []
    for row in rows:
        base_key = (
            row.get("signal_regime"),
            row.get("signal"),
            row.get("noise"),
            row.get("sigma"),
            row.get("seed"),
        )
        for method in methods:
            record = {
                "observation_key": base_key + (method,),
                "method": method,
            }
            for metric in GROUND_TRUTH_METRICS + PROXY_METRICS:
                val = _beneficial(_metric_value(row, method, metric), metric)
                if val is not None:
                    record[metric] = val
            records.append(record)

    out = []
    for proxy in PROXY_METRICS:
        for truth in GROUND_TRUTH_METRICS:
            for group_name, method_filter in [("all_methods", None)] + [(m, m) for m in methods]:
                x = []
                y = []
                for record in records:
                    if method_filter is not None and record["method"] != method_filter:
                        continue
                    if proxy in record and truth in record:
                        x.append(record[proxy])
                        y.append(record[truth])
                pearson = _corr(x, y, spearman=False)
                spearman = _corr(x, y, spearman=True)
                if pearson is None and spearman is None:
                    continue
                calibrated = bool(spearman is not None and abs(spearman) >= 0.50 and len(x) >= 20)
                out.append({
                    "proxy_metric": proxy,
                    "ground_truth_metric": truth,
                    "group": group_name,
                    "n_observations": int(len(x)),
                    "pearson_corr_beneficial_orientation": pearson,
                    "spearman_corr_beneficial_orientation": spearman,
                    "abs_spearman": None if spearman is None else abs(float(spearman)),
                    "calibrated_for_real_data": calibrated,
                    "calibration_rule": "abs_spearman>=0.50 and n>=20, after orienting all metrics so higher is better",
                })
    return out


def _proxy_recommendations(rows):
    by_proxy = defaultdict(list)
    for row in rows:
        if row.get("group") == "all_methods" and row.get("calibrated_for_real_data"):
            by_proxy[row["proxy_metric"]].append(row)
    out = []
    for proxy, vals in sorted(by_proxy.items()):
        vals = sorted(vals, key=lambda r: r.get("abs_spearman") or 0.0, reverse=True)
        best = vals[0]
        out.append({
            "proxy_metric": proxy,
            "recommended_for_real_data": True,
            "strongest_ground_truth_link": best["ground_truth_metric"],
            "strongest_abs_spearman": best["abs_spearman"],
            "n_supporting_links": int(len(vals)),
            "role": "validated_proxy_metric_not_synthetic_primary_endpoint",
        })
    return out


def run_proxy_metric_validation(output_root, cube_root):
    output_root = ensure_dir(output_root)
    cube_root = Path(cube_root)
    rows_path = cube_root / "unified_benchmark_cube_rows.csv"
    if not rows_path.exists():
        raise FileNotFoundError(f"Unified cube rows not found: {rows_path}")
    rows = _read_rows(rows_path)
    calibration = _calibration_rows(rows)
    recommendations = _proxy_recommendations(calibration)
    write_csv(calibration, output_root / "proxy_ground_truth_calibration.csv")
    write_json(calibration, output_root / "proxy_ground_truth_calibration.json")
    write_csv(recommendations, output_root / "validated_real_proxy_metrics.csv")
    write_json(recommendations, output_root / "validated_real_proxy_metrics.json")
    dashboard = {
        "section": "V5.21 proxy metric validation",
        "source_cube": str(cube_root),
        "ground_truth_metrics": list(GROUND_TRUTH_METRICS),
        "proxy_metrics": list(PROXY_METRICS),
        "n_calibration_rows": len(calibration),
        "n_validated_proxy_metrics": len(recommendations),
        "outputs": {
            "calibration": "proxy_ground_truth_calibration.csv",
            "validated_proxy_metrics": "validated_real_proxy_metrics.csv",
        },
        "interpretation": (
            "Proxy metrics are calibrated on synthetic ground truth and should "
            "be used on real data only as ground-truth-free evidence, not as "
            "synthetic primary endpoints."
        ),
    }
    write_json(dashboard, output_root / "proxy_metric_validation_dashboard.json")
    return dashboard
