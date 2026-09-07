#!/usr/bin/python
# coding: UTF-8

"""Section 7: mechanism and failure-case analysis for V4E."""

from collections import defaultdict
import numpy as np

from experiments.experiment_statistical_inference import LOWER_IS_BETTER, METRICS
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


MECHANISM_METRICS = (
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
)


def _finite(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return None


def _benefit_delta(row, metric):
    i = _finite(row.get("IRMF", {}).get(metric))
    e = _finite(row.get("EMD", {}).get(metric))
    if i is None or e is None:
        return None
    raw = i - e
    return -raw if metric in LOWER_IS_BETTER else raw


def _aggregate_by(rows, group_key):
    out = []
    for metric in MECHANISM_METRICS:
        grouped = defaultdict(list)
        for row in rows:
            delta = _benefit_delta(row, metric)
            if delta is None:
                continue
            grouped[row.get(group_key)].append(delta)
        for level, values in sorted(grouped.items(), key=lambda kv: str(kv[0])):
            arr = np.asarray(values, dtype=float)
            out.append({
                "grouping": group_key,
                group_key: level,
                "metric": metric,
                "n": int(arr.size),
                "mean_benefit_delta": float(np.mean(arr)),
                "median_benefit_delta": float(np.median(arr)),
                "win_rate": float(np.mean(arr > 0)),
                "loss_rate": float(np.mean(arr < 0)),
                "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            })
    return out


def _failure_cases(rows, top_n=30):
    failure_rows = []
    for idx, row in enumerate(rows):
        per_metric = {}
        benefits = []
        losses = 0
        for metric in METRICS:
            benefit = _benefit_delta(row, metric)
            if benefit is None:
                continue
            per_metric[f"benefit_{metric}"] = float(benefit)
            benefits.append(float(benefit))
            if benefit < 0:
                losses += 1
        if not benefits:
            continue
        arr = np.asarray(benefits, dtype=float)
        failure_rows.append({
            "case_index": idx,
            "signal": row.get("signal"),
            "noise": row.get("noise"),
            "sigma": row.get("sigma"),
            "mean_benefit_across_metrics": float(np.mean(arr)),
            "median_benefit_across_metrics": float(np.median(arr)),
            "n_metrics_available": int(arr.size),
            "n_metrics_where_irmf_underperforms": int(losses),
            "failure_severity_score": float(-np.mean(arr) + 0.1 * losses),
            **per_metric,
        })
    return sorted(failure_rows, key=lambda r: r["failure_severity_score"], reverse=True)[:int(top_n)]


def _theory_empirical_correlations(rows):
    candidates = (
        "local_theory_score",
        "operator_score",
        "theory_diagnostic_score",
        "final_trace_norm",
        "final_contraction",
        "operator_evolution_mean_contraction",
        "operator_evolution_final_trace_norm",
        "median_b0",
        "max_b0",
    )
    out = []
    for diag in candidates:
        xs = []
        by_metric = {metric: [] for metric in MECHANISM_METRICS}
        for row in rows:
            x = _finite(row.get("IRMF", {}).get(diag))
            if x is None:
                continue
            xs.append(x)
            for metric in MECHANISM_METRICS:
                by_metric[metric].append(_benefit_delta(row, metric))
        for metric, ys_raw in by_metric.items():
            pairs = [(x, y) for x, y in zip(xs, ys_raw) if y is not None and np.isfinite(y)]
            if len(pairs) < 3:
                continue
            xarr = np.asarray([p[0] for p in pairs], dtype=float)
            yarr = np.asarray([p[1] for p in pairs], dtype=float)
            if np.std(xarr) <= 1e-12 or np.std(yarr) <= 1e-12:
                corr = None
            else:
                corr = float(np.corrcoef(xarr, yarr)[0, 1])
            out.append({
                "diagnostic": diag,
                "metric": metric,
                "n": int(len(pairs)),
                "pearson_corr_with_benefit_delta": corr,
                "diagnostic_mean": float(np.mean(xarr)),
                "benefit_delta_mean": float(np.mean(yarr)),
            })
    return out


def _mechanism_interpretation(noise_rows, signal_rows, failure_rows):
    def best_levels(rows, grouping, metric, n=3, reverse=True):
        subset = [r for r in rows if r.get("grouping") == grouping and r.get("metric") == metric]
        return sorted(subset, key=lambda r: r["mean_benefit_delta"], reverse=reverse)[:n]

    noise_best_contam = best_levels(noise_rows, "noise", "contamination_resistance_score", n=3, reverse=True)
    noise_worst_struct = best_levels(noise_rows, "noise", "structural_fidelity_score", n=3, reverse=False)
    signal_best_recon = best_levels(signal_rows, "signal", "reconstruction_score", n=3, reverse=True)
    signal_worst_struct = best_levels(signal_rows, "signal", "structural_fidelity_score", n=3, reverse=False)
    return {
        "contamination_resistance_largest_noise_gains": noise_best_contam,
        "structural_fidelity_weakest_noise_regimes": noise_worst_struct,
        "reconstruction_largest_signal_gains": signal_best_recon,
        "structural_fidelity_weakest_signal_regimes": signal_worst_struct,
        "top_failure_cases": failure_rows[:5],
        "interpretation_note": (
            "Use these summaries to write the mechanism section: robust score clipping should be "
            "most visible under contaminated, impulsive, or burst noise; structural-fidelity losses "
            "identify regimes where decomposition geometry rather than denoising dominates."
        ),
    }


def run_mechanism_analysis(output_root, main_rows):
    output_root = ensure_dir(output_root)
    noise_rows = _aggregate_by(main_rows, "noise")
    signal_rows = _aggregate_by(main_rows, "signal")
    sigma_rows = _aggregate_by(main_rows, "sigma")
    failure_rows = _failure_cases(main_rows)
    theory_corr_rows = _theory_empirical_correlations(main_rows)
    interpretation = _mechanism_interpretation(noise_rows, signal_rows, failure_rows)

    write_csv(noise_rows, output_root / "mechanism_by_noise.csv")
    write_json(noise_rows, output_root / "mechanism_by_noise.json")
    write_csv(signal_rows, output_root / "mechanism_by_signal.csv")
    write_json(signal_rows, output_root / "mechanism_by_signal.json")
    write_csv(sigma_rows, output_root / "mechanism_by_sigma.csv")
    write_json(sigma_rows, output_root / "mechanism_by_sigma.json")
    write_csv(failure_rows, output_root / "failure_cases.csv")
    write_json(failure_rows, output_root / "failure_cases.json")
    write_csv(theory_corr_rows, output_root / "theory_metric_correlations.csv")
    write_json(theory_corr_rows, output_root / "theory_metric_correlations.json")
    write_json(interpretation, output_root / "mechanism_interpretation_summary.json")

    protocol = {
        "section": "7 Mechanism analysis",
        "purpose": "Translate metric-level benchmark outcomes into signal/noise mechanisms and transparent failure cases.",
        "main_outputs": [
            "mechanism_by_noise",
            "mechanism_by_signal",
            "mechanism_by_sigma",
            "failure_cases",
            "theory_metric_correlations",
        ],
        "note": "This module is explanatory. It does not retune IRMF or change any benchmark result.",
    }
    write_json(protocol, output_root / "mechanism_analysis_protocol.json")
    return {
        "n_noise_rows": len(noise_rows),
        "n_signal_rows": len(signal_rows),
        "n_sigma_rows": len(sigma_rows),
        "n_failure_cases": len(failure_rows),
        "n_theory_correlation_rows": len(theory_corr_rows),
        "interpretation": interpretation,
    }
