#!/usr/bin/python
# coding: UTF-8
"""V5.4 loss ablation, contamination tolerance, and optimization stability.

The experiment uses the same IRMF multiscale parameters for every loss.  Each
loss receives one globally selected tuning value from a pre-specified
representative development subset; no per-case tuning is allowed.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

from core_algorithms.robust_losses import LOSS_REGISTRY, evaluate_loss, loss_metadata
from experiments.experiment_utils import make_signal_noise_case, method_result_summary, run_fixed_irmf_case
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json
from project_config import (
    DEFAULT_FS, DEFAULT_N, GLOBAL_IRMF_PARAMS,
    LOSS_ABLATION_LOSSES, LOSS_ABLATION_NOISES, LOSS_ABLATION_SEEDS,
    LOSS_ABLATION_SIGMAS, LOSS_ABLATION_SIGNALS, LOSS_CONTAMINATION_LAMBDAS,
    LOSS_FAILURE_THRESHOLDS, LOSS_TUNING_GRIDS,
)

PRIMARY_METRICS = (
    "case_score", "reconstruction_score", "contamination_resistance_score",
    "imf_recovery_score", "denoise_nmse", "denoise_corr",
)


def _json_tuning(tuning):
    return {str(k): float(v) for k, v in dict(tuning).items()}


def _optimization_summary(scale_history):
    rows = [d for scale in scale_history for d in scale.get("optimization_diagnostics", [])]
    if not rows:
        return {"n_local_fits": 0}
    iterations = np.asarray([r.get("iterations", 0) for r in rows], dtype=float)
    converged = np.asarray([bool(r.get("converged", False)) for r in rows])
    nondesc = np.asarray([r.get("objective_non_descent_count", 0) for r in rows], dtype=float)
    curv = np.asarray([r.get("min_curvature", np.nan) for r in rows], dtype=float)
    hess = np.asarray([r.get("final_hessian", np.nan) for r in rows], dtype=float)
    return {
        "n_local_fits": int(len(rows)),
        "convergence_rate": float(np.mean(converged)),
        "failure_rate": float(1.0 - np.mean(converged)),
        "median_iterations": float(np.median(iterations)),
        "p95_iterations": float(np.quantile(iterations, 0.95)),
        "objective_non_descent_rate": float(np.mean(nondesc > 0)),
        "negative_curvature_rate": float(np.mean(curv < -1e-10)) if np.any(np.isfinite(curv)) else None,
        "near_singular_hessian_rate": float(np.mean(np.abs(hess) < 1e-8)) if np.any(np.isfinite(hess)) else None,
    }


def _run_loss_case(case, base_params, loss_name, tuning, run_id):
    params = dict(base_params)
    params.update({
        "loss_name": loss_name,
        "loss_tuning": dict(tuning),
        "collect_optimization_diagnostics": True,
    })
    if loss_name == "gaussian_smoothed_median" and "H" in tuning:
        params["H"] = float(tuning["H"])
    started = perf_counter()
    result = run_fixed_irmf_case(
        Y=case["Y"], X_clean=case["X_clean"], t=case["t"], fs=DEFAULT_FS,
        irmf_params=params, expected_noise_ratio=case["expected_noise_ratio"],
        true_components=case["true_components"], run_id=run_id,
    )
    runtime = perf_counter() - started
    summary = method_result_summary(result)
    summary.update({
        "loss_key": loss_name,
        "loss_tuning": _json_tuning(tuning),
        "runtime_seconds": float(runtime),
        **_optimization_summary(result.get("scale_history", [])),
    })
    return summary


def _selection_score(rows):
    """Balanced development score; failures and runtime are explicit penalties."""
    vals = defaultdict(list)
    for row in rows:
        for metric in PRIMARY_METRICS:
            value = row.get(metric)
            if value is not None and np.isfinite(value):
                vals[metric].append(float(value))
    reconstruction = np.mean(vals["reconstruction_score"]) if vals["reconstruction_score"] else 0.0
    contamination = np.mean(vals["contamination_resistance_score"]) if vals["contamination_resistance_score"] else 0.0
    recovery = np.mean(vals["imf_recovery_score"]) if vals["imf_recovery_score"] else 0.0
    failure = np.mean([r.get("failure_rate", 1.0) for r in rows]) if rows else 1.0
    runtime = np.median([r.get("runtime_seconds", np.nan) for r in rows]) if rows else np.inf
    runtime_penalty = 0.02 * np.log1p(max(float(runtime), 0.0))
    return float(0.35*reconstruction + 0.35*contamination + 0.30*recovery - 0.15*failure - runtime_penalty)


def calibrate_loss_tuning(output_root, base_params=GLOBAL_IRMF_PARAMS, n=DEFAULT_N, fs=DEFAULT_FS):
    output_root = ensure_dir(output_root)
    # Compact, pre-specified development subset distinct from the full ablation.
    dev_cases = [
        ("stationary_multi_sine", "gaussian", 0.10, 101),
        ("stationary_multi_sine", "huber_contamination", 0.20, 102),
        ("chirp", "student_t", 0.20, 103),
        ("impulsive_transient", "impulsive", 0.20, 104),
        ("intermittent_oscillation", "burst", 0.20, 105),
        ("close_frequencies", "gaussian", 0.10, 106),
    ]
    candidate_rows, selected = [], {}
    for loss_name in LOSS_ABLATION_LOSSES:
        best = None
        for tuning in LOSS_TUNING_GRIDS[loss_name]:
            rows = []
            for signal, noise, sigma, seed in dev_cases:
                case = make_signal_noise_case(signal, noise, sigma, n=n, fs=fs, seed=seed)
                rows.append(_run_loss_case(case, base_params, loss_name, tuning, f"cal_{loss_name}_{signal}_{noise}_{seed}"))
            score = _selection_score(rows)
            item = {
                "loss_key": loss_name, "tuning": _json_tuning(tuning),
                "selection_score": score, "n_cases": len(rows),
                "mean_case_score": float(np.nanmean([r.get("case_score", np.nan) for r in rows])),
                "mean_failure_rate": float(np.mean([r.get("failure_rate", 1.0) for r in rows])),
                "median_runtime_seconds": float(np.nanmedian([r.get("runtime_seconds", np.nan) for r in rows])),
            }
            candidate_rows.append(item)
            if best is None or item["selection_score"] > best["selection_score"]:
                best = item
        selected[loss_name] = dict(best["tuning"])
    write_csv(candidate_rows, output_root / "loss_tuning_candidates.csv")
    write_json(selected, output_root / "locked_loss_tuning.json")
    write_json({"development_cases": dev_cases, "selection_rule": "balanced fixed global score", "selected": selected}, output_root / "protocol.json")
    return selected, candidate_rows


def run_loss_ablation(output_root, base_params=GLOBAL_IRMF_PARAMS, selected_tuning=None, n=DEFAULT_N, fs=DEFAULT_FS):
    output_root = ensure_dir(output_root)
    if selected_tuning is None:
        selected_tuning, _ = calibrate_loss_tuning(output_root / "calibration", base_params, n=n, fs=fs)
    rows = []
    for signal in LOSS_ABLATION_SIGNALS:
        for noise in LOSS_ABLATION_NOISES:
            for sigma in LOSS_ABLATION_SIGMAS:
                for seed in LOSS_ABLATION_SEEDS:
                    case = make_signal_noise_case(signal, noise, sigma, n=n, fs=fs, seed=seed)
                    for loss_name in LOSS_ABLATION_LOSSES:
                        summary = _run_loss_case(case, base_params, loss_name, selected_tuning[loss_name], f"abl_{signal}_{noise}_{sigma}_{seed}_{loss_name}")
                        summary.update({"signal": signal, "noise": noise, "sigma": sigma, "seed": seed, **loss_metadata(loss_name)})
                        rows.append(summary)
    write_csv(rows, output_root / "loss_ablation_rows.csv")
    write_json(rows, output_root / "loss_ablation_rows.json")
    aggregate = _aggregate(rows)
    write_csv(aggregate, output_root / "loss_ablation_aggregate.csv")
    write_json(aggregate, output_root / "loss_ablation_aggregate.json")
    _write_figures(aggregate, output_root)
    return rows, aggregate, selected_tuning


def run_contamination_tolerance(output_root, base_params=GLOBAL_IRMF_PARAMS, selected_tuning=None, n=DEFAULT_N, fs=DEFAULT_FS):
    output_root = ensure_dir(output_root)
    if selected_tuning is None:
        selected_tuning, _ = calibrate_loss_tuning(output_root / "calibration", base_params, n=n, fs=fs)
    rows = []
    signals = ("stationary_multi_sine", "intermittent_oscillation", "impulsive_transient")
    for signal in signals:
        for lam in LOSS_CONTAMINATION_LAMBDAS:
            for seed in LOSS_ABLATION_SEEDS:
                case = make_signal_noise_case(signal, "huber_contamination", 0.20, n=n, fs=fs, seed=seed, noise_kwargs={"lam": lam})
                for loss_name in LOSS_ABLATION_LOSSES:
                    row = _run_loss_case(case, base_params, loss_name, selected_tuning[loss_name], f"cont_{signal}_{lam}_{seed}_{loss_name}")
                    nmse = row.get("denoise_nmse", np.inf)
                    corr = row.get("denoise_corr", -np.inf)
                    row.update({
                        "signal": signal, "lambda": lam, "seed": seed,
                        "empirical_failure": bool((not np.isfinite(nmse)) or nmse > LOSS_FAILURE_THRESHOLDS["denoise_nmse"] or corr < LOSS_FAILURE_THRESHOLDS["denoise_corr"]),
                    })
                    rows.append(row)
    write_csv(rows, output_root / "contamination_tolerance_rows.csv")
    aggregate = _aggregate(rows, group_keys=("loss_key", "lambda"))
    write_csv(aggregate, output_root / "contamination_tolerance_aggregate.csv")
    write_json({"rows": rows, "aggregate": aggregate, "failure_definition": LOSS_FAILURE_THRESHOLDS}, output_root / "contamination_tolerance.json")
    _write_contamination_figure(aggregate, output_root)
    return rows, aggregate


def _aggregate(rows, group_keys=("loss_key",)):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in group_keys)].append(row)
    out = []
    for key, subset in groups.items():
        item = {k: v for k, v in zip(group_keys, key)}
        item["n"] = len(subset)
        for metric in PRIMARY_METRICS + ("runtime_seconds", "failure_rate", "median_iterations", "objective_non_descent_rate", "negative_curvature_rate"):
            vals = [float(r[metric]) for r in subset if r.get(metric) is not None and np.isfinite(r.get(metric))]
            if vals:
                item[f"mean_{metric}"] = float(np.mean(vals))
                item[f"median_{metric}"] = float(np.median(vals))
                item[f"sd_{metric}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        failures = [bool(r.get("empirical_failure")) for r in subset if "empirical_failure" in r]
        if failures:
            item["empirical_failure_rate"] = float(np.mean(failures))
        out.append(item)
    return sorted(out, key=lambda r: tuple(str(r.get(k)) for k in group_keys))


def _write_figures(aggregate, output_root):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        write_json({"status": "skipped", "reason": str(exc)}, output_root / "figures_status.json")
        return
    xs = [max(r.get("median_runtime_seconds", np.nan), 1e-8) for r in aggregate]
    ys = [r.get("mean_case_score", np.nan) for r in aggregate]
    labels = [r["loss_key"] for r in aggregate]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    ax.scatter(xs, ys)
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("Median runtime (s, log scale)"); ax.set_ylabel("Mean case score")
    ax.set_title("Loss efficiency–performance frontier"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(output_root / "loss_performance_cost_frontier.png"); plt.close(fig)


def _write_contamination_figure(aggregate, output_root):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    losses = sorted(set(r["loss_key"] for r in aggregate))
    for loss in losses:
        part = sorted([r for r in aggregate if r["loss_key"] == loss], key=lambda r: float(r["lambda"]))
        ax.plot([r["lambda"] for r in part], [r.get("mean_denoise_nmse", np.nan) for r in part], marker="o", label=loss)
    ax.set_xlabel("Contamination rate λ"); ax.set_ylabel("Mean denoising NMSE"); ax.set_title("Empirical contamination tolerance")
    ax.grid(alpha=0.25); ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(output_root / "contamination_tolerance_nmse.png"); plt.close(fig)


def run_complete_loss_methodology(output_root, base_params=GLOBAL_IRMF_PARAMS, n=DEFAULT_N, fs=DEFAULT_FS):
    output_root = ensure_dir(output_root)
    selected, candidates = calibrate_loss_tuning(output_root / "01_calibration", base_params, n=n, fs=fs)
    _, aggregate, _ = run_loss_ablation(output_root / "02_ablation", base_params, selected, n=n, fs=fs)
    _, contamination = run_contamination_tolerance(output_root / "03_contamination_tolerance", base_params, selected, n=n, fs=fs)
    dashboard = {"locked_loss_tuning": selected, "n_calibration_candidates": len(candidates), "ablation": aggregate, "contamination": contamination}
    write_json(dashboard, output_root / "loss_methodology_dashboard.json")
    return dashboard
