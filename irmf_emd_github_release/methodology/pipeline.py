#!/usr/bin/python
# coding: UTF-8
"""Independent rho-function methodology pipeline.

This module deliberately contains no EMD/EEMD/CEEMDAN comparison.  It studies
loss properties and the effect of replacing the rho function inside the same
IRMF implementation.
"""
from pathlib import Path
import json
import numpy as np

from project_config import DEFAULT_FS, DEFAULT_N, GLOBAL_IRMF_PARAMS
from experiments.experiment_loss_function_analysis import run_loss_function_analysis
from experiments.experiment_loss_ablation import (
    calibrate_loss_tuning,
    run_loss_ablation,
    run_contamination_tolerance,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json, write_manifest
from methodology.loss_regime_map.pipeline import run_loss_regime_map


def run_loss_properties(output_root, H=1.0):
    return run_loss_function_analysis(ensure_dir(output_root), H=H)


def run_loss_calibration(output_root, base_params=None, n=DEFAULT_N, fs=DEFAULT_FS):
    return calibrate_loss_tuning(
        ensure_dir(output_root),
        base_params=dict(base_params or GLOBAL_IRMF_PARAMS), n=n, fs=fs,
    )


def _load_locked(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_locked(root, base_params, n, fs, locked_tuning=None):
    if locked_tuning is not None:
        return locked_tuning
    locked_path = Path(root) / "01_calibration" / "locked_loss_tuning.json"
    selected = _load_locked(locked_path)
    if selected is not None:
        return selected
    selected, _ = run_loss_calibration(Path(root) / "01_calibration", base_params, n=n, fs=fs)
    return selected


def run_loss_ablation_stage(output_root, base_params=None, n=DEFAULT_N, fs=DEFAULT_FS, locked_tuning=None):
    root = ensure_dir(output_root)
    params = dict(base_params or GLOBAL_IRMF_PARAMS)
    selected = _resolve_locked(root, params, n, fs, locked_tuning)
    return run_loss_ablation(root / "02_ablation", params, selected, n=n, fs=fs)


def run_contamination_stage(output_root, base_params=None, n=DEFAULT_N, fs=DEFAULT_FS, locked_tuning=None):
    root = ensure_dir(output_root)
    params = dict(base_params or GLOBAL_IRMF_PARAMS)
    selected = _resolve_locked(root, params, n, fs, locked_tuning)
    return run_contamination_tolerance(root / "03_contamination_tolerance", params, selected, n=n, fs=fs)


def run_optimization_stage(output_root, base_params=None, n=DEFAULT_N, fs=DEFAULT_FS, locked_tuning=None):
    """Create a focused optimization-stability report from the same ablation rows."""
    root = ensure_dir(output_root)
    params = dict(base_params or GLOBAL_IRMF_PARAMS)
    selected = _resolve_locked(root, params, n, fs, locked_tuning)
    rows_path = root / "02_ablation" / "loss_ablation_rows.json"
    if rows_path.exists():
        rows = json.loads(rows_path.read_text(encoding="utf-8"))
    else:
        rows, _, _ = run_loss_ablation(root / "02_ablation", params, selected, n=n, fs=fs)

    metrics = (
        "failure_rate", "median_iterations", "p95_iterations",
        "objective_non_descent_rate", "negative_curvature_rate",
        "near_singular_hessian_rate", "runtime_seconds",
    )
    grouped = {}
    for row in rows:
        grouped.setdefault(row["loss_key"], []).append(row)
    summary = []
    for loss, subset in sorted(grouped.items()):
        item = {"loss_key": loss, "n": len(subset)}
        for metric in metrics:
            vals = [float(r[metric]) for r in subset if r.get(metric) is not None and np.isfinite(r.get(metric))]
            if vals:
                item[f"mean_{metric}"] = float(np.mean(vals))
                item[f"median_{metric}"] = float(np.median(vals))
                item[f"p95_{metric}"] = float(np.quantile(vals, 0.95))
        summary.append(item)
    out = ensure_dir(root / "04_optimization_stability")
    write_csv(summary, out / "optimization_stability_summary.csv")
    write_json(summary, out / "optimization_stability_summary.json")
    write_json({"metrics": metrics, "source": "02_ablation/loss_ablation_rows.json"}, out / "protocol.json")
    return summary


def run_methodology_pipeline(output_root, base_params=None, n=DEFAULT_N, fs=DEFAULT_FS, quick_regime_map=False):
    root = ensure_dir(output_root)
    params = dict(base_params or GLOBAL_IRMF_PARAMS)
    properties = run_loss_properties(root / "00_loss_properties", H=float(params.get("H", 1.0)))
    selected, candidates = run_loss_calibration(root / "01_calibration", params, n=n, fs=fs)
    _, ablation, _ = run_loss_ablation(root / "02_ablation", params, selected, n=n, fs=fs)
    _, contamination = run_contamination_tolerance(root / "03_contamination_tolerance", params, selected, n=n, fs=fs)
    optimization = run_optimization_stage(root, params, n=n, fs=fs, locked_tuning=selected)
    regime_map = run_loss_regime_map(root / "05_loss_regime_map", params, n=n, fs=fs, quick=quick_regime_map)
    dashboard = {
        "scope": "rho_function_methodology_only",
        "emd_family_included": False,
        "loss_properties": properties,
        "locked_loss_tuning": selected,
        "n_calibration_candidates": len(candidates),
        "loss_ablation": ablation,
        "contamination_tolerance": contamination,
        "optimization_stability": optimization,
        "loss_regime_map": regime_map,
    }
    write_json(dashboard, root / "methodology_dashboard.json")
    write_manifest(root, {
        "pipeline_version": "V5.7",
        "scope": "rho_function_methodology_only",
        "emd_family_included": False,
        "completed_stages": ["loss_properties", "loss_calibration", "loss_ablation", "contamination_tolerance", "optimization_stability", "loss_regime_map"],
    })
    return dashboard
