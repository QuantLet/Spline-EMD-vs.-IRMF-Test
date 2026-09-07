from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from time import perf_counter
import numpy as np

from core_algorithms.robust_losses import loss_metadata
from diagnostics.shared_physical_diagnostics import reconstructed_signal
from experiments.experiment_utils import run_fixed_irmf_case, method_result_summary
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json
from signal_bank.synthetic_signals import SIGNAL_REGISTRY

CORE_LOSSES = ("huber", "pseudo_huber", "gaussian_smoothed_median")
EXTENDED_LOSSES = ("l2", "l1", "huber", "pseudo_huber", "fair", "tukey", "gaussian_smoothed_median")


def time_axis(n=500, fs=500.0):
    return np.arange(int(n), dtype=float) / float(fs)


def clean_signal(name, n=500, fs=500.0):
    t = time_axis(n, fs)
    return t, np.asarray(SIGNAL_REGISTRY[name](t), dtype=float)


def robust_scale(x):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)) + 1e-10)


def contamination_vector(n, fs, kind, center, width, rate, scale, seed):
    rng = np.random.default_rng(int(seed))
    z = np.zeros(int(n), dtype=float)
    center_idx = int(np.clip(round(float(center) * fs), 0, n - 1))
    width_pts = max(1, int(round(float(width) * fs)))
    if kind == "isolated_spike":
        count = max(1, int(round(float(rate) * n)))
        candidates = np.arange(n)
        idx = rng.choice(candidates, size=min(count, n), replace=False)
        z[idx] = scale * rng.choice((-1.0, 1.0), size=len(idx))
    elif kind in ("clustered_burst", "event_overlap"):
        count = max(width_pts, int(round(float(rate) * n)))
        start = int(np.clip(center_idx - count // 2, 0, max(0, n - count)))
        idx = np.arange(start, min(n, start + count))
        z[idx] = scale * rng.choice((-1.0, 1.0), size=len(idx)) * (0.75 + 0.5*rng.random(len(idx)))
    elif kind == "asymmetric_cluster":
        count = max(width_pts, int(round(float(rate) * n)))
        start = int(np.clip(center_idx - count // 2, 0, max(0, n - count)))
        idx = np.arange(start, min(n, start + count))
        z[idx] = scale * (0.75 + 0.5*rng.random(len(idx)))
    else:
        raise ValueError(f"Unknown contamination kind: {kind}")
    return z, np.flatnonzero(z != 0)


def event_spec(signal_name, t):
    if signal_name == "frequency_jump":
        return 0.5, (np.abs(t - 0.5) <= 0.04)
    if signal_name == "impulsive_transient":
        return 0.46, ((t >= 0.43) & (t <= 0.49))
    if signal_name == "intermittent_oscillation":
        mask = ((t >= 0.18) & (t <= 0.42)) | ((t >= 0.58) & (t <= 0.82))
        return 0.40, mask
    if signal_name == "close_frequencies":
        return 0.5, np.ones_like(t, dtype=bool)
    return 0.5, np.ones_like(t, dtype=bool)


def make_structural_case(signal_name, contamination_kind, distance_points, rate, outlier_scale, sigma, n, fs, seed):
    t, x = clean_signal(signal_name, n=n, fs=fs)
    event_center, event_mask = event_spec(signal_name, t)
    rng = np.random.default_rng(int(seed))
    gaussian = float(sigma) * rng.normal(size=n)
    center = event_center + float(distance_points) / float(fs)
    contamination, idx = contamination_vector(n, fs, contamination_kind, center, 0.02, rate, outlier_scale*sigma, seed+913)
    y0 = x + gaussian
    y = y0 + contamination
    contamination_mask = np.zeros(n, dtype=bool)
    contamination_mask[idx] = True
    neighborhood = contamination_mask.copy()
    radius = max(3, int(round(0.03*fs)))
    for j in idx:
        neighborhood[max(0, j-radius):min(n, j+radius+1)] = True
    return {"t": t, "X_clean": x, "Y": y, "Y_baseline": y0, "event_mask": event_mask,
            "contamination_mask": contamination_mask, "contamination_neighborhood": neighborhood,
            "noise": y-x, "expected_noise_ratio": float(np.var(y-x)/(np.var(y)+1e-12)), "true_components": None}


def optimization_summary(scale_history):
    rows = [d for scale in scale_history for d in scale.get("optimization_diagnostics", [])]
    if not rows:
        return {"n_local_fits": 0}
    def arr(key, default=np.nan): return np.asarray([r.get(key, default) for r in rows], dtype=float)
    iterations, hess, grad = arr("iterations", 0), arr("final_hessian"), arr("final_gradient_abs")
    converged = np.asarray([bool(r.get("converged", False)) for r in rows])
    return {
        "n_local_fits": len(rows), "convergence_rate": float(np.mean(converged)),
        "failure_rate": float(1-np.mean(converged)), "median_iterations": float(np.nanmedian(iterations)),
        "p95_iterations": float(np.nanquantile(iterations, .95)),
        "objective_non_descent_rate": float(np.mean(arr("objective_non_descent_count", 0)>0)),
        "line_search_activation_rate": float(np.mean(arr("line_search_activations", 0)>0)),
        "mean_line_search_steps": float(np.nanmean(arr("line_search_steps", 0))),
        "near_singular_hessian_rate": float(np.mean(np.abs(hess)<1e-8)),
        "negative_curvature_rate": float(np.mean(arr("min_curvature") < -1e-10)),
        "median_final_gradient_abs": float(np.nanmedian(grad)),
        "p95_final_gradient_abs": float(np.nanquantile(grad, .95)),
        "minimum_local_information": float(np.nanmin(hess)),
        "median_local_information": float(np.nanmedian(hess)),
        "local_information_cv": float(np.nanstd(hess)/(abs(np.nanmean(hess))+1e-12)),
    }


def run_loss_on_case(case, base_params, loss_name, tuning, run_id, initialization_offset=0.0):
    params = dict(base_params)
    params.update(loss_name=loss_name, loss_tuning=dict(tuning), collect_optimization_diagnostics=True,
                  initialization_offset=float(initialization_offset))
    if loss_name == "gaussian_smoothed_median" and "H" in tuning:
        params["H"] = float(tuning["H"])
    tic = perf_counter()
    result = run_fixed_irmf_case(case["Y"], case["X_clean"], case["t"],
                                 fs=1.0/(case["t"][1]-case["t"][0]), irmf_params=params,
                                 expected_noise_ratio=case.get("expected_noise_ratio"),
                                 true_components=case.get("true_components"), run_id=run_id)
    runtime = perf_counter()-tic
    rec = reconstructed_signal(case["Y"], result.get("imfs", []), result.get("residual"))
    row = method_result_summary(result)
    row.update(loss_metadata(loss_name)); row["loss_tuning"] = dict(tuning); row["runtime_seconds"] = runtime
    row.update(optimization_summary(result.get("scale_history", [])))
    return result, np.asarray(rec, dtype=float), row


def paired_local_metrics(x, rec_cont, rec_base, event_mask, contamination_neighborhood):
    x=np.asarray(x); rc=np.asarray(rec_cont); rb=np.asarray(rec_base)
    event=np.asarray(event_mask, bool); smooth=~event; outside=~np.asarray(contamination_neighborhood,bool)
    extra=np.maximum((rc-x)**2-(rb-x)**2, 0.0)
    return {
        "event_mse": float(np.mean((rc[event]-x[event])**2)) if np.any(event) else np.nan,
        "smooth_mse": float(np.mean((rc[smooth]-x[smooth])**2)) if np.any(smooth) else np.nan,
        "baseline_event_mse": float(np.mean((rb[event]-x[event])**2)) if np.any(event) else np.nan,
        "baseline_smooth_mse": float(np.mean((rb[smooth]-x[smooth])**2)) if np.any(smooth) else np.nan,
        "error_spread_ratio": float(np.sum(extra[outside])/(np.sum(extra)+1e-12)),
        "max_abs_error": float(np.max(np.abs(rc-x))),
        "event_amplitude_bias": float(np.max(np.abs(rc[event]))-np.max(np.abs(x[event]))) if np.any(event) else np.nan,
    }


def aggregate(rows, keys, metrics):
    groups=defaultdict(list)
    for r in rows: groups[tuple(r.get(k) for k in keys)].append(r)
    out=[]
    for key, subset in groups.items():
        item={k:v for k,v in zip(keys,key)}; item["n"]=len(subset)
        for m in metrics:
            vals=np.asarray([r.get(m,np.nan) for r in subset],float); vals=vals[np.isfinite(vals)]
            if vals.size:
                item[f"mean_{m}"]=float(np.mean(vals)); item[f"median_{m}"]=float(np.median(vals)); item[f"p95_{m}"]=float(np.quantile(vals,.95))
        out.append(item)
    return out
