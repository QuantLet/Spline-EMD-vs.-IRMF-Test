#!/usr/bin/python
# coding: UTF-8

from __future__ import annotations

import argparse, csv, json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt

from core_algorithms.strict_spokoiny_irmf import strict_spokoiny_irmf
from diagnostics.theoretical_diagnostics import compute_theoretical_diagnostics_from_scale_history, summarize_theoretical_diagnostics
from noise_bank.noise_models import generate_noise
from signal_bank.synthetic_signals import get_signal, list_synthetic_signals
from visualization.plot_theoretical_diagnostics import plot_theoretical_diagnostics

DEFAULT_SIGNALS = ["stationary_multi_sine", "chirp", "am_fm", "impulsive_transient", "frequency_jump"]
DEFAULT_NOISES = ["gaussian", "laplace", "student_t", "impulsive", "burst", "huber"]
AVAILABLE_NOISES = DEFAULT_NOISES + ["colored_ar1", "pink"]


def json_default(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if not np.isfinite(val) else val
    if isinstance(obj, np.ndarray): return obj.tolist()
    return str(obj)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=json_default)


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def parse_selection(value: str, default_items: List[str], all_items: List[str], label: str) -> List[str]:
    if value in ("all", "default", "defaults"):
        return list(default_items)
    selected = [x.strip() for x in value.split(",") if x.strip()]
    bad = [x for x in selected if x not in all_items]
    if bad:
        raise ValueError(f"Unknown {label}: {bad}. Available: {all_items}")
    return selected


def load_physical_best_params(summary_path: str | None = None) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Try to read physical-best parameters from comprehensive benchmark outputs.

    Accepts flexible JSON structures. Keys are normalized to signal -> noise.
    """
    candidate_paths = []
    if summary_path:
        candidate_paths.append(Path(summary_path))
    candidate_paths += list(Path(".").rglob("irmf_best_parameters.json"))
    candidate_paths += list(Path(".").rglob("main_noise_benchmark_summary.json"))
    candidate_paths += list(Path(".").rglob("*summary.json"))
    out: Dict[str, Dict[str, Dict[str, float]]] = {}

    def consider_record(rec):
        if not isinstance(rec, dict): return
        method = str(rec.get("method", rec.get("best_method", rec.get("algorithm", "")))).lower()
        # Do not require method; many summaries only contain IRMF best rows.
        signal = rec.get("signal") or rec.get("signal_name") or rec.get("Signal")
        noise = rec.get("noise") or rec.get("noise_name") or rec.get("Noise")
        params = rec.get("best_params") or rec.get("params") or rec
        try:
            h1 = float(params.get("h1")); a = float(params.get("a")); h_min = float(params.get("h_min")); H = float(params.get("H"))
        except Exception:
            return
        if not signal or not noise: return
        out.setdefault(str(signal), {})[str(noise)] = {"h1": h1, "a": a, "h_min": h_min, "H": H}

    def walk(obj):
        if isinstance(obj, dict):
            consider_record(obj)
            for v in obj.values(): walk(v)
        elif isinstance(obj, list):
            for v in obj: walk(v)

    seen = set()
    for p in candidate_paths:
        if not p.exists() or p in seen: continue
        seen.add(p)
        try:
            with p.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            walk(obj)
        except Exception:
            continue
    return out


def choose_params(signal, noise, args, best_map):
    fixed = {"h1": args.h1, "a": args.a, "h_min": args.h_min, "H": args.H}
    if args.param_source == "fixed":
        return fixed, "fixed"
    p = best_map.get(signal, {}).get(noise)
    if p is None:
        print(f"  WARNING: no physical-best params for {signal}+{noise}; using fixed params.")
        return fixed, "fixed_fallback"
    return p, "physical_best"


def plot_master_heatmap(rows, key, output_path, title):
    if not rows: return
    signals = [s for s in DEFAULT_SIGNALS if any(r.get("signal") == s for r in rows)]
    noises = [n for n in DEFAULT_NOISES if any(r.get("noise") == n for r in rows)]
    mat = np.full((len(signals), len(noises)), np.nan)
    for i, s in enumerate(signals):
        for j, n in enumerate(noises):
            vals = [r.get(key, np.nan) for r in rows if r.get("signal") == s and r.get("noise") == n]
            if vals: mat[i, j] = float(vals[0])
    plt.figure(figsize=(1.2*len(noises)+3, 0.6*len(signals)+3))
    im = plt.imshow(mat, aspect="auto"); plt.colorbar(im, label=key)
    plt.xticks(np.arange(len(noises)), noises, rotation=35, ha="right"); plt.yticks(np.arange(len(signals)), signals)
    plt.title(title)
    for i in range(len(signals)):
        for j in range(len(noises)):
            if np.isfinite(mat[i,j]): plt.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center", fontsize=8)
    plt.tight_layout(); output_path.parent.mkdir(parents=True, exist_ok=True); plt.savefig(output_path, dpi=180); plt.close()


def run_combo(args, signal, noise, best_map):
    params, param_source_used = choose_params(signal, noise, args, best_map)
    out_dir = args.output_dir / signal / noise / f"sigma_{args.sigma:g}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(args.n) / float(args.fs)
    t_unit = np.linspace(0, 1, args.n, endpoint=False)
    x = get_signal(signal, t_unit)
    eps = generate_noise(noise, args.n, sigma=args.sigma, seed=args.seed)
    y = x + eps
    imfs, residual, _, scale_history = strict_spokoiny_irmf(y, T=t, **params, boundary_mode=args.boundary_mode, verbose=False)
    sigma_trace = float(np.sum(eps**2))
    diag = compute_theoretical_diagnostics_from_scale_history(scale_history, T=t, H=params["H"], sigma_trace=sigma_trace, method="hutchinson", operator_mode=args.operator_mode, probe_count=args.probe_count, random_seed=args.seed+10000)
    rows = diag.get("rows", [])
    for r in rows:
        r.update({"signal": signal, "noise": noise, "sigma": args.sigma, "param_source_used": param_source_used, **params})
    summary = summarize_theoretical_diagnostics(diag)
    summary.update({"signal": signal, "noise": noise, "sigma": args.sigma, "n": args.n, "fs": args.fs, "imf_count": len(imfs), "sigma_trace": sigma_trace, "param_source_used": param_source_used, **params})
    write_csv(out_dir/"theoretical_diagnostics_rows.csv", rows)
    write_json(out_dir/"theoretical_diagnostics_rows.json", rows)
    write_json(out_dir/"theoretical_diagnostics_summary.json", summary)
    write_json(out_dir/"theoretical_diagnostics_full.json", {"summary": summary, "rows": rows, "raw": diag})
    plot_theoretical_diagnostics(diag, out_dir, name="theoretical_diagnostics")
    print(f"  using h1={params['h1']}, a={params['a']}, h_min={params['h_min']}, H={params['H']} ({param_source_used})")
    print(f"  IMF count={len(imfs)} risk={summary.get('theoretical_risk_score', np.nan):.6g} fisher={summary.get('fisher_step_l2_mean_over_scales', np.nan):.3g} wilks={summary.get('wilks_gap_mean_over_scales', np.nan):.3g}")
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--signal", default="all")
    p.add_argument("--noise", default="all")
    p.add_argument("--sigma", type=float, default=0.2)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--fs", type=float, default=500.0)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--h1", type=float, default=0.12); p.add_argument("--a", type=float, default=1.4); p.add_argument("--h-min", dest="h_min", type=float, default=0.005); p.add_argument("--H", type=float, default=0.8)
    p.add_argument("--param-source", choices=["physical_best", "fixed"], default="physical_best")
    p.add_argument("--best-summary", default=None)
    p.add_argument("--operator-mode", choices=["robust_influence", "linear_mean"], default="robust_influence")
    p.add_argument("--probe-count", type=int, default=64)
    p.add_argument("--boundary-mode", choices=["periodic", "mirror"], default="periodic")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/theoretical_diagnostics_standalone"))
    return p.parse_args()


def main():
    args = parse_args()
    signals = parse_selection(args.signal, DEFAULT_SIGNALS, list_synthetic_signals(), "signal")
    noises = parse_selection(args.noise, DEFAULT_NOISES, AVAILABLE_NOISES, "noise")
    best_map = load_physical_best_params(args.best_summary) if args.param_source == "physical_best" else {}
    print("="*80); print("Standalone IRMF theoretical diagnostics — FIX9 Spokoiny extra diagnostics"); print("="*80)
    print(f"signals={signals}\nnoises={noises}\nsigma={args.sigma}\nparam_source={args.param_source}\noperator_mode={args.operator_mode}\noutput_dir={args.output_dir}")
    master=[]; total=len(signals)*len(noises); c=0
    for s in signals:
        for n in noises:
            c+=1; print("\n"+"="*80); print(f"Combination {c}/{total}: {s} + {n}"); print("="*80)
            master.append(run_combo(args, s, n, best_map))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir/"master_theoretical_diagnostics_summary.csv", master)
    write_json(args.output_dir/"master_theoretical_diagnostics_summary.json", master)
    plot_master_heatmap(master, "theoretical_risk_score", args.output_dir/"heatmap_theoretical_risk_score.png", "Theoretical risk score")
    plot_master_heatmap(master, "fisher_step_l2_mean_over_scales", args.output_dir/"heatmap_fisher_step.png", "Fisher/Newton residual proxy")
    plot_master_heatmap(master, "wilks_gap_mean_over_scales", args.output_dir/"heatmap_wilks_gap.png", "Wilks quadratic gap proxy")
    print(f"\nSaved master summary: {args.output_dir/'master_theoretical_diagnostics_summary.csv'}")

if __name__ == "__main__":
    main()
