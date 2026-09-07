#!/usr/bin/python
# coding: UTF-8

"""Scale-sensitivity audit for V5.27 contamination spillover endpoint."""

from pathlib import Path

import numpy as np
import pandas as pd

from project_config import EVALUATION_METHODS, V527_PRIMARY_SCHEMA_VERSION
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


METHODS = tuple(EVALUATION_METHODS)
CONTAMINATION_NOISES = {"impulsive", "burst", "huber_contamination"}


def _rank(vals):
    items = [(m, v) for m, v in vals.items() if np.isfinite(v)]
    if len(items) < 2:
        return {}
    items.sort(key=lambda x: x[1])
    out = {}
    i = 0
    while i < len(items):
        j = i
        while j + 1 < len(items) and abs(items[j + 1][1] - items[i][1]) < 1e-12:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            out[items[k][0]] = float(avg)
        i = j + 1
    return out


def _finite(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def _normalized_from_score(score):
    score = _finite(score)
    if not np.isfinite(score) or score <= 0.0:
        return np.nan
    return float(-np.log(score))


def run_v527_spillover_scale_audit(output_root, cube_root):
    output_root = ensure_dir(output_root)
    rows_path = Path(cube_root) / "unified_benchmark_cube_rows.csv"
    if not rows_path.exists():
        raise FileNotFoundError(f"Unified cube rows not found: {rows_path}")
    df = pd.read_csv(rows_path, low_memory=False)
    df = df[df["noise"].isin(CONTAMINATION_NOISES)].copy()

    detail_rows = []
    rank_rows = []
    pair_rows = []
    for _, row in df.iterrows():
        block = "|".join(str(row.get(k)) for k in ("signal_regime", "signal", "noise", "sigma", "seed"))
        raw_vals = {}
        norm_vals = {}
        for method in METHODS:
            raw = _finite(row.get(f"{method}_contamination_spillover_error"))
            score = _finite(row.get(f"{method}_contamination_spillover_score"))
            norm = _normalized_from_score(score)
            raw_vals[method] = raw
            norm_vals[method] = norm
            detail_rows.append({
                "block_id": block,
                "signal_regime": row.get("signal_regime"),
                "signal": row.get("signal"),
                "noise": row.get("noise"),
                "sigma": row.get("sigma"),
                "seed": row.get("seed"),
                "method": method,
                "raw_contamination_spillover_error": raw,
                "normalized_spillover_loss_from_score": norm,
                "source_score": score,
                "normalization_formula": "-log(contamination_spillover_score)",
            })
        raw_rank = _rank(raw_vals)
        norm_rank = _rank(norm_vals)
        for method in METHODS:
            rank_rows.append({
                "block_id": block,
                "signal_regime": row.get("signal_regime"),
                "signal": row.get("signal"),
                "noise": row.get("noise"),
                "sigma": row.get("sigma"),
                "seed": row.get("seed"),
                "method": method,
                "raw_rank": raw_rank.get(method),
                "normalized_rank": norm_rank.get(method),
                "rank_equal": raw_rank.get(method) == norm_rank.get(method),
            })
        for baseline in [m for m in METHODS if m != "IRMF"]:
            raw_gain = raw_vals.get(baseline, np.nan) - raw_vals.get("IRMF", np.nan)
            norm_gain = norm_vals.get(baseline, np.nan) - norm_vals.get("IRMF", np.nan)
            pair_rows.append({
                "block_id": block,
                "comparison": f"IRMF_vs_{baseline}",
                "raw_paired_gain": raw_gain,
                "normalized_paired_gain": norm_gain,
                "raw_win": bool(np.isfinite(raw_gain) and raw_gain > 1e-12),
                "normalized_win": bool(np.isfinite(norm_gain) and norm_gain > 1e-12),
                "win_equal": (
                    bool(np.isfinite(raw_gain) and raw_gain > 1e-12)
                    == bool(np.isfinite(norm_gain) and norm_gain > 1e-12)
                ),
            })

    rank_equal_rate = float(np.mean([bool(r["rank_equal"]) for r in rank_rows])) if rank_rows else np.nan
    win_equal_rate = float(np.mean([bool(r["win_equal"]) for r in pair_rows])) if pair_rows else np.nan
    paired_summary = []
    for comp in sorted({r["comparison"] for r in pair_rows}):
        sub = [r for r in pair_rows if r["comparison"] == comp]
        raw = np.asarray([_finite(r["raw_paired_gain"]) for r in sub], dtype=float)
        norm = np.asarray([_finite(r["normalized_paired_gain"]) for r in sub], dtype=float)
        raw = raw[np.isfinite(raw)]
        norm = norm[np.isfinite(norm)]
        paired_summary.append({
            "comparison": comp,
            "n": int(min(raw.size, norm.size)),
            "raw_mean_paired_gain": float(np.mean(raw)) if raw.size else np.nan,
            "normalized_mean_paired_gain": float(np.mean(norm)) if norm.size else np.nan,
            "raw_win_rate": float(np.mean(raw > 1e-12)) if raw.size else np.nan,
            "normalized_win_rate": float(np.mean(norm > 1e-12)) if norm.size else np.nan,
        })

    dashboard = {
        "stage": "v527_spillover_scale_sensitivity_audit",
        "schema_version": V527_PRIMARY_SCHEMA_VERSION,
        "schema_status": "frozen",
        "n_contamination_rows": int(len(df)),
        "n_method_values": int(len(detail_rows)),
        "rank_equal_rate_raw_vs_normalized": rank_equal_rate,
        "win_equal_rate_raw_vs_normalized": win_equal_rate,
        "audit_status": "passed" if rank_equal_rate == 1.0 and win_equal_rate == 1.0 else "requires_review",
        "interpretation": (
            "Raw spillover MSE and normalized spillover loss share within-cell "
            "ordering when the normalization scale is case-specific and common "
            "across methods. Raw means remain scale-dependent; normalized loss "
            "is preferable for cross-case absolute summaries."
        ),
        "schema_change_recommendation": (
            "Do not silently change frozen V5.27. If normalized spillover is "
            "promoted to primary publication endpoint, record it as a V5.28 "
            "schema revision or an explicit V5.27.1 statistical-summary amendment."
        ),
    }
    write_csv(detail_rows, output_root / "v527_spillover_raw_vs_normalized_values.csv")
    write_csv(rank_rows, output_root / "v527_spillover_raw_vs_normalized_rank_audit.csv")
    write_csv(pair_rows, output_root / "v527_spillover_raw_vs_normalized_pairwise_audit.csv")
    write_csv(paired_summary, output_root / "v527_spillover_raw_vs_normalized_pairwise_summary.csv")
    write_json(dashboard, output_root / "v527_spillover_scale_audit_dashboard.json")
    return dashboard
