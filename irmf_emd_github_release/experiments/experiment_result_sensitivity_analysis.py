#!/usr/bin/python
# coding: UTF-8

"""Post-hoc result sensitivity analyses for the V5.2 manuscript revision."""

from collections import defaultdict
import math
import numpy as np

from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


METHODS = ("IRMF", "EMD", "EEMD", "CEEMDAN")
SCORE_FAMILIES = ("reconstruction_score", "structural_fidelity_score", "contamination_resistance_score")
WEIGHT_SCHEMES = {
    "original_0p40_0p35_0p25": {
        "reconstruction_score": 0.40,
        "structural_fidelity_score": 0.35,
        "contamination_resistance_score": 0.25,
    },
    "equal_family_weights": {
        "reconstruction_score": 1.0 / 3.0,
        "structural_fidelity_score": 1.0 / 3.0,
        "contamination_resistance_score": 1.0 / 3.0,
    },
    "reconstruction_emphasized": {
        "reconstruction_score": 0.60,
        "structural_fidelity_score": 0.20,
        "contamination_resistance_score": 0.20,
    },
    "structure_emphasized": {
        "reconstruction_score": 0.25,
        "structural_fidelity_score": 0.55,
        "contamination_resistance_score": 0.20,
    },
    "contamination_emphasized": {
        "reconstruction_score": 0.25,
        "structural_fidelity_score": 0.20,
        "contamination_resistance_score": 0.55,
    },
    "leave_reconstruction_out": {
        "reconstruction_score": 0.00,
        "structural_fidelity_score": 0.35 / 0.60,
        "contamination_resistance_score": 0.25 / 0.60,
    },
    "leave_structure_out": {
        "reconstruction_score": 0.40 / 0.65,
        "structural_fidelity_score": 0.00,
        "contamination_resistance_score": 0.25 / 0.65,
    },
    "leave_contamination_out": {
        "reconstruction_score": 0.40 / 0.75,
        "structural_fidelity_score": 0.35 / 0.75,
        "contamination_resistance_score": 0.00,
    },
}


def _finite(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return None


def _method_value(row, method, metric):
    data = row.get(method, {})
    if not isinstance(data, dict) or data.get("error"):
        return None
    return _finite(data.get(metric))


def _mean(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def _rank_values(values, higher_is_better=True):
    pairs = [(m, v) for m, v in values.items() if v is not None and np.isfinite(v)]
    if not pairs:
        return {}
    ordered = sorted(pairs, key=lambda x: x[1], reverse=higher_is_better)
    ranks = {}
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and abs(ordered[j + 1][1] - ordered[i][1]) < 1e-12:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[ordered[k][0]] = avg
        i = j + 1
    return ranks


def _weighted_composite(method_data, weights):
    vals = []
    ws = []
    for metric, weight in weights.items():
        value = _finite(method_data.get(metric))
        if value is not None and weight > 0:
            vals.append(value)
            ws.append(float(weight))
    if not vals:
        return None
    ws = np.asarray(ws, dtype=float)
    ws = ws / (np.sum(ws) + 1e-12)
    return float(np.sum(ws * np.asarray(vals, dtype=float)))


def run_composite_score_sensitivity(output_root, family_rows, weight_schemes=WEIGHT_SCHEMES):
    output_root = ensure_dir(output_root)
    case_rows = []
    summary_rows = []
    pair_rows = []
    for scheme, weights in weight_schemes.items():
        by_method = defaultdict(list)
        rank_by_method = defaultdict(list)
        for idx, row in enumerate(family_rows):
            scores = {}
            for method in METHODS:
                data = row.get(method, {})
                if isinstance(data, dict) and not data.get("error"):
                    scores[method] = _weighted_composite(data, weights)
            ranks = _rank_values(scores, higher_is_better=True)
            for method, score in scores.items():
                if score is None:
                    continue
                by_method[method].append(score)
                if method in ranks:
                    rank_by_method[method].append(ranks[method])
                case_rows.append({
                    "scheme": scheme,
                    "case_index": int(idx),
                    "signal": row.get("signal"),
                    "noise": row.get("noise"),
                    "sigma": row.get("sigma"),
                    "method": method,
                    "sensitivity_score": score,
                    "rank": ranks.get(method),
                })
        for method in METHODS:
            vals = by_method.get(method, [])
            ranks = rank_by_method.get(method, [])
            summary_rows.append({
                "scheme": scheme,
                "method": method,
                "n": int(len(vals)),
                "mean_score": _mean(vals),
                "median_score": float(np.median(vals)) if vals else None,
                "average_rank": _mean(ranks),
                "first_place_count": int(np.sum(np.asarray(ranks) == 1.0)) if ranks else 0,
                "last_place_count": int(np.sum(np.asarray(ranks) == len(METHODS))) if ranks else 0,
            })
        for baseline in ("EMD", "EEMD", "CEEMDAN"):
            diffs = []
            wins = []
            for row in family_rows:
                i = _weighted_composite(row.get("IRMF", {}), weights)
                b = _weighted_composite(row.get(baseline, {}), weights)
                if i is None or b is None:
                    continue
                diffs.append(i - b)
                wins.append(i > b)
            pair_rows.append({
                "scheme": scheme,
                "comparison": f"IRMF vs {baseline}",
                "n": int(len(diffs)),
                "mean_gain": _mean(diffs),
                "median_gain": float(np.median(diffs)) if diffs else None,
                "win_rate": float(np.mean(wins)) if wins else None,
            })

    write_csv(case_rows, output_root / "composite_sensitivity_case_scores.csv")
    write_json(case_rows, output_root / "composite_sensitivity_case_scores.json")
    write_csv(summary_rows, output_root / "composite_sensitivity_method_summary.csv")
    write_json(summary_rows, output_root / "composite_sensitivity_method_summary.json")
    write_csv(pair_rows, output_root / "composite_sensitivity_irmf_pairwise.csv")
    write_json(pair_rows, output_root / "composite_sensitivity_irmf_pairwise.json")
    write_json({
        "purpose": "Check whether EMD-family conclusions depend on the composite-score weights.",
        "score_families": list(SCORE_FAMILIES),
        "weight_schemes": weight_schemes,
        "interpretation": (
            "Composite score should be treated as a secondary summary outcome. "
            "Primary conclusions should also be reported by reconstruction, structural, "
            "contamination, IMF recovery, and runtime outcomes."
        ),
    }, output_root / "composite_sensitivity_protocol.json")
    return {
        "n_schemes": len(weight_schemes),
        "n_case_rows": len(case_rows),
        "n_summary_rows": len(summary_rows),
        "n_pairwise_rows": len(pair_rows),
    }


def run_structural_distribution_analysis(output_root, family_rows):
    output_root = ensure_dir(output_root)
    metric = "structural_fidelity_score"
    values_by_method = defaultdict(list)
    rank_by_method = defaultdict(list)
    rows_by_case = []
    for idx, row in enumerate(family_rows):
        vals = {method: _method_value(row, method, metric) for method in METHODS}
        ranks = _rank_values(vals, higher_is_better=True)
        for method, value in vals.items():
            if value is None:
                continue
            values_by_method[method].append(value)
            rank_by_method[method].append(ranks.get(method))
            rows_by_case.append({
                "case_index": int(idx),
                "signal": row.get("signal"),
                "noise": row.get("noise"),
                "sigma": row.get("sigma"),
                "method": method,
                metric: value,
                "rank": ranks.get(method),
            })

    summary = []
    for method in METHODS:
        vals = np.asarray(values_by_method.get(method, []), dtype=float)
        ranks = np.asarray([r for r in rank_by_method.get(method, []) if r is not None], dtype=float)
        if vals.size == 0:
            continue
        q10, q25, q50, q75, q90 = np.quantile(vals, [0.10, 0.25, 0.50, 0.75, 0.90])
        threshold = q10
        summary.append({
            "method": method,
            "n": int(vals.size),
            "mean": float(np.mean(vals)),
            "median": float(q50),
            "q10": float(q10),
            "q25": float(q25),
            "q75": float(q75),
            "q90": float(q90),
            "iqr": float(q75 - q25),
            "worst_decile_mean": float(np.mean(vals[vals <= threshold])) if np.any(vals <= threshold) else None,
            "average_rank": float(np.mean(ranks)) if ranks.size else None,
            "first_place_count": int(np.sum(ranks == 1.0)) if ranks.size else 0,
            "last_place_count": int(np.sum(ranks == len(METHODS))) if ranks.size else 0,
            "last_place_rate": float(np.mean(ranks == len(METHODS))) if ranks.size else None,
        })

    write_csv(rows_by_case, output_root / "structural_fidelity_case_distribution.csv")
    write_json(rows_by_case, output_root / "structural_fidelity_case_distribution.json")
    write_csv(summary, output_root / "structural_fidelity_distribution_summary.csv")
    write_json(summary, output_root / "structural_fidelity_distribution_summary.json")
    write_json({
        "purpose": "Diagnose why structural-fidelity means and ranks differ.",
        "metric": metric,
        "warning": (
            "Average rank alone should not be interpreted as dominance. "
            "Use quantiles, last-place rates, and pairwise tests together."
        ),
    }, output_root / "structural_distribution_protocol.json")
    return {"n_case_rows": len(rows_by_case), "n_summary_rows": len(summary)}


def run_performance_cost_frontier(output_root, family_rows):
    output_root = ensure_dir(output_root)
    rows = []
    for method in METHODS:
        runtimes = []
        metrics = defaultdict(list)
        for row in family_rows:
            data = row.get(method, {})
            if not isinstance(data, dict) or data.get("error"):
                continue
            rt = _finite(data.get("runtime_seconds"))
            if rt is not None:
                runtimes.append(rt)
            for metric in [
                "case_score",
                "reconstruction_score",
                "structural_fidelity_score",
                "contamination_resistance_score",
                "imf_recovery_score",
                "denoise_nmse",
            ]:
                value = _finite(data.get(metric))
                if value is not None:
                    metrics[metric].append(value)
        median_rt = float(np.median(runtimes)) if runtimes else None
        item = {
            "method": method,
            "n_runtime": int(len(runtimes)),
            "median_runtime_seconds": median_rt,
            "mean_runtime_seconds": float(np.mean(runtimes)) if runtimes else None,
        }
        for metric, vals in metrics.items():
            item[f"mean_{metric}"] = float(np.mean(vals)) if vals else None
        if median_rt and median_rt > 0:
            item["case_score_per_log_runtime"] = item.get("mean_case_score") / math.log1p(median_rt)
            item["imf_recovery_per_log_runtime"] = item.get("mean_imf_recovery_score") / math.log1p(median_rt)
        rows.append(item)
    baseline = next((r["median_runtime_seconds"] for r in rows if r["method"] == "EMD"), None)
    for row in rows:
        if baseline and row.get("median_runtime_seconds"):
            row["relative_median_runtime_vs_emd"] = row["median_runtime_seconds"] / baseline

    write_csv(rows, output_root / "performance_cost_frontier.csv")
    write_json(rows, output_root / "performance_cost_frontier.json")
    _write_frontier_plot(rows, output_root)
    return {"n_methods": len(rows)}


def _write_frontier_plot(rows, output_root):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        write_json({"plot_status": "skipped", "reason": str(exc)}, output_root / "performance_cost_plot_status.json")
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
    colors = {"IRMF": "#C53030", "EMD": "#2B6CB0", "EEMD": "#2F855A", "CEEMDAN": "#805AD5"}
    for row in rows:
        x = row.get("median_runtime_seconds")
        y = row.get("mean_case_score")
        if x is None or y is None:
            continue
        ax.scatter(x, y, s=70, color=colors.get(row["method"], "#333333"))
        ax.annotate(row["method"], (x, y), xytext=(5, 4), textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("median runtime seconds, log scale")
    ax.set_ylabel("mean composite score")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_root / "performance_cost_frontier.png")
    plt.close(fig)


def _base_signal_name(name):
    mapping = {
        "stationary_multi_sine": "stationary_multi_sine",
        "chirp": "chirp",
        "am_fm": "am_fm",
        "frequency_jump": "frequency_jump",
        "impulsive_transient": "impulsive_transient",
        "intermittent": "intermittent_oscillation",
        "close_freq": "close_frequencies",
    }
    for prefix, base in mapping.items():
        if str(name).startswith(prefix):
            return base
    return str(name)


def run_signal_variant_consistency_summary(output_root, variant_rows):
    output_root = ensure_dir(output_root)
    metrics = ["case_score", "reconstruction_score", "structural_fidelity_score", "contamination_resistance_score", "imf_recovery_score", "denoise_nmse"]
    lower = {"denoise_nmse"}
    grouped = defaultdict(list)
    for row in variant_rows or []:
        grouped[_base_signal_name(row.get("signal"))].append(row)
    out = []
    for base, rows in sorted(grouped.items()):
        for metric in metrics:
            gains = []
            wins = []
            for row in rows:
                i = _method_value(row, "IRMF", metric)
                e = _method_value(row, "EMD", metric)
                if i is None or e is None:
                    continue
                gain = e - i if metric in lower else i - e
                gains.append(gain)
                wins.append(gain > 0)
            if gains:
                out.append({
                    "base_signal": base,
                    "metric": metric,
                    "n_variant_cases": int(len(gains)),
                    "mean_irmf_gain_vs_emd": float(np.mean(gains)),
                    "median_irmf_gain_vs_emd": float(np.median(gains)),
                    "irmf_win_rate_vs_emd": float(np.mean(wins)),
                })
    write_csv(out, output_root / "signal_variant_consistency_summary.csv")
    write_json(out, output_root / "signal_variant_consistency_summary.json")
    write_json({
        "purpose": "Assess whether IRMF-vs-EMD signal-class conclusions persist across intra-class variants.",
        "limitation": "This module summarizes IRMF vs EMD variants only. It does not establish full EMD-family signal generalization unless EEMD/CEEMDAN are also run on variants.",
    }, output_root / "signal_variant_consistency_protocol.json")
    return {"n_rows": len(out), "n_base_signals": len(grouped)}


def run_v5_2_posthoc_result_checks(output_root, family_rows, variant_rows=None):
    output_root = ensure_dir(output_root)
    composite = run_composite_score_sensitivity(output_root / "composite_score_sensitivity", family_rows)
    structural = run_structural_distribution_analysis(output_root / "structural_fidelity_distribution", family_rows)
    frontier = run_performance_cost_frontier(output_root / "performance_cost_frontier", family_rows)
    variant = run_signal_variant_consistency_summary(output_root / "signal_variant_consistency", variant_rows or [])
    dashboard = {
        "composite_score_sensitivity": composite,
        "structural_fidelity_distribution": structural,
        "performance_cost_frontier": frontier,
        "signal_variant_consistency": variant,
    }
    write_json(dashboard, output_root / "v5_2_posthoc_dashboard.json")
    return dashboard

