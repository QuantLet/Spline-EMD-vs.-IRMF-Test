#!/usr/bin/python
# coding: UTF-8

"""CaseScore weighting sensitivity analysis.

This post-hoc stage does not redefine the primary benchmark metrics.  It checks
whether aggregate method rankings and IRMF-vs-baseline paired gains are stable
under plausible CaseScore weights over the three common metric dimensions:
Reconstruction, Structural Fidelity, and Robustness.
"""

from collections import defaultdict
import numpy as np

from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


METHODS = ("IRMF", "EMD", "EEMD", "CEEMDAN")
BASELINES = ("EMD", "EEMD", "CEEMDAN")
WEIGHT_SCHEMES = {
    "current_code_40_35_25": {
        "reconstruction_score": 0.40,
        "structural_fidelity_score": 0.35,
        "robustness_score": 0.25,
    },
    "recommended_decomposition_30_50_20": {
        "reconstruction_score": 0.30,
        "structural_fidelity_score": 0.50,
        "robustness_score": 0.20,
    },
    "robust_heavy_35_40_25": {
        "reconstruction_score": 0.35,
        "structural_fidelity_score": 0.40,
        "robustness_score": 0.25,
    },
    "legacy_metric_sheet_25_65_10": {
        "reconstruction_score": 0.25,
        "structural_fidelity_score": 0.65,
        "robustness_score": 0.10,
    },
    "equal_weight_33_33_33": {
        "reconstruction_score": 1.0 / 3.0,
        "structural_fidelity_score": 1.0 / 3.0,
        "robustness_score": 1.0 / 3.0,
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


def _dimension_value(method_data, key):
    if key == "robustness_score":
        return _finite(method_data.get("robustness_score", method_data.get("contamination_resistance_score")))
    return _finite(method_data.get(key))


def _weighted_score(method_data, weights):
    vals = []
    ws = []
    for key, weight in weights.items():
        value = _dimension_value(method_data, key)
        if value is not None:
            vals.append(value)
            ws.append(float(weight))
    if not vals:
        return None
    ws = np.asarray(ws, dtype=float)
    ws = ws / (np.sum(ws) + 1e-12)
    return float(np.sum(ws * np.asarray(vals, dtype=float)))


def _rank(values):
    items = [(m, v) for m, v in values.items() if v is not None and np.isfinite(v)]
    if not items:
        return {}
    items = sorted(items, key=lambda kv: kv[1], reverse=True)
    return {method: float(i + 1) for i, (method, _) in enumerate(items)}


def _spearman_from_ranks(a, b):
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return None
    x = np.asarray([a[k] for k in common], dtype=float)
    y = np.asarray([b[k] for k in common], dtype=float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def run_case_score_weighting_sensitivity(output_root, family_rows):
    output_root = ensure_dir(output_root)
    case_rows = []
    aggregate_rows = []
    ranking_rows = []

    for scheme_name, weights in WEIGHT_SCHEMES.items():
        method_scores = defaultdict(list)
        method_ranks = defaultdict(list)
        paired = defaultdict(list)

        for idx, row in enumerate(family_rows):
            values = {}
            for method in METHODS:
                method_data = row.get(method, {})
                if not isinstance(method_data, dict) or method_data.get("error"):
                    continue
                score = _weighted_score(method_data, weights)
                values[method] = score
                if score is not None:
                    method_scores[method].append(score)
                    case_rows.append({
                        "weight_scheme": scheme_name,
                        "case_index": int(idx),
                        "signal": row.get("signal"),
                        "noise": row.get("noise"),
                        "sigma": row.get("sigma"),
                        "method": method,
                        "weighted_case_score": score,
                        "reconstruction_weight": weights["reconstruction_score"],
                        "structural_fidelity_weight": weights["structural_fidelity_score"],
                        "robustness_weight": weights["robustness_score"],
                    })
            ranks = _rank(values)
            for method, rank in ranks.items():
                method_ranks[method].append(rank)
            iv = values.get("IRMF")
            if iv is not None:
                for baseline in BASELINES:
                    bv = values.get(baseline)
                    if bv is not None:
                        paired[baseline].append(iv - bv)

        for method in METHODS:
            vals = np.asarray(method_scores.get(method, []), dtype=float)
            ranks = np.asarray(method_ranks.get(method, []), dtype=float)
            aggregate_rows.append({
                "weight_scheme": scheme_name,
                "method": method,
                "n_cases": int(vals.size),
                "mean_weighted_case_score": float(np.mean(vals)) if vals.size else None,
                "median_weighted_case_score": float(np.median(vals)) if vals.size else None,
                "mean_rank": float(np.mean(ranks)) if ranks.size else None,
            })
        for baseline, deltas in paired.items():
            arr = np.asarray(deltas, dtype=float)
            aggregate_rows.append({
                "weight_scheme": scheme_name,
                "comparison": f"IRMF vs {baseline}",
                "n_cases": int(arr.size),
                "mean_paired_gain": float(np.mean(arr)) if arr.size else None,
                "median_paired_gain": float(np.median(arr)) if arr.size else None,
                "win_rate": float(np.mean(arr > 0)) if arr.size else None,
            })

    # Ranking stability against the current code weights.
    ranks_by_scheme = {}
    for scheme in WEIGHT_SCHEMES:
        subset = [r for r in aggregate_rows if r.get("weight_scheme") == scheme and r.get("method")]
        ranks_by_scheme[scheme] = {r["method"]: r.get("mean_rank") for r in subset if r.get("mean_rank") is not None}
    reference = ranks_by_scheme.get("current_code_40_35_25", {})
    for scheme, ranks in ranks_by_scheme.items():
        ranking_rows.append({
            "reference_scheme": "current_code_40_35_25",
            "comparison_scheme": scheme,
            "spearman_rank_correlation": _spearman_from_ranks(reference, ranks),
            "top_method": min(ranks, key=ranks.get) if ranks else None,
            "same_top_method_as_reference": (
                (min(ranks, key=ranks.get) == min(reference, key=reference.get))
                if ranks and reference else None
            ),
        })

    protocol = {
        "section": "CaseScore Weighting Sensitivity",
        "purpose": (
            "Assess whether conclusions based on the optional aggregate CaseScore "
            "are stable under plausible weights over reconstruction, structural "
            "fidelity, and robustness."
        ),
        "primary_metric_policy": (
            "The manuscript should primarily report the three dimensions separately. "
            "Weighted CaseScore is a parameter-selection and supplementary ranking criterion."
        ),
        "weight_schemes": WEIGHT_SCHEMES,
        "robustness_alias": "robustness_score is an alias for contamination_resistance_score in older outputs.",
    }
    write_json(protocol, output_root / "case_score_weighting_sensitivity_protocol.json")
    write_csv(case_rows, output_root / "case_score_weighting_sensitivity_rows.csv")
    write_json(case_rows, output_root / "case_score_weighting_sensitivity_rows.json")
    write_csv(aggregate_rows, output_root / "case_score_weighting_sensitivity_aggregate.csv")
    write_json(aggregate_rows, output_root / "case_score_weighting_sensitivity_aggregate.json")
    write_csv(ranking_rows, output_root / "case_score_weighting_ranking_stability.csv")
    write_json(ranking_rows, output_root / "case_score_weighting_ranking_stability.json")
    return {
        "n_rows": len(case_rows),
        "n_aggregate_rows": len(aggregate_rows),
        "n_ranking_rows": len(ranking_rows),
        "protocol": protocol,
    }
