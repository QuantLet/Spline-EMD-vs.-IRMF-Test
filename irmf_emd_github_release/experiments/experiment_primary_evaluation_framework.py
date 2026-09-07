#!/usr/bin/python
# coding: UTF-8

"""V5.14 primary endpoint, hierarchical rank, and redundancy analysis.

This module is intentionally post-processing only: it consumes four-method
benchmark rows and does not rerun IRMF/EMD-family algorithms.  The legacy
case_score remains available in input rows but is treated as a supplementary
sensitivity endpoint rather than the primary basis for ranking.
"""

from collections import defaultdict
import math

import numpy as np

from project_config import (
    CASE_SCORE_ROLE,
    CONDITIONAL_ENDPOINTS_BY_DIMENSION,
    CONDITIONAL_PRIMARY_ENDPOINTS_BY_DIMENSION,
    DIAGNOSTIC_METRICS_BY_DIMENSION,
    EVALUATION_DIMENSION_ESTIMANDS,
    EVALUATION_FRAMEWORK_VERSION,
    EVALUATION_METHODS,
    LOWER_IS_BETTER_METRICS,
    PRIMARY_ENDPOINTS_BY_DIMENSION,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


METHODS = tuple(EVALUATION_METHODS)
LOWER_IS_BETTER = set(LOWER_IS_BETTER_METRICS)


def _finite(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return None


def _rank_values(values, higher_is_better=True):
    items = [(m, v) for m, v in values.items() if v is not None and np.isfinite(v)]
    if len(items) < 2:
        return {}
    items = sorted(items, key=lambda kv: kv[1], reverse=higher_is_better)
    ranks = {}
    i = 0
    while i < len(items):
        j = i
        while j + 1 < len(items) and abs(items[j + 1][1] - items[i][1]) < 1e-12:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[items[k][0]] = float(avg)
        i = j + 1
    return ranks


def _block_id(row):
    return f"{row.get('signal')}|{row.get('noise')}|{row.get('sigma')}"


def _metric_value(row, method, metric):
    data = row.get(method, {})
    if not isinstance(data, dict) or data.get("error"):
        return None
    return _finite(data.get(metric))


def _metric_higher_is_better(metric):
    return metric not in LOWER_IS_BETTER


def _metric_applicable(row, dimension, metric):
    dcfg = CONDITIONAL_PRIMARY_ENDPOINTS_BY_DIMENSION.get(dimension, {})
    conditional_metrics = {
        item
        for metrics in dcfg.get("subdimensions", {}).values()
        for item in metrics
    }
    applicable_noises = set(dcfg.get("applicable_noises", ())) if metric in conditional_metrics else set()
    if not applicable_noises:
        return True
    if row.get("noise") not in applicable_noises:
        return False
    if metric in {"clean_region_nmse", "contaminated_region_nmse"}:
        return any(bool(row.get(method, {}).get("contamination_region_applicable")) for method in METHODS)
    return True


def _aggregate_seed_rows(rows):
    """Aggregate multiple seeds within signal-noise-sigma blocks by median."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[_block_id(row)].append(row)

    out = []
    for block, block_rows in grouped.items():
        first = block_rows[0]
        agg = {
            "block_id": block,
            "signal": first.get("signal"),
            "noise": first.get("noise"),
            "sigma": first.get("sigma"),
            "n_seed_rows": int(len(block_rows)),
            "seed_aggregation": "median_within_signal_noise_sigma_block",
        }
        for method in METHODS:
            merged = {}
            keys = set()
            for row in block_rows:
                data = row.get(method, {})
                if isinstance(data, dict):
                    keys.update(data.keys())
            for key in keys:
                vals = [_finite(row.get(method, {}).get(key)) for row in block_rows]
                vals = [v for v in vals if v is not None]
                if vals:
                    merged[key] = float(np.median(vals))
                else:
                    for row in block_rows:
                        value = row.get(method, {}).get(key) if isinstance(row.get(method), dict) else None
                        if value is not None:
                            merged[key] = value
                            break
            if merged:
                agg[method] = merged
        out.append(agg)
    return out


def _compute_hierarchical_ranks(case_rows):
    metric_rows = []
    subdimension_rows = []
    dimension_rows = []
    overall_rows = []

    by_block_dimension_method = defaultdict(list)
    by_block_method_dimension = defaultdict(dict)
    primary_dimension_configs = list(PRIMARY_ENDPOINTS_BY_DIMENSION.items()) + list(CONDITIONAL_PRIMARY_ENDPOINTS_BY_DIMENSION.items())

    for row in case_rows:
        block = row["block_id"]
        for dimension, dcfg in primary_dimension_configs:
            for subdimension, metrics in dcfg["subdimensions"].items():
                sub_values = []
                for metric in metrics:
                    if not _metric_applicable(row, dimension, metric):
                        continue
                    values = {method: _metric_value(row, method, metric) for method in METHODS}
                    ranks = _rank_values(values, higher_is_better=_metric_higher_is_better(metric))
                    for method, rank in ranks.items():
                        value = values.get(method)
                        metric_rows.append({
                            "block_id": block,
                            "signal": row.get("signal"),
                            "noise": row.get("noise"),
                            "sigma": row.get("sigma"),
                            "dimension": dimension,
                            "subdimension": subdimension,
                            "metric": metric,
                            "method": method,
                            "value": value,
                            "higher_is_better": _metric_higher_is_better(metric),
                            "within_case_metric_rank": rank,
                            "metric_role": "primary_endpoint",
                        })
                    if ranks:
                        sub_values.append(ranks)

                if not sub_values:
                    continue
                avg_metric_rank = {}
                for method in METHODS:
                    vals = [r[method] for r in sub_values if method in r]
                    if vals:
                        avg_metric_rank[method] = float(np.mean(vals))
                sub_ranks = _rank_values(avg_metric_rank, higher_is_better=False)
                for method, avg_rank in avg_metric_rank.items():
                    row_out = {
                        "block_id": block,
                        "signal": row.get("signal"),
                        "noise": row.get("noise"),
                        "sigma": row.get("sigma"),
                        "dimension": dimension,
                        "subdimension": subdimension,
                        "method": method,
                        "average_metric_rank": avg_rank,
                        "within_case_subdimension_rank": sub_ranks.get(method),
                    }
                    subdimension_rows.append(row_out)
                    if sub_ranks.get(method) is not None:
                        by_block_dimension_method[(block, dimension, method)].append(float(sub_ranks[method]))

        for (b, dimension, method), vals in list(by_block_dimension_method.items()):
            if b != block:
                continue
            avg_subdim_rank = float(np.mean(vals))
            by_block_method_dimension[(block, method)][dimension] = avg_subdim_rank

        for dimension, _ in primary_dimension_configs:
            values = {
                method: by_block_method_dimension.get((block, method), {}).get(dimension)
                for method in METHODS
            }
            dim_ranks = _rank_values(values, higher_is_better=False)
            for method, rank in dim_ranks.items():
                dimension_rows.append({
                    "block_id": block,
                    "signal": row.get("signal"),
                    "noise": row.get("noise"),
                    "sigma": row.get("sigma"),
                    "dimension": dimension,
                    "method": method,
                    "average_subdimension_rank": values.get(method),
                    "within_case_dimension_rank": rank,
                    "inferential_role": "dimension_specific_primary",
                })

        overall_values = {}
        for method in METHODS:
            vals = [
                r["within_case_dimension_rank"]
                for r in dimension_rows
                if r["block_id"] == block and r["method"] == method
            ]
            if vals:
                overall_values[method] = float(np.mean(vals))
        overall_ranks = _rank_values(overall_values, higher_is_better=False)
        for method, avg_rank in overall_values.items():
            overall_rows.append({
                "block_id": block,
                "signal": row.get("signal"),
                "noise": row.get("noise"),
                "sigma": row.get("sigma"),
                "method": method,
                "hierarchical_average_dimension_rank": avg_rank,
                "hierarchical_overall_rank": overall_ranks.get(method),
                "inferential_role": "descriptive_only",
            })

    return metric_rows, subdimension_rows, dimension_rows, overall_rows


def _chi_square_sf_approx(x, df):
    if x is None or df <= 0:
        return None
    x = max(float(x), 0.0)
    if x == 0:
        return 1.0
    z = ((x / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _friedman_rows(dimension_rows):
    out = []
    for dimension in sorted({r["dimension"] for r in dimension_rows}):
        rows = [r for r in dimension_rows if r["dimension"] == dimension]
        by_method = defaultdict(list)
        for row in rows:
            rank = _finite(row.get("within_case_dimension_rank"))
            if rank is not None:
                by_method[row["method"]].append(rank)
        methods = [m for m in METHODS if by_method.get(m)]
        n_blocks = min((len(by_method[m]) for m in methods), default=0)
        k = len(methods)
        if k < 2 or n_blocks < 2:
            continue
        avg_ranks = {m: float(np.mean(by_method[m])) for m in methods}
        q = 12.0 * n_blocks / (k * (k + 1.0)) * sum(
            (avg_ranks[m] - (k + 1.0) / 2.0) ** 2 for m in methods
        )
        p = _chi_square_sf_approx(q, k - 1)
        q_alpha = 2.569 if k == 4 else 1.96
        cd = float(q_alpha * math.sqrt(k * (k + 1.0) / (6.0 * n_blocks)))
        for method in methods:
            out.append({
                "dimension": dimension,
                "method": method,
                "average_rank": avg_ranks[method],
                "n_blocks": int(n_blocks),
                "friedman_chi_square_approx": float(q),
                "friedman_df": int(k - 1),
                "friedman_p_approx": p,
                "critical_difference_alpha_0p05": cd,
                "inferential_role": "dimension_specific_rank_summary",
            })
    return out


def _higher_better_metric_value(value, metric):
    value = _finite(value)
    if value is None:
        return None
    return -value if metric in LOWER_IS_BETTER else value


def _spearman_corr_matrix(table):
    names = list(table.keys())
    n = len(names)
    mat = np.full((n, n), np.nan, dtype=float)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            xa = np.asarray(table[a], dtype=float)
            xb = np.asarray(table[b], dtype=float)
            mask = np.isfinite(xa) & np.isfinite(xb)
            if np.sum(mask) < 3:
                continue
            ra = _rank_array(xa[mask])
            rb = _rank_array(xb[mask])
            mat[i, j] = float(np.corrcoef(ra, rb)[0, 1])
    return names, mat


def _rank_array(x):
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


def _correlation_rows(names, mat, view, dimension=None):
    rows = []
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            row = {
                "view": view,
                "metric_a": a,
                "metric_b": b,
                "spearman_rho": None if not np.isfinite(mat[i, j]) else float(mat[i, j]),
            }
            if dimension is not None:
                row["dimension"] = dimension
            rows.append(row)
    return rows


def _redundancy_clusters(names, mat, threshold=0.85, dimension=None):
    parent = {name: name for name in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(names):
        for j, b in enumerate(names[i + 1:], start=i + 1):
            rho = mat[i, j]
            if np.isfinite(rho) and abs(float(rho)) >= threshold:
                union(a, b)
    groups = defaultdict(list)
    for name in names:
        groups[find(name)].append(name)
    rows = []
    for idx, members in enumerate(sorted(groups.values(), key=lambda x: (-len(x), x[0])), start=1):
        row = {
            "cluster_id": int(idx),
            "n_metrics": int(len(members)),
            "metrics": ";".join(members),
            "threshold_abs_spearman_rho": float(threshold),
        }
        if dimension is not None:
            row["dimension"] = dimension
        rows.append(row)
    return rows


def _write_heatmap(names, mat, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    size = max(5.0, 0.45 * len(names))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _write_dendrogram(names, mat, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
    except Exception:
        return None
    if len(names) < 3:
        return None
    dist = 1.0 - np.nan_to_num(np.abs(mat), nan=0.0)
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="average")
    fig, ax = plt.subplots(figsize=(max(6.0, 0.5 * len(names)), 4.5))
    dendrogram(Z, labels=names, leaf_rotation=90, leaf_font_size=7, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("1 - abs(Spearman rho)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _redundancy_bundle(case_rows, dimension, metrics):
    pooled = defaultdict(list)
    for row in case_rows:
        for method in METHODS:
            data = row.get(method, {})
            if not isinstance(data, dict):
                continue
            for metric in metrics:
                value = _higher_better_metric_value(data.get(metric), metric)
                pooled[metric].append(value if value is not None else np.nan)
    names, mat = _spearman_corr_matrix(pooled)

    contrast_rows = []
    for baseline in ("EMD", "EEMD", "CEEMDAN"):
        table = defaultdict(list)
        for row in case_rows:
            for metric in metrics:
                iv = _metric_value(row, "IRMF", metric)
                bv = _metric_value(row, baseline, metric)
                if iv is None or bv is None:
                    table[metric].append(np.nan)
                else:
                    delta = iv - bv
                    table[metric].append(-delta if metric in LOWER_IS_BETTER else delta)
        cnames, cmat = _spearman_corr_matrix(table)
        contrast_rows.extend(_correlation_rows(cnames, cmat, f"paired_contrast_IRMF_vs_{baseline}", dimension=dimension))

    return {
        "dimension": dimension,
        "pooled_names": names,
        "pooled_matrix": mat,
        "pooled_rows": _correlation_rows(names, mat, "pooled_higher_is_better_values", dimension=dimension),
        "contrast_rows": contrast_rows,
        "clusters": _redundancy_clusters(names, mat, dimension=dimension),
    }


def _redundancy_analysis(case_rows):
    structural_metrics = (
        "imf_recovery_corr",
        "imf_recovery_nrmse",
        "component_splitting_index",
        "component_merging_index",
        "inter_imf_entanglement_index",
        "mode_mixing_index",
        "decomposition_count_error",
        "strict_io",
        "spectral_leakage",
        "frequency_overlap_max_offdiag",
        "frequency_separation_score",
        "over_decomposition_penalty",
        "under_decomposition_index",
        "imf_count",
        "effective_imf_count",
    )
    contamination_metrics = (
        "outlier_resistance_index",
        "noise_capture_corr",
        "noise_capture_corr_score",
        "clean_region_nmse",
        "contaminated_region_nmse",
        "contamination_spillover_error",
        "signal_leakage_into_noise",
        "signal_leakage_into_noise_score",
        "clean_region_signal_leakage",
    )
    bundles = {
        "structural_fidelity": _redundancy_bundle(case_rows, "structural_fidelity", structural_metrics),
        "contamination_resistance": _redundancy_bundle(case_rows, "contamination_resistance", contamination_metrics),
    }
    pooled_rows = []
    contrast_rows = []
    clusters = []
    for bundle in bundles.values():
        pooled_rows.extend(bundle["pooled_rows"])
        contrast_rows.extend(bundle["contrast_rows"])
        clusters.extend(bundle["clusters"])
    return {
        "bundles": bundles,
        "pooled_rows": pooled_rows,
        "contrast_rows": contrast_rows,
        "clusters": clusters,
    }


def _case_score_comparison(case_rows, overall_rows):
    rows = []
    by_block_overall = defaultdict(dict)
    for row in overall_rows:
        by_block_overall[row["block_id"]][row["method"]] = row.get("hierarchical_overall_rank")
    for row in case_rows:
        values = {method: _metric_value(row, method, "case_score") for method in METHODS}
        ranks = _rank_values(values, higher_is_better=True)
        for method in METHODS:
            if method not in ranks and method not in by_block_overall.get(row["block_id"], {}):
                continue
            rows.append({
                "block_id": row["block_id"],
                "signal": row.get("signal"),
                "noise": row.get("noise"),
                "sigma": row.get("sigma"),
                "method": method,
                "legacy_case_score": values.get(method),
                "legacy_case_score_rank": ranks.get(method),
                "case_score_role": CASE_SCORE_ROLE,
                "hierarchical_overall_rank": by_block_overall.get(row["block_id"], {}).get(method),
            })
    return rows


def run_primary_evaluation_framework(output_root, family_rows):
    output_root = ensure_dir(output_root)
    case_rows = _aggregate_seed_rows(family_rows)
    metric_rows, subdimension_rows, dimension_rows, overall_rows = _compute_hierarchical_ranks(case_rows)
    friedman = _friedman_rows(dimension_rows)
    redundancy = _redundancy_analysis(case_rows)
    case_score_rows = _case_score_comparison(case_rows, overall_rows)

    write_json(case_rows, output_root / "case_level_seed_aggregated_rows.json")
    write_csv(metric_rows, output_root / "metric_level_primary_ranks.csv")
    write_json(metric_rows, output_root / "metric_level_primary_ranks.json")
    write_csv(subdimension_rows, output_root / "subdimension_level_ranks.csv")
    write_json(subdimension_rows, output_root / "subdimension_level_ranks.json")
    write_csv(dimension_rows, output_root / "dimension_level_ranks.csv")
    write_json(dimension_rows, output_root / "dimension_level_ranks.json")
    write_csv(overall_rows, output_root / "hierarchical_overall_ranks_descriptive.csv")
    write_json(overall_rows, output_root / "hierarchical_overall_ranks_descriptive.json")
    write_csv(friedman, output_root / "dimension_specific_friedman_cd_ranking.csv")
    write_json(friedman, output_root / "dimension_specific_friedman_cd_ranking.json")
    write_csv(case_score_rows, output_root / "legacy_case_score_vs_hierarchical_rank.csv")
    write_json(case_score_rows, output_root / "legacy_case_score_vs_hierarchical_rank.json")

    write_csv(redundancy["pooled_rows"], output_root / "metric_redundancy_spearman_pooled.csv")
    write_csv(redundancy["contrast_rows"], output_root / "metric_redundancy_spearman_paired_contrasts.csv")
    write_csv(redundancy["clusters"], output_root / "metric_redundancy_clusters.csv")
    write_json(redundancy["clusters"], output_root / "metric_redundancy_clusters.json")
    for dimension, bundle in redundancy["bundles"].items():
        label = dimension.replace("_", " ").title()
        _write_heatmap(
            bundle["pooled_names"],
            bundle["pooled_matrix"],
            output_root / f"metric_correlation_heatmap_{dimension}.png",
            f"{label} Metric Redundancy: Spearman Correlation",
        )
        _write_dendrogram(
            bundle["pooled_names"],
            bundle["pooled_matrix"],
            output_root / f"metric_clustering_dendrogram_{dimension}.png",
            f"{label} Metric Redundancy Clustering",
        )

    protocol = {
        "section": "V5.20 Primary Evaluation Framework",
        "evaluation_framework_version": EVALUATION_FRAMEWORK_VERSION,
        "dimension_estimands": EVALUATION_DIMENSION_ESTIMANDS,
        "case_score_policy": {
            "case_score_retained": True,
            "case_score_role": CASE_SCORE_ROLE,
            "not_used_for_primary_ranking": True,
        },
        "ranking_unit": "signal x noise x sigma benchmark block; multiple seed rows are aggregated within block by median before ranking",
        "rank_flow": [
            "raw metric value",
            "within-case, within-metric method rank",
            "average metric ranks within subdimension",
            "within-case subdimension method rank",
            "average subdimension ranks within dimension",
            "within-case dimension method rank",
            "optional descriptive hierarchical overall rank",
        ],
        "primary_endpoints_by_dimension": PRIMARY_ENDPOINTS_BY_DIMENSION,
        "conditional_primary_endpoints_by_dimension": CONDITIONAL_PRIMARY_ENDPOINTS_BY_DIMENSION,
        "conditional_endpoints_by_dimension": CONDITIONAL_ENDPOINTS_BY_DIMENSION,
        "diagnostic_metrics_by_dimension": DIAGNOSTIC_METRICS_BY_DIMENSION,
        "contamination_policy": (
            "Universal primary endpoints include noise_capture_corr and "
            "signal_leakage_into_noise across all synthetic cases. "
            "Contamination-specific primary endpoints outlier_resistance_index "
            "and clean_region_nmse are evaluated only for explicit "
            "contamination-aware regimes. contaminated_region_nmse and "
            "contamination_spillover_error remain conditional diagnostics."
        ),
        "redundancy_analysis": {
            "purpose": "Validate that collinear structural and contamination diagnostics are not double-counted as primary endpoints.",
            "selection_policy": "Primary endpoints are pre-specified from scientific roles; redundancy analysis is confirmatory/diagnostic, not post-hoc metric shopping.",
            "correlation": "Spearman rank correlation after orienting metric values so higher is better.",
            "views": [
                "pooled method-case values",
                "paired IRMF-vs-baseline contrasts",
            ],
            "dimensions": list(redundancy["bundles"].keys()),
        },
        "overall_rank_policy": "hierarchical_overall_rank is descriptive_only; dimension-specific ranks are the inferential summaries.",
    }
    write_json(protocol, output_root / "primary_evaluation_framework_protocol.json")
    return {
        "n_case_blocks": len(case_rows),
        "n_metric_rank_rows": len(metric_rows),
        "n_subdimension_rank_rows": len(subdimension_rows),
        "n_dimension_rank_rows": len(dimension_rows),
        "n_overall_rank_rows": len(overall_rows),
        "n_friedman_rows": len(friedman),
        "n_redundancy_clusters": len(redundancy["clusters"]),
    }
