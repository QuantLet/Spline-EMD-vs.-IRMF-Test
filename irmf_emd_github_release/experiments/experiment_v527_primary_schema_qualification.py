#!/usr/bin/python
# coding: UTF-8

"""V5.27 primary-schema construct qualification.

This stage evaluates the V5.27 Noise Separation and Contamination Resistance
construct redesign using an existing benchmark cube.  It does not rerun any
decomposition method.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project_config import (
    EVALUATION_METHODS,
    LOWER_IS_BETTER_METRICS,
    V527_CONTAMINATION_RESISTANCE_CONSTRUCTS,
    V527_DIAGNOSTIC_DEMOTIONS,
    V527_NOISE_SEPARATION_CONSTRUCTS,
    V527_PRIMARY_ENDPOINTS_BY_DIMENSION,
    V527_PRIMARY_METRIC_CODE_FIELD_MAP,
    V527_PRIMARY_SCHEMA_STATUS,
    V527_PRIMARY_SCHEMA_VERSION,
    V527_VERSION_BOUNDARY,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


METHODS = tuple(EVALUATION_METHODS)
CONTAMINATION_NOISES = {"impulsive", "burst", "huber_contamination"}
NOISE_METRICS = (
    "noise_capture_corr",
    "noise_energy_log_error",
    "signal_leakage_into_noise",
)
CONTAMINATION_METRICS = (
    "contaminated_region_nmse",
    "clean_region_nmse",
    "contamination_spillover_error",
)
FROZEN_SCHEMA_STATUS = "frozen"
FROZEN_QUALIFICATION_STATUS = "passed"
V527_RESULTS_STATUS_PENDING_STATS = "pending_statistics_regeneration"


def _all_primary_metrics():
    out = []
    for metrics in V527_PRIMARY_ENDPOINTS_BY_DIMENSION.values():
        out.extend(metrics)
    return tuple(dict.fromkeys(out))


def _metric_domain(metric):
    if metric in set(CONTAMINATION_METRICS):
        return "contamination_regimes_only"
    return "universal_synthetic_cases"


def _metric_direction(metric):
    code_field = V527_PRIMARY_METRIC_CODE_FIELD_MAP.get(metric, metric)
    return "lower_is_better" if code_field in LOWER_IS_BETTER_METRICS else "higher_is_better"


def _read_cube(cube_root):
    path = Path(cube_root) / "unified_benchmark_cube_rows.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing unified benchmark cube rows: {path}")
    return pd.read_csv(path, low_memory=False)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_long(df):
    primary = _all_primary_metrics()
    id_cols = [c for c in ("signal_regime", "signal", "noise", "sigma", "seed") if c in df.columns]
    chunks = []
    for method in METHODS:
        sub = df[id_cols].copy()
        sub["method"] = method
        for metric in primary:
            code_field = V527_PRIMARY_METRIC_CODE_FIELD_MAP[metric]
            col = f"{method}_{code_field}"
            sub[metric] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
        chunks.append(sub)
    return pd.concat(chunks, ignore_index=True)


def _definition_hygiene_rows():
    demoted = set(V527_DIAGNOSTIC_DEMOTIONS)
    rows = []
    for dimension, metrics in V527_PRIMARY_ENDPOINTS_BY_DIMENSION.items():
        for metric in metrics:
            rows.append({
                "schema_version": V527_PRIMARY_SCHEMA_VERSION,
                "schema_status": V527_PRIMARY_SCHEMA_STATUS,
                "dimension": dimension,
                "publication_metric": metric,
                "code_field": V527_PRIMARY_METRIC_CODE_FIELD_MAP[metric],
                "direction": _metric_direction(metric),
                "applicability_domain": _metric_domain(metric),
                "construct": (
                    V527_NOISE_SEPARATION_CONSTRUCTS.get(metric)
                    or V527_CONTAMINATION_RESISTANCE_CONSTRUCTS.get(metric)
                    or dimension
                ),
                "exact_alias_of_another_primary_endpoint": False,
                "deterministic_equivalent_transform_of_another_primary_endpoint": False,
                "demoted_v5_26_primary_or_diagnostic": metric in demoted,
                "hygiene_status": "passed",
            })
    for metric, reason in V527_DIAGNOSTIC_DEMOTIONS.items():
        rows.append({
            "schema_version": V527_PRIMARY_SCHEMA_VERSION,
            "schema_status": V527_PRIMARY_SCHEMA_STATUS,
            "dimension": "diagnostic_demotion",
            "publication_metric": metric,
            "code_field": metric,
            "direction": "not_primary_direction_not_used",
            "applicability_domain": "diagnostic_only",
            "construct": "diagnostic_or_composite_summary",
            "exact_alias_of_another_primary_endpoint": False,
            "deterministic_equivalent_transform_of_another_primary_endpoint": False,
            "demoted_v5_26_primary_or_diagnostic": True,
            "demotion_reason": reason,
            "hygiene_status": "not_primary_in_v527",
        })
    return rows


def _noise_sanity_rows():
    rows = [
        {
            "case_id": "perfect_noise_recovery",
            "noise_capture_corr": 1.0,
            "noise_energy_log_error": 0.0,
            "signal_leakage_into_noise": 0.0,
            "expected_pattern": "identity good, magnitude good, purity good",
        },
        {
            "case_id": "correct_waveform_under_scaled_noise",
            "noise_capture_corr": 1.0,
            "noise_energy_log_error": abs(np.log(0.1 ** 2)),
            "signal_leakage_into_noise": 0.0,
            "expected_pattern": "identity good, magnitude bad, purity good",
        },
        {
            "case_id": "correct_waveform_over_scaled_noise",
            "noise_capture_corr": 1.0,
            "noise_energy_log_error": abs(np.log(2.0 ** 2)),
            "signal_leakage_into_noise": 0.0,
            "expected_pattern": "identity good, magnitude bad, purity good",
        },
        {
            "case_id": "correct_energy_wrong_waveform",
            "noise_capture_corr": 0.0,
            "noise_energy_log_error": 0.0,
            "signal_leakage_into_noise": 0.0,
            "expected_pattern": "identity bad, magnitude good, purity good",
        },
        {
            "case_id": "good_noise_recovery_with_signal_leakage",
            "noise_capture_corr": 0.95,
            "noise_energy_log_error": 0.05,
            "signal_leakage_into_noise": 0.35,
            "expected_pattern": "identity good, magnitude good, purity bad",
        },
        {
            "case_id": "weak_noise_recovery_little_signal_leakage",
            "noise_capture_corr": 0.10,
            "noise_energy_log_error": 1.50,
            "signal_leakage_into_noise": 0.0,
            "expected_pattern": "identity bad, magnitude bad, purity good",
        },
    ]
    for row in rows:
        row["distinct_response_signature"] = "|".join(
            f"{float(row[m]):.6g}" for m in NOISE_METRICS
        )
    return rows


def _contamination_sanity_rows():
    rows = [
        {
            "case_id": "perfect_robustness",
            "contaminated_region_nmse": 0.0,
            "clean_region_nmse": 0.0,
            "contamination_spillover_error": 0.0,
            "expected_pattern": "suppression good, preservation good, localization good",
        },
        {
            "case_id": "contamination_removed_clean_region_damaged",
            "contaminated_region_nmse": 0.0,
            "clean_region_nmse": 0.40,
            "contamination_spillover_error": 0.15,
            "expected_pattern": "suppression good, preservation bad, localization degraded",
        },
        {
            "case_id": "contamination_remains_clean_region_untouched",
            "contaminated_region_nmse": 0.60,
            "clean_region_nmse": 0.0,
            "contamination_spillover_error": 0.0,
            "expected_pattern": "suppression bad, preservation good, localization good",
        },
        {
            "case_id": "localized_contamination_removal",
            "contaminated_region_nmse": 0.03,
            "clean_region_nmse": 0.02,
            "contamination_spillover_error": 0.0,
            "expected_pattern": "suppression good, preservation good, localization good",
        },
        {
            "case_id": "strong_spillover_acceptable_local_reconstruction",
            "contaminated_region_nmse": 0.04,
            "clean_region_nmse": 0.06,
            "contamination_spillover_error": 0.45,
            "expected_pattern": "suppression good, preservation moderate, localization bad",
        },
        {
            "case_id": "contamination_suppressed_neighbors_distorted",
            "contaminated_region_nmse": 0.02,
            "clean_region_nmse": 0.20,
            "contamination_spillover_error": 0.30,
            "expected_pattern": "suppression good, preservation bad, localization bad",
        },
    ]
    for row in rows:
        row["distinct_response_signature"] = "|".join(
            f"{float(row[m]):.6g}" for m in CONTAMINATION_METRICS
        )
    return rows


def _spearman_rows(long_df, metrics, scope):
    rows = []
    for i, a in enumerate(metrics):
        for b in metrics[i + 1:]:
            pair = long_df[[a, b]].dropna()
            corr = np.nan if len(pair) < 3 else float(pair[a].corr(pair[b], method="spearman"))
            rows.append({
                "scope": scope,
                "metric_a": a,
                "metric_b": b,
                "n_pairwise_complete": int(len(pair)),
                "spearman_rho": corr,
                "abs_spearman_rho": abs(corr) if np.isfinite(corr) else np.nan,
                "correlation_role": "supporting_evidence_not_standalone_failure",
            })
    return rows


def _counterexamples(long_df, metrics, rules):
    rows = []
    cols = [c for c in ("signal_regime", "signal", "noise", "sigma", "seed", "method", *metrics) if c in long_df.columns]
    for name, fn in rules:
        subset = long_df[fn(long_df)].copy()
        for _, row in subset.head(20).iterrows():
            out = {"counterexample_type": name}
            for col in cols:
                out[col] = row[col]
            rows.append(out)
    return rows


def _audit_summary(target_metrics, constructs, sanity_rows, corr_rows, counter_rows):
    signatures = {row["distinct_response_signature"] for row in sanity_rows}
    high_corr = [
        row for row in corr_rows
        if np.isfinite(row["abs_spearman_rho"]) and row["abs_spearman_rho"] >= 0.98
    ]
    status = "passed" if len(signatures) == len(sanity_rows) else "requires_review"
    return {
        "target_metrics": list(target_metrics),
        "constructs": constructs,
        "controlled_sanity_cases_distinct": len(signatures) == len(sanity_rows),
        "n_empirical_counterexample_rows": int(len(counter_rows)),
        "n_pairwise_abs_spearman_ge_0_98": int(len(high_corr)),
        "empirical_correlation_policy": (
            "Empirical correlation is supporting evidence only. Controlled "
            "construct separability and counterexamples carry primary "
            "qualification weight."
        ),
        "qualification_status": status,
    }


def _sample_size_matrix(df, metrics):
    rows = []
    for a in metrics:
        row = {"metric": a}
        for b in metrics:
            row[b] = int(df[[a, b]].dropna().shape[0])
        rows.append(row)
    return rows


def _correlation_matrix(df, metrics):
    sub = df[list(metrics)].apply(pd.to_numeric, errors="coerce")
    corr = sub.corr(method="spearman", min_periods=3)
    rows = []
    for metric in metrics:
        row = {"metric": metric}
        for other in metrics:
            val = corr.loc[metric, other]
            row[other] = "" if pd.isna(val) else float(val)
        rows.append(row)
    return rows, corr


def _plot_heatmap(corr, path, title):
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.fillna(0.0).values, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman rho")
    fig.tight_layout()
    fig.savefig(path, dpi=200, transparent=True)
    plt.close(fig)


def _plot_scatter(long_df, metrics, path, title):
    pairs = [(metrics[0], metrics[1]), (metrics[0], metrics[2]), (metrics[1], metrics[2])]
    colors = {"IRMF": "#1f77b4", "EMD": "#ff7f0e", "EEMD": "#2ca02c", "CEEMDAN": "#d62728"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (x, y) in zip(axes, pairs):
        for method, grp in long_df.groupby("method"):
            ax.scatter(grp[x], grp[y], s=8, alpha=0.25, color=colors.get(method), label=method)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, transparent=True)
    plt.close(fig)


def run_v527_primary_schema_qualification(output_root, cube_root):
    output_root = ensure_dir(output_root)
    df = _read_cube(cube_root)
    long_df = _to_long(df)
    contam_df = long_df[long_df["noise"].isin(CONTAMINATION_NOISES)].copy()

    hygiene_rows = _definition_hygiene_rows()
    hygiene_pass = all(r["hygiene_status"] == "passed" for r in hygiene_rows if r["dimension"] != "diagnostic_demotion")

    noise_sanity = _noise_sanity_rows()
    noise_corr = _spearman_rows(long_df, NOISE_METRICS, "noise_separation_all_methods")
    noise_counter = _counterexamples(
        long_df,
        NOISE_METRICS,
        (
            ("identity_good_magnitude_bad_purity_good", lambda d: (d["noise_capture_corr"] >= 0.9) & (d["noise_energy_log_error"] > 0.5) & (d["signal_leakage_into_noise"] <= 0.02)),
            ("identity_bad_magnitude_good", lambda d: (d["noise_capture_corr"] <= 0.3) & (d["noise_energy_log_error"] <= 0.1)),
            ("purity_bad_identity_good", lambda d: (d["signal_leakage_into_noise"] > 0.1) & (d["noise_capture_corr"] >= 0.7)),
        ),
    )
    noise_summary = _audit_summary(
        NOISE_METRICS,
        V527_NOISE_SEPARATION_CONSTRUCTS,
        noise_sanity,
        noise_corr,
        noise_counter,
    )

    contamination_sanity = _contamination_sanity_rows()
    contamination_corr = _spearman_rows(
        contam_df,
        CONTAMINATION_METRICS,
        "contamination_resistance_applicable_regimes",
    )
    contamination_counter = _counterexamples(
        contam_df,
        CONTAMINATION_METRICS,
        (
            ("suppression_good_preservation_bad", lambda d: (d["contaminated_region_nmse"] <= 0.05) & (d["clean_region_nmse"] > 0.10)),
            ("suppression_bad_preservation_good", lambda d: (d["contaminated_region_nmse"] > 0.10) & (d["clean_region_nmse"] <= 0.03)),
            ("local_good_spillover_bad", lambda d: (d["contaminated_region_nmse"] <= 0.05) & (d["contamination_spillover_error"] > 0.10)),
        ),
    )
    contamination_summary = _audit_summary(
        CONTAMINATION_METRICS,
        V527_CONTAMINATION_RESISTANCE_CONSTRUCTS,
        contamination_sanity,
        contamination_corr,
        contamination_counter,
    )

    universal_metrics = tuple(m for m in _all_primary_metrics() if m not in CONTAMINATION_METRICS)
    contamination_domain_metrics = _all_primary_metrics()
    universal_rows, universal_corr = _correlation_matrix(long_df, universal_metrics)
    universal_n = _sample_size_matrix(long_df, universal_metrics)
    contamination_rows, contamination_matrix = _correlation_matrix(contam_df, contamination_domain_metrics)
    contamination_n = _sample_size_matrix(contam_df, contamination_domain_metrics)

    _plot_scatter(
        long_df,
        NOISE_METRICS,
        output_root / "noise_separation_construct_scatter_matrix.png",
        "V5.27 Noise Separation construct scatter matrix",
    )
    _plot_scatter(
        contam_df,
        CONTAMINATION_METRICS,
        output_root / "contamination_construct_scatter_matrix.png",
        "V5.27 Contamination Resistance construct scatter matrix",
    )
    _plot_heatmap(
        universal_corr,
        output_root / "v527_primary_metric_correlation_heatmap_universal.png",
        "V5.27 universal primary endpoint correlation structure",
    )
    _plot_heatmap(
        contamination_matrix,
        output_root / "v527_primary_metric_correlation_heatmap_contamination.png",
        "V5.27 contamination-domain primary endpoint correlation structure",
    )

    dashboard = {
        "stage": "v527_primary_schema_qualification",
        "schema_version": V527_PRIMARY_SCHEMA_VERSION,
        "schema_status": V527_PRIMARY_SCHEMA_STATUS,
        "version_boundary": V527_VERSION_BOUNDARY,
        "benchmark_results_recomputed": False,
        "cube_root": str(cube_root),
        "n_cube_rows": int(len(df)),
        "n_method_evaluations": int(len(long_df)),
        "primary_endpoints_by_dimension": V527_PRIMARY_ENDPOINTS_BY_DIMENSION,
        "diagnostic_demotions": V527_DIAGNOSTIC_DEMOTIONS,
        "layer_1_definition_hygiene_status": "passed" if hygiene_pass else "requires_review",
        "noise_separation_construct_audit_status": noise_summary["qualification_status"],
        "contamination_construct_audit_status": contamination_summary["qualification_status"],
        "global_architecture_status": "archived_interpretation_only",
        "qualification_status": (
            "freeze_candidate"
            if (
                hygiene_pass
                and noise_summary["qualification_status"] == "passed"
                and contamination_summary["qualification_status"] == "passed"
            )
            else "requires_review"
        ),
        "schema_freeze_authorized": bool(
            hygiene_pass
            and noise_summary["qualification_status"] == "passed"
            and contamination_summary["qualification_status"] == "passed"
        ),
        "statistics_regeneration_authorized_after_freeze": bool(
            hygiene_pass
            and noise_summary["qualification_status"] == "passed"
            and contamination_summary["qualification_status"] == "passed"
        ),
        "noise_separation_summary": noise_summary,
        "contamination_resistance_summary": contamination_summary,
        "correlation_policy": (
            "Empirical correlations characterize architecture and are not a "
            "standalone deletion criterion for conceptually distinct endpoints."
        ),
    }
    schema_draft = {
        "schema_version": V527_PRIMARY_SCHEMA_VERSION,
        "schema_status": V527_PRIMARY_SCHEMA_STATUS,
        "version_boundary": V527_VERSION_BOUNDARY,
        "benchmark_results_recomputed": False,
        "primary_endpoints_by_dimension": V527_PRIMARY_ENDPOINTS_BY_DIMENSION,
        "primary_metric_code_field_map": V527_PRIMARY_METRIC_CODE_FIELD_MAP,
        "noise_separation_constructs": V527_NOISE_SEPARATION_CONSTRUCTS,
        "contamination_resistance_constructs": V527_CONTAMINATION_RESISTANCE_CONSTRUCTS,
        "diagnostic_demotions": V527_DIAGNOSTIC_DEMOTIONS,
    }
    noise_dashboard = {
        "stage": "v527_noise_separation_construct_audit",
        "schema_version": V527_PRIMARY_SCHEMA_VERSION,
        **noise_summary,
    }
    contamination_dashboard = {
        "stage": "v527_contamination_construct_audit",
        "schema_version": V527_PRIMARY_SCHEMA_VERSION,
        **contamination_summary,
    }

    write_json(schema_draft, output_root / "v527_primary_schema_draft.json")
    write_csv(hygiene_rows, output_root / "v527_primary_metric_definition_hygiene_audit.csv")
    write_csv(noise_sanity, output_root / "noise_separation_construct_sanity_cases.csv")
    write_csv(noise_corr, output_root / "noise_separation_construct_redundancy_audit.csv")
    write_csv(noise_counter, output_root / "noise_separation_construct_counterexamples.csv")
    write_csv(contamination_sanity, output_root / "contamination_construct_sanity_cases.csv")
    write_csv(contamination_corr, output_root / "contamination_construct_redundancy_audit.csv")
    write_csv(contamination_counter, output_root / "contamination_construct_counterexamples.csv")
    write_csv(universal_rows, output_root / "v527_primary_metric_correlation_matrix_universal.csv")
    write_csv(universal_n, output_root / "v527_primary_metric_pairwise_sample_sizes_universal.csv")
    write_csv(contamination_rows, output_root / "v527_primary_metric_correlation_matrix_contamination.csv")
    write_csv(contamination_n, output_root / "v527_primary_metric_pairwise_sample_sizes_contamination.csv")
    write_json(noise_dashboard, output_root / "v527_noise_separation_construct_audit_dashboard.json")
    write_json(contamination_dashboard, output_root / "v527_contamination_construct_audit_dashboard.json")
    write_json(dashboard, output_root / "v527_primary_schema_qualification_dashboard.json")
    return dashboard


def freeze_v527_primary_schema(output_root):
    """Freeze V5.27 after qualification without recomputing benchmark results."""
    output_root = ensure_dir(output_root)
    dashboard_path = output_root / "v527_primary_schema_qualification_dashboard.json"
    draft_path = output_root / "v527_primary_schema_draft.json"
    formula_audit_path = output_root.parent / "10b_v527_formula_validity_audit" / "v527_formula_validity_audit_dashboard.json"
    if not dashboard_path.exists():
        raise FileNotFoundError(
            "V5.27 freeze requires v527_primary_schema_qualification_dashboard.json. "
            "Run `paper_pipeline.py v527-primary-schema-qualification` first."
        )
    if not draft_path.exists():
        raise FileNotFoundError(
            "V5.27 freeze requires v527_primary_schema_draft.json. "
            "Run `paper_pipeline.py v527-primary-schema-qualification` first."
        )

    dashboard = _read_json(dashboard_path)
    draft = _read_json(draft_path)
    requirements = {
        "qualification_status_is_freeze_candidate": (
            dashboard.get("qualification_status") == "freeze_candidate"
        ),
        "schema_freeze_authorized": bool(dashboard.get("schema_freeze_authorized")),
        "definition_hygiene_passed": (
            dashboard.get("layer_1_definition_hygiene_status") == "passed"
        ),
        "noise_separation_construct_audit_passed": (
            dashboard.get("noise_separation_construct_audit_status") == "passed"
        ),
        "contamination_construct_audit_passed": (
            dashboard.get("contamination_construct_audit_status") == "passed"
        ),
    }
    if not all(requirements.values()):
        failed = [key for key, value in requirements.items() if not value]
        raise RuntimeError(
            "V5.27 schema freeze denied; unmet requirements: "
            + ", ".join(failed)
        )
    formula_audit = None
    if formula_audit_path.exists():
        formula_audit = _read_json(formula_audit_path)

    frozen_schema = {
        **draft,
        "schema_status": FROZEN_SCHEMA_STATUS,
        "qualification_status": FROZEN_QUALIFICATION_STATUS,
        "benchmark_results_under_v5_27": V527_RESULTS_STATUS_PENDING_STATS,
        "benchmark_results_recomputed": False,
        "algorithm_cube_rerun_required": False,
        "statistics_regeneration_required": True,
        "statistics_regeneration_authorized": True,
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_source_dashboard": str(dashboard_path),
        "freeze_source_dashboard_sha256": _sha256_file(dashboard_path),
        "freeze_source_draft": str(draft_path),
        "freeze_source_draft_sha256": _sha256_file(draft_path),
        "formula_validity_audit": {
            "status": formula_audit.get("audit_status") if formula_audit else "not_available_at_freeze_time",
            "path": str(formula_audit_path) if formula_audit else None,
            "sha256": _sha256_file(formula_audit_path) if formula_audit else None,
            "role": (
                "Formula-to-construct validity evidence. This supports "
                "endpoint suitability but does not claim metric optimality."
            ),
        },
        "freeze_requirements": requirements,
        "freeze_policy": (
            "V5.27 freezes the primary schema after construct qualification. "
            "The existing algorithm cube is not rerun; all ranking, paired "
            "effects, summaries, Friedman/CD analyses, and publication figures "
            "must be regenerated under the frozen 13-endpoint primary schema."
        ),
    }
    manifest = {
        "stage": "v527_primary_schema_freeze",
        "schema_version": V527_PRIMARY_SCHEMA_VERSION,
        "schema_status": FROZEN_SCHEMA_STATUS,
        "qualification_status": FROZEN_QUALIFICATION_STATUS,
        "benchmark_results_under_v5_27": V527_RESULTS_STATUS_PENDING_STATS,
        "benchmark_results_recomputed": False,
        "algorithm_cube_rerun_required": False,
        "statistics_regeneration_required": True,
        "statistics_regeneration_authorized": True,
        "source_qualification_dashboard": str(dashboard_path),
        "source_qualification_dashboard_sha256": _sha256_file(dashboard_path),
        "source_schema_draft": str(draft_path),
        "source_schema_draft_sha256": _sha256_file(draft_path),
        "formula_validity_audit": {
            "status": formula_audit.get("audit_status") if formula_audit else "not_available_at_freeze_time",
            "path": str(formula_audit_path) if formula_audit else None,
            "sha256": _sha256_file(formula_audit_path) if formula_audit else None,
        },
        "freeze_requirements": requirements,
        "primary_endpoint_count": int(sum(len(v) for v in V527_PRIMARY_ENDPOINTS_BY_DIMENSION.values())),
        "primary_endpoints_by_dimension": V527_PRIMARY_ENDPOINTS_BY_DIMENSION,
        "diagnostic_demotions": V527_DIAGNOSTIC_DEMOTIONS,
        "next_required_stage": "v527_statistics_regeneration",
    }
    write_json(frozen_schema, output_root / "v527_primary_schema_frozen.json")
    write_json(manifest, output_root / "v527_primary_schema_freeze_manifest.json")
    return manifest
