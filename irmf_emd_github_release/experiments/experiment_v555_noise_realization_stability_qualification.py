#!/usr/bin/python
# coding: UTF-8

"""V5.55 qualification for V5.54 noise-realization stability outputs."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
from pathlib import Path

import numpy as np

from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


V555_NOISE_REALIZATION_STABILITY_QUALIFICATION_VERSION = (
    "V5.55_noise_realization_stability_qualification"
)
SOURCE_REL = Path("06_protocol_controlled_full_benchmark_rerun") / "unified_benchmark_cube_rows.csv"
V554_REL = Path("16w_v554_noise_realization_stability")
METHODS = ("IRMF", "EMD", "EEMD", "CEEMDAN")
PREFIXES = (5, 10, 15, 20)


def _finite(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return None


def _int_or_none(value):
    value = _finite(value)
    if value is None:
        return None
    return int(round(value))


def _read_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _target_snr(row):
    value = row.get("target_snr_db")
    if value in (None, ""):
        value = row.get("sigma")
    parsed = _finite(value)
    return str(int(parsed)) if parsed is not None and float(parsed).is_integer() else str(value)


def _group_key(row):
    return (
        row.get("signal_regime"),
        row.get("signal"),
        row.get("noise"),
        _target_snr(row),
    )


def _seed(row):
    return _int_or_none(row.get("data_seed") or row.get("seed"))


def _method_value(row, method, metric):
    return row.get(f"{method}_{metric}")


def _build_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[_group_key(row)].append(row)
    for key in groups:
        groups[key] = sorted(groups[key], key=lambda r: (_seed(r) is None, _seed(r)))
    return groups


def _component_count_fraction(rows, method, count_metric):
    counts = [_int_or_none(_method_value(row, method, count_metric)) for row in rows]
    counts = [v for v in counts if v is not None]
    if not counts:
        return None
    hist = Counter(counts)
    return float(hist.most_common(1)[0][1] / len(counts))


def _recovery_frequency(rows, method):
    recovered_total = 0.0
    true_total = 0.0
    for row in rows:
        k_true = _int_or_none(_method_value(row, method, "true_component_count"))
        missing = _int_or_none(_method_value(row, method, "missing_true_component_count"))
        matched = _int_or_none(_method_value(row, method, "imf_recovery_matched_count"))
        if k_true is None or k_true <= 0:
            continue
        if missing is not None:
            recovered = max(0, k_true - missing)
        elif matched is not None:
            recovered = min(k_true, matched)
        else:
            continue
        recovered_total += float(recovered)
        true_total += float(k_true)
    if true_total <= 0:
        return None
    return float(recovered_total / true_total)


def _summary(values):
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {
            "n": 0, "mean": None, "median": None, "q25": None,
            "q75": None, "min": None, "max": None,
        }
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _rank_methods(method_values, higher=True):
    clean = {m: v for m, v in method_values.items() if v is not None and np.isfinite(v)}
    return [
        {
            "method": method,
            "value": float(value),
            "rank": int(i + 1),
        }
        for i, (method, value) in enumerate(
            sorted(clean.items(), key=lambda item: item[1], reverse=higher)
        )
    ]


def _metric_validity_audit(v554_root):
    required = {
        "component_count": v554_root / "v554_component_count_stability_across_noise_realizations.csv",
        "recovery_frequency": v554_root / "v554_truth_conditioned_component_recovery_frequency.csv",
        "cross_realization_status": v554_root / "v554_cross_realization_matched_component_similarity_status.csv",
        "dashboard": v554_root / "v554_noise_realization_stability_dashboard.json",
    }
    rows = []
    for name, path in required.items():
        rows.append({
            "check": f"{name}_artifact_exists",
            "passed": path.exists(),
            "path": str(path),
        })
    status_rows = _read_rows(required["cross_realization_status"]) if required["cross_realization_status"].exists() else []
    status_ok = bool(status_rows) and status_rows[0].get("status") == "not_computable_from_current_compact_benchmark_rows"
    rows.append({
        "check": "cross_realization_similarity_not_falsely_computed",
        "passed": status_ok,
        "path": str(required["cross_realization_status"]),
    })
    return rows


def _aggregation_validity_audit(rows, groups):
    out = []
    seed_counts = [len({_seed(row) for row in case_rows if _seed(row) is not None}) for case_rows in groups.values()]
    out.append({
        "check": "source_rows_present",
        "passed": len(rows) > 0,
        "observed": len(rows),
        "expected": 16800,
    })
    out.append({
        "check": "signal_noise_snr_groups_present",
        "passed": len(groups) == 840,
        "observed": len(groups),
        "expected": 840,
    })
    out.append({
        "check": "all_groups_have_20_monte_carlo_realizations",
        "passed": bool(seed_counts) and min(seed_counts) == 20 and max(seed_counts) == 20,
        "observed_min": min(seed_counts) if seed_counts else None,
        "observed_max": max(seed_counts) if seed_counts else None,
        "expected": 20,
    })
    method_missing = []
    for key, case_rows in groups.items():
        for method in METHODS:
            if not any(_method_value(row, method, "denoise_nmse") not in (None, "") for row in case_rows):
                method_missing.append((*key, method))
    out.append({
        "check": "all_methods_present_in_each_group",
        "passed": len(method_missing) == 0,
        "observed_missing_method_groups": len(method_missing),
        "expected": 0,
    })
    return out


def _prefix_stability(rows, groups):
    rows_out = []
    ordering_rows = []
    for prefix in PREFIXES:
        metric_values = defaultdict(lambda: defaultdict(list))
        for key, case_rows in groups.items():
            prefix_rows = [row for row in case_rows if (_seed(row) is not None and _seed(row) < prefix)]
            if len(prefix_rows) != prefix:
                continue
            for method in METHODS:
                for count_metric in ("effective_imf_count", "protocol_selected_component_count"):
                    value = _component_count_fraction(prefix_rows, method, count_metric)
                    metric_values[("component_count_stability_across_noise_realizations", count_metric)][method].append(value)
                    rows_out.append({
                        "prefix_n_realizations": int(prefix),
                        "diagnostic": "component_count_stability_across_noise_realizations",
                        "submetric": count_metric,
                        "signal_regime": key[0],
                        "signal": key[1],
                        "noise": key[2],
                        "target_snr_db": key[3],
                        "method": method,
                        "value": value,
                    })
                value = _recovery_frequency(prefix_rows, method)
                metric_values[("truth_conditioned_component_recovery_frequency", "recovery_frequency")][method].append(value)
                rows_out.append({
                    "prefix_n_realizations": int(prefix),
                    "diagnostic": "truth_conditioned_component_recovery_frequency",
                    "submetric": "recovery_frequency",
                    "signal_regime": key[0],
                    "signal": key[1],
                    "noise": key[2],
                    "target_snr_db": key[3],
                    "method": method,
                    "value": value,
                })
        for (diagnostic, submetric), by_method in sorted(metric_values.items()):
            method_means = {
                method: _summary(vals)["mean"]
                for method, vals in by_method.items()
            }
            for ranked in _rank_methods(method_means, higher=True):
                ordering_rows.append({
                    "prefix_n_realizations": int(prefix),
                    "diagnostic": diagnostic,
                    "submetric": submetric,
                    **ranked,
                })
    drift_rows = []
    by_key = defaultdict(dict)
    for row in rows_out:
        by_key[(
            row["diagnostic"], row["submetric"], row["signal_regime"],
            row["signal"], row["noise"], row["target_snr_db"], row["method"],
        )][row["prefix_n_realizations"]] = row["value"]
    for key, vals in by_key.items():
        v20 = vals.get(20)
        for prefix in (5, 10, 15):
            vp = vals.get(prefix)
            drift = None if vp is None or v20 is None else float(vp - v20)
            drift_rows.append({
                "diagnostic": key[0],
                "submetric": key[1],
                "signal_regime": key[2],
                "signal": key[3],
                "noise": key[4],
                "target_snr_db": key[5],
                "method": key[6],
                "prefix_n_realizations": int(prefix),
                "prefix_minus_20_value": drift,
                "abs_prefix_minus_20_value": abs(drift) if drift is not None else None,
            })
    drift_summary = []
    drift_grouped = defaultdict(list)
    for row in drift_rows:
        drift_grouped[(row["diagnostic"], row["submetric"], row["method"], row["prefix_n_realizations"])].append(
            row["abs_prefix_minus_20_value"]
        )
    for key, vals in sorted(drift_grouped.items()):
        item = _summary(vals)
        drift_summary.append({
            "diagnostic": key[0],
            "submetric": key[1],
            "method": key[2],
            "prefix_n_realizations": int(key[3]),
            "n_cells": item["n"],
            "median_abs_drift_vs_20": item["median"],
            "q75_abs_drift_vs_20": item["q75"],
            "max_abs_drift_vs_20": item["max"],
        })
    return rows_out, ordering_rows, drift_rows, drift_summary


def _stratified_summaries(v554_root):
    count_rows = _read_rows(v554_root / "v554_component_count_stability_across_noise_realizations.csv")
    recovery_rows = _read_rows(v554_root / "v554_truth_conditioned_component_recovery_frequency.csv")
    out = []
    specs = [
        (count_rows, "component_count_stability_across_noise_realizations", "modal_count_fraction"),
        (recovery_rows, "truth_conditioned_component_recovery_frequency", "recovery_frequency"),
    ]
    for rows, diagnostic, value_field in specs:
        for stratifier in ("target_snr_db", "signal", "noise"):
            grouped = defaultdict(list)
            for row in rows:
                key = (row.get("method"), row.get(stratifier))
                grouped[key].append(_finite(row.get(value_field)))
            for (method, level), vals in sorted(grouped.items(), key=lambda item: str(item[0])):
                stats = _summary(vals)
                out.append({
                    "diagnostic": diagnostic,
                    "value_field": value_field,
                    "stratifier": stratifier,
                    "level": level,
                    "method": method,
                    "n_cells": stats["n"],
                    "mean": stats["mean"],
                    "median": stats["median"],
                    "q25": stats["q25"],
                    "q75": stats["q75"],
                    "min": stats["min"],
                    "max": stats["max"],
                })
    return out


def _seed_precision_table():
    out = []
    for r in (20, 50, 100):
        max_se = (0.25 / float(r)) ** 0.5
        out.append({
            "n_realizations": int(r),
            "recovery_frequency_step": float(1.0 / r),
            "max_binomial_se_at_p_0_5": float(max_se),
            "approx_95pct_margin_at_p_0_5": float(1.96 * max_se),
            "qualification_interpretation": (
                "available_now" if r == 20 else
                "requires additional seed-convergence execution"
            ),
        })
    return out


def run_v555_noise_realization_stability_qualification(algorithm_root):
    algorithm_root = ensure_dir(algorithm_root)
    output_root = ensure_dir(algorithm_root / "16x_v555_noise_realization_stability_qualification")
    source_path = algorithm_root / SOURCE_REL
    v554_root = algorithm_root / V554_REL
    if not source_path.exists():
        raise FileNotFoundError(f"Missing benchmark source rows: {source_path}")
    if not v554_root.exists():
        raise FileNotFoundError(f"Missing V5.54 outputs: {v554_root}")

    source_rows = _read_rows(source_path)
    groups = _build_groups(source_rows)
    metric_validity = _metric_validity_audit(v554_root)
    aggregation_validity = _aggregation_validity_audit(source_rows, groups)
    prefix_rows, ordering_rows, drift_rows, drift_summary = _prefix_stability(source_rows, groups)
    stratified = _stratified_summaries(v554_root)
    seed_precision = _seed_precision_table()

    passed = (
        all(bool(row.get("passed")) for row in metric_validity)
        and all(bool(row.get("passed")) for row in aggregation_validity)
    )
    dashboard = {
        "schema_version": V555_NOISE_REALIZATION_STABILITY_QUALIFICATION_VERSION,
        "created_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "qualification_status": "passed" if passed else "requires_attention",
        "v554_source": str(v554_root),
        "benchmark_source_rows": str(source_path),
        "fixed_input_repeatability_added": False,
        "primary_endpoint_or_ranking_impact": False,
        "audits": {
            "metric_validity": "v555_metric_validity_audit.csv",
            "aggregation_validity": "v555_aggregation_validity_audit.csv",
            "seed_prefix_stability": "v555_seed_prefix_stability_summary.csv",
            "stratified_summaries": "v555_stratified_stability_summaries.csv",
            "claim_boundary": "v555_claim_boundary.csv",
        },
        "claim_boundary": (
            "V5.54 is qualified as supplementary evidence for component-count "
            "stability and truth-conditioned component recovery frequency across "
            "the existing 20 Monte Carlo noise realizations. It does not support "
            "fixed-input stochastic repeatability claims. Cross-realization "
            "matched-component waveform similarity remains pending because compact "
            "rows do not store full component waveforms."
        ),
        "seed_count_interpretation": (
            "The current 20-seed data support supplementary stability summaries "
            "with 5 percentage-point recovery-frequency resolution. If recovery "
            "frequency becomes a major quantitative claim, run a 20->50->100 "
            "seed-convergence audit rather than changing the already frozen "
            "primary benchmark post hoc."
        ),
    }
    claim_boundary = [{
        "statement": "fixed_input_stochastic_repeatability",
        "authorized": False,
        "reason": "IRMF and standard EMD are deterministic; EEMD/CEEMDAN-only testing would be asymmetric.",
    }, {
        "statement": "component_count_stability_across_noise_realizations",
        "authorized": passed,
        "reason": "Computable from compact rows as modal count fraction across 20 Monte Carlo seeds.",
    }, {
        "statement": "truth_conditioned_component_recovery_frequency",
        "authorized": passed,
        "reason": "Computable from true and missing/matched component counts across 20 Monte Carlo seeds.",
    }, {
        "statement": "cross_realization_matched_component_similarity",
        "authorized": False,
        "reason": "Requires full component waveforms; compact benchmark rows are insufficient.",
    }]

    write_json(dashboard, output_root / "v555_noise_realization_stability_qualification_dashboard.json")
    write_csv(metric_validity, output_root / "v555_metric_validity_audit.csv")
    write_csv(aggregation_validity, output_root / "v555_aggregation_validity_audit.csv")
    write_csv(prefix_rows, output_root / "v555_seed_prefix_stability_rows.csv")
    write_csv(ordering_rows, output_root / "v555_seed_prefix_method_ordering.csv")
    write_csv(drift_rows, output_root / "v555_seed_prefix_drift_vs_20.csv")
    write_csv(drift_summary, output_root / "v555_seed_prefix_stability_summary.csv")
    write_csv(stratified, output_root / "v555_stratified_stability_summaries.csv")
    write_csv(seed_precision, output_root / "v555_seed_precision_table.csv")
    write_csv(claim_boundary, output_root / "v555_claim_boundary.csv")
    return dashboard
