#!/usr/bin/python
# coding: UTF-8

"""V5.61 Section 6.5 matching-rule robustness curve.

This stage upgrades the V5.60 correspondence audit from a small sanity check to
a prespecified Section 6.5 evaluation-protocol robustness analysis.  It keeps
the primary component matching definition frozen and evaluates whether
component-level conclusions are materially preserved under thresholded
Hungarian and mutual-nearest-neighbor alternatives.
"""

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from diagnostics.shared_physical_diagnostics import SYNTHETIC_RECONSTRUCTION_PROTOCOL_ID
from experiments.experiment_utils import run_fixed_method_family_case
from experiments.experiment_v560_matching_sensitivity_protocol import (
    V560_MATCHING_SENSITIVITY_PROTOCOL_VERSION,
)
from experiments.experiment_v560_matching_sensitivity_subset import (
    LOWER_IS_BETTER,
    METRICS,
    _finite,
    _load_locked_irmf_params,
    _rank_order,
    _rule_rows_for_result,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json
from project_config import (
    DEFAULT_FS,
    DEFAULT_METHOD_TIMEOUT_SECONDS,
    DEFAULT_N,
    EVALUATION_METHODS,
    GLOBAL_CEEMDAN_PARAMS,
    GLOBAL_EEMD_PARAMS,
    GLOBAL_EMD_PARAMS,
)


V561_SECTION6_5_VERSION = "V5.61_section6_5_matching_rule_robustness_curve"

SIGNALS = (
    "chirp",
    "close_frequencies",
    "impulsive_transient",
    "crossing_chirps",
    "time_varying_close_frequencies",
    "buried_weak_component",
)
SIGNAL_CLASS = {
    "chirp": "canonical",
    "close_frequencies": "canonical",
    "impulsive_transient": "canonical",
    "crossing_chirps": "challenging",
    "time_varying_close_frequencies": "challenging",
    "buried_weak_component": "challenging",
}
NOISES = ("gaussian", "impulsive", "huber_contamination")
TARGET_SNR_DB = (5.0, 15.0, 25.0)
SEEDS = (0, 1, 2)
TAU_GRID = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
ALGORITHM_SEED = 20260715


def _is_same_tau(a, b):
    av = _finite(a)
    bv = _finite(b)
    if av is None or bv is None:
        return False
    return abs(av - bv) <= 1e-12


def _rows_for_rules(result, true_components):
    rows = []
    for row in _rule_rows_for_result(result, true_components, tau_grid=TAU_GRID[1:]):
        if row["matching_rule"] == "primary_unthresholded_hungarian":
            row = dict(row)
            row["tau"] = 0.0
        rows.append(row)
    for row in _rule_rows_for_result(result, true_components, tau_grid=(0.0,)):
        if row["matching_rule"] == "mutual_nearest_neighbor":
            rows.append(row)
    return rows


def _summarize_by_rule(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["matching_rule"], float(row["tau"]), row["method"])].append(row)

    out = []
    for (rule, tau, method), vals in sorted(grouped.items()):
        item = {
            "matching_rule": rule,
            "tau": float(tau),
            "method": method,
            "n_rows": int(len(vals)),
        }
        for metric in METRICS:
            arr = np.asarray([
                _finite(v.get(metric))
                for v in vals
                if _finite(v.get(metric)) is not None
            ], dtype=float)
            item[f"{metric}_median"] = float(np.median(arr)) if arr.size else np.nan
            item[f"{metric}_mean"] = float(np.mean(arr)) if arr.size else np.nan
            item[f"{metric}_computable_rate"] = float(arr.size / max(len(vals), 1))

        coverage_true = np.asarray([
            _finite(v.get("valid_match_fraction_true"))
            for v in vals
            if _finite(v.get("valid_match_fraction_true")) is not None
        ], dtype=float)
        coverage_est = np.asarray([
            _finite(v.get("valid_match_fraction_estimated"))
            for v in vals
            if _finite(v.get("valid_match_fraction_estimated")) is not None
        ], dtype=float)
        corr = _finite(item.get("matched_corr_median"))
        coverage = float(np.median(coverage_true)) if coverage_true.size else np.nan
        item["valid_match_coverage_true_median"] = coverage
        item["valid_match_coverage_estimated_median"] = (
            float(np.median(coverage_est)) if coverage_est.size else np.nan
        )
        item["no_valid_match_rate"] = float(np.mean([
            int((_finite(v.get("n_matched_pairs")) or 0.0) <= 0.0) for v in vals
        ])) if vals else np.nan
        item["quality_coverage_product_median_corr_x_true_coverage"] = (
            float(corr * coverage)
            if corr is not None and np.isfinite(coverage)
            else np.nan
        )
        out.append(item)
    return out


def _summarize_by_signal_class(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(
            row["signal_class"],
            row["matching_rule"],
            float(row["tau"]),
            row["method"],
        )].append(row)
    out = []
    for (signal_class, rule, tau, method), vals in sorted(grouped.items()):
        item = {
            "signal_class": signal_class,
            "matching_rule": rule,
            "tau": float(tau),
            "method": method,
            "n_rows": int(len(vals)),
        }
        for metric in ("matched_corr", "matched_nrmse", "valid_match_fraction_true"):
            arr = np.asarray([
                _finite(v.get(metric))
                for v in vals
                if _finite(v.get(metric)) is not None
            ], dtype=float)
            item[f"{metric}_median"] = float(np.median(arr)) if arr.size else np.nan
        out.append(item)
    return out


def _rank_order_v561(summary_rows, rule, tau, metric):
    vals = []
    for row in summary_rows:
        if row["matching_rule"] != rule or not _is_same_tau(row["tau"], tau):
            continue
        val = _finite(row.get(f"{metric}_median"))
        if val is not None:
            vals.append((row["method"], val))
    reverse = metric not in LOWER_IS_BETTER
    return [m for m, _ in sorted(vals, key=lambda x: x[1], reverse=reverse)]


def _sign(value):
    value = _finite(value)
    if value is None:
        return "NA"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def _preservation_rows(summary_rows):
    out = []
    alternatives = sorted({
        (row["matching_rule"], float(row["tau"]))
        for row in summary_rows
        if row["matching_rule"] != "primary_unthresholded_hungarian"
    })
    for metric in METRICS:
        primary_order = _rank_order_v561(
            summary_rows,
            "primary_unthresholded_hungarian",
            0.0,
            metric,
        )
        primary_lookup = {
            row["method"]: _finite(row.get(f"{metric}_median"))
            for row in summary_rows
            if row["matching_rule"] == "primary_unthresholded_hungarian"
            and _is_same_tau(row["tau"], 0.0)
        }
        for rule, tau in alternatives:
            alt_order = _rank_order_v561(summary_rows, rule, tau, metric)
            alt_lookup = {
                row["method"]: _finite(row.get(f"{metric}_median"))
                for row in summary_rows
                if row["matching_rule"] == rule and _is_same_tau(row["tau"], tau)
            }
            direction_checks = []
            for baseline in ("EMD", "EEMD", "CEEMDAN"):
                p_irmf = primary_lookup.get("IRMF")
                p_base = primary_lookup.get(baseline)
                a_irmf = alt_lookup.get("IRMF")
                a_base = alt_lookup.get(baseline)
                if metric in LOWER_IS_BETTER:
                    p_delta = p_base - p_irmf if p_irmf is not None and p_base is not None else None
                    a_delta = a_base - a_irmf if a_irmf is not None and a_base is not None else None
                else:
                    p_delta = p_irmf - p_base if p_irmf is not None and p_base is not None else None
                    a_delta = a_irmf - a_base if a_irmf is not None and a_base is not None else None
                direction_checks.append({
                    "baseline": baseline,
                    "primary_benefit_sign": _sign(p_delta),
                    "alternative_benefit_sign": _sign(a_delta),
                    "preserved": bool(_sign(p_delta) == _sign(a_delta)),
                })
            out.append({
                "metric": metric,
                "alternative_rule": rule,
                "tau": float(tau),
                "primary_method_order": ">".join(primary_order),
                "alternative_method_order": ">".join(alt_order),
                "exact_order_preserved": bool(primary_order == alt_order),
                "effect_direction_preserved_all_irmf_vs_baselines": bool(
                    all(x["preserved"] for x in direction_checks)
                ),
                "effect_direction_details_json": str(direction_checks),
            })
    return out


def _preservation_summary(preservation):
    quality_rows = [
        r for r in preservation
        if r["metric"] in ("matched_corr", "matched_nrmse")
    ]
    allocation_rows = [
        r for r in preservation
        if r["metric"] in (
            "matching_rule_unmatched_true_energy_ratio",
            "matching_rule_unmatched_estimated_energy_ratio",
        )
    ]
    moderate_rows = [
        r for r in preservation
        if float(r["tau"]) in (0.3, 0.4, 0.5, 0.6)
    ]

    def rate(rows, key):
        return float(np.mean([bool(r[key]) for r in rows])) if rows else np.nan

    return {
        "n_preservation_checks": int(len(preservation)),
        "quality_effect_direction_preserved_rate": rate(
            quality_rows,
            "effect_direction_preserved_all_irmf_vs_baselines",
        ),
        "quality_exact_order_preserved_rate": rate(
            quality_rows,
            "exact_order_preserved",
        ),
        "allocation_effect_direction_preserved_rate": rate(
            allocation_rows,
            "effect_direction_preserved_all_irmf_vs_baselines",
        ),
        "moderate_tau_effect_direction_preserved_rate": rate(
            moderate_rows,
            "effect_direction_preserved_all_irmf_vs_baselines",
        ),
        "moderate_tau_exact_order_preserved_rate": rate(
            moderate_rows,
            "exact_order_preserved",
        ),
    }


def run_v561_section6_5_matching_rule_robustness(output_root, algorithm_root=None):
    output_root = ensure_dir(output_root)
    algorithm_root = Path(algorithm_root) if algorithm_root is not None else output_root.parent
    irmf_params = _load_locked_irmf_params(algorithm_root)

    rows = []
    failures = []
    for signal in SIGNALS:
        for noise in NOISES:
            for snr in TARGET_SNR_DB:
                for seed in SEEDS:
                    case, results, _ = run_fixed_method_family_case(
                        signal_name=signal,
                        noise_name=noise,
                        sigma=float(10.0 ** (-float(snr) / 20.0)),
                        irmf_params=irmf_params,
                        emd_params=GLOBAL_EMD_PARAMS,
                        eemd_params=GLOBAL_EEMD_PARAMS,
                        ceemdan_params=GLOBAL_CEEMDAN_PARAMS,
                        n=DEFAULT_N,
                        fs=DEFAULT_FS,
                        seed=int(seed),
                        algorithm_seed=ALGORITHM_SEED,
                        target_snr_db=float(snr),
                        run_id_prefix=f"v561_s65_{signal}_{noise}_snr{snr:g}_seed{seed}",
                        timeout_seconds=DEFAULT_METHOD_TIMEOUT_SECONDS,
                        reconstruction_protocol_id=SYNTHETIC_RECONSTRUCTION_PROTOCOL_ID,
                    )
                    for method in EVALUATION_METHODS:
                        result = results.get(method, {})
                        if result.get("imfs") is None or case.get("true_components") is None:
                            failures.append({
                                "signal": signal,
                                "noise": noise,
                                "target_snr_db": snr,
                                "seed": seed,
                                "method": method,
                                "failure_level": "missing_components_for_matching_audit",
                                "error": result.get("error"),
                            })
                            continue
                        for rule_row in _rows_for_rules(result, case.get("true_components")):
                            rows.append({
                                "signal": signal,
                                "signal_class": SIGNAL_CLASS.get(signal, "unknown"),
                                "noise": noise,
                                "target_snr_db": float(snr),
                                "seed": int(seed),
                                "method": method,
                                "realized_input_snr_db": case.get("realized_input_snr_db"),
                                "method_failure_level_flag": result.get("failure_level", "none"),
                                "method_any_failure_flag": bool(result.get("any_failure", False)),
                                **rule_row,
                                "primary_schema_changed": False,
                                "primary_matching_changed": False,
                            })

    method_summary = _summarize_by_rule(rows)
    signal_class_summary = _summarize_by_signal_class(rows)
    preservation = _preservation_rows(method_summary)
    preservation_summary = _preservation_summary(preservation)

    dashboard = {
        "schema_version": V561_SECTION6_5_VERSION,
        "created_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "parent_protocol_version": V560_MATCHING_SENSITIVITY_PROTOCOL_VERSION,
        "module_status": "section_6_5_matching_rule_robustness_execution_complete",
        "paper_section": "6.5 Matching-Rule Robustness",
        "primary_schema_changed": False,
        "primary_matching_changed": False,
        "not_retuning": True,
        "not_ranking_bearing": True,
        "subset_design": {
            "signals": list(SIGNALS),
            "signal_classes": SIGNAL_CLASS,
            "noises": list(NOISES),
            "target_snr_db": list(TARGET_SNR_DB),
            "seeds": list(SEEDS),
            "methods": list(EVALUATION_METHODS),
            "expected_method_evaluations": int(
                len(SIGNALS)
                * len(NOISES)
                * len(TARGET_SNR_DB)
                * len(SEEDS)
                * len(EVALUATION_METHODS)
            ),
            "expected_rule_rows": int(
                len(SIGNALS)
                * len(NOISES)
                * len(TARGET_SNR_DB)
                * len(SEEDS)
                * len(EVALUATION_METHODS)
                * (1 + (len(TAU_GRID) - 1) + len(TAU_GRID))
            ),
        },
        "threshold_grid": list(TAU_GRID),
        "rules": [
            "primary_unthresholded_hungarian_tau0_anchor",
            "thresholded_hungarian_tau_0p3_to_0p9",
            "mutual_nearest_neighbor_tau_0_to_0p9",
        ],
        "n_metric_rows": int(len(rows)),
        "n_failures": int(len(failures)),
        "locked_irmf_params_used": irmf_params,
        "preservation_summary": preservation_summary,
        "interpretation_boundary": (
            "Quality curves must be interpreted jointly with coverage because "
            "higher thresholds can mechanically increase matched-pair quality "
            "by discarding weak matches. Section 6.5 checks conclusion "
            "preservation for matched-component correspondence. It does not "
            "generate a new overall method ranking, does not recompute RCCE, "
            "and does not redefine the primary association-based missing or "
            "spurious energy metrics."
        ),
        "component_set_fidelity_boundary": (
            "RCCE is correspondence-independent and is not recomputed in this "
            "matching-rule audit. Primary missing_true_component_energy_ratio "
            "and spurious_mode_energy_ratio are association-definition metrics "
            "based on the full |corr|^2 association matrix plus frozen "
            "association/active-energy thresholds. The matching_rule_unmatched_* "
            "fields are supporting correspondence diagnostics only."
        ),
        "output_files": {
            "dashboard": "v561_section6_5_matching_rule_robustness_dashboard.json",
            "rows": "v561_section6_5_matching_rule_robustness_rows.csv",
            "method_summary": "v561_section6_5_matching_rule_robustness_method_summary.csv",
            "signal_class_summary": "v561_section6_5_matching_rule_robustness_signal_class_summary.csv",
            "preservation": "v561_section6_5_matching_rule_robustness_preservation.csv",
            "failures": "v561_section6_5_matching_rule_robustness_failures.csv",
        },
    }

    write_csv(rows, output_root / "v561_section6_5_matching_rule_robustness_rows.csv")
    write_csv(method_summary, output_root / "v561_section6_5_matching_rule_robustness_method_summary.csv")
    write_csv(signal_class_summary, output_root / "v561_section6_5_matching_rule_robustness_signal_class_summary.csv")
    write_csv(preservation, output_root / "v561_section6_5_matching_rule_robustness_preservation.csv")
    write_csv(failures, output_root / "v561_section6_5_matching_rule_robustness_failures.csv")
    write_json(dashboard, output_root / "v561_section6_5_matching_rule_robustness_dashboard.json")
    return {
        "dashboard": dashboard,
        "rows": rows,
        "method_summary": method_summary,
        "signal_class_summary": signal_class_summary,
        "preservation": preservation,
        "failures": failures,
    }
