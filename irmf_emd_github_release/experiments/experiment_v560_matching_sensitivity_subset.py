#!/usr/bin/python
# coding: UTF-8

"""V5.60 small executable subset for component-correspondence sensitivity.

This is a lightweight empirical check of the V5.60 Appendix protocol.  It
reruns a prespecified small synthetic subset so the full estimated-by-true
correlation matrix is available in memory.  It does not change primary metrics.
"""

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from diagnostics.shared_physical_diagnostics import SYNTHETIC_RECONSTRUCTION_PROTOCOL_ID
from experiments.experiment_utils import run_fixed_method_family_case
from experiments.experiment_v560_matching_sensitivity_protocol import (
    THRESHOLDED_HUNGARIAN_TAU_GRID,
    V560_MATCHING_SENSITIVITY_PROTOCOL_VERSION,
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


V560_SUBSET_EXECUTION_VERSION = (
    "V5.60_small_subset_matching_rule_robustness_execution"
)

SUBSET_SIGNALS = ("chirp", "crossing_chirps")
SUBSET_NOISES = ("gaussian", "huber_contamination")
SUBSET_TARGET_SNR_DB = (5.0, 15.0)
SUBSET_SEEDS = (0,)
ALGORITHM_SEED = 20260715
EPS = 1e-12

METRICS = (
    "matched_corr",
    "matched_nrmse",
    "matching_rule_unmatched_true_energy_ratio",
    "matching_rule_unmatched_estimated_energy_ratio",
)
LOWER_IS_BETTER = {
    "matched_nrmse",
    "matching_rule_unmatched_true_energy_ratio",
    "matching_rule_unmatched_estimated_energy_ratio",
}


def _finite(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return None


def _safe_corr(a, b):
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size != b.size or a.size < 3:
        return np.nan
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return np.nan
    if float(np.std(a)) <= EPS or float(np.std(b)) <= EPS:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _as_2d(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim != 2:
        return np.zeros((0, 0), dtype=float)
    return arr


def _corr_matrices(imfs, true_components):
    imfs = _as_2d(imfs)
    truth = _as_2d(true_components)
    if imfs.size == 0 or truth.size == 0 or imfs.shape[1] != truth.shape[1]:
        return (
            np.zeros((0, 0), dtype=float),
            np.zeros((0, 0), dtype=float),
            imfs,
            truth,
        )
    signed = np.full((imfs.shape[0], truth.shape[0]), np.nan, dtype=float)
    for i in range(imfs.shape[0]):
        for j in range(truth.shape[0]):
            signed[i, j] = _safe_corr(imfs[i], truth[j])
    return signed, np.abs(signed), imfs, truth


def _linear_sum_assignment(cost):
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(cost)
    return list(zip(rows, cols))


def _unthresholded_hungarian(abs_corr):
    if abs_corr.size == 0:
        return []
    score = np.nan_to_num(abs_corr, nan=0.0, posinf=0.0, neginf=0.0)
    return _linear_sum_assignment(1.0 - score)


def _thresholded_hungarian(abs_corr, tau):
    """Maximum-benefit one-to-one matching with optional unmatched components."""
    if abs_corr.size == 0:
        return []
    score = np.nan_to_num(abs_corr, nan=0.0, posinf=0.0, neginf=0.0)
    n_est, n_true = score.shape
    size = n_est + n_true
    cost = np.zeros((size, size), dtype=float)
    benefit = np.maximum(score - float(tau), 0.0)
    cost[:n_est, :n_true] = -benefit
    pairs = []
    for row, col in _linear_sum_assignment(cost):
        if row < n_est and col < n_true and score[row, col] >= float(tau):
            pairs.append((int(row), int(col)))
    return pairs


def _mutual_nearest_neighbor(abs_corr, tau=None):
    if abs_corr.size == 0:
        return []
    score = np.nan_to_num(abs_corr, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    if score.shape[0] == 0 or score.shape[1] == 0:
        return []
    row_best = np.argmax(score, axis=1)
    col_best = np.argmax(score, axis=0)
    pairs = []
    for i, j in enumerate(row_best):
        val = score[i, j]
        if not np.isfinite(val):
            continue
        if tau is not None and val < float(tau):
            continue
        if int(col_best[j]) == int(i):
            pairs.append((int(i), int(j)))
    return pairs


def _energy_ratio(arr):
    arr = _as_2d(arr)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    e = np.asarray([float(np.sum(x ** 2)) for x in arr], dtype=float)
    return e / max(float(np.sum(e)), EPS)


def _metrics_for_pairs(imfs, truth, signed_corr, abs_corr, pairs, rule, tau):
    n_est, n_true = int(imfs.shape[0]), int(truth.shape[0])
    matched_est = {int(i) for i, _ in pairs}
    matched_true = {int(j) for _, j in pairs}
    corrs = []
    nrmse = []
    signed_corrs = []
    for i, j in pairs:
        c = abs_corr[i, j]
        if np.isfinite(c):
            corrs.append(float(c))
        sc = signed_corr[i, j]
        if np.isfinite(sc):
            signed_corrs.append(float(sc))
        rmse = float(np.sqrt(np.mean((imfs[i] - truth[j]) ** 2)))
        denom = float(np.sqrt(np.mean(truth[j] ** 2)) + EPS)
        nrmse.append(float(rmse / denom))

    true_energy = _energy_ratio(truth)
    est_energy = _energy_ratio(imfs)
    unmatched_true_energy = float(
        np.sum([true_energy[j] for j in range(n_true) if j not in matched_true])
    ) if true_energy.size else np.nan
    unmatched_est_energy = float(
        np.sum([est_energy[i] for i in range(n_est) if i not in matched_est])
    ) if est_energy.size else np.nan

    threshold = float(tau) if tau is not None else ""
    weak_forced = np.nan
    if tau is not None and rule == "primary_unthresholded_hungarian":
        weak_forced = float(np.mean([abs_corr[i, j] < threshold for i, j in pairs])) if pairs else np.nan

    return {
        "matching_rule": rule,
        "tau": threshold,
        "n_estimated_components": n_est,
        "n_true_components": n_true,
        "n_matched_pairs": int(len(pairs)),
        "valid_match_fraction_true": float(len(matched_true) / max(n_true, 1)),
        "valid_match_fraction_estimated": float(len(matched_est) / max(n_est, 1)),
        "matched_corr": float(np.mean(corrs)) if corrs else np.nan,
        "matched_signed_corr_mean": float(np.mean(signed_corrs)) if signed_corrs else np.nan,
        "matched_nrmse": float(np.mean(nrmse)) if nrmse else np.nan,
        "matching_rule_unmatched_true_energy_ratio": unmatched_true_energy,
        "matching_rule_unmatched_estimated_energy_ratio": unmatched_est_energy,
        "weak_forced_match_fraction": weak_forced,
        "assignment_pairs": ";".join(f"est{i}->true{j}" for i, j in pairs),
    }


def _rule_rows_for_result(result, true_components, tau_grid=THRESHOLDED_HUNGARIAN_TAU_GRID):
    signed_corr, abs_corr, imfs, truth = _corr_matrices(
        result.get("imfs"),
        true_components,
    )
    rows = []
    primary_pairs = _unthresholded_hungarian(abs_corr)
    rows.append(_metrics_for_pairs(
        imfs, truth, signed_corr, abs_corr, primary_pairs,
        "primary_unthresholded_hungarian", None,
    ))
    for tau in tau_grid:
        rows.append(_metrics_for_pairs(
            imfs, truth, signed_corr, abs_corr,
            _thresholded_hungarian(abs_corr, tau),
            "thresholded_hungarian", tau,
        ))
        rows.append(_metrics_for_pairs(
            imfs, truth, signed_corr, abs_corr,
            _mutual_nearest_neighbor(abs_corr, tau=tau),
            "mutual_nearest_neighbor", tau,
        ))
    return rows


def _load_locked_irmf_params(algorithm_root):
    path = (
        Path(algorithm_root)
        / "01_global_parameter_selection"
        / "section_4_parameter_selection_protocol.json"
    )
    if path.exists():
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        selected = data.get("selected_global_irmf_params") or {}
        if selected.get("H_parameterization") == "relative_noise_scale":
            return {
                "h1": float(selected["h1"]),
                "a": float(selected["a"]),
                "h_min": float(selected["h_min"]),
                "H_parameterization": "relative_noise_scale",
                "c_H": float(selected["c_H"]),
                "boundary_mode": selected.get("boundary_mode", "periodic"),
                "min_support_points": int(selected.get("min_support_points", 3)),
            }
    from project_config import GLOBAL_IRMF_PARAMS

    return dict(GLOBAL_IRMF_PARAMS)


def _summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["matching_rule"], row["tau"], row["method"])].append(row)
    out = []
    for (rule, tau, method), vals in sorted(grouped.items(), key=lambda x: (x[0][0], str(x[0][1]), x[0][2])):
        item = {
            "matching_rule": rule,
            "tau": tau,
            "method": method,
            "n_rows": int(len(vals)),
        }
        for metric in METRICS:
            arr = np.asarray([_finite(v.get(metric)) for v in vals if _finite(v.get(metric)) is not None], dtype=float)
            item[f"{metric}_median"] = float(np.median(arr)) if arr.size else np.nan
            item[f"{metric}_mean"] = float(np.mean(arr)) if arr.size else np.nan
        out.append(item)
    return out


def _rank_order(summary_rows, rule, tau, metric):
    vals = []
    for row in summary_rows:
        if row["matching_rule"] != rule:
            continue
        if str(row["tau"]) != str(tau):
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
    primary_tau = ""
    alternatives = sorted({
        (row["matching_rule"], row["tau"])
        for row in summary_rows
        if row["matching_rule"] != "primary_unthresholded_hungarian"
    }, key=lambda x: (x[0], str(x[1])))
    for metric in METRICS:
        primary_order = _rank_order(
            summary_rows,
            "primary_unthresholded_hungarian",
            primary_tau,
            metric,
        )
        primary_lookup = {
            row["method"]: _finite(row.get(f"{metric}_median"))
            for row in summary_rows
            if row["matching_rule"] == "primary_unthresholded_hungarian"
        }
        for rule, tau in alternatives:
            alt_order = _rank_order(summary_rows, rule, tau, metric)
            alt_lookup = {
                row["method"]: _finite(row.get(f"{metric}_median"))
                for row in summary_rows
                if row["matching_rule"] == rule and str(row["tau"]) == str(tau)
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
                "tau": tau,
                "primary_method_order": ">".join(primary_order),
                "alternative_method_order": ">".join(alt_order),
                "exact_order_preserved": bool(primary_order == alt_order),
                "effect_direction_preserved_all_irmf_vs_baselines": bool(
                    all(x["preserved"] for x in direction_checks)
                ),
                "effect_direction_details_json": str(direction_checks),
            })
    return out


def run_v560_matching_sensitivity_subset(output_root, algorithm_root=None):
    output_root = ensure_dir(output_root)
    algorithm_root = Path(algorithm_root) if algorithm_root is not None else output_root.parent
    irmf_params = _load_locked_irmf_params(algorithm_root)

    rows = []
    failures = []
    for signal in SUBSET_SIGNALS:
        for noise in SUBSET_NOISES:
            for snr in SUBSET_TARGET_SNR_DB:
                for seed in SUBSET_SEEDS:
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
                        run_id_prefix=f"v560_subset_{signal}_{noise}_snr{snr:g}_seed{seed}",
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
                        for rule_row in _rule_rows_for_result(result, case.get("true_components")):
                            rows.append({
                                "signal": signal,
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

    summaries = _summary(rows)
    preservation = _preservation_rows(summaries)
    preserved_rates = {
        "n_preservation_checks": int(len(preservation)),
        "exact_order_preserved_rate": float(np.mean([r["exact_order_preserved"] for r in preservation])) if preservation else np.nan,
        "effect_direction_preserved_all_rate": float(np.mean([r["effect_direction_preserved_all_irmf_vs_baselines"] for r in preservation])) if preservation else np.nan,
    }
    dashboard = {
        "schema_version": V560_SUBSET_EXECUTION_VERSION,
        "created_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "parent_protocol_version": V560_MATCHING_SENSITIVITY_PROTOCOL_VERSION,
        "module_status": "small_subset_execution_complete",
        "paper_location": "Supplementary / Appendix protocol credibility audit",
        "primary_schema_changed": False,
        "primary_matching_changed": False,
        "not_ranking_bearing": True,
        "subset_design": {
            "signals": list(SUBSET_SIGNALS),
            "noises": list(SUBSET_NOISES),
            "target_snr_db": list(SUBSET_TARGET_SNR_DB),
            "seeds": list(SUBSET_SEEDS),
            "methods": list(EVALUATION_METHODS),
            "expected_method_evaluations": int(
                len(SUBSET_SIGNALS)
                * len(SUBSET_NOISES)
                * len(SUBSET_TARGET_SNR_DB)
                * len(SUBSET_SEEDS)
                * len(EVALUATION_METHODS)
            ),
        },
        "n_metric_rows": int(len(rows)),
        "n_failures": int(len(failures)),
        "threshold_grid": list(THRESHOLDED_HUNGARIAN_TAU_GRID),
        "locked_irmf_params_used": irmf_params,
        "preservation_summary": preserved_rates,
        "claim_boundary": (
            "This small subset tests execution feasibility and checks whether "
            "method ordering and IRMF-vs-baseline effect directions are "
            "materially preserved under alternative correspondence rules. It "
            "does not replace the frozen primary Hungarian matching definition. "
            "The matching_rule_unmatched_* fields are correspondence-rule "
            "supporting diagnostics and are not the primary "
            "missing_true_component_energy_ratio or spurious_mode_energy_ratio."
        ),
        "output_files": {
            "dashboard": "v560_matching_sensitivity_subset_dashboard.json",
            "rows": "v560_matching_sensitivity_subset_rows.csv",
            "method_summary": "v560_matching_sensitivity_subset_method_summary.csv",
            "preservation": "v560_matching_sensitivity_subset_preservation.csv",
            "failures": "v560_matching_sensitivity_subset_failures.csv",
        },
    }

    write_csv(rows, output_root / "v560_matching_sensitivity_subset_rows.csv")
    write_csv(summaries, output_root / "v560_matching_sensitivity_subset_method_summary.csv")
    write_csv(preservation, output_root / "v560_matching_sensitivity_subset_preservation.csv")
    write_csv(failures, output_root / "v560_matching_sensitivity_subset_failures.csv")
    write_json(dashboard, output_root / "v560_matching_sensitivity_subset_dashboard.json")
    return {
        "dashboard": dashboard,
        "rows": rows,
        "method_summary": summaries,
        "preservation": preservation,
        "failures": failures,
    }
