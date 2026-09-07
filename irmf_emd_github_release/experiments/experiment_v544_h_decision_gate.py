#!/usr/bin/python
# coding: UTF-8

"""V5.44 absolute-H versus relative-H decision gate.

This is a small protocol/data-generation audit.  It does not run IRMF/EMD
decompositions and does not authorize final benchmark claims.  Its purpose is
to decide whether the relative-H path is mature enough to supersede the V5.39
absolute-H development selection, or whether the project should proceed with
the already-amended absolute-H four-parameter selection.
"""

from collections import Counter
from datetime import datetime, timezone

import numpy as np

from experiments.experiment_utils import make_signal_noise_case
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json, write_manifest
from project_config import (
    DEFAULT_FS,
    DEFAULT_N,
    IRMF_RELATIVE_H_C_OPTIONS,
    IRMF_RELATIVE_H_PARAMETERIZATION_VERSION,
    IRMF_RELATIVE_H_SCALE_ESTIMATOR,
    PARAMETER_SELECTION_NOISES,
    PARAMETER_SELECTION_SIGNALS,
)


V544_H_DECISION_GATE_VERSION = "V5.44_absolute_vs_relative_H_decision_gate"
SNR_LEVELS_FOR_GATE = (5.0, 15.0, 25.0)
SEEDS_FOR_GATE = (0, 1)
STRESS_NOISES_FOR_GATE = ("huber_contamination", "heteroskedastic")


def first_difference_mad_noise_scale(y, eps=1e-12):
    y = np.asarray(y, dtype=float)
    out = {
        "noise_scale_hat": np.nan,
        "fallback_used": False,
        "estimator_status": "not_computable",
    }
    if y.size < 3:
        return out
    dy = np.diff(y)
    mad = float(np.median(np.abs(dy - np.median(dy))))
    sigma_hat = mad / (0.67448975 * np.sqrt(2.0))
    if np.isfinite(sigma_hat) and sigma_hat > eps:
        out["noise_scale_hat"] = float(sigma_hat)
        out["estimator_status"] = "finite_positive_first_difference_mad"
        return out

    mad_y = float(np.median(np.abs(y - np.median(y))))
    fallback = mad_y / 0.67448975
    out["fallback_used"] = True
    if np.isfinite(fallback) and fallback > eps:
        out["noise_scale_hat"] = float(fallback)
        out["estimator_status"] = "finite_positive_level_mad_fallback"
    return out


def _summarize(values):
    arr = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "min": np.nan,
            "max": np.nan,
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


def _group_summary(rows, group_fields):
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, []).append(row)
    out = []
    for key, group in sorted(groups.items()):
        item = {field: value for field, value in zip(group_fields, key)}
        finite = [g for g in group if bool(g["scale_estimator_finite_positive"])]
        item["n_cases"] = int(len(group))
        item["finite_positive_rate"] = float(len(finite) / max(len(group), 1))
        item.update({
            f"sigma_hat_to_true_noise_rms_{k}": v
            for k, v in _summarize(g["sigma_hat_to_true_noise_rms"] for g in finite).items()
        })
        item.update({
            f"abs_log_sigma_hat_ratio_{k}": v
            for k, v in _summarize(g["abs_log_sigma_hat_ratio"] for g in finite).items()
        })
        out.append(item)
    return out


def _build_rows():
    rows = []
    noises = tuple(PARAMETER_SELECTION_NOISES) + tuple(
        n for n in STRESS_NOISES_FOR_GATE if n not in PARAMETER_SELECTION_NOISES
    )
    for signal_name in PARAMETER_SELECTION_SIGNALS:
        for noise_name in noises:
            for target_snr_db in SNR_LEVELS_FOR_GATE:
                for seed in SEEDS_FOR_GATE:
                    case = make_signal_noise_case(
                        signal_name=signal_name,
                        noise_name=noise_name,
                        sigma=1.0,
                        n=DEFAULT_N,
                        fs=DEFAULT_FS,
                        seed=seed,
                        target_snr_db=target_snr_db,
                    )
                    y = np.asarray(case["Y"], dtype=float)
                    noise = np.asarray(case["noise"], dtype=float)
                    x = np.asarray(case["X_clean"], dtype=float)
                    est = first_difference_mad_noise_scale(y)
                    sigma_hat = float(est["noise_scale_hat"])
                    true_noise_rms = float(np.sqrt(np.mean(noise ** 2)))
                    clean_rms = float(np.sqrt(np.mean(x ** 2)))
                    ratio = (
                        float(sigma_hat / true_noise_rms)
                        if np.isfinite(sigma_hat) and true_noise_rms > 0
                        else np.nan
                    )
                    abs_log_ratio = float(abs(np.log(ratio))) if np.isfinite(ratio) and ratio > 0 else np.nan
                    for c_h in IRMF_RELATIVE_H_C_OPTIONS:
                        h_realized = float(c_h * sigma_hat) if np.isfinite(sigma_hat) else np.nan
                        rows.append({
                            "audit_version": V544_H_DECISION_GATE_VERSION,
                            "parameterization_under_test": "relative_H",
                            "scale_estimator_id": IRMF_RELATIVE_H_SCALE_ESTIMATOR,
                            "signal": signal_name,
                            "noise": noise_name,
                            "target_snr_db": float(target_snr_db),
                            "seed": int(seed),
                            "clean_rms": clean_rms,
                            "true_noise_rms": true_noise_rms,
                            "realized_input_snr_db": case["realized_input_snr_db"],
                            "noise_scale_hat": sigma_hat,
                            "sigma_hat_to_true_noise_rms": ratio,
                            "abs_log_sigma_hat_ratio": abs_log_ratio,
                            "scale_estimator_finite_positive": bool(np.isfinite(sigma_hat) and sigma_hat > 0),
                            "scale_estimator_status": est["estimator_status"],
                            "scale_estimator_fallback_used": bool(est["fallback_used"]),
                            "c_H": float(c_h),
                            "H_realized": h_realized,
                            "H_realized_to_true_noise_rms": (
                                float(h_realized / true_noise_rms)
                                if np.isfinite(h_realized) and true_noise_rms > 0
                                else np.nan
                            ),
                            "uses_clean_truth_or_true_noise_for_estimator": False,
                            "true_noise_used_for_audit_only": True,
                        })
    return rows


def run_v544_h_decision_gate(output_root):
    output_root = ensure_dir(output_root)
    rows = _build_rows()
    unique_cases = {}
    for row in rows:
        unique_cases[(row["signal"], row["noise"], row["target_snr_db"], row["seed"])] = row
    case_rows = list(unique_cases.values())
    finite_rows = [r for r in case_rows if r["scale_estimator_finite_positive"]]
    finite_rate = float(len(finite_rows) / max(len(case_rows), 1))
    abs_log_summary = _summarize(r["abs_log_sigma_hat_ratio"] for r in finite_rows)
    ratio_summary = _summarize(r["sigma_hat_to_true_noise_rms"] for r in finite_rows)
    fallback_count = int(sum(bool(r["scale_estimator_fallback_used"]) for r in case_rows))
    noise_summary = _group_summary(case_rows, ["noise"])
    snr_summary = _group_summary(case_rows, ["target_snr_db"])
    status_counts = dict(Counter(r["scale_estimator_status"] for r in case_rows))

    median_abs_log = abs_log_summary["median"]
    p75_abs_log = abs_log_summary["q75"]
    relative_h_smoke_pass = bool(
        finite_rate >= 0.99
        and np.isfinite(median_abs_log)
        and median_abs_log <= np.log(2.0)
        and np.isfinite(p75_abs_log)
        and p75_abs_log <= np.log(4.0)
    )
    dashboard = {
        "audit_version": V544_H_DECISION_GATE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "absolute-H versus relative-H decision gate before development-set rerun",
        "relative_H_schema_version": IRMF_RELATIVE_H_PARAMETERIZATION_VERSION,
        "benchmark_results_recomputed": False,
        "algorithm_decompositions_run": False,
        "statistics_regenerated": False,
        "claim_authorized": False,
        "n_case_rows": int(len(case_rows)),
        "n_c_H_expanded_rows": int(len(rows)),
        "signals": list(PARAMETER_SELECTION_SIGNALS),
        "noises": sorted({r["noise"] for r in case_rows}),
        "target_snr_db_levels": list(SNR_LEVELS_FOR_GATE),
        "seeds": list(SEEDS_FOR_GATE),
        "scale_estimator_id": IRMF_RELATIVE_H_SCALE_ESTIMATOR,
        "scale_estimator_status_counts": status_counts,
        "finite_positive_rate": finite_rate,
        "fallback_count": fallback_count,
        "sigma_hat_to_true_noise_rms_summary": ratio_summary,
        "abs_log_sigma_hat_ratio_summary": abs_log_summary,
        "relative_H_smoke_pass": relative_h_smoke_pass,
        "decision": {
            "absolute_H_path": {
                "status": "available_now_via_V5.39_four_parameter_development_selection",
                "advantage": "No additional scale-estimator qualification required before development selection.",
                "main_risk": "A single absolute H has scale-dependent interpretation across target-SNR cases.",
            },
            "relative_H_path": {
                "status": (
                    "promising_but_requires_full_scale_estimator_qualification"
                    if relative_h_smoke_pass
                    else "not_ready_without_revising_or_qualifying_scale_estimator"
                ),
                "advantage": "c_H is dimensionless and H_realized adapts mechanically to observed noise scale.",
                "main_risk": "first-difference MAD can be biased by signal dynamics, colored noise, heavy tails, and contamination.",
            },
            "recommended_next_step": (
                "Run full V5.42 scale-estimator qualification before replacing V5.39, or proceed with V5.39 absolute-H selection if the paper timeline prioritizes immediate benchmark execution."
            ),
        },
        "activation_rule": (
            "This smoke does not activate relative-H. Activation still requires a frozen "
            "scale-estimator qualification audit and then development selection with "
            "theta=(h1,a,h_min,c_H)."
        ),
    }

    write_csv(rows, output_root / "v544_relative_H_scale_estimator_rows.csv")
    write_csv(noise_summary, output_root / "v544_relative_H_scale_estimator_by_noise.csv")
    write_csv(snr_summary, output_root / "v544_relative_H_scale_estimator_by_snr.csv")
    write_json(dashboard, output_root / "v544_h_decision_gate_dashboard.json")
    write_manifest(output_root, {
        "audit_version": V544_H_DECISION_GATE_VERSION,
        "artifact_role": "H_parameterization_decision_gate",
        "benchmark_results_recomputed": False,
        "algorithm_decompositions_run": False,
    })
    return {
        "output_dir": str(output_root),
        "dashboard": dashboard,
    }

