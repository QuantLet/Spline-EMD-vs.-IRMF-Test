#!/usr/bin/python
# coding: UTF-8

"""V5.45 qualification for the V5.42 relative-H scale estimator.

This audit qualifies the truth-free first-difference MAD scale proxy used by
the proposed relative-H parameterization:

    H_case = c_H * sigma_hat(Y_case).

It does not run decomposition algorithms and does not choose c_H.  If it
passes, the next authorized step is development-set parameter selection with
theta=(h1, a, h_min, c_H).
"""

from collections import Counter
from datetime import datetime, timezone

import numpy as np

from experiments.experiment_utils import make_signal_noise_case
from experiments.experiment_v544_h_decision_gate import first_difference_mad_noise_scale
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json, write_manifest
from project_config import (
    DEFAULT_FS,
    DEFAULT_N,
    IRMF_RELATIVE_H_C_OPTIONS,
    IRMF_RELATIVE_H_PARAMETERIZATION_VERSION,
    IRMF_RELATIVE_H_SCALE_ESTIMATOR,
    NOISE_FAMILY,
    PARAMETER_SELECTION_SIGNALS,
    TARGET_SNR_DB_LEVELS,
)


V545_QUALIFICATION_VERSION = "V5.45_relative_H_scale_estimator_qualification"
QUALIFICATION_SEEDS = tuple(range(5))
RATIO_MEDIAN_LOWER = 0.40
RATIO_MEDIAN_UPPER = 2.50
OVERALL_MEDIAN_ABS_LOG_MAX = float(np.log(2.0))
OVERALL_Q75_ABS_LOG_MAX = float(np.log(4.0))


def _summarize(values):
    arr = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "q10": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "q90": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "q90": float(np.quantile(arr, 0.90)),
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
        finite = [g for g in group if bool(g["scale_estimator_finite_positive"])]
        item = {field: value for field, value in zip(group_fields, key)}
        item["n_cases"] = int(len(group))
        item["finite_positive_rate"] = float(len(finite) / max(len(group), 1))
        item["fallback_rate"] = float(
            sum(bool(g["scale_estimator_fallback_used"]) for g in group) / max(len(group), 1)
        )
        item.update({
            f"sigma_hat_to_true_noise_rms_{k}": v
            for k, v in _summarize(g["sigma_hat_to_true_noise_rms"] for g in finite).items()
        })
        item.update({
            f"abs_log_sigma_hat_ratio_{k}": v
            for k, v in _summarize(g["abs_log_sigma_hat_ratio"] for g in finite).items()
        })
        median_ratio = item["sigma_hat_to_true_noise_rms_median"]
        item["median_ratio_within_prespecified_range"] = bool(
            np.isfinite(median_ratio)
            and RATIO_MEDIAN_LOWER <= float(median_ratio) <= RATIO_MEDIAN_UPPER
        )
        return_note = "passed_range"
        if not item["median_ratio_within_prespecified_range"]:
            return_note = "outside_prespecified_range"
        item["range_audit_status"] = return_note
        out.append(item)
    return out


def _estimate_case(signal_name, noise_name, target_snr_db, seed):
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
        if np.isfinite(sigma_hat) and true_noise_rms > 0.0
        else np.nan
    )
    abs_log_ratio = float(abs(np.log(ratio))) if np.isfinite(ratio) and ratio > 0.0 else np.nan
    return {
        "audit_version": V545_QUALIFICATION_VERSION,
        "case_type": "noisy_target_snr",
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
        "scale_estimator_finite_positive": bool(np.isfinite(sigma_hat) and sigma_hat > 0.0),
        "scale_estimator_status": est["estimator_status"],
        "scale_estimator_fallback_used": bool(est["fallback_used"]),
        "H_realized_min_c_H": (
            float(min(IRMF_RELATIVE_H_C_OPTIONS) * sigma_hat)
            if np.isfinite(sigma_hat) else np.nan
        ),
        "H_realized_max_c_H": (
            float(max(IRMF_RELATIVE_H_C_OPTIONS) * sigma_hat)
            if np.isfinite(sigma_hat) else np.nan
        ),
        "uses_clean_truth_or_true_noise_for_estimator": False,
        "true_noise_used_for_audit_only": True,
    }


def _estimate_clean_case(signal_name, seed):
    case = make_signal_noise_case(
        signal_name=signal_name,
        noise_name="gaussian",
        sigma=1.0,
        n=DEFAULT_N,
        fs=DEFAULT_FS,
        seed=seed,
        target_snr_db=30.0,
    )
    x = np.asarray(case["X_clean"], dtype=float)
    est = first_difference_mad_noise_scale(x)
    sigma_hat = float(est["noise_scale_hat"])
    clean_rms = float(np.sqrt(np.mean(x ** 2)))
    ratio = float(sigma_hat / clean_rms) if np.isfinite(sigma_hat) and clean_rms > 0.0 else np.nan
    return {
        "audit_version": V545_QUALIFICATION_VERSION,
        "case_type": "clean_baseline_signal_only",
        "signal": signal_name,
        "noise": "clean",
        "target_snr_db": "clean",
        "seed": int(seed),
        "clean_rms": clean_rms,
        "true_noise_rms": 0.0,
        "realized_input_snr_db": np.inf,
        "noise_scale_hat": sigma_hat,
        "sigma_hat_to_true_noise_rms": np.nan,
        "abs_log_sigma_hat_ratio": np.nan,
        "scale_estimator_finite_positive": bool(np.isfinite(sigma_hat) and sigma_hat > 0.0),
        "scale_estimator_status": est["estimator_status"],
        "scale_estimator_fallback_used": bool(est["fallback_used"]),
        "sigma_hat_to_clean_rms": ratio,
        "uses_clean_truth_or_true_noise_for_estimator": False,
        "true_noise_used_for_audit_only": False,
    }


def _build_rows():
    noisy_rows = []
    for signal_name in PARAMETER_SELECTION_SIGNALS:
        for noise_name in NOISE_FAMILY:
            for target_snr_db in TARGET_SNR_DB_LEVELS:
                for seed in QUALIFICATION_SEEDS:
                    noisy_rows.append(_estimate_case(signal_name, noise_name, target_snr_db, seed))
    clean_rows = []
    for signal_name in PARAMETER_SELECTION_SIGNALS:
        for seed in QUALIFICATION_SEEDS:
            clean_rows.append(_estimate_clean_case(signal_name, seed))
    return noisy_rows, clean_rows


def run_v545_relative_h_scale_estimator_qualification(output_root):
    output_root = ensure_dir(output_root)
    noisy_rows, clean_rows = _build_rows()
    finite_noisy = [r for r in noisy_rows if bool(r["scale_estimator_finite_positive"])]
    finite_clean = [r for r in clean_rows if bool(r["scale_estimator_finite_positive"])]

    finite_noisy_rate = float(len(finite_noisy) / max(len(noisy_rows), 1))
    fallback_noisy_rate = float(
        sum(bool(r["scale_estimator_fallback_used"]) for r in noisy_rows) / max(len(noisy_rows), 1)
    )
    finite_clean_rate = float(len(finite_clean) / max(len(clean_rows), 1))
    overall_ratio_summary = _summarize(r["sigma_hat_to_true_noise_rms"] for r in finite_noisy)
    overall_abs_log_summary = _summarize(r["abs_log_sigma_hat_ratio"] for r in finite_noisy)
    by_noise = _group_summary(noisy_rows, ["noise"])
    by_snr = _group_summary(noisy_rows, ["target_snr_db"])
    by_signal = _group_summary(noisy_rows, ["signal"])
    failing_noise_groups = [
        row for row in by_noise if not bool(row["median_ratio_within_prespecified_range"])
    ]
    failing_snr_groups = [
        row for row in by_snr if not bool(row["median_ratio_within_prespecified_range"])
    ]
    failing_signal_groups = [
        row for row in by_signal if not bool(row["median_ratio_within_prespecified_range"])
    ]
    status_counts = dict(Counter(r["scale_estimator_status"] for r in noisy_rows + clean_rows))
    clean_summary = {
        "n_clean_cases": int(len(clean_rows)),
        "finite_positive_rate": finite_clean_rate,
        "fallback_rate": float(
            sum(bool(r["scale_estimator_fallback_used"]) for r in clean_rows) / max(len(clean_rows), 1)
        ),
        "sigma_hat_summary": _summarize(r["noise_scale_hat"] for r in finite_clean),
        "sigma_hat_to_clean_rms_summary": _summarize(r.get("sigma_hat_to_clean_rms") for r in finite_clean),
        "interpretation": (
            "Clean-baseline sigma_hat reflects signal local variation, not true noise. "
            "It is reported as a boundary behavior diagnostic and is not used to "
            "score noisy-case scale-estimator accuracy."
        ),
    }

    qualification_passed = bool(
        finite_noisy_rate >= 0.995
        and fallback_noisy_rate <= 0.01
        and np.isfinite(overall_abs_log_summary["median"])
        and overall_abs_log_summary["median"] <= OVERALL_MEDIAN_ABS_LOG_MAX
        and np.isfinite(overall_abs_log_summary["q75"])
        and overall_abs_log_summary["q75"] <= OVERALL_Q75_ABS_LOG_MAX
        and not failing_noise_groups
        and not failing_snr_groups
        and not failing_signal_groups
    )
    activation_status = (
        "qualified_for_relative_H_development_selection"
        if qualification_passed
        else "not_qualified_without_estimator_revision_or_narrower_claim"
    )
    dashboard = {
        "audit_version": V545_QUALIFICATION_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "relative_H_schema_version": IRMF_RELATIVE_H_PARAMETERIZATION_VERSION,
        "scale_estimator_id": IRMF_RELATIVE_H_SCALE_ESTIMATOR,
        "audit_scope": "truth-free scale estimator qualification for H_case = c_H * sigma_hat(Y)",
        "algorithm_decompositions_run": False,
        "benchmark_results_recomputed": False,
        "statistics_regenerated": False,
        "claim_authorized": False,
        "n_noisy_cases": int(len(noisy_rows)),
        "n_clean_cases": int(len(clean_rows)),
        "signals": list(PARAMETER_SELECTION_SIGNALS),
        "noises": list(NOISE_FAMILY),
        "target_snr_db_levels": list(TARGET_SNR_DB_LEVELS),
        "seeds": list(QUALIFICATION_SEEDS),
        "status_counts": status_counts,
        "finite_noisy_rate": finite_noisy_rate,
        "fallback_noisy_rate": fallback_noisy_rate,
        "overall_sigma_hat_to_true_noise_rms_summary": overall_ratio_summary,
        "overall_abs_log_sigma_hat_ratio_summary": overall_abs_log_summary,
        "clean_baseline_behavior": clean_summary,
        "pass_criteria": {
            "finite_noisy_rate_min": 0.995,
            "fallback_noisy_rate_max": 0.01,
            "overall_median_abs_log_ratio_max": OVERALL_MEDIAN_ABS_LOG_MAX,
            "overall_q75_abs_log_ratio_max": OVERALL_Q75_ABS_LOG_MAX,
            "by_group_median_ratio_range": [RATIO_MEDIAN_LOWER, RATIO_MEDIAN_UPPER],
            "note": (
                "The estimator is a truth-free scale proxy, not an oracle true-noise "
                "RMS estimator. Criteria require finite, stable, order-of-magnitude "
                "correct behavior rather than exact equality to true noise RMS."
            ),
        },
        "failing_noise_groups": [
            {k: row[k] for k in ("noise", "sigma_hat_to_true_noise_rms_median", "range_audit_status")}
            for row in failing_noise_groups
        ],
        "failing_snr_groups": [
            {k: row[k] for k in ("target_snr_db", "sigma_hat_to_true_noise_rms_median", "range_audit_status")}
            for row in failing_snr_groups
        ],
        "failing_signal_groups": [
            {k: row[k] for k in ("signal", "sigma_hat_to_true_noise_rms_median", "range_audit_status")}
            for row in failing_signal_groups
        ],
        "qualification_passed": qualification_passed,
        "activation_status": activation_status,
        "authorized_next_step": (
            "Run development-set parameter selection with theta=(h1,a,h_min,c_H) "
            "using the frozen objective, then lock one global c_H*."
            if qualification_passed
            else "Do not activate relative-H; revise or further qualify the scale estimator."
        ),
        "downstream_block_status": (
            "V5.30 benchmark and Section 6 full execution remain blocked until "
            "relative-H development selection has selected and locked c_H*."
        ),
    }

    write_csv(noisy_rows, output_root / "v545_relative_H_scale_estimator_noisy_rows.csv")
    write_csv(clean_rows, output_root / "v545_relative_H_scale_estimator_clean_rows.csv")
    write_csv(by_noise, output_root / "v545_relative_H_scale_estimator_by_noise.csv")
    write_csv(by_snr, output_root / "v545_relative_H_scale_estimator_by_snr.csv")
    write_csv(by_signal, output_root / "v545_relative_H_scale_estimator_by_signal.csv")
    write_json(dashboard, output_root / "v545_relative_H_scale_estimator_qualification_dashboard.json")
    write_manifest(output_root, {
        "audit_version": V545_QUALIFICATION_VERSION,
        "artifact_role": "relative_H_scale_estimator_qualification",
        "algorithm_decompositions_run": False,
        "benchmark_results_recomputed": False,
        "qualification_passed": qualification_passed,
        "activation_status": activation_status,
    })
    return {
        "output_dir": str(output_root),
        "dashboard": dashboard,
    }

