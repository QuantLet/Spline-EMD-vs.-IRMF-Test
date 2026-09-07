#!/usr/bin/python
# coding: UTF-8

"""V5.30 regular 5-dB target-SNR synthetic noise-severity design draft and audit."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from project_config import (
    DEFAULT_FS,
    DEFAULT_N,
    FULL_SIGMA_LEVELS,
    TARGET_SNR_DB_LEVELS,
    UNIFIED_BENCHMARK_REGIMES,
    V530_BENCHMARK_SEEDS,
    V530_CONVERGENCE_AUDIT_SEEDS,
    V530_SEED_CONVERGENCE_AUDIT_VERSION,
    V530_SEED_CONVERGENCE_PREFIXES,
    V530_SEED_POLICY_VERSION,
    V529_SYNTHETIC_NOISE_DESIGN_VERSION,
    V529_SUPERSEDED_GRID_NOTE,
)
from experiments.experiment_utils import make_signal_noise_case
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


def _sha256_obj(obj):
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return None


def _summarize(rows, group_keys, value_key):
    grouped = {}
    for row in rows:
        value = _finite(row.get(value_key))
        if value is None:
            continue
        key = tuple(row.get(k) for k in group_keys)
        grouped.setdefault(key, []).append(value)
    out = []
    for key, values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=float)
        item = {k: v for k, v in zip(group_keys, key)}
        item.update({
            "n": int(arr.size),
            "min": float(np.min(arr)),
            "median": float(np.median(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        })
        out.append(item)
    return out


def _snr_to_noise_rms_ratio(target_snr_db):
    return float(10.0 ** (-float(target_snr_db) / 20.0))


def _design_rows():
    rows = []
    for regime_name, cfg in UNIFIED_BENCHMARK_REGIMES.items():
        for signal_name in cfg["signals"]:
            for noise_name in cfg["noises"]:
                for target_snr_db in TARGET_SNR_DB_LEVELS:
                    sigma = _snr_to_noise_rms_ratio(target_snr_db)
                    for seed in V530_BENCHMARK_SEEDS:
                        fixed_case = make_signal_noise_case(
                            signal_name=signal_name,
                            noise_name=noise_name,
                            sigma=sigma,
                            n=DEFAULT_N,
                            fs=DEFAULT_FS,
                            seed=seed,
                        )
                        target_case = make_signal_noise_case(
                            signal_name=signal_name,
                            noise_name=noise_name,
                            sigma=sigma,
                            target_snr_db=target_snr_db,
                            n=DEFAULT_N,
                            fs=DEFAULT_FS,
                            seed=seed,
                        )
                        rows.append({
                            "signal_regime": regime_name,
                            "signal": signal_name,
                            "noise": noise_name,
                            "sigma_legacy_label": float(sigma),
                            "target_snr_db": target_snr_db,
                            "seed": int(seed),
                            "clean_signal_rms": fixed_case["clean_signal_rms"],
                            "fixed_sigma_noise_rms": fixed_case["noise_rms"],
                            "fixed_sigma_realized_snr_db": fixed_case["realized_input_snr_db"],
                            "target_snr_noise_scale_alpha": target_case["noise_scale_alpha"],
                            "target_snr_noise_rms": target_case["noise_rms"],
                            "target_snr_realized_snr_db": target_case["realized_input_snr_db"],
                            "target_snr_abs_error_db": abs(
                                float(target_case["realized_input_snr_db"]) - target_snr_db
                            ),
                        })
    return rows


def run_v529_snr_design_draft(output_root):
    output_root = ensure_dir(output_root)
    snr_grid = [
        {
            "target_snr_db": float(snr),
            "noise_rms_to_signal_rms_ratio": _snr_to_noise_rms_ratio(snr),
            "formula": "noise_rms / signal_rms = 10^(-target_snr_db/20)",
        }
        for snr in TARGET_SNR_DB_LEVELS
    ]
    draft = {
        "schema_version": V529_SYNTHETIC_NOISE_DESIGN_VERSION,
        "schema_status": "draft_requires_audit",
        "revision_target": "synthetic input noise-severity design",
        "benchmark_source_version": "V5.28",
        "supersedes": {
            "version": "V5.29_target_snr_noise_severity_design",
            "reason": (
                "Legacy-equivalent SNR levels were replaced before algorithmic "
                "evaluation by a regular 5-dB grid for interpretability and "
                "systematic severity coverage."
            ),
            "benchmark_results_generated_under_superseded_grid": False,
            "note": V529_SUPERSEDED_GRID_NOTE,
        },
        "benchmark_results_recomputed": False,
        "algorithm_cube_rerun_required_for_v530_results": True,
        "clean_signal_policy": "preserve original synthetic waveform amplitude; do not force unit-RMS normalization",
        "noise_policy": "generate standardized base noise Z, then rescale per signal/noise/seed to target input SNR",
        "snr_definition": {
            "formula": "SNR_in_db = 10 log10(sum_t X_t^2 / sum_t N_t^2)",
            "reference": "energy-ratio SNR definition used by ECG denoising literature",
        },
        "noise_generation_formula": {
            "observed_signal": "Y_t^(gamma) = X_t + N_t^(gamma)",
            "base_noise": "Z_t generated by the frozen noise model",
            "scaled_noise": "N_t^(gamma) = alpha_gamma Z_t",
            "alpha_gamma": "sqrt(sum_t X_t^2 / (10^(gamma/10) sum_t Z_t^2))",
            "guarantee": "10 log10(sum_t X_t^2 / sum_t (alpha_gamma Z_t)^2) = gamma",
        },
        "target_snr_db_levels": list(TARGET_SNR_DB_LEVELS),
        "seed_policy": {
            "version": V530_SEED_POLICY_VERSION,
            "primary_benchmark_seeds": list(V530_BENCHMARK_SEEDS),
            "n_primary_benchmark_seeds": int(len(V530_BENCHMARK_SEEDS)),
            "rationale": (
                "Use 20 Monte Carlo noise realizations for the primary V5.30 "
                "target-SNR benchmark, matching the established protocol while "
                "treating seed sufficiency as a separate Monte Carlo convergence "
                "qualification question."
            ),
            "planned_convergence_audit": {
                "version": V530_SEED_CONVERGENCE_AUDIT_VERSION,
                "candidate_prefixes": list(V530_SEED_CONVERGENCE_PREFIXES),
                "available_convergence_audit_seeds": list(V530_CONVERGENCE_AUDIT_SEEDS),
                "preferred_design": (
                    "Use the same full benchmark signal/noise/SNR cells and "
                    "extend only Monte Carlo seeds to 50, then evaluate nested "
                    "prefixes R=10,20,30,50."
                ),
                "fallback_design": (
                    "If full 50-seed extension is too costly, use a "
                    "representative subset spanning stationary, chirp, "
                    "close-frequency, transient, and weak-component signals; "
                    "Gaussian, heavy-tailed, colored, and impulsive noises; "
                    "and low, moderate, and high noise severities."
                ),
                "decision_rule": (
                    "If 10-, 20-, 30-, and 50-seed prefixes yield materially "
                    "stable metric summaries, paired effects, average ranks, "
                    "and method ordering, retain 20 seeds as sufficient. "
                    "Escalate the primary cube only if convergence evidence "
                    "indicates instability."
                ),
            },
        },
        "severity_grid": snr_grid,
        "clean_baseline_policy": {
            "recommended": True,
            "definition": "Y_t = X_t with no added noise",
            "role": "reference condition only; excluded from SNR-level averages",
            "implemented_in_algorithm_cube": False,
        },
        "version_boundary": {
            "changes": [
                "Replace fixed absolute sigma amplitude as the synthetic severity control with target input SNR.",
                "Replace legacy-equivalent V5.29 target-SNR levels with a regular 0:5:30 dB grid.",
                "Use sigma only as a derived noise-to-signal RMS-ratio label in target-SNR runs.",
            ],
            "unchanged": [
                "Synthetic signal formulas",
                "Noise model families",
                "Monte Carlo seeds",
                "Method parameters",
                "V5.28 primary metric formulas",
                "Protocol reconstruction rule",
            ],
        },
        "freeze_requirements": [
            "target_snr_realization_audit_passed",
            "parameter_lock_unchanged",
            "benchmark_eligibility_gate_rerun_for_v530",
            "protocol_controlled_full_benchmark_rerun_required",
            "v528_results_not_mixed_with_v530_statistics",
        ],
        "draft_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    draft["draft_checksum_sha256"] = _sha256_obj(draft)
    write_json(draft, output_root / "v530_snr_design_draft.json")
    return draft


def run_v530_snr_realization_audit(output_root):
    output_root = ensure_dir(output_root)
    rows = _design_rows()
    fixed_by_sigma = _summarize(rows, ["sigma_legacy_label"], "fixed_sigma_realized_snr_db")
    fixed_by_signal_sigma = _summarize(
        rows,
        ["signal_regime", "signal", "sigma_legacy_label"],
        "fixed_sigma_realized_snr_db",
    )
    target_by_snr = _summarize(rows, ["target_snr_db"], "target_snr_realized_snr_db")
    target_error_by_snr = _summarize(rows, ["target_snr_db"], "target_snr_abs_error_db")
    max_target_error = max(float(row["target_snr_abs_error_db"]) for row in rows)
    clean_rms_values = [float(row["clean_signal_rms"]) for row in rows]
    fixed_snr_values = [float(row["fixed_sigma_realized_snr_db"]) for row in rows]
    status = "passed" if max_target_error <= 1e-9 else "failed"
    dashboard = {
        "audit": "v530_snr_realization_audit",
        "schema_version": V529_SYNTHETIC_NOISE_DESIGN_VERSION,
        "audit_status": status,
        "n_signal_noise_snr_seed_cells": int(len(rows)),
        "n_target_snr_levels": int(len(TARGET_SNR_DB_LEVELS)),
        "n_primary_benchmark_seeds": int(len(V530_BENCHMARK_SEEDS)),
        "seed_policy_version": V530_SEED_POLICY_VERSION,
        "seed_convergence_audit_version": V530_SEED_CONVERGENCE_AUDIT_VERSION,
        "seed_convergence_candidate_prefixes": list(V530_SEED_CONVERGENCE_PREFIXES),
        "n_seed_convergence_audit_seeds": int(len(V530_CONVERGENCE_AUDIT_SEEDS)),
        "target_snr_db_levels": list(TARGET_SNR_DB_LEVELS),
        "clean_signal_unit_rms_normalized": False,
        "clean_signal_rms_min": float(np.min(clean_rms_values)),
        "clean_signal_rms_max": float(np.max(clean_rms_values)),
        "fixed_sigma_realized_snr_db_min": float(np.min(fixed_snr_values)),
        "fixed_sigma_realized_snr_db_max": float(np.max(fixed_snr_values)),
        "target_snr_max_abs_error_db": float(max_target_error),
        "algorithm_cube_rerun_required_for_v530_results": True,
        "statistics_regeneration_alone_sufficient": False,
        "interpretation": (
            "V5.28 fixed-sigma design standardizes noise but not clean signal RMS, "
            "so realized input SNR varies by signal. The V5.30 regular 5-dB "
            "target-SNR scaling attains the requested energy-ratio SNR for every "
            "synthetic case."
        ),
        "output_files": {
            "case_level": "v530_snr_realization_case_level.csv",
            "fixed_by_sigma": "v530_fixed_sigma_realized_snr_by_sigma.csv",
            "fixed_by_signal_sigma": "v530_fixed_sigma_realized_snr_by_signal_sigma.csv",
            "target_by_snr": "v530_target_snr_realized_by_level.csv",
            "target_error_by_snr": "v530_target_snr_abs_error_by_level.csv",
        },
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_csv(rows, output_root / "v530_snr_realization_case_level.csv")
    write_csv(fixed_by_sigma, output_root / "v530_fixed_sigma_realized_snr_by_sigma.csv")
    write_csv(fixed_by_signal_sigma, output_root / "v530_fixed_sigma_realized_snr_by_signal_sigma.csv")
    write_csv(target_by_snr, output_root / "v530_target_snr_realized_by_level.csv")
    write_csv(target_error_by_snr, output_root / "v530_target_snr_abs_error_by_level.csv")
    write_json(dashboard, output_root / "v530_snr_realization_audit_dashboard.json")
    return dashboard


def run_v530_snr_design_freeze(output_root):
    output_root = ensure_dir(output_root)
    draft_path = output_root / "v530_snr_design_draft.json"
    audit_path = output_root / "v530_snr_realization_audit_dashboard.json"
    if not draft_path.exists():
        run_v529_snr_design_draft(output_root)
    if not audit_path.exists():
        run_v530_snr_realization_audit(output_root)
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    checks = {
        "draft_schema_version_matches": draft.get("schema_version") == V529_SYNTHETIC_NOISE_DESIGN_VERSION,
        "realization_audit_passed": audit.get("audit_status") == "passed",
        "target_snr_error_within_tolerance": float(audit.get("target_snr_max_abs_error_db", np.inf)) <= 1e-9,
        "clean_signal_not_unit_rms_normalized_recorded": audit.get("clean_signal_unit_rms_normalized") is False,
        "algorithm_cube_rerun_required_recorded": audit.get("algorithm_cube_rerun_required_for_v530_results") is True,
        "statistics_only_regeneration_rejected": audit.get("statistics_regeneration_alone_sufficient") is False,
    }
    status = "frozen" if all(checks.values()) else "freeze_denied"
    frozen = dict(draft)
    frozen.update({
        "schema_status": status,
        "qualification_status": "passed" if status == "frozen" else "failed",
        "benchmark_results_under_v5_30": "not_generated",
        "benchmark_results_recomputed": False,
        "algorithm_cube_rerun_required": True,
        "statistics_regeneration_required_after_rerun": True,
        "source_realization_audit": str(audit_path),
        "source_realization_audit_sha256": _sha256_obj(audit),
        "freeze_checks": checks,
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    })
    manifest = {
        "stage": "v530_snr_design_freeze",
        "schema_version": V529_SYNTHETIC_NOISE_DESIGN_VERSION,
        "schema_status": status,
        "qualification_status": frozen["qualification_status"],
        "benchmark_results_under_v5_30": "not_generated",
        "benchmark_results_recomputed": False,
        "algorithm_cube_rerun_required": True,
        "statistics_regeneration_required_after_rerun": True,
        "freeze_checks": checks,
        "next_required_stage": "v530_benchmark_eligibility_gate_and_protocol_controlled_full_rerun",
    }
    write_json(frozen, output_root / "v530_snr_design_frozen.json")
    write_json(manifest, output_root / "v530_snr_design_freeze_manifest.json")
    if status != "frozen":
        failed = [k for k, v in checks.items() if not v]
        raise RuntimeError("V5.30 SNR-design freeze denied: " + ", ".join(failed))
    return manifest


# Backward-compatible entry points kept for CLI stages that were introduced
# before the V5.30 regular-grid refinement.
run_v529_snr_realization_audit = run_v530_snr_realization_audit
run_v529_snr_design_freeze = run_v530_snr_design_freeze
