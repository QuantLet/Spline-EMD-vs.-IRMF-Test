#!/usr/bin/python
# coding: UTF-8

"""V5.59 post-execution qualification gate for the V5.57 Section 6 amendment."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from project_config import (
    EMD_FAMILY_BENCHMARK_MODES,
    EMD_FAMILY_SENSITIVITY_NOISE_WIDTHS,
    EMD_FAMILY_SENSITIVITY_TRIALS,
    EMD_PARAMETER_SELECTION_GRID,
    CEEMDAN_PARAMETER_SELECTION_GRID,
    EVALUATION_METHODS,
    PARAMETER_SENSITIVITY_FACTORS,
    PARAMETER_SENSITIVITY_NOISES,
    PARAMETER_SENSITIVITY_SIGNALS,
    PARAMETER_SENSITIVITY_TARGET_SNR_DB_LEVELS,
    SIGNAL_VARIANT_FAMILY,
    SIGNAL_VARIANT_NOISES,
    SIGNAL_VARIANT_SEEDS,
    SIGNAL_VARIANT_TARGET_SNR_DB_LEVELS,
    V557_SECTION6_2_CONTAMINATION_SIGNALS,
    V557_SECTION6_3_CHALLENGING_SIGNAL_EXTENSION,
    V557_SECTION6_4_SCALING_SIGNALS,
)
from diagnostics.shared_physical_diagnostics import SYNTHETIC_RECONSTRUCTION_PROTOCOL_ID
from experiments.experiment_v551_section6_2_contamination_design import _designs_for_signal
from experiments.experiment_signal_variant_robustness import _target_snr_checkpoint_key
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json
from signal_bank.synthetic_signals import SIGNAL_VARIANT_METADATA


V559_SECTION6_POST_EXECUTION_QUALIFICATION_VERSION = (
    "V5.59_section6_post_execution_qualification"
)
METHODS = tuple(EVALUATION_METHODS)


def _read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_row_count(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8", newline="") as f:
        return max(0, sum(1 for _ in csv.reader(f)) - 1)


def _jsonl_row_count(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _active_scope_signal_variant_row_count(path):
    path = Path(path)
    if not path.exists():
        return None
    expected_ids = {
        _target_snr_checkpoint_key(signal_name, noise_name, target_snr_db, int(seed))
        for signal_name in SIGNAL_VARIANT_FAMILY
        for noise_name in SIGNAL_VARIANT_NOISES
        for target_snr_db in SIGNAL_VARIANT_TARGET_SNR_DB_LEVELS
        for seed in SIGNAL_VARIANT_SEEDS
    }
    completed_ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except Exception:
                continue
            cell_id = row.get("variant_cell_id")
            if cell_id in expected_ids:
                completed_ids.add(cell_id)
    return len(completed_ids)


def _product(options):
    out = 1
    for values in options:
        out *= len(values)
    return int(out)


def _emd_family_config_count():
    emd_count = _product((
        EMD_PARAMETER_SELECTION_GRID["nbsym_options"],
        EMD_PARAMETER_SELECTION_GRID["spline_kind_options"],
        EMD_PARAMETER_SELECTION_GRID["max_imf_options"],
        EMD_PARAMETER_SELECTION_GRID["std_thr_options"],
        EMD_PARAMETER_SELECTION_GRID["svar_thr_options"],
        EMD_PARAMETER_SELECTION_GRID["total_power_thr_options"],
        EMD_PARAMETER_SELECTION_GRID["range_thr_options"],
    ))
    eemd_count = len(EMD_FAMILY_SENSITIVITY_TRIALS) * len(EMD_FAMILY_SENSITIVITY_NOISE_WIDTHS)
    ceemdan_count = len(EMD_FAMILY_SENSITIVITY_TRIALS) * len(CEEMDAN_PARAMETER_SELECTION_GRID["epsilon_options"])
    return int(emd_count + eemd_count + ceemdan_count)


def _expected_counts():
    parameter_design_n = _product(PARAMETER_SENSITIVITY_FACTORS.values())
    contamination_design_n = sum(
        len(_designs_for_signal(signal_name))
        for signal_name in V557_SECTION6_2_CONTAMINATION_SIGNALS
    )
    s63b_cfg = EMD_FAMILY_BENCHMARK_MODES["representative"]
    n_grid = (250, 500, 1000, 2000, 4000)
    s64_seeds = (0, 1)
    s64_repeats = 3
    return {
        "6.1_rows": int(
            parameter_design_n
            * len(PARAMETER_SENSITIVITY_SIGNALS)
            * len(PARAMETER_SENSITIVITY_NOISES)
            * len(PARAMETER_SENSITIVITY_TARGET_SNR_DB_LEVELS)
        ),
        "6.1_case_blocks": int(
            len(PARAMETER_SENSITIVITY_SIGNALS)
            * len(PARAMETER_SENSITIVITY_NOISES)
            * len(PARAMETER_SENSITIVITY_TARGET_SNR_DB_LEVELS)
        ),
        "6.2_rows": int(contamination_design_n * 3 * len(METHODS)),
        "6.3A_case_rows": int(
            len(SIGNAL_VARIANT_FAMILY)
            * len(SIGNAL_VARIANT_NOISES)
            * len(SIGNAL_VARIANT_TARGET_SNR_DB_LEVELS)
            * len(SIGNAL_VARIANT_SEEDS)
        ),
        "6.3A_method_evaluations": int(
            len(SIGNAL_VARIANT_FAMILY)
            * len(SIGNAL_VARIANT_NOISES)
            * len(SIGNAL_VARIANT_TARGET_SNR_DB_LEVELS)
            * len(SIGNAL_VARIANT_SEEDS)
            * len(METHODS)
        ),
        "6.3B_rows": int(
            len(s63b_cfg["signals"])
            * len(s63b_cfg["noises"])
            * len(PARAMETER_SENSITIVITY_TARGET_SNR_DB_LEVELS)
            * _emd_family_config_count()
        ),
        "6.4_rows": int(
            len(n_grid)
            * len(V557_SECTION6_4_SCALING_SIGNALS)
            * len(s64_seeds)
            * len(METHODS)
            * s64_repeats
        ),
    }


def _signal_class(signal_name):
    if SIGNAL_VARIANT_METADATA.get(signal_name, {}).get("class") == "challenging":
        return "challenging"
    return "canonical_or_variant"


def _signal_difficulty_review_rows(algorithm_root):
    path = (
        algorithm_root / "09_signal_variant_robustness_target_snr"
        / "variant_family_reproducibility_summary.csv"
    )
    rows = []
    if not path.exists():
        return [{
            "review_id": "signal_family_summary_missing",
            "passed": False,
            "detail": str(path),
            "interpretation": "Cannot screen canonical-vs-challenging qualitative patterns until 6.3A has been rerun.",
        }]
    with open(path, "r", encoding="utf-8", newline="") as f:
        source_rows = list(csv.DictReader(f))
    metrics = (
        ("denoise_nmse", "denoise_nmse"),
        ("denoise_corr", "denoise_corr"),
        ("matched_component_corr", "imf_recovery_corr"),
        ("matched_component_nrmse", "imf_recovery_nrmse"),
    )
    for publication_metric, artifact_metric in metrics:
        metric_rows = [r for r in source_rows if r.get("metric") == artifact_metric]
        challenging_rows = [
            r for r in metric_rows
            if _signal_class(r.get("variant_family", "")) == "challenging"
        ]
        rows.append({
            "review_id": f"{publication_metric}_challenging_family_rows_present",
            "publication_metric": publication_metric,
            "artifact_metric": artifact_metric,
            "n_challenging_family_rows": len(challenging_rows),
            "passed": bool(challenging_rows),
            "interpretation": (
                "Presence check only. Final wording still requires inspecting whether "
                "challenging regimes show reversal or attenuation relative to canonical variants."
            ),
        })
    return rows


def run_v559_section6_post_execution_qualification(output_root):
    output_root = ensure_dir(output_root)
    algorithm_root = output_root.parent if output_root.name.startswith("16") else output_root
    expected = _expected_counts()

    v558 = _read_json(
        algorithm_root / "16aa_v558_section6_pre_execution_audit"
        / "v558_section6_pre_execution_audit_dashboard.json"
    )
    s61_status = _read_json(
        algorithm_root / "08_robustness_sensitivity_target_snr"
        / "6_3_parameter_sensitivity" / "section_6_3_parameter_sensitivity_status.json"
    )
    s61_case_blocked = _read_json(
        algorithm_root / "08_robustness_sensitivity_target_snr"
        / "6_3_parameter_sensitivity"
        / "section_6_3_primary_endpoint_case_blocked_factorial.json"
    )
    s62 = _read_json(
        algorithm_root / "16t_v551_section6_2_contamination_design"
        / "v551_section6_2_contamination_design_dashboard.json"
    )
    s63a = _read_json(
        algorithm_root / "09_signal_variant_robustness_target_snr"
        / "section_5_signal_family_reproducibility_dashboard.json"
    )
    s64 = _read_json(
        algorithm_root / "16u_v552_section6_4_computational_scaling"
        / "v552_section6_4_runtime_scaling_dashboard.json"
    )

    s63a_jsonl_path = (
        algorithm_root / "09_signal_variant_robustness_target_snr"
        / "section_5_signal_family_reproducibility_target_snr_rows.jsonl"
    )
    s63a_jsonl_rows = _active_scope_signal_variant_row_count(s63a_jsonl_path)
    s63a_jsonl_total_rows = _jsonl_row_count(s63a_jsonl_path)
    s63b_rows = _csv_row_count(
        algorithm_root / "11_emd_family_sensitivity_target_snr"
        / "appendix_D_emd_family_sensitivity.csv"
    )

    checks = [
        {
            "check_id": "v558_pre_execution_audit_passed",
            "passed": bool(v558 and v558.get("audit_status") == "passed"),
            "expected": "passed",
            "observed": None if v558 is None else v558.get("audit_status"),
        },
        {
            "check_id": "section6_1_v557_row_count_complete",
            "passed": bool(
                s61_status
                and int(s61_status.get("n_completed_rows", -1)) == expected["6.1_rows"]
                and int(s61_status.get("n_expected_rows", -2)) == expected["6.1_rows"]
                and s61_status.get("reconstruction_protocol_id") == SYNTHETIC_RECONSTRUCTION_PROTOCOL_ID
            ),
            "expected": expected["6.1_rows"],
            "observed": None if s61_status is None else s61_status.get("n_completed_rows"),
        },
        {
            "check_id": "section6_1_case_blocked_factorial_v557_case_blocks",
            "passed": bool(
                s61_case_blocked
                and int(s61_case_blocked.get("primary_endpoint_count", 0)) == 10
                and int(s61_case_blocked.get("n_factorial_settings", 0)) == 81
                and int(s61_case_blocked.get("n_case_blocks", -1)) == expected["6.1_case_blocks"]
            ),
            "expected": expected["6.1_case_blocks"],
            "observed": None if s61_case_blocked is None else s61_case_blocked.get("n_case_blocks"),
        },
        {
            "check_id": "section6_2_v557_row_count_complete",
            "passed": bool(
                s62
                and s62.get("statistics_complete") is True
                and int(s62.get("n_failed", 1)) == 0
                and int(s62.get("n_method_evaluations", -1)) == expected["6.2_rows"]
                and s62.get("normalized_spillover_formula_audit", {}).get("passed") is True
            ),
            "expected": expected["6.2_rows"],
            "observed": None if s62 is None else s62.get("n_method_evaluations"),
        },
        {
            "check_id": "section6_3A_v557_signal_variant_scope_complete",
            "passed": bool(
                s63a
                and s63a.get("protocol", {}).get("n_method_case_evaluations") == expected["6.3A_method_evaluations"]
                and s63a_jsonl_rows == expected["6.3A_case_rows"]
            ),
            "expected": expected["6.3A_case_rows"],
            "observed": s63a_jsonl_rows,
        },
        {
            "check_id": "section6_3B_canonical_comparator_scope_present",
            "passed": bool(s63b_rows == expected["6.3B_rows"]),
            "expected": expected["6.3B_rows"],
            "observed": s63b_rows,
        },
        {
            "check_id": "section6_4_v557_scaling_scope_complete",
            "passed": bool(
                s64
                and s64.get("statistics_complete") is True
                and int(s64.get("n_failed", 1)) == 0
                and int(s64.get("n_method_evaluations", -1)) == expected["6.4_rows"]
            ),
            "expected": expected["6.4_rows"],
            "observed": None if s64 is None else s64.get("n_method_evaluations"),
        },
    ]

    difficulty_rows = _signal_difficulty_review_rows(algorithm_root)
    all_passed = all(bool(row["passed"]) for row in checks)
    review_cues_passed = all(bool(row["passed"]) for row in difficulty_rows)
    dashboard = {
        "schema_version": V559_SECTION6_POST_EXECUTION_QUALIFICATION_VERSION,
        "created_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "qualification_status": (
            "passed_pending_manual_wording_review"
            if all_passed and review_cues_passed
            else "not_passed_pending_v557_reruns_or_review"
        ),
        "execution_integrity_authorized": bool(all_passed),
        "claim_authorized": bool(all_passed and review_cues_passed),
        "claim_authorization_boundary": (
            "V5.59 checks V5.57 post-execution completeness and provides "
            "canonical/challenging review cues. Even when passed, final Section 6 "
            "wording must be updated to reflect any attenuation or reversal in "
            "challenging regimes; V5.56 waveform stability remains separately qualified."
        ),
        "expected_counts": expected,
        "checkpoint_accounting_note": (
            "Section 6.3A completion is evaluated on active V5.66/V5.67 "
            "signal/noise/SNR/seed scope unique variant_cell_id rows; older "
            "out-of-scope checkpoint rows are retained for provenance but do "
            "not enter the completeness gate."
        ),
        "section6_3A_total_jsonl_rows_including_legacy_scope": s63a_jsonl_total_rows,
        "checks": checks,
        "signal_difficulty_review_cues": difficulty_rows,
        "required_next_steps_if_not_passed": [
            row["check_id"] for row in checks if not row["passed"]
        ] + [
            row["review_id"] for row in difficulty_rows if not row["passed"]
        ],
    }
    write_json(dashboard, output_root / "v559_section6_post_execution_qualification_dashboard.json")
    write_csv(checks, output_root / "v559_section6_post_execution_qualification_checks.csv")
    write_csv(difficulty_rows, output_root / "v559_signal_difficulty_review_cues.csv")
    return dashboard
