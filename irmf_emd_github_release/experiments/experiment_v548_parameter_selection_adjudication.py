#!/usr/bin/python
# coding: UTF-8

"""V5.48 adjudication for the V5.47 relative-H development selection.

This stage does not run decompositions and does not authorize benchmark claims.
It checks whether the completed development-selection artifacts are internally
consistent enough to lock one global IRMF parameter tuple.
"""

from pathlib import Path
import json
import math
from statistics import median

import numpy as np

from project_config import (
    IRMF_RELATIVE_H_SCALE_ESTIMATOR,
    PARAMETER_SELECTION_SEVERITY_DESIGN_VERSION,
    PARAMETER_SELECTION_TARGET_SNR_DB_LEVELS,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json, write_manifest


SELECTION_REL = Path("01_global_parameter_selection")
PROTOCOL_FILE = "section_4_parameter_selection_protocol.json"
CANDIDATE_SUMMARY_FILE = "section_4_candidate_summary.json"
STAGE1_SUMMARY_FILE = "section_4_stage_1_coarse_summary.json"
STAGE2_SUMMARY_FILE = "section_4_stage_2_refined_summary.json"
CASE_RESULTS_FILE = "section_4_candidate_case_results.json"

SNR_TOL_DB = 1e-8
RELATIVE_H_TOL = 1e-10


def _load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_float(value, default=np.nan):
    try:
        if value is None or value == "":
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _finite_values(rows, key):
    vals = [_to_float(row.get(key)) for row in rows]
    return [v for v in vals if math.isfinite(v)]


def _summary(values):
    values = [float(v) for v in values if math.isfinite(float(v))]
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _grid_values(grid, param):
    if not isinstance(grid, dict):
        return []
    values = grid.get(f"{param}_options", [])
    return [float(v) for v in values]


def _grid_count(grid, relative_h_active=True):
    if not isinstance(grid, dict):
        return 0
    required = ["h1_options", "a_options", "h_min_options"]
    h_key = "c_H_options" if relative_h_active else "H_options"
    keys = required + [h_key]
    if any(key not in grid for key in keys):
        return 0
    count = 1
    for key in keys:
        count *= len(grid.get(key, ()))
    return count


def _param_boundary_rows(selected, stage1_grid, stage2_grid):
    rows = []
    for param in ("h1", "a", "h_min", "c_H"):
        value = _to_float(selected.get(param))
        coarse_values = _grid_values(stage1_grid, param)
        refined_values = _grid_values(stage2_grid, param)
        row = {
            "parameter": param,
            "selected_value": value if math.isfinite(value) else None,
            "coarse_min": min(coarse_values) if coarse_values else None,
            "coarse_max": max(coarse_values) if coarse_values else None,
            "refined_min": min(refined_values) if refined_values else None,
            "refined_max": max(refined_values) if refined_values else None,
            "at_coarse_domain_boundary": False,
            "at_refined_grid_boundary": False,
        }
        if math.isfinite(value) and coarse_values:
            row["at_coarse_domain_boundary"] = (
                math.isclose(value, min(coarse_values), rel_tol=0.0, abs_tol=1e-12)
                or math.isclose(value, max(coarse_values), rel_tol=0.0, abs_tol=1e-12)
            )
        if math.isfinite(value) and refined_values:
            row["at_refined_grid_boundary"] = (
                math.isclose(value, min(refined_values), rel_tol=0.0, abs_tol=1e-12)
                or math.isclose(value, max(refined_values), rel_tol=0.0, abs_tol=1e-12)
            )
        rows.append(row)
    return rows


def _candidate_sort_key(row):
    return (
        _to_float(row.get("mean_case_score"), default=-np.inf),
        _to_float(row.get("mean_reconstruction_score"), default=-np.inf),
        _to_float(row.get("mean_structural_fidelity_score"), default=-np.inf),
        _to_float(row.get("mean_contamination_resistance_score"), default=-np.inf),
        -_to_float(row.get("c_H"), default=np.inf),
        -_to_float(row.get("h1"), default=np.inf),
        -_to_float(row.get("a"), default=np.inf),
        -_to_float(row.get("h_min"), default=np.inf),
        -_to_float(row.get("candidate_id"), default=np.inf),
    )


def _top_candidates(candidate_summaries, n=10):
    complete = [
        row for row in candidate_summaries
        if bool(row.get("candidate_complete_for_selection", False))
        and math.isfinite(_to_float(row.get("mean_case_score")))
    ]
    out = sorted(complete, key=_candidate_sort_key, reverse=True)[:n]
    return [
        {
            "rank": i + 1,
            "stage": row.get("stage"),
            "candidate_id": row.get("candidate_id"),
            "h1": row.get("h1"),
            "a": row.get("a"),
            "h_min": row.get("h_min"),
            "c_H": row.get("c_H"),
            "mean_case_score": row.get("mean_case_score"),
            "mean_reconstruction_score": row.get("mean_reconstruction_score"),
            "mean_structural_fidelity_score": row.get("mean_structural_fidelity_score"),
            "mean_contamination_resistance_score": row.get("mean_contamination_resistance_score"),
            "n_cases": row.get("n_cases"),
            "n_failed_or_nonfinite_cases": row.get("n_failed_or_nonfinite_cases"),
        }
        for i, row in enumerate(out)
    ]


def _group_summary(rows, group_key):
    groups = {}
    for row in rows:
        groups.setdefault(row.get(group_key), []).append(row)
    out = []
    for key, group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        score_values = _finite_values(group_rows, "case_score")
        out.append({
            group_key: key,
            "n_rows": len(group_rows),
            "n_finite_case_score": len(score_values),
            "mean_case_score": float(np.mean(score_values)) if score_values else None,
            "median_case_score": float(np.median(score_values)) if score_values else None,
            "mean_reconstruction_score": float(np.mean(_finite_values(group_rows, "reconstruction_score"))) if _finite_values(group_rows, "reconstruction_score") else None,
            "mean_structural_fidelity_score": float(np.mean(_finite_values(group_rows, "structural_fidelity_score"))) if _finite_values(group_rows, "structural_fidelity_score") else None,
            "mean_contamination_resistance_score": float(np.mean(_finite_values(group_rows, "contamination_resistance_score"))) if _finite_values(group_rows, "contamination_resistance_score") else None,
        })
    return out


def run_v548_parameter_selection_adjudication(output_root, algorithm_root=None):
    output_root = ensure_dir(output_root)
    algorithm_root = Path(algorithm_root) if algorithm_root is not None else output_root.parent
    selection_root = algorithm_root / SELECTION_REL
    protocol_path = selection_root / PROTOCOL_FILE
    protocol = _load_json(protocol_path, default=None)
    candidate_summaries = _load_json(selection_root / CANDIDATE_SUMMARY_FILE, default=None)
    stage1_summaries = _load_json(selection_root / STAGE1_SUMMARY_FILE, default=None)
    stage2_summaries = _load_json(selection_root / STAGE2_SUMMARY_FILE, default=None)
    case_rows = _load_json(selection_root / CASE_RESULTS_FILE, default=None)

    if not all(obj is not None for obj in (protocol, candidate_summaries, stage1_summaries, stage2_summaries, case_rows)):
        missing = [
            str(path.name)
            for path, obj in (
                (protocol_path, protocol),
                (selection_root / CANDIDATE_SUMMARY_FILE, candidate_summaries),
                (selection_root / STAGE1_SUMMARY_FILE, stage1_summaries),
                (selection_root / STAGE2_SUMMARY_FILE, stage2_summaries),
                (selection_root / CASE_RESULTS_FILE, case_rows),
            )
            if obj is None
        ]
        dashboard = {
            "protocol_version": "V5.48 parameter-selection adjudication",
            "adjudication_status": "selection_artifacts_missing",
            "parameter_lock_authorized": False,
            "claim_authorized": False,
            "missing_files": missing,
            "selection_root": str(selection_root),
            "benchmark_results_recomputed": False,
            "algorithm_cube_rerun_required": False,
        }
        write_json(dashboard, output_root / "v548_parameter_selection_adjudication_dashboard.json")
        write_manifest(output_root, {"stage": "v548_parameter_selection_adjudication"})
        return dashboard

    development = protocol.get("development_set", {})
    selected = protocol.get("selected_global_irmf_params") or {}
    stage1_grid = protocol.get("stage_1_coarse_grid") or {}
    stage2_grid = protocol.get("stage_2_refined_grid") or {}
    n_cases = int(development.get("n_development_cases", 0))
    relative_h_active = protocol.get("robust_loss", {}).get("H_parameterization") == "relative_noise_scale"
    stage1_expected_candidates = _grid_count(stage1_grid, relative_h_active=relative_h_active)
    stage2_expected_candidates = _grid_count(stage2_grid, relative_h_active=relative_h_active)
    expected_stage1_rows = stage1_expected_candidates * n_cases
    expected_stage2_rows = stage2_expected_candidates * n_cases
    actual_stage1_rows = sum(1 for row in case_rows if row.get("stage") == "stage_1_coarse")
    actual_stage2_rows = sum(1 for row in case_rows if row.get("stage") == "stage_2_refined")
    accounting_rows = [
        {
            "stage": "stage_1_coarse",
            "expected_candidates": stage1_expected_candidates,
            "actual_candidates": len(stage1_summaries),
            "expected_case_rows": expected_stage1_rows,
            "actual_case_rows": actual_stage1_rows,
            "accounting_closed": (
                stage1_expected_candidates == len(stage1_summaries)
                and expected_stage1_rows == actual_stage1_rows
            ),
        },
        {
            "stage": "stage_2_refined",
            "expected_candidates": stage2_expected_candidates,
            "actual_candidates": len(stage2_summaries),
            "expected_case_rows": expected_stage2_rows,
            "actual_case_rows": actual_stage2_rows,
            "accounting_closed": (
                stage2_expected_candidates == len(stage2_summaries)
                and expected_stage2_rows == actual_stage2_rows
            ),
        },
    ]
    accounting_passed = all(row["accounting_closed"] for row in accounting_rows)

    snr_errors = []
    for row in case_rows:
        target = _to_float(row.get("target_snr_db"))
        realized = _to_float(row.get("realized_input_snr_db"))
        if math.isfinite(target) and math.isfinite(realized):
            snr_errors.append(abs(realized - target))
    snr_max_error = max(snr_errors) if snr_errors else None
    snr_passed = bool(snr_errors) and snr_max_error <= SNR_TOL_DB

    relative_h_errors = []
    fallback_count = 0
    for row in case_rows:
        c_h = _to_float(row.get("c_H_candidate"))
        scale_hat = _to_float(row.get("scale_hat_case"))
        h_realized = _to_float(row.get("H_realized_case"))
        fallback_count += int(bool(row.get("scale_estimator_fallback_used", False)))
        if all(math.isfinite(v) for v in (c_h, scale_hat, h_realized)):
            relative_h_errors.append(abs(h_realized - c_h * scale_hat))
    relative_h_max_error = max(relative_h_errors) if relative_h_errors else None
    relative_h_passed = bool(relative_h_errors) and relative_h_max_error <= RELATIVE_H_TOL

    scale_estimator_id = protocol.get("robust_loss", {}).get("scale_estimator_id")
    no_oracle_leakage_protocol_passed = (
        scale_estimator_id == IRMF_RELATIVE_H_SCALE_ESTIMATOR
        and protocol.get("robust_loss", {}).get("H_parameterization") == "relative_noise_scale"
        and protocol.get("H_tuning_protocol", {}).get("no_per_case_tuning") is True
    )

    selected_candidate_id = selected.get("selected_from_candidate_id")
    selected_rows = [
        row for row in case_rows
        if row.get("candidate_id") == selected_candidate_id
    ]
    single_global_winner_passed = (
        selected_candidate_id is not None
        and len(selected_rows) == n_cases
        and selected.get("H_parameterization") == "relative_noise_scale"
        and selected.get("c_H") is not None
        and selected.get("h1") is not None
        and selected.get("a") is not None
        and selected.get("h_min") is not None
    )

    boundary_rows = _param_boundary_rows(selected, stage1_grid, stage2_grid)
    boundary_review_required = any(row["at_coarse_domain_boundary"] for row in boundary_rows)
    refined_boundary_flag = any(row["at_refined_grid_boundary"] for row in boundary_rows)

    top_rows = _top_candidates(candidate_summaries, n=10)
    top_scores = [_to_float(row.get("mean_case_score")) for row in top_rows]
    runner_up_gap = (
        float(top_scores[0] - top_scores[1])
        if len(top_scores) >= 2 and all(math.isfinite(v) for v in top_scores[:2])
        else None
    )
    best_score = top_scores[0] if top_scores and math.isfinite(top_scores[0]) else None
    near_optimal_001 = (
        sum(
            1 for row in candidate_summaries
            if math.isfinite(_to_float(row.get("mean_case_score")))
            and best_score is not None
            and best_score - _to_float(row.get("mean_case_score")) <= 0.001
        )
        if best_score is not None else None
    )
    near_optimal_005 = (
        sum(
            1 for row in candidate_summaries
            if math.isfinite(_to_float(row.get("mean_case_score")))
            and best_score is not None
            and best_score - _to_float(row.get("mean_case_score")) <= 0.005
        )
        if best_score is not None else None
    )

    nonfinite_rows = [
        row for row in case_rows
        if not math.isfinite(_to_float(row.get("case_score")))
    ]
    candidate_failures = [
        row for row in candidate_summaries
        if int(row.get("n_failed_or_nonfinite_cases", 0) or 0) > 0
    ]

    selected_by_snr = _group_summary(selected_rows, "target_snr_db")
    selected_by_signal = _group_summary(selected_rows, "signal")
    selected_by_noise = _group_summary(selected_rows, "noise")
    selected_scale_summary = _summary(_finite_values(selected_rows, "scale_hat_case"))
    selected_h_summary = _summary(_finite_values(selected_rows, "H_realized_case"))

    parameter_lock_authorized = (
        accounting_passed
        and snr_passed
        and relative_h_passed
        and no_oracle_leakage_protocol_passed
        and single_global_winner_passed
        and not boundary_review_required
        and len(nonfinite_rows) == 0
        and len(candidate_failures) == 0
    )
    if parameter_lock_authorized:
        adjudication_status = "passed_parameter_lock_authorized"
    elif boundary_review_required:
        adjudication_status = "requires_boundary_review"
    elif len(nonfinite_rows) or len(candidate_failures):
        adjudication_status = "requires_nonfinite_failure_review"
    else:
        adjudication_status = "failed_or_requires_review"

    dashboard = {
        "protocol_version": "V5.48 parameter-selection adjudication",
        "selection_protocol_version": protocol.get("protocol_version"),
        "selection_severity_design_version": development.get("severity_design_version"),
        "expected_severity_design_version": PARAMETER_SELECTION_SEVERITY_DESIGN_VERSION,
        "development_case_count": n_cases,
        "target_snr_db_levels": development.get("target_snr_db_levels"),
        "expected_target_snr_db_levels": list(PARAMETER_SELECTION_TARGET_SNR_DB_LEVELS),
        "stage_1_expected_candidates": stage1_expected_candidates,
        "stage_2_expected_candidates": stage2_expected_candidates,
        "stage_1_expected_case_rows": expected_stage1_rows,
        "stage_2_expected_case_rows": expected_stage2_rows,
        "total_expected_case_rows": expected_stage1_rows + expected_stage2_rows,
        "total_actual_case_rows": len(case_rows),
        "accounting_passed": accounting_passed,
        "snr_realization_passed": snr_passed,
        "snr_tolerance_db": SNR_TOL_DB,
        "snr_max_abs_error_db": snr_max_error,
        "snr_mean_abs_error_db": float(np.mean(snr_errors)) if snr_errors else None,
        "relative_H_provenance_passed": relative_h_passed,
        "relative_H_tolerance": RELATIVE_H_TOL,
        "relative_H_max_abs_error": relative_h_max_error,
        "relative_H_mean_abs_error": float(np.mean(relative_h_errors)) if relative_h_errors else None,
        "scale_estimator_fallback_count": fallback_count,
        "no_oracle_leakage_protocol_passed": no_oracle_leakage_protocol_passed,
        "scale_estimator_id": scale_estimator_id,
        "single_global_winner_passed": single_global_winner_passed,
        "selected_global_irmf_params": selected,
        "boundary_review_required": boundary_review_required,
        "refined_grid_boundary_flag": refined_boundary_flag,
        "runner_up_objective_gap": runner_up_gap,
        "near_optimal_candidate_count_within_0_001": near_optimal_001,
        "near_optimal_candidate_count_within_0_005": near_optimal_005,
        "n_nonfinite_case_rows": len(nonfinite_rows),
        "n_candidates_with_failed_or_nonfinite_cases": len(candidate_failures),
        "selected_scale_hat_case_summary": selected_scale_summary,
        "selected_H_realized_case_summary": selected_h_summary,
        "adjudication_status": adjudication_status,
        "parameter_lock_authorized": parameter_lock_authorized,
        "claim_authorized": False,
        "claim_boundary": (
            "This adjudication can authorize locking one global IRMF tuple "
            "(h1, a, h_min, c_H). It does not authorize benchmark performance claims."
        ),
        "benchmark_results_recomputed": False,
        "algorithm_cube_rerun_required": False,
    }

    write_json(dashboard, output_root / "v548_parameter_selection_adjudication_dashboard.json")
    write_csv(accounting_rows, output_root / "v548_selection_accounting.csv")
    write_csv(boundary_rows, output_root / "v548_boundary_audit.csv")
    write_csv(top_rows, output_root / "v548_top_candidate_summary.csv")
    write_csv(selected_by_snr, output_root / "v548_selected_candidate_by_snr.csv")
    write_csv(selected_by_signal, output_root / "v548_selected_candidate_by_signal.csv")
    write_csv(selected_by_noise, output_root / "v548_selected_candidate_by_noise.csv")
    write_json(nonfinite_rows[:200], output_root / "v548_nonfinite_case_rows_sample.json")
    write_json(candidate_failures[:200], output_root / "v548_candidate_failures_sample.json")
    write_manifest(output_root, {
        "stage": "v548_parameter_selection_adjudication",
        "selection_root": str(selection_root),
        "adjudication_status": adjudication_status,
        "parameter_lock_authorized": parameter_lock_authorized,
    })
    return dashboard


if __name__ == "__main__":
    run_v548_parameter_selection_adjudication(
        Path("IRMF_EMD_PAPER_RESULTS_V5_24_FULL_BENCHMARK")
        / "algorithm"
        / "16r_v548_parameter_selection_adjudication"
    )
