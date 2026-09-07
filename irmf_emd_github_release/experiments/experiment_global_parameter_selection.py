#!/usr/bin/python
# coding: UTF-8

"""
Section 4: Global Parameter Selection.

This module implements the V5.39 two-stage coarse-to-fine development-set
procedure:

    1. A 60-case stratified representative development set is used only for
       parameter selection.
    2. Stage I evaluates an expanded-domain coarse grid while constraining the
       geometric scale parameter a to the Spokoiny-consistent range (1, 2].
       The Gaussian-smoothed median loss width H is included as a fourth
       development-set hyperparameter because it controls the robust contrast
       scale.
    3. Stage II constructs a deterministic local grid around the Stage-I best
       configuration.
    4. The best Stage-II configuration is locked for all subsequent benchmark
       sections.

The procedure selects the best configuration inside a finite search design; it
does not claim global optimality over the continuous IRMF parameter space.
"""

from pathlib import Path
import itertools
import json
import time

import numpy as np

from project_config import (
    DEFAULT_N,
    DEFAULT_FS,
    DEFAULT_SEED,
    GLOBAL_IRMF_PARAMS,
    IRMF_RELATIVE_H_C_OPTIONS,
    IRMF_RELATIVE_H_PARAMETERIZATION_VERSION,
    IRMF_RELATIVE_H_SCALE_ESTIMATOR,
    PARAMETER_SELECTION_NOISES,
    PARAMETER_SELECTION_SIGMAS,
    PARAMETER_SELECTION_SIGNALS,
    PARAMETER_SELECTION_SEVERITY_DESIGN_VERSION,
    PARAMETER_SELECTION_TARGET_SNR_DB_LEVELS,
    PAPER_IRMF_COARSE_PARAMETER_GRID,
    PAPER_IRMF_REFINEMENT_STEPS,
    PAPER_OUTPUT_ROOT_NAME,
)
from experiments.experiment_utils import make_signal_noise_case, run_fixed_irmf_case, method_result_summary
from experiments.experiment_v544_h_decision_gate import first_difference_mad_noise_scale
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


PARAMETER_KEYS = ("h1", "a", "h_min", "c_H")
EXPECTED_DEVELOPMENT_CASE_COUNT = 60
MIN_COMPLETE_DEVELOPMENT_CASES = EXPECTED_DEVELOPMENT_CASE_COUNT
RELATIVE_H_PARAMETERIZATION_ACTIVE = True
TARGET_SNR_DEVELOPMENT_DESIGN_ACTIVE = True
SELECTION_COMPUTATION_PROTOCOL_VERSION = "V5.49_resumable_checkpointed_deterministic_selection_execution"


def _json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _append_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")


def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _checkpoint_paths(checkpoint_root, stage):
    checkpoint_root = Path(checkpoint_root)
    return {
        "case_rows": checkpoint_root / f"{stage}_candidate_case_results.jsonl",
        "summaries": checkpoint_root / f"{stage}_candidate_summaries.jsonl",
    }


def _checkpoint_fingerprint(signals, noises, sigmas, target_snr_db_levels, coarse_grid, seed, n, fs):
    return {
        "selection_computation_protocol_version": SELECTION_COMPUTATION_PROTOCOL_VERSION,
        "selection_protocol_version": "V5.47_target_snr_relative_H_development_selection",
        "signals": list(signals),
        "noises": list(noises),
        "sigmas": list(sigmas),
        "target_snr_db_levels": list(target_snr_db_levels) if target_snr_db_levels is not None else None,
        "coarse_grid": _json_safe(coarse_grid),
        "seed": int(seed),
        "n": int(n),
        "fs": float(fs),
        "expected_development_case_count": int(EXPECTED_DEVELOPMENT_CASE_COUNT),
    }


def _prepare_checkpoint_root(checkpoint_root, fingerprint):
    checkpoint_root = ensure_dir(checkpoint_root)
    manifest_path = checkpoint_root / "checkpoint_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != _json_safe(fingerprint):
            raise RuntimeError(
                "Existing V5.49 parameter-selection checkpoint manifest does not match "
                "the current frozen selection design. Use a new output root or archive "
                f"the stale checkpoint directory: {checkpoint_root}"
            )
    else:
        write_json({
            "checkpoint_status": "active_resumable_selection_checkpoint",
            "fingerprint": fingerprint,
            "candidate_completion_policy": (
                "Rows are checkpointed only after all required development cases for "
                "a candidate complete. Interrupted partial candidates are rerun."
            ),
        }, manifest_path)
    return checkpoint_root


def _h_options_from_grid(grid):
    if RELATIVE_H_PARAMETERIZATION_ACTIVE:
        return tuple(grid.get("c_H_options", IRMF_RELATIVE_H_C_OPTIONS))
    return tuple(grid["H_options"])


def _candidate_params(grid, base_params):
    for h1, a, h_min, h_scale in itertools.product(
            grid["h1_options"],
            grid["a_options"],
            grid["h_min_options"],
            _h_options_from_grid(grid),
    ):
        params = dict(base_params)
        params.update({
            "h1": float(h1),
            "a": float(a),
            "h_min": float(h_min),
        })
        if RELATIVE_H_PARAMETERIZATION_ACTIVE:
            params["c_H"] = float(h_scale)
            params["H_parameterization"] = "relative_noise_scale"
            params["scale_estimator_id"] = IRMF_RELATIVE_H_SCALE_ESTIMATOR
            params.pop("H", None)
            params.pop("loss_tuning", None)
        else:
            params["H"] = float(h_scale)
            params["H_parameterization"] = "absolute"
        yield params


def _grid_bounds(grid):
    h_key = "c_H" if RELATIVE_H_PARAMETERIZATION_ACTIVE else "H"
    h_options_key = "c_H_options" if RELATIVE_H_PARAMETERIZATION_ACTIVE else "H_options"
    return {
        "h1": (min(grid["h1_options"]), max(grid["h1_options"])),
        "a": (min(grid["a_options"]), max(grid["a_options"])),
        "h_min": (min(grid["h_min_options"]), max(grid["h_min_options"])),
        h_key: (min(grid[h_options_key]), max(grid[h_options_key])),
    }


def _rounded_unique(values, ndigits=12):
    out = []
    seen = set()
    for value in values:
        rounded = round(float(value), ndigits)
        if rounded in seen:
            continue
        seen.add(rounded)
        out.append(float(rounded))
    return tuple(out)


def _refined_levels(best_value, lower, upper, step):
    """
    Deterministic boundary-adaptive three-level refinement rule.

    Interior:
        {theta_best - Delta, theta_best, theta_best + Delta}
    Left boundary:
        {theta_min, theta_min + Delta, theta_min + 2 Delta}
    Right boundary:
        {theta_max - 2 Delta, theta_max - Delta, theta_max}

    For near-boundary values, candidates are clipped and completed inward so
    that the refined grid remains deterministic and as close as possible to
    three ordered levels inside the original coarse search interval.
    """
    best_value = float(best_value)
    lower = float(lower)
    upper = float(upper)
    step = float(step)
    tol = max(1e-12, 1e-9 * max(abs(lower), abs(upper), abs(best_value), 1.0))

    if best_value <= lower + tol:
        values = [lower, lower + step, lower + 2.0 * step]
    elif best_value >= upper - tol:
        values = [upper - 2.0 * step, upper - step, upper]
    else:
        values = [best_value - step, best_value, best_value + step]

    values = [min(max(v, lower), upper) for v in values]

    # If clipping created duplicates, fill deterministically from the interior.
    candidates = _rounded_unique(sorted(values))
    if len(candidates) < 3:
        grid_values = np.arange(lower, upper + 0.5 * step, step)
        ranked = sorted(
            [float(v) for v in grid_values if lower - tol <= v <= upper + tol],
            key=lambda v: (abs(v - best_value), v),
        )
        expanded = list(candidates)
        for value in ranked:
            if len(expanded) >= 3:
                break
            if all(abs(value - existing) > tol for existing in expanded):
                expanded.append(float(value))
        candidates = _rounded_unique(sorted(expanded))

    return candidates


def build_refined_grid(best_stage1, coarse_grid, refinement_steps):
    bounds = _grid_bounds(coarse_grid)
    h_key = "c_H" if RELATIVE_H_PARAMETERIZATION_ACTIVE else "H"
    h_options_key = "c_H_options" if RELATIVE_H_PARAMETERIZATION_ACTIVE else "H_options"
    return {
        "h1_options": _refined_levels(best_stage1["h1"], *bounds["h1"], refinement_steps["h1"]),
        "a_options": _refined_levels(best_stage1["a"], *bounds["a"], refinement_steps["a"]),
        "h_min_options": _refined_levels(best_stage1["h_min"], *bounds["h_min"], refinement_steps["h_min"]),
        h_options_key: _refined_levels(
            best_stage1[h_key],
            *bounds[h_key],
            refinement_steps[h_key],
        ),
    }


def _summarize_scores(scores):
    arr = np.asarray([v for v in scores if np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return None
    return float(np.mean(arr))


def _candidate_sort_key(row):
    """Deterministic winner and tie-breaking key.

    Primary objective is larger mean_case_score.  Subsequent entries are
    pre-specified tie-breakers, not a secondary tuning objective.
    """
    return (
        float(row["mean_case_score"]),
        float(row.get("mean_reconstruction_score", -np.inf)),
        float(row.get("mean_structural_fidelity_score", -np.inf)),
        float(row.get("mean_contamination_resistance_score", -np.inf)),
        -float(row.get("c_H", row.get("H", np.inf))),
        -float(row.get("h1", np.inf)),
        -float(row.get("a", np.inf)),
        -float(row.get("h_min", np.inf)),
        -int(row.get("candidate_id", 10**12)),
    )


def _case_params(params, case):
    out = dict(params)
    if out.get("H_parameterization") != "relative_noise_scale":
        return out, {
            "H_parameterization": out.get("H_parameterization", "absolute"),
            "c_H_candidate": np.nan,
            "scale_hat_case": np.nan,
            "H_realized_case": float(out.get("H", np.nan)),
            "scale_estimator_status": None,
            "scale_estimator_fallback_used": False,
        }
    est = first_difference_mad_noise_scale(case["Y"])
    scale_hat = float(est["noise_scale_hat"])
    c_h = float(out["c_H"])
    h_realized = float(c_h * scale_hat) if np.isfinite(scale_hat) else np.nan
    out["H"] = h_realized
    out["loss_tuning"] = {"H": h_realized}
    return out, {
        "H_parameterization": "relative_noise_scale",
        "c_H_candidate": c_h,
        "scale_hat_case": scale_hat,
        "H_realized_case": h_realized,
        "scale_estimator_status": est["estimator_status"],
        "scale_estimator_fallback_used": bool(est["fallback_used"]),
    }


def _run_parameter_grid(
        stage,
        grid,
        output_rows,
        signals,
        noises,
        sigmas,
        base_params,
        n,
        fs,
        seed,
        global_candidate_offset=0,
        target_snr_db_levels=None,
        checkpoint_root=None,
):
    checkpoint_paths = _checkpoint_paths(checkpoint_root, stage) if checkpoint_root is not None else None
    completed_summaries = _read_jsonl(checkpoint_paths["summaries"]) if checkpoint_paths else []
    completed_stage_ids = {
        int(row["stage_candidate_id"])
        for row in completed_summaries
        if row.get("stage") == stage
    }
    if checkpoint_paths:
        completed_case_rows = [
            row for row in _read_jsonl(checkpoint_paths["case_rows"])
            if int(row.get("stage_candidate_id", -1)) in completed_stage_ids
        ]
        output_rows.extend(completed_case_rows)
        if completed_summaries:
            print(
                f"[parameter-selection] {stage}: resuming with "
                f"{len(completed_summaries)} completed candidates from checkpoint",
                flush=True,
            )

    candidate_summaries = list(completed_summaries)
    candidates = list(_candidate_params(grid, base_params))

    for stage_candidate_id, params in enumerate(candidates, start=1):
        global_candidate_id = global_candidate_offset + stage_candidate_id
        if stage_candidate_id in completed_stage_ids:
            continue
        if stage_candidate_id == 1 or stage_candidate_id % 25 == 0 or stage_candidate_id == len(candidates):
            print(
                f"[parameter-selection] {stage}: candidate "
                f"{stage_candidate_id}/{len(candidates)} "
                f"(global {global_candidate_id})",
                flush=True,
            )
        case_scores = []
        reconstruction_scores = []
        structural_scores = []
        contamination_scores = []
        n_failed_or_nonfinite_cases = 0
        candidate_rows = []
        candidate_started = time.time()

        for signal_name in signals:
            for noise_name in noises:
                severity_levels = (
                    tuple((10.0 ** (-float(snr) / 20.0), float(snr)) for snr in target_snr_db_levels)
                    if target_snr_db_levels is not None
                    else tuple((float(sigma), None) for sigma in sigmas)
                )
                for sigma, target_snr_db in severity_levels:
                    case = make_signal_noise_case(
                        signal_name=signal_name,
                        noise_name=noise_name,
                        sigma=sigma,
                        n=n,
                        fs=fs,
                        seed=seed,
                        target_snr_db=target_snr_db,
                    )
                    case_params, h_provenance = _case_params(params, case)
                    if (
                        h_provenance["H_parameterization"] == "relative_noise_scale"
                        and not np.isfinite(h_provenance["H_realized_case"])
                    ):
                        result = {"case_score": np.nan}
                        summary = {
                            "case_score": np.nan,
                            "reconstruction_score": np.nan,
                            "structural_fidelity_score": np.nan,
                            "contamination_resistance_score": np.nan,
                        }
                    else:
                        result = run_fixed_irmf_case(
                            Y=case["Y"],
                            X_clean=case["X_clean"],
                            t=case["t"],
                            fs=fs,
                            irmf_params=case_params,
                            expected_noise_ratio=case.get("expected_noise_ratio"),
                            true_components=case.get("true_components"),
                            run_id=f"{stage}_candidate_{stage_candidate_id}",
                            compute_operator_diagnostics_flag=False,
                        )
                        summary = method_result_summary(result)
                    row = {
                        "stage": stage,
                        "stage_candidate_id": stage_candidate_id,
                        "candidate_id": global_candidate_id,
                        "signal": signal_name,
                        "noise": noise_name,
                        "sigma": sigma,
                        "target_snr_db": target_snr_db,
                        "noise_severity_design": (
                            "target_snr_energy_ratio"
                            if target_snr_db is not None
                            else "fixed_sigma"
                        ),
                        "realized_input_snr_db": case.get("realized_input_snr_db"),
                        "noise_scale_alpha": case.get("noise_scale_alpha"),
                        **h_provenance,
                        **summary,
                    }
                    candidate_rows.append(row)

                    if np.isfinite(result.get("case_score", np.nan)):
                        case_scores.append(float(result["case_score"]))
                    else:
                        n_failed_or_nonfinite_cases += 1
                    if np.isfinite(result.get("reconstruction_score", np.nan)):
                        reconstruction_scores.append(float(result["reconstruction_score"]))
                    if np.isfinite(result.get("structural_fidelity_score", np.nan)):
                        structural_scores.append(float(result["structural_fidelity_score"]))
                    if np.isfinite(result.get("contamination_resistance_score", np.nan)):
                        contamination_scores.append(float(result["contamination_resistance_score"]))

        candidate_summary = {
            "stage": stage,
            "stage_candidate_id": stage_candidate_id,
            "candidate_id": global_candidate_id,
            **params,
            "c_H": params.get("c_H", np.nan),
            "H": params.get("H", np.nan),
            "mean_case_score": _summarize_scores(case_scores),
            "mean_reconstruction_score": _summarize_scores(reconstruction_scores),
            "mean_structural_fidelity_score": _summarize_scores(structural_scores),
            "mean_contamination_resistance_score": _summarize_scores(contamination_scores),
            "n_development_cases": int(len(case_scores)),
            "expected_development_cases": int(EXPECTED_DEVELOPMENT_CASE_COUNT),
            "n_failed_or_nonfinite_cases": int(n_failed_or_nonfinite_cases),
            "candidate_complete_for_selection": (
                int(len(case_scores)) >= int(MIN_COMPLETE_DEVELOPMENT_CASES)
            ),
            "candidate_runtime_seconds": float(time.time() - candidate_started),
            "selection_computation_protocol_version": SELECTION_COMPUTATION_PROTOCOL_VERSION,
            "checkpoint_status": "candidate_complete",
        }
        candidate_summaries.append(candidate_summary)
        output_rows.extend(candidate_rows)
        if checkpoint_paths:
            _append_jsonl(checkpoint_paths["case_rows"], candidate_rows)
            _append_jsonl(checkpoint_paths["summaries"], [candidate_summary])
        print(
            f"[parameter-selection] {stage}: completed candidate "
            f"{stage_candidate_id}/{len(candidates)} "
            f"mean_case_score={candidate_summary['mean_case_score']} "
            f"runtime={candidate_summary['candidate_runtime_seconds']:.1f}s",
            flush=True,
        )

    return candidate_summaries


def _select_best(candidate_summaries):
    valid = [
        r for r in candidate_summaries
        if (
            r["mean_case_score"] is not None
            and bool(r.get("candidate_complete_for_selection"))
        )
    ]
    return max(valid, key=_candidate_sort_key) if valid else None


def _selected_params(best, selection_criterion):
    if best is None:
        return None
    selected = {
        "h1": best["h1"],
        "a": best["a"],
        "h_min": best["h_min"],
        "boundary_mode": best["boundary_mode"],
        "min_support_points": best["min_support_points"],
        "selected_from_stage": best["stage"],
        "selected_from_stage_candidate_id": best["stage_candidate_id"],
        "selected_from_candidate_id": best["candidate_id"],
        "selection_criterion": selection_criterion,
    }
    if best.get("H_parameterization") == "relative_noise_scale":
        selected.update({
            "H_parameterization": "relative_noise_scale",
            "c_H": best["c_H"],
            "scale_estimator_id": IRMF_RELATIVE_H_SCALE_ESTIMATOR,
            "H": None,
            "H_realized_rule": "H_case = c_H * sigma_hat(Y_case)",
        })
    else:
        selected.update({
            "H_parameterization": "absolute",
            "H": best["H"],
        })
    return selected


def run_global_parameter_selection(
        output_root,
        signals=PARAMETER_SELECTION_SIGNALS,
        noises=PARAMETER_SELECTION_NOISES,
        sigmas=PARAMETER_SELECTION_SIGMAS,
        target_snr_db_levels=PARAMETER_SELECTION_TARGET_SNR_DB_LEVELS,
        coarse_grid=PAPER_IRMF_COARSE_PARAMETER_GRID,
        refinement_steps=PAPER_IRMF_REFINEMENT_STEPS,
        base_params=GLOBAL_IRMF_PARAMS,
        n=DEFAULT_N,
        fs=DEFAULT_FS,
        seed=DEFAULT_SEED,
):
    output_root = ensure_dir(output_root)
    target_snr_db_levels = (
        tuple(float(v) for v in target_snr_db_levels)
        if TARGET_SNR_DEVELOPMENT_DESIGN_ACTIVE
        else None
    )
    n_severity_levels = (
        len(target_snr_db_levels)
        if target_snr_db_levels is not None
        else len(sigmas)
    )
    n_development_cases = int(len(signals) * len(noises) * n_severity_levels)
    if n_development_cases != EXPECTED_DEVELOPMENT_CASE_COUNT:
        raise ValueError(
            "Section 4 development-set case count changed unexpectedly: "
            f"{len(signals)} signals × {len(noises)} noises × {n_severity_levels} severity levels "
            f"= {n_development_cases}, expected {EXPECTED_DEVELOPMENT_CASE_COUNT}. "
            "Update the protocol version and development-set manifest before rerunning."
        )
    rows = []
    if RELATIVE_H_PARAMETERIZATION_ACTIVE:
        coarse_grid = dict(coarse_grid)
        if "c_H_options" not in coarse_grid:
            coarse_grid["c_H_options"] = tuple(IRMF_RELATIVE_H_C_OPTIONS)
    checkpoint_root = _prepare_checkpoint_root(
        output_root / "v549_resumable_selection_checkpoints",
        _checkpoint_fingerprint(
            signals=signals,
            noises=noises,
            sigmas=sigmas,
            target_snr_db_levels=target_snr_db_levels,
            coarse_grid=coarse_grid,
            seed=seed,
            n=n,
            fs=fs,
        ),
    )

    stage1_summaries = _run_parameter_grid(
        stage="stage_1_coarse",
        grid=coarse_grid,
        output_rows=rows,
        signals=signals,
        noises=noises,
        sigmas=sigmas,
        base_params=base_params,
        n=n,
        fs=fs,
        seed=seed,
        global_candidate_offset=0,
        target_snr_db_levels=target_snr_db_levels,
        checkpoint_root=checkpoint_root,
    )
    best_stage1 = _select_best(stage1_summaries)

    if best_stage1 is None:
        refined_grid = None
        stage2_summaries = []
        best_stage2 = None
    else:
        refined_grid = build_refined_grid(best_stage1, coarse_grid, refinement_steps)
        stage2_summaries = _run_parameter_grid(
            stage="stage_2_refined",
            grid=refined_grid,
            output_rows=rows,
            signals=signals,
            noises=noises,
            sigmas=sigmas,
            base_params=base_params,
            n=n,
            fs=fs,
            seed=seed,
            global_candidate_offset=len(stage1_summaries),
            target_snr_db_levels=target_snr_db_levels,
            checkpoint_root=checkpoint_root,
        )
        best_stage2 = _select_best(stage2_summaries)

    candidate_summaries = stage1_summaries + stage2_summaries
    best_final = best_stage2 if best_stage2 is not None else best_stage1
    selected = _selected_params(
        best_final,
        selection_criterion=(
            "maximum mean development-set case_score within the V5.47 target-SNR relative-H "
            "four-parameter smoothed-median theory-constrained expanded-domain "
            "two-stage search design"
            if RELATIVE_H_PARAMETERIZATION_ACTIVE
            else "maximum mean development-set case_score within the V5.39 four-parameter smoothed-median theory-constrained expanded-domain two-stage search design"
        ),
    )

    protocol = {
        "protocol_version": (
            "V5.47 canonical-development-v2 60-case target-SNR relative-H smoothed-median four-parameter two-stage parameter selection"
            if RELATIVE_H_PARAMETERIZATION_ACTIVE
            else "V5.39 canonical-development-v2 40-case smoothed-median four-parameter two-stage parameter selection"
        ),
        "selection_computation_protocol": {
            "protocol_version": SELECTION_COMPUTATION_PROTOCOL_VERSION,
            "checkpoint_root": str(checkpoint_root),
            "candidate_completion_policy": (
                "Each candidate is checkpointed only after all required development "
                "cases complete; interrupted partial candidates are rerun."
            ),
            "scientific_design_changed": False,
            "selection_objective_changed": False,
            "parallel_execution_active": False,
            "progress_reporting_unit": "candidate",
        },
        "non_overwrite_policy": (
            "The default paper output root is IRMF_EMD_PAPER_RESULTS_V4E_SMOOTHED_MEDIAN_H_FIXED_THEORY_CONSTRAINED, "
            "so earlier V1/V2/V3/V4B/V4C/V4D result directories are not overwritten."
        ),
        "robust_loss": {
            "name": "Gaussian-smoothed median loss",
            "formula": "rho_H(x) = sqrt(2/pi) H exp(-x^2/(2H^2)) + x [2 Phi(x/H) - 1]",
            "gradient": "rho'_H(x) = 2 Phi(x/H) - 1",
            "hessian": "rho''_H(x) = (2/H) phi(x/H)",
            "note": (
                "V5.47 selects a single global dimensionless c_H and derives "
                "case-specific H mechanically from a frozen truth-free scale proxy."
                if RELATIVE_H_PARAMETERIZATION_ACTIVE
                else "V5.39 treats H as a development-set hyperparameter rather than an untuned fixed constant."
            ),
            "H_options": list(coarse_grid.get("H_options", ())),
            "H_tuned": not RELATIVE_H_PARAMETERIZATION_ACTIVE,
            "c_H_options": list(coarse_grid.get("c_H_options", ())),
            "c_H_tuned": bool(RELATIVE_H_PARAMETERIZATION_ACTIVE),
            "H_parameterization": (
                "relative_noise_scale"
                if RELATIVE_H_PARAMETERIZATION_ACTIVE
                else "absolute"
            ),
            "relative_H_schema_version": (
                IRMF_RELATIVE_H_PARAMETERIZATION_VERSION
                if RELATIVE_H_PARAMETERIZATION_ACTIVE
                else None
            ),
            "scale_estimator_id": (
                IRMF_RELATIVE_H_SCALE_ESTIMATOR
                if RELATIVE_H_PARAMETERIZATION_ACTIVE
                else None
            ),
        },
        "development_set": {
            "design": "pre-specified 60-case target-SNR stratified canonical development set",
            "selection_dataset": "canonical_development_v2_60case_target_snr",
            "formula_instance_policy": "class-overlap allowed; development waveforms use distinct formula instances from the main benchmark",
            "generalization_role": (
                "This tier selects global parameters only.  The canonical benchmark "
                "is a held-out formula-instance evaluation within the canonical signal "
                "taxonomy, while the challenging suite is an out-of-development "
                "structural generalization benchmark."
            ),
            "excluded_from_selection": {
                "canonical_main_benchmark_results": True,
                "challenging_signal_suite_results": True,
                "emd_family_final_rankings": True,
                "failure_case_plots": True,
            },
            "signals": list(signals),
            "noises": list(noises),
            "sigmas": list(sigmas),
            "n_development_cases": n_development_cases,
            "n": n,
            "fs": fs,
            "seed": seed,
            "severity_design_version": (
                PARAMETER_SELECTION_SEVERITY_DESIGN_VERSION
                if target_snr_db_levels is not None
                else "legacy_fixed_sigma_development_severity"
            ),
            "severity_axis": (
                "target_snr_db"
                if target_snr_db_levels is not None
                else "fixed_sigma"
            ),
            "target_snr_db_levels": (
                list(target_snr_db_levels)
                if target_snr_db_levels is not None
                else None
            ),
        },
        "stage_1_coarse_grid": coarse_grid,
        "stage_1_domain_note": (
            "V5.39 keeps the expanded-domain search for h1 and h_min because the "
            "balanced-score V3 search selected boundary values.  Unlike the exploratory "
            "V4B search, V5.39 constrains the geometric scale factor to a in (1, 2], "
            "consistent with the Spokoiny-type multiscale filtering construction. "
            "V5.39 also promotes the Gaussian-smoothed median loss width H into "
            "the development-selection grid after auditing that H acts directly "
            "on local residual scale. "
            "This theory-constrained domain is used before locking the global IRMF "
            "parameter set."
        ),
        "H_tuning_protocol": {
            "rationale": (
                "H controls the Gaussian-smoothed-median robust contrast scale. "
                "Under V5.47, one global c_H is selected with h1, a, and h_min, "
                "and each case's H is then derived as H_case=c_H*sigma_hat(Y_case)."
                if RELATIVE_H_PARAMETERIZATION_ACTIVE
                else "H controls the Gaussian-smoothed-median robust contrast scale and therefore belongs to the common development-set parameter selection with h1, a, and h_min."
            ),
            "stage_1_levels": list(
                coarse_grid["c_H_options"]
                if RELATIVE_H_PARAMETERIZATION_ACTIVE
                else coarse_grid["H_options"]
            ),
            "stage_2_policy": (
                "c_H is refined deterministically around the Stage-I best level."
                if RELATIVE_H_PARAMETERIZATION_ACTIVE
                else "H is refined deterministically around the Stage-I best level."
            ),
            "stage_2_refinement_step": (
                refinement_steps["c_H"]
                if RELATIVE_H_PARAMETERIZATION_ACTIVE
                else refinement_steps["H"]
            ),
            "no_per_case_tuning": True,
            "global_tuned_quantity": (
                "c_H" if RELATIVE_H_PARAMETERIZATION_ACTIVE else "H"
            ),
            "case_specific_derived_quantity": (
                "H_case" if RELATIVE_H_PARAMETERIZATION_ACTIVE else None
            ),
            "provenance_fields_archived": [
                "c_H_candidate",
                "scale_hat_case",
                "H_realized_case",
                "scale_estimator_status",
                "scale_estimator_fallback_used",
            ] if RELATIVE_H_PARAMETERIZATION_ACTIVE else [],
        },
        "selection_objective": {
            "primary_objective": f"maximize mean_case_score across the {n_development_cases} frozen development cases",
            "objective_reused_from_prior_selection": True,
            "H_added_to_candidate_grid_without_changing_objective": not RELATIVE_H_PARAMETERIZATION_ACTIVE,
            "c_H_replaces_absolute_H_without_changing_objective": bool(RELATIVE_H_PARAMETERIZATION_ACTIVE),
            "case_score_formula": "0.40 * reconstruction_score + 0.35 * structural_fidelity_score + 0.25 * contamination_resistance_score",
            "cross_case_aggregation_rule": "arithmetic mean over all finite case_score values, with candidate eligibility requiring all 60 cases finite",
            "candidate_eligibility_rule": (
                "A candidate is selectable only when all 60 development cases "
                "produce finite case_score values; incomplete candidates remain "
                "archived but are excluded from winner selection."
            ),
            "failed_candidate_handling": (
                "Any non-finite case_score contributes to n_failed_or_nonfinite_cases "
                "and makes candidate_complete_for_selection=false."
            ),
            "tie_breaking_rule": [
                "larger mean_case_score",
                "larger mean_reconstruction_score",
                "larger mean_structural_fidelity_score",
                "larger mean_contamination_resistance_score",
                "smaller c_H" if RELATIVE_H_PARAMETERIZATION_ACTIVE else "smaller H",
                "smaller h1",
                "smaller a",
                "smaller h_min",
                "smaller candidate_id",
            ],
            "tie_breaking_role": "deterministic reproducibility only; not a secondary scientific endpoint",
        },
        "H_absolute_scale_diagnostic_plan": {
            "H_scale_semantics": (
                "case_derived_relative_noise_scale_parameter"
                if RELATIVE_H_PARAMETERIZATION_ACTIVE
                else "absolute_residual_scale_parameter"
            ),
            "diagnostic_only_not_used_for_selection": True,
            "question": (
                "Does the selected global c_H show systematic underperformance "
                "within development signal/noise/SNR scale strata?"
                if RELATIVE_H_PARAMETERIZATION_ACTIVE
                else "Does the selected global H show systematic underperformance within development signal-amplitude/RMS strata?"
            ),
            "planned_outputs": [
                "selected_c_H_performance_by_signal_noise_stratum"
                if RELATIVE_H_PARAMETERIZATION_ACTIVE else "selected_H_performance_by_signal_rms_stratum",
                "candidate_c_H_score_profiles_by_signal_family"
                if RELATIVE_H_PARAMETERIZATION_ACTIVE else "candidate_H_score_profiles_by_signal_family",
                "finite_case_score_rate_by_c_H_level"
                if RELATIVE_H_PARAMETERIZATION_ACTIVE else "finite_case_score_rate_by_H_level",
            ],
            "claim_boundary": (
                "This diagnostic evaluates whether one global c_H is acceptable; "
                "case-specific H_case is mechanically derived and is not optimized."
                if RELATIVE_H_PARAMETERIZATION_ACTIVE
                else "This diagnostic evaluates whether one absolute H is acceptable as a global locked parameter; it does not authorize case-wise H."
            ),
        },
        "a_theory_constraint": {
            "range": "(1, 2]",
            "stage_1_levels": list(coarse_grid["a_options"]),
            "stage_2_refinement_step": refinement_steps["a"],
            "rationale": "Keep the geometric scale progression within the Spokoiny-consistent admissible range.",
        },
        "stage_1_n_candidates": len(stage1_summaries),
        "stage_1_best": best_stage1,
        "stage_2_refinement_rule": {
            "description": "three-level deterministic refinement around the Stage-I best value; boundary optima are completed inward",
            "steps": refinement_steps,
            "global_optimality_note": (
                "The procedure selects the best configuration inside a finite pre-specified "
                "search design, rather than claiming global optimality over the continuous "
                "parameter space."
            ),
        },
        "stage_2_refined_grid": refined_grid,
        "stage_2_n_candidates": len(stage2_summaries),
        "stage_2_best": best_stage2,
        "selected_global_irmf_params": selected,
        "case_score_formula": {
            "case_score": "0.40 * reconstruction_score + 0.35 * structural_fidelity_score + 0.25 * contamination_resistance_score",
            "structural_fidelity_score": "0.30 * IMF recovery + 0.25 * orthogonality/leakage + 0.25 * frequency separation + 0.20 * local structure preservation, then OD/UD penalty",
            "contamination_resistance_score": "0.60 * outlier_resistance_index + 0.40 * noise_capture_corr_score",
            "all_components": "larger is better after score normalization",
        },
    }

    write_json(protocol, output_root / "section_4_parameter_selection_protocol.json")
    write_json(candidate_summaries, output_root / "section_4_candidate_summary.json")
    write_csv(candidate_summaries, output_root / "section_4_candidate_summary.csv")
    write_json(stage1_summaries, output_root / "section_4_stage_1_coarse_summary.json")
    write_csv(stage1_summaries, output_root / "section_4_stage_1_coarse_summary.csv")
    write_json(stage2_summaries, output_root / "section_4_stage_2_refined_summary.json")
    write_csv(stage2_summaries, output_root / "section_4_stage_2_refined_summary.csv")
    write_json(rows, output_root / "section_4_candidate_case_results.json")
    write_csv(rows, output_root / "section_4_candidate_case_results.csv")
    return selected, candidate_summaries, rows


if __name__ == "__main__":
    run_global_parameter_selection(Path(PAPER_OUTPUT_ROOT_NAME) / "section_4_global_parameter_selection")
