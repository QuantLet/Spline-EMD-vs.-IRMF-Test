#!/usr/bin/python
# coding: UTF-8
"""Parameter-transfer diagnostics for the fixed-parameter protocol.

These diagnostics turn generalization from an implicit chapter structure into
explicit, auditable experimental attributes:

1. development-test independence matrix;
2. leave-one-development-family-out selection stability;
3. held-out development-family transfer performance.
"""
from pathlib import Path
import math

import numpy as np

from project_config import (
    DEFAULT_FS,
    DEFAULT_N,
    DEFAULT_SEED,
    GLOBAL_IRMF_PARAMS,
    PARAMETER_SELECTION_NOISES,
    PARAMETER_SELECTION_SIGMAS,
    PARAMETER_SELECTION_SIGNALS,
)
from experiments.experiment_global_parameter_selection import run_global_parameter_selection
from experiments.experiment_utils import (
    make_signal_noise_case,
    method_result_summary,
    run_fixed_irmf_case,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


PARAMETER_KEYS = ("h1", "a", "h_min", "H")
LOWER_IS_BETTER = {
    "denoise_nmse",
    "component_splitting_index",
    "component_merging_index",
    "inter_imf_entanglement_index",
    "mode_mixing_index",
    "imf_recovery_nrmse",
    "decomposition_count_error",
}


def _base_family(signal_name):
    return str(signal_name).removesuffix("_dev")


def _development_test_independence_rows():
    return [
        {
            "evidence_layer": "Canonical benchmark",
            "new_realization": True,
            "new_formula_instance": True,
            "new_signal_family": "partial",
            "new_noise_model": "partial",
            "new_sigma_level": "partial",
            "new_domain": False,
            "interpretation": (
                "Tests held-out formula instances inside the canonical taxonomy; "
                "frequency_jump and intermittent_oscillation are fully held-out "
                "canonical families relative to the 40-case development set."
            ),
        },
        {
            "evidence_layer": "Signal-family reproducibility",
            "new_realization": True,
            "new_formula_instance": True,
            "new_signal_family": False,
            "new_noise_model": "design-dependent",
            "new_sigma_level": "design-dependent",
            "new_domain": False,
            "interpretation": (
                "Tests whether conclusions reproduce across unseen parameterizations "
                "within canonical signal families."
            ),
        },
        {
            "evidence_layer": "Robustness and protocol sensitivity",
            "new_realization": True,
            "new_formula_instance": "design-dependent",
            "new_signal_family": False,
            "new_noise_model": "partial",
            "new_sigma_level": True,
            "new_domain": False,
            "interpretation": (
                "Tests degradation trajectories and perturbation stability beyond "
                "the development-set operating points."
            ),
        },
        {
            "evidence_layer": "Challenging structural generalization",
            "new_realization": True,
            "new_formula_instance": True,
            "new_signal_family": True,
            "new_noise_model": "partial/full",
            "new_sigma_level": True,
            "new_domain": False,
            "interpretation": (
                "Tests out-of-development structural regimes outside the canonical "
                "signal taxonomy."
            ),
        },
        {
            "evidence_layer": "External real-data validation",
            "new_realization": True,
            "new_formula_instance": True,
            "new_signal_family": True,
            "new_noise_model": True,
            "new_sigma_level": True,
            "new_domain": True,
            "interpretation": (
                "Applies simulation-selected parameters to real data without "
                "truth-dependent retuning."
            ),
        },
    ]


def write_development_test_independence_matrix(output_root):
    output_root = ensure_dir(output_root)
    rows = _development_test_independence_rows()
    write_json(rows, output_root / "development_test_independence_matrix.json")
    write_csv(rows, output_root / "development_test_independence_matrix.csv")
    write_json({
        "purpose": (
            "Make explicit what information was unseen by the development set "
            "at each evidence layer."
        ),
        "development_set": {
            "signals": list(PARAMETER_SELECTION_SIGNALS),
            "noises": list(PARAMETER_SELECTION_NOISES),
            "sigmas": list(PARAMETER_SELECTION_SIGMAS),
            "n_cases": int(
                len(PARAMETER_SELECTION_SIGNALS)
                * len(PARAMETER_SELECTION_NOISES)
                * len(PARAMETER_SELECTION_SIGMAS)
            ),
        },
        "matrix": rows,
    }, output_root / "development_test_independence_dashboard.json")
    return rows


def _finite(value):
    try:
        return value is not None and np.isfinite(float(value))
    except Exception:
        return False


def _param_distance(selected, reference):
    values = {}
    squared = 0.0
    for key in PARAMETER_KEYS:
        sv = selected.get(key)
        rv = reference.get(key)
        if not (_finite(sv) and _finite(rv)):
            values[f"abs_delta_{key}"] = np.nan
            values[f"relative_delta_{key}"] = np.nan
            continue
        sv = float(sv)
        rv = float(rv)
        delta = abs(sv - rv)
        values[f"abs_delta_{key}"] = float(delta)
        values[f"relative_delta_{key}"] = float(delta / (abs(rv) + 1e-12))
        squared += values[f"relative_delta_{key}"] ** 2
    values["relative_l2_parameter_distance"] = float(math.sqrt(squared))
    return values


def _configuration_signature(params):
    return (
        f"h1={float(params.get('h1', np.nan)):.6g};"
        f"a={float(params.get('a', np.nan)):.6g};"
        f"h_min={float(params.get('h_min', np.nan)):.6g};"
        f"H={float(params.get('H', np.nan)):.6g}"
    )


def _metric_advantage(metric, candidate_value, reference_value):
    """Positive means the candidate improves over the reference."""
    if not (_finite(candidate_value) and _finite(reference_value)):
        return np.nan
    candidate_value = float(candidate_value)
    reference_value = float(reference_value)
    if metric in LOWER_IS_BETTER:
        return reference_value - candidate_value
    return candidate_value - reference_value


def _metric_regret(metric, candidate_value, reference_value):
    """Positive means the candidate is worse than the reference."""
    advantage = _metric_advantage(metric, candidate_value, reference_value)
    return -float(advantage) if _finite(advantage) else np.nan


def _evaluate_params_on_family(params, signal_name, noises, sigmas, n, fs, seed, label):
    rows = []
    for noise_name in noises:
        for sigma in sigmas:
            case = make_signal_noise_case(
                signal_name=signal_name,
                noise_name=noise_name,
                sigma=sigma,
                n=n,
                fs=fs,
                seed=seed,
            )
            result = run_fixed_irmf_case(
                Y=case["Y"],
                X_clean=case["X_clean"],
                t=case["t"],
                fs=fs,
                irmf_params=params,
                expected_noise_ratio=case.get("expected_noise_ratio"),
                true_components=case.get("true_components"),
                run_id=f"{label}_{signal_name}_{noise_name}_{sigma}",
            )
            rows.append({
                "evaluation_label": label,
                "signal": signal_name,
                "noise": noise_name,
                "sigma": sigma,
                **method_result_summary(result),
            })
    return rows


def _aggregate_metric(rows, metric):
    vals = [float(row[metric]) for row in rows if _finite(row.get(metric))]
    return float(np.mean(vals)) if vals else np.nan


def run_leave_one_development_family_out_analysis(
        output_root,
        reference_params=None,
        signals=PARAMETER_SELECTION_SIGNALS,
        noises=PARAMETER_SELECTION_NOISES,
        sigmas=PARAMETER_SELECTION_SIGMAS,
        n=DEFAULT_N,
        fs=DEFAULT_FS,
        seed=DEFAULT_SEED,
):
    output_root = ensure_dir(output_root)
    reference_params = dict(reference_params or GLOBAL_IRMF_PARAMS)
    summary_rows = []
    eval_rows = []

    for excluded_signal in signals:
        train_signals = tuple(signal for signal in signals if signal != excluded_signal)
        excluded_family = _base_family(excluded_signal)
        split_root = ensure_dir(output_root / f"exclude_{excluded_family}")
        selected, candidate_summaries, _ = run_global_parameter_selection(
            split_root,
            signals=train_signals,
            noises=noises,
            sigmas=sigmas,
            n=n,
            fs=fs,
            seed=seed,
        )
        if selected is None:
            continue

        selected_eval = _evaluate_params_on_family(
            selected,
            excluded_signal,
            noises=noises,
            sigmas=sigmas,
            n=n,
            fs=fs,
            seed=seed,
            label="lofo_selected",
        )
        reference_eval = _evaluate_params_on_family(
            reference_params,
            excluded_signal,
            noises=noises,
            sigmas=sigmas,
            n=n,
            fs=fs,
            seed=seed,
            label="full_development_reference",
        )
        eval_rows.extend({
            **row,
            "excluded_development_signal": excluded_signal,
            "excluded_family": excluded_family,
        } for row in selected_eval + reference_eval)

        metrics = (
            "case_score",
            "denoise_nmse",
            "denoise_corr",
            "imf_recovery_corr",
            "component_splitting_index",
            "component_merging_index",
            "contamination_resistance_score",
        )
        row = {
            "excluded_development_signal": excluded_signal,
            "excluded_family": excluded_family,
            "training_signals": list(train_signals),
            "n_training_cases": int(len(train_signals) * len(noises) * len(sigmas)),
            "n_candidates": int(len(candidate_summaries)),
            "selected_configuration_id": _configuration_signature(selected),
            "reference_configuration_id": _configuration_signature(reference_params),
            **{f"selected_{key}": selected.get(key) for key in PARAMETER_KEYS},
            **_param_distance(selected, reference_params),
        }
        for metric in metrics:
            lofo_mean = _aggregate_metric(selected_eval, metric)
            ref_mean = _aggregate_metric(reference_eval, metric)
            row[f"lofo_mean_{metric}"] = lofo_mean
            row[f"reference_mean_{metric}"] = ref_mean
            row[f"lofo_advantage_over_reference_{metric}"] = _metric_advantage(
                metric, lofo_mean, ref_mean
            )
            row[f"lofo_to_full_regret_{metric}"] = _metric_regret(
                metric, lofo_mean, ref_mean
            )
            row[f"lofo_to_oracle_regret_{metric}"] = None
            row[f"full_to_oracle_regret_{metric}"] = None
        summary_rows.append(row)

    write_json(summary_rows, output_root / "leave_one_development_family_out_summary.json")
    write_csv(summary_rows, output_root / "leave_one_development_family_out_summary.csv")
    write_json(eval_rows, output_root / "leave_one_development_family_out_case_results.json")
    write_csv(eval_rows, output_root / "leave_one_development_family_out_case_results.csv")

    dashboard = {
        "analysis": "leave_one_development_family_out_selection_stability",
        "purpose": (
            "Assess whether the selected global IRMF parameters depend strongly "
            "on a single development signal family."
        ),
        "reference_params": reference_params,
        "n_exclusions": len(summary_rows),
        "summary": summary_rows,
        "reported_diagnostics": [
            "selected configuration after excluding each development family",
            "distance from the full-development reference configuration",
            "held-out performance of the LOFO-selected configuration",
            "held-out performance of the full-development reference configuration",
            "LOFO-to-full regret with metric direction normalized",
            "oracle-regret placeholder fields for optional Appendix B integration",
        ],
        "oracle_regret_status": (
            "not_computed_in_this_stage; merge Appendix B oracle-adaptivity "
            "outputs if lofo-to-oracle and full-to-oracle regret are needed"
        ),
        "interpretation_rule": (
            "Changed selected parameters do not automatically imply instability. "
            "The central diagnostic is whether held-out performance remains close "
            "to the full-development reference configuration."
        ),
        "interpretation_limits": (
            "This is a development-family exclusion diagnostic, not a proof of "
            "universal parameter independence."
        ),
    }
    write_json(dashboard, output_root / "parameter_selection_stability_dashboard.json")
    return dashboard


def run_parameter_transfer_analysis(output_root, reference_params=None):
    output_root = ensure_dir(output_root)
    matrix = write_development_test_independence_matrix(output_root / "01_development_test_independence")
    lofo = run_leave_one_development_family_out_analysis(
        output_root / "02_leave_one_development_family_out",
        reference_params=reference_params,
    )
    dashboard = {
        "analysis": "parameter_transfer_and_generalization_diagnostics",
        "development_test_independence": matrix,
        "leave_one_development_family_out": lofo,
        "paper_claim_guidance": {
            "supported": [
                "development-test independence is explicitly documented",
                "local dependence on individual development families is quantified",
                "fixed-to-oracle regret should be interpreted as an adaptivity-gap diagnostic",
            ],
            "avoid_without_extra_experiments": [
                "universal parameter optimality",
                "complete scale-transfer generalization",
                "parameter independence from all possible development-set compositions",
            ],
        },
    }
    write_json(dashboard, output_root / "parameter_transfer_dashboard.json")
    return dashboard
