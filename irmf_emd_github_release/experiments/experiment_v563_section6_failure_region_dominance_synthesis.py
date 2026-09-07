#!/usr/bin/python
# coding: UTF-8

"""V5.63 cross-analysis failure-region and dominance synthesis.

This stage does not run algorithms.  It summarizes where the Section 5/6
comparative conclusions are preserved, attenuated, reversed, or not
identifiable from existing artifacts.
"""

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


V563_FAILURE_REGION_SYNTHESIS_VERSION = (
    "V5.63_section6_failure_region_and_dominance_synthesis"
)

LOWER_IS_BETTER = {
    "denoise_nmse",
    "imf_recovery_nrmse",
    "matched_nrmse",
    "clean_region_nmse",
    "contaminated_region_nmse",
    "normalized_contamination_spillover_loss",
    "noise_energy_log_error",
    "signal_leakage_into_noise",
    "decomposition_count_error",
    "relative_decomposition_count_error",
}
HIGHER_IS_BETTER = {
    "denoise_corr",
    "imf_recovery_corr",
    "matched_corr",
    "noise_capture_corr",
}


def _read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(value, default=np.nan):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _finite(value):
    value = _float(value)
    return value if np.isfinite(value) else None


def _fmt(value, digits=4):
    value = _finite(value)
    if value is None:
        return "NA"
    return f"{value:.{digits}g}"


def _metric_direction(metric):
    if metric in HIGHER_IS_BETTER:
        return "higher_is_better"
    if metric in LOWER_IS_BETTER:
        return "lower_is_better"
    return "unknown"


def _irmf_benefit(row):
    """Return positive values when IRMF beats the row's baseline."""
    diff = _finite(row.get("median_paired_difference"))
    if diff is None:
        diff = _finite(row.get("mean_paired_difference"))
    if diff is None:
        return None
    method_a = row.get("method_a")
    method_b = row.get("method_b")
    metric = row.get("metric")
    direction = _metric_direction(metric)
    if direction == "unknown" or "IRMF" not in {method_a, method_b}:
        return None
    # paired difference is method_a - method_b.
    irmf_minus_baseline = diff if method_a == "IRMF" else -diff
    if direction == "higher_is_better":
        return irmf_minus_baseline
    return -irmf_minus_baseline


def _status_from_loss_fraction(loss_fraction, median_benefit=None):
    if loss_fraction is None or not np.isfinite(loss_fraction):
        return "not_identifiable"
    if loss_fraction <= 0.10:
        return "preserved"
    if loss_fraction <= 0.35:
        return "attenuated"
    if median_benefit is not None and np.isfinite(median_benefit) and median_benefit < 0:
        return "reversed"
    return "attenuated_or_mixed"


def _paired_failure_summary(rows, source_label, metrics):
    grouped = defaultdict(list)
    for row in rows:
        metric = row.get("metric")
        if metric not in metrics:
            continue
        if "IRMF" not in {row.get("method_a"), row.get("method_b")}:
            continue
        baseline = row.get("method_b") if row.get("method_a") == "IRMF" else row.get("method_a")
        benefit = _irmf_benefit(row)
        if benefit is None:
            continue
        grouped[(metric, baseline)].append((benefit, row))

    out = []
    for (metric, baseline), vals in sorted(grouped.items()):
        benefits = np.asarray([v[0] for v in vals], dtype=float)
        loss_fraction = float(np.mean(benefits < 0.0)) if benefits.size else np.nan
        median_benefit = float(np.median(benefits)) if benefits.size else np.nan
        worst_decile = float(np.quantile(benefits, 0.10)) if benefits.size else np.nan
        worst = sorted(vals, key=lambda item: item[0])[:3]
        conditions = []
        for _, row in worst:
            conditions.append(
                "signal={signal}, noise={noise}, snr_or_sigma={sigma}, regime={signal_regime}".format(
                    signal=row.get("signal", "NA"),
                    noise=row.get("noise", "NA"),
                    sigma=row.get("target_snr_db", row.get("sigma", "NA")),
                    signal_regime=row.get("signal_regime", "NA"),
                )
            )
        out.append({
            "analysis_source": source_label,
            "metric": metric,
            "baseline": baseline,
            "direction": _metric_direction(metric),
            "n_condition_summaries": int(benefits.size),
            "median_irmf_benefit": median_benefit,
            "loss_fraction": loss_fraction,
            "worst_decile_irmf_benefit": worst_decile,
            "status": _status_from_loss_fraction(loss_fraction, median_benefit),
            "worst_conditions": " | ".join(conditions),
        })
    return out


def _contamination_failure_summary(rows):
    out = []
    grouped = defaultdict(list)
    for row in rows:
        metric = row.get("metric")
        if metric not in {
            "contaminated_region_nmse",
            "clean_region_nmse",
            "normalized_contamination_spillover_loss",
        }:
            continue
        baseline = row.get("baseline")
        win = str(row.get("irmf_win")).lower() == "true"
        diff = _finite(row.get("irmf_minus_baseline"))
        grouped[(metric, baseline)].append((win, diff, row))
    for (metric, baseline), vals in sorted(grouped.items()):
        wins = np.asarray([float(v[0]) for v in vals], dtype=float)
        benefit = []
        for _, diff, _ in vals:
            if diff is not None:
                benefit.append(-diff)  # all three contamination endpoints are lower-is-better.
        benefits = np.asarray(benefit, dtype=float)
        loss_fraction = float(1.0 - np.mean(wins)) if wins.size else np.nan
        median_benefit = float(np.median(benefits)) if benefits.size else np.nan
        worst_decile = float(np.quantile(benefits, 0.10)) if benefits.size else np.nan
        losses = [row for win, _, row in vals if not win][:5]
        conditions = [
            "axis={axis}, signal={signal}, lambda={lam}, kappa={kappa}, geometry={geo}, variance={var}".format(
                axis=row.get("axis", "NA"),
                signal=row.get("signal", "NA"),
                lam=row.get("lambda_target", "NA"),
                kappa=row.get("kappa", "NA"),
                geo=row.get("position_geometry", "NA"),
                var=row.get("variance_comparison", "NA"),
            )
            for row in losses
        ]
        out.append({
            "analysis_source": "6.2 contamination design",
            "metric": metric,
            "baseline": baseline,
            "direction": "lower_is_better",
            "n_condition_summaries": int(len(vals)),
            "median_irmf_benefit": median_benefit,
            "loss_fraction": loss_fraction,
            "worst_decile_irmf_benefit": worst_decile,
            "status": _status_from_loss_fraction(loss_fraction, median_benefit),
            "worst_conditions": " | ".join(conditions) if conditions else "no observed IRMF-loss cells",
        })
    return out


def _parameter_boundary_rows(rows):
    out = []
    for row in rows:
        metric = row.get("primary_endpoint")
        if not metric:
            continue
        regret = _finite(row.get("default_oracle_regret"))
        plateau = _finite(row.get("plateau_width_fraction_5pct_of_best"))
        interaction = str(row.get("interaction_materiality_flag")).lower() == "true"
        if plateau is None:
            status = "not_identifiable"
        elif plateau >= 0.50 and not interaction:
            status = "preserved"
        elif plateau >= 0.10:
            status = "attenuated"
        else:
            status = "boundary_sensitive"
        out.append({
            "analysis_source": "6.1 parameter sensitivity",
            "metric": metric,
            "baseline": "IRMF local parameter grid",
            "direction": row.get("direction", "NA"),
            "n_condition_summaries": int(_float(row.get("n_case_blocks"), 0)),
            "median_irmf_benefit": np.nan,
            "loss_fraction": np.nan,
            "worst_decile_irmf_benefit": np.nan,
            "status": status,
            "default_oracle_regret": regret,
            "plateau_width_fraction_5pct_of_best": plateau,
            "strongest_main_effect": row.get("strongest_main_effect", "NA"),
            "strongest_2way_interaction": row.get("strongest_2way_interaction", "NA"),
            "worst_conditions": (
                "parameter local grid; strongest main effect="
                f"{row.get('strongest_main_effect', 'NA')}; strongest interaction="
                f"{row.get('strongest_2way_interaction', 'NA')}"
            ),
        })
    return out


def _scaling_boundary_rows(algorithm_root):
    beta_rows = _read_csv(
        algorithm_root
        / "16u_v552_section6_4_computational_scaling"
        / "v552_runtime_scaling_exponents.csv"
    )
    runtime_rows = _read_csv(
        algorithm_root
        / "16u_v552_section6_4_computational_scaling"
        / "v552_runtime_by_method_and_n.csv"
    )
    beta = {row.get("method"): _finite(row.get("empirical_scaling_exponent_beta")) for row in beta_rows}
    at_4000 = {
        row.get("method"): _finite(row.get("median_runtime_seconds"))
        for row in runtime_rows if row.get("n") == "4000"
    }
    irmf_beta = beta.get("IRMF")
    max_other_beta = max([v for m, v in beta.items() if m != "IRMF" and v is not None], default=np.nan)
    status = "attenuated" if irmf_beta is not None and irmf_beta > max_other_beta else "preserved"
    return [{
        "analysis_source": "6.4 computational scaling",
        "metric": "runtime_seconds",
        "baseline": "EMD/EEMD/CEEMDAN",
        "direction": "lower_is_better",
        "n_condition_summaries": len(runtime_rows),
        "median_irmf_benefit": np.nan,
        "loss_fraction": np.nan,
        "worst_decile_irmf_benefit": np.nan,
        "status": status,
        "worst_conditions": (
            f"IRMF empirical beta={_fmt(irmf_beta)} vs max baseline beta={_fmt(max_other_beta)}; "
            f"n=4000 runtime IRMF={_fmt(at_4000.get('IRMF'))}s"
        ),
    }]


def _matching_boundary_rows(algorithm_root):
    preservation = _read_csv(
        algorithm_root
        / "16ae_v561_section6_5_matching_rule_robustness"
        / "v561_section6_5_matching_rule_robustness_preservation.csv"
    )
    out = []
    grouped = defaultdict(list)
    for row in preservation:
        grouped[row.get("metric")].append(row)
    for metric, vals in sorted(grouped.items()):
        flags = [str(v.get("effect_direction_preserved_all_irmf_vs_baselines")).lower() == "true" for v in vals]
        exact = [str(v.get("exact_order_preserved")).lower() == "true" for v in vals]
        taus = [_finite(v.get("tau")) for v in vals]
        high_tau_losses = [
            v for v in vals
            if _finite(v.get("tau")) is not None
            and _finite(v.get("tau")) >= 0.7
            and str(v.get("effect_direction_preserved_all_irmf_vs_baselines")).lower() != "true"
        ]
        preserve_rate = float(np.mean(flags)) if flags else np.nan
        exact_rate = float(np.mean(exact)) if exact else np.nan
        status = "preserved" if preserve_rate >= 0.80 else ("attenuated" if preserve_rate >= 0.50 else "reversed_or_rule_sensitive")
        out.append({
            "analysis_source": "6.5 matching-rule robustness",
            "metric": metric,
            "baseline": "alternative correspondence rules",
            "direction": "rule_preservation",
            "n_condition_summaries": len(vals),
            "median_irmf_benefit": np.nan,
            "loss_fraction": float(1.0 - preserve_rate) if np.isfinite(preserve_rate) else np.nan,
            "worst_decile_irmf_benefit": np.nan,
            "status": status,
            "effect_direction_preserved_rate": preserve_rate,
            "exact_order_preserved_rate": exact_rate,
            "worst_conditions": (
                "high-threshold or conservative matching restrictions: "
                + ", ".join(
                    f"{v.get('alternative_rule')} tau={v.get('tau')}"
                    for v in high_tau_losses[:5]
                )
                if high_tau_losses else "no preservation failure recorded"
            ),
        })
    return out


def _summarize_boundary(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row.get("analysis_source")].append(row)
    summary = []
    for source, vals in sorted(buckets.items()):
        statuses = defaultdict(int)
        for row in vals:
            statuses[row.get("status", "unknown")] += 1
        loss_values = [
            _finite(row.get("loss_fraction"))
            for row in vals
            if _finite(row.get("loss_fraction")) is not None
        ]
        summary.append({
            "analysis_source": source,
            "n_rows": len(vals),
            "status_counts_json": json.dumps(dict(statuses), sort_keys=True),
            "max_loss_fraction": max(loss_values) if loss_values else np.nan,
            "boundary_interpretation": _interpret_source(source, statuses),
        })
    return summary


def _interpret_source(source, statuses):
    if source.startswith("6.1"):
        return (
            "Signal recovery is generally preserved; component/noise allocation "
            "metrics define the local parameter trade-off boundary."
        )
    if source.startswith("6.2"):
        return (
            "Contamination-specific conclusions are largely preserved, with "
            "any losses localizing the design regimes that should be named."
        )
    if source.startswith("6.3"):
        return (
            "Signal/comparator robustness is the main generality test; losses "
            "identify metric-specific limits rather than overall retuning targets."
        )
    if source.startswith("6.4"):
        return (
            "Computational scalability is a real limitation: IRMF's runtime "
            "scales more steeply over the tested range."
        )
    if source.startswith("6.5"):
        return (
            "Matched-quality conclusions are more stable than unmatched-energy "
            "allocation diagnostics under correspondence-rule changes."
        )
    return "Manual interpretation required."


def run_v563_section6_failure_region_dominance_synthesis(output_root):
    output_root = ensure_dir(output_root)
    algorithm_root = output_root.parent if output_root.name.startswith("16") else output_root

    rows = []
    rows.extend(_parameter_boundary_rows(_read_csv(
        algorithm_root
        / "08_robustness_sensitivity_target_snr"
        / "6_3_parameter_sensitivity"
        / "section_6_3_primary_endpoint_case_blocked_factorial_summary.csv"
    )))
    rows.extend(_contamination_failure_summary(_read_csv(
        algorithm_root
        / "16t_v551_section6_2_contamination_design"
        / "v551_section6_2_paired_effects.csv"
    )))
    rows.extend(_paired_failure_summary(
        _read_csv(
            algorithm_root
            / "09_signal_variant_robustness_target_snr"
            / "variant_paired_method_difference_summary.csv"
        ),
        "6.3 signal-variant robustness",
        {"denoise_nmse", "denoise_corr", "imf_recovery_corr", "imf_recovery_nrmse"},
    ))
    rows.extend(_scaling_boundary_rows(algorithm_root))
    rows.extend(_matching_boundary_rows(algorithm_root))

    boundary_rows = [
        {
            "state": "preserved",
            "meaning": "IRMF advantage or conclusion direction persists across the audited perturbation.",
            "paper_language": "preserved / broadly stable",
        },
        {
            "state": "attenuated",
            "meaning": "IRMF advantage weakens or becomes conditional, without a clear median reversal.",
            "paper_language": "attenuated / trade-off / boundary condition",
        },
        {
            "state": "reversed",
            "meaning": "Comparator advantage appears for the metric or condition under the audited estimand.",
            "paper_language": "reversed for this metric or regime",
        },
        {
            "state": "not_identifiable",
            "meaning": "Current compact artifacts do not support this failure-region estimand.",
            "paper_language": "not evaluated by this analysis",
        },
    ]
    section_summary = _summarize_boundary(rows)

    write_csv(rows, output_root / "v563_section6_failure_region_dominance_map.csv")
    write_csv(section_summary, output_root / "v563_section6_boundary_condition_summary.csv")
    write_csv(boundary_rows, output_root / "v563_boundary_state_dictionary.csv")

    dashboard = {
        "schema_version": V563_FAILURE_REGION_SYNTHESIS_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "module_status": "failure_region_synthesis_complete_no_new_algorithm_runs",
        "no_algorithm_runs_performed": True,
        "no_primary_schema_change": True,
        "no_method_retuning": True,
        "no_second_overall_ranking": True,
        "recommended_section_label": "6.6 Synthesis of Robustness and Boundary Conditions",
        "scientific_question": "When, where, and how do the main comparative conclusions weaken?",
        "estimand_note": (
            "This synthesis uses existing Section 5/6 artifacts to summarize "
            "loss fractions, worst-decile paired effects where available, and "
            "boundary conditions. It is not a sixth algorithmic experiment."
        ),
        "status_taxonomy": [row["state"] for row in boundary_rows],
        "outputs": {
            "failure_region_dominance_map": str(
                output_root / "v563_section6_failure_region_dominance_map.csv"
            ),
            "boundary_condition_summary": str(
                output_root / "v563_section6_boundary_condition_summary.csv"
            ),
            "boundary_state_dictionary": str(
                output_root / "v563_boundary_state_dictionary.csv"
            ),
        },
        "section_summary": section_summary,
        "paper_ready_sentence": (
            "The robustness analyses do not imply uniform dominance; rather, "
            "they delineate the conditions under which the principal "
            "reconstruction advantage is preserved and the regimes in which "
            "component allocation, amplitude-sensitive recovery, computational "
            "scaling, or correspondence coverage become limiting."
        ),
    }
    write_json(dashboard, output_root / "v563_section6_failure_region_dominance_dashboard.json")
    return {
        "dashboard": dashboard,
        "failure_region_rows": rows,
        "section_summary": section_summary,
        "boundary_rows": boundary_rows,
    }
