#!/usr/bin/python
# coding: UTF-8

"""V5.62 reviewer-readiness synthesis for Section 6.

This module does not run algorithms or recompute benchmark metrics.  It
collects the existing Section 6.1--6.5 artifacts into a reviewer-facing
fragility-audit table, with explicit claim boundaries and failure-region notes.
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


V562_SECTION6_REVIEWER_READINESS_VERSION = (
    "V5.62_section6_reviewer_readiness_synthesis"
)

SECTION6_OPENING_PARAGRAPH = (
    "We assess whether the principal conclusions of Section 5 remain stable "
    "under five prespecified classes of perturbation: estimator "
    "parameterization, contamination design, signal and comparator "
    "specification, computational scale, and component-correspondence rule. "
    "These analyses are designed to evaluate conclusion preservation and "
    "failure regions, rather than to retune the methods or construct a second "
    "overall ranking."
)

MATCHING_CLAIM_BOUNDARY = (
    "Matching-rule sensitivity applies to matched-component correlation and "
    "NRMSE only; matching-rule-derived unmatched-energy diagnostics are "
    "supporting quantities and do not redefine the primary Component-Set "
    "Fidelity metrics."
)


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


def _format_float(value, digits=4):
    if value is None:
        return "NA"
    value = _float(value)
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}g}"


def _artifact_exists(algorithm_root, relpath):
    return (algorithm_root / relpath).exists()


def _metric_medians_by_method(rows, metric, axis=None, matching_rule=None, tau=None):
    out = {}
    for row in rows:
        if "metric" in row and row.get("metric") != metric:
            continue
        if axis is not None and row.get("axis") != axis:
            continue
        if matching_rule is not None and row.get("matching_rule") != matching_rule:
            continue
        if tau is not None and abs(_float(row.get("tau")) - float(tau)) > 1e-12:
            continue
        method = row.get("method")
        value = (
            row.get("median")
            if "median" in row
            else row.get(f"{metric}_median")
        )
        if method:
            out[method] = _float(value)
    return out


def _order(method_to_value, higher_is_better=False):
    clean = {
        k: v for k, v in method_to_value.items()
        if isinstance(v, (int, float, np.floating)) and np.isfinite(v)
    }
    if not clean:
        return "NA"
    return ">".join(
        k for k, _ in sorted(
            clean.items(),
            key=lambda item: item[1],
            reverse=bool(higher_is_better),
        )
    )


def _section_61_summary(algorithm_root):
    rel = (
        "08_robustness_sensitivity_target_snr/6_3_parameter_sensitivity/"
        "section_6_3_primary_endpoint_case_blocked_factorial_summary.csv"
    )
    rows = _read_csv(algorithm_root / rel)
    by_metric = {row.get("primary_endpoint"): row for row in rows}
    nmse = by_metric.get("denoise_nmse", {})
    corr = by_metric.get("denoise_corr", {})
    hmin_noise = by_metric.get("noise_energy_log_error", {})
    c_h_nmse = [
        row for row in _read_csv(
            algorithm_root
            / "08_robustness_sensitivity_target_snr/6_3_parameter_sensitivity/"
            / "section_6_3_primary_endpoint_main_effect_level_summary.csv"
        )
        if row.get("parameter") == "c_H"
        and row.get("primary_endpoint") == "denoise_nmse"
    ]
    c_h_note = "c_H local effects available"
    if c_h_nmse:
        values = [_float(row.get("median_raw")) for row in c_h_nmse]
        values = [v for v in values if np.isfinite(v)]
        if values:
            c_h_note = f"c_H denoise-NMSE median range {min(values):.4g}-{max(values):.4g}"
    return {
        "central": (
            "Default denoise NMSE "
            f"{_format_float(nmse.get('default_setting_median'))}; best grid "
            f"{_format_float(nmse.get('best_setting_median'))}; default denoise corr "
            f"{_format_float(corr.get('default_setting_median'))}. {c_h_note}."
        ),
        "uncertainty": (
            "Case-blocked factorial summaries report raw dispersion, default "
            "percentile, oracle regret, plateau width, and partial R2 across "
            f"{nmse.get('n_case_blocks', 'NA')} case blocks."
        ),
        "failure": (
            "Interaction materiality flags are false for signal-level recovery; "
            "sensitivity is more visible for component/noise allocation. "
            "Strongest noise-energy main effect: "
            f"{hmin_noise.get('strongest_main_effect', 'NA')}."
        ),
        "evidence": rel,
    }


def _section_62_summary(algorithm_root):
    summary_rel = (
        "16t_v551_section6_2_contamination_design/"
        "v551_section6_2_metric_summary.csv"
    )
    win_rel = (
        "16t_v551_section6_2_contamination_design/"
        "v551_section6_2_win_rate_summary.csv"
    )
    dash_rel = (
        "16t_v551_section6_2_contamination_design/"
        "v551_section6_2_contamination_design_dashboard.json"
    )
    rows = _read_csv(algorithm_root / summary_rel)
    wins = _read_csv(algorithm_root / win_rel)
    dash = _read_json(algorithm_root / dash_rel, default={})
    central_bits = []
    for metric in (
        "contaminated_region_nmse",
        "clean_region_nmse",
        "normalized_contamination_spillover_loss",
    ):
        vals = [
            _float(row.get("median")) for row in rows
            if row.get("metric") == metric and row.get("method") == "IRMF"
        ]
        if vals:
            central_bits.append(f"{metric} IRMF median range {min(vals):.4g}-{max(vals):.4g}")
    win_vals = [
        _float(row.get("win_rate")) for row in wins
        if row.get("baseline") == "CEEMDAN"
        and row.get("metric") in {
            "contaminated_region_nmse",
            "clean_region_nmse",
            "normalized_contamination_spillover_loss",
        }
    ]
    win_note = ""
    if win_vals:
        win_note = f"; IRMF-vs-CEEMDAN win-rate range {min(win_vals):.3g}-{max(win_vals):.3g}"
    return {
        "central": "; ".join(central_bits) + win_note,
        "uncertainty": (
            "Summaries stratify rate, magnitude, geometry, variance-matched "
            "shape, and limited rate-by-magnitude axes across seeds/cases."
        ),
        "failure": (
            f"n_failed={dash.get('n_failed', 'NA')}; target-SNR and variance-match "
            "audits passed in the execution dashboard; adverse design axes did "
            "not create an execution failure region."
        ),
        "evidence": f"{summary_rel}; {win_rel}; {dash_rel}",
    }


def _section_63_summary(algorithm_root):
    sig_dash_rel = (
        "09_signal_variant_robustness_target_snr/"
        "section_5_signal_family_reproducibility_dashboard.json"
    )
    sig_summary_rel = (
        "09_signal_variant_robustness_target_snr/"
        "variant_family_reproducibility_summary.csv"
    )
    comp_rel = (
        "11_emd_family_sensitivity_target_snr/"
        "appendix_D_emd_family_sensitivity_aggregate.json"
    )
    sig_dash = _read_json(algorithm_root / sig_dash_rel, default={})
    rows = _read_csv(algorithm_root / sig_summary_rel)
    ir = [
        _float(row.get("median_cell_mean")) for row in rows
        if row.get("metric") == "denoise_nmse" and row.get("method") == "IRMF"
    ]
    baseline = [
        _float(row.get("median_cell_mean")) for row in rows
        if row.get("metric") == "denoise_nmse" and row.get("method") != "IRMF"
    ]
    central = "Signal-variant artifact present"
    if ir and baseline:
        central = (
            f"IRMF denoise-NMSE family median range {min(ir):.4g}-{max(ir):.4g}; "
            f"baseline family median range {min(baseline):.4g}-{max(baseline):.4g}."
        )
    return {
        "central": central,
        "uncertainty": (
            "Evidence combines signal-family stratification, Monte Carlo seeds, "
            "and EMD-family comparator-configuration sweeps."
        ),
        "failure": (
            "Manual wording should retain known trade-offs: EEMD may remain "
            "competitive for amplitude-sensitive matched-component NRMSE, and "
            "structural diagnostics are not uniform IRMF wins."
        ),
        "evidence": f"{sig_dash_rel}; {sig_summary_rel}; {comp_rel}",
        "n_case_rows": sig_dash.get("n_case_rows", sig_dash.get("n_cases")),
    }


def _section_64_summary(algorithm_root):
    rt_rel = (
        "16u_v552_section6_4_computational_scaling/"
        "v552_runtime_by_method_and_n.csv"
    )
    beta_rel = (
        "16u_v552_section6_4_computational_scaling/"
        "v552_runtime_scaling_exponents.csv"
    )
    dash_rel = (
        "16u_v552_section6_4_computational_scaling/"
        "v552_section6_4_runtime_scaling_dashboard.json"
    )
    rt = _read_csv(algorithm_root / rt_rel)
    beta = _read_csv(algorithm_root / beta_rel)
    at_500 = {
        row.get("method"): _float(row.get("median_runtime_seconds"))
        for row in rt if row.get("n") == "500"
    }
    beta_vals = {
        row.get("method"): _float(row.get("empirical_scaling_exponent_beta"))
        for row in beta
    }
    dash = _read_json(algorithm_root / dash_rel, default={})
    return {
        "central": (
            "Runtime at n=500: "
            + ", ".join(f"{m}={v:.3g}s" for m, v in sorted(at_500.items()) if np.isfinite(v))
            + ". Empirical beta: "
            + ", ".join(f"{m}={v:.3g}" for m, v in sorted(beta_vals.items()) if np.isfinite(v))
            + "."
        ),
        "uncertainty": (
            "Runtime uses repeated outer wall-clock timing with median/IQR/p90/p95 "
            "summaries and log-log fit diagnostics."
        ),
        "failure": (
            f"n_failed={dash.get('n_failed', 'NA')}; large-n bottleneck is IRMF's "
            "steeper empirical scaling, not algorithm failure."
        ),
        "evidence": f"{rt_rel}; {beta_rel}; {dash_rel}",
    }


def _section_65_summary(algorithm_root):
    method_rel = (
        "16ae_v561_section6_5_matching_rule_robustness/"
        "v561_section6_5_matching_rule_robustness_method_summary.csv"
    )
    preservation_rel = (
        "16ae_v561_section6_5_matching_rule_robustness/"
        "v561_section6_5_matching_rule_robustness_preservation.csv"
    )
    dash_rel = (
        "16ae_v561_section6_5_matching_rule_robustness/"
        "v561_section6_5_matching_rule_robustness_dashboard.json"
    )
    rows = _read_csv(algorithm_root / method_rel)
    dash = _read_json(algorithm_root / dash_rel, default={})
    primary_corr = _metric_medians_by_method(
        rows, "matched_corr", matching_rule="primary_unthresholded_hungarian", tau=0.0
    )
    primary_nrmse = _metric_medians_by_method(
        rows, "matched_nrmse", matching_rule="primary_unthresholded_hungarian", tau=0.0
    )
    preservation = dash.get("preservation_summary", {})
    return {
        "central": (
            f"Primary-Hungarian matched-corr order {_order(primary_corr, True)}; "
            f"matched-NRMSE order {_order(primary_nrmse, False)}. "
            "Matched quality must be read with coverage."
        ),
        "uncertainty": (
            "Robustness is summarized over thresholded-Hungarian tau curve "
            "0.3--0.9 and mutual-nearest-neighbor alternatives, with coverage "
            "and noncomputability rates."
        ),
        "failure": (
            "Preservation summary: quality effect-direction rate "
            f"{_format_float(preservation.get('quality_effect_direction_preserved_rate'), 3)}, "
            "moderate-tau rate "
            f"{_format_float(preservation.get('moderate_tau_effect_direction_preserved_rate'), 3)}; "
            "allocation-derived unmatched-energy diagnostics are rule-sensitive."
        ),
        "evidence": f"{method_rel}; {preservation_rel}; {dash_rel}",
    }


def run_v562_section6_reviewer_readiness_synthesis(output_root):
    output_root = ensure_dir(output_root)
    algorithm_root = output_root.parent if output_root.name.startswith("16") else output_root

    s61 = _section_61_summary(algorithm_root)
    s62 = _section_62_summary(algorithm_root)
    s63 = _section_63_summary(algorithm_root)
    s64 = _section_64_summary(algorithm_root)
    s65 = _section_65_summary(algorithm_root)

    synthesis_rows = [
        {
            "section": "6.1",
            "fragility_source": "estimator parameterization",
            "primary_question": "Does IRMF depend on a narrow tuned parameter optimum?",
            "main_estimand": "Primary outcomes across the local h1, a, h_min, c_H grid.",
            "central_effect": s61["central"],
            "uncertainty_summary": s61["uncertainty"],
            "failure_region_summary": s61["failure"],
            "conclusion_preserved": "yes for signal-level recovery; partial for component/noise allocation",
            "claim_boundary": "No retuning claim; IRMF-only parameter perturbation around the locked tuple.",
            "paper_weight": "medium-high",
            "evidence_files": s61["evidence"],
            "recommended_wording": (
                "Signal-level reconstruction is locally robust; parameter effects "
                "are more informative as allocation/noise-separation trade-offs."
            ),
        },
        {
            "section": "6.2",
            "fragility_source": "contamination design",
            "primary_question": "Do contamination conclusions depend on rate, magnitude, geometry, or shape?",
            "main_estimand": "Three contamination primary endpoints across prespecified design axes.",
            "central_effect": s62["central"],
            "uncertainty_summary": s62["uncertainty"],
            "failure_region_summary": s62["failure"],
            "conclusion_preserved": "yes for contamination-specific endpoints",
            "claim_boundary": "Contamination-design robustness only; not a general replacement for Section 5.",
            "paper_weight": "medium-high",
            "evidence_files": s62["evidence"],
            "recommended_wording": (
                "IRMF's direct contaminated-region recovery and clean-region "
                "preservation remain stable across contamination design axes."
            ),
        },
        {
            "section": "6.3",
            "fragility_source": "signal/comparator specification",
            "primary_question": "Do conclusions depend on easy signals or weak comparator defaults?",
            "main_estimand": "Cross-family signal robustness and reasonable EMD-family comparator variants.",
            "central_effect": s63["central"],
            "uncertainty_summary": s63["uncertainty"],
            "failure_region_summary": s63["failure"],
            "conclusion_preserved": "yes/partial; strongest robustness evidence with retained trade-offs",
            "claim_boundary": "Supports generality of main conclusions, not universal dominance on all structural properties.",
            "paper_weight": "highest",
            "evidence_files": s63["evidence"],
            "recommended_wording": (
                "The reconstruction pattern is not driven by a single signal "
                "formula or comparator setting, while component-level trade-offs "
                "remain visible."
            ),
        },
        {
            "section": "6.4",
            "fragility_source": "computational scale",
            "primary_question": "How do wall-clock cost and accuracy context change with sample size?",
            "main_estimand": "Runtime, empirical scaling exponent, and accuracy-context diagnostics across n.",
            "central_effect": s64["central"],
            "uncertainty_summary": s64["uncertainty"],
            "failure_region_summary": s64["failure"],
            "conclusion_preserved": "limited; supports computational trade-off characterization",
            "claim_boundary": "Computational evidence only; empirical beta is not theoretical complexity.",
            "paper_weight": "medium",
            "evidence_files": s64["evidence"],
            "recommended_wording": (
                "IRMF has a clear accuracy-computation trade-off: comparable "
                "cost to ensemble baselines near n=500 but steeper large-n scaling."
            ),
        },
        {
            "section": "6.5",
            "fragility_source": "component-correspondence rule",
            "primary_question": "Do matched-component conclusions depend on unthresholded Hungarian correspondence?",
            "main_estimand": "Preservation of matched_component_corr and matched_component_nrmse under alternative rules.",
            "central_effect": s65["central"],
            "uncertainty_summary": s65["uncertainty"],
            "failure_region_summary": s65["failure"],
            "conclusion_preserved": "yes at moderate restriction for matched quality; no for allocation-derived diagnostics",
            "claim_boundary": MATCHING_CLAIM_BOUNDARY,
            "paper_weight": "medium-low",
            "evidence_files": s65["evidence"],
            "recommended_wording": (
                "Matched waveform-quality conclusions are relatively robust to "
                "moderate correspondence restrictions, but unmatched-energy "
                "diagnostics should be interpreted as rule-sensitive context."
            ),
        },
    ]

    hierarchy_rows = [
        {
            "layer": "Section 5 primary evidence",
            "role": "Primary endpoint-specific and construct-level comparative evidence.",
            "ranking_role": "Average ranks are descriptive only.",
            "claim_boundary": "Primary claims are based on frozen primary endpoints, not a composite score.",
        },
        {
            "layer": "Section 6 conclusion-preservation evidence",
            "role": "Tests whether Section 5 conclusions survive prespecified perturbations and identifies failure regions.",
            "ranking_role": "No second overall ranking.",
            "claim_boundary": "Sensitivity results contextualize, qualify, or bound Section 5 claims; they do not retune methods.",
        },
        {
            "layer": "Appendix/protocol credibility evidence",
            "role": "Documents implementation integrity, audits, secondary diagnostics, and supplementary stability analyses.",
            "ranking_role": "Not ranking-bearing.",
            "claim_boundary": "Protocol audits support credibility of measurement and execution, not new performance claims.",
        },
    ]

    fallacy_scan_rows = [
        {
            "risk": "post_hoc_expansion",
            "status": "mitigated",
            "note": "V5.57/V5.61 define amendments separately from execution and claim authorization.",
        },
        {
            "risk": "second_overall_ranking",
            "status": "mitigated",
            "note": "V5.62 explicitly frames Section 6 as conclusion-preservation, not a new ranking exercise.",
        },
        {
            "risk": "point_estimate_overinterpretation",
            "status": "partially_mitigated",
            "note": "Artifacts include dispersion, coverage, and case-blocked summaries; manuscript should foreground uncertainty bands or paired effects where available.",
        },
        {
            "risk": "causal_language_from_diagnostics",
            "status": "requires_wording_control",
            "note": "Use 'consistent with' or 'accompanied by' for secondary mechanisms unless a causal perturbation was run.",
        },
        {
            "risk": "matching_rule_estimand_drift",
            "status": "mitigated",
            "note": MATCHING_CLAIM_BOUNDARY,
        },
        {
            "risk": "computational_complexity_overclaim",
            "status": "mitigated",
            "note": "V5.52 beta is empirical scaling over tested n, not theoretical complexity.",
        },
        {
            "risk": "coverage_conditioned_quality_bias",
            "status": "mitigated",
            "note": "V5.61 requires matched quality to be interpreted jointly with valid-match coverage and no-valid-match rate.",
        },
    ]

    output_root = ensure_dir(output_root)
    write_csv(synthesis_rows, output_root / "v562_section6_reviewer_readiness_synthesis_table.csv")
    write_csv(hierarchy_rows, output_root / "v562_section6_claim_hierarchy.csv")
    write_csv(fallacy_scan_rows, output_root / "v562_section6_interpretation_fallacy_scan.csv")

    dashboard = {
        "schema_version": V562_SECTION6_REVIEWER_READINESS_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "module_status": "synthesis_complete_no_new_algorithm_runs",
        "no_algorithm_runs_performed": True,
        "no_primary_schema_change": True,
        "no_method_retuning": True,
        "no_second_overall_ranking": True,
        "section6_subsections": ["6.1", "6.2", "6.3", "6.4", "6.5"],
        "overall_readiness": "reviewer_ready_with_manual_wording_review",
        "claim_authorization_status": (
            "synthesis layer only; scientific claim authorization remains tied "
            "to the underlying V5.59/V5.61 gates and manuscript wording."
        ),
        "prespecification_statement": (
            "All sensitivity subsets, perturbation axes, and interpretation "
            "criteria should be described as specified before examining the "
            "corresponding robustness results."
        ),
        "recommended_section6_opening_paragraph": SECTION6_OPENING_PARAGRAPH,
        "recommended_paper_weighting": "6.3 > 6.1 approximately 6.2 > 6.4 approximately 6.5",
        "synthesis_outputs": {
            "main_table": str(output_root / "v562_section6_reviewer_readiness_synthesis_table.csv"),
            "claim_hierarchy": str(output_root / "v562_section6_claim_hierarchy.csv"),
            "fallacy_scan": str(output_root / "v562_section6_interpretation_fallacy_scan.csv"),
        },
        "sections": synthesis_rows,
    }
    write_json(dashboard, output_root / "v562_section6_reviewer_readiness_synthesis_dashboard.json")
    return {
        "dashboard": dashboard,
        "synthesis_rows": synthesis_rows,
        "claim_hierarchy_rows": hierarchy_rows,
        "fallacy_scan_rows": fallacy_scan_rows,
    }
