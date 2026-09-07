#!/usr/bin/python
# coding: UTF-8

"""V5.67 Section 6.3A analysis and qualification freeze.

This freezes how the V5.66 signal-specification robustness rerun will be read.
It does not change the signal registry, execute algorithms, retune methods, or
alter the primary metric/ranking schema.
"""

from datetime import datetime, timezone

from project_config import (
    EVALUATION_METHODS,
    SIGNAL_VARIANT_FAMILY,
    SIGNAL_VARIANT_NOISES,
    SIGNAL_VARIANT_SEEDS,
    SIGNAL_VARIANT_TARGET_SNR_DB_LEVELS,
    V557_SECTION6_3_CHALLENGING_SIGNAL_EXTENSION,
    V564_SECTION6_3A_CHALLENGING_FAMILY_VARIANTS,
    V566_SECTION6_3A_CANONICAL_TWO_AXIS_SIGNAL_SPECS,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


V567_SECTION6_3A_ANALYSIS_QUALIFICATION_FREEZE_VERSION = (
    "V5.67_section6_3a_analysis_and_qualification_freeze"
)


PRIMARY_CONTRASTS = (
    {"contrast": "IRMF_vs_EMD", "baseline": "EMD"},
    {"contrast": "IRMF_vs_EEMD", "baseline": "EEMD"},
    {"contrast": "IRMF_vs_CEEMDAN", "baseline": "CEEMDAN"},
)


FOCAL_ENDPOINTS = (
    {
        "endpoint": "denoise_nmse",
        "direction": "lower_is_better",
        "claim_role": "principal_signal_reconstruction",
    },
    {
        "endpoint": "denoise_corr",
        "direction": "higher_is_better",
        "claim_role": "principal_signal_reconstruction",
    },
    {
        "endpoint": "matched_component_corr",
        "artifact_alias": "imf_recovery_corr",
        "direction": "higher_is_better",
        "claim_role": "component_waveform_correspondence",
    },
    {
        "endpoint": "matched_component_nrmse",
        "artifact_alias": "imf_recovery_nrmse",
        "direction": "lower_is_better",
        "claim_role": "amplitude_sensitive_component_recovery",
    },
)


ANALYSIS_LAYERS = (
    {
        "layer": "overall_preservation",
        "question": "Do Section 5 method contrasts remain broadly preserved across the full 73-specification V5.66 scope?",
        "estimand": "paired IRMF-baseline effects across signal specifications, noises, SNR levels, and seeds",
        "summary_rule": "report central paired effect, uncertainty interval, loss fraction, and worst-decile effect; do not construct a new overall ranking",
    },
    {
        "layer": "family_level_qualification",
        "question": "Which of the 15 signal families preserve, attenuate, or reverse the anchor-relative conclusions?",
        "estimand": "family-stratified anchor-relative change in paired method contrast",
        "summary_rule": "classify each family and endpoint as preserved, attenuated, reversed, or not identifiable using the frozen state definitions",
    },
    {
        "layer": "canonical_axis_level_sensitivity",
        "question": "For canonical families, which prespecified structural axis most affects conclusion preservation?",
        "estimand": "axis-stratified effect change for Axis 1 and Axis 2 single-axis perturbations",
        "summary_rule": "report direction and magnitude change by family, endpoint, and axis; interpret as specification sensitivity, not retuning guidance",
    },
    {
        "layer": "joint_corner_qualification",
        "question": "Do prespecified joint perturbation corners show excess degradation beyond single-axis perturbations?",
        "estimand": "joint-corner degradation and, where estimable, Delta_AB - Delta_A - Delta_B or case-blocked A x B terms",
        "summary_rule": "use 'joint perturbation corner' by default; use 'interaction effect' only if explicit factorial contrast or case-blocked interaction estimate supports it",
    },
)


QUALIFICATION_STATES = (
    {
        "state": "preserved",
        "definition": "effect direction is unchanged and the effect magnitude is not materially reduced relative to the anchor/reference contrast",
        "allowed_wording": "preserved; broadly preserved; remained directionally consistent",
    },
    {
        "state": "attenuated",
        "definition": "effect direction is unchanged but the effect magnitude is materially reduced or uncertainty includes a practically weak effect",
        "allowed_wording": "attenuated; weakened; narrowed",
    },
    {
        "state": "reversed",
        "definition": "effect direction changes relative to the anchor/reference contrast",
        "allowed_wording": "reversed; comparator advantage emerged; direction changed",
    },
    {
        "state": "not_identifiable",
        "definition": "the endpoint is noncomputable, has insufficient support, or has coverage/validity limitations that prevent a stable preservation judgment",
        "allowed_wording": "not identifiable; not qualified; insufficient support",
    },
)


MATERIALITY_RULES = (
    {
        "rule_id": "direction_first",
        "rule": "A direction reversal overrides magnitude rules and is classified as reversed.",
    },
    {
        "rule_id": "material_reduction",
        "rule": "A same-direction effect is attenuated when the absolute paired effect is reduced by at least 50% relative to the anchor/reference contrast or falls below the endpoint-specific practical-effect threshold used in the Section 6 synthesis.",
    },
    {
        "rule_id": "uncertainty_aware",
        "rule": "Uncertainty intervals are reported alongside state labels; a preserved label should not be used as evidence of uniform dominance when intervals show material ambiguity.",
    },
    {
        "rule_id": "no_new_ranking",
        "rule": "Section 6.3A evaluates conclusion preservation and boundary conditions; it must not create a second overall ranking of methods.",
    },
)


LANGUAGE_RULES = (
    {
        "term": "joint perturbation corner",
        "rule": "Use for the two combined canonical conditions in V5.66 regardless of whether a statistical interaction is detected.",
    },
    {
        "term": "interaction effect",
        "rule": "Use only when supported by Delta_AB - Delta_A - Delta_B contrasts or by a case-blocked model containing an A x B term.",
    },
    {
        "term": "within-family specification robustness",
        "rule": "Use as the umbrella term for V5.66/V5.67 6.3A; distinguish canonical two-axis robustness from challenging difficulty-axis robustness.",
    },
)


def run_v567_section6_3a_analysis_qualification_freeze(output_root):
    output_root = ensure_dir(output_root)
    n_signal_specs = len(SIGNAL_VARIANT_FAMILY)
    n_case_rows = (
        n_signal_specs
        * len(SIGNAL_VARIANT_NOISES)
        * len(SIGNAL_VARIANT_TARGET_SNR_DB_LEVELS)
        * len(SIGNAL_VARIANT_SEEDS)
    )
    n_method_evaluations = n_case_rows * len(EVALUATION_METHODS)

    dashboard = {
        "schema_version": V567_SECTION6_3A_ANALYSIS_QUALIFICATION_FREEZE_VERSION,
        "created_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "module_status": "analysis_and_qualification_rules_frozen_execution_pending",
        "algorithm_runs_performed": False,
        "signal_registry_changed": False,
        "primary_metric_schema_changed": False,
        "section6_scientific_questions_changed": False,
        "ranking_procedure_changed": False,
        "not_retuning": True,
        "scope_basis": "V5.66 73-specification signal-scope design",
        "canonical_signal_specifications": int(len(V566_SECTION6_3A_CANONICAL_TWO_AXIS_SIGNAL_SPECS)),
        "challenging_signal_specifications": int(
            len(V557_SECTION6_3_CHALLENGING_SIGNAL_EXTENSION)
            + len(V564_SECTION6_3A_CHALLENGING_FAMILY_VARIANTS)
        ),
        "total_signal_specifications": int(n_signal_specs),
        "expected_6_3a_case_rows": int(n_case_rows),
        "expected_6_3a_method_evaluations": int(n_method_evaluations),
        "analysis_layers_frozen": [row["layer"] for row in ANALYSIS_LAYERS],
        "qualification_states_frozen": [row["state"] for row in QUALIFICATION_STATES],
        "language_guardrail": "joint corner != interaction effect",
        "post_rerun_required_sequence": [
            "full 6.3A rerun under V5.66 scope",
            "execution integrity audit",
            "overall preservation analysis",
            "family-level qualification",
            "canonical axis-level sensitivity",
            "joint-corner qualification",
            "post-execution claim authorization",
        ],
        "claim_boundary": (
            "V5.67 authorizes interpretation rules for future 6.3A execution. "
            "It does not authorize updated scientific claims until the V5.66 "
            "scope is rerun and the frozen qualification rules are applied."
        ),
    }

    write_json(dashboard, output_root / "v567_section6_3a_analysis_qualification_freeze_dashboard.json")
    write_csv(ANALYSIS_LAYERS, output_root / "v567_analysis_layers.csv")
    write_csv(QUALIFICATION_STATES, output_root / "v567_qualification_state_definitions.csv")
    write_csv(MATERIALITY_RULES, output_root / "v567_materiality_rules.csv")
    write_csv(LANGUAGE_RULES, output_root / "v567_language_rules.csv")
    write_csv(FOCAL_ENDPOINTS, output_root / "v567_focal_endpoints.csv")
    write_csv(PRIMARY_CONTRASTS, output_root / "v567_primary_method_contrasts.csv")
    return {
        "dashboard": dashboard,
        "analysis_layers": ANALYSIS_LAYERS,
        "qualification_states": QUALIFICATION_STATES,
        "materiality_rules": MATERIALITY_RULES,
        "language_rules": LANGUAGE_RULES,
    }
