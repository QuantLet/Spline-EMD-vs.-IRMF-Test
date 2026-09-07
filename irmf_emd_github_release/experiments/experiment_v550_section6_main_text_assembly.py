#!/usr/bin/python
# coding: UTF-8

"""V5.50 assembly map for the compact four-part Section 6 structure.

This module does not run sensitivity experiments.  It records which existing
target-SNR artifacts support the four main-text Section 6 subsections and which
legacy/protocol checks should be treated as supplementary material.
"""

from datetime import datetime, timezone
from pathlib import Path

from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


V550_SECTION6_MAIN_TEXT_ASSEMBLY_VERSION = "V5.50_section6_main_text_assembly"


def _exists(root, relpath):
    return (root / relpath).exists()


def _artifact(root, relpath, role="required"):
    path = root / relpath
    return {
        "path": relpath,
        "exists": bool(path.exists()),
        "role": role,
    }


def _all_exist(items):
    return all(bool(item.get("exists")) for item in items)


def _section_status(required, pending=None):
    pending = pending or []
    if not _all_exist(required):
        return "missing_required_artifacts"
    if pending:
        return "partial_execution_present_with_declared_pending_work"
    return "main_text_ready_from_existing_target_snr_artifacts"


def _readiness_layers(required, pending=None):
    execution_ready = _all_exist(required)
    scientific_ready = bool(execution_ready and not pending)
    claim_ready = False
    return {
        "execution_ready": execution_ready,
        "scientific_ready": scientific_ready,
        "claim_ready": claim_ready,
        "claim_ready_note": (
            "Claim readiness is intentionally false until V5.53 Section 6 "
            "scientific/claim qualification passes."
        ),
    }


def run_v550_section6_main_text_assembly(output_root):
    output_root = ensure_dir(output_root)
    algorithm_root = output_root.parent if output_root.name.startswith("16") else output_root
    v551_complete = _exists(
        algorithm_root,
        "16t_v551_section6_2_contamination_design/"
        "v551_section6_2_contamination_design_dashboard.json",
    )
    v552_complete = _exists(
        algorithm_root,
        "16u_v552_section6_4_computational_scaling/"
        "v552_section6_4_runtime_scaling_dashboard.json",
    )
    v561_complete = _exists(
        algorithm_root,
        "16ae_v561_section6_5_matching_rule_robustness/"
        "v561_section6_5_matching_rule_robustness_dashboard.json",
    )

    sections = [
        {
            "section": "6.1",
            "title": "IRMF Parameter Sensitivity and Robustness",
            "scientific_question": "Does IRMF depend on a narrow tuned optimum?",
            "main_text_role": "primary Section 6 robustness evidence",
            "required_artifacts": [
                _artifact(
                    algorithm_root,
                    "08_robustness_sensitivity_target_snr/6_3_parameter_sensitivity/"
                    "section_6_3_parameter_sensitivity_aggregate.json",
                ),
                _artifact(
                    algorithm_root,
                    "08_robustness_sensitivity_target_snr/6_3_parameter_sensitivity/"
                    "section_6_3_factorial_anova.json",
                ),
                _artifact(
                    algorithm_root,
                    "08_robustness_sensitivity_target_snr/6_3_parameter_sensitivity/"
                    "section_6_3_primary_endpoint_case_blocked_factorial.json",
                ),
                _artifact(
                    algorithm_root,
                    "08_robustness_sensitivity_target_snr/6_3_parameter_sensitivity/"
                    "section_6_3_primary_endpoint_case_blocked_factorial_summary.csv",
                ),
                _artifact(
                    algorithm_root,
                    "08_robustness_sensitivity_target_snr/6_3_parameter_sensitivity/"
                    "section_6_3_protocol.json",
                ),
            ],
            "output_mapping_note": (
                "Existing output directory keeps its historical name "
                "6_3_parameter_sensitivity, but it maps to final main-text "
                "Section 6.1."
            ),
            "parameterization_requirement": (
                "Must perturb h1, a, h_min, and c_H; H_case is derived "
                "mechanically as c_H times the frozen truth-free scale proxy."
            ),
            "statistical_requirement": (
                "Must include case-blocked factorial sensitivity analysis over "
                "the 10 non-contamination primary endpoints; legacy aggregate "
                "score ANOVA is descriptive only."
            ),
            "pending_work": [],
        },
        {
            "section": "6.2",
            "title": "Contamination Design Sensitivity",
            "scientific_question": (
                "Do contamination conclusions depend on rate, magnitude, or "
                "position/geometry?"
            ),
            "main_text_role": "primary Section 6 robustness evidence after full design execution",
            "required_artifacts": (
                [
                    _artifact(
                        algorithm_root,
                        "16t_v551_section6_2_contamination_design/"
                        "v551_section6_2_contamination_design_dashboard.json",
                    ),
                    _artifact(
                        algorithm_root,
                        "16t_v551_section6_2_contamination_design/"
                        "v551_section6_2_metric_summary.csv",
                    ),
                    _artifact(
                        algorithm_root,
                        "16t_v551_section6_2_contamination_design/"
                        "v551_section6_2_paired_effects.csv",
                    ),
                    _artifact(
                        algorithm_root,
                        "16t_v551_section6_2_contamination_design/"
                        "v551_section6_2_win_rate_summary.csv",
                    ),
                    _artifact(
                        algorithm_root,
                        "16t_v551_section6_2_contamination_design/"
                        "v551_section6_2_normalized_spillover_formula_audit.csv",
                    ),
                ]
                if v551_complete else [
                _artifact(
                    algorithm_root,
                    "08_robustness_sensitivity_target_snr/6_2_contamination_robustness/"
                    "section_6_2_contamination_robustness_aggregate.json",
                ),
                _artifact(
                    algorithm_root,
                    "08_robustness_sensitivity_target_snr/6_2_contamination_robustness/"
                    "section_6_2_protocol.json",
                ),
                ]
            ),
            "output_mapping_note": (
                "V5.51 full artifact covers target-SNR rate, magnitude, "
                "geometry, variance-matched shape, and lambda-by-magnitude axes."
                if v551_complete else
                "Current artifact covers the target-SNR contamination-rate "
                "axis. It is not yet the complete rate x magnitude x geometry "
                "Section 6.2 design."
            ),
            "pending_work": [] if v551_complete else [
                "full outlier magnitude axis",
                "full contamination position/geometry axis",
                "variance-matched contamination comparison",
                "selected magnitude-by-position interaction summary",
            ],
        },
        {
            "section": "6.3",
            "title": "Comparator and Signal Robustness",
            "scientific_question": (
                "Do conclusions depend on one signal family or one reasonable "
                "EMD-family comparator setting?"
            ),
            "main_text_role": "primary Section 6 robustness evidence",
            "required_artifacts": [
                _artifact(
                    algorithm_root,
                    "09_signal_variant_robustness_target_snr/"
                    "section_5_signal_family_reproducibility_aggregate.json",
                ),
                _artifact(
                    algorithm_root,
                    "09_signal_variant_robustness_target_snr/"
                    "section_5_signal_family_reproducibility_dashboard.json",
                ),
                _artifact(
                    algorithm_root,
                    "11_emd_family_sensitivity_target_snr/"
                    "appendix_D_emd_family_sensitivity_aggregate.json",
                ),
                _artifact(
                    algorithm_root,
                    "11_emd_family_sensitivity_target_snr/"
                    "appendix_D_emd_family_sensitivity_protocol.json",
                ),
            ],
            "output_mapping_note": (
                "Signal-variant and EMD-family outputs retain older artifact "
                "names but are assembled as final main-text Section 6.3."
            ),
            "pending_work": [],
        },
        {
            "section": "6.4",
            "title": "Computational Efficiency and Scaling",
            "scientific_question": (
                "What computational cost is associated with each locked method, "
                "and how does runtime scale with n?"
            ),
            "main_text_role": "pending full Section 6 runtime/scaling execution",
            "required_artifacts": (
                [
                    _artifact(
                        algorithm_root,
                        "16u_v552_section6_4_computational_scaling/"
                        "v552_section6_4_runtime_scaling_dashboard.json",
                    ),
                    _artifact(
                        algorithm_root,
                        "16u_v552_section6_4_computational_scaling/"
                        "v552_runtime_by_method.csv",
                    ),
                    _artifact(
                        algorithm_root,
                        "16u_v552_section6_4_computational_scaling/"
                        "v552_runtime_by_method_and_n.csv",
                    ),
                    _artifact(
                        algorithm_root,
                        "16u_v552_section6_4_computational_scaling/"
                        "v552_runtime_scaling_exponents.csv",
                    ),
                ]
                if v552_complete else [
                _artifact(
                    algorithm_root,
                    "16j_v538_section6_4_runtime_instrumentation_smoke/"
                    "v538_section6_4_smoke_status.json",
                    role="smoke_only",
                ),
                _artifact(
                    algorithm_root,
                    "16j_v538_section6_4_runtime_instrumentation_smoke/"
                    "v538_section6_4_environment_manifest.json",
                    role="smoke_only",
                ),
                ]
            ),
            "output_mapping_note": (
                "V5.52 full runtime/scaling artifact is present."
                if v552_complete else
                "Only instrumentation smoke is present. Main-text computational "
                "scaling still requires full runtime execution."
            ),
            "pending_work": [] if v552_complete else [
                "full runtime by method and n",
                "memory or peak-resource manifest if available",
                "log-runtime scaling slope",
                "accuracy-runtime trade-off summary",
            ],
        },
        {
            "section": "6.5",
            "title": "Matching-Rule Robustness",
            "scientific_question": (
                "Do component-level conclusions depend on the specific "
                "estimated-vs-truth correspondence rule?"
            ),
            "main_text_role": (
                "evaluation-protocol robustness evidence; not a replacement "
                "for frozen primary matching and not ranking-bearing"
            ),
            "required_artifacts": (
                [
                    _artifact(
                        algorithm_root,
                        "16ae_v561_section6_5_matching_rule_robustness/"
                        "v561_section6_5_matching_rule_robustness_dashboard.json",
                    ),
                    _artifact(
                        algorithm_root,
                        "16ae_v561_section6_5_matching_rule_robustness/"
                        "v561_section6_5_matching_rule_robustness_method_summary.csv",
                    ),
                    _artifact(
                        algorithm_root,
                        "16ae_v561_section6_5_matching_rule_robustness/"
                        "v561_section6_5_matching_rule_robustness_preservation.csv",
                    ),
                ]
                if v561_complete else [
                    _artifact(
                        algorithm_root,
                        "16ac_v560_matching_sensitivity_protocol/"
                        "v560_matching_sensitivity_protocol.json",
                        role="protocol_only",
                    ),
                ]
            ),
            "output_mapping_note": (
                "V5.61 full matching-rule robustness curve is present; "
                "quality must be interpreted jointly with valid-match coverage. "
                "The section targets matched_component_corr and "
                "matched_component_nrmse; matching-rule unmatched-energy fields "
                "are supporting diagnostics, not primary Component-Set Fidelity "
                "redefinitions."
                if v561_complete else
                "Only V5.60 protocol is present. Section 6.5 execution requires "
                "the prespecified matching-rule robustness curve."
            ),
            "pending_work": [] if v561_complete else [
                "thresholded Hungarian robustness curve",
                "mutual-nearest-neighbor alternative correspondence curve",
                "quality-coverage trade-off summary",
                "effect-direction and method-order preservation summary",
            ],
        },
    ]

    for item in sections:
        item["execution_status"] = _section_status(
            item["required_artifacts"],
            pending=item.get("pending_work"),
        )
        item.update(_readiness_layers(
            item["required_artifacts"],
            pending=item.get("pending_work"),
        ))

    supplementary = [
        {
            "name": "Performance across finite SNR levels",
            "source_artifact": (
                "08_robustness_sensitivity_target_snr/6_1_noise_robustness"
            ),
            "paper_role": (
                "Section 5 SNR-response or supplementary severity-curve "
                "material, not a standalone Section 6 main subsection."
            ),
            "exists": _exists(
                algorithm_root,
                "08_robustness_sensitivity_target_snr/6_1_noise_robustness/"
                "section_6_1_noise_robustness_aggregate.json",
            ),
        },
        {
            "name": "Boundary sensitivity",
            "source_artifact": (
                "08_robustness_sensitivity_target_snr/6_4_boundary_sensitivity"
            ),
            "paper_role": "Supplementary/protocol credibility check.",
            "exists": _exists(
                algorithm_root,
                "08_robustness_sensitivity_target_snr/6_4_boundary_sensitivity/"
                "section_6_4_boundary_sensitivity_aggregate.json",
            ),
        },
        {
            "name": "Noise-realization decomposition stability",
            "source_artifact": "16w_v554_noise_realization_stability",
            "qualification_artifact": "16x_v555_noise_realization_stability_qualification",
            "paper_role": (
                "Supplementary/protocol stability analysis across Monte Carlo "
                "noise realizations; not fixed-input stochastic repeatability "
                "and not ranking-bearing."
            ),
            "exists": _exists(
                algorithm_root,
                "16w_v554_noise_realization_stability/"
                "v554_noise_realization_stability_dashboard.json",
            ),
            "qualified": _exists(
                algorithm_root,
                "16x_v555_noise_realization_stability_qualification/"
                "v555_noise_realization_stability_qualification_dashboard.json",
            ),
        },
        {
            "name": "Cross-realization waveform-level stability",
            "source_artifact": "16y_v556_cross_realization_waveform_stability",
            "paper_role": (
                "Supplementary protocol for a future representative subset rerun "
                "that persists full component waveforms and applies "
                "cross-realization Hungarian component matching; not primary, "
                "not ranking-bearing, and not fixed-input stochastic repeatability."
            ),
            "exists": _exists(
                algorithm_root,
                "16y_v556_cross_realization_waveform_stability/"
                "v556_cross_realization_waveform_stability_protocol.json",
            ),
            "execution_status": "protocol_only_execution_pending",
        },
        {
            "name": "Primary component correspondence sensitivity",
            "source_artifact": "16ac_v560_matching_sensitivity_protocol",
            "paper_role": (
                "Supplementary/protocol credibility audit for thresholded "
                "Hungarian and mutual-nearest-neighbor alternatives to the "
                "frozen unthresholded primary estimated-vs-truth Hungarian "
                "matching rule; not primary, not ranking-bearing, and not a "
                "replacement for the 13-endpoint schema."
            ),
            "exists": _exists(
                algorithm_root,
                "16ac_v560_matching_sensitivity_protocol/"
                "v560_matching_sensitivity_protocol.json",
            ),
            "execution_status": "protocol_only_execution_pending",
        },
        {
            "name": "Primary component correspondence sensitivity subset",
            "source_artifact": "16ad_v560_matching_sensitivity_subset",
            "paper_role": (
                "Small supplementary subset execution checking whether method "
                "ordering, IRMF-vs-baseline effect directions, matched corr/NRMSE "
                "patterns, and thresholded unmatched-energy patterns are materially "
                "preserved under thresholded Hungarian and mutual-nearest-neighbor "
                "correspondence alternatives."
            ),
            "exists": _exists(
                algorithm_root,
                "16ad_v560_matching_sensitivity_subset/"
                "v560_matching_sensitivity_subset_dashboard.json",
            ),
            "execution_status": "small_subset_execution_if_present_not_primary",
        },
        {
            "name": "Section 6 challenging-signal extension",
            "source_artifact": "16z_v557_section6_challenging_signal_extension",
            "paper_role": (
                "Protocol amendment extending signal-difficulty coverage for "
                "Section 6.1-6.3 and adding one challenging scaling check to "
                "6.4; no primary metric or ranking change."
            ),
            "exists": _exists(
                algorithm_root,
                "16z_v557_section6_challenging_signal_extension/"
                "v557_section6_challenging_signal_extension_dashboard.json",
            ),
            "execution_status": "protocol_amendment_execution_pending",
        },
        {
            "name": "V5.57 pre-execution audit",
            "source_artifact": "16aa_v558_section6_pre_execution_audit",
            "paper_role": (
                "Protocol audit confirming signal taxonomy, true-component "
                "availability, expected-count accounting, common-anchor policy, "
                "and claim-gate continuity before V5.57 reruns."
            ),
            "exists": _exists(
                algorithm_root,
                "16aa_v558_section6_pre_execution_audit/"
                "v558_section6_pre_execution_audit_dashboard.json",
            ),
            "execution_status": "pre_execution_audit_only_no_algorithm_runs",
        },
        {
            "name": "V5.57 post-execution qualification",
            "source_artifact": "16ab_v559_section6_post_execution_qualification",
            "paper_role": (
                "Post-rerun qualification gate checking V5.57 execution counts, "
                "Section 6.1 case-blocked factorial scope, contamination and "
                "runtime execution integrity, and canonical/challenging wording cues."
            ),
            "exists": _exists(
                algorithm_root,
                "16ab_v559_section6_post_execution_qualification/"
                "v559_section6_post_execution_qualification_dashboard.json",
            ),
            "execution_status": "post_execution_gate_pending_v557_reruns",
        },
        {
            "name": "Runtime instrumentation smoke",
            "source_artifact": "16j_v538_section6_4_runtime_instrumentation_smoke",
            "paper_role": (
                "Implementation-integrity evidence only until full Section 6.4 "
                "runtime/scaling execution exists."
            ),
            "exists": _exists(
                algorithm_root,
                "16j_v538_section6_4_runtime_instrumentation_smoke/"
                "v538_section6_4_smoke_status.json",
            ),
        },
    ]

    n_ready = sum(
        1 for item in sections
        if item["execution_status"] == "main_text_ready_from_existing_target_snr_artifacts"
    )
    n_partial = sum(
        1 for item in sections
        if item["execution_status"] == "partial_execution_present_with_declared_pending_work"
    )
    status = {
        "schema_version": V550_SECTION6_MAIN_TEXT_ASSEMBLY_VERSION,
        "created_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "assembly_status": (
            "four_part_structure_recorded_with_pending_full_modules"
            if n_partial else "four_part_structure_ready"
        ),
        "main_text_section_count": len(sections),
        "main_text_ready_sections": int(n_ready),
        "partial_sections": int(n_partial),
        "execution_ready_sections": int(sum(1 for item in sections if item["execution_ready"])),
        "scientific_ready_sections": int(sum(1 for item in sections if item["scientific_ready"])),
        "claim_ready_sections": int(sum(1 for item in sections if item["claim_ready"])),
        "missing_sections": int(
            sum(1 for item in sections if item["execution_status"] == "missing_required_artifacts")
        ),
        "governance_principle": (
            "Main text contains robustness of scientific conclusions; "
            "supplement/protocol material contains implementation and "
            "evaluation credibility checks."
        ),
        "claim_boundary": (
            "This artifact is an assembly/provenance map. It does not create "
            "new performance evidence and does not authorize claims for "
            "sections marked partial."
        ),
        "sections": sections,
        "supplementary_protocol_checks": supplementary,
    }

    flat_sections = []
    for item in sections:
        flat_sections.append({
            "section": item["section"],
            "title": item["title"],
            "execution_status": item["execution_status"],
            "main_text_role": item["main_text_role"],
            "n_required_artifacts": len(item["required_artifacts"]),
            "n_existing_required_artifacts": sum(
                1 for artifact in item["required_artifacts"] if artifact["exists"]
            ),
            "n_pending_work_items": len(item.get("pending_work", [])),
            "execution_ready": item["execution_ready"],
            "scientific_ready": item["scientific_ready"],
            "claim_ready": item["claim_ready"],
            "pending_work": "; ".join(item.get("pending_work", [])),
            "output_mapping_note": item["output_mapping_note"],
        })

    write_json(status, output_root / "v550_section6_main_text_assembly_dashboard.json")
    write_csv(flat_sections, output_root / "v550_section6_main_text_sections.csv")
    write_csv(supplementary, output_root / "v550_section6_supplementary_protocol_checks.csv")
    return status
