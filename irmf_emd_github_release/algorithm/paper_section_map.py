#!/usr/bin/python
# coding: UTF-8
"""Map generated outputs to the frozen manuscript structure.

This module does not run experiments.  It writes a paper-facing index that
answers a practical question: which output files support each manuscript
section and appendix?
"""
from pathlib import Path
import json
from datetime import datetime, timezone

from experiments.paper_pipeline_utils import ensure_dir, write_json
from project_config import (
    EVALUATION_FRAMEWORK_FROZEN_DATE,
    EVALUATION_FRAMEWORK_STATUS,
    EVALUATION_FRAMEWORK_VERSION,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


POSITIONING = {
    "evaluation_framework_version": EVALUATION_FRAMEWORK_VERSION,
    "evaluation_framework_status": EVALUATION_FRAMEWORK_STATUS,
    "evaluation_framework_frozen_date": EVALUATION_FRAMEWORK_FROZEN_DATE,
    "frozen_positioning": (
        "This paper develops a reproducible benchmark for comparing robust "
        "signal decomposition methods under a locked-parameter evaluation "
        "protocol and incorporates a truth-aware validation strategy for "
        "synthetic, semi-synthetic, and real-world settings. The benchmark is "
        "demonstrated through a systematic comparison of an operationalized "
        "Spokoiny-inspired IRMF procedure with EMD-family methods."
    ),
    "originality_boundary": {
        "spokoiny_inspired_irmf_framework": "existing theoretical foundation / object of study",
        "operationalized_irmf_procedure": "reproducible computational realization used as an experimental object",
        "locked_parameter_evaluation_protocol": "benchmark-specific protocol contribution",
        "truth_aware_validation_strategy": "validation strategy incorporated into the benchmark",
        "unified_benchmark_and_statistics": "core benchmark contribution",
    },
    "wording_policy": {
        "avoid": [
            "We propose IRMF",
            "Our IRMF method",
            "The proposed IRMF algorithm",
            "This is the first IRMF implementation",
        ],
        "prefer": [
            "the implemented IRMF procedure",
            "the IRMF procedure considered in this study",
            "a Spokoiny-inspired IRMF procedure",
            "our hierarchical fixed-parameter evaluation methodology",
        ],
    },
}


EVIDENCE_HIERARCHY = [
    {
        "scientific_question": "Are parameters pre-specified?",
        "evidence_layer": "Protocol development",
        "representative_experiment": "40-case development protocol",
        "primary_outcome": "locked global method configurations",
    },
    {
        "scientific_question": "What can be concluded when ground truth is fully known?",
        "evidence_layer": "Synthetic benchmark",
        "representative_experiment": "canonical and challenging factorial synthetic benchmark",
        "primary_outcome": "signal recovery, component recovery, robustness, and statistical comparison",
    },
    {
        "scientific_question": "Are conclusions formula-dependent?",
        "evidence_layer": "Signal-family reproducibility",
        "representative_experiment": "canonical signal variants",
        "primary_outcome": "within-family reproducibility, difficulty-gradient robustness, and rank consistency",
    },
    {
        "scientific_question": "Do recovery claims transfer to real signal backgrounds when approximate truth is available?",
        "evidence_layer": "Semi-synthetic bridge validation",
        "representative_experiment": "real backgrounds with controlled synthetic contamination",
        "primary_outcome": "approximate-truth recovery, oracle upper bound, and adaptivity gap",
    },
    {
        "scientific_question": "Does evidence transfer to truth-free real-world signals?",
        "evidence_layer": "Pure real-world validation",
        "representative_experiment": "ECG and bearing truth-free validation",
        "primary_outcome": "internal validity, domain-specific validity, reproducibility, and practical utility",
    },
]


def _result(path):
    return {"kind": "result", "path": path}


def _code(path):
    return {"kind": "code", "path": path}


PAPER_STRUCTURE = [
    {
        "id": "section_01_introduction",
        "label": "1",
        "title": "Introduction",
        "scientific_role": "Freeze the originality boundary and state the empirical evaluation gap.",
        "subsections": [
            "1.1 Motivation",
            "1.2 Existing robust multiscale filtering and EMD-family methods",
            "1.3 Empirical evaluation gap and scientific questions",
            "1.4 Contributions",
        ],
        "sources": [],
    },
    {
        "id": "section_02_operationalized_irmf_procedure",
        "label": "2",
        "title": "Statistical Model and Operationalized IRMF Procedure",
        "scientific_role": "Describe the object of study without claiming a new IRMF theory.",
        "subsections": [
            "2.1 Signal-plus-noise model",
            "2.2 Spokoiny-inspired robust multiscale filtering framework",
            "2.3 Gaussian-smoothed median loss and local objective",
            "2.4 Operationalized IRMF procedure",
            "2.5 EMD, EEMD, and CEEMDAN baselines",
            "2.6 Fixed-parameter and no-per-case-tuning protocol",
        ],
        "sources": [
            _code("core_algorithms/strict_spokoiny_irmf.py"),
            _code("core_algorithms/robust_losses.py"),
            _code("core_algorithms/emd_wrapper.py"),
            _code("core_algorithms/eemd_wrapper.py"),
            _code("core_algorithms/ceemdan_wrapper.py"),
            _code("diagnostics/shared_physical_diagnostics.py"),
            _result("methodology/00_loss_properties"),
            _result("methodology/05_loss_regime_map"),
        ],
    },
    {
        "id": "section_03_evaluation_methodology",
        "label": "3",
        "title": "Evaluation Methodology: Hierarchical Fixed-Parameter Evidence Design",
        "scientific_role": "Define the reusable evaluation methodology, not merely a benchmark description.",
        "subsections": [
            "3.1 Scientific questions and evidence hierarchy",
            "3.2 Development-set protocol and parameter locking",
            "3.3 Factorial evaluation design",
            "3.4 Evaluation metric taxonomy",
            "3.5 Statistical inference protocol",
        ],
        "sources": [
            _code("project_config.py"),
            _code("experiments/experiment_global_parameter_selection.py"),
            _code("experiments/experiment_unified_benchmark_cube.py"),
            _code("experiments/experiment_unified_cube_statistics.py"),
            _code("experiments/experiment_structural_metric_validation.py"),
            _result("algorithm/locked_algorithm_parameters.json"),
            _result("algorithm/locked_parameter_audit.json"),
            _result("algorithm/03_unified_benchmark_cube/unified_benchmark_cube_protocol.json"),
            _result("algorithm/03_unified_benchmark_cube/unified_benchmark_cube_dashboard.json"),
            _result("algorithm/03_unified_benchmark_cube/monte_carlo_cell_summary.csv"),
            _result("algorithm/03_unified_benchmark_cube/paired_method_difference_summary.csv"),
            _result("algorithm/07_unified_cube_statistics/unified_cube_statistics_dashboard.json"),
            _result("algorithm/16_parameter_transfer_analysis/01_development_test_independence"),
        ],
    },
    {
        "id": "section_04_locked_parameter_protocol",
        "label": "4",
        "title": "Locked-Parameter Protocol Development",
        "scientific_role": "Document development-set parameter selection and the no-per-case-tuning lock before benchmark evaluation.",
        "subsections": [
            "4.1 Development-set design rationale",
            "4.2 Global parameter selection",
            "4.3 Baseline calibration under the same locked protocol",
            "4.4 Parameter-lock audit and reproducibility record",
        ],
        "sources": [
            _code("experiments/experiment_global_parameter_selection.py"),
            _code("experiments/experiment_v539_irmf_h_tuning_amendment.py"),
            _result("algorithm/01_global_parameter_selection"),
            _result("algorithm/16k_v539_irmf_h_tuning_amendment/v539_irmf_h_tuning_amendment.json"),
            _result("algorithm/02_emd_family_calibration"),
            _result("algorithm/locked_algorithm_parameters.json"),
            _result("algorithm/locked_parameter_audit.json"),
            _result("algorithm/16_parameter_transfer_analysis/01_development_test_independence"),
        ],
    },
    {
        "id": "section_05_synthetic_benchmark",
        "label": "5",
        "title": "Synthetic Benchmark",
        "scientific_role": "Known-truth evaluation of signal recovery, component recovery, robustness, challenging structure, and statistical comparison.",
        "subsections": [
            "5.1 Main factorial benchmark",
            "5.2 Robustness analyses",
            "5.3 Challenging structural regimes",
            "5.4 Statistical comparisons",
            "5.5 Integrated synthetic evidence summary",
        ],
        "sources": [
            _code("experiments/experiment_unified_benchmark_cube.py"),
            _code("experiments/experiment_unified_cube_statistics.py"),
            _code("experiments/experiment_controlled_challenging_diagnostics.py"),
            _code("experiments/experiment_v540_time_frequency_secondary_schema.py"),
            _result("algorithm/03_unified_benchmark_cube/section_5_canonical_main_slice"),
            _result("algorithm/03_unified_benchmark_cube/section_6_1_full_noise_robustness_slice"),
            _result("algorithm/03_unified_benchmark_cube/section_8_challenging_generalization_slice"),
            _result("algorithm/03_unified_benchmark_cube/robustness_auc_by_trajectory.csv"),
            _result("algorithm/03_unified_benchmark_cube/robustness_auc_aggregate.csv"),
            _result("algorithm/03_unified_benchmark_cube/failure_rates_by_cell.csv"),
            _result("algorithm/03_unified_benchmark_cube/monte_carlo_cell_summary.csv"),
            _result("algorithm/03_unified_benchmark_cube/paired_method_difference_summary.csv"),
            _result("algorithm/03_unified_benchmark_cube/seed_stability_summary.csv"),
            _result("algorithm/03_unified_benchmark_cube/method_win_tie_loss.csv"),
            _result("algorithm/07_unified_cube_statistics/method_metric_summary.csv"),
            _result("algorithm/07_unified_cube_statistics/dimension_average_ranks.csv"),
            _result("algorithm/07_unified_cube_statistics/paired_effect_sizes.csv"),
            _result("algorithm/07_unified_cube_statistics/friedman_cd_rank_summary.csv"),
            _result("algorithm/07_unified_cube_statistics/conditional_endpoint_summary.csv"),
            _result("algorithm/07_unified_cube_statistics/sigma_degradation_summary.csv"),
            _result("algorithm/07_unified_cube_statistics/sigma_degradation_trajectories.csv"),
            _result("algorithm/07e_structural_metric_validation/structural_metric_validation_dashboard.json"),
            _result("algorithm/07e_structural_metric_validation/range_audit.csv"),
            _result("algorithm/07e_structural_metric_validation/gating_audit.csv"),
            _result("algorithm/07e_structural_metric_validation/imf_count_dependency_audit.csv"),
            _result("algorithm/07e_structural_metric_validation/association_threshold_sensitivity_summary.csv"),
            _result("algorithm/16l_v540_time_frequency_secondary_schema/v540_time_frequency_secondary_schema.json"),
            _result("algorithm/08_robustness_sensitivity/6_2_contamination_robustness"),
            _result("algorithm/12_challenging_signal_suite"),
            _result("algorithm/13_controlled_challenging_diagnostics"),
        ],
        "legacy_output_note": (
            "Some output directories retain earlier section names such as "
            "section_6_1_full_noise_robustness_slice and "
            "section_8_challenging_generalization_slice. In the frozen evidence "
            "continuum these are known-truth synthetic benchmark slices."
        ),
    },
    {
        "id": "section_06_sensitivity_robustness",
        "label": "6",
        "title": "Sensitivity and Robustness Analyses",
        "scientific_role": (
            "Answer the main reviewer-facing robustness questions without turning "
            "the manuscript into an exhaustive audit catalogue. Endpoint-level "
            "and construct-level results remain the primary evidence; overall "
            "ranks are secondary summaries."
        ),
        "subsections": [
            "6.1 IRMF parameter sensitivity and robustness",
            "6.2 Contamination-design robustness",
            "6.3 Comparator and signal robustness",
            "6.4 Computational efficiency and scaling",
        ],
        "sources": [
            _code("experiments/experiment_v532_section6_scope.py"),
            _code("experiments/experiment_v533_irmf_parameter_sensitivity_protocol.py"),
            _code("experiments/experiment_v534_section6_executable_protocols.py"),
            _code("experiments/experiment_v535_section6_1_irmf_parameter_smoke.py"),
            _code("experiments/experiment_v536_section6_2_contamination_smoke.py"),
            _code("experiments/experiment_v537_section6_3_comparator_signal_smoke.py"),
            _code("experiments/experiment_v538_section6_4_runtime_smoke.py"),
            _code("experiments/experiment_v539_irmf_h_tuning_amendment.py"),
            _code("experiments/experiment_v550_section6_main_text_assembly.py"),
            _code("experiments/experiment_v551_section6_2_contamination_design.py"),
            _code("experiments/experiment_v552_section6_4_computational_scaling.py"),
            _code("experiments/experiment_v553_section6_claim_qualification.py"),
            _code("experiments/experiment_robustness_sensitivity.py"),
            _code("experiments/experiment_signal_variant_robustness.py"),
            _code("experiments/experiment_emd_family_benchmark.py"),
            _code("sensitivity_analysis/emd_family_sensitivity_analysis.py"),
            _result("algorithm/16d_v532_section6_scope/v532_section6_scope_protocol.json"),
            _result("algorithm/16e_v533_irmf_parameter_sensitivity/v533_irmf_parameter_sensitivity_protocol.json"),
            _result("algorithm/16f_v534_section6_executable_protocols/v534_section6_executable_protocols.json"),
            _result("algorithm/16g_v535_section6_1_irmf_parameter_smoke/v535_section6_1_smoke_status.json"),
            _result("algorithm/16h_v536_section6_2_contamination_design_smoke/v536_section6_2_smoke_status.json"),
            _result("algorithm/16i_v537_section6_3_comparator_signal_smoke/v537_section6_3_smoke_status.json"),
            _result("algorithm/16j_v538_section6_4_runtime_instrumentation_smoke/v538_section6_4_smoke_status.json"),
            _result("algorithm/16k_v539_irmf_h_tuning_amendment/v539_irmf_h_tuning_status.json"),
            _result("algorithm/16s_v550_section6_main_text_assembly/v550_section6_main_text_assembly_dashboard.json"),
            _result("algorithm/16t_v551_section6_2_contamination_design/v551_section6_2_contamination_design_dashboard.json"),
            _result("algorithm/16u_v552_section6_4_computational_scaling/v552_section6_4_runtime_scaling_dashboard.json"),
            _result("algorithm/16v_v553_section6_claim_qualification/v553_section6_claim_qualification_dashboard.json"),
            _result("algorithm/08_robustness_sensitivity_target_snr"),
            _result("algorithm/09_signal_variant_robustness_target_snr"),
            _result("algorithm/10_result_sensitivity"),
            _result("algorithm/11_emd_family_sensitivity_target_snr"),
        ],
        "legacy_output_note": (
            "Signal-family variants and EMD-family baseline checks are now "
            "integrated into Section 6.3 in the compact main-text structure. "
            "Detailed grids, distributions, seed convergence, discretization, "
            "association-threshold sensitivity, optional ranking robustness, and "
            "SNR realization audits belong in supplementary/protocol material."
        ),
    },
    {
        "id": "section_07_semisynthetic_bridge_validation",
        "label": "7",
        "title": "Semi-synthetic Bridge Validation",
        "scientific_role": "Bridge known-truth synthetic evaluation and truth-free real-world validation using real signal backgrounds with controlled contamination.",
        "subsections": [
            "7.1 Data construction",
            "7.2 Approximate-truth evaluation",
            "7.3 Fixed vs. oracle upper-bound comparison",
            "7.4 Adaptivity gap",
        ],
        "sources": [
            _code("experiments/experiment_real_data_semisynthetic.py"),
            _code("experiments/experiment_proxy_metric_validation.py"),
            _result("algorithm/07d_proxy_metric_validation"),
            _result("algorithm/14_semisynthetic_bridge_validation/section_7_semisynthetic_bridge_protocol.json"),
            _result("algorithm/14_semisynthetic_bridge_validation/section_7_semisynthetic_bridge_validation.csv"),
            _result("algorithm/14_oracle_adaptivity_analysis"),
        ],
        "legacy_output_note": (
            "If older result folders contain section_8b semi-synthetic filenames, "
            "they correspond to this Section 7 bridge-validation layer."
        ),
    },
    {
        "id": "section_08_pure_real_world_validation",
        "label": "8",
        "title": "Real-world Validation",
        "scientific_role": (
            "Truth-free validation using internal validity, domain-specific "
            "validity, robustness/reproducibility, practical utility, and "
            "event-locked residual-leakage evidence. No true-IMF, clean-signal "
            "NMSE, or oracle claims are made here."
        ),
        "subsections": [
            "8.1 Internal validity",
            "8.2 Domain-specific validity",
            "8.3 Robustness and reproducibility",
            "8.4 Practical utility",
            "8.5 Cross-domain truth-free synthesis",
        ],
        "sources": [
            _code("diagnostics/real_proxy_diagnostics.py"),
            _code("experiments/experiment_real_data_protocol.py"),
            _code("experiments/experiment_real_world_validation.py"),
            _result("algorithm/07d_proxy_metric_validation"),
            _result("algorithm/15a_real_data_protocol/real_data_protocol.yaml"),
            _result("algorithm/15a_real_data_protocol/split_registry.yaml"),
            _result("algorithm/15a_real_data_protocol/mitbih_record_subject_map.yaml"),
            _result("algorithm/15a_real_data_protocol/protocol_status.yaml"),
            _result("algorithm/15a_real_data_protocol/locked_reconstruction_rule.json"),
            _result("algorithm/15a_real_data_protocol/section_8_real_data_design_spec.json"),
            _result("algorithm/15a_real_data_protocol/section_8_audit_checklist.csv"),
            _result("algorithm/15a_real_data_protocol/section_8_evidence_matrix.csv"),
            _result("algorithm/15a_real_data_protocol/split_audit_report.json"),
            _result("algorithm/15a_real_data_protocol/split_audit_report.csv"),
            _result("algorithm/15b_real_data_split_audit/split_audit_report.json"),
            _result("algorithm/15b_real_data_split_audit/split_audit_report.csv"),
            _result("algorithm/15c_real_data_development_run/development_run_audit.json"),
            _result("algorithm/15c_real_data_development_run/development_run_audit.csv"),
            _result("algorithm/15c_real_data_development_run/development_run_status.json"),
            _result("algorithm/15c_real_data_development_run/development_run_status.yaml"),
            _result("algorithm/15c_real_data_development_run/environment_manifest.json"),
            _result("algorithm/15c_real_data_development_run/dataset_integrity_report.json"),
            _result("algorithm/15c_real_data_development_run/dataset_integrity_report.yaml"),
            _result("algorithm/15c_real_data_development_run/development_access_log.csv"),
            _result("algorithm/15c_real_data_development_run/development_access_log.json"),
            _result("algorithm/15_real_world_validation"),
            _result("algorithm/real_world_validation_dashboard.json"),
        ],
    },
    {
        "id": "section_09_discussion",
        "label": "9",
        "title": "Discussion",
        "scientific_role": "Interpret scientific and practical implications and the evaluation methodology itself.",
        "subsections": [
            "9.1 Scientific implications",
            "9.2 Practical implications",
            "9.3 Lessons for benchmark design",
            "9.4 Limitations",
            "9.5 Future adaptive IRMF directions",
            "9.6 Generalizability of the evaluation methodology",
        ],
        "sources": [
            _result("algorithm/07_unified_cube_statistics/unified_cube_statistics_dashboard.json"),
            _result("algorithm/14_oracle_adaptivity_analysis"),
            _result("algorithm/07d_proxy_metric_validation/proxy_metric_validation_dashboard.json"),
        ],
    },
    {
        "id": "section_10_conclusion",
        "label": "10",
        "title": "Conclusion",
        "scientific_role": "Close with the fixed-parameter empirical findings and reusable evaluation-methodology contribution.",
        "subsections": [],
        "sources": [
            _result("paper_sections/paper_section_manifest.json"),
        ],
    },
]


APPENDICES = [
    {
        "id": "appendix_A_spokoiny_theory_diagnostics",
        "label": "Appendix A",
        "title": "Spokoiny Theory Diagnostics and Implementation Fidelity",
        "sources": [
            _code("experiments/experiment_spokoiny_theory_validation.py"),
            _code("diagnostics/irmf_local_theory_diagnostics.py"),
            _code("diagnostics/irmf_operator_diagnostics.py"),
            _result("algorithm/appendix_A_spokoiny_theory_validation"),
            _result("algorithm/14_oracle_adaptivity_analysis"),
        ],
    },
    {
        "id": "appendix_B_oracle_adaptivity_gap",
        "label": "Appendix B",
        "title": "Oracle IRMF and Adaptivity-Gap Analysis",
        "sources": [
            _code("experiments/experiment_oracle_adaptivity_analysis.py"),
            _result("algorithm/14_oracle_adaptivity_analysis"),
            _result("algorithm/oracle_adaptivity_dashboard.json"),
        ],
    },
    {
        "id": "appendix_C_parameter_protocol_calibration",
        "label": "Appendix C",
        "title": "Parameter Protocol Calibration Details",
        "sources": [
            _result("algorithm/01_global_parameter_selection"),
            _result("algorithm/02_emd_family_calibration"),
            _result("algorithm/16_parameter_transfer_analysis"),
            _result("algorithm/locked_algorithm_parameters.json"),
            _result("algorithm/locked_parameter_audit.json"),
        ],
    },
    {
        "id": "appendix_D_metric_definitions_taxonomy",
        "label": "Appendix D",
        "title": "Metric Definitions and Taxonomy",
        "sources": [
            _code("experiments/experiment_metric_taxonomy.py"),
            _code("diagnostics/shared_physical_diagnostics.py"),
            _code("diagnostics/configuration_level_scoring.py"),
            _result("algorithm/05b_metric_taxonomy"),
            _result("algorithm/07_unified_cube_statistics/conditional_endpoint_values.csv"),
        ],
    },
    {
        "id": "appendix_E_full_benchmark_tables",
        "label": "Appendix E",
        "title": "Full Benchmark Tables",
        "sources": [
            _result("algorithm/03_unified_benchmark_cube/unified_benchmark_cube_rows.csv"),
            _result("algorithm/03_unified_benchmark_cube/unified_benchmark_cube_rows.json"),
            _result("algorithm/03_unified_benchmark_cube/unified_benchmark_cube_rows.jsonl"),
            _result("algorithm/03_unified_benchmark_cube/unified_benchmark_cube_aggregate.json"),
            _result("algorithm/03_unified_benchmark_cube/monte_carlo_cell_summary.csv"),
            _result("algorithm/03_unified_benchmark_cube/paired_method_difference_summary.csv"),
            _result("algorithm/03_unified_benchmark_cube/seed_stability_summary.csv"),
            _result("algorithm/03_unified_benchmark_cube/method_win_tie_loss.csv"),
            _result("algorithm/03_unified_benchmark_cube/failure_rates_by_cell.csv"),
        ],
    },
    {
        "id": "appendix_F_signal_family_variant_details",
        "label": "Appendix F",
        "title": "Signal-Family Variant Details",
        "sources": [
            _code("experiments/experiment_signal_variant_robustness.py"),
            _result("algorithm/09_signal_variant_robustness_target_snr"),
        ],
    },
    {
        "id": "appendix_G_emd_family_baseline_sensitivity",
        "label": "Appendix G",
        "title": "EMD-Family Baseline Sensitivity",
        "sources": [
            _code("experiments/experiment_emd_family_benchmark.py"),
            _code("sensitivity_analysis/emd_family_sensitivity_analysis.py"),
            _result("algorithm/11_emd_family_sensitivity_target_snr"),
        ],
    },
    {
        "id": "appendix_H_real_data_proxy_calibration",
        "label": "Appendix H",
        "title": "Real-Data Proxy Calibration",
        "sources": [
            _result("algorithm/07d_proxy_metric_validation/proxy_ground_truth_calibration.csv"),
            _result("algorithm/07d_proxy_metric_validation/validated_real_proxy_metrics.csv"),
            _result("algorithm/07d_proxy_metric_validation/proxy_metric_validation_dashboard.json"),
        ],
    },
    {
        "id": "appendix_I_reproducibility_package",
        "label": "Appendix I",
        "title": "Reproducibility Package",
        "sources": [
            _code("paper_pipeline.py"),
            _code("algorithm/pipeline.py"),
            _code("methodology/pipeline.py"),
            _code("project_config.py"),
            _result("algorithm/locked_algorithm_parameters.json"),
            _result("algorithm/locked_parameter_audit.json"),
            _result("algorithm/run_manifest.json"),
            _result("methodology/run_manifest.json"),
            _result("paper_sections/paper_section_manifest.json"),
        ],
    },
]


def _resolve_source(result_root, source):
    rel = Path(source["path"])
    if source["kind"] == "result":
        path = result_root / rel
    else:
        path = PROJECT_ROOT / rel
    generated_by_this_stage = rel.as_posix() == "paper_sections/paper_section_manifest.json"
    exists = path.exists() or generated_by_this_stage
    record = dict(source)
    record.update({
        "absolute_path": str(path),
        "exists": bool(exists),
        "is_dir": bool(path.is_dir()) if path.exists() else False,
    })
    if path.exists() and path.is_file():
        record["size_bytes"] = int(path.stat().st_size)
    if generated_by_this_stage:
        record["generated_by_section_map"] = True
    return record


def _materialize_entry(section_root, entry, records):
    out_dir = ensure_dir(section_root / entry["id"])
    write_json({
        "id": entry["id"],
        "label": entry["label"],
        "title": entry["title"],
        "scientific_role": entry.get("scientific_role"),
        "subsections": entry.get("subsections", []),
        "legacy_output_note": entry.get("legacy_output_note"),
        "sources": records,
        "all_required_sources_present": all(item["exists"] for item in records),
    }, out_dir / "sources.json")
    lines = [
        f"# {entry['label']}. {entry['title']}",
        "",
    ]
    if entry.get("scientific_role"):
        lines.extend([entry["scientific_role"], ""])
    if entry.get("subsections"):
        lines.append("## Subsections")
        lines.extend(f"- {item}" for item in entry["subsections"])
        lines.append("")
    if entry.get("legacy_output_note"):
        lines.extend(["## Note", entry["legacy_output_note"], ""])
    lines.append("## Sources")
    for item in records:
        status = "present" if item["exists"] else "missing"
        lines.append(f"- [{status}] `{item['kind']}` `{item['path']}`")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paper_section_map(result_root):
    """Write a paper-facing manifest under ``paper_sections/``.

    Parameters
    ----------
    result_root:
        The top-level run directory containing ``algorithm/`` and optionally
        ``methodology/``.  If an ``algorithm/`` directory is passed directly,
        the parent directory is used.
    """
    result_root = Path(result_root)
    if result_root.name == "algorithm":
        result_root = result_root.parent
    result_root = ensure_dir(result_root)
    section_root = ensure_dir(result_root / "paper_sections")

    sections = []
    for entry in PAPER_STRUCTURE:
        records = [_resolve_source(result_root, item) for item in entry.get("sources", [])]
        item = {
            "id": entry["id"],
            "label": entry["label"],
            "title": entry["title"],
            "scientific_role": entry.get("scientific_role"),
            "subsections": entry.get("subsections", []),
            "legacy_output_note": entry.get("legacy_output_note"),
            "sources": records,
            "all_required_sources_present": all(record["exists"] for record in records),
        }
        sections.append(item)
        _materialize_entry(section_root, entry, records)

    appendices = []
    appendix_root = ensure_dir(section_root / "appendices")
    for entry in APPENDICES:
        records = [_resolve_source(result_root, item) for item in entry.get("sources", [])]
        item = {
            "id": entry["id"],
            "label": entry["label"],
            "title": entry["title"],
            "sources": records,
            "all_required_sources_present": all(record["exists"] for record in records),
        }
        appendices.append(item)
        _materialize_entry(appendix_root, entry, records)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "result_root": str(result_root),
        "project_root": str(PROJECT_ROOT),
        "evaluation_framework_version": EVALUATION_FRAMEWORK_VERSION,
        "evaluation_framework_status": EVALUATION_FRAMEWORK_STATUS,
        "evaluation_framework_frozen_date": EVALUATION_FRAMEWORK_FROZEN_DATE,
        "positioning": POSITIONING,
        "evidence_hierarchy": EVIDENCE_HIERARCHY,
        "paper_sections": sections,
        "appendices": appendices,
        "single_source_of_truth": (
            "algorithm/03_unified_benchmark_cube/unified_benchmark_cube_rows.csv"
        ),
        "section_map_policy": (
            "This map aligns current output directories with the frozen "
            "manuscript structure. Missing entries indicate stages that have "
            "not been run yet, not failed scientific claims."
        ),
    }
    write_json(manifest, section_root / "paper_section_manifest.json")

    readme_lines = [
        "# Paper Section Output Map",
        "",
        f"Evaluation framework: `{EVALUATION_FRAMEWORK_VERSION}` "
        f"(status: {EVALUATION_FRAMEWORK_STATUS}; frozen date: {EVALUATION_FRAMEWORK_FROZEN_DATE})",
        "",
        POSITIONING["frozen_positioning"],
        "",
        "## Evidence Hierarchy",
        "",
        "| Scientific question | Evidence layer | Representative experiment | Primary outcome |",
        "|---|---|---|---|",
    ]
    for row in EVIDENCE_HIERARCHY:
        readme_lines.append(
            f"| {row['scientific_question']} | {row['evidence_layer']} | "
            f"{row['representative_experiment']} | {row['primary_outcome']} |"
        )
    readme_lines.extend(["", "## Manuscript Sections", ""])
    for item in sections:
        status = "complete" if item["all_required_sources_present"] else "partial"
        readme_lines.append(f"- {item['label']}. {item['title']} — {status}")
    readme_lines.extend(["", "## Appendices", ""])
    for item in appendices:
        status = "complete" if item["all_required_sources_present"] else "partial"
        readme_lines.append(f"- {item['label']}. {item['title']} — {status}")
    readme_lines.extend([
        "",
        "See `paper_section_manifest.json` and each section subdirectory for exact source paths.",
    ])
    (section_root / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    return manifest
