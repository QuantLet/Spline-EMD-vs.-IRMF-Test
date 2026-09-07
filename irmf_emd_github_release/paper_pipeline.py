#!/usr/bin/python
# coding: UTF-8
"""V5.7 two-layer reproducible research pipeline.

Scientific order:
  1. methodology - rho-function properties and within-IRMF loss experiments
  2. algorithm   - locked-loss IRMF versus EMD/EEMD/CEEMDAN
  3. paper-assets - contract validation and publication asset collection

The legacy V5.4/V5.5 mixed pipeline is available only through `legacy-paper`.
"""
from pathlib import Path
import argparse

from project_config import GLOBAL_IRMF_PARAMS
from methodology.pipeline import (
    run_loss_properties, run_loss_calibration, run_loss_ablation_stage,
    run_contamination_stage, run_optimization_stage, run_methodology_pipeline,
)

DEFAULT_ROOT = "IRMF_EMD_PAPER_RESULTS_V5_7_LOSS_REGIME_MAP"


def _parser():
    p = argparse.ArgumentParser(description="IRMF paper pipeline with separate methodology and algorithm layers.")
    p.add_argument("command", choices=(
        "loss-properties", "loss-calibration", "loss-ablation", "contamination",
        "optimization", "efficiency-calibration", "structure-contamination",
        "optimization-stress", "quadratic-diagnostics", "loss-regime-map", "methodology",
        "parameter-selection", "benchmark", "statistics", "sensitivity",
        "challenging-signals", "controlled-challenging", "oracle-adaptivity",
        "metric-taxonomy",
        "v526-primary-schema-qualification",
        "v527-primary-schema-qualification",
        "v527-freeze-primary-schema",
        "v527-formula-validity-audit",
        "v527-unified-statistics",
        "v527-directionality-audit",
        "v527-spillover-scale-audit",
        "v528-freeze-primary-schema",
        "v528-unified-statistics",
        "v528-final-statistics-validation",
        "v529-snr-design-draft",
        "v529-snr-realization-audit",
        "v529-freeze-snr-design",
        "v530-snr-design-draft",
        "v530-snr-realization-audit",
        "v530-freeze-snr-design",
        "v530-clean-baseline-schema",
        "v531-time-axis-semantics-audit",
        "v532-section6-scope",
        "v533-irmf-parameter-sensitivity-protocol",
        "v534-section6-executable-protocols",
        "v535-section6-1-irmf-parameter-smoke",
        "v536-section6-2-contamination-smoke",
        "v537-section6-3-comparator-signal-smoke",
        "v538-section6-4-runtime-smoke",
        "v539-irmf-h-tuning-amendment",
        "v540-time-frequency-secondary-schema",
        "v541-primary-secondary-closure",
        "v542-irmf-relative-h-parameterization",
        "v543-primary-metric-manual-audit",
        "v544-h-decision-gate",
        "v545-relative-h-scale-estimator-qualification",
        "v548-parameter-selection-adjudication",
        "v550-section6-main-text-assembly",
        "v551-section6-2-contamination-design",
        "v552-section6-4-computational-scaling",
        "v553-section6-claim-qualification",
        "v554-noise-realization-stability",
        "v555-noise-realization-stability-qualification",
        "v556-cross-realization-waveform-stability-protocol",
        "v557-section6-challenging-signal-extension",
        "v558-section6-pre-execution-audit",
        "v559-section6-post-execution-qualification",
        "v560-matching-sensitivity-protocol",
        "v560-matching-sensitivity-subset",
        "v561-section6-5-matching-rule-robustness",
        "v562-section6-reviewer-readiness-synthesis",
        "v563-section6-failure-region-dominance-synthesis",
        "v564-section6-3a-challenging-family-variant-extension",
        "v564a-section6-3a-canonical-anchor-reference",
        "v565-section6-3a-canonical-structured-perturbation",
        "v566-section6-3a-two-axis-interaction-design",
        "v567-section6-3a-analysis-qualification-freeze",
        "real-world-validation-schema-draft",
        "real-world-endpoint-formula-audit",
        "real-world-perturbation-protocol-freeze",
        "real-world-denominator-degeneracy-audit",
        "real-world-method-neutrality-audit",
        "real-world-validation-preflight",
        "real-world-v1-schema-smoke",
        "unified-cube", "unified-statistics", "proxy-validation",
        "benchmark-eligibility-gate", "benchmark-execution-audit",
        "ceemdan-noise-endpoint-adjudication",
        "rebuild-universal-noise-endpoints",
        "true-noise-separation-adapter-spec",
        "controlled-noise-endpoint-rerun-preflight",
        "synthetic-reconstruction-rule-discovery",
        "synthetic-reconstruction-protocol-amendment",
        "qualify-synthetic-reconstruction-protocol",
        "controlled-noise-endpoint-rerun",
        "noise-endpoint-merge-audit",
        "protocol-controlled-full-benchmark-rerun",
        "protocol-full-rerun-audit",
        "structural-metric-validation",
        "real-data-protocol", "real-data-split-audit",
        "real-data-development-run", "real-data-ecg-operational-smoke",
        "real-world-validation", "parameter-transfer", "paper-section-map",
        "algorithm", "paper-v518",
        "paper-assets", "paper", "legacy-paper",
    ))
    p.add_argument("--output-root", default=DEFAULT_ROOT)
    p.add_argument("--protocol-root", default=None,
                   help="Section 8 protocol result root for real-data-split-audit.")
    p.add_argument("--qualification-root", default=None,
                   help="V5.24 qualification result root for benchmark-eligibility-gate.")
    p.add_argument("--data-root", default=None,
                   help="MIT-BIH data root for real-data-development-run.")
    p.add_argument("--subject-map", default=None,
                   help="Optional record-to-subject CSV/JSON map for real-data-split-audit.")
    p.add_argument("--max-development-windows", type=int, default=None,
                   help="Optional smoke-run cap per development record for real-data-development-run.")
    p.add_argument("--real-data-dir", default=None)
    p.add_argument("--run-oracle", action="store_true")
    p.add_argument("--quick-regime-map", action="store_true",
                   help="Use a small seed subset for methodology loss-regime-map smoke runs.")
    p.add_argument("--skip-methodology", action="store_true",
                   help="For paper: reuse an existing completed methodology layer.")
    p.add_argument("--skip-algorithm", action="store_true",
                   help="For paper: reuse an existing completed algorithm layer.")
    p.add_argument("--skip-emd-family-calibration", action="store_true")
    p.add_argument("--skip-full-emd-family-benchmark", action="store_true")
    p.add_argument("--skip-emd-family-sensitivity", action="store_true")
    p.add_argument("--skip-challenging-signals", action="store_true")
    p.add_argument("--quick-challenging", action="store_true",
                   help="Run a small challenging-signal smoke benchmark.")
    p.add_argument("--quick-oracle", action="store_true",
                   help="Run a small oracle-adaptivity smoke analysis.")
    p.add_argument("--quick-unified-cube", action="store_true",
                   help="Run a small unified benchmark cube smoke test.")
    p.add_argument("--quick-structural-metric-validation", action="store_true",
                   help="Run a small V5.24 structural-metric validation smoke slice.")
    p.add_argument("--cube-seeds", default=None,
                   help="Comma-separated data seeds for unified-cube. Default: 0..19; V5.30 seed-convergence auditing up to 50 seeds is a separate qualification step.")
    p.add_argument("--timeout-seconds", type=int, default=120,
                   help="Per-method timeout threshold for unified-cube runs.")
    p.add_argument("--run-exploratory-ceemdan-oracle", action="store_true",
                   help="For oracle-adaptivity: also run exploratory IRMF-oracle vs CEEMDAN-oracle.")
    p.add_argument("--use-current-config-params", action="store_true",
                   help="Bypass Section 4 calibration and use project_config fixed parameters for compatible algorithm stages.")
    p.add_argument("--force-parameter-selection", action="store_true",
                   help="Rerun the current-code development-set parameter selection and overwrite locked_algorithm_parameters.json before eligible benchmark stages.")
    p.add_argument("--use-v530-target-snr-grid", action="store_true",
                   help="For protocol-controlled full rerun: use the frozen V5.30 regular 0:5:30 dB target-SNR input design.")
    return p


def main():
    args = _parser().parse_args()
    root = Path(args.output_root)
    method_root = root / "methodology"
    algorithm_root = root / "algorithm"
    cmd = args.command

    if cmd == "loss-properties":
        return run_loss_properties(method_root / "00_loss_properties", H=float(GLOBAL_IRMF_PARAMS.get("H", 1.0)))
    if cmd == "loss-calibration":
        return run_loss_calibration(method_root / "01_calibration")
    if cmd == "loss-ablation":
        return run_loss_ablation_stage(method_root)
    if cmd == "contamination":
        return run_contamination_stage(method_root)
    if cmd == "optimization":
        return run_optimization_stage(method_root)
    if cmd in {"efficiency-calibration", "structure-contamination", "optimization-stress", "quadratic-diagnostics", "loss-regime-map"}:
        from methodology.loss_regime_map.efficiency_calibration import run_efficiency_calibration
        from methodology.loss_regime_map.structure_contamination import run_structure_contamination
        from methodology.loss_regime_map.optimization_stress import run_optimization_stress
        from methodology.loss_regime_map.quadratic_diagnostics import run_quadratic_diagnostics
        from methodology.loss_regime_map.pipeline import run_loss_regime_map
        regime_root = method_root / "05_loss_regime_map"
        if cmd == "efficiency-calibration":
            return run_efficiency_calibration(regime_root / "00_efficiency_calibration")
        protocols, _ = run_efficiency_calibration(regime_root / "00_efficiency_calibration")
        locked = protocols["equal_gaussian_efficiency"]
        seeds = range(2) if args.quick_regime_map else range(10)
        if cmd == "structure-contamination":
            return run_structure_contamination(regime_root / "01_structure_contamination", GLOBAL_IRMF_PARAMS, locked, seeds=seeds)
        if cmd == "optimization-stress":
            return run_optimization_stress(regime_root / "02_optimization_stress", GLOBAL_IRMF_PARAMS, locked, seeds=seeds)
        if cmd == "quadratic-diagnostics":
            return run_quadratic_diagnostics(regime_root / "03_quadratic_diagnostics", GLOBAL_IRMF_PARAMS, locked, seeds=seeds)
        return run_loss_regime_map(regime_root, GLOBAL_IRMF_PARAMS, quick=args.quick_regime_map)
    if cmd == "methodology":
        return run_methodology_pipeline(method_root, quick_regime_map=args.quick_regime_map)

    if cmd in {
        "parameter-selection", "benchmark", "statistics", "sensitivity",
        "challenging-signals", "controlled-challenging", "oracle-adaptivity",
        "metric-taxonomy",
        "v526-primary-schema-qualification",
        "v527-primary-schema-qualification",
        "v527-freeze-primary-schema",
        "v527-formula-validity-audit",
        "v527-unified-statistics",
        "v527-directionality-audit",
        "v527-spillover-scale-audit",
        "v528-freeze-primary-schema",
        "v528-unified-statistics",
        "v528-final-statistics-validation",
        "v529-snr-design-draft",
        "v529-snr-realization-audit",
        "v529-freeze-snr-design",
        "v530-snr-design-draft",
        "v530-snr-realization-audit",
        "v530-freeze-snr-design",
        "v530-clean-baseline-schema",
        "v531-time-axis-semantics-audit",
        "v532-section6-scope",
        "v533-irmf-parameter-sensitivity-protocol",
        "v534-section6-executable-protocols",
        "v535-section6-1-irmf-parameter-smoke",
        "v536-section6-2-contamination-smoke",
        "v537-section6-3-comparator-signal-smoke",
        "v538-section6-4-runtime-smoke",
        "v539-irmf-h-tuning-amendment",
        "v540-time-frequency-secondary-schema",
        "v541-primary-secondary-closure",
        "v542-irmf-relative-h-parameterization",
        "v543-primary-metric-manual-audit",
        "v544-h-decision-gate",
        "v545-relative-h-scale-estimator-qualification",
        "v548-parameter-selection-adjudication",
        "v550-section6-main-text-assembly",
        "v551-section6-2-contamination-design",
        "v552-section6-4-computational-scaling",
        "v553-section6-claim-qualification",
        "v554-noise-realization-stability",
        "v555-noise-realization-stability-qualification",
        "v556-cross-realization-waveform-stability-protocol",
        "v557-section6-challenging-signal-extension",
        "v558-section6-pre-execution-audit",
        "v559-section6-post-execution-qualification",
        "v560-matching-sensitivity-protocol",
        "v560-matching-sensitivity-subset",
        "v561-section6-5-matching-rule-robustness",
        "v562-section6-reviewer-readiness-synthesis",
        "v563-section6-failure-region-dominance-synthesis",
        "v564-section6-3a-challenging-family-variant-extension",
        "v564a-section6-3a-canonical-anchor-reference",
        "v565-section6-3a-canonical-structured-perturbation",
        "v566-section6-3a-two-axis-interaction-design",
        "v567-section6-3a-analysis-qualification-freeze",
        "real-world-validation-schema-draft",
        "real-world-endpoint-formula-audit",
        "real-world-perturbation-protocol-freeze",
        "real-world-denominator-degeneracy-audit",
        "real-world-method-neutrality-audit",
        "real-world-validation-preflight",
        "real-world-v1-schema-smoke",
        "unified-cube", "unified-statistics", "proxy-validation",
        "benchmark-eligibility-gate", "benchmark-execution-audit",
        "ceemdan-noise-endpoint-adjudication",
        "rebuild-universal-noise-endpoints",
        "true-noise-separation-adapter-spec",
        "controlled-noise-endpoint-rerun-preflight",
        "synthetic-reconstruction-rule-discovery",
        "synthetic-reconstruction-protocol-amendment",
        "qualify-synthetic-reconstruction-protocol",
        "controlled-noise-endpoint-rerun",
        "noise-endpoint-merge-audit",
        "protocol-controlled-full-benchmark-rerun",
        "protocol-full-rerun-audit",
        "structural-metric-validation",
        "real-data-protocol", "real-data-split-audit",
        "real-data-development-run", "real-data-ecg-operational-smoke",
        "real-world-validation", "parameter-transfer", "paper-section-map",
        "algorithm", "paper-v518",
    }:
        from algorithm.pipeline import (
            run_parameter_selection_stage, run_benchmark_stage, run_statistics_stage,
            run_sensitivity_stage, run_challenging_signal_stage,
            run_controlled_challenging_diagnostics_stage,
            run_metric_taxonomy_stage,
            run_v526_primary_schema_qualification_stage,
            run_v527_primary_schema_qualification_stage,
            run_v527_primary_schema_freeze_stage,
            run_v527_formula_validity_audit_stage,
            run_v527_unified_statistics_stage,
            run_v527_directionality_audit_stage,
            run_v527_spillover_scale_audit_stage,
            run_v528_primary_schema_freeze_stage,
            run_v528_unified_statistics_stage,
            run_v528_final_statistics_validation_stage,
            run_v529_snr_design_draft_stage,
            run_v529_snr_realization_audit_stage,
            run_v529_snr_design_freeze_stage,
            run_v530_snr_design_draft_stage,
            run_v530_snr_realization_audit_stage,
            run_v530_snr_design_freeze_stage,
            run_v530_clean_baseline_schema_stage,
            run_v531_time_axis_semantics_audit_stage,
            run_v532_section6_scope_stage,
            run_v533_irmf_parameter_sensitivity_protocol_stage,
            run_v534_section6_executable_protocols_stage,
            run_v535_section6_1_irmf_parameter_smoke_stage,
            run_v536_section6_2_contamination_design_smoke_stage,
            run_v537_section6_3_comparator_signal_smoke_stage,
            run_v538_section6_4_runtime_instrumentation_smoke_stage,
            run_v539_irmf_h_tuning_amendment_stage,
            run_v540_time_frequency_secondary_schema_stage,
            run_v541_primary_secondary_closure_stage,
            run_v542_irmf_relative_h_parameterization_stage,
            run_v543_primary_metric_manual_code_literature_audit_stage,
            run_v544_h_decision_gate_stage,
            run_v545_relative_h_scale_estimator_qualification_stage,
            run_v548_parameter_selection_adjudication_stage,
            run_v550_section6_main_text_assembly_stage,
            run_v551_section6_2_contamination_design_stage,
            run_v552_section6_4_computational_scaling_stage,
            run_v553_section6_claim_qualification_stage,
            run_v554_noise_realization_stability_stage,
            run_v555_noise_realization_stability_qualification_stage,
            run_v556_cross_realization_waveform_stability_protocol_stage,
            run_v557_section6_challenging_signal_extension_stage,
            run_v558_section6_pre_execution_audit_stage,
            run_v559_section6_post_execution_qualification_stage,
            run_v560_matching_sensitivity_protocol_stage,
            run_v560_matching_sensitivity_subset_stage,
            run_v561_section6_5_matching_rule_robustness_stage,
            run_v562_section6_reviewer_readiness_synthesis_stage,
            run_v563_section6_failure_region_dominance_synthesis_stage,
            run_v564_section6_3a_challenging_family_variant_extension_stage,
            run_v564a_section6_3a_canonical_anchor_reference_stage,
            run_v565_section6_3a_canonical_structured_perturbation_stage,
            run_v566_section6_3a_two_axis_interaction_design_stage,
            run_v567_section6_3a_analysis_qualification_freeze_stage,
            run_real_world_validation_schema_draft_stage,
            run_real_world_endpoint_formula_audit_stage,
            run_real_world_perturbation_protocol_freeze_stage,
            run_real_world_denominator_degeneracy_audit_stage,
            run_real_world_method_neutrality_audit_stage,
            run_real_world_validation_preflight_stage,
            run_real_world_v1_schema_smoke_stage,
            run_oracle_adaptivity_stage, run_unified_benchmark_cube_stage,
            run_unified_cube_statistics_stage, run_algorithm_pipeline,
            run_proxy_metric_validation_stage, run_real_world_validation_stage,
            run_structural_metric_validation_stage,
            run_benchmark_eligibility_gate_stage,
            run_benchmark_execution_audit_stage,
            run_ceemdan_noise_endpoint_adjudication_stage,
            run_rebuild_universal_noise_endpoints_stage,
            run_true_noise_separation_adapter_spec_stage,
            run_controlled_noise_endpoint_rerun_preflight_stage,
            run_synthetic_reconstruction_rule_discovery_stage,
            run_synthetic_reconstruction_protocol_amendment_stage,
            run_qualify_synthetic_reconstruction_protocol_stage,
            run_controlled_noise_endpoint_re_evaluation_stage,
            run_noise_endpoint_merge_audit_stage,
            run_protocol_controlled_full_benchmark_rerun_stage,
            run_protocol_full_rerun_audit_stage,
            run_v518_paper_pipeline, run_paper_section_map_stage,
            run_parameter_transfer_stage, run_real_data_protocol_stage,
            run_real_data_split_audit_stage, run_real_data_development_run_stage,
            run_real_data_ecg_operational_smoke_stage,
        )
        calibrate = not args.skip_emd_family_calibration
        cube_seeds = (
            tuple(
                int(item.strip())
                for item in str(args.cube_seeds).split(",")
                if item.strip() != ""
            )
            if args.cube_seeds is not None
            else tuple(range(20))
        )
        if cmd == "parameter-selection":
            return run_parameter_selection_stage(algorithm_root, calibrate=calibrate)
        if cmd == "benchmark":
            return run_benchmark_stage(
                algorithm_root, calibrate=calibrate,
                full_family=not args.skip_full_emd_family_benchmark,
            )
        if cmd == "statistics":
            return run_statistics_stage(
                algorithm_root,
                calibrate=calibrate,
                use_current_config_params=args.use_current_config_params,
            )
        if cmd == "sensitivity":
            return run_sensitivity_stage(
                algorithm_root, calibrate=calibrate,
                emd_family_sensitivity=not args.skip_emd_family_sensitivity,
            )
        if cmd == "challenging-signals":
            return run_challenging_signal_stage(
                algorithm_root,
                calibrate=calibrate,
                quick=args.quick_challenging,
                use_current_config_params=args.use_current_config_params,
            )
        if cmd == "controlled-challenging":
            return run_controlled_challenging_diagnostics_stage(
                algorithm_root,
                calibrate=calibrate,
                quick=args.quick_challenging,
                use_current_config_params=args.use_current_config_params,
            )
        if cmd == "oracle-adaptivity":
            return run_oracle_adaptivity_stage(
                algorithm_root,
                calibrate=calibrate,
                quick=args.quick_oracle,
                use_current_config_params=args.use_current_config_params,
                run_exploratory_ceemdan_oracle=args.run_exploratory_ceemdan_oracle,
            )
        if cmd == "metric-taxonomy":
            return run_metric_taxonomy_stage(algorithm_root)
        if cmd == "v526-primary-schema-qualification":
            return run_v526_primary_schema_qualification_stage(algorithm_root)
        if cmd == "v527-primary-schema-qualification":
            return run_v527_primary_schema_qualification_stage(algorithm_root)
        if cmd == "v527-freeze-primary-schema":
            return run_v527_primary_schema_freeze_stage(algorithm_root)
        if cmd == "v527-formula-validity-audit":
            return run_v527_formula_validity_audit_stage(algorithm_root)
        if cmd == "v527-unified-statistics":
            return run_v527_unified_statistics_stage(algorithm_root)
        if cmd == "v527-directionality-audit":
            return run_v527_directionality_audit_stage(algorithm_root)
        if cmd == "v527-spillover-scale-audit":
            return run_v527_spillover_scale_audit_stage(algorithm_root)
        if cmd == "v528-freeze-primary-schema":
            return run_v528_primary_schema_freeze_stage(algorithm_root)
        if cmd == "v528-unified-statistics":
            return run_v528_unified_statistics_stage(algorithm_root)
        if cmd == "v528-final-statistics-validation":
            return run_v528_final_statistics_validation_stage(algorithm_root)
        if cmd == "v529-snr-design-draft":
            return run_v529_snr_design_draft_stage(algorithm_root)
        if cmd == "v529-snr-realization-audit":
            return run_v529_snr_realization_audit_stage(algorithm_root)
        if cmd == "v529-freeze-snr-design":
            return run_v529_snr_design_freeze_stage(algorithm_root)
        if cmd == "v530-snr-design-draft":
            return run_v530_snr_design_draft_stage(algorithm_root)
        if cmd == "v530-snr-realization-audit":
            return run_v530_snr_realization_audit_stage(algorithm_root)
        if cmd == "v530-freeze-snr-design":
            return run_v530_snr_design_freeze_stage(algorithm_root)
        if cmd == "v530-clean-baseline-schema":
            return run_v530_clean_baseline_schema_stage(algorithm_root)
        if cmd == "v531-time-axis-semantics-audit":
            return run_v531_time_axis_semantics_audit_stage(algorithm_root)
        if cmd == "v532-section6-scope":
            return run_v532_section6_scope_stage(algorithm_root)
        if cmd == "v533-irmf-parameter-sensitivity-protocol":
            return run_v533_irmf_parameter_sensitivity_protocol_stage(algorithm_root)
        if cmd == "v534-section6-executable-protocols":
            return run_v534_section6_executable_protocols_stage(algorithm_root)
        if cmd == "v535-section6-1-irmf-parameter-smoke":
            return run_v535_section6_1_irmf_parameter_smoke_stage(algorithm_root)
        if cmd == "v536-section6-2-contamination-smoke":
            return run_v536_section6_2_contamination_design_smoke_stage(algorithm_root)
        if cmd == "v537-section6-3-comparator-signal-smoke":
            return run_v537_section6_3_comparator_signal_smoke_stage(algorithm_root)
        if cmd == "v538-section6-4-runtime-smoke":
            return run_v538_section6_4_runtime_instrumentation_smoke_stage(algorithm_root)
        if cmd == "v539-irmf-h-tuning-amendment":
            return run_v539_irmf_h_tuning_amendment_stage(algorithm_root)
        if cmd == "v540-time-frequency-secondary-schema":
            return run_v540_time_frequency_secondary_schema_stage(algorithm_root)
        if cmd == "v541-primary-secondary-closure":
            return run_v541_primary_secondary_closure_stage(algorithm_root)
        if cmd == "v542-irmf-relative-h-parameterization":
            return run_v542_irmf_relative_h_parameterization_stage(algorithm_root)
        if cmd == "v543-primary-metric-manual-audit":
            return run_v543_primary_metric_manual_code_literature_audit_stage(algorithm_root)
        if cmd == "v544-h-decision-gate":
            return run_v544_h_decision_gate_stage(algorithm_root)
        if cmd == "v545-relative-h-scale-estimator-qualification":
            return run_v545_relative_h_scale_estimator_qualification_stage(algorithm_root)
        if cmd == "v548-parameter-selection-adjudication":
            return run_v548_parameter_selection_adjudication_stage(algorithm_root)
        if cmd == "v550-section6-main-text-assembly":
            return run_v550_section6_main_text_assembly_stage(algorithm_root)
        if cmd == "v551-section6-2-contamination-design":
            return run_v551_section6_2_contamination_design_stage(algorithm_root)
        if cmd == "v552-section6-4-computational-scaling":
            return run_v552_section6_4_computational_scaling_stage(algorithm_root)
        if cmd == "v553-section6-claim-qualification":
            return run_v553_section6_claim_qualification_stage(algorithm_root)
        if cmd == "v554-noise-realization-stability":
            return run_v554_noise_realization_stability_stage(algorithm_root)
        if cmd == "v555-noise-realization-stability-qualification":
            return run_v555_noise_realization_stability_qualification_stage(algorithm_root)
        if cmd == "v556-cross-realization-waveform-stability-protocol":
            return run_v556_cross_realization_waveform_stability_protocol_stage(algorithm_root)
        if cmd == "v557-section6-challenging-signal-extension":
            return run_v557_section6_challenging_signal_extension_stage(algorithm_root)
        if cmd == "v558-section6-pre-execution-audit":
            return run_v558_section6_pre_execution_audit_stage(algorithm_root)
        if cmd == "v559-section6-post-execution-qualification":
            return run_v559_section6_post_execution_qualification_stage(algorithm_root)
        if cmd == "v560-matching-sensitivity-protocol":
            return run_v560_matching_sensitivity_protocol_stage(algorithm_root)
        if cmd == "v560-matching-sensitivity-subset":
            return run_v560_matching_sensitivity_subset_stage(algorithm_root)
        if cmd == "v561-section6-5-matching-rule-robustness":
            return run_v561_section6_5_matching_rule_robustness_stage(algorithm_root)
        if cmd == "v562-section6-reviewer-readiness-synthesis":
            return run_v562_section6_reviewer_readiness_synthesis_stage(algorithm_root)
        if cmd == "v563-section6-failure-region-dominance-synthesis":
            return run_v563_section6_failure_region_dominance_synthesis_stage(algorithm_root)
        if cmd == "v564-section6-3a-challenging-family-variant-extension":
            return run_v564_section6_3a_challenging_family_variant_extension_stage(algorithm_root)
        if cmd == "v564a-section6-3a-canonical-anchor-reference":
            return run_v564a_section6_3a_canonical_anchor_reference_stage(algorithm_root)
        if cmd == "v565-section6-3a-canonical-structured-perturbation":
            return run_v565_section6_3a_canonical_structured_perturbation_stage(algorithm_root)
        if cmd == "v566-section6-3a-two-axis-interaction-design":
            return run_v566_section6_3a_two_axis_interaction_design_stage(algorithm_root)
        if cmd == "v567-section6-3a-analysis-qualification-freeze":
            return run_v567_section6_3a_analysis_qualification_freeze_stage(algorithm_root)
        if cmd == "real-world-validation-schema-draft":
            return run_real_world_validation_schema_draft_stage(algorithm_root)
        if cmd == "real-world-endpoint-formula-audit":
            return run_real_world_endpoint_formula_audit_stage(algorithm_root)
        if cmd == "real-world-perturbation-protocol-freeze":
            return run_real_world_perturbation_protocol_freeze_stage(algorithm_root)
        if cmd == "real-world-denominator-degeneracy-audit":
            return run_real_world_denominator_degeneracy_audit_stage(algorithm_root)
        if cmd == "real-world-method-neutrality-audit":
            return run_real_world_method_neutrality_audit_stage(algorithm_root)
        if cmd == "real-world-validation-preflight":
            return run_real_world_validation_preflight_stage(algorithm_root)
        if cmd == "real-world-v1-schema-smoke":
            return run_real_world_v1_schema_smoke_stage(
                algorithm_root,
                data_root=args.data_root,
                calibrate=calibrate,
                use_current_config_params=args.use_current_config_params,
                timeout_seconds=args.timeout_seconds,
            )
        if cmd == "benchmark-eligibility-gate":
            return run_benchmark_eligibility_gate_stage(
                algorithm_root,
                calibrate=calibrate,
                use_current_config_params=args.use_current_config_params,
                qualification_root=args.qualification_root,
                seeds=cube_seeds,
                force_parameter_selection=args.force_parameter_selection,
            )
        if cmd == "unified-cube":
            return run_unified_benchmark_cube_stage(
                algorithm_root,
                calibrate=calibrate,
                quick=args.quick_unified_cube,
                seeds=cube_seeds,
                timeout_seconds=args.timeout_seconds,
                use_current_config_params=args.use_current_config_params,
                force_parameter_selection=args.force_parameter_selection,
            )
        if cmd == "unified-statistics":
            return run_unified_cube_statistics_stage(algorithm_root)
        if cmd == "benchmark-execution-audit":
            return run_benchmark_execution_audit_stage(algorithm_root)
        if cmd == "ceemdan-noise-endpoint-adjudication":
            return run_ceemdan_noise_endpoint_adjudication_stage(algorithm_root)
        if cmd == "rebuild-universal-noise-endpoints":
            return run_rebuild_universal_noise_endpoints_stage(algorithm_root)
        if cmd == "true-noise-separation-adapter-spec":
            return run_true_noise_separation_adapter_spec_stage(algorithm_root)
        if cmd == "controlled-noise-endpoint-rerun-preflight":
            return run_controlled_noise_endpoint_rerun_preflight_stage(algorithm_root)
        if cmd == "synthetic-reconstruction-rule-discovery":
            return run_synthetic_reconstruction_rule_discovery_stage(algorithm_root)
        if cmd == "synthetic-reconstruction-protocol-amendment":
            return run_synthetic_reconstruction_protocol_amendment_stage(algorithm_root)
        if cmd == "qualify-synthetic-reconstruction-protocol":
            return run_qualify_synthetic_reconstruction_protocol_stage(algorithm_root)
        if cmd == "controlled-noise-endpoint-rerun":
            return run_controlled_noise_endpoint_re_evaluation_stage(
                algorithm_root,
                calibrate=calibrate,
                use_current_config_params=args.use_current_config_params,
                timeout_seconds=args.timeout_seconds,
                seeds=cube_seeds,
            )
        if cmd == "noise-endpoint-merge-audit":
            return run_noise_endpoint_merge_audit_stage(algorithm_root)
        if cmd == "protocol-controlled-full-benchmark-rerun":
            return run_protocol_controlled_full_benchmark_rerun_stage(
                algorithm_root,
                calibrate=calibrate,
                use_current_config_params=args.use_current_config_params,
                timeout_seconds=args.timeout_seconds,
                seeds=(
                    None
                    if args.use_v530_target_snr_grid and args.cube_seeds is None
                    else cube_seeds
                ),
                force_parameter_selection=args.force_parameter_selection,
                use_v530_target_snr_grid=args.use_v530_target_snr_grid,
            )
        if cmd == "protocol-full-rerun-audit":
            return run_protocol_full_rerun_audit_stage(algorithm_root)
        if cmd == "proxy-validation":
            return run_proxy_metric_validation_stage(algorithm_root)
        if cmd == "structural-metric-validation":
            return run_structural_metric_validation_stage(
                algorithm_root,
                calibrate=calibrate,
                quick=args.quick_structural_metric_validation,
                timeout_seconds=args.timeout_seconds,
                use_current_config_params=args.use_current_config_params,
            )
        if cmd == "real-data-protocol":
            return run_real_data_protocol_stage(
                algorithm_root,
                calibrate=calibrate,
                use_current_config_params=args.use_current_config_params,
            )
        if cmd == "real-data-split-audit":
            protocol_root = Path(args.protocol_root) if args.protocol_root else root
            audit_algorithm_root = protocol_root / "algorithm"
            return run_real_data_split_audit_stage(
                audit_algorithm_root,
                protocol_root=protocol_root,
                subject_map_path=args.subject_map,
            )
        if cmd == "real-data-development-run":
            protocol_root = Path(args.protocol_root) if args.protocol_root else root
            audit_algorithm_root = protocol_root / "algorithm"
            return run_real_data_development_run_stage(
                audit_algorithm_root,
                protocol_root=protocol_root,
                data_root=args.data_root,
                calibrate=calibrate,
                use_current_config_params=args.use_current_config_params,
                max_windows_per_record=args.max_development_windows,
            )
        if cmd == "real-data-ecg-operational-smoke":
            protocol_root = Path(args.protocol_root) if args.protocol_root else root
            audit_algorithm_root = protocol_root / "algorithm"
            return run_real_data_ecg_operational_smoke_stage(
                audit_algorithm_root,
                protocol_root=protocol_root,
                data_root=args.data_root,
                calibrate=calibrate,
                use_current_config_params=args.use_current_config_params,
                max_windows_per_record=args.max_development_windows or 1,
            )
        if cmd == "real-world-validation":
            return run_real_world_validation_stage(
                algorithm_root,
                data_dir=args.real_data_dir,
                calibrate=calibrate,
                use_current_config_params=args.use_current_config_params,
            )
        if cmd == "parameter-transfer":
            return run_parameter_transfer_stage(
                algorithm_root,
                calibrate=calibrate,
                use_current_config_params=args.use_current_config_params,
            )
        if cmd == "paper-section-map":
            return run_paper_section_map_stage(algorithm_root)
        if cmd == "paper-v518":
            return run_v518_paper_pipeline(
                algorithm_root,
                run_methodology=not args.skip_methodology,
                quick_regime_map=args.quick_regime_map,
                calibrate=calibrate,
                quick_unified_cube=args.quick_unified_cube,
                seeds=cube_seeds,
                timeout_seconds=args.timeout_seconds,
                use_current_config_params=args.use_current_config_params,
                run_oracle=args.run_oracle,
                quick_oracle=args.quick_oracle,
                run_exploratory_ceemdan_oracle=args.run_exploratory_ceemdan_oracle,
                run_assets=True,
            )
        return run_algorithm_pipeline(
            algorithm_root,
            calibrate=calibrate,
            full_family=not args.skip_full_emd_family_benchmark,
            sensitivity=not args.skip_emd_family_sensitivity,
            challenging=not args.skip_challenging_signals,
            quick_challenging=args.quick_challenging,
        )

    if cmd == "paper-assets":
        from paper_assets import run_paper_assets
        return run_paper_assets(root)

    if cmd == "paper":
        if not args.skip_methodology:
            run_methodology_pipeline(method_root, quick_regime_map=args.quick_regime_map)
        if not args.skip_algorithm:
            from algorithm.pipeline import run_algorithm_pipeline
            run_algorithm_pipeline(
                algorithm_root,
                calibrate=not args.skip_emd_family_calibration,
                full_family=not args.skip_full_emd_family_benchmark,
                sensitivity=not args.skip_emd_family_sensitivity,
                challenging=not args.skip_challenging_signals,
                quick_challenging=args.quick_challenging,
            )
        from paper_assets import run_paper_assets
        return run_paper_assets(root)

    # Explicit backward-compatible entry only. Not the recommended paper workflow.
    from legacy_full_pipeline import run_paper_pipeline as run_legacy_full_pipeline
    return run_legacy_full_pipeline(
        output_root=root / "legacy_paper_full",
        run_oracle=args.run_oracle,
        real_data_dir=args.real_data_dir,
        run_emd_family_sensitivity=not args.skip_emd_family_sensitivity,
        run_methodology_layer=True,
        run_emd_family_calibration=not args.skip_emd_family_calibration,
        run_full_emd_family_benchmark=not args.skip_full_emd_family_benchmark,
        run_v5_2_posthoc=True,
        run_loss_ablation=True,
    )


if __name__ == "__main__":
    main()
