#!/usr/bin/python
# coding: UTF-8

"""
Paper-oriented experiment pipeline.

Sections:
  2b Loss Function Analysis
  4  Global Parameter Selection
  5  Main Fixed-Parameter Benchmark
  6  Robustness and Sensitivity Analyses
  7  Statistical Inference, Mechanism Analysis, and Signal-Variant Robustness Study
  7b EMD-Family Baseline Comparison
  8  Real Data Illustration
  A/F Appendices

The oracle comparison is intentionally not run by default because it performs
per-case grid search and belongs only in Appendix C.

V5.3 fairness-calibration upgrade:
  EMD, EEMD, and CEEMDAN are calibrated on the same 32-case development set
  using predefined global grids, then locked before all benchmark sections.
"""

from pathlib import Path
import argparse

from project_config import (
    GLOBAL_CEEMDAN_PARAMS,
    GLOBAL_EEMD_PARAMS,
    GLOBAL_EMD_PARAMS,
    GLOBAL_IRMF_PARAMS,
    PAPER_OUTPUT_ROOT_NAME,
)
from experiments.experiment_appendices import (
    run_method_faithfulness_checks,
    run_oracle_comparison_appendix,
    write_reproducibility_appendix,
)
from experiments.experiment_global_parameter_selection import run_global_parameter_selection
from experiments.experiment_emd_family_benchmark import (
    run_emd_family_baseline_sensitivity,
    run_emd_family_parameter_calibration,
    run_fixed_emd_family_benchmark,
)
from experiments.experiment_loss_function_analysis import run_loss_function_analysis
from experiments.experiment_loss_ablation import run_complete_loss_methodology
from experiments.experiment_main_fixed_parameter_benchmark import run_main_fixed_parameter_benchmark
from experiments.experiment_mechanism_analysis import run_mechanism_analysis
from experiments.experiment_real_data_illustration import run_real_data_illustration
from experiments.experiment_real_data_semisynthetic import run_real_data_semisynthetic_contamination
from experiments.experiment_robustness_sensitivity import run_all_robustness_sensitivity
from experiments.experiment_signal_variant_robustness import run_signal_variant_robustness
from experiments.experiment_spokoiny_theory_validation import run_spokoiny_theory_validation
from experiments.experiment_statistical_inference import run_statistical_inference
from experiments.experiment_repeated_measures_statistics import run_repeated_measures_statistics
from experiments.experiment_result_sensitivity_analysis import run_v5_2_posthoc_result_checks
from experiments.paper_pipeline_utils import ensure_dir, write_json


def run_paper_pipeline(
        output_root=PAPER_OUTPUT_ROOT_NAME,
        run_oracle=False,
        real_data_dir=None,
        run_emd_family_sensitivity=True,
        run_methodology_layer=True,
        run_emd_family_calibration=True,
        run_full_emd_family_benchmark=True,
        run_v5_2_posthoc=True,
        run_loss_ablation=True,
):
    output_root = ensure_dir(output_root)

    loss_analysis = None
    loss_ablation = None
    if run_methodology_layer:
        loss_analysis = run_loss_function_analysis(
            output_root / "section_2b_loss_function_analysis",
            H=1.0,
        )
        if run_loss_ablation:
            loss_ablation = run_complete_loss_methodology(
                output_root / "section_6_loss_methodology_upgrade",
                base_params=GLOBAL_IRMF_PARAMS,
            )

    selected_params, candidate_summary, _ = run_global_parameter_selection(
        output_root / "section_4_global_parameter_selection"
    )
    irmf_params = dict(GLOBAL_IRMF_PARAMS)
    if selected_params is not None:
        for key in [
            "h1", "a", "h_min", "H", "boundary_mode", "min_support_points",
            "robust_scale_mode", "robust_scale_floor",
        ]:
            if key in selected_params:
                irmf_params[key] = selected_params[key]

    eemd_params = dict(GLOBAL_EEMD_PARAMS)
    ceemdan_params = dict(GLOBAL_CEEMDAN_PARAMS)
    emd_params = dict(GLOBAL_EMD_PARAMS)
    emd_family_calibration = None
    if run_emd_family_calibration:
        emd_family_calibration = run_emd_family_parameter_calibration(
            output_root / "section_4b_emd_family_parameter_calibration",
            emd_params=emd_params,
        )
        if isinstance(emd_family_calibration, dict):
            emd_params.update(emd_family_calibration.get("EMD", {}))
            eemd_params.update(emd_family_calibration.get("EEMD", {}))
            ceemdan_params.update(emd_family_calibration.get("CEEMDAN", {}))

    main_rows, main_summary = run_main_fixed_parameter_benchmark(
        output_root / "section_5_main_fixed_parameter_benchmark",
        irmf_params=irmf_params,
        emd_params=emd_params,
    )
    full_emd_family_rows = None
    full_emd_family_summary = None
    repeated_measures_statistics = None
    if run_full_emd_family_benchmark:
        full_emd_family_rows, full_emd_family_summary = run_fixed_emd_family_benchmark(
            output_root / "section_5b_main_emd_family_benchmark",
            irmf_params=irmf_params,
            emd_params=emd_params,
            eemd_params=eemd_params,
            ceemdan_params=ceemdan_params,
            mode="full",
            return_rows=True,
        )
        repeated_measures_statistics = run_repeated_measures_statistics(
            output_root / "section_7_repeated_measures_statistics",
            family_rows=full_emd_family_rows,
        )
    statistical_inference = None
    mechanism_analysis = None
    if run_methodology_layer:
        statistical_inference = run_statistical_inference(
            output_root / "section_7_statistical_inference",
            main_rows=main_rows,
        )
        mechanism_analysis = run_mechanism_analysis(
            output_root / "section_7_mechanism_analysis",
            main_rows=main_rows,
        )
    robustness_summary = run_all_robustness_sensitivity(
        output_root / "section_6_robustness_sensitivity",
        irmf_params=irmf_params,
        emd_params=emd_params,
    )
    variant_rows, variant_summary = run_signal_variant_robustness(
        output_root / "section_7_signal_variant_robustness",
        irmf_params=irmf_params,
        emd_params=emd_params,
    )
    v5_2_posthoc = None
    if run_v5_2_posthoc and full_emd_family_rows is not None:
        v5_2_posthoc = run_v5_2_posthoc_result_checks(
            output_root / "section_6b_v5_2_result_sensitivity_checks",
            family_rows=full_emd_family_rows,
            variant_rows=variant_rows,
        )
    emd_family_summary = run_fixed_emd_family_benchmark(
        output_root / "section_7b_emd_family_comparison",
        irmf_params=irmf_params,
        emd_params=emd_params,
        eemd_params=eemd_params,
        ceemdan_params=ceemdan_params,
        mode="representative",
    )
    real_rows, real_summary = run_real_data_illustration(
        output_root / "section_8_real_data_illustration",
        data_dir=real_data_dir,
        irmf_params=irmf_params,
        emd_params=emd_params,
    )
    real_semisynthetic = run_real_data_semisynthetic_contamination(
        output_root / "section_8b_real_semisynthetic_contamination",
        data_dir=real_data_dir,
        irmf_params=irmf_params,
        emd_params=emd_params,
        eemd_params=eemd_params,
        ceemdan_params=ceemdan_params,
    )
    sanity = run_method_faithfulness_checks(
        output_root / "appendices" / "appendix_A_method_faithfulness_checks",
        irmf_params=irmf_params,
    )
    spokoiny_theory = run_spokoiny_theory_validation(
        output_root / "appendices" / "appendix_A_spokoiny_theory_validation",
        irmf_params=irmf_params,
    )
    emd_family_sensitivity = None
    if run_emd_family_sensitivity:
        emd_family_sensitivity = run_emd_family_baseline_sensitivity(
            output_root / "appendices" / "appendix_D_emd_family_baseline_sensitivity",
            irmf_params=irmf_params,
            emd_params=emd_params,
            mode="representative",
        )
    reproducibility = write_reproducibility_appendix(
        output_root / "appendices" / "appendix_F_reproducibility",
        extra={
            "selected_irmf_params_for_this_run": irmf_params,
            "selected_emd_params_for_this_run": emd_params,
            "selected_eemd_params_for_this_run": eemd_params,
            "selected_ceemdan_params_for_this_run": ceemdan_params,
        },
    )
    oracle_rows = None
    if run_oracle:
        oracle_rows = run_oracle_comparison_appendix(
            output_root / "appendices" / "appendix_C_oracle_comparison"
        )

    dashboard = {
        "selected_irmf_params": irmf_params,
        "selected_emd_params": emd_params,
        "selected_eemd_params": eemd_params,
        "selected_ceemdan_params": ceemdan_params,
        "section_2b_loss_function_analysis": loss_analysis,
        "section_6_loss_methodology_upgrade": loss_ablation,
        "section_4_n_candidates": len(candidate_summary),
        "section_4b_emd_family_calibration": emd_family_calibration,
        "section_5_main_summary": main_summary,
        "section_5b_full_emd_family_summary": full_emd_family_summary,
        "section_6_robustness_summary": robustness_summary,
        "section_7_statistical_inference": statistical_inference,
        "section_7_mechanism_analysis": mechanism_analysis,
        "section_7_repeated_measures_statistics": repeated_measures_statistics,
        "section_7_variant_summary": variant_summary,
        "section_6b_v5_2_posthoc_result_checks": v5_2_posthoc,
        "section_7b_emd_family_summary": emd_family_summary,
        "section_8_real_data_summary": real_summary,
        "section_8b_real_semisynthetic_summary": real_semisynthetic,
        "appendix_A_sanity": sanity,
        "appendix_A_spokoiny_theory_summary": spokoiny_theory,
        "appendix_D_emd_family_sensitivity_run": bool(run_emd_family_sensitivity),
        "appendix_D_emd_family_sensitivity_summary": emd_family_sensitivity,
        "appendix_F_reproducibility": reproducibility,
        "appendix_C_oracle_run": bool(run_oracle),
        "appendix_C_n_rows": len(oracle_rows) if oracle_rows is not None else 0,
    }
    write_json(dashboard, output_root / "paper_pipeline_dashboard.json")
    return dashboard


def parse_args():
    parser = argparse.ArgumentParser(description="Run the IRMF/EMD paper experiment pipeline.")
    parser.add_argument("--output-root", default=PAPER_OUTPUT_ROOT_NAME)
    parser.add_argument("--real-data-dir", default=None)
    parser.add_argument("--run-oracle", action="store_true", help="Run Appendix C per-case oracle grid-search results.")
    parser.add_argument(
        "--skip-emd-family-sensitivity",
        action="store_true",
        help="Skip Appendix D EEMD/CEEMDAN baseline sensitivity.",
    )
    parser.add_argument(
        "--skip-methodology-layer",
        action="store_true",
        help="Skip V5 loss analysis, statistical inference, and mechanism-analysis outputs.",
    )
    parser.add_argument(
        "--skip-emd-family-calibration",
        action="store_true",
        help="Use default EMD/EEMD/CEEMDAN parameters instead of 32-case development-set calibration.",
    )
    parser.add_argument(
        "--skip-full-emd-family-benchmark",
        action="store_true",
        help="Skip the full 168-case IRMF/EMD/EEMD/CEEMDAN benchmark and repeated-measures ranking.",
    )
    parser.add_argument(
        "--skip-loss-ablation",
        action="store_true",
        help="Skip loss calibration, ablation, contamination tolerance, and optimization stability.",
    )
    parser.add_argument(
        "--skip-v5-2-posthoc",
        action="store_true",
        help="Skip composite-score sensitivity, structural distribution, cost frontier, and variant consistency outputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_paper_pipeline(
        output_root=Path(args.output_root),
        run_oracle=args.run_oracle,
        real_data_dir=args.real_data_dir,
        run_emd_family_sensitivity=not args.skip_emd_family_sensitivity,
        run_methodology_layer=not args.skip_methodology_layer,
        run_emd_family_calibration=not args.skip_emd_family_calibration,
        run_full_emd_family_benchmark=not args.skip_full_emd_family_benchmark,
        run_v5_2_posthoc=not args.skip_v5_2_posthoc,
        run_loss_ablation=not args.skip_loss_ablation,
    )
