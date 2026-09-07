#!/usr/bin/python
# coding: UTF-8

"""V5.57 Section 6 challenging-signal extension amendment."""

from datetime import datetime, timezone

from project_config import (
    EMD_FAMILY_BENCHMARK_MODES,
    EMD_FAMILY_SENSITIVITY_METHODS,
    EMD_FAMILY_SENSITIVITY_NOISE_WIDTHS,
    EMD_FAMILY_SENSITIVITY_TRIALS,
    EMD_PARAMETER_SELECTION_GRID,
    CEEMDAN_PARAMETER_SELECTION_GRID,
    PARAMETER_SENSITIVITY_FACTORS,
    PARAMETER_SENSITIVITY_NOISES,
    PARAMETER_SENSITIVITY_SIGNALS,
    PARAMETER_SENSITIVITY_TARGET_SNR_DB_LEVELS,
    SIGNAL_VARIANT_FAMILY,
    SIGNAL_VARIANT_NOISES,
    SIGNAL_VARIANT_SEEDS,
    SIGNAL_VARIANT_TARGET_SNR_DB_LEVELS,
    V556_WAVEFORM_STABILITY_LOW_SIMILARITY_PRIMARY_THRESHOLD,
    V556_WAVEFORM_STABILITY_LOW_SIMILARITY_THRESHOLD_SENSITIVITY,
    V556_WAVEFORM_STABILITY_METHODS,
    V556_WAVEFORM_STABILITY_NOISES,
    V556_WAVEFORM_STABILITY_SEEDS,
    V556_WAVEFORM_STABILITY_SIGNALS,
    V556_WAVEFORM_STABILITY_TARGET_SNR_DB_LEVELS,
    V557_SECTION6_2_CONTAMINATION_SIGNALS,
    V557_SECTION6_3_CHALLENGING_SIGNAL_EXTENSION,
    V557_SECTION6_4_SCALING_SIGNALS,
    V557_SECTION6_AMENDMENT_VERSION,
    V557_SECTION6_COMMON_ANCHOR_NOISES,
    V557_SECTION6_COMMON_ANCHOR_SIGNALS,
    V557_SECTION6_COMMON_ANCHOR_TARGET_SNR_DB_LEVELS,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


METHODS = ("IRMF", "EMD", "EEMD", "CEEMDAN")


def _product(options):
    out = 1
    for values in options:
        out *= len(values)
    return int(out)


def _emd_family_config_count():
    emd_count = _product((
        EMD_PARAMETER_SELECTION_GRID["nbsym_options"],
        EMD_PARAMETER_SELECTION_GRID["spline_kind_options"],
        EMD_PARAMETER_SELECTION_GRID["max_imf_options"],
        EMD_PARAMETER_SELECTION_GRID["std_thr_options"],
        EMD_PARAMETER_SELECTION_GRID["svar_thr_options"],
        EMD_PARAMETER_SELECTION_GRID["total_power_thr_options"],
        EMD_PARAMETER_SELECTION_GRID["range_thr_options"],
    ))
    eemd_count = len(EMD_FAMILY_SENSITIVITY_TRIALS) * len(EMD_FAMILY_SENSITIVITY_NOISE_WIDTHS)
    ceemdan_count = len(EMD_FAMILY_SENSITIVITY_TRIALS) * len(CEEMDAN_PARAMETER_SELECTION_GRID["epsilon_options"])
    return {
        "EMD": int(emd_count),
        "EEMD": int(eemd_count),
        "CEEMDAN": int(ceemdan_count),
        "total_baseline_configurations": int(emd_count + eemd_count + ceemdan_count),
    }


def run_v557_section6_challenging_signal_extension_amendment(output_root):
    output_root = ensure_dir(output_root)
    parameter_design_n = _product(PARAMETER_SENSITIVITY_FACTORS.values())
    s63b_cfg = EMD_FAMILY_BENCHMARK_MODES["representative"]
    s63b_configs = _emd_family_config_count()
    expected_counts = {
        "6.1_parameter_sensitivity_irmf_runs": int(
            parameter_design_n
            * len(PARAMETER_SENSITIVITY_SIGNALS)
            * len(PARAMETER_SENSITIVITY_NOISES)
            * len(PARAMETER_SENSITIVITY_TARGET_SNR_DB_LEVELS)
        ),
        "6.2_contamination_design_method_runs": int(
            len(V557_SECTION6_2_CONTAMINATION_SIGNALS)
            * 24
            * 3
            * len(METHODS)
        ),
        "6.3A_signal_variant_method_runs": int(
            len(SIGNAL_VARIANT_FAMILY)
            * len(SIGNAL_VARIANT_NOISES)
            * len(SIGNAL_VARIANT_TARGET_SNR_DB_LEVELS)
            * len(SIGNAL_VARIANT_SEEDS)
            * len(METHODS)
        ),
        "6.3B_comparator_config_case_runs": int(
            len(s63b_cfg["signals"])
            * len(s63b_cfg["noises"])
            * len(PARAMETER_SENSITIVITY_TARGET_SNR_DB_LEVELS)
            * s63b_configs["total_baseline_configurations"]
        ),
        "6.4_computational_scaling_runs": int(
            5 * len(V557_SECTION6_4_SCALING_SIGNALS) * 1 * 1 * 2 * 3 * len(METHODS)
        ),
        "V5.56_waveform_stability_method_evaluations_when_executed": int(
            len(V556_WAVEFORM_STABILITY_SIGNALS)
            * len(V556_WAVEFORM_STABILITY_NOISES)
            * len(V556_WAVEFORM_STABILITY_TARGET_SNR_DB_LEVELS)
            * len(V556_WAVEFORM_STABILITY_SEEDS)
            * len(V556_WAVEFORM_STABILITY_METHODS)
        ),
    }
    rows = [
        {
            "module": "6.1",
            "title": "IRMF Parameter Sensitivity and Robustness",
            "updated_signal_scope": ";".join(PARAMETER_SENSITIVITY_SIGNALS),
            "updated_noise_scope": ";".join(PARAMETER_SENSITIVITY_NOISES),
            "target_snr_db_levels": ";".join(str(x) for x in PARAMETER_SENSITIVITY_TARGET_SNR_DB_LEVELS),
            "expected_runs": expected_counts["6.1_parameter_sensitivity_irmf_runs"],
            "execution_status": "requires_rerun_for_v557_claims",
        },
        {
            "module": "6.2",
            "title": "Contamination Design Sensitivity",
            "updated_signal_scope": ";".join(V557_SECTION6_2_CONTAMINATION_SIGNALS),
            "updated_noise_scope": "huber_contamination_design_axes",
            "target_snr_db_levels": "15.0",
            "expected_runs": expected_counts["6.2_contamination_design_method_runs"],
            "execution_status": "requires_rerun_for_v557_claims",
        },
        {
            "module": "6.3A",
            "title": "Signal Variant Robustness",
            "updated_signal_scope": ";".join(SIGNAL_VARIANT_FAMILY),
            "challenging_extension": ";".join(V557_SECTION6_3_CHALLENGING_SIGNAL_EXTENSION),
            "updated_noise_scope": ";".join(SIGNAL_VARIANT_NOISES),
            "target_snr_db_levels": ";".join(str(x) for x in SIGNAL_VARIANT_TARGET_SNR_DB_LEVELS),
            "expected_runs": expected_counts["6.3A_signal_variant_method_runs"],
            "execution_status": "requires_rerun_for_v557_claims",
        },
        {
            "module": "6.3B",
            "title": "EMD-family Baseline Sensitivity",
            "updated_signal_scope": ";".join(s63b_cfg["signals"]),
            "updated_noise_scope": ";".join(s63b_cfg["noises"]),
            "target_snr_db_levels": ";".join(str(x) for x in PARAMETER_SENSITIVITY_TARGET_SNR_DB_LEVELS),
            "expected_runs": expected_counts["6.3B_comparator_config_case_runs"],
            "execution_status": "canonical_representative_mode_retained",
            "note": "Comparator-configuration robustness remains canonical-representative unless a future amendment adds challenging comparator anchors.",
        },
        {
            "module": "6.4",
            "title": "Computational Efficiency and Scaling",
            "updated_signal_scope": ";".join(V557_SECTION6_4_SCALING_SIGNALS),
            "updated_noise_scope": "gaussian",
            "target_snr_db_levels": "15.0",
            "expected_runs": expected_counts["6.4_computational_scaling_runs"],
            "execution_status": "requires_rerun_for_v557_claims",
        },
        {
            "module": "V5.56",
            "title": "Cross-Realization Waveform-Level Stability",
            "updated_signal_scope": ";".join(V556_WAVEFORM_STABILITY_SIGNALS),
            "updated_noise_scope": ";".join(V556_WAVEFORM_STABILITY_NOISES),
            "target_snr_db_levels": ";".join(str(x) for x in V556_WAVEFORM_STABILITY_TARGET_SNR_DB_LEVELS),
            "expected_runs": expected_counts["V5.56_waveform_stability_method_evaluations_when_executed"],
            "execution_status": "protocol_frozen_execution_pending",
        },
    ]
    dashboard = {
        "schema_version": V557_SECTION6_AMENDMENT_VERSION,
        "created_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "amendment_status": "protocol_amended_execution_pending",
        "primary_metric_schema_changed": False,
        "section6_scientific_questions_changed": False,
        "ranking_procedure_changed": False,
        "reason": (
            "Improve signal-difficulty coverage and align sensitivity analyses "
            "with the canonical/challenging taxonomy without changing primary "
            "metrics or main benchmark claims."
        ),
        "common_anchor_core": {
            "signals": list(V557_SECTION6_COMMON_ANCHOR_SIGNALS),
            "noises": list(V557_SECTION6_COMMON_ANCHOR_NOISES),
            "target_snr_db_levels": list(V557_SECTION6_COMMON_ANCHOR_TARGET_SNR_DB_LEVELS),
            "policy": "Use where scientifically applicable; do not force 6.4 into a signal-robustness experiment.",
        },
        "module_updates": rows,
        "expected_counts": expected_counts,
        "V5.56_balanced_design": {
            "canonical": ["chirp", "close_frequencies", "impulsive_transient"],
            "challenging": [
                "crossing_chirps",
                "time_varying_close_frequencies",
                "piecewise_am_fm_discontinuity",
            ],
            "low_similarity_primary_threshold": float(
                V556_WAVEFORM_STABILITY_LOW_SIMILARITY_PRIMARY_THRESHOLD
            ),
            "threshold_sensitivity": list(
                V556_WAVEFORM_STABILITY_LOW_SIMILARITY_THRESHOLD_SENSITIVITY
            ),
        },
        "claim_boundary": (
            "Existing Section 6 results remain valid for their original executed "
            "subsets. V5.57-updated claims require rerunning affected modules "
            "under this amended signal-scope design."
        ),
    }
    write_json(dashboard, output_root / "v557_section6_challenging_signal_extension_dashboard.json")
    write_csv(rows, output_root / "v557_section6_challenging_signal_extension_module_updates.csv")
    return dashboard
