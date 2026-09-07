#!/usr/bin/python
# coding: UTF-8

"""V5.30 synthetic clean-baseline reference-stage specification."""

from datetime import datetime, timezone

from project_config import (
    EVALUATION_METHODS,
    TARGET_SNR_DB_LEVELS,
    UNIFIED_BENCHMARK_REGIMES,
    V529_SYNTHETIC_NOISE_DESIGN_VERSION,
)
from experiments.paper_pipeline_utils import ensure_dir, write_json


V530_CLEAN_BASELINE_SCHEMA_VERSION = "V5.30_synthetic_clean_baseline_reference_stage"


def run_v530_clean_baseline_schema(output_root):
    output_root = ensure_dir(output_root)
    signal_count = sum(len(cfg["signals"]) for cfg in UNIFIED_BENCHMARK_REGIMES.values())
    spec = {
        "schema_version": V530_CLEAN_BASELINE_SCHEMA_VERSION,
        "schema_status": "frozen",
        "qualification_status": "design_recorded",
        "benchmark_results_under_clean_baseline": "not_generated",
        "related_snr_schema_version": V529_SYNTHETIC_NOISE_DESIGN_VERSION,
        "clean_baseline_definition": {
            "observed_signal": "Y_t = X_t",
            "added_noise": "N_t = 0",
            "snr": "infinity",
            "target_snr_db": None,
        },
        "scientific_question": (
            "When no observational noise is added, how well do the methods "
            "decompose and reconstruct the clean synthetic structure itself?"
        ),
        "reporting_policy": {
            "display_with_snr_curves": True,
            "curve_order": ["clean", *list(reversed(TARGET_SNR_DB_LEVELS))],
            "excluded_from_snr_level_aggregation": True,
            "excluded_from_average_over_snr": True,
            "excluded_from_snr_friedman_rank_summary": True,
            "reported_as_reference_condition": True,
        },
        "execution_design": {
            "methods": list(EVALUATION_METHODS),
            "signal_regimes": {
                name: list(cfg["signals"])
                for name, cfg in UNIFIED_BENCHMARK_REGIMES.items()
            },
            "n_signals": int(signal_count),
            "noise_models": "not_applicable_no_added_noise",
            "seeds": "not_applicable_deterministic_clean_input",
            "expected_method_evaluations": int(signal_count * len(EVALUATION_METHODS)),
        },
        "provenance_guardrails": {
            "must_not_be_encoded_as_eighth_snr_level": True,
            "must_not_use_target_snr_scaling": True,
            "must_not_be_mixed_with_v530_snr_cube_statistics": True,
        },
        "created_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(spec, output_root / "v530_clean_baseline_schema.json")
    return spec
