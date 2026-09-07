#!/usr/bin/python
# coding: UTF-8

"""V5.58 pre-execution audit for the V5.57 Section 6 amendment."""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from project_config import (
    DEFAULT_FS,
    DEFAULT_N,
    EMD_FAMILY_BENCHMARK_MODES,
    EMD_FAMILY_SENSITIVITY_NOISE_WIDTHS,
    EMD_FAMILY_SENSITIVITY_TRIALS,
    EMD_PARAMETER_SELECTION_GRID,
    CEEMDAN_PARAMETER_SELECTION_GRID,
    EVALUATION_METHODS,
    PARAMETER_SENSITIVITY_FACTORS,
    PARAMETER_SENSITIVITY_NOISES,
    PARAMETER_SENSITIVITY_SIGNALS,
    PARAMETER_SENSITIVITY_TARGET_SNR_DB_LEVELS,
    SIGNAL_FAMILY,
    SIGNAL_VARIANT_FAMILY,
    SIGNAL_VARIANT_NOISES,
    SIGNAL_VARIANT_SEEDS,
    SIGNAL_VARIANT_TARGET_SNR_DB_LEVELS,
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
from experiments.experiment_v551_section6_2_contamination_design import _designs_for_signal
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json
from signal_bank.synthetic_signals import SIGNAL_VARIANT_METADATA, get_signal, get_true_components


V558_SECTION6_PRE_EXECUTION_AUDIT_VERSION = "V5.58_section6_pre_execution_audit"
METHODS = tuple(EVALUATION_METHODS)


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
    return int(emd_count + eemd_count + ceemdan_count)


def _signal_class(signal_name):
    if signal_name in SIGNAL_FAMILY:
        return "canonical"
    meta = SIGNAL_VARIANT_METADATA.get(signal_name, {})
    if meta.get("class") == "challenging":
        return "challenging"
    if signal_name in SIGNAL_VARIANT_FAMILY:
        return "canonical_variant"
    return "unknown"


def _audit_signal_truth_components(signals):
    rows = []
    t = np.linspace(0.0, 1.0, int(DEFAULT_N), endpoint=False)
    for signal_name in sorted(set(signals)):
        row = {
            "signal": signal_name,
            "taxonomy_class": _signal_class(signal_name),
            "registry_signal_finite": False,
            "true_components_available": False,
            "true_components_finite": False,
            "n_true_components": 0,
            "n_samples": int(DEFAULT_N),
            "error": "",
        }
        try:
            x = np.asarray(get_signal(signal_name, t), dtype=float)
            row["registry_signal_finite"] = bool(x.shape == t.shape and np.all(np.isfinite(x)))
            comps = get_true_components(signal_name, t)
            if comps is not None:
                comps = np.asarray(comps, dtype=float)
                row["true_components_available"] = bool(comps.ndim == 2 and comps.shape[1] == t.size and comps.shape[0] > 0)
                row["true_components_finite"] = bool(row["true_components_available"] and np.all(np.isfinite(comps)))
                row["n_true_components"] = int(comps.shape[0]) if comps.ndim == 2 else 0
        except Exception as exc:
            row["error"] = repr(exc)
        row["passed"] = bool(
            row["taxonomy_class"] != "unknown"
            and row["registry_signal_finite"]
            and row["true_components_available"]
            and row["true_components_finite"]
        )
        rows.append(row)
    return rows


def _expected_count_rows():
    parameter_design_n = _product(PARAMETER_SENSITIVITY_FACTORS.values())
    contamination_design_counts = {
        signal_name: len(_designs_for_signal(signal_name))
        for signal_name in V557_SECTION6_2_CONTAMINATION_SIGNALS
    }
    s63b_cfg = EMD_FAMILY_BENCHMARK_MODES["representative"]
    comparator_config_count = _emd_family_config_count()
    n_grid = (250, 500, 1000, 2000, 4000)
    s64_seeds = (0, 1)
    s64_repeats = 3
    rows = [
        {
            "module": "6.1",
            "calculation": "parameter_designs * signals * noises * target_snr_levels",
            "dimension_detail": (
                f"{parameter_design_n} * {len(PARAMETER_SENSITIVITY_SIGNALS)} * "
                f"{len(PARAMETER_SENSITIVITY_NOISES)} * {len(PARAMETER_SENSITIVITY_TARGET_SNR_DB_LEVELS)}"
            ),
            "expected_rows": int(
                parameter_design_n
                * len(PARAMETER_SENSITIVITY_SIGNALS)
                * len(PARAMETER_SENSITIVITY_NOISES)
                * len(PARAMETER_SENSITIVITY_TARGET_SNR_DB_LEVELS)
            ),
            "expected_case_blocks": int(
                len(PARAMETER_SENSITIVITY_SIGNALS)
                * len(PARAMETER_SENSITIVITY_NOISES)
                * len(PARAMETER_SENSITIVITY_TARGET_SNR_DB_LEVELS)
            ),
        },
        {
            "module": "6.2",
            "calculation": "signals * contamination_designs_per_signal * seeds * methods",
            "dimension_detail": (
                f"{len(V557_SECTION6_2_CONTAMINATION_SIGNALS)} * "
                f"{sorted(set(contamination_design_counts.values()))} * 3 * {len(METHODS)}"
            ),
            "expected_rows": int(sum(contamination_design_counts.values()) * 3 * len(METHODS)),
            "expected_case_blocks": int(sum(contamination_design_counts.values()) * 3),
        },
        {
            "module": "6.3A",
            "calculation": "signals * noises * target_snr_levels * seeds * methods",
            "dimension_detail": (
                f"{len(SIGNAL_VARIANT_FAMILY)} * {len(SIGNAL_VARIANT_NOISES)} * "
                f"{len(SIGNAL_VARIANT_TARGET_SNR_DB_LEVELS)} * {len(SIGNAL_VARIANT_SEEDS)} * {len(METHODS)}"
            ),
            "expected_rows": int(
                len(SIGNAL_VARIANT_FAMILY)
                * len(SIGNAL_VARIANT_NOISES)
                * len(SIGNAL_VARIANT_TARGET_SNR_DB_LEVELS)
                * len(SIGNAL_VARIANT_SEEDS)
                * len(METHODS)
            ),
            "expected_case_blocks": int(
                len(SIGNAL_VARIANT_FAMILY)
                * len(SIGNAL_VARIANT_NOISES)
                * len(SIGNAL_VARIANT_TARGET_SNR_DB_LEVELS)
                * len(SIGNAL_VARIANT_SEEDS)
            ),
        },
        {
            "module": "6.3B",
            "calculation": "representative_signals * noises * target_snr_levels * baseline_configurations",
            "dimension_detail": (
                f"{len(s63b_cfg['signals'])} * {len(s63b_cfg['noises'])} * "
                f"{len(PARAMETER_SENSITIVITY_TARGET_SNR_DB_LEVELS)} * {comparator_config_count}"
            ),
            "expected_rows": int(
                len(s63b_cfg["signals"])
                * len(s63b_cfg["noises"])
                * len(PARAMETER_SENSITIVITY_TARGET_SNR_DB_LEVELS)
                * comparator_config_count
            ),
            "expected_case_blocks": int(
                len(s63b_cfg["signals"])
                * len(s63b_cfg["noises"])
                * len(PARAMETER_SENSITIVITY_TARGET_SNR_DB_LEVELS)
            ),
        },
        {
            "module": "6.4",
            "calculation": "n_grid * signals * seeds * methods * timing_repeats",
            "dimension_detail": (
                f"{len(n_grid)} * {len(V557_SECTION6_4_SCALING_SIGNALS)} * "
                f"{len(s64_seeds)} * {len(METHODS)} * {s64_repeats}"
            ),
            "expected_rows": int(
                len(n_grid)
                * len(V557_SECTION6_4_SCALING_SIGNALS)
                * len(s64_seeds)
                * len(METHODS)
                * s64_repeats
            ),
            "expected_case_blocks": int(len(n_grid) * len(V557_SECTION6_4_SCALING_SIGNALS) * len(s64_seeds)),
        },
        {
            "module": "V5.56",
            "calculation": "signals * noises * target_snr_levels * seeds * methods",
            "dimension_detail": (
                f"{len(V556_WAVEFORM_STABILITY_SIGNALS)} * {len(V556_WAVEFORM_STABILITY_NOISES)} * "
                f"{len(V556_WAVEFORM_STABILITY_TARGET_SNR_DB_LEVELS)} * "
                f"{len(V556_WAVEFORM_STABILITY_SEEDS)} * {len(V556_WAVEFORM_STABILITY_METHODS)}"
            ),
            "expected_rows": int(
                len(V556_WAVEFORM_STABILITY_SIGNALS)
                * len(V556_WAVEFORM_STABILITY_NOISES)
                * len(V556_WAVEFORM_STABILITY_TARGET_SNR_DB_LEVELS)
                * len(V556_WAVEFORM_STABILITY_SEEDS)
                * len(V556_WAVEFORM_STABILITY_METHODS)
            ),
            "expected_case_blocks": int(
                len(V556_WAVEFORM_STABILITY_SIGNALS)
                * len(V556_WAVEFORM_STABILITY_NOISES)
                * len(V556_WAVEFORM_STABILITY_TARGET_SNR_DB_LEVELS)
                * len(V556_WAVEFORM_STABILITY_METHODS)
            ),
        },
    ]
    for row in rows:
        row["passed"] = bool(row["expected_rows"] > 0 and row["expected_case_blocks"] > 0)
    return rows


def _common_anchor_rows():
    checks = [
        ("6.1", PARAMETER_SENSITIVITY_SIGNALS, PARAMETER_SENSITIVITY_NOISES, PARAMETER_SENSITIVITY_TARGET_SNR_DB_LEVELS),
        ("V5.56", V556_WAVEFORM_STABILITY_SIGNALS, V556_WAVEFORM_STABILITY_NOISES, V556_WAVEFORM_STABILITY_TARGET_SNR_DB_LEVELS),
    ]
    rows = []
    for module, signals, noises, snrs in checks:
        missing_signals = sorted(set(V557_SECTION6_COMMON_ANCHOR_SIGNALS) - set(signals))
        missing_noises = sorted(set(V557_SECTION6_COMMON_ANCHOR_NOISES) - set(noises))
        missing_snrs = sorted(set(V557_SECTION6_COMMON_ANCHOR_TARGET_SNR_DB_LEVELS) - set(snrs))
        rows.append({
            "module": module,
            "anchor_policy": "common anchor where scientifically applicable",
            "missing_anchor_signals": ";".join(missing_signals),
            "missing_anchor_noises": ";".join(missing_noises),
            "missing_anchor_target_snr_db_levels": ";".join(str(x) for x in missing_snrs),
            "passed": bool(not missing_signals and not missing_noises and not missing_snrs),
        })
    missing_noises = sorted(set(V557_SECTION6_COMMON_ANCHOR_NOISES) - set(SIGNAL_VARIANT_NOISES))
    missing_snrs = sorted(set(V557_SECTION6_COMMON_ANCHOR_TARGET_SNR_DB_LEVELS) - set(SIGNAL_VARIANT_TARGET_SNR_DB_LEVELS))
    has_canonical_variants = any(
        _signal_class(signal_name) == "canonical_variant"
        for signal_name in SIGNAL_VARIANT_FAMILY
    )
    has_challenging_extension = all(
        signal_name in SIGNAL_VARIANT_FAMILY
        for signal_name in V557_SECTION6_3_CHALLENGING_SIGNAL_EXTENSION
    )
    rows.append({
        "module": "6.3A",
        "anchor_policy": (
            "signal-robustness section uses canonical-family variants plus explicit challenging extension; "
            "base anchor signals are not required"
        ),
        "missing_anchor_signals": "",
        "missing_anchor_noises": ";".join(missing_noises),
        "missing_anchor_target_snr_db_levels": ";".join(str(x) for x in missing_snrs),
        "has_canonical_variants": bool(has_canonical_variants),
        "has_challenging_extension": bool(has_challenging_extension),
        "passed": bool(
            not missing_noises
            and not missing_snrs
            and has_canonical_variants
            and has_challenging_extension
        ),
    })
    rows.extend([
        {
            "module": "6.2",
            "anchor_policy": "not forced; contamination-design estimand uses huber contamination by construction",
            "missing_anchor_signals": "",
            "missing_anchor_noises": "gaussian;impulsive",
            "missing_anchor_target_snr_db_levels": "5.0;25.0",
            "passed": True,
        },
        {
            "module": "6.4",
            "anchor_policy": "not forced; controlled runtime scaling uses gaussian at 15 dB plus one challenging signal",
            "missing_anchor_signals": "close_frequencies;impulsive_transient",
            "missing_anchor_noises": "impulsive;huber_contamination",
            "missing_anchor_target_snr_db_levels": "5.0;25.0",
            "passed": True,
        },
    ])
    return rows


def _claim_gate_rows(algorithm_root):
    v553_path = algorithm_root / "16v_v553_section6_claim_qualification" / "v553_section6_claim_qualification_dashboard.json"
    v557_path = algorithm_root / "16z_v557_section6_challenging_signal_extension" / "v557_section6_challenging_signal_extension_dashboard.json"
    v553_source = Path(__file__).with_name("experiment_v553_section6_claim_qualification.py")
    return [
        {
            "check_id": "v557_amendment_artifact_exists",
            "passed": bool(v557_path.exists()),
            "detail": str(v557_path),
        },
        {
            "check_id": "v553_claim_gate_source_exists",
            "passed": bool(v553_source.exists()),
            "detail": str(v553_source),
        },
        {
            "check_id": "previous_v553_dashboard_detected",
            "passed": bool(v553_path.exists()),
            "detail": str(v553_path),
            "interpretation": (
                "Existing V5.53 dashboard may describe the pre-V5.57 executed subset. "
                "V5.57-updated claims require rerunning affected modules before claim qualification."
            ),
        },
        {
            "check_id": "aggregation_rule_not_redefined_by_v558",
            "passed": True,
            "detail": "V5.58 performs pre-execution audit only and does not introduce new metrics, ranking, or aggregation rules.",
        },
        {
            "check_id": "v553_expected_counts_need_v557_refresh_after_rerun",
            "passed": True,
            "detail": (
                "After V5.57 reruns, V5.53 or its successor must validate updated 6.1 case blocks "
                "and amended 6.2/6.3A/6.4 counts before authorizing claims."
            ),
        },
    ]


def run_v558_section6_pre_execution_audit(output_root):
    output_root = ensure_dir(output_root)
    algorithm_root = output_root.parent if output_root.name.startswith("16") else output_root
    all_signals = (
        list(PARAMETER_SENSITIVITY_SIGNALS)
        + list(V557_SECTION6_2_CONTAMINATION_SIGNALS)
        + list(SIGNAL_VARIANT_FAMILY)
        + list(V557_SECTION6_4_SCALING_SIGNALS)
        + list(V556_WAVEFORM_STABILITY_SIGNALS)
    )
    signal_rows = _audit_signal_truth_components(all_signals)
    expected_rows = _expected_count_rows()
    anchor_rows = _common_anchor_rows()
    claim_gate_rows = _claim_gate_rows(algorithm_root)

    signal_passed = all(bool(r["passed"]) for r in signal_rows)
    expected_passed = all(bool(r["passed"]) for r in expected_rows)
    anchor_passed = all(bool(r["passed"]) for r in anchor_rows)
    claim_gate_passed = all(bool(r["passed"]) for r in claim_gate_rows)
    all_passed = bool(signal_passed and expected_passed and anchor_passed and claim_gate_passed)

    dashboard = {
        "schema_version": V558_SECTION6_PRE_EXECUTION_AUDIT_VERSION,
        "created_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "amended_protocol": V557_SECTION6_AMENDMENT_VERSION,
        "audit_status": "passed" if all_passed else "not_passed",
        "v557_execution_protocol_frozen": bool(all_passed),
        "algorithm_runs_performed": False,
        "default_n": int(DEFAULT_N),
        "default_fs": float(DEFAULT_FS),
        "checks": {
            "signal_taxonomy_and_truth_components_passed": bool(signal_passed),
            "expected_count_accounting_passed": bool(expected_passed),
            "common_anchor_policy_passed": bool(anchor_passed),
            "claim_gate_continuity_passed": bool(claim_gate_passed),
        },
        "expected_counts": {
            row["module"]: int(row["expected_rows"])
            for row in expected_rows
        },
        "expected_case_blocks": {
            row["module"]: int(row["expected_case_blocks"])
            for row in expected_rows
        },
        "claim_boundary": (
            "V5.58 audits whether the V5.57 amended Section 6 execution protocol "
            "is internally coherent before rerun. It does not execute algorithms, "
            "does not authorize scientific claims, and does not change primary "
            "metrics, ranking, or aggregation logic."
        ),
        "required_next_step_if_passed": (
            "Rerun affected V5.57 Section 6 modules, then rerun Section 6 claim qualification "
            "against the updated artifacts."
        ),
    }
    write_json(dashboard, output_root / "v558_section6_pre_execution_audit_dashboard.json")
    write_csv(signal_rows, output_root / "v558_signal_taxonomy_and_truth_component_audit.csv")
    write_csv(expected_rows, output_root / "v558_expected_count_audit.csv")
    write_csv(anchor_rows, output_root / "v558_common_anchor_coverage_audit.csv")
    write_csv(claim_gate_rows, output_root / "v558_claim_gate_continuity_audit.csv")
    return dashboard
