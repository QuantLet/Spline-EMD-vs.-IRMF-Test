#!/usr/bin/python
# coding: UTF-8

"""V5.64 Section 6.3A challenging-family variant extension.

This is a protocol amendment and registry audit only.  It symmetrizes the
signal-variant robustness design by adding controlled within-family variants
for selected challenging signal regimes.  It does not run method evaluations.
"""

from datetime import datetime, timezone

import numpy as np

from project_config import (
    DEFAULT_FS,
    DEFAULT_N,
    EVALUATION_METHODS,
    SIGNAL_VARIANT_FAMILY,
    SIGNAL_VARIANT_NOISES,
    SIGNAL_VARIANT_SEEDS,
    SIGNAL_VARIANT_TARGET_SNR_DB_LEVELS,
    V557_SECTION6_3_CHALLENGING_SIGNAL_EXTENSION,
    V564_SECTION6_3A_CHALLENGING_FAMILY_CONDITIONS,
    V564_SECTION6_3A_CHALLENGING_FAMILY_VARIANTS,
    V564_SECTION6_3A_CHALLENGING_VARIANT_EXTENSION_VERSION,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json
from signal_bank.synthetic_signals import (
    SIGNAL_VARIANT_METADATA,
    get_signal,
    get_true_components,
    get_true_frequencies,
)


def _variant_family(signal_name):
    if signal_name.startswith("crossing_chirps"):
        return "crossing_chirps"
    if signal_name.startswith("time_varying_close_frequencies"):
        return "time_varying_close_frequencies"
    if signal_name.startswith("piecewise_am_fm_discontinuity"):
        return "piecewise_am_fm_discontinuity"
    if signal_name.startswith("damped_oscillation"):
        return "damped_oscillation"
    if signal_name.startswith("trend_plus_oscillation"):
        return "trend_plus_oscillation"
    if signal_name.startswith("buried_weak_component"):
        return "buried_weak_component"
    if signal_name.startswith("non_sinusoidal_periodic"):
        return "non_sinusoidal_periodic"
    if signal_name.startswith("transient_train"):
        return "transient_train"
    return signal_name


def _difficulty_axis(signal_name):
    meta = SIGNAL_VARIANT_METADATA.get(signal_name, {})
    return meta.get("difficulty_axis", "anchor_default")


def _signal_audit_rows():
    t = np.arange(int(DEFAULT_N), dtype=float) / float(DEFAULT_N)
    rows = []
    for signal_name in V564_SECTION6_3A_CHALLENGING_FAMILY_CONDITIONS:
        signal = get_signal(signal_name, t)
        components = get_true_components(signal_name, t)
        frequencies = get_true_frequencies(signal_name, t)
        meta = SIGNAL_VARIANT_METADATA.get(signal_name, {})
        rows.append({
            "signal": signal_name,
            "family": _variant_family(signal_name),
            "role": "anchor" if signal_name in V557_SECTION6_3_CHALLENGING_SIGNAL_EXTENSION else "controlled_variant",
            "difficulty_axis": _difficulty_axis(signal_name),
            "metadata_class": meta.get("class"),
            "metadata_regime": meta.get("regime"),
            "n_samples": int(signal.shape[0]) if hasattr(signal, "shape") else 0,
            "n_true_components": int(components.shape[0]) if components is not None and components.ndim == 2 else 0,
            "n_true_frequency_curves": int(len(frequencies or [])),
            "signal_all_finite": bool(np.all(np.isfinite(signal))),
            "components_all_finite": bool(
                components is not None and components.size > 0 and np.all(np.isfinite(components))
            ),
            "registry_passed": bool(
                signal.shape[0] == int(DEFAULT_N)
                and components is not None
                and components.ndim == 2
                and components.shape[1] == int(DEFAULT_N)
                and np.all(np.isfinite(signal))
                and np.all(np.isfinite(components))
            ),
        })
    return rows


def run_v564_section6_3a_challenging_family_variant_extension(output_root):
    output_root = ensure_dir(output_root)
    audit_rows = _signal_audit_rows()
    family_rows = []
    for family in sorted({_variant_family(s) for s in V564_SECTION6_3A_CHALLENGING_FAMILY_CONDITIONS}):
        names = [s for s in V564_SECTION6_3A_CHALLENGING_FAMILY_CONDITIONS if _variant_family(s) == family]
        family_rows.append({
            "challenging_family": family,
            "n_conditions": int(len(names)),
            "conditions": ";".join(names),
            "difficulty_axes": ";".join(sorted({_difficulty_axis(s) for s in names})),
        })

    n_signals = len(SIGNAL_VARIANT_FAMILY)
    n_case_rows = (
        n_signals
        * len(SIGNAL_VARIANT_NOISES)
        * len(SIGNAL_VARIANT_TARGET_SNR_DB_LEVELS)
        * len(SIGNAL_VARIANT_SEEDS)
    )
    n_method_evaluations = n_case_rows * len(EVALUATION_METHODS)
    old_signal_count = n_signals - len(V564_SECTION6_3A_CHALLENGING_FAMILY_VARIANTS)
    incremental_case_rows = (
        len(V564_SECTION6_3A_CHALLENGING_FAMILY_VARIANTS)
        * len(SIGNAL_VARIANT_NOISES)
        * len(SIGNAL_VARIANT_TARGET_SNR_DB_LEVELS)
        * len(SIGNAL_VARIANT_SEEDS)
    )

    dashboard = {
        "schema_version": V564_SECTION6_3A_CHALLENGING_VARIANT_EXTENSION_VERSION,
        "created_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "module_status": "protocol_amended_registry_audit_passed" if all(r["registry_passed"] for r in audit_rows) else "registry_audit_failed",
        "algorithm_runs_performed": False,
        "primary_metric_schema_changed": False,
        "section6_scientific_questions_changed": False,
        "ranking_procedure_changed": False,
        "amendment_role": "Section 6.3A signal-family and variant robustness extension",
        "reason": (
            "Symmetrize the signal-robustness design by evaluating within-family "
            "perturbations for challenging signal structures, rather than only "
            "reusing fixed challenging anchors."
        ),
        "not_outcome_driven_statement": (
            "The extension is a pre-execution design amendment to improve "
            "challenging-family coverage; it is not introduced as a response to "
            "a favorable or unfavorable result."
        ),
        "challenging_anchor_signals": list(V557_SECTION6_3_CHALLENGING_SIGNAL_EXTENSION),
        "new_challenging_family_variants": list(V564_SECTION6_3A_CHALLENGING_FAMILY_VARIANTS),
        "challenging_family_conditions": list(V564_SECTION6_3A_CHALLENGING_FAMILY_CONDITIONS),
        "new_signal_variant_family_size": int(n_signals),
        "previous_signal_variant_family_size": int(old_signal_count),
        "expected_6_3a_case_rows_after_amendment": int(n_case_rows),
        "expected_6_3a_method_evaluations_after_amendment": int(n_method_evaluations),
        "incremental_case_rows_from_v564_variants": int(incremental_case_rows),
        "incremental_method_evaluations_from_v564_variants": int(
            incremental_case_rows * len(EVALUATION_METHODS)
        ),
        "noises": list(SIGNAL_VARIANT_NOISES),
        "target_snr_db_levels": list(SIGNAL_VARIANT_TARGET_SNR_DB_LEVELS),
        "seeds": list(SIGNAL_VARIANT_SEEDS),
        "methods": list(EVALUATION_METHODS),
        "default_n": int(DEFAULT_N),
        "default_fs": float(DEFAULT_FS),
        "claim_boundary": (
            "Existing V5.57/V5.59 Section 6.3A claims remain scoped to the "
            "previous executed signal set. Claims about challenging-family "
            "variant robustness require rerunning 6.3A under this amended signal "
            "scope and re-running post-execution qualification."
        ),
    }

    write_json(dashboard, output_root / "v564_section6_3a_challenging_family_variant_extension_dashboard.json")
    write_csv(audit_rows, output_root / "v564_challenging_family_variant_registry_audit.csv")
    write_csv(family_rows, output_root / "v564_challenging_family_variant_design_summary.csv")
    return {
        "dashboard": dashboard,
        "audit_rows": audit_rows,
        "family_rows": family_rows,
    }
