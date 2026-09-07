#!/usr/bin/python
# coding: UTF-8

"""V5.64A Section 6.3A canonical-anchor reference amendment.

This is a protocol amendment and registry audit only.  It adds the seven
Section 5 canonical signal specifications as reference anchors for the existing
canonical-family variants, without changing metrics, ranking, or scientific
questions.
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
    V564_SECTION6_3A_CHALLENGING_FAMILY_VARIANTS,
    V564A_SECTION6_3A_CANONICAL_ANCHOR_REFERENCE_VERSION,
    V564A_SECTION6_3A_CANONICAL_ANCHORS,
    V564A_SECTION6_3A_CANONICAL_FAMILY_VARIANTS,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json
from signal_bank.synthetic_signals import (
    SIGNAL_VARIANT_METADATA,
    get_signal,
    get_true_components,
    get_true_frequencies,
)


def _canonical_family(signal_name):
    if signal_name.startswith("stationary_multi_sine"):
        return "stationary_multi_sine"
    if signal_name.startswith("chirp"):
        return "chirp"
    if signal_name.startswith("am_fm"):
        return "am_fm"
    if signal_name.startswith("frequency_jump"):
        return "frequency_jump"
    if signal_name.startswith("impulsive_transient"):
        return "impulsive_transient"
    if signal_name.startswith("intermittent"):
        return "intermittent_oscillation"
    if signal_name.startswith("close_freq") or signal_name == "close_frequencies":
        return "close_frequencies"
    return signal_name


def _challenging_family(signal_name):
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


def _registry_status(signal_name, role, family_type):
    t = np.arange(int(DEFAULT_N), dtype=float) / float(DEFAULT_N)
    signal = get_signal(signal_name, t)
    components = get_true_components(signal_name, t)
    frequencies = get_true_frequencies(signal_name, t)
    meta = SIGNAL_VARIANT_METADATA.get(signal_name, {})
    return {
        "signal": signal_name,
        "family_type": family_type,
        "family": _canonical_family(signal_name) if family_type == "canonical" else _challenging_family(signal_name),
        "role": role,
        "metadata_class": meta.get("class", "anchor"),
        "metadata_regime": meta.get("regime", ""),
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
    }


def _family_rows():
    rows = []
    for family in sorted({_canonical_family(s) for s in V564A_SECTION6_3A_CANONICAL_ANCHORS}):
        anchor = [s for s in V564A_SECTION6_3A_CANONICAL_ANCHORS if _canonical_family(s) == family]
        variants = [s for s in V564A_SECTION6_3A_CANONICAL_FAMILY_VARIANTS if _canonical_family(s) == family]
        rows.append({
            "family_type": "canonical",
            "family": family,
            "anchor": anchor[0] if anchor else "",
            "n_variants": int(len(variants)),
            "variants": ";".join(variants),
            "design_role": "anchor_to_multiple_specification_robustness",
        })
    for family in sorted({_challenging_family(s) for s in V557_SECTION6_3_CHALLENGING_SIGNAL_EXTENSION}):
        anchor = [s for s in V557_SECTION6_3_CHALLENGING_SIGNAL_EXTENSION if _challenging_family(s) == family]
        variants = [s for s in V564_SECTION6_3A_CHALLENGING_FAMILY_VARIANTS if _challenging_family(s) == family]
        rows.append({
            "family_type": "challenging",
            "family": family,
            "anchor": anchor[0] if anchor else "",
            "n_variants": int(len(variants)),
            "variants": ";".join(variants),
            "design_role": "anchor_centered_difficulty_perturbation",
        })
    return rows


def run_v564a_section6_3a_canonical_anchor_reference(output_root):
    output_root = ensure_dir(output_root)
    audit_rows = []
    for signal_name in V564A_SECTION6_3A_CANONICAL_ANCHORS:
        audit_rows.append(_registry_status(signal_name, "anchor", "canonical"))
    for signal_name in V564A_SECTION6_3A_CANONICAL_FAMILY_VARIANTS:
        audit_rows.append(_registry_status(signal_name, "controlled_variant", "canonical"))
    for signal_name in V557_SECTION6_3_CHALLENGING_SIGNAL_EXTENSION:
        audit_rows.append(_registry_status(signal_name, "anchor", "challenging"))
    for signal_name in V564_SECTION6_3A_CHALLENGING_FAMILY_VARIANTS:
        audit_rows.append(_registry_status(signal_name, "controlled_variant", "challenging"))

    family_rows = _family_rows()
    n_signals = len(SIGNAL_VARIANT_FAMILY)
    n_case_rows = (
        n_signals
        * len(SIGNAL_VARIANT_NOISES)
        * len(SIGNAL_VARIANT_TARGET_SNR_DB_LEVELS)
        * len(SIGNAL_VARIANT_SEEDS)
    )
    n_method_evaluations = n_case_rows * len(EVALUATION_METHODS)
    incremental_anchor_case_rows = (
        len(V564A_SECTION6_3A_CANONICAL_ANCHORS)
        * len(SIGNAL_VARIANT_NOISES)
        * len(SIGNAL_VARIANT_TARGET_SNR_DB_LEVELS)
        * len(SIGNAL_VARIANT_SEEDS)
    )

    dashboard = {
        "schema_version": V564A_SECTION6_3A_CANONICAL_ANCHOR_REFERENCE_VERSION,
        "created_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "module_status": "protocol_amended_registry_audit_passed" if all(r["registry_passed"] for r in audit_rows) else "registry_audit_failed",
        "algorithm_runs_performed": False,
        "primary_metric_schema_changed": False,
        "section6_scientific_questions_changed": False,
        "ranking_procedure_changed": False,
        "amendment_role": "Section 6.3A canonical-anchor reference amendment",
        "reason": (
            "Add the seven Section 5 canonical anchor specifications as reference "
            "conditions for the existing 19 canonical-family variants, so 6.3A "
            "can evaluate anchor-to-variant preservation for canonical and "
            "challenging families under a common within-family robustness frame."
        ),
        "not_outcome_driven_statement": (
            "The amendment is a pre-execution design clarification introduced to "
            "complete the canonical anchor structure; it is not introduced as a "
            "response to a favorable or unfavorable result."
        ),
        "section5_role": "across challenging signal types",
        "section6_3a_role": "within-family specification robustness",
        "canonical_structure": "7 canonical anchors + 19 controlled canonical-family variants",
        "challenging_structure": "5 challenging anchors + 10 controlled challenging-family variants",
        "canonical_anchors": list(V564A_SECTION6_3A_CANONICAL_ANCHORS),
        "canonical_variants": list(V564A_SECTION6_3A_CANONICAL_FAMILY_VARIANTS),
        "challenging_anchors": list(V557_SECTION6_3_CHALLENGING_SIGNAL_EXTENSION),
        "challenging_variants": list(V564_SECTION6_3A_CHALLENGING_FAMILY_VARIANTS),
        "new_signal_variant_family_size": int(n_signals),
        "previous_v564_signal_variant_family_size": int(n_signals - len(V564A_SECTION6_3A_CANONICAL_ANCHORS)),
        "expected_6_3a_case_rows_after_amendment": int(n_case_rows),
        "expected_6_3a_method_evaluations_after_amendment": int(n_method_evaluations),
        "incremental_case_rows_from_canonical_anchors": int(incremental_anchor_case_rows),
        "incremental_method_evaluations_from_canonical_anchors": int(
            incremental_anchor_case_rows * len(EVALUATION_METHODS)
        ),
        "noises": list(SIGNAL_VARIANT_NOISES),
        "target_snr_db_levels": list(SIGNAL_VARIANT_TARGET_SNR_DB_LEVELS),
        "seeds": list(SIGNAL_VARIANT_SEEDS),
        "methods": list(EVALUATION_METHODS),
        "default_n": int(DEFAULT_N),
        "default_fs": float(DEFAULT_FS),
        "claim_boundary": (
            "Existing V5.57/V5.59/V5.64 Section 6.3A claims remain scoped to "
            "their executed signal sets. Claims about the V5.64A anchor-centered "
            "41-specification scope require rerunning 6.3A and post-execution "
            "qualification."
        ),
    }

    write_json(dashboard, output_root / "v564a_section6_3a_canonical_anchor_reference_dashboard.json")
    write_csv(audit_rows, output_root / "v564a_signal_specification_registry_audit.csv")
    write_csv(family_rows, output_root / "v564a_anchor_variant_family_design_summary.csv")
    return {
        "dashboard": dashboard,
        "audit_rows": audit_rows,
        "family_rows": family_rows,
    }
