#!/usr/bin/python
# coding: UTF-8

"""V5.66 Section 6.3A two-axis interaction design audit.

This audit freezes the signal-specification robustness scope as:
    7 canonical families x 7 specifications
    8 challenging families x 3 specifications
for 73 total signal specifications.  It performs no method evaluations.
"""

from collections import defaultdict
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
    V566_SECTION6_3A_CANONICAL_INTERACTION_VARIANTS,
    V566_SECTION6_3A_CANONICAL_TWO_AXIS_INTERACTION_VERSION,
    V566_SECTION6_3A_CANONICAL_TWO_AXIS_SIGNAL_SPECS,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json
from signal_bank.synthetic_signals import (
    SIGNAL_VARIANT_METADATA,
    get_signal,
    get_true_components,
    get_true_frequencies,
)


CANONICAL_FAMILIES = (
    "stationary_multi_sine",
    "chirp",
    "am_fm",
    "frequency_jump",
    "impulsive_transient",
    "intermittent_oscillation",
    "close_frequencies",
)
CHALLENGING_FAMILIES = (
    "crossing_chirps",
    "time_varying_close_frequencies",
    "piecewise_am_fm_discontinuity",
    "damped_oscillation",
    "trend_plus_oscillation",
    "buried_weak_component",
    "non_sinusoidal_periodic",
    "transient_train",
)


def _family(signal_name):
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


def _role(signal_name):
    meta = SIGNAL_VARIANT_METADATA.get(signal_name, {})
    if signal_name in CANONICAL_FAMILIES:
        return "canonical_anchor"
    if signal_name in V557_SECTION6_3_CHALLENGING_SIGNAL_EXTENSION:
        return "challenging_anchor"
    return meta.get("variant_role", "unclassified")


def _family_type(signal_name):
    return "canonical" if _family(signal_name) in CANONICAL_FAMILIES else "challenging"


def _registry_rows():
    t = np.arange(int(DEFAULT_N), dtype=float) / float(DEFAULT_N)
    rows = []
    for signal_name in SIGNAL_VARIANT_FAMILY:
        signal = get_signal(signal_name, t)
        components = get_true_components(signal_name, t)
        frequencies = get_true_frequencies(signal_name, t)
        meta = SIGNAL_VARIANT_METADATA.get(signal_name, {})
        rows.append({
            "signal": signal_name,
            "family": _family(signal_name),
            "family_type": _family_type(signal_name),
            "role": _role(signal_name),
            "difficulty_axis": meta.get("difficulty_axis", "anchor"),
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


def _family_rows(registry_rows):
    rows = []
    for family in sorted(set(CANONICAL_FAMILIES + CHALLENGING_FAMILIES)):
        fam_rows = [r for r in registry_rows if r["family"] == family]
        roles = defaultdict(int)
        axes = set()
        for row in fam_rows:
            roles[row["role"]] += 1
            axes.add(row["difficulty_axis"])
        family_type = "canonical" if family in CANONICAL_FAMILIES else "challenging"
        expected = 7 if family_type == "canonical" else 3
        rows.append({
            "family": family,
            "family_type": family_type,
            "n_signal_specifications": int(len(fam_rows)),
            "expected_signal_specifications": int(expected),
            "family_count_passed": bool(len(fam_rows) == expected),
            "n_anchors": int(roles.get("canonical_anchor", 0) + roles.get("challenging_anchor", 0)),
            "n_single_axis_or_existing_variants": int(
                roles.get("existing_canonical_variant", 0)
                + roles.get("canonical_structured_perturbation", 0)
                + roles.get("challenging_family_variant", 0)
            ),
            "n_interaction_corners": int(roles.get("canonical_interaction_corner", 0)),
            "axes": ";".join(sorted(axes)),
            "signals": ";".join(r["signal"] for r in fam_rows),
        })
    return rows


def run_v566_section6_3a_two_axis_interaction_design(output_root):
    output_root = ensure_dir(output_root)
    registry_rows = _registry_rows()
    family_rows = _family_rows(registry_rows)
    n_signals = len(SIGNAL_VARIANT_FAMILY)
    n_case_rows = (
        n_signals
        * len(SIGNAL_VARIANT_NOISES)
        * len(SIGNAL_VARIANT_TARGET_SNR_DB_LEVELS)
        * len(SIGNAL_VARIANT_SEEDS)
    )
    n_method_evaluations = n_case_rows * len(EVALUATION_METHODS)
    dashboard = {
        "schema_version": V566_SECTION6_3A_CANONICAL_TWO_AXIS_INTERACTION_VERSION,
        "created_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "module_status": (
            "protocol_amended_registry_audit_passed"
            if all(r["registry_passed"] for r in registry_rows)
            and all(r["family_count_passed"] for r in family_rows)
            and n_signals == 73
            else "registry_or_count_audit_failed"
        ),
        "algorithm_runs_performed": False,
        "primary_metric_schema_changed": False,
        "section6_scientific_questions_changed": False,
        "ranking_procedure_changed": False,
        "amendment_role": "Section 6.3A canonical two-axis interaction extension",
        "canonical_design": "7 families x 7 specs = anchor + 4 single-axis perturbations + 2 interaction corners",
        "challenging_design": "8 families x 3 specs = anchor + 2 controlled difficulty variants",
        "canonical_signal_specifications": int(len(V566_SECTION6_3A_CANONICAL_TWO_AXIS_SIGNAL_SPECS)),
        "challenging_signal_specifications": int(
            len(V557_SECTION6_3_CHALLENGING_SIGNAL_EXTENSION)
            + len(V564_SECTION6_3A_CHALLENGING_FAMILY_VARIANTS)
        ),
        "total_signal_specifications": int(n_signals),
        "expected_6_3a_case_rows_after_amendment": int(n_case_rows),
        "expected_6_3a_method_evaluations_after_amendment": int(n_method_evaluations),
        "canonical_interaction_variants": list(V566_SECTION6_3A_CANONICAL_INTERACTION_VARIANTS),
        "noises": list(SIGNAL_VARIANT_NOISES),
        "target_snr_db_levels": list(SIGNAL_VARIANT_TARGET_SNR_DB_LEVELS),
        "seeds": list(SIGNAL_VARIANT_SEEDS),
        "methods": list(EVALUATION_METHODS),
        "not_outcome_driven_statement": (
            "The amendment is a pre-execution design refinement that tests "
            "joint departures along two prespecified canonical specification "
            "axes per family. It does not respond to observed robustness outcomes."
        ),
        "claim_boundary": (
            "Claims about the V5.66 73-specification signal-robustness scope "
            "require rerunning 6.3A and post-execution qualification."
        ),
    }
    write_json(dashboard, output_root / "v566_section6_3a_two_axis_interaction_design_dashboard.json")
    write_csv(registry_rows, output_root / "v566_signal_specification_registry_audit.csv")
    write_csv(family_rows, output_root / "v566_family_count_and_axis_audit.csv")
    return {
        "dashboard": dashboard,
        "registry_rows": registry_rows,
        "family_rows": family_rows,
    }
