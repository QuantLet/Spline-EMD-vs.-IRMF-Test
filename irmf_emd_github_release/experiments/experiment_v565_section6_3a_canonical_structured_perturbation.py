#!/usr/bin/python
# coding: UTF-8

"""V5.65 Section 6.3A canonical structured perturbation extension.

This protocol audit adds anchor-referenced canonical perturbation axes without
running method evaluations or changing the primary metric schema.
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
    V564A_SECTION6_3A_CANONICAL_ANCHORS,
    V564A_SECTION6_3A_CANONICAL_FAMILY_VARIANTS,
    V565_SECTION6_3A_CANONICAL_STRUCTURED_PERTURBATION_VERSION,
    V565_SECTION6_3A_CANONICAL_STRUCTURED_VARIANTS,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json
from signal_bank.synthetic_signals import (
    SIGNAL_VARIANT_METADATA,
    get_signal,
    get_true_components,
    get_true_frequencies,
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
    if signal_name in V564A_SECTION6_3A_CANONICAL_ANCHORS:
        return "canonical_anchor"
    if signal_name in V564A_SECTION6_3A_CANONICAL_FAMILY_VARIANTS:
        return "existing_canonical_variant"
    if signal_name in V565_SECTION6_3A_CANONICAL_STRUCTURED_VARIANTS:
        return "canonical_structured_perturbation"
    if signal_name in V557_SECTION6_3_CHALLENGING_SIGNAL_EXTENSION:
        return "challenging_anchor"
    if signal_name in V564_SECTION6_3A_CHALLENGING_FAMILY_VARIANTS:
        return "challenging_controlled_variant"
    return "other"


def _family_type(signal_name):
    return "canonical" if _role(signal_name).startswith("canonical") or _role(signal_name).startswith("existing_canonical") else "challenging"


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
            "difficulty_axis": meta.get("difficulty_axis", "anchor_or_existing_axis"),
            "metadata_class": meta.get("class", "anchor"),
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


def _axis_rows(registry_rows):
    by_family_axis = defaultdict(list)
    for row in registry_rows:
        if row["family_type"] != "canonical":
            continue
        by_family_axis[(row["family"], row["difficulty_axis"])].append(row["signal"])
    rows = []
    for (family, axis), signals in sorted(by_family_axis.items()):
        rows.append({
            "family": family,
            "axis": axis,
            "n_signal_specifications": int(len(signals)),
            "signals": ";".join(signals),
        })
    return rows


def _family_rows(registry_rows):
    rows = []
    for family in sorted({_family(s) for s in SIGNAL_VARIANT_FAMILY}):
        fam_rows = [r for r in registry_rows if r["family"] == family]
        roles = defaultdict(int)
        axes = set()
        for row in fam_rows:
            roles[row["role"]] += 1
            axes.add(row["difficulty_axis"])
        rows.append({
            "family": family,
            "family_type": fam_rows[0]["family_type"] if fam_rows else "",
            "n_signal_specifications": int(len(fam_rows)),
            "n_anchors": int(roles.get("canonical_anchor", 0) + roles.get("challenging_anchor", 0)),
            "n_existing_canonical_variants": int(roles.get("existing_canonical_variant", 0)),
            "n_canonical_structured_perturbations": int(roles.get("canonical_structured_perturbation", 0)),
            "n_challenging_controlled_variants": int(roles.get("challenging_controlled_variant", 0)),
            "axes": ";".join(sorted(axes)),
            "signals": ";".join(r["signal"] for r in fam_rows),
        })
    return rows


def run_v565_section6_3a_canonical_structured_perturbation(output_root):
    output_root = ensure_dir(output_root)
    registry_rows = _registry_rows()
    axis_rows = _axis_rows(registry_rows)
    family_rows = _family_rows(registry_rows)
    n_signals = len(SIGNAL_VARIANT_FAMILY)
    n_case_rows = (
        n_signals
        * len(SIGNAL_VARIANT_NOISES)
        * len(SIGNAL_VARIANT_TARGET_SNR_DB_LEVELS)
        * len(SIGNAL_VARIANT_SEEDS)
    )
    n_method_evaluations = n_case_rows * len(EVALUATION_METHODS)
    incremental_case_rows = (
        len(V565_SECTION6_3A_CANONICAL_STRUCTURED_VARIANTS)
        * len(SIGNAL_VARIANT_NOISES)
        * len(SIGNAL_VARIANT_TARGET_SNR_DB_LEVELS)
        * len(SIGNAL_VARIANT_SEEDS)
    )
    dashboard = {
        "schema_version": V565_SECTION6_3A_CANONICAL_STRUCTURED_PERTURBATION_VERSION,
        "created_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "module_status": "protocol_amended_registry_audit_passed" if all(r["registry_passed"] for r in registry_rows) else "registry_audit_failed",
        "algorithm_runs_performed": False,
        "primary_metric_schema_changed": False,
        "section6_scientific_questions_changed": False,
        "ranking_procedure_changed": False,
        "amendment_role": "Section 6.3A canonical structured perturbation-axis extension",
        "reason": (
            "Improve canonical-family coverage by adding prespecified "
            "anchor-referenced structural perturbation axes rather than "
            "arbitrarily increasing named variant counts."
        ),
        "not_outcome_driven_statement": (
            "The extension is a pre-execution design amendment to improve "
            "perturbation-axis coverage and was not introduced in response to "
            "observed robustness outcomes."
        ),
        "canonical_structure": "7 anchors + 19 existing variants + 19 structured perturbation variants",
        "challenging_structure": "8 anchors + 16 controlled challenging-family variants",
        "new_signal_variant_family_size": int(n_signals),
        "previous_v564a_signal_variant_family_size": int(n_signals - len(V565_SECTION6_3A_CANONICAL_STRUCTURED_VARIANTS)),
        "expected_6_3a_case_rows_after_amendment": int(n_case_rows),
        "expected_6_3a_method_evaluations_after_amendment": int(n_method_evaluations),
        "incremental_case_rows_from_v565_variants": int(incremental_case_rows),
        "incremental_method_evaluations_from_v565_variants": int(incremental_case_rows * len(EVALUATION_METHODS)),
        "noises": list(SIGNAL_VARIANT_NOISES),
        "target_snr_db_levels": list(SIGNAL_VARIANT_TARGET_SNR_DB_LEVELS),
        "seeds": list(SIGNAL_VARIANT_SEEDS),
        "methods": list(EVALUATION_METHODS),
        "claim_boundary": (
            "Existing Section 6.3A claims remain scoped to previously executed "
            "signal sets. Claims about canonical structured perturbation-axis "
            "robustness require rerunning 6.3A and post-execution qualification."
        ),
    }
    write_json(dashboard, output_root / "v565_section6_3a_canonical_structured_perturbation_dashboard.json")
    write_csv(registry_rows, output_root / "v565_signal_specification_registry_audit.csv")
    write_csv(family_rows, output_root / "v565_family_design_summary.csv")
    write_csv(axis_rows, output_root / "v565_canonical_axis_coverage_summary.csv")
    return {
        "dashboard": dashboard,
        "registry_rows": registry_rows,
        "family_rows": family_rows,
        "axis_rows": axis_rows,
    }
