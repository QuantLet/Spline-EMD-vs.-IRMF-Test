#!/usr/bin/python
# coding: UTF-8

"""V5.60 supplementary component-correspondence sensitivity protocol.

This stage freezes an Appendix/protocol audit for the primary estimated-vs-truth
component matching rule.  It does not alter the 13 primary endpoint definitions,
does not rerun algorithms, and does not recompute primary statistics.
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


V560_MATCHING_SENSITIVITY_PROTOCOL_VERSION = (
    "V5.60_primary_component_correspondence_sensitivity_protocol"
)

THRESHOLDED_HUNGARIAN_TAU_GRID = (0.50, 0.70, 0.80)


def _read_csv_header(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return []


def _artifact_inventory(algorithm_root):
    algorithm_root = Path(algorithm_root)
    candidate_artifacts = [
        {
            "artifact_role": "protocol_controlled_main_benchmark_compact_rows",
            "relative_path": (
                "06_protocol_controlled_full_benchmark_rerun/"
                "unified_benchmark_cube_rows.csv"
            ),
        },
        {
            "artifact_role": "section_6_1_parameter_sensitivity_rows",
            "relative_path": (
                "08_robustness_sensitivity_target_snr/6_3_parameter_sensitivity/"
                "section_6_3_parameter_sensitivity.csv"
            ),
        },
        {
            "artifact_role": "section_6_3_signal_variant_rows",
            "relative_path": (
                "09_signal_variant_robustness_target_snr/"
                "section_5_signal_family_reproducibility.csv"
            ),
        },
    ]

    inventory = []
    for item in candidate_artifacts:
        path = algorithm_root / item["relative_path"]
        header = _read_csv_header(path)
        lower_header = [h.lower() for h in header]
        inventory.append({
            **item,
            "exists": bool(path.exists()),
            "n_columns": int(len(header)),
            "has_primary_hungarian_assignment_pairs": any(
                "imf_recovery_assignment_pairs" in h for h in lower_header
            ),
            "has_primary_matched_corr_summary": any(
                "imf_recovery_corr" in h for h in lower_header
            ),
            "has_primary_matched_nrmse_summary": any(
                "imf_recovery_nrmse" in h for h in lower_header
            ),
            "has_full_estimated_true_abs_corr_matrix": any(
                "estimated_true_abs_corr_matrix" in h
                or "imf_true_abs_corr_matrix" in h
                for h in lower_header
            ),
            "has_full_estimated_true_signed_corr_matrix": any(
                "estimated_true_signed_corr_matrix" in h
                or "imf_true_signed_corr_matrix" in h
                for h in lower_header
            ),
            "has_true_component_energy_vector": any(
                "true_component_energy" in h
                or "true_component_energies" in h
                for h in lower_header
            ),
            "has_estimated_component_energy_vector": any(
                "estimated_component_energy" in h
                or "estimated_component_energies" in h
                or "imf_component_energy" in h
                for h in lower_header
            ),
            "has_split_merge_secondary_diagnostics": any(
                "component_splitting_index" in h
                or "component_merging_index" in h
                or "association_strength" in h
                for h in lower_header
            ),
        })
    return inventory


def _feasibility_summary(inventory):
    any_assignment = any(
        row["has_primary_hungarian_assignment_pairs"] for row in inventory
    )
    any_full_abs = any(
        row["has_full_estimated_true_abs_corr_matrix"] for row in inventory
    )
    any_full_signed = any(
        row["has_full_estimated_true_signed_corr_matrix"] for row in inventory
    )
    any_true_energy = any(
        row["has_true_component_energy_vector"] for row in inventory
    )
    any_est_energy = any(
        row["has_estimated_component_energy_vector"] for row in inventory
    )
    any_split_merge = any(
        row["has_split_merge_secondary_diagnostics"] for row in inventory
    )

    exact_thresholded = bool(
        any_full_abs and any_true_energy and any_est_energy
    )
    exact_mnn = bool(any_full_abs)
    exact_energy_unmatched = bool(
        any_full_abs and any_true_energy and any_est_energy
    )
    partial_split_merge = bool(any_split_merge)

    return {
        "primary_unthresholded_hungarian_summaries_present": any_assignment,
        "exact_thresholded_hungarian_from_current_compact_rows": exact_thresholded,
        "exact_mutual_nearest_neighbor_from_current_compact_rows": exact_mnn,
        "exact_unmatched_energy_recalculation_from_current_compact_rows": exact_energy_unmatched,
        "split_merge_context_available_as_existing_secondary_diagnostics": partial_split_merge,
        "requires_future_full_pairwise_correlation_artifact": not (
            exact_thresholded and exact_mnn
        ),
        "reason_if_not_exact": (
            "Current compact rows preserve selected Hungarian pairs and "
            "aggregate matched corr/NRMSE, but not the full estimated-by-true "
            "correlation matrix plus component-energy vectors needed to run "
            "thresholded Hungarian or mutual-nearest-neighbor correspondence "
            "exactly."
            if not (exact_thresholded and exact_mnn)
            else ""
        ),
    }


def _method_rows():
    return [
        {
            "audit_method": "thresholded_hungarian_estimated_vs_truth",
            "role": "supplementary_matching_sensitivity",
            "matching_target": "estimated_components_vs_true_components",
            "rule": (
                "Build S_ij = |corr(est_i, true_j)|; perform globally optimal "
                "one-to-one assignment with invalid edges below tau left "
                "unmatched via dummy nodes or an equivalent thresholded "
                "assignment formulation."
            ),
            "tau_grid": ";".join(str(v) for v in THRESHOLDED_HUNGARIAN_TAU_GRID),
            "primary_outputs": (
                "valid_match_fraction; thresholded_matched_corr; "
                "thresholded_matched_nrmse; weak_forced_match_fraction; "
                "matching_rule_unmatched_true_energy_ratio; "
                "matching_rule_unmatched_estimated_energy_ratio"
            ),
            "claim_boundary": (
                "Checks whether component-level conclusions depend on forced "
                "low-similarity Hungarian assignments; does not replace primary "
                "unthresholded Hungarian endpoints. The matching_rule_unmatched_* "
                "fields are not the primary association-based missing/spurious "
                "energy endpoints."
            ),
        },
        {
            "audit_method": "mutual_nearest_neighbor_estimated_vs_truth",
            "role": "supplementary_matching_sensitivity",
            "matching_target": "estimated_components_vs_true_components",
            "rule": (
                "Accept est_i <-> true_j only if true_j is est_i's highest "
                "absolute-correlation target and est_i is true_j's highest "
                "absolute-correlation estimate."
            ),
            "tau_grid": "optional_same_tau_grid_for_validity_gate",
            "primary_outputs": (
                "mnn_match_fraction; mnn_matched_corr; mnn_matched_nrmse; "
                "discarded_hungarian_pair_fraction"
            ),
            "claim_boundary": (
                "Conservative correspondence check; expected to leave more "
                "components unmatched and used only to assess dependence on "
                "the global assignment rule."
            ),
        },
        {
            "audit_method": "many_to_many_split_merge_context",
            "role": "secondary_allocation_context",
            "matching_target": "association_structure_between_estimated_and_true_sets",
            "rule": (
                "Use pre-existing association/splitting/merging diagnostics or "
                "a future explicit allocation matrix to characterize one-to-many "
                "and many-to-one structure."
            ),
            "tau_grid": "not_a_primary_thresholded_assignment_rule",
            "primary_outputs": (
                "true_component_splitting_max; estimated_component_merging_max; "
                "component_splitting_index; component_merging_index; "
                "association_strength_distributions"
            ),
            "claim_boundary": (
                "Explains split/merge allocation mechanisms; not a replacement "
                "for the one-to-one primary correspondence rule."
            ),
        },
    ]


def _required_future_fields():
    return [
        {
            "field": "estimated_true_abs_corr_matrix",
            "purpose": (
                "Exact thresholded Hungarian and mutual-nearest-neighbor matching."
            ),
        },
        {
            "field": "estimated_true_signed_corr_matrix",
            "purpose": "Sign-aware secondary interpretation of accepted matches.",
        },
        {
            "field": "true_component_energy_vector",
            "purpose": "Thresholded unmatched true-component energy ratios.",
        },
        {
            "field": "estimated_component_energy_vector",
            "purpose": "Thresholded unmatched estimated-component energy ratios.",
        },
        {
            "field": "unthresholded_hungarian_pairs",
            "purpose": (
                "Direct comparison against the frozen primary correspondence."
            ),
        },
        {
            "field": "component_index_and_case_provenance",
            "purpose": (
                "Prevent mixing rows across methods, seeds, SNR levels, and "
                "signal/noise cases."
            ),
        },
    ]


def run_v560_matching_sensitivity_protocol(output_root, algorithm_root=None):
    output_root = ensure_dir(output_root)
    output_root = Path(output_root)
    algorithm_root = Path(algorithm_root) if algorithm_root is not None else output_root.parent

    inventory = _artifact_inventory(algorithm_root)
    feasibility = _feasibility_summary(inventory)
    method_rows = _method_rows()
    required_fields = _required_future_fields()

    protocol = {
        "schema_version": V560_MATCHING_SENSITIVITY_PROTOCOL_VERSION,
        "created_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "module_status": "protocol_frozen_execution_pending",
        "paper_location": "Supplementary / Appendix protocol credibility audit",
        "scientific_question": (
            "Are component-level conclusions robust to the correspondence rule, "
            "or are they driven by forced low-similarity Hungarian matches?"
        ),
        "primary_schema_policy": {
            "primary_13_metrics_changed": False,
            "primary_matching_changed": False,
            "primary_correspondence_rule": (
                "Unthresholded global Hungarian matching between estimated "
                "components and true components with cost C_ij = 1 - |corr|."
            ),
            "reason_primary_not_changed": (
                "Thresholding changes the matched-component estimand and can "
                "move low-quality matches into missing/spurious penalties. "
                "Therefore it is treated as sensitivity/qualification rather "
                "than as a replacement for frozen primary metrics."
            ),
        },
        "supplementary_correspondence_methods": method_rows,
        "threshold_grid": list(THRESHOLDED_HUNGARIAN_TAU_GRID),
        "current_compact_artifact_feasibility": feasibility,
        "artifact_inventory": inventory,
        "required_future_artifact_fields": required_fields,
        "execution_policy": {
            "no_algorithm_runs_performed_by_this_stage": True,
            "no_primary_statistic_recomputation_performed": True,
            "exact_thresholded_reanalysis_requires_full_pairwise_matrix": True,
            "not_ranking_bearing": True,
            "not_retuning": True,
        },
        "claim_boundary": (
            "V5.60 authorizes a supplementary correspondence-sensitivity "
            "protocol only. It cannot authorize a scientific conclusion until "
            "future artifacts persist the full estimated-by-true correlation "
            "matrices and component-energy vectors required for exact execution."
        ),
        "recommended_paper_language": (
            "Primary component recovery was evaluated using globally optimal "
            "one-to-one Hungarian correspondence. Supplementary "
            "matching-sensitivity analyses were pre-specified to evaluate "
            "whether conclusions depend on forced low-similarity assignments; "
            "these analyses are not primary and do not alter the frozen "
            "endpoint definitions."
        ),
    }

    dashboard = {
        "schema_version": V560_MATCHING_SENSITIVITY_PROTOCOL_VERSION,
        "created_timestamp_utc": protocol["created_timestamp_utc"],
        "module_status": protocol["module_status"],
        "primary_schema_changed": False,
        "primary_matching_changed": False,
        "no_algorithm_runs_performed": True,
        "claim_authorized": False,
        "exact_recomputation_from_current_compact_rows": bool(
            feasibility["exact_thresholded_hungarian_from_current_compact_rows"]
            and feasibility["exact_mutual_nearest_neighbor_from_current_compact_rows"]
        ),
        "requires_full_pairwise_correlation_artifact": bool(
            feasibility["requires_future_full_pairwise_correlation_artifact"]
        ),
        "threshold_grid": list(THRESHOLDED_HUNGARIAN_TAU_GRID),
        "output_files": {
            "protocol": "v560_matching_sensitivity_protocol.json",
            "artifact_inventory": "v560_matching_sensitivity_artifact_inventory.csv",
            "method_definitions": "v560_matching_sensitivity_method_definitions.csv",
            "required_future_fields": "v560_matching_sensitivity_required_future_fields.csv",
            "dashboard": "v560_matching_sensitivity_dashboard.json",
        },
    }

    write_json(protocol, output_root / "v560_matching_sensitivity_protocol.json")
    write_json(dashboard, output_root / "v560_matching_sensitivity_dashboard.json")
    write_csv(inventory, output_root / "v560_matching_sensitivity_artifact_inventory.csv")
    write_csv(method_rows, output_root / "v560_matching_sensitivity_method_definitions.csv")
    write_csv(required_fields, output_root / "v560_matching_sensitivity_required_future_fields.csv")

    return {
        "protocol": protocol,
        "dashboard": dashboard,
        "artifact_inventory": inventory,
        "method_definitions": method_rows,
        "required_future_fields": required_fields,
    }
