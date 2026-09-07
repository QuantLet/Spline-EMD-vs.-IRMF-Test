#!/usr/bin/python
# coding: UTF-8

"""
Backfill fixed-parameter EMD-family comparisons into existing paper results.

This script is intended for V4D and V4E result folders that were produced
before Section 7b and Appendix D existed.  It does not rerun Sections 4-7.
Instead, it reads the already locked IRMF parameters from each version's
dashboard and writes:

    section_7b_emd_family_comparison/
    appendices/appendix_D_emd_family_baseline_sensitivity/

The comparison remains fixed-parameter: IRMF uses the Section 4 global
configuration for that version, and EMD/EEMD/CEEMDAN use pre-specified baseline
settings.  No method is tuned separately on individual test cases.
"""

from pathlib import Path
import argparse
import json

from project_config import GLOBAL_EMD_PARAMS
from experiments.experiment_emd_family_benchmark import (
    run_emd_family_baseline_sensitivity,
    run_fixed_emd_family_benchmark,
)
from experiments.paper_pipeline_utils import write_json


VERSION_ROOTS = {
    "V4D": "IRMF_EMD_PAPER_RESULTS_V4D_SMOOTHED_MEDIAN_THEORY_CONSTRAINED",
    "V4E": "IRMF_EMD_PAPER_RESULTS_V4E_SMOOTHED_MEDIAN_H_FIXED_THEORY_CONSTRAINED",
}


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_locked_params(result_root):
    result_root = Path(result_root)
    dashboard_path = result_root / "paper_pipeline_dashboard.json"
    if dashboard_path.exists():
        dashboard = _load_json(dashboard_path)
        params = dashboard.get("selected_irmf_params")
        emd_params = dashboard.get("emd_params", dict(GLOBAL_EMD_PARAMS))
        if params:
            return dict(params), dict(emd_params), dashboard

    protocol_path = (
        result_root
        / "section_4_global_parameter_selection"
        / "section_4_parameter_selection_protocol.json"
    )
    if protocol_path.exists():
        protocol = _load_json(protocol_path)
        params = protocol.get("selected_global_irmf_params")
        if params:
            return dict(params), dict(GLOBAL_EMD_PARAMS), {}

    raise FileNotFoundError(
        f"Could not find locked IRMF params in {result_root}. "
        "Expected paper_pipeline_dashboard.json or Section 4 protocol."
    )


def backfill_one_version(
        version_label,
        result_root,
        run_section_7b=True,
        run_appendix_d=True,
        mode="representative",
        force=False,
):
    result_root = Path(result_root)
    if not result_root.exists():
        raise FileNotFoundError(f"Result root does not exist: {result_root}")

    irmf_params, emd_params, dashboard = _load_locked_params(result_root)

    # V4D/V4E are raw-residual smoothed-median implementations.  If these keys
    # are absent, keep them absent from manuscript tables but pass explicit
    # defaults to the current runner so the V4F local-MAD path is not used.
    runner_irmf_params = dict(irmf_params)
    runner_irmf_params.setdefault("robust_scale_mode", "none")
    runner_irmf_params.setdefault("robust_scale_floor", 1e-3)

    outputs = {
        "version": version_label,
        "result_root": str(result_root),
        "locked_irmf_params": runner_irmf_params,
        "emd_params": emd_params,
        "section_7b_run": False,
        "appendix_D_run": False,
    }

    section_7b_root = result_root / "section_7b_emd_family_comparison"
    appendix_d_root = result_root / "appendices" / "appendix_D_emd_family_baseline_sensitivity"

    if run_section_7b:
        aggregate_path = section_7b_root / "section_7b_emd_family_comparison_aggregate.json"
        if aggregate_path.exists() and not force:
            outputs["section_7b_summary"] = _load_json(aggregate_path)
            outputs["section_7b_skipped_existing"] = True
        else:
            outputs["section_7b_summary"] = run_fixed_emd_family_benchmark(
                section_7b_root,
                irmf_params=runner_irmf_params,
                emd_params=emd_params,
                mode=mode,
            )
            outputs["section_7b_run"] = True

    if run_appendix_d:
        aggregate_path = appendix_d_root / "appendix_D_emd_family_sensitivity_aggregate.json"
        if aggregate_path.exists() and not force:
            outputs["appendix_D_summary"] = _load_json(aggregate_path)
            outputs["appendix_D_skipped_existing"] = True
        else:
            outputs["appendix_D_summary"] = run_emd_family_baseline_sensitivity(
                appendix_d_root,
                irmf_params=runner_irmf_params,
                emd_params=emd_params,
                mode=mode,
            )
            outputs["appendix_D_run"] = True

    dashboard_path = result_root / "paper_pipeline_dashboard.json"
    if dashboard_path.exists():
        dashboard = _load_json(dashboard_path)
        if "section_7b_summary" in outputs:
            dashboard["section_7b_emd_family_summary"] = outputs["section_7b_summary"]
        dashboard["appendix_D_emd_family_sensitivity_run"] = bool(run_appendix_d)
        if "appendix_D_summary" in outputs:
            dashboard["appendix_D_emd_family_sensitivity_summary"] = outputs["appendix_D_summary"]
        write_json(dashboard, dashboard_path)

    write_json(outputs, result_root / "emd_family_backfill_manifest.json")
    return outputs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backfill Section 7b and Appendix D into existing V4D/V4E result folders."
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        default=["V4D", "V4E"],
        choices=sorted(VERSION_ROOTS.keys()),
    )
    parser.add_argument("--mode", default="representative")
    parser.add_argument("--skip-section-7b", action="store_true")
    parser.add_argument("--skip-appendix-d", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute even if output files already exist.")
    return parser.parse_args()


def main():
    args = parse_args()
    here = Path(__file__).resolve().parent
    outputs = []
    for version in args.versions:
        outputs.append(backfill_one_version(
            version_label=version,
            result_root=here / VERSION_ROOTS[version],
            run_section_7b=not args.skip_section_7b,
            run_appendix_d=not args.skip_appendix_d,
            mode=args.mode,
            force=args.force,
        ))
    write_json(outputs, here / "emd_family_backfill_summary.json")
    return outputs


if __name__ == "__main__":
    main()
