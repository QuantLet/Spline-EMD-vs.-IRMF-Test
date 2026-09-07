#!/usr/bin/python
# coding: UTF-8

"""V5.53 Section 6 scientific and claim qualification gate."""

import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json
from diagnostics.shared_physical_diagnostics import SYNTHETIC_RECONSTRUCTION_PROTOCOL_ID


V553_SECTION6_CLAIM_QUALIFICATION_VERSION = "V5.53_section6_claim_qualification"


def _read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(root, relpath):
    path = root / relpath
    return {
        "path": relpath,
        "exists": bool(path.exists()),
    }


def run_v553_section6_claim_qualification(output_root):
    output_root = ensure_dir(output_root)
    algorithm_root = output_root.parent if output_root.name.startswith("16") else output_root

    assembly_path = (
        algorithm_root / "16s_v550_section6_main_text_assembly"
        / "v550_section6_main_text_assembly_dashboard.json"
    )
    assembly = _read_json(assembly_path)
    v551 = _read_json(
        algorithm_root / "16t_v551_section6_2_contamination_design"
        / "v551_section6_2_contamination_design_dashboard.json"
    )
    v552 = _read_json(
        algorithm_root / "16u_v552_section6_4_computational_scaling"
        / "v552_section6_4_runtime_scaling_dashboard.json"
    )
    section61_protocol = _read_json(
        algorithm_root / "08_robustness_sensitivity_target_snr"
        / "6_3_parameter_sensitivity" / "section_6_3_protocol.json"
    )
    section61_status = _read_json(
        algorithm_root / "08_robustness_sensitivity_target_snr"
        / "6_3_parameter_sensitivity" / "section_6_3_parameter_sensitivity_status.json"
    )
    section61_primary_case_blocked = _read_json(
        algorithm_root / "08_robustness_sensitivity_target_snr"
        / "6_3_parameter_sensitivity"
        / "section_6_3_primary_endpoint_case_blocked_factorial.json"
    )

    checks = [
        {
            "check_id": "section6_assembly_exists",
            "passed": assembly is not None,
            "detail": str(assembly_path),
        },
        {
            "check_id": "section6_four_main_sections_declared",
            "passed": bool(assembly and int(assembly.get("main_text_section_count", 0)) == 4),
            "detail": "Expected four compact main-text sections.",
        },
        {
            "check_id": "section6_2_full_contamination_design_exists",
            "passed": bool(
                v551
                and v551.get("statistics_complete")
                and int(v551.get("n_failed", 1)) == 0
                and v551.get("normalized_spillover_formula_audit", {}).get("passed") is True
            ),
            "detail": (
                "Requires V5.51 full rate/magnitude/geometry/variance-matched "
                "execution and normalized spillover loss formula audit "
                "loss=-log(score)."
            ),
        },
        {
            "check_id": "section6_4_full_runtime_scaling_exists",
            "passed": bool(v552 and v552.get("statistics_complete") and int(v552.get("n_failed", 1)) == 0),
            "detail": "Requires V5.52 full runtime/scaling execution.",
        },
        {
            "check_id": "target_snr_artifacts_present",
            "passed": all(item["exists"] for item in [
                _artifact(algorithm_root, "08_robustness_sensitivity_target_snr"),
                _artifact(algorithm_root, "09_signal_variant_robustness_target_snr"),
                _artifact(algorithm_root, "11_emd_family_sensitivity_target_snr"),
            ]),
            "detail": "Section 6 main evidence must use target-SNR artifacts.",
        },
        {
            "check_id": "relative_c_H_protocol_present",
            "passed": bool(_artifact(
                algorithm_root,
                "16f_v534_section6_executable_protocols/v534_section6_executable_protocols.json",
            )["exists"]),
            "detail": "Protocol must describe c_H perturbation rather than absolute-H sensitivity.",
        },
        {
            "check_id": "section6_1_uses_frozen_synthetic_reconstruction_protocol",
            "passed": bool(
                section61_protocol
                and section61_status
                and section61_protocol.get("reconstruction_protocol_id")
                == SYNTHETIC_RECONSTRUCTION_PROTOCOL_ID
                and section61_status.get("reconstruction_protocol_id")
                == SYNTHETIC_RECONSTRUCTION_PROTOCOL_ID
                and int(section61_status.get("n_expected_rows", -1))
                == int(section61_status.get("n_completed_rows", -2))
            ),
            "detail": (
                "Section 6.1 parameter sensitivity must use the same frozen "
                "synthetic reconstruction protocol as the V5.30 benchmark, "
                f"{SYNTHETIC_RECONSTRUCTION_PROTOCOL_ID}, and complete all "
                "expected rows."
            ),
        },
        {
            "check_id": "section6_1_primary_endpoint_case_blocked_factorial_complete",
            "passed": bool(
                section61_primary_case_blocked
                and int(section61_primary_case_blocked.get("primary_endpoint_count", 0)) == 10
                and int(section61_primary_case_blocked.get("n_factorial_settings", 0)) == 81
                and int(section61_primary_case_blocked.get("n_case_blocks", 0)) == 48
            ),
            "detail": (
                "Section 6.1 must include case-blocked joint sensitivity analysis "
                "for the 10 non-contamination primary endpoints, with 81 "
                "factorial settings and 48 case blocks."
            ),
        },
    ]
    all_passed = all(bool(c["passed"]) for c in checks)
    dashboard = {
        "schema_version": V553_SECTION6_CLAIM_QUALIFICATION_VERSION,
        "created_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "qualification_status": "passed" if all_passed else "not_passed_pending_required_artifacts",
        "claim_authorized": bool(all_passed),
        "section": "6 Sensitivity and Robustness Analyses",
        "claim_boundary": (
            "Passing this gate authorizes Section 6 main-text claims only within "
            "the frozen four-subsection scope. It does not authorize new metrics, "
            "new sensitivity categories, or post-hoc retuning."
        ),
        "checks": checks,
        "required_next_steps_if_not_passed": [
            c["check_id"] for c in checks if not c["passed"]
        ],
    }
    write_json(dashboard, output_root / "v553_section6_claim_qualification_dashboard.json")
    write_csv(checks, output_root / "v553_section6_claim_qualification_checks.csv")
    return dashboard
