#!/usr/bin/python
# coding: UTF-8
"""Collect validated outputs from the two scientific layers.

This stage never reruns scientific experiments. It validates the two-layer
contract, copies publication-ready figures/tables, and writes reproducibility
manifests and a final summary.
"""
from pathlib import Path
import hashlib
import json
import shutil

from experiments.paper_pipeline_utils import ensure_dir, write_json, write_manifest


def _read_json(path, required=True):
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required result is missing: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_files(source_root, destination, suffixes):
    source_root = Path(source_root)
    destination = ensure_dir(destination)
    records = []
    if not source_root.exists():
        return records
    for source in sorted(source_root.rglob("*")):
        if not source.is_file() or source.suffix.lower() not in suffixes:
            continue
        rel = source.relative_to(source_root)
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append({
            "source": str(source),
            "paper_asset": str(target),
            "sha256": _sha256(target),
            "size_bytes": target.stat().st_size,
        })
    return records


def run_paper_assets(output_root):
    root = Path(output_root)
    methodology_root = root / "methodology"
    algorithm_root = root / "algorithm"
    assets_root = ensure_dir(root / "paper_assets")

    methodology = _read_json(methodology_root / "methodology_dashboard.json")
    algorithm = _read_json(algorithm_root / "algorithm_dashboard.json")

    if methodology.get("emd_family_included") is not False:
        raise ValueError("Methodology layer contract violated: EMD-family results were included.")
    lock = algorithm.get("rho_lock", {})
    if lock.get("loss_name") != "gaussian_smoothed_median" or float(lock.get("H", -1)) != 1.0:
        raise ValueError(f"Algorithm layer contract violated: invalid rho lock {lock!r}")

    figure_records = []
    figure_records += _collect_files(methodology_root, assets_root / "figures" / "methodology", {".png", ".pdf", ".svg"})
    figure_records += _collect_files(algorithm_root, assets_root / "figures" / "algorithm", {".png", ".pdf", ".svg"})
    table_records = []
    table_records += _collect_files(methodology_root, assets_root / "tables" / "methodology", {".csv", ".tex"})
    table_records += _collect_files(algorithm_root, assets_root / "tables" / "algorithm", {".csv", ".tex"})

    final_summary = {
        "pipeline_version": "V5.6",
        "scientific_order": [
            "methodology: evaluate rho function",
            "lock gaussian_smoothed_median with H=1",
            "algorithm: compare IRMF with EMD-family",
            "paper_assets: validate and collect outputs",
        ],
        "methodology_contract": {
            "scope": methodology.get("scope"),
            "emd_family_included": methodology.get("emd_family_included"),
            "locked_loss_tuning": methodology.get("locked_loss_tuning"),
        },
        "algorithm_contract": {
            "scope": algorithm.get("scope"),
            "rho_function_fixed": algorithm.get("rho_function_fixed"),
            "rho_lock": lock,
            "selected_irmf_params": algorithm.get("selected_irmf_params"),
            "selected_emd_params": algorithm.get("selected_emd_params"),
            "selected_eemd_params": algorithm.get("selected_eemd_params"),
            "selected_ceemdan_params": algorithm.get("selected_ceemdan_params"),
        },
        "n_figure_assets": len(figure_records),
        "n_table_assets": len(table_records),
    }
    write_json(final_summary, assets_root / "final_result_summary.json")
    write_json(figure_records, assets_root / "manifests" / "figure_assets.json")
    write_json(table_records, assets_root / "manifests" / "table_assets.json")
    write_manifest(assets_root / "manifests", {
        "pipeline_version": "V5.6",
        "scope": "paper_assets",
        "methodology_dashboard_sha256": _sha256(methodology_root / "methodology_dashboard.json"),
        "algorithm_dashboard_sha256": _sha256(algorithm_root / "algorithm_dashboard.json"),
        "n_figure_assets": len(figure_records),
        "n_table_assets": len(table_records),
    })
    return final_summary
