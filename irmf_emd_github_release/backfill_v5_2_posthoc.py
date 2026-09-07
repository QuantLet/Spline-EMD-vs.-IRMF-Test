#!/usr/bin/python
# coding: UTF-8

"""Backfill V5.2 post-hoc result checks from an existing V5.1 run."""

from pathlib import Path
import argparse
import json

from project_config import PAPER_OUTPUT_ROOT_NAME
from experiments.experiment_result_sensitivity_analysis import run_v5_2_posthoc_result_checks


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Backfill V5.2 post-hoc analyses from V5.1 outputs.")
    parser.add_argument("--output-root", default=PAPER_OUTPUT_ROOT_NAME)
    args = parser.parse_args()
    root = Path(args.output_root)
    family_path = root / "section_5b_main_emd_family_benchmark" / "section_7b_emd_family_comparison.json"
    variant_path = root / "section_7_signal_variant_robustness" / "section_7_signal_variant_robustness.json"
    if not family_path.exists():
        raise FileNotFoundError(f"Missing full EMD-family benchmark rows: {family_path}")
    family_rows = _load_json(family_path)
    variant_rows = _load_json(variant_path) if variant_path.exists() else []
    summary = run_v5_2_posthoc_result_checks(
        root / "section_6b_v5_2_result_sensitivity_checks",
        family_rows=family_rows,
        variant_rows=variant_rows,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

