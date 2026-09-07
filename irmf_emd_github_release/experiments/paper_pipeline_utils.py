#!/usr/bin/python
# coding: UTF-8

"""Utilities shared by the paper-oriented experiment pipeline."""

from pathlib import Path
import csv
import json
import platform
import sys
from datetime import datetime, timezone

import numpy as np

from experiments.experiment_utils import aggregate_method_family_rows, aggregate_paired_rows, paper_json_safe


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(paper_json_safe(obj), f, indent=2)


def flatten_row(row):
    """Flatten nested IRMF/EMD summaries into one CSV row."""
    out = {}
    for key, value in row.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, (dict, list, tuple)):
                    out[f"{key}_{subkey}"] = json.dumps(paper_json_safe(subvalue))
                else:
                    out[f"{key}_{subkey}"] = subvalue
        else:
            out[key] = value
    return out


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = [flatten_row(row) for row in rows]
    if not flat:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in flat for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in flat:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_section_outputs(rows, output_dir, stem):
    output_dir = ensure_dir(output_dir)
    write_json(rows, output_dir / f"{stem}.json")
    write_csv(rows, output_dir / f"{stem}.csv")
    has_family_baselines = any(
        isinstance(row.get("EEMD"), dict) or isinstance(row.get("CEEMDAN"), dict)
        for row in rows
    )
    summary = aggregate_method_family_rows(rows) if has_family_baselines else aggregate_paired_rows(rows)
    write_json(summary, output_dir / f"{stem}_aggregate.json")
    return summary


def write_manifest(output_dir, extra=None):
    output_dir = ensure_dir(output_dir)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    if extra:
        manifest.update(extra)
    write_json(manifest, output_dir / "run_manifest.json")
    return manifest
