#!/usr/bin/python
# coding: UTF-8

"""Section 8: Real Data Illustration."""

from pathlib import Path

import numpy as np

from project_config import DEFAULT_FS, GLOBAL_EMD_PARAMS, GLOBAL_IRMF_PARAMS
from experiments.experiment_utils import (
    paired_method_summary,
    run_fixed_emd_case,
    run_fixed_irmf_case,
)
from experiments.paper_pipeline_utils import ensure_dir, write_json, write_section_outputs
from signal_bank.real_data_loader import load_real_signal_csv


def run_one_real_data_case(path, irmf_params, emd_params, fs=None):
    T, Y, inferred_fs = load_real_signal_csv(path, fs=fs)
    if len(T) > 1:
        t_unit = (T - T[0]) / (T[-1] - T[0])
    else:
        t_unit = np.asarray([0.0])
    # Real data have no clean ground truth, so reconstruction metrics that
    # require X_clean will be NaN.  Structural and residual diagnostics remain
    # useful for illustration.
    irmf = run_fixed_irmf_case(
        Y=Y,
        X_clean=None,
        t=t_unit,
        fs=inferred_fs,
        irmf_params=irmf_params,
        expected_noise_ratio=None,
        true_components=None,
        run_id=f"real_irmf_{Path(path).stem}",
    )
    emd = run_fixed_emd_case(
        Y=Y,
        X_clean=None,
        t=t_unit,
        fs=inferred_fs,
        emd_params=emd_params,
        true_components=None,
        run_id=f"real_emd_{Path(path).stem}",
    )
    row = paired_method_summary(irmf, emd, {
        "dataset": Path(path).stem,
        "path": str(path),
        "fs": inferred_fs,
        "n": int(len(Y)),
    })
    return row


def run_real_data_illustration(
        output_root,
        data_dir=None,
        irmf_params=GLOBAL_IRMF_PARAMS,
        emd_params=GLOBAL_EMD_PARAMS,
        fs=DEFAULT_FS,
):
    output_root = ensure_dir(output_root)
    data_dir = Path(data_dir) if data_dir is not None else None
    rows = []
    if data_dir is not None and data_dir.exists():
        for path in sorted(data_dir.glob("*.csv")):
            rows.append(run_one_real_data_case(path, irmf_params, emd_params, fs=fs))

    if not rows:
        template = (
            "Real-data CSV format:\n"
            "  either one column: y values, with fs supplied to run_real_data_illustration;\n"
            "  or two columns: time,y.\n"
            "Place CSV files in data/real/ and rerun this section.\n"
        )
        (output_root / "README_real_data_template.txt").write_text(template, encoding="utf-8")

    write_json({
        "section": "8 Real Data Illustration",
        "data_dir": str(data_dir) if data_dir is not None else None,
        "note": "Real-data metrics requiring clean ground truth are reported as NaN.",
    }, output_root / "section_8_protocol.json")
    aggregate = write_section_outputs(rows, output_root, "section_8_real_data_illustration")
    return rows, aggregate


if __name__ == "__main__":
    run_real_data_illustration(Path("IRMF_EMD_PAPER_RESULTS") / "section_8_real_data_illustration")
