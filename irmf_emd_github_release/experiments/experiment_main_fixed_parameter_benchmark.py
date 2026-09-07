#!/usr/bin/python
# coding: UTF-8

"""Classical EMD reference contrast.

The paper-facing main benchmark is the four-method EMD-family benchmark.  This
module is retained as a focused IRMF-vs-classical-EMD contrast for backward
compatibility and historical comparison.
"""

from pathlib import Path

from project_config import (
    COMPREHENSIVE_NOISE_FAMILY,
    COMPREHENSIVE_SIGMA_LEVELS,
    COMPREHENSIVE_SIGNAL_FAMILY,
    DEFAULT_FS,
    DEFAULT_N,
    DEFAULT_SEED,
    GLOBAL_EMD_PARAMS,
    GLOBAL_IRMF_PARAMS,
)
from experiments.experiment_utils import run_fixed_pair_case
from experiments.paper_pipeline_utils import ensure_dir, write_json, write_section_outputs


def run_main_fixed_parameter_benchmark(
        output_root,
        irmf_params=GLOBAL_IRMF_PARAMS,
        emd_params=GLOBAL_EMD_PARAMS,
        signals=COMPREHENSIVE_SIGNAL_FAMILY,
        noises=COMPREHENSIVE_NOISE_FAMILY,
        sigmas=COMPREHENSIVE_SIGMA_LEVELS,
        n=DEFAULT_N,
        fs=DEFAULT_FS,
        seed=DEFAULT_SEED,
):
    output_root = ensure_dir(output_root)
    rows = []
    for signal_name in signals:
        for noise_name in noises:
            for sigma in sigmas:
                _, irmf, emd, row = run_fixed_pair_case(
                    signal_name=signal_name,
                    noise_name=noise_name,
                    sigma=sigma,
                    irmf_params=irmf_params,
                    emd_params=emd_params,
                    n=n,
                    fs=fs,
                    seed=seed,
                    run_id_prefix=f"{signal_name}_{noise_name}_{sigma}",
                )
                rows.append(row)

    protocol = {
        "section": "5.x Classical EMD Reference Contrast",
        "design": "7 signal classes x 8 noise models x 3 sigma levels = 168 cases",
        "paper_role": (
            "supplementary classical-reference contrast; the main Section 5 "
            "benchmark is the four-method IRMF/EMD/EEMD/CEEMDAN comparison"
        ),
        "signals": list(signals),
        "noises": list(noises),
        "sigmas": list(sigmas),
        "n": n,
        "fs": fs,
        "seed": seed,
        "irmf_params": dict(irmf_params),
        "emd_params": dict(emd_params),
        "important_note": "No per-case parameter tuning is used; do not interpret this two-method contrast as the sole main benchmark.",
    }
    write_json(protocol, output_root / "classical_emd_reference_contrast_protocol.json")
    aggregate = write_section_outputs(rows, output_root, "classical_emd_reference_contrast")
    return rows, aggregate


if __name__ == "__main__":
    run_main_fixed_parameter_benchmark(Path("IRMF_EMD_PAPER_RESULTS") / "section_5_main_fixed_parameter_benchmark")
