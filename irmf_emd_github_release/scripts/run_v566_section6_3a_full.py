#!/usr/bin/env python3
"""Resume the V5.66/V5.67 Section 6.3A full signal-variant rerun."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.experiment_signal_variant_robustness import run_signal_variant_robustness


def main():
    output_root = (
        Path("IRMF_EMD_PAPER_RESULTS_V5_24_FULL_BENCHMARK")
        / "algorithm"
        / "09_signal_variant_robustness_target_snr"
    )
    print("V5.66/V5.67 Section 6.3A full rerun resume", flush=True)
    print(f"output_root={output_root}", flush=True)
    rows, variants = run_signal_variant_robustness(output_root)
    print("V5.66/V5.67 Section 6.3A full rerun complete", flush=True)
    print(f"case_rows={len(rows)}", flush=True)
    print(f"variant_family_summary_rows={len(variants)}", flush=True)


if __name__ == "__main__":
    main()
