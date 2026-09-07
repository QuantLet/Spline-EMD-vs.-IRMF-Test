# IRMF / EMD-Family Benchmark Suite

This repository contains the executable code for a paper-oriented benchmark of
a Spokoiny-inspired iterative robust multiscale filtering procedure (IRMF)
against EMD-family decomposition methods: EMD, EEMD, and CEEMDAN.

The main contribution of this codebase is the fixed-parameter evaluation
framework: target-SNR benchmark design, locked IRMF parameter provenance,
primary/secondary metric taxonomy, repeated-measures statistics, Section 6
robustness analyses, and a real-world validation scaffold.

## What Is Included

- `core_algorithms/`: IRMF and EMD-family wrappers.
- `signal_bank/`: synthetic signal generators, including canonical and
  challenging signal families.
- `noise_bank/`: synthetic noise and contamination models.
- `diagnostics/`: primary and secondary metric implementations.
- `experiments/`: benchmark, qualification, Section 6, Section 8, and audit
  modules.
- `algorithm/` and `methodology/`: orchestration helpers for the paper
  pipeline.
- `visualization/`: plotting utilities.
- `tests/`: lightweight regression tests.
- `scripts/`: convenience scripts for long-running stages.

Large result folders, local data, caches, logs, and generated artifacts are not
included in this GitHub-ready export.

## Current Protocol Snapshot

The current code reflects the later protocol family used in the manuscript
development:

- V5.30 target-SNR synthetic benchmark design.
- V5.45 truth-free scale proxy qualification for relative-H.
- V5.47/V5.49 relative-H development parameter selection workflow.
- V5.48 parameter-selection adjudication gate.
- V5.40/V5.41 secondary TF/IF and primary-to-secondary diagnostic closure.
- V5.51-V5.63 Section 6 robustness, reviewer-readiness, and boundary-condition
  synthesis.
- V5.66/V5.67 Section 6.3A two-axis canonical and challenging-family
  signal-specification robustness design and analysis freeze.

For IRMF, the robust-loss scale is parameterized as a case-adaptive relative
scale:

```text
H_case = c_H * scale_hat(Y_case)
```

where `scale_hat` is the frozen truth-free first-difference MAD scale proxy.
The globally selected parameter is `c_H`, not a case-specific absolute `H`.

## Installation

Create a fresh Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On machines where Matplotlib cannot write to the default user cache directory,
set a local cache directory before running the pipeline:

```bash
export MPLCONFIGDIR="$PWD/.mplconfig"
```

## Main Entry Point

Most workflows are dispatched through:

```bash
python paper_pipeline.py --help
```

Example smoke checks:

```bash
python paper_pipeline.py v545-relative-h-scale-estimator-qualification \
  --output-root IRMF_EMD_PAPER_RESULTS_V5_24_FULL_BENCHMARK

python paper_pipeline.py v552-section6-4-computational-scaling \
  --output-root IRMF_EMD_PAPER_RESULTS_V5_24_FULL_BENCHMARK
```

Example full target-SNR benchmark/statistics flow:

```bash
python paper_pipeline.py protocol-controlled-full-benchmark-rerun \
  --output-root IRMF_EMD_PAPER_RESULTS_V5_24_FULL_BENCHMARK \
  --timeout-seconds 120 \
  --use-v530-target-snr-grid

python paper_pipeline.py v528-unified-statistics \
  --output-root IRMF_EMD_PAPER_RESULTS_V5_24_FULL_BENCHMARK
```

Long-running stages are checkpointed where the corresponding protocol module
supports resumable execution.

## Section 6 Robustness Commands

```bash
python paper_pipeline.py sensitivity \
  --output-root IRMF_EMD_PAPER_RESULTS_V5_24_FULL_BENCHMARK

python paper_pipeline.py v551-section6-2-contamination-design \
  --output-root IRMF_EMD_PAPER_RESULTS_V5_24_FULL_BENCHMARK

python paper_pipeline.py v552-section6-4-computational-scaling \
  --output-root IRMF_EMD_PAPER_RESULTS_V5_24_FULL_BENCHMARK

python scripts/run_v566_section6_3a_full.py

python paper_pipeline.py v559-section6-post-execution-qualification \
  --output-root IRMF_EMD_PAPER_RESULTS_V5_24_FULL_BENCHMARK

python paper_pipeline.py v561-section6-5-matching-rule-robustness \
  --output-root IRMF_EMD_PAPER_RESULTS_V5_24_FULL_BENCHMARK

python paper_pipeline.py v562-section6-reviewer-readiness-synthesis \
  --output-root IRMF_EMD_PAPER_RESULTS_V5_24_FULL_BENCHMARK

python paper_pipeline.py v563-section6-failure-region-dominance-synthesis \
  --output-root IRMF_EMD_PAPER_RESULTS_V5_24_FULL_BENCHMARK
```

## Real-World Validation

The real-world validation code is scaffolded separately from the synthetic
benchmark. Held-out real-data performance claims should only be made after the
real-world preflight, split audit, development-record threshold locking, and
held-out execution gates pass.

```bash
python paper_pipeline.py real-world-validation-schema-draft \
  --output-root IRMF_EMD_PAPER_RESULTS_V5_24_FULL_BENCHMARK

python paper_pipeline.py real-world-v1-schema-smoke \
  --output-root IRMF_EMD_PAPER_RESULTS_V5_24_FULL_BENCHMARK
```

Commands that need external ECG data require the user to supply a local data
path through the relevant `--data-root` or `--real-data-dir` argument.

## Result Policy

This repository is intended as a code release. Generated benchmark outputs are
excluded by default because full runs can be large. If publishing result
artifacts, place them in a separate release archive or an external data
repository and document their checksums.

## License

No open-source license file is included in this export. Add a `LICENSE` file
before public release if you want others to reuse the code under explicit
terms.
