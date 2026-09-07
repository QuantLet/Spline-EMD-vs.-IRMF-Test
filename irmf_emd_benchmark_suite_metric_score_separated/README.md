# IRMF / EMD Benchmark Suite

A modular benchmark framework for evaluating Intrinsic Robust Multiscale Filtering (IRMF) against EMD-family decomposition methods on synthetic noisy signals.

## Evaluation framework

```text
Layer 1 — IRMF Performance Evaluation
Layer 2 — Robust Estimation
Layer 3 — Decomposition Quality Evaluation
Layer 4 — Supporting Diagnostics
```

See `EVALUATION_FRAMEWORK.md` for the full metric structure.

## Run

```bash
python main.py
```

Outputs are written to:

```text
IRMF_EMD_BENCHMARK_SUITE_RESULTS/
```
