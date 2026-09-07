# Evaluation Framework

## Layer 1 — IRMF Performance Evaluation

Implemented in:

```text
diagnostics/layer1_irmf_performance.py
```

### Level A — Core local/operator behaviour

```text
mean_b0_like
trace_norm
operator_norm
contraction_ratio
```

### Level B — Multiscale evolution

```text
residual_energy_ratio
b0_evolution_mean
b0_evolution_final
operator_monotonicity_score
```

### Level C — Hessian diagnostics

```text
hessian_condition_proxy
hessian_positive_ratio
```

Composite score:

```text
irmf_performance_score
```

## Layer 2 — Robust Estimation

Implemented in:

```text
diagnostics/layer2_robust_estimation.py
```

```text
denoise_psnr
denoise_mse
denoise_corr
input_snr_db
output_snr_db
snr_gain_db
imf_recovery_rmse
imf_recovery_corr
noise_capture_corr
noise_capture_energy_ratio
```

Composite score:

```text
robust_estimation_score
```

## Layer 3 — Decomposition Quality Evaluation

Implemented in:

```text
diagnostics/layer3_decomposition_quality.py
```

```text
strict_io
spectral_leakage
frequency_overlap_mean_offdiag
frequency_overlap_max_offdiag
frequency_separation_score
residual_whiteness
imf_count
```

Composite score:

```text
decomposition_quality_score
```

## Layer 4 — Supporting Diagnostics

Implemented in:

```text
diagnostics/layer4_supporting_diagnostics.py
```

Layer 4 diagnostics are reported for interpretation and are not used as composite sorting scores.

```text
frequency_spacing_min_ratio
frequency_spacing_mean_ratio
frequency_spacing_penalty
ifs
energy_ratio
energy_concentration_penalty
residual_autocorrelation_score
```
