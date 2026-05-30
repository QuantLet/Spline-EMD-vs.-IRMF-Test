# Evaluation Framework

## Composite scores

```text
irmf_performance_score
robust_estimation_score
decomposition_quality_score
```

Raw metrics are kept separate from score components. For example, `denoise_psnr`
is always reported as a raw PSNR value in dB.

Layer 4 diagnostics are reported for interpretation and are not used as a composite sorting score.

## Layer 1 — IRMF Performance Evaluation

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

## Layer 2 — Robust Estimation

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

## Layer 3 — Decomposition Quality Evaluation

```text
strict_io
spectral_leakage
frequency_overlap_mean_offdiag
frequency_overlap_max_offdiag
frequency_separation_score
residual_whiteness
imf_count
```

## Layer 4 — Supporting Diagnostics

```text
frequency_spacing_min_ratio
frequency_spacing_mean_ratio
frequency_spacing_penalty
ifs
energy_ratio
energy_concentration_penalty
residual_autocorrelation_score
```
