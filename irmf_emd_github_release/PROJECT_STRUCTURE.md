# Project Structure

The codebase is organized around two research layers and one paper-facing
mapping layer.

## 1. Methodology Layer

```text
methodology/
├── pipeline.py
└── loss_regime_map/
```

Purpose:

```text
rho-function properties
loss calibration
loss ablation
contamination tolerance
optimization stability
local quadratic diagnostics
```

This layer studies the rho/loss choice inside IRMF.  It does not compare
against EMD/EEMD/CEEMDAN.

## 2. Algorithm Evaluation Layer

```text
algorithm/
├── pipeline.py
└── paper_section_map.py
```

Purpose:

```text
fixed-parameter IRMF evaluation
four-method EMD-family benchmark
unified nested factorial benchmark cube
statistics and rankings
sensitivity analyses
challenging structural generalization
oracle/adaptivity-gap appendix
real-data proxy validation
```

The algorithm layer fixes the IRMF loss:

```text
loss_name = gaussian_smoothed_median
H = 1.0
```

## 3. Paper-Facing Output Map

Every result root can contain:

```text
paper_sections/
├── README.md
├── paper_section_manifest.json
├── section_01_introduction/
├── section_02_operationalized_irmf_procedure/
├── section_03_evaluation_methodology/
├── section_04_canonical_benchmark_performance/
├── section_05_signal_family_reproducibility/
├── section_06_robustness_protocol_sensitivity/
├── section_07_challenging_structural_generalization/
├── section_08_external_validation_real_data/
├── section_09_discussion/
├── section_10_conclusion/
└── appendices/
```

Generate or refresh it with:

```bash
python paper_pipeline.py paper-section-map --output-root <RESULT_ROOT>
```

## Core Source Directories

```text
core_algorithms/
    strict_spokoiny_irmf.py
    robust_losses.py
    emd_wrapper.py
    eemd_wrapper.py
    ceemdan_wrapper.py

signal_bank/
    synthetic_signals.py
    real_data_loader.py

noise_bank/
    noise_models.py

diagnostics/
    shared_physical_diagnostics.py
    real_proxy_diagnostics.py
    irmf_local_theory_diagnostics.py
    irmf_operator_diagnostics.py
    ...

experiments/
    experiment_global_parameter_selection.py
    experiment_unified_benchmark_cube.py
    experiment_unified_cube_statistics.py
    experiment_signal_variant_robustness.py
    experiment_robustness_sensitivity.py
    experiment_controlled_challenging_diagnostics.py
    experiment_oracle_adaptivity_analysis.py
    experiment_proxy_metric_validation.py
    experiment_real_world_validation.py
    ...
```

## Manuscript Alignment

The frozen manuscript structure is:

```text
1. Introduction
2. Statistical Model and Operationalized IRMF Procedure
3. Evaluation Methodology: Hierarchical Fixed-Parameter Evidence Design
4. Canonical Benchmark Performance
5. Signal-Family Reproducibility
6. Robustness and Protocol Sensitivity
7. Challenging Structural Generalization
8. External Validation on Real Data
9. Discussion
10. Conclusion
```

The section map is the authoritative bridge between code outputs and this
manuscript structure.

