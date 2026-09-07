# GitHub Release Manifest

This folder is a cleaned source-code export from the active manuscript
development working tree. Local absolute paths, result directories, caches, and
logs were excluded from the export.

## Included Source Directories

- `algorithm/`
- `core_algorithms/`
- `diagnostics/`
- `experimental/`
- `experiments/`
- `irmf_prior_theory/`
- `methodology/`
- `noise_bank/`
- `parameter_search/`
- `robustness/`
- `scripts/`
- `sensitivity_analysis/`
- `signal_bank/`
- `tests/`
- `visualization/`

## Included Top-Level Entrypoints

- `paper_pipeline.py`
- `main.py`
- `paper_assets.py`
- `project_config.py`
- `run_theoretical_diagnostics.py`
- `run_theoretical_vs_empirical.py`
- `legacy_full_pipeline.py`
- `backfill_emd_family_for_versions.py`
- `backfill_v5_2_posthoc.py`

## Excluded From This Export

- Full benchmark result directories: `IRMF_EMD_PAPER_RESULTS*/`
- Local data directory: `data/`
- Python bytecode/cache directories
- Local logs and PID files
- Historical zip archives
- macOS `.DS_Store` files

The export is intended to be small enough for GitHub source control. Full
experimental results should be stored separately as release assets or in a data
repository.
