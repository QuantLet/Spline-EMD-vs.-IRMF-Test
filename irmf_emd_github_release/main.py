#!/usr/bin/python
# coding: UTF-8

"""Run the IRMF / EMD benchmark suite."""

from pathlib import Path

from project_config import (
    OUTPUT_ROOT_NAME,
    DEFAULT_N,
    DEFAULT_FS,
    DEFAULT_SEED,
    DEFAULT_SEARCH_MODE,
    DEFAULT_MONTE_CARLO_TRIALS,
)
from experiments.experiment_operator_propagation import run_operator_propagation_experiment
from experiments.experiment_comprehensive_signal_noise_benchmark import run_comprehensive_signal_noise_benchmark
from experiments.experiment_monte_carlo_robustness import run_monte_carlo_robustness_experiment
from experiments.experiment_emd_family_benchmark import run_emd_family_benchmark
from experiments.experiment_boundary_effects import run_boundary_effects_experiment


def main():
    output_root = Path(OUTPUT_ROOT_NAME)
    output_root.mkdir(parents=True, exist_ok=True)

    run_operator_propagation_experiment(
        output_root=output_root / "4_1_operator_propagation",
        n=DEFAULT_N,
        fs=DEFAULT_FS,
        seed=DEFAULT_SEED,
    )

    run_comprehensive_signal_noise_benchmark(
        output_root=output_root / "4_2_comprehensive_signal_noise_benchmark",
        search_mode=DEFAULT_SEARCH_MODE,
        n=DEFAULT_N,
        fs=DEFAULT_FS,
        seed=DEFAULT_SEED,
        run_huber_lambda_sweep=False,
    )

    run_monte_carlo_robustness_experiment(
        output_root=output_root / "4_3_monte_carlo_robustness",
        n_trials=DEFAULT_MONTE_CARLO_TRIALS,
        search_mode=DEFAULT_SEARCH_MODE,
        n=DEFAULT_N,
        fs=DEFAULT_FS,
        base_seed=1000,
        run_huber_breakdown=True,
    )

    run_emd_family_benchmark(
        output_root=output_root / "4_4_emd_family_benchmark",
        mode="representative",
        search_mode=DEFAULT_SEARCH_MODE,
        n=DEFAULT_N,
        fs=DEFAULT_FS,
        seed=DEFAULT_SEED,
    )

    run_boundary_effects_experiment(
        output_root=output_root / "4_5_boundary_effects",
        mode="representative",
        search_mode=DEFAULT_SEARCH_MODE,
        n=DEFAULT_N,
        fs=DEFAULT_FS,
        seed=DEFAULT_SEED,
    )


if __name__ == "__main__":
    main()
