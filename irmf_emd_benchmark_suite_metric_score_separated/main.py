#!/usr/bin/python
# coding: UTF-8

"""Run the IRMF / EMD benchmark suite."""

from pathlib import Path

from experiments.experiment_irmf_vs_emd import run_irmf_vs_emd_multisigma_experiment
from experiments.experiment_signal_bank import run_signal_bank_multisigma_experiment
from experiments.experiment_boundary_effects import run_boundary_effects_multisigma_experiment
from experiments.experiment_apriori_vs_posteriori import run_apriori_vs_posteriori_experiment
from experiments.experiment_operator_propagation import run_operator_propagation_signalbank_experiment
from experiments.experiment_local_smoothness_adaptation import run_local_smoothness_adaptation_multisigma_experiment
from experiments.experiment_emd_family_benchmark import run_emd_family_multisigma_benchmark
from experiments.experiment_monte_carlo_robustness import run_monte_carlo_robustness_grid_experiment
from experiments.experiment_full_signal_noise_robustness import run_full_signal_noise_robustness_experiment


OUTPUT_ROOT = Path("IRMF_EMD_BENCHMARK_SUITE_RESULTS")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    search_mode = "quick"
    sigma_levels = (0.05, 0.20)

    # ------------------------------------------------------------------
    # 1. Basic IRMF vs EMD comparison
    # ------------------------------------------------------------------
    run_irmf_vs_emd_multisigma_experiment(
        output_root=OUTPUT_ROOT,
        signal_name="stationary_multi_sine",
        sigma_levels=sigma_levels,
        search_mode=search_mode
    )

    # ------------------------------------------------------------------
    # 2. Signal-bank benchmark
    # ------------------------------------------------------------------
    run_signal_bank_multisigma_experiment(
        output_root=OUTPUT_ROOT,
        sigma_levels=sigma_levels,
        search_mode=search_mode
    )

    # ------------------------------------------------------------------
    # 3. Boundary effects
    # ------------------------------------------------------------------
    run_boundary_effects_multisigma_experiment(
        output_root=OUTPUT_ROOT,
        sigma_levels=sigma_levels,
        search_mode=search_mode
    )

    # ------------------------------------------------------------------
    # 4. A-priori vs posteriori bandwidth/theory comparison
    # ------------------------------------------------------------------
    run_apriori_vs_posteriori_experiment(
        output_root=OUTPUT_ROOT,
        signal_name="stationary_multi_sine",
        sigma=0.05,
        search_mode=search_mode
    )

    # ------------------------------------------------------------------
    # 5. Operator propagation curves
    # ------------------------------------------------------------------
    run_operator_propagation_signalbank_experiment(
        output_root=OUTPUT_ROOT,
        signal_names=("stationary_multi_sine", "chirp", "am_fm", "frequency_jump", "impulsive_transient"),
        sigma_levels=sigma_levels,
        search_mode=search_mode
    )

    # ------------------------------------------------------------------
    # 6. Local smoothness adaptation
    # ------------------------------------------------------------------
    run_local_smoothness_adaptation_multisigma_experiment(
        output_root=OUTPUT_ROOT,
        sigma_levels=sigma_levels,
        search_mode=search_mode
    )

    # ------------------------------------------------------------------
    # 7. EMD-family benchmark: EMD / optional EEMD / optional CEEMDAN
    # ------------------------------------------------------------------
    run_emd_family_multisigma_benchmark(
        output_root=OUTPUT_ROOT,
        signal_name="stationary_multi_sine",
        sigma_levels=sigma_levels,
        search_mode=search_mode
    )

    # ------------------------------------------------------------------
    # 8. Monte-Carlo robustness grid
    #    5 signals × 5 noises × 2 sigmas, with repeated trials.
    #    n_trials can be increased for final paper runs.
    # ------------------------------------------------------------------
    run_monte_carlo_robustness_grid_experiment(
        output_root=OUTPUT_ROOT,
        signal_names=("stationary_multi_sine", "chirp", "am_fm", "frequency_jump", "impulsive_transient"),
        noise_names=("gaussian", "student_t", "laplace", "impulsive", "burst"),
        sigma_levels=sigma_levels,
        n_trials=20
    )

    # ------------------------------------------------------------------
    # 9. Full deterministic robustness grid
    #    5 signals × 5 noises × 2 sigmas = 40 cases.
    # ------------------------------------------------------------------
    run_full_signal_noise_robustness_experiment(
        output_root=OUTPUT_ROOT,
        signal_names=("stationary_multi_sine", "chirp", "am_fm", "frequency_jump", "impulsive_transient"),
        noise_names=("gaussian", "student_t", "laplace", "impulsive", "burst"),
        sigma_levels=sigma_levels,
        search_mode=search_mode
    )

    print("\n" + "=" * 120)
    print("ALL BENCHMARK SUITE EXPERIMENTS FINISHED")
    print("=" * 120)


if __name__ == "__main__":
    main()
