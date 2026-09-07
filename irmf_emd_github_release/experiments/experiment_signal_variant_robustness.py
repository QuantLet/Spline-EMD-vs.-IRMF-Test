#!/usr/bin/python
# coding: UTF-8

"""Section 5: Signal-Family Reproducibility Study.

This study is deliberately kept separate from the primary 600-cell unified
benchmark cube.  It tests whether conclusions reproduce across alternative
formula/parameter instances within the same canonical signal family.
"""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from project_config import (
    DEFAULT_FS,
    DEFAULT_METHOD_TIMEOUT_SECONDS,
    DEFAULT_N,
    EVALUATION_METHODS,
    GLOBAL_CEEMDAN_PARAMS,
    GLOBAL_EEMD_PARAMS,
    GLOBAL_EMD_PARAMS,
    GLOBAL_IRMF_PARAMS,
    LOWER_IS_BETTER_METRICS,
    SIGNAL_VARIANT_FAMILY,
    SIGNAL_VARIANT_NOISES,
    SIGNAL_VARIANT_SEEDS,
    SIGNAL_VARIANT_SIGMAS,
    SIGNAL_VARIANT_TARGET_SNR_DB_LEVELS,
)
from experiments.experiment_utils import run_fixed_method_family_case
from experiments.experiment_unified_benchmark_cube import (
    MONTE_CARLO_SUMMARY_METRICS,
    _algorithm_seed_for,
    _finite,
    method_win_tie_loss_rows,
    monte_carlo_cell_summary_rows,
    paired_method_difference_summary_rows,
    seed_stability_summary_rows,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json, write_section_outputs


METHODS = tuple(EVALUATION_METHODS)
LOWER_IS_BETTER = set(LOWER_IS_BETTER_METRICS)


def _variant_family(signal_name):
    if signal_name.startswith("stationary_multi_sine"):
        return "stationary_multi_sine"
    if signal_name.startswith("chirp"):
        return "chirp"
    if signal_name.startswith("am_fm"):
        return "am_fm"
    if signal_name.startswith("frequency_jump"):
        return "frequency_jump"
    if signal_name.startswith("impulsive_transient"):
        return "impulsive_transient"
    if signal_name.startswith("intermittent"):
        return "intermittent_oscillation"
    if signal_name.startswith("close_freq"):
        return "close_frequencies"
    if signal_name.startswith("crossing_chirps"):
        return "crossing_chirps"
    if signal_name.startswith("time_varying_close_frequencies"):
        return "time_varying_close_frequencies"
    if signal_name.startswith("piecewise_am_fm_discontinuity"):
        return "piecewise_am_fm_discontinuity"
    if signal_name.startswith("damped_oscillation"):
        return "damped_oscillation"
    if signal_name.startswith("trend_plus_oscillation"):
        return "trend_plus_oscillation"
    if signal_name.startswith("buried_weak_component"):
        return "buried_weak_component"
    if signal_name.startswith("non_sinusoidal_periodic"):
        return "non_sinusoidal_periodic"
    if signal_name.startswith("transient_train"):
        return "transient_train"
    return signal_name


def _checkpoint_key(signal_name, noise_name, sigma, data_seed):
    return f"{signal_name}|{noise_name}|{float(sigma):.12g}|{int(data_seed)}"


def _target_snr_checkpoint_key(signal_name, noise_name, target_snr_db, data_seed):
    return f"{signal_name}|{noise_name}|snr{float(target_snr_db):.12g}|{int(data_seed)}"


def _sigma_label_for_snr(target_snr_db):
    return float(10.0 ** (-float(target_snr_db) / 20.0))


def _load_checkpoint_rows(path):
    rows = []
    path = Path(path)
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except Exception:
                continue
    return rows


def _append_checkpoint_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _summarize_family_reproducibility(rows, metrics=MONTE_CARLO_SUMMARY_METRICS):
    """Aggregate variant Monte Carlo summaries by canonical family.

    This summary is descriptive.  It is not a replacement for the primary
    600-cell benchmark ranking.
    """
    mc_rows = monte_carlo_cell_summary_rows(rows, metrics=metrics)
    grouped = defaultdict(list)
    for row in mc_rows:
        family = _variant_family(row.get("signal", ""))
        grouped[(family, row["method"], row["metric"])].append(row)
    out = []
    for (family, method, metric), vals in sorted(grouped.items(), key=lambda item: str(item[0])):
        means = np.asarray([v["mc_mean"] for v in vals if _finite(v.get("mc_mean")) is not None], dtype=float)
        sds = np.asarray([v["mc_sd"] for v in vals if _finite(v.get("mc_sd")) is not None], dtype=float)
        if means.size == 0:
            continue
        lower = metric in LOWER_IS_BETTER
        out.append({
            "variant_family": family,
            "method": method,
            "metric": metric,
            "higher_is_better": not lower,
            "n_variant_noise_sigma_cells": int(len(vals)),
            "median_cell_mean": float(np.median(means)),
            "mean_cell_mean": float(np.mean(means)),
            "best_cell_mean": float(np.min(means) if lower else np.max(means)),
            "worst_cell_mean": float(np.max(means) if lower else np.min(means)),
            "between_variant_iqr": float(np.quantile(means, 0.75) - np.quantile(means, 0.25)) if means.size > 1 else 0.0,
            "median_within_cell_mc_sd": float(np.median(sds)) if sds.size else None,
            "note": (
                "Family-level reproducibility summary over variant cells. "
                "This output is not merged into the primary factorial benchmark ranking."
            ),
        })
    return out


def run_signal_variant_robustness(
        output_root,
        irmf_params=GLOBAL_IRMF_PARAMS,
        emd_params=GLOBAL_EMD_PARAMS,
        eemd_params=GLOBAL_EEMD_PARAMS,
        ceemdan_params=GLOBAL_CEEMDAN_PARAMS,
        signals=SIGNAL_VARIANT_FAMILY,
        noises=SIGNAL_VARIANT_NOISES,
        sigmas=SIGNAL_VARIANT_SIGMAS,
        target_snr_db_levels=SIGNAL_VARIANT_TARGET_SNR_DB_LEVELS,
        seeds=SIGNAL_VARIANT_SEEDS,
        n=DEFAULT_N,
        fs=DEFAULT_FS,
        timeout_seconds=DEFAULT_METHOD_TIMEOUT_SECONDS,
):
    output_root = ensure_dir(output_root)
    signals = tuple(signals)
    noises = tuple(noises)
    target_snr_db_levels = tuple(target_snr_db_levels)
    seeds = tuple(seeds)
    checkpoint_path = output_root / "section_5_signal_family_reproducibility_target_snr_rows.jsonl"
    expected_cell_ids = {
        _target_snr_checkpoint_key(signal_name, noise_name, target_snr_db, int(seed))
        for signal_name in signals
        for noise_name in noises
        for target_snr_db in target_snr_db_levels
        for seed in seeds
    }
    loaded_rows = _load_checkpoint_rows(checkpoint_path)
    rows = [
        row for row in loaded_rows
        if row.get("variant_cell_id") in expected_cell_ids
    ]
    completed = {
        row.get("variant_cell_id")
        for row in rows
        if row.get("variant_cell_id")
    }
    expected_n = len(expected_cell_ids)
    print(
        f"[signal-variant] active-scope completed {len(completed)}/{expected_n} "
        f"case rows; remaining {expected_n - len(completed)}",
        flush=True,
    )
    for signal_name in signals:
        for noise_name in noises:
            for target_snr_db in target_snr_db_levels:
                sigma = _sigma_label_for_snr(target_snr_db)
                for seed in seeds:
                    data_seed = int(seed)
                    variant_cell_id = _target_snr_checkpoint_key(
                        signal_name, noise_name, target_snr_db, data_seed
                    )
                    if variant_cell_id in completed:
                        continue
                    algorithm_seed_base = _algorithm_seed_for(data_seed)
                    _, _, row = run_fixed_method_family_case(
                        signal_name=signal_name,
                        noise_name=noise_name,
                        sigma=sigma,
                        target_snr_db=target_snr_db,
                        irmf_params=irmf_params,
                        emd_params=emd_params,
                        eemd_params=eemd_params,
                        ceemdan_params=ceemdan_params,
                        n=n,
                        fs=fs,
                        seed=data_seed,
                        algorithm_seed=algorithm_seed_base,
                        timeout_seconds=timeout_seconds,
                        run_id_prefix=(
                            f"variant_{signal_name}_{noise_name}_"
                            f"snr{float(target_snr_db):g}dB_dataseed{data_seed}"
                        ),
                    )
                    row["signal_regime"] = "canonical_variant"
                    row["variant_family"] = _variant_family(signal_name)
                    row["variant_signal"] = signal_name
                    row["seed"] = data_seed
                    row["data_seed"] = data_seed
                    row["algorithm_seed"] = algorithm_seed_base
                    row["algorithm_seed_base"] = algorithm_seed_base
                    row["algorithm_seed_EEMD"] = algorithm_seed_base
                    row["algorithm_seed_CEEMDAN"] = algorithm_seed_base + 50000
                    row["seed_role"] = "data_generation_seed"
                    row["algorithm_seed_policy"] = "EEMD uses 100000 + data_seed; CEEMDAN uses 150000 + data_seed"
                    row["variant_cell_id"] = variant_cell_id
                    rows.append(row)
                    completed.add(variant_cell_id)
                    _append_checkpoint_row(checkpoint_path, row)
                    if len(completed) % 25 == 0 or len(completed) == expected_n:
                        print(
                            f"[signal-variant] active-scope completed "
                            f"{len(completed)}/{expected_n} case rows",
                            flush=True,
                        )

    protocol = {
        "section": "5 Signal-Family Reproducibility Study",
        "purpose": (
            "Test whether benchmark conclusions reproduce across alternative "
            "formula/parameter instances within each canonical signal family."
        ),
        "signals": list(signals),
        "variant_families": sorted({_variant_family(s) for s in signals}),
        "noises": list(noises),
        "sigmas": list(sigmas),
        "target_snr_db_levels": list(target_snr_db_levels),
        "noise_severity_design": "target_snr_energy_ratio",
        "sigma_role": "snr_equivalent_label_not_fixed_noise_amplitude",
        "seeds": list(seeds),
        "methods": list(METHODS),
        "n_variant_noise_sigma_cells": int(len(signals) * len(noises) * len(target_snr_db_levels)),
        "n_monte_carlo_replications_per_cell": int(len(seeds)),
        "n_signal_noise_sigma_seed_cells": int(len(signals) * len(noises) * len(target_snr_db_levels) * len(seeds)),
        "n_method_case_evaluations": int(len(signals) * len(noises) * len(target_snr_db_levels) * len(seeds) * len(METHODS)),
        "analysis_role": "separate_signal_family_reproducibility_layer",
        "not_primary_ranking": True,
        "primary_cube_policy": (
            "Variant cells are not merged into the 600-cell primary factorial benchmark. "
            "They are analyzed by canonical family to avoid implicit overweighting of "
            "families with more variants."
        ),
        "noise_sigma_rationale": (
            "Representative noise mechanisms and three prespecified target-SNR "
            "levels are used to test within-family reproducibility without turning "
            "this layer into a second full seven-SNR factorial benchmark."
        ),
        "checkpoint_policy": {
            "enabled": True,
            "checkpoint_file": checkpoint_path.name,
            "resume_behavior": "completed variant_cell_id rows are skipped on rerun",
            "current_scope_filter": "checkpoint rows outside the active signal/noise/SNR/seed scope are ignored in generated summaries",
            "loaded_checkpoint_rows": int(len(loaded_rows)),
            "active_scope_checkpoint_rows": int(len(rows)),
            "write_granularity": "one variant-noise-sigma-data_seed cell after all methods finish",
        },
        "seed_policy": {
            "data_seed": "controls signal/noise realization, contamination locations, AR innovations, and heteroskedastic innovations",
            "algorithm_seed": "controls EEMD and CEEMDAN internal ensemble noise only",
            "EEMD_algorithm_seed": "100000 + data_seed",
            "CEEMDAN_algorithm_seed": "150000 + data_seed",
            "paired_design": "all methods process the same observed signal within each data_seed replication",
        },
    }
    write_json(protocol, output_root / "section_5_signal_family_reproducibility_protocol.json")

    aggregate = write_section_outputs(rows, output_root, "section_5_signal_family_reproducibility")
    mc_summary = monte_carlo_cell_summary_rows(rows)
    paired_mc = paired_method_difference_summary_rows(rows)
    seed_stability = seed_stability_summary_rows(rows)
    win_tie_loss = method_win_tie_loss_rows(rows)
    family_summary = _summarize_family_reproducibility(rows)

    write_csv(mc_summary, output_root / "variant_monte_carlo_cell_summary.csv")
    write_json(mc_summary, output_root / "variant_monte_carlo_cell_summary.json")
    write_csv(paired_mc, output_root / "variant_paired_method_difference_summary.csv")
    write_json(paired_mc, output_root / "variant_paired_method_difference_summary.json")
    write_csv(seed_stability, output_root / "variant_seed_stability_summary.csv")
    write_json(seed_stability, output_root / "variant_seed_stability_summary.json")
    write_csv(win_tie_loss, output_root / "variant_method_win_tie_loss.csv")
    write_json(win_tie_loss, output_root / "variant_method_win_tie_loss.json")
    write_csv(family_summary, output_root / "variant_family_reproducibility_summary.csv")
    write_json(family_summary, output_root / "variant_family_reproducibility_summary.json")

    dashboard = {
        "protocol": protocol,
        "aggregate_rows": len(aggregate),
        "monte_carlo_cell_summary_rows": len(mc_summary),
        "paired_method_difference_summary_rows": len(paired_mc),
        "seed_stability_summary_rows": len(seed_stability),
        "method_win_tie_loss_rows": len(win_tie_loss),
        "variant_family_reproducibility_summary_rows": len(family_summary),
        "outputs": {
            "rows": "section_5_signal_family_reproducibility.csv",
            "checkpoint_rows_jsonl": checkpoint_path.name,
            "monte_carlo_cell_summary": "variant_monte_carlo_cell_summary.csv",
            "paired_method_difference_summary": "variant_paired_method_difference_summary.csv",
            "seed_stability_summary": "variant_seed_stability_summary.csv",
            "method_win_tie_loss": "variant_method_win_tie_loss.csv",
            "family_reproducibility_summary": "variant_family_reproducibility_summary.csv",
        },
    }
    write_json(dashboard, output_root / "section_5_signal_family_reproducibility_dashboard.json")
    return rows, dashboard


if __name__ == "__main__":
    run_signal_variant_robustness(Path("IRMF_EMD_PAPER_RESULTS") / "section_5_signal_family_reproducibility")
