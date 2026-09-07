#!/usr/bin/python
# coding: UTF-8

"""V5.54 supplementary noise-realization decomposition stability.

This stage intentionally does not evaluate fixed-input stochastic
repeatability.  IRMF and standard EMD are deterministic under fixed inputs,
and EEMD/CEEMDAN fixed-input repeatability would create an asymmetric
method-specific supplement.  Instead, this module summarizes stability across
Monte Carlo noise realizations using the existing target-SNR benchmark rows.
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
from pathlib import Path

import numpy as np

from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


V554_NOISE_REALIZATION_STABILITY_VERSION = (
    "V5.54_supplementary_noise_realization_decomposition_stability"
)

SOURCE_REL = Path("06_protocol_controlled_full_benchmark_rerun") / "unified_benchmark_cube_rows.csv"
METHODS = ("IRMF", "EMD", "EEMD", "CEEMDAN")


def _finite(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return None


def _int_or_none(value):
    value = _finite(value)
    if value is None:
        return None
    return int(round(value))


def _read_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def _group_key(row):
    snr = row.get("target_snr_db")
    if snr in (None, ""):
        snr = row.get("sigma")
    return (
        row.get("signal_regime"),
        row.get("signal"),
        row.get("noise"),
        str(snr),
    )


def _seed_value(row):
    return row.get("data_seed") or row.get("seed")


def _method_value(row, method, metric):
    return row.get(f"{method}_{metric}")


def _component_count_stability(groups):
    out = []
    for (signal_regime, signal, noise, snr), method_rows in sorted(groups.items()):
        for method, rows in sorted(method_rows.items()):
            seeds = sorted({_seed_value(row) for row in rows})
            for count_metric in ("effective_imf_count", "protocol_selected_component_count"):
                counts = [
                    _int_or_none(_method_value(row, method, count_metric))
                    for row in rows
                ]
                counts = [v for v in counts if v is not None]
                n = len(counts)
                hist = Counter(counts)
                modal_count, modal_n = (None, 0)
                if hist:
                    modal_count, modal_n = hist.most_common(1)[0]
                out.append({
                    "diagnostic": "component_count_stability_across_noise_realizations",
                    "count_metric": count_metric,
                    "signal_regime": signal_regime,
                    "signal": signal,
                    "noise": noise,
                    "target_snr_db": snr,
                    "method": method,
                    "n_realizations": int(len(seeds)),
                    "n_finite_counts": int(n),
                    "modal_count": modal_count,
                    "modal_count_fraction": float(modal_n / n) if n else None,
                    "unique_count_values": ";".join(str(k) for k in sorted(hist)),
                    "count_value_histogram": ";".join(
                        f"{k}:{hist[k]}" for k in sorted(hist)
                    ),
                    "mean_count": float(np.mean(counts)) if n else None,
                    "sd_count": float(np.std(counts, ddof=1)) if n > 1 else 0.0 if n else None,
                    "interpretation": (
                        "Fraction of Monte Carlo noise realizations yielding the "
                        "same component count under fixed signal/noise/SNR/method."
                    ),
                })
    return out


def _truth_conditioned_recovery_frequency(groups):
    out = []
    for (signal_regime, signal, noise, snr), method_rows in sorted(groups.items()):
        for method, rows in sorted(method_rows.items()):
            recovered_counts = []
            true_counts = []
            fractions = []
            seeds = sorted({_seed_value(row) for row in rows})
            for row in rows:
                k_true = _int_or_none(_method_value(row, method, "true_component_count"))
                missing = _int_or_none(_method_value(row, method, "missing_true_component_count"))
                matched = _int_or_none(_method_value(row, method, "imf_recovery_matched_count"))
                if k_true is None or k_true <= 0:
                    continue
                if missing is not None:
                    recovered = max(0, k_true - missing)
                elif matched is not None:
                    recovered = min(k_true, matched)
                else:
                    continue
                recovered_counts.append(float(recovered))
                true_counts.append(float(k_true))
                fractions.append(float(recovered / k_true))
            n = len(fractions)
            total_true = float(np.sum(true_counts)) if true_counts else 0.0
            total_recovered = float(np.sum(recovered_counts)) if recovered_counts else 0.0
            out.append({
                "diagnostic": "truth_conditioned_component_recovery_frequency",
                "signal_regime": signal_regime,
                "signal": signal,
                "noise": noise,
                "target_snr_db": snr,
                "method": method,
                "n_realizations": int(len(seeds)),
                "n_computable_realizations": int(n),
                "aggregate_recovered_true_components": total_recovered if n else None,
                "aggregate_true_components": total_true if n else None,
                "recovery_frequency": float(total_recovered / total_true) if total_true else None,
                "median_recovery_fraction_per_realization": float(np.median(fractions)) if n else None,
                "q25_recovery_fraction_per_realization": float(np.quantile(fractions, 0.25)) if n else None,
                "q75_recovery_fraction_per_realization": float(np.quantile(fractions, 0.75)) if n else None,
                "interpretation": (
                    "Truth-conditioned frequency with which true components are "
                    "successfully represented across independent noise realizations; "
                    "this is not fixed-input algorithmic repeatability."
                ),
            })
    return out


def _method_level_summary(rows, value_field):
    grouped = defaultdict(list)
    for row in rows:
        value = _finite(row.get(value_field))
        if value is None:
            continue
        grouped[(row.get("diagnostic"), row.get("method"))].append(value)
    out = []
    for (diagnostic, method), vals in sorted(grouped.items()):
        arr = np.asarray(vals, dtype=float)
        out.append({
            "diagnostic": diagnostic,
            "method": method,
            "n_cells": int(arr.size),
            f"median_{value_field}": float(np.median(arr)),
            f"q25_{value_field}": float(np.quantile(arr, 0.25)),
            f"q75_{value_field}": float(np.quantile(arr, 0.75)),
            f"mean_{value_field}": float(np.mean(arr)),
        })
    return out


def run_v554_noise_realization_decomposition_stability(algorithm_root):
    algorithm_root = ensure_dir(algorithm_root)
    output_root = ensure_dir(algorithm_root / "16w_v554_noise_realization_stability")
    source_path = algorithm_root / SOURCE_REL
    if not source_path.exists():
        raise FileNotFoundError(
            f"V5.54 requires existing target-SNR benchmark rows at {source_path}"
        )

    groups = defaultdict(lambda: defaultdict(list))
    n_source_rows = 0
    for row in _read_rows(source_path):
        n_source_rows += 1
        key = _group_key(row)
        for method in METHODS:
            if _method_value(row, method, "method") or _method_value(row, method, "denoise_nmse"):
                groups[key][method].append(row)

    count_rows = _component_count_stability(groups)
    recovery_rows = _truth_conditioned_recovery_frequency(groups)
    count_summary = _method_level_summary(count_rows, "modal_count_fraction")
    recovery_summary = _method_level_summary(recovery_rows, "recovery_frequency")

    cross_realization_status = [{
        "diagnostic": "cross_realization_matched_component_similarity",
        "status": "not_computable_from_current_compact_benchmark_rows",
        "reason": (
            "Existing unified benchmark rows intentionally store compact method "
            "summaries and do not include full component waveforms for pairwise "
            "matching across Monte Carlo realizations."
        ),
        "required_artifact_for_future_execution": (
            "full component arrays or a dedicated small supplementary rerun that "
            "persists matched components for each noise realization"
        ),
        "not_substituted_by": (
            "matched_component_corr, because that metric is truth-referenced "
            "within one realization rather than a cross-realization component "
            "similarity estimand"
        ),
    }]

    seed_precision = []
    for r in (20, 50, 100):
        max_se = (0.25 / float(r)) ** 0.5
        seed_precision.append({
            "n_realizations": int(r),
            "recovery_frequency_step": float(1.0 / r),
            "max_binomial_se_at_p_0_5": float(max_se),
            "approx_95pct_margin_at_p_0_5": float(1.96 * max_se),
            "role": (
                "current compact benchmark" if r == 20 else
                "candidate seed-convergence extension, not yet executed here"
            ),
        })

    dashboard = {
        "schema_version": V554_NOISE_REALIZATION_STABILITY_VERSION,
        "created_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "paper_role": "Supplementary/protocol stability analysis, not primary endpoint and not ranking-bearing.",
        "fixed_input_repeatability_policy": (
            "Not added. IRMF and standard EMD are deterministic under fixed "
            "inputs; fixed-input repeatability would be asymmetric for "
            "EEMD/CEEMDAN and is not the scientific perturbation axis used here."
        ),
        "stability_axis": (
            "independent Monte Carlo noise realizations at fixed clean signal, "
            "noise family, target SNR, method parameters, and implementation"
        ),
        "source_rows": str(source_path),
        "n_source_rows": int(n_source_rows),
        "n_signal_noise_snr_groups": int(len(groups)),
        "diagnostics": [
            "component_count_stability_across_noise_realizations",
            "truth_conditioned_component_recovery_frequency",
            "cross_realization_matched_component_similarity",
        ],
        "computed_from_current_artifacts": [
            "component_count_stability_across_noise_realizations",
            "truth_conditioned_component_recovery_frequency",
        ],
        "not_computable_from_current_artifacts": [
            "cross_realization_matched_component_similarity",
        ],
        "seed_count_decision": {
            "current_benchmark_realizations": 20,
            "interpretation": (
                "Twenty seeds can support the current supplementary summaries, "
                "but recovery-frequency estimates have 5 percentage-point "
                "resolution. A 20->50->100 convergence audit is recommended "
                "before making recovery-frequency precision a headline claim."
            ),
            "recommended_if_promoted_to_quantitative_stability_claim": (
                "Use 50 seeds as the practical main Monte Carlo count and reserve "
                "100 seeds for convergence auditing or selected cells."
            ),
        },
        "claim_boundary": (
            "This artifact supports noise-realization stability summaries for "
            "component counts and truth-conditioned recovery frequency from "
            "existing compact rows. It does not authorize fixed-input stochastic "
            "repeatability claims or cross-realization waveform-similarity claims."
        ),
        "outputs": {
            "component_count_stability": "v554_component_count_stability_across_noise_realizations.csv",
            "truth_conditioned_recovery_frequency": "v554_truth_conditioned_component_recovery_frequency.csv",
            "cross_realization_similarity_status": "v554_cross_realization_matched_component_similarity_status.csv",
            "seed_precision_table": "v554_seed_precision_table.csv",
        },
    }

    write_json(dashboard, output_root / "v554_noise_realization_stability_dashboard.json")
    write_csv(count_rows, output_root / "v554_component_count_stability_across_noise_realizations.csv")
    write_csv(recovery_rows, output_root / "v554_truth_conditioned_component_recovery_frequency.csv")
    write_csv(count_summary, output_root / "v554_component_count_stability_method_summary.csv")
    write_csv(recovery_summary, output_root / "v554_truth_conditioned_recovery_frequency_method_summary.csv")
    write_csv(cross_realization_status, output_root / "v554_cross_realization_matched_component_similarity_status.csv")
    write_csv(seed_precision, output_root / "v554_seed_precision_table.csv")
    return dashboard
