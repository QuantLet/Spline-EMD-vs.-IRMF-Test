#!/usr/bin/python
# coding: UTF-8

"""Section 7: statistical inference layer for fixed-parameter comparisons."""

from collections import defaultdict
import math
import numpy as np

from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


METRICS = (
    "case_score",
    "reconstruction_score",
    "structural_fidelity_score",
    "contamination_resistance_score",
    "denoise_nmse",
    "imf_recovery_score",
    "component_splitting_index",
    "component_merging_index",
    "noise_capture_corr",
    "outlier_resistance_index",
)

LOWER_IS_BETTER = {"denoise_nmse", "component_splitting_index", "component_merging_index"}


def _as_float(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return None


def _paired_records(rows, metric):
    records = []
    for idx, row in enumerate(rows):
        i = _as_float(row.get("IRMF", {}).get(metric))
        e = _as_float(row.get("EMD", {}).get(metric))
        if i is None or e is None:
            continue
        delta = i - e
        records.append({
            "case_id": row.get("case_id", idx),
            "signal": row.get("signal"),
            "noise": row.get("noise"),
            "sigma": row.get("sigma"),
            "metric": metric,
            "irmf": i,
            "emd": e,
            "delta_raw_irmf_minus_emd": delta,
            "delta_benefit": -delta if metric in LOWER_IS_BETTER else delta,
        })
    return records


def _bootstrap_ci(values, rng, n_boot=2000, alpha=0.05):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, None
    if values.size == 1:
        return float(values[0]), float(values[0])
    idx = rng.integers(0, values.size, size=(int(n_boot), values.size))
    means = np.mean(values[idx], axis=1)
    return (
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    )


def _normal_two_sided_p_from_z(z):
    return float(math.erfc(abs(float(z)) / math.sqrt(2.0)))


def _wilcoxon_signed_rank_approx(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values[np.abs(values) > 1e-12]
    n = values.size
    if n < 2:
        return {"wilcoxon_n": int(n), "wilcoxon_z": None, "wilcoxon_p_approx": None}
    order = np.argsort(np.abs(values))
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    w_plus = float(np.sum(ranks[values > 0]))
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    z = (w_plus - mean) / math.sqrt(var + 1e-12)
    return {
        "wilcoxon_n": int(n),
        "wilcoxon_w_plus": w_plus,
        "wilcoxon_z": float(z),
        "wilcoxon_p_approx": _normal_two_sided_p_from_z(z),
    }


def _benjamini_hochberg(rows, p_key="wilcoxon_p_approx", q_key="wilcoxon_q_bh"):
    valid = [(idx, row[p_key]) for idx, row in enumerate(rows) if row.get(p_key) is not None]
    if not valid:
        return rows
    valid_sorted = sorted(valid, key=lambda x: x[1])
    m = len(valid_sorted)
    adjusted = [None] * len(rows)
    running = 1.0
    for rank_from_end, (idx, p) in enumerate(reversed(valid_sorted), start=1):
        rank = m - rank_from_end + 1
        running = min(running, p * m / rank)
        adjusted[idx] = float(min(running, 1.0))
    for idx, q in enumerate(adjusted):
        if q is not None:
            rows[idx][q_key] = q
    return rows


def _cliffs_delta(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float((np.sum(values > 0) - np.sum(values < 0)) / values.size)


def _group_summary(records, group_key):
    grouped = defaultdict(list)
    for rec in records:
        grouped[rec.get(group_key)].append(rec["delta_benefit"])
    rows = []
    for key, vals in sorted(grouped.items(), key=lambda kv: str(kv[0])):
        arr = np.asarray(vals, dtype=float)
        rows.append({
            group_key: key,
            "metric": records[0]["metric"] if records else None,
            "n": int(arr.size),
            "mean_benefit_delta": float(np.mean(arr)),
            "median_benefit_delta": float(np.median(arr)),
            "win_rate": float(np.mean(arr > 0)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        })
    return rows


def _ols_delta_model(records):
    """
    Small no-dependency fixed-effect model:
        delta_benefit ~ 1 + signal + noise + sigma

    This is not a replacement for a full mixed-effects model, but it gives a
    reproducible inference table when statsmodels/R are unavailable.
    """
    if not records:
        return [], {}
    y = np.asarray([r["delta_benefit"] for r in records], dtype=float)
    signals = sorted({r.get("signal") for r in records})
    noises = sorted({r.get("noise") for r in records})
    sigmas = sorted({r.get("sigma") for r in records})
    cols = [("intercept", None)]
    cols.extend(("signal", s) for s in signals[1:])
    cols.extend(("noise", n) for n in noises[1:])
    cols.extend(("sigma", s) for s in sigmas[1:])
    X = np.ones((len(records), len(cols)), dtype=float)
    for j, (kind, level) in enumerate(cols[1:], start=1):
        X[:, j] = [1.0 if r.get(kind) == level else 0.0 for r in records]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    dof = max(len(y) - X.shape[1], 1)
    sigma2 = float(np.sum(resid ** 2) / dof)
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
    coef_rows = []
    for (kind, level), b, s in zip(cols, beta, se):
        z = float(b / (s + 1e-12))
        coef_rows.append({
            "term": kind if kind == "intercept" else f"{kind}[{level}]",
            "estimate": float(b),
            "std_error": float(s),
            "z_approx": z,
            "p_approx": _normal_two_sided_p_from_z(z),
        })
    model_summary = {
        "n": int(len(y)),
        "n_parameters": int(X.shape[1]),
        "residual_std": float(math.sqrt(sigma2)),
        "r_squared": float(1.0 - np.sum(resid ** 2) / (np.sum((y - np.mean(y)) ** 2) + 1e-12)),
        "baseline_signal": signals[0] if signals else None,
        "baseline_noise": noises[0] if noises else None,
        "baseline_sigma": sigmas[0] if sigmas else None,
    }
    return coef_rows, model_summary


def run_statistical_inference(output_root, main_rows, n_boot=2000, seed=12345):
    output_root = ensure_dir(output_root)
    rng = np.random.default_rng(seed)
    metric_rows = []
    all_group_rows = []
    all_coef_rows = []
    model_summaries = []

    for metric in METRICS:
        records = _paired_records(main_rows, metric)
        values = np.asarray([r["delta_benefit"] for r in records], dtype=float)
        if values.size == 0:
            continue
        ci_low, ci_high = _bootstrap_ci(values, rng=rng, n_boot=n_boot)
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        se = float(sd / math.sqrt(values.size)) if values.size else None
        z = mean / (se + 1e-12) if se is not None else None
        infer = {
            "metric": metric,
            "direction": "lower is better" if metric in LOWER_IS_BETTER else "higher is better",
            "n_paired": int(values.size),
            "mean_benefit_delta": mean,
            "median_benefit_delta": float(np.median(values)),
            "std_benefit_delta": sd,
            "se_benefit_delta": se,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "normal_z_approx": float(z) if z is not None else None,
            "normal_p_approx": _normal_two_sided_p_from_z(z) if z is not None else None,
            "win_rate": float(np.mean(values > 0)),
            "cliffs_delta_vs_zero": _cliffs_delta(values),
        }
        infer.update(_wilcoxon_signed_rank_approx(values))
        metric_rows.append(infer)

        for group_key in ("signal", "noise", "sigma"):
            for row in _group_summary(records, group_key):
                row["grouping"] = group_key
                all_group_rows.append(row)

        coef_rows, model_summary = _ols_delta_model(records)
        for row in coef_rows:
            row["metric"] = metric
            all_coef_rows.append(row)
        model_summary["metric"] = metric
        model_summaries.append(model_summary)

    metric_rows = _benjamini_hochberg(metric_rows)
    write_csv(metric_rows, output_root / "paired_effect_sizes_and_ci.csv")
    write_json(metric_rows, output_root / "paired_effect_sizes_and_ci.json")
    write_csv(all_group_rows, output_root / "method_effect_by_signal_noise_sigma.csv")
    write_json(all_group_rows, output_root / "method_effect_by_signal_noise_sigma.json")
    write_csv(all_coef_rows, output_root / "fixed_effect_delta_model_coefficients.csv")
    write_json(all_coef_rows, output_root / "fixed_effect_delta_model_coefficients.json")
    write_csv(model_summaries, output_root / "fixed_effect_delta_model_summaries.csv")
    write_json(model_summaries, output_root / "fixed_effect_delta_model_summaries.json")

    protocol = {
        "section": "7.x Classical-reference paired inference",
        "paper_role": (
            "focused IRMF-vs-classical-EMD contrast retained for interpretability; "
            "the main cross-method inference is the four-method repeated-measures "
            "and primary-evaluation framework."
        ),
        "paired_design": "Each row is one signal-noise-sigma case evaluated by IRMF-fixed and classical EMD-fixed.",
        "delta_definition": "delta_benefit is IRMF-EMD for higher-is-better metrics and EMD-IRMF for lower-is-better metrics.",
        "bootstrap_repetitions": int(n_boot),
        "p_values": "Normal and Wilcoxon signed-rank approximations are provided for screening; confirm final manuscript statistics in R/statsmodels if desired.",
        "fixed_effect_model": "delta_benefit ~ signal + noise + sigma, implemented by no-dependency least squares.",
    }
    write_json(protocol, output_root / "statistical_inference_protocol.json")
    return {
        "n_metrics": len(metric_rows),
        "metrics": metric_rows,
        "n_group_rows": len(all_group_rows),
        "n_model_coefficients": len(all_coef_rows),
    }
