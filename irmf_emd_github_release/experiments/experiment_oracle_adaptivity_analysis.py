#!/usr/bin/python
# coding: UTF-8

"""IRMF oracle upper-bound and adaptivity-gap analysis.

The oracle analysis estimates algorithmic potential under ideal per-case
parameter selection.  It is not a deployable protocol and is not used for the
main fixed-parameter claims.  Its primary estimand is the adaptivity gap:

    gap = oracle performance - fixed global-parameter performance

with metric direction normalized so positive values mean the oracle improves
over the fixed global IRMF configuration.
"""

import numpy as np

from project_config import (
    DEFAULT_FS,
    DEFAULT_N,
    DEFAULT_SEED,
    GLOBAL_EMD_PARAMS,
    GLOBAL_IRMF_PARAMS,
    ORACLE_CEEMDAN_EXPLORATORY_EPSILONS,
    ORACLE_CEEMDAN_EXPLORATORY_MAX_IMF,
    ORACLE_CEEMDAN_EXPLORATORY_QUICK_EPSILONS,
    ORACLE_CEEMDAN_EXPLORATORY_QUICK_MAX_IMF,
    ORACLE_CEEMDAN_EXPLORATORY_QUICK_TRIALS,
    ORACLE_CEEMDAN_EXPLORATORY_TRIALS,
    ORACLE_ADAPTIVITY_NOISES,
    ORACLE_ADAPTIVITY_QUICK_NOISES,
    ORACLE_ADAPTIVITY_QUICK_SIGMAS,
    ORACLE_ADAPTIVITY_QUICK_SIGNALS,
    ORACLE_ADAPTIVITY_SEARCH_MODE,
    ORACLE_ADAPTIVITY_SIGMAS,
    ORACLE_ADAPTIVITY_SIGNALS,
)
from core_algorithms.ceemdan_wrapper import run_ceemdan
from diagnostics.shared_physical_diagnostics import evaluate_shared_physical_diagnostics
from experiments.experiment_utils import (
    default_irmf_grid,
    make_signal_noise_case,
    method_result_summary,
    run_fixed_irmf_case,
    select_irmf_best,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json
from parameter_search.irmf_parameter_search import run_irmf_parameter_search


METRICS = (
    "case_score",
    "reconstruction_score",
    "structural_fidelity_score",
    "contamination_resistance_score",
    "denoise_nmse",
    "denoise_corr",
    "imf_recovery_score",
    "component_splitting_index",
    "component_merging_index",
    "inter_imf_entanglement_index",
    "mode_mixing_index",
    "outlier_resistance_index",
    "noise_capture_corr",
)
LOWER_IS_BETTER = {
    "denoise_nmse",
    "component_splitting_index",
    "component_merging_index",
    "inter_imf_entanglement_index",
    "mode_mixing_index",
}


def _finite(value):
    try:
        return value is not None and np.isfinite(float(value))
    except Exception:
        return False


def _metric_gap(metric, fixed_value, oracle_value):
    if not (_finite(fixed_value) and _finite(oracle_value)):
        return np.nan
    fixed_value = float(fixed_value)
    oracle_value = float(oracle_value)
    if metric in LOWER_IS_BETTER:
        return fixed_value - oracle_value
    return oracle_value - fixed_value


def _method_advantage(metric, method_value, baseline_value):
    """Positive value means the first method is better than the baseline."""
    if not (_finite(method_value) and _finite(baseline_value)):
        return np.nan
    method_value = float(method_value)
    baseline_value = float(baseline_value)
    if metric in LOWER_IS_BETTER:
        return baseline_value - method_value
    return method_value - baseline_value


def _score(result, metric):
    value = result.get(metric)
    return float(value) if _finite(value) else np.nan


def _summarize(rows, group_keys=None):
    group_keys = [] if group_keys is None else list(group_keys)
    groups = {}
    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        groups.setdefault(key, []).append(row)

    out = []
    for key, subset in sorted(groups.items(), key=lambda item: str(item[0])):
        item = {k: v for k, v in zip(group_keys, key)}
        item["n_cases"] = int(len(subset))
        for metric in METRICS:
            fixed_vals, oracle_vals, gaps = [], [], []
            for row in subset:
                fv = row.get(f"fixed_{metric}")
                ov = row.get(f"oracle_{metric}")
                gv = row.get(f"gap_{metric}")
                if _finite(fv):
                    fixed_vals.append(float(fv))
                if _finite(ov):
                    oracle_vals.append(float(ov))
                if _finite(gv):
                    gaps.append(float(gv))
            if gaps:
                item[f"fixed_mean_{metric}"] = float(np.mean(fixed_vals)) if fixed_vals else np.nan
                item[f"oracle_mean_{metric}"] = float(np.mean(oracle_vals)) if oracle_vals else np.nan
                item[f"adaptivity_gap_mean_{metric}"] = float(np.mean(gaps))
                item[f"adaptivity_gap_median_{metric}"] = float(np.median(gaps))
                item[f"adaptivity_gap_p90_{metric}"] = float(np.quantile(gaps, 0.90))
                item[f"positive_gap_rate_{metric}"] = float(np.mean(np.asarray(gaps) > 0.0))
        out.append(item)
    return out


def _candidate_usage(rows):
    keys = ("oracle_h1", "oracle_a", "oracle_h_min", "oracle_H")
    groups = {}
    for row in rows:
        key = tuple(row.get(k) for k in keys)
        groups.setdefault(key, []).append(row)
    out = []
    for key, subset in sorted(groups.items(), key=lambda item: str(item[0])):
        item = {k: v for k, v in zip(keys, key)}
        item["n_cases"] = int(len(subset))
        item["case_fraction"] = float(len(subset) / max(len(rows), 1))
        vals = [row.get("gap_case_score") for row in subset if _finite(row.get("gap_case_score"))]
        if vals:
            item["mean_gap_case_score"] = float(np.mean(vals))
        out.append(item)
    return out


def _oracle_search(case, fs, search_mode, boundary_mode, min_support_points):
    grid = default_irmf_grid(search_mode)
    results = run_irmf_parameter_search(
        Y=case["Y"],
        X_clean=case["X_clean"],
        fs=fs,
        T=case["t"],
        expected_noise_ratio=case.get("expected_noise_ratio"),
        boundary_mode=boundary_mode,
        min_support_points=min_support_points,
        true_components=case.get("true_components"),
        **grid,
    )
    best = select_irmf_best(results)
    return results, best["physical_best"]


def _ceemdan_grid(quick=False):
    trials = ORACLE_CEEMDAN_EXPLORATORY_QUICK_TRIALS if quick else ORACLE_CEEMDAN_EXPLORATORY_TRIALS
    epsilons = ORACLE_CEEMDAN_EXPLORATORY_QUICK_EPSILONS if quick else ORACLE_CEEMDAN_EXPLORATORY_EPSILONS
    max_imfs = ORACLE_CEEMDAN_EXPLORATORY_QUICK_MAX_IMF if quick else ORACLE_CEEMDAN_EXPLORATORY_MAX_IMF
    return [
        {"trials": int(t), "epsilon": float(e), "max_imf": int(m)}
        for t in trials
        for e in epsilons
        for m in max_imfs
    ]


def _ceemdan_oracle_search(case, fs, algorithm_seed=20260715, quick=False):
    candidates = []
    for params in _ceemdan_grid(quick=quick):
        raw = run_ceemdan(
            case["Y"],
            max_imf=params["max_imf"],
            trials=params["trials"],
            epsilon=params["epsilon"],
            random_seed=algorithm_seed,
        )
        physical = evaluate_shared_physical_diagnostics(
            Y_observed=case["Y"],
            X_clean=case["X_clean"],
            imfs=raw["imfs"],
            residual=raw["residual"],
            fs=fs,
            residual_penalty_mode="none",
            true_components=case.get("true_components"),
        )
        result = {
            "method": "CEEMDAN-oracle-candidate",
            "run_id": (
                f"ceemdan_oracle_trials_{params['trials']}"
                f"_epsilon_{params['epsilon']}_maximf_{params['max_imf']}"
            ),
            **raw,
            **physical,
            **params,
        }
        candidates.append(result)
    best = sorted(
        candidates,
        key=lambda r: (
            -float(r.get("case_score", -np.inf)),
            float(r.get("denoise_nmse", np.inf)),
        ),
    )[0]
    return candidates, best


def _write_oracle_plots(output_root, rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {"plots_created": False, "reason": "matplotlib_unavailable"}

    output_root = ensure_dir(output_root)
    created = []
    gaps = np.asarray([r.get("gap_case_score") for r in rows if _finite(r.get("gap_case_score"))], dtype=float)
    if len(gaps):
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.hist(gaps, bins=min(20, max(5, len(gaps))), color="#4c78a8", edgecolor="white")
        ax.axvline(float(np.mean(gaps)), color="#f58518", linewidth=2, label="mean")
        ax.set_xlabel("Adaptivity gap in case_score")
        ax.set_ylabel("Number of cases")
        ax.set_title("IRMF oracle adaptivity gap")
        ax.legend(frameon=False)
        fig.tight_layout()
        path = output_root / "adaptivity_gap_case_score_histogram.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        created.append(str(path))

    for group_key, filename, xlabel in [
        ("signal", "adaptivity_gap_by_signal_boxplot.png", "Signal class"),
        ("noise", "adaptivity_gap_by_noise_boxplot.png", "Noise model"),
        ("sigma", "adaptivity_gap_by_sigma_boxplot.png", "Sigma"),
    ]:
        groups = {}
        for row in rows:
            if _finite(row.get("gap_case_score")):
                groups.setdefault(str(row.get(group_key)), []).append(float(row["gap_case_score"]))
        if not groups:
            continue
        labels = sorted(groups)
        values = [groups[label] for label in labels]
        fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * len(labels)), 4.2))
        ax.boxplot(values, labels=labels, showfliers=False)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Adaptivity gap in case_score")
        ax.set_title(f"Adaptivity gap by {group_key}")
        ax.tick_params(axis="x", rotation=45 if len(labels) > 4 else 0)
        fig.tight_layout()
        path = output_root / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        created.append(str(path))
    return {"plots_created": True, "files": created}


def run_irmf_oracle_adaptivity_analysis(
        output_root,
        irmf_params=GLOBAL_IRMF_PARAMS,
        signals=ORACLE_ADAPTIVITY_SIGNALS,
        noises=ORACLE_ADAPTIVITY_NOISES,
        sigmas=ORACLE_ADAPTIVITY_SIGMAS,
        search_mode=ORACLE_ADAPTIVITY_SEARCH_MODE,
        n=DEFAULT_N,
        fs=DEFAULT_FS,
        seed=DEFAULT_SEED,
        quick=False,
        run_exploratory_ceemdan_oracle=False,
        algorithm_seed=20260715,
):
    """Run IRMF fixed-vs-oracle adaptivity-gap analysis."""
    output_root = ensure_dir(output_root)
    if quick:
        signals = ORACLE_ADAPTIVITY_QUICK_SIGNALS
        noises = ORACLE_ADAPTIVITY_QUICK_NOISES
        sigmas = ORACLE_ADAPTIVITY_QUICK_SIGMAS

    grid = default_irmf_grid(search_mode)
    n_candidates = int(
        len(grid["h1_options"])
        * len(grid["a_options"])
        * len(grid["h_min_options"])
        * len(grid["H_options"])
    )
    rows = []
    exploratory_rows = []
    for signal_name in signals:
        for noise_name in noises:
            for sigma in sigmas:
                case = make_signal_noise_case(
                    signal_name=signal_name,
                    noise_name=noise_name,
                    sigma=sigma,
                    n=n,
                    fs=fs,
                    seed=seed,
                )
                fixed = run_fixed_irmf_case(
                    Y=case["Y"],
                    X_clean=case["X_clean"],
                    t=case["t"],
                    fs=fs,
                    irmf_params=irmf_params,
                    expected_noise_ratio=case.get("expected_noise_ratio"),
                    true_components=case.get("true_components"),
                    run_id=f"fixed_{signal_name}_{noise_name}_{sigma}",
                )
                _, oracle = _oracle_search(
                    case=case,
                    fs=fs,
                    search_mode=search_mode,
                    boundary_mode=irmf_params.get("boundary_mode", "periodic"),
                    min_support_points=irmf_params.get("min_support_points", 3),
                )
                fixed_summary = method_result_summary(fixed)
                oracle_summary = method_result_summary(oracle)
                ceemdan_oracle_summary = None
                if run_exploratory_ceemdan_oracle:
                    _, ceemdan_oracle = _ceemdan_oracle_search(
                        case,
                        fs=fs,
                        algorithm_seed=algorithm_seed,
                        quick=quick,
                    )
                    ceemdan_oracle_summary = method_result_summary(ceemdan_oracle)
                row = {
                    "section": "oracle_upper_bound_and_adaptivity_gap",
                    "analysis_role": "algorithm_potential_not_main_claim",
                    "oracle_target": "attainable_upper_performance_bound_under_ideal_parameter_selection",
                    "main_estimand": "adaptivity_gap",
                    "signal": signal_name,
                    "noise": noise_name,
                    "sigma": float(sigma),
                    "seed": int(seed),
                    "search_mode": search_mode,
                    "n_oracle_candidates": n_candidates,
                    "quick": bool(quick),
                    "exploratory_ceemdan_oracle_run": bool(run_exploratory_ceemdan_oracle),
                    "fixed_h1": fixed_summary.get("h1"),
                    "fixed_a": fixed_summary.get("a"),
                    "fixed_h_min": fixed_summary.get("h_min"),
                    "fixed_H": fixed_summary.get("H"),
                    "oracle_h1": oracle_summary.get("h1"),
                    "oracle_a": oracle_summary.get("a"),
                    "oracle_h_min": oracle_summary.get("h_min"),
                    "oracle_H": oracle_summary.get("H"),
                }
                for metric in METRICS:
                    fv = _score(fixed_summary, metric)
                    ov = _score(oracle_summary, metric)
                    row[f"fixed_{metric}"] = fv
                    row[f"oracle_{metric}"] = ov
                    row[f"gap_{metric}"] = _metric_gap(metric, fv, ov)
                rows.append(row)
                if ceemdan_oracle_summary is not None:
                    exploratory = {
                        "section": "exploratory_oracle_irmf_vs_oracle_ceemdan",
                        "analysis_role": "exploratory_only_not_main_evidence",
                        "exploratory_only": True,
                        "not_used_for_main_claims": True,
                        "signal": signal_name,
                        "noise": noise_name,
                        "sigma": float(sigma),
                        "seed": int(seed),
                        "algorithm_seed": int(algorithm_seed),
                        "quick": bool(quick),
                        "irmf_oracle_h1": oracle_summary.get("h1"),
                        "irmf_oracle_a": oracle_summary.get("a"),
                        "irmf_oracle_h_min": oracle_summary.get("h_min"),
                        "irmf_oracle_H": oracle_summary.get("H"),
                        "ceemdan_oracle_trials": ceemdan_oracle_summary.get("trials"),
                        "ceemdan_oracle_epsilon": ceemdan_oracle_summary.get("epsilon"),
                        "ceemdan_oracle_max_imf": ceemdan_oracle_summary.get("max_imf"),
                    }
                    for metric in METRICS:
                        iv = _score(oracle_summary, metric)
                        cv = _score(ceemdan_oracle_summary, metric)
                        exploratory[f"irmf_oracle_{metric}"] = iv
                        exploratory[f"ceemdan_oracle_{metric}"] = cv
                        exploratory[f"irmf_oracle_advantage_over_ceemdan_oracle_{metric}"] = _method_advantage(metric, iv, cv)
                    exploratory_rows.append(exploratory)
                print(
                    "ORACLE ADAPTIVITY DONE | "
                    f"signal={signal_name} | noise={noise_name} | sigma={sigma} | "
                    f"gap_case_score={row.get('gap_case_score')}",
                    flush=True,
                )

    overall = _summarize(rows)
    by_signal = _summarize(rows, ["signal"])
    by_noise = _summarize(rows, ["noise"])
    by_sigma = _summarize(rows, ["sigma"])
    by_signal_noise = _summarize(rows, ["signal", "noise"])
    candidate_usage = _candidate_usage(rows)
    exploratory_summary = _summarize_exploratory(exploratory_rows)
    plot_manifest = _write_oracle_plots(output_root / "figures", rows)
    protocol = {
        "section": "Oracle Upper-Bound and Adaptivity-Gap Analysis",
        "purpose": (
            "Estimate IRMF's attainable performance ceiling and quantify how "
            "far the locked global parameter configuration is from the per-case "
            "oracle best configuration."
        ),
        "not_main_claim": True,
        "oracle_is_deployable_method": False,
        "evaluation_target": (
            "The oracle changes the evaluation target from deployable fixed-"
            "parameter performance to ideal per-case parameter-selection potential."
        ),
        "recommended_paper_language": (
            "Oracle evaluation estimates the attainable upper performance bound "
            "under ideal parameter selection.  It is used to quantify the "
            "adaptivity gap, not to support the main fixed-parameter benchmark claim."
        ),
        "adaptivity_gap_definition": (
            "For higher-is-better metrics, gap = oracle - fixed.  For lower-is-"
            "better metrics, gap = fixed - oracle.  Positive gap indicates "
            "recoverable performance through ideal parameter adaptation."
        ),
        "signals": list(signals),
        "noises": list(noises),
        "sigmas": list(sigmas),
        "search_mode": search_mode,
        "n_oracle_candidates": n_candidates,
        "grid": {k: list(map(float, v)) for k, v in grid.items()},
        "fixed_irmf_params": dict(irmf_params),
        "quick": bool(quick),
        "figures": plot_manifest,
        "exploratory_oracle_irmf_vs_oracle_ceemdan": {
            "run": bool(run_exploratory_ceemdan_oracle),
            "exploratory_only": True,
            "not_used_for_main_claims": True,
            "rationale": (
                "CEEMDAN oracle comparison is optional because the IRMF and "
                "CEEMDAN parameter spaces are structurally different.  It is "
                "reported only as a sensitivity/upper-bound reference."
            ),
            "ceemdan_grid": _ceemdan_grid(quick=quick) if run_exploratory_ceemdan_oracle else None,
        },
    }

    write_json(protocol, output_root / "oracle_adaptivity_protocol.json")
    write_json(rows, output_root / "oracle_adaptivity_rows.json")
    write_csv(rows, output_root / "oracle_adaptivity_rows.csv")
    write_json(overall, output_root / "oracle_adaptivity_gap_summary.json")
    write_csv(overall, output_root / "oracle_adaptivity_gap_summary.csv")
    write_json(by_signal, output_root / "oracle_adaptivity_by_signal.json")
    write_csv(by_signal, output_root / "oracle_adaptivity_by_signal.csv")
    write_json(by_noise, output_root / "oracle_adaptivity_by_noise.json")
    write_csv(by_noise, output_root / "oracle_adaptivity_by_noise.csv")
    write_json(by_sigma, output_root / "oracle_adaptivity_by_sigma.json")
    write_csv(by_sigma, output_root / "oracle_adaptivity_by_sigma.csv")
    write_json(by_signal_noise, output_root / "oracle_adaptivity_by_signal_noise.json")
    write_csv(by_signal_noise, output_root / "oracle_adaptivity_by_signal_noise.csv")
    write_json(candidate_usage, output_root / "oracle_adaptivity_candidate_usage.json")
    write_csv(candidate_usage, output_root / "oracle_adaptivity_candidate_usage.csv")
    if run_exploratory_ceemdan_oracle:
        write_json(exploratory_rows, output_root / "exploratory_oracle_irmf_vs_ceemdan_rows.json")
        write_csv(exploratory_rows, output_root / "exploratory_oracle_irmf_vs_ceemdan_rows.csv")
        write_json(exploratory_summary, output_root / "exploratory_oracle_irmf_vs_ceemdan_summary.json")
        write_csv(exploratory_summary, output_root / "exploratory_oracle_irmf_vs_ceemdan_summary.csv")
    return {
        "rows": rows,
        "overall": overall,
        "by_signal": by_signal,
        "by_noise": by_noise,
        "by_sigma": by_sigma,
        "by_signal_noise": by_signal_noise,
        "candidate_usage": candidate_usage,
        "exploratory_oracle_irmf_vs_ceemdan_rows": exploratory_rows,
        "exploratory_oracle_irmf_vs_ceemdan_summary": exploratory_summary,
        "figures": plot_manifest,
        "protocol": protocol,
    }


def _summarize_exploratory(rows):
    if not rows:
        return []
    groups = {(): rows}
    out = []
    for key, subset in groups.items():
        item = {
            "comparison": "IRMF_oracle_vs_CEEMDAN_oracle",
            "exploratory_only": True,
            "not_used_for_main_claims": True,
            "n_cases": int(len(subset)),
        }
        for metric in METRICS:
            vals = [
                row.get(f"irmf_oracle_advantage_over_ceemdan_oracle_{metric}")
                for row in subset
                if _finite(row.get(f"irmf_oracle_advantage_over_ceemdan_oracle_{metric}"))
            ]
            if vals:
                vals = np.asarray(vals, dtype=float)
                item[f"mean_irmf_oracle_advantage_{metric}"] = float(np.mean(vals))
                item[f"median_irmf_oracle_advantage_{metric}"] = float(np.median(vals))
                item[f"irmf_oracle_win_rate_{metric}"] = float(np.mean(vals > 0.0))
        out.append(item)
    return out


if __name__ == "__main__":
    run_irmf_oracle_adaptivity_analysis(
        "IRMF_EMD_PAPER_RESULTS/oracle_adaptivity_analysis",
        quick=True,
    )
