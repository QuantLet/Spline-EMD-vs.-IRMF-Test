#!/usr/bin/python
# coding: UTF-8

"""Challenging signal-regime benchmark for the algorithm layer.

This experiment is deliberately separate from the canonical 168-case main
benchmark.  It asks whether conclusions remain stable on harder synthetic
signals: crossing chirps, time-varying close frequencies, trends, damping,
weak components, harmonic-rich periodicity, and irregular transient trains.
"""

from time import perf_counter

from project_config import (
    CHALLENGING_SIGNAL_FAMILY,
    CHALLENGING_SIGNAL_NOISES,
    CHALLENGING_SIGNAL_QUICK_FAMILY,
    CHALLENGING_SIGNAL_QUICK_NOISES,
    CHALLENGING_SIGNAL_QUICK_SIGMAS,
    CHALLENGING_SIGNAL_SIGMAS,
    DEFAULT_FS,
    DEFAULT_N,
    DEFAULT_SEED,
    GLOBAL_CEEMDAN_PARAMS,
    GLOBAL_EEMD_PARAMS,
    GLOBAL_EMD_PARAMS,
    GLOBAL_IRMF_PARAMS,
)
from experiments.experiment_emd_family_benchmark import (
    _family_aggregate,
    _run_locked_emd_family_method,
)
from experiments.experiment_utils import (
    make_signal_noise_case,
    method_result_summary,
    run_fixed_emd_case,
    run_fixed_irmf_case,
)
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


def run_challenging_signal_suite(
        output_root,
        irmf_params=GLOBAL_IRMF_PARAMS,
        emd_params=GLOBAL_EMD_PARAMS,
        eemd_params=GLOBAL_EEMD_PARAMS,
        ceemdan_params=GLOBAL_CEEMDAN_PARAMS,
        signals=CHALLENGING_SIGNAL_FAMILY,
        noises=CHALLENGING_SIGNAL_NOISES,
        sigmas=CHALLENGING_SIGNAL_SIGMAS,
        n=DEFAULT_N,
        fs=DEFAULT_FS,
        data_seed=DEFAULT_SEED,
        algorithm_seed=20260715,
        quick=False,
        return_rows=False,
):
    """Run fixed IRMF/EMD/EEMD/CEEMDAN on harder synthetic signal regimes."""
    output_root = ensure_dir(output_root)
    if quick:
        signals = CHALLENGING_SIGNAL_QUICK_FAMILY
        noises = CHALLENGING_SIGNAL_QUICK_NOISES
        sigmas = CHALLENGING_SIGNAL_QUICK_SIGMAS

    rows = []
    for signal_name in signals:
        for noise_name in noises:
            for sigma in sigmas:
                case = make_signal_noise_case(
                    signal_name=signal_name,
                    noise_name=noise_name,
                    sigma=sigma,
                    n=n,
                    fs=fs,
                    seed=data_seed,
                )
                true_components = case.get("true_components", None)
                row = {
                    "section": "challenging_signal_generalization_benchmark",
                    "benchmark_tier": "Tier 3",
                    "benchmark_role": "out_of_development_structural_generalization",
                    "signal": signal_name,
                    "noise": noise_name,
                    "sigma": float(sigma),
                    "data_seed": int(data_seed),
                    "algorithm_seed": int(algorithm_seed),
                    "quick": bool(quick),
                }

                start = perf_counter()
                irmf = run_fixed_irmf_case(
                    Y=case["Y"],
                    X_clean=case["X_clean"],
                    t=case["t"],
                    fs=fs,
                    irmf_params=irmf_params,
                    expected_noise_ratio=case.get("expected_noise_ratio"),
                    true_components=true_components,
                    run_id=f"challenging_{signal_name}_{noise_name}_{sigma}_irmf",
                )
                irmf["runtime_seconds"] = perf_counter() - start
                row["IRMF"] = method_result_summary(irmf)

                start = perf_counter()
                emd = run_fixed_emd_case(
                    Y=case["Y"],
                    X_clean=case["X_clean"],
                    t=case["t"],
                    fs=fs,
                    emd_params=emd_params,
                    true_components=true_components,
                    run_id=f"challenging_{signal_name}_{noise_name}_{sigma}_emd",
                )
                emd["runtime_seconds"] = perf_counter() - start
                row["EMD"] = method_result_summary(emd)

                for method_name in ("EEMD", "CEEMDAN"):
                    try:
                        result = _run_locked_emd_family_method(
                            method_name,
                            case,
                            fs=fs,
                            emd_params=emd_params,
                            eemd_params=eemd_params,
                            ceemdan_params=ceemdan_params,
                            algorithm_seed=algorithm_seed,
                        )
                        result["run_id"] = f"challenging_{signal_name}_{noise_name}_{sigma}_{method_name.lower()}"
                        row[method_name] = method_result_summary(result)
                    except Exception as exc:
                        row[method_name] = {"method": method_name, "error": str(exc)}
                rows.append(row)
                print(
                    "CHALLENGING SIGNAL DONE | "
                    f"signal={signal_name} | noise={noise_name} | sigma={sigma}",
                    flush=True,
                )

    summary = _family_aggregate(rows)
    protocol = {
        "section": "Challenging Signal Generalization Benchmark",
        "benchmark_tier": "Tier 3",
        "benchmark_role": "out-of-development structural generalization benchmark",
        "purpose": (
            "Evaluate whether fixed-parameter IRMF conclusions generalize from "
            "canonical held-out formula instances to structurally more difficult "
            "and previously unseen signal regimes."
        ),
        "design": {
            "signals": list(signals),
            "noises": list(noises),
            "sigmas": list(sigmas),
            "n_cases": int(len(rows)),
            "methods": ["IRMF", "EMD", "EEMD", "CEEMDAN"],
        },
        "matched_noise_design": (
            "The challenging benchmark uses the same noise models and sigma "
            "levels as the canonical benchmark when quick=False, so the main "
            "design contrast is canonical versus challenging signal structure."
        ),
        "target_estimand": (
            "Fixed-parameter performance under unseen structural regimes, not "
            "parameter selection and not case-specific oracle tuning."
        ),
        "irmf_params": dict(irmf_params),
        "emd_params": dict(emd_params),
        "eemd_params": dict(eemd_params),
        "ceemdan_params": dict(ceemdan_params),
        "data_seed": int(data_seed),
        "algorithm_seed": int(algorithm_seed),
        "quick": bool(quick),
        "note": (
            "This experiment is not used for parameter selection and does not "
            "alter the 168-case canonical main benchmark.  It should be interpreted "
            "as a structural generalization benchmark."
        ),
    }
    write_json(protocol, output_root / "challenging_signal_suite_protocol.json")
    write_json(rows, output_root / "challenging_signal_suite_rows.json")
    write_csv(rows, output_root / "challenging_signal_suite_rows.csv")
    write_json(summary, output_root / "challenging_signal_suite_aggregate.json")
    if return_rows:
        return rows, summary
    return summary


if __name__ == "__main__":
    run_challenging_signal_suite("IRMF_EMD_PAPER_RESULTS/challenging_signal_suite", quick=True)
