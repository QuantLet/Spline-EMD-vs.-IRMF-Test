#!/usr/bin/python
# coding: UTF-8

"""
EMD sensitivity analysis.

EMD does not have theoretical parameter search like IRMF.
This scans implementation settings only.
"""

import itertools

from core_algorithms.emd_core import run_emd_decomposition
from diagnostics.shared_physical_diagnostics import evaluate_shared_physical_diagnostics
from diagnostics.emd_specific_diagnostics import compute_emd_specific_diagnostics


def evaluate_emd_run(
        run_id,
        Y,
        T,
        X_clean,
        fs,
        nbsym,
        spline_kind,
        max_imf,
        std_thr=None,
        svar_thr=None,
        total_power_thr=None,
        range_thr=None,
        true_components=None
):
    imfs, residual, all_components, config = run_emd_decomposition(
        Y=Y,
        T=T,
        nbsym=nbsym,
        spline_kind=spline_kind,
        max_imf=max_imf,
        std_thr=std_thr,
        svar_thr=svar_thr,
        total_power_thr=total_power_thr,
        range_thr=range_thr
    )

    physical = evaluate_shared_physical_diagnostics(
        Y_observed=Y,
        X_clean=X_clean,
        imfs=imfs,
        residual=residual,
        fs=fs,
        residual_penalty_mode="none",
        true_components=true_components
    )

    emd_specific = compute_emd_specific_diagnostics(residual, fs)

    out = {
        "method": "EMD",
        "run_id": run_id,
        "nbsym": nbsym,
        "spline_kind": spline_kind,
        "max_imf": max_imf,
        "std_thr": std_thr,
        "svar_thr": svar_thr,
        "total_power_thr": total_power_thr,
        "range_thr": range_thr,
        "config": config,
        "imfs": imfs,
        "residual": residual,
        "all_components": all_components,
        "theory_diagnostic_score": None,
    }

    out.update(physical)
    out.update(emd_specific)

    return out


def pareto_front(results, keys):
    front = []

    for i, r in enumerate(results):
        dominated = False

        for j, q in enumerate(results):
            if i == j:
                continue

            better_or_equal_all = all(q[k] <= r[k] for k in keys)
            strictly_better_one = any(q[k] < r[k] for k in keys)

            if better_or_equal_all and strictly_better_one:
                dominated = True
                break

        if not dominated:
            front.append(r)

    return front


def run_emd_sensitivity_analysis(
        Y,
        T,
        X_clean,
        fs,
        nbsym_options,
        spline_kind_options,
        max_imf_options,
        std_thr_options=(None,),
        svar_thr_options=(None,),
        total_power_thr_options=(None,),
        range_thr_options=(None,),
        true_components=None
):
    grid = list(itertools.product(
        nbsym_options,
        spline_kind_options,
        max_imf_options,
        std_thr_options,
        svar_thr_options,
        total_power_thr_options,
        range_thr_options
    ))

    results = []

    print(f"Total EMD runs: {len(grid)}")

    for run_id, (
        nbsym,
        spline_kind,
        max_imf,
        std_thr,
        svar_thr,
        total_power_thr,
        range_thr
    ) in enumerate(grid, start=1):

        print("=" * 90)
        print(
            f"EMD Run {run_id}/{len(grid)} | "
            f"nbsym={nbsym}, spline={spline_kind}, max_imf={max_imf}, "
            f"std_thr={std_thr}, svar_thr={svar_thr}, "
            f"total_power_thr={total_power_thr}, range_thr={range_thr}"
        )

        r = evaluate_emd_run(
            run_id=run_id,
            Y=Y,
            T=T,
            X_clean=X_clean,
            fs=fs,
            nbsym=nbsym,
            spline_kind=spline_kind,
            max_imf=max_imf,
            std_thr=std_thr,
            svar_thr=svar_thr,
            total_power_thr=total_power_thr,
            range_thr=range_thr,
            true_components=true_components
        )

        results.append(r)

        print(f"IMF count              : {r['imf_count']}")
        print(f"General physical score : {r['general_physical_score']:.6f}")
        print(f"Strict IO              : {r['strict_io']:.6f}")
        print(f"Spectral leakage       : {r['spectral_leakage']:.6f}")
        print(f"IFS                    : {r['ifs']:.6f}")
        print("Center freqs           :", [round(x, 3) for x in r["center_freqs"]])

    return results
