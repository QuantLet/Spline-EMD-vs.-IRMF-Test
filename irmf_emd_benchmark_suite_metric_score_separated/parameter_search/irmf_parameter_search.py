#!/usr/bin/python
# coding: UTF-8

"""
IRMF parameter search.

IRMF has true algorithm-defining parameters:
h1, a, h_min, H.
"""

import itertools
import numpy as np

from core_algorithms.strict_spokoiny_irmf import strict_spokoiny_irmf
from diagnostics.shared_physical_diagnostics import evaluate_shared_physical_diagnostics
from diagnostics.irmf_local_theory_diagnostics import (
    compute_irmf_local_theory_diagnostics,
    summarize_irmf_local_theory_score
)
from diagnostics.irmf_operator_evolution import compute_operator_evolution, summarize_operator_evolution
from diagnostics.irmf_operator_diagnostics import (
    compute_irmf_operator_diagnostics,
    summarize_irmf_operator_score
)


from diagnostics.layered_scores import attach_irmf_performance_score

def evaluate_irmf_run(
        run_id,
        Y,
        X_clean,
        fs,
        h1,
        a,
        h_min,
        H,
        T=None,
        expected_noise_ratio=None,
        boundary_mode="periodic",
        min_support_points=3,
        true_components=None
):
    imfs, residual, residual_history, scale_history = strict_spokoiny_irmf(
        Y=Y,
        T=T,
        h1=h1,
        a=a,
        h_min=h_min,
        H=H,
        verbose=False,
        boundary_mode=boundary_mode,
        min_support_points=min_support_points
    )

    physical = evaluate_shared_physical_diagnostics(
        Y_observed=Y,
        X_clean=X_clean,
        imfs=imfs,
        residual=residual,
        fs=fs,
        residual_penalty_mode="whiteness",
        true_components=true_components
    )

    local_theory_diag = compute_irmf_local_theory_diagnostics(Y, scale_history, H)
    local_theory_summary = summarize_irmf_local_theory_score(
        local_theory_diag,
        expected_noise_ratio=expected_noise_ratio
    )

    operator_diag = compute_irmf_operator_diagnostics(Y, scale_history, H)
    operator_summary = summarize_irmf_operator_score(operator_diag)

    operator_evolution_rows = compute_operator_evolution(scale_history)
    operator_evolution_summary = summarize_operator_evolution(operator_evolution_rows)

    theory_summary = {}
    theory_summary.update(local_theory_summary)
    theory_summary.update(operator_summary)
    theory_summary['theory_diagnostic_score'] = (
        0.65 * local_theory_summary['local_theory_score']
        + 0.35 * operator_summary['operator_score']
    )

    out = {
        "method": "IRMF",
        "run_id": run_id,
        "h1": h1,
        "a": a,
        "h_min": h_min,
        "H": H,
        "boundary_mode": boundary_mode,
        "min_support_points": min_support_points,
        "imfs": imfs,
        "residual": residual,
        "scale_history": scale_history,
        "local_theory_diagnostics": local_theory_diag,
        "operator_diagnostics": operator_diag,
        "operator_evolution": operator_evolution_rows,
    }

    out.update(physical)
    out.update(theory_summary)
    out.update(operator_evolution_summary)
    out = attach_irmf_performance_score(out)

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


def run_irmf_parameter_search(
        Y,
        X_clean,
        fs,
        h1_options,
        a_options,
        h_min_options,
        H_options,
        T=None,
        expected_noise_ratio=None,
        boundary_mode="periodic",
        min_support_points=3,
        true_components=None
):
    grid = list(itertools.product(h1_options, a_options, h_min_options, H_options))
    results = []

    print(f"Total IRMF runs: {len(grid)}")

    for run_id, (h1, a, h_min, H) in enumerate(grid, start=1):
        print("=" * 90)
        print(
            f"IRMF Run {run_id}/{len(grid)} | "
            f"h1={h1:.3f}, a={a:.4f}, h_min={h_min:.4f}, H={H:.3f}, "
            f"boundary={boundary_mode}"
        )

        r = evaluate_irmf_run(
            run_id=run_id,
            Y=Y,
            X_clean=X_clean,
            fs=fs,
            h1=h1,
            a=a,
            h_min=h_min,
            H=H,
            T=T,
            expected_noise_ratio=expected_noise_ratio,
            boundary_mode=boundary_mode,
            min_support_points=min_support_points,
            true_components=true_components
        )

        results.append(r)

        print(f"IMF count              : {r['imf_count']}")
        print(f"General physical score : {r['general_physical_score']:.6f}")
        print(f"Theory score           : {r['theory_diagnostic_score']:.6f}")
        print(f"Strict IO              : {r['strict_io']:.6f}")
        print(f"Spectral leakage       : {r['spectral_leakage']:.6f}")
        print(f"IFS                    : {r['ifs']:.6f}")
        print("Center freqs           :", [round(x, 3) for x in r["center_freqs"]])

    return results
