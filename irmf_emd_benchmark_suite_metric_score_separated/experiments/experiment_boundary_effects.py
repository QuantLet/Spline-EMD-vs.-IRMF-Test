#!/usr/bin/python
# coding: UTF-8
"""Boundary effects experiment with full diagnostic report."""
import numpy as np
from signal_bank.synthetic_signals import get_signal
from diagnostics.boundary_diagnostics import compute_boundary_diagnostics
from experiments.experiment_reporting import print_irmf_full_report, print_emd_full_report, print_compact_comparison, print_extended_comparison
from experiments.experiment_utils import ensure_dir, make_observed_signal, run_single_irmf_case, run_single_emd_case, serialize_result_summary, write_json

def make_edge_discontinuous_signal(t):
    base=get_signal('stationary_multi_sine',t); trend=1.5*t; jump=np.zeros_like(t); jump[t>0.75]=0.8
    return base+trend+jump

def attach_boundary_diagnostics(result,Y,edge_fraction=0.10):
    result.update(compute_boundary_diagnostics(Y,result['imfs'],result['residual'],edge_fraction=edge_fraction)); return result

def run_boundary_effects_experiment(output_root, sigma=0.05, n=500, fs=500.0, search_mode='quick', seed=0):
    output_root=ensure_dir(output_root); t=np.linspace(0,1,n,endpoint=False)
    X_clean=make_edge_discontinuous_signal(t); Y,noise,expected_noise_ratio=make_observed_signal(X_clean,sigma,seed=seed)
    case_root=output_root/'boundary_effects'/f'sigma_{sigma:.2f}'; case_root.mkdir(parents=True,exist_ok=True)
    summary={}; comparison_rows=[]
    for boundary_mode in ['periodic','mirror']:
        print('\n'+'#'*120); print(f'Boundary experiment | IRMF boundary={boundary_mode}'); print('#'*120)
        _,irmf_best=run_single_irmf_case(Y=Y,X_clean=X_clean,t=t,fs=fs,output_dir=case_root/f'irmf_{boundary_mode}',search_mode=search_mode,expected_noise_ratio=expected_noise_ratio,boundary_mode=boundary_mode)
        best=attach_boundary_diagnostics(irmf_best['physical_best'],Y)
        print_irmf_full_report(f'BOUNDARY EFFECTS | IRMF {boundary_mode.upper()} FULL REPORT',best)
        summary[f'IRMF_{boundary_mode}']=serialize_result_summary(best); comparison_rows.append((f'IRMF_{boundary_mode}',best))
    _,emd_best=run_single_emd_case(Y=Y,X_clean=X_clean,t=t,fs=fs,output_dir=case_root/'emd',search_mode=search_mode)
    emd=attach_boundary_diagnostics(emd_best['physical_best'],Y)
    print_emd_full_report('BOUNDARY EFFECTS | EMD FULL REPORT',emd)
    summary['EMD']=serialize_result_summary(emd); comparison_rows.append(('EMD',emd))
    print_compact_comparison('BOUNDARY EFFECTS COMPACT COMPARISON',comparison_rows)
    write_json(summary,case_root/'boundary_effects_summary.json')
    return summary


def run_boundary_effects_multisigma_experiment(
        output_root,
        sigma_levels=(0.05, 0.20),
        n=500,
        fs=500.0,
        search_mode="quick",
        seed=0
):
    """
    Multi-sigma wrapper for boundary effects benchmark.
    """
    summary = {}

    for sigma in sigma_levels:
        print("\n" + "#" * 120)
        print(f"Boundary effects multi-sigma benchmark | sigma={sigma}")
        print("#" * 120)

        summary[str(sigma)] = run_boundary_effects_experiment(
            output_root=output_root,
            sigma=sigma,
            n=n,
            fs=fs,
            search_mode=search_mode,
            seed=seed
        )

    return summary
