#!/usr/bin/python
# coding: UTF-8

"""
IRMF local theory diagnostics.

Layer 1: strict paper-style local quantities closest to the
Spokoiny Fisher/Wilks perturbation framework.

Contains:
- local Fisher metric / Hessian positivity
- local perturbation radius b0_i
- tau3 * b0 condition
- Wilks/Fisher remainder scale proxy

Does NOT contain operator trace / operator norm / contraction ratio.
"""

import numpy as np


def hessian_stability(hess_vec):
    h = np.array(hess_vec, dtype=float)
    h = h[np.isfinite(h)]
    if len(h) == 0:
        return {
            'hessian_min': 0.0,
            'hessian_median': 0.0,
            'hessian_mean': 0.0,
            'hessian_positive_ratio': 0.0,
            'hessian_condition_proxy': 999.0,
        }
    h_min = float(np.min(h))
    h_median = float(np.median(h))
    h_mean = float(np.mean(h))
    h_max = float(np.max(h))
    return {
        'hessian_min': h_min,
        'hessian_median': h_median,
        'hessian_mean': h_mean,
        'hessian_positive_ratio': float(np.mean(h > 1e-10)),
        'hessian_condition_proxy': float(h_max / (h_min + 1e-10)),
    }


def local_b0_statistics(grad_vec, hess_vec):
    grad_vec = np.asarray(grad_vec, dtype=float)
    hess_vec = np.asarray(hess_vec, dtype=float)
    safe_hess = np.maximum(hess_vec, 1e-10)
    b0_local = np.abs(grad_vec) / np.sqrt(safe_hess)
    return {
        'b0_local': b0_local,
        'b0_local_max': float(np.max(b0_local)),
        'b0_local_median': float(np.median(b0_local)),
        'b0_local_mean': float(np.mean(b0_local)),
        'b0_local_q90': float(np.quantile(b0_local, 0.90)),
    }


def tau3_condition_statistics(b0_local, h, n, c3=0.2):
    effective_n = max(float(n) * float(h), 1.0)
    tau3 = c3 / np.sqrt(effective_n)
    condition = tau3 * np.asarray(b0_local)
    return {
        'tau3': float(tau3),
        'tau3_b0_max': float(np.max(condition)),
        'tau3_b0_median': float(np.median(condition)),
        'tau3_b0_mean': float(np.mean(condition)),
        'tau3_b0_valid_ratio': float(np.mean(condition < (4.0 / 9.0))),
    }


def wilks_remainder_scale_statistics(b0_local, h, n, c3=0.2):
    effective_n = max(float(n) * float(h), 1.0)
    tau3 = c3 / np.sqrt(effective_n)
    remainder = tau3 * (np.asarray(b0_local) ** 3)
    return {
        'wilks_remainder_scale_max': float(np.max(remainder)),
        'wilks_remainder_scale_median': float(np.median(remainder)),
        'wilks_remainder_scale_mean': float(np.mean(remainder)),
    }


def compute_irmf_local_theory_diagnostics(Y, scale_history, H, c3=0.2):
    n = len(Y)
    initial_energy = np.sum(Y ** 2) + 1e-10
    diagnostics = {
        'bandwidths': [],
        'b0_local_max': [],
        'b0_local_median': [],
        'b0_local_mean': [],
        'b0_local_q90': [],
        'b0_like': [],
        'tau3': [],
        'tau3_b0_max': [],
        'tau3_b0_median': [],
        'tau3_b0_mean': [],
        'tau3_b0_valid_ratio': [],
        'wilks_remainder_scale_max': [],
        'wilks_remainder_scale_median': [],
        'wilks_remainder_scale_mean': [],
        'mean_hessian': [],
        'min_hessian': [],
        'median_hessian': [],
        'hessian_positive_ratio': [],
        'hessian_condition_proxy': [],
        'mean_abs_grad': [],
        'residual_energy': [],
        'residual_energy_ratio': [],
    }
    for state in scale_history:
        h = state['h']
        Y_after = state['Y_after']
        grad_vec = state['grad_vec']
        hess_vec = state['hess_vec']
        b0_stats = local_b0_statistics(grad_vec, hess_vec)
        tau_stats = tau3_condition_statistics(b0_stats['b0_local'], h, n, c3=c3)
        wilks_stats = wilks_remainder_scale_statistics(b0_stats['b0_local'], h, n, c3=c3)
        hess_stats = hessian_stability(hess_vec)
        res_energy = float(np.sum(Y_after ** 2))
        diagnostics['bandwidths'].append(h)
        diagnostics['b0_local_max'].append(b0_stats['b0_local_max'])
        diagnostics['b0_local_median'].append(b0_stats['b0_local_median'])
        diagnostics['b0_local_mean'].append(b0_stats['b0_local_mean'])
        diagnostics['b0_local_q90'].append(b0_stats['b0_local_q90'])
        diagnostics['b0_like'].append(b0_stats['b0_local_median'])
        diagnostics['tau3'].append(tau_stats['tau3'])
        diagnostics['tau3_b0_max'].append(tau_stats['tau3_b0_max'])
        diagnostics['tau3_b0_median'].append(tau_stats['tau3_b0_median'])
        diagnostics['tau3_b0_mean'].append(tau_stats['tau3_b0_mean'])
        diagnostics['tau3_b0_valid_ratio'].append(tau_stats['tau3_b0_valid_ratio'])
        diagnostics['wilks_remainder_scale_max'].append(wilks_stats['wilks_remainder_scale_max'])
        diagnostics['wilks_remainder_scale_median'].append(wilks_stats['wilks_remainder_scale_median'])
        diagnostics['wilks_remainder_scale_mean'].append(wilks_stats['wilks_remainder_scale_mean'])
        diagnostics['mean_hessian'].append(hess_stats['hessian_mean'])
        diagnostics['min_hessian'].append(hess_stats['hessian_min'])
        diagnostics['median_hessian'].append(hess_stats['hessian_median'])
        diagnostics['hessian_positive_ratio'].append(hess_stats['hessian_positive_ratio'])
        diagnostics['hessian_condition_proxy'].append(hess_stats['hessian_condition_proxy'])
        diagnostics['mean_abs_grad'].append(float(np.mean(np.abs(grad_vec))))
        diagnostics['residual_energy'].append(res_energy)
        diagnostics['residual_energy_ratio'].append(float(res_energy / initial_energy))
    return diagnostics


def summarize_irmf_local_theory_score(local_diagnostics, expected_noise_ratio=None):
    median_b0 = float(np.median(local_diagnostics['b0_local_median']))
    mean_b0 = float(np.mean(local_diagnostics['b0_local_mean']))
    max_b0 = float(np.max(local_diagnostics['b0_local_max']))
    final_tau3_b0_median = float(local_diagnostics['tau3_b0_median'][-1])
    final_tau3_b0_max = float(local_diagnostics['tau3_b0_max'][-1])
    final_tau3_valid_ratio = float(local_diagnostics['tau3_b0_valid_ratio'][-1])
    final_wilks_remainder_median = float(local_diagnostics['wilks_remainder_scale_median'][-1])
    final_wilks_remainder_max = float(local_diagnostics['wilks_remainder_scale_max'][-1])
    final_residual_energy_ratio = float(local_diagnostics['residual_energy_ratio'][-1])
    final_hessian_condition = float(local_diagnostics['hessian_condition_proxy'][-1])
    final_hessian_positive_ratio = float(local_diagnostics['hessian_positive_ratio'][-1])
    if expected_noise_ratio is None:
        residual_energy_penalty = final_residual_energy_ratio
    else:
        residual_energy_penalty = abs(final_residual_energy_ratio - expected_noise_ratio)
    local_theory_score = (
        0.25 * (median_b0 / (median_b0 + 1.0))
        + 0.20 * min(final_tau3_b0_median / (4.0 / 9.0), 10.0)
        + 0.15 * (final_wilks_remainder_median / (final_wilks_remainder_median + 1.0))
        + 0.15 * residual_energy_penalty
        + 0.15 * (final_hessian_condition / (final_hessian_condition + 1.0))
        + 0.10 * (1.0 - final_hessian_positive_ratio)
    )
    return {
        'mean_b0': mean_b0,
        'median_b0': median_b0,
        'max_b0': max_b0,
        'final_tau3_b0_median': final_tau3_b0_median,
        'final_tau3_b0_max': final_tau3_b0_max,
        'final_tau3_valid_ratio': final_tau3_valid_ratio,
        'final_wilks_remainder_median': final_wilks_remainder_median,
        'final_wilks_remainder_max': final_wilks_remainder_max,
        'final_residual_energy_ratio': final_residual_energy_ratio,
        'residual_energy_penalty': residual_energy_penalty,
        'final_hessian_condition': final_hessian_condition,
        'final_hessian_positive_ratio': final_hessian_positive_ratio,
        'local_theory_score': float(local_theory_score),
    }
