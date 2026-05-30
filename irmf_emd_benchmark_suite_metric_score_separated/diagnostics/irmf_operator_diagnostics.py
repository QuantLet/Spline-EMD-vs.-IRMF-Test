#!/usr/bin/python
# coding: UTF-8

"""
IRMF operator-inspired diagnostics.

Layer 2: engineering diagnostics inspired by operator propagation.
These are not strict Spokoiny PDF quantities.

Contains:
- trace evolution
- operator norm proxy
- contraction ratio
- noise amplification
"""

import numpy as np
from core_algorithms.strict_spokoiny_irmf import spokoiny_kernel, rho_spline_grad_hess


def indexed_boundary_extension(Y, t, boundary_mode='periodic'):
    n = len(Y)
    if boundary_mode == 'periodic':
        return (
            np.concatenate([Y, Y, Y]),
            np.concatenate([t - 1.0, t, t + 1.0]),
            np.concatenate([np.arange(n), np.arange(n), np.arange(n)]),
        )
    if boundary_mode == 'mirror':
        idx = np.arange(n)
        return (
            np.concatenate([Y[::-1], Y, Y[::-1]]),
            np.concatenate([-t[::-1], t, 2.0 - t[::-1]]),
            np.concatenate([idx[::-1], idx, idx[::-1]]),
        )
    raise ValueError("boundary_mode must be 'periodic' or 'mirror'.")


def robust_influence_apply(Y_before, S_k, h, H, V, boundary_mode='periodic'):
    n = len(Y_before)
    t = np.linspace(0, 1, n, endpoint=False)
    Y_ext, t_ext, idx_ext = indexed_boundary_extension(Y_before, t, boundary_mode=boundary_mode)
    WV = np.zeros_like(V)
    for i in range(n):
        u = (t[i] - t_ext) / h
        kh = spokoiny_kernel(u)
        valid = kh > 0
        if not np.any(valid):
            continue
        y_local = Y_ext[valid]
        idx_local = idx_ext[valid]
        residual = y_local - S_k[i]
        _, _, hess_vals = rho_spline_grad_hess(residual, H)
        weights = kh[valid] * hess_vals
        denom = np.sum(weights) + 1e-10
        WV[i, :] = (weights / denom) @ V[idx_local, :]
    return WV


def compute_irmf_operator_diagnostics(Y, scale_history, H, trace_probe_count=16, random_seed=123):
    n = len(Y)
    rng = np.random.default_rng(random_seed)
    Z0 = rng.choice([-1.0, 1.0], size=(n, trace_probe_count))
    AZ = Z0.copy()
    prev_trace = None
    diagnostics = {
        'bandwidths': [],
        'noise_traces': [],
        'noise_trace_norms': [],
        'operator_norms': [],
        'contraction_ratios': [],
        'noise_amplification': [],
        'trace_estimator': 'hutchinson_rademacher',
        'trace_probe_count': trace_probe_count,
    }
    for state in scale_history:
        h = state['h']
        Y_before = state['Y_before']
        S_k = state['S_k']
        boundary_mode = state.get('boundary_mode', 'periodic')
        W_AZ = robust_influence_apply(Y_before, S_k, h, H, AZ, boundary_mode=boundary_mode)
        AZ = AZ - W_AZ
        probe_norms_sq = np.sum(AZ ** 2, axis=0)
        noise_trace = float(np.mean(probe_norms_sq))
        noise_trace_norm = noise_trace / n
        contraction_ratio = 1.0 if prev_trace is None else noise_trace / (prev_trace + 1e-10)
        prev_trace = noise_trace
        z_norms = np.sqrt(np.sum(Z0 ** 2, axis=0)) + 1e-10
        az_norms = np.sqrt(np.sum(AZ ** 2, axis=0))
        operator_norm_proxy = float(np.max(az_norms / z_norms))
        diagnostics['bandwidths'].append(h)
        diagnostics['noise_traces'].append(noise_trace)
        diagnostics['noise_trace_norms'].append(noise_trace_norm)
        diagnostics['operator_norms'].append(operator_norm_proxy)
        diagnostics['contraction_ratios'].append(float(contraction_ratio))
        diagnostics['noise_amplification'].append(float(noise_trace_norm))
    return diagnostics


def summarize_irmf_operator_score(operator_diagnostics):
    final_trace = float(operator_diagnostics['noise_traces'][-1])
    final_trace_norm = float(operator_diagnostics['noise_trace_norms'][-1])
    final_operator_norm = float(operator_diagnostics['operator_norms'][-1])
    final_contraction = float(operator_diagnostics['contraction_ratios'][-1])
    operator_score = (
        0.40 * final_trace_norm
        + 0.25 * abs(1.0 - final_contraction)
        + 0.25 * (final_operator_norm / (final_operator_norm + 1.0))
        + 0.10 * max(0.0, final_contraction - 1.0)
    )
    return {
        'final_trace': final_trace,
        'final_trace_norm': final_trace_norm,
        'final_operator_norm': final_operator_norm,
        'final_contraction': final_contraction,
        'operator_score': float(operator_score),
    }
