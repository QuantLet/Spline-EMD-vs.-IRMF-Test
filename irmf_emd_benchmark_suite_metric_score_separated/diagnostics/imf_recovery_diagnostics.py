#!/usr/bin/python
# coding: UTF-8

import numpy as np


def _corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) == 0:
        return np.nan
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) == 0:
        return np.nan
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _assignment(cost):
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(cost)
        return list(zip(rows, cols))
    except Exception:
        pairs = []
        used_rows = set()
        used_cols = set()
        while len(used_rows) < cost.shape[0] and len(used_cols) < cost.shape[1]:
            best = None
            best_val = np.inf
            for i in range(cost.shape[0]):
                if i in used_rows:
                    continue
                for j in range(cost.shape[1]):
                    if j in used_cols:
                        continue
                    if cost[i, j] < best_val:
                        best_val = cost[i, j]
                        best = (i, j)
            if best is None:
                break
            i, j = best
            pairs.append((i, j))
            used_rows.add(i)
            used_cols.add(j)
        return pairs


def compute_imf_recovery_diagnostics(imfs, true_components=None):
    if true_components is None:
        return {
            "imf_recovery_rmse": np.nan,
            "imf_recovery_corr": np.nan,
            "imf_recovery_matched_count": 0,
        }

    imfs = np.asarray(imfs, dtype=float)
    true_components = np.asarray(true_components, dtype=float)

    if imfs.ndim != 2 or true_components.ndim != 2:
        return {
            "imf_recovery_rmse": np.nan,
            "imf_recovery_corr": np.nan,
            "imf_recovery_matched_count": 0,
        }

    if imfs.shape[1] != true_components.shape[1] or imfs.shape[0] == 0 or true_components.shape[0] == 0:
        return {
            "imf_recovery_rmse": np.nan,
            "imf_recovery_corr": np.nan,
            "imf_recovery_matched_count": 0,
        }

    K, M = imfs.shape[0], true_components.shape[0]
    cost = np.ones((K, M), dtype=float)

    for i in range(K):
        for j in range(M):
            c = _corr(imfs[i], true_components[j])
            if np.isfinite(c):
                cost[i, j] = 1.0 - abs(c)

    pairs = _assignment(cost)
    rmses = []
    corrs = []

    for i, j in pairs:
        rmses.append(_rmse(imfs[i], true_components[j]))
        c = _corr(imfs[i], true_components[j])
        if np.isfinite(c):
            corrs.append(abs(c))

    return {
        "imf_recovery_rmse": float(np.nanmean(rmses)) if rmses else np.nan,
        "imf_recovery_corr": float(np.nanmean(corrs)) if corrs else np.nan,
        "imf_recovery_matched_count": int(len(pairs)),
    }
