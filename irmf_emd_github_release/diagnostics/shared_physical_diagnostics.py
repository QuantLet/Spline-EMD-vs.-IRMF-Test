#!/usr/bin/python
# coding: UTF-8

"""
Shared diagnostics for IRMF and EMD-family decompositions.

FIX10 paper-final metrics framework.

Case-level evaluation:
    1. Reconstruction
       - denoise_corr_score
       - nmse_score
       - spectral_corr_score

    2. Structural Fidelity
       - IMF Recovery
       - Orthogonality & Leakage
       - Frequency Separation
       - Local Structure Preservation
       - OD/UD hard penalty

    3. Contamination Resistance
       - Outlier Resistance Index (ORI)
       - Noise Capture Correlation

Important convention:
    - Raw metrics keep their physical meaning and units.
    - *_score fields are normalized to [0, 1], where 1 = good.
    - general_physical_score is kept backward-compatible as a minimization objective:
          general_physical_score = 1 - case_score_final
      Therefore existing "ascending best run" logic still works.
"""

import numpy as np
from scipy.signal import hilbert, medfilt

from project_config import (
    EVALUATION_FRAMEWORK_FROZEN_DATE,
    EVALUATION_FRAMEWORK_STATUS,
    EVALUATION_FRAMEWORK_VERSION,
)


# ============================================================
# Basic utilities
# ============================================================

EPS = 1e-12
NOISE_ENERGY_EPS = EPS
NOISE_ENERGY_BIAS_TOLERANCE = 0.05
METRIC_DEFINITION_VERSION = "V5.24"
STRUCTURAL_METRIC_SCHEMA_VERSION = "TRUE_COMPONENT_MIXING_V1.0"
SYNTHETIC_RECONSTRUCTION_PROTOCOL_ID = "SYNTHETIC_COMPONENT_SELECTION_AND_RECONSTRUCTION_V1.0"
SYNTHETIC_RECONSTRUCTION_ASSOCIATION_THRESHOLD = 0.05
LEGACY_METRIC_ALIAS_STATUS = {
    "mode_mixing_index": "backward_compatible_alias_of_inter_imf_entanglement_index",
}


def _as_2d(imfs):
    imfs = np.asarray(imfs, dtype=float)
    if imfs.ndim == 1:
        imfs = imfs[None, :]
    if imfs.size == 0:
        return np.empty((0, 0), dtype=float)
    return imfs


def _safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) != len(b) or len(a) == 0:
        return np.nan

    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def _corr_score(a, b, absolute=False):
    c = _safe_corr(a, b)
    if not np.isfinite(c):
        return np.nan
    if absolute:
        return float(np.clip(abs(c), 0.0, 1.0))
    return float(np.clip((c + 1.0) / 2.0, 0.0, 1.0))


def _safe_psnr(x_true, x_hat):
    """
    Peak signal-to-noise ratio in dB.
    Raw diagnostic only; not used directly in CaseScore.
    """
    x_true = np.asarray(x_true, dtype=float)
    x_hat = np.asarray(x_hat, dtype=float)

    mse = np.mean((x_hat - x_true) ** 2)
    if mse <= 1e-20:
        return float("inf")

    peak = np.max(np.abs(x_true)) + 1e-12
    return float(20.0 * np.log10(peak / np.sqrt(mse)))


def _safe_db(x):
    return 10.0 * np.log10(max(float(x), 1e-20))


def _nanmean(values, default=np.nan):
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    return float(np.mean(arr)) if len(arr) else default


def _clip01(x, default=np.nan):
    try:
        x = float(x)
        if not np.isfinite(x):
            return default
        return float(np.clip(x, 0.0, 1.0))
    except Exception:
        return default


def _positive_to_score(x):
    """
    Convert nonnegative loss-like value to [0,1] score.
    0 -> 1, larger values -> smaller score.
    """
    try:
        x = float(x)
        if not np.isfinite(x):
            return np.nan
        return float(np.exp(-max(x, 0.0)))
    except Exception:
        return np.nan


def _loss_to_unit_score(x):
    """
    Convert approximately [0,1] loss into score.
    """
    try:
        x = float(x)
        if not np.isfinite(x):
            return np.nan
        return float(np.clip(1.0 - x, 0.0, 1.0))
    except Exception:
        return np.nan


def reconstructed_signal(Y_observed, imfs, residual):
    """
    Clean-signal proxy used for denoising/recovery metrics.

    For Y = signal + noise and a decomposition Y = sum(IMFs) + residual,
    the denoised reconstruction is interpreted as Y - residual.
    """
    try:
        return np.asarray(Y_observed, dtype=float) - np.asarray(residual, dtype=float)
    except Exception:
        imfs = _as_2d(imfs)
        if imfs.size == 0:
            return None
        return np.sum(imfs, axis=0)


def protocol_component_reconstruction(
        Y_observed,
        imfs,
        residual,
        true_components=None,
        association_threshold=SYNTHETIC_RECONSTRUCTION_ASSOCIATION_THRESHOLD,
):
    """Truth-assisted synthetic reconstruction under the frozen V5.24 amendment.

    Truth is used only to decide whether a complete estimated component is
    eligible for retention.  Retained components are not projected, rescaled,
    sign-corrected, or otherwise purified.
    """
    Y_observed = np.asarray(Y_observed, dtype=float)
    imfs = _as_2d(imfs)
    candidates = imfs
    residual_candidate_included = False
    try:
        r = np.asarray(residual, dtype=float)
        if r.shape == Y_observed.shape:
            candidates = np.vstack([candidates, r[None, :]]) if candidates.size else r[None, :]
            residual_candidate_included = True
    except Exception:
        pass

    out = {
        "reconstruction_protocol_id": SYNTHETIC_RECONSTRUCTION_PROTOCOL_ID,
        "reconstruction_protocol_status": "specification_frozen",
        "reconstruction_protocol_truth_assisted": True,
        "reconstruction_protocol_association_threshold": float(association_threshold),
        "reconstruction_protocol_threshold_operator": ">",
        "reconstruction_protocol_waveform_modification": "none",
        "protocol_component_candidate_count": int(candidates.shape[0]) if candidates.size else 0,
        "protocol_imf_component_count": int(imfs.shape[0]) if imfs.size else 0,
        "protocol_residual_or_trend_bearing_candidate_included": bool(residual_candidate_included),
        "protocol_selected_component_indices": "",
        "protocol_selected_component_count": 0,
        "protocol_selected_fraction": 0.0,
        "protocol_selection_status_code": "not_computable_missing_true_components",
        "protocol_zero_variance_noise_flag": True,
        "protocol_noise_reason_code": "not_computable_missing_true_components",
        "protocol_estimated_signal_energy": np.nan,
        "protocol_estimated_noise_energy": np.nan,
        "protocol_reconstruction_closure_max_abs_error": np.nan,
    }
    if true_components is None or candidates.size == 0:
        return None, None, out
    true_components = _as_2d(true_components)
    if true_components.size == 0 or true_components.shape[1] != Y_observed.size:
        return None, None, out

    assoc = np.zeros((candidates.shape[0], true_components.shape[0]), dtype=float)
    for i in range(candidates.shape[0]):
        for j in range(true_components.shape[0]):
            c = _safe_corr(candidates[i], true_components[j])
            assoc[i, j] = 0.0 if not np.isfinite(c) else abs(float(c)) ** 2
    selected = np.max(assoc, axis=1) > float(association_threshold)
    selected_indices = [int(i) for i, flag in enumerate(selected) if flag]
    if selected_indices:
        rec = np.sum(candidates[selected], axis=0)
    else:
        rec = np.zeros_like(Y_observed)
    noise = Y_observed - rec
    if not selected_indices:
        status = "valid_complete_signal_rejection"
    elif len(selected_indices) == candidates.shape[0]:
        status = "all_components_selected"
    else:
        status = "partial_selection"
    zero_var = bool(np.std(noise) < EPS)
    reason = "not_computable_zero_variance_protocol_noise" if zero_var else "computable_protocol_noise"
    out.update({
        "protocol_selected_component_indices": ";".join(str(i) for i in selected_indices),
        "protocol_selected_component_count": int(len(selected_indices)),
        "protocol_selected_fraction": float(len(selected_indices) / max(candidates.shape[0], 1)),
        "protocol_selection_status_code": status,
        "protocol_zero_variance_noise_flag": zero_var,
        "protocol_noise_reason_code": reason,
        "protocol_estimated_signal_energy": float(np.sum(rec ** 2)),
        "protocol_estimated_noise_energy": float(np.sum(noise ** 2)),
        "protocol_reconstruction_closure_max_abs_error": float(np.max(np.abs(Y_observed - (rec + noise)))),
    })
    return rec, noise, out


def _periodogram(x):
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    p = np.abs(np.fft.rfft(x)) ** 2
    return p / (np.sum(p) + EPS)


# ============================================================
# Reconstruction metrics
# ============================================================

def spectral_correlation(X_clean, X_recovered):
    """
    Spectral reconstruction metric:
        corr(P_true, P_recovered)
    where P is the normalized periodogram.
    """
    if X_clean is None or X_recovered is None:
        return np.nan

    X_clean = np.asarray(X_clean, dtype=float)
    X_recovered = np.asarray(X_recovered, dtype=float)

    if len(X_clean) != len(X_recovered) or len(X_clean) == 0:
        return np.nan

    return _safe_corr(_periodogram(X_clean), _periodogram(X_recovered))


def reconstruction_accuracy(Y_observed, X_clean, imfs, residual, reconstructed=None):
    rec = reconstructed if reconstructed is not None else reconstructed_signal(Y_observed, imfs, residual)

    if X_clean is None or rec is None:
        return {
            "denoise_psnr": np.nan,
            "denoise_mse": np.nan,
            "denoise_nmse": np.nan,
            "denoise_corr": np.nan,
            "spectral_corr": np.nan,
            "denoise_corr_score": np.nan,
            "nmse_score": np.nan,
            "spectral_corr_score": np.nan,
            "reconstruction_score": np.nan,
        }

    X_clean = np.asarray(X_clean, dtype=float)
    rec = np.asarray(rec, dtype=float)

    if len(X_clean) != len(rec):
        return {
            "denoise_psnr": np.nan,
            "denoise_mse": np.nan,
            "denoise_nmse": np.nan,
            "denoise_corr": np.nan,
            "spectral_corr": np.nan,
            "denoise_corr_score": np.nan,
            "nmse_score": np.nan,
            "spectral_corr_score": np.nan,
            "reconstruction_score": np.nan,
        }

    mse = float(np.mean((rec - X_clean) ** 2))
    nmse = float(np.sum((rec - X_clean) ** 2) / (np.sum(X_clean ** 2) + EPS))
    corr = _safe_corr(X_clean, rec)
    scorr = spectral_correlation(X_clean, rec)

    corr_score = _corr_score(X_clean, rec, absolute=False)
    nmse_score = _positive_to_score(nmse)
    spectral_corr_score = _corr_score(_periodogram(X_clean), _periodogram(rec), absolute=False)

    reconstruction_score = _nanmean(
        [corr_score, nmse_score, spectral_corr_score],
        default=np.nan,
    )

    return {
        "denoise_psnr": _safe_psnr(X_clean, rec),
        "denoise_mse": mse,
        "denoise_nmse": nmse,
        "denoise_corr": corr,
        "spectral_corr": scorr,
        "denoise_corr_score": corr_score,
        "nmse_score": nmse_score,
        "spectral_corr_score": spectral_corr_score,
        "reconstruction_score": reconstruction_score,
    }


def snr_gain_diagnostics(Y_observed, X_clean, imfs, residual, reconstructed=None):
    """
    Raw SNR diagnostics. snr_gain_db is reported but not used directly in CaseScore.
    """
    rec = reconstructed if reconstructed is not None else reconstructed_signal(Y_observed, imfs, residual)

    if X_clean is None or rec is None:
        return {
            "input_snr_db": np.nan,
            "output_snr_db": np.nan,
            "snr_gain_db": np.nan,
        }

    Y_observed = np.asarray(Y_observed, dtype=float)
    X_clean = np.asarray(X_clean, dtype=float)
    rec = np.asarray(rec, dtype=float)

    if len(Y_observed) != len(X_clean) or len(rec) != len(X_clean):
        return {
            "input_snr_db": np.nan,
            "output_snr_db": np.nan,
            "snr_gain_db": np.nan,
        }

    signal_power = np.mean(X_clean ** 2) + 1e-20
    input_noise_power = np.mean((Y_observed - X_clean) ** 2) + 1e-20
    output_error_power = np.mean((rec - X_clean) ** 2) + 1e-20

    input_snr = _safe_db(signal_power / input_noise_power)
    output_snr = _safe_db(signal_power / output_error_power)

    return {
        "input_snr_db": float(input_snr),
        "output_snr_db": float(output_snr),
        "snr_gain_db": float(output_snr - input_snr),
    }


# ============================================================
# IMF recovery with Hungarian matching
# ============================================================

def _assignment(cost):
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(cost)
        return list(zip(rows, cols))
    except Exception:
        # Greedy fallback if scipy.optimize is unavailable.
        pairs = []
        used_i = set()
        used_j = set()

        while len(used_i) < cost.shape[0] and len(used_j) < cost.shape[1]:
            best = None
            best_val = np.inf

            for i in range(cost.shape[0]):
                if i in used_i:
                    continue
                for j in range(cost.shape[1]):
                    if j in used_j:
                        continue
                    if cost[i, j] < best_val:
                        best_val = cost[i, j]
                        best = (i, j)

            if best is None:
                break

            i, j = best
            pairs.append((i, j))
            used_i.add(i)
            used_j.add(j)

        return pairs


def _effective_imf_count(imfs, energy_threshold=0.01):
    imfs = _as_2d(imfs)
    if imfs.size == 0:
        return 0
    e = np.sum(imfs ** 2, axis=1)
    total = np.sum(e) + EPS
    return int(np.sum((e / total) >= energy_threshold))


def _normalized_gini_simpson(weights, allocation_floor=0.01):
    weights = np.asarray(weights, dtype=float)
    weights = weights[np.isfinite(weights) & (weights > 0.0)]
    if len(weights):
        weights = weights / (np.sum(weights) + EPS)
        weights = weights[weights >= float(allocation_floor)]
    if len(weights) <= 1:
        return 0.0
    weights = weights / (np.sum(weights) + EPS)
    raw = 1.0 - float(np.sum(weights ** 2))
    max_raw = 1.0 - 1.0 / float(len(weights))
    return float(np.clip(raw / (max_raw + EPS), 0.0, 1.0))


def true_component_mixing_diagnostics(
        imfs,
        true_components=None,
        energy_threshold=0.01,
        association_threshold=0.05,
        allocation_floor=0.01,
):
    """
    Synthetic/semi-synthetic true-component mixing diagnostics.

    component_splitting_index asks whether one true component is distributed
    across several estimated components.  The association matrix has rows =
    estimated components and columns = true components:

        A[j, k] = |corr(estimated_j, true_k)|^2

    Splitting normalizes each true-component column over active estimated
    components; merging normalizes each active estimated-component row over
    true components.  Both primary indices use a normalized Gini-Simpson
    concentration loss in [0, 1], where 0 means association is concentrated in
    a single component and 1 means maximally diffuse allocation.  Components
    with near-zero association are not interpreted as splitting/merging; they
    are reported as missing/spurious diagnostics.
    """
    common = {
        "metric_definition_version": METRIC_DEFINITION_VERSION,
        "evaluation_framework_version": EVALUATION_FRAMEWORK_VERSION,
        "evaluation_framework_status": EVALUATION_FRAMEWORK_STATUS,
        "evaluation_framework_frozen_date": EVALUATION_FRAMEWORK_FROZEN_DATE,
        "structural_metric_schema_version": STRUCTURAL_METRIC_SCHEMA_VERSION,
        "legacy_metric_alias_status": LEGACY_METRIC_ALIAS_STATUS,
        "true_component_metrics_available": False,
        "component_mixing_basis": "abs_zero_lag_corr_squared_true_estimated_components",
        "component_mixing_index_formula": "gated_normalized_gini_simpson_concentration_loss",
        "component_mixing_association_threshold": float(association_threshold),
        "component_mixing_energy_threshold": float(energy_threshold),
        "component_mixing_allocation_floor": float(allocation_floor),
        "true_component_association_strength": np.asarray([], dtype=float),
        "estimated_component_association_strength": np.asarray([], dtype=float),
        "missing_true_component_count": np.nan,
        "missing_true_component_fraction": np.nan,
        "missing_true_component_rate": np.nan,
        "missing_true_component_energy_ratio": np.nan,
        "unmatched_estimated_component_count": np.nan,
        "unmatched_estimated_component_fraction": np.nan,
        "unmatched_estimated_component_rate": np.nan,
        "spurious_mode_energy_ratio": np.nan,
    }
    empty = {
        **common,
        "component_splitting_index": np.nan,
        "component_merging_index": np.nan,
        "true_component_mixing_matrix": np.zeros((0, 0)),
        "true_component_splitting_max": np.nan,
        "estimated_component_merging_max": np.nan,
    }
    if true_components is None:
        return empty

    if imfs is None:
        out = dict(empty)
        true_components = _as_2d(true_components)
        if true_components.size:
            out["true_component_metrics_available"] = True
            out["missing_true_component_count"] = int(true_components.shape[0])
            out["missing_true_component_fraction"] = 1.0
        return out

    imfs = _as_2d(imfs)
    true_components = _as_2d(true_components)
    if imfs.size == 0 or true_components.size == 0 or imfs.shape[1] != true_components.shape[1]:
        out = dict(empty)
        out["true_component_metrics_available"] = bool(true_components.size and imfs.shape[1:] == true_components.shape[1:])
        if true_components.size:
            out["missing_true_component_count"] = int(true_components.shape[0])
            out["missing_true_component_fraction"] = 1.0
            out["missing_true_component_rate"] = 1.0
            out["missing_true_component_energy_ratio"] = 1.0
        if imfs.size:
            out["unmatched_estimated_component_count"] = int(imfs.shape[0])
            out["unmatched_estimated_component_fraction"] = 1.0
            out["unmatched_estimated_component_rate"] = 1.0
            energy = np.sum(imfs ** 2, axis=1)
            out["spurious_mode_energy_ratio"] = float(np.sum(energy) / (np.sum(energy) + EPS))
        return out

    K_est, K_true = imfs.shape[0], true_components.shape[0]
    C = np.zeros((K_est, K_true), dtype=float)
    for i in range(K_est):
        for j in range(K_true):
            c = _safe_corr(imfs[i], true_components[j])
            C[i, j] = 0.0 if not np.isfinite(c) else abs(float(c)) ** 2

    energy = np.sum(imfs ** 2, axis=1)
    energy_ratio = energy / (np.sum(energy) + EPS)
    active_estimated = energy_ratio >= float(energy_threshold)

    gated_C = C.copy()
    gated_C[~active_estimated, :] = 0.0
    true_strength = np.sum(gated_C, axis=0)
    estimated_strength = np.sum(C, axis=1)
    associated_true = true_strength > float(association_threshold)
    associated_estimated = active_estimated & (estimated_strength > float(association_threshold))

    splitting_vals = []
    splitting_weights = []
    for j in range(K_true):
        if not associated_true[j]:
            continue
        col = gated_C[:, j]
        total = float(np.sum(col))
        weights = col / total
        splitting_vals.append(_normalized_gini_simpson(weights, allocation_floor=allocation_floor))
        splitting_weights.append(float(np.sum(true_components[j] ** 2)))

    merging_vals = []
    merging_weights = []
    for i in range(K_est):
        if not associated_estimated[i]:
            continue
        row = C[i, :]
        total = float(np.sum(row))
        weights = row / total
        merging_vals.append(_normalized_gini_simpson(weights, allocation_floor=allocation_floor))
        merging_weights.append(float(energy_ratio[i]))

    if splitting_vals:
        sw = np.asarray(splitting_weights, dtype=float)
        sv = np.asarray(splitting_vals, dtype=float)
        component_splitting = float(np.sum(sv * sw) / (np.sum(sw) + EPS))
        splitting_max = float(np.max(sv))
    else:
        component_splitting = np.nan
        splitting_max = np.nan

    if merging_vals:
        mw = np.asarray(merging_weights, dtype=float)
        mv = np.asarray(merging_vals, dtype=float)
        component_merging = float(np.sum(mv * mw) / (np.sum(mw) + EPS))
        merging_max = float(np.max(mv))
    else:
        component_merging = np.nan
        merging_max = np.nan

    missing_count = int(np.sum(~associated_true))
    true_energy = np.sum(true_components ** 2, axis=1)
    true_energy_ratio = true_energy / (np.sum(true_energy) + EPS)
    missing_energy_ratio = float(np.sum(true_energy_ratio[~associated_true]))
    unmatched_count = int(np.sum(active_estimated & (estimated_strength <= float(association_threshold))))
    spurious_energy = float(np.sum(energy_ratio[active_estimated & (estimated_strength <= float(association_threshold))]))
    missing_fraction = float(missing_count / max(K_true, 1))
    unmatched_fraction = float(unmatched_count / max(int(np.sum(active_estimated)), 1))

    return {
        **common,
        "true_component_metrics_available": True,
        "component_splitting_index": component_splitting,
        "component_merging_index": component_merging,
        "true_component_mixing_matrix": C,
        "true_component_splitting_max": splitting_max,
        "estimated_component_merging_max": merging_max,
        "true_component_association_strength": true_strength,
        "estimated_component_association_strength": estimated_strength,
        "missing_true_component_count": missing_count,
        "missing_true_component_fraction": missing_fraction,
        "missing_true_component_rate": missing_fraction,
        "missing_true_component_energy_ratio": missing_energy_ratio,
        "unmatched_estimated_component_count": unmatched_count,
        "unmatched_estimated_component_fraction": unmatched_fraction,
        "unmatched_estimated_component_rate": unmatched_fraction,
        "spurious_mode_energy_ratio": spurious_energy,
    }


def imf_recovery_diagnostics(imfs, true_components=None):
    """
    Synthetic-benchmark-only metric.

    Uses Hungarian matching / Munkres assignment on absolute correlations:
        cost_ij = 1 - |corr(IMF_i, true_component_j)|

    This avoids assuming that estimated IMF ordering equals true component
    ordering.  This matters because IRMF is interpreted as a top-down
    decomposition while EMD-family methods are bottom-up decompositions.
    """
    alignment_rule = "permutation_invariant_hungarian_abs_corr_not_extraction_index"
    if true_components is None:
        return {
            "imf_recovery_alignment_rule": alignment_rule,
            "imf_recovery_rmse": np.nan,
            "imf_recovery_nrmse": np.nan,
            "imf_recovery_corr": np.nan,
            "imf_recovery_corr_score": np.nan,
            "imf_recovery_rmse_score": np.nan,
            "imf_recovery_score": np.nan,
            "imf_recovery_matched_count": 0,
            "imf_recovery_assignment_pairs": "",
            "true_component_count": np.nan,
            "effective_imf_count": np.nan,
            "decomposition_count_error": np.nan,
            "relative_decomposition_count_error": np.nan,
            "decomposition_adequacy_score": np.nan,
            "over_decomposition_penalty": 0.0,
            "under_decomposition_index": 0.0,
            **true_component_mixing_diagnostics(np.zeros((0, 0)), true_components=None),
        }

    imfs = _as_2d(imfs)
    true_components = _as_2d(true_components)

    if imfs.size == 0 or true_components.size == 0:
        return {
            "imf_recovery_alignment_rule": alignment_rule,
            "imf_recovery_rmse": np.nan,
            "imf_recovery_nrmse": np.nan,
            "imf_recovery_corr": np.nan,
            "imf_recovery_corr_score": np.nan,
            "imf_recovery_rmse_score": np.nan,
            "imf_recovery_score": np.nan,
            "imf_recovery_matched_count": 0,
            "imf_recovery_assignment_pairs": "",
            "true_component_count": int(true_components.shape[0]) if true_components.ndim == 2 else 0,
            "effective_imf_count": 0,
            "decomposition_count_error": 1.0,
            "relative_decomposition_count_error": 1.0,
            "decomposition_adequacy_score": 0.0,
            "over_decomposition_penalty": 0.0,
            "under_decomposition_index": 1.0,
            **true_component_mixing_diagnostics(imfs, true_components=true_components),
        }

    if imfs.shape[1] != true_components.shape[1]:
        return {
            "imf_recovery_alignment_rule": alignment_rule,
            "imf_recovery_rmse": np.nan,
            "imf_recovery_nrmse": np.nan,
            "imf_recovery_corr": np.nan,
            "imf_recovery_corr_score": np.nan,
            "imf_recovery_rmse_score": np.nan,
            "imf_recovery_score": np.nan,
            "imf_recovery_matched_count": 0,
            "imf_recovery_assignment_pairs": "",
            "true_component_count": int(true_components.shape[0]),
            "effective_imf_count": int(imfs.shape[0]),
            "decomposition_count_error": np.nan,
            "relative_decomposition_count_error": np.nan,
            "decomposition_adequacy_score": np.nan,
            "over_decomposition_penalty": 0.0,
            "under_decomposition_index": 0.0,
            **true_component_mixing_diagnostics(imfs, true_components=true_components),
        }

    K_est, K_true = imfs.shape[0], true_components.shape[0]
    cost = np.ones((K_est, K_true), dtype=float)

    for i in range(K_est):
        for j in range(K_true):
            c = _safe_corr(imfs[i], true_components[j])
            if np.isfinite(c):
                cost[i, j] = 1.0 - abs(c)

    pairs = _assignment(cost)

    rmses = []
    nrmse_vals = []
    corrs = []

    for i, j in pairs:
        rmse = float(np.sqrt(np.mean((imfs[i] - true_components[j]) ** 2)))
        denom = float(np.sqrt(np.mean(true_components[j] ** 2)) + EPS)
        rmses.append(rmse)
        nrmse_vals.append(rmse / denom)

        c = _safe_corr(imfs[i], true_components[j])
        if np.isfinite(c):
            corrs.append(abs(c))

    imf_recovery_rmse = float(np.nanmean(rmses)) if rmses else np.nan
    imf_recovery_nrmse = float(np.nanmean(nrmse_vals)) if nrmse_vals else np.nan
    imf_recovery_corr = float(np.nanmean(corrs)) if corrs else np.nan

    corr_score = _clip01(imf_recovery_corr)
    rmse_score = _positive_to_score(imf_recovery_nrmse)
    recovery_score = _nanmean([corr_score, rmse_score], default=np.nan)

    # OD/UD based on effective IMF count, normalized by true component count.
    K_eff = _effective_imf_count(imfs, energy_threshold=0.01)
    denom = float(max(K_true, 1))
    od = max(0.0, K_eff - K_true) / denom
    ud = max(0.0, K_true - K_eff) / denom
    decomposition_count_error = abs(float(K_eff) - float(K_true)) / denom
    decomposition_adequacy_score = 1.0 - min(1.0, decomposition_count_error)

    component_mixing = true_component_mixing_diagnostics(
        imfs,
        true_components=true_components,
    )

    return {
        "imf_recovery_alignment_rule": alignment_rule,
        "imf_recovery_rmse": imf_recovery_rmse,
        "imf_recovery_nrmse": imf_recovery_nrmse,
        "imf_recovery_corr": imf_recovery_corr,
        "imf_recovery_corr_score": corr_score,
        "imf_recovery_rmse_score": rmse_score,
        "imf_recovery_score": recovery_score,
        "imf_recovery_matched_count": int(len(pairs)),
        "imf_recovery_assignment_pairs": ";".join(f"est{i}->true{j}" for i, j in pairs),
        "true_component_count": int(K_true),
        "effective_imf_count": int(K_eff),
        "decomposition_count_error": float(decomposition_count_error),
        "relative_decomposition_count_error": float(decomposition_count_error),
        "decomposition_adequacy_score": float(decomposition_adequacy_score),
        "over_decomposition_penalty": float(od),
        "under_decomposition_index": float(ud),
        **component_mixing,
    }


# ============================================================
# Contamination resistance
# ============================================================

def noise_capture_diagnostics(Y_observed, X_clean, residual, estimated_noise=None):
    """
    Synthetic-benchmark-only residual/noise alignment metric.
    """
    empty = {
        "noise_capture_corr": np.nan,
        "noise_capture_corr_score": np.nan,
        "noise_capture_energy_ratio": np.nan,
        "noise_energy_ratio": np.nan,
        "noise_energy_log_error": np.nan,
        "remaining_noise_energy_ratio": np.nan,
        "noise_energy_bias_class": "not_computable",
        "noise_energy_eps": NOISE_ENERGY_EPS,
        "noise_energy_bias_tolerance": NOISE_ENERGY_BIAS_TOLERANCE,
        "noise_energy_diagnostic_role": "secondary_universal_noise_separation_diagnostic",
        "remaining_noise_energy_ratio_role": "secondary_universal_noise_separation_diagnostic",
        "noise_energy_not_computable_reason": None,
    }
    if X_clean is None:
        out = dict(empty)
        out["noise_energy_not_computable_reason"] = "missing_clean_reference"
        return out

    Y_observed = np.asarray(Y_observed, dtype=float)
    X_clean = np.asarray(X_clean, dtype=float)
    residual = np.asarray(estimated_noise if estimated_noise is not None else residual, dtype=float)

    if len(Y_observed) != len(X_clean) or len(residual) != len(X_clean):
        out = dict(empty)
        out["noise_energy_not_computable_reason"] = "length_mismatch"
        return out

    true_noise = Y_observed - X_clean
    c = _safe_corr(residual, true_noise)
    true_noise_energy = float(np.sum(true_noise ** 2))
    estimated_noise_energy = float(np.sum(residual ** 2))
    remaining_noise_energy = float(np.sum((true_noise - residual) ** 2))
    ratio = float(estimated_noise_energy / (true_noise_energy + NOISE_ENERGY_EPS))
    if true_noise_energy <= NOISE_ENERGY_EPS:
        log_error = np.nan
        remaining_ratio = np.nan
        not_computable = "not_computable_zero_energy_true_noise"
        bias_class = "not_computable"
    else:
        log_error = float(abs(np.log((estimated_noise_energy + NOISE_ENERGY_EPS) / (true_noise_energy + NOISE_ENERGY_EPS))))
        remaining_ratio = float(remaining_noise_energy / (true_noise_energy + NOISE_ENERGY_EPS))
        not_computable = None
        if ratio < 1.0 - NOISE_ENERGY_BIAS_TOLERANCE:
            bias_class = "under_recovery"
        elif ratio > 1.0 + NOISE_ENERGY_BIAS_TOLERANCE:
            bias_class = "over_recovery"
        else:
            bias_class = "approximately_correct"

    return {
        "noise_capture_corr": c,
        "noise_capture_corr_score": _clip01((c + 1.0) / 2.0) if np.isfinite(c) else np.nan,
        "noise_capture_energy_ratio": ratio,
        "noise_energy_ratio": ratio,
        "noise_energy_log_error": log_error,
        "remaining_noise_energy_ratio": remaining_ratio,
        "noise_energy_bias_class": bias_class,
        "noise_energy_eps": NOISE_ENERGY_EPS,
        "noise_energy_bias_tolerance": NOISE_ENERGY_BIAS_TOLERANCE,
        "noise_energy_diagnostic_role": "secondary_universal_noise_separation_diagnostic",
        "remaining_noise_energy_ratio_role": "secondary_universal_noise_separation_diagnostic",
        "noise_energy_not_computable_reason": not_computable,
    }


def outlier_resistance_index(Y_observed, X_clean, imfs, residual, q=0.95, reconstructed=None):
    """
    Outlier Resistance Index (ORI).

    e(t) = X(t) - X_hat(t)

    ORI = 1 - Q_q(|e|) / (range(X) + eps)

    Higher is better. The result is clipped to [0, 1].
    """
    rec = reconstructed if reconstructed is not None else reconstructed_signal(Y_observed, imfs, residual)
    if X_clean is None or rec is None:
        return {
            "outlier_resistance_index": np.nan,
            "ori": np.nan,
        }

    X_clean = np.asarray(X_clean, dtype=float)
    rec = np.asarray(rec, dtype=float)

    if len(X_clean) != len(rec) or len(X_clean) == 0:
        return {
            "outlier_resistance_index": np.nan,
            "ori": np.nan,
        }

    e = np.abs(X_clean - rec)
    qerr = float(np.quantile(e, q))
    scale = float(np.max(X_clean) - np.min(X_clean) + EPS)

    ori = 1.0 - qerr / scale
    ori = float(np.clip(ori, 0.0, 1.0))

    return {
        "outlier_resistance_index": ori,
        "ori": ori,
        "ori_q": float(q),
        "ori_abs_error_quantile": qerr,
    }


def contamination_region_diagnostics(
        Y_observed,
        X_clean,
        imfs,
        residual,
        contamination_mask=None,
        spillover_radius=5,
        reconstructed=None,
):
    """
    Region-level contamination diagnostics.

    These metrics are only applicable when the synthetic noise model has
    explicit contaminated locations.  They are diagnostics for contamination
    mechanisms and are not folded into the legacy case_score.
    """
    rec = reconstructed if reconstructed is not None else reconstructed_signal(Y_observed, imfs, residual)
    out = {
        "contamination_region_applicable": False,
        "contaminated_point_count": 0,
        "contaminated_fraction": np.nan,
        "contaminated_region_nmse": np.nan,
        "contaminated_region_preservation_score": np.nan,
        "clean_region_nmse": np.nan,
        "clean_region_preservation_score": np.nan,
        "contamination_spillover_error": np.nan,
        "contamination_spillover_score": np.nan,
        "spillover_radius": int(spillover_radius),
    }
    if X_clean is None or rec is None or contamination_mask is None:
        return out

    X_clean = np.asarray(X_clean, dtype=float)
    rec = np.asarray(rec, dtype=float)
    mask = np.asarray(contamination_mask, dtype=bool)
    if len(X_clean) != len(rec) or len(mask) != len(X_clean) or len(mask) == 0:
        return out

    n_cont = int(np.sum(mask))
    out["contaminated_point_count"] = n_cont
    out["contaminated_fraction"] = float(n_cont / max(len(mask), 1))
    if n_cont <= 0:
        return out

    err2 = (rec - X_clean) ** 2
    clean_mask = ~mask
    cont_denom = float(np.sum(X_clean[mask] ** 2) + EPS)
    cont_nmse = float(np.sum(err2[mask]) / cont_denom)
    out["contaminated_region_nmse"] = cont_nmse
    out["contaminated_region_preservation_score"] = _positive_to_score(cont_nmse)

    if np.any(clean_mask):
        clean_denom = float(np.sum(X_clean[clean_mask] ** 2) + EPS)
        clean_nmse = float(np.sum(err2[clean_mask]) / clean_denom)
        out["clean_region_nmse"] = clean_nmse
        out["clean_region_preservation_score"] = _positive_to_score(clean_nmse)

    radius = int(max(0, spillover_radius))
    if radius > 0:
        expanded = mask.copy()
        idx = np.flatnonzero(mask)
        for i in idx:
            lo = max(0, int(i) - radius)
            hi = min(len(mask), int(i) + radius + 1)
            expanded[lo:hi] = True
        spill_mask = expanded & (~mask)
        if np.any(spill_mask):
            out["contamination_spillover_error"] = float(np.mean(err2[spill_mask]))
            scale = float(np.mean(X_clean ** 2) + EPS)
            out["contamination_spillover_score"] = _positive_to_score(out["contamination_spillover_error"] / scale)

    out["contamination_region_applicable"] = True
    return out


def _projection_energy_ratio(vector, basis_rows):
    vector = np.asarray(vector, dtype=float)
    basis = _as_2d(basis_rows)
    if vector.size == 0 or basis.size == 0 or basis.shape[1] != vector.size:
        return np.nan
    # Rows are components; QR expects columns as basis vectors.
    S = basis.T
    try:
        Q, r = np.linalg.qr(S, mode="reduced")
    except Exception:
        return np.nan
    if Q.size == 0:
        return np.nan
    diag = np.abs(np.diag(r)) if r.ndim == 2 else np.asarray([])
    if diag.size:
        keep = diag > 1e-10
        Q = Q[:, keep]
    if Q.size == 0:
        return np.nan
    projected = Q @ (Q.T @ vector)
    return float(np.sum(projected ** 2) / (np.sum(vector ** 2) + EPS))


def signal_leakage_into_noise_diagnostics(
        Y_observed,
        X_clean,
        residual,
        true_components=None,
        contamination_mask=None,
        residual_energy_eps=1e-10,
        estimated_noise=None,
):
    """
    Projection-energy signal leakage into the estimated noise/residual.

    signal_leakage_into_noise = ||P_S residual||^2 / ||residual||^2,
    where S is the span of true intrinsic components when available.  Lower is
    better.  Degenerate near-zero residual energy is reported as NaN with a
    flag, so a method that removes almost nothing does not receive a false
    perfect score.
    """
    out = {
        "signal_leakage_into_noise": np.nan,
        "signal_leakage_into_noise_score": np.nan,
        "signal_leakage_residual_energy": np.nan,
        "signal_leakage_observed_energy": np.nan,
        "signal_leakage_degenerate_flag": True,
        "signal_leakage_basis": None,
        "clean_region_signal_leakage": np.nan,
        "clean_region_signal_leakage_score": np.nan,
    }
    if X_clean is None or (residual is None and estimated_noise is None):
        return out

    residual = np.asarray(estimated_noise if estimated_noise is not None else residual, dtype=float)
    X_clean = np.asarray(X_clean, dtype=float)
    Y_observed = np.asarray(Y_observed, dtype=float)
    if len(residual) != len(X_clean) or len(Y_observed) != len(X_clean) or len(residual) == 0:
        return out

    residual_energy = float(np.sum(residual ** 2))
    observed_energy = float(np.sum(Y_observed ** 2) + EPS)
    out["signal_leakage_residual_energy"] = residual_energy
    out["signal_leakage_observed_energy"] = observed_energy
    if residual_energy < residual_energy_eps * observed_energy:
        return out
    out["signal_leakage_degenerate_flag"] = False

    if true_components is not None:
        basis = _as_2d(true_components)
        out["signal_leakage_basis"] = "true_component_span"
    else:
        basis = X_clean[None, :]
        out["signal_leakage_basis"] = "clean_signal_vector"

    leakage = _projection_energy_ratio(residual, basis)
    if np.isfinite(leakage):
        leakage = float(np.clip(leakage, 0.0, 1.0))
        out["signal_leakage_into_noise"] = leakage
        out["signal_leakage_into_noise_score"] = float(1.0 - leakage)

    if contamination_mask is not None:
        mask = np.asarray(contamination_mask, dtype=bool)
        if len(mask) == len(residual) and np.any(~mask):
            clean_residual = residual[~mask]
            if np.sum(clean_residual ** 2) >= residual_energy_eps * observed_energy:
                if true_components is not None:
                    clean_basis = _as_2d(true_components)[:, ~mask]
                else:
                    clean_basis = X_clean[None, ~mask]
                clean_leakage = _projection_energy_ratio(clean_residual, clean_basis)
                if np.isfinite(clean_leakage):
                    clean_leakage = float(np.clip(clean_leakage, 0.0, 1.0))
                    out["clean_region_signal_leakage"] = clean_leakage
                    out["clean_region_signal_leakage_score"] = float(1.0 - clean_leakage)
    return out


def contamination_resistance_score(result):
    """
    Case-level contamination resistance.

    C = 0.60 ORI + 0.40 NoiseCaptureCorrScore
    """
    ori = result.get("outlier_resistance_index", np.nan)
    ncc = result.get("noise_capture_corr_score", np.nan)

    values = []
    weights = []

    if np.isfinite(ori):
        values.append(ori)
        weights.append(0.60)

    if np.isfinite(ncc):
        values.append(ncc)
        weights.append(0.40)

    if not values:
        return np.nan

    weights = np.asarray(weights, dtype=float)
    weights = weights / (np.sum(weights) + EPS)
    return float(np.sum(weights * np.asarray(values, dtype=float)))


# ============================================================
# Structural fidelity metrics
# ============================================================

def strict_io(imfs):
    """
    Raw strict orthogonality index. Lower is better.
    """
    imfs = _as_2d(imfs)
    K = imfs.shape[0]

    if K <= 1:
        return 0.0

    vals = []

    for i in range(K):
        for j in range(i + 1, K):
            denom = np.linalg.norm(imfs[i]) * np.linalg.norm(imfs[j]) + EPS
            vals.append(abs(float(np.dot(imfs[i], imfs[j]) / denom)))

    return float(np.mean(vals)) if vals else 0.0


def classical_io_with_residual(Y_observed, imfs, residual):
    imfs = _as_2d(imfs)

    if imfs.size == 0:
        return 0.0

    components = list(imfs) + [np.asarray(residual, dtype=float)]
    vals = []

    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            denom = np.linalg.norm(components[i]) * np.linalg.norm(components[j]) + EPS
            vals.append(abs(float(np.dot(components[i], components[j]) / denom)))

    return float(np.mean(vals)) if vals else 0.0


def spectral_leakage(imfs):
    """
    Raw spectral leakage. Lower is better.
    """
    imfs = _as_2d(imfs)
    K = imfs.shape[0]

    if K <= 1:
        return 0.0

    psds = []

    for imf in imfs:
        p = np.abs(np.fft.rfft(imf)) ** 2
        p = p / (np.sum(p) + EPS)
        psds.append(p)

    P = np.vstack(psds)
    M = P @ P.T
    offdiag = ~np.eye(K, dtype=bool)

    return float(np.mean(M[offdiag]))


def frequency_overlap_matrix(imfs):
    imfs = _as_2d(imfs)
    K = imfs.shape[0]

    if K == 0:
        return np.zeros((0, 0))

    psds = []

    for imf in imfs:
        p = np.abs(np.fft.rfft(imf)) ** 2
        p = p / (np.sqrt(np.sum(p ** 2)) + EPS)
        psds.append(p)

    P = np.vstack(psds)
    return np.clip(P @ P.T, 0.0, 1.0)


def frequency_overlap_statistics(imfs):
    M = frequency_overlap_matrix(imfs)
    K = M.shape[0]

    if K <= 1:
        return {
            "frequency_overlap_matrix": M,
            "frequency_overlap_mean_offdiag": 0.0,
            "frequency_overlap_max_offdiag": 0.0,
            "frequency_overlap_adjacent_mean": 0.0,
            "frequency_overlap_adjacent_max": 0.0,
        }

    offdiag = ~np.eye(K, dtype=bool)
    adjacent = np.array([M[i, i + 1] for i in range(K - 1)])

    return {
        "frequency_overlap_matrix": M,
        "frequency_overlap_mean_offdiag": float(np.mean(M[offdiag])),
        "frequency_overlap_max_offdiag": float(np.max(M[offdiag])),
        "frequency_overlap_adjacent_mean": float(np.mean(adjacent)),
        "frequency_overlap_adjacent_max": float(np.max(adjacent)),
    }


def center_frequencies(imfs, fs):
    imfs = _as_2d(imfs)
    out = []

    for imf in imfs:
        p = np.abs(np.fft.rfft(imf)) ** 2
        f = np.fft.rfftfreq(len(imf), d=1.0 / fs)
        out.append(float(np.sum(f * p) / (np.sum(p) + EPS)))

    return np.asarray(out, dtype=float)


def imf_energy_ratios(imfs):
    imfs = _as_2d(imfs)

    if imfs.size == 0:
        return np.array([])

    e = np.sum(imfs ** 2, axis=1)
    return e / (np.sum(e) + EPS)


def dominant_freq_energy_pairs(center_freqs, imfs):
    cf = np.asarray(center_freqs, dtype=float)
    er = imf_energy_ratios(imfs)
    n = min(len(cf), len(er))

    return [(float(cf[i]), float(er[i])) for i in range(n)]


def frequency_spacing_diagnostics(center_freqs):
    cf = np.asarray(center_freqs, dtype=float)
    cf = cf[np.isfinite(cf)]
    cf = cf[cf > 1e-12]

    if len(cf) <= 1:
        return {
            "frequency_spacing_min_ratio": 0.0,
            "frequency_spacing_mean_ratio": 0.0,
            "frequency_spacing_penalty": 0.0,
            "frequency_separation_score": 1.0,
        }

    cf = np.sort(cf)
    ratios = np.diff(cf) / (cf[1:] + EPS)

    min_ratio = float(np.min(ratios))
    mean_ratio = float(np.mean(ratios))
    threshold = 0.25
    penalty = float(max(0.0, (threshold - min_ratio) / threshold))

    return {
        "frequency_spacing_min_ratio": min_ratio,
        "frequency_spacing_mean_ratio": mean_ratio,
        "frequency_spacing_penalty": penalty,
        "frequency_separation_score": float(np.clip(1.0 - penalty, 0.0, 1.0)),
    }


def _mode_mixing_diagnostics(imfs, fs):
    """
    Wrapper around diagnostics.mode_mixing_diagnostics.
    """
    try:
        from diagnostics.mode_mixing_diagnostics import compute_mode_mixing_diagnostics
        d = compute_mode_mixing_diagnostics(_as_2d(imfs), fs)
        if "mode_mixing_index" not in d:
            d["mode_mixing_index"] = d.get("mode_mixing_score", np.nan)
        d["inter_imf_entanglement_index"] = d.get("mode_mixing_index", d.get("mode_mixing_score", np.nan))
        return d
    except Exception:
        return {
            "mode_mixing_score": np.nan,
            "mode_mixing_index": np.nan,
            "inter_imf_entanglement_index": np.nan,
            "imf_corr_mean_offdiag": np.nan,
            "imf_corr_max_offdiag": np.nan,
            "ridge_overlap_mean_offdiag": np.nan,
            "ridge_overlap_max_offdiag": np.nan,
            "local_frequency_crossing_rate": np.nan,
        }


def transient_smearing_diagnostics(imfs, true_components=None):
    """
    Local Structure Preservation metric.

    For sparse/local true components, match the local component to the estimated IMF
    with maximal absolute correlation and measure the fraction of matched-IMF energy
    outside the true active support.

    Returns 0 for signals without a clearly local component.
    """
    if true_components is None:
        return {
            "transient_smearing_index": 0.0,
            "transient_preservation_score": 1.0,
        }

    imfs = _as_2d(imfs)
    true_components = _as_2d(true_components)

    if imfs.size == 0 or true_components.size == 0 or imfs.shape[1] != true_components.shape[1]:
        return {
            "transient_smearing_index": np.nan,
            "transient_preservation_score": np.nan,
        }

    # Identify sparse/local components by active support fraction.
    local_candidates = []
    for j, comp in enumerate(true_components):
        amp = np.abs(comp)
        if np.max(amp) < EPS:
            continue
        active = amp > 0.05 * np.max(amp)
        support_frac = float(np.mean(active))
        if 0.0 < support_frac <= 0.45:
            local_candidates.append((j, active, support_frac))

    if not local_candidates:
        return {
            "transient_smearing_index": 0.0,
            "transient_preservation_score": 1.0,
        }

    smear_vals = []

    for j, active, support_frac in local_candidates:
        # Match by highest absolute correlation. If correlation is unavailable,
        # fall back to maximum energy concentration on active support.
        best_i = None
        best_score = -np.inf

        for i in range(imfs.shape[0]):
            c = _safe_corr(imfs[i], true_components[j])
            if np.isfinite(c):
                score = abs(c)
            else:
                e_total = np.sum(imfs[i] ** 2) + EPS
                score = float(np.sum(imfs[i][active] ** 2) / e_total)

            if score > best_score:
                best_score = score
                best_i = i

        if best_i is None:
            continue

        e_total = np.sum(imfs[best_i] ** 2) + EPS
        outside_energy = np.sum(imfs[best_i][~active] ** 2)
        smear_vals.append(float(np.clip(outside_energy / e_total, 0.0, 1.0)))

    if not smear_vals:
        return {
            "transient_smearing_index": 0.0,
            "transient_preservation_score": 1.0,
        }

    smearing = float(np.mean(smear_vals))
    return {
        "transient_smearing_index": smearing,
        "transient_preservation_score": float(np.clip(1.0 - smearing, 0.0, 1.0)),
    }


def residual_whiteness_penalty(residual, max_lag=20):
    """
    Diagnostic only. Lower is better.
    """
    residual = np.asarray(residual, dtype=float)
    residual = residual - np.mean(residual)

    if len(residual) <= 3:
        return 0.0

    denom = np.sum(residual ** 2) + EPS
    vals = []

    for lag in range(1, min(max_lag, len(residual) - 1) + 1):
        vals.append(float(np.sum(residual[:-lag] * residual[lag:]) / denom) ** 2)

    return float(np.mean(vals)) if vals else 0.0


def instantaneous_frequency_smoothness(imfs):
    """
    Diagnostic only. Lower is usually smoother.
    """
    imfs = _as_2d(imfs)
    vals = []

    for imf in imfs:
        if np.std(imf) < 1e-12:
            continue

        phase = np.unwrap(np.angle(hilbert(imf)))
        om = np.diff(phase)

        if len(om) >= 5:
            om = medfilt(om, kernel_size=5)

        denom = np.mean(np.abs(om)) ** 2 + EPS
        vals.append(float(np.var(om) / denom))

    return float(np.mean(vals)) if vals else 0.0


def energy_ratio(Y_observed, imfs, residual):
    """
    Diagnostic only.
    """
    imfs = _as_2d(imfs)
    Y_observed = np.asarray(Y_observed, dtype=float)

    e_imfs = np.sum(imfs ** 2)
    e_y = np.sum(Y_observed ** 2) + EPS
    return float(e_imfs / e_y)


def imf_energy_concentration_penalty(imfs):
    """
    Diagnostic only.
    """
    r = imf_energy_ratios(imfs)

    if len(r) == 0:
        return 0.0

    entropy = -np.sum(r * np.log(r + EPS)) / np.log(len(r) + EPS)
    return float(max(0.0, np.max(r) - 0.85) + max(0.0, 0.20 - entropy))


# ============================================================
# Case-level composite scoring
# ============================================================

def structural_fidelity_score(result):
    """
    Structural Fidelity legacy composite.

    S_raw =
        0.45 IMF / component recovery
      + 0.35 true-component mixing control
      + 0.20 local structure preservation

    Decomposition count and inter-IMF entanglement are returned as diagnostics,
    not multiplicative penalties in this legacy composite.
    """
    # B1 IMF Recovery
    recovery = result.get("imf_recovery_score", np.nan)

    # Secondary internal diagnostics retained for mechanism interpretation.
    strict_orth_score = _loss_to_unit_score(result.get("strict_io", np.nan))
    leakage_score = _loss_to_unit_score(result.get("spectral_leakage", np.nan))
    overlap_score = _loss_to_unit_score(result.get("frequency_overlap_max_offdiag", np.nan))
    orth_leakage = _nanmean(
        [strict_orth_score, leakage_score, overlap_score],
        default=np.nan,
    )

    # Legacy inter-IMF entanglement diagnostic, not the primary true-component
    # mixing endpoint.
    sep_score = result.get("frequency_separation_score", np.nan)
    mm_raw = result.get(
        "inter_imf_entanglement_index",
        result.get("mode_mixing_index", result.get("mode_mixing_score", np.nan)),
    )
    mm_score = _loss_to_unit_score(mm_raw)
    frequency_sep = _nanmean([sep_score, mm_score], default=np.nan)

    split_score = _loss_to_unit_score(result.get("component_splitting_index", np.nan))
    merge_score = _loss_to_unit_score(result.get("component_merging_index", np.nan))
    true_component_mixing = _nanmean([split_score, merge_score], default=np.nan)

    local_preservation = result.get("transient_preservation_score", np.nan)

    groups = []
    weights = []

    for val, w in [
        (recovery, 0.45),
        (true_component_mixing, 0.35),
        (local_preservation, 0.20),
    ]:
        if np.isfinite(val):
            groups.append(val)
            weights.append(w)

    if not groups:
        raw = np.nan
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / (np.sum(weights) + EPS)
        raw = float(np.sum(weights * np.asarray(groups, dtype=float)))

    od = float(result.get("over_decomposition_penalty", 0.0) or 0.0)
    ud = float(result.get("under_decomposition_index", 0.0) or 0.0)
    penalty = float(np.exp(-od - 2.0 * ud))
    final = raw

    return {
        "imf_recovery_group_score": recovery,
        "true_component_mixing_group_score": true_component_mixing,
        "orthogonality_leakage_group_score": orth_leakage,
        "frequency_separation_group_score": frequency_sep,
        "local_structure_preservation_group_score": local_preservation,
        "structural_fidelity_raw": raw,
        "od_ud_penalty_factor": penalty,
        "structural_fidelity_score_policy": "truth_component_recovery_mixing_primary_count_entanglement_diagnostic",
        "structural_fidelity_score": final,
    }


def case_level_score(result):
    """
    CaseScore =
        0.40 Reconstruction
      + 0.35 Structural Fidelity
      + 0.25 Contamination Resistance

    All inputs are [0,1], where 1 = good.
    """
    R = result.get("reconstruction_score", np.nan)
    S = result.get("structural_fidelity_score", np.nan)
    C = result.get("contamination_resistance_score", np.nan)

    values = []
    weights = []

    for val, w in [(R, 0.40), (S, 0.35), (C, 0.25)]:
        if np.isfinite(val):
            values.append(val)
            weights.append(w)

    if not values:
        return np.nan

    weights = np.asarray(weights, dtype=float)
    weights = weights / (np.sum(weights) + EPS)
    return float(np.sum(weights * np.asarray(values, dtype=float)))


# ============================================================
# Backward-compatible composite scores
# ============================================================

def robust_estimation_score(result):
    """
    Backward-compatible minimization score.
    Smaller is better.

    It now represents reconstruction loss:
        robust_estimation_score = 1 - reconstruction_score
    """
    score = result.get("reconstruction_score", np.nan)
    if np.isfinite(score):
        return float(1.0 - score), {
            "reconstruction_loss_score_component": float(1.0 - score),
        }
    return np.nan, {"reconstruction_loss_score_component": np.nan}


def decomposition_quality_score(result):
    """
    Backward-compatible minimization score.
    Smaller is better.

    It now represents structural fidelity loss:
        decomposition_quality_score = 1 - structural_fidelity_score
    """
    score = result.get("structural_fidelity_score", np.nan)
    if np.isfinite(score):
        return float(1.0 - score), {
            "structural_fidelity_loss_score_component": float(1.0 - score),
        }
    return np.nan, {"structural_fidelity_loss_score_component": np.nan}


# ============================================================
# Main entry point
# ============================================================

def evaluate_shared_physical_diagnostics(
        Y_observed,
        X_clean,
        imfs,
        residual,
        fs,
        residual_penalty_mode="whiteness",
        true_components=None,
        contamination_mask=None,
        reconstruction_protocol_id=None,
):
    imfs = _as_2d(imfs)
    try:
        from diagnostics.real_proxy_diagnostics import compute_real_proxy_diagnostics
        proxy_stats = compute_real_proxy_diagnostics(Y_observed, imfs, residual, fs)
    except Exception:
        proxy_stats = {}

    protocol_rec = protocol_noise = None
    protocol_stats = {}
    if reconstruction_protocol_id == SYNTHETIC_RECONSTRUCTION_PROTOCOL_ID:
        protocol_rec, protocol_noise, protocol_stats = protocol_component_reconstruction(
            Y_observed,
            imfs,
            residual,
            true_components=true_components,
        )

    rec_stats = reconstruction_accuracy(Y_observed, X_clean, imfs, residual, reconstructed=protocol_rec)
    snr_stats = snr_gain_diagnostics(Y_observed, X_clean, imfs, residual, reconstructed=protocol_rec)
    recovery_stats = imf_recovery_diagnostics(imfs, true_components=true_components)
    noise_stats = noise_capture_diagnostics(Y_observed, X_clean, residual, estimated_noise=protocol_noise)
    ori_stats = outlier_resistance_index(Y_observed, X_clean, imfs, residual, reconstructed=protocol_rec)
    contamination_region_stats = contamination_region_diagnostics(
        Y_observed,
        X_clean,
        imfs,
        residual,
        contamination_mask=contamination_mask,
        reconstructed=protocol_rec,
    )
    leakage_stats = signal_leakage_into_noise_diagnostics(
        Y_observed,
        X_clean,
        residual,
        true_components=true_components,
        contamination_mask=contamination_mask,
        estimated_noise=protocol_noise,
    )

    cf = center_frequencies(imfs, fs)
    fover = frequency_overlap_statistics(imfs)
    spacing = frequency_spacing_diagnostics(cf)
    mixing = _mode_mixing_diagnostics(imfs, fs)
    transient = transient_smearing_diagnostics(imfs, true_components=true_components)

    io = strict_io(imfs)
    cio = classical_io_with_residual(Y_observed, imfs, residual)
    leak = spectral_leakage(imfs)
    white = residual_whiteness_penalty(residual)
    ifs = instantaneous_frequency_smoothness(imfs)
    er = energy_ratio(Y_observed, imfs, residual)
    energy_penalty = imf_energy_concentration_penalty(imfs)

    result = {
        **rec_stats,
        **snr_stats,
        **recovery_stats,
        **noise_stats,
        **ori_stats,
        **contamination_region_stats,
        **leakage_stats,
        **protocol_stats,
        **proxy_stats,
        "strict_io": io,
        "io": io,
        "classical_io": cio,
        "spectral_leakage": leak,
        "center_freqs": cf,
        "imf_energy_ratios": imf_energy_ratios(imfs),
        "dominant_freq_energy_pairs": dominant_freq_energy_pairs(cf, imfs),
        "residual_whiteness": white,
        "residual_autocorrelation_score": white,
        "ifs": ifs,
        "instantaneous_frequency_smoothness": ifs,
        "energy_ratio": er,
        "energy_concentration_penalty": energy_penalty,
        "imf_count": int(imfs.shape[0]),
        **fover,
        **spacing,
        **mixing,
        **transient,
    }

    # Ensure these fields are always available under the final names.
    if "mode_mixing_index" not in result:
        result["mode_mixing_index"] = result.get("mode_mixing_score", np.nan)
    if "inter_imf_entanglement_index" not in result:
        result["inter_imf_entanglement_index"] = result.get("mode_mixing_index", np.nan)
    result["mode_mixing_index_legacy"] = result.get("inter_imf_entanglement_index", np.nan)
    result["legacy_metric_used_in_primary_analysis"] = False
    result["metric_definition_version"] = METRIC_DEFINITION_VERSION
    result["evaluation_framework_version"] = EVALUATION_FRAMEWORK_VERSION
    result["evaluation_framework_status"] = EVALUATION_FRAMEWORK_STATUS
    result["evaluation_framework_frozen_date"] = EVALUATION_FRAMEWORK_FROZEN_DATE
    result["structural_metric_schema_version"] = STRUCTURAL_METRIC_SCHEMA_VERSION
    result["legacy_metric_alias_status"] = LEGACY_METRIC_ALIAS_STATUS

    result["frequency_separation_score"] = float(
        max(0.0, 1.0 - result.get("frequency_overlap_max_offdiag", 0.0))
    )

    result["contamination_resistance_score"] = contamination_resistance_score(result)
    # Paper-facing alias.  The legacy field name is kept for compatibility, but
    # the manuscript can report this dimension as Robustness because it covers
    # outlier resistance and residual/noise capture under heavy-tailed, colored,
    # burst, contamination, and heteroskedastic regimes.
    result["robustness_score"] = result["contamination_resistance_score"]
    result["robustness_score_legacy_field"] = "contamination_resistance_score"

    struct_scores = structural_fidelity_score(result)
    result.update(struct_scores)

    cscore = case_level_score(result)
    result["case_score"] = cscore
    result["case_score_final"] = cscore
    result["case_score_role"] = "composite_sensitivity_endpoint_only"

    robust_score, robust_components = robust_estimation_score(result)
    decomp_score, decomp_components = decomposition_quality_score(result)

    result.update(robust_components)
    result.update(decomp_components)

    result["robust_estimation_score"] = robust_score
    result["decomposition_quality_score"] = decomp_score

    # Backward-compatible field name. Existing code sorts ascending.
    result["general_physical_score"] = float(1.0 - cscore) if np.isfinite(cscore) else np.inf

    return result
