import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product
from scipy.signal import hilbert
from scipy.optimize import linear_sum_assignment, minimize_scalar
from scipy.special import ndtr


# ============================================================
# 0. Output settings
# ============================================================
OUTPUT_DIR = Path("IRMF17S_maxK_sensitivity_geometric_constraint_plots")
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams["figure.facecolor"] = "none"
plt.rcParams["axes.facecolor"] = "none"
plt.rcParams["savefig.transparent"] = True


def _safe_filename(title):
    safe = "".join(char.lower() if char.isalnum() else "_" for char in title)
    safe = "_".join(part for part in safe.split("_") if part)
    return safe[:160] or "plot"


def save_current_plot(title):
    output_path = OUTPUT_DIR / f"{_safe_filename(title)}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
    print(f"Saved figure: {output_path}")
    plt.close()


# ============================================================
# 1. Kernel
# ============================================================
def kernel_document_version(z):
    return 0.75 * (1 - np.abs(z))**2 * (np.abs(z) <= 1)


def kernel_standard_epanechnikov(z):
    return 0.75 * (1 - z**2) * (np.abs(z) <= 1)


def kernel_epanechnikov(z):
    return kernel_standard_epanechnikov(z)


# ============================================================
# 2. Geometric bandwidth cascade constraint
# ============================================================
def build_h_list(h1, a, max_K, min_h=None):
    if not (1 < a <= 2):
        raise ValueError(f"a must satisfy 1 < a <= 2, but got a={a}")

    h_list = []
    for k in range(max_K):
        h = h1 / (a**k)
        if min_h is not None and h < min_h:
            break
        h_list.append(h)

    if len(h_list) >= 2:
        ratios = np.array([h_list[i] / h_list[i + 1] for i in range(len(h_list) - 1)])
        if not np.all((ratios > 1) & (ratios <= 2)):
            raise ValueError("Invalid bandwidth cascade: all h_k / h_{k+1} must be in (1, 2].")

    return h_list


def check_geometric_cascade(h_list, a, atol=1e-10):
    if not (1 < a <= 2):
        return False
    if len(h_list) < 2:
        return True
    ratios = np.array([h_list[i] / h_list[i + 1] for i in range(len(h_list) - 1)])
    return bool(np.all(np.abs(ratios - a) <= atol) and np.all((ratios > 1) & (ratios <= 2)))


def bandwidth_ratios(h_list):
    if len(h_list) < 2:
        return []
    return [h_list[i] / h_list[i + 1] for i in range(len(h_list) - 1)]


# ============================================================
# 3. Periodic distance
# ============================================================
def periodic_distance(t, u):
    return ((t - u + 0.5) % 1.0) - 0.5


# ============================================================
# 4. Contrast functions and local contrast smoother
# ============================================================
def rho_l2(y):
    return 0.5 * y**2


def rho_huber(y, H=1.0):
    abs_y = np.abs(y)
    return np.where(abs_y <= H, 0.5 * y**2, H * abs_y - 0.5 * H**2)


def rho_smooth_l1_document(y, H=1.0):
    z = y / H
    return y * (2 * ndtr(z) - 1) + (2 * H / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z**2)


def weighted_local_contrast_objective(x0, values, weights, rho_type="l2", H=1.0):
    r = values - x0

    if rho_type == "l2":
        return np.sum(weights * rho_l2(r))
    if rho_type == "huber":
        return np.sum(weights * rho_huber(r, H=H))
    if rho_type == "smooth_l1":
        return np.sum(weights * rho_smooth_l1_document(r, H=H))

    raise ValueError("rho_type must be one of: 'l2', 'huber', 'smooth_l1'.")


def weighted_local_robust_fit(values, weights, rho_type="l2", H=1.0):
    weight_sum = np.sum(weights)

    if weight_sum <= 1e-14:
        return 0.0

    if rho_type == "l2":
        return np.sum(weights * values) / weight_sum

    lo = np.min(values) - 5.0 * H
    hi = np.max(values) + 5.0 * H

    result = minimize_scalar(
        weighted_local_contrast_objective,
        bounds=(lo, hi),
        method="bounded",
        args=(values, weights, rho_type, H),
        options={"xatol": 1e-8},
    )

    if not result.success:
        return np.sum(weights * values) / weight_sum

    return result.x


def local_contrast_smooth_periodic(t_grid, Z, h, rho_type="l2", H=1.0):
    S = np.zeros_like(Z)

    for i, t in enumerate(t_grid):
        d = periodic_distance(t, t_grid)
        w = kernel_epanechnikov(d / h) / h
        S[i] = weighted_local_robust_fit(values=Z, weights=w, rho_type=rho_type, H=H)

    return S


def local_mean_smooth_periodic(t_grid, Z, h):
    return local_contrast_smooth_periodic(t_grid=t_grid, Z=Z, h=h, rho_type="l2", H=1.0)


# ============================================================
# 5. Linear operators and theory diagnostics
# ============================================================
def build_smoothing_matrix_periodic(t_grid, h):
    n = len(t_grid)
    W = np.zeros((n, n))

    for i, t in enumerate(t_grid):
        d = periodic_distance(t, t_grid)
        w = kernel_epanechnikov(d / h) / h
        s = np.sum(w)

        if s > 1e-14:
            W[i, :] = w / s

    return W


def compute_noise_propagation_trace(t_grid, h_list):
    n = len(t_grid)
    I = np.eye(n)
    A = np.eye(n)

    trace_list = []
    W_list = []
    A_list = []

    for h in h_list:
        W = build_smoothing_matrix_periodic(t_grid, h)
        A = (I - W) @ A

        W_list.append(W)
        A_list.append(A.copy())
        trace_list.append(float(np.trace(A @ A.T)))

    return {
        "trace_AAT": trace_list,
        "W_list": W_list,
        "A_list": A_list,
    }


# ============================================================
# 6. Spectral tools
# ============================================================
def spectral_centroid(signal, dx):
    signal = signal - np.mean(signal)
    fft = np.fft.rfft(signal)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(len(signal), d=dx)

    if np.sum(power) == 0:
        return 0.0

    return np.sum(freqs * power) / np.sum(power)


def spectral_bandwidth(signal, dx):
    signal = signal - np.mean(signal)
    fft = np.fft.rfft(signal)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(len(signal), d=dx)

    if np.sum(power) == 0:
        return 0.0

    center = np.sum(freqs * power) / np.sum(power)
    return np.sum((freqs - center) ** 2 * power) / np.sum(power)


def spectral_flatness(signal):
    signal = signal - np.mean(signal)
    power = np.abs(np.fft.rfft(signal)) ** 2
    power = power[1:]
    power = power + 1e-12
    return float(np.exp(np.mean(np.log(power))) / np.mean(power))


def autocorrelation(x, max_lag=30):
    x = np.asarray(x) - np.mean(x)
    den = np.dot(x, x) + 1e-12
    ac = []
    for lag in range(1, max_lag + 1):
        ac.append(float(np.dot(x[:-lag], x[lag:]) / den))
    return np.array(ac)


def residual_autocorrelation_score(residual, max_lag=30):
    ac = autocorrelation(residual, max_lag=max_lag)
    return float(np.mean(ac**2))


# ============================================================
# 7. IRMF residual cascade
# ============================================================
def intrinsic_multiscale_filtering_autoK_cascade(
    Y,
    t_grid,
    h1,
    a,
    min_K=1,
    max_K=10,
    min_h=None,
    centroid_gain_tol=0.01,
    bandwidth_gain_tol=0.01,
    relative_gain_tol=0.001,
    patience=1,
    rho_type="l2",
    H=1.0,
    compute_theory_diagnostics=False,
    verbose=False,
):
    residual = Y.copy()
    components = []
    residuals = []

    centroid_list = []
    bandwidth_list = []
    centroid_gain_list = []
    bandwidth_gain_list = []

    if min_h is None:
        min_h = 1 / len(t_grid)

    h_list = build_h_list(h1=h1, a=a, max_K=max_K, min_h=min_h)

    dx = t_grid[1] - t_grid[0]

    prev_centroid = spectral_centroid(residual, dx=dx)
    prev_bandwidth = spectral_bandwidth(residual, dx=dx)

    for k, h in enumerate(h_list):
        S = local_contrast_smooth_periodic(t_grid=t_grid, Z=residual, h=h, rho_type=rho_type, H=H)
        residual = residual - S

        curr_centroid = spectral_centroid(residual, dx=dx)
        curr_bandwidth = spectral_bandwidth(residual, dx=dx)

        centroid_gain = curr_centroid - prev_centroid
        bandwidth_gain = curr_bandwidth - prev_bandwidth

        components.append(S)
        residuals.append(residual.copy())

        centroid_list.append(curr_centroid)
        bandwidth_list.append(curr_bandwidth)
        centroid_gain_list.append(centroid_gain)
        bandwidth_gain_list.append(bandwidth_gain)

        if verbose:
            print(
                f"k={k + 1}, h={h:.5f}, "
                f"centroid={curr_centroid:.6f}, "
                f"centroid_gain={centroid_gain:.6f}, "
                f"bandwidth={curr_bandwidth:.6f}, "
                f"bandwidth_gain={bandwidth_gain:.6f}"
            )

        prev_centroid = curr_centroid
        prev_bandwidth = curr_bandwidth

    diagnostics = {
        "centroid_list": centroid_list,
        "bandwidth_list": bandwidth_list,
        "centroid_gain_list": centroid_gain_list,
        "bandwidth_gain_list": bandwidth_gain_list,
        "noise_propagation": None,
        "geometric_constraint_ok": check_geometric_cascade(h_list, a),
        "bandwidth_ratios": bandwidth_ratios(h_list),
    }

    if compute_theory_diagnostics and rho_type == "l2":
        diagnostics["noise_propagation"] = compute_noise_propagation_trace(t_grid=t_grid, h_list=h_list)

    return components, residuals, residual, h_list, diagnostics


def intrinsic_multiscale_filtering_l2_autoK_cascade(
    Y,
    t_grid,
    h1,
    a,
    min_K=1,
    max_K=10,
    min_h=None,
    centroid_gain_tol=0.01,
    bandwidth_gain_tol=0.01,
    relative_gain_tol=0.001,
    patience=1,
    verbose=False,
):
    return intrinsic_multiscale_filtering_autoK_cascade(
        Y=Y,
        t_grid=t_grid,
        h1=h1,
        a=a,
        min_K=min_K,
        max_K=max_K,
        min_h=min_h,
        centroid_gain_tol=centroid_gain_tol,
        bandwidth_gain_tol=bandwidth_gain_tol,
        relative_gain_tol=relative_gain_tol,
        patience=patience,
        rho_type="l2",
        H=1.0,
        compute_theory_diagnostics=False,
        verbose=verbose,
    )


# ============================================================
# 8. Unsupervised metrics
# ============================================================
def orthogonality_index(components):
    total = 0.0
    count = 0

    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            num = np.dot(components[i], components[j])
            den = np.linalg.norm(components[i]) * np.linalg.norm(components[j])

            if den > 0:
                total += abs(num / den)
                count += 1

    return total / count if count > 0 else 0.0


def component_energy(signal):
    return np.sum(signal**2)


def reconstruction_error(Y, components, residual):
    reconstruction = np.sum(components, axis=0) + residual
    return np.max(np.abs(Y - reconstruction))


def evaluate_unsupervised_metrics(components, residual, Y, x, residual_ac_max_lag=30):
    dx = x[1] - x[0]

    avg_component_bandwidth = np.mean([spectral_bandwidth(c, dx=dx) for c in components])
    ortho = orthogonality_index(components)
    residual_bw = spectral_bandwidth(residual, dx=dx)
    residual_flat = spectral_flatness(residual)
    residual_ac = residual_autocorrelation_score(residual, max_lag=residual_ac_max_lag)
    recon_error = reconstruction_error(Y, components, residual)

    return {
        "component_bandwidth": avg_component_bandwidth,
        "orthogonality": ortho,
        "residual_bandwidth": residual_bw,
        "residual_flatness": residual_flat,
        "residual_ac_score": residual_ac,
        "reconstruction_error": recon_error,
        "K": len(components),
        "component_energies": np.array([component_energy(c) for c in components]),
    }


# ============================================================
# 9. Optional simulation recovery check
# ============================================================
def l2_error_relative(est, true, eps=1e-12):
    return np.linalg.norm(est - true) / (np.linalg.norm(true) + eps)


def sup_error_relative(est, true, eps=1e-12):
    return np.max(np.abs(est - true)) / (np.max(np.abs(true)) + eps)


def compute_population_proxy_from_clean_signal(
    signal_clean,
    x,
    h1,
    a,
    max_K,
    min_h,
    rho_type="l2",
    H=1.0,
):
    return intrinsic_multiscale_filtering_autoK_cascade(
        Y=signal_clean,
        t_grid=x,
        h1=h1,
        a=a,
        min_K=1,
        max_K=max_K,
        min_h=min_h,
        rho_type=rho_type,
        H=H,
        compute_theory_diagnostics=False,
        verbose=False,
    )


def attach_simulation_recovery_check(result, signal_clean, x, rho_type="l2", H=1.0):
    min_h = 1 / len(x)
    pop_components, pop_residuals, pop_residual, pop_h_list, pop_diag = compute_population_proxy_from_clean_signal(
        signal_clean=signal_clean,
        x=x,
        h1=result["h1"],
        a=result["a"],
        max_K=len(result["h_list"]),
        min_h=min_h,
        rho_type=rho_type,
        H=H,
    )

    K_common = min(len(result["components"]), len(pop_components))
    l2_errors = []
    sup_errors = []

    for k in range(K_common):
        l2_errors.append(l2_error_relative(result["components"][k], pop_components[k]))
        sup_errors.append(sup_error_relative(result["components"][k], pop_components[k]))

    result["component_l2_recovery_errors"] = l2_errors
    result["component_sup_recovery_errors"] = sup_errors
    result["mean_component_l2_recovery_error"] = float(np.mean(l2_errors)) if l2_errors else np.inf
    result["mean_component_sup_recovery_error"] = float(np.mean(sup_errors)) if sup_errors else np.inf
    result["max_component_l2_recovery_error"] = float(np.max(l2_errors)) if l2_errors else np.inf
    result["max_component_sup_recovery_error"] = float(np.max(sup_errors)) if sup_errors else np.inf
    result["residual_l2_recovery_error"] = l2_error_relative(result["residual"], pop_residual)
    result["residual_sup_recovery_error"] = sup_error_relative(result["residual"], pop_residual)
    result["population_proxy_components"] = pop_components
    result["population_proxy_residual"] = pop_residual
    result["population_proxy_residuals"] = pop_residuals

    return result


# ============================================================
# 10. AM-FM validation metrics
# ============================================================
def instantaneous_frequency(signal, x, pad_ratio=0.5):
    n = len(signal)
    pad = int(pad_ratio * n)

    if pad < 2:
        pad = 2

    left = signal[1:pad + 1][::-1]
    right = signal[-pad - 1:-1][::-1]
    signal_ext = np.concatenate([left, signal, right])

    dx = x[1] - x[0]
    x_ext = np.arange(len(signal_ext)) * dx

    analytic = hilbert(signal_ext)
    phase = np.unwrap(np.angle(analytic))
    freq_ext = np.gradient(phase, x_ext) / (2 * np.pi)

    return freq_ext[pad:pad + n]


def amplitude_envelope(signal, pad_ratio=0.5):
    n = len(signal)
    pad = int(pad_ratio * n)

    if pad < 2:
        pad = 2

    left = signal[1:pad + 1][::-1]
    right = signal[-pad - 1:-1][::-1]
    signal_ext = np.concatenate([left, signal, right])
    analytic = hilbert(signal_ext)
    envelope_ext = np.abs(analytic)

    return envelope_ext[pad:pad + n]


def generate_adjacent_component_candidates(components, max_merge_len=3, energy_threshold_ratio=0.02):
    K = len(components)
    comp_energies = np.array([component_energy(c) for c in components])

    if K == 0:
        return [], comp_energies

    max_energy = np.max(comp_energies)
    candidates = []

    for start in range(K):
        merged = np.zeros_like(components[0])

        for length in range(1, max_merge_len + 1):
            end = start + length

            if end > K:
                break

            merged = merged + components[end - 1]
            candidate_energy = component_energy(merged)

            if candidate_energy >= energy_threshold_ratio * max_energy:
                candidates.append(
                    {
                        "signal": merged.copy(),
                        "start_index": start,
                        "end_index": end - 1,
                        "component_indices": list(range(start, end)),
                        "component_numbers": list(range(start + 1, end + 1)),
                        "component_range": f"{start + 1}-{end}",
                        "merge_len": length,
                        "energy": candidate_energy,
                    }
                )

    if len(candidates) == 0:
        for k, c in enumerate(components):
            candidates.append(
                {
                    "signal": c.copy(),
                    "start_index": k,
                    "end_index": k,
                    "component_indices": [k],
                    "component_numbers": [k + 1],
                    "component_range": f"{k + 1}",
                    "merge_len": 1,
                    "energy": component_energy(c),
                }
            )

    return candidates, comp_energies


def amfm_component_error(component, x, true_if, true_amp=None, trim=30, w_if=1.0, w_amp=1.0):
    f_hat = instantaneous_frequency(component, x)

    f_hat_inner = f_hat[trim:-trim]
    true_if_inner = true_if[trim:-trim]

    if_error = np.sqrt(np.mean((f_hat_inner - true_if_inner) ** 2))
    if_error_rel = if_error / (np.mean(true_if_inner) + 1e-12)

    if true_amp is None:
        return if_error_rel, if_error_rel, None, f_hat, None, 1.0

    amp_hat = amplitude_envelope(component)

    amp_hat_inner = amp_hat[trim:-trim]
    true_amp_inner = true_amp[trim:-trim]

    amp_scale = np.dot(amp_hat_inner, true_amp_inner) / (np.dot(amp_hat_inner, amp_hat_inner) + 1e-12)
    amp_hat_scaled = amp_scale * amp_hat
    amp_hat_scaled_inner = amp_hat_scaled[trim:-trim]

    amp_error = np.sqrt(np.mean((amp_hat_scaled_inner - true_amp_inner) ** 2)) / (
        np.mean(true_amp_inner) + 1e-12
    )

    total_error = w_if * if_error_rel + w_amp * amp_error

    return total_error, if_error_rel, amp_error, f_hat, amp_hat_scaled, amp_scale


def amfm_matching_score(
    components,
    x,
    true_ifs,
    true_amps=None,
    energy_threshold_ratio=0.02,
    unmatched_penalty=1.0,
    extra_candidate_penalty=0.25,
    merge_penalty=0.10,
    max_merge_len=3,
    trim=30,
    w_if=1.0,
    w_amp=1.0,
):
    candidates, comp_energies = generate_adjacent_component_candidates(
        components=components,
        max_merge_len=max_merge_len,
        energy_threshold_ratio=energy_threshold_ratio,
    )

    if len(candidates) == 0:
        return {
            "amfm_matching_score": np.inf,
            "matched_pairs": [],
            "component_energies": comp_energies,
            "candidate_infos": [],
            "best_candidate_signal": None,
            "best_candidate_indices": None,
            "best_component_index": None,
            "best_estimated_freq": None,
            "best_estimated_amp": None,
            "best_amp_scale": None,
            "all_amfm_errors": [],
            "all_if_errors": [],
            "all_amp_errors": [],
            "num_signal_components": 0,
        }

    if true_amps is None:
        true_amps = [None] * len(true_ifs)

    cost = np.zeros((len(candidates), len(true_ifs)))
    raw_amfm_matrix = np.zeros_like(cost)
    merge_penalty_matrix = np.zeros_like(cost)
    if_error_matrix = np.zeros_like(cost)
    amp_error_matrix = np.zeros_like(cost)

    estimated_freqs = {}
    estimated_amps = {}
    amp_scales = {}

    for i, cand in enumerate(candidates):
        candidate_signal = cand["signal"]

        for j in range(len(true_ifs)):
            raw_err, if_err, amp_err, f_hat, amp_hat_scaled, amp_scale = amfm_component_error(
                component=candidate_signal,
                x=x,
                true_if=true_ifs[j],
                true_amp=true_amps[j],
                trim=trim,
                w_if=w_if,
                w_amp=w_amp,
            )

            cand_merge_penalty = merge_penalty * (cand["merge_len"] - 1)
            total_err_with_penalty = raw_err + cand_merge_penalty

            cost[i, j] = total_err_with_penalty
            raw_amfm_matrix[i, j] = raw_err
            merge_penalty_matrix[i, j] = cand_merge_penalty
            if_error_matrix[i, j] = if_err
            amp_error_matrix[i, j] = amp_err if amp_err is not None else 0.0

            estimated_freqs[(i, j)] = f_hat
            estimated_amps[(i, j)] = amp_hat_scaled
            amp_scales[(i, j)] = amp_scale

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_costs = cost[row_ind, col_ind]
    mean_match_cost = np.mean(matched_costs)

    num_unmatched_true = max(0, len(true_ifs) - len(row_ind))
    num_extra_candidates = max(0, len(candidates) - len(true_ifs))

    score = mean_match_cost + unmatched_penalty * num_unmatched_true + extra_candidate_penalty * num_extra_candidates

    best_pair_local = int(np.argmin(matched_costs))
    best_row = row_ind[best_pair_local]
    best_col = col_ind[best_pair_local]
    best_candidate = candidates[best_row]

    best_candidate_signal = best_candidate["signal"]
    best_candidate_indices = best_candidate["component_indices"]
    best_component_index = best_candidate["component_indices"][0]

    best_estimated_freq = estimated_freqs[(best_row, best_col)]
    best_estimated_amp = estimated_amps[(best_row, best_col)]
    best_amp_scale = amp_scales[(best_row, best_col)]

    matched_pairs = []

    for r, c in zip(row_ind, col_ind):
        cand = candidates[r]

        matched_pairs.append(
            {
                "candidate_index": int(r),
                "component_indices": cand["component_indices"],
                "component_numbers": cand["component_numbers"],
                "component_range": cand["component_range"],
                "merge_len": cand["merge_len"],
                "matched_mode": c + 1,
                "raw_amfm_error": float(raw_amfm_matrix[r, c]),
                "merge_penalty": float(merge_penalty_matrix[r, c]),
                "amfm_error_with_merge_penalty": float(cost[r, c]),
                "if_error": float(if_error_matrix[r, c]),
                "amp_error": float(amp_error_matrix[r, c]),
                "amp_scale": float(amp_scales[(r, c)]),
                "energy": float(cand["energy"]),
            }
        )

    candidate_infos = []

    for i, cand in enumerate(candidates):
        best_mode_for_candidate = int(np.argmin(cost[i, :]))

        raw_err = raw_amfm_matrix[i, best_mode_for_candidate]
        merge_p = merge_penalty_matrix[i, best_mode_for_candidate]
        total_err = cost[i, best_mode_for_candidate]

        candidate_infos.append(
            {
                "candidate_index": i,
                "component_indices": cand["component_indices"],
                "component_numbers": cand["component_numbers"],
                "component_range": cand["component_range"],
                "merge_len": cand["merge_len"],
                "energy": float(cand["energy"]),
                "raw_amfm_error": float(raw_err),
                "merge_penalty": float(merge_p),
                "best_amfm_error_with_penalty": float(total_err),
                "best_if_error": float(if_error_matrix[i, best_mode_for_candidate]),
                "best_amp_error": float(amp_error_matrix[i, best_mode_for_candidate]),
            }
        )

    candidate_infos = sorted(candidate_infos, key=lambda d: d["best_amfm_error_with_penalty"])

    all_best_errors = np.min(cost, axis=1)
    all_best_if_errors = np.min(if_error_matrix, axis=1)
    all_best_amp_errors = np.min(amp_error_matrix, axis=1)

    return {
        "amfm_matching_score": score,
        "matched_pairs": matched_pairs,
        "component_energies": comp_energies,
        "candidate_infos": candidate_infos,
        "best_candidate_signal": best_candidate_signal,
        "best_candidate_indices": best_candidate_indices,
        "best_component_index": best_component_index,
        "best_estimated_freq": best_estimated_freq,
        "best_estimated_amp": best_estimated_amp,
        "best_amp_scale": best_amp_scale,
        "all_amfm_errors": all_best_errors,
        "all_if_errors": all_best_if_errors,
        "all_amp_errors": all_best_amp_errors,
        "num_signal_components": len(candidates),
    }


def attach_amfm_validation_metrics(result, x, true_ifs, true_amps):
    amfm = amfm_matching_score(
        components=result["components"],
        x=x,
        true_ifs=true_ifs,
        true_amps=true_amps,
        energy_threshold_ratio=0.02,
        unmatched_penalty=1.0,
        extra_candidate_penalty=0.25,
        merge_penalty=0.10,
        max_merge_len=3,
        trim=30,
        w_if=1.0,
        w_amp=1.0,
    )

    result["amfm_matching_score_validation_only"] = amfm["amfm_matching_score"]
    result["matched_pairs_validation_only"] = amfm["matched_pairs"]
    result["candidate_infos_validation_only"] = amfm["candidate_infos"]
    result["best_candidate_signal_validation_only"] = amfm["best_candidate_signal"]
    result["best_candidate_indices_validation_only"] = amfm["best_candidate_indices"]
    result["best_component_index_validation_only"] = amfm["best_component_index"]
    result["best_estimated_freq_validation_only"] = amfm["best_estimated_freq"]
    result["best_estimated_amp_validation_only"] = amfm["best_estimated_amp"]
    result["best_amp_scale_validation_only"] = amfm["best_amp_scale"]
    result["all_amfm_errors_validation_only"] = amfm["all_amfm_errors"]
    result["all_if_errors_validation_only"] = amfm["all_if_errors"]
    result["all_amp_errors_validation_only"] = amfm["all_amp_errors"]
    result["num_signal_components_validation_only"] = amfm["num_signal_components"]

    return result


# ============================================================
# 11. Standardized unsupervised score with lambda_K
# ============================================================
def add_unsupervised_standardized_scores(raw_results, lambda_K=0.05, weights=None):
    if weights is None:
        weights = {
            "component_bandwidth": 1.0,
            "orthogonality": 1.0,
            "residual_ac_score": 1.0,
            "negative_residual_flatness": 1.0,
        }

    for r in raw_results:
        r["negative_residual_flatness"] = -r["residual_flatness"]

    metric_names = list(weights.keys()) + ["K"]

    means = {}
    stds = {}

    for m in metric_names:
        vals = np.array([r[m] for r in raw_results], dtype=float)
        means[m] = np.mean(vals)
        stds[m] = np.std(vals) + 1e-12

    scored_results = []

    for r in raw_results:
        r_new = r.copy()
        score = 0.0

        for m, w in weights.items():
            z = (r[m] - means[m]) / stds[m]
            r_new[f"z_{m}"] = z
            score += w * z

        z_K = (r["K"] - means["K"]) / stds["K"]
        r_new["z_K"] = z_K
        score += lambda_K * z_K

        r_new["lambda_K"] = lambda_K
        r_new["unsupervised_score"] = score

        scored_results.append(r_new)

    return scored_results


# ============================================================
# 12. Theory-consistent grid search over h1, a, max_K, lambda_K
# ============================================================
def grid_search_h1_a_maxK_lambdaK_theory_consistent(
    Y,
    x,
    h1_candidates,
    a_candidates,
    max_K_candidates,
    lambda_K_candidates,
    min_K=1,
    centroid_gain_tol=0.01,
    bandwidth_gain_tol=0.01,
    relative_gain_tol=0.001,
    patience=1,
    score_weights=None,
    rho_type="l2",
    H=1.0,
):
    raw_results = []

    for h1, a, max_K in product(h1_candidates, a_candidates, max_K_candidates):
        if not (1 < a <= 2):
            continue

        components, residuals, residual, h_list, diagnostics = intrinsic_multiscale_filtering_autoK_cascade(
            Y=Y,
            t_grid=x,
            h1=h1,
            a=a,
            min_K=min_K,
            max_K=max_K,
            min_h=1 / len(x),
            centroid_gain_tol=centroid_gain_tol,
            bandwidth_gain_tol=bandwidth_gain_tol,
            relative_gain_tol=relative_gain_tol,
            patience=patience,
            rho_type=rho_type,
            H=H,
            compute_theory_diagnostics=False,
            verbose=False,
        )

        if len(components) == 0:
            continue

        metrics = evaluate_unsupervised_metrics(
            components=components,
            residual=residual,
            Y=Y,
            x=x,
            residual_ac_max_lag=30,
        )

        raw_results.append(
            {
                "h1": h1,
                "a": a,
                "max_K_candidate": max_K,
                "h_list": h_list,
                "rho_type": rho_type,
                "H": H,
                "geometric_constraint_ok": check_geometric_cascade(h_list, a),
                "bandwidth_ratios": bandwidth_ratios(h_list),
                "components": components,
                "residuals": residuals,
                "residual": residual,
                "diagnostics": diagnostics,
                **metrics,
            }
        )

    all_results = []

    for lambda_K in lambda_K_candidates:
        scored = add_unsupervised_standardized_scores(
            raw_results,
            lambda_K=lambda_K,
            weights=score_weights,
        )
        all_results.extend(scored)

    return sorted(all_results, key=lambda r: r["unsupervised_score"]), raw_results


# ============================================================
# 13. Plot helpers
# ============================================================
def plot_components(x, components, h_list, title):
    K = len(components)
    plt.figure(figsize=(10, 2.5 * K))

    for k, S in enumerate(components, start=1):
        plt.subplot(K, 1, k)
        plt.plot(x, S)
        plt.title(rf"{title}: component $\tilde S^{{({k})}}$, h={h_list[k - 1]:.5f}")
        plt.xlabel("x")
        plt.ylabel("value")

    plt.tight_layout()
    save_current_plot(title + " components")


def plot_residuals(x, residuals, title):
    K = len(residuals)
    plt.figure(figsize=(10, 2.5 * K))

    for k, R in enumerate(residuals, start=1):
        plt.subplot(K, 1, k)
        plt.plot(x, R)
        plt.title(rf"{title}: residual $Y^{{({k + 1})}}$")
        plt.xlabel("x")
        plt.ylabel("value")

    plt.tight_layout()
    save_current_plot(title + " residuals")


def plot_reconstruction(x, Y, components, residual, title):
    reconstruction = np.sum(components, axis=0) + residual

    plt.figure(figsize=(10, 4))
    plt.plot(x, Y, label="Y")
    plt.plot(x, reconstruction, "--", label="reconstruction")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    save_current_plot(title)


def plot_best_candidate_validation(x, result, title):
    plt.figure(figsize=(10, 4))
    plt.plot(x, result["best_candidate_signal_validation_only"])
    nums = [i + 1 for i in result["best_candidate_indices_validation_only"]]
    plt.title(f"{title}: validation-only best merged candidate components {nums}")
    plt.xlabel("x")
    plt.ylabel("value")
    plt.tight_layout()
    save_current_plot(title + " validation only best merged candidate")


def plot_if_validation(x, true_freq, est_freq, title):
    plt.figure(figsize=(10, 4))
    plt.plot(x, true_freq, label="true frequency")
    plt.plot(x, est_freq, "--", label="estimated IF")
    plt.title(title + " [validation only]")
    plt.xlabel("x")
    plt.ylabel("frequency")
    plt.legend()
    plt.tight_layout()
    save_current_plot(title + " validation only IF")


def plot_amplitude_envelope_validation(x, true_amp, est_amp, title):
    plt.figure(figsize=(10, 4))
    plt.plot(x, true_amp, label="true amplitude envelope")
    plt.plot(x, est_amp, "--", label="estimated amplitude envelope, scaled")
    plt.title(title + " [validation only]")
    plt.xlabel("x")
    plt.ylabel("amplitude")
    plt.legend()
    plt.tight_layout()
    save_current_plot(title + " validation only amplitude envelope")


def plot_component_energy(result, title):
    energies = result["component_energies"]

    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(1, len(energies) + 1), energies)
    plt.xlabel("component index")
    plt.ylabel("energy")
    plt.title(title + ": component energy")
    plt.tight_layout()
    save_current_plot(title + " component energy")


def plot_candidate_errors_validation(result, title, top_n=None):
    infos = result["candidate_infos_validation_only"]

    if top_n is not None:
        infos = infos[:top_n]

    labels = [info["component_range"] for info in infos]
    total_errors = [info["best_amfm_error_with_penalty"] for info in infos]
    raw_errors = [info["raw_amfm_error"] for info in infos]
    merge_penalties = [info["merge_penalty"] for info in infos]

    plt.figure(figsize=(12, 4))
    plt.plot(np.arange(len(total_errors)), total_errors, marker="o", label="AM-FM error + merge penalty")
    plt.plot(np.arange(len(raw_errors)), raw_errors, marker="o", linestyle="--", label="raw AM-FM error")
    plt.xticks(np.arange(len(total_errors)), labels, rotation=90)
    plt.xlabel("candidate component range")
    plt.ylabel("error")
    plt.title(title + ": validation-only adjacent-merge candidate errors")
    plt.legend()
    plt.tight_layout()
    save_current_plot(title + " validation only adjacent merge candidate errors")

    plt.figure(figsize=(12, 4))
    plt.bar(np.arange(len(merge_penalties)), merge_penalties)
    plt.xticks(np.arange(len(merge_penalties)), labels, rotation=90)
    plt.xlabel("candidate component range")
    plt.ylabel("merge penalty")
    plt.title(title + ": validation-only candidate merge penalty")
    plt.tight_layout()
    save_current_plot(title + " validation only candidate merge penalty")


def plot_cascade_diagnostics(diagnostics, title):
    centroid_list = diagnostics["centroid_list"]
    bandwidth_list = diagnostics["bandwidth_list"]
    centroid_gain_list = diagnostics["centroid_gain_list"]
    bandwidth_gain_list = diagnostics["bandwidth_gain_list"]
    k_grid = np.arange(1, len(centroid_list) + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(k_grid, centroid_list, marker="o")
    plt.title(title + ": residual spectral centroid")
    plt.xlabel("k")
    plt.ylabel("centroid")
    plt.tight_layout()
    save_current_plot(title + " residual spectral centroid")

    plt.figure(figsize=(10, 4))
    plt.plot(k_grid, centroid_gain_list, marker="o", label="centroid gain")
    plt.plot(k_grid, bandwidth_gain_list, marker="o", label="bandwidth gain")
    plt.axhline(0, linestyle="--")
    plt.title(title + ": cascade gains")
    plt.xlabel("k")
    plt.ylabel("gain")
    plt.legend()
    plt.tight_layout()
    save_current_plot(title + " cascade gains")

    plt.figure(figsize=(10, 4))
    plt.plot(k_grid, bandwidth_list, marker="o")
    plt.title(title + ": residual spectral bandwidth")
    plt.xlabel("k")
    plt.ylabel("bandwidth")
    plt.tight_layout()
    save_current_plot(title + " residual spectral bandwidth")


def plot_noise_propagation_trace(result, title):
    noise_diag = result["diagnostics"].get("noise_propagation", None)

    if noise_diag is None:
        print(f"Skip noise propagation trace for {title}: available only for rho_type='l2'.")
        return

    trace_AAT = np.array(noise_diag["trace_AAT"])
    k_grid = np.arange(1, len(trace_AAT) + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(k_grid, trace_AAT, marker="o")
    plt.title(title + r": theory diagnostic $\mathrm{tr}(A^{(k)}A^{(k)\top})$")
    plt.xlabel("k")
    plt.ylabel(r"$\mathrm{tr}(A^{(k)}A^{(k)\top})$")
    plt.tight_layout()
    save_current_plot(title + " theory noise propagation trace")

    plt.figure(figsize=(10, 4))
    plt.plot(k_grid, trace_AAT / len(result["residual"]), marker="o")
    plt.title(title + r": normalized noise propagation trace")
    plt.xlabel("k")
    plt.ylabel(r"$\mathrm{tr}(A^{(k)}A^{(k)\top}) / n$")
    plt.tight_layout()
    save_current_plot(title + " normalized theory noise propagation trace")


def plot_residual_whiteness(result, title):
    residual = result["residual"]
    ac = autocorrelation(residual, max_lag=30)

    plt.figure(figsize=(10, 4))
    plt.stem(np.arange(1, len(ac) + 1), ac)
    plt.axhline(0, linestyle="--")
    plt.title(title + ": residual autocorrelation")
    plt.xlabel("lag")
    plt.ylabel("autocorrelation")
    plt.tight_layout()
    save_current_plot(title + " residual autocorrelation")


def plot_simulation_recovery_components(x, result, title):
    if "population_proxy_components" not in result:
        return

    K = min(len(result["components"]), len(result["population_proxy_components"]))
    plt.figure(figsize=(10, 2.5 * K))

    for k in range(K):
        plt.subplot(K, 1, k + 1)
        plt.plot(x, result["population_proxy_components"][k], label=rf"population proxy $S^{{({k + 1})}}$")
        plt.plot(x, result["components"][k], "--", label=rf"estimated $\tilde S^{{({k + 1})}}$")
        plt.title(rf"{title}: optional simulation recovery check component {k + 1}")
        plt.xlabel("x")
        plt.ylabel("value")
        plt.legend()

    plt.tight_layout()
    save_current_plot(title + " optional simulation recovery components")


def plot_simulation_recovery_errors(result, title):
    if "component_l2_recovery_errors" not in result:
        return

    l2_errors = result["component_l2_recovery_errors"]
    sup_errors = result["component_sup_recovery_errors"]
    k_grid = np.arange(1, len(l2_errors) + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(k_grid, l2_errors, marker="o", label="relative L2 error")
    plt.plot(k_grid, sup_errors, marker="o", linestyle="--", label="relative sup error")
    plt.title(title + ": optional simulation recovery errors")
    plt.xlabel("component index")
    plt.ylabel("relative error")
    plt.legend()
    plt.tight_layout()
    save_current_plot(title + " optional simulation recovery errors")


def print_candidate_ranking_validation(result, title, top_n=10):
    print("\n" + "-" * 100)
    print(f"TOP {top_n} AM-FM VALIDATION CANDIDATES: {title}")
    print("-" * 100)
    print(
        f"{'rank':>4} | {'components':>10} | {'merge_len':>9} | "
        f"{'raw_err':>10} | {'merge_pen':>10} | {'total_err':>10} | "
        f"{'IF_err':>10} | {'AMP_err':>10} | {'energy':>10}"
    )
    print("-" * 100)

    for rank, info in enumerate(result["candidate_infos_validation_only"][:top_n], start=1):
        print(
            f"{rank:>4} | "
            f"{info['component_range']:>10} | "
            f"{info['merge_len']:>9} | "
            f"{info['raw_amfm_error']:>10.6f} | "
            f"{info['merge_penalty']:>10.6f} | "
            f"{info['best_amfm_error_with_penalty']:>10.6f} | "
            f"{info['best_if_error']:>10.6f} | "
            f"{info['best_amp_error']:>10.6f} | "
            f"{info['energy']:>10.6f}"
        )


def print_theory_consistent_summary(result, title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print("rho_type:", result["rho_type"], "H:", result["H"])
    print("Selected K:", result["K"])
    print("max_K candidate:", result.get("max_K_candidate", None))
    print("lambda_K:", result["lambda_K"])
    print("z_K:", result["z_K"])
    print("h_list:", [round(float(h), 5) for h in result["h_list"]])
    print("h1:", round(float(result["h1"]), 5), "a:", round(float(result["a"]), 5))
    print("constraint 1 < a <= 2:", 1 < result["a"] <= 2)
    print("geometric constraint ok:", result.get("geometric_constraint_ok", None))
    print("h_k / h_{k+1} ratios:", [round(float(r), 6) for r in result.get("bandwidth_ratios", [])])
    print("unsupervised score:", result["unsupervised_score"])
    print("component bandwidth:", result["component_bandwidth"])
    print("orthogonality:", result["orthogonality"])
    print("residual bandwidth:", result["residual_bandwidth"])
    print("residual flatness:", result["residual_flatness"])
    print("residual autocorrelation score:", result["residual_ac_score"])
    print("reconstruction error:", result["reconstruction_error"])

    if "mean_component_l2_recovery_error" in result:
        print("\nOptional simulation recovery check, not used for selection:")
        print("mean component L2 recovery error:", result["mean_component_l2_recovery_error"])
        print("mean component sup recovery error:", result["mean_component_sup_recovery_error"])
        print("max component L2 recovery error:", result["max_component_l2_recovery_error"])
        print("max component sup recovery error:", result["max_component_sup_recovery_error"])
        print("residual L2 recovery error:", result["residual_l2_recovery_error"])
        print("residual sup recovery error:", result["residual_sup_recovery_error"])

    noise_diag = result["diagnostics"].get("noise_propagation", None)
    if noise_diag is not None:
        print("\ntheory noise trace tr(A A^T):", [round(v, 4) for v in noise_diag["trace_AAT"]])
        print("normalized theory noise trace:", [round(v / len(result["residual"]), 6) for v in noise_diag["trace_AAT"]])
    else:
        print("\ntheory noise trace: not computed because rho_type is nonlinear robust contrast")

    print("\nAM-FM validation only, not used for model selection:")
    print("AM-FM matching score:", result["amfm_matching_score_validation_only"])
    print("best validation candidate component indices:", result["best_candidate_indices_validation_only"])
    print("best validation candidate component numbers:", [i + 1 for i in result["best_candidate_indices_validation_only"]])
    print("best validation amplitude scale:", result["best_amp_scale_validation_only"])

    print("\nMatched AM-FM validation pairs:")
    for pair in result["matched_pairs_validation_only"]:
        print(
            f"  components {pair['component_range']} -> mode {pair['matched_mode']}: "
            f"merge_len={pair['merge_len']}, "
            f"raw AMFM error={pair['raw_amfm_error']:.6f}, "
            f"merge penalty={pair['merge_penalty']:.6f}, "
            f"AMFM error+penalty={pair['amfm_error_with_merge_penalty']:.6f}, "
            f"IF error={pair['if_error']:.6f}, "
            f"AMP error={pair['amp_error']:.6f}, "
            f"AMP scale={pair['amp_scale']:.6f}, "
            f"energy={pair['energy']:.6f}"
        )

    print_candidate_ranking_validation(result, title, top_n=10)


# ============================================================
# 14. Signal: AM-FM chirp
# ============================================================
n = 500
x = np.arange(n) / n

true_amp = 1 + 0.5 * np.sin(4 * np.pi * x)
true_freq = 6 + 24 * x

signal_clean = true_amp * np.sin(2 * np.pi * (6 * x + 12 * x**2))

true_ifs = [true_freq]
true_amps = [true_amp]

np.random.seed(0)
eps = np.random.randn(n)

noise_sigma = 0.05
Y = signal_clean + noise_sigma * eps


# ============================================================
# 15. Main configuration
# ============================================================
rho_type = "l2"
H = 1.0 * noise_sigma

h1_candidates = np.linspace(0.04, 0.28, 50)
a_candidates = np.linspace(1.05, 2.0, 50)

max_K_candidates = [4, 5, 6, 7, 8, 9, 10]

lambda_K_candidates = [0.0, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0]

score_weights = {
    "component_bandwidth": 1.0,
    "orthogonality": 1.0,
    "residual_ac_score": 1.0,
    "negative_residual_flatness": 1.0,
}

RUN_SIMULATION_RECOVERY_CHECK = True


# ============================================================
# 16. Theory-consistent grid search
# ============================================================
grid_results, raw_results = grid_search_h1_a_maxK_lambdaK_theory_consistent(
    Y=Y,
    x=x,
    h1_candidates=h1_candidates,
    a_candidates=a_candidates,
    max_K_candidates=max_K_candidates,
    lambda_K_candidates=lambda_K_candidates,
    min_K=1,
    centroid_gain_tol=0.01,
    bandwidth_gain_tol=0.01,
    relative_gain_tol=0.001,
    patience=1,
    score_weights=score_weights,
    rho_type=rho_type,
    H=H,
)


# ============================================================
# 17. Model selection without using AM-FM truth or clean signal
# ============================================================
best_theory_score = min(grid_results, key=lambda r: r["unsupervised_score"])

best_residual_whiteness = min(grid_results, key=lambda r: r["residual_ac_score"])

if rho_type == "l2":
    best_theory_score["diagnostics"]["noise_propagation"] = compute_noise_propagation_trace(
        t_grid=x,
        h_list=best_theory_score["h_list"],
    )
    best_residual_whiteness["diagnostics"]["noise_propagation"] = compute_noise_propagation_trace(
        t_grid=x,
        h_list=best_residual_whiteness["h_list"],
    )

if RUN_SIMULATION_RECOVERY_CHECK:
    best_theory_score = attach_simulation_recovery_check(
        best_theory_score,
        signal_clean=signal_clean,
        x=x,
        rho_type=rho_type,
        H=H,
    )
    best_residual_whiteness = attach_simulation_recovery_check(
        best_residual_whiteness,
        signal_clean=signal_clean,
        x=x,
        rho_type=rho_type,
        H=H,
    )

best_theory_score = attach_amfm_validation_metrics(
    best_theory_score,
    x=x,
    true_ifs=true_ifs,
    true_amps=true_amps,
)

best_residual_whiteness = attach_amfm_validation_metrics(
    best_residual_whiteness,
    x=x,
    true_ifs=true_ifs,
    true_amps=true_amps,
)

print_theory_consistent_summary(
    best_theory_score,
    "THEORY-CONSISTENT BEST BY UNSUPERVISED STANDARDIZED SCORE",
)

print_theory_consistent_summary(
    best_residual_whiteness,
    "REFERENCE: BEST BY RESIDUAL WHITENESS ONLY",
)


# ============================================================
# 18. Plot theory-consistent best
# ============================================================
plot_components(
    x,
    best_theory_score["components"],
    best_theory_score["h_list"],
    "Theory-consistent best by unsupervised score",
)

plot_residuals(
    x,
    best_theory_score["residuals"],
    "Theory-consistent best by unsupervised score",
)

plot_reconstruction(
    x,
    Y,
    best_theory_score["components"],
    best_theory_score["residual"],
    "Reconstruction: theory-consistent best",
)

plot_component_energy(
    best_theory_score,
    "Theory-consistent best",
)

plot_cascade_diagnostics(
    best_theory_score["diagnostics"],
    "Diagnostics: theory-consistent best",
)

plot_noise_propagation_trace(
    best_theory_score,
    "Diagnostics: theory-consistent best",
)

plot_residual_whiteness(
    best_theory_score,
    "Diagnostics: theory-consistent best",
)

if RUN_SIMULATION_RECOVERY_CHECK:
    plot_simulation_recovery_components(
        x,
        best_theory_score,
        "Theory-consistent best",
    )
    plot_simulation_recovery_errors(
        best_theory_score,
        "Theory-consistent best",
    )

plot_best_candidate_validation(
    x,
    best_theory_score,
    "Theory-consistent best",
)

plot_if_validation(
    x,
    true_freq,
    best_theory_score["best_estimated_freq_validation_only"],
    "IF: theory-consistent best",
)

plot_amplitude_envelope_validation(
    x,
    true_amp,
    best_theory_score["best_estimated_amp_validation_only"],
    "Amplitude envelope: theory-consistent best",
)

plot_candidate_errors_validation(
    best_theory_score,
    "Theory-consistent best",
    top_n=15,
)


# ============================================================
# 19. Plot reference: best by residual whiteness only
# ============================================================
plot_components(
    x,
    best_residual_whiteness["components"],
    best_residual_whiteness["h_list"],
    "Reference best by residual whiteness only",
)

plot_residuals(
    x,
    best_residual_whiteness["residuals"],
    "Reference best by residual whiteness only",
)

plot_reconstruction(
    x,
    Y,
    best_residual_whiteness["components"],
    best_residual_whiteness["residual"],
    "Reconstruction: reference best by residual whiteness only",
)

plot_component_energy(
    best_residual_whiteness,
    "Reference best by residual whiteness only",
)

plot_cascade_diagnostics(
    best_residual_whiteness["diagnostics"],
    "Diagnostics: reference best by residual whiteness only",
)

plot_noise_propagation_trace(
    best_residual_whiteness,
    "Diagnostics: reference best by residual whiteness only",
)

plot_residual_whiteness(
    best_residual_whiteness,
    "Diagnostics: reference best by residual whiteness only",
)

if RUN_SIMULATION_RECOVERY_CHECK:
    plot_simulation_recovery_components(
        x,
        best_residual_whiteness,
        "Reference best by residual whiteness only",
    )
    plot_simulation_recovery_errors(
        best_residual_whiteness,
        "Reference best by residual whiteness only",
    )

plot_best_candidate_validation(
    x,
    best_residual_whiteness,
    "Reference best by residual whiteness only",
)

plot_if_validation(
    x,
    true_freq,
    best_residual_whiteness["best_estimated_freq_validation_only"],
    "IF: reference best by residual whiteness only",
)

plot_amplitude_envelope_validation(
    x,
    true_amp,
    best_residual_whiteness["best_estimated_amp_validation_only"],
    "Amplitude envelope: reference best by residual whiteness only",
)

plot_candidate_errors_validation(
    best_residual_whiteness,
    "Reference best by residual whiteness only",
    top_n=15,
)


# ============================================================
# 20. 2D heatmaps from theory-consistent grid search
# ============================================================
h1_grid = np.array(sorted(set([r["h1"] for r in raw_results])))
a_grid = np.array(sorted(set([r["a"] for r in raw_results])))

best_lambda_K = best_theory_score["lambda_K"]
score_heatmap_results = [r for r in grid_results if r["lambda_K"] == best_lambda_K]

score_matrix = np.full((len(a_grid), len(h1_grid)), np.nan)
K_matrix = np.full((len(a_grid), len(h1_grid)), np.nan)
maxK_matrix = np.full((len(a_grid), len(h1_grid)), np.nan)
flatness_matrix = np.full((len(a_grid), len(h1_grid)), np.nan)
ac_matrix = np.full((len(a_grid), len(h1_grid)), np.nan)
orthogonality_matrix = np.full((len(a_grid), len(h1_grid)), np.nan)

for i_a, a_val in enumerate(a_grid):
    for j_h, h1_val in enumerate(h1_grid):
        cell_results = [
            r for r in score_heatmap_results
            if r["a"] == a_val and r["h1"] == h1_val
        ]

        if len(cell_results) == 0:
            continue

        best_cell = min(cell_results, key=lambda r: r["unsupervised_score"])

        score_matrix[i_a, j_h] = best_cell["unsupervised_score"]
        K_matrix[i_a, j_h] = best_cell["K"]
        maxK_matrix[i_a, j_h] = best_cell["max_K_candidate"]
        flatness_matrix[i_a, j_h] = best_cell["residual_flatness"]
        ac_matrix[i_a, j_h] = best_cell["residual_ac_score"]
        orthogonality_matrix[i_a, j_h] = best_cell["orthogonality"]


plt.figure(figsize=(10, 6))
plt.imshow(
    score_matrix,
    origin="lower",
    aspect="auto",
    extent=[h1_grid[0], h1_grid[-1], a_grid[0], a_grid[-1]],
)
plt.colorbar(label=f"best unsupervised standardized score, lambda_K={best_lambda_K}")
plt.scatter(
    best_theory_score["h1"],
    best_theory_score["a"],
    marker="x",
    s=120,
    label="theory-consistent best",
)
plt.xlabel(r"$h_1$")
plt.ylabel(r"$a$")
plt.title(r"Heatmap: best score over $(h_1,a)$ after optimizing max_K")
plt.legend()
plt.tight_layout()
save_current_plot("Heatmap best score over h1 a after optimizing maxK")


plt.figure(figsize=(10, 6))
plt.imshow(
    K_matrix,
    origin="lower",
    aspect="auto",
    extent=[h1_grid[0], h1_grid[-1], a_grid[0], a_grid[-1]],
)
plt.colorbar(label="selected K")
plt.scatter(
    best_theory_score["h1"],
    best_theory_score["a"],
    marker="x",
    s=120,
    label="theory-consistent best",
)
plt.xlabel(r"$h_1$")
plt.ylabel(r"$a$")
plt.title(r"Heatmap: selected K over $(h_1,a)$ after optimizing max_K")
plt.legend()
plt.tight_layout()
save_current_plot("Heatmap selected K over h1 a after optimizing maxK")


plt.figure(figsize=(10, 6))
plt.imshow(
    maxK_matrix,
    origin="lower",
    aspect="auto",
    extent=[h1_grid[0], h1_grid[-1], a_grid[0], a_grid[-1]],
)
plt.colorbar(label="best max_K candidate")
plt.scatter(
    best_theory_score["h1"],
    best_theory_score["a"],
    marker="x",
    s=120,
    label="theory-consistent best",
)
plt.xlabel(r"$h_1$")
plt.ylabel(r"$a$")
plt.title(r"Heatmap: best max_K candidate over $(h_1,a)$")
plt.legend()
plt.tight_layout()
save_current_plot("Heatmap best maxK candidate over h1 a")


plt.figure(figsize=(10, 6))
plt.imshow(
    flatness_matrix,
    origin="lower",
    aspect="auto",
    extent=[h1_grid[0], h1_grid[-1], a_grid[0], a_grid[-1]],
)
plt.colorbar(label="residual spectral flatness")
plt.scatter(
    best_theory_score["h1"],
    best_theory_score["a"],
    marker="x",
    s=120,
    label="theory-consistent best",
)
plt.xlabel(r"$h_1$")
plt.ylabel(r"$a$")
plt.title(r"Heatmap: residual spectral flatness over $(h_1,a)$")
plt.legend()
plt.tight_layout()
save_current_plot("Heatmap residual spectral flatness over h1 a after optimizing maxK")


plt.figure(figsize=(10, 6))
plt.imshow(
    ac_matrix,
    origin="lower",
    aspect="auto",
    extent=[h1_grid[0], h1_grid[-1], a_grid[0], a_grid[-1]],
)
plt.colorbar(label="residual autocorrelation score")
plt.scatter(
    best_theory_score["h1"],
    best_theory_score["a"],
    marker="x",
    s=120,
    label="theory-consistent best",
)
plt.xlabel(r"$h_1$")
plt.ylabel(r"$a$")
plt.title(r"Heatmap: residual autocorrelation score over $(h_1,a)$")
plt.legend()
plt.tight_layout()
save_current_plot("Heatmap residual autocorrelation score over h1 a after optimizing maxK")


plt.figure(figsize=(10, 6))
plt.imshow(
    orthogonality_matrix,
    origin="lower",
    aspect="auto",
    extent=[h1_grid[0], h1_grid[-1], a_grid[0], a_grid[-1]],
)
plt.colorbar(label="orthogonality index")
plt.scatter(
    best_theory_score["h1"],
    best_theory_score["a"],
    marker="x",
    s=120,
    label="theory-consistent best",
)
plt.xlabel(r"$h_1$")
plt.ylabel(r"$a$")
plt.title(r"Heatmap: orthogonality over $(h_1,a)$")
plt.legend()
plt.tight_layout()
save_current_plot("Heatmap orthogonality over h1 a after optimizing maxK")


# ============================================================
# 21. Lambda_K sensitivity plot
# ============================================================
best_by_lambda = []

for lam in lambda_K_candidates:
    subset = [r for r in grid_results if r["lambda_K"] == lam]
    best_lam = min(subset, key=lambda r: r["unsupervised_score"])
    best_by_lambda.append(best_lam)

lambda_arr = np.array([r["lambda_K"] for r in best_by_lambda])
K_arr = np.array([r["K"] for r in best_by_lambda])
maxK_arr = np.array([r["max_K_candidate"] for r in best_by_lambda])
score_arr = np.array([r["unsupervised_score"] for r in best_by_lambda])
flatness_arr = np.array([r["residual_flatness"] for r in best_by_lambda])
ac_arr = np.array([r["residual_ac_score"] for r in best_by_lambda])

plt.figure(figsize=(10, 4))
plt.plot(lambda_arr, K_arr, marker="o", label="selected K")
plt.plot(lambda_arr, maxK_arr, marker="o", linestyle="--", label="selected max_K candidate")
plt.title(r"Sensitivity: selected $K$ and max_K vs $\lambda_K$")
plt.xlabel(r"$\lambda_K$")
plt.ylabel("K")
plt.legend()
plt.tight_layout()
save_current_plot("Sensitivity selected K and maxK vs lambda_K")


plt.figure(figsize=(10, 4))
plt.plot(lambda_arr, score_arr, marker="o")
plt.title(r"Sensitivity: unsupervised score vs $\lambda_K$")
plt.xlabel(r"$\lambda_K$")
plt.ylabel("unsupervised standardized score")
plt.tight_layout()
save_current_plot("Sensitivity unsupervised score vs lambda_K")


plt.figure(figsize=(10, 4))
plt.plot(lambda_arr, flatness_arr, marker="o")
plt.title(r"Sensitivity: residual spectral flatness vs $\lambda_K$")
plt.xlabel(r"$\lambda_K$")
plt.ylabel("residual spectral flatness")
plt.tight_layout()
save_current_plot("Sensitivity residual spectral flatness vs lambda_K")


plt.figure(figsize=(10, 4))
plt.plot(lambda_arr, ac_arr, marker="o")
plt.title(r"Sensitivity: residual autocorrelation score vs $\lambda_K$")
plt.xlabel(r"$\lambda_K$")
plt.ylabel("residual autocorrelation score")
plt.tight_layout()
save_current_plot("Sensitivity residual autocorrelation score vs lambda_K")


# ============================================================
# 22. max_K sensitivity plot
# ============================================================
best_by_maxK = []

for mK in max_K_candidates:
    subset = [r for r in grid_results if r["max_K_candidate"] == mK]
    best_mK = min(subset, key=lambda r: r["unsupervised_score"])
    best_by_maxK.append(best_mK)

maxK_sens_arr = np.array([r["max_K_candidate"] for r in best_by_maxK])
K_sens_arr = np.array([r["K"] for r in best_by_maxK])
score_sens_arr = np.array([r["unsupervised_score"] for r in best_by_maxK])
lambda_sens_arr = np.array([r["lambda_K"] for r in best_by_maxK])

plt.figure(figsize=(10, 4))
plt.plot(maxK_sens_arr, K_sens_arr, marker="o")
plt.title(r"Sensitivity: selected $K$ vs max_K candidate")
plt.xlabel("max_K candidate")
plt.ylabel("selected K")
plt.tight_layout()
save_current_plot("Sensitivity selected K vs maxK candidate")


plt.figure(figsize=(10, 4))
plt.plot(maxK_sens_arr, score_sens_arr, marker="o")
plt.title(r"Sensitivity: best unsupervised score vs max_K candidate")
plt.xlabel("max_K candidate")
plt.ylabel("best unsupervised standardized score")
plt.tight_layout()
save_current_plot("Sensitivity best score vs maxK candidate")


plt.figure(figsize=(10, 4))
plt.plot(maxK_sens_arr, lambda_sens_arr, marker="o")
plt.title(r"Sensitivity: selected $\lambda_K$ vs max_K candidate")
plt.xlabel("max_K candidate")
plt.ylabel(r"selected $\lambda_K$")
plt.tight_layout()
save_current_plot("Sensitivity selected lambda_K vs maxK candidate")


# ============================================================
# 23. Theory-stopping note
# ============================================================
# In this version, the decomposition depth is selected by model selection:
#
#     choose best among h1 x a x max_K x lambda_K
#
# The cascade itself still follows:
#
#     h_k = h1 / a^k,     1 < a <= 2
#
# It stops only when:
#
#     1) k reaches max_K candidate, or
#     2) h_k < min_h.
#
# residual flatness, autocorrelation, AM-FM validation, and recovery check
# are diagnostics / model-selection / validation layers, not internal stopping rules.


# ============================================================
# 24. Optional robust run
# ============================================================
# To run robust IRMF, change in Section 15:
#
#     rho_type = "huber"
#     H = 1.0 * noise_sigma
#
# or
#
#     rho_type = "smooth_l1"
#     H = 1.0 * noise_sigma
#
# Note: for robust nonlinear filtering, the linear theory diagnostic
# tr(A A^T) is skipped by design.