import os
from pathlib import Path

import numpy as np
from scipy.signal import argrelextrema
import scipy.interpolate as spi

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".matplotlib-cache")
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Define colors
# ============================================================
color_data   = "#1f77b4"
color_maxima = "#ff0000"
color_minima = "#0000ff"
color_upper  = "#ff7f0e"
color_lower  = "#2ca02c"
color_mean   = "#d62728"
color_imf    = "#00cfd5"
color_res    = "#444444"

LINE_WIDTH = 3.0
OUTPUT_DIR = Path(__file__).resolve().parent / "EMD Visualization 4 PyEMD mirror plots"


def save_current_figure(filename):
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================
# Generate signal
# ============================================================
np.random.seed(100)
data = np.random.random(100) - 0.5

index = np.arange(len(data))
trim = slice(2, -3)

NBSYM = 2


# ============================================================
# Helper functions
# ============================================================
def find_local_extrema(signal):
    max_peaks_tuple = argrelextrema(signal, np.greater)
    min_peaks_tuple = argrelextrema(signal, np.less)

    max_peaks = np.asarray(max_peaks_tuple[0], dtype=int)
    min_peaks = np.asarray(min_peaks_tuple[0], dtype=int)

    return max_peaks, min_peaks, max_peaks_tuple, min_peaks_tuple


def sort_unique_extrema(pos, val):
    pos = np.asarray(pos, dtype=float)
    val = np.asarray(val, dtype=float)

    order = np.argsort(pos)
    pos = pos[order]
    val = val[order]

    unique_pos, unique_idx = np.unique(pos, return_index=True)
    unique_val = val[unique_idx]

    return unique_pos, unique_val


def prepare_points_simple_like_pyemd(signal, max_pos, min_pos, nbsym=2):
    """
    PyEMD-style extrema mirroring.

    This mirrors extrema positions, not the whole signal.
    It is closer to the EMD boundary treatment used in PyEMD.
    """
    n = len(signal)
    T = np.arange(n, dtype=float)

    ind_max = np.asarray(max_pos, dtype=int)
    ind_min = np.asarray(min_pos, dtype=int)

    if len(ind_max) < 2 or len(ind_min) < 2:
        return None, None, None, None

    end_max = len(ind_max)
    end_min = len(ind_min)

    # ========================================================
    # Left boundary
    # ========================================================
    if ind_max[0] < ind_min[0]:
        if signal[0] > signal[ind_min[0]]:
            lmax = ind_max[1:min(end_max, nbsym + 1)][::-1]
            lmin = ind_min[0:min(end_min, nbsym)][::-1]
            lsym = ind_max[0]
        else:
            lmax = ind_max[0:min(end_max, nbsym)][::-1]
            lmin = np.append(ind_min[0:min(end_min, nbsym - 1)][::-1], 0)
            lsym = 0
    else:
        if signal[0] < signal[ind_max[0]]:
            lmax = ind_max[0:min(end_max, nbsym)][::-1]
            lmin = ind_min[1:min(end_min, nbsym + 1)][::-1]
            lsym = ind_min[0]
        else:
            lmax = np.append(ind_max[0:min(end_max, nbsym - 1)][::-1], 0)
            lmin = ind_min[0:min(end_min, nbsym)][::-1]
            lsym = 0

    # ========================================================
    # Right boundary
    # ========================================================
    if ind_max[-1] < ind_min[-1]:
        if signal[-1] < signal[ind_max[-1]]:
            rmax = ind_max[max(end_max - nbsym, 0):][::-1]
            rmin = ind_min[max(end_min - nbsym - 1, 0):-1][::-1]
            rsym = ind_min[-1]
        else:
            rmax = np.append(ind_max[max(end_max - nbsym + 1, 0):], n - 1)[::-1]
            rmin = ind_min[max(end_min - nbsym, 0):][::-1]
            rsym = n - 1
    else:
        if signal[-1] > signal[ind_min[-1]]:
            rmax = ind_max[max(end_max - nbsym - 1, 0):-1][::-1]
            rmin = ind_min[max(end_min - nbsym, 0):][::-1]
            rsym = ind_max[-1]
        else:
            rmax = ind_max[max(end_max - nbsym, 0):][::-1]
            rmin = np.append(ind_min[max(end_min - nbsym + 1, 0):], n - 1)[::-1]
            rsym = n - 1

    if len(lmin) == 0:
        lmin = ind_min
    if len(rmin) == 0:
        rmin = ind_min
    if len(lmax) == 0:
        lmax = ind_max
    if len(rmax) == 0:
        rmax = ind_max

    # ========================================================
    # Mirror positions
    # ========================================================
    tlmin = 2 * T[lsym] - T[lmin]
    tlmax = 2 * T[lsym] - T[lmax]
    trmin = 2 * T[rsym] - T[rmin]
    trmax = 2 * T[rsym] - T[rmax]

    # Fallback if mirrored points are not outside the signal range
    if len(tlmin) > 0 and len(tlmax) > 0:
        if tlmin[0] > T[0] or tlmax[0] > T[0]:
            lsym = 0
            lmax = ind_max[0:min(end_max, nbsym)][::-1]
            lmin = ind_min[0:min(end_min, nbsym)][::-1]
            tlmin = 2 * T[lsym] - T[lmin]
            tlmax = 2 * T[lsym] - T[lmax]

    if len(trmin) > 0 and len(trmax) > 0:
        if trmin[-1] < T[-1] or trmax[-1] < T[-1]:
            rsym = n - 1
            rmax = ind_max[max(end_max - nbsym, 0):][::-1]
            rmin = ind_min[max(end_min - nbsym, 0):][::-1]
            trmin = 2 * T[rsym] - T[rmin]
            trmax = 2 * T[rsym] - T[rmax]

    zlmax = signal[lmax]
    zlmin = signal[lmin]
    zrmax = signal[rmax]
    zrmin = signal[rmin]

    tmax = np.concatenate((tlmax, T[ind_max], trmax))
    zmax = np.concatenate((zlmax, signal[ind_max], zrmax))

    tmin = np.concatenate((tlmin, T[ind_min], trmin))
    zmin = np.concatenate((zlmin, signal[ind_min], zrmin))

    tmax, zmax = sort_unique_extrema(tmax, zmax)
    tmin, zmin = sort_unique_extrema(tmin, zmin)

    return tmax, zmax, tmin, zmin


def spline_from_mirrored_extrema(pos, val, n):
    if pos is None or val is None or len(pos) < 2:
        return np.zeros(n)

    k = min(3, len(pos) - 1)

    try:
        tck = spi.splrep(pos, val, k=k, s=0)
        return spi.splev(np.arange(n), tck)
    except Exception:
        return np.interp(np.arange(n), pos, val)


def cubic_envelopes(signal, nbsym=NBSYM):
    max_peaks, min_peaks, max_peaks_tuple, min_peaks_tuple = find_local_extrema(signal)

    if len(max_peaks) < 2 or len(min_peaks) < 2:
        upper = np.zeros_like(signal)
        lower = np.zeros_like(signal)
        mean = np.zeros_like(signal)
        return upper, lower, mean, max_peaks, min_peaks, max_peaks_tuple, min_peaks_tuple

    tmax, zmax, tmin, zmin = prepare_points_simple_like_pyemd(
        signal=signal,
        max_pos=max_peaks,
        min_pos=min_peaks,
        nbsym=nbsym
    )

    upper = spline_from_mirrored_extrema(tmax, zmax, len(signal))
    lower = spline_from_mirrored_extrema(tmin, zmin, len(signal))
    mean = 0.5 * (upper + lower)

    return upper, lower, mean, max_peaks, min_peaks, max_peaks_tuple, min_peaks_tuple


def sifting_once(signal):
    upper, lower, mean, max_peaks, min_peaks, max_peaks_tuple, min_peaks_tuple = cubic_envelopes(signal)

    proto_imf = signal - mean

    return {
        "upper": upper,
        "lower": lower,
        "mean": mean,
        "proto_imf": proto_imf,
        "max_peaks": max_peaks,
        "min_peaks": min_peaks,
        "max_peaks_tuple": max_peaks_tuple,
        "min_peaks_tuple": min_peaks_tuple,
    }


def count_zero_crossings(signal):
    return np.sum(signal[:-1] * signal[1:] < 0)


def is_imf(signal, mean_ratio_thr=0.05, extrema_zero_tol=1):
    max_peaks, min_peaks, _, _ = find_local_extrema(signal)

    num_extrema = len(max_peaks) + len(min_peaks)
    num_zero_crossings = count_zero_crossings(signal)

    condition_1 = abs(num_extrema - num_zero_crossings) <= extrema_zero_tol

    upper, lower, mean, *_ = cubic_envelopes(signal)

    if np.mean(np.abs(signal)) < 1e-12:
        condition_2 = False
    else:
        condition_2 = np.mean(np.abs(mean)) < mean_ratio_thr * np.mean(np.abs(signal))

    return condition_1 and condition_2


def should_stop_decomposition(
    residual,
    original_signal,
    amplitude_ratio_thr=0.08,
    absolute_amplitude_thr=1e-3,
    min_extrema_to_continue=8,
):
    max_peaks, min_peaks, _, _ = find_local_extrema(residual)
    num_extrema = len(max_peaks) + len(min_peaks)

    if num_extrema < min_extrema_to_continue:
        return True

    residual_range = np.max(residual) - np.min(residual)
    original_range = np.max(original_signal) - np.min(original_signal)

    if residual_range < absolute_amplitude_thr:
        return True

    if residual_range < amplitude_ratio_thr * original_range:
        return True

    return False


# ============================================================
# Plot functions
# ============================================================
def plot_extrema(signal, title):
    max_peaks, min_peaks, max_peaks_tuple, min_peaks_tuple = find_local_extrema(signal)

    plt.figure(figsize=(18, 6), facecolor="none")

    plt.plot(signal, color=color_data, linewidth=LINE_WIDTH)

    plt.scatter(
        max_peaks_tuple,
        signal[max_peaks_tuple],
        c=color_maxima
    )

    plt.scatter(
        min_peaks_tuple,
        signal[min_peaks_tuple],
        c=color_minima
    )

    plt.gca().set_facecolor("none")
    plt.title(title)
    save_current_figure("01_original_local_extrema.png")


def plot_sifting_iteration(original_residual, current_signal, result, imf_no, iteration):
    proto_imf = result["proto_imf"]
    current_residual = original_residual - proto_imf

    fig, axes = plt.subplots(2, 1, figsize=(18, 9), facecolor="none")

    ax = axes[0]

    ax.plot(current_signal[trim], color=color_data, linewidth=LINE_WIDTH)
    ax.plot(result["upper"][trim], color=color_upper, linewidth=LINE_WIDTH)
    ax.plot(result["lower"][trim], color=color_lower, linewidth=LINE_WIDTH)
    ax.plot(result["mean"][trim], color=color_mean, linewidth=LINE_WIDTH)
    ax.plot(proto_imf[trim], color=color_imf, linewidth=LINE_WIDTH)

    trim_start = 2
    trim_end = len(current_signal) - 3

    max_peaks_trimmed = [p for p in result["max_peaks"] if trim_start <= p < trim_end]
    min_peaks_trimmed = [p for p in result["min_peaks"] if trim_start <= p < trim_end]

    ax.scatter(
        np.array(max_peaks_trimmed) - trim_start,
        current_signal[max_peaks_trimmed],
        c=color_maxima,
        zorder=5
    )

    ax.scatter(
        np.array(min_peaks_trimmed) - trim_start,
        current_signal[min_peaks_trimmed],
        c=color_minima,
        zorder=5
    )

    ax.set_title(f"IMF {imf_no}; sifting iteration {iteration}")
    ax.set_facecolor("none")

    ax = axes[1]
    ax.plot(current_residual[trim], color=color_res, linewidth=LINE_WIDTH)
    ax.set_title("Current residual")
    ax.set_facecolor("none")

    plt.tight_layout()
    save_current_figure(f"04_imf_{imf_no}_sifting_iteration_{iteration:02d}.png")


# ============================================================
# Automatic IMF extraction
# ============================================================
def extract_imf_auto(
    signal,
    imf_no,
    max_sift_iterations=20,
    sd_thr=0.2,
    mean_ratio_thr=0.05,
):
    original_residual = signal.copy()
    current = signal.copy()

    for iteration in range(max_sift_iterations):
        result = sifting_once(current)
        proto_imf = result["proto_imf"]

        plot_sifting_iteration(
            original_residual=original_residual,
            current_signal=current,
            result=result,
            imf_no=imf_no,
            iteration=iteration
        )

        sd = np.sum((current - proto_imf) ** 2) / (np.sum(current ** 2) + 1e-12)

        current = proto_imf.copy()

        if is_imf(current, mean_ratio_thr=mean_ratio_thr) and sd < sd_thr:
            print(f"IMF {imf_no} stopped at iteration {iteration + 1}, SD={sd:.6f}")
            break

    imf = current.copy()
    residual = original_residual - imf

    return imf, residual


# ============================================================
# Automatic EMD decomposition
# ============================================================
def emd_auto(
    data,
    max_imfs=10,
    max_sift_iterations=20,
    amplitude_ratio_thr=0.08,
    absolute_amplitude_thr=1e-3,
    min_extrema_to_continue=8,
    sd_thr=0.2,
    mean_ratio_thr=0.05,
):
    imfs = []
    residuals = []

    current_residual = data.copy()

    for imf_no in range(1, max_imfs + 1):

        if should_stop_decomposition(
            residual=current_residual,
            original_signal=data,
            amplitude_ratio_thr=amplitude_ratio_thr,
            absolute_amplitude_thr=absolute_amplitude_thr,
            min_extrema_to_continue=min_extrema_to_continue,
        ):
            print(
                f"Stop before IMF {imf_no}: residual is trend-like, "
                f"too small, or has too few extrema."
            )
            break

        imf_k, next_residual = extract_imf_auto(
            signal=current_residual,
            imf_no=imf_no,
            max_sift_iterations=max_sift_iterations,
            sd_thr=sd_thr,
            mean_ratio_thr=mean_ratio_thr,
        )

        imfs.append(imf_k)
        residuals.append(next_residual)

        current_residual = next_residual.copy()

    return imfs, residuals, current_residual


# ============================================================
# 1. Original local extrema plot
# ============================================================
plot_extrema(data, "Find Local Extrema: original data")


# ============================================================
# 2. Original cubic spline interpolation plot
# ============================================================
result0 = sifting_once(data)

plt.figure(figsize=(18, 6), facecolor="none")

plt.plot(data[trim], color=color_data, linewidth=LINE_WIDTH)
plt.plot(result0["upper"][trim], color=color_upper, linewidth=LINE_WIDTH)
plt.plot(result0["lower"][trim], color=color_lower, linewidth=LINE_WIDTH)
plt.plot(result0["mean"][trim], color=color_mean, linewidth=LINE_WIDTH)

plt.gca().set_facecolor("none")
plt.title("Cubic Spline Interpolation with PyEMD-style mirror extension")

save_current_figure("02_original_cubic_spline_interpolation_mirror.png")


# ============================================================
# 3. Original IMF1 one-step plot
# ============================================================
imf1_one_step = data - result0["mean"]

plt.figure(figsize=(18, 6), facecolor="none")

plt.plot(data[trim], color=color_data, linewidth=LINE_WIDTH)
plt.plot(imf1_one_step[trim], color=color_imf, linewidth=LINE_WIDTH)
plt.plot(result0["mean"][trim], color=color_mean, linewidth=LINE_WIDTH)

plt.gca().set_facecolor("none")
plt.title("IMF1: one sifting step with PyEMD-style mirror extension")

save_current_figure("03_imf1_one_sifting_step_mirror.png")


# ============================================================
# 4. Run automatic EMD
# ============================================================
imfs, residuals, final_residual = emd_auto(
    data=data,
    max_imfs=10,
    max_sift_iterations=20,
    amplitude_ratio_thr=0.08,
    absolute_amplitude_thr=1e-3,
    min_extrema_to_continue=8,
    sd_thr=0.2,
    mean_ratio_thr=0.05,
)


# ============================================================
# 5. Final decomposition summary
# ============================================================
K = len(imfs)

plt.figure(figsize=(18, 2.8 * (K + 2)), facecolor="none")

plt.subplot(K + 2, 1, 1)
plt.plot(data[trim], color=color_data, linewidth=LINE_WIDTH)
plt.title("Original data")
plt.gca().set_facecolor("none")

for k, imf_k in enumerate(imfs, start=1):
    plt.subplot(K + 2, 1, k + 1)
    plt.plot(imf_k[trim], color=color_imf, linewidth=LINE_WIDTH)
    plt.title(f"Extracted IMF{k}")
    plt.gca().set_facecolor("none")

plt.subplot(K + 2, 1, K + 2)
plt.plot(final_residual[trim], color=color_res, linewidth=LINE_WIDTH)
plt.title(f"Final residual after {K} IMFs")
plt.gca().set_facecolor("none")

plt.tight_layout()
save_current_figure("05_final_decomposition_summary_mirror.png")


# ============================================================
# 6. Reconstruction check
# ============================================================
if K > 0:
    reconstruction = np.sum(imfs, axis=0) + final_residual
else:
    reconstruction = final_residual

plt.figure(figsize=(18, 6), facecolor="none")

plt.plot(data[trim], color=color_data, linewidth=LINE_WIDTH)
plt.plot(
    reconstruction[trim],
    "--",
    color=color_res,
    linewidth=LINE_WIDTH,
)

plt.gca().set_facecolor("none")
plt.title("Reconstruction Check")

save_current_figure("06_reconstruction_check_mirror.png")