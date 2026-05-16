#!/usr/bin/python
# coding: UTF-8

import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
from PyEMD import EMD


# ============================================================
# Plot settings
# ============================================================
plt.rcParams["figure.facecolor"] = "none"
plt.rcParams["axes.facecolor"] = "none"


def save_current_plot(title):
    plt.show()


# ============================================================
# Colors
# ============================================================
color_data   = "#1f77b4"
color_maxima = "#ff0000"
color_minima = "#0000ff"
color_upper  = "#ff7f0e"
color_lower  = "#2ca02c"
color_mean   = "#d62728"
color_imf    = "#00cfd5"
color_res    = "#444444"

LINE_WIDTH = 2.0
trim = slice(2, -3)


# ============================================================
# Helper functions for visualization
# ============================================================
def count_zero_crossings(signal):
    return int(np.sum(signal[:-1] * signal[1:] < 0))


def plot_sifting_iteration(
    T,
    original_residual,
    current_signal,
    proto_imf,
    upper,
    lower,
    mean,
    imf_no,
    iteration,
):
    current_residual = original_residual - proto_imf

    max_peaks = argrelextrema(current_signal, np.greater)[0]
    min_peaks = argrelextrema(current_signal, np.less)[0]

    fig, axes = plt.subplots(2, 1, figsize=(18, 9), facecolor="none")

    ax = axes[0]
    ax.plot(T[trim], current_signal[trim], color=color_data, linewidth=LINE_WIDTH)
    ax.plot(T[trim], upper[trim], color=color_upper, linewidth=LINE_WIDTH)
    ax.plot(T[trim], lower[trim], color=color_lower, linewidth=LINE_WIDTH)
    ax.plot(T[trim], mean[trim], color=color_mean, linewidth=LINE_WIDTH)
    ax.plot(T[trim], proto_imf[trim], color=color_imf, linewidth=LINE_WIDTH)

    trim_start = 2
    trim_end = len(current_signal) - 3

    max_peaks_trimmed = [p for p in max_peaks if trim_start <= p < trim_end]
    min_peaks_trimmed = [p for p in min_peaks if trim_start <= p < trim_end]

    ax.scatter(
        T[max_peaks_trimmed],
        current_signal[max_peaks_trimmed],
        c=color_maxima,
        s=25,
        zorder=5,
    )

    ax.scatter(
        T[min_peaks_trimmed],
        current_signal[min_peaks_trimmed],
        c=color_minima,
        s=25,
        zorder=5,
    )

    ax.set_title(f"IMF {imf_no}; sifting iteration {iteration}")
    ax.set_facecolor("none")

    ax = axes[1]
    ax.plot(T[trim], current_residual[trim], color=color_res, linewidth=LINE_WIDTH)
    ax.set_title("Current residual")
    ax.set_facecolor("none")

    plt.tight_layout()
    save_current_plot(f"04 IMF {imf_no} sifting iteration {iteration:02d}")


def generate_sifting_process_plots(S, T, emd, max_imf=-1):
    """
    Visualization-only PyEMD-style sifting loop.
    The final decomposition is still computed by emd.emd().
    """
    T_index = np.arange(len(S), dtype=S.dtype)

    IMF = np.empty((0, len(S)))
    imf_no = 0
    finished = False
    extNo = -1

    while not finished:
        residue = S - np.sum(IMF[:imf_no], axis=0)
        imf = residue.copy()

        n = 0
        n_h = 0

        while True:
            n += 1

            if n >= emd.MAX_ITERATION:
                print(f"IMF {imf_no + 1}: max iterations reached.")
                break

            ext_res = emd.find_extrema(T_index, imf)
            max_pos, min_pos, indzer = ext_res[0], ext_res[2], ext_res[4]

            extNo = len(max_pos) + len(min_pos)
            nzm = len(indzer)

            if extNo > 2:
                max_env, min_env, eMax, eMin = emd.extract_max_min_spline(T_index, imf)
                mean = 0.5 * (max_env + min_env)

                imf_old = imf.copy()
                imf = imf - mean

                plot_sifting_iteration(
                    T=T,
                    original_residual=residue,
                    current_signal=imf_old,
                    proto_imf=imf,
                    upper=max_env,
                    lower=min_env,
                    mean=mean,
                    imf_no=imf_no + 1,
                    iteration=n,
                )

                if emd.FIXE:
                    if n >= emd.FIXE:
                        break

                elif emd.FIXE_H:
                    tmp_residue = emd.find_extrema(T_index, imf)
                    max_pos_new, min_pos_new, ind_zer_new = (
                        tmp_residue[0],
                        tmp_residue[2],
                        tmp_residue[4],
                    )

                    extNo = len(max_pos_new) + len(min_pos_new)
                    nzm = len(ind_zer_new)

                    if n == 1:
                        continue

                    n_h = n_h + 1 if abs(extNo - nzm) < 2 else 0

                    if n_h >= emd.FIXE_H:
                        print(
                            f"IMF {imf_no + 1} stopped by FIXE_H={emd.FIXE_H} "
                            f"at iteration {n}"
                        )
                        break

                else:
                    break

            else:
                finished = True
                break

        IMF = np.vstack((IMF, imf.copy()))
        imf_no += 1

        if emd.end_condition(S, IMF) or imf_no == max_imf:
            finished = True
            break

    print("Finished generating double-panel sifting plots.")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    max_imf = -1
    DTYPE = np.float64

    # --------------------------------------------------------
    # Your signal
    # --------------------------------------------------------
    np.random.seed(100)
    data = np.random.random(100) - 0.5
    data = data.astype(DTYPE)

    N = len(data)
    T = np.arange(N, dtype=DTYPE)

    tMin, tMax = 0, N - 1
    S = data.copy()

    print("Input S.dtype:", S.dtype)
    print("Signal length:", N)

    # --------------------------------------------------------
    # PyEMD settings
    # --------------------------------------------------------
    emd = EMD()
    emd.FIXE_H = 5
    emd.nbsym = 2
    emd.spline_kind = "cubic"
    emd.extrema_detection = "simple"
    emd.DTYPE = DTYPE

    # ========================================================
    # Generate double-panel sifting process plots
    # ========================================================
    generate_sifting_process_plots(
        S=S,
        T=T,
        emd=emd,
        max_imf=max_imf,
    )

    # ========================================================
    # Run official PyEMD decomposition
    # ========================================================
    imfs_with_residue = emd.emd(S, T, max_imf)
    imfs, residue = emd.get_imfs_and_residue()

    imfNo = imfs.shape[0]

    print("Number of IMFs:", imfNo)
    print("nbsym:", emd.nbsym)
    print("FIXE_H:", emd.FIXE_H)
    print("spline_kind:", emd.spline_kind)

    # ========================================================
    # 1. Original signal
    # ========================================================
    plt.figure(figsize=(18, 6), facecolor="none")
    plt.plot(T, S, color=color_data, linewidth=LINE_WIDTH)
    plt.title("Original random signal")
    plt.xlabel("index")
    plt.ylabel("value")
    plt.gca().set_facecolor("none")
    plt.tight_layout()
    save_current_plot("01 original random signal")

    # ========================================================
    # 2. Local extrema of original signal
    # ========================================================
    max_peaks = argrelextrema(S, np.greater)[0]
    min_peaks = argrelextrema(S, np.less)[0]

    plt.figure(figsize=(18, 6), facecolor="none")
    plt.plot(T, S, color=color_data, linewidth=LINE_WIDTH)
    plt.scatter(T[max_peaks], S[max_peaks], c=color_maxima, s=30)
    plt.scatter(T[min_peaks], S[min_peaks], c=color_minima, s=30)
    plt.title("Find local extrema")
    plt.xlabel("index")
    plt.ylabel("value")
    plt.gca().set_facecolor("none")
    plt.tight_layout()
    save_current_plot("02 local extrema original signal")

    # ========================================================
    # 3. First envelope construction
    # ========================================================
    try:
        T_index = np.arange(N, dtype=DTYPE)

        max_env, min_env, eMax, eMin = emd.extract_max_min_spline(T_index, S)
        mean_env = 0.5 * (max_env + min_env)

        plt.figure(figsize=(18, 6), facecolor="none")
        plt.plot(T, S, color=color_data, linewidth=LINE_WIDTH)
        plt.plot(T, max_env, color=color_upper, linewidth=LINE_WIDTH)
        plt.plot(T, min_env, color=color_lower, linewidth=LINE_WIDTH)
        plt.plot(T, mean_env, color=color_mean, linewidth=LINE_WIDTH)
        plt.title("Cubic spline envelopes with PyEMD mirror extension")
        plt.xlabel("index")
        plt.ylabel("value")
        plt.gca().set_facecolor("none")
        plt.tight_layout()
        save_current_plot("03 first envelope construction")

        imf1_one_step = S - mean_env

        plt.figure(figsize=(18, 6), facecolor="none")
        plt.plot(T, S, color=color_data, linewidth=LINE_WIDTH)
        plt.plot(T, imf1_one_step, color=color_imf, linewidth=LINE_WIDTH)
        plt.plot(T, mean_env, color=color_mean, linewidth=LINE_WIDTH)
        plt.title("IMF1: one sifting step")
        plt.xlabel("index")
        plt.ylabel("value")
        plt.gca().set_facecolor("none")
        plt.tight_layout()
        save_current_plot("03b IMF1 one sifting step")

    except Exception as exc:
        print("Envelope plot skipped:", exc)

    # ========================================================
    # 5. Decomposition summary
    # ========================================================
    rows = imfNo + 2

    plt.figure(figsize=(18, 2.8 * rows), facecolor="none")

    plt.subplot(rows, 1, 1)
    plt.plot(T, S, color=color_data, linewidth=LINE_WIDTH)
    plt.xlim((tMin, tMax))
    plt.title("Original random signal")
    plt.ylabel("signal")
    plt.gca().set_facecolor("none")

    for k in range(imfNo):
        plt.subplot(rows, 1, k + 2)
        plt.plot(T, imfs[k], color=color_imf, linewidth=LINE_WIDTH)
        plt.xlim((tMin, tMax))
        plt.ylabel(f"IMF {k + 1}")
        plt.gca().set_facecolor("none")

    plt.subplot(rows, 1, rows)
    plt.plot(T, residue, color=color_res, linewidth=LINE_WIDTH)
    plt.xlim((tMin, tMax))
    plt.ylabel("residue")
    plt.xlabel("index")
    plt.gca().set_facecolor("none")

    plt.tight_layout()
    save_current_plot("05 Spline EMD decomposition random signal")

    # ========================================================
    # 6. Reconstruction check
    # ========================================================
    reconstruction = np.sum(imfs, axis=0) + residue
    recon_error = np.max(np.abs(S - reconstruction))

    print("Max reconstruction error:", recon_error)

    plt.figure(figsize=(18, 6), facecolor="none")
    plt.plot(T, S, color=color_data, linewidth=LINE_WIDTH)
    plt.plot(T, reconstruction, "--", color=color_res, linewidth=LINE_WIDTH)
    plt.title("Reconstruction check")
    plt.xlabel("index")
    plt.ylabel("value")
    plt.gca().set_facecolor("none")
    plt.tight_layout()
    save_current_plot("06 reconstruction check")
