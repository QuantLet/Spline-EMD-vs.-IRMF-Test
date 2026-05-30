#!/usr/bin/python
# coding: UTF-8

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def ensure_output_dir(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _legend_if_any(ax, fontsize=8):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=fontsize)


def _residual_evolution_from_imfs(Y_observed, imfs):
    """
    Residual evolution:
        R_0 = Y
        R_k = Y - sum_{j=1}^k IMF_j
    """
    if Y_observed is None:
        return []

    residuals = [np.asarray(Y_observed, dtype=float)]
    current = np.asarray(Y_observed, dtype=float).copy()

    for imf in imfs:
        current = current - imf
        residuals.append(current.copy())

    return residuals


def plot_imfs(
        imfs,
        residual,
        t,
        output_dir,
        name="imfs",
        Y_observed=None,
        X_clean=None,
        true_noise=None
):
    """
    Academic two-column decomposition plot.

    Left column:
        row 0: observed signal + clean reference
        rows 1..K: IMF_i

    Right column:
        row 0: initial residual R0 = observed signal
        rows 1..K: residual evolution R_i
        last row uses final residual and optional true noise.

    This follows the user's preferred decomposition-process layout.
    """
    output_dir = ensure_output_dir(output_dir)

    K = len(imfs)

    if K == 0:
        fig, axes = plt.subplots(1, 2, figsize=(15, 2.8), facecolor="none")
        fig.patch.set_alpha(0)
        fig.suptitle(f"{name} (No IMFs)", fontsize=12, fontweight="bold")

        if Y_observed is not None:
            axes[0].plot(t, Y_observed, alpha=0.55, label="Observed Signal $Y(t)$")
        if X_clean is not None:
            axes[0].plot(t, X_clean, linestyle="--", linewidth=1.2, label="Clean Signal")
        axes[0].set_title("Input signal")
        axes[0].grid(True, linestyle=":", alpha=0.6)

        if Y_observed is not None:
            axes[1].plot(t, Y_observed, alpha=0.7, label="Initial Residual $R_0$")
        axes[1].plot(t, residual, linewidth=1.2, label="Final Residual")
        axes[1].set_title("Residual")
        axes[1].grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_dir / f"{name}_imfs.png", dpi=300, transparent=True)
        plt.close()
        return

    residuals_history = _residual_evolution_from_imfs(Y_observed, imfs)

    fig, axes = plt.subplots(
        K + 1,
        2,
        figsize=(15, max(1.8 * (K + 1), 5)),
        facecolor="none",
        sharex=True
    )
    fig.patch.set_alpha(0)
    fig.suptitle(name, fontsize=13, fontweight="bold")

    # Row 0: original / clean and initial residual
    if Y_observed is not None:
        axes[0, 0].plot(t, Y_observed, alpha=0.55, label="Observed Signal $Y(t)$")
    if X_clean is not None:
        axes[0, 0].plot(t, X_clean, linestyle="--", linewidth=1.2, label="Clean Signal")
    axes[0, 0].set_title("Signal and extracted IMFs")
    axes[0, 0].set_xlim([float(np.min(t)), float(np.max(t))])
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    if Y_observed is not None:
        axes[0, 1].plot(t, Y_observed, alpha=0.7, label="Initial Residual $R_0$")
    axes[0, 1].set_title("Residual evolution")
    axes[0, 1].set_xlim([float(np.min(t)), float(np.max(t))])
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    # IMF rows and residual evolution
    for i in range(K):
        axes[i + 1, 0].plot(t, imfs[i], linewidth=1.0, label=f"IMF {i + 1}")
        axes[i + 1, 0].set_xlim([float(np.min(t)), float(np.max(t))])
        axes[i + 1, 0].grid(True, linestyle=":", alpha=0.6)

        # Residual after extracting IMF i+1
        if i + 1 < len(residuals_history):
            r_to_plot = residuals_history[i + 1]
            label = f"Residual $R_{i + 1}$"
        else:
            r_to_plot = residual
            label = "Final Residual"

        if i == K - 1:
            axes[i + 1, 1].plot(t, residual, linewidth=1.2, label="Final Residual")
            if true_noise is not None:
                axes[i + 1, 1].plot(t, true_noise, linestyle=":", alpha=0.55, label="True Noise")
        else:
            axes[i + 1, 1].plot(t, r_to_plot, linewidth=1.0, label=label)

        axes[i + 1, 1].set_xlim([float(np.min(t)), float(np.max(t))])
        axes[i + 1, 1].grid(True, linestyle=":", alpha=0.6)

    axes[-1, 0].set_xlabel("Time")
    axes[-1, 1].set_xlabel("Time")

    plt.tight_layout(rect=[0.04, 0, 1, 0.96])
    clean_name = str(name).replace("(", "").replace(")", "")
    plt.savefig(output_dir / f"{clean_name}_imfs.png", dpi=300, transparent=True)
    plt.close()


def plot_input_signal(Y_observed, t, output_dir, name="input_signal", X_clean=None):
    output_dir = ensure_output_dir(output_dir)

    plt.figure(figsize=(10, 4))
    plt.plot(t, Y_observed, linewidth=1.1, label="Observed")
    if X_clean is not None:
        plt.plot(t, X_clean, linewidth=1.1, linestyle="--", label="Clean reference")
        plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.title("Input signal")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}.png", dpi=300, transparent=True)
    plt.close()
