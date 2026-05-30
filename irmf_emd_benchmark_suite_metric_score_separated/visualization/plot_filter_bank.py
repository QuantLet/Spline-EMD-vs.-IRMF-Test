#!/usr/bin/python
# coding: UTF-8

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def ensure_output_dir(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _spectrum(x, fs):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return np.array([]), np.array([])
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    amp = np.abs(np.fft.rfft(x))
    amp = amp / (np.max(amp) + 1e-12)
    return freq, amp


def _bottom_legend(fig, ax, ncol=5):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(ncol, max(1, len(labels))),
            frameon=False,
            fontsize=8
        )


def plot_filter_bank(imfs, fs, output_dir, name="filter_bank"):
    """
    Plot normalized spectra of IMFs.

    Legend is placed below the figure, outside the plotting frame.
    """
    output_dir = ensure_output_dir(output_dir)

    if len(imfs) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, imf in enumerate(imfs):
        freq, amp = _spectrum(imf, fs)
        if len(freq) == 0:
            continue
        ax.plot(freq, amp, linewidth=1.0, label=f"IMF {i + 1}")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title("IMF filter-bank spectra")
    ax.grid(True, alpha=0.4)
    ax.set_xlim(left=0)

    _bottom_legend(fig, ax, ncol=5)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(output_dir / f"{name}_filter_bank.png", dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)
