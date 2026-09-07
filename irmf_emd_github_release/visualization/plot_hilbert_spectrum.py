#!/usr/bin/python
# coding: UTF-8

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import hilbert, medfilt


def ensure_output_dir(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _if_curve(imf, fs):
    analytic = hilbert(imf)
    phase = np.unwrap(np.angle(analytic))
    freq = fs * np.diff(phase) / (2.0 * np.pi)

    if len(freq) >= 5:
        k = min(5, len(freq))
        if k % 2 == 0:
            k -= 1
        if k >= 3:
            freq = medfilt(freq, kernel_size=k)

    if len(freq) > 0:
        cap = np.quantile(np.abs(freq), 0.98)
        freq = np.clip(freq, -cap, cap)

    return freq


def _normalize_true_frequency_items(true_frequencies, t):
    if true_frequencies is None:
        return []

    if isinstance(true_frequencies, dict):
        iterable = [{"label": k, "freq": v} for k, v in true_frequencies.items()]
    else:
        iterable = true_frequencies

    out = []
    for i, item in enumerate(iterable):
        if isinstance(item, dict):
            label = item.get("label", f"true {i + 1}")
            freq = item.get("freq", None)
        else:
            label = f"true {i + 1}"
            freq = item

        if freq is None:
            continue

        if np.isscalar(freq):
            arr = np.ones_like(t, dtype=float) * float(freq)
        else:
            arr = np.asarray(freq, dtype=float)
            if len(arr) != len(t):
                old_x = np.linspace(0.0, 1.0, len(arr))
                new_x = np.linspace(0.0, 1.0, len(t))
                arr = np.interp(new_x, old_x, arr)

        out.append((label, arr))

    return out


def _bottom_legend(fig, ax, ncol=4):
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


def plot_hilbert_ridges(
        imfs,
        t,
        fs,
        output_dir,
        name="hilbert_ridges",
        true_frequencies=None
):
    """
    Plot Hilbert instantaneous-frequency ridges.

    Legend is placed below the figure, outside the plotting frame.
    """
    output_dir = ensure_output_dir(output_dir)

    if len(imfs) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, imf in enumerate(imfs):
        freq = _if_curve(imf, fs)
        if len(freq) == 0:
            continue
        tt = t[:len(freq)]
        ax.plot(tt, freq, linewidth=1.0, label=f"IMF {i + 1}")

    true_items = _normalize_true_frequency_items(true_frequencies, t)
    for label, true_freq in true_items:
        ax.plot(t, true_freq, linestyle="--", linewidth=2.0, label=label)

    ax.set_xlabel("Time")
    ax.set_ylabel("Instantaneous frequency (Hz)")
    title = "Hilbert instantaneous frequency ridges"
    if true_items:
        title += " with true/reference IF"
    ax.set_title(title)
    ax.grid(True, alpha=0.4)

    _bottom_legend(fig, ax, ncol=4)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(output_dir / f"{name}_hilbert_ridges.png", dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)
