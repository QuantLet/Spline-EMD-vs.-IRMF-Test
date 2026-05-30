#!/usr/bin/python
# coding: UTF-8

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def ensure_output_dir(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_matrix_heatmap(matrix, output_dir, name, title, colorbar_label="value"):
    output_dir = ensure_output_dir(output_dir)
    M = np.asarray(matrix)

    if M.size == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(M, vmin=0.0, vmax=max(1.0, float(np.max(M))), aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("IMF index")
    ax.set_ylabel("IMF index")

    labels = [str(i + 1) for i in range(M.shape[0])]
    ax.set_xticks(np.arange(M.shape[0]))
    ax.set_yticks(np.arange(M.shape[0]))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=300, transparent=True)
    plt.close(fig)
