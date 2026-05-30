#!/usr/bin/python
# coding: UTF-8
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def ensure_output_dir(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_metric_bar_with_error(summary, metric, output_dir, name):
    output_dir = ensure_output_dir(output_dir)
    labels, means, stds = [], [], []
    for method, stats in summary.items():
        if isinstance(stats, dict) and metric in stats:
            labels.append(method)
            means.append(stats[metric]['mean'])
            stds.append(stats[metric]['std'])
    if not labels:
        return
    x = np.arange(len(labels))
    plt.figure(figsize=(8, 5))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, labels)
    plt.ylabel(metric)
    plt.title(f"Monte-Carlo {metric}: mean ± std")
    plt.grid(True, axis='y', alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_{metric}_bar.png", dpi=300, transparent=True)
    plt.close()
