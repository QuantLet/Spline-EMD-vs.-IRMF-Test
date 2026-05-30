#!/usr/bin/python
# coding: UTF-8
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def ensure_output_dir(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _curve(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


def plot_operator_evolution(rows, output_dir, name="operator_evolution"):
    output_dir = ensure_output_dir(output_dir)
    if not rows:
        return
    k = _curve(rows, "k")
    curves = [
        ("trace_norm_proxy", "Trace norm proxy"),
        ("operator_norm_proxy", "Operator norm proxy"),
        ("contraction_ratio", "Contraction ratio"),
        ("residual_energy_ratio", "Residual energy ratio"),
        ("component_energy_ratio", "Component energy ratio"),
        ("b0_scale_proxy", "b0 scale proxy"),
    ]
    for key, title in curves:
        plt.figure(figsize=(8, 5))
        plt.plot(k, _curve(rows, key), marker="o")
        plt.xlabel("Scale index k")
        plt.ylabel(title)
        plt.title(f"{title} across IRMF scales")
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(output_dir / f"{name}_{key}.png", dpi=300, transparent=True)
        plt.close()
    plt.figure(figsize=(8, 5))
    plt.plot(k, _curve(rows, "h"), marker="o")
    plt.xlabel("Scale index k")
    plt.ylabel("Bandwidth h")
    plt.title("Bandwidth decay across IRMF scales")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_bandwidth_decay.png", dpi=300, transparent=True)
    plt.close()
