#!/usr/bin/python
# coding: UTF-8
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_theoretical_diagnostics(theoretical_diagnostics, output_dir, name="theoretical_diagnostics"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = theoretical_diagnostics.get("rows", [])
    if not rows:
        return
    k = np.asarray([r["k"] for r in rows], dtype=int)
    h = np.asarray([r["h"] for r in rows], dtype=float)
    imf = np.asarray([r["imf_trace_risk_norm"] for r in rows], dtype=float)
    resid = np.asarray([r["residual_trace_risk_norm"] for r in rows], dtype=float)
    contraction = np.asarray([r["residual_contraction_ratio"] for r in rows], dtype=float)
    fisher = np.asarray([r.get("fisher_step_l2_mean", np.nan) for r in rows], dtype=float)
    wilks = np.asarray([r.get("wilks_gap_mean", np.nan) for r in rows], dtype=float)
    hzero = np.asarray([r.get("hessian_zero_fraction", np.nan) for r in rows], dtype=float)

    plt.figure(figsize=(12, 10))
    plt.subplot(3, 2, 1); plt.plot(k, imf, marker="o"); plt.title("IMF trace risk / n"); plt.xlabel("k"); plt.grid(True)
    plt.subplot(3, 2, 2); plt.plot(k, resid, marker="o"); plt.title("Residual trace risk / n"); plt.xlabel("k"); plt.grid(True)
    plt.subplot(3, 2, 3); plt.plot(k, contraction, marker="o"); plt.axhline(1, linestyle="--"); plt.title("Residual contraction ratio"); plt.xlabel("k"); plt.grid(True)
    plt.subplot(3, 2, 4); plt.plot(k, h, marker="o"); plt.title("Bandwidth path"); plt.xlabel("k"); plt.grid(True)
    plt.subplot(3, 2, 5); plt.plot(k, fisher, marker="o", label="Fisher/Newton step"); plt.plot(k, wilks, marker="s", label="Wilks proxy"); plt.title("Appendix-A proxies"); plt.xlabel("k"); plt.legend(); plt.grid(True)
    plt.subplot(3, 2, 6); plt.plot(k, hzero, marker="o"); plt.title("Hessian zero fraction"); plt.xlabel("k"); plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}.png", dpi=180)
    plt.close()
