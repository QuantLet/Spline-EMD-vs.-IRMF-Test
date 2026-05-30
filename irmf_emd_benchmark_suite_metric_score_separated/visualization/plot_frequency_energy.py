import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def ensure_output_dir(output_dir):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True); return output_dir

def compute_imf_energy_ratios(imfs):
    if imfs is None or len(imfs) == 0: return np.array([])
    imfs = np.asarray(imfs, dtype=float)
    e = np.sum(imfs**2, axis=1)
    return e / (np.sum(e) + 1e-12)

def plot_frequency_energy_map(center_freqs, imfs, output_dir, name="frequency_energy_map"):
    output_dir = ensure_output_dir(output_dir)
    cf = np.asarray(center_freqs, dtype=float)
    er = compute_imf_energy_ratios(imfs)
    if len(cf) == 0 or len(er) == 0: return
    n = min(len(cf), len(er)); cf = cf[:n]; er = er[:n]
    x = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.scatter(x, cf, s=80 + 900 * er, alpha=0.7)
    for xi, fi, ei in zip(x, cf, er):
        ax.text(xi, fi, f"  {ei:.2f}", fontsize=8, va="center")
    ax.set_xlabel("IMF index"); ax.set_ylabel("Center frequency (Hz)")
    ax.set_title("IMF frequency-energy map")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)

def plot_center_frequency_energy_table(center_freqs, imfs, output_dir, name="center_frequency_energy_table"):
    output_dir = ensure_output_dir(output_dir)
    cf = np.asarray(center_freqs, dtype=float); er = compute_imf_energy_ratios(imfs)
    if len(cf) == 0 or len(er) == 0: return
    n = min(len(cf), len(er))
    rows = [[f"IMF {i+1}", f"{cf[i]:.3f}", f"{er[i]:.3f}"] for i in range(n)]
    fig, ax = plt.subplots(figsize=(6.5, max(2.5, 0.35*n + 1.2)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=["Mode", "Center frequency", "Energy ratio"], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.25)
    ax.set_title("Center frequency and energy ratio")
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)
