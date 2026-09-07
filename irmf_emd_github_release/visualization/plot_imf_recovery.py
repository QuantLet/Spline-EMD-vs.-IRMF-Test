import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def ensure_output_dir(output_dir):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True); return output_dir

def _corr(a,b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12: return np.nan
    return float(np.corrcoef(a,b)[0,1])

def _match(imfs, true_components):
    try:
        from scipy.optimize import linear_sum_assignment
        cost = np.ones((imfs.shape[0], true_components.shape[0]))
        for i in range(imfs.shape[0]):
            for j in range(true_components.shape[0]):
                c = _corr(imfs[i], true_components[j])
                if np.isfinite(c): cost[i,j] = 1 - abs(c)
        rows, cols = linear_sum_assignment(cost)
        return list(zip(rows, cols))
    except Exception:
        return [(i, i) for i in range(min(imfs.shape[0], true_components.shape[0]))]

def plot_imf_recovery_pairs(imfs, true_components, t, output_dir, name="imf_recovery_pairs"):
    output_dir = ensure_output_dir(output_dir)
    if true_components is None or imfs is None or len(imfs) == 0: return
    imfs = np.asarray(imfs, dtype=float); tc = np.asarray(true_components, dtype=float)
    if imfs.ndim != 2 or tc.ndim != 2 or imfs.shape[1] != tc.shape[1]: return
    pairs = _match(imfs, tc)
    if not pairs: return
    fig, axes = plt.subplots(len(pairs), 1, figsize=(10, max(2*len(pairs), 3)), sharex=True)
    if len(pairs) == 1: axes = [axes]
    for ax, (i,j) in zip(axes, pairs):
        ax.plot(t, tc[j], linestyle="--", linewidth=1.2, label=f"True component {j+1}")
        ax.plot(t, imfs[i], linewidth=1.0, label=f"Recovered IMF {i+1}")
        c = _corr(imfs[i], tc[j])
        ax.set_ylabel(f"T{j+1}/I{i+1}")
        ax.set_title(f"Matched mode pair | corr={c:.3f}" if np.isfinite(c) else "Matched mode pair")
        ax.grid(True, alpha=0.35)
    axes[-1].set_xlabel("Time")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(output_dir / f"{name}.png", dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)
