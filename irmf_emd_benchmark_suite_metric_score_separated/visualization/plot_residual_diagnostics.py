import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def ensure_output_dir(output_dir):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True); return output_dir

def residual_autocorrelation(residual, max_lag=40):
    r = np.asarray(residual, dtype=float); r = r - np.mean(r)
    denom = np.sum(r**2) + 1e-12
    lags = np.arange(0, min(max_lag, len(r)-1) + 1)
    acf = [1.0 if lag == 0 else float(np.sum(r[:-lag]*r[lag:]) / denom) for lag in lags]
    return lags, np.asarray(acf)

def plot_residual_autocorrelation(residual, output_dir, name="residual_autocorrelation", max_lag=40):
    output_dir = ensure_output_dir(output_dir)
    if residual is None or len(residual) < 3: return
    lags, acf = residual_autocorrelation(residual, max_lag=max_lag)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.stem(lags, acf); ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Lag"); ax.set_ylabel("Autocorrelation"); ax.set_title("Residual autocorrelation")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)
