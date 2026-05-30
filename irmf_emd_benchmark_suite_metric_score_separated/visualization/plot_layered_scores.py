import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def ensure_output_dir(output_dir):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True); return output_dir

def _v(r, k):
    try: return float(r.get(k, np.nan))
    except Exception: return np.nan

def plot_layered_score_comparison(rows, output_dir, name="layered_score_comparison"):
    output_dir = ensure_output_dir(output_dir)
    keys = [("robust_estimation_score","Robust Estimation"), ("decomposition_quality_score","Decomposition Quality"), ("irmf_performance_score","IRMF Performance")]
    methods = [m for m,_ in rows]; x = np.arange(len(methods)); width = 0.24
    fig, ax = plt.subplots(figsize=(10,4.8))
    for i,(k,label) in enumerate(keys):
        ax.bar(x+(i-1)*width, [_v(r,k) for _,r in rows], width=width, label=label)
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel("Score"); ax.set_title("Layered score comparison")
    ax.grid(True, axis="y", alpha=0.3)
    h,l = ax.get_legend_handles_labels()
    if h:
        fig.legend(h,l,loc="lower center",bbox_to_anchor=(0.5,-0.02),ncol=min(3,len(l)),frameon=False,fontsize=8)
    fig.tight_layout(rect=[0,0.12,1,1])
    fig.savefig(output_dir / f"{name}.png", dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)
