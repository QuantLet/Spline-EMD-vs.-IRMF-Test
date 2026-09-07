
from pathlib import Path
import matplotlib.pyplot as plt

def ensure_output_dir(output_dir): output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True); return output_dir
def plot_metric_vs_noise(results_by_noise,metric_key,output_dir,name):
    output_dir=ensure_output_dir(output_dir); sig=sorted(results_by_noise.keys()); vals=[results_by_noise[s][metric_key] for s in sig]
    plt.figure(figsize=(7,4)); plt.plot(sig,vals,marker='o'); plt.xlabel('Noise sigma'); plt.ylabel(metric_key); plt.grid(True); plt.tight_layout(); plt.savefig(output_dir/f'{name}_{metric_key}_vs_noise.png',dpi=300,transparent=True); plt.close()
