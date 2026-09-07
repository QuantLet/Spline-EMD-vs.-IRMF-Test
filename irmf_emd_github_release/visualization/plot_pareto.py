
from pathlib import Path
import matplotlib.pyplot as plt

def ensure_output_dir(output_dir): output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True); return output_dir
def plot_pareto_front(results,output_dir,x_key='spectral_leakage',y_key='strict_io',name='pareto_front'):
    output_dir=ensure_output_dir(output_dir); plt.figure(figsize=(7,5)); plt.scatter([r[x_key] for r in results],[r[y_key] for r in results],s=18,alpha=.7); plt.xlabel(x_key); plt.ylabel(y_key); plt.title('Pareto diagnostic space'); plt.grid(True); plt.tight_layout(); plt.savefig(output_dir/f'{name}.png',dpi=300,transparent=True); plt.close()
