
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt

def ensure_output_dir(output_dir): output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True); return output_dir
def plot_apriori_risk_bound(bounds,output_dir,name='irmf_apriori_risk_bound'):
    output_dir=ensure_output_dir(output_dir); h=bounds['h']; plt.figure(figsize=(12,10))
    plt.subplot(2,2,1); plt.plot(h,bounds['physical_error_bound'],label='risk bound'); plt.axvline(h[np.argmin(bounds['physical_error_bound'])],ls='--'); plt.grid(True); plt.legend(); plt.title('IRMF a priori error bound')
    plt.subplot(2,2,2); plt.plot(h,bounds['b_0'],label='b0'); plt.plot(h,bounds['stochastic_component'],'--',label='stoch'); plt.plot(h,bounds['bias_component'],'--',label='bias'); plt.grid(True); plt.legend(); plt.title('b0 decomposition')
    plt.subplot(2,2,3); plt.plot(h,bounds['wilks_functional_bound']); plt.grid(True); plt.title('Wilks-style bound')
    plt.subplot(2,2,4); plt.plot(h,bounds['applicability_metric']); plt.axhline(4/9,ls='-.'); plt.grid(True); plt.title('Applicability')
    plt.tight_layout(); plt.savefig(output_dir/f'{name}.png',dpi=300,transparent=True); plt.close()
