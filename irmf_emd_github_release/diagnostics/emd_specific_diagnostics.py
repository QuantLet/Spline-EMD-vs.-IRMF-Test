
import numpy as np

def residual_trend_smoothness(residual):
    if len(residual)<3: return 0.
    d2=np.diff(residual,n=2); return float(np.mean(d2**2)/(np.var(residual)+1e-10))
def residual_low_frequency_dominance(residual,fs,cutoff_ratio=.10):
    if len(residual)<4: return 0.
    freqs=np.fft.rfftfreq(len(residual),d=1/fs); psd=np.abs(np.fft.rfft(residual))**2; cutoff=cutoff_ratio*(fs/2); return float(np.sum(psd[freqs<=cutoff])/(np.sum(psd)+1e-10))
def compute_emd_specific_diagnostics(residual,fs): return {'emd_residual_trend_smoothness':residual_trend_smoothness(residual),'emd_residual_low_frequency_dominance':residual_low_frequency_dominance(residual,fs)}
