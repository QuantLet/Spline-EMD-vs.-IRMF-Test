
import numpy as np

def _nearest_match_relative_drift(a,b):
    if len(a)==0 or len(b)==0: return np.nan
    a=np.array(a); b=np.array(b); return float(np.mean([abs(b[np.argmin(abs(b-f))]-f)/(abs(f)+1e-10) for f in a]))
def compute_pairwise_noise_robustness(low,high):
    return {'imf_count_change':int(high['imf_count']-low['imf_count']),'center_frequency_relative_drift':_nearest_match_relative_drift(low['center_freqs'],high['center_freqs']),'general_score_change':float(high['general_physical_score']-low['general_physical_score']),'spectral_leakage_growth':float(high['spectral_leakage']-low['spectral_leakage']),'io_growth':float(high['strict_io']-low['strict_io']),'ifs_growth':float(high['ifs']-low['ifs'])}
def summarize_robustness_by_signal(results_by_noise):
    sigmas=sorted(results_by_noise.keys())
    return {} if len(sigmas)<2 else compute_pairwise_noise_robustness(results_by_noise[sigmas[0]],results_by_noise[sigmas[-1]])
