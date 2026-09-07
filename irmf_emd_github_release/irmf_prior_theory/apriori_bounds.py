
import numpy as np

def compute_apriori_risk_bound(h_grid,n,sigma,c3,bias_amplitude):
    Nh=n*h_grid; tau3=c3/np.sqrt(Nh); st=sigma/np.sqrt(Nh); bias=bias_amplitude*h_grid**2; b0=st+bias
    app=tau3*b0
    return {'h':h_grid,'tau_3':tau3,'b_0':b0,'applicability_metric':app,'is_applicable':app<4/9,'metric_distance_bound':1.5*b0,'physical_error_bound':1.5*b0/np.sqrt(Nh),'wilks_functional_bound':b0**2+tau3*b0**3,'stochastic_component':st,'bias_component':bias}
