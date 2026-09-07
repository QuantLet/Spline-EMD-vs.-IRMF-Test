#!/usr/bin/python
# coding: UTF-8
import numpy as np

def boundary_mask(n, edge_fraction=0.10):
    edge_n=max(1,int(n*edge_fraction)); m=np.zeros(n,dtype=bool); m[:edge_n]=True; m[-edge_n:]=True; return m

def boundary_energy_ratio(x, edge_fraction=0.10):
    x=np.asarray(x,dtype=float); m=boundary_mask(len(x),edge_fraction); return float(np.sum(x[m]**2)/(np.sum(x**2)+1e-10))

def boundary_imf_energy_ratio(imfs, edge_fraction=0.10):
    if len(imfs)==0: return 0.0
    return float(np.mean([boundary_energy_ratio(imf,edge_fraction) for imf in imfs]))

def boundary_reconstruction_error_ratio(Y_observed, imfs, residual, edge_fraction=0.10):
    recon=(np.sum(imfs,axis=0)+residual) if len(imfs)>0 else residual
    return boundary_energy_ratio(recon-Y_observed, edge_fraction)

def endpoint_jump_amplification(x, edge_fraction=0.10):
    x=np.asarray(x,dtype=float); n=len(x); edge_n=max(2,int(n*edge_fraction))
    left=np.mean(x[:edge_n]); right=np.mean(x[-edge_n:])
    interior=x[edge_n:-edge_n] if n>2*edge_n else x
    return float(abs(right-left)/(np.std(interior)+1e-10))

def compute_boundary_diagnostics(Y_observed, imfs, residual, edge_fraction=0.10):
    return {
        'boundary_signal_energy_ratio': boundary_energy_ratio(Y_observed,edge_fraction),
        'boundary_imf_energy_ratio': boundary_imf_energy_ratio(imfs,edge_fraction),
        'boundary_residual_energy_ratio': boundary_energy_ratio(residual,edge_fraction),
        'boundary_reconstruction_error_ratio': boundary_reconstruction_error_ratio(Y_observed,imfs,residual,edge_fraction),
        'endpoint_jump_amplification_observed': endpoint_jump_amplification(Y_observed,edge_fraction),
        'endpoint_jump_amplification_residual': endpoint_jump_amplification(residual,edge_fraction),
    }
