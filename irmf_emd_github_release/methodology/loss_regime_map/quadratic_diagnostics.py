from __future__ import annotations
import numpy as np
from core_algorithms.robust_losses import evaluate_loss
from core_algorithms.strict_spokoiny_irmf import make_boundary_extension, spokoiny_kernel
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json
from .common import CORE_LOSSES, make_structural_case, run_loss_on_case, aggregate


def objective_terms(y,w,theta,loss,tuning):
    rho,psi,curv=evaluate_loss(y-theta,loss_name=loss,tuning=tuning)
    return float(np.sum(w*rho)),float(np.sum(-w*psi)),float(np.sum(w*curv))

def local_diagnostics(result, case, loss, tuning, delta_levels=(.01,.02,.05,.10,.20), probes=24):
    rows=[]; t=case['t']; n=len(t)
    for scale_idx,scale in enumerate(result.get('scale_history',[])):
        yext,text=make_boundary_extension(scale['Y_before'],t,scale.get('boundary_mode','periodic'))
        indices=np.unique(np.linspace(0,n-1,min(probes,n)).astype(int))
        for i in indices:
            w=spokoiny_kernel((t[i]-text)/scale['h']); valid=w>0
            y=yext[valid]; ww=w[valid]; theta=float(scale['S_k'][i]); loc_scale=float(np.std(y)+1e-10)
            L0,g,h=objective_terms(y,ww,theta,loss,tuning)
            _,_,curv=evaluate_loss(y-theta,loss_name=loss,tuning=tuning)
            info_terms=ww*curv
            concentration=float(np.max(np.abs(info_terms))/(np.sum(np.abs(info_terms))+1e-12)) if len(info_terms) else np.nan
            for dl in delta_levels:
                for sign in (-1.0,1.0):
                    delta=sign*dl*loc_scale; L1,_,_=objective_terms(y,ww,theta+delta,loss,tuning)
                    rem=abs(L1-L0-g*delta-.5*h*delta*delta)
                    rows.append({"loss_key":loss,"scale_index":scale_idx,"h":scale['h'],"probe_index":int(i),"delta_level":dl,"delta":delta,
                        "absolute_cubic_remainder":float(rem/(abs(delta)**3+1e-12)),
                        "relative_quadratic_remainder":float(rem/(.5*abs(h*delta*delta)+1e-12)),
                        "local_information":h,"curvature_concentration":concentration,"gradient_abs":abs(g),"support_count":int(np.sum(valid))})
    return rows

def run_quadratic_diagnostics(output_root, base_params, locked_tuning, n=500, fs=500.0,
        signals=("frequency_jump","impulsive_transient","intermittent_oscillation"), contamination_rates=(.0,.10,.20,.30), seeds=range(10), losses=CORE_LOSSES):
    out=ensure_dir(output_root); rows=[]
    for signal in signals:
     for rate in contamination_rates:
      for seed in seeds:
       case=make_structural_case(signal,'event_overlap',0,rate,15.0,.10,n,fs,seed)
       for loss in losses:
        result,_,_=run_loss_on_case(case,base_params,loss,locked_tuning[loss],f'quad_{signal}_{rate}_{seed}_{loss}')
        local=local_diagnostics(result,case,loss,locked_tuning[loss])
        for r in local:r.update({"signal":signal,"contamination_rate":rate,"seed":seed})
        rows.extend(local)
    metrics=("absolute_cubic_remainder","relative_quadratic_remainder","local_information","curvature_concentration","gradient_abs","support_count")
    agg=aggregate(rows,("loss_key","signal","contamination_rate","delta_level"),metrics)
    write_csv(rows,out/'quadratic_diagnostic_rows.csv'); write_csv(agg,out/'quadratic_diagnostic_aggregate.csv')
    write_json({"rows":rows,"aggregate":agg,"delta_levels":[.01,.02,.05,.10,.20],"definitions":{"absolute_cubic_remainder":"|R3|/(|Delta|^3+eps)","relative_quadratic_remainder":"|R3|/(0.5|Hessian Delta^2|+eps)"}},out/'quadratic_diagnostics.json')
    return rows,agg
