#!/usr/bin/python
# coding: UTF-8

from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Dict, List, Any, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt

from core_algorithms.strict_spokoiny_irmf import strict_spokoiny_irmf
from diagnostics.theoretical_diagnostics import compute_theoretical_diagnostics_from_scale_history
from noise_bank.noise_models import generate_noise
from signal_bank.synthetic_signals import get_signal, list_synthetic_signals
from run_theoretical_diagnostics import load_physical_best_params, parse_selection, DEFAULT_SIGNALS, DEFAULT_NOISES, AVAILABLE_NOISES, write_csv, write_json, choose_params

EPS = 1e-14


def finite_pair(x, y):
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float); m=np.isfinite(x)&np.isfinite(y); return x[m],y[m]

def safe_corr(x,y):
    x,y=finite_pair(x,y)
    if len(x)<2 or np.std(x)<EPS or np.std(y)<EPS: return float('nan')
    return float(np.corrcoef(x,y)[0,1])

def average_ranks(x):
    x=np.asarray(x,dtype=float); order=np.argsort(x,kind='mergesort'); ranks=np.empty(len(x),dtype=float); i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]: j+=1
        ranks[order[i:j]]=0.5*(i+1+j); i=j
    return ranks

def safe_spearman(x,y):
    x,y=finite_pair(x,y)
    if len(x)<2: return float('nan')
    return safe_corr(average_ranks(x), average_ranks(y))

def norm_share(x):
    x=np.asarray(x,dtype=float); out=np.full_like(x,np.nan,dtype=float); m=np.isfinite(x)&(x>=0); s=float(np.sum(x[m]))
    if s>EPS: out[m]=x[m]/s
    return out

def rank_consistency(x,y):
    x,y=finite_pair(x,y); n=len(x); total=0; agree=0
    for i in range(n):
        for j in range(i+1,n):
            sx=np.sign(x[i]-x[j]); sy=np.sign(y[i]-y[j])
            if sx==0 or sy==0: continue
            total+=1; agree+=int(sx==sy)
    return float(agree/total) if total else float('nan')

def mse(x,y): return float(np.mean((np.asarray(x)-np.asarray(y))**2))

def run_single_rep(x_clean, t, clean_imfs, signal, noise, sigma, seed, params, args):
    eps=generate_noise(noise, len(x_clean), sigma=sigma, seed=seed)
    y=x_clean+eps
    noisy_imfs, residual, _, scale_history = strict_spokoiny_irmf(y, T=t, **params, boundary_mode=args.boundary_mode, verbose=False)
    sigma_trace=float(np.sum(eps**2))
    diag=compute_theoretical_diagnostics_from_scale_history(scale_history, T=t, H=params['H'], sigma_trace=sigma_trace, method='hutchinson', operator_mode=args.operator_mode, probe_count=args.probe_count, random_seed=seed+100000)
    rows=[]; kmax=min(len(clean_imfs),len(noisy_imfs),len(diag.get('rows',[])))
    for i in range(kmax):
        th=diag['rows'][i]; e_mse=mse(noisy_imfs[i], clean_imfs[i]); e_rmse=float(np.sqrt(e_mse)); risk=float(th.get('imf_trace_risk_norm',np.nan))
        rows.append({
            'rep_seed': int(seed), 'signal': signal, 'noise': noise, 'sigma': float(sigma), 'k': int(i+1), 'h': float(th.get('h',np.nan)),
            'empirical_imf_mse': e_mse, 'empirical_imf_rmse': e_rmse, 'empirical_imf_rmse_sq': e_rmse**2,
            'theoretical_imf_trace_risk_norm': risk, 'theoretical_residual_trace_risk_norm': float(th.get('residual_trace_risk_norm',np.nan)),
            'fisher_step_l2_mean': float(th.get('fisher_step_l2_mean',np.nan)), 'wilks_gap_mean': float(th.get('wilks_gap_mean',np.nan)),
            'hessian_zero_fraction': float(th.get('hessian_zero_fraction',np.nan)), 'risk_to_mse_ratio': float(risk/(e_mse+EPS)) if np.isfinite(risk) else float('nan')
        })
    risk=[r['theoretical_imf_trace_risk_norm'] for r in rows]; emse=[r['empirical_imf_mse'] for r in rows]
    rshare=norm_share(risk); eshare=norm_share(emse)
    for i,r in enumerate(rows):
        r['theoretical_risk_share']=float(rshare[i]); r['empirical_mse_share']=float(eshare[i]); r['share_gap_theory_minus_empirical']=float(rshare[i]-eshare[i])
    rec=np.sum(np.asarray(noisy_imfs),axis=0) if noisy_imfs else np.zeros_like(x_clean)
    return {'rows':rows,'global':{
        'rep_seed':int(seed),'signal':signal,'noise':noise,'sigma':float(sigma),'k_count_noisy':int(len(noisy_imfs)),
        'denoise_mse_vs_clean_signal':mse(rec,x_clean),'corr_risk_mse_by_k':safe_corr(risk,emse),'spearman_risk_mse_by_k':safe_spearman(risk,emse),
        'rank_consistency_risk_mse_by_k':rank_consistency(risk,emse),'corr_normalized_share_by_k':safe_corr(rshare,eshare)
    }}

def aggregate(rows):
    out=[]
    for k in sorted(set(int(r['k']) for r in rows)):
        g=[r for r in rows if int(r['k'])==k]
        def mean(key): return float(np.nanmean([x.get(key,np.nan) for x in g]))
        def std(key):
            vals=np.asarray([x.get(key,np.nan) for x in g],dtype=float); vals=vals[np.isfinite(vals)]
            return float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0
        first=g[0]
        out.append({'signal':first['signal'],'noise':first['noise'],'sigma':first['sigma'],'k':k,'h_mean':mean('h'),
                    'empirical_imf_mse_mean':mean('empirical_imf_mse'),'empirical_imf_mse_std':std('empirical_imf_mse'),
                    'empirical_imf_rmse_mean':mean('empirical_imf_rmse'),'empirical_imf_rmse_sq_mean':mean('empirical_imf_rmse_sq'),
                    'theoretical_imf_trace_risk_norm_mean':mean('theoretical_imf_trace_risk_norm'),
                    'theoretical_residual_trace_risk_norm_mean':mean('theoretical_residual_trace_risk_norm'),
                    'fisher_step_l2_mean':mean('fisher_step_l2_mean'),'wilks_gap_mean':mean('wilks_gap_mean'),
                    'hessian_zero_fraction_mean':mean('hessian_zero_fraction'),'risk_to_mse_ratio_mean':mean('risk_to_mse_ratio'),'n_rep':len(g)})
    risk=norm_share([r['theoretical_imf_trace_risk_norm_mean'] for r in out]); err=norm_share([r['empirical_imf_mse_mean'] for r in out])
    for i,r in enumerate(out):
        r['theoretical_risk_share_from_means']=float(risk[i]); r['empirical_mse_share_from_means']=float(err[i]); r['share_gap_from_means']=float(risk[i]-err[i])
    return out

def summarize(agg, global_rows, meta):
    risk=[r['theoretical_imf_trace_risk_norm_mean'] for r in agg]; mse_v=[r['empirical_imf_mse_mean'] for r in agg]; rmse=[r['empirical_imf_rmse_mean'] for r in agg]
    rshare=[r['theoretical_risk_share_from_means'] for r in agg]; eshare=[r['empirical_mse_share_from_means'] for r in agg]
    out=dict(meta)
    out.update({'corr_theory_empirical_mse_level':safe_corr(risk,mse_v),'corr_theory_empirical_rmse_squared':safe_corr(risk,[x*x for x in rmse]),
                'corr_log_theory_log_empirical_mse':safe_corr(np.log(np.asarray(risk)+EPS),np.log(np.asarray(mse_v)+EPS)),
                'spearman_theory_empirical_mse':safe_spearman(risk,mse_v),'normalized_share_corr_theory_empirical_mse':safe_corr(rshare,eshare),
                'rank_consistency_theory_empirical_mse':rank_consistency(risk,mse_v),
                'mean_rep_corr_risk_mse_by_k':float(np.nanmean([g['corr_risk_mse_by_k'] for g in global_rows])),
                'mean_rep_spearman_risk_mse_by_k':float(np.nanmean([g['spearman_risk_mse_by_k'] for g in global_rows]))})
    return out

def plot_combo(agg, out_dir):
    k=np.asarray([r['k'] for r in agg]); risk=np.asarray([r['theoretical_imf_trace_risk_norm_mean'] for r in agg]); err=np.asarray([r['empirical_imf_mse_mean'] for r in agg])
    plt.figure(figsize=(8,5)); plt.plot(k,risk,marker='s',label='Theoretical risk/n'); plt.plot(k,err,marker='o',label='Empirical IMF MSE'); plt.xlabel('IMF k'); plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(out_dir/'risk_vs_error_curve.png',dpi=180); plt.close()
    plt.figure(figsize=(6,5)); plt.scatter(risk,err); [plt.annotate(str(int(kk)),(risk[i],err[i])) for i,kk in enumerate(k)]; plt.xlabel('Theoretical risk/n'); plt.ylabel('Empirical MSE'); plt.tight_layout(); plt.savefig(out_dir/'risk_vs_error_scatter.png',dpi=180); plt.close()

def plot_heat(rows,key,path,title):
    signals=[s for s in DEFAULT_SIGNALS if any(r['signal']==s for r in rows)]; noises=[n for n in DEFAULT_NOISES if any(r['noise']==n for r in rows)]
    mat=np.full((len(signals),len(noises)),np.nan)
    for i,s in enumerate(signals):
        for j,n in enumerate(noises):
            vals=[r.get(key,np.nan) for r in rows if r['signal']==s and r['noise']==n]
            if vals: mat[i,j]=float(vals[0])
    plt.figure(figsize=(10,5)); im=plt.imshow(mat,aspect='auto'); plt.colorbar(im,label=key); plt.xticks(np.arange(len(noises)),noises,rotation=35,ha='right'); plt.yticks(np.arange(len(signals)),signals); plt.title(title)
    for i in range(len(signals)):
        for j in range(len(noises)):
            if np.isfinite(mat[i,j]): plt.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',fontsize=8)
    plt.tight_layout(); path.parent.mkdir(parents=True,exist_ok=True); plt.savefig(path,dpi=180); plt.close()

def run_combo(args, signal, noise, best_map):
    params, src=choose_params(signal,noise,args,best_map)
    out_dir=args.output_dir/signal/noise/f"sigma_{args.sigma:g}"; out_dir.mkdir(parents=True,exist_ok=True)
    t=np.arange(args.n)/float(args.fs); t_unit=np.linspace(0,1,args.n,endpoint=False); x=get_signal(signal,t_unit)
    clean_imfs,_,_,_=strict_spokoiny_irmf(x,T=t,**params,boundary_mode=args.boundary_mode,verbose=False)
    all_rows=[]; global_rows=[]
    print(f"\n[{signal} | {noise} | sigma={args.sigma:g} | params={src}] clean IMF count: {len(clean_imfs)}")
    print(f"  using h1={params['h1']}, a={params['a']}, h_min={params['h_min']}, H={params['H']}")
    for rep in range(args.mc):
        res=run_single_rep(x,t,clean_imfs,signal,noise,args.sigma,args.seed+rep,params,args)
        all_rows.extend(res['rows']); global_rows.append(res['global'])
        print(f"  replication {rep+1:>3}/{args.mc} done")
    agg=aggregate(all_rows); meta={'signal':signal,'noise':noise,'sigma':args.sigma,'n':args.n,'fs':args.fs,'mc':args.mc,'param_source_used':src,**params}
    summ=summarize(agg,global_rows,meta)
    write_csv(out_dir/'risk_vs_error_replications.csv',all_rows); write_json(out_dir/'risk_vs_error_replications.json',all_rows)
    write_csv(out_dir/'risk_vs_error_by_imf.csv',agg); write_json(out_dir/'risk_vs_error_by_imf.json',agg)
    write_csv(out_dir/'global_replication_summary.csv',global_rows); write_json(out_dir/'global_replication_summary.json',global_rows)
    write_json(out_dir/'correlation_summary.json',summ); write_json(out_dir/'theory_empirical_full.json',{'summary':summ,'by_imf':agg,'replications':all_rows,'global_replications':global_rows})
    plot_combo(agg,out_dir)
    print(f"  corr risk vs MSE = {summ['corr_theory_empirical_mse_level']:.4f}; spearman = {summ['spearman_theory_empirical_mse']:.4f}; rank = {summ['rank_consistency_theory_empirical_mse']:.4f}")
    return summ

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--signal',default='all'); p.add_argument('--noise',default='all'); p.add_argument('--sigma',type=float,default=0.2); p.add_argument('--n',type=int,default=500); p.add_argument('--fs',type=float,default=500.0); p.add_argument('--seed',type=int,default=100); p.add_argument('--mc',type=int,default=20)
    p.add_argument('--h1',type=float,default=0.12); p.add_argument('--a',type=float,default=1.4); p.add_argument('--h-min',dest='h_min',type=float,default=0.005); p.add_argument('--H',type=float,default=0.8); p.add_argument('--param-source',choices=['physical_best','fixed'],default='physical_best'); p.add_argument('--best-summary',default=None)
    p.add_argument('--operator-mode',choices=['robust_influence','linear_mean'],default='robust_influence'); p.add_argument('--probe-count',type=int,default=64); p.add_argument('--boundary-mode',choices=['periodic','mirror'],default='periodic'); p.add_argument('--output-dir',type=Path,default=Path('outputs/theoretical_vs_empirical'))
    return p.parse_args()

def main():
    args=parse_args(); signals=parse_selection(args.signal,DEFAULT_SIGNALS,list_synthetic_signals(),'signal'); noises=parse_selection(args.noise,DEFAULT_NOISES,AVAILABLE_NOISES,'noise'); best_map=load_physical_best_params(args.best_summary) if args.param_source=='physical_best' else {}
    print('='*80); print('IRMF theoretical vs empirical validation — FIX9 Spokoiny extra diagnostics'); print('='*80); print(f'signals={signals}\nnoises={noises}\nsigma={args.sigma}\nmc={args.mc}\nparam_source={args.param_source}\noperator_mode={args.operator_mode}\noutput_dir={args.output_dir}')
    master=[]; total=len(signals)*len(noises); c=0
    for s in signals:
        for n in noises:
            c+=1; print('\n'+'='*80); print(f'Combination {c}/{total}: {s} + {n}'); print('='*80); master.append(run_combo(args,s,n,best_map))
    args.output_dir.mkdir(parents=True,exist_ok=True); write_csv(args.output_dir/'master_correlation_summary.csv',master); write_json(args.output_dir/'master_correlation_summary.json',master)
    plot_heat(master,'corr_theory_empirical_mse_level',args.output_dir/'heatmap_corr_theory_empirical_mse.png','Corr: risk vs empirical MSE')
    plot_heat(master,'spearman_theory_empirical_mse',args.output_dir/'heatmap_spearman_theory_empirical_mse.png','Spearman: risk vs empirical MSE')
    plot_heat(master,'rank_consistency_theory_empirical_mse',args.output_dir/'heatmap_rank_consistency.png','Rank consistency')
    print(f"\nSaved master summary: {args.output_dir/'master_correlation_summary.csv'}")

if __name__=='__main__': main()
