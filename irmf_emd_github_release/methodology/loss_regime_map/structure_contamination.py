from __future__ import annotations
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json
from .common import CORE_LOSSES, make_structural_case, run_loss_on_case, paired_local_metrics, aggregate


def run_structure_contamination(output_root, base_params, locked_tuning, n=500, fs=500.0,
        signals=("frequency_jump","impulsive_transient","intermittent_oscillation"),
        kinds=("isolated_spike","clustered_burst","event_overlap","asymmetric_cluster"),
        distances=(-20,-10,-5,0,5,10,20), rates=(.02,.05,.10,.20), scales=(5.0,10.0,20.0), seeds=range(10), losses=CORE_LOSSES):
    out=ensure_dir(output_root); rows=[]
    for signal in signals:
      for kind in kinds:
       for distance in distances:
        for rate in rates:
         for scale in scales:
          for seed in seeds:
           case=make_structural_case(signal,kind,distance,rate,scale,.10,n,fs,seed)
           base=dict(case); base['Y']=case['Y_baseline']
           for loss in losses:
            _,rec0,_=run_loss_on_case(base,base_params,loss,locked_tuning[loss],f'base_{signal}_{seed}_{loss}')
            _,rec,row=run_loss_on_case(case,base_params,loss,locked_tuning[loss],f'struct_{signal}_{kind}_{distance}_{rate}_{scale}_{seed}_{loss}')
            row.update({"signal":signal,"contamination_kind":kind,"distance_points":distance,"contamination_rate":rate,"outlier_scale":scale,"seed":seed})
            row.update(paired_local_metrics(case['X_clean'],rec,rec0,case['event_mask'],case['contamination_neighborhood']))
            rows.append(row)
    metrics=("denoise_nmse","denoise_corr","event_mse","smooth_mse","error_spread_ratio","event_amplitude_bias","failure_rate","runtime_seconds")
    agg=aggregate(rows,("loss_key","signal","contamination_kind","distance_points","contamination_rate","outlier_scale"),metrics)
    write_csv(rows,out/'structure_contamination_rows.csv'); write_csv(agg,out/'structure_contamination_aggregate.csv')
    write_json({"rows":rows,"aggregate":agg,"paired_baseline":"same seed Gaussian-noise case without added contamination"},out/'structure_contamination.json')
    return rows,agg
