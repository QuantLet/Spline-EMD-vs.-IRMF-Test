from __future__ import annotations
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json
from .common import CORE_LOSSES, make_structural_case, run_loss_on_case, aggregate


def run_optimization_stress(output_root, base_params, locked_tuning, n=500, fs=500.0,
        signals=("frequency_jump","impulsive_transient","close_frequencies"),
        h_min_values=(.012,.008,.004), min_support_values=(3,5,9), initialization_offsets=(-2.0,-1.0,0.0,1.0,2.0),
        contamination_rates=(.10,.20,.30), seeds=range(10), losses=CORE_LOSSES):
    out=ensure_dir(output_root); rows=[]
    for signal in signals:
     for hmin in h_min_values:
      for support in min_support_values:
       for init in initialization_offsets:
        for rate in contamination_rates:
         for seed in seeds:
          case=make_structural_case(signal,'clustered_burst',0,rate,15.0,.10,n,fs,seed)
          params=dict(base_params,h_min=float(hmin),min_support_points=int(support))
          for loss in losses:
           _,_,row=run_loss_on_case(case,params,loss,locked_tuning[loss],f'opt_{signal}_{hmin}_{support}_{init}_{rate}_{seed}_{loss}',initialization_offset=init)
           row.update({"signal":signal,"h_min_stress":hmin,"min_support_stress":support,"initialization_offset":init,"contamination_rate":rate,"seed":seed})
           rows.append(row)
    metrics=("failure_rate","median_iterations","p95_iterations","objective_non_descent_rate","line_search_activation_rate","mean_line_search_steps","near_singular_hessian_rate","negative_curvature_rate","median_final_gradient_abs","p95_final_gradient_abs","minimum_local_information","local_information_cv","denoise_nmse","runtime_seconds")
    agg=aggregate(rows,("loss_key","signal","h_min_stress","min_support_stress","initialization_offset","contamination_rate"),metrics)
    write_csv(rows,out/'optimization_stress_rows.csv'); write_csv(agg,out/'optimization_stress_aggregate.csv')
    write_json({"rows":rows,"aggregate":agg,"solver":"common safeguarded Newton interface; initialization perturbation measured in local standard deviations"},out/'optimization_stress.json')
    return rows,agg
