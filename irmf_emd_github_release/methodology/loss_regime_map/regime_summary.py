from __future__ import annotations
from collections import defaultdict
import json, numpy as np
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json

LOWER_BETTER={"denoise_nmse","event_mse","smooth_mse","error_spread_ratio","failure_rate","p95_iterations","line_search_activation_rate","near_singular_hessian_rate","absolute_cubic_remainder","relative_quadratic_remainder","curvature_concentration"}

def _load(path):
    p=path
    try:return json.loads(open(p,encoding='utf-8').read())
    except Exception:return None

def _rank_rows(rows, regime_keys, metrics):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r.get(k) for k in regime_keys)].append(r)
    output=[]
    for regime,subset in groups.items():
        byloss=defaultdict(list)
        for r in subset:byloss[r['loss_key']].append(r)
        for metric in metrics:
            scores=[]
            for loss,rs in byloss.items():
                vals=np.asarray([x.get(metric,np.nan) for x in rs],float);vals=vals[np.isfinite(vals)]
                if vals.size:scores.append((loss,float(np.mean(vals))))
            if not scores:continue
            reverse=metric not in LOWER_BETTER;scores=sorted(scores,key=lambda x:x[1],reverse=reverse)
            best=scores[0][1]
            for rank,(loss,val) in enumerate(scores,1):
                output.append({**{k:v for k,v in zip(regime_keys,regime)},"metric":metric,"loss_key":loss,"rank":rank,"mean_value":val,"gap_to_best":abs(val-best)})
    return output

def run_regime_summary(output_root, methodology_root):
    out=ensure_dir(output_root); root=str(methodology_root)
    sources={
      'structure':_load(root+'/05_loss_regime_map/structure_contamination/structure_contamination.json'),
      'optimization':_load(root+'/05_loss_regime_map/optimization_stress/optimization_stress.json'),
      'quadratic':_load(root+'/05_loss_regime_map/quadratic_diagnostics/quadratic_diagnostics.json')}
    rankings=[]
    if sources['structure']:
        rankings+=_rank_rows(sources['structure']['rows'],('signal','contamination_kind','distance_points','contamination_rate','outlier_scale'),('denoise_nmse','event_mse','smooth_mse','error_spread_ratio','failure_rate'))
    if sources['optimization']:
        rankings+=_rank_rows(sources['optimization']['rows'],('signal','h_min_stress','min_support_stress','initialization_offset','contamination_rate'),('failure_rate','p95_iterations','line_search_activation_rate','near_singular_hessian_rate','denoise_nmse'))
    if sources['quadratic']:
        rankings+=_rank_rows(sources['quadratic']['rows'],('signal','contamination_rate','delta_level'),('absolute_cubic_remainder','relative_quadratic_remainder','local_information','curvature_concentration'))
    wins=defaultdict(lambda:defaultdict(int)); totals=defaultdict(lambda:defaultdict(int))
    for r in rankings:
        totals[r['metric']][r['loss_key']]+=1
        if r['rank']==1:wins[r['metric']][r['loss_key']]+=1
    summary=[]
    for metric,losses in totals.items():
        for loss,n in losses.items():summary.append({'metric':metric,'loss_key':loss,'n_regimes':n,'wins':wins[metric][loss],'win_rate':wins[metric][loss]/n})
    write_csv(rankings,out/'regime_rankings.csv');write_csv(summary,out/'loss_regime_win_rates.csv')
    write_json({'rankings':rankings,'win_rates':summary,'interpretation_rule':'Map advantages and disadvantages; do not claim universal dominance.'},out/'loss_regime_map.json')
    return {'n_rankings':len(rankings),'win_rates':summary}
