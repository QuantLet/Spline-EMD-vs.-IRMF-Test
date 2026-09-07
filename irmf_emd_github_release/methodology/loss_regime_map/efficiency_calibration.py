from __future__ import annotations
import numpy as np
from core_algorithms.robust_losses import evaluate_loss, LOSS_REGISTRY
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json

TUNING_PARAMETER = {"huber":"delta", "pseudo_huber":"delta", "fair":"c", "tukey":"c", "gaussian_smoothed_median":"H"}
SEARCH_GRIDS = {
    "huber": np.linspace(.25, 4.0, 300), "pseudo_huber": np.linspace(.15, 5.0, 300),
    "fair": np.linspace(.15, 5.0, 300), "tukey": np.linspace(1.0, 10.0, 300),
    "gaussian_smoothed_median": np.linspace(.10, 5.0, 300),
}

def gaussian_expectation(values, nodes, weights):
    return float(np.sum(weights*values)/np.sqrt(np.pi))

def loss_moments(loss, tuning, order=120):
    nodes, weights=np.polynomial.hermite.hermgauss(order); z=np.sqrt(2.0)*nodes
    _,psi,curv=evaluate_loss(z, loss_name=loss, tuning=tuning)
    a=gaussian_expectation(psi**2,nodes,weights); b=gaussian_expectation(curv,nodes,weights)
    avar=a/(b*b+1e-15); are=1.0/avar
    _,_,c0=evaluate_loss(np.asarray([0.0]),loss_name=loss,tuning=tuning)
    grid=np.linspace(-30,30,20001); _,pg,_=evaluate_loss(grid,loss_name=loss,tuning=tuning)
    return {"gaussian_asymptotic_variance":avar,"gaussian_are":are,"origin_curvature":float(c0[0]),"score_bound_proxy":float(np.max(np.abs(pg)))}

def calibrate_by_target(loss, target, criterion):
    if loss in ("l1","l2"):
        tuning=dict(LOSS_REGISTRY[loss].default_tuning); m=loss_moments(loss,tuning); return tuning,m
    param=TUNING_PARAMETER[loss]; best=None
    for v in SEARCH_GRIDS[loss]:
        tuning={param:float(v)}; m=loss_moments(loss,tuning)
        gap=abs(m[criterion]-target)
        if best is None or gap<best[0]: best=(gap,tuning,m)
    return best[1],best[2]

def run_efficiency_calibration(output_root, losses=("l2","l1","huber","pseudo_huber","fair","tukey","gaussian_smoothed_median"), target_are=.95):
    out=ensure_dir(output_root); rows=[]; protocols={}
    modes=(("equal_gaussian_efficiency","gaussian_are",target_are),
           ("equal_origin_curvature","origin_curvature",1.0),
           ("equal_score_bound","score_bound_proxy",1.0))
    for mode,criterion,target in modes:
        selected={}
        for loss in losses:
            tuning,m=calibrate_by_target(loss,target,criterion)
            selected[loss]=tuning
            rows.append({"calibration_mode":mode,"loss_key":loss,"tuning":tuning,"target":target,"criterion":criterion,"criterion_gap":abs(m[criterion]-target),**m})
        protocols[mode]=selected
    write_csv(rows,out/'fair_calibration_table.csv'); write_json(rows,out/'fair_calibration_table.json')
    write_json(protocols,out/'locked_tuning_by_calibration_mode.json')
    write_json({"main_mode":"equal_gaussian_efficiency","target_are":target_are,"note":"L1 and L2 have no robustness tuning and are reported at their natural efficiencies."},out/'protocol.json')
    return protocols,rows
