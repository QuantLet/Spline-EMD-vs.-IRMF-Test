from experiments.paper_pipeline_utils import ensure_dir, write_json
from .efficiency_calibration import run_efficiency_calibration
from .structure_contamination import run_structure_contamination
from .optimization_stress import run_optimization_stress
from .quadratic_diagnostics import run_quadratic_diagnostics
from .regime_summary import run_regime_summary


def run_loss_regime_map(output_root, base_params, n=500, fs=500.0, quick=False):
    root=ensure_dir(output_root)
    protocols,cal_rows=run_efficiency_calibration(root/'00_efficiency_calibration')
    locked=protocols['equal_gaussian_efficiency']
    kwargs={}
    if quick:
        kwargs=dict(seeds=range(2))
    srows,sagg=run_structure_contamination(root/'01_structure_contamination',base_params,locked,n=n,fs=fs,**kwargs)
    orows,oagg=run_optimization_stress(root/'02_optimization_stress',base_params,locked,n=n,fs=fs,**kwargs)
    qrows,qagg=run_quadratic_diagnostics(root/'03_quadratic_diagnostics',base_params,locked,n=n,fs=fs,**kwargs)
    # Keep canonical aliases expected by summary.
    import shutil
    aliases=((root/'01_structure_contamination',root/'structure_contamination'),(root/'02_optimization_stress',root/'optimization_stress'),(root/'03_quadratic_diagnostics',root/'quadratic_diagnostics'))
    for src,dst in aliases:
        if dst.exists(): shutil.rmtree(dst)
        shutil.copytree(src,dst)
    summary=run_regime_summary(root/'04_regime_summary',root.parent)
    dashboard={'main_calibration_mode':'equal_gaussian_efficiency','locked_tuning':locked,'n_calibration_rows':len(cal_rows),'n_structure_rows':len(srows),'n_optimization_rows':len(orows),'n_quadratic_rows':len(qrows),'regime_summary':summary,'quick':quick}
    write_json(dashboard,root/'loss_regime_dashboard.json')
    return dashboard
