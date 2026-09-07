"""Loss-methodology pipeline: rho-function theory, ablation, contamination and optimization diagnostics."""
from .pipeline import (
    run_loss_properties,
    run_loss_calibration,
    run_loss_ablation_stage,
    run_contamination_stage,
    run_optimization_stage,
    run_methodology_pipeline,
)
