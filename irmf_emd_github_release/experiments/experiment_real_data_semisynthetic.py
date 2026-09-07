#!/usr/bin/python
# coding: UTF-8

"""Semi-synthetic real-data contamination experiment."""

from pathlib import Path
import numpy as np

from project_config import (
    GLOBAL_CEEMDAN_PARAMS,
    GLOBAL_EEMD_PARAMS,
    GLOBAL_EMD_PARAMS,
    GLOBAL_IRMF_PARAMS,
)
from core_algorithms.eemd_wrapper import run_eemd
from core_algorithms.ceemdan_wrapper import run_ceemdan
from diagnostics.shared_physical_diagnostics import evaluate_shared_physical_diagnostics
from experiments.experiment_utils import method_result_summary, run_fixed_emd_case, run_fixed_irmf_case
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json
from noise_bank.noise_models import generate_noise
from signal_bank.real_data_loader import load_real_signal_csv


def _normalize_time(T):
    T = np.asarray(T, dtype=float)
    if len(T) <= 1:
        return np.asarray([0.0])
    return (T - T[0]) / (T[-1] - T[0])


def _run_eemd_like(method, Y, X_clean, fs, emd_params, eemd_params, ceemdan_params, algorithm_seed):
    method = method.upper()
    if method == "EEMD":
        raw = run_eemd(
            Y,
            max_imf=eemd_params.get("max_imf", emd_params.get("max_imf", -1)),
            trials=eemd_params.get("trials", 100),
            noise_width=eemd_params.get("noise_width", 0.05),
            parallel=eemd_params.get("parallel", False),
            random_seed=algorithm_seed,
        )
    elif method == "CEEMDAN":
        raw = run_ceemdan(
            Y,
            max_imf=ceemdan_params.get("max_imf", emd_params.get("max_imf", -1)),
            trials=ceemdan_params.get("trials", 100),
            epsilon=ceemdan_params.get("epsilon", 0.005),
            parallel=ceemdan_params.get("parallel", False),
            random_seed=algorithm_seed,
        )
    else:
        raise ValueError(method)
    physical = evaluate_shared_physical_diagnostics(
        Y_observed=Y,
        X_clean=X_clean,
        imfs=raw["imfs"],
        residual=raw["residual"],
        fs=fs,
        residual_penalty_mode="none",
        true_components=None,
    )
    return {"method": method, **raw, **physical}


def run_real_data_semisynthetic_contamination(
        output_root,
        data_dir=None,
        irmf_params=GLOBAL_IRMF_PARAMS,
        emd_params=GLOBAL_EMD_PARAMS,
        eemd_params=GLOBAL_EEMD_PARAMS,
        ceemdan_params=GLOBAL_CEEMDAN_PARAMS,
        noises=("gaussian", "laplace", "huber_contamination", "impulsive"),
        sigmas=(0.10, 0.20),
        fs=None,
        data_seed=0,
        algorithm_seed=20260715,
):
    """
    Use real signals as clean references, add controlled synthetic noise, and
    compare IRMF/EMD/EEMD/CEEMDAN under known contamination.
    """
    output_root = ensure_dir(output_root)
    data_dir = Path(data_dir) if data_dir is not None else None
    rows = []
    if data_dir is not None and data_dir.exists():
        for path in sorted(data_dir.glob("*.csv")):
            T, X_clean, inferred_fs = load_real_signal_csv(path, fs=fs)
            t_unit = _normalize_time(T)
            X_clean = np.asarray(X_clean, dtype=float)
            for noise_name in noises:
                for sigma in sigmas:
                    noise = generate_noise(noise_name, len(X_clean), sigma=sigma, seed=data_seed)
                    Y = X_clean + noise
                    row = {
                        "dataset": path.stem,
                        "path": str(path),
                        "noise": noise_name,
                        "sigma": sigma,
                        "data_seed": data_seed,
                        "algorithm_seed": algorithm_seed,
                        "fs": inferred_fs,
                        "n": int(len(X_clean)),
                    }
                    irmf = run_fixed_irmf_case(
                        Y=Y,
                        X_clean=X_clean,
                        t=t_unit,
                        fs=inferred_fs,
                        irmf_params=irmf_params,
                        expected_noise_ratio=float(np.sum(noise ** 2) / (np.sum(Y ** 2) + 1e-12)),
                        true_components=None,
                        run_id=f"real_semisynth_{path.stem}_{noise_name}_{sigma}_irmf",
                    )
                    emd = run_fixed_emd_case(
                        Y=Y,
                        X_clean=X_clean,
                        t=t_unit,
                        fs=inferred_fs,
                        emd_params=emd_params,
                        true_components=None,
                        run_id=f"real_semisynth_{path.stem}_{noise_name}_{sigma}_emd",
                    )
                    row["IRMF"] = method_result_summary(irmf)
                    row["EMD"] = method_result_summary(emd)
                    for method in ("EEMD", "CEEMDAN"):
                        try:
                            result = _run_eemd_like(method, Y, X_clean, inferred_fs, emd_params, eemd_params, ceemdan_params, algorithm_seed)
                            row[method] = method_result_summary(result)
                        except Exception as exc:
                            row[method] = {"method": method, "error": str(exc)}
                    rows.append(row)

    if not rows:
        (output_root / "README_real_semisynthetic_template.txt").write_text(
            "Place clean real-signal CSV files in a directory and pass --real-data-dir. "
            "CSV format: one column y with fs supplied, or two columns time,y.\n",
            encoding="utf-8",
        )
    write_json({
        "section": "7 Semi-synthetic Bridge Validation",
        "data_dir": str(data_dir) if data_dir is not None else None,
        "noises": list(noises),
        "sigmas": list(sigmas),
        "purpose": "Real clean signals with controlled contamination, enabling reconstruction and robustness metrics.",
    }, output_root / "section_7_semisynthetic_bridge_protocol.json")
    write_json(rows, output_root / "section_7_semisynthetic_bridge_validation.json")
    write_csv(rows, output_root / "section_7_semisynthetic_bridge_validation.csv")
    return {"n_cases": len(rows)}
