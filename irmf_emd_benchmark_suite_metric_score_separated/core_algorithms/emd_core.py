#!/usr/bin/python
# coding: UTF-8

"""
EMD core decomposition.

This module contains only EMD execution.
EMD has no IRMF bandwidth h and no Spokoiny theory diagnostics.

Updates in v2:
- Exposes common PyEMD stopping thresholds for fairer sensitivity analysis.
"""

import numpy as np
from PyEMD import EMD


def run_emd_decomposition(
        Y,
        T,
        nbsym=2,
        spline_kind="cubic",
        max_imf=-1,
        dtype=np.float64,
        std_thr=None,
        svar_thr=None,
        total_power_thr=None,
        range_thr=None
):
    """
    Standard EMD decomposition using PyEMD.

    Parameters
    ----------
    nbsym : int
        Boundary mirroring points.
    spline_kind : str
        Envelope interpolation type.
    max_imf : int
        Maximum IMF count. -1 means PyEMD default/no explicit limit.
    std_thr, svar_thr, total_power_thr, range_thr : float or None
        Optional PyEMD stopping thresholds. If None, PyEMD defaults are used.
    """
    emd = EMD()
    emd.nbsym = nbsym
    emd.spline_kind = spline_kind
    emd.DTYPE = dtype

    if std_thr is not None:
        emd.std_thr = std_thr
    if svar_thr is not None:
        emd.svar_thr = svar_thr
    if total_power_thr is not None:
        emd.total_power_thr = total_power_thr
    if range_thr is not None:
        emd.range_thr = range_thr

    all_components = emd.emd(Y.astype(dtype), T.astype(dtype), max_imf)

    if len(all_components) <= 1:
        imfs = np.empty((0, len(Y)), dtype=dtype)
        residual = all_components[0] if len(all_components) == 1 else np.zeros_like(Y)
    else:
        imfs = all_components[:-1]
        residual = all_components[-1]

    config = {
        "nbsym": nbsym,
        "spline_kind": spline_kind,
        "max_imf": max_imf,
        "dtype": str(dtype),
        "std_thr": std_thr,
        "svar_thr": svar_thr,
        "total_power_thr": total_power_thr,
        "range_thr": range_thr
    }

    return imfs, residual, all_components, config


def run_pyemd_family_decomposition(
        Y,
        T,
        method="EMD",
        nbsym=2,
        spline_kind="cubic",
        max_imf=-1,
        dtype=np.float64,
        trials=50,
        noise_width=0.05,
        random_seed=0,
        **kwargs
):
    """
    Optional PyEMD-family baseline runner.

    method:
    - "EMD": always available if PyEMD.EMD works.
    - "EEMD": used only if PyEMD.EEMD is available.
    - "CEEMDAN": used only if PyEMD.CEEMDAN is available.

    Returns same structure as run_emd_decomposition plus config.
    """
    method = method.upper()

    if method == "EMD":
        imfs, residual, all_components, config = run_emd_decomposition(
            Y=Y,
            T=T,
            nbsym=nbsym,
            spline_kind=spline_kind,
            max_imf=max_imf,
            dtype=dtype,
            **kwargs
        )
        config["method"] = "EMD"
        return imfs, residual, all_components, config

    if method == "EEMD":
        try:
            from PyEMD import EEMD
        except Exception as exc:
            raise ImportError("PyEMD.EEMD is not available in this environment.") from exc

        model = EEMD(trials=trials, noise_width=noise_width)
        model.noise_seed(random_seed)
        comps = model.eemd(Y.astype(dtype), T.astype(dtype), max_imf=max_imf)

    elif method == "CEEMDAN":
        try:
            from PyEMD import CEEMDAN
        except Exception as exc:
            raise ImportError("PyEMD.CEEMDAN is not available in this environment.") from exc

        model = CEEMDAN(trials=trials, noise_width=noise_width)
        model.noise_seed(random_seed)
        comps = model.ceemdan(Y.astype(dtype), T.astype(dtype), max_imf=max_imf)

    else:
        raise ValueError("method must be EMD, EEMD, or CEEMDAN.")

    # EEMD/CEEMDAN typically returns IMFs without explicit residual.
    imfs = comps
    residual = Y - np.sum(imfs, axis=0) if len(imfs) > 0 else Y.copy()
    all_components = np.vstack([imfs, residual]) if len(imfs) > 0 else np.array([residual])

    config = {
        "method": method,
        "nbsym": nbsym,
        "spline_kind": spline_kind,
        "max_imf": max_imf,
        "dtype": str(dtype),
        "trials": trials,
        "noise_width": noise_width,
        "random_seed": random_seed,
    }

    return imfs, residual, all_components, config
