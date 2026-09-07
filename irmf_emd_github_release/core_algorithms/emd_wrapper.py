#!/usr/bin/python
# coding: UTF-8

"""Classical EMD wrapper."""

import numpy as np


def final_residual(y, imfs):
    imfs = np.asarray(imfs, dtype=float)
    if imfs.ndim == 1:
        imfs = imfs[None, :]
    if imfs.size == 0:
        return np.asarray(y, dtype=float)
    return np.asarray(y, dtype=float) - np.sum(imfs, axis=0)


def run_emd(
        y,
        max_imf=-1,
        spline_kind="cubic",
        nbsym=2,
        std_thr=None,
        svar_thr=None,
        total_power_thr=None,
        range_thr=None,
):
    from PyEMD import EMD

    emd = EMD()
    emd.nbsym = nbsym

    if spline_kind is not None:
        try:
            emd.spline_kind = spline_kind
        except Exception:
            pass

    if std_thr is not None:
        emd.std_thr = std_thr
    if svar_thr is not None:
        emd.svar_thr = svar_thr
    if total_power_thr is not None:
        emd.total_power_thr = total_power_thr
    if range_thr is not None:
        emd.range_thr = range_thr

    imfs = emd.emd(np.asarray(y, dtype=float), max_imf=max_imf)
    residual = final_residual(y, imfs)

    return {
        "method": "EMD",
        "imfs": imfs,
        "residual": residual,
        "nbsym": nbsym,
        "spline_kind": spline_kind,
        "max_imf": max_imf,
        "std_thr": std_thr,
        "svar_thr": svar_thr,
        "total_power_thr": total_power_thr,
        "range_thr": range_thr,
    }
