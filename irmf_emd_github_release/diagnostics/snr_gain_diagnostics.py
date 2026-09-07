#!/usr/bin/python
# coding: UTF-8

import numpy as np


def _safe_db(x):
    return 10.0 * np.log10(max(float(x), 1e-20))


def compute_snr_gain_diagnostics(Y_observed, X_clean, recovered):
    if X_clean is None or recovered is None:
        return {
            "input_snr_db": np.nan,
            "output_snr_db": np.nan,
            "snr_gain_db": np.nan,
        }

    Y_observed = np.asarray(Y_observed, dtype=float)
    X_clean = np.asarray(X_clean, dtype=float)
    recovered = np.asarray(recovered, dtype=float)

    input_noise = Y_observed - X_clean
    output_error = recovered - X_clean

    signal_power = np.mean(X_clean ** 2) + 1e-20
    input_noise_power = np.mean(input_noise ** 2) + 1e-20
    output_error_power = np.mean(output_error ** 2) + 1e-20

    input_snr_db = _safe_db(signal_power / input_noise_power)
    output_snr_db = _safe_db(signal_power / output_error_power)

    return {
        "input_snr_db": float(input_snr_db),
        "output_snr_db": float(output_snr_db),
        "snr_gain_db": float(output_snr_db - input_snr_db),
    }
