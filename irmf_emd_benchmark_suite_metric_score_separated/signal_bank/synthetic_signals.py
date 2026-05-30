from pathlib import Path

import numpy as np

def signal_stationary_multi_sine(t): return 1.2*np.sin(2*np.pi*1*t)+0.5*np.sin(2*np.pi*3*t)+0.25*np.sin(2*np.pi*9*t)
def signal_chirp(t):
    f0=3.; k=22.; return np.sin(2*np.pi*(f0*t+0.5*k*t**2))
def signal_am_fm(t):
    amp=1+0.45*np.sin(2*np.pi*t); phase=2*np.pi*(8*t+2*np.sin(2*np.pi*.8*t)); return amp*np.sin(phase)
def signal_impulsive_transient(t):
    x=signal_stationary_multi_sine(t); idx=(t>.45)&(t<.47); x=x.copy(); x[idx]+=1.5*np.sin(2*np.pi*80*t[idx]); return x
def signal_frequency_jump(t):
    x=np.zeros_like(t); x[t<.5]=np.sin(2*np.pi*5*t[t<.5]); x[t>=.5]=np.sin(2*np.pi*20*t[t>=.5]); return x
SIGNAL_REGISTRY={'stationary_multi_sine':signal_stationary_multi_sine,'chirp':signal_chirp,'am_fm':signal_am_fm,'impulsive_transient':signal_impulsive_transient,'frequency_jump':signal_frequency_jump}
def get_signal(name,t): return SIGNAL_REGISTRY[name](t)
def list_synthetic_signals(): return list(SIGNAL_REGISTRY.keys())


def signal_piecewise_holder(t):
    x = np.zeros_like(t, dtype=float)
    left = t < 0.5
    right = ~left
    x[left] = 1.2 * np.sin(2 * np.pi * 2 * t[left])
    x[right] = 0.7 * np.sin(2 * np.pi * 9 * t[right]) + 0.25 * np.sin(2 * np.pi * 21 * t[right]) + 0.5
    return x


def signal_heteroscedastic_smoothness(t):
    base = np.sin(2 * np.pi * 2 * t)
    local_rough = np.zeros_like(t)
    mask = (t >= 0.45) & (t <= 0.75)
    local_rough[mask] = 0.45 * np.sin(2 * np.pi * 18 * t[mask])
    jump = 0.35 * (t > 0.6)
    return base + local_rough + jump


SIGNAL_REGISTRY['piecewise_holder'] = signal_piecewise_holder
SIGNAL_REGISTRY['heteroscedastic_smoothness'] = signal_heteroscedastic_smoothness


# ============================================================
# True/reference instantaneous frequencies for synthetic signals
# ============================================================

def get_true_frequencies(signal_name, t):
    """
    Return true/reference instantaneous frequency curves for synthetic signals.

    Format:
        [
            {"label": "...", "freq": scalar_or_array},
            ...
        ]

    These curves are used only for Hilbert-ridge visualization.
    If no meaningful true frequency exists, returns [].
    """
    signal_name = str(signal_name)

    if signal_name == "stationary_multi_sine":
        return [
            {"label": "true 1 Hz", "freq": np.ones_like(t) * 1.0},
            {"label": "true 3 Hz", "freq": np.ones_like(t) * 3.0},
            {"label": "true 9 Hz", "freq": np.ones_like(t) * 9.0},
        ]

    if signal_name == "chirp":
        return [
            {"label": "true chirp IF", "freq": 3.0 + 22.0 * t},
        ]

    if signal_name == "am_fm":
        return [
            {"label": "true AM-FM IF", "freq": 8.0 + 3.2 * np.pi * np.cos(2 * np.pi * 0.8 * t)},
        ]

    if signal_name == "impulsive_transient":
        return [
            {"label": "true 1 Hz", "freq": np.ones_like(t) * 1.0},
            {"label": "true 3 Hz", "freq": np.ones_like(t) * 3.0},
            {"label": "true 9 Hz", "freq": np.ones_like(t) * 9.0},
            {"label": "transient 80 Hz", "freq": np.ones_like(t) * 80.0},
        ]

    if signal_name == "frequency_jump":
        return [
            {"label": "true jump IF", "freq": np.where(t < 0.5, 5.0, 12.0)},
        ]

    if signal_name == "piecewise_holder":
        return [
            {"label": "left 2 Hz", "freq": np.where(t < 0.5, 2.0, np.nan)},
            {"label": "right 9 Hz", "freq": np.where(t >= 0.5, 9.0, np.nan)},
            {"label": "right 21 Hz", "freq": np.where(t >= 0.5, 21.0, np.nan)},
        ]

    if signal_name == "heteroscedastic_smoothness":
        return [
            {"label": "base 2 Hz", "freq": np.ones_like(t) * 2.0},
            {"label": "local 18 Hz", "freq": np.where((t >= 0.45) & (t <= 0.75), 18.0, np.nan)},
        ]

    return []


def infer_signal_name_from_path(path):
    """
    Infer synthetic signal name from an output path.
    This is used only for plotting true-frequency overlays.
    """
    known = [
        "stationary_multi_sine",
        "chirp",
        "am_fm",
        "impulsive_transient",
        "frequency_jump",
        "piecewise_holder",
        "heteroscedastic_smoothness",
    ]

    parts = [str(p) for p in Path(path).parts]

    for name in known:
        if name in parts:
            return name

    joined = str(path)
    for name in known:
        if name in joined:
            return name

    return None


# ============================================================
# V11 true/reference instantaneous frequencies for synthetic signals
# ============================================================

def get_true_frequencies(signal_name, t):
    """
    Return true/reference instantaneous frequency curves for synthetic signals.

    Format:
        [
            {"label": "...", "freq": scalar_or_array},
            ...
        ]

    These curves are used only for Hilbert-ridge visualization.
    If no meaningful true frequency exists, returns [].
    """
    signal_name = str(signal_name)

    if signal_name == "stationary_multi_sine":
        return [
            {"label": "true 1 Hz", "freq": np.ones_like(t) * 1.0},
            {"label": "true 3 Hz", "freq": np.ones_like(t) * 3.0},
            {"label": "true 9 Hz", "freq": np.ones_like(t) * 9.0},
        ]

    if signal_name == "chirp":
        return [
            {"label": "true chirp IF", "freq": 3.0 + 22.0 * t},
        ]

    if signal_name == "am_fm":
        return [
            {"label": "true AM-FM IF", "freq": 8.0 + 3.2 * np.pi * np.cos(2 * np.pi * 0.8 * t)},
        ]

    if signal_name == "impulsive_transient":
        return [
            {"label": "true 1 Hz", "freq": np.ones_like(t) * 1.0},
            {"label": "true 3 Hz", "freq": np.ones_like(t) * 3.0},
            {"label": "true 9 Hz", "freq": np.ones_like(t) * 9.0},
            {"label": "transient 80 Hz", "freq": np.ones_like(t) * 80.0},
        ]

    if signal_name == "frequency_jump":
        return [
            {"label": "true jump IF", "freq": np.where(t < 0.5, 5.0, 12.0)},
        ]

    if signal_name == "piecewise_holder":
        return [
            {"label": "left 2 Hz", "freq": np.where(t < 0.5, 2.0, np.nan)},
            {"label": "right 9 Hz", "freq": np.where(t >= 0.5, 9.0, np.nan)},
            {"label": "right 21 Hz", "freq": np.where(t >= 0.5, 21.0, np.nan)},
        ]

    if signal_name == "heteroscedastic_smoothness":
        return [
            {"label": "base 2 Hz", "freq": np.ones_like(t) * 2.0},
            {"label": "local 18 Hz", "freq": np.where((t >= 0.45) & (t <= 0.75), 18.0, np.nan)},
        ]

    return []


def infer_signal_name_from_path(path):
    """
    Infer synthetic signal name from an output path.

    Used only for plot annotation when experiments do not explicitly pass
    signal_name into plotting helpers.
    """
    known = [
        "stationary_multi_sine",
        "chirp",
        "am_fm",
        "frequency_jump",
        "impulsive_transient",
        "piecewise_holder",
        "heteroscedastic_smoothness",
    ]

    parts = [str(p) for p in Path(path).parts]

    for name in known:
        if name in parts:
            return name

    joined = str(path)
    for name in known:
        if name in joined:
            return name

    return None

def get_true_components(signal_name, t):
    """
    Return true component modes for synthetic signals when available.
    Used only for optional IMF recovery diagnostics.
    """
    signal_name = str(signal_name)

    if signal_name == "stationary_multi_sine":
        return np.vstack([
            1.2 * np.sin(2 * np.pi * 1 * t),
            0.5 * np.sin(2 * np.pi * 3 * t),
            0.25 * np.sin(2 * np.pi * 9 * t),
        ])

    if signal_name == "chirp":
        f0 = 3.0
        k = 22.0
        return np.vstack([
            np.sin(2 * np.pi * (f0 * t + 0.5 * k * t ** 2))
        ])

    if signal_name == "am_fm":
        amp = 1 + 0.45 * np.sin(2 * np.pi * t)
        phase = 2 * np.pi * (8 * t + 2 * np.sin(2 * np.pi * 0.8 * t))
        return np.vstack([
            amp * np.sin(phase)
        ])

    if signal_name == "frequency_jump":
        return np.vstack([
            np.where(t < 0.5, np.sin(2 * np.pi * 5 * t), np.sin(2 * np.pi * 12 * t))
        ])

    if signal_name == "impulsive_transient":
        transient = np.zeros_like(t)
        idx = (t > 0.45) & (t < 0.47)
        transient[idx] = 1.5 * np.sin(2 * np.pi * 80 * t[idx])
        return np.vstack([
            1.2 * np.sin(2 * np.pi * 1 * t),
            0.5 * np.sin(2 * np.pi * 3 * t),
            0.25 * np.sin(2 * np.pi * 9 * t),
            transient,
        ])

    return None
