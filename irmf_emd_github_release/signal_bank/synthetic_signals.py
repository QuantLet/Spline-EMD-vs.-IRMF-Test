#!/usr/bin/python
# coding: UTF-8

"""
Synthetic Signal Bank for the IRMF vs EMD benchmark suite.

Updated FIX9+ benchmark design:
    7 signals × 8 noise models = 56 main benchmark cases

Signal Bank:
    1. stationary_multi_sine      — baseline stable decomposition
    2. chirp                      — smooth nonstationary frequency tracking
    3. am_fm                      — amplitude/frequency modulation and Hilbert analysis
    4. frequency_jump             — abrupt regime change and mode mixing
    5. impulsive_transient        — local high-frequency transient
    6. intermittent_oscillation   — classical EMD intermittency/mode-mixing test
    7. close_frequencies          — frequency resolution test
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

DTYPE = np.float64


# ============================================================
# Core synthetic signals
# ============================================================

def signal_stationary_multi_sine(t: np.ndarray) -> np.ndarray:
    """
    Stable multi-sine baseline.

    Components:
        1 Hz, 3 Hz, 9 Hz

    Tests:
        Basic IMF separation, reconstruction accuracy,
        IMF recovery correlation/RMSE.
    """
    t = np.asarray(t, dtype=DTYPE)
    return (
        1.2 * np.sin(2 * np.pi * 1.0 * t)
        + 0.5 * np.sin(2 * np.pi * 3.0 * t)
        + 0.25 * np.sin(2 * np.pi * 9.0 * t)
    ).astype(DTYPE)


def signal_stationary_multi_sine_variant(
        t: np.ndarray,
        freqs=(1.0, 3.0, 9.0),
        amps=(1.2, 0.5, 0.25),
) -> np.ndarray:
    """Multi-sine variant used for within-class spacing studies."""
    t = np.asarray(t, dtype=DTYPE)
    x = np.zeros_like(t, dtype=DTYPE)
    for amp, freq in zip(amps, freqs):
        x += float(amp) * np.sin(2 * np.pi * float(freq) * t)
    return x.astype(DTYPE)


def signal_chirp(t: np.ndarray, f0: float = 3.0, k: float = 22.0) -> np.ndarray:
    """
    Linear chirp with continuously increasing instantaneous frequency.

    Phase:
        phi(t) = 2π(f0 t + 0.5 k t^2)

    Instantaneous frequency:
        f(t) = f0 + k t

    Tests:
        Smooth nonstationarity and frequency tracking.
    """
    t = np.asarray(t, dtype=DTYPE)
    return np.sin(2 * np.pi * (f0 * t + 0.5 * k * t ** 2)).astype(DTYPE)


def signal_am_fm(t: np.ndarray) -> np.ndarray:
    """
    AM-FM signal with time-varying envelope and phase.

    Tests:
        Hilbert analysis, envelope recovery, instantaneous frequency quality.
    """
    t = np.asarray(t, dtype=DTYPE)
    amp = 1.0 + 0.45 * np.sin(2 * np.pi * 1.0 * t)
    phase = 2 * np.pi * (8.0 * t + 2.0 * np.sin(2 * np.pi * 0.8 * t))
    return (amp * np.sin(phase)).astype(DTYPE)


def signal_am_fm_variant(
        t: np.ndarray,
        amp_depth: float = 0.45,
        phase_depth: float = 2.0,
) -> np.ndarray:
    """AM-FM variant with controllable modulation strength."""
    t = np.asarray(t, dtype=DTYPE)
    amp = 1.0 + float(amp_depth) * np.sin(2 * np.pi * 1.0 * t)
    phase = 2 * np.pi * (8.0 * t + float(phase_depth) * np.sin(2 * np.pi * 0.8 * t))
    return (amp * np.sin(phase)).astype(DTYPE)


def signal_frequency_jump(
    t: np.ndarray,
    f_left: float = 5.0,
    f_right: float = 15.0,
    jump_time: float = 0.5,
) -> np.ndarray:
    """
    Abrupt frequency regime change.

    Left half:  f_left Hz
    Right half: f_right Hz

    Tests:
        Regime-change sensitivity, mode mixing, spectral leakage.
    """
    t = np.asarray(t, dtype=DTYPE)
    x = np.zeros_like(t, dtype=DTYPE)
    left = t < jump_time
    x[left] = np.sin(2 * np.pi * f_left * t[left])
    x[~left] = np.sin(2 * np.pi * f_right * t[~left])
    return x.astype(DTYPE)


def signal_frequency_jump_multiple(t: np.ndarray) -> np.ndarray:
    """Frequency-jump variant with multiple abrupt regimes."""
    t = np.asarray(t, dtype=DTYPE)
    x = np.zeros_like(t, dtype=DTYPE)
    seg1 = t < 0.33
    seg2 = (t >= 0.33) & (t < 0.66)
    seg3 = t >= 0.66
    x[seg1] = np.sin(2 * np.pi * 5.0 * t[seg1])
    x[seg2] = np.sin(2 * np.pi * 15.0 * t[seg2])
    x[seg3] = np.sin(2 * np.pi * 8.0 * t[seg3])
    return x.astype(DTYPE)


def signal_impulsive_transient(t: np.ndarray) -> np.ndarray:
    """
    Stable multi-sine signal with a short local high-frequency transient.

    Transient:
        80 Hz active only on 0.45 < t < 0.47

    Tests:
        Local robustness, leakage control, transient localization.
    """
    t = np.asarray(t, dtype=DTYPE)
    x = signal_stationary_multi_sine(t).copy()
    idx = (t > 0.45) & (t < 0.47)
    x[idx] += 1.5 * np.sin(2 * np.pi * 80.0 * t[idx])
    return x.astype(DTYPE)


def signal_impulsive_transient_variant(
        t: np.ndarray,
        start: float = 0.45,
        stop: float = 0.47,
        amplitude: float = 1.5,
        freq: float = 80.0,
        base_freqs=(1.0, 3.0, 9.0),
        base_amps=(1.2, 0.5, 0.25),
) -> np.ndarray:
    """Transient variant with controllable duration and amplitude."""
    t = np.asarray(t, dtype=DTYPE)
    x = signal_stationary_multi_sine_variant(t, freqs=base_freqs, amps=base_amps).copy()
    idx = (t > float(start)) & (t < float(stop))
    x[idx] += float(amplitude) * np.sin(2 * np.pi * float(freq) * t[idx])
    return x.astype(DTYPE)


def signal_intermit_oscillation(t: np.ndarray, f: float = 10.0) -> np.ndarray:
    """
    Intermittent oscillation.

    Oscillation appears only in two separated time windows:
        0.20 < t < 0.40
        0.60 < t < 0.80

    Tests:
        Classical EMD mode mixing caused by intermittency.
    """
    t = np.asarray(t, dtype=DTYPE)
    x = np.zeros_like(t, dtype=DTYPE)
    idx1 = (t > 0.20) & (t < 0.40)
    idx2 = (t > 0.60) & (t < 0.80)
    x[idx1] = np.sin(2 * np.pi * f * t[idx1])
    x[idx2] = np.sin(2 * np.pi * f * t[idx2])
    return x.astype(DTYPE)


def signal_intermittent_windows(
        t: np.ndarray,
        windows,
        f: float = 10.0,
) -> np.ndarray:
    """Intermittent oscillation over a supplied list of active intervals."""
    t = np.asarray(t, dtype=DTYPE)
    x = np.zeros_like(t, dtype=DTYPE)
    active = np.zeros_like(t, dtype=bool)
    for start, stop in windows:
        active |= (t > float(start)) & (t < float(stop))
    x[active] = np.sin(2 * np.pi * float(f) * t[active])
    return x.astype(DTYPE)


def signal_close_frequencies(
        t: np.ndarray,
        f1: float = 10.0,
        f2: float = 11.0,
        amp1: float = 1.0,
        amp2: float = 1.0,
) -> np.ndarray:
    """
    Two close sinusoidal components.

    x(t) = sin(2π f1 t) + sin(2π f2 t)

    Because f1 and f2 are close, the signal shows beating.

    Tests:
        Frequency resolution, IMF separation under near-overlapping spectra.
    """
    t = np.asarray(t, dtype=DTYPE)
    return (
        float(amp1) * np.sin(2 * np.pi * f1 * t)
        + float(amp2) * np.sin(2 * np.pi * f2 * t)
    ).astype(DTYPE)


# ============================================================
# Challenging signal regimes for stress/variant studies
# ============================================================

def _phase_from_frequency(t: np.ndarray, freq: np.ndarray) -> np.ndarray:
    """Numerically integrate an instantaneous-frequency curve into phase."""
    t = np.asarray(t, dtype=DTYPE)
    freq = np.asarray(freq, dtype=DTYPE)
    if len(t) <= 1:
        return np.zeros_like(t, dtype=DTYPE)
    dt = float(np.median(np.diff(t)))
    return 2 * np.pi * np.cumsum(freq) * dt


def signal_crossing_chirps_variant_components(
        t: np.ndarray,
        up_slope: float = 24.0,
        down_slope: float = 18.0,
        upper_start: float = 5.0,
        lower_start: float = 29.0,
        second_amplitude: float = 0.85,
) -> np.ndarray:
    """Two chirps with configurable crossing geometry."""
    t = np.asarray(t, dtype=DTYPE)
    f_up = float(upper_start) + float(up_slope) * t
    f_down = float(lower_start) - float(down_slope) * t
    c1 = np.sin(_phase_from_frequency(t, f_up))
    c2 = float(second_amplitude) * np.sin(_phase_from_frequency(t, f_down) + 0.4)
    return np.vstack([c1, c2]).astype(DTYPE)


def signal_crossing_chirps_components(t: np.ndarray) -> np.ndarray:
    """Two chirps with crossing instantaneous frequencies."""
    return signal_crossing_chirps_variant_components(t)


def signal_crossing_chirps(t: np.ndarray) -> np.ndarray:
    return np.sum(signal_crossing_chirps_components(t), axis=0).astype(DTYPE)


def signal_time_varying_close_frequencies_variant_components(
        t: np.ndarray,
        min_separation: float = 0.35,
        curvature: float = 2.2,
        center_modulation: float = 1.5,
        second_amplitude: float = 0.9,
) -> np.ndarray:
    """Two close components with configurable minimum IF separation."""
    t = np.asarray(t, dtype=DTYPE)
    center = 11.0 + float(center_modulation) * np.sin(2 * np.pi * 0.45 * t)
    separation = float(min_separation) + float(curvature) * (t - 0.5) ** 2
    f1 = center - 0.5 * separation
    f2 = center + 0.5 * separation
    c1 = np.sin(_phase_from_frequency(t, f1))
    c2 = float(second_amplitude) * np.sin(_phase_from_frequency(t, f2) + 0.25)
    return np.vstack([c1, c2]).astype(DTYPE)


def signal_time_varying_close_frequencies_components(t: np.ndarray) -> np.ndarray:
    """Two close components whose separation narrows around the middle."""
    return signal_time_varying_close_frequencies_variant_components(t)


def signal_time_varying_close_frequencies(t: np.ndarray) -> np.ndarray:
    return np.sum(signal_time_varying_close_frequencies_components(t), axis=0).astype(DTYPE)


def signal_piecewise_am_fm_discontinuity_variant_components(
        t: np.ndarray,
        split_time: float = 0.52,
        right_amp_base: float = 0.65,
        right_amp_depth: float = 0.45,
        right_freq_base: float = 13.0,
        right_freq_depth: float = 2.8,
) -> np.ndarray:
    """AM-FM component with configurable discontinuity strength."""
    t = np.asarray(t, dtype=DTYPE)
    amp = np.where(
        t < float(split_time),
        1.0 + 0.25 * np.sin(2 * np.pi * t),
        float(right_amp_base) + float(right_amp_depth) * np.sin(2 * np.pi * 1.4 * t),
    )
    freq = np.where(
        t < float(split_time),
        6.0 + 2.0 * np.sin(2 * np.pi * 0.8 * t),
        float(right_freq_base) + float(right_freq_depth) * np.cos(2 * np.pi * 0.7 * t),
    )
    carrier = amp * np.sin(_phase_from_frequency(t, freq))
    low = 0.35 * np.sin(2 * np.pi * 1.2 * t)
    return np.vstack([low, carrier]).astype(DTYPE)


def signal_piecewise_am_fm_discontinuity_components(t: np.ndarray) -> np.ndarray:
    """AM-FM component with a local amplitude and frequency discontinuity."""
    return signal_piecewise_am_fm_discontinuity_variant_components(t)


def signal_piecewise_am_fm_discontinuity(t: np.ndarray) -> np.ndarray:
    return np.sum(signal_piecewise_am_fm_discontinuity_components(t), axis=0).astype(DTYPE)


def signal_damped_oscillation_variant_components(
        t: np.ndarray,
        fast_decay: float = 2.6,
        slow_decay: float = 0.7,
) -> np.ndarray:
    """Damped oscillations with configurable damping rates."""
    t = np.asarray(t, dtype=DTYPE)
    c1 = 1.25 * np.exp(-float(fast_decay) * t) * np.sin(2 * np.pi * 14.0 * t)
    c2 = 0.55 * np.exp(-float(slow_decay) * t) * np.sin(2 * np.pi * 4.0 * t + 0.2)
    return np.vstack([c2, c1]).astype(DTYPE)


def signal_damped_oscillation_components(t: np.ndarray) -> np.ndarray:
    """Damped medium/high-frequency oscillations plus a weak low component."""
    return signal_damped_oscillation_variant_components(t)


def signal_damped_oscillation(t: np.ndarray) -> np.ndarray:
    return np.sum(signal_damped_oscillation_components(t), axis=0).astype(DTYPE)


def signal_trend_plus_oscillation_variant_components(
        t: np.ndarray,
        trend_curvature: float = 1.4,
        trend_slope: float = -0.25,
) -> np.ndarray:
    """Trend plus oscillations with configurable trend strength."""
    t = np.asarray(t, dtype=DTYPE)
    trend = float(trend_curvature) * (t - 0.5) ** 2 + float(trend_slope) * t
    slow = 0.55 * np.sin(2 * np.pi * 2.0 * t)
    fast = 0.22 * np.sin(2 * np.pi * 18.0 * t + 0.3)
    return np.vstack([trend, slow, fast]).astype(DTYPE)


def signal_trend_plus_oscillation_components(t: np.ndarray) -> np.ndarray:
    """Nonlinear trend combined with two oscillatory scales."""
    return signal_trend_plus_oscillation_variant_components(t)


def signal_trend_plus_oscillation(t: np.ndarray) -> np.ndarray:
    return np.sum(signal_trend_plus_oscillation_components(t), axis=0).astype(DTYPE)


def signal_buried_weak_component_variant_components(
        t: np.ndarray,
        weak_amplitude: float = 0.12,
        modulation_amplitude: float = 0.18,
) -> np.ndarray:
    """Strong component with configurable weak-component amplitude."""
    t = np.asarray(t, dtype=DTYPE)
    strong = 1.35 * np.sin(2 * np.pi * 3.0 * t)
    weak = float(weak_amplitude) * np.sin(2 * np.pi * 31.0 * t + 0.5)
    mod = float(modulation_amplitude) * np.sin(2 * np.pi * 7.0 * t)
    return np.vstack([strong, mod, weak]).astype(DTYPE)


def signal_buried_weak_component_components(t: np.ndarray) -> np.ndarray:
    """Strong low-frequency component with a weak high-frequency component."""
    return signal_buried_weak_component_variant_components(t)


def signal_buried_weak_component(t: np.ndarray) -> np.ndarray:
    return np.sum(signal_buried_weak_component_components(t), axis=0).astype(DTYPE)


def signal_non_sinusoidal_periodic_variant_components(
        t: np.ndarray,
        harmonic_scale: float = 1.0,
) -> np.ndarray:
    """Periodic signal with configurable harmonic richness."""
    t = np.asarray(t, dtype=DTYPE)
    fundamental = np.sin(2 * np.pi * 5.0 * t)
    scale = float(harmonic_scale)
    h2 = 0.45 * scale * np.sin(2 * np.pi * 10.0 * t)
    h3 = 0.28 * scale * np.sin(2 * np.pi * 15.0 * t)
    h5 = 0.18 * scale * np.sin(2 * np.pi * 25.0 * t)
    return np.vstack([fundamental, h2, h3, h5]).astype(DTYPE)


def signal_non_sinusoidal_periodic_components(t: np.ndarray) -> np.ndarray:
    """Harmonic-rich periodic signal represented by explicit harmonics."""
    return signal_non_sinusoidal_periodic_variant_components(t)


def signal_non_sinusoidal_periodic(t: np.ndarray) -> np.ndarray:
    return np.sum(signal_non_sinusoidal_periodic_components(t), axis=0).astype(DTYPE)


def signal_transient_train_variant_components(t: np.ndarray, events=None) -> np.ndarray:
    """Transient train with configurable event spacing and amplitudes."""
    t = np.asarray(t, dtype=DTYPE)
    base = 0.55 * np.sin(2 * np.pi * 2.5 * t)
    transient = np.zeros_like(t, dtype=DTYPE)
    if events is None:
        events = (
            (0.18, 0.020, 1.0, 55.0),
            (0.37, 0.015, 0.75, 72.0),
            (0.63, 0.030, 1.15, 48.0),
            (0.84, 0.018, 0.90, 85.0),
        )
    for center, width, amp, freq in events:
        env = np.exp(-0.5 * ((t - center) / width) ** 2)
        transient += amp * env * np.sin(2 * np.pi * freq * t)
    return np.vstack([base, transient]).astype(DTYPE)


def signal_transient_train_components(t: np.ndarray) -> np.ndarray:
    """Irregular local transient train on top of a smooth carrier."""
    return signal_transient_train_variant_components(t)


def signal_transient_train(t: np.ndarray) -> np.ndarray:
    return np.sum(signal_transient_train_components(t), axis=0).astype(DTYPE)


# Backward-compatible alias in case older scripts used the shorter name.
signal_intermittent_oscillation = signal_intermit_oscillation


SIGNAL_REGISTRY: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "stationary_multi_sine": signal_stationary_multi_sine,
    "chirp": signal_chirp,
    "am_fm": signal_am_fm,
    "frequency_jump": signal_frequency_jump,
    "impulsive_transient": signal_impulsive_transient,
    "intermittent_oscillation": signal_intermit_oscillation,
    "close_frequencies": signal_close_frequencies,
    # Section 4 development-only waveform instances.  These share benchmark
    # signal regimes but use distinct formulas/parameters for parameter
    # selection, so the fixed-parameter benchmark does not reuse exact
    # development waveforms.
    "stationary_multi_sine_dev": lambda t: signal_stationary_multi_sine_variant(
        t, freqs=(1.5, 4.0, 10.0), amps=(1.0, 0.55, 0.30)
    ),
    "chirp_dev": lambda t: signal_chirp(t, f0=4.0, k=18.0),
    "am_fm_dev": lambda t: signal_am_fm_variant(t, amp_depth=0.35, phase_depth=1.35),
    "impulsive_transient_dev": lambda t: signal_impulsive_transient_variant(
        t,
        start=0.34,
        stop=0.365,
        amplitude=1.3,
        freq=70.0,
        base_freqs=(1.5, 4.5, 8.5),
        base_amps=(1.0, 0.45, 0.30),
    ),
    "close_frequencies_dev": lambda t: signal_close_frequencies(t, f1=9.0, f2=11.5),
    # Section 7 signal-variant robustness study.
    "stationary_multi_sine_wide": lambda t: signal_stationary_multi_sine_variant(t, freqs=(1.0, 4.0, 12.0)),
    "stationary_multi_sine_medium": lambda t: signal_stationary_multi_sine_variant(t, freqs=(1.0, 3.0, 9.0)),
    "stationary_multi_sine_close": lambda t: signal_stationary_multi_sine_variant(t, freqs=(6.0, 8.0, 10.0), amps=(1.0, 0.7, 0.45)),
    "stationary_multi_sine_balanced_amplitudes": lambda t: signal_stationary_multi_sine_variant(t, freqs=(1.0, 3.0, 9.0), amps=(0.8, 0.8, 0.8)),
    "stationary_multi_sine_high_imbalance": lambda t: signal_stationary_multi_sine_variant(t, freqs=(1.0, 3.0, 9.0), amps=(1.4, 0.35, 0.12)),
    "stationary_multi_sine_wide_balanced": lambda t: signal_stationary_multi_sine_variant(t, freqs=(1.0, 4.0, 12.0), amps=(0.8, 0.8, 0.8)),
    "stationary_multi_sine_close_high_imbalance": lambda t: signal_stationary_multi_sine_variant(t, freqs=(6.0, 8.0, 10.0), amps=(1.4, 0.35, 0.12)),
    "chirp_slow": lambda t: signal_chirp(t, f0=3.0, k=10.0),
    "chirp_fast": lambda t: signal_chirp(t, f0=3.0, k=35.0),
    "chirp_narrow_band": lambda t: signal_chirp(t, f0=6.0, k=12.0),
    "chirp_broad_band": lambda t: signal_chirp(t, f0=1.5, k=30.0),
    "chirp_slow_narrow_band": lambda t: signal_chirp(t, f0=6.0, k=10.0),
    "chirp_fast_broad_band": lambda t: signal_chirp(t, f0=1.5, k=35.0),
    "am_fm_weak": lambda t: signal_am_fm_variant(t, amp_depth=0.20, phase_depth=0.80),
    "am_fm_strong": lambda t: signal_am_fm_variant(t, amp_depth=0.70, phase_depth=3.00),
    "am_fm_am_weak": lambda t: signal_am_fm_variant(t, amp_depth=0.20, phase_depth=2.00),
    "am_fm_am_strong": lambda t: signal_am_fm_variant(t, amp_depth=0.70, phase_depth=2.00),
    "am_fm_fm_weak": lambda t: signal_am_fm_variant(t, amp_depth=0.45, phase_depth=0.80),
    "am_fm_fm_strong": lambda t: signal_am_fm_variant(t, amp_depth=0.45, phase_depth=3.00),
    "frequency_jump_up": lambda t: signal_frequency_jump(t, f_left=5.0, f_right=15.0),
    "frequency_jump_down": lambda t: signal_frequency_jump(t, f_left=15.0, f_right=5.0),
    "frequency_jump_multiple": signal_frequency_jump_multiple,
    "frequency_jump_small_magnitude": lambda t: signal_frequency_jump(t, f_left=8.0, f_right=12.0),
    "frequency_jump_large_magnitude": lambda t: signal_frequency_jump(t, f_left=3.0, f_right=21.0),
    "frequency_jump_early": lambda t: signal_frequency_jump(t, f_left=5.0, f_right=15.0, jump_time=0.35),
    "frequency_jump_late": lambda t: signal_frequency_jump(t, f_left=5.0, f_right=15.0, jump_time=0.65),
    "frequency_jump_small_early": lambda t: signal_frequency_jump(t, f_left=8.0, f_right=12.0, jump_time=0.35),
    "frequency_jump_large_late": lambda t: signal_frequency_jump(t, f_left=3.0, f_right=21.0, jump_time=0.65),
    "impulsive_transient_short": lambda t: signal_impulsive_transient_variant(t, start=0.45, stop=0.465, amplitude=1.5),
    "impulsive_transient_long": lambda t: signal_impulsive_transient_variant(t, start=0.42, stop=0.50, amplitude=1.5),
    "impulsive_transient_strong": lambda t: signal_impulsive_transient_variant(t, start=0.45, stop=0.47, amplitude=2.5),
    "impulsive_transient_weak": lambda t: signal_impulsive_transient_variant(t, start=0.45, stop=0.47, amplitude=0.75),
    "impulsive_transient_early": lambda t: signal_impulsive_transient_variant(t, start=0.25, stop=0.27, amplitude=1.5),
    "impulsive_transient_late": lambda t: signal_impulsive_transient_variant(t, start=0.68, stop=0.70, amplitude=1.5),
    "impulsive_transient_weak_early": lambda t: signal_impulsive_transient_variant(t, start=0.25, stop=0.27, amplitude=0.75),
    "impulsive_transient_strong_late": lambda t: signal_impulsive_transient_variant(t, start=0.68, stop=0.70, amplitude=2.5),
    "intermittent_one_interval": lambda t: signal_intermittent_windows(t, windows=((0.30, 0.55),)),
    "intermittent_two_intervals": lambda t: signal_intermittent_windows(t, windows=((0.20, 0.40), (0.60, 0.80))),
    "intermittent_irregular": lambda t: signal_intermittent_windows(t, windows=((0.12, 0.22), (0.37, 0.48), (0.72, 0.91))),
    "intermittent_sparse_duty": lambda t: signal_intermittent_windows(t, windows=((0.24, 0.34), (0.66, 0.76))),
    "intermittent_dense_duty": lambda t: signal_intermittent_windows(t, windows=((0.12, 0.42), (0.56, 0.90))),
    "intermittent_sparse_one_interval": lambda t: signal_intermittent_windows(t, windows=((0.38, 0.50),)),
    "intermittent_dense_irregular": lambda t: signal_intermittent_windows(t, windows=((0.10, 0.32), (0.40, 0.62), (0.70, 0.92))),
    "close_freq_10_12": lambda t: signal_close_frequencies(t, f1=10.0, f2=12.0),
    "close_freq_10_11": lambda t: signal_close_frequencies(t, f1=10.0, f2=11.0),
    "close_freq_10_10p5": lambda t: signal_close_frequencies(t, f1=10.0, f2=10.5),
    "close_freq_second_weak": lambda t: signal_close_frequencies(t, f1=10.0, f2=11.0, amp1=1.0, amp2=0.5),
    "close_freq_second_moderate": lambda t: signal_close_frequencies(t, f1=10.0, f2=11.0, amp1=1.0, amp2=0.8),
    "close_freq_wide_second_moderate": lambda t: signal_close_frequencies(t, f1=10.0, f2=12.0, amp1=1.0, amp2=0.8),
    "close_freq_veryclose_second_weak": lambda t: signal_close_frequencies(t, f1=10.0, f2=10.5, amp1=1.0, amp2=0.5),
    # Challenging signal suite: not used for Section 5 parameter selection or
    # the main 168-case benchmark.  These stress local structure, frequency
    # crossing, weak-component recovery, trends, damping, and harmonic content.
    "crossing_chirps": signal_crossing_chirps,
    "crossing_chirps_shallow": lambda t: np.sum(
        signal_crossing_chirps_variant_components(t, up_slope=16.0, down_slope=12.0),
        axis=0,
    ).astype(DTYPE),
    "crossing_chirps_steep": lambda t: np.sum(
        signal_crossing_chirps_variant_components(t, up_slope=32.0, down_slope=24.0),
        axis=0,
    ).astype(DTYPE),
    "time_varying_close_frequencies": signal_time_varying_close_frequencies,
    "time_varying_close_frequencies_moderate_gap": lambda t: np.sum(
        signal_time_varying_close_frequencies_variant_components(t, min_separation=0.70),
        axis=0,
    ).astype(DTYPE),
    "time_varying_close_frequencies_severe_gap": lambda t: np.sum(
        signal_time_varying_close_frequencies_variant_components(t, min_separation=0.20),
        axis=0,
    ).astype(DTYPE),
    "piecewise_am_fm_discontinuity": signal_piecewise_am_fm_discontinuity,
    "piecewise_am_fm_discontinuity_mild": lambda t: np.sum(
        signal_piecewise_am_fm_discontinuity_variant_components(
            t, right_amp_base=0.80, right_amp_depth=0.30, right_freq_base=10.0, right_freq_depth=1.8,
        ),
        axis=0,
    ).astype(DTYPE),
    "piecewise_am_fm_discontinuity_severe": lambda t: np.sum(
        signal_piecewise_am_fm_discontinuity_variant_components(
            t, right_amp_base=0.45, right_amp_depth=0.55, right_freq_base=16.0, right_freq_depth=3.6,
        ),
        axis=0,
    ).astype(DTYPE),
    "damped_oscillation": signal_damped_oscillation,
    "damped_oscillation_slow_decay": lambda t: np.sum(
        signal_damped_oscillation_variant_components(t, fast_decay=1.4, slow_decay=0.35),
        axis=0,
    ).astype(DTYPE),
    "damped_oscillation_fast_decay": lambda t: np.sum(
        signal_damped_oscillation_variant_components(t, fast_decay=4.2, slow_decay=1.1),
        axis=0,
    ).astype(DTYPE),
    "trend_plus_oscillation": signal_trend_plus_oscillation,
    "trend_plus_oscillation_mild_trend": lambda t: np.sum(
        signal_trend_plus_oscillation_variant_components(t, trend_curvature=0.7, trend_slope=-0.125),
        axis=0,
    ).astype(DTYPE),
    "trend_plus_oscillation_strong_trend": lambda t: np.sum(
        signal_trend_plus_oscillation_variant_components(t, trend_curvature=2.4, trend_slope=-0.45),
        axis=0,
    ).astype(DTYPE),
    "buried_weak_component": signal_buried_weak_component,
    "buried_weak_component_mild": lambda t: np.sum(
        signal_buried_weak_component_variant_components(t, weak_amplitude=0.20),
        axis=0,
    ).astype(DTYPE),
    "buried_weak_component_severe": lambda t: np.sum(
        signal_buried_weak_component_variant_components(t, weak_amplitude=0.06),
        axis=0,
    ).astype(DTYPE),
    "non_sinusoidal_periodic": signal_non_sinusoidal_periodic,
    "non_sinusoidal_periodic_mild_harmonics": lambda t: np.sum(
        signal_non_sinusoidal_periodic_variant_components(t, harmonic_scale=0.55),
        axis=0,
    ).astype(DTYPE),
    "non_sinusoidal_periodic_strong_harmonics": lambda t: np.sum(
        signal_non_sinusoidal_periodic_variant_components(t, harmonic_scale=1.55),
        axis=0,
    ).astype(DTYPE),
    "transient_train": signal_transient_train,
    "transient_train_sparse": lambda t: np.sum(
        signal_transient_train_variant_components(
            t,
            events=((0.20, 0.020, 0.95, 55.0), (0.55, 0.024, 1.05, 62.0), (0.84, 0.018, 0.85, 80.0)),
        ),
        axis=0,
    ).astype(DTYPE),
    "transient_train_dense": lambda t: np.sum(
        signal_transient_train_variant_components(
            t,
            events=((0.18, 0.020, 1.0, 55.0), (0.30, 0.015, 0.85, 72.0), (0.42, 0.018, 0.95, 66.0), (0.58, 0.026, 1.10, 48.0), (0.70, 0.015, 0.80, 88.0), (0.84, 0.018, 0.90, 85.0)),
        ),
        axis=0,
    ).astype(DTYPE),
}


SIGNAL_VARIANT_METADATA = {
    "stationary_multi_sine_dev": {"class": "stationary_multi_sine", "freqs": (1.5, 4.0, 10.0), "amps": (1.0, 0.55, 0.30), "role": "development"},
    "chirp_dev": {"class": "chirp", "f0": 4.0, "k": 18.0, "role": "development"},
    "am_fm_dev": {"class": "am_fm", "amp_depth": 0.35, "phase_depth": 1.35, "role": "development"},
    "impulsive_transient_dev": {
        "class": "impulsive_transient",
        "start": 0.34,
        "stop": 0.365,
        "amplitude": 1.3,
        "freq": 70.0,
        "base_freqs": (1.5, 4.5, 8.5),
        "base_amps": (1.0, 0.45, 0.30),
        "role": "development",
    },
    "close_frequencies_dev": {"class": "close_frequencies", "f1": 9.0, "f2": 11.5, "role": "development"},
    "stationary_multi_sine_wide": {"class": "stationary_multi_sine", "freqs": (1.0, 4.0, 12.0), "amps": (1.2, 0.5, 0.25), "variant_role": "existing_canonical_variant", "difficulty_axis": "frequency_spacing"},
    "stationary_multi_sine_medium": {"class": "stationary_multi_sine", "freqs": (1.0, 3.0, 9.0), "amps": (1.2, 0.5, 0.25), "variant_role": "existing_canonical_variant", "difficulty_axis": "frequency_spacing"},
    "stationary_multi_sine_close": {"class": "stationary_multi_sine", "freqs": (6.0, 8.0, 10.0), "amps": (1.0, 0.7, 0.45), "variant_role": "existing_canonical_variant", "difficulty_axis": "frequency_spacing"},
    "stationary_multi_sine_balanced_amplitudes": {"class": "stationary_multi_sine", "freqs": (1.0, 3.0, 9.0), "amps": (0.8, 0.8, 0.8), "variant_role": "canonical_structured_perturbation", "difficulty_axis": "amplitude_imbalance"},
    "stationary_multi_sine_high_imbalance": {"class": "stationary_multi_sine", "freqs": (1.0, 3.0, 9.0), "amps": (1.4, 0.35, 0.12), "variant_role": "canonical_structured_perturbation", "difficulty_axis": "amplitude_imbalance"},
    "stationary_multi_sine_wide_balanced": {"class": "stationary_multi_sine", "freqs": (1.0, 4.0, 12.0), "amps": (0.8, 0.8, 0.8), "variant_role": "canonical_interaction_corner", "difficulty_axis": "frequency_spacing_x_amplitude_imbalance"},
    "stationary_multi_sine_close_high_imbalance": {"class": "stationary_multi_sine", "freqs": (6.0, 8.0, 10.0), "amps": (1.4, 0.35, 0.12), "variant_role": "canonical_interaction_corner", "difficulty_axis": "frequency_spacing_x_amplitude_imbalance"},
    "chirp_slow": {"class": "chirp", "f0": 3.0, "k": 10.0, "variant_role": "existing_canonical_variant", "difficulty_axis": "chirp_rate"},
    "chirp_fast": {"class": "chirp", "f0": 3.0, "k": 35.0, "variant_role": "existing_canonical_variant", "difficulty_axis": "chirp_rate"},
    "chirp_narrow_band": {"class": "chirp", "f0": 6.0, "k": 12.0, "variant_role": "canonical_structured_perturbation", "difficulty_axis": "sweep_bandwidth"},
    "chirp_broad_band": {"class": "chirp", "f0": 1.5, "k": 30.0, "variant_role": "canonical_structured_perturbation", "difficulty_axis": "sweep_bandwidth"},
    "chirp_slow_narrow_band": {"class": "chirp", "f0": 6.0, "k": 10.0, "variant_role": "canonical_interaction_corner", "difficulty_axis": "chirp_rate_x_sweep_bandwidth"},
    "chirp_fast_broad_band": {"class": "chirp", "f0": 1.5, "k": 35.0, "variant_role": "canonical_interaction_corner", "difficulty_axis": "chirp_rate_x_sweep_bandwidth"},
    "am_fm_weak": {"class": "am_fm", "amp_depth": 0.20, "phase_depth": 0.80, "variant_role": "canonical_interaction_corner", "difficulty_axis": "am_depth_x_fm_depth"},
    "am_fm_strong": {"class": "am_fm", "amp_depth": 0.70, "phase_depth": 3.00, "variant_role": "canonical_interaction_corner", "difficulty_axis": "am_depth_x_fm_depth"},
    "am_fm_am_weak": {"class": "am_fm", "amp_depth": 0.20, "phase_depth": 2.00, "variant_role": "canonical_structured_perturbation", "difficulty_axis": "am_depth"},
    "am_fm_am_strong": {"class": "am_fm", "amp_depth": 0.70, "phase_depth": 2.00, "variant_role": "canonical_structured_perturbation", "difficulty_axis": "am_depth"},
    "am_fm_fm_weak": {"class": "am_fm", "amp_depth": 0.45, "phase_depth": 0.80, "variant_role": "canonical_structured_perturbation", "difficulty_axis": "fm_depth"},
    "am_fm_fm_strong": {"class": "am_fm", "amp_depth": 0.45, "phase_depth": 3.00, "variant_role": "canonical_structured_perturbation", "difficulty_axis": "fm_depth"},
    "frequency_jump_up": {"class": "frequency_jump", "f_left": 5.0, "f_right": 15.0, "jump_time": 0.5, "variant_role": "existing_canonical_variant", "difficulty_axis": "jump_direction"},
    "frequency_jump_down": {"class": "frequency_jump", "f_left": 15.0, "f_right": 5.0, "jump_time": 0.5, "variant_role": "existing_canonical_variant", "difficulty_axis": "jump_direction"},
    "frequency_jump_multiple": {"class": "frequency_jump", "multi": True, "variant_role": "existing_canonical_variant", "difficulty_axis": "jump_complexity"},
    "frequency_jump_small_magnitude": {"class": "frequency_jump", "f_left": 8.0, "f_right": 12.0, "jump_time": 0.5, "variant_role": "canonical_structured_perturbation", "difficulty_axis": "jump_magnitude"},
    "frequency_jump_large_magnitude": {"class": "frequency_jump", "f_left": 3.0, "f_right": 21.0, "jump_time": 0.5, "variant_role": "canonical_structured_perturbation", "difficulty_axis": "jump_magnitude"},
    "frequency_jump_early": {"class": "frequency_jump", "f_left": 5.0, "f_right": 15.0, "jump_time": 0.35, "variant_role": "canonical_structured_perturbation", "difficulty_axis": "jump_timing"},
    "frequency_jump_late": {"class": "frequency_jump", "f_left": 5.0, "f_right": 15.0, "jump_time": 0.65, "variant_role": "canonical_structured_perturbation", "difficulty_axis": "jump_timing"},
    "frequency_jump_small_early": {"class": "frequency_jump", "f_left": 8.0, "f_right": 12.0, "jump_time": 0.35, "variant_role": "canonical_interaction_corner", "difficulty_axis": "jump_magnitude_x_jump_timing"},
    "frequency_jump_large_late": {"class": "frequency_jump", "f_left": 3.0, "f_right": 21.0, "jump_time": 0.65, "variant_role": "canonical_interaction_corner", "difficulty_axis": "jump_magnitude_x_jump_timing"},
    "impulsive_transient_short": {"class": "impulsive_transient", "start": 0.45, "stop": 0.465, "amplitude": 1.5, "freq": 80.0, "variant_role": "existing_canonical_variant", "difficulty_axis": "transient_width"},
    "impulsive_transient_long": {"class": "impulsive_transient", "start": 0.42, "stop": 0.50, "amplitude": 1.5, "freq": 80.0, "variant_role": "existing_canonical_variant", "difficulty_axis": "transient_width"},
    "impulsive_transient_strong": {"class": "impulsive_transient", "start": 0.45, "stop": 0.47, "amplitude": 2.5, "freq": 80.0, "variant_role": "existing_canonical_variant", "difficulty_axis": "transient_amplitude"},
    "impulsive_transient_weak": {"class": "impulsive_transient", "start": 0.45, "stop": 0.47, "amplitude": 0.75, "freq": 80.0, "variant_role": "canonical_structured_perturbation", "difficulty_axis": "transient_amplitude"},
    "impulsive_transient_early": {"class": "impulsive_transient", "start": 0.25, "stop": 0.27, "amplitude": 1.5, "freq": 80.0, "variant_role": "canonical_structured_perturbation", "difficulty_axis": "transient_location"},
    "impulsive_transient_late": {"class": "impulsive_transient", "start": 0.68, "stop": 0.70, "amplitude": 1.5, "freq": 80.0, "variant_role": "canonical_structured_perturbation", "difficulty_axis": "transient_location"},
    "impulsive_transient_weak_early": {"class": "impulsive_transient", "start": 0.25, "stop": 0.27, "amplitude": 0.75, "freq": 80.0, "variant_role": "canonical_interaction_corner", "difficulty_axis": "transient_amplitude_x_transient_location"},
    "impulsive_transient_strong_late": {"class": "impulsive_transient", "start": 0.68, "stop": 0.70, "amplitude": 2.5, "freq": 80.0, "variant_role": "canonical_interaction_corner", "difficulty_axis": "transient_amplitude_x_transient_location"},
    "intermittent_one_interval": {"class": "intermittent_oscillation", "windows": ((0.30, 0.55),), "f": 10.0, "variant_role": "existing_canonical_variant", "difficulty_axis": "interval_count"},
    "intermittent_two_intervals": {"class": "intermittent_oscillation", "windows": ((0.20, 0.40), (0.60, 0.80)), "f": 10.0, "variant_role": "existing_canonical_variant", "difficulty_axis": "interval_count"},
    "intermittent_irregular": {"class": "intermittent_oscillation", "windows": ((0.12, 0.22), (0.37, 0.48), (0.72, 0.91)), "f": 10.0, "variant_role": "existing_canonical_variant", "difficulty_axis": "interval_regularity"},
    "intermittent_sparse_duty": {"class": "intermittent_oscillation", "windows": ((0.24, 0.34), (0.66, 0.76)), "f": 10.0, "variant_role": "canonical_structured_perturbation", "difficulty_axis": "duty_cycle"},
    "intermittent_dense_duty": {"class": "intermittent_oscillation", "windows": ((0.12, 0.42), (0.56, 0.90)), "f": 10.0, "variant_role": "canonical_structured_perturbation", "difficulty_axis": "duty_cycle"},
    "intermittent_sparse_one_interval": {"class": "intermittent_oscillation", "windows": ((0.38, 0.50),), "f": 10.0, "variant_role": "canonical_interaction_corner", "difficulty_axis": "duty_cycle_x_interval_pattern"},
    "intermittent_dense_irregular": {"class": "intermittent_oscillation", "windows": ((0.10, 0.32), (0.40, 0.62), (0.70, 0.92)), "f": 10.0, "variant_role": "canonical_interaction_corner", "difficulty_axis": "duty_cycle_x_interval_pattern"},
    "close_freq_10_12": {"class": "close_frequencies", "f1": 10.0, "f2": 12.0, "variant_role": "existing_canonical_variant", "difficulty_axis": "frequency_separation"},
    "close_freq_10_11": {"class": "close_frequencies", "f1": 10.0, "f2": 11.0, "variant_role": "existing_canonical_variant", "difficulty_axis": "frequency_separation"},
    "close_freq_10_10p5": {"class": "close_frequencies", "f1": 10.0, "f2": 10.5, "variant_role": "existing_canonical_variant", "difficulty_axis": "frequency_separation"},
    "close_freq_second_weak": {"class": "close_frequencies", "f1": 10.0, "f2": 11.0, "amp1": 1.0, "amp2": 0.5, "variant_role": "canonical_structured_perturbation", "difficulty_axis": "amplitude_ratio"},
    "close_freq_second_moderate": {"class": "close_frequencies", "f1": 10.0, "f2": 11.0, "amp1": 1.0, "amp2": 0.8, "variant_role": "canonical_structured_perturbation", "difficulty_axis": "amplitude_ratio"},
    "close_freq_wide_second_moderate": {"class": "close_frequencies", "f1": 10.0, "f2": 12.0, "amp1": 1.0, "amp2": 0.8, "variant_role": "canonical_interaction_corner", "difficulty_axis": "frequency_separation_x_amplitude_ratio"},
    "close_freq_veryclose_second_weak": {"class": "close_frequencies", "f1": 10.0, "f2": 10.5, "amp1": 1.0, "amp2": 0.5, "variant_role": "canonical_interaction_corner", "difficulty_axis": "frequency_separation_x_amplitude_ratio"},
    "crossing_chirps": {"class": "challenging", "regime": "crossing_chirps"},
    "crossing_chirps_shallow": {"class": "challenging", "regime": "crossing_chirps", "up_slope": 16.0, "down_slope": 12.0, "variant_role": "challenging_family_variant", "difficulty_axis": "crossing_angle_chirp_rate"},
    "crossing_chirps_steep": {"class": "challenging", "regime": "crossing_chirps", "up_slope": 32.0, "down_slope": 24.0, "variant_role": "challenging_family_variant", "difficulty_axis": "crossing_angle_chirp_rate"},
    "time_varying_close_frequencies": {"class": "challenging", "regime": "time_varying_close_frequencies"},
    "time_varying_close_frequencies_moderate_gap": {"class": "challenging", "regime": "time_varying_close_frequencies", "min_separation": 0.70, "variant_role": "challenging_family_variant", "difficulty_axis": "minimum_frequency_separation"},
    "time_varying_close_frequencies_severe_gap": {"class": "challenging", "regime": "time_varying_close_frequencies", "min_separation": 0.20, "variant_role": "challenging_family_variant", "difficulty_axis": "minimum_frequency_separation"},
    "piecewise_am_fm_discontinuity": {"class": "challenging", "regime": "piecewise_am_fm_discontinuity"},
    "piecewise_am_fm_discontinuity_mild": {"class": "challenging", "regime": "piecewise_am_fm_discontinuity", "right_amp_base": 0.80, "right_amp_depth": 0.30, "right_freq_base": 10.0, "right_freq_depth": 1.8, "variant_role": "challenging_family_variant", "difficulty_axis": "discontinuity_magnitude"},
    "piecewise_am_fm_discontinuity_severe": {"class": "challenging", "regime": "piecewise_am_fm_discontinuity", "right_amp_base": 0.45, "right_amp_depth": 0.55, "right_freq_base": 16.0, "right_freq_depth": 3.6, "variant_role": "challenging_family_variant", "difficulty_axis": "discontinuity_magnitude"},
    "damped_oscillation": {"class": "challenging", "regime": "damped_oscillation"},
    "damped_oscillation_slow_decay": {"class": "challenging", "regime": "damped_oscillation", "fast_decay": 1.4, "slow_decay": 0.35, "variant_role": "challenging_family_variant", "difficulty_axis": "damping_rate"},
    "damped_oscillation_fast_decay": {"class": "challenging", "regime": "damped_oscillation", "fast_decay": 4.2, "slow_decay": 1.1, "variant_role": "challenging_family_variant", "difficulty_axis": "damping_rate"},
    "trend_plus_oscillation": {"class": "challenging", "regime": "trend_plus_oscillation"},
    "trend_plus_oscillation_mild_trend": {"class": "challenging", "regime": "trend_plus_oscillation", "trend_curvature": 0.7, "trend_slope": -0.125, "variant_role": "challenging_family_variant", "difficulty_axis": "trend_strength"},
    "trend_plus_oscillation_strong_trend": {"class": "challenging", "regime": "trend_plus_oscillation", "trend_curvature": 2.4, "trend_slope": -0.45, "variant_role": "challenging_family_variant", "difficulty_axis": "trend_strength"},
    "buried_weak_component": {"class": "challenging", "regime": "buried_weak_component"},
    "buried_weak_component_mild": {"class": "challenging", "regime": "buried_weak_component", "weak_amplitude": 0.20, "variant_role": "challenging_family_variant", "difficulty_axis": "weak_component_amplitude"},
    "buried_weak_component_severe": {"class": "challenging", "regime": "buried_weak_component", "weak_amplitude": 0.06, "variant_role": "challenging_family_variant", "difficulty_axis": "weak_component_amplitude"},
    "non_sinusoidal_periodic": {"class": "challenging", "regime": "non_sinusoidal_periodic"},
    "non_sinusoidal_periodic_mild_harmonics": {"class": "challenging", "regime": "non_sinusoidal_periodic", "harmonic_scale": 0.55, "variant_role": "challenging_family_variant", "difficulty_axis": "harmonic_richness"},
    "non_sinusoidal_periodic_strong_harmonics": {"class": "challenging", "regime": "non_sinusoidal_periodic", "harmonic_scale": 1.55, "variant_role": "challenging_family_variant", "difficulty_axis": "harmonic_richness"},
    "transient_train": {"class": "challenging", "regime": "transient_train"},
    "transient_train_sparse": {"class": "challenging", "regime": "transient_train", "events": ((0.20, 0.020, 0.95, 55.0), (0.55, 0.024, 1.05, 62.0), (0.84, 0.018, 0.85, 80.0)), "variant_role": "challenging_family_variant", "difficulty_axis": "transient_spacing_density"},
    "transient_train_dense": {"class": "challenging", "regime": "transient_train", "events": ((0.18, 0.020, 1.0, 55.0), (0.30, 0.015, 0.85, 72.0), (0.42, 0.018, 0.95, 66.0), (0.58, 0.026, 1.10, 48.0), (0.70, 0.015, 0.80, 88.0), (0.84, 0.018, 0.90, 85.0)), "variant_role": "challenging_family_variant", "difficulty_axis": "transient_spacing_density"},
}


# ============================================================
# Public API
# ============================================================

def get_signal(name: str, t: np.ndarray) -> np.ndarray:
    name = str(name)
    if name not in SIGNAL_REGISTRY:
        raise ValueError(
            f"Unknown signal: {name}. Available signals: {list(SIGNAL_REGISTRY.keys())}"
        )
    return SIGNAL_REGISTRY[name](t).astype(DTYPE)


def list_synthetic_signals() -> List[str]:
    return list(SIGNAL_REGISTRY.keys())


# ============================================================
# True/reference instantaneous frequencies
# ============================================================

def get_true_frequencies(signal_name: str, t: np.ndarray):
    """
    Return true/reference instantaneous frequency curves.

    Format:
        [
            {"label": "...", "freq": scalar_or_array},
            ...
        ]

    These curves are used only for Hilbert-ridge visualization.
    """
    signal_name = str(signal_name)
    t = np.asarray(t, dtype=DTYPE)

    meta = SIGNAL_VARIANT_METADATA.get(signal_name)
    if meta is not None:
        cls = meta["class"]
        if cls == "stationary_multi_sine":
            return [
                {"label": f"true {freq:g} Hz", "freq": np.ones_like(t) * float(freq)}
                for freq in meta["freqs"]
            ]
        if cls == "chirp":
            return [
                {"label": "true chirp IF", "freq": float(meta["f0"]) + float(meta["k"]) * t},
            ]
        if cls == "am_fm":
            phase_depth = float(meta["phase_depth"])
            return [
                {"label": "true AM-FM IF", "freq": 8.0 + phase_depth * 1.6 * np.pi * np.cos(2 * np.pi * 0.8 * t)},
            ]
        if cls == "frequency_jump" and meta.get("multi"):
            freq = np.where(t < 0.33, 5.0, np.where(t < 0.66, 15.0, 8.0))
            return [{"label": "true multi-jump IF", "freq": freq}]
        if cls == "frequency_jump":
            return [
                {
                    "label": "true jump IF",
                    "freq": np.where(t < float(meta["jump_time"]), float(meta["f_left"]), float(meta["f_right"])),
                },
            ]
        if cls == "impulsive_transient":
            active = (t > float(meta["start"])) & (t < float(meta["stop"]))
            base_freqs = meta.get("base_freqs", (1.0, 3.0, 9.0))
            return [
                *[
                    {"label": f"true {float(freq):g} Hz", "freq": np.ones_like(t) * float(freq)}
                    for freq in base_freqs
                ],
                {"label": f"transient {float(meta['freq']):g} Hz", "freq": np.where(active, float(meta["freq"]), np.nan)},
            ]
        if cls == "intermittent_oscillation":
            active = np.zeros_like(t, dtype=bool)
            for start, stop in meta["windows"]:
                active |= (t > float(start)) & (t < float(stop))
            return [
                {"label": f"intermittent {float(meta['f']):g} Hz", "freq": np.where(active, float(meta["f"]), np.nan)},
            ]
        if cls == "close_frequencies":
            return [
                {"label": f"true {float(meta['f1']):g} Hz", "freq": np.ones_like(t) * float(meta["f1"])},
                {"label": f"true {float(meta['f2']):g} Hz", "freq": np.ones_like(t) * float(meta["f2"])},
            ]
        if cls == "challenging":
            regime = meta["regime"]
            if regime == "crossing_chirps":
                up_slope = float(meta.get("up_slope", 24.0))
                down_slope = float(meta.get("down_slope", 18.0))
                upper_start = float(meta.get("upper_start", 5.0))
                lower_start = float(meta.get("lower_start", 29.0))
                return [
                    {"label": "upward chirp", "freq": upper_start + up_slope * t},
                    {"label": "downward chirp", "freq": lower_start - down_slope * t},
                ]
            if regime == "time_varying_close_frequencies":
                center = 11.0 + float(meta.get("center_modulation", 1.5)) * np.sin(2 * np.pi * 0.45 * t)
                separation = float(meta.get("min_separation", 0.35)) + float(meta.get("curvature", 2.2)) * (t - 0.5) ** 2
                return [
                    {"label": "close IF lower", "freq": center - 0.5 * separation},
                    {"label": "close IF upper", "freq": center + 0.5 * separation},
                ]
            if regime == "piecewise_am_fm_discontinuity":
                freq = np.where(
                    t < float(meta.get("split_time", 0.52)),
                    6.0 + 2.0 * np.sin(2 * np.pi * 0.8 * t),
                    float(meta.get("right_freq_base", 13.0))
                    + float(meta.get("right_freq_depth", 2.8)) * np.cos(2 * np.pi * 0.7 * t),
                )
                return [
                    {"label": "low 1.2 Hz", "freq": np.ones_like(t) * 1.2},
                    {"label": "piecewise AM-FM IF", "freq": freq},
                ]
            if regime == "damped_oscillation":
                return [
                    {"label": "damped 4 Hz", "freq": np.ones_like(t) * 4.0},
                    {"label": "damped 14 Hz", "freq": np.ones_like(t) * 14.0},
                ]
            if regime == "trend_plus_oscillation":
                return [
                    {"label": "trend", "freq": np.full_like(t, np.nan)},
                    {"label": "2 Hz", "freq": np.ones_like(t) * 2.0},
                    {"label": "18 Hz", "freq": np.ones_like(t) * 18.0},
                ]
            if regime == "buried_weak_component":
                return [
                    {"label": "3 Hz strong", "freq": np.ones_like(t) * 3.0},
                    {"label": "7 Hz modulation", "freq": np.ones_like(t) * 7.0},
                    {"label": "31 Hz weak", "freq": np.ones_like(t) * 31.0},
                ]
            if regime == "non_sinusoidal_periodic":
                return [
                    {"label": "5 Hz fundamental", "freq": np.ones_like(t) * 5.0},
                    {"label": "10 Hz harmonic", "freq": np.ones_like(t) * 10.0},
                    {"label": "15 Hz harmonic", "freq": np.ones_like(t) * 15.0},
                    {"label": "25 Hz harmonic", "freq": np.ones_like(t) * 25.0},
                ]
            if regime == "transient_train":
                freq = np.full_like(t, np.nan)
                events = meta.get("events", (
                    (0.18, 0.020, 1.0, 55.0),
                    (0.37, 0.015, 0.75, 72.0),
                    (0.63, 0.030, 1.15, 48.0),
                    (0.84, 0.018, 0.90, 85.0),
                ))
                for center, width, _, f in events:
                    freq[np.abs(t - center) <= 2.5 * width] = f
                return [
                    {"label": "base 2.5 Hz", "freq": np.ones_like(t) * 2.5},
                    {"label": "transient train IF", "freq": freq},
                ]

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
        # phase/(2π) = 8t + 2 sin(2π0.8t)
        # IF = derivative = 8 + 2*(2π*0.8)*cos(2π0.8t)
        return [
            {"label": "true AM-FM IF", "freq": 8.0 + 3.2 * np.pi * np.cos(2 * np.pi * 0.8 * t)},
        ]

    if signal_name == "frequency_jump":
        return [
            {"label": "true jump IF", "freq": np.where(t < 0.5, 5.0, 15.0)},
        ]

    if signal_name == "impulsive_transient":
        return [
            {"label": "true 1 Hz", "freq": np.ones_like(t) * 1.0},
            {"label": "true 3 Hz", "freq": np.ones_like(t) * 3.0},
            {"label": "true 9 Hz", "freq": np.ones_like(t) * 9.0},
            {"label": "transient 80 Hz", "freq": np.where((t > 0.45) & (t < 0.47), 80.0, np.nan)},
        ]

    if signal_name == "intermittent_oscillation":
        active = ((t > 0.20) & (t < 0.40)) | ((t > 0.60) & (t < 0.80))
        return [
            {"label": "intermittent 10 Hz", "freq": np.where(active, 10.0, np.nan)},
        ]

    if signal_name == "close_frequencies":
        return [
            {"label": "true 10 Hz", "freq": np.ones_like(t) * 10.0},
            {"label": "true 11 Hz", "freq": np.ones_like(t) * 11.0},
        ]

    return []


# ============================================================
# True/reference signal components
# ============================================================

def get_true_components(signal_name: str, t: np.ndarray) -> Optional[np.ndarray]:
    """
    Return true component modes when available.

    Shape:
        (n_components, n_samples)

    Used for optional IMF recovery diagnostics.
    """
    signal_name = str(signal_name)
    t = np.asarray(t, dtype=DTYPE)

    meta = SIGNAL_VARIANT_METADATA.get(signal_name)
    if meta is not None:
        cls = meta["class"]
        if cls == "stationary_multi_sine":
            return np.vstack([
                float(amp) * np.sin(2 * np.pi * float(freq) * t)
                for amp, freq in zip(meta["amps"], meta["freqs"])
            ]).astype(DTYPE)
        if cls == "chirp":
            return np.vstack([
                signal_chirp(t, f0=float(meta["f0"]), k=float(meta["k"])),
            ]).astype(DTYPE)
        if cls == "am_fm":
            return np.vstack([
                signal_am_fm_variant(
                    t,
                    amp_depth=float(meta["amp_depth"]),
                    phase_depth=float(meta["phase_depth"]),
                ),
            ]).astype(DTYPE)
        if cls == "frequency_jump" and meta.get("multi"):
            return np.vstack([signal_frequency_jump_multiple(t)]).astype(DTYPE)
        if cls == "frequency_jump":
            return np.vstack([
                signal_frequency_jump(
                    t,
                    f_left=float(meta["f_left"]),
                    f_right=float(meta["f_right"]),
                    jump_time=float(meta["jump_time"]),
                ),
            ]).astype(DTYPE)
        if cls == "impulsive_transient":
            transient = np.zeros_like(t, dtype=DTYPE)
            idx = (t > float(meta["start"])) & (t < float(meta["stop"]))
            transient[idx] = float(meta["amplitude"]) * np.sin(2 * np.pi * float(meta["freq"]) * t[idx])
            base_freqs = meta.get("base_freqs", (1.0, 3.0, 9.0))
            base_amps = meta.get("base_amps", (1.2, 0.5, 0.25))
            return np.vstack([
                *[
                    float(amp) * np.sin(2 * np.pi * float(freq) * t)
                    for amp, freq in zip(base_amps, base_freqs)
                ],
                transient,
            ]).astype(DTYPE)
        if cls == "intermittent_oscillation":
            return np.vstack([
                signal_intermittent_windows(t, windows=meta["windows"], f=float(meta["f"])),
            ]).astype(DTYPE)
        if cls == "close_frequencies":
            return np.vstack([
                float(meta.get("amp1", 1.0)) * np.sin(2 * np.pi * float(meta["f1"]) * t),
                float(meta.get("amp2", 1.0)) * np.sin(2 * np.pi * float(meta["f2"]) * t),
            ]).astype(DTYPE)
        if cls == "challenging":
            regime = meta["regime"]
            if regime == "crossing_chirps":
                return signal_crossing_chirps_variant_components(
                    t,
                    up_slope=float(meta.get("up_slope", 24.0)),
                    down_slope=float(meta.get("down_slope", 18.0)),
                    upper_start=float(meta.get("upper_start", 5.0)),
                    lower_start=float(meta.get("lower_start", 29.0)),
                    second_amplitude=float(meta.get("second_amplitude", 0.85)),
                )
            if regime == "time_varying_close_frequencies":
                return signal_time_varying_close_frequencies_variant_components(
                    t,
                    min_separation=float(meta.get("min_separation", 0.35)),
                    curvature=float(meta.get("curvature", 2.2)),
                    center_modulation=float(meta.get("center_modulation", 1.5)),
                    second_amplitude=float(meta.get("second_amplitude", 0.9)),
                )
            if regime == "piecewise_am_fm_discontinuity":
                return signal_piecewise_am_fm_discontinuity_variant_components(
                    t,
                    split_time=float(meta.get("split_time", 0.52)),
                    right_amp_base=float(meta.get("right_amp_base", 0.65)),
                    right_amp_depth=float(meta.get("right_amp_depth", 0.45)),
                    right_freq_base=float(meta.get("right_freq_base", 13.0)),
                    right_freq_depth=float(meta.get("right_freq_depth", 2.8)),
                )
            if regime == "damped_oscillation":
                return signal_damped_oscillation_variant_components(
                    t,
                    fast_decay=float(meta.get("fast_decay", 2.6)),
                    slow_decay=float(meta.get("slow_decay", 0.7)),
                )
            if regime == "trend_plus_oscillation":
                return signal_trend_plus_oscillation_variant_components(
                    t,
                    trend_curvature=float(meta.get("trend_curvature", 1.4)),
                    trend_slope=float(meta.get("trend_slope", -0.25)),
                )
            if regime == "buried_weak_component":
                return signal_buried_weak_component_variant_components(
                    t,
                    weak_amplitude=float(meta.get("weak_amplitude", 0.12)),
                    modulation_amplitude=float(meta.get("modulation_amplitude", 0.18)),
                )
            if regime == "non_sinusoidal_periodic":
                return signal_non_sinusoidal_periodic_variant_components(
                    t,
                    harmonic_scale=float(meta.get("harmonic_scale", 1.0)),
                )
            if regime == "transient_train":
                return signal_transient_train_variant_components(
                    t,
                    events=meta.get("events"),
                )

    if signal_name == "stationary_multi_sine":
        return np.vstack([
            1.2 * np.sin(2 * np.pi * 1.0 * t),
            0.5 * np.sin(2 * np.pi * 3.0 * t),
            0.25 * np.sin(2 * np.pi * 9.0 * t),
        ]).astype(DTYPE)

    if signal_name == "chirp":
        return np.vstack([
            signal_chirp(t),
        ]).astype(DTYPE)

    if signal_name == "am_fm":
        return np.vstack([
            signal_am_fm(t),
        ]).astype(DTYPE)

    if signal_name == "frequency_jump":
        return np.vstack([
            signal_frequency_jump(t),
        ]).astype(DTYPE)

    if signal_name == "impulsive_transient":
        transient = np.zeros_like(t, dtype=DTYPE)
        idx = (t > 0.45) & (t < 0.47)
        transient[idx] = 1.5 * np.sin(2 * np.pi * 80.0 * t[idx])
        return np.vstack([
            1.2 * np.sin(2 * np.pi * 1.0 * t),
            0.5 * np.sin(2 * np.pi * 3.0 * t),
            0.25 * np.sin(2 * np.pi * 9.0 * t),
            transient,
        ]).astype(DTYPE)

    if signal_name == "intermittent_oscillation":
        return np.vstack([
            signal_intermit_oscillation(t),
        ]).astype(DTYPE)

    if signal_name == "close_frequencies":
        return np.vstack([
            np.sin(2 * np.pi * 10.0 * t),
            np.sin(2 * np.pi * 11.0 * t),
        ]).astype(DTYPE)

    return None


# ============================================================
# Helper for plotting true frequency overlays
# ============================================================

def infer_signal_name_from_path(path) -> Optional[str]:
    """
    Infer synthetic signal name from an output path.
    Used only for plot annotation when experiments do not explicitly pass signal_name.
    """
    parts = [str(p) for p in Path(path).parts]
    for name in SIGNAL_REGISTRY.keys():
        if name in parts:
            return name

    joined = str(path)
    for name in SIGNAL_REGISTRY.keys():
        if name in joined:
            return name

    return None
