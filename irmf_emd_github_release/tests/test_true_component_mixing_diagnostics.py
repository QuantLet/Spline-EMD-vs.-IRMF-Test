#!/usr/bin/python
# coding: UTF-8

"""Scientific unit tests for true-component splitting/merging diagnostics."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.shared_physical_diagnostics import true_component_mixing_diagnostics
from diagnostics.shared_physical_diagnostics import imf_recovery_diagnostics


def _basis(n=2048):
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    c1 = np.sin(2.0 * np.pi * 7.0 * t)
    c2 = np.cos(2.0 * np.pi * 13.0 * t)
    c3 = np.sin(2.0 * np.pi * 23.0 * t + 0.3)
    return np.vstack([c1, c2, c3])


def _assert_near_zero(value, name, tol=1e-6):
    assert np.isfinite(value), f"{name} should be finite, got {value!r}"
    assert abs(float(value)) <= tol, f"{name} should be near zero, got {value!r}"


def test_perfect_one_to_one_mapping():
    true = _basis()[:2]
    out = true_component_mixing_diagnostics(true.copy(), true)
    _assert_near_zero(out["component_splitting_index"], "splitting")
    _assert_near_zero(out["component_merging_index"], "merging")
    assert out["missing_true_component_count"] == 0
    assert out["unmatched_estimated_component_count"] == 0
    assert out["true_component_metrics_available"] is True


def test_true_component_split_across_two_estimated_modes():
    true = _basis()[:2]
    estimated = np.vstack([0.6 * true[0], 0.4 * true[0], true[1]])
    out = true_component_mixing_diagnostics(estimated, true)
    assert out["component_splitting_index"] > 0.4
    _assert_near_zero(out["component_merging_index"], "merging")
    assert out["missing_true_component_count"] == 0


def test_two_true_components_merged_into_one_estimated_mode():
    true = _basis()[:2]
    estimated = np.vstack([true[0] + true[1]])
    out = true_component_mixing_diagnostics(estimated, true)
    _assert_near_zero(out["component_splitting_index"], "splitting")
    assert out["component_merging_index"] > 0.9
    assert out["missing_true_component_count"] == 0


def test_simultaneous_splitting_and_merging():
    true = _basis()[:3]
    estimated = np.vstack([
        0.6 * true[0] + 0.5 * true[1],
        0.4 * true[0],
        true[2],
    ])
    out = true_component_mixing_diagnostics(estimated, true)
    assert out["component_splitting_index"] > 0.2
    assert out["component_merging_index"] > 0.2


def test_unmatched_noise_mode_is_spurious_not_merging():
    true = _basis()[:2]
    rng = np.random.default_rng(0)
    noise = rng.normal(size=true.shape[1])
    noise = noise / (np.std(noise) + 1e-12)
    estimated = np.vstack([true[0], true[1], noise])
    out = true_component_mixing_diagnostics(estimated, true)
    _assert_near_zero(out["component_splitting_index"], "splitting")
    _assert_near_zero(out["component_merging_index"], "merging")
    assert out["unmatched_estimated_component_count"] == 1
    assert out["spurious_mode_energy_ratio"] > 0.0


def test_missing_true_component_is_not_counted_as_splitting():
    true = _basis()[:2]
    estimated = np.vstack([true[0]])
    out = true_component_mixing_diagnostics(estimated, true)
    _assert_near_zero(out["component_splitting_index"], "splitting")
    _assert_near_zero(out["component_merging_index"], "merging")
    assert out["missing_true_component_count"] == 1
    assert out["missing_true_component_fraction"] == 0.5


def test_low_energy_extra_mode_does_not_change_primary_indices():
    true = _basis()[:2]
    estimated = np.vstack([true[0], true[1], 1e-3 * true[0]])
    out = true_component_mixing_diagnostics(estimated, true)
    _assert_near_zero(out["component_splitting_index"], "splitting")
    _assert_near_zero(out["component_merging_index"], "merging")


def test_permutation_invariance_for_estimated_and_true_order():
    true = _basis()[:3]
    estimated = np.vstack([
        0.6 * true[0] + 0.5 * true[1],
        0.4 * true[0],
        true[2],
    ])
    base = true_component_mixing_diagnostics(estimated, true)
    permuted = true_component_mixing_diagnostics(estimated[[2, 0, 1]], true[[1, 2, 0]])
    for key in (
        "component_splitting_index",
        "component_merging_index",
        "missing_true_component_count",
        "unmatched_estimated_component_count",
        "spurious_mode_energy_ratio",
    ):
        assert np.allclose(base[key], permuted[key], equal_nan=True), key


def test_sign_invariance():
    true = _basis()[:2]
    estimated = np.vstack([0.6 * true[0], 0.4 * true[0], true[1]])
    base = true_component_mixing_diagnostics(estimated, true)
    flipped = true_component_mixing_diagnostics(
        np.vstack([-estimated[0], estimated[1], -estimated[2]]),
        np.vstack([-true[0], true[1]]),
    )
    for key in ("component_splitting_index", "component_merging_index"):
        assert np.allclose(base[key], flipped[key], equal_nan=True), key


def test_scale_behavior_allocation_vs_amplitude_recovery():
    true = _basis()[:1]
    for scale in (1.0, 0.5, 0.05):
        estimated = np.vstack([scale * true[0]])
        mixing = true_component_mixing_diagnostics(estimated, true)
        recovery = imf_recovery_diagnostics(estimated, true)
        _assert_near_zero(mixing["component_splitting_index"], "splitting")
        _assert_near_zero(mixing["component_merging_index"], "merging")
        assert mixing["missing_true_component_count"] == 0
        assert recovery["imf_recovery_corr"] > 0.99
        assert recovery["imf_recovery_nrmse"] > 0.0 if scale < 1.0 else recovery["imf_recovery_nrmse"] < 1e-9


def test_duplicate_mode_is_splitting_not_merging_or_spurious():
    true = _basis()[:2]
    estimated = np.vstack([true[0], true[0], true[1]])
    out = true_component_mixing_diagnostics(estimated, true)
    assert out["component_splitting_index"] > 0.4
    _assert_near_zero(out["component_merging_index"], "merging")
    assert out["unmatched_estimated_component_count"] == 0
    assert out["spurious_mode_energy_ratio"] == 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("true-component mixing diagnostic tests passed")
