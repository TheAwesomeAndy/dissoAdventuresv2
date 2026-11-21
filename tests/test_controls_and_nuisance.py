"""arspinet_v2.controls and arspinet_v2.nuisance."""

import numpy as np
import pytest

from arspinet_v2 import controls, nuisance
from arspinet_v2 import evaluation as ev


def _blocks(code):
    return code.reshape(len(code), code.shape[1], 6, -1)


def test_collapse_removes_timing_and_keeps_the_mean(code):
    b = _blocks(controls.collapse(code))
    assert np.allclose(b, b[:, :, :1])
    assert np.allclose(b[:, :, 0], _blocks(code).mean(axis=2), atol=1e-5)


def test_bin_shuffle_only_reorders_bins(code):
    shuffled = controls.shuffle_bins(code, np.random.RandomState(1))
    assert not np.array_equal(shuffled, code)
    assert np.allclose(
        np.sort(_blocks(shuffled).sum(axis=3), axis=2), np.sort(_blocks(code).sum(axis=3), axis=2)
    )


def test_fixed_bin_permutation_is_not_the_identity_and_changes_no_fit(code, cohort):
    _, y, g = cohort
    fixed = controls.fixed_bin_permutation(code, np.random.RandomState(2))
    assert not np.array_equal(fixed, code)
    a = ev.evaluate(fixed.reshape(len(y), -1), y, g, n_repeats=1)
    b = ev.evaluate(code.reshape(len(y), -1), y, g, n_repeats=1)
    assert (a["pred"] == b["pred"]).mean() >= 0.97


def test_electrode_shuffle_only_reorders_electrodes(code):
    shuffled = controls.shuffle_electrodes(code, np.random.RandomState(3))
    assert not np.array_equal(shuffled, code)
    assert np.allclose(np.sort(shuffled.sum(axis=2), axis=1), np.sort(code.sum(axis=2), axis=1))


def test_controls_do_not_modify_their_input(code):
    before = code.copy()
    controls.collapse(code)
    controls.shuffle_bins(code, np.random.RandomState(0))
    controls.shuffle_electrodes(code, np.random.RandomState(0))
    assert np.array_equal(code, before)


def test_noise_has_no_variance_components(cohort):
    _, y, g = cohort
    vc = nuisance.variance_components(
        np.random.RandomState(4).standard_normal((len(y), 300)), y, g, n_perm=200
    )
    assert abs(vc["fraction_subject"]) < 0.03 and abs(vc["fraction_condition"]) < 0.03
    assert vc["subject_p_value"] > 0.01 and vc["condition_p_value"] > 0.01
    assert vc["uncalibrated_v1_style_ratio"] > 5  # the ratio v1 reported has no noise floor


def test_planted_subject_effect_is_recovered(cohort):
    _, y, g = cohort
    rng = np.random.RandomState(6)
    X = rng.standard_normal((len(y), 300)) + rng.standard_normal((len(np.unique(g)), 300))[g]
    vc = nuisance.variance_components(X, y, g, n_perm=200)
    assert 0.4 < vc["fraction_subject"] < 0.6  # equal variances -> one half
    assert vc["subject_p_value"] == pytest.approx(1 / 201)
    assert vc["fraction_subject"] + vc["fraction_condition"] + vc["fraction_residual"] == pytest.approx(1.0)


def test_planted_condition_effect_is_recovered(cohort):
    _, y, g = cohort
    rng = np.random.RandomState(7)
    X = rng.standard_normal((len(y), 300)) + rng.standard_normal((3, 300))[y]
    vc = nuisance.variance_components(X, y, g, n_perm=200)
    assert vc["fraction_condition"] > 0.2 and vc["condition_p_value"] == pytest.approx(1 / 201)


def test_incomplete_panels_are_rejected(cohort):
    _, y, g = cohort
    with pytest.raises(ValueError):
        nuisance.variance_components(np.zeros((len(y) - 1, 5)), y[:-1], g[:-1])
