"""experiments/zscore_artifact: which normalizations can leak, shown on synthetic trials.

The mechanism (experiments/zscore_artifact/MECHANISM.md): z-scoring a trial over its
epoch subtracts the epoch mean, which contains the evoked response, from every sample,
including the pre-stimulus ones.
"""
import numpy as np
import pytest

import run_erpcore as R

TIMES = 1000.0 * (-0.2 + np.arange(256) / 256.0)        # -200 .. 796 ms, as in ERP CORE
PRE = TIMES < 0


def trials(n=60, n_ch=8, seed=0, late_effect=0.0):
    """Baseline-corrected trials: noise, plus a sustained post-stimulus deflection whose
    size differs between two conditions by ``late_effect``."""
    rng = np.random.RandomState(seed)
    y = np.tile([0, 1], n // 2)
    x = rng.standard_normal((n, 256, n_ch))
    late = ((TIMES > 300) & (TIMES < 700)).astype(float)
    topo = rng.standard_normal(n_ch)
    x += (1.0 + late_effect * y)[:, None, None] * late[None, :, None] * topo[None, None, :]
    x -= x[:, PRE].mean(axis=1, keepdims=True)
    return x, y


@pytest.mark.parametrize("how,uses_post", [("none", False), ("baseline_z", False),
                                           ("epoch_z_full", True), ("epoch_z_post", True)])
def test_which_normalizations_let_the_future_change_the_past(how, uses_post):
    x, _ = trials()
    altered = x.copy()
    altered[:, ~PRE] += 5.0                                 # change only post-stimulus samples
    a, b = R.normalize(x, TIMES, how)[:, PRE], R.normalize(altered, TIMES, how)[:, PRE]
    assert (not np.allclose(a, b)) is uses_post


def test_epoch_z_post_is_the_shape_v1_convention():
    """v1 z-scores with post-stimulus statistics; on the post-stimulus part the two agree."""
    x, _ = trials()
    post = x[:, ~PRE]
    v1_style = (post - post.mean(axis=1, keepdims=True)) / post.std(axis=1, keepdims=True)
    assert np.allclose(R.normalize(x, TIMES, "epoch_z_post")[:, ~PRE], v1_style)


def _pre_stimulus_accuracy(x, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    feats = x[:, PRE].mean(axis=1)                          # one pre-stimulus scalp pattern per trial
    pred = cross_val_predict(LogisticRegression(max_iter=1000), feats, y, cv=5)
    return R.bal_acc(y, pred)


def test_late_difference_becomes_decodable_before_onset_only_after_z_scoring():
    x, y = trials(n=200, late_effect=1.5, seed=3)
    clean = _pre_stimulus_accuracy(R.normalize(x, TIMES, "baseline_z"), y)
    leaked = _pre_stimulus_accuracy(R.normalize(x, TIMES, "epoch_z_full"), y)
    assert abs(clean - 0.5) < 0.1
    assert leaked > 0.8


def test_no_condition_difference_means_no_leak():
    x, y = trials(n=200, late_effect=0.0, seed=4)
    assert abs(_pre_stimulus_accuracy(R.normalize(x, TIMES, "epoch_z_full"), y) - 0.5) < 0.12


def test_binning_and_balanced_accuracy():
    x, _ = trials(n=4)
    b = R.binned(x)
    assert b.shape == (4, 64, 8) and np.allclose(b[:, 0], x[:, :4].mean(axis=1))
    assert R.bal_acc(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])) == 0.75
