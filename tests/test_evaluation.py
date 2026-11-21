"""arspinet_v2.evaluation: splits, readout, bootstrap and permutation test."""
import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

from arspinet_v2 import evaluation as ev
from arspinet_v2 import features, reservoir


def test_participants_never_straddle_a_split(cohort):
    _, y, g = cohort
    for _, tr, te in ev.make_splits(y, g):
        assert not np.intersect1d(g[tr], g[te]).size


def test_every_repeat_predicts_every_observation_once(cohort):
    _, y, g = cohort
    splits = ev.make_splits(y, g)
    assert len(splits) == ev.N_REPEATS * ev.N_FOLDS
    for r in range(ev.N_REPEATS):
        te = np.concatenate([t for k, _, t in splits if k == r])
        assert sorted(te.tolist()) == list(range(len(y)))


def test_direct_fit_reproduces_the_v1_estimator(cohort):
    """Same splits, same standardization, same readout as v1's evaluate_features."""
    import confirmatory_pipeline as cp
    raw, y, g = cohort
    X = features.amplitude_bins(raw, 6).reshape(len(y), -1)
    v1 = cp.evaluate_features(X, y, g, base_seed=7)
    v2 = ev.evaluate(X, y, g, base_seed=7, reduce=False)
    assert np.array_equal(v1["oof_pred"], v2["pred"].reshape(-1))
    assert v1["balanced_accuracy"] == pytest.approx(v2["balanced_accuracy"])


def test_row_space_fit_is_the_same_model(projected, cohort):
    _, y, g = cohort
    assert projected.shape[1] > len(y)                      # the p > n regime the reduction is for
    row, direct = ev.evaluate(projected, y, g), ev.evaluate(projected, y, g, reduce=False)
    assert (row["pred"] == direct["pred"]).mean() >= 0.97
    assert row["balanced_accuracy"] > 0.5


def test_balanced_accuracy_matches_sklearn():
    rng = np.random.RandomState(0)
    y, p = rng.randint(0, 3, 200), rng.randint(0, 3, 200)
    assert ev.balanced_accuracy(y, p) == pytest.approx(balanced_accuracy_score(y, p))


def test_vectorized_bootstrap_equals_explicit_resampling(projected, cohort):
    _, y, g = cohort
    res = ev.evaluate(projected, y, g, n_repeats=2)
    w = ev._weights(len(np.unique(g)), 40, 3)
    slow = []
    for b in range(40):
        idx = np.concatenate([np.flatnonzero(g == u) for u, k in zip(np.unique(g), w[b], strict=True)
                              for _ in range(int(k))])
        slow.append(np.mean([(res["pred"][:, idx][:, y[idx] == c] == c).mean() for c in np.unique(y)]))
    assert np.allclose(ev._boot_ba(res, w), slow)


def test_bootstrap_intervals(projected, cohort):
    _, y, g = cohort
    res = ev.evaluate(projected, y, g, n_repeats=2)
    lo, hi = ev.bootstrap_ci(res)
    assert lo <= res["balanced_accuracy"] <= hi
    assert ev.bootstrap_ci(res) == [lo, hi]                 # seeded
    assert ev.bootstrap_difference_ci(res, res) == [0.0, 0.0]


def test_test_only_mode_scores_altered_held_out_rows(projected, cohort):
    _, y, g = cohort
    intact = ev.evaluate(projected, y, g, n_repeats=1)
    same = ev.evaluate(projected, y, g, n_repeats=1, X_test=projected)
    zeroed = ev.evaluate(projected, y, g, n_repeats=1, X_test=np.zeros_like(projected))
    assert np.array_equal(intact["pred"], same["pred"])
    assert zeroed["balanced_accuracy"] < intact["balanced_accuracy"]


def test_non_finite_features_are_rejected(cohort):
    _, y, g = cohort
    X = np.zeros((len(y), 4))
    X[0, 0] = np.nan
    with pytest.raises(ValueError):
        ev.evaluate(X, y, g)


def test_permutation_keeps_each_participants_labels(cohort):
    _, y, g = cohort
    idx = [np.flatnonzero(g == u) for u in np.unique(g)]
    yp = ev._permute_within(y, idx, np.random.RandomState(0))
    assert all(sorted(yp[i]) == sorted(y[i]) for i in idx) and not np.array_equal(yp, y)


def test_permutation_test_detects_signal(projected, cohort):
    _, y, g = cohort
    p = ev.permutation_test(projected, y, g, n_perm=99)
    assert p["p_value"] == p["p_value_floor"] == pytest.approx(0.01)
    assert p["n_exceedances"] == 0 and p["observed_z_vs_null"] > 3


def test_permutation_test_is_calibrated_under_the_null(null_cohort, srp):
    raw, y, g = null_cohort
    X = features.project(reservoir.encode(raw, rho=0.0), srp)
    p = ev.permutation_test(X, y, g, n_perm=99)
    assert p["p_value"] > 0.05
    assert abs(p["null_mean"] - 1 / 3) < 0.03
