"""Headline estimator, participant bootstrap and permutation test.

The protocol is v1's: repeated participant-grouped stratified K-fold, per-feature
standardization and an L2 logistic readout (C = 1) fitted on training participants only.

Two things change.

1. The readout is fitted in the row space of the standardized training matrix when there
   are more features than training observations. An L2-penalized linear model with an
   unpenalized intercept is invariant to that rotation, so the fitted decision function
   is the same model, up to solver tolerance, at a fraction of the cost. The
   decomposition does not depend on labels, so it is computed once per fold and reused.

2. Because of (1), the permutation test can afford the headline estimator. v1 used a
   ridge classifier that scored 3 to 8 points below the model it stood in for, with 100
   permutations, so its p-value was pinned at the floor 1/101.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

N_REPEATS = 5
N_FOLDS = 5
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 12345
C_READOUT = 1.0


def make_splits(y, groups, n_repeats=N_REPEATS, n_folds=N_FOLDS, base_seed=0):
    """v1's splits: StratifiedGroupKFold(shuffle, random_state=base_seed + repeat)."""
    y, groups = np.asarray(y), np.asarray(groups)
    dummy = np.zeros((len(y), 1))
    out = []
    for r in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=base_seed + r)
        for tr, te in sgkf.split(dummy, y, groups):
            if np.intersect1d(groups[tr], groups[te]).size:
                raise RuntimeError("participant appears in both train and test")
            out.append((r, tr, te))
    return out


class FoldDesign:
    """Label-free part of one fold: standardization and row-space reduction."""

    def __init__(self, X, tr, te, X_test=None, reduce=True):
        Xtr = np.asarray(X[tr], dtype=np.float64)
        self.mean = Xtr.mean(axis=0)
        scale = Xtr.std(axis=0)
        self.scale = np.where(scale > 0, scale, 1.0)
        Ztr = (Xtr - self.mean) / self.scale
        Xte = np.asarray((X if X_test is None else X_test)[te], dtype=np.float64)
        Zte = (Xte - self.mean) / self.scale
        if reduce and Ztr.shape[1] > Ztr.shape[0]:
            u, s, vt = np.linalg.svd(Ztr, full_matrices=False)
            keep = s > s[0] * 1e-12
            self.Ztr, self.Zte = u[:, keep] * s[keep], Zte @ vt[keep].T
        else:
            self.Ztr, self.Zte = Ztr, Zte
        self.tr, self.te = tr, te

    def predict(self, y, C=C_READOUT):
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(self.Ztr, y[self.tr])
        return clf.predict(self.Zte)


def balanced_accuracy(y, pred, n_classes=None):
    y, pred = np.asarray(y), np.asarray(pred)
    classes = np.unique(y) if n_classes is None else np.arange(n_classes)
    return float(np.mean([(pred[y == c] == c).mean() for c in classes]))


def evaluate(X, y, groups, n_repeats=N_REPEATS, n_folds=N_FOLDS, base_seed=0,
             X_test=None, reduce=True, C=C_READOUT):
    """Repeated grouped CV. ``X_test`` lets a model trained on ``X`` be scored on an
    altered copy of the held-out rows (test-only destruction controls)."""
    y, groups = np.asarray(y), np.asarray(groups)
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values")
    pred = np.full((n_repeats, len(y)), -1, dtype=int)
    for r, tr, te in make_splits(y, groups, n_repeats, n_folds, base_seed):
        pred[r, te] = FoldDesign(X, tr, te, X_test=X_test, reduce=reduce).predict(y, C=C)
    if (pred < 0).any():
        raise RuntimeError("an observation did not receive an out-of-fold prediction")
    per_repeat = [balanced_accuracy(y, p) for p in pred]
    return {"balanced_accuracy": float(np.mean(per_repeat)), "per_repeat": per_repeat,
            "pred": pred, "y": y, "groups": groups}


# ---------------------------------------------------------------- participant bootstrap
def _recall_tables(result):
    y, groups, pred = result["y"], result["groups"], result["pred"]
    ug, gi = np.unique(groups, return_inverse=True)
    classes = np.unique(y)
    correct = np.zeros((len(ug), len(classes)))
    total = np.zeros((len(ug), len(classes)))
    for k, c in enumerate(classes):
        m = y == c
        np.add.at(total[:, k], gi[m], pred.shape[0])
        np.add.at(correct[:, k], gi[m], (pred[:, m] == c).sum(axis=0))
    return correct, total


def _weights(n_groups, n_boot, seed):
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n_groups, size=(n_boot, n_groups))
    w = np.zeros((n_boot, n_groups))
    for b in range(n_boot):
        np.add.at(w[b], idx[b], 1.0)
    return w


def _boot_ba(result, w):
    correct, total = _recall_tables(result)
    return ((w @ correct) / (w @ total)).mean(axis=1)


def bootstrap_ci(result, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED, alpha=0.05):
    """Participant-cluster percentile interval over the pooled repeated predictions.

    The interval conditions on the fitted models (participants are resampled, models
    are not refitted), so it is somewhat narrow. v1 had the same property.
    """
    w = _weights(len(np.unique(result["groups"])), n_boot, seed)
    s = _boot_ba(result, w)
    return [float(np.percentile(s, 100 * alpha / 2)), float(np.percentile(s, 100 * (1 - alpha / 2)))]


def bootstrap_difference_ci(a, b, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED, alpha=0.05):
    """Paired participant-bootstrap interval for BA(a) - BA(b)."""
    w = _weights(len(np.unique(a["groups"])), n_boot, seed)
    s = _boot_ba(a, w) - _boot_ba(b, w)
    return [float(np.percentile(s, 100 * alpha / 2)), float(np.percentile(s, 100 * (1 - alpha / 2)))]


# --------------------------------------------------------------------- permutation test
def _permute_within(y, groups_index, rng):
    out = y.copy()
    for idx in groups_index:
        out[idx] = rng.permutation(y[idx])
    return out


def _perm_chunk(designs, y, groups_index, seeds):
    scores = []
    for s in seeds:
        yp = _permute_within(y, groups_index, np.random.RandomState(s))
        pred = np.full(len(y), -1, dtype=int)
        for d in designs:
            pred[d.te] = d.predict(yp)
        scores.append(balanced_accuracy(yp, pred))
    return scores


def permutation_test(X, y, groups, n_perm=1000, n_folds=N_FOLDS, base_seed=0, n_jobs=1,
                     perm_seed=777):
    """Within-participant label permutation with the headline estimator.

    One repeat of the grouped K-fold (the first repeat of the headline splits) is used
    for both the observed statistic and every permutation, so the two are comparable.
    Every participant contributes one observation per class, so permuting labels within
    a participant leaves the stratified folds unchanged.
    """
    from joblib import Parallel, delayed

    y, groups = np.asarray(y), np.asarray(groups)
    designs = [FoldDesign(X, tr, te) for _, tr, te in make_splits(y, groups, 1, n_folds, base_seed)]
    groups_index = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    pred = np.full(len(y), -1, dtype=int)
    for d in designs:
        pred[d.te] = d.predict(y)
    observed = balanced_accuracy(y, pred)
    seeds = np.random.RandomState(perm_seed + base_seed).randint(0, 2**31 - 1, size=n_perm)
    chunks = np.array_split(seeds, max(1, min(n_perm, 8 * n_jobs)))
    res = Parallel(n_jobs=n_jobs)(
        delayed(_perm_chunk)(designs, y, groups_index, c) for c in chunks if len(c))
    null = np.concatenate([np.asarray(r) for r in res])
    return {
        "estimator": "headline: standardize + L2 logistic (C=1), one repeat of grouped 5-fold",
        "observed_balanced_accuracy": float(observed),
        "n_permutations": int(len(null)),
        "n_exceedances": int((null >= observed).sum()),
        "p_value": float((1 + (null >= observed).sum()) / (1 + len(null))),
        "p_value_floor": float(1.0 / (1 + len(null))),
        "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
        "null_max": float(null.max()),
        "observed_z_vs_null": float((observed - null.mean()) / null.std(ddof=1)),
    }
