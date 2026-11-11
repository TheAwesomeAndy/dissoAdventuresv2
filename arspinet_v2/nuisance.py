"""Calibrated subject and condition variance components.

v1 summarized subject dominance with a ratio rho of between-subject to between-condition
variance. That ratio has no noise floor: features with no structure at all give a large
value, because a subject mean over K observations always has variance sigma^2 / K. The
review measured rho near 105 on pure noise.

For a complete subjects x conditions panel (one observation per cell) the two-way
layout gives, per feature,

    sigma2_subject   = (MS_subject   - MS_residual) / K
    sigma2_condition = (MS_condition - MS_residual) / S
    sigma2_residual  =  MS_residual

These are unbiased, so structureless features give components near zero. Features are
standardized first and components are averaged over features. Permutation nulls are
provided for both effects.
"""
from __future__ import annotations

import numpy as np


def _panel(X, y, groups):
    X = np.asarray(X, dtype=np.float64)
    ug, classes = np.unique(groups), np.unique(y)
    S, K = len(ug), len(classes)
    gi = np.searchsorted(ug, groups)
    ki = np.searchsorted(classes, y)
    if len(X) != S * K or len(set(zip(gi.tolist(), ki.tolist()))) != S * K:
        raise ValueError("variance components need one observation per subject x condition")
    sd = X.std(axis=0)
    Xs = (X[:, sd > 0] - X[:, sd > 0].mean(axis=0)) / sd[sd > 0]
    P = np.empty((S, K, Xs.shape[1]))
    P[gi, ki] = Xs
    return P


def _components(P):
    S, K, _ = P.shape
    grand = P.mean(axis=(0, 1), keepdims=True)
    ms_s = K * ((P.mean(axis=1, keepdims=True) - grand) ** 2).sum(axis=(0, 1)) / (S - 1)
    ms_c = S * ((P.mean(axis=0, keepdims=True) - grand) ** 2).sum(axis=(0, 1)) / (K - 1)
    resid = P - P.mean(axis=1, keepdims=True) - P.mean(axis=0, keepdims=True) + grand
    ms_r = (resid ** 2).sum(axis=(0, 1)) / ((S - 1) * (K - 1))
    v_s, v_c, v_r = (ms_s - ms_r) / K, (ms_c - ms_r) / S, ms_r
    tot = (v_s + v_c + v_r).mean()
    return {"subject": float(v_s.mean() / tot), "condition": float(v_c.mean() / tot),
            "residual": float(v_r.mean() / tot)}


def variance_components(X, y, groups, n_perm=1000, seed=0):
    """Fractions of total variance, with permutation nulls.

    Subject null: observations are reassigned to subjects within each condition.
    Condition null: condition labels are permuted within each subject.
    """
    P = _panel(X, y, groups)
    S, K, _ = P.shape
    obs = _components(P)
    rng = np.random.RandomState(seed)
    null_s, null_c = np.empty(n_perm), np.empty(n_perm)
    for i in range(n_perm):
        Ps = np.stack([P[rng.permutation(S), k] for k in range(K)], axis=1)
        null_s[i] = _components(Ps)["subject"]
        Pc = np.stack([P[s, rng.permutation(K)] for s in range(S)], axis=0)
        null_c[i] = _components(Pc)["condition"]
    naive_ratio = (P.mean(axis=1).var(axis=0, ddof=1).mean()
                   / P.mean(axis=0).var(axis=0, ddof=1).mean())
    return {
        "fraction_subject": obs["subject"], "fraction_condition": obs["condition"],
        "fraction_residual": obs["residual"],
        "subject_null_mean": float(null_s.mean()), "subject_null_p95": float(np.percentile(null_s, 95)),
        "subject_p_value": float((1 + (null_s >= obs["subject"]).sum()) / (1 + n_perm)),
        "condition_null_mean": float(null_c.mean()), "condition_null_p95": float(np.percentile(null_c, 95)),
        "condition_p_value": float((1 + (null_c >= obs["condition"]).sum()) / (1 + n_perm)),
        "n_permutations": int(n_perm),
        "uncalibrated_v1_style_ratio": float(naive_ratio),
    }
