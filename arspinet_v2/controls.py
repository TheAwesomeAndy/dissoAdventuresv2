"""Destruction controls on the electrode-resolved binned code.

v1 destroyed the whole dataset before cross-validation with an independent permutation
per observation. That mixes two effects: removing information, and injecting
observation-level noise into training. The bin shuffle then cost more accuracy than the
collapse, although the collapse removes strictly more temporal information.

v2 reports each control in two modes:

``train_and_test``  the v1 design. The model is refitted on destroyed features.
``test_only``       the model is fitted on intact features and scored on destroyed
                    held-out rows. This measures what the fitted readout relies on.

The interpretable temporal control is ``collapse`` under ``train_and_test``: it asks how
much a model can recover with no within-epoch timing at all. ``fixed_bin_permutation``
is a sanity control: one permutation shared by all observations only relabels the code,
so under ``train_and_test`` it should move accuracy by no more than a different
projection draw would.
"""
from __future__ import annotations

import numpy as np


def _blocks(code, n_bins):
    n, c, d = code.shape
    return code.reshape(n, c, n_bins, d // n_bins)


def collapse(code, n_bins=6):
    b = _blocks(code, n_bins)
    return np.repeat(b.mean(axis=2, keepdims=True), n_bins, axis=2).reshape(code.shape)


def shuffle_bins(code, rng, n_bins=6):
    b = _blocks(code, n_bins).copy()
    n, c = b.shape[:2]
    for i in range(n):
        for ch in range(c):
            b[i, ch] = b[i, ch][rng.permutation(n_bins)]
    return b.reshape(code.shape)


def fixed_bin_permutation(code, rng, n_bins=6):
    perm = rng.permutation(n_bins)
    while np.array_equal(perm, np.arange(n_bins)):
        perm = rng.permutation(n_bins)
    return _blocks(code, n_bins)[:, :, perm, :].reshape(code.shape)


def shuffle_electrodes(code, rng):
    out = code.copy()
    for i in range(len(out)):
        out[i] = out[i][rng.permutation(out.shape[1])]
    return out
