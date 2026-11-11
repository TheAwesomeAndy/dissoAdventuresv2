"""Projection, matched non-reservoir controls and windowing.

The controls answer the question v1 never asked: does the spiking reservoir add
anything beyond coarse temporal binning of the input? Each control goes through the
same downstream chain as the reservoir code wherever that is possible.
"""
from __future__ import annotations

import numpy as np
from sklearn.random_projection import SparseRandomProjection

SRP_SEED = 20240517          # v1 value
N_SRP = 64
FS = 256.0                   # Hz after decimation; sample 0 is stimulus onset


def make_srp(n_features, seed=SRP_SEED, n_components=N_SRP):
    srp = SparseRandomProjection(n_components=n_components, random_state=seed)
    srp.fit(np.ones((2, n_features)))
    return srp


def project(blocks, srp):
    """Apply one shared projection to every electrode block. (N, C, D) -> (N, C * k)."""
    blocks = np.asarray(blocks)
    n, c, d = blocks.shape
    z = srp.transform(blocks.reshape(n * c, d).astype(np.float64))
    return np.asarray(z).reshape(n, c * srp.n_components)


def amplitude_bins(raw, n_bins):
    """Mean amplitude in ``n_bins`` equal bins. (N, T, C) -> (N, C, n_bins).

    With ``n_bins=6`` the bin edges are v1's BSC6 edges (T // 6 samples per bin, trailing
    samples dropped), so this is the reservoir code with the reservoir removed.
    """
    raw = np.asarray(raw, dtype=np.float64)
    n, T, c = raw.shape
    size = T // n_bins
    x = raw[:, :size * n_bins, :].reshape(n, n_bins, size, c).mean(axis=2)
    return x.transpose(0, 2, 1)


def ms_window(raw, lo_ms, hi_ms, fs=FS):
    """Slice by time relative to stimulus onset. v1 sliced by fraction and mislabelled
    the first quarter as 39 to 273 ms; it is 0 to 250 ms."""
    lo, hi = int(round(lo_ms * fs / 1000.0)), int(round(hi_ms * fs / 1000.0))
    return np.asarray(raw)[:, lo:hi, :]


WINDOWS_MS = {
    "0-250ms": (0, 250),
    "250-500ms": (250, 500),
    "500-750ms": (500, 750),
    "750-1000ms": (750, 1000),
    "0-1000ms_full": (0, 1000),
}
