"""Shared fixtures. Everything here is synthetic; no test needs restricted data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
# v1 and experiment modules are scripts, not packages. Tests import them by path, the same
# way the scripts import each other.
for sub in (
    "chapter5Experiments",
    "experiments/confirmatory",
    "experiments/ch5_4class",
    "experiments/v2",
    "experiments/zscore_artifact",
    "scripts",
):
    p = str(REPO / sub)
    if p not in sys.path:
        sys.path.insert(0, p)


def make_cohort(n_subj=40, n_cls=3, n_time=256, n_ch=6, seed=0, effect=1.0):
    """ERP-like epochs shaped like preprocessed SHAPE data: (observations, time, channels),
    z-scored per epoch and channel, one observation per participant and condition.

    Each observation is a participant-specific early deflection, a condition-specific
    late deflection of size ``effect`` and unit noise.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_time) / 256.0
    late = np.exp(-0.5 * ((t - 0.5) / 0.12) ** 2)
    early = np.exp(-0.5 * ((t - 0.15) / 0.04) ** 2)
    topo = rng.normal(0, 1, (n_cls, n_ch))
    X, y, g = [], [], []
    for s in range(n_subj):
        subj = rng.normal(0, 1, n_ch)
        for c in range(n_cls):
            sig = (
                subj[None] * early[:, None]
                + effect * topo[c][None] * late[:, None]
                + rng.normal(0, 1, (n_time, n_ch))
            )
            X.append((sig - sig.mean(0)) / sig.std(0))
            y.append(c)
            g.append(s)
    return np.asarray(X), np.asarray(y), np.asarray(g)


@pytest.fixture(scope="session", autouse=True)
def _limit_blas_threads():
    """Small problems run slower, not faster, when BLAS oversubscribes shared runners."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover
        yield
        return
    with threadpool_limits(limits=2):
        yield


@pytest.fixture(scope="session")
def repo() -> Path:
    return REPO


@pytest.fixture(scope="session")
def cohort():
    return make_cohort()


@pytest.fixture(scope="session")
def null_cohort():
    return make_cohort(effect=0.0, seed=5)


@pytest.fixture(scope="session")
def code(cohort):
    """Feedforward (rho = 0) shared-reservoir code of the synthetic cohort."""
    from arspinet_v2 import reservoir

    return reservoir.encode(cohort[0], rho=0.0)


@pytest.fixture(scope="session")
def srp():
    from arspinet_v2 import features

    return features.make_srp(6 * 256)


@pytest.fixture(scope="session")
def projected(code, srp):
    from arspinet_v2 import features

    return features.project(code, srp)
