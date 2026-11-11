"""SHAPE loading for the v2 runner.

Preprocessing is v1's audited path, imported unchanged from
experiments/confirmatory/run_confirmatory_validation.py: drop the 205 pre-stimulus
samples, anti-alias decimate by four to 256 samples at 256 Hz, z-score each epoch and
channel. Exclusions are v1's (participant 127 from both streams, 195 from the
four-category stream). The preprocessed arrays can be cached outside the repository.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "experiments", "confirmatory"))
sys.path.insert(0, _REPO)

import run_confirmatory_validation as v1  # noqa: E402


def load(data3, data4, cache=None):
    out = {}
    for name, path, loader in (("three_condition", data3, v1.load_3class),
                               ("four_category", data4, v1.load_4class)):
        if not path:
            continue
        f = os.path.join(cache, f"{name}.npz") if cache else None
        if f and os.path.exists(f):
            z = np.load(f)
            out[name] = (z["raw"], z["y"], z["groups"])
            continue
        raw, y, groups = loader(path)
        if f:
            os.makedirs(cache, exist_ok=True)
            np.savez(f, raw=raw, y=y, groups=groups)
        out[name] = (raw, y, groups)
    return out
