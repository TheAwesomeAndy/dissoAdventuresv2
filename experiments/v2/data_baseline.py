"""Non-z-scored SHAPE inputs for the v2 baseline-input runner.

Same files, cohort, exclusions, order and anti-alias decimation as data.py and v1,
without per-epoch z-scoring. Kept separate so data.py stays identical to the file
hashed in v2_results.json.
"""
import os

import numpy as np


# ---------------------------------------------------------------- non-z-scored inputs
import re

from scipy.signal import decimate

_PATTERNS = {
    "three_condition": (re.compile(r"SHAPE_Community_(\d+)_IAPS(Neg|Neu|Pos)_BC\.txt$"),
                        ("Neg", "Neu", "Pos"), {127}),
    "four_category": (re.compile(r"SHAPE_Community_(\d+)_IAPS(?:Neg|Pos)_(Threat|Mutilation|Cute|Erotic)_BC\.txt$"),
                      ("Threat", "Mutilation", "Cute", "Erotic"), {127, 195}),
}
PRESTIM = 205


def _decimated(path):
    """Baseline (204 samples) and post-stimulus (1024 samples), each decimated by four
    with the same zero-phase anti-alias filter v1 uses. Returns (51, C) and (256, C)."""
    x = np.loadtxt(path)
    dec = lambda seg: np.column_stack([decimate(seg[:, c], 4, zero_phase=True)  # noqa: E731
                                       for c in range(seg.shape[1])])
    return dec(x[1:PRESTIM]), dec(x[PRESTIM:PRESTIM + 1024])


def load_baseline_referenced(name, data_dir, cache=None):
    """The delivered files (already baseline-corrected) with v1's cohort, order and
    decimation, but without per-epoch z-scoring.

    Returns raw_uv (N, 256, C) in microvolts and baseline_sd (N, C), the standard
    deviation of each epoch's own pre-stimulus interval. Neither uses any sample after
    a given latency to transform that latency.
    """
    f = os.path.join(cache, f"{name}_baseline_referenced.npz") if cache else None
    if f and os.path.exists(f):
        z = np.load(f)
        return z["raw_uv"], z["baseline_sd"], z["y"], z["groups"]
    pat, labels, excluded = _PATTERNS[name]
    found = {}
    for root, _, files in os.walk(data_dir):
        for fn in files:
            m = pat.search(fn)
            if m and int(m.group(1)) not in excluded:
                found[(int(m.group(1)), m.group(2))] = os.path.join(root, fn)
    subjects = sorted({s for s, _ in found})
    raw, bsd, y, g = [], [], [], []
    for sid in subjects:
        for k, lab in enumerate(labels):
            b, p = _decimated(found[(sid, lab)])
            raw.append(p)
            bsd.append(b.std(axis=0))
            y.append(k)
            g.append(sid)
    raw, bsd, y, g = np.asarray(raw), np.asarray(bsd), np.asarray(y), np.asarray(g)
    if f:
        os.makedirs(cache, exist_ok=True)
        np.savez(f, raw_uv=raw, baseline_sd=bsd, y=y, groups=g)
    return raw, bsd, y, g
