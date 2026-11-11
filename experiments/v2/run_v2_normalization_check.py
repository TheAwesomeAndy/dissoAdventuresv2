#!/usr/bin/env python3
"""Does per-epoch z-scoring smear late information into early latencies?

v1 z-scores every epoch and channel over the whole post-stimulus second. Subtracting the
epoch mean writes a copy of the late potentials (with opposite sign) into every sample,
including the first 30 ms, where no cortical response to picture content exists yet.

This check repeats the time-resolved scalp-pattern decoding under two normalizations:

    epoch_z       v1: z-score over the post-stimulus epoch, per channel
    baseline_only the files as delivered (already baseline-corrected to the 200 ms
                  pre-stimulus interval), decimated identically, no further scaling

If early-latency decoding is far above chance under ``epoch_z`` and at chance under
``baseline_only``, the early accuracy is produced by the normalization.

    python experiments/v2/run_v2_normalization_check.py --data3 ./batch_data \
        --out experiments/v2/v2_normalization_check.json
"""
import argparse
import json
import os
import re
import sys

import numpy as np
from scipy.signal import decimate

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import data as shape_data  # noqa: E402
from arspinet_v2 import features  # noqa: E402
from arspinet_v2 import evaluation as ev  # noqa: E402

LABELS = {"Neg": 0, "Neu": 1, "Pos": 2}
EXCLUDED = {127}


def load_baseline_only(data_dir):
    pat = re.compile(r"SHAPE_Community_(\d+)_IAPS(Neg|Neu|Pos)_BC\.txt$")
    found = {}
    for root, _, files in os.walk(data_dir):
        for f in files:
            m = pat.search(f)
            if m and int(m.group(1)) not in EXCLUDED:
                found[(int(m.group(1)), m.group(2))] = os.path.join(root, f)
    X, y, g = [], [], []
    for sid in sorted({s for s, _ in found}):
        for cond in ("Neg", "Neu", "Pos"):
            x = np.loadtxt(found[(sid, cond)])[205:205 + 1024]
            X.append(np.column_stack([decimate(x[:, c], 4, zero_phase=True) for c in range(x.shape[1])]))
            y.append(LABELS[cond])
            g.append(sid)
    return np.asarray(X), np.asarray(y), np.asarray(g)


def profile(raw, y, g):
    E = lambda X: ev.evaluate(X, y, g, base_seed=0)  # noqa: E731
    bins = []
    for b, pattern in enumerate(features.time_resolved_patterns(raw, 32)):
        r = E(pattern)
        ci = ev.bootstrap_ci(r)
        bins.append({"start_ms": round(b * 31.25, 1), "end_ms": round((b + 1) * 31.25, 1),
                     "balanced_accuracy_percent": round(100 * r["balanced_accuracy"], 2),
                     "ci95_percent": [round(100 * ci[0], 2), round(100 * ci[1], 2)]})
    whole = {}
    for name, X in (("raw_erp16", features.amplitude_bins(raw, 16).reshape(len(y), -1)),
                    ("amplitude6", features.amplitude_bins(raw, 6).reshape(len(y), -1)),
                    ("lpp_300_1000ms", features.lpp_mean_amplitude(raw))):
        r = E(X)
        ci = ev.bootstrap_ci(r)
        whole[name] = {"balanced_accuracy_percent": round(100 * r["balanced_accuracy"], 2),
                       "ci95_percent": [round(100 * ci[0], 2), round(100 * ci[1], 2)]}
    return {"time_resolved": bins, "whole_epoch": whole}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data3", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default=os.path.join(_HERE, "v2_normalization_check.json"))
    args = ap.parse_args()
    raw_z, y, g = shape_data.load(args.data3, None, args.cache)["three_condition"]
    raw_b, yb, gb = load_baseline_only(args.data3)
    assert np.array_equal(y, yb) and np.array_equal(g, gb)
    z = (raw_b - raw_b.mean(axis=1, keepdims=True)) / raw_b.std(axis=1, keepdims=True)
    out = {"cohort": "three_condition", "n_participants": int(len(np.unique(g))), "chance_percent": 33.33,
           "epoch_z_reconstruction_max_abs_error": float(np.abs(z - raw_z).max()),
           "epoch_z": profile(raw_z, y, g), "baseline_only": profile(raw_b, y, g)}
    for k in ("epoch_z", "baseline_only"):
        t = out[k]["time_resolved"]
        print(k, "0-31 ms", t[0]["balanced_accuracy_percent"], "| 31-62", t[1]["balanced_accuracy_percent"],
              "| peak", max(b["balanced_accuracy_percent"] for b in t), out[k]["whole_epoch"])
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
