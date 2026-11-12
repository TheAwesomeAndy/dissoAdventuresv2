#!/usr/bin/env python3
"""Permutation null for the across-participant regime of run_erpcore.py.

Statistic: mean balanced accuracy over the pre-stimulus bins (-200 to 0 ms), one
participant-grouped 5-fold split, standardization + L2 logistic (C = 1), as in
run_erpcore.across. Null: each participant's two condition labels are swapped with
probability 1/2, the only permutation that keeps one average per condition per
participant. The same splits are used for observed and null values.

The observed value here uses one CV repeat, so it differs slightly from the five-repeat
value in erpcore_results.json.

    python experiments/zscore_artifact/run_erpcore_across_null.py --data ./erpcore --out across_null.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_erpcore as R  # noqa: E402


def pre_acc(avg, y, splits, pre_bins):
    accs = []
    for b in pre_bins:
        pred = np.full(len(y), -1)
        for tr, te in splits:
            sc = StandardScaler().fit(avg[tr, b])
            clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(avg[tr, b]), y[tr])
            pred[te] = clf.predict(sc.transform(avg[te, b]))
        accs.append(R.bal_acc(y, pred))
    return float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    out = {"design": {"statistic": "mean balanced accuracy over pre-stimulus bins, across-participant regime",
                      "null": "condition labels swapped within participant with probability 1/2",
                      "n_permutations": args.n_perm}, "paradigms": {}}
    for para, (ka, kb) in R.PARADIGMS.items():
        root = os.path.join(args.data, para)
        sids = sorted((d for d in os.listdir(root) if d.isdigit()), key=int)
        avgs = {n: [] for n in R.NORMS}
        g = []
        for sid in sids:
            x, times, bini, rej, labels = R.load_participant(os.path.join(root, sid, f"{sid}_{para}_ar.set"))
            a, b = R.pick(labels, ka), R.pick(labels, kb)
            keep = ~rej & np.isin(bini, (a, b))
            x, yy = x[keep], (bini[keep] == b).astype(int)
            for n in R.NORMS:
                xb = R.binned(R.normalize(x, times, n))
                avgs[n] += [xb[yy == 0].mean(axis=0), xb[yy == 1].mean(axis=0)]
            g += [int(sid), int(sid)]
        centers = times[:(len(times) // R.BIN) * R.BIN].reshape(-1, R.BIN).mean(axis=1)
        pre_bins = np.flatnonzero((centers >= R.PRE[0]) & (centers < R.PRE[1]))
        y = np.tile([0, 1], len(sids))
        g = np.asarray(g)
        splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, g))
        rng = np.random.RandomState(2026)
        flips = rng.randint(0, 2, size=(args.n_perm, len(sids)))
        out["paradigms"][para] = {}
        for n in R.NORMS:
            A = np.asarray(avgs[n])
            obs = pre_acc(A, y, splits, pre_bins)
            perm_y = [np.where(np.repeat(f, 2) == 1, 1 - y, y) for f in flips]
            null = np.asarray(Parallel(n_jobs=args.jobs)(delayed(pre_acc)(A, yp, splits, pre_bins) for yp in perm_y))
            rec = {"observed_percent": round(100 * obs, 2), "null_mean_percent": round(100 * float(null.mean()), 2),
                   "null_p95_percent": round(100 * float(np.percentile(null, 95)), 2),
                   "p_value": float((1 + (null >= obs).sum()) / (1 + len(null)))}
            out["paradigms"][para][n] = rec
            print(f"[{time.strftime('%H:%M:%S')}] {para} {n:13s} pre-stimulus {rec['observed_percent']:.1f} "
                  f"null {rec['null_mean_percent']:.1f} (95th {rec['null_p95_percent']:.1f}) p {rec['p_value']:.4f}", flush=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
