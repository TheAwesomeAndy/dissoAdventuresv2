#!/usr/bin/env python3
"""Pre-specified depression analysis (requires authorized SHAPE data and SCID-5 labels).

The source laboratory suggested asking whether the ERP data differentiate depression
groups, taking four SCID-5 variables in a fixed order:

    1. MDE              major depressive episode (absent / present)
    2. MDD              major depressive disorder, no bipolar (absent / present)
    3. Depression_Type  0 absent, 1 single episode, 2 chronic or persistent
    4. MDD_Recurrent    recurrent depression (absent / present)

v1 ran its clinical analyses as an uncorrected screen. Here the family is fixed in
advance: those four variables, three participant-level representations, one estimator.
Twelve tests, Holm-corrected together.

Each participant contributes one feature vector: their three condition responses
(Negative, Neutral, Pleasant) concatenated. Representations:

    lpp        LPP mean amplitude 300 to 1000 ms per electrode (the laboratory's score)
    amplitude6 six-bin amplitudes per electrode
    bsc6_srp64 the v2 headline reservoir code

Estimator: standardization and class-weighted L2 logistic regression (C = 1) fitted on
training participants only, 5 repeats of stratified 5-fold over participants. Statistic:
balanced accuracy. Null: labels permuted across participants, same estimator.

Only aggregate results are written. No participant-level value leaves this script.

    python experiments/v2/run_v2_depression.py --data3 ./batch_data --labels labels.csv \
        --cache ./v2_cache --out experiments/v2/v2_depression.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)

import data as shape_data  # noqa: E402
from arspinet_v2 import features, reservoir  # noqa: E402
from arspinet_v2 import evaluation as ev  # noqa: E402

TARGETS = ("MDE", "MDD", "Depression_Type", "MDD_Recurrent")     # the laboratory's order
N_REPEATS, N_FOLDS = 5, 5


def participant_matrix(X, y, groups):
    """(observations, d) -> (participants, 3 d), conditions in label order."""
    ids = np.unique(groups)
    out = np.empty((len(ids), len(np.unique(y)) * X.shape[1]))
    for i, g in enumerate(ids):
        m = groups == g
        out[i] = X[m][np.argsort(y[m])].reshape(-1)
    return ids, out


def designs(X, t, n_repeats, seed):
    out = []
    for r in range(n_repeats):
        for tr, te in StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed + r).split(X, t):
            out.append((r, ev.FoldDesign(X, tr, te)))
    return out


def score(ds, t, n_repeats):
    pred = np.full((n_repeats, len(t)), -1, dtype=int)
    for r, d in ds:
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
        clf.fit(d.Ztr, t[d.tr])
        pred[r, d.te] = clf.predict(d.Zte)
    return float(np.mean([ev.balanced_accuracy(t, p) for p in pred])), pred


def analyse(X, t, n_perm, seed):
    ds = designs(X, t, N_REPEATS, seed)
    observed, pred = score(ds, t, N_REPEATS)
    # participant bootstrap of the pooled out-of-fold predictions
    rng = np.random.RandomState(seed + 11)
    boot = []
    for _ in range(2000):
        idx = rng.randint(0, len(t), len(t))
        boot.append(np.mean([(pred[:, idx][:, t[idx] == c] == c).mean() for c in np.unique(t)
                             if (t[idx] == c).any()]))
    ds1 = [(0, d) for r, d in ds if r == 0]
    obs1, _ = score(ds1, t, 1)
    rng = np.random.RandomState(seed + 777)
    null = np.array([score(ds1, rng.permutation(t), 1)[0] for _ in range(n_perm)])
    return {
        "balanced_accuracy_percent": round(100 * observed, 2),
        "ci95_percent": [round(100 * float(np.percentile(boot, 2.5)), 2),
                         round(100 * float(np.percentile(boot, 97.5)), 2)],
        "chance_percent": round(100.0 / len(np.unique(t)), 2),
        "permutation": {"observed_one_repeat_percent": round(100 * obs1, 2),
                        "null_mean_percent": round(100 * float(null.mean()), 2),
                        "null_sd_percent": round(100 * float(null.std(ddof=1)), 2),
                        "n_permutations": int(n_perm),
                        "p_value": float((1 + (null >= obs1).sum()) / (1 + n_perm))},
    }


def holm(pvals):
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data3", required=True)
    ap.add_argument("--labels", required=True, help="CSV or XLSX with an ID column and the SCID-5 variables")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--rho", type=float, default=None, help="headline recurrent gain (default: read from v2_results.json)")
    ap.add_argument("--out", default=os.path.join(_HERE, "v2_depression.json"))
    ap.add_argument("--n-perm", type=int, default=1000)
    args = ap.parse_args()

    rho = args.rho
    if rho is None:
        rho = json.load(open(os.path.join(_HERE, "v2_results.json")))["fading_memory"]["selected_rho"]
    raw, y, groups = shape_data.load(args.data3, None, args.cache)["three_condition"]
    lab = pd.read_excel(args.labels) if args.labels.lower().endswith("xlsx") else pd.read_csv(args.labels)
    lab = lab.set_index("ID")

    code = reservoir.encode(raw, rho=rho)
    reps = {
        "lpp": features.lpp_mean_amplitude(raw),
        "amplitude6": features.amplitude_bins(raw, 6).reshape(len(y), -1),
        "bsc6_srp64": features.project(code, features.make_srp(code.shape[2])),
    }
    out = {"design": {"targets_in_order": list(TARGETS), "representations": list(reps),
                      "estimator": "standardize + class-weighted L2 logistic (C=1), 5x5 stratified CV over participants",
                      "family": "12 tests (4 targets x 3 representations), Holm-corrected together",
                      "headline_rho": rho,
                      "covariates": "not adjusted: age and sex are missing for roughly half of the cohort"},
           "targets": {}}
    keys, pvals = [], []
    for ti, target in enumerate(TARGETS):
        out["targets"][target] = {}
        for name, X in reps.items():
            ids, P = participant_matrix(X, y, groups)
            t = lab.reindex(ids)[target].to_numpy(dtype=float)
            keep = np.isfinite(t)
            t = t[keep].astype(int)
            res = analyse(P[keep], t, args.n_perm, seed=1000 + ti)
            res["n_participants"] = int(keep.sum())
            res["class_counts"] = {str(c): int((t == c).sum()) for c in np.unique(t)}
            out["targets"][target][name] = res
            keys.append((target, name))
            pvals.append(res["permutation"]["p_value"])
            print(f"[{time.strftime('%H:%M:%S')}] {target:16s} {name:11s} BA {res['balanced_accuracy_percent']:.1f} "
                  f"{res['ci95_percent']} chance {res['chance_percent']} p {res['permutation']['p_value']:.4f}", flush=True)
    for (target, name), p_adj in zip(keys, holm(np.array(pvals))):
        out["targets"][target][name]["permutation"]["p_value_holm_12"] = float(p_adj)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
