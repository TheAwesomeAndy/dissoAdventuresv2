#!/usr/bin/env python3
"""Is the comparison between the reservoir and the amplitude baselines fair to both?

Every v2 accuracy uses one readout setting (standardized features, C = 1). The reservoir
code has 2,176 features, raw ERP-16 has 544, six-bin amplitudes 204 and the LPP score 34,
so one regularization strength does not suit them equally. The tuned-PCA control showed
how much that can matter. This runner gives each representation its own regularization
strength, chosen by nested participant-grouped cross-validation, and asks whether the
ordering of representations changes.

Representations, on the dissertation's preprocessing (per-epoch z-scored input):

    v2_headline_bsc6_srp64   reservoir code, shared reservoir, selected gain
    raw_erp16                mean amplitude in 16 bins per electrode
    amplitude6_matched_bins  mean amplitude in the reservoir's six bins
    lpp_300_1000ms           LPP mean amplitude per electrode

Each is scored with C = 1 and with C tuned. Tuning uses the grid, the inner folds and
the selection rule of run_v2_pca_tuned.py: within each outer training set, 4-fold
participant-grouped CV over C; the best inner balanced accuracy picks C. Outer splits
are v1's. Nothing is chosen on outer-test accuracy.

    python experiments/v2/run_v2_baseline_tuned.py --cache ./v2_cache --jobs 2
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
sys.path.insert(0, _HERE)

from arspinet_v2 import evaluation as ev
from arspinet_v2 import features
from arspinet_v2 import provenance as prov

import run_v2_pca_tuned as tuned
import run_v2_validation as v2

CODE_FILES = [
    "experiments/v2/run_v2_baseline_tuned.py",
    "experiments/v2/run_v2_pca_tuned.py",
    "experiments/v2/config.toml",
    "arspinet_v2/features.py",
    "arspinet_v2/evaluation.py",
]
REPRESENTATIONS = ("v2_headline_bsc6_srp64", "raw_erp16", "amplitude6_matched_bins", "lpp_300_1000ms")


def outer_fold(X, y, g, r, tr, te, seed):
    """C = 1 and nested-tuned predictions for every representation, one outer fold."""
    out = {}
    for name, Xr in X.items():
        Ztr, Zte = tuned._design(Xr[tr], Xr[te], True)
        out[f"{name}_C1"] = (tuned._fit_predict(Ztr, y[tr], Zte, 1.0), 1.0)
        inner = StratifiedGroupKFold(tuned.N_INNER, shuffle=True, random_state=seed + 1000 + r)
        scores = np.zeros(len(tuned.C_GRID))
        for itr, ite in inner.split(np.zeros(len(tr)), y[tr], g[tr]):
            a, b = tr[itr], tr[ite]
            Zi_tr, Zi_te = tuned._design(Xr[a], Xr[b], True)
            for j, C in enumerate(tuned.C_GRID):
                scores[j] += ev.balanced_accuracy(y[b], tuned._fit_predict(Zi_tr, y[a], Zi_te, C))
        C_best = tuned.C_GRID[int(np.argmax(scores))]
        out[f"{name}_Ctuned"] = (tuned._fit_predict(Ztr, y[tr], Zte, C_best), C_best)
    return r, te, out


def run(name, raw, code, y, g, jobs):
    X = {
        "v2_headline_bsc6_srp64": features.project(code, features.make_srp(code.shape[2])),
        "raw_erp16": features.amplitude_bins(raw, 16).reshape(len(y), -1),
        "amplitude6_matched_bins": features.amplitude_bins(raw, 6).reshape(len(y), -1),
        "lpp_300_1000ms": features.lpp_mean_amplitude(raw),
    }
    base = v2.BASE_SEED[name]
    res = Parallel(n_jobs=jobs)(
        delayed(outer_fold)(X, y, g, r, tr, te, base) for r, tr, te in ev.make_splits(y, g, base_seed=base)
    )
    preds, chosen = {}, {}
    for r, te, out in res:
        for k, (p, C) in out.items():
            preds.setdefault(k, np.full((ev.N_REPEATS, len(y)), -1, dtype=int))[r, te] = p
            chosen.setdefault(k, []).append(C)
    results = {
        k: {
            "balanced_accuracy": float(np.mean([ev.balanced_accuracy(y, q) for q in p])),
            "pred": p,
            "y": y,
            "groups": g,
        }
        for k, p in preds.items()
    }

    def pct(v):
        return round(100 * float(v), 4)

    def contrast(a, b):
        return {
            "difference_pp": pct(results[a]["balanced_accuracy"] - results[b]["balanced_accuracy"]),
            "ci95_pp": [pct(v) for v in ev.bootstrap_difference_ci(results[a], results[b])],
        }

    table = {}
    for k, r in results.items():
        ci = ev.bootstrap_ci(r)
        Cs = np.asarray(chosen[k])
        table[k] = {
            "balanced_accuracy_percent": pct(r["balanced_accuracy"]),
            "ci95_percent": [pct(ci[0]), pct(ci[1])],
            "n_features": int(X[k.rsplit("_C", 1)[0]].shape[1]),
            "C_chosen": {str(c): int((Cs == c).sum()) for c in np.unique(Cs)},
        }
    head = "v2_headline_bsc6_srp64"
    return {
        "variants": table,
        "tuning_gain_pp": {n: contrast(f"{n}_Ctuned", f"{n}_C1") for n in REPRESENTATIONS},
        "contrasts_tuned": {
            f"{n}_minus_headline": contrast(f"{n}_Ctuned", f"{head}_Ctuned")
            for n in REPRESENTATIONS
            if n != head
        },
        "contrasts_C1": {
            f"{n}_minus_headline": contrast(f"{n}_C1", f"{head}_C1") for n in REPRESENTATIONS if n != head
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--cache", required=True, help="directory holding preprocessed epochs and v2 headline codes"
    )
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--out", default=os.path.join(_HERE, "v2_baseline_tuned.json"))
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    if args.rho is None:
        with open(os.path.join(_HERE, "v2_results.json")) as fh:
            args.rho = json.load(fh)["fading_memory"]["selected_rho"]
    out = {
        "design": {
            "C_grid": list(tuned.C_GRID),
            "inner_folds": tuned.N_INNER,
            "outer": "v1 splits, 5 x 5 participant-grouped",
            "scaling": "per-feature standardization fitted on training participants",
            "input": "v1 preprocessing (per-epoch z-scored)",
            "headline_rho": args.rho,
        },
        "granularities": {},
    }
    for name in ("three_condition", "four_category"):
        z = np.load(os.path.join(args.cache, f"{name}.npz"))
        code = np.load(os.path.join(args.cache, f"{name}_rho{args.rho}_s42_sh_spk_full.npy"))
        t = time.time()
        out["granularities"][name] = run(name, z["raw"], code, z["y"], z["groups"], args.jobs)
        gr = out["granularities"][name]
        print(f"[{time.strftime('%H:%M:%S')}] {name} ({time.time() - t:.0f} s)", flush=True)
        for k, v in gr["variants"].items():
            print(
                f"    {k:38s} {v['balanced_accuracy_percent']:.1f} {v['ci95_percent']} C {v['C_chosen']}",
                flush=True,
            )
        print("    tuned:", gr["contrasts_tuned"], flush=True)
        out["provenance"] = prov.provenance(v2._REPO, CODE_FILES)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
