#!/usr/bin/env python3
"""Does random projection still beat PCA when PCA gets a fair chance?

v2 found SRP-64 ahead of fold-local PCA-64 by 8.6 and 9.7 pp with a fixed readout
(standardized features, C = 1). The review asked for the control that could close
the gap: a tuned regularization strength, with whitening on and off.

Design, on the v2 headline reservoir code:

    compressor   SRP-64 (fixed, data-independent) or PCA-64 (basis pooled over
                 electrodes, fitted on the outer training participants only)
    scaling      standardized (each component scaled to unit variance on the training
                 rows, which whitens PCA) or centered only (PCA keeps its eigenvalue
                 spectrum, so the L2 penalty favours the leading components)
    C            fixed at 1 (the v2 default) or tuned by nested cross-validation

Nested selection: within each outer training set, 4-fold participant-grouped CV over
C in {1e-4, ..., 1e2}; the best inner balanced accuracy picks C, which is refitted on
the full outer training set. Outer splits are v1's.

Choosing the best PCA variant after seeing outer-test accuracy favours PCA. That bias
is deliberate: if SRP still wins, the result does not depend on how PCA was set up.

    python experiments/v2/run_v2_pca_tuned.py --cache ./v2_cache --jobs 4
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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

sys.path.insert(0, _HERE)
import run_v2_validation as v2  # noqa: E402
from arspinet_v2 import features  # noqa: E402
from arspinet_v2 import evaluation as ev  # noqa: E402

C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
N_INNER = 4
BASE_SEED = {"three_condition": 0, "four_category": 100}


def _design(Xtr, Xte, standardize):
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0) if standardize else np.ones(Xtr.shape[1])
    sd = np.where(sd > 0, sd, 1.0)
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    if Ztr.shape[1] > Ztr.shape[0]:
        u, s, vt = np.linalg.svd(Ztr, full_matrices=False)
        keep = s > s[0] * 1e-12
        return u[:, keep] * s[keep], Zte @ vt[keep].T
    return Ztr, Zte


def _fit_predict(Ztr, ytr, Zte, C):
    return LogisticRegression(C=C, max_iter=5000).fit(Ztr, ytr).predict(Zte)


def _pca_transform(code, tr, k):
    n, c, d = code.shape
    x = code[tr].reshape(-1, d).astype(np.float64)
    mu = x.mean(axis=0)
    w, v = np.linalg.eigh((x - mu).T @ (x - mu) / (len(x) - 1))
    basis = v[:, ::-1][:, :k]
    return ((code.reshape(n * c, d) - mu) @ basis).reshape(n, c * k)


def outer_fold(code, srp_X, y, g, r, tr, te, seed):
    """All eight variants for one outer fold. Returns predictions and chosen C."""
    X = {"srp64": srp_X, "pca64": _pca_transform(code, tr, features.N_SRP)}
    out = {}
    for comp, Xc in X.items():
        for standardize in (True, False):
            Ztr, Zte = _design(Xc[tr], Xc[te], standardize)
            key = f"{comp}_{'standardized' if standardize else 'centered'}"
            out[f"{key}_C1"] = (_fit_predict(Ztr, y[tr], Zte, 1.0), 1.0)
            inner = StratifiedGroupKFold(N_INNER, shuffle=True, random_state=seed + 1000 + r)
            scores = np.zeros(len(C_GRID))
            for itr, ite in inner.split(np.zeros(len(tr)), y[tr], g[tr]):
                a, b = tr[itr], tr[ite]
                # PCA basis refitted on the inner training participants
                Xi = _pca_transform(code, a, features.N_SRP) if comp == "pca64" else Xc
                Zi_tr, Zi_te = _design(Xi[a], Xi[b], standardize)
                for j, C in enumerate(C_GRID):
                    scores[j] += ev.balanced_accuracy(y[b], _fit_predict(Zi_tr, y[a], Zi_te, C))
            C_best = C_GRID[int(np.argmax(scores))]
            out[f"{key}_Ctuned"] = (_fit_predict(Ztr, y[tr], Zte, C_best), C_best)
    return r, te, out


def run(name, code, y, g, jobs):
    srp_X = features.project(code, features.make_srp(code.shape[2]))
    splits = ev.make_splits(y, g, base_seed=BASE_SEED[name])
    res = Parallel(n_jobs=jobs)(delayed(outer_fold)(code, srp_X, y, g, r, tr, te, BASE_SEED[name])
                                for r, tr, te in splits)
    preds, chosen = {}, {}
    for r, te, out in res:
        for k, (p, C) in out.items():
            preds.setdefault(k, np.full((ev.N_REPEATS, len(y)), -1, dtype=int))[r, te] = p
            chosen.setdefault(k, []).append(C)
    results = {k: {"balanced_accuracy": float(np.mean([ev.balanced_accuracy(y, q) for q in p])),
                   "per_repeat": [ev.balanced_accuracy(y, q) for q in p], "pred": p, "y": y, "groups": g}
               for k, p in preds.items()}
    pct = lambda v: round(100 * float(v), 4)  # noqa: E731
    table = {}
    for k, r in results.items():
        ci = ev.bootstrap_ci(r)
        Cs = np.asarray(chosen[k])
        table[k] = {"balanced_accuracy_percent": pct(r["balanced_accuracy"]), "ci95_percent": [pct(ci[0]), pct(ci[1])],
                    "C_chosen": {str(c): int((Cs == c).sum()) for c in np.unique(Cs)}}
    best_pca = max((k for k in results if k.startswith("pca")), key=lambda k: results[k]["balanced_accuracy"])
    best_srp = max((k for k in results if k.startswith("srp")), key=lambda k: results[k]["balanced_accuracy"])
    contrast = lambda a, b: {"difference_pp": pct(results[a]["balanced_accuracy"] - results[b]["balanced_accuracy"]),  # noqa: E731
                             "ci95_pp": [pct(v) for v in ev.bootstrap_difference_ci(results[a], results[b])]}
    return {"variants": table, "best_pca_variant": best_pca, "best_srp_variant": best_srp,
            "contrasts": {"srp_default_minus_best_pca": contrast("srp64_standardized_C1", best_pca),
                          "best_srp_minus_best_pca": contrast(best_srp, best_pca),
                          "srp_default_minus_pca_default": contrast("srp64_standardized_C1", "pca64_standardized_C1")}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="directory holding the v2 headline codes")
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--out", default=os.path.join(_HERE, "v2_pca_tuned.json"))
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    rho = args.rho if args.rho is not None else json.load(open(os.path.join(_HERE, "v2_results.json")))["fading_memory"]["selected_rho"]
    out = {"design": {"C_grid": list(C_GRID), "inner_folds": N_INNER, "outer": "v1 splits, 5 x 5 participant-grouped",
                      "code": f"v2 headline, shared reservoir, rho={rho}, seed 42", "k": features.N_SRP},
           "granularities": {}}
    for name in ("three_condition", "four_category"):
        z = np.load(os.path.join(args.cache, f"{name}.npz"))
        code = np.load(os.path.join(args.cache, f"{name}_rho{rho}_s42_sh_spk_full.npy"))
        t = time.time()
        out["granularities"][name] = run(name, code, z["y"], z["groups"], args.jobs)
        g = out["granularities"][name]
        print(f"[{time.strftime('%H:%M:%S')}] {name} ({time.time() - t:.0f} s)", flush=True)
        for k, v in g["variants"].items():
            print(f"    {k:28s} {v['balanced_accuracy_percent']:.1f} {v['ci95_percent']} C {v['C_chosen']}", flush=True)
        print("    ", g["contrasts"], flush=True)
        out["provenance"] = {"code_sha256": {f: v2._sha(os.path.join(v2._REPO, f)) for f in (
            "experiments/v2/run_v2_pca_tuned.py", "arspinet_v2/features.py", "arspinet_v2/evaluation.py")},
            "git_commit": v2._git("rev-parse", "HEAD")}
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
