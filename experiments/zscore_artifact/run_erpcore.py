#!/usr/bin/env python3
"""Per-epoch z-scoring artefact on public data (ERP CORE, Kappenman et al. 2021).

Claim under test: z-scoring each epoch and channel over the epoch, a common EEG
machine-learning preprocessing step, makes every sample a function of the whole epoch.
Late condition differences are written into early samples, including the pre-stimulus
interval, so time-resolved decoding becomes above chance before anything has happened.

Data: the ERP CORE epoched, ICA-corrected, artefact-flagged single trials
(``*_epoch_interp_ar.set/.fdt``), 256 Hz, -200 to 800 ms, baseline-corrected, 28 scalp
channels. Trials flagged for artefacts are dropped. Paradigms and contrasts:

    N170   faces vs cars                (early effect, about 170 ms)
    P3     rare vs frequent targets     (late effect, about 300 to 600 ms)

Normalizations, each applied per trial and per channel before decoding:

    none           the delivered baseline-corrected microvolts
    baseline_z     divided by the SD of the trial's own pre-stimulus interval
    epoch_z_full   z-scored over the whole epoch, -200 to 800 ms
    epoch_z_post   z-scored with the mean and SD of the post-stimulus part only (the
                   SHAPE v1 convention), applied to the whole epoch

Only ``epoch_z_*`` use post-stimulus samples to transform earlier ones.

Regimes:

    within   single trials, each participant separately: 16 ms bins, 28-channel
             pattern, fold-local standardization + shrinkage LDA, stratified 5-fold
             repeated twice. Group statistic: participant mean of balanced accuracy.
    across   one average per participant and condition, participant-grouped 5-fold
             repeated 5 times, standardization + L2 logistic (C = 1).

Primary test, fixed in advance: mean accuracy over the pre-stimulus window (-200 to
0 ms), one value per participant, one-sample t-test against 50% (within regime).

    python experiments/zscore_artifact/run_erpcore.py --data ./erpcore --out erpcore_results.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.io as sio
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PARADIGMS = {"N170": (("faces, correct",), ("cars, correct",)),
             "P3": (("rare",), ("frequent",))}
N_SCALP = 28
BIN = 4            # samples per decoding bin: 15.6 ms at 256 Hz
NORMS = ("none", "baseline_z", "epoch_z_full", "epoch_z_post")
PRE = (-200.0, 0.0)


def load_participant(path_set):
    m = sio.loadmat(path_set, squeeze_me=True, struct_as_record=False)["EEG"]
    fdt = os.path.join(os.path.dirname(path_set), os.path.basename(path_set)[:-4] + ".fdt")
    x = np.fromfile(fdt, dtype="<f4").reshape(m.trials, m.pnts, m.nbchan)[:, :, :N_SCALP].astype(np.float64)
    times = 1000.0 * (m.xmin + np.arange(m.pnts) / m.srate)
    labels = [str(b).strip().lower() for b in [d.description for d in np.atleast_1d(m.EVENTLIST.bdf)]]
    bini = []
    for e in np.atleast_1d(m.epoch):
        b = np.atleast_1d(e.eventbini)[0]
        bini.append(int(np.atleast_1d(b)[0]))
    rej = np.asarray(m.reject.rejmanual, dtype=bool) if np.size(m.reject.rejmanual) else np.zeros(m.trials, bool)
    return x, times, np.asarray(bini), rej, labels


def pick(labels, keys):
    hits = [i + 1 for i, lab in enumerate(labels) if any(lab.startswith(k) for k in keys)]
    if len(hits) != 1:
        raise ValueError(f"bin match for {keys} in {labels}: {hits}")
    return hits[0]


def normalize(x, times, how):
    pre, post = times < 0, times >= 0
    if how == "none":
        return x
    if how == "baseline_z":
        return x / x[:, pre].std(axis=1, keepdims=True)
    if how == "epoch_z_full":
        return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
    if how == "epoch_z_post":
        return (x - x[:, post].mean(axis=1, keepdims=True)) / x[:, post].std(axis=1, keepdims=True)
    raise ValueError(how)


def binned(x):
    n, T, c = x.shape
    nb = T // BIN
    return x[:, :nb * BIN].reshape(n, nb, BIN, c).mean(axis=2)


def bal_acc(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def within(xb, y, seed):
    nb = xb.shape[1]
    acc = np.zeros(nb)
    for r in range(2):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed + r).split(xb[:, 0], y):
            for b in range(nb):
                clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
                clf.fit(xb[tr, b], y[tr])
                acc[b] += bal_acc(y[te], clf.predict(xb[te, b])) / 10.0
    return acc


def across(avg, y, g, seed):
    nb = avg.shape[1]
    acc = np.zeros(nb)
    for r in range(5):
        pred = np.full((nb, len(y)), -1)
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed + r).split(avg[:, 0], y, g):
            for b in range(nb):
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
                pred[b, te] = clf.fit(avg[tr, b], y[tr]).predict(avg[te, b])
        acc += np.array([bal_acc(y, pred[b]) for b in range(nb)]) / 5.0
    return acc


def summarize(per_participant, centers):
    A = np.asarray(per_participant)          # participants x bins
    pre = (centers >= PRE[0]) & (centers < PRE[1])
    early = (centers >= 0) & (centers < 62.5)
    pre_vals = A[:, pre].mean(axis=1)
    t, p = stats.ttest_1samp(pre_vals, 0.5)
    w = stats.wilcoxon(pre_vals - 0.5)
    mean = A.mean(axis=0)
    sem = A.std(axis=0, ddof=1) / np.sqrt(len(A))
    return {
        "time_course": {"bin_center_ms": [round(float(c), 2) for c in centers],
                        "mean_percent": [round(100 * float(v), 2) for v in mean],
                        "sem_percent": [round(100 * float(v), 2) for v in sem]},
        "prestimulus_mean_percent": round(100 * float(pre_vals.mean()), 2),
        "prestimulus_ci95_percent": [round(100 * float(pre_vals.mean() - 1.96 * pre_vals.std(ddof=1) / np.sqrt(len(pre_vals))), 2),
                                     round(100 * float(pre_vals.mean() + 1.96 * pre_vals.std(ddof=1) / np.sqrt(len(pre_vals))), 2)],
        "prestimulus_t": round(float(t), 3), "prestimulus_p": float(p), "prestimulus_wilcoxon_p": float(w.pvalue),
        "prestimulus_participants_above_50": int((pre_vals > 0.5).sum()),
        "first_62ms_mean_percent": round(100 * float(A[:, early].mean()), 2),
        "peak_percent": round(100 * float(mean.max()), 2),
        "peak_ms": round(float(centers[int(mean.argmax())]), 2),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder with N170/<id>/<id>_N170_ar.set and P3/<id>/...")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = {"design": {"normalizations": list(NORMS), "bin_ms": 1000.0 * BIN / 256, "prestimulus_window_ms": list(PRE),
                      "primary_test": "within regime: participant mean accuracy over -200 to 0 ms, one-sample t-test vs 50%",
                      "source": "ERP CORE (Kappenman et al. 2021, NeuroImage 225:117465), doi:10.18115/D5JW4R"},
           "paradigms": {}}
    for para, (ka, kb) in PARADIGMS.items():
        root = os.path.join(args.data, para)
        sids = sorted((d for d in os.listdir(root) if d.isdigit()), key=int)
        W = {n: [] for n in NORMS}
        avgs = {n: [] for n in NORMS}
        ys, gs, counts = [], [], []
        centers = None
        for sid in sids:
            x, times, bini, rej, labels = load_participant(os.path.join(root, sid, f"{sid}_{para}_ar.set"))
            a, b = pick(labels, ka), pick(labels, kb)
            keep = ~rej & np.isin(bini, (a, b))
            x, y = x[keep], (bini[keep] == b).astype(int)
            counts.append([int((y == 0).sum()), int((y == 1).sum())])
            tb = times[:(len(times) // BIN) * BIN].reshape(-1, BIN).mean(axis=1)
            centers = tb
            for n in NORMS:
                xb = binned(normalize(x, times, n))
                W[n].append(within(xb, y, seed=int(sid)))
                for c in (0, 1):
                    avgs[n].append(xb[y == c].mean(axis=0))
            ys += [0, 1]
            gs += [int(sid), int(sid)]
        y_, g_ = np.asarray(ys), np.asarray(gs)
        P = {"n_participants": len(sids), "contrast": [ka[0], kb[0]],
             "trials_per_class_median": [int(np.median([c[0] for c in counts])), int(np.median([c[1] for c in counts]))],
             "within": {}, "across": {}}
        for n in NORMS:
            P["within"][n] = summarize(W[n], centers)
            acc = across(np.asarray(avgs[n]), y_, g_, seed=0)
            pre = (centers >= PRE[0]) & (centers < PRE[1])
            P["across"][n] = {"time_course": {"bin_center_ms": [round(float(c), 2) for c in centers],
                                              "accuracy_percent": [round(100 * float(v), 2) for v in acc]},
                              "prestimulus_mean_percent": round(100 * float(acc[pre].mean()), 2),
                              "first_62ms_mean_percent": round(100 * float(acc[(centers >= 0) & (centers < 62.5)].mean()), 2),
                              "peak_percent": round(100 * float(acc.max()), 2)}
            w = P["within"][n]
            print(f"[{time.strftime('%H:%M:%S')}] {para} {n:13s} within: pre {w['prestimulus_mean_percent']:.1f} "
                  f"{w['prestimulus_ci95_percent']} t {w['prestimulus_t']:.2f} p {w['prestimulus_p']:.2g} "
                  f"({w['prestimulus_participants_above_50']}/{len(sids)} above 50) 0-62 {w['first_62ms_mean_percent']:.1f} "
                  f"peak {w['peak_percent']:.1f}@{w['peak_ms']:.0f} | across: pre {P['across'][n]['prestimulus_mean_percent']:.1f} "
                  f"0-62 {P['across'][n]['first_62ms_mean_percent']:.1f} peak {P['across'][n]['peak_percent']:.1f}", flush=True)
        out["paradigms"][para] = P
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
