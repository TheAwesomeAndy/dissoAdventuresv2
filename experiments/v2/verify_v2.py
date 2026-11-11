#!/usr/bin/env python3
"""Data-independent verification of the v2 machinery (no SHAPE data required).

These checks establish that the code does what V2_UPGRADE.md says it does. They do not
establish any empirical value; those come from run_v2_validation.py on authorized data.
"""
import os
import sys
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "chapter5Experiments"))
warnings.filterwarnings("ignore")

from arspinet_v2 import controls, dynamics, features, nuisance, reservoir  # noqa: E402
from arspinet_v2 import evaluation as ev  # noqa: E402

PASSED, FAILED = [], []


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def synthetic_cohort(n_subj=40, n_cls=3, T=256, n_ch=6, seed=0, effect=1.0):
    """ERP-like epochs: subject waveform + condition-specific late deflection + noise,
    z-scored per epoch and channel like the SHAPE preprocessing."""
    rng = np.random.RandomState(seed)
    t = np.arange(T) / 256.0
    late = np.exp(-0.5 * ((t - 0.5) / 0.12) ** 2)
    early = np.exp(-0.5 * ((t - 0.15) / 0.04) ** 2)
    topo = rng.normal(0, 1, (n_cls, n_ch))
    X, y, g = [], [], []
    for s in range(n_subj):
        subj = rng.normal(0, 1, n_ch)
        for c in range(n_cls):
            sig = (subj[None] * early[:, None] + effect * topo[c][None] * late[:, None]
                   + rng.normal(0, 1, (T, n_ch)))
            X.append((sig - sig.mean(0)) / sig.std(0))
            y.append(c)
            g.append(s)
    return np.asarray(X), np.asarray(y), np.asarray(g)


def main():
    raw, y, g = synthetic_cohort()

    print("Reservoir")
    from experiment_zero import LIFReservoir as V1Reservoir, bsc6_encode
    pc = reservoir.encode(raw[:6], rho=0.9, seed=42, shared=False)
    ref = np.stack([[bsc6_encode(V1Reservoir(seed=42 + ch * 17).forward(raw[i, :, ch]))
                     for ch in range(raw.shape[2])] for i in range(6)])
    check("batched per-electrode code equals the v1 operator exactly", np.array_equal(pc, ref.astype(np.float32)))
    sh = reservoir.encode(raw[:6], rho=0.9, seed=42, shared=True)
    ref = np.stack([[bsc6_encode(V1Reservoir(seed=42).forward(raw[i, :, ch]))
                     for ch in range(raw.shape[2])] for i in range(6)])
    check("batched shared code equals the v1 operator exactly", np.array_equal(sh, ref.astype(np.float32)))
    check("rho = 0 removes the recurrent matrix", reservoir.LIFReservoir(rho=0.0).w_rec is None)
    r5 = reservoir.LIFReservoir(rho=0.5)
    check("rho sets the spectral radius", abs(np.abs(np.linalg.eigvals(r5.w_rec)).max() - 0.5) < 1e-9)
    check("rho rescales one fixed weight draw",
          np.allclose(reservoir.LIFReservoir(rho=0.9).w_rec * (0.5 / 0.9), r5.w_rec))
    lin = reservoir.encode(raw[:4], rho=0.0, spiking=False)
    check("non-spiking control is non-negative and not integer-valued",
          lin.min() >= 0 and not np.allclose(lin, np.round(lin)))
    chunked = reservoir.encode(raw[:6], rho=0.3, chunk=7)
    check("chunking does not change the code", np.array_equal(chunked, reservoir.encode(raw[:6], rho=0.3)))

    print("Fading memory")
    x = raw[:20].transpose(0, 2, 1).reshape(-1, raw.shape[1])
    rep0, rep9 = dynamics.fading_memory_report(x, 0.0), dynamics.fading_memory_report(x, 0.9)
    check("no recurrence passes the fading-memory rule", rep0["passes"],
          f"esp {rep0['esp_converged_fraction']:.2f}, r {rep0['reliability_median_r']:.3f}")
    check("the v1 gain fails the fading-memory rule", not rep9["passes"],
          f"esp {rep9['esp_converged_fraction']:.2f}, r {rep9['reliability_median_r']:.3f}, "
          f"autonomous {rep9['autonomous_rate_after_input_off']:.3f}")
    check("selection returns the largest passing gain", dynamics.select_rho(
        [{"rho": 0.0, "passes": True}, {"rho": 0.1, "passes": True}, {"rho": 0.2, "passes": False}]) == 0.1)
    check("selection returns None when nothing passes", dynamics.select_rho([{"rho": 0.3, "passes": False}]) is None)

    print("Features")
    a6 = features.amplitude_bins(raw, 6)
    check("amplitude bins use the BSC6 bin edges",
          np.allclose(a6[3, 2, 4], raw[3, 4 * 42:5 * 42, 2].mean()) and a6.shape == (len(raw), raw.shape[2], 6))
    check("windows are sliced in true milliseconds",
          features.ms_window(raw, 0, 250).shape[1] == 64 and features.ms_window(raw, 250, 500).shape[1] == 64
          and np.array_equal(features.ms_window(raw, 250, 500), raw[:, 64:128]))
    srp = features.make_srp(1536)
    check("projection is data-independent and shared across electrodes",
          np.allclose(features.project(sh, srp)[:, :64], np.asarray(srp.transform(sh[:, 0].astype(float))))
          and np.allclose(features.project(sh * 0, srp), 0))

    print("Evaluation")
    splits = ev.make_splits(y, g)
    check("no participant is in both train and test", all(not np.intersect1d(g[tr], g[te]).size for _, tr, te in splits))
    check("every repeat predicts every observation once",
          all(sorted(np.concatenate([te for r, _, te in splits if r == k]).tolist()) == list(range(len(y)))
              for k in range(ev.N_REPEATS)))
    code = reservoir.encode(raw, rho=0.0)
    X = features.project(code, srp)
    row, direct = ev.evaluate(X, y, g), ev.evaluate(X, y, g, reduce=False)
    agree = (row["pred"] == direct["pred"]).mean()
    check("row-space fit is the same model as the direct fit", agree >= 0.97, f"prediction agreement {agree:.3f}")
    check("signal is decodable above chance", row["balanced_accuracy"] > 0.5, f"BA {row['balanced_accuracy']:.3f}")
    w = ev._weights(len(np.unique(g)), 50, 3)
    fast = ev._boot_ba(row, w)
    slow = []
    for b in range(50):
        idx = np.concatenate([np.flatnonzero(g == u) for u, k in zip(np.unique(g), w[b]) for _ in range(int(k))])
        slow.append(np.mean([(row["pred"][:, idx][:, y[idx] == c] == c).mean() for c in np.unique(y)]))
    check("vectorized participant bootstrap equals explicit resampling", np.allclose(fast, slow))
    ci = ev.bootstrap_ci(row)
    check("bootstrap interval brackets the estimate", ci[0] <= row["balanced_accuracy"] <= ci[1])
    d = ev.bootstrap_difference_ci(row, row)
    check("paired difference of a result with itself is zero", d == [0.0, 0.0])

    print("Permutation test")
    p_sig = ev.permutation_test(X, y, g, n_perm=99)
    check("signal reaches the permutation floor", p_sig["p_value"] == p_sig["p_value_floor"], f"p {p_sig['p_value']}")
    rawn, yn, gn = synthetic_cohort(effect=0.0, seed=5)
    p_null = ev.permutation_test(features.project(reservoir.encode(rawn, rho=0.0), srp), yn, gn, n_perm=99)
    check("no condition effect gives a non-significant p-value", p_null["p_value"] > 0.05, f"p {p_null['p_value']}")
    rng = np.random.RandomState(0)
    gi = [np.flatnonzero(g == u) for u in np.unique(g)]
    yp = ev._permute_within(y, gi, rng)
    check("permutation keeps each participant's label set",
          all(sorted(yp[i]) == sorted(y[i]) for i in gi) and not np.array_equal(yp, y))

    print("Destruction controls")
    col = controls.collapse(code)
    b = col.reshape(len(code), code.shape[1], 6, -1)
    check("collapse removes all between-bin variation and keeps the mean",
          np.allclose(b, b[:, :, :1]) and np.allclose(b[:, :, 0], code.reshape(b.shape).mean(axis=2), atol=1e-5))
    shf = controls.shuffle_bins(code, np.random.RandomState(1))
    check("bin shuffle preserves every bin block", np.allclose(
        np.sort(shf.reshape(b.shape).sum(axis=3), axis=2), np.sort(code.reshape(b.shape).sum(axis=3), axis=2)))
    fx = controls.fixed_bin_permutation(code, np.random.RandomState(2))
    r_fx = ev.evaluate(fx.reshape(len(y), -1), y, g, n_repeats=1)
    r_in = ev.evaluate(code.reshape(len(y), -1), y, g, n_repeats=1)
    check("a fixed bin permutation leaves the unprojected fit unchanged",
          (r_fx["pred"] == r_in["pred"]).mean() >= 0.97)
    el = controls.shuffle_electrodes(code, np.random.RandomState(3))
    check("electrode shuffle preserves each observation's blocks",
          np.allclose(np.sort(el.sum(axis=2), axis=1), np.sort(code.sum(axis=2), axis=1)))
    r_to = ev.evaluate(X, y, g, X_test=features.project(col, srp), n_repeats=1)
    check("test-only mode scores a model trained on intact features", r_to["pred"].shape == (1, len(y)))

    print("Variance components")
    noise = np.random.RandomState(4).standard_normal((len(y), 300))
    vc = nuisance.variance_components(noise, y, g, n_perm=200)
    check("noise has no subject or condition component",
          abs(vc["fraction_subject"]) < 0.03 and abs(vc["fraction_condition"]) < 0.03 and vc["subject_p_value"] > 0.01,
          f"subject {vc['fraction_subject']:+.3f}, condition {vc['fraction_condition']:+.3f}")
    check("the uncalibrated v1-style ratio is large on noise", vc["uncalibrated_v1_style_ratio"] > 5,
          f"ratio {vc['uncalibrated_v1_style_ratio']:.1f}")
    planted = noise + np.random.RandomState(6).standard_normal((len(np.unique(g)), 300))[g]
    vp = nuisance.variance_components(planted, y, g, n_perm=200)
    check("a planted subject effect of equal variance is recovered",
          0.4 < vp["fraction_subject"] < 0.6 and vp["subject_p_value"] == 1 / 201,
          f"subject {vp['fraction_subject']:.3f}")

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("FAILED: " + "; ".join(FAILED))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
