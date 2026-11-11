#!/usr/bin/env python3
"""ARSPI-Net v2 validation runner (requires authorized SHAPE data).

    python experiments/v2/run_v2_validation.py \
        --data3 ./batch_data --data4 ./categories \
        --cache ./v2_cache --out experiments/v2/v2_results.json --n-perm 5000 --jobs 4

The permutation stage is independent of the others. It can be skipped with --n-perm 0
and added afterwards with --permutation-only.

Stages, per granularity:

  v1_reproduction   the committed v1 configuration, recomputed by the v2 code
  fading_memory     dynamics tests on the deployed reservoir across the recurrent gain
  rho_sweep         held-out accuracy across the recurrent gain (shared reservoir)
  representations   the v2 headline and its matched controls, with paired contrasts
  seeds             reservoir weight draws and projection draws
  windows           windows stated in true milliseconds
  time_resolved     scalp-pattern decoding at each latency, no reservoir
  destruction       controls in train-and-test and test-only modes
  compressors       SRP against fold-local PCA, with and without subject centering
  variance          calibrated subject and condition variance components
  permutation       within-participant permutation test with the headline estimator

The v2 headline recurrent gain is not tuned on accuracy. It is the largest value on the
grid that passes the label-free fading-memory rule in arspinet_v2/dynamics.py.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)

import data as shape_data  # noqa: E402
from arspinet_v2 import controls, dynamics, features, nuisance, reservoir  # noqa: E402
from arspinet_v2 import evaluation as ev  # noqa: E402

RHO_GRID = (0.0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
RESERVOIR_SEEDS = (42, 1, 2, 3, 4, 5, 6, 7, 8, 9)
SRP_SEEDS = tuple(range(10))
DYNAMICS_SAMPLE_SEED = 2026
DYNAMICS_N_SEQUENCES = 300
BASE_SEED = {"three_condition": 0, "four_category": 100}     # v1 values


def pct(v):
    return round(float(v) * 100.0, 4)


def record(result, seed_offset=0):
    ci = ev.bootstrap_ci(result, seed=ev.BOOTSTRAP_SEED + seed_offset)
    return {"balanced_accuracy_percent": pct(result["balanced_accuracy"]),
            "ci95_percent": [pct(ci[0]), pct(ci[1])],
            "per_repeat_percent": [pct(v) for v in result["per_repeat"]]}


def contrast(a, b):
    ci = ev.bootstrap_difference_ci(a, b)
    return {"difference_pp": pct(a["balanced_accuracy"] - b["balanced_accuracy"]),
            "ci95_pp": [pct(ci[0]), pct(ci[1])]}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Encoder:
    """Reservoir codes with an on-disk cache keyed by configuration."""

    def __init__(self, name, raw, cache):
        self.name, self.raw, self.cache = name, raw, cache

    def code(self, rho, seed=42, shared=True, spiking=True, tag="full", raw=None):
        key = f"{self.name}_rho{rho}_s{seed}_{'sh' if shared else 'pc'}_{'spk' if spiking else 'lin'}_{tag}"
        f = os.path.join(self.cache, key + ".npy") if self.cache else None
        if f and os.path.exists(f):
            return np.load(f)
        c = reservoir.encode(self.raw if raw is None else raw, rho=rho, seed=seed,
                             shared=shared, spiking=spiking)
        if f:
            os.makedirs(self.cache, exist_ok=True)
            np.save(f, c)
        return c


def fading_memory(raw):
    rng = np.random.RandomState(DYNAMICS_SAMPLE_SEED)
    i = rng.randint(0, len(raw), DYNAMICS_N_SEQUENCES)
    c = rng.randint(0, raw.shape[2], DYNAMICS_N_SEQUENCES)
    x = raw[i, :, c]
    sweep = [dynamics.fading_memory_report(x, rho) for rho in RHO_GRID]
    rho_star = dynamics.select_rho(sweep)
    across_seeds = {}
    for rho in sorted({rho_star, reservoir.V1_RHO} - {None}):
        reps = [dynamics.fading_memory_report(x, rho, seed=s) for s in RESERVOIR_SEEDS]
        across_seeds[str(rho)] = {
            "n_seeds": len(reps),
            "n_passing": int(sum(r["passes"] for r in reps)),
            "esp_converged_fraction_range": [min(r["esp_converged_fraction"] for r in reps),
                                             max(r["esp_converged_fraction"] for r in reps)],
            "reliability_median_r_range": [min(r["reliability_median_r"] for r in reps),
                                           max(r["reliability_median_r"] for r in reps)],
            "autonomous_rate_range": [min(r["autonomous_rate_after_input_off"] for r in reps),
                                      max(r["autonomous_rate_after_input_off"] for r in reps)],
        }
    return {"rule": dynamics.RULE, "sample": {"seed": DYNAMICS_SAMPLE_SEED,
            "n_sequences": DYNAMICS_N_SEQUENCES, "source": "three_condition input, label-free"},
            "sweep_seed42": sweep, "selected_rho": rho_star, "across_reservoir_seeds": across_seeds}


def _pca_basis(blocks, k):
    x = blocks.reshape(-1, blocks.shape[-1]).astype(np.float64)
    mu = x.mean(axis=0)
    cov = (x - mu).T @ (x - mu) / (len(x) - 1)
    w, v = np.linalg.eigh(cov)
    return mu, v[:, ::-1][:, :k]


def compressors(code, y, groups, base_seed, k=features.N_SRP):
    """SRP-64 against fold-local PCA-64. The PCA basis is fitted on training
    participants only, pooled over electrodes, with or without removing each training
    participant's own mean first. Features given to the readout are never
    subject-centered, so every row is an inductive protocol."""
    n, c, d = code.shape
    pred = {m: np.full((ev.N_REPEATS, n), -1, dtype=int)
            for m in ("pca64_fold_local", "pca64_fold_local_subject_centered_basis")}
    for r, tr, te in ev.make_splits(y, groups, base_seed=base_seed):
        btr = code[tr]
        mu, basis = _pca_basis(btr, k)
        centered = btr.astype(np.float64).copy()
        for gid in np.unique(groups[tr]):
            m = groups[tr] == gid
            centered[m] -= centered[m].mean(axis=0, keepdims=True)
        _, basis_c = _pca_basis(centered, k)
        for name, b in (("pca64_fold_local", basis),
                        ("pca64_fold_local_subject_centered_basis", basis_c)):
            X = ((code.reshape(n * c, d) - mu) @ b).reshape(n, c * k)
            pred[name][r, te] = ev.FoldDesign(X, tr, te).predict(y)
    out = {}
    for name, p in pred.items():
        per = [ev.balanced_accuracy(y, q) for q in p]
        out[name] = {"balanced_accuracy": float(np.mean(per)), "per_repeat": per,
                     "pred": p, "y": y, "groups": groups}
    out["none_full_1536_per_electrode"] = ev.evaluate(code.reshape(n, c * d), y, groups,
                                                      base_seed=base_seed)
    return out


def run_permutation(name, raw, y, groups, rho_star, cache, n_perm, jobs):
    Xh = features.project(Encoder(name, raw, cache).code(rho_star), features.make_srp(6 * reservoir.N_RES))
    out = {"v2_headline_bsc6_srp64": ev.permutation_test(Xh, y, groups, n_perm,
                                                         base_seed=BASE_SEED[name], n_jobs=jobs)}
    log(f"{name} permutation {out['v2_headline_bsc6_srp64']}")
    return out


def run_granularity(name, raw, y, groups, rho_star, cache, n_perm, jobs, n_var_perm):
    base = BASE_SEED[name]
    enc = Encoder(name, raw, cache)
    srp = features.make_srp(6 * reservoir.N_RES)
    res = {"n_participants": int(len(np.unique(groups))), "n_observations": int(len(y))}
    E = lambda X, **kw: ev.evaluate(X, y, groups, base_seed=base, **kw)  # noqa: E731

    # ---- v1 reproduction ---------------------------------------------------------------
    Xv1 = features.project(enc.code(reservoir.V1_RHO, shared=False), srp)
    v1_direct = E(Xv1, reduce=False)
    v1_row = E(Xv1)
    res["v1_reproduction"] = {
        "configuration": "per-electrode reservoirs (seed 42 + 17 * channel), rho 0.9, full epoch, SRP-64",
        "v1_estimator_direct_fit": record(v1_direct),
        "v2_estimator_row_space_fit": record(v1_row),
        "prediction_agreement_between_fits": round(float((v1_direct["pred"] == v1_row["pred"]).mean()), 6),
    }
    log(f"{name} v1 reproduction {res['v1_reproduction']['v1_estimator_direct_fit']['balanced_accuracy_percent']}")

    # ---- recurrent gain sweep, shared reservoir -----------------------------------------
    sweep_eval = {rho: E(features.project(enc.code(rho), srp)) for rho in RHO_GRID}
    res["rho_sweep_shared_reservoir"] = {str(rho): record(r) for rho, r in sweep_eval.items()}
    log(f"{name} rho sweep " + " ".join(f"{rho}:{pct(r['balanced_accuracy']):.1f}" for rho, r in sweep_eval.items()))

    # ---- representations and matched controls -------------------------------------------
    head_code = enc.code(rho_star)
    head = sweep_eval[rho_star]
    amp6 = features.amplitude_bins(raw, 6)
    evals = {
        "v2_headline_bsc6_srp64": head,
        "raw_erp16": E(shape_data.v1.raw_erp_bins(raw, n_bins=16)),
        "amplitude6_matched_bins": E(amp6.reshape(len(y), -1)),
        "nonspiking_bank_srp64": E(features.project(enc.code(0.0, spiking=False), srp)),
        "bsc6_srp64_no_recurrence": sweep_eval[0.0],
        "bsc6_srp64_v1_gain_shared": sweep_eval[reservoir.V1_RHO],
        "bsc6_srp64_v1_gain_per_electrode": v1_row,
        "conventional": E(shape_data.v1.conventional_features(raw)),
        "lpp_300_1000ms_per_electrode": E(features.lpp_mean_amplitude(raw)),
    }
    res["representations"] = {k: record(v) for k, v in evals.items()}
    res["feature_counts"] = {"v2_headline_bsc6_srp64": 34 * 64, "raw_erp16": 34 * 16,
                             "amplitude6_matched_bins": 34 * 6, "nonspiking_bank_srp64": 34 * 64,
                             "conventional": 34 * 8, "lpp_300_1000ms_per_electrode": 34}
    res["contrasts"] = {
        "headline_minus_amplitude6": contrast(head, evals["amplitude6_matched_bins"]),
        "headline_minus_nonspiking_bank": contrast(head, evals["nonspiking_bank_srp64"]),
        "headline_minus_no_recurrence": contrast(head, evals["bsc6_srp64_no_recurrence"]),
        "headline_minus_v1_gain_shared": contrast(head, evals["bsc6_srp64_v1_gain_shared"]),
        "headline_minus_v1_configuration": contrast(head, v1_row),
        "headline_minus_conventional": contrast(head, evals["conventional"]),
        "raw_erp16_minus_headline": contrast(evals["raw_erp16"], head),
        "headline_minus_lpp": contrast(head, evals["lpp_300_1000ms_per_electrode"]),
        "v1_gain_recurrence_effect_shared": contrast(evals["bsc6_srp64_v1_gain_shared"],
                                                     evals["bsc6_srp64_no_recurrence"]),
        "v1_gain_per_electrode_minus_shared": contrast(v1_row, evals["bsc6_srp64_v1_gain_shared"]),
    }
    for k, v in res["representations"].items():
        log(f"{name} {k:36s} {v['balanced_accuracy_percent']:.1f} {v['ci95_percent']}")

    # ---- seeds --------------------------------------------------------------------------
    seeds = {}
    for rho in (rho_star,):
        vals = [pct(E(features.project(enc.code(rho, seed=s), srp))["balanced_accuracy"])
                for s in RESERVOIR_SEEDS]
        seeds[f"reservoir_seeds_rho{rho}"] = {"seeds": list(RESERVOIR_SEEDS), "percent": vals,
                                              "mean": round(float(np.mean(vals)), 4),
                                              "sd": round(float(np.std(vals, ddof=1)), 4),
                                              "min": min(vals), "max": max(vals)}
    vals = [pct(E(features.project(head_code, features.make_srp(6 * reservoir.N_RES, seed=s)))["balanced_accuracy"])
            for s in SRP_SEEDS]
    seeds["projection_seeds_headline"] = {"seeds": list(SRP_SEEDS), "percent": vals,
                                          "mean": round(float(np.mean(vals)), 4),
                                          "sd": round(float(np.std(vals, ddof=1)), 4),
                                          "min": min(vals), "max": max(vals)}
    res["seeds"] = seeds
    log(f"{name} seeds done")

    # ---- windows in true milliseconds ---------------------------------------------------
    win = {}
    for wname, (lo, hi) in features.WINDOWS_MS.items():
        w = features.ms_window(raw, lo, hi)
        win[wname] = {
            "v2_headline_bsc6_srp64": pct(E(features.project(
                enc.code(rho_star, tag=wname, raw=w), srp))["balanced_accuracy"]),
            "amplitude6_matched_bins": pct(E(features.amplitude_bins(w, 6).reshape(len(y), -1))["balanced_accuracy"]),
        }
    res["windows_ms"] = win

    # ---- time-resolved decoding of the scalp pattern -------------------------------------
    n_tb = 32
    tr = []
    for b, pattern in enumerate(features.time_resolved_patterns(raw, n_tb)):
        r = E(pattern)
        ci = ev.bootstrap_ci(r)
        tr.append({"start_ms": round(1000.0 * b * (raw.shape[1] // n_tb) / features.FS, 1),
                   "end_ms": round(1000.0 * (b + 1) * (raw.shape[1] // n_tb) / features.FS, 1),
                   "balanced_accuracy_percent": pct(r["balanced_accuracy"]),
                   "ci95_percent": [pct(ci[0]), pct(ci[1])]})
    res["time_resolved_decoding"] = {
        "features": "34 electrode amplitudes in one 31.25 ms bin, no reservoir",
        "chance_percent": round(100.0 / len(np.unique(y)), 4), "bins": tr}
    log(f"{name} time-resolved peak {max(t['balanced_accuracy_percent'] for t in tr):.1f}")

    # ---- destruction controls -----------------------------------------------------------
    rng = np.random.RandomState(base + 101)
    destroyed = {
        "temporal_collapse": controls.collapse(head_code),
        "temporal_bin_shuffle_per_observation": controls.shuffle_bins(head_code, rng),
        "fixed_bin_permutation_sanity": controls.fixed_bin_permutation(head_code, rng),
        "electrode_shuffle_per_observation": controls.shuffle_electrodes(head_code, rng),
    }
    Xh = features.project(head_code, srp)
    dc = {}
    for cname, code_d in destroyed.items():
        Xd = features.project(code_d, srp)
        dc[cname] = {"train_and_test": contrast(E(Xd), head),
                     "test_only": contrast(E(Xh, X_test=Xd), head)}
        log(f"{name} control {cname:38s} both {dc[cname]['train_and_test']['difference_pp']:+.1f} "
            f"test-only {dc[cname]['test_only']['difference_pp']:+.1f}")
    res["destruction_controls"] = dc
    del destroyed

    # ---- compressors --------------------------------------------------------------------
    comp = compressors(head_code, y, groups, base)
    comp["srp64"] = head
    res["compressors"] = {k: record(v) for k, v in comp.items()}
    res["compressor_contrasts"] = {
        "srp64_minus_pca64_fold_local": contrast(head, comp["pca64_fold_local"]),
        "srp64_minus_pca64_subject_centered_basis": contrast(
            head, comp["pca64_fold_local_subject_centered_basis"]),
        "srp64_minus_no_compression": contrast(head, comp["none_full_1536_per_electrode"]),
    }
    log(f"{name} compressors " + " ".join(f"{k}:{v['balanced_accuracy_percent']:.1f}" for k, v in res["compressors"].items()))

    # ---- calibrated variance components -------------------------------------------------
    res["variance_components"] = {
        "raw_erp16": nuisance.variance_components(shape_data.v1.raw_erp_bins(raw, 16), y, groups, n_var_perm),
        "amplitude6_matched_bins": nuisance.variance_components(amp6.reshape(len(y), -1), y, groups, n_var_perm),
        "v2_headline_bsc6_srp64": nuisance.variance_components(Xh, y, groups, n_var_perm),
        "v1_configuration_bsc6_srp64": nuisance.variance_components(Xv1, y, groups, n_var_perm),
        "gaussian_noise_same_shape_as_headline": nuisance.variance_components(
            np.random.RandomState(base + 5).standard_normal(Xh.shape), y, groups, n_var_perm),
    }
    log(f"{name} variance components done")

    if n_perm:
        res["permutation"] = run_permutation(name, raw, y, groups, rho_star, cache, n_perm, jobs)
    return res


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args):
    try:
        return subprocess.run(["git", "-C", _REPO, *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def provenance(args):
    import scipy
    import sklearn
    code = ["arspinet_v2/reservoir.py", "arspinet_v2/dynamics.py", "arspinet_v2/features.py",
            "arspinet_v2/evaluation.py", "arspinet_v2/controls.py", "arspinet_v2/nuisance.py",
            "experiments/v2/run_v2_validation.py", "experiments/v2/data.py",
            "experiments/confirmatory/run_confirmatory_validation.py"]
    count = lambda p: sum(f.endswith(".txt") for _, _, fs in os.walk(p) for f in fs) if p else 0  # noqa: E731
    return {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git("rev-parse", "HEAD"),
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        "code_sha256": {c: _sha(os.path.join(_REPO, c)) for c in code},
        "input_file_counts": {"data3": count(args.data3), "data4": count(args.data4)},
        "note": "SHAPE data are restricted and are not distributed with the repository.",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data3", required=True)
    ap.add_argument("--data4", default=None)
    ap.add_argument("--cache", default=None, help="directory for preprocessed arrays and codes (outside the repo)")
    ap.add_argument("--out", default=os.path.join(_HERE, "v2_results.json"))
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--n-variance-perm", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--permutation-only", action="store_true",
                    help="add the permutation stage to an existing --out file (the stage is "
                         "independent of the others, so it can run as a separate process)")
    args = ap.parse_args()

    cohorts = shape_data.load(args.data3, args.data4, args.cache)
    if args.permutation_only:
        with open(args.out) as fh:
            results = json.load(fh)
        rho_star = results["fading_memory"]["selected_rho"]
        for name, (raw, y, groups) in cohorts.items():
            results["granularities"][name]["permutation"] = run_permutation(
                name, raw, y, groups, rho_star, args.cache, args.n_perm, args.jobs)
            results["provenance"]["permutation_stage"] = provenance(args)
            with open(args.out, "w") as fh:
                json.dump(results, fh, indent=2, sort_keys=True)
                fh.write("\n")
        log(f"updated {args.out}")
        return 0
    results = {
        "version": "2.0.0",
        "protocol": {
            "cv": "5 repeats x 5-fold participant-grouped stratified CV (v1 splits)",
            "readout": "per-feature standardization + L2 logistic regression (C=1), fitted on training participants only, in the row space of the training matrix",
            "compression": f"fixed data-independent SparseRandomProjection 1536->64, seed={features.SRP_SEED}, shared across electrodes",
            "reservoir": "one LIF reservoir shared by all electrodes: N=256, beta=0.05, theta=0.5; recurrent gain chosen by the label-free fading-memory rule",
            "intervals": f"participant-cluster percentile bootstrap, {ev.N_BOOTSTRAP} resamples, pooled repeated out-of-fold predictions",
            "rho_grid": list(RHO_GRID),
        },
    }
    log("fading-memory tests")
    results["fading_memory"] = fading_memory(cohorts["three_condition"][0])
    rho_star = results["fading_memory"]["selected_rho"]
    if rho_star is None:
        raise SystemExit("no recurrent gain on the grid passes the fading-memory rule")
    results["protocol"]["headline_rho"] = rho_star
    log(f"selected rho = {rho_star}")
    results["granularities"] = {}
    for name, (raw, y, groups) in cohorts.items():
        results["granularities"][name] = run_granularity(
            name, raw, y, groups, rho_star, args.cache, args.n_perm, args.jobs, args.n_variance_perm)
        results["provenance"] = provenance(args)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2, sort_keys=True)
            fh.write("\n")
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
