#!/usr/bin/env python3
"""ARSPI-Net v2 with reservoir input that is not z-scored per epoch.

v1 and the first v2 run z-score each epoch and channel over the whole post-stimulus
second. That writes late activity into early samples: the scalp pattern in the first
31 ms decodes condition at 54% (chance 33%), and at chance without z-scoring
(v2_normalization_check.json). This runner repeats the v2 reservoir analysis on two
inputs that never use a later sample to transform an earlier one:

    baseline_z    each epoch and channel divided by the standard deviation of its own
                  pre-stimulus interval (the delivered files are already baseline-
                  corrected, so the baseline mean is zero)
    global_scale  microvolts divided by one constant for the whole study: the pooled
                  post-stimulus SD of the three-condition cohort. A units choice; it
                  uses no labels and no per-observation information

For each input: the label-free fading-memory rule picks the recurrent gain, then
held-out accuracy for the reservoir at that gain, recurrence off and v1's gain; six-
bin amplitudes and raw ERP-16 on the same input; ten reservoir weight draws; windows
in milliseconds including 0-62 ms, where no picture-content response should exist.
The v1-style ``epoch_z`` input is included for the window comparison.

    python experiments/v2/run_v2_baseline_input.py --data3 ./batch_data --data4 ./categories \
        --cache ./v2_cache --out experiments/v2/v2_baseline_input.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import data as shape_data  # noqa: E402
import data_baseline  # noqa: E402
import run_v2_validation as v2  # noqa: E402
from arspinet_v2 import features, reservoir  # noqa: E402
from arspinet_v2 import evaluation as ev  # noqa: E402

WINDOWS_MS = {"0-62ms": (0, 62.5), **features.WINDOWS_MS}
SEEDS = v2.RESERVOIR_SEEDS


def inputs_for(name, args, scale):
    raw_uv, bsd, y, g = data_baseline.load_baseline_referenced(
        name, args.data3 if name == "three_condition" else args.data4, args.cache)
    rz, yz, gz = shape_data.load(args.data3, args.data4, args.cache)[name]
    if not (np.array_equal(y, yz) and np.array_equal(g, gz)):
        raise RuntimeError("cohort order differs from the v1 loader")
    return {"baseline_z": raw_uv / bsd[:, None, :], "global_scale": raw_uv / scale, "epoch_z": rz}, y, g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data3", required=True)
    ap.add_argument("--data4", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=os.path.join(_HERE, "v2_baseline_input.json"))
    args = ap.parse_args()

    raw3_uv, _, _, _ = data_baseline.load_baseline_referenced("three_condition", args.data3, args.cache)
    scale = float(raw3_uv.std())
    out = {"design": {"inputs": ["baseline_z", "global_scale", "epoch_z (v1, windows only)"],
                      "global_scale_uv": scale, "rho_grid": list(v2.RHO_GRID), "reservoir_seeds": list(SEEDS),
                      "fading_memory_rule": v2.dynamics.RULE},
           "fading_memory": {}, "granularities": {}}
    inp3, _, _ = inputs_for("three_condition", args, scale)
    rho = {}
    for k in ("baseline_z", "global_scale"):
        fm = v2.fading_memory(inp3[k])
        out["fading_memory"][k] = fm
        rho[k] = fm["selected_rho"]
        v2.log(f"{k}: selected rho {rho[k]}  input SD {inp3[k].std():.2f}  "
               f"driven rate at selected {next(s['driven_population_rate'] for s in fm['sweep_seed42'] if s['rho'] == rho[k]):.3f}")
    rho["epoch_z"] = json.load(open(os.path.join(_HERE, "v2_results.json")))["fading_memory"]["selected_rho"]
    out["selected_rho"] = rho
    del inp3

    for name in ("three_condition", "four_category"):
        inp, y, g = inputs_for(name, args, scale)
        base = v2.BASE_SEED[name]
        E = lambda X: ev.evaluate(X, y, g, base_seed=base)  # noqa: E731
        srp = features.make_srp(6 * reservoir.N_RES)
        G = {}
        for k in ("baseline_z", "global_scale"):
            raw = inp[k]
            enc = v2.Encoder(f"{name}_{k}", raw, args.cache)
            r_sel = rho[k]
            ev_ = {"reservoir_selected_rho": E(features.project(enc.code(r_sel), srp)),
                   "reservoir_no_recurrence": E(features.project(enc.code(0.0), srp)),
                   "reservoir_v1_gain": E(features.project(enc.code(reservoir.V1_RHO), srp)),
                   "amplitude6_matched_bins": E(features.amplitude_bins(raw, 6).reshape(len(y), -1)),
                   "raw_erp16": E(features.amplitude_bins(raw, 16).reshape(len(y), -1))}
            rec = {"selected_rho": r_sel, "representations": {kk: v2.record(v) for kk, v in ev_.items()}}
            rec["contrasts"] = {
                "reservoir_minus_amplitude6": v2.contrast(ev_["reservoir_selected_rho"], ev_["amplitude6_matched_bins"]),
                "reservoir_minus_no_recurrence": v2.contrast(ev_["reservoir_selected_rho"], ev_["reservoir_no_recurrence"]),
                "reservoir_minus_v1_gain": v2.contrast(ev_["reservoir_selected_rho"], ev_["reservoir_v1_gain"]),
                "raw_erp16_minus_reservoir": v2.contrast(ev_["raw_erp16"], ev_["reservoir_selected_rho"]),
            }
            for kk, v in rec["representations"].items():
                v2.log(f"{name} {k:12s} {kk:26s} {v['balanced_accuracy_percent']:.1f} {v['ci95_percent']}")
            vals = [v2.pct(E(features.project(enc.code(r_sel, seed=s), srp))["balanced_accuracy"]) for s in SEEDS]
            rec["reservoir_seeds"] = {"seeds": list(SEEDS), "percent": vals, "mean": round(float(np.mean(vals)), 4),
                                      "sd": round(float(np.std(vals, ddof=1)), 4), "min": min(vals), "max": max(vals)}
            v2.log(f"{name} {k:12s} seeds mean {rec['reservoir_seeds']['mean']:.1f} sd {rec['reservoir_seeds']['sd']:.2f}")
            G[k] = rec
        win = {}
        for k in ("baseline_z", "global_scale", "epoch_z"):
            raw = inp[k]
            enc = v2.Encoder(f"{name}_{k}" if k != "epoch_z" else name, raw, args.cache)
            win[k] = {}
            for w, (lo, hi) in WINDOWS_MS.items():
                xw = features.ms_window(raw, lo, hi)
                win[k][w] = {
                    "reservoir": v2.record(E(features.project(enc.code(rho[k], tag=w, raw=xw), srp))),
                    "amplitude6": v2.record(E(features.amplitude_bins(xw, 6).reshape(len(y), -1))),
                }
            v2.log(f"{name} windows {k:12s} " + " ".join(
                f"{w}:{v['reservoir']['balanced_accuracy_percent']:.1f}/{v['amplitude6']['balanced_accuracy_percent']:.1f}"
                for w, v in win[k].items()))
        G["windows_ms"] = win
        G["chance_percent"] = round(100.0 / len(np.unique(y)), 4)
        out["granularities"][name] = G
        out["provenance"] = v2.provenance(args)
        for f in ("experiments/v2/run_v2_baseline_input.py", "experiments/v2/data_baseline.py"):
            out["provenance"]["code_sha256"][f] = v2._sha(os.path.join(v2._REPO, f))
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
            fh.write("\n")
    v2.log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
