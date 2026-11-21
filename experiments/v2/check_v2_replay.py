#!/usr/bin/env python3
"""Replay part of the v2 record on the authorized data and compare exactly.

``v2_results.json`` took about 2 h 40 min to produce. This script recomputes a subset in
a few minutes and checks that the values equal the committed record. It answers two
questions the record cannot answer about itself: is the run deterministic, and does the
code in the working tree still reproduce it?

Replayed, per granularity, with the runner's own functions and seeds:

    fading-memory sweep (three-condition input) and the selected gain
    v1 reproduction with the v1 estimator (direct fit)
    v2 headline, recurrence off, v1 gain, raw ERP-16, six-bin amplitudes,
    LPP mean amplitude, conventional features, each with its bootstrap interval

Not replayed: seed sweeps, windows, destruction controls, compressors, variance
components, permutation test.

    python experiments/v2/check_v2_replay.py --data3 ./batch_data --data4 ./categories --cache ./v2_cache

Exit status 0 means every replayed value equals the record to the recorded precision.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from arspinet_v2 import evaluation as ev  # noqa: E402
from arspinet_v2 import features, reservoir  # noqa: E402

import data as shape_data  # noqa: E402
import run_v2_validation as v2  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data3", required=True)
    ap.add_argument("--data4", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--record", default=os.path.join(_HERE, "v2_results.json"))
    args = ap.parse_args()
    with open(args.record) as fh:
        rec = json.load(fh)
    cohorts = shape_data.load(args.data3, args.data4, args.cache)
    mismatches: list[str] = []
    n_values = 0

    def compare(label: str, got, want) -> None:
        nonlocal n_values
        n_values += 1
        ok = got == want
        print(f"  [{'same' if ok else 'DIFF'}] {label}")
        if not ok:
            mismatches.append(f"{label}: replay {got} != record {want}")

    v2.log("fading-memory sweep")
    fm = v2.fading_memory(cohorts["three_condition"][0])
    compare("fading_memory.selected_rho", fm["selected_rho"], rec["fading_memory"]["selected_rho"])
    compare("fading_memory.sweep_seed42", fm["sweep_seed42"], rec["fading_memory"]["sweep_seed42"])
    compare("fading_memory.across_reservoir_seeds", fm["across_reservoir_seeds"],
            rec["fading_memory"]["across_reservoir_seeds"])
    rho_star = rec["fading_memory"]["selected_rho"]

    for name, (raw, y, groups) in cohorts.items():
        v2.log(name)
        want = rec["granularities"][name]
        base = v2.BASE_SEED[name]
        enc = v2.Encoder(name, raw, args.cache)
        srp = features.make_srp(6 * reservoir.N_RES)
        E = lambda X, **kw: v2.record(ev.evaluate(X, y, groups, base_seed=base, **kw))  # noqa: E731, B023
        x_v1 = features.project(enc.code(reservoir.V1_RHO, shared=False), srp)
        compare(f"{name}.v1_reproduction.v1_estimator_direct_fit", E(x_v1, reduce=False),
                want["v1_reproduction"]["v1_estimator_direct_fit"])
        reps = want["representations"]
        compare(f"{name}.v2_headline_bsc6_srp64", E(features.project(enc.code(rho_star), srp)),
                reps["v2_headline_bsc6_srp64"])
        compare(f"{name}.bsc6_srp64_no_recurrence", E(features.project(enc.code(0.0), srp)),
                reps["bsc6_srp64_no_recurrence"])
        compare(f"{name}.bsc6_srp64_v1_gain_shared", E(features.project(enc.code(reservoir.V1_RHO), srp)),
                reps["bsc6_srp64_v1_gain_shared"])
        compare(f"{name}.raw_erp16", E(shape_data.v1.raw_erp_bins(raw, n_bins=16)), reps["raw_erp16"])
        compare(f"{name}.amplitude6_matched_bins", E(features.amplitude_bins(raw, 6).reshape(len(y), -1)),
                reps["amplitude6_matched_bins"])
        compare(f"{name}.lpp_300_1000ms_per_electrode", E(features.lpp_mean_amplitude(raw)),
                reps["lpp_300_1000ms_per_electrode"])
        compare(f"{name}.conventional", E(shape_data.v1.conventional_features(raw)), reps["conventional"])

    print(f"\n{n_values - len(mismatches)} of {n_values} replayed values equal the record")
    for m in mismatches:
        print("  - " + m)
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
