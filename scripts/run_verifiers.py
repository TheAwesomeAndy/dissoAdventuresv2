#!/usr/bin/env python3
"""Run the legacy verifier suites without touching the working tree.

Why this exists. Two of the v1 suites regenerate figures and one result file from
synthetic data and write them over tracked files:

    chapter4Experiments/verify_chapter4.py   13 PDFs in pictures/chLSMEmbeddings/
    chapter7Experiments/verify_chapter7.py    7 PDFs in pictures/chSynthesis/ and
                                              chapter7Experiments/chapter7_results/ch7_expB_results.npz

Run in place and then committed, they replace dissertation figures with synthetic
stand-ins. CI is unaffected because it runs in a throwaway checkout. This runner gives
the same isolation locally: it copies the tree (tracked files plus untracked,
non-ignored files, so uncommitted work is tested) to a temporary directory and runs
every suite there. The provenance validator needs git history, and is read-only, so it
runs in place.

    python scripts/run_verifiers.py               # all suites, 4 at a time
    python scripts/run_verifiers.py --only v2 ch5zero
    python scripts/run_verifiers.py --list

Exit status is non-zero if a suite fails or reports fewer checks than the recorded
baseline (a suite that silently loses checks is a regression too).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# id, argv, runs in place. Must stay in step with .github/workflows/verify.yml
# (tests/test_tooling.py checks that).
VERIFIERS: list[tuple[str, list[str], bool]] = [
    ("ch4", ["chapter4Experiments/verify_chapter4.py"], False),
    ("ch5", ["chapter5Experiments/verify_chapter5.py"], False),
    ("ch5zero", ["chapter5Experiments/verify_experiment_zero.py"], False),
    ("ch5repro", ["chapter5Experiments/verify_reproduce_chapter5.py"], False),
    ("ch5base", ["chapter5Experiments/verify_baselines.py"], False),
    ("ch6", ["chapter6Experiments/verify_chapter6.py"], False),
    ("ch6repro", ["chapter6Experiments/verify_reproduce_chapter6.py"], False),
    ("ch7", ["chapter7Experiments/verify_chapter7.py"], False),
    ("ch7util", ["chapter7Experiments/verify_extract_utilities.py"], False),
    ("fourcat", ["experiments/ch5_4class/verify_ch5_4class.py"], False),
    ("threecond", ["experiments/ch6_ch7_3class/verify_ch6_ch7_3class.py"], False),
    ("ablation", ["experiments/ablation/verify_ablation.py"], False),
    ("validators", ["validation/verify_validators.py"], False),
    ("confirmatory", ["experiments/confirmatory/verify_confirmatory.py"], False),
    ("manifest", ["experiments/confirmatory/check_against_manifest.py",
                  "experiments/confirmatory/confirmatory_results.json", "--tol", "0.05"], False),
    ("provenance", ["experiments/confirmatory/validate_provenance.py",
                    "experiments/confirmatory/confirmatory_results.json"], True),
    ("novel", ["experiments/novel/verify_novel.py"], False),
    ("nuisance", ["experiments/novel/verify_nuisance.py"], False),
    ("external", ["experiments/external/verify_external.py"], False),
    ("novelmanifest", ["experiments/novel/check_novel_manifest.py"], False),
    ("v2", ["experiments/v2/verify_v2.py"], False),
    ("claims", ["validation/validate_dissertation_claims.py"], False),
]

# Checks each suite reported at commit 913d457 (2026-11-21, Python 3.13.18). None means
# the suite prints no count. A suite may gain checks; it may not lose them.
BASELINE_CHECKS: dict[str, int | None] = {
    "ch4": 31, "ch5": 32, "ch5zero": 37, "ch5repro": 33, "ch5base": 55, "ch6": 31, "ch6repro": 42,
    "ch7": 38, "ch7util": 40, "fourcat": 25, "threecond": 28, "ablation": 23, "validators": 20,
    "confirmatory": 18, "manifest": None, "provenance": None, "novel": 13, "nuisance": 14,
    "external": 11, "novelmanifest": 42, "v2": 32, "claims": 207,
}

_COUNT_PATTERNS = (
    re.compile(r"(\d+) passed, (\d+) failed"),
    re.compile(r"RESULT: (\d+)/\d+ PASS, (\d+)/\d+ FAIL"),
    re.compile(r"(\d+) checked, (\d+) failure"),
)


def parse_counts(text: str) -> tuple[int, int] | None:
    """(passed, failed) from the last summary line a suite printed, if it printed one."""
    found = None
    for line in text.splitlines():
        for pat in _COUNT_PATTERNS:
            m = pat.search(line)
            if m:
                found = (int(m.group(1)), int(m.group(2)))
    return found


def isolated_copy(dst: Path) -> None:
    """Copy tracked and untracked-but-not-ignored files, preserving paths."""
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-co", "--exclude-standard", "-z"],
                         capture_output=True, check=True).stdout
    for rel in filter(None, out.decode().split("\0")):
        src = REPO / rel
        if src.is_file():
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)


def run_one(vid: str, argv: list[str], cwd: Path, log_dir: Path) -> dict:
    env = dict(os.environ, PYTHONUTF8="1", MPLBACKEND="Agg")
    start = time.time()
    proc = subprocess.run([sys.executable, *argv], cwd=cwd, env=env, capture_output=True, text=True)
    text = proc.stdout + proc.stderr
    (log_dir / f"{vid}.log").write_text(text)
    counts = parse_counts(text)
    base = BASELINE_CHECKS.get(vid)
    ok = proc.returncode == 0 and (counts is None or counts[1] == 0)
    lost = counts is not None and base is not None and counts[0] < base
    return {"id": vid, "ok": ok and not lost, "returncode": proc.returncode, "counts": counts,
            "baseline": base, "lost_checks": lost, "seconds": time.time() - start}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", metavar="ID", help="run only these suites")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--log-dir", type=Path, help="keep logs here (default: a temporary directory)")
    ap.add_argument("--list", action="store_true", help="list suite ids and exit")
    args = ap.parse_args()
    if args.list:
        for vid, argv, in_place in VERIFIERS:
            print(f"{vid:14s} {' '.join(argv)}{'   [in place]' if in_place else ''}")
        return 0
    chosen = [v for v in VERIFIERS if not args.only or v[0] in args.only]
    unknown = set(args.only or []) - {v[0] for v in VERIFIERS}
    if unknown:
        raise SystemExit(f"unknown suite id(s): {sorted(unknown)}")

    with tempfile.TemporaryDirectory(prefix="arspinet-verify-") as tmp:
        copy = Path(tmp) / "tree"
        log_dir = args.log_dir or Path(tmp) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        isolated_copy(copy)
        with cf.ThreadPoolExecutor(max(1, args.jobs)) as pool:
            futures = [pool.submit(run_one, vid, argv, REPO if in_place else copy, log_dir)
                       for vid, argv, in_place in chosen]
            results = [f.result() for f in futures]
        failed = [r for r in results if not r["ok"]]
        total = 0
        for r in results:
            c = r["counts"]
            total += c[0] if c else 0
            status = "ok" if r["ok"] else ("LOST CHECKS" if r["lost_checks"] else "FAIL")
            detail = f"{c[0]} passed, {c[1]} failed" if c else "no count"
            print(f"  {r['id']:14s} {status:12s} {detail:22s} baseline {r['baseline']!s:5s} {r['seconds']:6.1f} s")
        for r in failed:
            print(f"\n--- last lines of {r['id']} ---")
            print("\n".join((log_dir / f"{r['id']}.log").read_text().splitlines()[-15:]))
        print(f"\n{len(results) - len(failed)} of {len(results)} suites ok, {total} checks passed"
              + (f"; logs in {log_dir}" if args.log_dir else ""))
        return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
