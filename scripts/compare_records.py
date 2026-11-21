#!/usr/bin/env python3
"""Compare two result records for equality, ignoring their provenance blocks.

Use after regenerating a record to show that a code change did not change a result:

    python scripts/compare_records.py experiments/v2/v2_results.json /tmp/regenerated.json

Every value present in both records must be equal. Keys that exist only in the new
record are listed but are not a failure (a newer runner may record more). Keys that exist
only in the old record are a failure. Exit status 0 means the old record is reproduced.

    --tol 1e-9   allow this absolute difference between numbers (default 0: exact)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

IGNORED = ("provenance",)


def diff(old: Any, new: Any, tol: float, path: str = "") -> tuple[list[str], list[str]]:
    """Return (differences, additions) between two JSON values."""
    if isinstance(old, dict) and isinstance(new, dict):
        problems: list[str] = []
        added = [f"{path}/{k}" for k in new if k not in old and not (path == "" and k in IGNORED)]
        for k, v in old.items():
            if path == "" and k in IGNORED:
                continue
            if k not in new:
                problems.append(f"{path}/{k}: missing from the new record")
                continue
            p, a = diff(v, new[k], tol, f"{path}/{k}")
            problems += p
            added += a
        return problems, added
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return [f"{path}: length {len(old)} != {len(new)}"], []
        problems, added = [], []
        for i, (a, b) in enumerate(zip(old, new, strict=True)):
            p, ad = diff(a, b, tol, f"{path}[{i}]")
            problems += p
            added += ad
        return problems, added
    if (
        isinstance(old, bool)
        or isinstance(new, bool)
        or not all(isinstance(v, int | float) for v in (old, new))
    ):
        return ([] if old == new else [f"{path}: {old!r} != {new!r}"]), []
    if math.isnan(old) and math.isnan(new):
        return [], []
    return ([] if abs(old - new) <= tol else [f"{path}: {old!r} != {new!r}"]), []


def count_leaves(value: Any) -> int:
    if isinstance(value, dict):
        return sum(count_leaves(v) for v in value.values())
    if isinstance(value, list):
        return sum(count_leaves(v) for v in value)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--tol", type=float, default=0.0)
    args = ap.parse_args()
    with open(args.old) as fh:
        old = json.load(fh)
    with open(args.new) as fh:
        new = json.load(fh)
    problems, added = diff(old, new, args.tol)
    n = count_leaves({k: v for k, v in old.items() if k not in IGNORED})
    if added:
        print(
            f"only in the new record ({len(added)}): "
            + ", ".join(added[:8])
            + (" ..." if len(added) > 8 else "")
        )
    if problems:
        print(f"DIFFERENT: {len(problems)} of {n} values differ")
        for p in problems[:25]:
            print("  - " + p)
        return 1
    print(f"REPRODUCED: all {n} values of the old record equal the new one (tolerance {args.tol})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
