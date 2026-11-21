#!/usr/bin/env python3
"""Check that every committed result record still matches the code that produced it.

A *record* is a committed JSON file of results from a run on restricted or large data
that CI cannot repeat. The index of records is ``experiments/records.lock.json``. For
each record the lock stores

    sha256        the record file itself, so a hand edit is detected
    code_sha256   the files that produced it, so a silent code change is detected
    code_commit   the commit whose blobs carry those hashes, so the code can be recovered

Checks, per record:

    1. the record file hash equals the lock;
    2. every code file in the working tree equals the lock (the files are *frozen*);
    3. a ``provenance.code_sha256`` block inside the record, if present, equals the lock;
    4. with git history available, the blobs at ``code_commit`` equal the lock.

The lock also lists ``unverified_records``: committed result files whose producing code
changed after they were written, or that were annotated by hand. Only their own hash is
checked. They are evidence of what was once computed, not of what the code computes now.

Changing a frozen file is allowed only together with a rerun on the authorized data and
a regenerated record. Then refresh the lock entry:

    python scripts/check_records.py --update experiments/v2/v2_results.json

Usage:

    python scripts/check_records.py              # verify everything, exit 1 on failure
    python scripts/check_records.py --list-frozen
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOCK = REPO / "experiments" / "records.lock.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True)


def blob_sha256(commit: str, path: str) -> str | None:
    """Hash of ``path`` at ``commit``; None if the commit or path is not available."""
    out = git("show", f"{commit}:{path}")
    return sha256_bytes(out.stdout) if out.returncode == 0 else None


def load_lock() -> dict:
    return json.loads(LOCK.read_text())


def frozen_files(lock: dict | None = None) -> list[str]:
    lock = lock or load_lock()
    return sorted({f for rec in lock["records"].values() for f in rec["code_sha256"]})


def embedded_hashes(record_path: Path) -> dict[str, str]:
    """``provenance.code_sha256`` (and the permutation-stage block) from a record."""
    try:
        prov = json.loads(record_path.read_text()).get("provenance", {})
    except (json.JSONDecodeError, AttributeError):
        return {}
    return dict(prov.get("code_sha256") or {})


def check(lock: dict | None = None, use_history: bool = True) -> list[str]:
    """Return a list of human-readable problems. Empty means every record verifies."""
    lock = lock or load_lock()
    problems: list[str] = []
    for name, rec in lock["records"].items():
        path = REPO / name
        if not path.exists():
            problems.append(f"{name}: record file is missing")
            continue
        if sha256_file(path) != rec["sha256"]:
            problems.append(f"{name}: record file differs from the lock (edited by hand, or regenerated "
                            f"without `check_records.py --update`)")
        embedded = embedded_hashes(path)
        for code, want in rec["code_sha256"].items():
            cpath = REPO / code
            if not cpath.exists():
                problems.append(f"{name}: producing file {code} is missing")
                continue
            if sha256_file(cpath) != want:
                problems.append(f"{name}: frozen file {code} has changed since the record was produced")
            if code in embedded and embedded[code] != want:
                problems.append(f"{name}: {code} hash inside the record disagrees with the lock")
            if use_history:
                at = blob_sha256(rec["code_commit"], code)
                if at is not None and at != want:
                    problems.append(f"{name}: {code} at commit {rec['code_commit'][:10]} does not have the locked hash")
        missing = set(embedded) - set(rec["code_sha256"])
        if missing:
            problems.append(f"{name}: record hashes files the lock does not list: {sorted(missing)}")
    for name, rec in lock.get("unverified_records", {}).items():
        path = REPO / name
        if not path.exists():
            problems.append(f"{name}: unverified record file is missing")
        elif sha256_file(path) != rec["sha256"]:
            problems.append(f"{name}: unverified record file differs from the lock")
    return problems


def update(name: str) -> None:
    """Refresh one lock entry after a record has been regenerated."""
    lock = load_lock()
    rec = lock["records"].get(name)
    if rec is None:
        raise SystemExit(f"{name} is not in {LOCK.relative_to(REPO)}; add an entry by hand first "
                         f"(produced_by, command, data, tier, code_sha256 keys)")
    path = REPO / name
    embedded = embedded_hashes(path)
    files = sorted(set(rec["code_sha256"]) | set(embedded))
    rec["sha256"] = sha256_file(path)
    rec["code_sha256"] = {f: sha256_file(REPO / f) for f in files}
    prov = json.loads(path.read_text()).get("provenance", {})
    commit = prov.get("git_commit") or git("rev-parse", "HEAD").stdout.decode().strip()
    rec["code_commit"] = commit
    rec["provenance"] = "embedded" if embedded else "reconstructed"
    LOCK.write_text(json.dumps(lock, indent=2) + "\n")
    print(f"updated {name}: code_commit {commit[:10]}, {len(files)} code files")
    for f in files:
        if blob_sha256(commit, f) not in (None, rec["code_sha256"][f]):
            print(f"  WARNING: {f} in the working tree differs from commit {commit[:10]}; commit the code first")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update", metavar="RECORD", help="refresh the lock entry of a regenerated record")
    ap.add_argument("--list-frozen", action="store_true", help="print the frozen code files and exit")
    ap.add_argument("--no-history", action="store_true", help="skip the check against git blobs")
    args = ap.parse_args()
    if args.update:
        update(args.update)
        return 0
    lock = load_lock()
    if args.list_frozen:
        print("\n".join(frozen_files(lock)))
        return 0
    problems = check(lock, use_history=not args.no_history)
    n = len(lock["records"])
    if problems:
        print(f"record check: {len(problems)} problem(s) across {n} records")
        for p in problems:
            print("  - " + p)
        return 1
    print(f"record check: OK ({n} records, {len(frozen_files(lock))} frozen code files, "
          f"{len(lock.get('unverified_records', {}))} unverified legacy records)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
