"""Provenance blocks for result records.

Every runner ends by writing a block that says which code, which commit and which
environment produced its record. ``scripts/check_records.py`` later verifies those
hashes against the working tree and against git.

A record is only as good as this block, so two things are recorded beyond the hashes:

``code_matches_commit``  True when every hashed file is byte-identical to its blob at
                         ``git_commit``. A run started from uncommitted code says so.
``environment``          interpreter, platform and the versions of the numerical stack.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

PACKAGES = ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(repo: Path, *args: str) -> bytes | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return out.stdout


def git_commit(repo: str | Path) -> str:
    out = _git(Path(repo), "rev-parse", "HEAD")
    return out.decode().strip() if out else "unknown"


def environment() -> dict[str, str]:
    env = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in PACKAGES:
        try:
            env[pkg.replace("-", "_")] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            env[pkg.replace("-", "_")] = "not installed"
    return env


def provenance(repo: str | Path, code_files: Iterable[str], **extra: Any) -> dict[str, Any]:
    """Provenance block for a record produced by ``code_files`` (paths relative to ``repo``).

    ``extra`` is merged in unchanged (input file counts, command line, notes).
    """
    repo = Path(repo)
    files = sorted(set(code_files))
    hashes = {f: sha256_file(repo / f) for f in files}
    commit = git_commit(repo)
    matches: bool | None = None
    if commit != "unknown":
        matches = True
        for f, want in hashes.items():
            blob = _git(repo, "show", f"{commit}:{f}")
            if blob is None or hashlib.sha256(blob).hexdigest() != want:
                matches = False
    env = environment()
    block: dict[str, Any] = {
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": commit,
        "code_matches_commit": matches,
        "code_sha256": hashes,
        "environment": env,
        # flat copies kept for readers of earlier records
        "python": env["python"],
        "numpy": env["numpy"],
        "scipy": env["scipy"],
        "scikit_learn": env["scikit_learn"],
    }
    block.update(extra)
    return block


def count_files(path: str | Path | None, suffix: str = ".txt") -> int:
    """Number of files with ``suffix`` under ``path`` (0 if ``path`` is None)."""
    if not path:
        return 0
    return sum(1 for p in Path(path).rglob(f"*{suffix}") if p.is_file())


def strip_provenance(record: Mapping[str, Any]) -> dict[str, Any]:
    """A record without its provenance block, for comparing two runs for equality."""
    return {k: v for k, v in record.items() if k != "provenance"}
