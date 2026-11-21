"""Repository-level invariants: sensitive data, documentation links, tooling in step."""
import json
import re
import subprocess

import pytest
import run_verifiers


def _tracked(repo):
    out = subprocess.run(["git", "-C", str(repo), "ls-files", "-z"], capture_output=True, check=True).stdout
    return [f for f in out.decode().split("\0") if f]


@pytest.fixture(scope="module")
def tracked(repo):
    return _tracked(repo)


# Raw SHAPE epochs, clinical tables and anything else that must never enter git.
FORBIDDEN = [r"(^|/)SHAPE_Community_\d+_IAPS.*\.txt$", r"(^|/)batch_data/", r"(^|/)categories(batch\d)?/",
             r"(^|/)clinical_profile\.csv$", r"Psychopathology.*\.xlsx$", r"(^|/)ParticipantInfo\.csv$",
             r"\.(bdf|vhdr|vmrk|eeg|set|fdt)$", r"(^|/)\.env$"]


def test_no_raw_or_clinical_data_is_tracked(tracked):
    bad = [f for f in tracked for pat in FORBIDDEN if re.search(pat, f)]
    assert not bad, f"restricted data tracked in git: {bad}"


def test_known_participant_level_files_are_the_only_ones(tracked, repo):
    """Participant-keyed derived files are tracked today (docs/REPOSITORY_AUDIT.md, finding S1).
    Whether they stay is the data owner's decision. Until then the set must not grow."""
    allow = set(json.loads((repo / "docs/participant_level_files.json").read_text())["files"])
    tabular = [f for f in tracked if f.endswith((".csv", ".pkl", ".npz", ".npy", ".parquet", ".h5", ".mat"))]
    assert set(tabular) <= allow | set(json.loads(
        (repo / "docs/participant_level_files.json").read_text())["aggregate_only"]), \
        f"new tabular/binary data file(s) tracked; classify them first: {sorted(set(tabular) - allow)}"


DOCS = ["README.md", "AGENTS.md", "CLAUDE.md", "docs/ARCHITECTURE.md", "docs/RESEARCH_MAP.md",
        "docs/EXPERIMENT_REGISTRY.md", "docs/REPRODUCIBILITY.md", "docs/AGENT_HANDOFF.md",
        "docs/REPOSITORY_AUDIT.md"]
_PATHLIKE = re.compile(r"`([A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)+/?|[A-Za-z0-9_\-]+\.(?:py|md|json|toml|txt|yml|cff|tex))`")


@pytest.mark.parametrize("doc", DOCS)
def test_paths_named_in_the_documentation_exist(doc, repo):
    """Every path written in backticks in the entry-point documents must exist.
    Paths that are deliberately absent are written with a leading ``./`` (user-supplied
    data) or listed in the document's own 'does not exist' sections without backticks."""
    text = (repo / doc).read_text()
    missing = sorted({m for m in _PATHLIKE.findall(text)
                      if not (repo / m).exists() and not m.startswith(("origin/", "refs/", "./"))})
    assert not missing, f"{doc} names paths that do not exist: {missing}"


def test_experiment_registry_is_consistent(repo):
    reg = json.loads((repo / "docs/experiment_registry.json").read_text())
    ids = [e["id"] for e in reg["experiments"]]
    assert len(ids) == len(set(ids))
    tiers = set(reg["tiers"])
    for e in reg["experiments"]:
        assert e["tier"] in tiers, e["id"]
        for key in ("scripts", "records", "verified_by"):
            for path in e.get(key, []):
                assert (repo / path).exists(), f"{e['id']}: {path} does not exist"
        for cmd in e.get("commands", []):
            script = next((tok for tok in cmd.split() if tok.endswith(".py")), None)
            assert script and (repo / script).exists(), f"{e['id']}: command names a missing script: {cmd}"


def test_registry_document_is_generated(repo, tmp_path):
    import build_registry_doc
    assert build_registry_doc.render(json.loads((repo / "docs/experiment_registry.json").read_text())) == \
        (repo / "docs/EXPERIMENT_REGISTRY.md").read_text()


def test_registry_covers_every_locked_record(repo):
    reg = json.loads((repo / "docs/experiment_registry.json").read_text())
    lock = json.loads((repo / "experiments/records.lock.json").read_text())
    registered = {r for e in reg["experiments"] for r in e.get("records", [])}
    assert set(lock["records"]) <= registered


def test_runner_and_ci_run_the_same_suites(repo):
    workflow = (repo / ".github/workflows/verify.yml").read_text().split("\n  tests:")[0]   # the verify job
    in_ci = set(re.findall(r"python ((?:[\w./-]+/)?(?:verify|validate|check)_[\w]+\.py)", workflow))
    in_runner = {argv[0] for _, argv, _ in run_verifiers.VERIFIERS}
    assert in_ci == in_runner
    assert set(run_verifiers.BASELINE_CHECKS) == {v[0] for v in run_verifiers.VERIFIERS}


def test_count_parser():
    assert run_verifiers.parse_counts("x\nCHAPTER 4 VERIFICATION COMPLETE: 31 passed, 0 failed\n") == (31, 0)
    assert run_verifiers.parse_counts("RESULT: 55/55 PASS, 0/55 FAIL") == (55, 0)
    assert run_verifiers.parse_counts("42 checked, 1 failure(s)") == (42, 1)
    assert run_verifiers.parse_counts("OK") is None


def test_agent_documents_share_one_source_of_truth(repo):
    claude = (repo / "CLAUDE.md").read_text()
    assert "AGENTS.md" in claude
    assert len(claude.splitlines()) < 60, "CLAUDE.md should point at AGENTS.md, not restate it"
