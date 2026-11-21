"""Committed result records: integrity, provenance and agreement with the documents.

These tests need no data. They guard the link between records, the code that produced
them and the numbers quoted in prose, which is where this repository drifted before.
"""

import json
import os
import re
import subprocess
import sys

import check_records
import pytest


@pytest.fixture(scope="module")
def lock():
    return check_records.load_lock()


@pytest.fixture(scope="module")
def v2(repo):
    return json.loads((repo / "experiments/v2/v2_results.json").read_text())


def test_records_and_frozen_code_match_the_lock(lock):
    assert check_records.check(lock, use_history=False) == []


@pytest.mark.history
def test_frozen_code_is_recoverable_from_git(lock, repo):
    shallow = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--is-shallow-repository"], capture_output=True, text=True
    ).stdout.strip()
    if shallow != "false":
        pytest.skip("needs full git history")
    assert check_records.check(lock, use_history=True) == []
    for name, rec in lock["records"].items():
        for code in rec["code_sha256"]:
            assert check_records.blob_sha256(rec["code_commit"], code) is not None, (name, code)


def test_every_record_json_in_the_v2_trees_is_locked(lock, repo):
    found = {
        str(p.relative_to(repo))
        for d in ("experiments/v2", "experiments/zscore_artifact")
        for p in (repo / d).glob("*.json")
    }
    assert found <= set(lock["records"]), f"unlocked records: {sorted(found - set(lock['records']))}"


def test_lock_entries_are_complete(lock, repo):
    for name, rec in lock["records"].items():
        assert {
            "produced_by",
            "command",
            "data",
            "tier",
            "provenance",
            "code_commit",
            "sha256",
            "code_sha256",
        } <= set(rec), name
        assert (repo / rec["produced_by"]).exists(), name
        assert rec["produced_by"] in rec["code_sha256"], name
        assert rec["produced_by"] in rec["command"], name


def test_no_record_awaits_regeneration_on_main(lock, repo):
    """A record may be pending while its rerun is in progress on a branch. Never on main."""
    branch = (
        os.environ.get("GITHUB_REF_NAME")
        or subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    )
    if branch == "main":
        assert check_records.pending(lock) == []


def test_v2_results_report_is_generated_from_the_record(repo, tmp_path):
    out = tmp_path / "V2_RESULTS.md"
    subprocess.run(
        [
            sys.executable,
            str(repo / "experiments/v2/make_report.py"),
            str(repo / "experiments/v2/v2_results.json"),
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    assert out.read_text() == (repo / "experiments/v2/V2_RESULTS.md").read_text()


def test_cohorts_are_the_audited_ones(v2):
    g = v2["granularities"]
    assert (g["three_condition"]["n_participants"], g["three_condition"]["n_observations"]) == (211, 633)
    assert (g["four_category"]["n_participants"], g["four_category"]["n_observations"]) == (210, 840)


def test_v2_reproduces_the_v1_confirmatory_record(v2, repo):
    v1 = json.loads((repo / "experiments/confirmatory/confirmatory_results.json").read_text())
    for name in ("three_condition", "four_category"):
        a = v2["granularities"][name]["v1_reproduction"]["v1_estimator_direct_fit"]
        b = v1["granularities"][name]["representations"]["bsc6_srp64"]
        assert a["balanced_accuracy_percent"] == b["balanced_accuracy_percent"]
        assert a["per_repeat_percent"] == b["per_repeat_percent"]
        raw_v2 = v2["granularities"][name]["representations"]["raw_erp16"]["balanced_accuracy_percent"]
        raw_v1 = v1["granularities"][name]["representations"]["raw_erp16"]["balanced_accuracy_percent"]
        assert abs(raw_v2 - raw_v1) < 0.5  # same features, row-space vs direct fit


def test_headline_gain_is_the_one_the_rule_selected(v2):
    fm = v2["fading_memory"]
    passing = [s["rho"] for s in fm["sweep_seed42"] if s["passes"]]
    assert fm["selected_rho"] == max(passing) == v2["protocol"]["headline_rho"] == 0.025


def _fmt(r):
    return f"{r['balanced_accuracy_percent']:.1f}% [{r['ci95_percent'][0]:.1f}, {r['ci95_percent'][1]:.1f}]"


def _pp(r):
    return f"{r['difference_pp']:+.1f} pp [{r['ci95_pp'][0]:+.1f}, {r['ci95_pp'][1]:+.1f}]"


def test_numbers_quoted_in_v2_upgrade_come_from_the_records(v2, repo):
    """The 'Results in brief' table of V2_UPGRADE.md, cell by cell."""
    text = (repo / "V2_UPGRADE.md").read_text()
    pca = json.loads((repo / "experiments/v2/v2_pca_tuned.json").read_text())["granularities"]
    g3, g4 = v2["granularities"]["three_condition"], v2["granularities"]["four_category"]
    expected = []
    for g in (g3, g4):
        r, c = g["representations"], g["contrasts"]
        expected += [
            _fmt(g["v1_reproduction"]["v1_estimator_direct_fit"]),
            _fmt(r["v2_headline_bsc6_srp64"]),
            _fmt(r["amplitude6_matched_bins"]),
            _fmt(r["raw_erp16"]),
            _fmt(r["lpp_300_1000ms_per_electrode"]),
            _pp(c["headline_minus_amplitude6"]).replace("+-", "-"),
            _pp(c["raw_erp16_minus_headline"]),
        ]
        s = g["seeds"]["reservoir_seeds_rho0.025"]
        expected.append(f"{s['mean']:.1f}% (sd {s['sd']:.2f})")
    for name in ("three_condition", "four_category"):
        expected.append(_pp(pca[name]["contrasts"]["best_srp_minus_best_pca"]))
    missing = [e for e in expected if e not in text]
    assert not missing, f"V2_UPGRADE.md does not contain these record values: {missing}"


def test_zscore_results_quote_the_records(repo):
    text = (repo / "experiments/zscore_artifact/RESULTS.md").read_text()
    null = json.loads((repo / "experiments/zscore_artifact/across_null.json").read_text())["paradigms"]
    for para in ("N170", "P3"):
        for norm, rec in null[para].items():
            assert f"{rec['observed_percent']:.1f}%" in text, (para, norm)
    assert re.search(r"1,000\s+permutations", text)


def test_readme_table_comes_from_the_records(v2, repo):
    """The 'What the data say' table in the root README, row by row."""
    text = (repo / "README.md").read_text()
    g3, g4 = v2["granularities"]["three_condition"], v2["granularities"]["four_category"]

    def acc(g, key):
        return g["representations"][key]["balanced_accuracy_percent"]

    rows = {
        "Raw ERP, 16 time bins": [acc(g, "raw_erp16") for g in (g3, g4)],
        "Six-bin amplitudes, no reservoir": [acc(g, "amplitude6_matched_bins") for g in (g3, g4)],
        "v2 reservoir code, mean of 10 weight draws": [
            g["seeds"]["reservoir_seeds_rho0.025"]["mean"] for g in (g3, g4)
        ],
        "v1 reservoir configuration": [
            g["v1_reproduction"]["v1_estimator_direct_fit"]["balanced_accuracy_percent"] for g in (g3, g4)
        ],
        "LPP mean amplitude, 300 to 1000 ms": [acc(g, "lpp_300_1000ms_per_electrode") for g in (g3, g4)],
        "Band power and Hjorth features": [acc(g, "conventional") for g in (g3, g4)],
    }
    for label, (a, b) in rows.items():
        assert f"| {label} | {a:.1f}% | {b:.1f}% |" in text, label
    null = json.loads((repo / "experiments/zscore_artifact/across_null.json").read_text())["paradigms"]
    for para in ("N170", "P3"):
        assert f"{null[para]['epoch_z_full']['observed_percent']:.1f}%" in text
