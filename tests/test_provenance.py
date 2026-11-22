"""The provenance block every v2 record embeds."""

from pathlib import Path

from arspinet_v2 import provenance as prov

REPO = Path(__file__).resolve().parents[1]


def test_block_names_commit_code_and_environment():
    block = prov.provenance(REPO, ["arspinet_v2/provenance.py"], note="x")
    assert set(block["code_sha256"]) == {"arspinet_v2/provenance.py"}
    assert len(block["code_sha256"]["arspinet_v2/provenance.py"]) == 64
    assert block["environment"]["numpy"] == block["numpy"]
    assert block["note"] == "x"


def test_block_records_thread_settings(monkeypatch):
    # Values that pass through LAPACK can change in the last recorded digit with the
    # BLAS thread count, so a record must say what it was produced with.
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    threads = prov.provenance(REPO, [])["threads"]
    assert threads["OPENBLAS_NUM_THREADS"] == "2"
    assert threads["OMP_NUM_THREADS"] is None
    assert set(prov.THREAD_VARIABLES) <= set(threads)
    assert "blas" in threads


def test_strip_provenance_leaves_the_results():
    assert prov.strip_provenance({"a": 1, "provenance": {"b": 2}}) == {"a": 1}
