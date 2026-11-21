"""End-to-end smoke test of the v2 runner's stages on synthetic data.

The real run needs the restricted SHAPE files and takes hours. This drives the same
stage functions with a small synthetic cohort and shrunken grids, so a refactor that
breaks the pipeline is caught without data. It checks structure and sanity, not values.
"""
import json
import subprocess
import sys

import numpy as np
import pytest
from conftest import make_cohort

pytestmark = pytest.mark.slow

STAGES = ("v1_reproduction", "rho_sweep_shared_reservoir", "representations", "contrasts", "seeds",
          "windows_ms", "time_resolved_decoding", "destruction_controls", "compressors",
          "compressor_contrasts", "variance_components", "permutation")


@pytest.fixture(scope="module")
def synthetic_run():
    import run_v2_validation as v2

    mp = pytest.MonkeyPatch()
    mp.setattr(v2, "RHO_GRID", (0.0, 0.025, 0.9))
    mp.setattr(v2, "RESERVOIR_SEEDS", (42, 1))
    mp.setattr(v2, "SRP_SEEDS", (0, 1))
    mp.setattr(v2, "DYNAMICS_N_SEQUENCES", 40)
    try:
        raw, y, g = make_cohort(n_subj=30, n_ch=4, seed=11)
        fm = v2.fading_memory(raw)
        gran = v2.run_granularity("three_condition", raw, y, g, fm["selected_rho"], cache=None,
                                  n_perm=19, jobs=1, n_var_perm=20)
    finally:
        mp.undo()
    return v2, fm, gran


def test_fading_memory_stage(synthetic_run):
    _, fm, _ = synthetic_run
    assert [s["rho"] for s in fm["sweep_seed42"]] == [0.0, 0.025, 0.9]
    assert fm["sweep_seed42"][0]["passes"] and not fm["sweep_seed42"][-1]["passes"]
    assert fm["selected_rho"] in (0.0, 0.025)


def test_every_stage_reports(synthetic_run):
    _, _, res = synthetic_run
    assert (res["n_participants"], res["n_observations"]) == (30, 90)
    for stage in STAGES:
        assert stage in res, stage
    for name, r in res["representations"].items():
        lo, hi = r["ci95_percent"]
        assert 0 <= lo <= r["balanced_accuracy_percent"] <= hi <= 100, name
    # the synthetic condition effect is late and large: decodable from amplitudes
    assert res["representations"]["raw_erp16"]["balanced_accuracy_percent"] > 60
    assert len(res["time_resolved_decoding"]["bins"]) == 32
    assert set(res["destruction_controls"]["temporal_collapse"]) == {"train_and_test", "test_only"}
    perm = res["permutation"]["v2_headline_bsc6_srp64"]
    assert perm["n_permutations"] == 19 and 0 < perm["p_value"] <= 1
    assert abs(res["variance_components"]["gaussian_noise_same_shape_as_headline"]["fraction_subject"]) < 0.1


def test_report_renders_whatever_the_runner_emits(synthetic_run, tmp_path):
    v2, fm, gran = synthetic_run
    record = {"version": "test", "fading_memory": fm, "granularities": {"three_condition": gran},
              "protocol": {"cv": "synthetic", "readout": "synthetic", "intervals": "synthetic"},
              "provenance": {"generated_utc": "n/a", "git_commit": "0" * 40, "python": "n/a",
                             "numpy": np.__version__, "scipy": "n/a", "scikit_learn": "n/a",
                             "input_file_counts": {"data3": 0, "data4": 0}}}
    src, dst = tmp_path / "r.json", tmp_path / "r.md"
    src.write_text(json.dumps(record))
    subprocess.run([sys.executable, v2._REPO + "/experiments/v2/make_report.py", str(src), str(dst)],
                   check=True, capture_output=True)
    text = dst.read_text()
    assert "## 4. Representations and matched controls" in text and "## 10." in text
