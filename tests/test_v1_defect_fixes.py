"""Regression tests for the defects fixed in the v1 record in November 2026.

Each test fails on the code as it was before the fix. The defects, their effect on
reported numbers and what was and was not regenerated are in docs/v1/DEFECT_FIXES.md.
"""

import importlib.util
import os

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")


def load(repo, rel):
    """Import a v1 script by path without running its main()."""
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_").removesuffix(".py"), repo / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_d1_fading_memory_is_measured_on_spikes(repo, monkeypatch):
    """Defense experiment C unpacked run_full's (membrane, spikes) as (spikes, _).

    Replacing the membrane trace with NaN must not change the result. With the old
    unpacking it turned every deviation into NaN."""
    import sys

    monkeypatch.syspath_prepend(str(repo / "defense_figures"))
    monkeypatch.setattr(sys, "argv", ["x"])
    m = load(repo, "defense_figures/experiment_c_memory_capacity/make_experiment_c_figures.py")
    reference = m.fading_memory_tau(beta=0.118)
    real = m.LIFReservoir.run_full

    def without_membrane(self, X):
        membrane, spikes = real(self, X)
        assert set(np.unique(spikes)) <= {0.0, 1.0}
        return np.full_like(membrane, np.nan), spikes

    monkeypatch.setattr(m.LIFReservoir, "run_full", without_membrane)
    assert np.isfinite(reference)
    assert m.fading_memory_tau(beta=0.118) == reference


def test_d2_saliency_input_is_channels_first(repo):
    """The EEGNet saliency script fed (N, time, channels) for either input layout.

    The script exits at import without PyTorch, so the helper is taken from its source."""
    import ast

    source = (repo / "experiments/interpretability/run_eegnet_saliency_comparison.py").read_text()
    func = next(
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "channels_first"
    )
    namespace: dict = {}
    exec(compile(ast.Module(body=[func], type_ignores=[]), "channels_first", "exec"), namespace)
    channels_first = namespace["channels_first"]
    assert channels_first(np.zeros((5, 34, 256))).shape == (5, 34, 256)
    assert channels_first(np.zeros((5, 256, 34))).shape == (5, 34, 256)


def test_d3_clustering_is_the_onnela_coefficient(repo):
    """Chapter 7 experiment A returned twice the Onnela weighted clustering."""
    m = load(repo, "chapter7Experiments/run_chapter7_experiment_A.py")
    rng = np.random.RandomState(0)
    w = rng.uniform(0.05, 1.0, (9, 9))
    w = (w + w.T) / 2
    np.fill_diagonal(w, 0.0)
    got = m.extract_topology(w)
    n = len(w)
    expected = np.zeros(n)
    for i in range(n):
        total = sum(
            (w[i, j] * w[i, h] * w[j, h]) ** (1 / 3)
            for j in range(n)
            for h in range(n)
            if len({i, j, h}) == 3
        )
        expected[i] = total / ((n - 1) * (n - 2))
    assert np.allclose(got[:, 0], w.sum(axis=1))
    assert np.allclose(got[:, 1], expected)
    assert got[:, 1].max() <= 1.0  # a clustering coefficient cannot exceed 1


def test_d3_fix_does_not_change_rank_statistics(repo):
    """Halving the clustering values leaves every Spearman-based quantity unchanged, so
    the stored coupling matrices and kappa are unaffected by the fix."""
    from scipy.stats import spearmanr

    rng = np.random.RandomState(1)
    clustering, descriptor = rng.rand(34), rng.rand(34)
    assert spearmanr(descriptor, clustering)[0] == spearmanr(descriptor, 2.0 * clustering)[0]


def test_d5_attention_pooling_does_not_see_test_labels(repo):
    """reproduce_chapter5 scored channels on the held-out fold's labels. On pure noise,
    where the true accuracy is one third, that code returned 0.55."""
    m = load(repo, "chapter5Experiments/reproduce_chapter5.py")
    rng = np.random.RandomState(0)
    n_subj = 60
    subjects, y = np.repeat(np.arange(n_subj), 3), np.tile([0, 1, 2], n_subj)
    noise = rng.standard_normal((len(y), 34, 8))
    assert m.attention_pooling_cv(noise, y, subjects) < 0.42


@pytest.mark.parametrize(
    "rel,forbidden",
    [
        ("chapter6Experiments/run_chapter6_exp5_interaction.py", "p_int=np.mean(np.array(F_null)>=F_obs)"),
        ("experiments/ch6_ch7_3class/ch7_04_experiments.py", ").sum() / N_PERM"),
        (
            "defense_figures/experiment_d_channel_permutation/make_experiment_d_figures.py",
            "(null >= obs).mean()",
        ),
        (
            "experiments/interpretability/run_arspinet_v2_attention_prototype.py",
            ">= np.abs(obs_diff)).mean()",
        ),
    ],
)
def test_d4_permutation_p_values_use_the_plus_one_form(repo, rel, forbidden):
    """A permutation p-value is (1 + count) / (1 + n). The scripts listed computed
    count / n, which reports exactly zero when nothing exceeds the observed value. They
    cannot be run here (they need restricted data), so the check is on the source."""
    assert forbidden not in (repo / rel).read_text()


def test_d6_fano_bound_includes_the_k_minus_one_term(repo):
    text = (repo / "experiments/ch6_ch7_3class/ch6_03_experiments.py").read_text()
    assert "I_decoded = np.log2(K) - H_b - P_e * np.log2(K - 1)" in text


def test_d9_chapter7_conclusions_are_computed(repo):
    """Experiments D and E printed fixed conclusions regardless of their results."""
    d = (repo / "chapter7Experiments/run_chapter7_experiment_D.py").read_text()
    e = (repo / "chapter7Experiments/run_chapter7_experiment_E.py").read_text()
    assert "are smaller than d ≈ 0.12" not in d and "max_abs_d" in d
    assert "CONCATENATION NEVER IMPROVES" not in e and "Computed summary" in e
    for text in (d, e):
        assert "/mnt/user-data" not in text
