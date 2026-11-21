"""arspinet_v2.features: projection, matched controls and windows."""

import numpy as np

from arspinet_v2 import features


def test_projection_matches_the_v1_confirmatory_pipeline(code):
    import confirmatory_pipeline as cp

    v1 = cp.project_bsc(code, cp.make_srp(code.shape[2]))
    assert np.allclose(features.project(code, features.make_srp(code.shape[2])), v1)
    assert (features.SRP_SEED, features.N_SRP) == (cp.SRP_SEED, cp.N_SRP)


def test_projection_is_data_independent():
    a, b = features.make_srp(1536), features.make_srp(1536)
    assert (a.components_ != b.components_).nnz == 0
    assert (a.components_ != features.make_srp(1536, seed=1).components_).nnz > 0


def test_one_projection_is_shared_by_all_electrodes(code, srp):
    X = features.project(code, srp)
    assert X.shape == (len(code), code.shape[1] * 64)
    for ch in (0, code.shape[1] - 1):
        assert np.allclose(
            X[:, ch * 64 : (ch + 1) * 64], np.asarray(srp.transform(code[:, ch].astype(float)))
        )


def test_amplitude_bins_use_the_bsc6_edges(cohort):
    raw = cohort[0]
    a6 = features.amplitude_bins(raw, 6)
    assert a6.shape == (len(raw), raw.shape[2], 6)
    assert np.allclose(a6[3, 2, 4], raw[3, 4 * 42 : 5 * 42, 2].mean())
    assert np.allclose(a6[0, 0, 5], raw[0, 210:252, 0].mean())  # samples 252..255 dropped


def test_raw_erp16_matches_v1(cohort):
    import run_confirmatory_validation as v1

    raw = cohort[0]
    assert np.allclose(
        features.amplitude_bins(raw, 16).reshape(len(raw), -1), v1.raw_erp_bins(raw, n_bins=16)
    )


def test_windows_are_true_milliseconds(cohort):
    raw = cohort[0]
    assert np.array_equal(features.ms_window(raw, 0, 250), raw[:, :64])
    assert np.array_equal(features.ms_window(raw, 250, 500), raw[:, 64:128])
    assert features.WINDOWS_MS["0-1000ms_full"] == (0, 1000)


def test_lpp_is_the_300_to_1000_ms_mean(cohort):
    raw = cohort[0]
    assert features.LPP_WINDOW_MS == (300, 1000)
    assert np.allclose(features.lpp_mean_amplitude(raw), raw[:, 77:256].mean(axis=1))


def test_time_resolved_patterns(cohort):
    raw = cohort[0]
    p = features.time_resolved_patterns(raw, 32)
    assert p.shape == (32, len(raw), raw.shape[2])
    assert np.allclose(p[3], raw[:, 24:32].mean(axis=1))
