"""arspinet_v2.reservoir: parity with the v1 operator and pinned dynamics."""

import numpy as np
import pytest

from arspinet_v2 import reservoir


def _v1_code(raw, seed_for_channel):
    from experiment_zero import LIFReservoir as V1
    from experiment_zero import bsc6_encode

    return np.stack(
        [
            [bsc6_encode(V1(seed=seed_for_channel(ch)).forward(raw[i, :, ch])) for ch in range(raw.shape[2])]
            for i in range(len(raw))
        ]
    ).astype(np.float32)


def test_per_electrode_mode_is_bit_identical_to_v1(cohort):
    raw = cohort[0][:5]
    got = reservoir.encode(raw, rho=0.9, seed=42, shared=False)
    assert np.array_equal(got, _v1_code(raw, lambda ch: 42 + 17 * ch))


def test_shared_mode_is_bit_identical_to_v1(cohort):
    raw = cohort[0][:5]
    got = reservoir.encode(raw, rho=0.9, seed=42, shared=True)
    assert np.array_equal(got, _v1_code(raw, lambda ch: 42))


def test_dynamics_are_pinned():
    """Golden values. If this fails the LIF update rule, the weight draw or the binning
    changed, and every committed reservoir result is no longer reproduced by the code."""
    x = np.random.RandomState(0).standard_normal((4, 256))
    out = {rho: reservoir.LIFReservoir(rho=rho, seed=42).run(x, n_bins=6)["counts"] for rho in (0.0, 0.9)}
    assert out[0.0].shape == (4, 6, 256)
    assert int(out[0.0].sum()) == GOLDEN[0.0]["total"]
    assert int(out[0.9].sum()) == GOLDEN[0.9]["total"]
    assert out[0.0].sum(axis=(0, 2)).astype(int).tolist() == GOLDEN[0.0]["per_bin"]
    assert out[0.9].sum(axis=(0, 2)).astype(int).tolist() == GOLDEN[0.9]["per_bin"]


# Computed with the code at commit 913d457 (numpy 2.5.3). RandomState streams are stable
# across numpy versions, and spike counts are integers, so these are exact.
GOLDEN = {
    0.0: {"total": 3192, "per_bin": [478, 594, 593, 542, 592, 393]},
    0.9: {"total": 31363, "per_bin": [4786, 5295, 5380, 5368, 5229, 5305]},
}


def test_rho_zero_has_no_recurrent_matrix():
    assert reservoir.LIFReservoir(rho=0.0).w_rec is None


@pytest.mark.parametrize("rho", [0.025, 0.5, 0.9])
def test_rho_is_the_spectral_radius(rho):
    w = reservoir.LIFReservoir(rho=rho).w_rec
    assert abs(np.abs(np.linalg.eigvals(w)).max() - rho) < 1e-9


def test_rho_rescales_one_weight_draw():
    a, b = reservoir.LIFReservoir(rho=0.9).w_rec, reservoir.LIFReservoir(rho=0.3).w_rec
    assert np.allclose(a / 0.9, b / 0.3)


def test_input_weights_do_not_depend_on_rho():
    assert np.array_equal(reservoir.LIFReservoir(rho=0.0).w_in, reservoir.LIFReservoir(rho=0.9).w_in)


def test_chunking_is_invisible(cohort):
    raw = cohort[0][:6]
    assert np.array_equal(reservoir.encode(raw, rho=0.3, chunk=7), reservoir.encode(raw, rho=0.3))


def test_trailing_samples_are_dropped_like_v1():
    """256 samples in 6 bins: 42 per bin, the last 4 samples are not counted."""
    res = reservoir.LIFReservoir(rho=0.0)
    x = np.zeros((1, 256))
    x[0, 252:] = 50.0  # huge drive only in the dropped tail
    assert res.run(x, n_bins=6)["counts"].sum() == 0
    assert res.run(x, return_spikes=True)["spikes"].sum() > 0


def test_non_spiking_control(cohort):
    lin = reservoir.encode(cohort[0][:4], rho=0.0, spiking=False)
    assert lin.min() >= 0 and not np.allclose(lin, np.round(lin))


def test_spikes_and_counts_agree(cohort):
    x = cohort[0][:3, :, 0]
    out = reservoir.LIFReservoir(rho=0.5).run(x, n_bins=6, return_spikes=True, return_rate=True)
    assert np.array_equal(out["spikes"][:, :252].reshape(3, 6, 42, 256).sum(axis=2), out["counts"])
    assert np.allclose(out["rate"], out["spikes"].mean(axis=2))


def test_bad_input_is_rejected():
    with pytest.raises(ValueError):
        reservoir.LIFReservoir().run(np.zeros(10))
    with pytest.raises(ValueError):
        reservoir.LIFReservoir().run(np.zeros((1, 3)), n_bins=6)
