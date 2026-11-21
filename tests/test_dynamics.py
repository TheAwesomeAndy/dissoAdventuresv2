"""arspinet_v2.dynamics: the label-free fading-memory tests and the gain rule."""
import math

import numpy as np
import pytest

from arspinet_v2 import dynamics


@pytest.fixture(scope="module")
def signals(cohort):
    raw = cohort[0][:20]
    return raw.transpose(0, 2, 1).reshape(-1, raw.shape[1])


@pytest.fixture(scope="module")
def reports(signals):
    return {rho: dynamics.fading_memory_report(signals, rho) for rho in (0.0, 0.9)}


def test_feedforward_reservoir_passes(reports):
    r = reports[0.0]
    assert r["passes"]
    assert r["esp_converged_fraction"] == 1.0
    assert r["autonomous_rate_after_input_off"] == 0.0
    assert r["recurrent_to_input_current_ratio"] == 0.0


def test_v1_gain_fails(reports):
    r = reports[0.9]
    assert not r["passes"]
    assert r["esp_converged_fraction"] < 0.5
    assert r["autonomous_rate_after_input_off"] > dynamics.RULE["autonomous_rate_max"]
    assert r["reliability_median_r"] < reports[0.0]["reliability_median_r"]


def test_report_is_deterministic(signals):
    a, b = (dynamics.fading_memory_report(signals[:30], 0.5) for _ in range(2))
    assert a == b


def test_rule_thresholds_are_the_published_ones():
    """The rule that selected the v2 gain. Changing it changes the headline configuration."""
    assert dynamics.RULE == {"esp_converged_fraction_min": 0.95, "autonomous_rate_max": 1e-3,
                             "reliability_median_r_min": 0.95}


@pytest.mark.parametrize("key,value,expected", [
    ("esp_converged_fraction", 0.94, False), ("esp_converged_fraction", 0.95, True),
    ("autonomous_rate_after_input_off", 0.002, False), ("reliability_median_r", 0.94, False),
    ("reliability_median_r", math.nan, False),
])
def test_passes_applies_each_criterion(key, value, expected):
    report = {"esp_converged_fraction": 1.0, "autonomous_rate_after_input_off": 0.0,
              "reliability_median_r": 1.0}
    report[key] = value
    assert dynamics.passes(report) is expected


def test_select_rho():
    sweep = [{"rho": 0.0, "passes": True}, {"rho": 0.1, "passes": True}, {"rho": 0.2, "passes": False}]
    assert dynamics.select_rho(sweep) == 0.1
    assert dynamics.select_rho([{"rho": 0.3, "passes": False}]) is None


def test_no_label_enters_the_report(signals):
    """The report takes signals only; a permutation of the sequences changes nothing
    that the rule reads."""
    a = dynamics.fading_memory_report(signals[:40], 0.0)
    b = dynamics.fading_memory_report(signals[:40][::-1], 0.0)
    for k in ("esp_converged_fraction", "autonomous_rate_after_input_off"):
        assert a[k] == b[k]
    assert np.isclose(a["driven_population_rate"], b["driven_population_rate"])
