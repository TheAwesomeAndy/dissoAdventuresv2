"""Fading-memory tests on the deployed reservoir.

v1 reported echo-state and Lyapunov evidence from a different, much weaker network than
the one that produced the confirmatory features, and used an infinitesimal perturbation
that cannot flip a threshold. Every test here runs on the network that is deployed, on
the input it actually receives, with finite perturbations. None of them uses a label.

``fading_memory_report`` returns four measurements:

``esp``           same input, two initial membrane states. Fraction of sequences whose
                  spike trains are identical over the final quarter, and the mean
                  per-step spike disagreement there, relative to the spike count.
``autonomous``    population rate at the end of a zero-input tail appended to the drive.
``perturbation``  a 1% change to the first 10 input samples. Spike disagreement over the
                  final quarter.
``reliability``   per-feature test-retest correlation of the binned code under 1%
                  additive input noise.

``passes`` applies one fixed rule to those numbers. The rule is label-free, so choosing
the recurrent gain with it cannot leak condition information.
"""
from __future__ import annotations

import numpy as np

from .reservoir import LIFReservoir

RULE = {
    "esp_converged_fraction_min": 0.95,
    "autonomous_rate_max": 1e-3,
    "reliability_median_r_min": 0.95,
}


def _disagreement(a, b, lo):
    """Per-sequence spike disagreement over [lo:], and whether the trains are identical."""
    diff = (a[:, lo:] != b[:, lo:]).sum(axis=(1, 2)).astype(float)
    total = (a[:, lo:].sum(axis=(1, 2)) + b[:, lo:].sum(axis=(1, 2))).astype(float)
    return diff / np.maximum(total, 1.0), diff == 0


def fading_memory_report(signals, rho, seed=42, noise_seed=0, tail=256, n_bins=6):
    """signals : (B, T) sample of the real per-electrode input."""
    x = np.asarray(signals, dtype=np.float64)
    B, T = x.shape
    rng = np.random.RandomState(noise_seed)
    res = LIFReservoir(rho=rho, seed=seed)
    last = T - T // 4

    base = res.run(x, return_spikes=True, n_bins=n_bins)
    mem0 = rng.uniform(0.0, res.theta, (B, res.n_res))
    other = res.run(x, mem0=mem0, return_spikes=True)
    esp_d, esp_same = _disagreement(base["spikes"], other["spikes"], last)

    xp = x.copy()
    xp[:, :10] *= 1.01
    pert = res.run(xp, return_spikes=True)
    per_d, per_same = _disagreement(base["spikes"], pert["spikes"], last)

    drive = np.concatenate([x, np.zeros((B, tail))], axis=1)
    rate = res.run(drive, return_rate=True)["rate"]
    driven_rate = float(rate[:, :T].mean())
    autonomous = float(rate[:, -(tail // 4):].mean())

    noisy = x + 0.01 * x.std(axis=1, keepdims=True) * rng.standard_normal(x.shape)
    c0 = base["counts"].reshape(B, -1).astype(np.float64)
    c1 = res.run(noisy, n_bins=n_bins)["counts"].reshape(B, -1).astype(np.float64)
    keep = (c0.std(axis=0) > 0) & (c1.std(axis=0) > 0)
    a = c0[:, keep] - c0[:, keep].mean(axis=0)
    b = c1[:, keep] - c1[:, keep].mean(axis=0)
    r = (a * b).sum(axis=0) / np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))

    # share of synaptic drive that is recurrent, on the driven segment
    if res.w_rec is not None:
        sp = base["spikes"].reshape(-1, res.n_res).astype(np.float64)
        rec = np.abs(sp @ res.w_rec.T).mean()
        inp = np.abs(x.reshape(-1, 1) * res.w_in[None, :]).mean()
        rec_to_in = float(rec / inp)
    else:
        rec_to_in = 0.0

    report = {
        "rho": float(rho),
        "n_sequences": int(B),
        "driven_population_rate": driven_rate,
        "recurrent_to_input_current_ratio": rec_to_in,
        "esp_converged_fraction": float(esp_same.mean()),
        "esp_mean_disagreement": float(esp_d.mean()),
        "perturbation_unchanged_fraction": float(per_same.mean()),
        "perturbation_mean_disagreement": float(per_d.mean()),
        "autonomous_rate_after_input_off": autonomous,
        "reliability_median_r": float(np.median(r)) if r.size else float("nan"),
        "reliability_p10_r": float(np.percentile(r, 10)) if r.size else float("nan"),
        "n_active_features": int(keep.sum()),
    }
    report["passes"] = passes(report)
    return report


def passes(report, rule=RULE):
    return bool(
        report["esp_converged_fraction"] >= rule["esp_converged_fraction_min"]
        and report["autonomous_rate_after_input_off"] <= rule["autonomous_rate_max"]
        and report["reliability_median_r"] >= rule["reliability_median_r_min"]
    )


def select_rho(reports):
    """Largest recurrent gain that passes the fading-memory rule (None if none pass)."""
    ok = [r["rho"] for r in reports if r["passes"]]
    return max(ok) if ok else None
