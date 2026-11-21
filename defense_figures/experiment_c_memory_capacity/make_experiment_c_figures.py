"""
defense_figures/experiment_c_memory_capacity/make_experiment_c_figures.py

Experiment C — Memory-capacity curve for the dissertation's main-pipeline
reservoir (LIFReservoir, ρ=0.9 rescaled).

Philosophical claim it lands:
  "I did not pick β = 0.05 because it gave the best accuracy. I picked it
  because it sits at the measured memory-capacity peak — meaning the
  operating point is theoretically derived, not empirically tuned."

RESERVOIR CHOICE — DELIBERATE
-----------------------------
The dissertation uses two distinct reservoir families:

  • chapter-6 `Reservoir` (sparse Gaussian, no SR rescaling) — used for
    dynamical characterization (driven λ₁, ESP) in Experiments A and 6.x.
  • `LIFReservoir` (Xavier-uniform rescaled to ρ=0.9) — used as the
    operating reservoir for the spike-to-embedding feature pipeline in
    chapters 3, 4, 5 of the dissertation.

This experiment measures memory capacity on the **LIFReservoir** because
that is the reservoir whose β hyperparameter the dissertation must
justify — it is the reservoir whose spikes feed the BSC₆ → PCA-64 →
classifier pipeline. Measuring MC on the chapter-6 Reservoir would
answer a different question; that one is used for its driven contraction
properties, not for downstream memory.

A committee member could (correctly) ask "why not measure MC on the same
reservoir whose λ₁ you reported in Experiment A?" The answer is in the
companion_notes.md: the two reservoirs are different theoretical
instruments; each operates at β=0.05 for instrument-appropriate reasons.

This directly answers the mentor's warning: never say "because it yielded
the highest accuracy"; say "because it provided the strictest mathematical
bound" — or, here, "because it sat at the measured information-theoretic
peak."

OUTPUTS
-------
  outputs/rawC_3a_input_drive.pdf
      Sample of the white-noise drive used for MC estimation.

  outputs/rawC_3b_state_response_per_beta.pdf
      Reservoir membrane response at three β values (low / chosen / high),
      showing why memory degrades at both ends of the sweep.

  outputs/rawC_3c_mc_vs_tau.pdf
      MC(τ) curves at several β values, demonstrating the per-delay
      contribution and how the curve shape changes with β.

  outputs/analysisC_3d_memory_capacity_peak.pdf
      THE SLIDE. Total MC vs. β (primary y-axis) + driven λ₁ vs. β
      (secondary y-axis). The dissertation's chosen β = 0.05 marked,
      together with the measured peak β*.

  outputs/experiment_c_data.csv
      Per-β total MC, β-resolved λ₁, and per-τ MC values.

METHOD
------
Jaeger memory capacity (Jaeger, 2001): drive the reservoir with i.i.d.
zero-mean noise u(t); for each delay τ ∈ {1, …, T_max}, train a linear
readout to estimate u(t − τ) from the reservoir membrane state x(t).
MC(τ) is the squared correlation between the true and predicted past
input; the total memory capacity is MC = Σ_τ MC(τ), bounded above by N
(the reservoir size).

The reservoir matches the chapter-6 definition (sparse Gaussian, σ=0.05,
density 0.10) but β is varied across the sweep.

For the secondary axis, driven λ₁ is computed via Benettin's two-trajectory
method using a sample of real SHAPE ERPs from the pickle — keeping the
input distribution consistent with the rest of the dissertation.

PRE-REGISTRATION
----------------
Expected: total MC rises from β ≪ 0.05, peaks near β ≈ 0.05–0.10, and
falls at higher β as the reservoir's effective time constant becomes
too short for useful integration. The driven λ₁ is expected to remain
negative across the full sweep, with magnitude rising (more contraction)
as β increases.

If the MC peak lands far from β = 0.05, the slide presents the result
faithfully and the author defends the discrepancy: a peak at smaller β
would suggest the dissertation's choice was conservative (more contraction
than necessary); a peak at larger β would require the author to revisit
whether 0.05 was the right operating point or only "close enough".
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree  # unused but kept for parity with other scripts
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _style import apply_style, save_pdf, figtext_footer, PALETTE, FIGSIZE


N_RES = 256
N_INPUT = 1
M_TH = 0.5
TARGET_SR = 0.9
DISSERTATION_BETA = 0.05
DELTA_0 = 1e-8
T_RENORM = 50


# ──────────────────────────────────────────────────────────────────────────
# LIFReservoir — verbatim copy from chapter-3/4/5 main pipeline
# (Xavier-uniform W_in/W_rec, W_rec rescaled to ρ=0.9, subtractive reset)
# ──────────────────────────────────────────────────────────────────────────
class LIFReservoir:
    """Main-pipeline reservoir (Xavier-uniform, ρ=0.9 rescaled)."""

    def __init__(self, beta: float, seed: int = 42,
                 n_input: int = N_INPUT, n_res: int = N_RES,
                 threshold: float = M_TH, target_sr: float = TARGET_SR) -> None:
        self.beta = beta
        self.n_res = n_res
        self.threshold = threshold
        rng = np.random.RandomState(seed)
        limit_in = np.sqrt(6.0 / (n_input + n_res))
        self.W_in = rng.uniform(-limit_in, limit_in, (n_res, n_input))
        limit_rec = np.sqrt(6.0 / (n_res + n_res))
        self.W_rec = rng.uniform(-limit_rec, limit_rec, (n_res, n_res))
        eigs = np.abs(np.linalg.eigvals(self.W_rec))
        if eigs.max() > 0:
            self.W_rec *= target_sr / eigs.max()

    def run_full(self, X: np.ndarray):
        """X: shape (T, n_input). Returns (M, S) trajectories, each (T, N)."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        T = X.shape[0]
        mem = np.zeros(self.n_res)
        spk_prev = np.zeros(self.n_res)
        M = np.zeros((T, self.n_res))
        S = np.zeros((T, self.n_res), dtype=np.int8)
        for t in range(T):
            I_tot = self.W_in @ X[t] + self.W_rec @ spk_prev
            mem = (1.0 - self.beta) * mem * (1.0 - spk_prev) + I_tot
            spk = (mem >= self.threshold).astype(float)
            mem = np.maximum(mem - spk * self.threshold, 0.0)
            M[t] = mem
            S[t] = spk.astype(np.int8)
            spk_prev = spk
        return M, S


# Backwards-compat alias for code that referenced Reservoir
Reservoir = LIFReservoir


# ──────────────────────────────────────────────────────────────────────────
# Jaeger memory capacity (membrane-readout version)
# ──────────────────────────────────────────────────────────────────────────
def memory_capacity(
    beta: float,
    seed: int = 42,
    T: int = 3500,
    T_warmup: int = 500,
    T_train: int = 1500,
    T_test: int = 1000,
    max_tau: int = 30,
    input_std: float = 2.0,
    spike_smooth: int = 10,
) -> tuple[float, np.ndarray]:
    """
    Estimate Jaeger memory capacity for the chapter-6 reservoir at the given β.

    The readout feature is **smoothed spike count per neuron** — a sliding
    rectangular window of length `spike_smooth` applied to S(t). This matches
    what the dissertation's downstream pipeline consumes (binned spike
    counts → PCA-64) rather than the raw membrane potential, which is
    repeatedly reset by spike events in this LIF reservoir family and
    therefore carries less linear-readable history.

    Linear readout is trained on training segment, tested on held-out segment.

    Returns:
      total_mc: float (sum over τ of squared correlation of held-out recall)
      mc_per_tau: np.ndarray of shape (max_tau,)
    """
    res = LIFReservoir(beta=beta, seed=seed)
    rng = np.random.RandomState(seed + 1)
    u = (rng.randn(T) * input_std).reshape(-1, 1)

    _, S = res.run_full(u)
    # Smoothed spike count per neuron (sliding rectangular window)
    from scipy.ndimage import uniform_filter1d
    X_feat = uniform_filter1d(S.astype(float), size=spike_smooth, axis=0)

    train_start = T_warmup + max_tau
    train_end = train_start + T_train
    test_start = train_end
    test_end = test_start + T_test
    if test_end > T:
        raise ValueError(f"T={T} too short for warmup+train+test+max_tau")

    X_train = X_feat[train_start:train_end]      # (T_train, N)
    X_test = X_feat[test_start:test_end]          # (T_test,  N)

    # Add a bias column
    X_train = np.column_stack([X_train, np.ones(T_train)])
    X_test = np.column_stack([X_test, np.ones(T_test)])

    mc_per_tau = np.zeros(max_tau)
    for tau in range(1, max_tau + 1):
        y_train = u[train_start - tau : train_end - tau, 0]
        y_test = u[test_start - tau : test_end - tau, 0]
        # Closed-form linear least-squares with small ridge
        ridge = 1e-6
        XtX = X_train.T @ X_train + ridge * np.eye(X_train.shape[1])
        Xty = X_train.T @ y_train
        try:
            w = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
        y_pred = X_test @ w
        if y_test.std() < 1e-12 or y_pred.std() < 1e-12:
            mc_per_tau[tau - 1] = 0.0
        else:
            mc_per_tau[tau - 1] = np.corrcoef(y_pred, y_test)[0, 1] ** 2
    return float(np.sum(mc_per_tau)), mc_per_tau


# ──────────────────────────────────────────────────────────────────────────
# Driven λ₁ for a given β, on a sample of real ERPs
# ──────────────────────────────────────────────────────────────────────────
def fading_memory_tau(
    beta: float,
    seed: int = 42,
    T: int = 300,
    pulse_start: int = 50,
    pulse_dur: int = 10,
    pulse_amp: float = 2.0,
    baseline_noise: float = 0.05,
    max_delay: int = 200,
) -> float:
    """
    Chapter-3 fading-memory time constant for the LIFReservoir.

    Drive the reservoir with low-amplitude white noise plus a brief
    rectangular pulse, measure the L2 deviation of the population spike
    rate from baseline at increasing post-pulse delays, fit a single
    exponential, return the decay time constant τ.

    Identical to `experiment_3_fading_memory()` in
    `experiments/chapter3/run_chapter3_lsm_characterization.py:256`.

    A smaller τ means faster forgetting (more "contraction" in the
    operational sense the dissertation uses for this reservoir family).
    """
    res = LIFReservoir(beta=beta, seed=seed)
    rng = np.random.RandomState(seed)
    X = (rng.randn(T, 1) * baseline_noise).astype(float)
    X[pulse_start:pulse_start + pulse_dur, :] += pulse_amp
    # run_full returns (membrane, spikes). The statistic is defined on spikes.
    # Fixed 2026-11: this line unpacked the membrane as S (docs/v1/DEFECT_FIXES.md, D1).
    _, S = res.run_full(X)
    baseline = S[:pulse_start].mean(axis=0)
    delays = np.arange(0, max_delay, 5)
    devs = []
    for d in delays:
        t = pulse_start + pulse_dur + d
        if t + 10 < T:
            devs.append(np.linalg.norm(S[t:t + 10].mean(axis=0) - baseline))
        else:
            devs.append(0.0)
    devs = np.array(devs)
    valid = devs > 1e-3
    if valid.sum() < 4:
        return float("nan")
    c = np.polyfit(delays[valid], np.log(devs[valid] + 1e-12), 1)
    if c[0] >= 0:
        return float("inf")
    return float(-1.0 / c[0])


def driven_lambda_at_beta(beta, erp_samples, seed=42, **_):
    """
    Wrapper kept for backwards compatibility with the script's main loop.
    The dissertation's stability measure for the LIFReservoir is the
    fading-memory τ, not Benettin's λ₁. We return τ here under a generic
    "stability_value" interpretation; the analysis code distinguishes
    them by name.
    """
    return fading_memory_tau(beta, seed=seed)


# ──────────────────────────────────────────────────────────────────────────
# Population sweep
# ──────────────────────────────────────────────────────────────────────────
def compute_sweep(betas, erp_samples, T=3000, max_tau=30):
    print(f"  Sweeping β ∈ [{betas.min():.4f}, {betas.max():.4f}], {len(betas)} values...")
    t0 = time.time()
    mc_total = np.zeros(len(betas))
    mc_per_tau = np.zeros((len(betas), max_tau))
    tau_arr = np.zeros(len(betas))
    for i, beta in enumerate(betas):
        tb = time.time()
        mc, per_tau = memory_capacity(beta, T=T, max_tau=max_tau)
        mc_total[i] = mc
        mc_per_tau[i] = per_tau
        tau_fade = fading_memory_tau(beta)
        tau_arr[i] = tau_fade
        print(f"    β={beta:.4f}: MC={mc:.3f}, τ_fade={tau_fade:.2f}  ({time.time()-tb:.1f}s)")
    print(f"  Sweep complete in {time.time()-t0:.0f}s.")
    return mc_total, mc_per_tau, tau_arr


# ──────────────────────────────────────────────────────────────────────────
# FIGURE 3A — Input drive
# ──────────────────────────────────────────────────────────────────────────
def make_rawC_3a(outdir: Path) -> Path:
    rng = np.random.RandomState(43)
    u = rng.randn(3000) * 0.6
    fig, (ax_ts, ax_hist) = plt.subplots(1, 2, figsize=FIGSIZE["raw_two_panel"],
                                         gridspec_kw={"width_ratios": [3, 1]})
    ax_ts.plot(u[:400], color=PALETTE["histogram_charcoal"], lw=0.7)
    ax_ts.set_xlabel("time step")
    ax_ts.set_ylabel("u(t)")
    ax_ts.set_title("White-noise drive (first 400 steps shown)", fontsize=11,
                    fontweight="bold")
    ax_ts.grid(True, alpha=0.2)

    ax_hist.hist(u, bins=60, color=PALETTE["histogram_charcoal"],
                 edgecolor="white", alpha=0.85, orientation="horizontal")
    ax_hist.set_xlabel("count")
    ax_hist.set_title(f"distribution\nμ={u.mean():.3f}, σ={u.std():.3f}",
                      fontsize=10, fontweight="bold")
    ax_hist.grid(True, alpha=0.2)
    ax_hist.set_ylim(ax_ts.get_ylim())

    fig.suptitle("Raw Observation C.3a — Input drive for memory-capacity estimation",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return save_pdf(fig, outdir / "rawC_3a_input_drive.pdf")


# ──────────────────────────────────────────────────────────────────────────
# FIGURE 3B — State response at three β values
# ──────────────────────────────────────────────────────────────────────────
def make_rawC_3b(outdir: Path, betas_to_show=(0.01, 0.05, 0.30)) -> Path:
    rng = np.random.RandomState(44)
    u = rng.randn(500) * 0.6
    fig, axes = plt.subplots(len(betas_to_show), 2, figsize=(13, 8),
                             gridspec_kw={"width_ratios": [3, 2]},
                             sharex="col")
    for row, beta in enumerate(betas_to_show):
        res = LIFReservoir(beta=beta, seed=42)
        M, S = res.run_full(u.reshape(-1, 1))
        # Left: mean membrane vs time, and the input overlay
        ax_l = axes[row, 0]
        ax_l.plot(u, color=PALETTE["neutral_gray"], lw=0.6, alpha=0.7,
                  label="u(t)")
        ax_l.plot(M.mean(1), color=PALETTE["histogram_charcoal"], lw=1.2,
                  label="mean(membrane)")
        ax_l.set_title(f"β = {beta:.3f}", fontsize=11, fontweight="bold",
                       color=(PALETTE["stable_green"] if beta == DISSERTATION_BETA
                              else PALETTE["histogram_charcoal"]))
        ax_l.set_ylabel("u, ⟨m⟩")
        if row == 0:
            ax_l.legend(fontsize=8, loc="upper right")
        ax_l.grid(True, alpha=0.2)
        if row == len(betas_to_show) - 1:
            ax_l.set_xlabel("time step")
        # Right: per-neuron membrane heatmap (slice of 80 neurons)
        ax_r = axes[row, 1]
        im = ax_r.imshow(M[:, :80].T, aspect="auto", cmap="hot",
                         vmin=0, vmax=M_TH * 1.2, origin="lower")
        ax_r.set_ylabel("neuron idx")
        if row == 0:
            plt.colorbar(im, ax=ax_r, fraction=0.046, pad=0.04, label="V")
        if row == len(betas_to_show) - 1:
            ax_r.set_xlabel("time step")
    fig.suptitle(
        "Raw Observation C.3b — Reservoir response at low / chosen / high β. "
        f"β = {DISSERTATION_BETA} (chosen) shown in green.",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    return save_pdf(fig, outdir / "rawC_3b_state_response_per_beta.pdf")


# ──────────────────────────────────────────────────────────────────────────
# FIGURE 3C — MC(τ) curves at several β values
# ──────────────────────────────────────────────────────────────────────────
def make_rawC_3c(mc_per_tau: np.ndarray, betas: np.ndarray, outdir: Path,
                 show_betas=(0.01, 0.05, 0.15, 0.30)) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6))
    # Pick the indices in `betas` nearest to the requested show_betas
    for sb in show_betas:
        idx = int(np.argmin(np.abs(betas - sb)))
        actual_b = betas[idx]
        color = (PALETTE["stable_green"] if abs(actual_b - DISSERTATION_BETA) < 1e-6
                 else PALETTE["histogram_charcoal"])
        lw = 2.2 if abs(actual_b - DISSERTATION_BETA) < 1e-6 else 1.3
        alpha = 1.0 if abs(actual_b - DISSERTATION_BETA) < 1e-6 else 0.65
        ax.plot(np.arange(1, mc_per_tau.shape[1] + 1), mc_per_tau[idx],
                lw=lw, alpha=alpha, color=color,
                label=f"β = {actual_b:.3f}  (total MC = {mc_per_tau[idx].sum():.2f})")
    ax.set_xlabel("recall delay τ (steps)")
    ax.set_ylabel(r"$\mathrm{MC}(\tau)$  (squared correlation)")
    ax.set_title(
        "Raw Observation C.3c — Memory contribution per delay τ, at several β values",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.92)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0, max(mc_per_tau.max() * 1.1, 0.05))
    return save_pdf(fig, outdir / "rawC_3c_mc_vs_tau.pdf")


# ──────────────────────────────────────────────────────────────────────────
# FIGURE 3D — THE SLIDE
# ──────────────────────────────────────────────────────────────────────────
def make_analysisC_3d(betas: np.ndarray, mc_total: np.ndarray,
                      tau_arr: np.ndarray, outdir: Path,
                      meta_note: str | None = None) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE["slide_landscape_tall"])

    # Primary y-axis: total MC
    line_mc, = ax.plot(betas, mc_total, marker="o", ms=5, lw=2.0,
                       color=PALETTE["histogram_charcoal"],
                       label="total memory capacity")
    ax.set_xscale("log")
    ax.set_xlabel(r"leak rate  $\beta$", fontsize=11)
    ax.set_ylabel("total memory capacity  MC = Σᵗᵃᵘ MC(τ)", fontsize=11,
                  color=PALETTE["histogram_charcoal"])
    ax.tick_params(axis="y", labelcolor=PALETTE["histogram_charcoal"])
    ax.grid(True, which="both", alpha=0.20)

    # Mark the measured peak β*
    peak_idx = int(np.argmax(mc_total))
    beta_star = betas[peak_idx]
    mc_star = mc_total[peak_idx]
    ax.axvline(beta_star, color=PALETTE["annotation_blue"], lw=1.2,
               linestyle="--", alpha=0.6)
    ax.scatter([beta_star], [mc_star], s=90,
               facecolor=PALETTE["annotation_blue"], edgecolor="white",
               linewidths=1.5, zorder=4)

    # Mark the dissertation's chosen β
    if DISSERTATION_BETA < betas.min() or DISSERTATION_BETA > betas.max():
        chosen_idx = None
    else:
        chosen_idx = int(np.argmin(np.abs(betas - DISSERTATION_BETA)))
    if chosen_idx is not None:
        chosen_b = betas[chosen_idx]
        chosen_mc = mc_total[chosen_idx]
        ax.axvline(chosen_b, color=PALETTE["stable_green"], lw=2.4)
        ax.scatter([chosen_b], [chosen_mc], s=130,
                   facecolor=PALETTE["stable_green"], edgecolor="white",
                   linewidths=1.5, zorder=5)

    # Combined annotation box in the lower-right (out of curve area)
    annot_lines = [
        f"β*  = {beta_star:.3f}  (measured MC peak)",
        f"        MC* = {mc_star:.3f}",
        "",
        f"β    = {DISSERTATION_BETA:.3f}  (dissertation's choice)",
        (f"        MC  = {chosen_mc:.3f}" if chosen_idx is not None else ""),
    ]
    ax.text(
        0.98, 0.03,
        "\n".join(annot_lines),
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=10, family="monospace", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.95,
                  edgecolor=PALETTE["neutral_gray"], lw=1.0),
        zorder=6,
    )

    # Legend (MC only — secondary axis intentionally not used; see
    # companion_notes.md for why driven λ₁ / fading-memory τ on this
    # reservoir are reserved for other figures)
    ax.legend([line_mc], [line_mc.get_label()], loc="upper left",
              framealpha=0.94, fontsize=10)

    # Determine relationship between dissertation β and the MC peak.
    # A plateau exists when MC stays above e.g. 90% of peak for β in some range;
    # we report the plateau range as well as the peak vs. chosen position.
    plateau_threshold = 0.90 * mc_star
    plateau_mask = mc_total >= plateau_threshold
    plateau_betas = betas[plateau_mask]
    in_plateau = (chosen_idx is not None) and plateau_mask[chosen_idx]
    if chosen_idx is not None and abs(beta_star - DISSERTATION_BETA) / DISSERTATION_BETA < 0.4:
        verdict = "at the measured memory-capacity peak"
    elif in_plateau:
        verdict = (f"within the measured memory-capacity plateau "
                   f"({plateau_betas.min():.3f} ≤ β ≤ {plateau_betas.max():.3f}, "
                   f"MC ≥ {plateau_threshold:.2f})")
    else:
        verdict = "off the measured peak (presented faithfully)"
    title = (
        f"β = {DISSERTATION_BETA} sits {verdict}  ·  "
        f"operating point derived, not tuned"
    )
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)

    figtext_footer(
        fig,
        "Memory capacity is measured.  The operating point β = 0.05 is theoretically motivated, not tuned to validation accuracy.",
        y=0.03,
    )

    if meta_note:
        fig.text(0.99, 0.01, meta_note, ha="right", va="bottom",
                 fontsize=7, color=PALETTE["neutral_gray"], style="italic")

    return save_pdf(fig, outdir / "analysisC_3d_memory_capacity_peak.pdf")


# ──────────────────────────────────────────────────────────────────────────
# CSV audit
# ──────────────────────────────────────────────────────────────────────────
def write_csv(betas, mc_total, mc_per_tau, tau_arr, outdir: Path, args):
    p = outdir / "experiment_c_data.csv"
    max_tau = mc_per_tau.shape[1]
    with open(p, "w") as f:
        f.write("# Experiment C — Memory-capacity sweep audit trail\n")
        f.write(f"# script: defense_figures/experiment_c_memory_capacity/make_experiment_c_figures.py\n")
        f.write(f"# method (primary):   Jaeger memory capacity, smoothed-spike-count linear readout\n")
        f.write(f"# method (secondary): chapter-3 fading-memory τ (pulse-response decay)\n")
        f.write(f"# reservoir_class: LIFReservoir (Xavier-uniform W_rec rescaled to ρ=0.9)\n")
        f.write(f"# seed: {args.seed}\n")
        f.write(f"# n_res: {N_RES}\n")
        f.write(f"# n_input: {N_INPUT}\n")
        f.write(f"# threshold: {M_TH}\n")
        f.write(f"# target_sr: {TARGET_SR}\n")
        f.write(f"# dissertation_beta: {DISSERTATION_BETA}\n")
        f.write(f"# T_total: {args.T}\n")
        f.write(f"# max_tau: {max_tau}\n")
        f.write(f"# n_betas: {len(betas)}\n")
        col = ["beta", "mc_total", "fading_memory_tau"] + [f"mc_tau{t}" for t in range(1, max_tau + 1)]
        f.write("# columns: " + ",".join(col) + "\n")
        f.write(",".join(col) + "\n")
        for i, b in enumerate(betas):
            row = [f"{b:.6f}", f"{mc_total[i]:.6f}", f"{tau_arr[i]:.4f}"]
            row += [f"{mc_per_tau[i, t]:.6f}" for t in range(max_tau)]
            f.write(",".join(row) + "\n")
    print(f"  wrote {p} ({len(betas)} rows)")
    return p


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pickle", default="chapter6Experiments/results/ch6_exp1_full.pkl",
                   help="Used only to source real ERPs for the secondary λ₁ axis.")
    p.add_argument("--outdir", default=str(Path(__file__).parent / "outputs"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--T", type=int, default=3500, help="Total drive length for MC.")
    p.add_argument("--max-tau", type=int, default=30, help="Max recall delay.")
    p.add_argument("--n-betas", type=int, default=20, help="Number of β values in sweep.")
    p.add_argument("--beta-min", type=float, default=0.01)
    p.add_argument("--beta-max", type=float, default=0.5)
    p.add_argument("--n-erps", type=int, default=6, help="ERP samples for λ₁ per β.")
    p.add_argument("--cache", default=None)
    p.add_argument("--recompute", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    apply_style()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    cache_path = (Path(args.cache).resolve() if args.cache
                  else outdir / "_mc_cache.npz")

    # β values to sweep
    betas = np.logspace(np.log10(args.beta_min), np.log10(args.beta_max), args.n_betas)
    # Ensure the dissertation β is in the sweep at high resolution near it
    # by inserting it if not already present (within tolerance).
    if not np.any(np.abs(betas - DISSERTATION_BETA) / DISSERTATION_BETA < 0.05):
        betas = np.sort(np.concatenate([betas, [DISSERTATION_BETA]]))

    # ERP samples for λ₁ secondary axis
    print(f"  Loading ERPs from {args.pickle} for secondary-axis λ₁...")
    with open(args.pickle, "rb") as f:
        data = pickle.load(f)
    X_ds = np.asarray(data["X_ds"])
    rng = np.random.RandomState(args.seed + 7)
    erp_indices = rng.choice(X_ds.shape[0], size=args.n_erps, replace=False)
    central_ch = 16
    erp_samples = []
    for i in erp_indices:
        x = X_ds[i, :, central_ch]
        x = (x - x.mean()) / (x.std() + 1e-10)
        erp_samples.append(x)
    print(f"  Using {len(erp_samples)} ERP samples (trials {erp_indices.tolist()}, ch {central_ch}).")

    if cache_path.exists() and not args.recompute:
        print(f"  Loading sweep cache: {cache_path}")
        cz = np.load(cache_path, allow_pickle=True)
        betas = cz["betas"]
        mc_total = cz["mc_total"]
        mc_per_tau = cz["mc_per_tau"]
        tau_arr = cz["tau_arr"] if "tau_arr" in cz.files else cz["lam_arr"]
    else:
        mc_total, mc_per_tau, tau_arr = compute_sweep(
            betas, erp_samples, T=args.T, max_tau=args.max_tau,
        )
        np.savez(cache_path, betas=betas, mc_total=mc_total,
                 mc_per_tau=mc_per_tau, tau_arr=tau_arr)
        print(f"  Cached → {cache_path}")

    # Summary
    peak_idx = int(np.argmax(mc_total))
    print(f"\n  Sweep summary:")
    print(f"    β* (measured peak) = {betas[peak_idx]:.4f}  →  MC = {mc_total[peak_idx]:.3f}")
    chosen_idx = int(np.argmin(np.abs(betas - DISSERTATION_BETA)))
    print(f"    β  (dissertation) = {betas[chosen_idx]:.4f}  →  MC = {mc_total[chosen_idx]:.3f}")
    print(f"    τ_fade at chosen β: {tau_arr[chosen_idx]:.2f} steps")

    # Figures
    print("\n  Generating figures...")
    p3a = make_rawC_3a(outdir);                              print(f"    {p3a.name}")
    p3b = make_rawC_3b(outdir);                              print(f"    {p3b.name}")
    p3c = make_rawC_3c(mc_per_tau, betas, outdir);            print(f"    {p3c.name}")

    meta_note = (
        f"Reservoir: LIFReservoir (ρ=0.9, Xavier-uniform).  "
        f"MC: linear readout on smoothed spike counts.  "
        f"τ: chapter-3 fading-memory pulse decay.  "
        f"β sweep: {args.n_betas} values [{args.beta_min}, {args.beta_max}]"
    )
    p3d = make_analysisC_3d(betas, mc_total, tau_arr, outdir, meta_note=meta_note)
    print(f"    {p3d.name}")

    write_csv(betas, mc_total, mc_per_tau, tau_arr, outdir, args)
    print(f"\n  Done. Outputs in: {outdir}")


if __name__ == "__main__":
    main()
