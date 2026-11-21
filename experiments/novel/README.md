# Novel experiments: what the inspectable chain buys you

> **Status note, added 2026-11-21.** This document is part of the v1 record and is kept
> as written. Its N1 claim (random projection beats PCA) and its PCA-nuisance
> mechanism are superseded: see `experiments/v2/v2_pca_tuned.json` and
> `docs/RESEARCH_MAP.md`, item 9. The two results files in this directory already retract
> parts of this README. The committed JSON files here cannot be tied to the code that
> wrote them (`experiments/records.lock.json`, unverified records).

These experiments target the hardest reviewer question for an
interpretability-first neuromorphic model that is deliberately **not** an
accuracy leader: *if the raw ERP classifies affect more accurately, why use the
fixed, inspectable measurement chain?* Each experiment answers with a
quantitative property that a black-box model cannot expose, computed on the same
verified operators as the confirmatory stream and reported at honest scope.

## The three contributions

### N1 — A data-independent random projection beats data-fitted PCA
The confirmatory pipeline replaced a globally fitted PCA (which leaks held-out
participants) with a fixed, data-independent SRP-64. N1 shows this is not merely
a leak-free compromise — it is *better*:

- **SRP-64 > fold-local PCA-64.** Against a *fold-local* (leak-free) PCA fitted
  only on training participants, the fixed random projection is significantly
  more accurate — the paired participant-bootstrap difference *excludes zero* at
  every tested dimension (on the three-condition cohort, +4.4pp at 64 dims,
  interval [1.5, 7.2]). The mechanism: on subject-entangled EEG the
  variance-maximizing PCA directions align to subject nuisance (the ρ≈7–10
  entanglement quantified in N3), while an isotropic random projection preserves
  condition geometry. Leak-free *and* more discriminative.
- **Johnson–Lindenstrauss dimensionality law.** Balanced accuracy across
  compression dimensions {16, 32, 64, 128} traces the expected geometry-
  preservation curve.
- **Seed stability.** The reported SRP-64 is not a lucky draw: accuracy across
  many random projection seeds has a small spread (~±1pp).

*Why it matters:* on nuisance-dominated grouped clinical EEG, unsupervised
variance-maximizing compression is actively counterproductive because it fits
the nuisance; a data-independent random projection is leak-free by construction
*and* more discriminative — a clean, reusable methods result and a concrete
argument against the default PCA step.

### N2 — The reservoir is a calibrated instrument (at the decision level)
The dissertation frames the fixed reservoir as a measurement instrument. N2
quantifies that — and is careful about *where* the reproducibility lives:

- **Reproducibility is decision-level, not feature-level.** Individual SRP
  coordinates have near-zero cross-seed ICC(2,1) (≈0.00 on real data — expected
  for a random projection: no individual coordinate is privileged). What *is*
  reproducible is the readout: accuracy is stable across projection and reservoir
  seeds (~±1pp). We report both, and do not overclaim feature-wise reliability.
- **Transfer function.** Total spike output is a monotonic function of input
  amplitude (Spearman ρ = 1.0) with a quantified dynamic range — a calibratable
  dose–response.
- **Graceful degradation.** Balanced accuracy under channel dropout and additive
  noise versus the raw-ERP reference. The reservoir readout is notably
  *noise-insensitive* (near-flat under additive noise up to 1 SD), a robustness
  property of the fixed nonlinear transform.

*Why it matters:* reframes "fixed weights" from a limitation into an instrument
guarantee — decision-level reproducibility, monotonic calibration, and known
noise behavior — while stating honestly that individual random features are not
themselves reproducible.

### N3 — Representational necessity and subject/condition information
Extends the destruction controls into a systematic map of *which structure is
necessary for what*:

- For the intact SRP-64 features and under each destruction (temporal-bin order,
  temporal resolution, electrode identity), N3 reports both the **held-out
  condition-accuracy drop** and the **subject-to-condition variance ratio ρ**.
- ρ ties directly to the dissertation's reported ρ ≈ 7.2 subject-covariance
  entanglement.

*Why it matters:* an interpretability *necessity map* — evidence that the
inspectable factors are causally load-bearing for condition decoding, and a
quantitative account of how subject nuisance is distributed across them.

## Follow-on study — the nuisance-alignment account (M1–M5)

N1 showed *that* a data-independent random projection beats data-fitted PCA on
this cohort. The follow-on study explains *why*, and maps *when* it holds — the
publishable core.

- **M2 (real data, the mechanism).** On SHAPE BSC features, the PCA-64 subspace
  is almost collinear with the subject-nuisance subspace (**minimum principal
  angle 1.34°**, vs 3.38° to the condition subspace; mean 3.95° vs 10.19°) and
  concentrates subject variance (compressed subject/condition ratio **ρ = 44**
  vs ambient 11), while SRP-64 stays **below** ambient (ρ = 7.3). PCA spends its
  budget on nuisance; the isotropic random projection does not.
- **M1a (controlled, the generalization).** As the subject-nuisance magnitude
  grows, the PCA subspace rotates *monotonically* away from the condition
  subspace and toward the subject subspace (e.g. 26.6° → 0.3° to subject; 0.7° →
  29.4° to condition). This is dataset-independent: PCA maximizes variance,
  which is the nuisance once the nuisance dominates.
- **M1b (controlled, honest boundary).** The *accuracy* consequence is
  **regime-dependent**: SRP overtakes PCA only once compression is aggressive
  enough that the condition signal falls below the retained nuisance spectrum.
  We report the crossover rather than claim a universal win — the real SHAPE
  regime happens to satisfy it (+4.4pp for SRP in N1).
- **M3 (compressor comparison).** SRP vs fold-local PCA vs supervised in-fold
  LDA vs random coordinate subsampling under identical grouped CV.
- **M5 (honest null).** The neuromorphic transform does **not** itself reduce
  entanglement: raw-ERP ρ (8.7) ≤ BSC6 ρ (11.4). The disentanglement, where it
  occurs, comes from the *isotropic projection*, not the reservoir. Reported
  plainly.

*Publishable claim:* on subject-grouped clinical EEG, unsupervised
variance-maximizing compression aligns to the grouping nuisance by construction;
a data-independent random projection is leak-free and avoids that alignment.
This is stated with its mechanism (principal angles), its boundary (M1b regime),
and an honest null (M5), which is exactly the register a careful reviewer wants.

Files: `nuisance_study.py`, `run_nuisance_study.py`, `verify_nuisance.py`,
results in `NUISANCE_RESULTS.md` / `nuisance_results.json`.

## Files

| File | Purpose |
|---|---|
| `novel_pipeline.py` | Estimators (fold-local PCA, ICC, transfer function, robustness, variance ratio, necessity map), reusing the committed confirmatory operators |
| `run_novel_experiments.py` | Runs N1–N3 on authorized SHAPE data → `novel_results.json` |
| `verify_novel.py` | Synthetic verification of every estimator (CI-safe, no SHAPE data) |

## Running

```text
# machinery (no data, CI-safe)
python experiments/novel/verify_novel.py

# real data (restricted; not shipped)
python experiments/novel/run_novel_experiments.py \
    --data3 ./batch_data --granularity three_condition --out novel_results.json
```

Add `--fast` for a quicker pass (fewer dimensions and projection seeds).
`--granularity four_category` runs the four-category cohort.

## Scope

`verify_novel.py` establishes estimator behavior on synthetic data: fold-local
PCA does not leak, ICC recovers reliability, the transfer function is monotonic,
and the variance decomposition recovers a planted subject-nuisance structure. It
does **not** establish the empirical SHAPE results, which require running the
committed pipeline on the authorized restricted dataset. These experiments are
positioned as **exploratory, hypothesis-strengthening** analyses; like the rest
of the repository, passing tests verify code behavior, not scientific validity.
