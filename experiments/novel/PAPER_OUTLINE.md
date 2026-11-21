# Short-paper outline — "Random projection vs PCA on subject-grouped EEG"

> **Status note, added 2026-11-21.** This document is part of the v1 record and is kept
> as written. The paper it outlines was not pursued further: its central result, a
> random-projection advantage over PCA, disappears when PCA is left unwhitened and the
> readout is tuned (`experiments/v2/v2_pca_tuned.json`). See `docs/RESEARCH_MAP.md`, item 9.

A concrete skeleton assembled from the committed results. Target: a methods
short paper / letter (e.g. an ML-for-health
workshop, or a NeurIPS/ICML workshop on trustworthy ML for health). This
outline reports only what the experiments actually show, including a negative
control that bounds the mechanism, and flags the remaining gaps explicitly.

**Working title:** *When does unsupervised compression align to the grouping
nuisance? A mechanism, its operator boundary, and a leak-free alternative for
subject-grouped clinical EEG.*

## Claim (one sentence)
On participant-grouped EEG, **aggressive** unsupervised variance-maximizing
compression (global PCA) aligns its leading components to the between-subject
nuisance subspace by construction; a fixed data-independent random projection
avoids that alignment and is leak-free — and while the gross alignment does not
by itself explain the accuracy gap under the mild per-electrode operator we use,
the random projection still edges fold-local PCA on real data (+4.36pp) on
leak-safety plus mild-regime grounds.

## 1. Motivation
- Grouped clinical EEG is dominated by between-subject variance (here ρ ≈ 8–17
  subject:condition). The field's default reduce-then-classify pipeline fits PCA
  — often globally, which also leaks — before a grouped readout.
- Question: is a *fitted* basis even the right choice when the fitted variance is
  the nuisance? Contribution: a mechanism, its operator boundary (a negative
  control), a practical consequence, and honest nulls.

## 2. Setup
- Dataset: SHAPE affective EEG under two label granularities of the *same*
  cohort (3-condition 211/633, 4-category 210/840 — same participants and
  acquisition, not independent cohorts). Fixed LIF-reservoir BSC6 features
  (1,536/electrode). One genuinely independent external cohort: TCRZEM IAPS
  (228 subj) — see `experiments/external/`.
- Estimator: 5×5 participant-grouped `StratifiedGroupKFold`, fold-local
  standardization, linear readout; participant-cluster bootstrap intervals.
- Two compression operators are distinguished throughout: a **global** operator
  (flatten the full ~52k-dim representation, reduce to 64) and a **blockwise**
  per-electrode operator (each electrode 1,536→64, concatenated to 2,176) — the
  one the model actually uses.
- Compressors compared at matched output dimension: fixed sparse random
  projection (SRP), **fold-local** PCA (leak-free comparator), supervised in-fold
  LDA, random coordinate subsampling.

## 3. Result 1 — accuracy (real, both cohorts + external)
- SRP-64 > fold-local PCA-64: **+4.36pp [1.5, 7.2]** (3-cond), **+3.9pp
  [1.5, 6.4]** (4-cat); interval excludes zero. Advantage present at dims 16/32/64.
- **External (TCRZEM, 228 subj):** SRP-64 > PCA-64 by **+5.0pp [2.0, 8.0]** — the
  accuracy consequence replicates on an independent cohort.
- Figure: dimensionality sweep, SRP vs PCA with bootstrap bands, all cohorts.
- Table: SRP-64 seed-stability (64.0±1.0 / 54.1±1.1) — not a lucky projection.

## 4. Result 2 — mechanism, and its operator boundary (real + controlled)
- **Mechanism, global operator (real):** PCA-64 minimum principal angle to the
  subject subspace **1.3–1.6°** (vs 3.2–5.3° to condition); compressed
  subject:condition ρ inflated to 32–44 (ambient 11–17), SRP *below* ambient.
- **Controlled:** as nuisance grows, the (global) PCA subspace rotates
  monotonically toward subject (26.6°→0.3°) and away from condition (0.7°→29.4°).
  Dataset-independent (PCA maximizes variance = nuisance once nuisance dominates).
- **Negative control — blockwise operator (real, the key honesty result):** with
  the *same* per-electrode operator the model uses, PCA-64 sits **23.9°** from
  the subject subspace (both granularities) and its ρ (7.2 / 15.5) is *below*
  ambient. The gross alignment **does not transfer** to N1's operator. We report
  this as a measured *weaker* alignment of the shared per-electrode PCA
  read-subspace with the global subject subspace (23.9° vs the global 1.3°), not
  as a structural impossibility — the downstream linear readout can still combine
  coordinates across electrodes. **We therefore do not claim the alignment
  mechanism explains the accuracy gap.**
- Figure: principal-angle rotation curve (controlled) + real-data angle/ρ bars
  for BOTH the global and blockwise operators, side by side.

## 5. Result 3 — boundary and honest nulls
- **Operator dependence (the headline boundary):** the alignment mechanism is a
  property of aggressive global compression, not of mild per-electrode
  compression. State it as a scope limit, not a universal law.
- **Regime dependence (M1b/M3):** the controlled model does not reproduce an
  SRP>PCA accuracy crossover; in mild regimes fold-local PCA is the best
  compressor. Do not claim universality.
- **Reservoir is not the disentangler (M5):** raw-ERP ρ ≤ BSC6 ρ on both
  cohorts; any ρ reduction comes from the isotropic projection, not the reservoir.
- **Necessity map (N3):** temporal-bin order is the load-bearing factor
  (−26 to −28pp when destroyed); reported for interpretability, not as a causal
  account of the accuracy gap.

## 6. Practical recommendation
- For grouped-EEG pipelines that reduce reservoir/BSC6-style high-dimensional
  representations before a grouped readout, prefer a fixed data-independent
  projection to a fitted PCA front-end: it removes a transductive-leakage foot-gun
  and, across the SHAPE cohort (two label granularities) and one independent
  external cohort, does not lose (and modestly gains) grouped accuracy. Scope the
  recommendation to the measured BSC6 regimes rather than all clinical EEG. The
  strongest argument is leak-safety; the accuracy gain is a bonus, not attributed
  to gross nuisance alignment under the operator used.

## 7. Limitations (state plainly)
- **Modest absolute accuracy.** The contribution is the compression *choice* and
  its geometry, not state-of-the-art affect decoding (raw ERP is stronger).
- **Accuracy-mechanism gap.** The clean alignment mechanism is demonstrated for
  the global operator; the accuracy gap we report uses the blockwise operator,
  where the gross mechanism is absent. The paper's honesty hinges on presenting
  both, not conflating them.
- **Linear readout only; one feature family** (reservoir BSC6). The mechanism
  argument extends to any high-dim grouped feature space but is shown here on one.
- **Descriptive-ρ reproducibility.** Accuracies reproduce exactly; SVD-based ρ can
  drift ~1 unit across BLAS/LAPACK builds. Conclusions are invariant to that.

## 8. Reproducibility
- All estimators verified on synthetic data without the restricted dataset
  (`verify_novel.py` 13/13, `verify_nuisance.py` 14/14, `verify_external.py`
  11/11, in CI); every number is in `results_manifest.json` with its generating
  script and cross-checked by `check_novel_manifest.py`; no participant-level
  data is distributed.

## Figures / tables checklist
- [x] F1 dimensionality sweep SRP vs PCA (both cohorts) — data in `novel_results_*.json`
- [x] F2 principal-angle rotation (controlled) — `nuisance_results.json` M1a
- [x] F3 real-data angle + ρ bars, global vs blockwise (both cohorts) — `nuisance_results*.json` M2 + M2_blockwise
- [x] T1 compressor comparison — M3
- [x] F4 external-cohort replication (TCRZEM) — `experiments/external/external_results.json`

## What is NOT claimed
- Not that ARSPI-Net is an accuracy leader (it is not; raw ERP is stronger).
- Not that SRP always beats PCA (regime-bounded).
- Not that the subject-alignment mechanism explains the accuracy gap under the
  operator used (blockwise M2 refutes that); the mechanism is scoped to aggressive
  global compression.
- Not any clinical or biomarker claim.
