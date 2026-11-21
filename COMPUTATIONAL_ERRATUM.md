# Computational erratum — confirmatory classification values

> **See also, added 2026-11-21:** `DISSERTATION_ERRATUM.md` lists every other statement in
> the dissertation affected by the v2 validation and the repository audit. This file
> remains the record for the confirmatory classification values.

**Status:** authoritative for code-derived numbers. This document records a
reconciliation between the values reported in the submitted dissertation and the
values the committed pipeline in this repository actually produces.

## Why this document exists

The dissertation is a submitted degree document. Its text and tables report the
confirmatory BSC₆/SRP-64 balanced accuracy as **60.6%** (three conditions) and
**53.2%** (four categories), taken from the original computational record that
accompanied the submission (see `dissertation/main.tex`, `dissertation/chgraph.tex`,
`dissertation/chConclusion.tex`).

The committed canonical pipeline
(`experiments/confirmatory/run_confirmatory_validation.py` +
`confirmatory_pipeline.py`) is an **independent reconstruction** of that
protocol. The original analysis code is not part of this repository, and the
SRP/reservoir path was reimplemented from the described method. Run on the
authorized SHAPE data it produces **64.3%** and **54.6%** — it is close to the
dissertation on the raw-ERP and conventional-baseline rows (within ≈1.5 pp) but
diverges on the reservoir/SRP rows and, more strongly, on the destruction
controls.

Both statements are true at once. They are reconciled here rather than by
silently overwriting either record.

## Authority model (Model B: frozen dissertation + erratum)

- The **dissertation PDF and LaTeX source** remain the historical degree
  document. Their reported confirmatory values (60.6% / 53.2%) are **not edited**
  to match the reconstruction; they stand as the submitted record.
- The **committed pipeline and its execution record**
  (`experiments/confirmatory/confirmatory_results.json`) are **authoritative for
  code-derived numbers**. Where a number in this repository describes what the
  committed code produces, it is the reconstruction value.
- `results_manifest.json` stores the reconstruction values under `value` and the
  submitted values under `dissertation_reported`, with a top-level
  `confirmatory_reconciliation` note.

The `dissertation/` tree is "authoritative" for the **language- and
consistency-corrected manuscript**, not for asserting that the reported
confirmatory *values* equal the committed runner's output. That equality does
not hold and is not claimed.

## Reconciliation table (BSC₆/SRP-64, balanced accuracy)

Format `3-cond; 4-cat`. **Agree** = within ≈1.5 pp; **Diverge** = larger.

| Quantity | Dissertation (submitted) | Committed runner (reproduced) | Status |
|---|---|---|---|
| Raw ERP-16 | 68.1%; 60.5% | 69.6%; 60.6% | agree (≤1.5 pp) |
| Conventional baseline | 49.4%; 36.7% | 48.7%; 36.1% | agree (≤0.7 pp) |
| BSC6/SRP-64 | 60.6% [57.8, 63.4]; 53.2% [50.6, 55.8] | **64.3% [61.1, 67.2]; 54.6% [51.9, 57.3]** | diverge |
| BSC6/SRP-64 + conventional (fusion) | 62.0%; 53.5% | 65.1%; 55.0% | diverge |
| Advantage over conventional | +11.3 pp; +16.5 pp | +15.7 pp; +18.5 pp | diverge |
| Participant-label permutation | p = 0.0099 (both) | p = 0.0099 (both) | **exact** |
| Temporal-bin-shuffle drop | −7.4 pp; −12.9 pp | −24.8 pp; −27.8 pp | diverge (large) |
| Temporal-collapse drop | −4.9 pp; −8.6 pp | −17.6 pp; −22.4 pp | diverge (large) |
| Electrode-shuffle drop | −13.9 pp; −20.1 pp | −24.1 pp; −22.0 pp | diverge |

Every confirmatory row is shown. The agreeing rows (raw ERP-16, conventional
baseline, permutation p) and the diverging rows (reservoir/SRP accuracies, fusion,
and all destruction-control magnitudes) are both present, so the scope of the
divergence is explicit. The divergence is concentrated on exactly the
SRP/reservoir path that was reimplemented and on the destruction controls that
depend on it. The largest single accuracy-row gap is BSC6/SRP-64 (≈3.7 pp,
3-cond); the raw-ERP gap is ≈1.5 pp.

## What reproduces exactly, and what drifts

- **Accuracies and the permutation p reproduce exactly** across library versions
  (verified: three-condition SRP-64 = 64.329%, PCA-64 = 59.968%, gap +4.36 pp,
  byte-identical on reruns).
- **SVD-based descriptive variance ratios (ρ) in the exploratory study** drift at
  the ~1-unit level across BLAS/LAPACK builds; the committed JSONs and manifest
  hold the values produced by the environment recorded in the execution record's
  provenance block (`package_versions`, `platform`, `python_version`), and the
  qualitative conclusions are invariant to that drift. Note the CI workflow
  installs dependency *lower bounds*, not an exact lock file, so the recorded
  environment — not the CI spec — is the authoritative version record.

## Verification

`experiments/confirmatory/check_against_manifest.py confirmatory_results.json --tol 0.05`
compares every confirmatory manifest estimand against the committed execution
record and must pass with no mismatches; CI runs exactly this. Passing tests
verify code behavior, not scientific validity.
