# Per-epoch z-scoring artefact: results

Two datasets, two decoding regimes. Values are balanced accuracy over the pre-stimulus
interval (-200 to 0 ms), where nothing about the condition can yet be in the brain, so
the correct value is chance. Records: `erpcore_results.json`, `across_null.json`, and
`../v2/v2_normalization_check.json` and `../v2/v2_baseline_input.json` for SHAPE.

## Across participants (one average per participant and condition)

Permutation null: each participant's two labels swapped with probability 1/2, 1,000
permutations, same splits.

| Input | ERP CORE N170 (faces vs cars) | ERP CORE P3 (rare vs frequent) |
|---|---:|---:|
| Baseline-corrected microvolts | 46.7% (p = 0.91) | 51.4% (p = 0.31) |
| Divided by pre-stimulus SD | 48.5% (p = 0.75) | 51.8% (p = 0.23) |
| z-scored over the whole epoch | **58.6% (p = 0.001)** | **70.8% (p = 0.001)** |
| z-scored with post-stimulus statistics | **59.6% (p = 0.003)** | **70.4% (p = 0.001)** |
| Null 95th percentile | 53.6% to 55.0% | 53.8% to 56.1% |

40 participants per paradigm. p = 0.001 is the floor for 1,000 permutations.

SHAPE (211 participants, three conditions, chance 33.3%) has no pre-stimulus samples in
its delivered epochs, so the first post-stimulus bin stands in: 54.3% for the scalp
pattern in 0 to 31 ms under v1's z-scoring, 33.8% [31.0, 36.8] without it. The
reservoir driven by 0 to 62 ms only: 44.7% z-scored, 35.1% [32.4, 37.9] without.

## Within participants (single trials)

ERP CORE, shrinkage LDA per 16 ms bin, stratified 5-fold repeated twice, participant
mean with 95% interval and one-sample t-test against 50% (the primary test fixed in
`run_erpcore.py`).

| Input | N170 | P3 |
|---|---:|---:|
| Baseline-corrected microvolts | 50.8% [49.2, 52.4], p = 0.36 | 50.0% [49.8, 50.3], p = 0.84 |
| Divided by pre-stimulus SD | 50.5% [48.9, 52.1], p = 0.53 | 49.9% [49.7, 50.1], p = 0.32 |
| z-scored over the whole epoch | 51.5% [49.9, 53.1], p = 0.084 | 50.4% [50.0, 50.9], p = 0.052 |
| z-scored with post-stimulus statistics | 52.0% [50.3, 53.7], p = 0.024 | 50.7% [50.1, 51.2], p = 0.018 |

Post-stimulus timing also moves. On P3, z-scoring creates a decoding peak near 200 ms
(58.4% against 54.3% without), ahead of the P3 itself (see `zscore_artifact.png`).

## Against the prediction

`MECHANISM.md` was committed (69b6c97) before these results were seen. It predicted a
leak under both z-scorings, none under pre-stimulus scaling, and a larger leak for P3
than N170.

- **Confirmed across participants**, all three parts: leak under both z-scorings
  (p <= 0.003), none under pre-stimulus scaling, and P3 (about 71%) larger than N170
  (about 59%).
- **Only partly confirmed within participants.** The leak is present but small (0.7 to
  2.0 points), reaches p < 0.05 only for the post-stimulus variant, and is not larger
  for P3. A likely reason, not tested here: on single trials the epoch mean is
  dominated by slow trial-to-trial fluctuation, which swamps the condition difference;
  averaging removes that noise and leaves the leaked difference. Two tests, uncorrected.

## What this supports

- Per-epoch z-scoring makes above-chance decoding appear before stimulus onset in two
  public paradigms and in SHAPE. The effect is large when decoding uses averaged
  responses across participants, the setting of SHAPE and of many clinical and
  individual-differences studies, and small for single-trial within-participant decoding.
- Scaling by the trial's own pre-stimulus SD, or not scaling, gives no leak.
- Any latency or onset claim from z-scored data needs to be rechecked.

## Limits

- Two public paradigms, 40 participants each. One classifier family per regime.
- The across-participant regime has 80 observations per paradigm, so its time courses
  are noisy. The permutation test covers the pre-stimulus mean, not individual bins.
- The literature check was three web searches, not a systematic review.
