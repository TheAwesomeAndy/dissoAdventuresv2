# External-cohort replication — results

> **Status note, added 2026-11-21.** This document is part of the v1 record and is kept
> as written. The comparison below standardizes PCA scores before the readout,
> the setup later shown to handicap PCA (`experiments/v2/v2_pca_tuned.json`). It has not
> been rerun with the tuned control, so the random-projection advantage reported here
> should not be cited.

The two dataset-independent claims replicate on a **genuinely external** IAPS
affective-picture ERP cohort: **DataVerseNL `doi:10.34894/TCRZEM`** (Revers et
al.), 228 complete-panel participants, different lab, different acquisition
(BrainVision segmented `.vhdr/.dat`, 41 channels), same paradigm (passive IAPS
viewing, negative / neutral / positive valence). Machine-readable:
[`external_results.json`](external_results.json). Exploratory; passing tests
verify code behavior, not scientific validity.

## N1 — a data-independent random projection beats data-fitted PCA (replicated)

Balanced accuracy, 5×5 participant-grouped fold-local CV; paired
participant-bootstrap 95% interval on the difference.

| Dim | SRP | fold-local PCA | SRP − PCA | 95% interval |
|---:|---:|---:|---:|---:|
| 16 | 43.7% | 37.0% | **+6.7 pp** | [3.3, 10.2] |
| 32 | 43.7% | 38.1% | **+5.6 pp** | [2.3, 8.9] |
| 64 | 45.5% | 40.5% | **+5.0 pp** | [2.0, 8.0] |

The interval excludes zero at every dimension — the same direction and a
**larger** effect than on SHAPE (+3.9–4.4 pp).

## M2 — global PCA aligns to the subject-nuisance subspace (replicated, starker)

These are the **global** operator angles (full flattened representation → 64),
directly comparable to SHAPE's *global* M2 — **not** the blockwise per-electrode
operator N1 uses.

| Quantity (global operator) | SHAPE (211) | TCRZEM external (228) |
|---|---:|---:|
| PCA-64 min principal angle → **subject** | 1.33° | **0.89°** |
| PCA-64 min principal angle → condition | 3.21° | 21.15° |
| ρ compressed: PCA / SRP / ambient | 44 / 7 / 11 | 133 / 111 / 111 |

Under the global operator PCA-64 is **within 0.9° of the subject subspace** and
21° from the condition subspace — even more collinear with the nuisance than on
SHAPE.

> **Operator scope (same caveat as SHAPE).** External N1 uses the **blockwise**
> per-electrode operator; external M2 above uses the **global** operator. On
> SHAPE, the blockwise geometry did *not* show the gross alignment
> (`nuisance_results.json` `M2_geometry_blockwise`), so the global external
> angle here does **not** establish that alignment causes the external
> blockwise accuracy gap. The external blockwise geometry has not been run
> (the external data are not staged in this environment), so — exactly as on
> SHAPE — no causal "alignment explains the accuracy gap" claim is made for the
> external cohort.

## Honest reading of the absolute accuracies

Absolute balanced accuracies here are **lower and near three-class chance**
(33.3%): SRP-64 reaches 45.5%. This is expected and consistent with the theory,
not a contradiction:

- The external cohort is **far more subject-dominated** (ambient ρ ≈ **111** vs
  SHAPE's ≈ 11). The condition (valence) signal is a much smaller fraction of
  the variance, so *every* linear decoder is closer to chance.
- That extreme nuisance dominance is exactly the regime in which the
  nuisance-alignment mechanism predicts the **largest** PCA penalty — and indeed
  the SRP-over-PCA gap is *bigger* here (+5–6.7 pp) and the PCA→subject angle is
  *smaller* (0.89°).
- The reservoir operating point and the arousal-pooling were fixed to match
  SHAPE, not tuned to this cohort; the point of the external run is the
  **relative** compression comparison and the mechanism, both of which hold.

So the *comparative* claims (SRP > data-fitted PCA; PCA aligns to nuisance)
replicate and strengthen on an independent cohort, while the *absolute* affect-
decoding accuracy is dataset-specific and low here — reported plainly.

## What this establishes

The paper's single-dataset limitation is resolved for its central claims:

> Across two independent IAPS affective-picture EEG cohorts (SHAPE, N=211;
> Revers/TCRZEM, N=228) — different labs, montages, and acquisition systems —
> a fixed data-independent random projection significantly outperforms a
> fold-local data-fitted PCA under participant-grouped evaluation (+4–7 pp,
> intervals excluding zero). Separately, **global** PCA is strongly collinear
> with the between-subject nuisance subspace on both cohorts (principal angle
> 0.9–1.3°). The present experiments do not establish that the global alignment
> *causes* the blockwise accuracy difference; the blockwise geometry that would
> test this has not been run on the external cohort.

## Reproduce

```text
python experiments/external/verify_external.py            # machinery, no data
python experiments/external/run_tcrzem_replication.py \
    --data ./external_data/tcrzem --out external_results.json
```

The external EEG is restricted-download (CC BY-NC 4.0) and gitignored; only the
summary statistics above are committed.
