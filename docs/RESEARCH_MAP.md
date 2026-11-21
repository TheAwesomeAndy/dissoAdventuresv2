# Research map

Where each scientific idea is implemented, how it was tested, what evidence supports it,
and what its standing is today. Experiment IDs refer to `docs/EXPERIMENT_REGISTRY.md`.

Standing uses five words, and each is a statement about evidence, not about quality:

- **Confirmed**: holds under the leak-free protocol, with a committed record that
  verifies against its code.
- **Not supported**: tested properly and the claim did not hold.
- **Withdrawn**: the supporting statistic or comparison was found invalid.
- **Archived**: legacy analysis kept for provenance; known leakage, legacy input or
  uncorrected statistics. Not tested under the corrected protocol.
- **Open** or **Ongoing**: not yet settled; what is missing is stated.

Nothing here was inferred from a file name. v2 statements cite committed records. v1
statements come from the code reading in `docs/REPOSITORY_AUDIT.md`.

## 1. Summary

| # | Idea | Standing | Evidence |
|---|---|---|---|
| 1 | A leak-free participant-grouped evaluation protocol | Confirmed | CONF-01, V2-01 |
| 2 | The reservoir code decodes affective condition on held-out participants and beats band-power features | Confirmed | CONF-01, V2-01 |
| 3 | Raw ERP amplitudes decode better than the reservoir code | Confirmed | CONF-01, V2-01 |
| 4 | The spiking reservoir adds information beyond coarse temporal binning | Not supported | V2-01, V2-05 |
| 5 | The deployed reservoir has fading memory (echo-state property) | Not supported at the v1 gain | V2-01 |
| 6 | Recurrence contributes to decoding | Not supported | V2-01 |
| 7 | Temporal order and electrode identity carry the decodable signal | Confirmed, with corrected controls | V2-01 |
| 8 | Subject variance dominates condition variance in the embeddings | Confirmed qualitatively; the v1 statistic is withdrawn | V2-01 |
| 9 | Random projection beats PCA under subject nuisance | Withdrawn | V2-04 |
| 10 | Per-epoch z-scoring fabricates early and pre-stimulus decoding | Confirmed on two datasets; ongoing | V2-02, V2-05, ZS-01, ZS-02 |
| 11 | Four-level interpretability audit: the Level-1 and Level-3 statistic r = 0.82 | Withdrawn (recommended) | V2-02, INT-01 |
| 12 | Dynamical descriptors of reservoir activity | Archived | CH6-01, CH6-02 |
| 13 | Structure-function coupling kappa | Archived; null invalid; open | CH7-01, CH7-02 |
| 14 | Graph propagation over electrodes | Archived | CH5-01, CH5-4C |
| 15 | Clinical correlates | Not detected under a corrected family; legacy screens archived | V2-03, CH6-02, CH7-03 |
| 16 | External replication of the compression result | Exploratory; inherits a known handicap | EXT-01 |
| 17 | Energy efficiency of the neuromorphic front end | Never measured | none |
| 18 | Theoretical bounds on dynamics | Illustrative; labels contradicted by their data | DEF-01 |

## 2. Detail

### 1. Leak-free evaluation protocol: confirmed

- **Idea.** Every participant's observations stay in one fold; scaling, projection
  basis and readout see training participants only; the projection is data-independent.
- **Implementation.** `experiments/confirmatory/confirmatory_pipeline.py` (v1) and
  `arspinet_v2/evaluation.py` (v2, same splits and model, fitted in the training row
  space).
- **Evidence.** A code review found no train-test leakage. The v2 direct fit reproduces
  the v1 predictions exactly (`tests/test_evaluation.py`), and on SHAPE reproduces the
  v1 headline to four decimals (`tests/test_records.py`).
- **Records.** `experiments/confirmatory/confirmatory_results.json`,
  `experiments/v2/v2_results.json`.
- **Dissertation.** Chapter 5. The dissertation's own reported values differ from the
  committed pipeline's; see `COMPUTATIONAL_ERRATUM.md`.

### 2 and 3. What decodes, and how well: confirmed

- **Numbers** (balanced accuracy, three conditions and four categories): v1
  configuration 64.3% and 54.6%; conventional band power and Hjorth 48.6% and 36.0%; raw
  ERP in 16 bins 69.6% and 60.5%.
- **Statistics.** Within-participant permutation test with the headline model, 2,000
  permutations, zero exceedances.
- **Implementation.** `arspinet_v2/reservoir.py`, `arspinet_v2/features.py`;
  conventional features in `experiments/confirmatory/run_confirmatory_validation.py`.
- **Limits.** One cohort. Intervals condition on fitted models.

### 4, 5, 6. The reservoir's contribution: not supported

- **Claim tested.** That a recurrent spiking reservoir with fading memory adds
  discriminative structure to the EEG.
- **Findings.**
  - At the v1 gain (0.9) the deployed network sustains activity without input, never
    converges from two initial states, and gives a feature test-retest correlation of
    0.67 under 1% input noise, on 10 of 10 weight draws. The gain selected by the
    label-free rule is 0.025.
  - Accuracy is flat across gains from 0 to 0.9.
  - Six-bin amplitudes on the same bin edges match the reservoir code on z-scored input
    and beat it by about 3 points on input without z-scoring.
- **Implementation.** `arspinet_v2/dynamics.py` (tests and rule),
  `arspinet_v2/features.py` (`amplitude_bins`, the matched control).
- **Records.** `experiments/v2/v2_results.json`, `experiments/v2/v2_baseline_input.json`.
- **Relation to v1.** The echo-state and Lyapunov evidence in Chapter 6 (CH6-01) was
  measured on a different, weaker network (reservoir class C in
  `docs/ARCHITECTURE.md`), with a perturbation too small to flip a threshold.
- **Limits.** The rule was written after its author had seen the dynamics sweep; it
  uses no labels and the full sweep is reported.

### 7. Destruction controls: confirmed with a corrected design

- **Idea.** Remove one kind of structure from the code and measure the loss.
- **Finding.** Collapsing all timing costs 15.6 and 19.6 points; shuffling electrodes
  costs 22.6 and 21.2. A fixed bin permutation, which only relabels features, costs
  nothing (-0.8 and -0.7), as it should.
- **Implementation.** `arspinet_v2/controls.py`. v1's version mixed information removal
  with noise injection and found the shuffle costlier than the collapse.
- **Limit.** Electrode identity is unlabeled, so no anatomical reading is possible.

### 8. Subject variance: confirmed qualitatively, statistic withdrawn

- **Finding.** In the v2 reservoir code, subject accounts for 44.5% of feature variance
  and condition for 10.3% (three conditions), both far outside their permutation nulls.
- **Withdrawn.** The v1 ratio rho (reported as 7.2) has no noise floor: structureless
  features give 69. Use `arspinet_v2/nuisance.py`.

### 9. Random projection against PCA: withdrawn

- **Original claim** (NOV-01, EXT-01): SRP-64 beats fold-local PCA-64 by about 4 to 5
  points; v2 first measured 9.
- **What V2-04 found.** The gap is produced by standardizing PCA scores before a
  regularized readout, which gives near-noise directions the weight of the signal. With
  PCA unwhitened and the regularization tuned by nested CV, the best of each differ by
  +0.7 and +0.3 points with intervals spanning zero.
- **What survives.** SRP is insensitive to that scaling choice. v1's PCA-64 rows used
  the handicapped setup.
- **Record.** `experiments/v2/v2_pca_tuned.json`.
- **Stale documents.** `experiments/novel/NOVEL_RESULTS.md`, `experiments/novel/README.md`
  and `results_manifest.json` still state the advantage.

### 10. Per-epoch z-scoring artefact: confirmed on two datasets, ongoing

- **Idea.** Z-scoring an epoch over its own duration subtracts the epoch mean, which
  contains the evoked response, from every sample. Late condition differences appear at
  early and pre-stimulus latencies.
- **Mechanism and prediction.** `experiments/zscore_artifact/MECHANISM.md`, committed
  before the public-data results were seen.
- **Evidence.**
  - SHAPE: the scalp pattern in 0 to 31 ms decodes at 54.3% under v1 preprocessing and
    33.8% without it (chance 33.3%).
  - ERP CORE: across participant averages, pre-stimulus decoding is 58.6% (N170) and
    70.8% (P3) under z-scoring, permutation p at or below 0.003, and at chance without.
  - Within participants on single trials the effect is at most 2 points.
- **Implementation.** `experiments/zscore_artifact/run_erpcore.py` (`normalize`), unit
  demonstration in `tests/test_zscore_artifact.py`.
- **Records.** `experiments/v2/v2_normalization_check.json`,
  `experiments/zscore_artifact/erpcore_results.json`,
  `experiments/zscore_artifact/across_null.json`.
- **Consequence for v1.** Every window-specific number computed on z-scored epochs is
  uninterpretable as timing.
- **Open.** Systematic literature review; a cluster test over the time course; a third
  dataset; why the single-trial effect is small.

### 11. Interpretability audit statistics: withdrawal recommended

- **v1 claim.** Level 1 and Level 3 report a pre-reservoir correlation of 0.82 and a
  post-reservoir absolute correlation of 0.23.
- **Problem.** The 0.82 correlates two windows of raw z-scored EEG and never involves
  the reservoir; non-overlapping windows of a zero-mean epoch anticorrelate by
  construction; neither mapped script computes it (INT-01). The 0.23 is below the
  dissertation's own threshold.
- **Standing.** The four-level taxonomy is a framework the dissertation proposes. Its
  Level-1 and Level-3 numbers should not be cited. This is a recommendation for the
  erratum; the dissertation text is not edited here.

### 12 and 13. Dynamical descriptors and coupling: archived, open

- **Implementation.** `chapter6Experiments/`, `chapter7Experiments/`,
  `experiments/ch6_ch7_3class/`. There is no single authoritative implementation: the
  same descriptor names are defined differently in different scripts.
- **Problems.** Computed on reservoir class C, not the classification reservoir. The
  kappa null permutes electrodes as if exchangeable, which a review showed gives a 90%
  false-positive rate for spatially smooth independent fields. kappa's reliability
  across categories is near zero (ICC 0.059). Most of this stream uses legacy input.
- **What would settle it.** The electrode coordinate map from the data provider (for a
  spatially valid null) and a rerun on corrected input with one descriptor definition.

### 14. Graph propagation: archived

- **Problem.** The graph models are random-weight propagation followed by a classifier,
  never trained, with adjacency chosen on test folds in one script and test labels
  setting attention weights in another. They support no claim about graph neural
  networks.

### 15. Clinical correlates: not detected

- **Legacy.** Chapters 6 and 7 ran uncorrected screens over diagnoses without
  covariates (CH6-02, CH7-03, CH67-3C).
- **Corrected.** V2-03 fixed a family of 12 tests in the order the data provider
  suggested. Smallest Holm-corrected p = 0.34.
- **Reading.** Not detected at this sample size without covariates. Not evidence of
  absence.

### 16, 17, 18

- **External replication (EXT-01).** Run with the standardized-PCA setup later shown to
  be handicapped; accuracy near chance. Would need rerunning with V2-04's controls.
- **Energy efficiency.** Stated as motivation. No hardware energy or latency was ever
  measured, and no code for it exists.
- **Theoretical bounds (DEF-01).** Presentation figures. The measured values fall
  outside the labelled bounds.

## 3. Authoritative implementation by concept

| Concept | Use this | Not these |
|---|---|---|
| LIF reservoir | `arspinet_v2/reservoir.py` | the 16 copies in v1 scripts |
| Spike-count code | `arspinet_v2.reservoir.encode` | the neuron-major variant in `experiments/ch5_4class/` |
| Projection | `arspinet_v2/features.py` | global PCA in Chapters 4 and 5 |
| Evaluation, intervals, permutation test | `arspinet_v2/evaluation.py` | ridge-based permutation test in v1 |
| Subject and condition variance | `arspinet_v2/nuisance.py` | the ratio in `experiments/novel/` |
| Fading memory | `arspinet_v2/dynamics.py` | Chapter 6 experiment 1 |
| SHAPE loading and preprocessing | `experiments/v2/data.py` (calls the v1 confirmatory loader); `experiments/v2/data_baseline.py` for input without z-scoring | loaders in the other v1 scripts |
| Dynamical descriptors, coupling | none yet | every existing copy |

## 4. Publications and what could be verified

| Item | In the repository | Verifiable link to code |
|---|---|---|
| PhD dissertation (Lane, 2026) | LaTeX source in `dissertation/`; no PDF | Reported confirmatory values differ from the committed pipeline; reconciled in `COMPUTATIONAL_ERRATUM.md` |
| Two IEEE LISAT conference papers (2023, 2024) | not present | none |
| A. Lane, W. Tang, and B. Nelson, "On the Edge of Stability: Spiking Reservoir State-Space Encoding of Affective EEG", IEEE BIBM 2026, Dallas, TX (accepted; derived from and built on ARSPI-Net) | not present | none |
| Defense slides | `ARSPI-Net_Defense.tex`, `ARSPI-Net_Defense.pdf` | figures from `defense_figures/` |
