# ARSPI-Net: An Interpretable Neuromorphic Framework for Clinical EEG Analysis

[![verify](https://github.com/TheAwesomeAndy/dissoAdventuresv2/actions/workflows/verify.yml/badge.svg)](https://github.com/TheAwesomeAndy/dissoAdventuresv2/actions/workflows/verify.yml)
[![License: MIT](https://img.shields.io/badge/code%20license-MIT-blue.svg)](LICENSE)
<!-- DOI badge added after the frozen release is archived. See RELEASE.md. -->

> **This is ARSPI-Net v2.** It keeps the v1 confirmatory protocol and tests what an
> external methods review found untested: the reservoir's fading memory, matched
> non-reservoir controls, the destruction controls, the permutation test and the
> subject-variance statistics. All of it was run on the authorized SHAPE data. The short
> version: the v1 reservoir has no fading memory, the v2 reservoir does, recurrence adds
> nothing measurable, six-bin amplitudes match or beat the reservoir code, random projection
> ties PCA once PCA is left unwhitened and tuned, and v1's per-epoch z-scoring destroys
> temporal localisation.
> Read [`V2_UPGRADE.md`](V2_UPGRADE.md) first. Tables are in
> [`experiments/v2/V2_RESULTS.md`](experiments/v2/V2_RESULTS.md). The text below is the
> v1 README, kept for provenance. Where it and `V2_UPGRADE.md` disagree, v2 governs.

**Dissertation:** Lane, A. A. (2026). *Affective Reservoir-Spike Processing and Inference Network (ARSPI-Net): A Four-Level Interpretable Neuromorphic Framework for Clinical EEG Analysis.* PhD Dissertation, Department of Electrical and Computer Engineering, Stony Brook University.

ARSPI-Net is a neuromorphic signal-processing framework for transforming multichannel EEG into explicit temporal, dynamical, and graph-organized representations. A fixed-weight leaky integrate-and-fire reservoir maps continuous EEG to spike trains. A six-bin spike-count code, BSC6, preserves coarse temporal order. A fixed data-independent projection compresses each electrode representation. Named dynamical descriptors, graph quantities, and structure-function coupling statistics remain available for direct inspection.

The primary contribution is a characterized measurement chain, not a state-of-the-art classification claim. On the SHAPE Community cohort, the reported raw ERP reference exceeds the reported reservoir representation. ARSPI-Net is therefore positioned as a decomposable neuromorphic transform whose intermediate variables support mechanistic tests of preserved, transformed, and suppressed signal structure.

## Scope

The repository and corrected dissertation use the following scope restrictions.

- The four-level interpretability taxonomy is an audit framework proposed in the dissertation. It is not an externally standardized taxonomy.
- Cross-architecture comparisons are conceptual unless they use a stated common protocol and dataset.
- Clinical-label effects are cohort-level, exploratory, and hypothesis-generating. They do not establish diagnostic subtypes, patient-level interpretation, or validated biomarkers.
- The corrected classification estimands use participant-grouped cross-validation, fold-local model fitting, and a fixed data-independent SRP-64 projection.
- Legacy PCA-64, subject-centered, 10-fold, and message-passing results are retained as archived descriptive results. They are not used for the primary inference.
- Generalization is restricted to the measured cohort until independently validated.
- Energy efficiency motivates the architecture but is not a measured result because hardware energy and latency were not evaluated.

## Publications

1. Lane, A., Nelson, B. D., and Tang, K. W. (2023). “Towards ARSPI-Net: Development of an Efficient Hybrid Deep Learning Framework.” *Proceedings of the IEEE Long Island Systems, Applications and Technology Conference*.
2. Lane, A., Nelson, B. D., and Tang, K. W. (2024). “Towards ARSPI-NET: Advancing EEG Feature Extraction with Neuromorphic Algorithms.” *Proceedings of the IEEE Long Island Systems, Applications and Technology Conference*.
3. Lane, A. A., Tang, K. W., and Nelson, B. D. (2026). “Subject-Covariance Entanglement in Affective EEG Embeddings.” *IEEE Signal Processing Letters*, submitted and under review.

## Authoritative dissertation source

The LaTeX source is under `dissertation/`. The authoritative files for the **language- and consistency-corrected manuscript** are `main.tex`, the chapter files, `Appendixa.tex`, and `Disso.bib`. `dissertation/AUDIT_REPORT.md` records the language and consistency audit. `ArspiNetFinalVer_.zip` is retained only as an earlier packaged source. Where the package differs from `dissertation/`, the corrected `dissertation/` source governs.

**Authority split for confirmatory numbers.** The `dissertation/` tree is authoritative for the corrected *manuscript*, not for asserting that its reported confirmatory *values* equal the committed runner's output. The dissertation reports BSC₆/SRP-64 = 60.6% / 53.2% (the submitted record); the committed pipeline is an independent reconstruction that reproduces 64.3% / 54.6%. These are reconciled in [`COMPUTATIONAL_ERRATUM.md`](COMPUTATIONAL_ERRATUM.md): the dissertation source is **not** edited to the reconstruction values, and the committed execution record is authoritative for code-derived numbers.

## Dissertation map

| Chapter | Scientific object | Source | Primary code and outputs |
|---|---|---|---|
| 1 | Motivation and problem formulation | `dissertation/ch1.tex` | `pictures/ch1/` |
| 2 | Neuromorphic, graph, and EEG foundations | `dissertation/ch2.tex` | `pictures/ch2/` |
| 3 | LIF reservoir dynamics and operating point | `dissertation/chLSM.tex` | `experiments/chapter3/`, `pictures/chLSM/` |
| 4 | Spike-to-embedding transformation | `dissertation/chLSMEmbeddings.tex` | `chapter4Experiments/`, `pictures/chLSMEmbeddings/` |
| 5 | Classification and graph propagation | `dissertation/chgraph.tex` | `chapter5Experiments/`, `experiments/ch5_4class/` |
| 6 | Dynamical descriptors | `dissertation/chDynamics.tex` | `chapter6Experiments/`, `experiments/ch6_ch7_3class/` |
| 7 | Structure-function coupling | `dissertation/chSynthesis.tex` | `chapter7Experiments/`, `experiments/ablation/` |
| 8 | Synthesis and limitations | `dissertation/chConclusion.tex` | `pictures/conclusion/` |
| A | Supplementary analyses | `dissertation/Appendixa.tex` | chapter-specific scripts and figures |

Every indexed numerical result is recorded in `results_manifest.json`. The script-to-result mapping and inferential tiers are documented in `docs/REPRODUCTION_MAP.md`.

## Confirmatory estimands (reproduced vs dissertation-reported)

**Protocol:** five repeats of five-fold participant-grouped cross-validation. Each participant’s complete condition panel remains in one fold. Per-feature standardization and the linear readout are fitted on training participants only. BSC6 is compressed from 1,536 to 64 coordinates with a fixed `SparseRandomProjection` generated from a prespecified seed. Reported intervals are participant-cluster bootstrap intervals over repeated out-of-fold predictions. A separate deterministic ridge readout is used for 100 within-participant label permutations.

**Cohorts:** 211 participants and 633 observations for Negative, Neutral, and Pleasant; 210 participants and 840 observations for Threat, Mutilation, Cute, and Erotic.

> **Reproduction note (important).** The numbers below are what the **committed pipeline** (`experiments/confirmatory/run_confirmatory_validation.py`) actually produces on the authorized SHAPE data, recorded in `experiments/confirmatory/confirmatory_results.json` and checked into `results_manifest.json` (`check_against_manifest.py confirmatory_results.json --tol 0.05` passes). The **dissertation reports different figures** (e.g. BSC6/SRP-64 60.6% / 53.2%), preserved in the manifest under `dissertation_reported`. The committed pipeline is an **independent reconstruction**: it matches the dissertation on the raw-ERP and conventional rows (within ~1 pp) but runs **~1.4–3.7 pp higher on the BSC6/SRP-64 and fusion rows** and gives markedly larger destruction-control drops, because the original analysis code is not in this repository and the SRP/reservoir path is reimplemented. Where this README states a confirmatory number, it is the reproduced value; the dissertation value is given in parentheses. The full authority model and a row-by-row reconciliation are in [`COMPUTATIONAL_ERRATUM.md`](COMPUTATIONAL_ERRATUM.md).

| Representation | Three conditions (dissertation) | Four categories (dissertation) |
|---|---:|---:|
| Raw ERP, 16 temporal bins | 69.6% [66.5, 72.5] (68.1) | 60.6% [57.9, 63.1] (60.5) |
| Conventional spectral and Hjorth features | 48.7% [45.9, 51.5] (49.4) | 36.1% [33.9, 38.4] (36.7) |
| **BSC6 / SRP-64** | **64.3% [61.1, 67.2]** (60.6) | **54.6% [51.9, 57.3]** (53.2) |
| BSC6 / SRP-64 plus conventional features | 65.1% [61.9, 68.0] (62.0) | 55.0% [52.1, 57.8] (53.5) |
| Chance | 33.3% | 25.0% |

Reproduced BSC6/SRP-64 advantage over the conventional baseline: +15.7 pp [11.9, 19.5] for three conditions (dissertation +11.3) and +18.5 pp [15.2, 21.7] for four categories (dissertation +16.5). The participant-label permutation result **reproduces exactly**: p = 0.0099 at both granularities.

### Mechanistic destruction controls (reproduced; dissertation in parentheses)

| Control | Three conditions | Four categories |
|---|---:|---:|
| Shuffle temporal-bin order within observation and electrode | -24.8 pp [-28.8, -21.1] (-7.4) | -27.8 pp [-31.1, -24.4] (-12.9) |
| Collapse temporal resolution | -17.6 pp (-4.9) | -22.4 pp (-8.6) |
| Shuffle electrode identity | -24.1 pp (-13.9) | -22.0 pp (-20.1) |

Ordered temporal structure and stable electrode identity contribute to held-out discrimination (every destruction reduces accuracy). The reproduced drops are substantially larger than the dissertation-reported ones; the two use different control implementations. These do not establish an anatomical interpretation for unlabeled channel identities.

## Confirmatory reproduction status

The canonical implementation is committed under `experiments/confirmatory/`:

- `confirmatory_pipeline.py` defines the participant-grouped estimator, fixed projection, repeated out-of-fold pooling, bootstrap estimators, permutation test, and destruction controls.
- `run_confirmatory_validation.py` reconstructs the audited SHAPE epochs, applies the specified exclusions, and writes `confirmatory_results.json`.
- `verify_confirmatory.py` tests the machinery on synthetic data without requiring restricted records.
- `check_against_manifest.py` compares an authorized-data execution against every confirmatory result recorded in `results_manifest.json`.

The SHAPE data are restricted and are not included. Continuous integration verifies implementation behavior and preprocessing boundaries, not the empirical headline values. Exact agreement with the reported estimates must be established on the authorized data before the frozen release:

```text
python experiments/confirmatory/run_confirmatory_validation.py \
    --data3 ./batch_data \
    --data4 ./categories \
    --out confirmatory_results.json

python experiments/confirmatory/check_against_manifest.py \
    confirmatory_results.json --tol 0.05
```

## Interpretability audit measurements

| Level | Observable | Reported measurement | Chapter |
|---|---|---|---|
| 1 | Temporal traceability | pre-reservoir LPP r = 0.82; post-reservoir absolute r = 0.23 | 4 |
| 2 | Representation geometry | subject-to-condition variance ratio rho = 7.2 | 5 |
| 3 | Dynamical correspondence | amplitude-domain absolute r = 0.82; spike-domain absolute r = 0.23 | 6 |
| 4 | Structure-function coupling | per-observation median kappa = 0.273; subject-level median approximately 0.22 | 7 |

The two coupling values are distinct estimands. The 0.273 value is the median across 844 four-category observations. The approximately 0.22 value is the median across 211 subject-averaged three-condition values. They differ in sample unit, granularity, and processing stream.

## Archived results

The repository retains legacy results for provenance. These include globally fitted PCA, transductive subject-mean centering, legacy input slicing, dependent folds, and the original message-passing sweep. They are descriptive and must not be interpreted as prospective performance.

| Archived method | Uncentered | Transductive centered |
|---|---:|---:|
| Band power plus SVM | 47.7% | 61.0% |
| Raw EEG PCA-200 plus logistic regression | 64.9% | 86.4% |
| Raw EEG plus logistic regression | 70.5% | 88.4% |
| LSTM | 58.0% | 71.1% |
| GRU | 59.9% | 78.4% |
| EEGNet | 72.0% | 89.1% |
| Reservoir plus logistic regression | 59.4% | 78.8% |

The archived message-passing sweep reports 63.4% for PCA-64 concatenation plus SVM and 49.8%, 56.4%, and 48.5% for GIN, GCN, and EdgeConv. Global preprocessing and dependent folds prevent a universal graph-regime claim.

## Repository structure

```text
dissoAdventuresv2/
├── README.md
├── LICENSE
├── LICENSES.md
├── CITATION.cff
├── RELEASE.md
├── results_manifest.json
├── dissertation/
├── chapter4Experiments/
├── chapter5Experiments/
├── chapter6Experiments/
├── chapter7Experiments/
├── experiments/
│   ├── chapter3/
│   ├── confirmatory/
│   ├── novel/            # leak-free compression + nuisance-alignment study
│   ├── external/         # independent-cohort replication (TCRZEM)
│   ├── ch5_4class/
│   ├── ch6_ch7_3class/
│   ├── ablation/
│   └── interpretability/
├── validation/
├── docs/
├── scripts/
└── pictures/
```

## Data format

The SHAPE study is described at <https://lab-can.com/shape/>. The restricted files are not distributed in this repository.

**Three-condition directory, `batch_data/`:**

- filename: `SHAPE_Community_{SUBJECT_ID}_IAPS{Neg,Neu,Pos}_BC.txt`
- shape: 1229 time samples by 34 channels
- format: baseline-corrected text, read with `numpy.loadtxt`

**Four-category directory, `categories/`:**

- filename: `SHAPE_Community_{SUBJECT_ID}_IAPS{Neg,Pos}_{Threat,Mutilation,Cute,Erotic}_BC.txt`
- shape: 1229 time samples by 34 channels
- format: baseline-corrected text, read with `numpy.loadtxt`

The canonical runner removes samples 0 through 204, decimates samples 205 through 1228 by four to 256 samples, and z-scores each epoch and channel. Subject 127 is excluded from both granularities. Subject 195 is additionally excluded from the four-category cohort.

## Dependencies

```text
numpy>=2.0
scipy>=1.13
scikit-learn>=1.5
pandas>=2.2.3
matplotlib>=3.9
torch>=1.12  # canonical PyTorch baselines only
```

## Verification

The GitHub Actions workflow runs the 491 declared data-independent checks and uploads per-suite logs. These checks establish that the scripts execute, the operators satisfy their stated implementation contracts, and fixed seeds produce deterministic outputs. They do not establish that an empirical result is true, clinically valid, externally generalizable, or reproduced from restricted data.

| Suite | Checks |
|---|---:|
| Chapter 4 | 31 |
| Chapter 5 core | 32 |
| Chapter 5 Experiment Zero | 37 |
| Chapter 5 reproduction | 33 |
| Chapter 5 baselines | 55 |
| Chapter 6 core | 31 |
| Chapter 6 reproduction | 42 |
| Chapter 7 experiments | 38 |
| Chapter 7 utilities | 40 |
| Four-category extension | 25 |
| Three-condition Chapter 6/7 extension | 28 |
| Ablation | 23 |
| Data validators | 20 |
| Confirmatory pipeline | 18 |
| Novel compression experiments | 13 |
| Nuisance-alignment study | 14 |
| External-replication harness | 11 |
| **Total** | **491** |

The workflow additionally runs three consistency gates that compare committed
result files against `results_manifest.json` rather than exercising code:
`experiments/confirmatory/check_against_manifest.py` (confirmatory rows),
`experiments/novel/check_novel_manifest.py` (novel, nuisance, and external
rows), and `validation/validate_dissertation_claims.py` (repository-to-
dissertation consistency). A green run therefore also establishes that the
committed JSON outputs and the manifest agree.

See `docs/VERIFICATION_METHODOLOGY.md`, `docs/REPRODUCTION_MAP.md`, and `results_manifest.json` for the verification boundary and result provenance.

## Licensing and citation

Software and scripts are released under the MIT License. Dissertation text, original figures, third-party material, and restricted data have separate terms described in `LICENSES.md`. Citation metadata are in `CITATION.cff`. The frozen release and DOI procedure is specified in `RELEASE.md`.
