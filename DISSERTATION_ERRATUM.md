# Erratum to the dissertation

**Dissertation.** Lane, A. A. (2026). *Affective Reservoir-Spike Processing and Inference
Network (ARSPI-Net): A Four-Level Interpretable Neuromorphic Framework for Clinical EEG
Analysis.* PhD dissertation, Stony Brook University.

**Status.** Draft prepared on 2026-11-21 for the author's review. It has not been filed
with the university or any repository of record. The dissertation source in
`dissertation/` is not edited by this erratum or by any change that accompanies it.

**Scope.** This erratum lists statements in the dissertation that later work in this
repository contradicts, qualifies or cannot support. Three sources:

1. the v2 validation, which reran the dissertation's central analyses on the same
   SHAPE data with matched controls (`V2_UPGRADE.md`);
2. a line-by-line reading of the analysis code (`docs/REPOSITORY_AUDIT.md`);
3. defects found in that code and since fixed (`docs/v1/DEFECT_FIXES.md`).

The earlier `COMPUTATIONAL_ERRATUM.md` covers one further point, the confirmatory
classification values, and remains in force (item 11 below).

**How to read it.** Locations are `file:line` in `dissertation/` as committed at
`8573cf0`, with the chapter in brackets: ch1.tex [1], ch2.tex [2], chLSM.tex [3],
chLSMEmbeddings.tex [4], chgraph.tex [5], chDynamics.tex [6], chSynthesis.tex [7],
chConclusion.tex [8], Appendixa.tex [A], main.tex [abstract]. Each item gives
what is stated, what is wrong, a corrected statement, and the basis. "Measured" means
computed on the SHAPE data with a committed record. "Read" means established from the
code. "Inferred" marks reasoning that was not run.

**What is not affected.** The corrected evaluation protocol (participant-grouped
cross-validation with fold-local fitting), the finding that the reservoir code decodes
affective condition far above chance and above band-power features, the finding that
raw ERP amplitudes decode better still, and the dissertation's own scope restrictions.
Many passages already carry "archived" or "descriptive" wording; this erratum does not
repeat points the text already concedes.

---

## Part I. Corrections of substance

### 1. The classification reservoir does not have fading memory

| Location | Statement |
|---|---|
| `ch1.tex:184` [1] | the framework "evaluates them through echo-state verification, cross-initialization reliability, and surrogate testing" |
| `chLSM.tex:543` [3] | "The spiking regime validated in Module 2 is more likely to satisfy ESP than the subthreshold regime" |
| `chDynamics.tex:316` [6] | "echo-state and generalized-synchronization arguments establishing initial-condition independence and input determinism" |
| `chDynamics.tex:427` [6] | echo-state verification is listed as a required step; no result is reported |
| `chConclusion.tex:69` [8] | the pipeline "establishes local fading of initial conditions" |
| `chgraph.tex:947` [5] | the classification reservoir has "spectral radius = 0.9" |

**What is wrong.** The dissertation reports no echo-state measurement on the network
that produced the classification features. Measured on that network, with the EEG input
it actually receives, at spectral radius 0.9:

- the recurrent current is 3.6 times the input current;
- activity sustains itself after the input stops (population rate 0.106);
- none of 300 test sequences converges from two different initial states;
- the feature test-retest correlation under 1% input noise is 0.67;
- all of this holds for 10 of 10 weight draws.

The dynamical analyses of Chapter 6 were run on a different network (read from
`chapter6Experiments/`): sparse, Gaussian, with no spectral-radius scaling and a
different reset rule. `chDynamics.tex:84` describes "sparse random connectivity" without
saying that this differs from the dense network of Chapter 5. The Benettin estimator
described at `chDynamics.tex:150` uses an initial separation of 1e-8, which cannot flip
a spike threshold and therefore measures leak and reset only (inferred from the method).

**Corrected statement.** The reservoir used for classification does not have the
echo-state property on SHAPE input at spectral radius 0.9. Local contraction
(`thm:conditional_lif_contraction`, `chDynamics.tex:367`) holds only on regions where
spike decisions do not change, and that condition fails here. A recurrent gain of 0.025
passes a label-free fading-memory test on 10 of 10 draws. Held-out accuracy is the same
at every gain from 0 to 0.9, so recurrence contributes nothing measurable to decoding.

**Basis.** Measured: `experiments/v2/v2_results.json`, sections 1 and 3 of
`experiments/v2/V2_RESULTS.md`.

### 2. The reservoir does not add separability over its input

| Location | Statement |
|---|---|
| `chLSMEmbeddings.tex:71` [4] | section title "Quantitative Verification of Separability Enhancement" |
| `chLSMEmbeddings.tex:82` [4] | Fisher ratio 0.13 for raw and filtered input, 1158 for the reservoir code |
| `chLSMEmbeddings.tex:224` [4] | "an 8,900x separability improvement over the raw input" |
| `chgraph.tex:262` [5] | "The pipeline functions as a progressive signal extractor." |
| `chgraph.tex:353` [5] | "The reservoir improves all three classes" |
| `chgraph.tex:817` [5] | "confirming the reservoir's advantage under linear readout" |
| `chgraph.tex:922` [5] | reservoir features provide characterization "that raw amplitudes cannot" |

**What is wrong.**

- The Fisher ratio is inflated whenever features outnumber trials. A review of the
  Chapter 4 code obtained a ratio of 296 from pure noise at 1,536 features and 400
  trials, against 0.01 at 198 features (simulated by the reviewer; not rerun here). The
  ratio of 1158 to 0.13 compares representations of different dimension and is not an
  effect size, as `chLSMEmbeddings.tex:87` itself notes.
- Against a matched control the reservoir shows no advantage. Six-bin mean amplitudes on
  the reservoir's own bin edges, with one tenth of the features, score 67.1% and 55.1%
  (three conditions, four categories). The reservoir code scores 65.1% and 55.4%
  averaged over ten weight draws. On input that is not z-scored per epoch (item 4) the
  reservoir loses to the same control by 3.2 and 3.0 points, with intervals excluding
  zero.

**Corrected statement.** The reservoir code is statistically indistinguishable from
six-bin amplitude averages of its input on the dissertation's preprocessing, and worse
on preprocessing that does not z-score each epoch. Raw ERP amplitudes in 16 bins are
better than both, as the dissertation already reports.

**Basis.** Measured: `experiments/v2/v2_results.json`,
`experiments/v2/v2_baseline_input.json`.

### 3. The correlation r = 0.82 does not describe the reservoir

| Location | Statement |
|---|---|
| `ch2.tex:380` [2] | Level 1 criterion "at least 0.5"; "Pre-reservoir: |r| = 0.82 (LPP) ... Post-reservoir (BSC6 spike bins): |r| = 0.23 (LPP)" |
| `ch2.tex:384` [2] | Level 3, same criterion and values |
| `ch2.tex:373` [2] | "All thresholds were specified before validation." |
| `chLSMEmbeddings.tex:265`, `285` [4] | the bin profile "corresponds to conventional ERP summaries with ... |r| = 0.82 for LPP" |
| `chgraph.tex:922`, `972` [5] | pre-reservoir 0.82, post-reservoir 0.23 |
| `chDynamics.tex:575` [6] | table row "Mean amplitude ... -0.820" |
| `chConclusion.tex:140`, `152`, `154` [8] | "LPP (|r| = 0.82)" |

**What is wrong.**

- The 0.82 value correlates amplitude bins of the ERP with late positive potential
  amplitude. Both are properties of the input. The reservoir is not involved, yet
  `chLSMEmbeddings.tex:285` attributes the value to the architecture's Level 1.
- The value is negative (`chDynamics.tex:575`) and is printed unsigned elsewhere.
- Under per-epoch z-scoring, two non-overlapping windows of a zero-mean epoch are
  anticorrelated by construction (inferred; the mechanism is item 4).
- The post-reservoir value, 0.23, is below the 0.5 criterion that `ch2.tex:380` and
  `384` set. By the dissertation's own thresholds the reservoir representation does not
  meet Level 1 or Level 3.
- In the repository, neither script the results manifest names for this value computes
  it; in one it is a printed constant (read).

**Corrected statement.** Levels 1 and 3 of the interpretability audit are not met by the
reservoir representation (post-reservoir |r| = 0.23 against a criterion of 0.5). The
0.82 value describes the input and should not be cited as a property of ARSPI-Net.

**Basis.** Read: `docs/REPOSITORY_AUDIT.md`, finding I2. Measured:
`experiments/v2/v2_normalization_check.json`.

### 4. Statements about timing rest on a normalization that moves information in time

| Location | Statement |
|---|---|
| `chgraph.tex:150` [5] | "per-channel z-score normalization" of each epoch |
| `chgraph.tex:426` [5] | "60.5% for 39-273 ms, 56.9% for 273-508 ms, 56.7% for 508-742 ms, 48.5% for 742-977 ms ... the earliest window is the strongest isolated segment" |
| `main.tex:114` [abstract] | "discriminative information is distributed across the post-stimulus response" |
| `chLSMEmbeddings.tex:275` [4], `chgraph.tex:930` [5] | bin-resolution sweep with peak bins at 234-273 ms, 176-195 ms and 172-180 ms |
| `chgraph.tex:953` [5] | attention peaks in the "first window (39-78 ms ...)" and "final window (234-273 ms ...)" |
| `ch2.tex:354` [2] | attention "peaks within the ERP window (176-254 ms)" |

**What is wrong.** Z-scoring each epoch over its own duration subtracts the epoch mean,
which contains the late evoked response, from every sample. Late condition differences
are thereby written into early samples.

- On the dissertation's preprocessing, the scalp pattern in the first 31 ms after
  picture onset decodes the condition at 54.3% (chance 33.3%). No cortical response to
  picture content exists at that latency.
- On the same data without per-epoch z-scoring, that bin is at chance (33.8%, interval
  31.0 to 36.8). Decoding rises from about 125 ms and peaks at 62.2% between 560 and
  690 ms.
- With input that is not z-scored, the reservoir driven by the first 62 ms alone is at
  chance (35.1% and 23.9%), against 44.7% and 36.3% with z-scoring.
- The effect replicates on public data (ERP CORE): across participant averages,
  pre-stimulus decoding is 58.6% and 70.8% under per-epoch z-scoring against a 50%
  chance level, and at chance without it.

Without z-scoring, the windows order the other way: amplitude decoding is 46.5% for 0 to
250 ms, 63.2% for 250 to 500 ms and 63.1% for 500 to 750 ms. The statement that the
earliest window is the strongest is an effect of the normalization.

**Corrected statement.** Window-specific accuracies, peak-bin latencies and attention
latencies computed on per-epoch z-scored data cannot be interpreted as timing. With
baseline-referenced input, decodable information is concentrated between roughly 250 and
750 ms, the range of the late positive potential. Whole-epoch accuracy is not affected
by the normalization (69.6% against 68.8% for raw ERP).

**Basis.** Measured: `experiments/v2/v2_normalization_check.json`,
`experiments/v2/v2_baseline_input.json`,
`experiments/zscore_artifact/erpcore_results.json`,
`experiments/zscore_artifact/across_null.json`.

### 5. PCA is not shown to be aligned with subject variance or to be lossy

| Location | Statement |
|---|---|
| `ch1.tex:182` [1] | "PCA-based nuisance regression is provably lossy" |
| `chgraph.tex:575` [5] | subject and condition information "share the same principal components ... they are entangled" |
| `chgraph.tex:915` [5] | "perfect alignment: the first five principal angles ... are all 0 degrees" |
| `chgraph.tex:917` [5] | "identifies PCA compression, not the reservoir transformation, as the stage that destroys ERP-component traceability" |
| `chgraph.tex:974` [5] | "PCA-64 features cannot be centered without losing condition signal" |

**What is wrong.**

- Principal angles near zero and a large subject-to-condition ratio also arise from
  structureless data of this shape; a review obtained angles near 1 degree and a ratio
  near 105 from pure noise (simulated by the reviewer). The dissertation reports no null
  for either quantity. The repository's own later study measured 23.9 degrees for the
  fold-local, per-electrode PCA (`experiments/novel/NUISANCE_RESULTS.md`).
- PCA-64 loses 8.6 and 9.7 points to a fixed random projection only when its component
  scores are standardized before the readout, which gives near-noise directions the
  weight of the signal. Left at their natural scale, with the readout regularization
  tuned by nested cross-validation, PCA-64 reaches 67.2% and 57.3% against 67.9% and
  57.6% for the random projection. The differences are +0.7 and +0.3 points with
  intervals spanning zero.
- Fitting the PCA basis after removing each training participant's mean does not change
  its accuracy (57.8% and 46.7%).

**Corrected statement.** The data do not show that PCA is aligned with the subject
subspace or that it discards condition information. The accuracy loss attributed to PCA
is produced by whitening its scores before a regularized readout.

**Basis.** Measured: `experiments/v2/v2_pca_tuned.json`,
`experiments/v2/v2_results.json`.

### 6. The subject-to-condition ratio of 7.2 has no noise floor

| Location | Statement |
|---|---|
| `ch2.tex:382` [2] | "rho = SS_subj / SS_cond ... rho = 7.2" |
| `chgraph.tex:556` [5] | subject 62.6%, condition 8.7%, residual 28.7%, "a 7x subject-to-condition ratio" |
| `ch1.tex:182`, `ch2.tex:172`, `chgraph.tex:980`, `chConclusion.tex:140`, `153`, `167`, `173`, `Appendixa.tex:48` | the same ratio, or 19.6x for four categories |

**What is wrong.** A ratio of between-subject to between-condition variance is large for
features with no structure at all, because a mean over a few observations always has
variance. Gaussian noise of the same shape gives a ratio of 69. The real embedding gives
5.6, which is smaller.

**Corrected statement.** Unbiased variance components from the two-way subject by
condition layout, which are zero for structureless features: subject 44.5%, condition
10.3%, residual 45.2% for three conditions, and 38.7%, 5.1%, 56.3% for four categories,
all far outside their permutation nulls. Subject variance does exceed condition
variance. The ratio 7.2 should not be used to quantify it.

**Basis.** Measured: `experiments/v2/v2_results.json`, section 9 of
`experiments/v2/V2_RESULTS.md`.

### 7. Destruction controls

| Location | Statement |
|---|---|
| `chgraph.tex:424` [5], `main.tex:114`, `chConclusion.tex:90` [8] | bin shuffle lowers accuracy by 7.4 and 12.9 points; collapse by 4.9 and 8.6; electrode shuffle by 13.9 and 20.1 |
| `main.tex:114`, `chgraph.tex:877`, `1022` | a "population-count" control is named |

**What is wrong.** These controls destroy training and test data with an independent
permutation for each observation, which mixes the removal of information with the
injection of noise. A control that removes strictly more temporal information (collapse)
then appears to cost less than one that removes less (shuffle). No value is reported for
the population-count control anywhere in the text.

**Corrected statement.** With the model refitted on destroyed data, removing all
within-epoch timing costs 15.6 and 19.6 points, and shuffling electrode identity costs
22.6 and 21.2. A fixed bin permutation, which only relabels features, costs nothing
(-0.8 and -0.7), as it should. Ordered timing and stable electrode identity both carry
decodable signal. The magnitudes in the dissertation should be replaced by these.

**Basis.** Measured: `experiments/v2/v2_results.json`, section 7 of
`experiments/v2/V2_RESULTS.md`.

### 8. The permutation p-value is a floor

| Location | Statement |
|---|---|
| `chgraph.tex:401`, `403` [5]; `main.tex:112`; `chgraph.tex:13`, `794`, `1020`; `chConclusion.tex:15` | "The ridge permutation tests give p = 0.0099 at both granularities" from 100 permutations |

**What is wrong.** 0.0099 is 1/101, the smallest value 100 permutations can give. The
test used a ridge readout that scores 3 to 8 points below the model whose accuracy is
reported.

**Corrected statement.** With the reported model and 2,000 within-participant
permutations there are no exceedances at either granularity (p = 0.0005, again the
floor); the observed accuracy lies 16.9 and 19.2 null standard deviations above the
null mean. The conclusion is unchanged.

**Basis.** Measured: `experiments/v2/v2_results.json`.

### 9. Structure-function coupling

| Location | Statement |
|---|---|
| `chSynthesis.tex:68` [7] | the null permutes electrode rows; "For each of 5,000 permutations" |
| `chSynthesis.tex:138` [7] | "confirmatory subject-level analysis ... (p < 0.001, one-sample Wilcoxon against zero)" |
| `chSynthesis.tex:154` [7] | observed median 0.273 against a null median 0.139 |
| `chSynthesis.tex:539` [7] | table row "Permutation p ... 0.000 ... 5.0e-113" |
| `ch2.tex:386` [2], `chConclusion.tex:155` [8] | "median kappa = 0.273, p = 4.98e-113 (844 obs); confirmed at N = 211" |

**What is wrong.**

- Permuting electrodes treats them as exchangeable. Neighbouring electrodes are
  correlated, so two spatially smooth quantities correlate across electrodes without any
  coupling between them. A review obtained a 90% false-positive rate from this null on
  two independent smooth fields (simulated by the reviewer). A valid null needs the
  electrode coordinates, which the data files do not carry.
- kappa is a norm and cannot be negative. A one-sample test against zero
  (`chSynthesis.tex:138`) rejects for any data (inferred from the definition at
  `chSynthesis.tex:46`).
- The value 5.0e-113 is a Wilcoxon p over 844 rows that are not independent, which
  `chSynthesis.tex:171` itself calls anti-conservative. It is not a permutation p. A
  permutation p cannot be 0.000; with 5,000 permutations its floor is 0.0002.
- The committed result file for this analysis holds 2,000 permutations per observation,
  not 5,000 (read).
- These analyses use 211 participants and 844 observations. Participant 195, excluded
  from the corrected four-category cohort for a flat channel, is included.
- Reliability across categories is near zero (ICC 0.059, `chSynthesis.tex:249`-`256`), which
  the text reports.

**Corrected statement.** The existence of structure-function coupling is not
established. The observed kappa exceeds an electrode-permutation null that is not valid
for spatially correlated electrodes. The statistic should be treated as exploratory
until it is tested against a spatially constrained null.

**Basis.** Read: `chapter7Experiments/run_chapter7_experiment_A.py`,
`docs/REPOSITORY_AUDIT.md`. A defect in the clustering computation
(`docs/v1/DEFECT_FIXES.md`, D3) does not affect kappa.

### 10. Graph models, deep baselines and saliency

| Location | Statement |
|---|---|
| `chgraph.tex:39`, `464`, `481` [5] | GCN, GIN and EdgeConv "evaluated ... as five-seed ensembles", with accuracies 56.4%, 49.8%, 48.5% |
| `chgraph.tex:386`, `391` [5] | "inter-channel variance decreases by 33% after two layers"; cosine similarity 0.21 at K = 0 |
| `chgraph.tex:497`, `512` [5] | "Deep models use 5-seed ensembles with majority voting"; LSTM 58.0%, GRU 59.9%, EEGNet 72.0% |
| `chgraph.tex:989`, `991` [5]; `ch2.tex:354` [2] | EEGNet saliency maxima at 691, 441 and 402 ms; "The approximately 20 pp accuracy gap is the measured cost of intrinsic interpretability" |

**What is wrong (read from the code).**

- The graph models in the repository are propagation through fixed random weights
  followed by a classifier. No graph weight is trained. In one script the adjacency is
  chosen by its cross-validated accuracy, and in another the attention weights were
  computed from the test fold's labels (`docs/v1/DEFECT_FIXES.md`, D5; on pure noise
  that function returned 55% where chance is 33%).
- The 33% figure is a constant in the script (190 and 99), not a computed quantity.
  `chgraph.tex:650` gives 34% for the same setting and `chgraph.tex:652` gives 0.138 for
  the same cosine similarity.
- The deep-baseline values match an implementation that trains only the output layer of
  each network. The script written for full training cannot read the data files.
- The saliency script passed the network its input with time and channels exchanged
  (`docs/v1/DEFECT_FIXES.md`, D2). The latencies it reported cannot be attributed to a
  correctly oriented model.
- `chgraph.tex:1004` already says the accuracy gap "cannot be interpreted as the cost of
  interpretability", which contradicts `ch2.tex:354`.

**Corrected statement.** The graph results describe random propagation and support no
conclusion about graph neural networks as a model class. The deep-baseline accuracies
are lower bounds from partly trained models. The EEGNet saliency latencies are
withdrawn. No accuracy difference in the dissertation measures a cost of
interpretability.

**Basis.** Read: `docs/audit/appendix_A_chapters3-5_interpretability.md`,
`docs/audit/appendix_B_confirmatory_novel_external_ablation.md`.

### 11. Confirmatory classification values

`main.tex:112` and `chgraph.tex:403` report 60.6% and 53.2% for the reservoir code, with
an advantage of 11.3 and 16.5 points over conventional features. The pipeline committed
with the dissertation produces 64.3% and 54.6%, and 15.7 and 18.5 points. The two are
reconciled in `COMPUTATIONAL_ERRATUM.md`. The v2 code reproduces the committed pipeline
exactly.

### 12. Clinical associations

| Location | Statement |
|---|---|
| `chgraph.tex:700`, `714`-`718` [5] | channel-level effects by diagnosis, for example "SUD x Strength Reactivity (p = 0.0004, d = -0.52)" |
| `chgraph.tex:720` [5] | the pattern "provides a graph-topological phenotype" |
| `chgraph.tex:1012` [5], `chDynamics.tex:538` [6] | "dynamical metrics (D block) carry the strongest SUD and PTSD clinical signals" |
| `chDynamics.tex:480` [6] | "27 significant channel-metric pairs across diagnoses" |

**What is wrong.**

- The tests are screens over channels, metrics and diagnoses with no correction for the
  number tested and no adjustment for age, sex, medication or comorbidity. No covariate
  adjustment is mentioned anywhere in the dissertation.
- The headline effect does not survive the dissertation's own check: the same SUD effect
  is d = -0.52, p = 0.0004 at `chgraph.tex:714` and d = -0.34, p = 0.073 under
  permutation at `chgraph.tex:722`. `chgraph.tex:885` then says the results "do not
  establish ... a coherent SUD phenotype", which contradicts `chgraph.tex:720`.
- `chDynamics.tex:471` reports that zero of 49 tests are significant, which contradicts
  the summary at `chgraph.tex:1012`.
- A family of twelve tests fixed in advance (four SCID-5 depression variables, three
  representations, Holm correction) found nothing: smallest corrected p = 0.34.
- A rerun of the Chapter 7 diagnosis analyses on the committed tables reproduces the
  dissertation's values: no diagnosis effect on kappa (p from 0.32 to 0.92).

**Corrected statement.** No clinical association reported in the dissertation survives
correction for multiple comparisons, and none is adjusted for covariates. All of them
are hypothesis-generating. The phrases "phenotype" and "carry the strongest clinical
signals" should be struck.

**Basis.** Measured: `experiments/v2/v2_depression.json`. Replayed:
`docs/v1/DEFECT_FIXES.md`, D9.

### 13. An inequality stated without one of its terms

`chDynamics.tex:99` [6] gives the bound `I >= log2(K) - H_b(P_e)`. Fano's inequality for
K classes is `I >= log2(K) - H_b(P_e) - P_e log2(K - 1)`. The last term vanishes only
for K = 2. For the three-class case with the error rate used, the bound is lower by
0.206 bits. Comparisons between groups are unaffected, because the missing term is a
constant. The code has been corrected (`docs/v1/DEFECT_FIXES.md`, D6).

---

## Part II. Statements that disagree with each other

These pairs were found by reading the source. Where the code or the data settle the
matter, the resolution is given. Otherwise the author must decide.

| # | Statement A | Statement B | Resolution |
|---:|---|---|---|
| 1 | Compressor is a fixed sparse random projection (`main.tex:112`, `chgraph.tex:401`) | "BSC6-PCA-64 ... the primary feature representation used throughout" (`chgraph.tex:913`); PCA in the specification table (`chLSMEmbeddings.tex:249`) | The corrected analysis uses the random projection. PCA-64 belongs to the archived analyses |
| 2 | "each of 34 electrodes drives its own reservoir" (`chLSMEmbeddings.tex:59`) | "one saved reservoir across channels" (`chLSMEmbeddings.tex:195`, `258`) | The committed confirmatory code uses one reservoir per electrode. Measured difference between the two: -0.8 and +1.8 points, within noise |
| 3 | "Raw SHAPE inputs were not included in the package, so the corrected real-data analysis could not be rerun" (`chLSMEmbeddings.tex:258`) | "rebuilt from the raw SHAPE files" (`chConclusion.tex:8`, `chgraph.tex:152`, `401`) | The corrected analysis was run on the raw files |
| 4 | Echo-state verification performed (`ch1.tex:184`) or promised (`ch2.tex:236`, `chLSM.tex:11`) | listed as required, no result (`chDynamics.tex:427`) | Not performed on the classification reservoir; see item 1 |
| 5 | Four-category cohort 210 participants, 840 observations (`main.tex:112`, `chgraph.tex:152`, `771`) | 211 and 844 (`chSynthesis.tex:122`, `Appendixa.tex:55`, `78`, `ch2.tex:386`) | 210 and 840 is the corrected cohort. Chapter 7 and Appendix A include participant 195 |
| 6 | "All 211 subjects have complete diagnostic metadata" (`chgraph.tex:139`) | "206 of 211 (5 missing)" (`chSynthesis.tex:124`, `Appendixa.tex:229`) | 206 of 211 (confirmed on the diagnostic table) |
| 7 | 172 female and 33 male (`chgraph.tex:139`) | N = 211 | The counts sum to 205. Author to resolve |
| 8 | 27 healthy controls from incomplete panels (`chDynamics.tex:506`) | 27 from a "broader definition" (`chSynthesis.tex:124`); 27 still used at `chDynamics.tex:624` against 22 at `chDynamics.tex:515` | 22 participants have none of the five primary diagnoses and complete data |
| 9 | "over 40" diagnostic variables (`ch1.tex:26`) | 15 (`chSynthesis.tex:586`); eight (`chgraph.tex:144`). MDD 67.3% (`chgraph.tex:139`) or 69% (`chSynthesis.tex:590`) | Author to resolve. 142 of 206 assessed is 68.9%; 142 of 211 is 67.3% |
| 10 | PCA components form "an interpretable basis ... onset (PC1)" (`chLSMEmbeddings.tex:228`) | "should not be assigned fixed semantic labels" (`chLSMEmbeddings.tex:182`, `289`) | The second |
| 11 | "perfect alignment", "provably lossy" (`chgraph.tex:915`, `917`, `ch1.tex:182`) | "does not prove" (`chLSMEmbeddings.tex:283`, `chConclusion.tex:86`) | The second; see item 5 |
| 12 | "8,900x separability improvement" (`chLSMEmbeddings.tex:224`) | "not an effect-size ratio" (`chLSMEmbeddings.tex:87`) | The second; see item 2 |
| 13 | Synthetic accuracy 99.3% (`chLSMEmbeddings.tex:193`) | 98.8% (`chLSMEmbeddings.tex:230`). PCA variance 50.5% (`chLSMEmbeddings.tex:167`) or 50.8% (`chgraph.tex:342`) | Author to resolve |
| 14 | Reservoir size "saturates at 256" (`chLSM.tex:497`, `515`) | "rather than as a universal saturation point" (`chLSMEmbeddings.tex:61`) | The second |
| 15 | Attention "peaks within the ERP window ... aligned with P300, LPP, and N2"; "measured cost of intrinsic interpretability" (`ch2.tex:354`) | "do not establish distinct temporal strategies"; "cannot be interpreted as the cost of interpretability" (`chgraph.tex:993`, `1004`) | The second; see items 4 and 10 |
| 16 | A bin at 234-273 ms labelled "LPP" (`chgraph.tex:930`, `938`) | LPP defined as 500-800 ms (`chgraph.tex:922`, `chLSMEmbeddings.tex:265`) | The label is wrong. The laboratory scores the LPP from 300 to 1000 ms |
| 17 | "earliest window is the strongest isolated segment" (`chgraph.tex:426`) | "distributed across the post-stimulus response" (`main.tex:114`) | Neither; see item 4 |
| 18 | Variance drop 33% (`chgraph.tex:386`) | 34% for the same setting (`chgraph.tex:650`). Cosine similarity 0.21 (`chgraph.tex:386`) or 0.138 (`chgraph.tex:652`). Edges about 112 (`chgraph.tex:29`) or 113 (`chgraph.tex:197`) | The 33% is a constant in the code; see item 10 |
| 19 | "six operator families" (`chgraph.tex:686`) | seven are named in the same sentence | Seven |
| 20 | SUD strength effect d = -0.52 (`chgraph.tex:714`) | d = -0.34 for the same effect (`chgraph.tex:722`) | See item 12 |
| 21 | "provides a graph-topological phenotype" (`chgraph.tex:720`) | "do not establish ... a coherent SUD phenotype" (`chgraph.tex:885`) | The second |
| 22 | Zero of 49 tests significant (`chDynamics.tex:471`) | "dynamical metrics ... carry the strongest SUD and PTSD clinical signals" (`chgraph.tex:1012`, `chDynamics.tex:538`); Chapter 7's account of Chapter 6 diagnosis patterns (`chSynthesis.tex:331`, `383`) | The first |
| 23 | "No participant-level Phi comparison is reported" (`chDynamics.tex:506`) | a Phi row with d = -0.10, p = 0.59 (`chDynamics.tex:611`) | Author to resolve |
| 24 | Chapter 7 cites Experiments 6.4, 6.5 and 6.6 (`chSynthesis.tex:189`, `258`, `276`, `319`, `331`) | Experiments 6.4, 6.5 and 6.6 as written in Chapter 6 are different analyses (`chDynamics.tex:482`, `493`, `504`); a sentence quoted from Chapter 6 at `chSynthesis.tex:395` is not in Chapter 6 | The cross-references predate a renumbering. Author to correct |
| 25 | Caption names three metrics as "the four significant cells" (`chSynthesis.tex:145`) | text names two (`chSynthesis.tex:189`) | Author to resolve |
| 26 | "Permutation p ... 0.000 ... 5.0e-113" (`chSynthesis.tex:539`) | "rather than p = 0" (`chSynthesis.tex:468`); the 1e-113 value is a Wilcoxon p called anti-conservative (`chSynthesis.tex:171`). Difference in kappa +0.030 (`chSynthesis.tex:225`) or +0.031 (`chSynthesis.tex:258`) | See item 9 |
| 27 | "These are not replications" (`chSynthesis.tex:13`) | "Three-Class Replication", "This replicates the four-class finding" (`chSynthesis.tex:461`, `492`, `497`, `521`) | The first. Same participants, different stimulus grouping |
| 28 | Mean comorbidity 2.8 (`ch1.tex:28`, `chgraph.tex:139`, `chConclusion.tex:6`) | 2.0 (`chSynthesis.tex:124`, `333`) | Both, for different sets of diagnoses, as `chSynthesis.tex:124` explains. 2.0 for the five primary diagnoses was confirmed on rerun |
| 29 | Footnote cites `tab:baselines` for the EEGNet value (`chgraph.tex:987`) | the value is in `tab:conventional_baselines` (`chgraph.tex:514`) | Reference to correct |
| 30 | The author's earlier papers cited under one pair of keys (`ch1.tex:164`) | under another pair with different titles (`ch2.tex:404`); two bibliography entries name only "Proceedings of the IEEE Conference" | Author to correct against the published records |
| 31 | The label "60.6%" for the corrected random-projection result (`chgraph.tex:403`) | the same number for an archived result at N = 115 (`chgraph.tex:488`). A 59.4% result called "10-fold" (`chgraph.tex:821`) beside 5-fold and 5x5 protocols | Different analyses with coinciding values. State the protocol with each |

---

## Part III. Statements the repository cannot support

These are not shown to be wrong. The code or input that would reproduce them is not in
the repository, so they cannot be checked.

| Statement | Location | Why it cannot be checked |
|---|---|---|
| Chapter 4 synthetic experiments were rerun with "Scaling and PCA ... fitted within each training fold" | `chLSMEmbeddings.tex:61`, `66`, `80`, `123`, `193` | the committed Chapter 4 script fits PCA on all trials before the split |
| Layer-ablation accuracies | `chgraph.tex:817`-`865`, `Appendixa.tex:150`, `164` | the script needs a feature file that no committed script writes; some of its values are constants in the code |
| Temporal-window values 60.5%, 56.9%, 56.7%, 48.5% | `chgraph.tex:426` | the committed pipeline slices quarters of the 0 to 1000 ms epoch and gives different values; the original code is not in the repository. Item 4 applies in either case |
| Chapter 7 per-subject feature tables used for diagnosis classification | `chSynthesis.tex:432` | their generating script is not in the repository. The analyses that read them do reproduce (`docs/v1/DEFECT_FIXES.md`, D9) |
| Diagnosis by category interaction p-values in Chapter 6 | `chDynamics.tex:482` | the script's permutation p-value lacked the +1 term (`docs/v1/DEFECT_FIXES.md`, D4) and was not rerun. The reported result, zero of 35 significant, can only move further from significance |

## Part IV. Points that need no correction

- **Energy.** `ch1.tex:149` and `210` state that the energy advantage is motivational
  and not measured. That remains accurate.
- **Echo-state theory.** The conditional contraction theorem and the statement that a
  spectral radius below one is neither necessary nor sufficient for a driven system
  (`ch2.tex:236`) are consistent with item 1.
- **Raw ERP exceeds the reservoir.** Stated at `chgraph.tex:529`, `541`, `799` and
  `1026`, and confirmed.
- **Scope restrictions.** The statements that clinical results are cohort-level and
  hypothesis-generating, and that legacy results are descriptive, stand. Part I, item
  12 asks only that the passages which contradict them be struck.
