# Defects fixed in the v1 record, November 2026

The 2026-11-21 audit (`docs/REPOSITORY_AUDIT.md`, finding I6) listed defects in archived
and illustrative v1 experiments. The repository owner asked for them to be fixed. This
file records each one: what was wrong, what changed, what the fix does to reported
numbers, and whether outputs were regenerated.

Every fix has a regression test in `tests/test_v1_defect_fixes.py` that fails on the
code as it was. The dissertation text is not edited; affected statements are listed in
`DISSERTATION_ERRATUM.md`.

**Regenerated** means the script was rerun and its committed outputs replaced.
**Not regenerated** means the code is fixed but the committed outputs, and any number
the dissertation takes from them, predate the fix. The reason is given each time.

| ID | Where | Defect | Effect on reported numbers | Outputs |
|---|---|---|---|---|
| D1 | `defense_figures/experiment_c_memory_capacity/make_experiment_c_figures.py` | The fading-memory time constant was computed on membrane potentials, not spikes | The `fading_memory_tau` column and one panel change. Memory-capacity values do not | Regenerated |
| D2 | `experiments/interpretability/run_eegnet_saliency_comparison.py` | An inverted axis test fed the network time as channels | The saliency peaks reported for EEGNet (402 to 691 ms) cannot have come from a correctly oriented input | Not regenerated |
| D3 | `chapter7Experiments/run_chapter7_experiment_A.py` | Weighted clustering was twice the Onnela coefficient | None on any statistic. Stored clustering values are 2x | Not regenerated |
| D4 | four scripts, below | Permutation p-values computed as count / n, without the +1 | p rises by at most 1/(n+1); a reported 0 becomes 1/(n+1) | Not regenerated |
| D5 | `chapter5Experiments/reproduce_chapter5.py` | Held-out labels set the attention weights | The attention-pooling accuracy from this script is inflated | Not regenerated |
| D6 | `experiments/ch6_ch7_3class/ch6_03_experiments.py` | Fano bound without the P_e log2(K-1) term | The bound is lower by a constant. No group comparison changes | Not regenerated |
| D7 | `experiments/ch6_ch7_3class/ch7_04_experiments.py` | A variance share was labelled ICC(3,1) | Label only | Not applicable |
| D8 | `chapter5Experiments/verify_reproduce_chapter5.py`, two READMEs | "PCA fitted per fold (no leakage)" asserted from a string match; the code fits PCA on all data | The archived Chapter 5 results were described as leak-free and are not | Not applicable |
| D9 | `chapter7Experiments/run_chapter7_experiment_D.py`, `chapter7Experiments/run_chapter7_experiment_E.py` | Conclusions printed as fixed text regardless of results; figure path hardcoded to a directory that exists only on the original machine | None. The computed tables are identical before and after | Replayed, see below |

## D1. Fading-memory time constant on the wrong variable

`run_full` returns `(membrane, spikes)`. The function unpacked it as `S, _`, so the
"population spike rate" deviation was a membrane deviation.

- **Check before the fix.** The unfixed script, run on 2026-11-21, reproduced the
  committed `experiment_c_data.csv` byte for byte. The script draws synthetic noise and
  does not use the SHAPE samples it loads, so this is a full reproduction.
- **After the fix.** Only the `fading_memory_tau` column changes. The other 32 columns
  are identical.
- **The corrected quantity is still not informative.** Of 20 leak values, 7 give an
  infinite time constant and several give no estimate; finite values range from 9 to
  4,100 steps. The single-seed exponential fit is unstable. Do not cite it.
- **Regenerated:** `experiment_c_data.csv` and `analysisC_3d_memory_capacity_peak.pdf`
  in `defense_figures/experiment_c_memory_capacity/outputs/`. The copy of that panel
  under `pictures/defense_audit/`, which the defense deck includes, was left as
  presented.

## D2. EEGNet saliency input orientation

`if eeg.shape[1] > eeg.shape[2]: X = eeg  # already (N, C, T)` treated a
`(N, 256, 34)` array as channels-first. Either layout reached the network as 256
"channels" by 34 "time samples". The fix is `channels_first()`.

- **Not regenerated.** The script needs a feature file that no committed script writes,
  and PyTorch. It also trains on all data with no seed.
- **Consequence.** The dissertation's EEGNet saliency latencies (Chapter 5 and
  Chapter 2) rest on this script. They are unsupported until it is rerun.

## D3. Clustering coefficient doubled

`diag(W^3)` over cube-root weights already sums ordered neighbour pairs, which the
`1 / (k (k - 1))` normalization accounts for. An extra factor 2 doubled every value.

- **No statistic changes.** The coupling matrix and kappa are Spearman correlations,
  which are invariant to the factor. Experiment E standardizes features inside each
  fold, which removes it.
- **Not regenerated.** Regenerating needs the full experiment A run on the restricted
  data (about 59 million rank correlations) to change numbers that do not affect any
  result. The stored `t_clustering_*` columns in
  `chapter7Experiments/chapter7_results/subject_features.csv` and
  `observation_features.csv`, and `topo_profiles` in `ch7_full_results.pkl`, are twice
  the Onnela value.

## D4. Permutation p-values without the +1

A permutation p-value is `(1 + count) / (1 + n)`. These computed `count / n`:

| Script | n | Largest possible change |
|---|---:|---:|
| `chapter6Experiments/run_chapter6_exp5_interaction.py` | 500 | +0.002 |
| `experiments/ch6_ch7_3class/ch7_04_experiments.py` (three tests) | 5,000 | +0.0002 |
| `defense_figures/experiment_d_channel_permutation/make_experiment_d_figures.py` | 500 | +0.002 |
| `experiments/interpretability/run_arspinet_v2_attention_prototype.py` | 10,000 | +0.0001 |

- The corrected p is never smaller than the old one, so no result that was not
  significant becomes significant.
- A reported p of 0 becomes the floor. Defense experiment D's committed CSV shows
  0.000000 where the corrected value is 0.002. The dissertation's Chapter 7 table row
  "Permutation p 0.000" should read p = 0.0002.
- **Not regenerated.** Each script needs the restricted data and intermediate files.

## D5. Test labels in the attention weights

`attention_pooling_cv` scored every channel on the held-out fold and used those scores
as attention weights and to pick the top channels for that same fold.

- **Size of the leak.** On pure noise, where the true accuracy is 33.3%, the unfixed
  function returns 55.0%. The fixed function returns 28.9%.
- **Fix.** Channel scores come from a split of the training participants.
- **Not regenerated.** The script reads a feature file no committed script writes.
- **Consequence.** Any attention-pooling accuracy taken from this script is inflated.

## D6, D7, D8

- **D6.** Fano's inequality for K classes is
  `I >= log2 K - H_b(P_e) - P_e log2(K - 1)`. The last term was missing. It also uses a
  hardcoded accuracy of 79.4%, which is a transductive centered value; that is not
  changed here.
- **D7.** The quantity is var(subject means) over that plus the mean within-subject
  variance. It is now labelled as a between-subject variance share.
- **D8.** The verifier check and two README lines now say what the code does. The
  number of checks the suite reports is unchanged.

## D9. Fixed conclusions and a machine-specific path

Experiments D and E printed paragraphs of conclusions that did not depend on the
computation. They now print statements computed from the run. The figure directory is
`ARSPI_FIGURE_DIR`, defaulting to a `figures` folder beside the script.

- **Replayed on 2026-11-21.** Both scripts were run before and after the change on the
  committed Chapter 7 tables and the SCID-5 spreadsheet. The result tables are
  identical before and after, and they match the dissertation: experiment D's five
  diagnosis tests give p from 0.32 to 0.92, and experiment E's best AUC is 0.622 (ADHD,
  dynamical features only).
- The old fixed text happened to agree with the data. Its statements are now computed:
  largest observed |d| = 0.115, below the minimum detectable effect for all five
  diagnoses.

## Reviewed and not changed

| Report | Why it was left |
|---|---|
| Chapter 6 experiment 5 reseeds its generator identically for all 45 tests | Not a defect. Each test's p-value is valid; sharing the permutation sequence makes tests dependent, which matters only for a correction that was not applied |
| Global PCA before the split in Chapters 4 and 5 and the four-category extension | A protocol choice of the archived analyses, superseded by `experiments/confirmatory/`. Changing it would produce different experiments, not corrected ones. Documented (D8) |
| Chapter 6 echo-state analysis runs on a different network from the classifier's | Not a coding error. Superseded by `experiments/v2/run_v2_validation.py`; covered in the erratum |
| Mislabelled window names in `experiments/confirmatory/confirmatory_pipeline.py` | The file is hashed in the v1 confirmatory record and the labels are keys of that record. Documented in the registry and the erratum |
| `chapter5Experiments/canonical_pytorch_baselines.py` cannot load the SHAPE text files | It is a template whose loader was never adapted. No reported number comes from it |
| Legacy verifier tautologies | Wrapped and counted by `scripts/run_verifiers.py`; to be replaced as areas are next worked on |
