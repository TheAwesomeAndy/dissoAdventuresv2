# ARSPI-Net v2: what changed and what the data say

v2 answers an external methods review of v1 (this repository at `b0fb317`, the
head of `dissoAdventureExperiments`). The review found the corrected v1 confirmatory
protocol sound and leak-free, and most of what was built on it untested or wrong.
v2 keeps the protocol and tests the rest on the authorized SHAPE data.

v1 files are unchanged. Everything new is in `arspinet_v2/` and `experiments/v2/`.

This repository is self-contained. It carries the complete history of the v1 `main`
branch and a copy of every other v1 branch under the same name. The commit that v1
tagged `v1.0-defense` (`9496ba4`) is kept as the branch `archive/v1.0-defense-tag`.

Every number below is read from a committed execution record:
`experiments/v2/v2_results.json` (tables in `experiments/v2/V2_RESULTS.md`),
`experiments/v2/v2_normalization_check.json` and `experiments/v2/v2_depression.json`.
Values are balanced accuracy with participant-bootstrap 95% intervals. Cohorts are v1's:
211 participants and 633 observations for three conditions, 210 and 840 for four
categories.

## Results in brief

| Question | Three conditions | Four categories |
|---|---:|---:|
| v1 configuration, recomputed by the v2 code | 64.3% [61.1, 67.2] | 54.6% [51.9, 57.3] |
| v2 reservoir (shared, gain 0.025), default draw | 66.4% [63.6, 69.1] | 56.2% [53.6, 58.8] |
| v2 reservoir, mean over 10 weight draws | 65.1% (sd 0.82) | 55.4% (sd 0.70) |
| Six-bin amplitudes, no reservoir (204 features) | 67.1% [64.3, 70.0] | 55.1% [52.6, 57.7] |
| Raw ERP, 16 bins (544 features) | 69.6% [66.5, 72.5] | 60.5% [57.9, 63.0] |
| LPP mean amplitude, 300 to 1000 ms (34 features) | 58.4% [55.6, 61.0] | 45.8% [43.3, 48.4] |
| v2 reservoir minus six-bin amplitudes | -0.7 pp [-3.2, +1.8] | +1.1 pp [-1.2, +3.4] |
| Raw ERP-16 minus v2 reservoir | +3.2 pp [+0.4, +6.0] | +4.3 pp [+1.7, +6.8] |
| SRP-64 minus fold-local PCA-64 | +8.6 pp [+5.5, +11.7] | +9.7 pp [+7.0, +12.6] |

Five conclusions follow.

1. **The v1 reservoir has no fading memory. The v2 reservoir does.** At v1's recurrent
   gain of 0.9 the recurrent current is 3.6 times the input current, activity sustains
   itself after the input stops (population rate 0.106), no test sequence converges from
   two initial states, and the median feature test-retest correlation under 1% input
   noise is 0.67. This holds for 10 of 10 weight draws. At gain 0.025 all 10 draws pass,
   with reliability 0.998.
2. **Recurrence contributes nothing measurable to accuracy.** Across gains from 0 to 0.9
   accuracy stays between 64.7% and 67.2% (three conditions) and 52.6% and 56.4% (four
   categories). At v1's gain, recurrence costs 2.6 pp [-5.3, -0.0] on four categories.
3. **The reservoir does not beat its own matched control.** Six-bin amplitudes on the
   same bin edges, with a tenth of the features, are statistically indistinguishable from
   the reservoir code. Raw ERP in 16 bins is better than both. The v2 gain over v1 is
   within draw-to-draw variation: the default weight draw is the best of ten, so 65.1%
   and 55.4% are the fair summaries, not 66.4% and 56.2%.
4. **Random projection beats PCA by 9 points, and subject centering does not explain
   it.** PCA-64 fitted on training participants loses 8.6 and 9.7 pp to SRP-64. Fitting
   the PCA basis after removing each training participant's mean changes nothing (57.8%
   and 46.7%). SRP-64 loses nothing against no compression at all (-0.0 and -0.6 pp).
   This is the strongest positive result in the repository.
5. **v1's per-epoch z-scoring destroys temporal localisation.** Under v1 preprocessing
   the scalp pattern in the first 31 ms after picture onset decodes the condition at
   54.3% (chance 33.3%), which no cortical response can produce. With the files as
   delivered (baseline-corrected, no z-scoring) the same bin is at chance, 33.8%
   [31.0, 36.8], decoding rises from about 125 ms and peaks at 62.2% between 560 and
   690 ms, the late positive potential range. Whole-epoch accuracy is unchanged (69.6%
   against 68.8% for raw ERP-16). Every window-specific number computed on z-scored
   epochs, in v1 and in section 6 of `V2_RESULTS.md`, is therefore uninterpretable as
   timing.

## Finding-by-finding record

"Done" means implemented, verified on synthetic data in CI, and run on SHAPE.

| Review finding | v2 action | Status |
|---|---|---|
| F1. Confirmatory reservoir lacks fading memory. Echo-state evidence came from a different, weaker network | `arspinet_v2/dynamics.py`: four label-free tests on the deployed network with finite perturbations. Gain chosen as the largest grid value that passes a fixed rule | Done. Confirmed for v1. v2 gain 0.025 |
| F2. No matched non-reservoir control | Six-bin amplitudes on the BSC6 bin edges, recurrence off, non-spiking leaky bank with the same input weights, all through the same chain | Done. Reservoir does not beat six-bin amplitudes |
| M1. Committed code used a different reservoir per electrode. The dissertation describes one shared reservoir. First window mislabelled as 39 to 273 ms | Sharing is an explicit option and v2 uses one shared reservoir. Windows are sliced in milliseconds | Done. Sharing changes accuracy by -0.8 and +1.8 pp, both within noise |
| M2. Destruction controls mix information removal with noise injection | Each control in two modes: refit on destroyed data, and intact-trained model scored on destroyed held-out rows. Fixed-permutation sanity control added | Done. Sanity control moves accuracy by -0.8 and -0.7 pp. Collapse costs 15.6 and 19.6 pp |
| M3. Permutation test used a ridge stand-in and 100 permutations, so p sat at the floor | Headline estimator, 2,000 within-participant permutations | Done. Zero exceedances, observed 16.9 and 19.2 null SDs above the null mean. p = 0.0005 is still the floor for 2,000 |
| Subject-variance ratio has no noise floor | `arspinet_v2/nuisance.py`: unbiased two-way variance components with permutation nulls | Done. Noise gives components of 0.0% and a v1-style ratio of 69. The real code gives subject 44.5%, condition 10.3% and a ratio of 5.6 |
| PCA control missing: PCA after train-fold subject centering | Fold-local PCA-64 with and without a subject-centered basis, and no compression | Done. See conclusion 4 |
| Clinical analyses were an uncorrected screen | `experiments/v2/run_v2_depression.py`: four SCID-5 variables in the laboratory's order, three representations, Holm correction over the 12 tests | Done. Nothing survives correction (smallest corrected p = 0.34) |
| Level-1 statistic r = 0.82 never touches the reservoir, and non-overlapping windows of a zero-mean epoch anticorrelate by construction | Normalization check shows the mechanism directly | Done as a diagnosis. The v1 statistic should be withdrawn |
| Bootstrap intervals condition on fitted models | Documented in `evaluation.py`. Not changed, so v1 and v2 intervals are comparable | Open |
| Coupling statistic kappa: electrode-permutation null is invalid for spatially smooth fields | Needs electrode coordinates | Open. The laboratory has offered the column-to-electrode map |
| "GNNs" are untrained random propagation with adjacency picked on test folds | Not addressed | Open |
| Chapters 5 to 7 rest on legacy input | Not rerun | Open |
| M4. Theory figure labels (Lyapunov "bound", Takens "lower bound", memory capacity) | Not edited. The dissertation is a submitted record | Open, for the erratum |
| M5. Chapter 4 Fisher ratio is a p > n artefact | Not addressed | Open |
| Reservoir input is z-scored per epoch | Found by v2. A baseline-referenced input needs a new input scaling, which is a design decision | Open |
| Dissertation and slide inconsistencies (four accuracy values, PCA-64 wording, 840 against 844 observations, cohort descriptors) | Not edited | Open, for the erratum |

## What v2 changes in the code

- `arspinet_v2/reservoir.py`. One batched LIF implementation. It reproduces the v1
  operator bit for bit (checked in CI and on SHAPE) and exposes recurrent gain, reservoir
  sharing and a non-spiking mode.
- `arspinet_v2/evaluation.py`. v1's splits and readout. The readout is fitted in the row
  space of the standardized training matrix, which is the same model at a fraction of
  the cost. On the v1 features the two fits give 64.3% and 64.4%, with 99.4% and 98.1%
  of predictions identical.
- `arspinet_v2/features.py`, `controls.py`, `nuisance.py`, `dynamics.py`. As in the table.
- `experiments/v2/verify_v2.py`. 32 data-independent checks, wired into CI.
- `data/clinical_profile.csv` is no longer tracked. `.gitignore` already listed it as
  private participant data. It remains in the git history of this repository and of v1.

## Reading the results with the task in mind

The cohort performed the emotional interrupt task, with the late positive potential as
the standard measure (Weinberg and Hajcak 2011, J. Cogn. Neurosci. 23:2994,
doi:10.1162/jocn.2011.21630. Nelson et al. 2015, Psychol. Med. 45:2545,
doi:10.1017/S0033291715000471, where the LPP is scored from 300 to 1000 ms at pooled
occipital and parietal sites). Three points follow from that literature.

- The LPP score alone, kept per electrode, reaches 58.4% and 45.8%. The multivariate
  representations add 8 to 11 points to it.
- Time-resolved decoding of the scalp pattern follows Bae and Luck (2018, J. Neurosci.
  38:409, doi:10.1523/JNEUROSCI.2860-17.2017). Their method decodes within a participant
  from averages of trial subsets. SHAPE is delivered as one average per participant and
  condition, so decoding here is across participants. The two answer different questions.
- The delivered files have no channel header. Occipital and parietal pooling, spherical
  spline interpolation and a spatially valid null for kappa all need the column map.

## Limits of this record

- One cohort. Nothing here is validated on a second dataset.
- The fading-memory rule was written in the session that ran it. It uses no labels, and
  the gain sweep is reported in full so the choice can be checked against any other rule.
- 2,000 permutations, not the 5,000 the review asked for. The runner's default is 5,000.
- The depression analysis does not adjust for age or sex. Both are missing for about
  half of the cohort.
- The interval caveat above applies to every interval in this document.

## Reproducing

```text
python experiments/v2/verify_v2.py                      # no data required

python experiments/v2/run_v2_validation.py --data3 ./batch_data --data4 ./categories \
    --cache ./v2_cache --out experiments/v2/v2_results.json --n-perm 2000 --jobs 4
python experiments/v2/run_v2_normalization_check.py --data3 ./batch_data --cache ./v2_cache
python experiments/v2/run_v2_depression.py --data3 ./batch_data --labels labels.xlsx --cache ./v2_cache
python experiments/v2/make_report.py
```

The full run took about 2 hours 40 minutes on 4 cores.
