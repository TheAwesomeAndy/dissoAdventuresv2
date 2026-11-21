<!-- Audit appendix C. Repository state: commit 913d457, 2026-11-21.
     Written by a read-only reviewing agent that read the code and ran nothing in the repository.
     Reproduced unedited. Statements are that reviewer's reading of the code, cited by file and line.
     They were not each re-verified; docs/REPOSITORY_AUDIT.md says which ones were. -->

# Audit: chapter6Experiments, chapter7Experiments, experiments/ch6_ch7_3class

Read-only; nothing was run, so no numeric result is verified. Path prefixes: `C6` = `/home/user/dissoAdventuresv2/chapter6Experiments`, `C7` = `/home/user/dissoAdventuresv2/chapter7Experiments`, `X3` = `/home/user/dissoAdventuresv2/experiments/ch6_ch7_3class`. No git commands were used, so "tracked" means present in the tree and not matched by `.gitignore`.

## 1. Reservoir definitions (equivalence classes)

All use N=256, β=0.05, θ=0.5, one input per channel, seed 42 unless noted.

| Class | Where | Weights | Update | vs `experiment_zero.py:66-98` |
|---|---|---|---|---|
| A: dense uniform | `C6/reproduce_chapter6.py:61-83` (`LIFReservoirFull`); inline copy in Lyapunov `:107-116` | W_in U(±√(6/(1+N))), W_rec U(±√(6/2N)), scaled to ρ=0.9 | (1−s) reset, then subtract θ, then floor 0 | Identical init order and equations; also returns post-reset membrane. Seeds `42+17·ch` (`:315`) |
| B: dense Gaussian | `X3/ch6_ch7_01_feature_extraction.py:125-163` | W_in as A; W_rec = randn/√N scaled to ρ=0.9 | floor, threshold, subtract | Update is algebraically equivalent; W_rec distribution differs. Docstring `:47` says "Xavier uniform"; code is Gaussian. Arrays are (N,T) |
| C: sparse Gaussian | `C6/run_chapter6_exp1_esp.py:39-50`, `exp2:38-49`, `exp3_surrogate:47-58`, `exp3_valueadd:80-90`, `exp4:53-64`, `exp5:61-72`, `exp6:54-65`; `C7/run_chapter7_experiment_A.py:83-118` | W_in = randn·0.3; 10% mask, no self-loops; W_rec = randn·0.05·mask; no spectral scaling | (1−s) reset only; no subtraction, no floor; membrane stored pre-reset | Different weights and different equations |
| D: non-LIF comparators | `C6/run_chapter6_exp3_valueadd.py:91-106` | TanhRNN reuses C weights; RandFourier W = randn·3 | tanh; cos | n/a |

There is no `class LIFReservoir` in scope; the classes are `Reservoir`, `Res`, `LIFRes`, `LIFReservoirFull`, plus functions. `C6/README.md:40` says class C has spectral radius ~0.8. No code sets it; a circular-law estimate (0.05·√25.6 ≈ 0.25) disagrees. Not computed.

Inputs differ by family:
- Class C: full 1229-sample epoch at 1024 Hz, baseline included, z-scored per channel.
- Class B: 205 baseline rows dropped, decimated ×4, 256 steps, z-scored (`01:169-204`).
- Class A: `X_ds` from `shape_features.pkl`, no normalisation in the script; upstream scaling not determined.

## 2. Per-script facts

### C6 4-class runners (`run_chapter6_exp*.py`)
Shared:
- Raw files `SHAPE_Community_<id>_IAPS(Neg|Pos)_(Threat|Mutilation|Cute|Erotic)_BC.txt`; subject 127 excluded.
- CLI `--category-dirs` (required), `--output-dir figures`, `--results-dir results` (relative to CWD).
- Outputs are PDFs plus one pickle. No CV, no covariates, no multiple-comparison correction.

Descriptors (`exp2:51-87`, duplicated verbatim in exp3_surrogate and exp4):
- total_spikes, mean_firing_rate.
- rate_entropy: mean per-neuron binary entropy.
- phi_proxy: rate_entropy / (spikes·(mean fan-out+3))·1e6.
- population_sparsity, temporal_sparsity (pop rate < 1/N), rate_variance (across neurons).
- LZ: dictionary phrase count on 10 neurons chosen by `RandomState(0)`.
- Permutation entropy: d=4, τ=1, on mean membrane.
- tau_relax: 1/e decay from the smoothed peak after step 100.
- tau_ac: first lag with ACF ≤ 1/e, cap 200.

Per script:
- **exp1_esp**: Benettin λ1 with δ0=1e-8, renormalisation every 50 steps, direction `RandomState(42)`. 211 subjects × 4 categories × channels [0,8,16,24,33]. The perturbed spike state is overwritten by the reference at each renormalisation (`:182`). Writes `ch6_exp1_full.pkl`. Docstring hardcodes "λ1 = −0.054".
- **exp2_reliability**: 10 seeds (`seed*1000+42`), 60 subjects (`RandomState(42)`), channels [0,16,33]. ICC(3,1) over (subject, category, channel) rows; gates at 0.75 and 0.5. Writes `ch6_exp2_full.pkl`.
- **exp3_surrogate**: 40 subjects; phase-randomised, time-shuffled and block-shuffled (50) surrogates with RNG `RandomState(sid*100+ch)`. "Sensitive" means pooled-SD |d|>0.2 and Wilcoxon p<0.05; the gate is ≥2 of 3 (`:204`). Writes `ch6_exp3_full.pkl`.
- **exp3_valueadd**: also labelled "Experiment 6.3". 80 subjects, channel 16, four paths; permutation entropy, tau_ac and LZ on mean trajectories; paired d_z and Wilcoxon. Writes `ch6_exp3_unified.pkl`.
- **exp4_dissociation**: all subjects, 5 channels, 9 "VALIDATED" metrics (drops LZ and population_sparsity); d_z and Wilcoxon, 18 tests. Writes `ch6_exp4_full.pkl`.
- **exp5_interaction**: requires `--psychopathology` xlsx and `--exp4-results`. Columns `ID`, MDD/SUD/PTSD/GAD/ADHD, ==1 vs ==0. Statistic is the variance across categories of the group difference in reactivity; 500 group-label permutations; p has no +1 (`:272`); RNG is reseeded identically for all 45 tests (`:267`). `mannwhitneyu` is imported but unused.
- **exp6_temporal**: channel 16, window 150, stride 50 (22 windows). Permutation entropy changes to d=3 on population rate (`:77-78`). 308 Wilcoxon tests.

### `C6/reproduce_chapter6.py` (reproduction script)
- Docstring claims every Chapter 6 result and 10 experiments; `main()` implements only 1, 2 and 7 plus figures.
- Dead code: `compute_tau_ac`, `compute_icc31`, `phase_randomize`, `fdr_bh`, the SVC branch, and the `hc, mdd` masks (`:312`).
- Inputs: `--features` pickle (keys `y`, `subjects`, `X_ds`), optional `--labels` xlsx (MDD only).
- 3-class (Neg/Neu/Pos). Channels [11,21,7,13,31]. Metrics: Phi = Var[pop]/activity, permutation entropy of membrane PC1 (order 4, delay 3), tau_relax (stim_ref=70). Kruskal–Wallis treats observations as independent.
- Sliding window: 13 steps, 3 spike-count bins, then `PCA(32)` fitted on all observations before CV (`:363`) and with no `random_state`. LogReg C=0.1, `StratifiedGroupKFold(5, seed 42)` by subject.
- Lyapunov: renormalisation every 10 steps, RNG 99.
- Writes `ch6_all_results.pkl` to CWD (`:425`, not gitignored) and figures to `./figures/ch6`.

### C7 4-class
- **experiment_A** (runner plus library-style functions):
  - Input: raw files under `C7/../categories/categoriesbatch{1..4}`. Positional args `START BATCH`, or `--analyze`.
  - Graph: theta 4–8 Hz, Butterworth order 3, filtfilt, Hilbert, time-averaged PLV; fully weighted, no threshold, diagonal 0. T = strength and Onnela clustering.
  - D: 7 metrics per channel; tau_ac cap is 100 here versus 200 in C6 (`:140`).
  - C: 7×2 Spearman across 34 electrodes. κ = ‖C‖_F/√14 (`:263`).
  - Null: permute electrode rows of T jointly, `N_PERM=5000` (`:40`), seed = subject ID; p = (1+#≥)/(1+n).
  - Group test: one-sided Wilcoxon of κ minus null median over 844 observations, treated as independent. Bonferroni over 14 cells.
  - Writes `chapter7_results/ch7_full_results.pkl` and `ch7_expA_analysis.pkl` (tracked dir). Figures go to hardcoded `/mnt/user-data/outputs/pictures/chSynthesis` (`:50`).
- **experiment_B**: reads `kappa_matrix.csv` and asserts shape (211,4). Sum-of-squares partition; within-subject label permutation ×5000 (seed 42); ICC(3,1); Friedman; 2 Wilcoxon contrasts. Writes `chapter7_results/ch7_expB_results.npz` and figures to `pictures/chSynthesis` (both tracked; `ARSPI_FIGURE_DIR` overrides).
- **experiment_C**: reads `C_matrices.csv`. Cellwise Wilcoxon, Bonferroni 14 per contrast; sign-flip Frobenius test ×5000. Figures only. d_z uses ddof=0; B uses ddof=1.
- **experiment_D**: reads `C_matrices.csv` and `C7/../SHAPE_Community_Andrew_Psychopathology.xlsx`. Subject-mean κ; Mann–Whitney for 5 diagnoses with Bonferroni 5; interaction permutation ×5000, uncorrected; bootstrap 2000 via global `np.random.seed`. Figures to `/mnt/...`; no results file. Conclusion text is hardcoded regardless of output (`:153`, `:415-417`).
- **experiment_E**: reads `subject_features.csv`, `C_matrices.csv`, xlsx. Feature sets T (68), D (238), T+D, T+D+κ. LogReg C=1, `StratifiedKFold(5)`×10 repeats (one row per subject, scaler in-fold); AUC, balanced accuracy, F1; Wilcoxon over 50 dependent folds, uncorrected. Hardcoded conclusions (`:369-377`, `:488-506`). Figures to `/mnt/...`.
- **extract_C_matrices.py** (utility, no main guard): docstring says the output is `/mnt/user-data/outputs/C_matrices.csv`; the code writes `chapter7_results/C_matrices.csv` (`:18`).
- **extract_kappa_matrix.py**: prints to stdout only; `C7/README.md:47` redirects it.

### X3 3-class (all paths `./` relative to CWD; 02–04 execute at import)
- **01_feature_extraction**:
  - Input `./batch_data/`; output `./ch6_ch7_3class_features.pkl` (gitignored), which includes participant IDs.
  - Same-named metrics are defined differently from 4-class: rate_entropy is the entropy of a 20-bin histogram of neuron rates (`:295-303`); rate_variance is variance over time of population rate (`:306`); tau_ac uses a different estimator with cap 64.
  - Extras: LZ76 on 32 neurons, `lambda_proxy` (mean log ratio of successive |Δ mean membrane|, not Benettin), tau_relax, T_RTB (α=0.1, sustain 5).
  - tPLV has diagonal 1; clustering is 2·Σ(unordered)/(k(k−1)).
  - Duplicate files resolve by `os.walk` order (`:116`).
- **02_raw_observations**: optional `./clinical_profile.csv` (ID, MDD, PTSD, GAD, SUD, ADHD, EAT, Mania, OCD, PDA, SAD, Psychiatric_Medication, Assigned_Sex). Docstring promises 8 `.npz`; the code writes one containing only metric names (`:658`).
- **03_experiments**:
  - Friedman and Wilcoxon; Mann–Whitney for 5 diagnoses plus medication and sex (1 vs 2) as grouping variables; no correction anywhere.
  - Φ = constant/total_spikes, with the constant from hardcoded 79.4% accuracy (`:542-548`).
  - HC = zero of 5 primary diagnoses, NaN dropped.
  - SVC-RBF with `StratifiedGroupKFold(10)` by subject, scaler in-fold.
  - Writes `./ch6_results.pkl` and `./ch6_figures/`.
- **04_experiments**:
  - κ as in A. The null is group-level: mean κ over 633 observations with every observation's electrodes permuted, ×5000; p has no +1 (`:219`).
  - "ICC(3,1)" is var(subject means)/(that + mean within-subject variance) (`:360-363`), not the ICC(3,1) formula.
  - Mann–Whitney by diagnosis, uncorrected.
  - LogReg with `StratifiedGroupKFold(5)`, single run, balanced accuracy.
  - EXP-7.5 is a Neg−Pos cell contrast (Bonferroni 14, sign-flip ×5000); the docstring and README describe 4-class Cute–Erotic.
  - Writes `./ch7_3class_results.pkl` and `./ch7_figures/`.

## 3. Summary table

| Script | Kind | Restricted data? | Writes tracked files? | Status (source) |
|---|---|---|---|---|
| `C6/run_chapter6_exp1`…`exp6` (7) | runner | yes (exp5 also xlsx) | no (CWD `figures/`, `results/`) | Archived (map) / "4-class exploration" (README) |
| `C6/reproduce_chapter6.py` | reproduction | yes | no | no tier assigned |
| `C7/run_chapter7_experiment_A.py` | runner | yes | yes (`chapter7_results/*.pkl`) | exploratory (manifest[14]) and archived (manifest[21], map) |
| `C7/…_B.py`, `_C.py` | runner on committed CSV | no | yes (B: npz; both: `pictures/chSynthesis`) | Archived |
| `C7/…_D.py`, `_E.py` | runner | yes (xlsx) | no (`/mnt/...`) | Archived |
| `C7/extract_C_matrices.py` | utility | no (needs pkl) | yes | none |
| `C7/extract_kappa_matrix.py` | utility | no | stdout only | none |
| `X3/…01…`–`04…` | runner | yes | no | Tier 2 Exploratory |
| 5 `verify_*.py` | verifier | no | `verify_chapter7.py` yes, via B and C | CI (`.github/workflows/verify.yml:83-118`) |

Imports: `verify_chapter6.py:83` loads exp1; `verify_reproduce_chapter6.py:61` loads reproduce; `verify_ch6_ch7_3class.py:74` loads 01; `verify_chapter7.py:141,219` runs B and C as subprocesses. Nothing else in scope is imported.

## 4. `chapter7_results/` metadata

None of the seven files contains a clinical, diagnostic or demographic column or key. Subject identifiers are the raw SHAPE IDs (non-sequential, max >211, 127 absent).

| File | Size (bytes) | Shape / keys | Keyed by participant ID? | Writer |
|---|---|---|---|---|
| `C_matrices.csv` | 122,509 | 844×16: `subject`, `category`, 14 `<dyn>_x_<topo>` | yes | `extract_C_matrices.py` |
| `kappa_matrix.csv` | 8,391 | 211×5: `subject`, Threat, Mutilation, Cute, Erotic | yes | `extract_kappa_matrix.py` stdout redirect |
| `observation_features.csv` | 2,513,025 | 844×308: `subject`, `category`, 238 `d_<metric>_<ch>`, 68 `t_<metric>_<ch>` | yes | no script in repo |
| `subject_features.csv` | 630,491 | 211×307: `subject` plus the same 306 | yes | no script in repo |
| `ch7_expA_analysis.pkl` | 131,731 | dict: `all_kappa`, `all_null_median`, `all_pvals` (844,), `all_C` (844,7,2), `mean_C`, `pval_C`, `sig_mask` (7,2), 3 scalars, `cat_kappas` (4 lists of 211) | no (implicit order only) | Exp A `--analyze`; keys match |
| `ch7_expB_results.npz` | 58,132 | 23 arrays incl. `K` (211,4), `subjects` (211,), `null_F` (5000,) | yes | keys do not match current B (`:290-302`); written by an earlier version |
| `ch7_full_results.pkl` | 23,697,325 | dicts keyed `(int id, str category)`, 844 each: `dyn_profiles` (34,7), `topo_profiles` (34,2), `plv_matrices` (34,34), `coupling_C`, `coupling_kappa`, `null_kappa` (2000,); plus `subjects`, `completed_subjects`, names, `n_permutations`=2000 | yes | Exp A batch mode, earlier parameterisation |

Both pickles were opened with a numpy-only restricted unpickler.

## 5. Docs mapping

- `results_manifest.json` results[14] (tier exploratory) maps to `run_chapter7_experiment_A.py`. It states 5000 permutations per observation; the committed pickle holds 2000 (`C7/README.md:33` also says 2000).
- results[15] (exploratory) maps to `ch7_04_experiments.py`. It describes a "median across 211 subject-averaged kappa" and a "one-sample Wilcoxon against zero"; the script computes neither.
- results[21] (archived) covers the `run_chapter6_exp*.py` and `run_chapter7_experiment_*.py` globs. It says 210 participants / 840 observations; scripts and committed files have 211 / 844.
- `docs/REPRODUCTION_MAP.md` puts X3 01–04 in Tier 2 Exploratory and the 4-class C6/C7 runners in Tier 3 Archived. Its verifier counts (31/42/38/40/28) match my counts.
- Not mentioned by map or manifest: `reproduce_chapter6.py` and both `extract_*` (only in `docs/VERIFICATION_METHODOLOGY.md`).
- References to missing files:
  - `extract_features_for_expE.py` (`C7/README.md:328,437`; Exp E docstring).
  - `CHAPTER6_VERIFICATION_REPORT.md` reviews `run_chapter6_exp*_ (1).py` and a 340-line gain-ratio valueadd; the current file has 268 lines and no gain. `C6/README.md:119-120` "6.8×/0.59×" comes from that version.
  - X3 README says `clinical_profile.csv` is in `data/`; the scripts read `./`.
- Other doc/code mismatches:
  - `C7/README.md:232` says 2000 interaction permutations; D uses 5000.
  - X3 README "Key Results" for Chapter 7 are worded as "expected".
  - `C6/README.md:108` says "all three" surrogates; the code gate is 2 of 3.
- No script carries an in-code legacy, archived or superseded marker; "legacy input slicing" is not attributed to specific scripts anywhere I read.

## 6. Verifiers

| Verifier | Checks | Substantive numerical | Notes |
|---|---|---|---|
| `verify_chapter6.py` | 31 | 5 (ESP convergence on the real class; 4 surrogate properties on inline code) | 16 existence/syntax/shape, 10 weak range. Metrics are re-implemented inline, not the scripts' functions |
| `verify_reproduce_chapter6.py` | 42 | 13 (7 on repo code: spectral radius, λ<0, floor, determinism; 6 inline) | 15 are source-string checks (`"pe" in source` is always true). Unseeded `np.random`; the "ESP convergence" block never compares distances (`:171-172` unused) |
| `verify_chapter7.py` | 38 | 9 (7 regressions against archived values, 2 range) | 29 existence/shape/exit-code. Does not check the npz |
| `verify_extract_utilities.py` | 40 | 0 | Never imports or executes the extract scripts; all logic is re-typed on mock data. 4 unconditional `True` (`:97,104,130,131`); `k_from_C` unused |
| `verify_ch6_ch7_3class.py` | 28 | 0 against repo metric or tPLV functions | Inline re-implementations with different definitions; 3 constant checks |

## 7. Confident bugs and maintainer notes

1. Exp A clustering is 2× the Onnela value: `2·diag(W^⅓ cubed)/(k(k−1))` with an ordered-pair diagonal (`A:223-229`). C and κ are unaffected (rank-based); stored `topo_profiles` and the `t_clustering_*` columns are affected. X3 01 is correct.
2. X3 03 "Fano bound" omits the `P_e·log2(K−1)` term (`:544`). It is a constant factor, so tests are unaffected.
3. Running `verify_chapter7.py` (it is in CI) or B overwrites the committed `ch7_expB_results.npz` with a different key schema.
4. `defense_figures/*` read `X_ds`, `y`, `subjects` from `chapter6Experiments/results/ch6_exp1_full.pkl`; exp1 writes none of those keys (`exp1:221`).
5. "7 metrics" means three different sets: C6 exp6, C7 A, and X3 with redefined metrics.
6. X3 04 recomputes paired arrays inside 5000×14 loops (`:705-716`) and makes about 44M `spearmanr` calls in EXP-7.1. Unused imports and variables: `SVC`, `RepeatedStratifiedKFold`, `neg_vals`.
7. exp4, exp6 and A index hardcoded subject positions and assume all four categories exist per subject; paired arrays would misalign otherwise (`exp4:158-167`).
8. X3 02 lists OBS-4/5/6 twice (`:645-650`).

Not determined: the generator of the two `*_features.csv` files; whether class B equals the Chapter 4 reservoir; any reported numeric value.

