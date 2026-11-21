<!-- Audit appendix B. Repository state: commit 913d457, 2026-11-21.
     Written by a read-only reviewing agent that read the code and ran nothing in the repository.
     Reproduced unedited. Statements are that reviewer's reading of the code, cited by file and line.
     They were not each re-verified; docs/REPOSITORY_AUDIT.md says which ones were. -->

# Audit: experiments/{confirmatory, ch5_4class, novel, external, ablation}

Root: `/home/user/dissoAdventuresv2/` (paths below are relative to it). Nothing in the repo was executed or edited. Read-only extras I ran: `sha256sum`, JSON parsing, `git cat-file/merge-base/log`, and one `git status --porcelain` (may refresh the index stat cache; no content change).

## Findings that would mislead a registry

1. **`ablation/layer_ablation.py` cannot be regenerated from the repo.** It needs `shape_features_211.pkl` (l.142), and no `.py` in the repo writes that file. `ablation/README.md` says `chapter5Experiments/run_chapter5_experiments.py` generates it, but that script has no `pickle`/`open(` write.
2. **Manifest and execution record disagree on the generating commit.** `results_manifest.json` says `f6a0c66` throughout; `confirmatory_results.json` provenance says `05acc31ca480…`. Both commits exist.
3. **Manifest entry `exp_4class_classification` is not produced by its listed script.** Its value is `{"centered":0.52,"raw":0.404}`, but `ch5_4class_03` has no centering code. The same figures appear as hardcoded strings in `layer_ablation.py:651-666` (`regime_table`), which are pickled as results.
4. **`ch5_4class/README.md` claims "PCA fitted per fold on training data only"; the code fits on all data.** `01:358-361`: `pca = PCA(...); pooled_pca = pca.fit_transform(pooled)`. `03:187-188`: `F_rel = pca_rel.fit_transform(F_rel_raw)`. The README also names LogReg/MLP/GAT and GCN/GraphSAGE; the code uses RBF-SVC and untrained random-weight GIN/ChebNet/EdgeConv/Transformer.
5. **`novel/README.md` is stale against the two results files.** It still asserts the PCA-nuisance mechanism for N1, "calibrated instrument", "graceful degradation", dim 128, and angles 1.34°/3.38°; `NOVEL_RESULTS.md` and `NUISANCE_RESULTS.md` retract these. The committed `novel_results_*.json` `interpretation` strings (hardcoded at `run_novel_experiments.py:75-81,142-151`) still carry the retracted wording.
6. **Committed novel/external JSONs are not byte-reproducible from the runners.** All three carry a `note` field no runner writes, and the 3-class interpretation text differs from the code string.
7. **`confirmatory_pipeline.TEMPORAL_WINDOWS` labels do not match the slices** (l.306-312). Labels read "39-273ms" etc.; the code slices fractions 0–25/25–50/… of the 256 samples.
8. **Two different encodings share the name "BSC6".**

| | Confirmatory / novel / external | `ch5_4class` |
|---|---|---|
| Reservoir | per-channel, seed `reservoir_seed + ch*17` (`confirmatory_pipeline.py:85-87`) | one shared, seed 42 (`01:292`) |
| Window | full 256 samples, bin 42 | t=10–70, bin 10 (`01:63-64`) |
| Layout | bin-major | neuron-major |
| Prestimulus | 205 samples removed | not removed (`01:232-238`) |
| Compression | fixed SRP-64 | global PCA-64 |
| Exclusions | 127 (+195 for 4-cat) | 127 only |

## Shared confirmatory machinery

- **Reservoir** (`chapter5Experiments/experiment_zero.py:68-81`): N=256, β=0.05, θ=0.5, spectral radius 0.9.
- **Compression**: SparseRandomProjection 1536→64, seed 20240517, fitted on a ones-matrix (l.59-61), shared across electrodes.
- **CV**: `StratifiedGroupKFold(5, shuffle, random_state=base_seed+repeat)` × 5 repeats; `StandardScaler().fit(X[train_idx])`; `LogisticRegression(max_iter=2000, C=1.0)` (l.132-146). No step is fitted before the split.
- **Intervals**: participant-cluster percentile bootstrap, n=2000, seed 12345, over OOF predictions pooled across repeats; paired version for contrasts.
- **Permutation**: within-participant label shuffle, n=100, RidgeClassifier, 2 repeats (l.234, 252), p=(1+k)/(1+n). No multiple-comparison handling.
- **Seeds**: base 0 (3-cond), 100 (4-cat); reservoir 42, robustness 7/42/99; destruction rng base+101. Deterministic.

## Per script

**confirmatory/**
- `confirmatory_pipeline.py`: library; docstring matches code. Imported by `run_confirmatory_validation`, `verify_confirmatory`, all of novel/ except `check_novel_manifest`, and `external_replication`. Its `sys.path` entry for `experiments/ch5_4class` (l.34-39) is dead: nothing is imported from there.
- `run_confirmatory_validation.py`: runner and SHAPE loader.
  - CLI: `--data3 ./batch_data --data4 ./categories --out confirmatory_results.json --skip4`.
  - Preprocessing: drop 205 samples, zero-phase `decimate(4)` to 256, per-epoch/channel z-score (l.66-88). Hard asserts 211/633 and 210/840.
  - Representations: raw-ERP16, 5-band+Hjorth (fs=256), SRP-64, fusion; three destruction controls; seed and window sweeps.
  - Writes the JSON with a provenance block (l.428-479).
  - Also imported by `novel/run_*` and `experiments/v2/data.py`.
- `verify_confirmatory.py`: synthetic verifier, 18 checks.
- `check_against_manifest.py`: manifest checker.
  - No argument: schema-only (11 expected ids, required fields, script path).
  - With a results file: 56 comparisons — cohort counts (exact), 8 accuracies with CIs, BSC6−conventional with CI, permutation n and p (tolerance 1e-4), 3 controls with CIs. Tolerance 0.05 pp absolute.
  - Not compared: fusion−BSC6, rawERP−BSC6, seed robustness, temporal windows, per-repeat values.
- `validate_provenance.py`: provenance checker.
  - Checks 11 required fields are present; recomputes sha256 of `{protocol, granularities}`; re-hashes every file listed in the record's `code_sha256`; checks `git merge-base --is-ancestor commit HEAD`.
  - Hashed files: `run_confirmatory_validation.py`, `confirmatory_pipeline.py`, `chapter5Experiments/experiment_zero.py`.
  - Ancestry only warns (exit 0) if git or the commit object is absent.
  - It does not hash data, the manifest or verifiers, does not require a clean tree, and passes vacuously on an empty hash dict.
  - Current state, which I checked: all three hashes match, the payload digest matches, and `05acc31` is an ancestor of HEAD. Any edit to those three files breaks the record until a restricted-data rerun.

**ch5_4class/** (hardcoded cwd-relative paths, no CLI)
- `01_feature_extraction.py`: runner with its own duplicated `LIFReservoir`.
  - Input `./categories/categoriesbatch{1-4}`; output `./shape_features_4class.pkl`.
  - `FS_ORIGINAL = 1000` is marked "assumed" (l.72). Incomplete panels only warn.
  - Docstring cites non-existent filenames (`ch5_4class_feature_extraction.py`, `ch5_4class_classification.py`).
- `02_raw_observations.py`: descriptive only, no CV and no seeds. Writes 7 PDFs and 7 npz to `./ch5_4class_raw_data/`. The README says "tPLV"; the code computes Pearson correlation of PCA embeddings.
- `03_classification_full.py`: runner; writes `ch5_4class_results.pkl` and 9 PDFs.
  - CV: subject-level `StratifiedKFold(10, seed 42)`, single repeat (l.129-131). The stratification label is `int(median(y))`, constant for complete panels.
  - Classifier: `SVC(rbf, class_weight='balanced')`; fold-local scaler.
  - GNN adjacency is chosen by best CV accuracy (l.302), then Wilcoxon against M3 with a bare `except → p=1`.
  - Clinical tests: Mann-Whitney/Spearman at raw p<0.05, no correction.
  - `rng` at l.263 is unused.
- `verify_ch5_4class.py`: 25 checks. Docstring claims PCA and classification are tested; they are not. Scripts 02 and 03 get only `ast.parse`.

**novel/**
- `novel_pipeline.py`: library; imported by `run_novel`, `verify_novel`, `nuisance_study`, `external_replication`. Contains fold-local blockwise PCA (fit on train electrodes only, l.112-115), ICC(2,1), a transfer function, perturbations (seed 777), a subject/condition SS ratio computed on all data (descriptive), and the necessity map (rng base+4242).
- `run_novel_experiments.py`: runner on SHAPE.
  - CLI: `--data3 --data4 --granularity --out novel_results.json --fast`.
  - Dims 16/32/64(/128); SRP seeds 1000–1009 (fast) or 1000–1024; transfer template is `raw[0,:,0]`.
  - No provenance block.
- `nuisance_study.py`: library; imported by `run_nuisance_study`, `verify_nuisance`, `external_replication`. PCA has no `random_state` at l.83, 244, 320 (possible nondeterminism; `NUISANCE_RESULTS.md` admits ρ drift). Dead placeholder at l.246. Geometry is fitted on the full matrix by design.
- `run_nuisance_study.py`: runner. M1a, M1b and M3 are synthetic (generator seeds 0 and 1); M2 and M5 need SHAPE. No provenance block.
- `verify_novel.py` (13 checks), `verify_nuisance.py` (14 checks): synthetic.
- `check_novel_manifest.py`: 42 comparisons at absolute tolerance 0.001; all 5 JSONs mandatory. Covers N1 (both granularities), M2 global and blockwise, external N1 and M2. Manifest entries for N2, N3, M1a and M5 exist but are not checked; M1b and M3 are not in the manifest at all.

**external/**
- `load_tcrzem.py`: loader for BrainVision int16 files. Imported only by `run_tcrzem_replication`. `load_trials` is unused by the runner. The full 1792-sample epoch (−1000..+2500 ms) is Fourier-resampled to 256 with no prestimulus removal (l.160). No verifier touches this file.
- `external_replication.py`: library (`prepare_epochs`, `replicate`). N1 is blockwise; M2 is global only (l.126).
- `run_tcrzem_replication.py`: runner. CLI: `--data ./external_data/tcrzem --out external_results.json --max-subjects --dims`. Needs TCRZEM.
- `verify_external.py`: 11 checks. A comment promises a noise-reduction test; the check is only `isfinite`.

**ablation/**
- `layer_ablation.py`: top-level script with no main guard; hardcoded `./` inputs.
  - Emotion task: `StratifiedGroupKFold(10, seed 42)`, single repeat, fold-local scaler, `LogisticRegression(C=1, balanced, multi_class='multinomial')` (l.266). That keyword may be rejected by recent sklearn; not executed.
  - Clinical task: 5-fold on subject-mean features, five diagnoses (l.501).
  - No statistical tests. Whether the E block's upstream PCA was global: not determined (producer absent).
  - The README's A5–A7 and C1–C6 definitions contradict the code.
- `verify_ablation.py`: 23 checks. It never imports `layer_ablation`; it re-implements the logic inline.

## Summary table

| Script | Kind | Restricted data | Committed output | Tier |
|---|---|---|---|---|
| `confirmatory_pipeline` | library | no | – | confirmatory |
| `run_confirmatory_validation` | runner + loader | SHAPE | `confirmatory_results.json` | confirmatory |
| `verify_confirmatory` | verifier | no | – | – |
| `check_against_manifest`, `validate_provenance` | checkers | no | – | – |
| `ch5_4class_01/02/03` | runners | SHAPE (+ clinical csv) | none | manifest "exploratory", described "archived preprocessing" |
| `verify_ch5_4class` | verifier | no | – | – |
| `novel_pipeline`, `nuisance_study` | libraries | no | – | exploratory |
| `run_novel_experiments` | runner | SHAPE | `novel_results_{3,4}class.json` | exploratory |
| `run_nuisance_study` | runner | SHAPE for M2/M5 | `nuisance_results{,_4class}.json` | exploratory |
| `verify_novel`, `verify_nuisance`, `check_novel_manifest` | verifiers / checker | no | – | – |
| `load_tcrzem`, `external_replication` | loader / library | TCRZEM / no | – | exploratory |
| `run_tcrzem_replication` | runner | TCRZEM | `external_results.json` | exploratory |
| `verify_external` | verifier | no | – | – |
| `layer_ablation` | runner | SHAPE-derived pickles | none | manifest "archived"; own README "keystone" |
| `verify_ablation` | verifier | no | – | – |

No file inside `ch5_4class/` or `ablation/` labels itself archived. "Pre-registered" appears nowhere in scope; the nearest are "prespecified seed" and "PREDEFINED DECISION RULES" in the `layer_ablation` docstring.

## Result JSONs

- **`confirmatory_results.json`**: keys `granularities{three_condition, four_category}`, `protocol`, `provenance`, `schema_version` 2.0.0. Full provenance: commit `05acc31`, clean tree, 3 code hashes, numpy 2.4.6 / scipy 1.17.1 / sklearn 1.9.0, py 3.11.15, seeds, file counts 636/848.

| | 3-cond | 4-cat |
|---|---|---|
| SRP-64 | 64.33 [61.14, 67.24] | 54.64 [51.90, 57.33] |
| Raw ERP16 | 69.61 | 60.57 |
| Conventional | 48.66 | 36.14 |
| Fusion | 65.06 | 54.98 |
| BSC6 − conventional | +15.67 | +18.50 |
| Bin shuffle / collapse / electrode shuffle | −24.77 / −17.57 / −24.08 | −27.76 / −22.40 / −22.05 |

  Permutation p = 0.0099 in both (the minimum attainable with n=100).
- **`novel_results_3class.json` / `_4class.json`**: keys `granularity`, `protocol`, `results{N1,N2,N3}`, `note`; fast mode; no provenance.
  - N1 d64: 64.329 vs 59.968, +4.361 [1.453, 7.173]; 4-cat 54.643 vs 50.738, +3.905 [1.476, 6.405].
  - SRP seed stability: 64.009 ± 1.03; 54.064 ± 1.097.
  - N2: ICC 0.0025 / 0.0027; Spearman 1.0.
  - N3: intact ρ 9.816 / 14.926; bin-shuffle −25.846 / −28.238.
- **`nuisance_results.json` / `_4class.json`**: keys `study`, `components{M1a, M1b, M3, M2_geometry, M2_geometry_blockwise, M5}`, `granularity`; fast ratios; no provenance.
  - M2 global: angle to subject 1.33° / 1.58°, ρ_pca 43.554 / 31.798.
  - M2 blockwise: 23.9° in both, ρ_pca 7.159 / 15.528.
  - M1b crossover: null. M3: PCA 98.778, SRP 91.889.
  - M5: 8.679 / 11.407 / 9.816.
  - The M1a curve has ρ_srp > ρ_pca at ratios 2 and 4.
- **`external_results.json`**: keys `dataset`, `paradigm`, `cohort`, `results`, `note`; no provenance. 228 participants / 684 observations / 41 channels. d64: 45.497 vs 40.468, +5.029 [2.047, 7.953]. Global angle 0.89°, ρ 132.8 / 111.2 / 111.1.

## Dependency graph

- `chapter5Experiments/experiment_zero.py` ← `confirmatory_pipeline` (lazy import, l.79) and `novel_pipeline` (l.212).
- `confirmatory/` ← `novel/*`, `external/external_replication`, `experiments/v2/data.py`.
- `novel/` ← `external/external_replication`.
- `check_novel_manifest` reads `external/external_results.json`.
- `ch5_4class` and `ablation` import nothing cross-directory and nothing imports them. `ablation` consumes pickles from `ch6_ch7_3class/ch6_ch7_01` and the missing chapter-5 producer.
- All linkage is `sys.path.insert`; there are no packages.

## What the studies' own .md files claim (repo tier: exploratory)

**`NOVEL_RESULTS.md`**
- SRP beats fold-local PCA at every dimension: +3.9 / +5.3 / +4.4 pp, intervals exclude zero.
- The four-category result is a "cross-granularity consistency check… not an independent-cohort replication".
- N2 is "preliminary"; the crossed design is "not yet done".
- N3: temporal-bin order is most necessary (−25.8 pp); ρ 9.8 → 97.
- Numbers come from a `--fast` run.

**`NUISANCE_RESULTS.md`**
- Global PCA is near-collinear with the subject subspace (1.33°, ρ 43.55); blockwise is not (23.9°, ρ 7.16).
- "We therefore do not claim M2 explains the N1 accuracy gap."
- M1b: PCA beats SRP at every nuisance level.
- M5: the reservoir does not disentangle (8.68 vs 11.41).

**`PAPER_OUTLINE.md`**
- Target is a methods letter: leak-safety plus a mild-regime +4.36 pp.
- States "every number is in results_manifest.json"; this does not hold for M1b and M3.

**`EXTERNAL_RESULTS.md`**
- SRP − PCA is +6.7 / +5.6 / +5.0 pp.
- Absolute accuracy is near chance (45.5%).
- M2 is global-operator only; external blockwise geometry is "not run".

## Doc gaps

- **Missing files referenced**: `novel_results.json` (`novel/README`); `shape_features_211.pkl` producer (`ablation/README`); the wrong script names in the `01` and `03` docstrings.
- **Scripts no .md mentions**: `validate_provenance.py` (only `.github/workflows/verify.yml`). `load_tcrzem.py` appears only in its local README.
- **`experiments/README.md`** omits confirmatory/, novel/ and external/ entirely.
- **`docs/REPRODUCTION_MAP.md`** has no result rows for the novel or external runners, and files `layer_ablation` (3-class) under "Archived four-category".
- **`docs/VERIFICATION_METHODOLOGY.md`** rates `layer_ablation` "Component + integration" and `03` "Syntax + component"; neither script's code is exercised. Its coverage map omits confirmatory/novel/external.

## Verifier checks (104 in scope)

Substantive means a behavioural or numerical assertion, by my classification. None compares against real-data reference values.

| Verifier | Checks | Substantive | Notes |
|---|---|---|---|
| `verify_confirmatory` | 18 | ~15 | |
| `verify_novel` | 13 | 7 | |
| `verify_nuisance` | 14 | 7 | |
| `verify_external` | 11 | 1 | |
| `verify_ch5_4class` | 25 | 8 weak invariants | 7 constant-equality, 3 syntax |
| `verify_ablation` | 23 | ≤3 | on inline re-implementations; 9 imports, 8 tautologies such as `34*64 == 2176` or `check(..., True)` |

## Not determined

- Runtime behaviour of any script.
- sklearn's solver choice for the unseeded PCAs.
- Contents and provenance of the ablation pickles.
- Whether the `raw: 0.404` manifest value comes from `03`.

