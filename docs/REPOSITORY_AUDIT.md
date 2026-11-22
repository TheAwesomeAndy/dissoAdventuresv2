# Repository audit

Audit date 2026-11-21. Repository state audited: commit `913d457` on branch
`ccr-0221a213-rhj47r` of `dissoAdventuresv2` (566 tracked files, 109 Python files, about
36,000 lines of Python, 214 MB of git history).

This document records what was found, what was changed in response, and what is proposed
but not done. The last section separates those three explicitly.

## 1. Method and baseline

- **Baseline.** All 22 CI verifier commands were run in a scratch clone of `913d457`
  before anything was changed. All 22 exited 0 and reported 772 checks. The per-suite
  counts are stored in `scripts/run_verifiers.py` as the baseline a suite may not fall
  below.
- **Code reading.** Four independent read-only reviewers read every Python file and
  README outside the v2 code and reported per-script facts with file and line citations.
  Their reports are reproduced unedited in `docs/audit/`. They ran nothing in the
  repository.
- **Spot checks.** Eight of the most consequential reviewer findings were re-verified
  against the code by a second reading (marked "verified" below). The rest are the
  reviewers' readings and are marked "reported".
- **v2 code.** The v2 package and experiments were written in November 2026 and audited
  directly: every committed record was checked against the code that produced it, and a
  subset of the main record was replayed on the data (section 6).
- **Not done.** No v1 experiment was rerun on the SHAPE data. No reported v1 number was
  re-derived. Nothing in `dissertation/` was read for scientific content.

## 2. What the repository is

Three layers of different age and authority share one tree.

| Layer | Where | Status |
|---|---|---|
| **v1 dissertation record** | `chapter4Experiments/` to `chapter7Experiments/`, `experiments/chapter3/`, `experiments/ch5_4class/`, `experiments/ch6_ch7_3class/`, `experiments/ablation/`, `experiments/confirmatory/`, `experiments/interpretability/`, `experiments/novel/`, `experiments/external/`, `defense_figures/`, `validation/`, `dissertation/`, `pictures/` | The defended work and its corrective audit. Frozen. |
| **v2 validation** | `arspinet_v2/`, `experiments/v2/`, `V2_UPGRADE.md` | Tests of the v1 claims on the authorized data. The current scientific position. |
| **z-scoring artefact study** | `experiments/zscore_artifact/` | A new line of work on public data that grew out of v2. Ongoing. |

The authoritative statement of what the data show is `V2_UPGRADE.md`. Where it and any
v1 document disagree, v2 governs.

## 3. Findings

Severity: **S** sensitive data, **I** scientific integrity and traceability, **E**
engineering. Within each group the order is by priority.

### S. Sensitive data and security

**S1. Participant-level derived data is tracked, keyed by real participant IDs
(verified).** Nine files, listed in `docs/participant_level_files.json`: seven under
`chapter7Experiments/chapter7_results/` and two defense-figure data files. They hold
per-electrode features, coupling matrices and per-trial statistics for all 211
participants. None has a clinical or demographic column. `LICENSES.md` states that no
participant-level data is included.

**S2. The clinical table is published (verified).** The file data/clinical_profile.csv
(211 rows, 54 columns: participant ID, SCID-5 diagnoses, age, sex, race, ethnicity,
medication, substance use) was committed on 2026-03-20. It was untracked on this branch
on 2026-11-11 but remains at the tip of `main` and of 15 other branches in this
repository, at the tip of `main` in the original repository, and in the history of
both. Both repositories are public. Joined with S1 on the shared ID column, it links
every tracked feature row to a diagnosis.

**S3. CI depends on the participant-level files (reported).**
`chapter7Experiments/verify_chapter7.py` asserts their presence and minimum sizes.
Removing them needs a matching change to that verifier.

**S4. No credentials were found** in tracked files or in pickaxe searches of history
for common key and token patterns (reported). Commit metadata carries three author
email addresses. An old revision of the bibliography carried a session token in a
publisher URL.

**S5. Third-party and personal material (reported).** Two images
(`pictures/ch1/OIP.jpg`, `pictures/ch2/LSMArchitecture.png`) have no recorded source or
licence and are referenced by no source file. `validation/validate_shape_data.py`
quotes a private email from the data provider in its docstring.

S1 and S2 need a decision by the data owner. Nothing was deleted and no history was
rewritten in this pass. See section 8.

### I. Scientific integrity and traceability

**I1. Results whose generating code is not in the repository (verified by search).**
No committed script writes any of these, yet committed results or other scripts depend
on them:

- shape_features_211.pkl, needed by five scripts and by `experiments/ablation/layer_ablation.py`;
- `chapter7Experiments/chapter7_results/subject_features.csv` and `observation_features.csv`;
- `chapter5Experiments/results/baseline_deep_results.pkl`;
- the 633-trial pickle used by defense experiments A, B and D (the committed writer of
  that filename produces different keys);
- the "PCA-200" baseline row (the repository's own documentation says it has no
  committed generator).

**I2. The results manifest maps numbers to scripts that do not compute them
(reported, three verified).** In `results_manifest.json`:

- the Level-1 value r = 0.82 is mapped to a script whose docstring says it does not
  compute it; in the Level-3 script it is a printed literal;
- the Level-2 value rho = 7.2 is mapped to a baseline script that computes no variance
  decomposition;
- `arch_pca64_svm` and `arch_gnn_sweep` are mapped to a script with no SVM row, GIN or
  EdgeConv;
- `exp_4class_classification` reports values that appear only as hardcoded strings in
  another script;
- the "33% inter-channel variance" figure is hardcoded (190 to 99), not computed;
- the manifest cites generating commit `f6a0c66` in 21 places; the confirmatory record
  itself says `05acc31` (verified).

**I3. Documentation claims leak-free fitting where the code fits on all data
(verified).** `chapter5Experiments/README.md` and `experiments/ch5_4class/README.md`
say PCA is fitted per fold. The code calls `fit_transform` on all observations before
the split. `chapter5Experiments/verify_reproduce_chapter5.py` asserts "PCA fitted per
fold (no leakage)" by checking that the string `fit_transform` occurs in the source,
which is true of the leaking code.

**I4. Held-out labels set model weights (verified).**
`chapter5Experiments/reproduce_chapter5.py` line 279 scores each channel on the test
fold's labels and uses those scores as attention weights for that fold.

**I5. Committed result files that current code does not write (reported).** The
committed Chapter 7 experiment B file has keys the current script does not produce. The
Chapter 7 pickle holds 2,000 permutations per observation where script and manifest say
5,000. The five exploratory JSON files under `experiments/novel/` and
`experiments/external/` carry a hand-added field, and their producing code changed
after they were committed (verified from git history). They are now tracked as
"unverified records" in `experiments/records.lock.json`.

**I6. Confident code defects (two verified, the rest reported).**

- `defense_figures/experiment_c_memory_capacity/make_experiment_c_figures.py` line 266
  unpacks a `(membrane, spikes)` return as `(spikes, _)`, so the fading-memory time
  constant is computed on membranes (verified).
- `experiments/interpretability/run_eegnet_saliency_comparison.py` line 191 has an
  inverted axis test, so the network receives time as channels for either input layout
  (verified).
- Chapter 7 experiment A computes clustering at twice the Onnela value (coupling and
  kappa are rank-based and unaffected).
- Chapter 6 experiment 5 reseeds its generator identically for all 45 tests.
- Several permutation p-values omit the +1 correction and can report exactly zero.
- Conclusion text in Chapter 7 experiments D and E is hardcoded.

None of these was changed. Each alters a reported or archived result, so each needs a
decision (section 8).

**I7. One name, several definitions (reported).**

- *Reservoir.* Sixteen `LIFReservoir`-style definitions fall into four classes: the
  dense uniform v1 operator (deployed for classification); a dense Gaussian variant
  (three-condition extension); a sparse Gaussian network with no spectral scaling and a
  different reset rule (all of Chapter 6, Chapter 7 experiment A, defense experiment
  A); and non-spiking comparators. The Chapter 6 echo-state and Lyapunov evidence
  therefore describes a different network from the one that produced the
  classification features.
- *BSC6.* The confirmatory code uses per-electrode reservoirs, the full epoch, bin-major
  layout and SRP-64. The four-category extension uses one shared reservoir, samples 10
  to 70, neuron-major layout and global PCA-64, and does not remove the baseline.
- *"Seven metrics"* means three different sets, and several same-named metrics are
  defined differently in the three-condition extension.

**I8. Contradictions between documents (reported).** Eleven are listed in
`docs/audit/appendix_D_figures_docs_security.md`. The ones that matter most: the
manifest and v1 README report r = 0.82 where `V2_UPGRADE.md` withdraws it; the manifest
and `experiments/novel/NOVEL_RESULTS.md` report SRP over PCA by 4.4 points where v2
found no difference; the check count is stated as 491, 380, 102 and 76 in different
files (the measured baseline is 772); `RELEASE.md` calls the 104-page defense deck the
dissertation PDF, and no dissertation PDF is tracked.

**I9. Scientific assumptions that live only in code.** Collected in section 5.

### E. Engineering

**E1. Verifiers overwrite tracked files (verified).** Running the CI suites in a
working tree modifies 21 tracked files: 13 PDFs in `pictures/chLSMEmbeddings/`, 7 PDFs
in `pictures/chSynthesis/` and `chapter7Experiments/chapter7_results/ch7_expB_results.npz`.
Committing afterwards replaces dissertation figures with synthetic stand-ins. This
happened once and was caught. *Mitigated:* `scripts/run_verifiers.py`.

**E2. No packaging, no dependency declaration, no test framework.** Dependencies were
stated in four places that disagreed. *Fixed:* `pyproject.toml`,
`requirements-lock.txt`, `tests/`.

**E3. No packages; 39 files manipulate `sys.path`.** Scripts import each other by
inserting directories into the path. Cross-directory dependencies are invisible to
tools. The graph is in `docs/ARCHITECTURE.md`.

**E4. Hardcoded and working-directory-relative paths.** Six files hardcode absolute
paths from the original environment (`/mnt/user-data/...`, `/workspaces/...`). The
four-category, three-condition and ablation scripts read and write `./` relative paths
with no command line, and three of them execute at import.

**E5. Verifier quality (reported).** Of roughly 590 legacy checks reviewed, a minority
are substantive numerical assertions on repository code. Many are file-exists, shape or
source-string checks; several are tautologies; three verifiers re-implement the logic
they claim to test and never import the script. No verifier compares against a
real-data reference value. No verifier covers `defense_figures/`,
`experiments/interpretability/`, `experiments/chapter3/` or the animations.

**E6. Duplicated code.** Sixteen reservoir definitions; descriptor functions copied
verbatim between Chapter 6 scripts; loaders in seven files.

**E7. Documentation sprawl and drift.** 41 Markdown files. Directory READMEs list a
fraction of what exists (`experiments/README.md` named 3 of 10 subdirectories). There
was no entry point that said which document governs.

**E8. No agent instructions** other than a 400-line handoff inside `defense_figures/`.
*Fixed:* `AGENTS.md`, `CLAUDE.md`, `docs/AGENT_HANDOFF.md`.

**E9. Large binaries in git.** A 22.6 MB pickle, a 17 MB zip of the dissertation source
(matched by a `.gitignore` rule but tracked), 26 MB of animations, and 343 PDFs.

**E10. Inconsistent layout.** Chapters 4 to 7 are top-level `chapterNExperiments/`
directories; Chapter 3 is `experiments/chapter3/`.

**E11. Unseeded randomness (reported).** Most PCA calls have no `random_state`; two
validators and one verifier use the global NumPy generator unseeded.

**E12. Undeclared dependencies.** PyTorch (never installed in CI), PyMuPDF, openpyxl,
ffmpeg and pillow are needed by some scripts and were declared nowhere. openpyxl is now
an optional extra.

**E13. v2 records without embedded provenance.** Four of the eight v2-era records were
written without a provenance block. *Fixed* by reconstruction from git history in
`experiments/records.lock.json`.

**E14. v2 core code cannot be reformatted.** The six modules of `arspinet_v2/` and nine
experiment scripts are hashed in committed records. Any byte change breaks the link
between a record and its code. They are therefore excluded from the linter until a
record is regenerated.

## 4. Proposed architecture

### What was decided

**The tree was not physically restructured.** Moving the v1 directories would break,
together: the three file paths hashed in the v1 confirmatory record; the script paths
in `results_manifest.json` and `docs/REPRODUCTION_MAP.md`; 51 `sys.path` lines; the
paths the defended dissertation's reproducibility statements point to; and the
`v1.0-defense` reference. The v1 tree is a published record, and its paths are part of
what is cited. The cost of moving it is certain and the benefit is cosmetic.

Instead the architecture is expressed by **layering and indexing**:

```text
dissoAdventuresv2/
├── README.md  AGENTS.md  CLAUDE.md        entry points
├── V2_UPGRADE.md                          current scientific position
├── pyproject.toml  requirements-lock.txt  packaging, dependencies, tool config
├── arspinet_v2/                           AUTHORITATIVE library (frozen bytes)
├── experiments/
│   ├── records.lock.json                  index of records and the code behind them
│   ├── v2/                                v2 runners and records
│   ├── zscore_artifact/                   ongoing public-data study
│   └── confirmatory/ novel/ external/ ... v1 record
├── chapter{4,5,6,7}Experiments/           v1 record
├── defense_figures/ dissertation/ pictures/ validation/     v1 record
├── tests/                                 pytest suite (no data needed)
├── scripts/                               repository tooling
└── docs/                                  architecture, registry, research map, audit
```

Rules that make this navigable:

1. One authoritative implementation per concept, named in `docs/RESEARCH_MAP.md`.
2. Every experiment has an entry in `docs/experiment_registry.json` with its tier.
3. Every committed record is in `experiments/records.lock.json`, tied to its code.
4. New reusable code goes in a package; new experiments go under `experiments/<study>/`.

### Target structure, if a restructure is ever approved

A package under src/arspinet with data, reservoir, features, evaluation, controls,
nuisance and dynamics modules; one directory per study under experiments for runners
with configuration files; an archive/v1 directory holding the v1 tree unchanged; a
results directory for records. This requires regenerating the v2 records on the authorized data (about four
hours of compute), because their code paths and hashes would change. It should be done
in the same change as the next scientific rerun, not as a standalone cleanup.

## 5. Scientific assumptions embedded in code

These shape results and were not stated in prose until v2 or this audit.

| Assumption | Where | Consequence |
|---|---|---|
| Each epoch and channel is z-scored over the post-stimulus second | `experiments/confirmatory/run_confirmatory_validation.py` | Writes late activity into early samples. Window-specific results are uninterpretable (`experiments/v2/v2_normalization_check.json`). |
| The 200 ms baseline is discarded before normalization | same | No pre-stimulus reference survives in the model input. |
| Six bins of 42 samples; the last 4 of 256 samples are dropped | `chapter5Experiments/experiment_zero.py` | The final 16 ms never enters the code. |
| A different reservoir per electrode (seed 42 + 17 x channel) in the committed v1 runner | `experiments/confirmatory/confirmatory_pipeline.py` | The dissertation text describes one shared reservoir. v2 measured the difference: within noise. |
| Recurrent gain 0.9 | `chapter5Experiments/experiment_zero.py` | No fading memory on the deployed network (`experiments/v2/v2_results.json`). |
| Membrane is reset multiplicatively, then theta is subtracted, then clamped at zero | same | The soft reset is overridden by the hard one. No refractory period. |
| The projection is fitted on a matrix of ones | `experiments/confirmatory/confirmatory_pipeline.py` | It is data-independent by construction. |
| Participants 127 (both streams) and 195 (four-category) are excluded | loaders | Flat channels. Confirmed by the data provider and by a file check. Chapter 7 legacy results include 195. |
| Bootstrap intervals resample participants but do not refit models | `experiments/confirmatory/confirmatory_pipeline.py`, `arspinet_v2/evaluation.py` | Intervals are somewhat narrow. |
| With complete panels, stratified grouped folds are the same for any within-participant label permutation | `arspinet_v2/evaluation.py` | Lets the permutation test reuse fold designs. Fails silently for incomplete panels. |
| The electrode order in the data files is unknown | all loaders | No anatomical statement is possible. A spatially valid null for kappa cannot be built. |
| Sampling rate "assumed" 1000 Hz in the four-category extension | `experiments/ch5_4class/ch5_4class_01_feature_extraction.py` | The data are 1024 Hz. Affects that archived stream only. |

## 6. Reproducibility assessment

Evidence levels used here and in `docs/REPRODUCIBILITY.md`:

- **Replayed**: recomputed on the data in this audit and equal to the record.
- **Locked**: the record is tied by hash to code recoverable from git; not rerun.
- **Structural**: synthetic tests show the code behaves as described; no data rerun.
- **Unverifiable**: the generating code or its input is not in the repository.

| Result set | Level | Basis |
|---|---|---|
| v2 fading-memory sweep; v1-configuration and seven other representation rows per cohort | Replayed | `experiments/v2/check_v2_replay.py`, run 2026-11-21 with fresh reservoir encodings (preprocessed epochs reused from cache): all replayed values equal the record |
| v1 confirmatory headline (64.3%, 54.6%) | Replayed | recomputed bit-identically by the v2 code in November 2026 and again in the replay |
| Rest of `experiments/v2/v2_results.json`; the other four v2 records; both ERP CORE records | Locked | `experiments/records.lock.json` |
| v2 library behaviour | Structural | `tests/`, `experiments/v2/verify_v2.py` |
| v1 chapter results (Chapters 3 to 7, ablation, interpretability, defense figures) | Structural at best | legacy verifiers; no data rerun in this audit |
| Novel, nuisance and external JSON records | Unverifiable from repository state | code changed after the records were written (I5) |
| Results listed under I1 and I2 | Unverifiable | no generating code |

## 7. Risks to scientific preservation

1. **A cleanup that edits a frozen file** silently detaches a record from its code.
   Guarded by `tests/test_records.py`.
2. **A test run followed by a commit** overwrites figures (E1). Guarded by the isolated
   runner and stated first in `AGENTS.md`.
3. **Fixing a v1 defect in place** (I6) changes an archived result without a new
   record. v1 files are frozen; defects are recorded here and in the registry instead.
4. **Reading archived numbers as current.** Several v1 numbers are superseded or
   withdrawn by v2. The registry tiers and `docs/RESEARCH_MAP.md` state the status of
   each.
5. **Removing participant-level files** (S1) without updating the Chapter 7 verifier
   breaks CI and loses the only copy of results that cannot be regenerated without the
   restricted data.

## 8. Plan and status

### Done in this pass

| Change | Files |
|---|---|
| Baseline recorded; all suites run in isolation | `scripts/run_verifiers.py` |
| Packaging, consolidated dependencies, exact environment | `pyproject.toml`, `requirements-lock.txt` |
| Record index with hash verification against the tree and against git | `experiments/records.lock.json`, `scripts/check_records.py` |
| Unit, integration, record and hygiene tests | `tests/` |
| Replay check of the v2 record on the data | `experiments/v2/check_v2_replay.py` |
| Experiment registry (machine-readable, with generated document) | `docs/experiment_registry.json`, `docs/EXPERIMENT_REGISTRY.md`, `scripts/build_registry_doc.py` |
| Architecture, research map, reproducibility guide | `docs/ARCHITECTURE.md`, `docs/RESEARCH_MAP.md`, `docs/REPRODUCIBILITY.md` |
| Agent instructions and handoff protocol | `AGENTS.md`, `CLAUDE.md`, `docs/AGENT_HANDOFF.md`, `docs/handoffs/` |
| Entry-point README; v1 README preserved | `README.md`, `docs/v1/README_v1.md` |
| CI job for lint, tests and record check | `.github/workflows/verify.yml` |
| Classification of tracked data files, with a test that the set cannot grow | `docs/participant_level_files.json` |

No file of the v1 record was modified, moved or deleted, except that the v1 README text
moved to `docs/v1/README_v1.md`. No frozen v2 file was modified.

### Needs the owner's decision (not done)

1. **S2, S1: the published clinical table and participant-keyed files.** Options, in
   order of containment: make both repositories private; remove the clinical table from
   every branch and purge it from history; decide whether the participant-keyed derived
   files may stay, be re-keyed with study-internal pseudonyms, or be removed (with the
   Chapter 7 verifier updated); inform the data provider and the IRB.
2. **I6: code defects in archived experiments.** Fix and regenerate, or annotate as
   known-defective. Fixing changes archived results.
3. **I2, I8: manifest and v1 documents that contradict the code or v2.** Correct the
   manifest entries, or mark them superseded. They belong to the v1 record.
4. **Dissertation erratum.** Chapter 6 echo-state evidence (I7), the r = 0.82
   statistic, the PCA-64 comparisons and window-specific statements are affected by v2
   findings.
5. **S5: unlicensed images and the quoted private email.**
6. **Repository housekeeping from earlier sessions:** recreate the `v1.0-defense` tag
   in this repository; make `main` the default branch after merging this branch.

### Proposed (not started)

1. Regenerate the v2 records after formatting and typing `arspinet_v2/`, in one change,
   verified by the replay script. Move the runner constants to configuration files at
   the same time.
2. Obtain the electrode map from the data provider. It unlocks standard LPP sites, a
   spatially valid kappa null and spline interpolation.
3. Replace the legacy verifiers' tautological and string checks with behavioural tests
   as each area is next touched; do not rewrite them wholesale.
4. Move large binaries (animations, the dissertation zip, the Chapter 7 pickle) out of
   git history if the history is being rewritten anyway for S2.
5. Tune readout regularization for the raw-ERP and amplitude baselines (open in
   `V2_UPGRADE.md`).
6. Systematic literature review for the z-scoring artefact before writing it up.

## 9. Follow-up after the owner's decisions (2026-11-21)

The owner reviewed section 8 and decided as follows. Sections 1 to 8 above are left as
written on the day of the audit; this section records what changed afterwards.

| Decision in section 8 | Owner's decision | What was done |
|---|---|---|
| 1. Published clinical table and participant-keyed files (S1, S2) | No action | Nothing was removed and no history was rewritten. The clinical table is absent from this branch's tip. `tests/test_repo_hygiene.py` still prevents new data files from being tracked |
| 2. Code defects in archived experiments (I6) | Fix them | Nine defects fixed, each with a regression test: `docs/v1/DEFECT_FIXES.md` |
| 3. Manifest and v1 documents that contradict the code or v2 (I2, I8) | Accepted the recommendation | Every `results_manifest.json` entry carries a `standing_2026_11` field; its generating commit was corrected; eight stale v1 documents carry a status note; two READMEs that claimed fold-local PCA were corrected |
| 4. Dissertation erratum | Prepare it | `DISSERTATION_ERRATUM.md` |
| A letter listed as under review | It was not accepted; remove it | References removed |
| 6. Housekeeping | Merge to `main` | This branch was merged to `main` at the end of the pass |

Proposed items from section 8 that were also carried out:

- **Proposed 1, regenerate the v2 records after formatting and typing the library.**
  `arspinet_v2/` and every v2 runner are now typed, formatted and linted, runner
  parameters are in `experiments/v2/config.toml`, and every runner embeds a provenance
  block. All seven v2-era records were regenerated from the raw files and compared with
  the committed values using `scripts/compare_records.py`. All seven are equal in every
  value, given the BLAS thread count each was first produced with. The regeneration
  also found that `experiments/v2/v2_baseline_input.json` differs by one observation in
  some rows when run with four threads where the original used two (at most 0.24
  points; no conclusion changes). Thread settings are now recorded per record and in
  every new provenance block. The outcome table is in `docs/REPRODUCIBILITY.md`,
  section 5. Finding E14 no longer applies, and E13 is resolved at the source. In the
  table of section 6, the rows marked Locked for the v2 and ERP CORE records are now
  Replayed.
- **Proposed 5, tune the readout for the amplitude baselines.**
  `experiments/v2/run_v2_baseline_tuned.py`; result in `V2_UPGRADE.md`.

Still open: the electrode map (proposed 2), replacing weak legacy checks beyond the
areas touched here (proposed 3), a systematic literature review for the z-scoring
artefact (proposed 6), recreating the `v1.0-defense` tag, and the two images of unknown
licence (S5).

Two things the fixes brought to light:

- The unfixed attention-pooling function in `chapter5Experiments/reproduce_chapter5.py`
  scores 55% on pure noise where chance is 33%. That quantifies the leak the audit
  found by reading.
- Chapter 7 experiments D and E reproduce the dissertation's values exactly when rerun
  on the committed tables and the diagnostic spreadsheet. Those two analyses move from
  "structural at best" to "replayed" in section 6.
