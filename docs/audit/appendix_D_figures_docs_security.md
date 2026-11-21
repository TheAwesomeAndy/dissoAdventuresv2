<!-- Audit appendix D. Repository state: commit 913d457, 2026-11-21.
     Written by a read-only reviewing agent that read the code and ran nothing in the repository.
     Reproduced unedited. Statements are that reviewer's reading of the code, cited by file and line.
     They were not each re-verified; docs/REPOSITORY_AUDIT.md says which ones were. -->

## Audit report: /home/user/dissoAdventuresv2 (read-only, HEAD 913d457, branch ccr-0221a213-rhj47r)

Nothing in the repo was run or edited. During the audit another process added untracked files (`pyproject.toml`, `requirements-lock.txt`, `scripts/check_records.py`, `scripts/run_verifiers.py`, `tests/`, `experiments/records.lock.json`, `arspinet.egg-info/`) and modified `.gitignore`. Findings describe tracked files at HEAD unless stated.

### Top findings
1. **Participant-level SHAPE-derived data is tracked, keyed by the real participant IDs.** Six CSVs carry a `subject` column whose 211 values equal the `ID` set of `data/clinical_profile.csv` (non-contiguous, not re-indexed).
2. **`data/clinical_profile.csv` is removed only on this branch.** It was added in e8ccefb (2026-03-20) and deleted in b700327 (2026-11-11). It is still present at the tip of 32 of 34 refs, including `origin/main` (b0fb317), `origin/archive/v1.0-defense-tag` and tag `v1.0-defense`. It has 211 rows x 54 columns: ID, diagnoses, age, sex, race, ethnicity, medication, substance use. Joined with item 1, history gives ID-to-diagnosis linkage for every tracked feature row.
3. **CI requires the participant-level files.** `chapter7Experiments/verify_chapter7.py:68-80,93,117,282` asserts their presence and minimum sizes. `LICENSES.md` ("Not included and not redistributed"; "no participant-level data") and `verify.yml:4` do not acknowledge this.
4. **A README restructure breaks one check:** `validation/validate_dissertation_claims.py:364` needs the literal substrings `0.82` and `0.23` in `README.md`. `V2_UPGRADE.md` says that 0.82 statistic "should be withdrawn".
5. **No credentials found** in tracked files or in pickaxe searches of history.

---

## Scope A: code

### defense_figures (18 scripts, 15 subfolders)
- **Tracking:** every `outputs/` folder is tracked (PDFs, 5 CSVs, 1 GIF, 1 MP4, `outputs/deck_manifest.json`).
- **Caches:** the `_*_cache.npz` files do not exist and are gitignored (`.gitignore:15-16`), but `defense_figures/README.md:121-124` says they "are committed".
- **Dissertation use:** no output filename appears in `dissertation/*.tex`. The Beamer deck `ARSPI-Net_Defense.tex` includes 20 of the 26 byte-identical copies in `pictures/defense_audit/`: A (1e, 1b, 1c, 1d), B (2e, 2b, 2d), C (3d, 3a-c), D (4d, 4a-c), TB, G, AM (two), AC. Copied but not included: F, IP, J, K, MR, OQ.
- **Input pickle:** A, B, C, D, TB and `make_animation.py` default to `chapter6Experiments/results/ch6_exp1_full.pkl` (absent, gitignored) and need keys `X_ds`, `y`, `subjects` (D also `conv_feats`). The committed writer of that filename (`chapter6Experiments/run_chapter6_exp1_esp.py:221`) dumps `lyapunov_results, demo_sid, demo_ch, subjects, categories` only. Which script produced the 633-trial pickle actually used: not determined.

Reference reservoir (`experiment_zero.py:66-98`): N=256, beta=0.05, theta=0.5, dense uniform W_rec scaled to 0.9, one scalar input, subtractive reset plus clamp at 0, seed 42.

| Folder | Computes | Data | Reservoir vs reference | Seeds |
|---|---|---|---|---|
| `experiment_a_autonomous_vs_driven/` | rho(W) and Benettin driven lambda1 on 633 trials x 5 channels; `make_animation.py` shows 5 initial conditions converging | SHAPE pickle | **Differs** (`make_experiment_a_figures.py:101-121`): W_in Gaussian x0.3; W_rec Gaussian sigma 0.05, 10% density, zero diagonal, no rescaling (rho = 0.2648); no subtractive reset or clamp. Same N, beta, theta, scalar input | reservoir 42, perturbation 42, sampling 42, animation ICs 7 |
| `experiment_b_takens_dimension/` | Kennel FNN embedding dimension m* of ERPs (tau 5, max m 20) | SHAPE pickle | None defined; constants 256 and 64 are reference lines only | 42, 43, 44 (plot sampling) |
| `experiment_c_memory_capacity/` | Jaeger memory capacity on smoothed spike counts and fading-memory tau, over 20 log-spaced beta in [0.01, 0.5] | Synthetic noise. Loads the SHAPE pickle (L590) but the ERP samples are never used (L300-312) | **Same as reference** (L122-158), beta swept | reservoir 42, drive 43, pulse 42, plot 43 |
| `experiment_d_channel_permutation/` | One-vs-rest logistic regression AUC against a per-trial channel-shuffle null, 500 permutations, 5-fold StratifiedGroupKFold | SHAPE pickle (`conv_feats`), stimulus labels | None | 42 |
| `figure_TB_theoretical_bounds/` | Bound-versus-measured panels; Grassberger-Procaccia dimension on 80 trials | The three absent caches plus SHAPE pickle | None; mixes A's reservoir (panel A) with C's (panel C) | 0 |
| `figure_{K,F,J,G,AM,AC,MR,IP,OQ,QA}/` | Static text and diagram slides | None | None | None |
| `build_deck.py` | Concatenates PDFs into a 15-page main and 16-page appendix deck | Needs PyMuPDF (`:37`), undeclared | n/a | n/a |

Stated claims (`defense_figures/README.md:41-44`):
- A: "Autonomous spectral-radius criteria are insufficient unless paired with a measurement under input drive."
- B: "Takens-motivated question; FNN-measured answer."
- C: "β = 0.05 sits inside the measured memory-capacity plateau ... but is **not** at the measured peak."
- D: "**Stimulus-class** ... classification survives per-trial channel-permutation null ... **Not** a clinical-disorder validation."
- Companion notes for A-D are skeletons; each has 2-3 `AUTHOR WRITES HERE` markers.

Code and claim problems:
- **Exp C bug:** `make_experiment_c_figures.py:266` has `S, _ = res.run_full(X)`, but `run_full` returns `(M, S)` (L158). The fading-memory tau is therefore computed on membranes, not spikes, despite "Identical to experiment_3_fading_memory" (L255). In the tracked CSV, tau is `inf` for 6 of 20 beta values and in the thousands of steps for several others.
- **Exp C docstring** (L7-10, L69-74) says beta "sits at the measured memory-capacity peak", describes the chapter-6 reservoir, and promises a lambda1 axis. Code and README contradict all three.
- **Exp C numbers:** beta = 0.05 is not in the sweep; the "0.763" value is at beta = 0.0519. The curve is single-seed and non-monotone (0.835, 0.735, 0.740 at adjacent betas).
- **TB:** in `experiment_tb_data.csv` the measured lambda1 (-0.062) lies above the "autonomous upper bound" (-1.329), and the measured m* (4) lies below the "lower bound" (6.16). `V2_UPGRADE.md` lists these labels as open finding M4.
- **Exp A:** all 3,165 lambda1 values lie in [-0.0636, -0.0614], median -0.0619; the pre-registered expectation was -0.054.
- **Exp D:** `p_perm = (null >= obs).mean()` (L441) has no +1 correction, so the CSV reports 0.000000. Null mean AUC is about 0.62 for classes 0 and 1.
- **Exp D clinical mode:** `--label-source csv` expects columns `subject_id,disorder_label` (L424-427); `data/clinical_profile.csv` has `ID`, `MDD`, and so on, so the command in its `companion_notes.md` would fail.
- **Exp D docstring** (L36) mentions an outputs README that does not exist.
- `make_animation.py:13` names `_poster.pdf`; the code writes `.pdf` (L277).
- `make_experiment_a_figures.py:8` and `HANDOFF_TO_NEXT_AGENT.md:344` cite `/root/.claude/plans/experiment-a-autonomous-foamy-bear.md`, which is outside the repo.

### validation/ and scripts/
- **`validate_dissertation_claims.py`** (CI gate, no data). Sections: existence of 36 hard-coded scripts (L67-108, all present); `ast.parse` of every `**/*.py` including untracked (L119-129); synthetic LIF, BSC6, permutation-entropy and kappa checks (L166 uses unseeded `np.random`); file-content checks.
- **`validate_shape_data.py`**: 10 QC checks on three SHAPE zips (`--batch1/2/3` required), optional `ParticipantInfo.csv` and psychopathology `.xlsx`. Writes `qc_report.txt`, `qc_file_metrics.csv`, `qc_flagged_subjects.csv` and PDFs to `qc_output/`, which is not gitignored.
- **`validate_subcategory_data.py`**: 12 checks on four-category files; exits 1 on critical failures.
- **`verify_validators.py`**: 20 checks. It never executes `validate_shape_data.py` (L71), and its "mock data" checks (L99-156) test arrays built in the verifier, not validator functions.
- **`validation/README.md`**: the usage line omits required arguments, and its check table does not match the script (script checks 9 and 10 are duplicate detection and clinical cross-reference).
- **`scripts/make_release_manifest.py`**: SHA-256 of `README.md`, `CITATION.cff`, `LICENSE`, `LICENSES.md`, the two docs, `results_manifest.json` (L74-78), `dissertation/*.tex|bib`, `ARSPI-Net_Defense.pdf`, and one digest over `pictures/`. Missing files are skipped silently; no `RELEASE_MANIFEST.json` is committed.

**README-dependent checks (complete list across all tracked `.py` and `.yml`):**
1. `validate_dissertation_claims.py:360-366`: `README.md` must contain `0.82` and `0.23` anywhere. They occur only at README L118 and L120. Skipped silently if the file is absent.
2. `validate_dissertation_claims.py:347-355`: fails if `0.661` appears in `README.md` or `docs/REPRODUCTION_MAP.md` with "LPP prediction" in the preceding 50 characters. Neither file contains it.
3. `make_release_manifest.py:75`: hashes README; no assertion.

Nothing machine-checks the "491" count or the suite table. Other dissertation-text checks: no `0.661` in `main.tex`, `ch2.tex`, `chDynamics.tex`, `chConclusion.tex` (L327-336); no `%%%ANDREW` in any `.tex` (L339-344); literal seed, GroupKFold and `N_RES/BETA/THRESHOLD` strings in 11 scripts (L382-443). All currently hold (evaluated statically).

---

## Scope B: documentation and metadata

**Authority**
- `README.md`: banner says v2; the body is "the v1 README, kept for provenance. Where it and `V2_UPGRADE.md` disagree, v2 governs."
- `V2_UPGRADE.md`: v2 findings and open items. Tables are in `experiments/v2/V2_RESULTS.md` (generated; "Do not edit by hand").
- `COMPUTATIONAL_ERRATUM.md`: "authoritative for code-derived numbers"; `dissertation/` governs the manuscript text only.
- `results_manifest.json`: v1.1.0. Top-level keys: `$schema_note, manifest_version, generated_for, authoritative_source, dataset, test_caveat, results, confirmatory_pipeline, reproduction_status, confirmatory_reconciliation`. `results` is a flat list of 33 entries with a `tier` field: confirmatory 11, exploratory 18, archived 4. No v2 entries.
- `docs/REPRODUCTION_MAP.md`: script-to-result map by tier. `docs/VERIFICATION_METHODOLOGY.md`: what tests prove. `docs/methodology_rules.md`: 12 advisor-derived rules.
- `RELEASE.md`: freeze and DOI procedure. `dissertation/AUDIT_REPORT.md`: language-only audit. `LICENSE`, `LICENSES.md`, `CITATION.cff`: terms and citation.

**Contradictions**
1. README L118/120 and the manifest report r = 0.82; `V2_UPGRADE.md` says it "never touches the reservoir" and should be withdrawn.
2. Manifest `novel_n1_srp_vs_pca` and `experiments/novel/NOVEL_RESULTS.md` report SRP-64 over PCA-64 by +4.4 pp; `V2_UPGRADE.md` conclusion 4 says "The SRP advantage disappears."
3. `results_manifest.json` cites "generating commit f6a0c66" (12 places); `confirmatory_results.json` provenance records `05acc31`.
4. Check counts: 491 (README L202-223, RELEASE, REPRODUCTION_MAP, manifest); "380 tests ... 12 verification scripts" (`VERIFICATION_METHODOLOGY.md:187-210`); "102/102" (`chapter5Experiments/README.md:320`); "76/76" (`experiments/README.md`). CI also runs `verify_v2.py` (32 checks per V2_UPGRADE), which no count includes.
5. `VERIFICATION_METHODOLOGY.md` says "Scripts with zero coverage | 0" and parameters "consistent across all scripts" with spectral radius 0.9. `defense_figures/`, `experiments/interpretability/`, `experiments/zscore_artifact/` and `canonical_pytorch_baselines.py` have no verifier, and the chapter-6 reservoir is unrescaled.
6. `V2_UPGRADE.md` "Limits": "Nothing here is validated on a second dataset". The same file reports ERP CORE replication, and `experiments/external/` holds a TCRZEM replication.
7. `RELEASE.md` and `make_release_manifest.py` call `ARSPI-Net_Defense.pdf` the dissertation PDF; it is the 104-page Beamer deck. No dissertation PDF is tracked.
8. `RELEASE.md` tags `v1.0.0`; the existing tag is `v1.0-defense`; README says v2; `CITATION.cff` says `0.0.0-unreleased`, date `2026-01-01`, DOI `10.5281/zenodo.XXXXXXX`.
9. Deck size: 104 pages (`qa_index.md`, `timing_map.md`, actual) versus 105 (`visual_audit_checklist.md`). `HANDOFF_TO_NEXT_AGENT.md:97` says the pickle holds 4,220 lambda1 values; Exp A used 3,165.
10. `methodology_rules.md` Rule 11 calls the SUD result "notable and real"; V2 says clinical analyses were "an uncorrected screen" and nothing survives correction.
11. Raw ERP-16, four categories: 60.6% [57.9, 63.1] in README, 60.5% [57.9, 63.0] in V2.

**Stale references (verified with ls)**
- README tree (L143-169) omits `arspinet_v2/`, `experiments/v2/`, `experiments/zscore_artifact/`, `defense_figures/`, `data/`, `V2_UPGRADE.md`, `COMPUTATIONAL_ERRATUM.md`.
- `experiments/README.md` lists 3 of 10 subdirectories.
- `RELEASE.md` section 2 omits `validate_provenance.py` and `verify_v2.py`.
- `defense_figures/visual_audit_checklist.md` L115-173 names four nonexistent files, for example `figureG_failure_gallery.pdf`.
- `chapter7Experiments/README.md:328` cites `extract_features_for_expE.py` (absent); L460 and three scripts hard-code `/mnt/user-data/outputs/...`.
- `chapter5Experiments/README.md` cites `figures/ch5/*.pdf` (absent).
- `experiments/novel/README.md:116` cites `novel_results.json`; the files are `_3class` and `_4class`.
- Data directory names disagree: `categories/` (README) versus `categoriesbatch{1-4}/` (`validation/README.md`).

**Dependencies**

| Source | Declares |
|---|---|
| `verify.yml:37-38` (Python 3.11) | numpy>=2.0, scipy>=1.13, scikit-learn>=1.5, pandas>=2.2.3, matplotlib>=3.9, joblib>=1.4 |
| `README.md:192-197` | Same without joblib, plus torch>=1.12 |
| `docs/VERIFICATION_METHODOLOGY.md:191` | Unpinned, no joblib |
| `experiments/chapter3/animations_README.md:104` | Adds pillow, imageio-ffmpeg |

- No requirements file is tracked at HEAD.
- Undeclared: PyMuPDF, openpyxl (six `pd.read_excel` calls), ffmpeg. torch is never installed in CI.
- Recorded environments: confirmatory record Python 3.11.15, numpy 2.4.6, scipy 1.17.1, sklearn 1.9.0; v2 records Python 3.13.18, numpy 2.5.3, scipy 1.18.1, sklearn 1.9.1.
- The untracked `pyproject.toml` and `requirements-lock.txt` match the CI bounds and v2 record; `pyproject.toml` cites `AGENTS.md`, which does not exist.

**Licensing**
- Code: MIT (`LICENSE`, (c) 2026 Andrew A. Lane). `LICENSES.md` also puts JSON manifests under MIT.
- Dissertation text: "All rights reserved" and "CC BY 4.0" in the same cell.
- Original figures: CC BY 4.0; no CC licence text is included.
- Third-party: `pictures/ch1/OIP.jpg` (474x316) and `pictures/ch2/LSMArchitecture.png` are referenced by no tracked `.tex`, `.py` or `.md` other than `LICENSES.md`. Sources: not determined.
- Data: declared restricted and not included, contradicted by top finding 1.
- `ArspiNetFinalVer_.zip` (17 MB, 294 entries): 10 root `.tex` plus `Disso.bib`, and `pictures/` with 283 entries (appendixA 59, ch1 2, ch2 1, chDynamics 56, chGraphNeuralNetworks 71, chLSM 28, chLSMEmbeddings 37, chSynthesis 28, conclusion 1). The only entry not in the repo is nested `pictures/chLSMEmbeddings/files.zip` (6 PDFs and 1 `.tex`, names already in the repo). No data files. Whether the zip's `.tex` files differ from `dissertation/`: not determined.

---

## Scope C: security and sensitive data

**Credentials**
- None in tracked files: no key or token patterns, no `.env`, no URLs with credentials, no email addresses.
- History pickaxe (`api_key`, `ghp_`, `AKIA`, `BEGIN RSA/OPENSSH`, `sk-ant`, `Bearer`, `SECRET`): 0 commits. `password`: prose only. `token=`: an Elsevier reader URL session token in old `Disso.bib` revisions (36a097c, 28671f7).
- Commit author metadata exposes three personal or institutional email addresses (two gmail, one stonybrook.edu).

**Participant-level tracked files** (no values printed)

| File | Shape | Key | Content |
|---|---|---|---|
| `chapter7Experiments/chapter7_results/subject_features.csv` | 211 x 307 | SHAPE ID | Per-electrode features |
| `.../observation_features.csv` | 844 x 308 | ID + category | Same, per observation |
| `.../C_matrices.csv` | 844 x 16 | ID + category | Coupling correlations |
| `.../kappa_matrix.csv` | 211 x 5 | ID | Kappa per category |
| `.../ch7_full_results.pkl` (22.6 MB) | dict | `subjects` array | Per-subject profiles, PLV matrices |
| `.../ch7_expB_results.npz` | 23 arrays | `subjects` (211,) | K (211, 4), per-subject means |
| `.../ch7_expA_analysis.pkl` | dict | Per-observation arrays | Row order not determined |
| `defense_figures/experiment_a_autonomous_vs_driven/outputs/experiment_a_data.csv` | 3165 x 5 | ID, trial, class | lambda1 |
| `defense_figures/experiment_b_takens_dimension/outputs/experiment_b_data.csv` | 3165 x 25 | ID, trial, class | FNN |

- None of these files has clinical or demographic columns.
- Aggregate only: `chapter5Experiments/results/baseline_deep_results.pkl`, the other three defense CSVs, and all 15 JSONs.
- No text file pairs an individual ID with a diagnosis. Subjects 127 and 195 are named as exclusions.
- `validation/validate_shape_data.py:9-17` quotes a private email from the data provider by first name.
- Deleted history archive `arspi_net_chapter5_complete.tar.gz` (c3f15cf to 54e0f27) contains `data/ch5_raw_observations/obs0*_data.npz`; contents not determined.
- Whether any of the 343 PDFs show individual participants: not determined.

**Tracked files over 1 MB** (pack size 211 MiB)

| File | Size | Class |
|---|---|---|
| `chapter7Experiments/chapter7_results/ch7_full_results.pkl` | 22.6 MB | Regenerable only with SHAPE (`run_chapter7_experiment_A.py`) |
| `ArspiNetFinalVer_.zip` | 17.0 MB | Redundant archive |
| `pictures/animations/*.gif` (6) | 2.1-15.8 MB | Generated: `experiments/chapter3/animate_lsm_dynamics.py`, `chapter4Experiments/animate_ch4_dynamics.py`; several need SHAPE |
| `pictures/animations/*.mp4` (3) | 1.4-1.9 MB | Same |
| `ARSPI-Net_Defense.pdf` | 4.8 MB | Built from `ARSPI-Net_Defense.tex` |
| `pictures/chDynamics/membrane_evolution_top50.pdf`, `raw6_1d_esp_convergence.pdf` | 3.2, 1.4 MB | `run_chapter6_exp1_esp.py` (SHAPE) |
| `defense_figures/outputs/ARSPI_Net_Defense_Appendix.pdf` | 3.2 MB | `build_deck.py` |
| `rawA_1f_contraction_animation.gif` | 3.2 MB | `make_animation.py` (SHAPE) |
| `rawC_3b_state_response_per_beta.pdf` (two copies) | 2.6 MB each | `make_experiment_c_figures.py` |
| `chapter7_results/observation_features.csv` | 2.4 MB | Generator script not in repo |

**.gitignore versus tracked** (`git ls-files -ci --exclude-standard`): only `ArspiNetFinalVer_.zip` (rule `*.zip`, L23).

---

## Agent files and conventions
- No `AGENTS.md`, `CLAUDE.md`, `.claude/`, `.cursor*`, copilot instructions or `CONTRIBUTING` exists in the tree or history.
- De facto agent instructions: `defense_figures/HANDOFF_TO_NEXT_AGENT.md` (400 lines, addressed to "the next AI agent").
- Stated conventions:
  - `docs/methodology_rules.md` Rules 1-12, including linear readouts for content comparison, clinical claims exploratory, and state conditional results together.
  - `defense_figures/README.md`: all figures import `_style.py`; seed 42; each analysis figure ships a `*_data.csv`.
  - Companion notes: "Do not delegate the writing of this file."
  - `COMPUTATIONAL_ERRATUM.md`: dissertation source is not edited to match reconstruction values.
  - `V2_UPGRADE.md`: "v1 files are unchanged."
  - `experiments/v2/make_report.py:43`: `V2_RESULTS.md` is generated.

## Not determined
- Whether CI currently passes (nothing was executed).
- Content differences between the zip's `.tex` files and `dissertation/`.
- Origin of the two third-party images.
- Which script produced the pickle used for defense experiments A, B and D.
