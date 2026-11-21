<!-- Audit appendix A. Repository state: commit 913d457, 2026-11-21.
     Written by a read-only reviewing agent that read the code and ran nothing in the repository.
     Reproduced unedited. Statements are that reviewer's reading of the code, cited by file and line.
     They were not each re-verified; docs/REPOSITORY_AUDIT.md says which ones were. -->

All paths are under `/home/user/dissoAdventuresv2/`. I read all 21 scripts and 3 READMEs and ran nothing. "Tracked" was checked with `git ls-files`. SGKF = StratifiedGroupKFold.

## Scope-wide facts
- Every LIF copy hardcodes N=256, β=0.05, θ=0.5, dense Xavier-uniform weights, spectral radius 0.9, 6 bins. The one exception is the sparse reservoir in the Ch4 animation.
- No script in scope uses SRP; all compression is PCA.
- No committed script anywhere writes `shape_features_211.pkl`. Five scripts need it.
- PCA is built without `random_state` everywhere except `run_chapter4_observations.py`. Whether sklearn picks the randomized solver was not determined.

## Per script

**chapter4Experiments/**
- **`run_chapter4_experiments.py`** — runner, synthetic, no CLI.
  - Task: 200 trials/class, 33 inputs, T=150, window 10–70; `main()` sets noise 0.5, jitter ±5 (535-540).
  - Six experiments: size, FDR, coding × LogReg(C=0.1)/SVC-RBF, PCA-k, 10 seeds, β×θ grid.
  - Validation: `StratifiedKFold(5, seed 42)`, fold-local scaler, but `PCA(...).fit_transform(features)` on all trials before CV (225-227, 373, 429-431, 494-496).
  - Metrics: accuracy mean±SD, FDR.
  - Writes 7 tracked PDFs in `pictures/chLSMEmbeddings/`.
  - Docstring says "All random states are fixed"; PCA is unseeded.
- **`run_chapter4_observations.py`** — figure generator, synthetic; `--output_dir` defaults to the same folder (6 tracked PDFs).
  - Its stimulus generator differs from the runner's (pulses at 15/30/45, width 12, one jitter per trial; 148-190). The README lists only the runner's parameters.
  - Legend text "50.5% (64 PCs)" and "5 PCs (99.5% acc)" is hardcoded (425-429).
- **`animate_ch4_dynamics.py`** — animation generator; needs the pickle (`X_ds`, `y`, `subjects`).
  - Defaults: `--pkl /tmp/shape_features_211.pkl`, `--outdir pictures/animations` (9 tracked files).
  - The Lyapunov cache `chapter4Experiments/cache/…npz` is reused silently.
  - BSC6 spans all 256 samples; one shared 1-input reservoir drives channel 8.
- **`verify_chapter4.py`** — verifier; subprocess-runs both scripts from the repo root, so every run rewrites the 13 tracked PDFs.

**experiments/chapter3/**
- **`run_chapter3_lsm_characterization.py`** — runner, synthetic; `--outdir ./ch3_results` (6 untracked PDFs, referenced nowhere).
  - Exps 4 and 6: `feats_s = scaler.fit_transform(feats)` on all data (343-344, 454-455), then `StratifiedKFold(5)`, LogReg, balanced accuracy.
  - The "same as Ch4" task uses noise 0.15, jitter ±3, not what Ch4's `main()` passes.
  - Exp 1 "fire" check is a heuristic (`mem[t-1] >= 0.7θ`, line 174).
- **`animate_lsm_dynamics.py`** — animation generator. Without `--pkl`, animation 1 falls back to synthetic data yet writes to `pictures/animations/`, overwriting the tracked real-data `lsm_graph_diffusion.*`.

**experiments/interpretability/** — all four need `--data_dir`; outputs go to `./interpretability_results` (untracked PDFs, no numeric file). Loaders want pickle key `raw_eeg` or `X_raw`, which nothing in the repo writes. None has a verifier or CI step.
- **`run_level1_temporal_traceability.py`**
  - Analysis 1: one shared reservoir for all channels; median signed Pearson r between LPP mean (samples 102–179) and BSC bins 2–4.
  - Analysis 2: one 34-input reservoir, RidgeCV, `GroupKFold(10)`.
  - Docstring: Ridge result "exploratory… NOT cited"; R²=0.661 "removed" (actual −0.747).
  - No centering in code; the manifest says "centered ERP".
- **`run_level3_descriptor_erp_alignment.py`**
  - Per-channel reservoirs (seed 42+17·ch).
  - Docstring says seven descriptors; code computes six (120-147), all spike-derived.
  - Docstring claims |r|=0.82 for spike mean vs LPP; level1's docstring says 0.82 is pre-reservoir. 0.82 appears only as a printed literal (238).
- **`run_eegnet_saliency_comparison.py`**
  - EEGNet trained on all data, no split, no torch seed.
  - The ARSPI-Net window is hardcoded: `return 176, 254` (178).
  - Bug: line 191 `if eeg.shape[1] > eeg.shape[2]: X = eeg  # already (N, C, T)` is inverted. Either layout gives 256 "channels" × 34 "samples" (~130 ms), so the docstring's 402–691 ms peak cannot come from this code.
- **`run_arspinet_v2_attention_prototype.py`**
  - BSC from samples 10–70 of the 256-sample epoch, averaged over channels; the figure labels bins 0–1000 ms (263).
  - The "standard linear readout" is a ReLU MLP.
  - SGKF(10, seed 42); no torch seed.
  - The permutation test shuffles 20 fold accuracies between methods (249), not the manifest's "condition-profile permutation".

**chapter5Experiments/**
- **`run_chapter5_experiments.py`** — runner.
  - Demo mode still needs a real SHAPE file (`--sample_file` default `/mnt/user-data/uploads/SHAPE_Community_007_IAPSNeg_BC.txt`).
  - Preprocessing: rows 205+, decimate ×4, z-score; reservoirs seeded 42+17·ch.
  - Validation: SGKF(10, seed 42), fold-local scaler. But PCA-64 is fitted on all observations ("for inspection", 746-751) and feeds every row. The fold-local `apply_pca_per_channel` (326) is dead code.
  - GAT weights are random and untrained. There is no GIN, EdgeConv or SVM row.
  - Metrics: pooled accuracy, balanced accuracy, macro-F1; no SD, no MCC.
  - Writes `figures/ch5/*.pdf` (untracked; the directory is created at import); no results file.
- **`reproduce_chapter5.py`** — docstring says it "reproduces EVERY figure, table, and statistical result".
  - Code runs only Experiments 1 and 3. The GNN figure plots hardcoded values (519-538) and `'var_before': 190.0, 'var_after': 99.0,  # from earlier analysis` (654).
  - Leakage: global PCA (594, 631); greedy channel selection by CV on all data; test-fold labels set attention weights: `scores[ch] = clf.score(sc.transform(node_feats[te, ch]), y[te])` (279).
  - `--labels` is unused. Writes `ch5_all_results.pkl`, not the `shape_features.pkl` the README lists.
- **`experiment_zero.py`** — runner and library.
  - `LIFReservoir`/`bsc6_encode` are imported by `experiments/confirmatory/confirmatory_pipeline.py:79`, `experiments/novel/novel_pipeline.py:212` and `experiments/v2/verify_v2.py:54`.
  - Its own loader keeps the baseline: it decimates all 1229 rows, then `X_ds = X_ds[:256, :]` (148-159), i.e. raw rows 0–1023.
  - Fitted on all data: `pca.fit_transform(bsc_all[:, ch, :])` (215) and `subject_center` (337-338).
  - Then SGKF(10, seed 42), LogReg C=0.1, balanced accuracy.
  - Writes `experiment_zero_results.pkl`. `--skip_reservoir` is unused.
- **`canonical_pytorch_baselines.py`**
  - Loader globs `*.mat` under "# Adapt to your SHAPE data format" (191-193); it cannot read the `.txt` layout.
  - Docstring says "50 runs"; code is 10 seeds × 10 folds × 5 inits, averaging accuracies (no majority vote).
  - Centering uses all data (221); validation is `X_train[:n_val]`, ungrouped (345).
  - No output file, MCC or variance decomposition.
- **`deprecated/eegnet_gru_lstm_baselines.py`** — NumPy; output layer only; SGKF(10). Docstring: 72.0/59.9/58.0 "are FROM THESE simplified implementations".
- **`sklearn_baselines.py`** — 8 classifiers on BandPower+Hjorth, SGKF(10), fold-local scaler, no global fit. Writes `sklearn_baseline_results.pkl`.
- **`graph_diffusion_oversmoothing.py`**
  - Needs the pickle. Computes Dirichlet energy and cosine for K=0–8, not inter-channel variance.
  - Overwrites tracked `pictures/chGraphNeuralNetworks/exp03b_diffusion_overlay.pdf`; writes `diffusion_proxy.npz`.
  - Its "verbatim" functional adjacency uses signed correlation; the runner's uses absolute.
- **`results/baseline_deep_results.pkl`** (tracked; disassembled, not unpickled) — 10 fold accuracies plus MCC per model, means 72.0/59.9/58.0. No committed script writes this schema, filename or `fig_deep_baselines.pdf`.

## Summary table
| Script | Kind | SHAPE? | Writes tracked? | Status in repo |
|---|---|---|---|---|
| `run_chapter4_experiments` | runner | no | yes (7) | exploratory |
| `run_chapter4_observations` | figures | no | yes (6) | exploratory |
| `animate_ch4_dynamics` | animation | yes | yes (9) | unassigned |
| `verify_chapter4` | verifier | no | yes, via subprocess | CI |
| `run_chapter3_lsm_characterization` | runner | no | no | exploratory |
| `animate_lsm_dynamics` | animation | optional | yes (9) | unassigned |
| four interpretability scripts | runners | yes | no | exploratory |
| `run_chapter5_experiments` | runner | yes | no | archived |
| `reproduce_chapter5` | runner | yes | no | unassigned |
| `experiment_zero` | runner + library | yes | no | archived; reservoir reused by confirmatory |
| `canonical_pytorch_baselines` | runner (template loader) | yes | no | archived |
| `sklearn_baselines` | runner | yes | no | archived |
| `graph_diffusion_oversmoothing` | runner + figure | yes | yes (1) | archived |
| `deprecated/eegnet_gru_lstm_baselines` | runner | yes | no | deprecated |
| four Ch5 `verify_*` | verifiers | no | no | CI |

## Reservoir definitions
- **Class A — same equations** (leak ×(1−β), multiplied by (1−previous spike), then subtract θ, floor at 0).
  - Multi-input signature: `run_chapter4_experiments.py:43`, `run_chapter4_observations.py:75`, `animate_ch4_dynamics.py:84`, `run_chapter3_lsm_characterization.py:63`, `animate_lsm_dynamics.py:91`, level1 `:86`, level3 `:78`, attention `:81`.
  - Single-input signature: `run_chapter5_experiments.py:226`, `reproduce_chapter5.py:74`, `experiment_zero.py:66`, `graph_diffusion_oversmoothing.py:49`, `animate_lsm_dynamics.py:149`.
  - Weights match for one input and the same seed. Summation order differs (`+ I_in + I_rec` vs `+ I`); bit-identical spikes not determined.
- **Class B — different**: `ReservoirCh6` plus `compute_driven_lyapunov` (`animate_ch4_dynamics.py:130, 141`): 10% sparse Gaussian, no spectral-radius scaling, no subtractive reset, no floor.
- **Usage differs within class A**: per-channel seeds; one shared reservoir (level1, attention, Ch4 animation); 34-input (level1 Analysis 2); 33-input (synthetic).
- **BSC windows differ**: 10–70 of 150 (synthetic); 10–70 of 256 (attention); full epoch dropping the last 4 samples (runner, `experiment_zero`, diffusion); full epoch with a 46-sample last bin (`reproduce_chapter5`, Ch4 animation).

## Docs to scripts
- **Tier 2 exploratory** (`REPRODUCTION_MAP.md`): Ch3 and Ch4 scripts and all four interpretability scripts. Ch3, Ch4 and saliency have no manifest entry.
- **`exp_interp_L1_temporal`**: maps `r_pre_reservoir` 0.82 to level1, which says it does not compute it.
- **`exp_interp_L2_geometric`** (ρ=7.2): maps to `canonical_pytorch_baselines.py (--centered)`, which computes no variance decomposition.
- **`exp_interp_L3_dynamical`**: maps "amplitude-domain" 0.82 to level3, whose descriptors are all spike-domain.
- **`arch_baseline_centered_table`** (archived): the uncentered deep rows match the tracked 10-fold pickle and the deprecated docstring. The mapped PyTorch script cannot load the data as committed and has no "majority voting". The PCA-200 row has "no canonical committed generator" (the repo's words).
- **`arch_pca64_svm`, `arch_gnn_sweep`** (archived): mapped to `run_chapter5_experiments.py`, which has no SVM row, GIN, EdgeConv, MCC or ±SD. GIN/EdgeConv exist only in `experiments/ch5_4class/`.
- **"33% inter-channel variance" at K=2**: computed by neither mapped script. `reproduce_chapter5.py` hardcodes 190→99; `animations_README.md` says Dirichlet "~87%".
- **Mapped to no result**: `reproduce_chapter5.py`, both animation scripts.
- **Missing from `chapter5Experiments/README.md`**: canonical, sklearn, diffusion, deprecated, `verify_baselines.py`.
- **`experiments/README.md`** omits `chapter3/` and `interpretability/`.
- **References to things that do not exist**:
  - `arspi_net_chapter5_complete (1).zip` and `figures/ch5/` (`chapter5Experiments/README.md`).
  - Fold-local PCA (same README).
  - `shape_features_211.pkl` "generated by run_chapter5_experiments.py" (`experiments/ablation/README.md:125`).

## Verifiers
Counts match the docs.

| Verifier | Checks | Numerical | Rest |
|---|---:|---:|---|
| `verify_chapter4` | 31 | 16 thresholds regex-parsed from stdout | 13 file-exists, 2 exit codes |
| `verify_chapter5` | 32 | 11 invariants | 18 shape/key, 3 literal `True` |
| `verify_experiment_zero` | 37 | 15 | 12 shape, 10 syntax/import/source-string |
| `verify_reproduce_chapter5` | 33 | 3 on the module | 13 source-string, 8 on code written inside the verifier, 9 structural |
| `verify_baselines` | 55 | 19 (4 compare the committed pickle to constants) | 14 shape, 22 instantiate/syntax/key |

- `verify_chapter5` gives each sample its own subject, so grouping is never exercised.
- `verify_experiment_zero` never tests the loader or the PCA.
- `verify_reproduce_chapter5` asserts "PCA fitted per fold (no leakage)" via `"fit_transform" in source`. Its GNN checks run code inside the verifier; the module has no GNN functions, despite `VERIFICATION_METHODOLOGY.md` §7.
- No verifier covers the PyTorch baselines, the diffusion script, Ch3, the animations or interpretability.
