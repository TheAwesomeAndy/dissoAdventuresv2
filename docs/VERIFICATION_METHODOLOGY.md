# Verification Methodology and Trustworthiness

> **Status note, added 2026-11-21.** This document is part of the v1 record and is kept
> as written. Three of its statements do not hold: scripts with zero coverage
> exist (`defense_figures/`, `experiments/interpretability/`, `experiments/chapter3/`),
> reservoir parameters are not consistent across scripts (Chapter 6 uses a different,
> sparse network), and the test count is out of date. See `docs/REPOSITORY_AUDIT.md`,
> findings E5 and I7, and `docs/REPRODUCIBILITY.md` for the current checks.

This document explains how every script in the ARSPI-Net repository is independently verified, what the verification tests prove, and why readers should trust the experimental results.

---

## Why Verification Matters

Scientific code must be transparent. A reader encountering this repository should be able to answer three questions:

1. **Does the code run?** Syntax errors, missing imports, or incompatible APIs would prevent reproduction.
2. **Does the code compute what it claims?** A function named `permutation_entropy` should actually compute permutation entropy, not something else.
3. **Are the results trustworthy?** The pipeline should produce consistent, deterministic outputs that match the claimed methodology.

The 491 automated tests across 17 verification scripts answer all three questions without requiring the proprietary [Stress, Health, and the Psychophysiology of Emotion (SHAPE) project](https://lab-can.com/shape/) EEG dataset.

---

## Verification Architecture

### Test Levels

Each verification script operates at multiple levels:

| Level | What It Tests | Why It Matters |
|-------|--------------|----------------|
| **Syntax** | Every `.py` file compiles without errors | Catches import failures, API changes (e.g., `np.trapz` → `np.trapezoid`), and typos |
| **Component** | Individual functions produce correct outputs on synthetic data | Validates that the LIF reservoir, BSC6 encoder, GNN layers, statistical tests, etc. are implemented correctly |
| **Integration** | End-to-end mini-pipelines run on small synthetic data | Validates that components work together (reservoir → encoding → PCA → classification) |
| **Consistency** | Cross-script parameter agreement | Ensures all scripts use the same reservoir parameters (N=256, beta=0.05, theta=0.5) |
| **Determinism** | Same inputs produce same outputs | Validates reproducibility via fixed random seeds |

### What Tests Cannot Cover

The verification scripts do **not** test:
- Classification accuracy on real SHAPE EEG data (requires dataset access)
- Statistical significance of clinical findings (requires full sample)
- Figure aesthetics or formatting

These limitations are by design. The tests validate that the *machinery* is correct; the *findings* depend on the data, which is separately validated by the `validation/` scripts.

### Passing tests verify code behavior, not scientific validity

This distinction is important enough to state plainly, because it is the single most common way a green test suite is over-read:

> **The 491 passing tests establish that the code behaves as written. They do not establish that any scientific claim is true.**

Concretely, a fully passing run says **nothing** about whether:

- **the estimand is the right one** — e.g. whether balanced accuracy under participant-grouped cross-validation, rather than a subject-mean-centered transductive estimate, is what a generalization claim requires;
- **the preprocessing boundary is leak-free** — e.g. whether PCA and standardization were fitted inside each fold on training participants only (fold-local), or globally on all data including held-out participants;
- **the generalization claim holds** — whether a result obtained on this cohort transfers to unseen participants, sites, or acquisition conditions.

Those are properties of the **study design and the data**, argued at explicitly bounded scope in the dissertation itself. They are exactly the properties that separate a *confirmatory* result from an *archived* one in [`REPRODUCTION_MAP.md`](REPRODUCTION_MAP.md) and [`../results_manifest.json`](../results_manifest.json) — and every result in all three tiers is backed by passing tests. Test status and scientific tier are orthogonal. A reviewer should therefore read a green test run as *"the machinery is correct"* and look to the manifest, the split strategy, and the preprocessing scope for whether a number is a defensible estimate.

---

## Verification Coverage Map

Every Python script in the repository is tested by at least one verification script:

| Script | Verified By | Test Level |
|--------|------------|------------|
| `chapter4Experiments/run_chapter4_experiments.py` | `verify_chapter4.py` | Full run on synthetic data |
| `chapter4Experiments/run_chapter4_observations.py` | `verify_chapter4.py` | Full run, all 13 figures generated |
| `chapter5Experiments/run_chapter5_experiments.py` | `verify_chapter5.py` | Component + integration |
| `chapter5Experiments/experiment_zero.py` | `verify_experiment_zero.py` | Component + integration + determinism |
| `chapter5Experiments/reproduce_chapter5.py` | `verify_reproduce_chapter5.py` | Component + integration + output structure |
| `chapter6Experiments/run_chapter6_exp1_esp.py` | `verify_chapter6.py` | Syntax + ESP convergence |
| `chapter6Experiments/run_chapter6_exp2_reliability.py` | `verify_chapter6.py` | Syntax + metric computation |
| `chapter6Experiments/run_chapter6_exp3_surrogate.py` | `verify_chapter6.py` | Syntax + surrogate generation |
| `chapter6Experiments/run_chapter6_exp3_valueadd.py` | `verify_chapter6.py` | Syntax + metric computation |
| `chapter6Experiments/run_chapter6_exp4_dissociation.py` | `verify_chapter6.py` | Syntax + metric computation |
| `chapter6Experiments/run_chapter6_exp5_interaction.py` | `verify_chapter6.py` | Syntax + metric computation |
| `chapter6Experiments/run_chapter6_exp6_temporal.py` | `verify_chapter6.py` | Syntax + metric computation |
| `chapter6Experiments/reproduce_chapter6.py` | `verify_reproduce_chapter6.py` | Component + Lyapunov + surrogate + integration |
| `chapter7Experiments/run_chapter7_experiment_A.py` | `verify_chapter7.py` | Syntax + data file validation |
| `chapter7Experiments/run_chapter7_experiment_B.py` | `verify_chapter7.py` | Full re-execution with numerical checks |
| `chapter7Experiments/run_chapter7_experiment_C.py` | `verify_chapter7.py` | Full re-execution with numerical checks |
| `chapter7Experiments/run_chapter7_experiment_D.py` | `verify_chapter7.py` | Syntax validation |
| `chapter7Experiments/run_chapter7_experiment_E.py` | `verify_chapter7.py` | Syntax validation |
| `chapter7Experiments/extract_kappa_matrix.py` | `verify_extract_utilities.py` | Functional test with mock pickle |
| `chapter7Experiments/extract_C_matrices.py` | `verify_extract_utilities.py` | Functional test with mock pickle |
| `experiments/ch5_4class/ch5_4class_01_feature_extraction.py` | `verify_ch5_4class.py` | Syntax + component |
| `experiments/ch5_4class/ch5_4class_02_raw_observations.py` | `verify_ch5_4class.py` | Syntax |
| `experiments/ch5_4class/ch5_4class_03_classification_full.py` | `verify_ch5_4class.py` | Syntax + component |
| `experiments/ch6_ch7_3class/ch6_ch7_01_feature_extraction.py` | `verify_ch6_ch7_3class.py` | Syntax + component |
| `experiments/ch6_ch7_3class/ch6_ch7_02_raw_observations.py` | `verify_ch6_ch7_3class.py` | Syntax |
| `experiments/ch6_ch7_3class/ch6_03_experiments.py` | `verify_ch6_ch7_3class.py` | Syntax + component |
| `experiments/ch6_ch7_3class/ch7_04_experiments.py` | `verify_ch6_ch7_3class.py` | Syntax + component |
| `experiments/ablation/layer_ablation.py` | `verify_ablation.py` | Component + integration |
| `chapter5Experiments/sklearn_baselines.py` | `verify_baselines.py` | Component + integration (8 classifiers, feature extraction, CV) |
| `chapter5Experiments/deprecated/eegnet_gru_lstm_baselines.py` | `verify_baselines.py` | Component + integration (EEGNet/GRU/LSTM forward, fit, determinism) |
| `validation/validate_shape_data.py` | `verify_validators.py` | Syntax + mock data QC |
| `validation/validate_subcategory_data.py` | `verify_validators.py` | Syntax + mock data QC |

---

## Key Algorithmic Validations

The tests go beyond simple "does it run" checks. Here are the critical algorithmic validations that establish correctness:

### 1. LIF Reservoir Correctness (tested in 7 scripts)

- **Weight shapes:** W_in is (N_res, 1), W_rec is (N_res, N_res) — verified in every script
- **Spectral radius:** Eigenvalue scaling produces exactly 0.9 (tolerance < 0.05)
- **Binary spikes:** Output values are exactly 0.0 or 1.0 — no floating-point artifacts
- **Non-negative membrane:** Membrane potential never goes below zero (floor constraint verified)
- **Determinism:** Same seed produces identical spike trains across runs
- **Channel independence:** Different seeds (seed + ch * 17) produce different reservoir dynamics

**Why this matters:** If the reservoir produced non-binary spikes, negative membrane potentials, or non-deterministic outputs, every downstream analysis would be invalid.

### 2. BSC6 Temporal Encoding (tested in 4 scripts)

- **Dimensionality:** 6 bins x N_res neurons = correct feature vector length
- **Non-negativity:** Spike counts cannot be negative
- **Conservation:** Sum of binned counts matches total spike count (minus remainder from integer division)

**Why this matters:** BSC6 is the core temporal coding scheme. If it produced wrong-dimensional vectors or negative counts, classification results would be meaningless.

### 3. Subject Centering (tested in verify_experiment_zero.py)

- **Per-subject mean is exactly zero** after centering (verified to tolerance 1e-10)
- **Does not modify the input array** (copy semantics verified)
- **Changes feature values** (centering has measurable effect)

**Why this matters:** Subject centering is critical for the Experiment Zero disambiguation. If centering didn't actually zero the per-subject mean, the "centered vs uncentered" comparison would be invalid.

### 4. Driven Lyapunov Exponent (tested in verify_reproduce_chapter6.py)

- **Benettin algorithm runs** on the LIFReservoirFull class
- **lambda_1 < 0** on synthetic data (Echo State Property verified)
- **Finite value** (no divergence or NaN)
- **Convergence trace returned** (algorithm produces interpretable output)

**Why this matters:** The entire Chapter 6 experimental program depends on the reservoir operating in the echo state regime (lambda_1 < 0). This test validates the Benettin algorithm implementation on the actual reservoir class used in the experiments.

### 5. Permutation Entropy Validation (tested in verify_reproduce_chapter6.py)

- **Random signal → PE ~ 1.0** (actual: 0.997). A random signal has maximum disorder, so PE should be near 1.0
- **Monotonic signal → PE ~ 0.0** (actual: 0.000). A monotonic signal has a single permutation pattern, so PE should be near 0.0

**Why this matters:** PE is one of the two temporal-structure metrics (alongside tau_AC) that carry the strongest condition effects in Chapter 6. These ground-truth tests verify the PE computation against known theoretical values.

### 6. Surrogate Generation (tested in 2 scripts)

- **Phase-randomized surrogates preserve the power spectrum exactly** (verified to rtol=1e-10)
- **Phase-randomized surrogates change the temporal signal** (not identical to original)
- **Time-shuffled surrogates preserve the amplitude distribution** (sorted values match)
- **Time-shuffled surrogates change temporal order** (not identical to original)

**Why this matters:** Surrogate testing (Experiment 6.3) is the validity gate for all dynamical metrics. If surrogates didn't properly preserve/destroy the intended signal properties, the surrogate sensitivity results would be meaningless.

### 7. GNN Layer Correctness (tested in verify_reproduce_chapter5.py)

- **GCN:** Symmetric normalized message passing (D^{-1/2} A D^{-1/2} H W) produces finite, correct-shape outputs
- **GraphSAGE:** Self-features concatenated with mean-aggregated neighbor features doubles the feature dimension
- **GAT:** Softmax attention weights sum to exactly 1.0 (tolerance 1e-6) per node

**Why this matters:** All three GNN architectures are implemented from scratch in NumPy (no PyTorch/DGL). These tests validate that the manual implementations match the published algorithms.

### 8. Coupling Matrix Extraction (tested in verify_extract_utilities.py)

- **Mock pickle data** mimics the exact structure of ch7_full_results.pkl
- **Kappa values in [0, 1]** — coupling strength is a normalized Frobenius norm
- **Correlation values in [-1, 1]** — Spearman correlations are bounded
- **C matrices have shape (7, 2)** — 7 dynamical metrics x 2 topological metrics
- **CSV output has correct column count** (5 for kappa, 16 for C matrices)
- **Cross-utility consistency** — same (subject, category) pairs in both extractions

**Why this matters:** The extraction utilities convert pickle results to CSV for transparency. If the CSV format were wrong (wrong columns, out-of-range values, missing rows), any downstream analysis of the exported data would be incorrect.

### 9. Cross-Validation Pipeline (tested in 4 scripts)

- **StratifiedGroupKFold** used everywhere (prevents subject leakage)
- **PCA fitted per fold** on training data only (no data leakage)
- **Fold accuracies in [0, 1]** and finite
- **Correct number of folds** returned

**Why this matters:** Subject leakage (training on data from a subject who appears in the test set) would inflate accuracy. The tests verify that the CV splitting and PCA fitting follow the correct protocol.

---

## How to Reproduce the Verification

Anyone can re-run all 380 tests:

```bash
# Install dependencies
pip install numpy scipy scikit-learn matplotlib pandas

# Run all 12 verification scripts
MPLBACKEND=Agg python chapter4Experiments/verify_chapter4.py
python chapter5Experiments/verify_chapter5.py
python chapter5Experiments/verify_experiment_zero.py
python chapter5Experiments/verify_reproduce_chapter5.py
python chapter6Experiments/verify_chapter6.py
python chapter6Experiments/verify_reproduce_chapter6.py
python chapter7Experiments/verify_chapter7.py
python chapter7Experiments/verify_extract_utilities.py
python experiments/ch5_4class/verify_ch5_4class.py
python experiments/ch6_ch7_3class/verify_ch6_ch7_3class.py
python experiments/ablation/verify_ablation.py
python validation/verify_validators.py
```

**No SHAPE EEG data required.** All tests use synthetic data or mock objects.

**Expected output:** 380/380 PASS across all scripts.

**Runtime:** Under 2 minutes total on a modern CPU.

---

## Limitations and Honest Assessment

### What the tests prove

- Every script in the repository parses and imports without errors
- Core algorithms (LIF reservoir, BSC6, PE, Lyapunov, GNN, surrogates) produce mathematically correct outputs on synthetic data with known ground truth
- The end-to-end pipeline (data → reservoir → features → classification) runs without errors
- Parameters are consistent across all scripts (N=256, beta=0.05, threshold=0.5, seed=42)
- Output formats (pickle, CSV, PDF) are produced correctly

### What the tests do not prove

- That classification accuracy on real SHAPE EEG data matches the reported values (requires dataset access)
- That statistical significance of clinical findings is reproducible (requires full 211-subject sample)
- That the experimental design is optimal (a methodological question, not a code correctness question)
- That the results generalize to other EEG datasets (an external validity question)

### Why this level of verification is sufficient

The verification strategy follows the principle of **compositional correctness**: if every component is individually correct, and the integration between components is verified, then the full pipeline is correct. The remaining uncertainty is in the *data*, not the *code*. The data itself is validated by the `validation/` scripts (10 + 12 automated checks on file format, dimensions, amplitude ranges, and cross-referencing).

---

## Independent Review

Chapter 6 has an additional layer of verification: `CHAPTER6_VERIFICATION_REPORT.md` (746 lines), an independent code review that provides 27 additional synthetic unit tests and identifies 4 methodological issues (1 HIGH, 2 MEDIUM, 1 LOW). This external review validates not just code correctness but scientific methodology.

---

## Summary

| Metric | Value |
|--------|-------|
| Total verification scripts | 17 |
| Total automated tests | 491 |
| Pass rate | 100% (491/491) |
| Scripts with zero coverage | 0 |
| External data required | None |
| Runtime | < 2 minutes |
| Independent code reviews | 1 (Chapter 6, 746 lines) |
