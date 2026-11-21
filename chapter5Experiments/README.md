# Chapter 5: Clinical EEG Classification with ARSPI-Net

This folder contains the experimental scripts, results archive, and figures for Chapter 5 of the ARSPI-Net dissertation. Chapter 5 validates the full ARSPI-Net pipeline on real clinical EEG data — classifying affective responses (Negative/Neutral/Pleasant) from the [Stress, Health, and the Psychophysiology of Emotion (SHAPE) project](https://lab-can.com/shape/).

## Experiment Overview

| Experiment | Script | Purpose |
|------------|--------|---------|
| 5.1 | `run_chapter5_experiments.py` | Complete pipeline: 7-row baseline table + GNN architecture comparison + sparsity sweep + depth ablation |
| 5.2 | `reproduce_chapter5.py` | Standalone reproducibility pipeline — regenerates every figure, table, and statistical result in Chapter 5 |

---

## The ARSPI-Net Architecture

The full ARSPI-Net pipeline validated in Chapter 5 has five stages:

```
Raw EEG (1024 Hz) → Preprocess → LIF Reservoir → Feature Coding → Graph Construction → GNN Classification
```

### Stage 1: Preprocessing

- **Baseline removal:** Discard first 205 samples (200 ms at 1024 Hz)
- **Downsampling:** Decimate 4x (1024 Hz → 256 Hz), producing (N, 256, 34) arrays
- **Normalization:** Z-score per epoch per channel (zero mean, unit variance)

### Stage 2: LIF Reservoir

Each of 34 EEG channels is processed by an independent 256-neuron Leaky Integrate-and-Fire reservoir. Canonical parameters from Chapter 3 characterization:

| Parameter | Value |
|-----------|-------|
| Reservoir neurons (N_RES) | 256 |
| Membrane leak (beta) | 0.05 |
| Firing threshold | 0.5 |
| Spectral radius | 0.9 |
| Input weights | Xavier-uniform, (256, 1) |
| Recurrent weights | Xavier-uniform, (256, 256), scaled to spectral radius |

The membrane update rule:

```
mem[t] = (1 - beta) * mem[t-1] * (1 - spk[t-1]) + W_in * x[t] + W_rec @ spk[t-1]
spk[t] = (mem[t] >= threshold)
mem[t] = max(mem[t] - spk[t] * threshold, 0)
```

Each reservoir is seeded deterministically (`seed + ch * 17`) so that channel-specific weights are reproducible but independent across channels.

### Stage 3: Feature Coding

Four feature extraction methods are compared:

| Feature | Extraction | Per-Channel Dim | Total Dim (34 ch) |
|---------|-----------|----------------|-------------------|
| **BSC₆ + PCA-64** | 6 temporal bins x 256 neurons = 1536 → PCA to 64 | 64 | 2,176 |
| **MFR** | Mean firing rate across timesteps | 256 | 8,704 |
| **BandPower** | Welch PSD in 5 frequency bands (delta 1-4, theta 4-8, alpha 8-13, beta 13-30, gamma 30-100 Hz) | 5 | 170 |
| **Hjorth** | Activity (variance), Mobility (sqrt(var(dx)/var(x))), Complexity | 3 | 102 |

**BSC₆ (Binned Spike Counts):** The spike train from each reservoir is divided into 6 equal temporal bins. Spike counts per bin per neuron are concatenated to produce a 1536-dimensional vector. PCA reduces this to 64 dimensions — the primary ARSPI-Net embedding of these archived scripts. **Correction (2026-11):** this README used to say the PCA was fitted on training data only per fold. In the scripts in this directory it is fitted on all observations before the split (`run_chapter5_experiments.py`, `reproduce_chapter5.py`, `experiment_zero.py`), so these results are not leak-free. The leak-free protocol is `experiments/confirmatory/`; see `docs/v1/DEFECT_FIXES.md`.

**MFR (Mean Firing Rate):** Average spike count per neuron across the full window. This rate-based code discards temporal structure — it serves as a control to prove temporal coding matters.

### Stage 4: Graph Construction

Two adjacency strategies define how electrode-level features interact:

**Spatial adjacency (primary):** k-nearest neighbors (k=5) from standard 10-20 3D electrode positions. The 34-channel montage uses: Fp1, Fp2, F7, F3, Fz, F4, F8, FC5, FC1, FC2, FC6, T7, C3, Cz, C4, T8, CP5, CP1, CP2, CP6, P7, P3, Pz, P4, P8, PO7, PO3, POz, PO4, PO8, O1, Oz, O2, REF. Adjacency is symmetric and binary.

**Functional adjacency (secondary):** Absolute Pearson correlation of node feature vectors across electrodes, thresholded at the 75th percentile. This is data-driven and varies with the embedding.

### Stage 5: GNN Classification

All GNN implementations are **from-scratch NumPy** — no PyTorch, TensorFlow, or DGL dependencies:

| Architecture | Propagation Rule | Key Property |
|-------------|-----------------|-------------|
| **GCN** | `H' = D^{-1/2} A_tilde D^{-1/2} @ H` | Symmetric normalized message passing |
| **GraphSAGE** | `H' = [H; mean(H_neighbors)]` | Concatenates self with mean-aggregated neighbors |
| **GAT** | Multi-head attention (4 heads), LeakyReLU, softmax | Attention-weighted neighbor aggregation |

After GNN propagation (2 layers by default), a mean-pooling graph readout produces a single graph-level embedding per sample. This is fed to a downstream classifier (LogReg or MLP).

**Validation:** Subject-stratified 10-fold cross-validation (`StratifiedGroupKFold`) — all conditions from one subject stay in the same fold to prevent data leakage. PCA is fitted on training folds only.

---

## Experiment 1: The 7-Row Baseline Table

### Research Question

What is the individual and combined contribution of (a) LSM spiking embeddings vs conventional EEG features, and (b) graph-structured classification vs flat classifiers, to affective EEG classification?

### Method

Seven experimental conditions are arranged to isolate each component's contribution through systematic ablation:

| Row | Features | Classifier | Graph | Purpose |
|-----|----------|-----------|-------|---------|
| 1 | BandPower + Hjorth | LogReg | No | Conventional baseline |
| 2 | BandPower + Hjorth | MLP | No | Nonlinear conventional baseline |
| 3 | LSM-BSC₆-PCA64 | MLP | No | LSM benefit without graph structure |
| 4 | BandPower + Hjorth | GAT | Spatial | Graph benefit on conventional features |
| **5** | **LSM-BSC₆-PCA64** | **GAT** | **Spatial** | **Full ARSPI-Net (best expected)** |
| 6 | LSM-BSC₆-PCA64 | GAT | Functional | Functional adjacency variant |
| 7 | LSM-MFR | GAT | Spatial | Rate-based variant (expected worse) |

### Key Comparisons

Three pairwise differences isolate the contribution of each architectural component:

- **Graph structure value (Row 5 - Row 3):** Does spatial graph structure improve classification beyond flat LSM embeddings?
- **LSM embedding value (Row 5 - Row 4):** Do spiking reservoir features outperform conventional EEG features when both use the same graph architecture?
- **Temporal coding value (Row 5 - Row 7):** Does the temporally-coded BSC₆ representation outperform the rate-coded MFR representation? This validates that temporal structure — not just spiking activity level — carries affective information.

### Usage

```bash
# Full pipeline with real SHAPE data:
python run_chapter5_experiments.py --data_dir /path/to/shape_eeg_files/

# Demo mode with synthetic data (pipeline verification):
python run_chapter5_experiments.py --demo
```

### Output Figures

- `figures/ch5/baseline_comparison.pdf` — 7-row bar chart with balanced accuracy per condition
- `figures/ch5/confusion_matrix.pdf` — 3x3 confusion matrix for Row 5 (full ARSPI-Net)

---

## Experiment 2: GNN Architecture Comparison

### Research Question

Which GNN propagation mechanism (GCN, GraphSAGE, GAT) is most effective for electrode-graph classification when using LSM-BSC₆-PCA64 features on the spatial adjacency graph?

### Method

All three GNN architectures are evaluated under identical conditions:
- Features: LSM-BSC₆-PCA64 (the best embedding from Experiment 1)
- Graph: Spatial k-NN (k=5)
- Classifier: LogReg
- Layers: 2
- Validation: Subject-stratified 10-fold CV

The comparison tests whether attention-based neighbor weighting (GAT) provides benefit over uniform aggregation (GCN) or concatenation-based aggregation (GraphSAGE) for EEG electrode graphs.

### Output Figures

- `figures/ch5/architecture_comparison.pdf` — Bar chart of GCN vs GraphSAGE vs GAT balanced accuracy

---

## Experiment 3: Graph Sparsity Sweep

### Research Question

How sensitive is classification performance to graph sparsity? The spatial adjacency uses k-nearest neighbors — what is the optimal k?

### Method

The spatial adjacency is reconstructed for k ∈ {3, 5, 7, 10, 15} neighbors. For each k value, the full subject-stratified 10-fold CV is re-run using GCN propagation with LSM-BSC₆-PCA64 features.

- k=3: Very sparse graph (each electrode connects to ~3 nearest neighbors)
- k=5: Default setting
- k=15: Nearly half of all 34 electrodes are neighbors — approaches a fully connected graph

### Interpretation

If accuracy is stable across k, the graph adds value through its existence (distinguishing connected from unconnected) rather than through precise topology. If accuracy peaks at a specific k and degrades at extremes, the spatial neighborhood radius matters and over-smoothing (too many neighbors) or under-connection (too few) hurts.

### Output Figures

- `figures/ch5/sparsity_sweep.pdf` — Line plot of balanced accuracy vs k

---

## Experiment 4: GNN Depth Ablation

### Research Question

How many GNN propagation layers are optimal? More layers allow information to propagate further across the electrode graph, but risk over-smoothing (all node representations converging to the same vector).

### Method

GCN propagation with 1, 2, 3, and 4 layers using LSM-BSC₆-PCA64 features on the spatial adjacency (k=5). Full subject-stratified 10-fold CV for each depth.

- **1 layer:** Each node sees only direct neighbors (1-hop)
- **2 layers (default):** Each node aggregates 2-hop neighborhood
- **3-4 layers:** Broader receptive field, but increasingly homogeneous representations

### Interpretation

Over-smoothing is a known failure mode for deep GNNs on small graphs. With 34 electrodes and k=5 spatial neighbors, 3+ layers mean every node's representation contains information from most of the graph. If accuracy degrades at depth 3-4, this confirms that the 34-electrode EEG graph is too small for deep GNN propagation.

### Output Figures

- `figures/ch5/depth_ablation.pdf` — Line plot of balanced accuracy vs GNN depth

---

## Data Specification

### SHAPE EEG Dataset

| Property | Value |
|----------|-------|
| Per file dimensions | 1229 rows x 34 columns (microvolts) |
| Sampling rate | 1024 Hz |
| Rows 0-204 | 200 ms pre-stimulus baseline (already baseline-corrected) |
| Rows 205-1228 | 1000 ms post-stimulus |
| Conditions | IAPSNeg (Negative), IAPSNeu (Neutral), IAPSPos (Pleasant) |
| Subjects | 80+ |
| Files per subject | 3 (one per condition) |
| Naming convention | `SHAPE_Community_{SUBJECT_ID}_{CONDITION}_BC.txt` |
| Channels | 34 (standard 10-20 extended montage) |

### Preprocessing Pipeline

```
Raw (1229, 34) → Remove baseline rows 0-204 → (1024, 34) → Decimate 4x → (256, 34) → Z-score per epoch per channel
```

---

## Shared Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Reservoir neurons | 256 | Chapter 3 characterization |
| Leak rate (beta) | 0.05 | Chapter 3 characterization |
| Spike threshold | 0.5 | Chapter 3 characterization |
| Spectral radius | 0.9 | Chapter 3 characterization |
| BSC temporal bins | 6 | Chapter 4 validation |
| PCA components | 64 | Chapter 4 validation |
| Feature window | Timesteps 10-70 of 256 (for some analyses) | Post-transient steady state |
| Spatial k-NN | k=5 | Experiment 3 sweep |
| Functional threshold | 75th percentile | Standard practice |
| GNN layers | 2 | Experiment 4 ablation |
| GAT attention heads | 4 | Standard multi-head attention |
| CV folds | 10 | Subject-stratified (`StratifiedGroupKFold`) |
| LogReg regularization | C=0.1, max_iter=2000 | Default |
| MLP architecture | (128, 64), early stopping | Default |
| Random seed | 42 | Reproducibility |
| Figure DPI | 300 | Publication quality |
| Figure format | PDF | Vector graphics |

---

## Files

### Scripts

| File | Lines | Description |
|------|-------|-------------|
| `run_chapter5_experiments.py` | 1,050 | Complete Chapter 5 pipeline: data loading, preprocessing, LIF reservoir, 4 feature extractors, 2 graph constructors, 3 GNN architectures, 7-row baseline table, architecture comparison, sparsity sweep, depth ablation, 5 publication figures |
| `reproduce_chapter5.py` | 672 | Standalone reproducibility script: regenerates every figure, table, and statistical result. Runtime ~15-20 min on modern CPU. Outputs `ch5_all_results.pkl` + all figures |

### Archive

| File | Size | Description |
|------|------|-------------|
| `arspi_net_chapter5_complete (1).zip` | 6.1 MB | Complete experimental package: 13 exploration/classification/interpretability scripts, 7 NPZ data files (8.4 MB raw observations), 49 PDF figures (2.2 MB), LaTeX source (`chgraph_final.tex`), experimental log, negative results documentation |

### ZIP Archive Contents

The archive contains the full experimental development history:

```
arspi_net_chapter5_complete/
├── github_repo/
│   ├── README.md
│   ├── chgraph_final.tex              # LaTeX figure generation (36 KB)
│   ├── docs/
│   │   ├── experimental_log.md        # Experimental development history
│   │   └── negative_results_summary.md # Failed approaches (documented)
│   ├── data/ch5_raw_observations/     # 7 NPZ files (8.4 MB)
│   │   ├── obs01_data.npz             # Raw EEG waveforms
│   │   ├── obs02_data.npz             # Spike trains
│   │   ├── obs03_data.npz             # BSC₆ PCA-64 embeddings
│   │   ├── obs04_data.npz             # Connectivity matrices
│   │   ├── obs05_data.npz             # Clinical distributions
│   │   ├── obs06_data.npz             # Condition differences
│   │   └── obs07_data.npz             # Graph variability
│   ├── figures/
│   │   ├── ch5_raw_data/              # 7 observation PDFs
│   │   └── chGraphNeuralNetworks/     # 40+ publication figures
│   └── scripts/
│       ├── classification/            # Baseline + GNN classifiers (incl. failed approaches)
│       ├── exploration/               # Raw data visualization
│       ├── gnn_experiments/           # Personality topology analysis
│       └── interpretability/          # Clinical graph biomarkers + figure generation
```

### Expected Output Figures (from `run_chapter5_experiments.py`)

| File | Description |
|------|-------------|
| `figures/ch5/baseline_comparison.pdf` | 7-row bar chart: balanced accuracy per experimental condition |
| `figures/ch5/architecture_comparison.pdf` | GCN vs GraphSAGE vs GAT balanced accuracy |
| `figures/ch5/sparsity_sweep.pdf` | Balanced accuracy vs spatial k-NN k value |
| `figures/ch5/depth_ablation.pdf` | Balanced accuracy vs GNN layer depth |
| `figures/ch5/confusion_matrix.pdf` | 3x3 confusion matrix for full ARSPI-Net (Row 5) |

### Expected Output Data (from `reproduce_chapter5.py`)

| File | Description |
|------|-------------|
| `ch5_all_results.pkl` | Complete numerical results: accuracies, F1, confusion matrices for all conditions |
| `shape_features.pkl` | Preprocessed features passed forward to Chapter 6 |

---

## Verification Results

<!-- Last run: 2026-03-27, Result: 102/102 PASS across 3 verification scripts -->

### `verify_chapter5.py` — Core Infrastructure (32/32 PASS)

```bash
python chapter5Experiments/verify_chapter5.py
```

Verified components:
- **Script import:** `run_chapter5_experiments.py` imports successfully
- **LIF Reservoir (7 tests):** Instantiation, weight shapes, forward pass shapes, binary spikes, non-zero activity, sparsity
- **Feature extraction (4 tests):** BSC6 produces 384-dim vector with non-negative values; MFR produces 64-dim vector in [0,1]
- **Conventional features (3 tests):** BandPower shape (2, 34, 5), non-negative; Hjorth shape (2, 34, 3)
- **Graph construction (6 tests):** 34 electrode positions, spatial adjacency (symmetric, binary, no self-loops), functional adjacency
- **GNN propagation (5 tests):** GCN preserves shape, GraphSAGE doubles features, GAT returns features + attention matrices, attention rows sum to ~1
- **Graph readout (1 test):** Mean readout produces 64-dim vector
- **Classification (3 tests):** Full CV pipeline runs, returns accuracy and predictions

### `verify_experiment_zero.py` — Baseline Disambiguation (37/37 PASS)

```bash
python chapter5Experiments/verify_experiment_zero.py
```

Verified components:
- **Syntax validation (1 test):** `experiment_zero.py` parses without errors
- **Import validation (1 test):** All core functions importable (LIFReservoir, bsc6_encode, flatten_raw, subject_center, evaluate)
- **LIF Reservoir (7 tests):** Instantiation, weight shapes, spectral radius = 0.9, forward pass shape, binary spikes, non-silent
- **BSC6 encoding (4 tests):** Correct dimensionality (384-dim), non-negative, nonzero entries, sum consistency
- **Flatten raw (2 tests):** Shape transformation (N,256,34) -> (N,8704), value preservation
- **Subject centering (4 tests):** Correct shape, values change, per-subject mean is zero, does not modify input
- **CV pipeline (4 tests):** Returns numpy array, correct fold count, accuracies in [0,1], finite mean
- **Determinism (2 tests):** Same seed produces identical spikes, different seed produces different spikes
- **Channel seeding (1 test):** Different channels produce different reservoir responses
- **End-to-end (4 tests):** Raw and reservoir paths with centering produce correct shapes
- **Script structure (7 tests):** All 4 experimental conditions defined, SVM cross-check present, pickle output saved

### `verify_reproduce_chapter5.py` — Reproducibility Pipeline (33/33 PASS)

```bash
python chapter5Experiments/verify_reproduce_chapter5.py
```

Verified components:
- **Syntax and import (2 tests):** Module parses and imports successfully
- **LIF Reservoir (8 tests):** Instantiation, weight shapes, spectral radius, forward pass, binary spikes, non-silent, returns spikes only (not tuple)
- **Feature extraction (2 tests):** BSC6 encoding present, band power referenced
- **GNN implementations (7 tests):** GCN, GraphSAGE/skip, GAT/attention present in source; manual GCN/SAGE/attention produce valid shapes and finite outputs; attention weights sum to 1
- **Graph construction (3 tests):** Graph structure, electrode topology, and k-NN referenced
- **Classification pipeline (3 tests):** StratifiedGroupKFold CV runs, returns correct fold count, finite accuracies
- **Graph readout (1 test):** Mean pooling produces correct-dim vector
- **Split checks (2 tests):** StratifiedGroupKFold used; `fit_transform` present. The second check was labelled "PCA fitted per fold" until 2026-11. It is a source-string match and the PCA in this script is fitted on all data (`docs/v1/DEFECT_FIXES.md`, D8)
- **Output structure (3 tests):** Pickle saved, PDF figures generated, Agg backend used
- **Deep baselines (2 tests):** Baseline results pickle exists and contains data

Full end-to-end verification requires the SHAPE EEG dataset. Use `--demo` mode for pipeline testing:
```bash
python chapter5Experiments/run_chapter5_experiments.py --demo
```

### Relationship to Extended 4-Class Experiments

The `experiments/ch5_4class/` directory extends this 3-class pipeline to 4 IAPS subcategories (Threat, Mutilation, Cute, Erotic). That extension includes its own verification script (`verify_ch5_4class.py`, 25/25 PASS) testing the same infrastructure components plus band power extraction and 4-category configuration.

---

## Relationship to Other Chapters

- **Chapter 3** established the LIF reservoir parameters (256 neurons, beta=0.05, threshold=0.5, spectral radius=0.9) through systematic characterization.
- **Chapter 4** validated BSC₆ + PCA-64 as the optimal feature coding (>90% on synthetic data; MFR ~50% — proving temporal coding is necessary).
- **Chapter 5** applies the full pipeline to real clinical EEG for the first time, adding graph-structured classification.
- **Chapter 6** uses the Chapter 5 feature pipeline to characterize reservoir dynamics per affective condition (Lyapunov exponents, permutation entropy, relaxation time).
- **Chapter 7** couples Chapter 5's topological descriptors with Chapter 6's dynamical descriptors to study their alignment structure.

## Sample

- 80+ subjects from the [SHAPE project](https://lab-can.com/shape/)
- 3 affective conditions: Negative, Neutral, Pleasant (IAPS picture viewing)
- Subject-stratified cross-validation prevents data leakage
- Demo mode available with `--demo` flag for pipeline testing without SHAPE data access
