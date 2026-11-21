# Chapter 5 — 4-Class Classification Extension

## Overview

These scripts extend the Chapter 5 clinical EEG classification pipeline from 3-class (Negative / Neutral / Pleasant) to 4-class (Threat / Mutilation / Cute / Erotic). This tests whether within-valence subcategory pairs carry distinct spatiotemporal signatures in the ARSPI-Net embedding space.

---

## Dissertation Context

Chapter 5 originally classified 3 broad affective conditions. The 4-class extension is critical because it reveals whether the 3-class collapse lost discriminative structure. If Threat and Mutilation (both negative) produce different reservoir embeddings, the temporal coding captures finer affective distinctions than valence alone. This finding feeds directly into Chapter 6's within-valence dissociation analysis and Chapter 7's category-conditioned coupling structure.

The 4-class regime operates in a weaker signal environment (2.4% condition variance vs 8.7% at 3-class), making it harder but more informative — any structure detected here is robust.

---

## Scripts (run in order)

### 1. `ch5_4class_01_feature_extraction.py` — Feature Extraction

**Purpose:** Extract BSC6-PCA64 and band-power features from all 4 IAPS subcategories.
- **Input:** Raw EEG from `categoriesbatch{1-4}/` (844 observations: 211 subjects x 4 categories)
- **Output:** `shape_features_4class.pkl` containing downsampled EEG, BSC6 features, PCA-64 embeddings, band-power features, MFR vectors
- **Runtime:** ~20-40 minutes
- **Key parameters:** N_res=256, beta=0.05, threshold=0.5, PCA components=64, BSC bins=6

### 2. `ch5_4class_02_raw_observations.py` — Raw Observations

**Purpose:** Characterize the 4-class data at every pipeline stage BEFORE classification modeling.
- **Input:** `shape_features_4class.pkl`, `clinical_profile.csv`
- **Output:** 7 PDF figures + 7 .npz data files in `ch5_4class_raw_data/`
- **Runtime:** ~3-5 minutes
- **Observations:**
  1. Grand-average ERPs by subcategory
  2. Spike statistics (total spikes, firing rates per category)
  3. Embedding structure (PCA projections, between-class distances)
  4. Inter-channel connectivity (tPLV matrices)
  5. Graph variability across subjects
  6. Clinical metadata coverage
  7. Within-vs-between valence distances (key diagnostic of 4-class separability)

### 3. `ch5_4class_03_classification_full.py` — Full Experimental Program

**Purpose:** Complete 4-class experimental program with 11 experiments.
- **Input:** `shape_features_4class.pkl`, `clinical_profile.csv`
- **Output:** `ch5_4class_results.pkl`, PDF figures in `ch5_4class_figures/`
- **Validation:** Subject-stratified 10-fold CV. **Correction (2026-11):** this line used to say PCA was fitted per fold on training data only. The code fits PCA on all observations before the split (`ch5_4class_01_feature_extraction.py` lines 358-361, `ch5_4class_03_classification_full.py` lines 187-188), so these results are not leak-free (`docs/v1/DEFECT_FIXES.md`, D8)

**Classification experiments (EXP 1-6):**

| Exp | Question | Method |
|-----|----------|--------|
| 1 | Baseline performance at 4-class? | BandPower+Hjorth vs BSC6-PCA64, LogReg/MLP/GAT |
| 2 | Which GNN architecture? | GCN vs GraphSAGE vs GAT comparison |
| 3 | Confusion structure? | 3x3 and 4x4 confusion matrices |
| 4 | Pairwise contrasts? | All 6 pairwise subcategory classifications |
| 5 | Variance decomposition? | Subject vs condition vs residual partition |
| 6 | Per-channel discrimination? | Channel-level classification maps |

**Clinical experiments (EXP 7-11):**

| Exp | Question | Method |
|-----|----------|--------|
| 7 | Channel-level biomarkers? | Per-electrode clinical group contrasts |
| 8 | Condition x clinical interactions? | SUD/PTSD x subcategory interaction tests |
| 9 | Edge-level biomarkers? | Graph connectivity clinical comparisons |
| 10 | Comorbidity burden? | Correlation of biomarker effects with comorbidity count |
| 11 | Within-valence x clinical? | Threat-Mutilation and Cute-Erotic x diagnosis interactions |

---

## Key Results and Interpretation

### Classification Performance

The 4-class problem is substantially harder than 3-class (chance = 25% vs 33%). Expected performance hierarchy:

| Condition | Expected Accuracy | Interpretation |
|-----------|------------------|----------------|
| BandPower + LogReg baseline | ~35-40% | Conventional features capture minimal subcategory structure |
| BSC6-PCA64 + MLP | ~45-55% | Reservoir embeddings detect within-valence differences |
| BSC6-PCA64 + GAT (spatial) | ~50-60% | Graph structure adds spatial context |

### Pairwise Contrast Hierarchy

The 6 pairwise contrasts reveal which subcategory pairs are most and least distinguishable:
- **Between-valence pairs** (Threat-Cute, Threat-Erotic, Mutilation-Cute, Mutilation-Erotic): Expected higher accuracy, driven by the valence difference already captured at 3-class
- **Within-valence pairs** (Threat-Mutilation, Cute-Erotic): The critical test — any accuracy above chance here proves the reservoir captures finer affective distinctions than valence

### Variance Decomposition

Expected structure: subject identity dominates (>60%), condition accounts for ~2.4% at 4-class, with the remainder in residual. The low condition fraction is consistent with the 3.6x weaker signal relative to 3-class.

### Clinical Findings

The 4-class regime uniquely enables:
- **PTSD x Threat-Mutilation interaction:** Whether threat-specific hypervigilance appears in the embedding space
- **Feature hierarchy inversion:** Whether the most discriminative metric family (amplitude vs temporal) differs between within-valence pairs
- **SUD x within-positive interaction:** Whether substance use alters the Cute-Erotic embedding distinction

These findings complement the 3-class clinical analyses and require within-valence resolution that the 3-class collapse cannot provide.

---

## Verification Results

<!-- Last run: 2026-03-20, Result: 25/25 PASS -->

```bash
python experiments/ch5_4class/verify_ch5_4class.py
```

**Result: 25/25 PASS.** Infrastructure tests on synthetic data without requiring [Stress, Health, and the Psychophysiology of Emotion (SHAPE) project](https://lab-can.com/shape/) EEG data:

- **Syntax validation (3 tests):** All 3 scripts parse without errors
- **LIF Reservoir (8 tests):** Module imports, instantiation, W_in (64,1) and W_rec (64,64) shapes, forward pass shapes (256,64), binary spikes, finite membrane, non-negative membrane, non-silent spikes
- **BSC extraction (3 tests):** Produces 384-dim vector (64 neurons x 6 bins), non-negative, nonzero entries
- **Band power (3 tests):** Produces 5-dim vector per channel, non-negative, nonzero entries
- **Configuration consistency (7 tests):** N_RES=256, BETA=0.05, THRESHOLD=0.5, SEED=42, BSC_N_BINS=6, PCA_N_COMPONENTS=64, 4 categories defined

Note: One numpy compatibility fix was applied — `np.trapz` replaced with `np.trapezoid` for numpy >= 2.0.

## Dependencies

numpy, scipy, scikit-learn, matplotlib, pandas, pickle

## Data Requirements

SHAPE EEG subcategory files in `categoriesbatch{1-4}/` directories. Each file: (1229, 34) float64, baseline-corrected, trial-averaged microvolts.
