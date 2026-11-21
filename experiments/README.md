# Extended Experiments

> **Navigation note, added 2026-11-21.** This file is part of the v1 record and
> describes three of the directories below. The complete, current index of every
> experiment in the repository, with tier, command, protocol and known problems, is
> `docs/EXPERIMENT_REGISTRY.md`. The index of committed result records is
> `experiments/records.lock.json`.

This directory contains experimental extensions that build on the core chapter experiments. These scripts were added in March 2026 to address cross-chapter questions, expand analyses to different class granularities, and directly test the dissertation's central thesis.

---

## Overview

| Pipeline | Directory | Granularity | Signal Strength | Experiments | Verification |
|----------|-----------|-------------|-----------------|-------------|-------------|
| 4-Class Classification | `ch5_4class/` | 4-class (Threat/Mutilation/Cute/Erotic) | 2.4% condition variance | 11 experiments (6 classification + 5 clinical) | 25/25 PASS |
| 3-Class Dynamical + Coupling | `ch6_ch7_3class/` | 3-class (Negative/Neutral/Pleasant) | 8.7% condition variance | 12 experiments (7 Ch6 + 5 Ch7) | 28/28 PASS |
| Layer Ablation | `ablation/` | 3-class | 8.7% condition variance | 16 conditions (A0-A9 emotion + C1-C6 clinical) | 23/23 PASS |

**Total verification: 76/76 PASS across all extended experiments.**

---

## Subdirectories

### `ch5_4class/` — 4-Class Classification Extension

Extends Chapter 5's 3-class clinical classification to 4 IAPS subcategories (Threat, Mutilation, Cute, Erotic). Tests whether within-valence pairs carry distinct spatiotemporal signatures in the reservoir embedding space.

**Key questions answered:**
- Do Threat and Mutilation (both negative) produce distinguishable embeddings?
- Is the PTSD x Threat-Mutilation interaction detectable in the embedding space?
- Does the feature hierarchy (amplitude vs temporal) invert between within-valence pairs?

**Scripts:** 3 sequential scripts (feature extraction, raw observations, full experimental program)

### `ch6_ch7_3class/` — 3-Class Dynamical and Coupling Pipeline

Consolidated pipeline for Chapters 6 and 7 at 3-class granularity (Negative, Neutral, Pleasant). The 3-class regime provides a 3.6x signal advantage over 4-class (8.7% vs 2.4% condition variance), enabling stronger statistical tests of dynamical condition sensitivity and coupling structure.

**Key questions answered:**
- Are dynamical metrics condition-sensitive at 3-class? (Ch6 experiments)
- Which metric family (amplitude vs temporal) is more condition-sensitive?
- Is coupling observation-specific (not a trait)? (Ch7 experiments)
- Do descriptor families carry complementary discriminative information?

**Scripts:** 4 sequential scripts (feature extraction, raw observations, Ch6 experiments, Ch7 experiments)

### `ablation/` — Layer Ablation (Keystone Experiment)

The dissertation's keystone experiment testing whether ARSPI-Net's three response layers (discriminative embedding E, dynamical trajectory D, spatial topology T) are redundant or complementary. Directly tests the central thesis by systematically ablating and combining feature blocks from all chapters.

**Key questions answered:**
- Are the three layers operationally distinct?
- Do different layers dominate for different clinical dimensions?
- Does combining layers improve beyond the best individual layer?

**Script:** Single comprehensive ablation script (A0-A9 emotion + C1-C6 clinical conditions)

---

## Relationship to Chapter Directories

```
Repository Root
├── chapter4Experiments/    ← Synthetic proof-of-concept (Ch4)
├── chapter5Experiments/    ← 3-class clinical classification (Ch5)
├── chapter6Experiments/    ← 4-class dynamical characterization (Ch6)
├── chapter7Experiments/    ← 4-class coupling analysis (Ch7)
│
└── experiments/            ← Extended analyses (this directory)
    ├── ch5_4class/         ← 4-class classification extension
    ├── ch6_ch7_3class/     ← 3-class dynamical + coupling pipeline
    └── ablation/           ← Cross-chapter layer ablation
```

The `chapter*Experiments/` directories contain the original per-chapter experiments. This `experiments/` directory contains the newer cross-chapter analyses that emerged from the March 2026 methodology review:

- **ch5_4class** extends Chapter 5 to finer granularity (4-class)
- **ch6_ch7_3class** consolidates Chapters 6-7 at coarser granularity (3-class, stronger signal)
- **ablation** tests the system-level thesis that spans all chapters

The two granularities are complementary, not redundant: 3-class provides statistical power, 4-class provides within-valence resolution.

---

## Data Requirements

All scripts require data from the [Stress, Health, and the Psychophysiology of Emotion (SHAPE) project](https://lab-can.com/shape/) (not included in repository). See the root README for data format specifications and the `validation/` directory for data quality-control scripts.
