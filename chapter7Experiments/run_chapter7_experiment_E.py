#!/usr/bin/env python3
"""
Chapter 7 — Experiment 7.5: Augmentation Ablation
==================================================
Scientific Cycle: Motivation → Design → Raw Observation → Analysis → Next Question
Prerequisites:
  - subject_features.csv from extract_features_for_expE.py
  - C_matrices.csv from extract_C_matrices.py
  - SHAPE_Community_Andrew_Psychopathology.xlsx (clinical metadata)
Usage:
  python3 run_chapter7_experiment_E.py
Inputs:
  subject_features.csv
    211 rows × 307 columns (subject ID + 238 D features + 68 T features)
    D features: 34 channels × 7 dual-gate validated dynamical metrics
    T features: 34 channels × 2 primary topological metrics (strength, clustering)
  C_matrices.csv
    844 rows × 16 columns (for reconstructing subject-averaged κ)
  SHAPE_Community_Andrew_Psychopathology.xlsx
    Binary diagnosis columns (MDD, SUD, PTSD, GAD, ADHD)
Outputs:
  Figures (Raw Observation):
    fig7_E1_raw_pca_projections.pdf   — PCA scatter for SUD/ADHD × T/D/T+D
    fig7_E2_raw_univariate_effects.pdf — Per-feature Cohen's d profiles
    fig7_E3_pca_variance_curves.pdf   — Effective dimensionality curves
  Figures (Analysis):
    fig7_E4_classification_auc.pdf    — AUC bar charts × conditions × diagnoses
    fig7_E5_paired_auc_detail.pdf     — Paired ΔAUC distributions
Mathematical Motivation:
  Experiments 7.1-7.4 established that coupling between dynamical and
  topological descriptor families exists (d_z = 1.06), is observation-
  specific (ICC = 0.059), and does not differ by diagnosis at the scalar
  κ level. Chapter 6 established that dynamical descriptors are analytically
  informative but not discriminatively additive at the linear-readout level.
  The question: does combining topological and dynamical descriptor families
  in a single flat feature vector improve clinical detection beyond either
  family alone? Or is their value organizational (coupling structure,
  Experiments A-D) rather than discriminatively complementary?
  The raw observation step examines the feature spaces — their dimensionality
  structure, group separation geometry, and univariate effect profiles —
  before any classifier is trained, so that the classification results can
  be understood as consequences of observable feature-space properties.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from scipy.stats import wilcoxon, mannwhitneyu
import os
import time
# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUBJECT_FEATURES_FILE = os.path.join(SCRIPT_DIR, 'chapter7_results', 'subject_features.csv')
C_MATRIX_FILE = os.path.join(SCRIPT_DIR, 'chapter7_results', 'C_matrices.csv')
PSYCH_FILE = os.path.join(SCRIPT_DIR, '..', 'SHAPE_Community_Andrew_Psychopathology.xlsx')
FIGURE_DIR = os.environ.get('ARSPI_FIGURE_DIR', os.path.join(SCRIPT_DIR, 'figures'))
DX_LIST = ['SUD', 'MDD', 'PTSD', 'GAD', 'ADHD']
N_REPEATS = 10
N_SPLITS = 5
RANDOM_SEED = 42
# ═══════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════
def run_cv(X, y, n_repeats=N_REPEATS, n_splits=N_SPLITS, seed=RANDOM_SEED):
    """Run repeated stratified K-fold logistic regression."""
    results = {'auc': [], 'bacc': [], 'f1': []}
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=seed + rep)
        for train_idx, test_idx in skf.split(X, y):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])
            clf = LogisticRegression(C=1.0, max_iter=2000,
                                     solver='lbfgs', random_state=seed)
            clf.fit(X_tr, y[train_idx])
            y_prob = clf.predict_proba(X_te)[:, 1]
            y_pred = clf.predict(X_te)
            try:
                auc = roc_auc_score(y[test_idx], y_prob)
            except ValueError:
                auc = 0.5
            results['auc'].append(auc)
            results['bacc'].append(balanced_accuracy_score(y[test_idx],
                                                           y_pred))
            results['f1'].append(f1_score(y[test_idx], y_pred,
                                          average='macro'))
    return {k: np.array(v) for k, v in results.items()}
def cohens_d_twosample(group1, group2):
    """Cohen's d with pooled standard deviation."""
    n1, n2 = len(group1), len(group2)
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) /
                     (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / (pooled + 1e-10)
# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    print("=" * 70)
    print("EXPERIMENT 7.5: Augmentation Ablation")
    print("Following: Motivation → Design → Raw Observation → Analysis")
    print("=" * 70)
    # ── Load features ──
    sf = pd.read_csv(SUBJECT_FEATURES_FILE)
    d_cols = [c for c in sf.columns if c.startswith('d_')]
    t_cols = [c for c in sf.columns if c.startswith('t_')]
    subjects = sf['subject'].values
    X_D = sf[d_cols].values  # (211, 238)
    X_T = sf[t_cols].values  # (211, 68)
    X_TD = np.hstack([X_T, X_D])  # (211, 306)
    # ── Reconstruct subject-averaged kappa ──
    C_df = pd.read_csv(C_MATRIX_FILE)
    corr_cols = [c for c in C_df.columns if '_x_' in c]
    subj_kappa = {}
    for sid in subjects:
        rows = C_df[C_df.subject == int(sid)]
        kappas = []
        for _, row in rows.iterrows():
            C = row[corr_cols].values.astype(float).reshape(7, 2)
            kappas.append(np.linalg.norm(C, 'fro') / np.sqrt(14))
        subj_kappa[int(sid)] = np.mean(kappas)
    kappa_vec = np.array([subj_kappa.get(int(s), 0)
                          for s in subjects]).reshape(-1, 1)
    X_TDK = np.hstack([X_TD, kappa_vec])  # (211, 307)
    # ── Load diagnosis labels ──
    psych = pd.read_excel(PSYCH_FILE)
    psych_dict = {int(row['ID']): row for _, row in psych.iterrows()}
    dx_labels = {}
    for dx in DX_LIST:
        dx_labels[dx] = {}
        for sid in subjects:
            if int(sid) in psych_dict:
                val = psych_dict[int(sid)].get(dx, np.nan)
                if not np.isnan(val):
                    dx_labels[dx][int(sid)] = int(val)
    all_feature_cols = t_cols + d_cols
    X_all = np.hstack([X_T, X_D])
    conditions = {
        'T-only': X_T,
        'D-only': X_D,
        'T+D': X_TD,
        'T+D+kappa': X_TDK,
    }
    cond_names = list(conditions.keys())
    # ==============================================================
    # 1. MATHEMATICAL MOTIVATION
    # ==============================================================
    print("\n" + "=" * 60)
    print("1. MATHEMATICAL MOTIVATION")
    print("=" * 60)
    print("""
Experiments 7.1-7.4 characterized coupling between dynamical and
topological descriptor families. The coupling exists (d_z = 1.06),
is observation-specific (ICC = 0.059), and does not differ by
diagnosis at the scalar κ level.
Chapter 6 established that dynamical descriptors are analytically
informative rather than discriminatively additive at the linear level.
The question: when the two descriptor families are combined in a flat
feature vector, does the combination improve clinical detection beyond
either family alone? Or is their value organizational rather than
discriminatively complementary?
Before running any classifier, the raw observation step examines what
the feature spaces look like: their dimensionality structure, their
group separation geometry, and whether the two families occupy
overlapping or distinct regions of the feature space.
""")
    # ==============================================================
    # 2. EXPERIMENTAL DESIGN
    # ==============================================================
    print("=" * 60)
    print("2. EXPERIMENTAL DESIGN")
    print("=" * 60)
    print(f"""
Feature conditions:
  T-only:      {X_T.shape[1]} features (34 electrodes × 2 topology)
  D-only:      {X_D.shape[1]} features (34 electrodes × 7 dynamics)
  T+D:         {X_TD.shape[1]} features (concatenation)
  T+D+κ:       {X_TDK.shape[1]} features (+ subject-averaged κ̄)
Primary task: SUD detection (strongest prior signal)
Secondary: MDD, PTSD, GAD, ADHD
Readout: L2-regularized logistic regression (C=1.0)
CV: Subject-level stratified {N_SPLITS}-fold, repeated {N_REPEATS} times
    ({N_SPLITS * N_REPEATS} total folds)
Metrics: AUC, balanced accuracy, macro-F1
""")
    # ==============================================================
    # 3. RAW OBSERVATION
    # ==============================================================
    print("=" * 60)
    print("3. RAW OBSERVATION")
    print("=" * 60)
    # 3a. Feature matrix structure
    print("\n--- 3a. Feature matrix structure ---")
    print(f"  X_T shape:  {X_T.shape}")
    print(f"  X_D shape:  {X_D.shape}")
    print(f"  X_TD shape: {X_TD.shape}")
    print(f"  N/p ratio:  T={len(subjects)/X_T.shape[1]:.1f}  "
          f"D={len(subjects)/X_D.shape[1]:.2f}  "
          f"TD={len(subjects)/X_TD.shape[1]:.2f}")
    print(f"  Matrix rank: T={np.linalg.matrix_rank(X_T)}/{X_T.shape[1]}  "
          f"D={np.linalg.matrix_rank(X_D)}/{X_D.shape[1]}  "
          f"TD={np.linalg.matrix_rank(X_TD)}/{X_TD.shape[1]}")
    # 3b. PCA variance structure
    print("\n--- 3b. PCA variance structure ---")
    for name, X in [('T-only', X_T), ('D-only', X_D), ('T+D', X_TD)]:
        Xs = StandardScaler().fit_transform(X)
        pca = PCA().fit(Xs)
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        n90 = np.searchsorted(cum_var, 0.90) + 1
        n95 = np.searchsorted(cum_var, 0.95) + 1
        n99 = np.searchsorted(cum_var, 0.99) + 1
        print(f"  {name:<8} PCs for 90%: {n90}  95%: {n95}  99%: {n99}  "
              f"PC1: {pca.explained_variance_ratio_[0]:.3f}")
    # 3c. Group separation in PCA space
    print("\n--- 3c. Group separation in PCA projection ---")
    for dx in ['SUD', 'ADHD']:
        y_arr = np.array([dx_labels[dx].get(int(s), -1) for s in subjects])
        valid = y_arr >= 0
        y = y_arr[valid]
        print(f"  {dx}:")
        for name, X in [('T-only', X_T), ('D-only', X_D), ('T+D', X_TD)]:
            Xs = StandardScaler().fit_transform(X[valid])
            pca_proj = PCA(n_components=5).fit_transform(Xs)
            pos_cent = pca_proj[y == 1, :2].mean(axis=0)
            neg_cent = pca_proj[y == 0, :2].mean(axis=0)
            dist = np.linalg.norm(pos_cent - neg_cent)
            spread = (np.std(pca_proj[y == 1, :2], axis=0).mean() +
                      np.std(pca_proj[y == 0, :2], axis=0).mean())
            print(f"    {name:<8} Centroid dist: {dist:.3f}  "
                  f"Spread: {spread:.3f}  "
                  f"Ratio: {dist / (spread + 1e-10):.3f}")
    # 3d. Univariate screening
    for dx in ['SUD', 'ADHD']:
        y_arr = np.array([dx_labels[dx].get(int(s), -1) for s in subjects])
        valid = y_arr >= 0
        y = y_arr[valid]
        print(f"\n--- 3d. Univariate feature screening ({dx}, "
              f"top 10 by |d|) ---")
        scores = []
        for i, col in enumerate(all_feature_cols):
            pos = X_all[valid][y == 1, i]
            neg = X_all[valid][y == 0, i]
            d = cohens_d_twosample(pos, neg)
            _, p = mannwhitneyu(pos, neg, alternative='two-sided')
            fam = 'T' if col.startswith('t_') else 'D'
            scores.append((abs(d), d, p, col, fam))
        scores.sort(reverse=True)
        print(f"  {'Rank':<5} {'Feature':<35} {'Fam':>4} "
              f"{'d':>8} {'p':>12}")
        for rank, (_, d, p, col, fam) in enumerate(scores[:10]):
            print(f"  {rank+1:<5} {col:<35} {fam:>4} "
                  f"{d:>8.4f} {p:>12.4e}")
        n_sig_T = sum(1 for _, _, p, _, f in scores
                      if p < 0.05 and f == 'T')
        n_sig_D = sum(1 for _, _, p, _, f in scores
                      if p < 0.05 and f == 'D')
        print(f"\n  Uncorrected p<0.05: T={n_sig_T}/{len(t_cols)}, "
              f"D={n_sig_D}/{len(d_cols)}")
    # ── RAW OBSERVATION FIGURES ──
    # Figure E1: PCA projections
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for row, dx in enumerate(['SUD', 'ADHD']):
        y_arr = np.array([dx_labels[dx].get(int(s), -1) for s in subjects])
        valid = y_arr >= 0
        y = y_arr[valid]
        for col_idx, (name, X) in enumerate([
                ('T-only', X_T), ('D-only', X_D), ('T+D', X_TD)]):
            ax = axes[row, col_idx]
            Xs = StandardScaler().fit_transform(X[valid])
            proj = PCA(n_components=2).fit_transform(Xs)
            ax.scatter(proj[y == 0, 0], proj[y == 0, 1],
                       alpha=0.3, s=15, c='steelblue', label='Dx−')
            ax.scatter(proj[y == 1, 0], proj[y == 1, 1],
                       alpha=0.3, s=15, c='red', label='Dx+')
            for lbl, clr in [(0, 'steelblue'), (1, 'red')]:
                c = proj[y == lbl].mean(axis=0)
                ax.scatter(*c, s=200, c=clr, edgecolors='black',
                           linewidths=2, marker='X', zorder=10)
            ax.set_xlabel('PC1', fontsize=10)
            ax.set_ylabel('PC2' if col_idx == 0 else '', fontsize=10)
            ax.set_title(f'{dx} — {name}', fontsize=11)
            if row == 0 and col_idx == 0:
                ax.legend(fontsize=9)
    plt.suptitle('Raw Observation: PCA Projections Before Classification\n'
                 '(× = group centroid)', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(f'{FIGURE_DIR}/fig7_E1_raw_pca_projections.pdf',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: fig7_E1_raw_pca_projections.pdf")
    # Figure E2: Univariate effect profiles
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    for ax, dx in zip(axes, ['SUD', 'ADHD']):
        y_arr = np.array([dx_labels[dx].get(int(s), -1) for s in subjects])
        valid = y_arr >= 0
        y = y_arr[valid]
        t_ds = []
        d_ds = []
        for i, col in enumerate(all_feature_cols):
            pos = X_all[valid][y == 1, i]
            neg = X_all[valid][y == 0, i]
            d = cohens_d_twosample(pos, neg)
            if col.startswith('t_'):
                t_ds.append(d)
            else:
                d_ds.append(d)
        t_idx = list(range(len(t_ds)))
        d_idx = list(range(len(t_ds), len(t_ds) + len(d_ds)))
        ax.bar(t_idx, t_ds, color='#4878CF', alpha=0.7, width=1.0,
               label='Topology')
        ax.bar(d_idx, d_ds, color='#D65F5F', alpha=0.7, width=1.0,
               label='Dynamics')
        ax.axhline(0, color='black', lw=0.5)
        ax.axhline(0.2, color='gray', ls=':', lw=1, alpha=0.5)
        ax.axhline(-0.2, color='gray', ls=':', lw=1, alpha=0.5)
        ax.axvline(len(t_ds) - 0.5, color='black', ls='--', lw=1)
        ax.set_ylabel("Cohen's d", fontsize=11)
        ax.set_title(f'{dx}: Per-Feature Effect Sizes', fontsize=12)
        ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(f'{FIGURE_DIR}/fig7_E2_raw_univariate_effects.pdf',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig7_E2_raw_univariate_effects.pdf")
    # Figure E3: PCA variance curves
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for name, X, color in [('T-only', X_T, '#4878CF'),
                             ('D-only', X_D, '#D65F5F'),
                             ('T+D', X_TD, '#6ACC65')]:
        Xs = StandardScaler().fit_transform(X)
        pca = PCA().fit(Xs)
        cum = np.cumsum(pca.explained_variance_ratio_)
        ax.plot(range(1, len(cum) + 1), cum, '-', color=color,
                lw=2, label=name)
    ax.axhline(0.90, color='gray', ls=':', lw=1)
    ax.axhline(0.95, color='gray', ls=':', lw=1)
    ax.axvline(len(subjects), color='black', ls='--', lw=1, alpha=0.5,
               label=f'N={len(subjects)}')
    ax.set_xlabel('Number of PCs', fontsize=11)
    ax.set_ylabel('Cumulative variance explained', fontsize=11)
    ax.set_title('Effective Dimensionality of Feature Spaces', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 250)
    plt.tight_layout()
    fig.savefig(f'{FIGURE_DIR}/fig7_E3_pca_variance_curves.pdf',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig7_E3_pca_variance_curves.pdf")
    print("\n--- Raw observation complete. "
          "Proceeding to quantitative analysis. ---")
    # ==============================================================
    # 4. QUANTITATIVE ANALYSIS
    # ==============================================================
    print("\n" + "=" * 60)
    print("4. QUANTITATIVE ANALYSIS")
    print("=" * 60)
    # Until 2026-11 a fixed paragraph here stated what "the raw observation predicted"
    # whatever the data showed (docs/v1/DEFECT_FIXES.md, D9).
    print("""
Testing whether the patterns seen in the raw observations above
translate to multivariate classification performance.
""")
    all_results = {}
    for dx in DX_LIST:
        y_full = np.array([dx_labels[dx].get(int(s), -1)
                           for s in subjects])
        valid = y_full >= 0
        y = y_full[valid]
        print(f"\n  ── {dx} Detection "
              f"(n+={sum(y==1)}, n-={sum(y==0)}) ──")
        print(f"  {'Cond':<12} {'AUC':>8} {'±':>4} {'BAcc':>8} "
              f"{'±':>4} {'F1':>8} {'±':>4}")
        dx_res = {}
        for cond_name, X_full in conditions.items():
            X = X_full[valid]
            res = run_cv(X, y)
            dx_res[cond_name] = res
            print(f"  {cond_name:<12} {res['auc'].mean():>8.4f} "
                  f"{'±':>2}{res['auc'].std():>5.3f} "
                  f"{res['bacc'].mean():>8.4f} "
                  f"{'±':>2}{res['bacc'].std():>5.3f} "
                  f"{res['f1'].mean():>8.4f} "
                  f"{'±':>2}{res['f1'].std():>5.3f}")
        all_results[dx] = dx_res
        # Paired comparisons
        for c1, c2 in [('T+D', 'T-only'), ('T+D', 'D-only'),
                        ('T+D+kappa', 'T+D')]:
            diff = dx_res[c1]['auc'] - dx_res[c2]['auc']
            if np.std(diff) > 0:
                _, p = wilcoxon(diff)
            else:
                p = 1.0
            d_z = np.mean(diff) / (np.std(diff) + 1e-10)
            print(f"    {c1} vs {c2}: ΔAUC = {np.mean(diff):+.4f}, "
                  f"d_z = {d_z:.3f}, p = {p:.4e}")
    # ── ANALYSIS FIGURES ──
    # Figure E4: AUC comparison
    fig, axes = plt.subplots(1, len(DX_LIST), figsize=(20, 5))
    colors = ['#4878CF', '#D65F5F', '#6ACC65', '#B47CC7']
    for ax, dx in zip(axes, DX_LIST):
        means = [all_results[dx][c]['auc'].mean() for c in cond_names]
        stds = [all_results[dx][c]['auc'].std() for c in cond_names]
        bars = ax.bar(range(len(cond_names)), means, yerr=stds,
                      color=colors, edgecolor='black', width=0.7,
                      capsize=4)
        ax.axhline(0.5, color='gray', ls='--', lw=1)
        ax.set_xticks(range(len(cond_names)))
        ax.set_xticklabels(cond_names, fontsize=7, rotation=35)
        ax.set_ylabel('AUC' if dx == DX_LIST[0] else '', fontsize=11)
        ax.set_title(f'{dx}', fontsize=13)
        ax.set_ylim(0.35, 0.75)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=7)
    plt.suptitle('Quantitative Analysis: AUC by Feature Condition '
                 'and Diagnosis', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(f'{FIGURE_DIR}/fig7_E4_classification_auc.pdf',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: fig7_E4_classification_auc.pdf")
    # Figure E5: Paired ΔAUC for SUD and ADHD
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row, dx in enumerate(['SUD', 'ADHD']):
        for col, (c1, c2) in enumerate([('T+D', 'T-only'),
                                         ('T+D', 'D-only'),
                                         ('T+D+kappa', 'T+D')]):
            ax = axes[row, col]
            diff = all_results[dx][c1]['auc'] - all_results[dx][c2]['auc']
            ax.hist(diff, bins=25, color='steelblue', alpha=0.7,
                    edgecolor='black', density=True)
            ax.axvline(0, color='black', ls='-', lw=1.5)
            ax.axvline(np.mean(diff), color='red', ls='--', lw=2)
            if np.std(diff) > 0:
                _, p = wilcoxon(diff)
            else:
                p = 1.0
            ax.set_title(f'{dx}: {c1} vs {c2}\n'
                         f'ΔAUC={np.mean(diff):+.4f}, p={p:.4f}',
                         fontsize=10)
            ax.set_xlabel('ΔAUC', fontsize=9)
            if col == 0:
                ax.set_ylabel('Density', fontsize=9)
    plt.suptitle('Paired ΔAUC Distributions '
                 f'({N_SPLITS * N_REPEATS} folds)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(f'{FIGURE_DIR}/fig7_E5_paired_auc_detail.pdf',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig7_E5_paired_auc_detail.pdf")
    # ==============================================================
    # 5. OBSERVATION SUMMARY AND NEXT QUESTION
    # ==============================================================
    print("\n" + "=" * 60)
    print("5. OBSERVATION SUMMARY AND NEXT QUESTION")
    print("=" * 60)
    print("\nBest condition per diagnosis:")
    print(f"{'Dx':<8}", end='')
    for c in cond_names:
        print(f" {c:>12}", end='')
    print(f" {'Best':>12}")
    for dx in DX_LIST:
        print(f"{dx:<8}", end='')
        means = {}
        for c in cond_names:
            m = all_results[dx][c]['auc'].mean()
            means[c] = m
            print(f" {m:>12.4f}", end='')
        best = max(means, key=means.get)
        print(f" {best:>12}")
    # Computed summary. Until 2026-11 a fixed five-point narrative was printed here
    # regardless of the results (docs/v1/DEFECT_FIXES.md, D9).
    print("\nComputed summary:")
    for dx in DX_LIST:
        m = {c: all_results[dx][c]['auc'].mean() for c in cond_names}
        best = max(m, key=m.get)
        single = max(m['T-only'], m['D-only'])
        print(f"  {dx}: best condition {best} (AUC {m[best]:.3f}); "
              f"T+D {'exceeds' if m['T+D'] > single else 'does not exceed'} "
              f"the better single family ({single:.3f}); "
              f"adding kappa changes T+D by {m['T+D+kappa'] - m['T+D']:+.3f}")

    # ==============================================================
    # SUMMARY TABLE
    # ==============================================================
    print("=" * 60)
    print("TABLE 7.E1: Augmentation Ablation Results (AUC ± SD)")
    print("=" * 60)
    print(f"{'Dx':<8} {'T-only':>12} {'D-only':>12} "
          f"{'T+D':>12} {'T+D+κ':>12}")
    for dx in DX_LIST:
        vals = []
        for c in cond_names:
            r = all_results[dx][c]
            vals.append(f"{r['auc'].mean():.3f}±{r['auc'].std():.3f}")
        print(f"{dx:<8} {vals[0]:>12} {vals[1]:>12} "
              f"{vals[2]:>12} {vals[3]:>12}")
    print(f"\nCompleted in {time.time() - t0:.1f}s")
if __name__ == '__main__':
    main()
