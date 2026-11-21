#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
ARSPI-Net — Script 03: Chapter 6 Experiments (Dynamical Characterization)
═══════════════════════════════════════════════════════════════════════

SCIENTIFIC QUESTION
-------------------
What are the dynamical properties of the reservoir's internal trajectory
when driven by affective EEG, and do those properties vary with emotional
condition and clinical status?

Chapter 6 opens Stage 1 and characterizes the driven trajectory as an
independent source of interpretable descriptors. The 7 experiments
progress deductively: first establish condition sensitivity (EXP-6.1),
decompose which metric family carries the signal (EXP-6.2), survey
transdiagnostic clinical associations (EXP-6.3–6.4), characterize the
energy-information tradeoff (EXP-6.5), test the specific HC vs MDD
predictions from the chapter specification (EXP-6.6), and quantify
discriminative value (EXP-6.7).

Each experiment follows the five-step cycle: mathematical motivation,
experimental design, raw observation, analysis, and the next question
that the observation motivates.

INPUT
-----
  ch6_ch7_3class_features.pkl (from Script 01)
  clinical_profile.csv

OUTPUT
------
  ch6_results.pkl — all numerical results
  7 PDF figures in ./ch6_figures/
  Terminal output with quantitative observations and statistics

RUN COMMAND
-----------
  python ch6_03_experiments.py
"""

import numpy as np
import pickle
import os
import time
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu, spearmanr, friedmanchisquare
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════
INPUT_FILE = './ch6_ch7_3class_features.pkl'
CLINICAL_FILE = './clinical_profile.csv'
RESULTS_FILE = './ch6_results.pkl'
FIG_DIR = './ch6_figures'
os.makedirs(FIG_DIR, exist_ok=True)

METRIC_NAMES = ['total_spikes', 'MFR', 'rate_entropy', 'rate_variance',
                'temporal_sparsity', 'perm_entropy', 'tau_AC']
EXTRA_NAMES = ['CLZ', 'lambda_proxy', 'tau_relax', 'T_RTB']
COND_NAMES_MAP = {0: 'Negative', 1: 'Neutral', 2: 'Pleasant'}
COND_COLORS = ['#d63031', '#636e72', '#00b894']

AMPLITUDE_METRICS = [0, 1, 2, 3]
TEMPORAL_METRICS = [5, 6]
SPARSITY_METRICS = [4]

RANDOM_STATE = 42

# ══════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("ARSPI-NET — CHAPTER 6 EXPERIMENTS (DYNAMICAL CHARACTERIZATION)")
print("=" * 70)

with open(INPUT_FILE, 'rb') as f:
    data = pickle.load(f)

D = data['D']
D_extra = data['D_extra']
pop_rate_ts = data['pop_rate_ts']
y = data['y']
subjects = data['subjects']
COND_NAMES = data['cond_names']

N_obs, N_ch, _ = D.shape
unique_subjects = np.unique(subjects)
N_subj = len(unique_subjects)

print(f"\nData: {N_obs} obs, {N_subj} subjects, {N_ch} channels, 3 classes")
print(f"Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

import pandas as pd
try:
    df_clin = pd.read_csv(CLINICAL_FILE)
    df_clin = df_clin.drop_duplicates(subset='ID', keep='first')
    clin_ids = set(df_clin['ID'].values) & set(unique_subjects)
    print(f"Clinical metadata: {len(clin_ids)} subjects")
    HAS_CLINICAL = True
except Exception as e:
    print(f"WARNING: Clinical metadata not found ({e}). EXP 6.3–6.6 skipped.")
    HAS_CLINICAL = False

D_avg = D.mean(axis=1)
D_extra_avg = D_extra.mean(axis=1)

all_results = {}


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    var1, var2 = g1.var(ddof=1), g2.var(ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (g1.mean() - g2.mean()) / pooled if pooled > 1e-12 else 0.0


def paired_dz(x1, x2):
    diff = x1 - x2
    return diff.mean() / diff.std() if diff.std() > 1e-12 else 0.0


def get_scm(metric_idx, is_extra=False):
    """Subject × condition means. Returns (N_subj, 3)."""
    arr = D_extra_avg if is_extra else D_avg
    result = np.zeros((N_subj, 3))
    for si, sid in enumerate(unique_subjects):
        for c in range(3):
            mask = (subjects == sid) & (y == c)
            if mask.sum() > 0:
                result[si, c] = arr[mask, metric_idx].mean()
    return result


def get_clinical_groups(var_name):
    if not HAS_CLINICAL or var_name not in df_clin.columns:
        return None, None, 0, 0
    if var_name == 'Assigned_Sex':
        yes_ids = set(df_clin[df_clin[var_name] == 1]['ID'].values)
        no_ids = set(df_clin[df_clin[var_name] == 2]['ID'].values)
    else:
        yes_ids = set(df_clin[df_clin[var_name] == 1]['ID'].values)
        no_ids = set(df_clin[df_clin[var_name] == 0]['ID'].values)
    yes_s = np.array([s for s in unique_subjects if s in yes_ids])
    no_s = np.array([s for s in unique_subjects if s in no_ids])
    return yes_s, no_s, len(yes_s), len(no_s)


def subject_metric_mean(subs, metric_idx, is_extra=False):
    """Per-subject grand mean (averaged across conditions and channels)."""
    arr = D_extra_avg if is_extra else D_avg
    return np.array([arr[subjects == sid, metric_idx].mean() for sid in subs])


# ══════════════════════════════════════════════════════════════════
# EXP-6.1: CONDITION EFFECTS ON DYNAMICAL METRICS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXP-6.1: CONDITION EFFECTS ON DYNAMICAL METRICS")
print("=" * 70)
print(f"\n  Mathematical motivation: The reservoir processes different emotional")
print(f"  inputs. The data processing inequality constrains what information")
print(f"  the driven trajectory can carry. This experiment measures whether")
print(f"  the 7 core metrics differ across conditions.\n")

# ── Step 3: Raw observation ──
print(f"  Raw observation (per-subject, channel-averaged means ± SEM):\n")
print(f"  {'Metric':20s}  {'Negative':>12s}  {'Neutral':>12s}  {'Pleasant':>12s}")
print(f"  {'─' * 60}")

exp61 = {}
for k, name in enumerate(METRIC_NAMES):
    scm = get_scm(k)
    means = [scm[:, c].mean() for c in range(3)]
    sems = [scm[:, c].std() / np.sqrt(N_subj) for c in range(3)]
    print(f"  {name:20s}  {means[0]:>8.4f}±{sems[0]:.4f}  "
          f"{means[1]:>8.4f}±{sems[1]:.4f}  {means[2]:>8.4f}±{sems[2]:.4f}")

# ── Step 4: Analysis ──
print(f"\n  Analysis (Friedman test + paired Wilcoxon follow-ups):\n")
for k, name in enumerate(METRIC_NAMES):
    scm = get_scm(k)
    try:
        stat, p_fried = friedmanchisquare(scm[:, 0], scm[:, 1], scm[:, 2])
    except:
        stat, p_fried = 0, 1.0

    pairs = [(0, 1, 'Neg-Neu'), (0, 2, 'Neg-Pos'), (1, 2, 'Neu-Pos')]
    pair_results = {}
    for c1, c2, label in pairs:
        try:
            _, p = wilcoxon(scm[:, c1], scm[:, c2])
        except:
            p = 1.0
        d_z = paired_dz(scm[:, c1], scm[:, c2])
        pair_results[label] = {'p': p, 'd_z': d_z}

    exp61[name] = {'friedman_p': p_fried,
                   'means': {COND_NAMES[c]: float(scm[:, c].mean()) for c in range(3)},
                   'pairs': pair_results}

    sig = '*' if p_fried < 0.05 else ' '
    print(f"  {name:20s}: Friedman p={p_fried:.4f}{sig}")
    for label, res in pair_results.items():
        sig2 = '*' if res['p'] < 0.05 else ' '
        print(f"    {label:8s}: d_z={res['d_z']:+.3f}, p={res['p']:.4f}{sig2}")

all_results['exp61'] = exp61

# Figure
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
for k in range(7):
    ax = axes[k]
    scm = get_scm(k)
    means = [scm[:, c].mean() for c in range(3)]
    sems = [scm[:, c].std() / np.sqrt(N_subj) for c in range(3)]
    ax.bar([0, 1, 2], means, yerr=sems, color=COND_COLORS, capsize=5, edgecolor='white')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Neg', 'Neu', 'Pos'])
    p = exp61[METRIC_NAMES[k]]['friedman_p']
    ax.set_title(f'{METRIC_NAMES[k]}\n(Friedman p={p:.3f})', fontsize=10, fontweight='bold')
axes[7].axis('off')
fig.suptitle('EXP-6.1: Condition Effects on Dynamical Metrics', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{FIG_DIR}/fig01_condition_effects.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"\n  → fig01_condition_effects.pdf")


# ══════════════════════════════════════════════════════════════════
# EXP-6.2: METRIC FAMILY DECOMPOSITION
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXP-6.2: METRIC FAMILY DECOMPOSITION (AMPLITUDE vs TEMPORAL)")
print("=" * 70)
print(f"\n  Mathematical motivation: The LIF reservoir's distinctive")
print(f"  contribution is temporal structure encoding. This experiment")
print(f"  decomposes the Neg–Pos contrast into amplitude-tracking and")
print(f"  temporal-structure families.\n")

# ── Step 3: Raw observation ──
print(f"  Raw observation — Neg−Pos paired differences (d_z):\n")
print(f"  {'Metric':20s}  {'d_z':>8s}  {'p':>8s}  Family")
print(f"  {'─' * 55}")

exp62 = {'amplitude': {}, 'temporal': {}, 'sparsity': {}}
for k, name in enumerate(METRIC_NAMES):
    scm = get_scm(k)
    d_z = paired_dz(scm[:, 0], scm[:, 2])
    try:
        _, p = wilcoxon(scm[:, 0], scm[:, 2])
    except:
        p = 1.0

    if k in AMPLITUDE_METRICS:
        family = 'amplitude'
    elif k in TEMPORAL_METRICS:
        family = 'temporal'
    else:
        family = 'sparsity'

    exp62[family][name] = {'d_z': d_z, 'p': p}
    sig = '*' if p < 0.05 else ' '
    print(f"  {name:20s}  {d_z:+8.4f}  {p:8.4f}{sig}  {family}")

# ── Step 4: Analysis ──
print(f"\n  Analysis — aggregate effect sizes by family:")
for fam in ['amplitude', 'temporal', 'sparsity']:
    if exp62[fam]:
        dz_vals = [abs(v['d_z']) for v in exp62[fam].values()]
        print(f"    {fam.upper():12s}: mean |d_z| = {np.mean(dz_vals):.4f} "
              f"(n={len(dz_vals)} metrics)")

all_results['exp62'] = exp62

# Figure
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
fam_colors = {'amplitude': '#74b9ff', 'temporal': '#e17055', 'sparsity': '#ffeaa7'}
names_plot, dz_plot, col_plot = [], [], []
for k, name in enumerate(METRIC_NAMES):
    scm = get_scm(k)
    d_z = paired_dz(scm[:, 0], scm[:, 2])
    names_plot.append(name)
    dz_plot.append(d_z)
    fam = 'amplitude' if k in AMPLITUDE_METRICS else ('temporal' if k in TEMPORAL_METRICS else 'sparsity')
    col_plot.append(fam_colors[fam])
ax.bar(range(len(names_plot)), dz_plot, color=col_plot, edgecolor='white')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(range(len(names_plot)))
ax.set_xticklabels([n[:12] for n in names_plot], rotation=45, ha='right')
ax.set_ylabel("Cohen's d_z (Neg − Pos)")
ax.set_title('EXP-6.2: Metric Family Decomposition', fontsize=13, fontweight='bold')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=v, label=k.capitalize()) for k, v in fam_colors.items()], loc='upper right')
fig.tight_layout()
fig.savefig(f'{FIG_DIR}/fig02_metric_families.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"  → fig02_metric_families.pdf")


# ══════════════════════════════════════════════════════════════════
# EXP-6.3: TRANSDIAGNOSTIC CLINICAL COMPARISONS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXP-6.3: TRANSDIAGNOSTIC CLINICAL COMPARISONS")
print("=" * 70)

if not HAS_CLINICAL:
    print("  SKIPPED: No clinical metadata.")
else:
    print(f"\n  Mathematical motivation: Chapter 5 established disorder-specific")
    print(f"  graph-topological signatures. The dynamical metrics measure a")
    print(f"  different property — temporal trajectory behavior. This experiment")
    print(f"  tests whether temporal dynamics also carry clinical information.\n")

    clinical_vars = {'MDD': 'Major Depressive Disorder', 'PTSD': 'PTSD',
                     'GAD': 'Generalized Anxiety', 'SUD': 'Substance Use Disorder',
                     'ADHD': 'ADHD'}
    if 'Psychiatric_Medication' in df_clin.columns:
        clinical_vars['Psychiatric_Medication'] = 'Medication Status'
    if 'Assigned_Sex' in df_clin.columns:
        clinical_vars['Assigned_Sex'] = 'Biological Sex'

    exp63 = {}
    n_sig_total = 0

    for var, label in clinical_vars.items():
        yes_s, no_s, n_yes, n_no = get_clinical_groups(var)
        if yes_s is None or n_yes < 10 or n_no < 10:
            continue

        print(f"  {label} (n={n_no} vs {n_yes}):")

        # ── Step 3: Raw observation — group means BEFORE any tests ──
        print(f"    {'Metric':20s}  {'No':>10s}  {'Yes':>10s}  {'Δ':>10s}")
        print(f"    {'─' * 55}")

        exp63[var] = {'n_no': n_no, 'n_yes': n_yes, 'metrics': {}}
        n_sig = 0

        for k, name in enumerate(METRIC_NAMES):
            yes_vals = subject_metric_mean(yes_s, k)
            no_vals = subject_metric_mean(no_s, k)

            # Raw observation: just the means
            delta = yes_vals.mean() - no_vals.mean()
            print(f"    {name:20s}  {no_vals.mean():>10.4f}  "
                  f"{yes_vals.mean():>10.4f}  {delta:>+10.4f}")

            # Step 4: Analysis
            try:
                _, p = mannwhitneyu(no_vals, yes_vals, alternative='two-sided')
            except:
                p = 1.0
            d = cohens_d(yes_vals, no_vals)
            exp63[var]['metrics'][name] = {'d': d, 'p': p,
                                           'mean_yes': float(yes_vals.mean()),
                                           'mean_no': float(no_vals.mean())}
            if p < 0.05:
                n_sig += 1
                n_sig_total += 1

        # Print significant results after showing all raw data
        sig_metrics = [(n, r) for n, r in exp63[var]['metrics'].items() if r['p'] < 0.05]
        if sig_metrics:
            print(f"    Significant (p<0.05):")
            for n, r in sig_metrics:
                print(f"      {n:20s}: d={r['d']:+.3f}, p={r['p']:.4f}")
        else:
            print(f"    (no significant metrics at p<0.05)")
        print()

    print(f"  Total significant metric-diagnosis pairs: {n_sig_total}")
    all_results['exp63'] = exp63

    # Figure: heatmap
    diag_labels = [v for k, v in clinical_vars.items() if k in exp63]
    diag_keys = [k for k in clinical_vars if k in exp63]
    n_diag = len(diag_keys)

    if n_diag > 0:
        effect_matrix = np.zeros((n_diag, 7))
        pval_matrix = np.ones((n_diag, 7))
        for di, dk in enumerate(diag_keys):
            for k, name in enumerate(METRIC_NAMES):
                if name in exp63[dk]['metrics']:
                    effect_matrix[di, k] = exp63[dk]['metrics'][name]['d']
                    pval_matrix[di, k] = exp63[dk]['metrics'][name]['p']

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        vmax = max(0.5, np.abs(effect_matrix).max())
        im = ax.imshow(effect_matrix, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(7))
        ax.set_xticklabels([n[:12] for n in METRIC_NAMES], rotation=45, ha='right')
        ax.set_yticks(range(n_diag))
        ax.set_yticklabels([l[:20] for l in diag_labels])
        for di in range(n_diag):
            for k in range(7):
                if pval_matrix[di, k] < 0.05:
                    ax.text(k, di, '*', ha='center', va='center', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label="Cohen's d")
        ax.set_title('EXP-6.3: Clinical Effects on Dynamical Metrics', fontsize=12, fontweight='bold')
        fig.tight_layout()
        fig.savefig(f'{FIG_DIR}/fig03_clinical_heatmap.pdf', bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  → fig03_clinical_heatmap.pdf")


# ══════════════════════════════════════════════════════════════════
# EXP-6.4: CONDITION × CLINICAL INTERACTIONS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXP-6.4: CONDITION × CLINICAL INTERACTIONS")
print("=" * 70)

if not HAS_CLINICAL:
    print("  SKIPPED: No clinical metadata.")
else:
    print(f"\n  Mathematical motivation: Chapter 5's strongest clinical finding")
    print(f"  is a condition × SUD interaction (p=0.0004). This experiment tests")
    print(f"  the dynamical analog: does the change in reservoir dynamics between")
    print(f"  conditions differ by diagnosis?\n")

    interaction_vars = ['MDD', 'PTSD', 'GAD', 'SUD']
    if 'Psychiatric_Medication' in df_clin.columns:
        interaction_vars.append('Psychiatric_Medication')

    exp64 = {}
    n_sig = 0

    for var in interaction_vars:
        yes_s, no_s, n_yes, n_no = get_clinical_groups(var)
        if yes_s is None or n_yes < 15 or n_no < 15:
            continue

        label = clinical_vars.get(var, var)

        # ── Step 3: Raw observation — reactivity values ──
        print(f"  {label} — Reactivity (Neg − Pos):")
        print(f"    {'Metric':20s}  {'No group':>10s}  {'Yes group':>10s}  {'Δ':>10s}")
        print(f"    {'─' * 55}")

        exp64[var] = {}

        for k, name in enumerate(METRIC_NAMES):
            react_yes, react_no = [], []
            for sid in yes_s:
                neg_m = (subjects == sid) & (y == 0)
                pos_m = (subjects == sid) & (y == 2)
                if neg_m.sum() > 0 and pos_m.sum() > 0:
                    react_yes.append(D_avg[neg_m, k].mean() - D_avg[pos_m, k].mean())
            for sid in no_s:
                neg_m = (subjects == sid) & (y == 0)
                pos_m = (subjects == sid) & (y == 2)
                if neg_m.sum() > 0 and pos_m.sum() > 0:
                    react_no.append(D_avg[neg_m, k].mean() - D_avg[pos_m, k].mean())

            r_yes = np.array(react_yes)
            r_no = np.array(react_no)

            # Raw observation first
            print(f"    {name:20s}  {r_no.mean():>+10.4f}  {r_yes.mean():>+10.4f}  "
                  f"{r_yes.mean()-r_no.mean():>+10.4f}")

            # Step 4: Analysis
            try:
                _, p = mannwhitneyu(r_no, r_yes, alternative='two-sided')
            except:
                p = 1.0
            d = cohens_d(r_yes, r_no)
            exp64[var][name] = {'d': d, 'p': p,
                                'react_yes': float(r_yes.mean()),
                                'react_no': float(r_no.mean())}
            if p < 0.05:
                n_sig += 1

        # Print significant after raw data
        sig_int = [(n, r) for n, r in exp64[var].items() if isinstance(r, dict) and r.get('p', 1) < 0.05]
        if sig_int:
            print(f"    Significant (p<0.05):")
            for n, r in sig_int:
                print(f"      {n:20s}: d={r['d']:+.3f}, p={r['p']:.4f}")
        print()

    if n_sig == 0:
        print("  No significant interactions at p<0.05")
    else:
        print(f"  Total significant interactions: {n_sig}")

    all_results['exp64'] = exp64

    # Figure
    sig_interactions = [(v, n, r) for v in exp64 for n, r in exp64[v].items()
                        if isinstance(r, dict) and r.get('p', 1) < 0.05]
    n_panels = max(1, min(len(sig_interactions), 6))
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]
    if sig_interactions:
        for pi, (var, name, res) in enumerate(sig_interactions[:n_panels]):
            ax = axes[pi]
            ax.bar([0, 1], [res['react_no'], res['react_yes']],
                   color=['#74b9ff', '#e17055'], edgecolor='white')
            ax.set_xticks([0, 1])
            ax.set_xticklabels([f'No {var}', f'Yes {var}'], fontsize=9)
            ax.set_ylabel('Reactivity (Neg − Pos)')
            ax.set_title(f'{name}\np={res["p"]:.4f}', fontsize=10, fontweight='bold')
            ax.axhline(0, color='black', linewidth=0.5)
    else:
        axes[0].text(0.5, 0.5, 'No significant interactions', ha='center', va='center',
                     fontsize=14, transform=axes[0].transAxes)
    fig.suptitle('EXP-6.4: Condition × Clinical Interactions', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/fig04_interactions.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  → fig04_interactions.pdf")


# ══════════════════════════════════════════════════════════════════
# EXP-6.5: SPARSE CODING EFFICIENCY (Φ)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXP-6.5: SPARSE CODING EFFICIENCY (Φ)")
print("=" * 70)
print(f"\n  Mathematical motivation: Φ = I_decoded / SynOps measures bits")
print(f"  of stimulus information per synaptic operation — the energy-")
print(f"  information tradeoff of the neuromorphic representation.\n")

K = 3
P_e = 1.0 - 0.794
H_b = -P_e * np.log2(P_e + 1e-12) - (1 - P_e) * np.log2(1 - P_e + 1e-12)
# Fano: I >= log2(K) - H_b(P_e) - P_e * log2(K - 1). The last term is zero only for
# K = 2 and was missing until 2026-11 (docs/v1/DEFECT_FIXES.md, D6). It lowers the bound
# by a constant, so every comparison of Phi between groups is unchanged.
I_decoded = np.log2(K) - H_b - P_e * np.log2(K - 1)
print(f"  I_decoded ≥ {I_decoded:.4f} bits (from 79.4% accuracy, Fano bound)")

total_spikes_obs = D[:, :, 0].mean(axis=1)
Phi = I_decoded / (total_spikes_obs + 1e-8)

# ── Step 3: Raw observation ──
print(f"\n  Raw observation — Φ by condition:\n")
exp65 = {}
for c in range(3):
    phi_c = Phi[y == c]
    exp65[COND_NAMES[c]] = {'mean': float(phi_c.mean()), 'std': float(phi_c.std())}
    print(f"    {COND_NAMES[c]:10s}: Φ = {phi_c.mean():.6f} ± {phi_c.std():.6f} bits/spike")

# ── Step 4: Analysis ──
scm_phi = np.zeros((N_subj, 3))
for si, sid in enumerate(unique_subjects):
    for c in range(3):
        mask = (subjects == sid) & (y == c)
        if mask.sum() > 0:
            scm_phi[si, c] = Phi[mask].mean()
try:
    _, p_phi = wilcoxon(scm_phi[:, 0], scm_phi[:, 2])
except:
    p_phi = 1.0
d_phi = paired_dz(scm_phi[:, 0], scm_phi[:, 2])
print(f"\n  Analysis — Neg vs Pos: d_z={d_phi:+.4f}, p={p_phi:.4f}")
exp65['neg_pos_test'] = {'d_z': d_phi, 'p': p_phi}

if HAS_CLINICAL:
    print(f"\n  Clinical Φ (raw observation then test):")
    for var in ['MDD', 'PTSD', 'SUD', 'GAD']:
        yes_s, no_s, n_yes, n_no = get_clinical_groups(var)
        if yes_s is None or n_yes < 10:
            continue
        phi_yes = np.array([Phi[subjects == s].mean() for s in yes_s])
        phi_no = np.array([Phi[subjects == s].mean() for s in no_s])
        print(f"    {var:5s}: No={phi_no.mean():.6f}, Yes={phi_yes.mean():.6f}, "
              f"Δ={phi_yes.mean()-phi_no.mean():+.6f}")
        try:
            _, p = mannwhitneyu(phi_no, phi_yes, alternative='two-sided')
        except:
            p = 1.0
        d = cohens_d(phi_yes, phi_no)
        sig = '*' if p < 0.05 else ' '
        print(f"          d={d:+.3f}, p={p:.4f}{sig}")
        exp65[f'clinical_{var}'] = {'d': d, 'p': p}

    # HC vs all clinical
    primary = [c for c in ['MDD', 'PTSD', 'GAD', 'SUD', 'ADHD'] if c in df_clin.columns]
    df_eeg = df_clin[df_clin['ID'].isin(unique_subjects)]
    hc_mask = df_eeg[primary].apply(lambda r: r.dropna().astype(int).sum() == 0, axis=1)
    hc_ids = set(df_eeg[hc_mask]['ID'].values)
    clin_all = set(df_eeg[~hc_mask]['ID'].values)
    phi_hc = np.array([Phi[subjects == s].mean() for s in unique_subjects if s in hc_ids])
    phi_cl = np.array([Phi[subjects == s].mean() for s in unique_subjects if s in clin_all])
    if len(phi_hc) >= 5:
        print(f"\n    HC (n={len(phi_hc)}): {phi_hc.mean():.6f}")
        print(f"    Clinical (n={len(phi_cl)}): {phi_cl.mean():.6f}")
        try:
            _, p = mannwhitneyu(phi_hc, phi_cl, alternative='two-sided')
        except:
            p = 1.0
        d = cohens_d(phi_hc, phi_cl)
        print(f"    d={d:+.3f}, p={p:.4f}")
        exp65['hc_vs_clinical'] = {'d': d, 'p': p, 'n_hc': len(phi_hc)}

all_results['exp65'] = exp65

# Figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for c in range(3):
    axes[0].hist(Phi[y == c], bins=30, alpha=0.5, color=COND_COLORS[c], label=COND_NAMES[c])
axes[0].set_xlabel('Φ (bits/spike)')
axes[0].set_ylabel('Count')
axes[0].set_title('Φ Distribution by Condition', fontweight='bold')
axes[0].legend()
axes[1].bar([0, 1, 2], [exp65[COND_NAMES[c]]['mean'] for c in range(3)],
            yerr=[exp65[COND_NAMES[c]]['std'] / np.sqrt(211) for c in range(3)],
            color=COND_COLORS, capsize=5)
axes[1].set_xticks([0, 1, 2])
axes[1].set_xticklabels(['Neg', 'Neu', 'Pos'])
axes[1].set_ylabel('Φ (bits/spike)')
axes[1].set_title(f'Mean Φ ± SEM (Neg-Pos p={p_phi:.4f})', fontweight='bold')
fig.suptitle('EXP-6.5: Sparse Coding Efficiency', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{FIG_DIR}/fig05_sparse_coding.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"  → fig05_sparse_coding.pdf")


# ══════════════════════════════════════════════════════════════════
# EXP-6.6: HC vs MDD HYPOTHESIS TEST
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXP-6.6: HC vs MDD HYPOTHESIS TEST")
print("=" * 70)

if not HAS_CLINICAL:
    print("  SKIPPED: No clinical metadata.")
else:
    print(f"\n  Mathematical motivation: Chapter 6 predicts H1 (Φ_HC > Φ_MDD),")
    print(f"  H2 (Λ_HC > Λ_MDD), H3 (τ_MDD > τ_HC).\n")

    primary = [c for c in ['MDD', 'PTSD', 'GAD', 'SUD', 'ADHD'] if c in df_clin.columns]
    df_eeg = df_clin[df_clin['ID'].isin(unique_subjects)]
    hc_mask = df_eeg[primary].apply(lambda r: r.dropna().astype(int).sum() == 0, axis=1)
    hc_ids = set(df_eeg[hc_mask]['ID'].values)
    mdd_ids = set(df_eeg[df_eeg['MDD'] == 1]['ID'].values) if 'MDD' in df_eeg.columns else set()
    hc_subs = np.array([s for s in unique_subjects if s in hc_ids])
    mdd_subs = np.array([s for s in unique_subjects if s in mdd_ids])

    print(f"  Groups: HC n={len(hc_subs)}, MDD n={len(mdd_subs)}")
    exp66 = {'n_hc': len(hc_subs), 'n_mdd': len(mdd_subs), 'tests': {}}

    if len(hc_subs) >= 5 and len(mdd_subs) >= 5:
        tests = [
            ('H1_Phi', 'Φ', lambda s: np.array([Phi[subjects == sid].mean() for sid in s]), 'greater'),
            ('H2_perm_entropy', 'Ĥ_π', lambda s: subject_metric_mean(s, 5), 'greater'),
            ('H2_CLZ', 'C_LZ', lambda s: subject_metric_mean(s, 0, is_extra=True), 'greater'),
            ('H3_tau_relax', 'τ_relax', lambda s: subject_metric_mean(s, 2, is_extra=True), 'less'),
            ('H3_tau_AC', 'τ_AC', lambda s: subject_metric_mean(s, 6), 'less'),
        ]

        # ── Step 3: Raw observation ──
        print(f"\n  Raw observation — group means:")
        print(f"    {'Test':20s}  {'HC':>12s}  {'MDD':>12s}  {'Δ':>10s}")
        print(f"    {'─' * 58}")

        for key, label, fn, alt in tests:
            hc_v = fn(hc_subs)
            mdd_v = fn(mdd_subs)
            print(f"    {label:20s}  {hc_v.mean():>12.4f}  {mdd_v.mean():>12.4f}  "
                  f"{hc_v.mean()-mdd_v.mean():>+10.4f}")

        # ── Step 4: Analysis ──
        print(f"\n  Analysis — directional tests:")
        for key, label, fn, alt in tests:
            hc_v = fn(hc_subs)
            mdd_v = fn(mdd_subs)
            if alt == 'greater':
                try:
                    _, p = mannwhitneyu(hc_v, mdd_v, alternative='greater')
                except:
                    p = 1.0
                d = cohens_d(hc_v, mdd_v)
                hyp = f'{label}_HC > {label}_MDD'
            else:
                try:
                    _, p = mannwhitneyu(mdd_v, hc_v, alternative='greater')
                except:
                    p = 1.0
                d = cohens_d(mdd_v, hc_v)
                hyp = f'{label}_MDD > {label}_HC'

            exp66['tests'][key] = {'d': d, 'p': p}
            confirmed = '✓' if p < 0.05 else '✗'
            print(f"    {hyp:30s}: d={d:+.3f}, p={p:.4f} {confirmed}")
    else:
        print(f"  Insufficient HC subjects (n={len(hc_subs)}).")

    all_results['exp66'] = exp66

    # Figure
    if len(hc_subs) >= 5:
        test_metrics = [('perm_entropy', 5, False), ('tau_AC', 6, False),
                        ('CLZ', 0, True), ('tau_relax', 2, True)]
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for pi, (name, idx, is_extra) in enumerate(test_metrics):
            ax = axes[pi]
            hc_v = subject_metric_mean(hc_subs, idx, is_extra)
            mdd_v = subject_metric_mean(mdd_subs, idx, is_extra)
            parts = ax.violinplot([hc_v, mdd_v], positions=[0, 1], showmeans=True)
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(['#00b894', '#d63031'][i])
                pc.set_alpha(0.6)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([f'HC\n(n={len(hc_subs)})', f'MDD\n(n={len(mdd_subs)})'])
            ax.set_title(name, fontsize=11, fontweight='bold')
        fig.suptitle('EXP-6.6: HC vs MDD Hypothesis Tests', fontsize=13, fontweight='bold')
        fig.tight_layout()
        fig.savefig(f'{FIG_DIR}/fig06_hc_vs_mdd.pdf', bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  → fig06_hc_vs_mdd.pdf")


# ══════════════════════════════════════════════════════════════════
# EXP-6.7: DYNAMICAL METRIC DISCRIMINATIVE VALUE
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXP-6.7: DYNAMICAL METRIC DISCRIMINATIVE VALUE")
print("=" * 70)
print(f"\n  Mathematical motivation: The dynamical metrics are designed as")
print(f"  interpretable descriptors. This experiment measures whether they")
print(f"  also carry discriminative information — above-chance accuracy on")
print(f"  emotion classification and clinical detection tasks.\n")

exp67 = {}

X_dyn = D_avg.copy()
X_dyn_extra = np.hstack([D_avg, D_extra_avg])
X_dyn_ch = D.reshape(N_obs, -1)

cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

def classify(X, y_in, subjects_in, label):
    accs = []
    for tr, te in cv.split(X, y_in, groups=subjects_in):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        clf = SVC(kernel='rbf', class_weight='balanced', random_state=RANDOM_STATE)
        clf.fit(Xtr, y_in[tr])
        accs.append(balanced_accuracy_score(y_in[te], clf.predict(Xte)))
    accs = np.array(accs)
    print(f"    {label:40s}: {accs.mean()*100:.1f}% ± {accs.std()*100:.1f}")
    return accs

print("  3-class emotion classification (chance = 33.3%):")
acc_7 = classify(X_dyn, y, subjects, "7 core metrics (channel-avg)")
acc_11 = classify(X_dyn_extra, y, subjects, "11 metrics (core+extra)")
acc_ch = classify(X_dyn_ch, y, subjects, "7 metrics × 34 channels (238d)")

exp67['emotion_3class'] = {
    '7_core': float(acc_7.mean()),
    '11_all': float(acc_11.mean()),
    '238_perchannel': float(acc_ch.mean()),
}

if HAS_CLINICAL:
    print(f"\n  Binary clinical detection (chance = 50%):")
    for var in ['SUD', 'MDD', 'PTSD', 'GAD']:
        yes_s, no_s, n_yes, n_no = get_clinical_groups(var)
        if yes_s is None or n_yes < 15 or n_no < 15:
            continue
        X_c, y_c, s_c = [], [], []
        for sid in unique_subjects:
            feat = D_avg[subjects == sid].mean(axis=0)
            if sid in set(yes_s):
                X_c.append(feat); y_c.append(1); s_c.append(sid)
            elif sid in set(no_s):
                X_c.append(feat); y_c.append(0); s_c.append(sid)
        X_c, y_c, s_c = np.array(X_c), np.array(y_c), np.array(s_c)
        if len(X_c) < 30:
            continue
        cv2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        accs = []
        for tr, te in cv2.split(X_c, y_c, groups=s_c):
            sc = StandardScaler()
            clf = SVC(kernel='rbf', class_weight='balanced', random_state=RANDOM_STATE)
            clf.fit(sc.fit_transform(X_c[tr]), y_c[tr])
            accs.append(balanced_accuracy_score(y_c[te], clf.predict(sc.transform(X_c[te]))))
        accs = np.array(accs)
        print(f"    {var:5s} (n={n_no}vs{n_yes}): {accs.mean()*100:.1f}% ± {accs.std()*100:.1f}")
        exp67[f'clinical_{var}'] = float(accs.mean())

all_results['exp67'] = exp67

# Figure
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
labels = ['7 core\n(ch-avg)', '11 all\n(ch-avg)', '238d\n(per-ch)']
vals = [exp67['emotion_3class'][k] * 100 for k in ['7_core', '11_all', '238_perchannel']]
ax.bar(range(3), vals, color=['#74b9ff', '#0984e3', '#2d3436'], edgecolor='white')
ax.axhline(33.3, color='red', linestyle='--', label='Chance (33.3%)')
ax.set_xticks(range(3))
ax.set_xticklabels(labels)
ax.set_ylabel('Balanced Accuracy (%)')
ax.set_title('EXP-6.7: 3-Class Emotion from Dynamical Metrics', fontsize=12, fontweight='bold')
ax.legend()
fig.tight_layout()
fig.savefig(f'{FIG_DIR}/fig07_classification.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"  → fig07_classification.pdf")


# ══════════════════════════════════════════════════════════════════
# SAVE & SUMMARY
# ══════════════════════════════════════════════════════════════════
with open(RESULTS_FILE, 'wb') as f:
    pickle.dump(all_results, f, protocol=4)

print(f"\n{'=' * 70}")
print("COMPLETE EXPERIMENTAL SUMMARY")
print(f"{'=' * 70}")
print(f"\n  Dataset: {N_subj} subjects × 3 conditions = {N_obs} observations")

n_sig_cond = sum(1 for v in exp61.values() if v['friedman_p'] < 0.05)
print(f"\n  EXP-6.1: {n_sig_cond}/7 metrics condition-sensitive (Friedman p<0.05)")

for fam in ['amplitude', 'temporal']:
    if exp62.get(fam):
        dz = [abs(v['d_z']) for v in exp62[fam].values()]
        print(f"  EXP-6.2: {fam} family mean |d_z| = {np.mean(dz):.4f}")

if HAS_CLINICAL:
    print(f"  EXP-6.3: {n_sig_total} significant metric-diagnosis pairs")
    n_int = sum(1 for v in exp64.values() for r in v.values()
                if isinstance(r, dict) and r.get('p', 1) < 0.05)
    print(f"  EXP-6.4: {n_int} significant interactions")
    print(f"  EXP-6.5: Φ Neg-Pos d_z={exp65['neg_pos_test']['d_z']:+.4f}, p={exp65['neg_pos_test']['p']:.4f}")

print(f"  EXP-6.7: 3-class (7 metrics): {exp67['emotion_3class']['7_core']*100:.1f}%")
print(f"  EXP-6.7: 3-class (238d):      {exp67['emotion_3class']['238_perchannel']*100:.1f}%")

print(f"\n  Figures: {len(os.listdir(FIG_DIR))} PDFs in {FIG_DIR}/")
print(f"  Results: {RESULTS_FILE}")
print(f"\n  ── 7 EXPERIMENTS COMPLETE ──")
print("=" * 70)
