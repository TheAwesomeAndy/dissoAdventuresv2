#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
ARSPI-Net — Script 04: Chapter 7 Experiments (Structure-Function Coupling)
═══════════════════════════════════════════════════════════════════════

SCIENTIFIC QUESTION
-------------------
Are the temporal properties (reservoir dynamics) and spatial properties
(graph topology) of ARSPI-Net statistically coupled across electrodes,
and does that coupling carry information beyond either property alone?

Chapter 6 established that the dynamical metrics encode condition
information (emotional vs non-emotional) but not clinical information.
Chapter 5 established that the graph topology encodes clinical
information (SUD, PTSD, etc.) but requires subject-centering to access
condition information. This chapter measures the RELATIONSHIP between
these two information layers across the 34-electrode array.

Five experiments progress deductively: establish that coupling exists
(EXP-7.1), decompose its variance (EXP-7.2), test whether it differs
by diagnosis (EXP-7.3), test whether combining the two descriptor
families improves clinical detection (EXP-7.4), and present the
within-valence coupling findings from the existing 4-class analysis
(EXP-7.5).

Each experiment follows the five-step cycle: mathematical motivation,
experimental design, raw observation, analysis, and the next question.

INPUT
-----
  ch6_ch7_3class_features.pkl (from Script 01)
  clinical_profile.csv

OUTPUT
------
  ch7_3class_results.pkl — all numerical results
  5 PDF figures in ./ch7_figures/
  Terminal output with quantitative observations and statistics

RUN COMMAND
-----------
  python ch7_04_experiments.py
"""

import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu, spearmanr, friedmanchisquare
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, RepeatedStratifiedKFold

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════
INPUT_FILE = './ch6_ch7_3class_features.pkl'
CLINICAL_FILE = './clinical_profile.csv'
RESULTS_FILE = './ch7_3class_results.pkl'
FIG_DIR = './ch7_figures'
os.makedirs(FIG_DIR, exist_ok=True)

METRIC_NAMES = ['total_spikes', 'MFR', 'rate_entropy', 'rate_variance',
                'temporal_sparsity', 'perm_entropy', 'tau_AC']
TOPO_NAMES = ['weighted_strength', 'weighted_clustering']
COND_COLORS = ['#d63031', '#636e72', '#00b894']

RANDOM_STATE = 42

# ══════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("ARSPI-NET — CHAPTER 7 EXPERIMENTS (STRUCTURE-FUNCTION COUPLING)")
print("=" * 70)

with open(INPUT_FILE, 'rb') as f:
    data = pickle.load(f)

D = data['D']                   # (633, 34, 7) dynamical
T_topo = data['T_topo']         # (633, 34, 2) topological
tPLV_mats = data['tPLV_mats']  # (633, 34, 34)
y = data['y']                   # (633,)
subjects = data['subjects']     # (633,)
COND_NAMES = data['cond_names']

N_obs, N_ch, N_dyn = D.shape
unique_subjects = np.unique(subjects)
N_subj = len(unique_subjects)

print(f"\nData: {N_obs} obs, {N_subj} subjects, {N_ch} channels, 3 classes")
print(f"Dynamical descriptors: {N_dyn} metrics per channel")
print(f"Topological descriptors: 2 metrics per channel")

import pandas as pd
try:
    df_clin = pd.read_csv(CLINICAL_FILE)
    df_clin = df_clin.drop_duplicates(subset='ID', keep='first')
    clin_ids = set(df_clin['ID'].values) & set(unique_subjects)
    print(f"Clinical metadata: {len(clin_ids)} subjects")
    HAS_CLINICAL = True
except Exception as e:
    print(f"WARNING: Clinical metadata not found ({e}).")
    HAS_CLINICAL = False

all_results = {}


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    v1, v2 = g1.var(ddof=1), g2.var(ddof=1)
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return (g1.mean() - g2.mean()) / pooled if pooled > 1e-12 else 0.0


def compute_coupling_matrix(D_obs, T_obs):
    """
    Compute 7×2 coupling matrix C for one observation.
    C[j,k] = Spearman correlation between dynamical metric j and
    topological metric k across 34 electrodes.
    """
    n_dyn = D_obs.shape[1]
    n_topo = T_obs.shape[1]
    C = np.zeros((n_dyn, n_topo))
    for j in range(n_dyn):
        for k in range(n_topo):
            rho, _ = spearmanr(D_obs[:, j], T_obs[:, k])
            C[j, k] = rho if np.isfinite(rho) else 0.0
    return C


def coupling_scalar(C):
    """Scalar coupling strength κ = ||C||_F / sqrt(p*q) (Frobenius-normalized)."""
    p, q = C.shape
    return float(np.linalg.norm(C, 'fro') / np.sqrt(p * q))


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


# ══════════════════════════════════════════════════════════════════
# COMPUTE COUPLING FOR ALL OBSERVATIONS
# ══════════════════════════════════════════════════════════════════
print(f"\nComputing coupling matrices for {N_obs} observations...")

C_all = np.zeros((N_obs, N_dyn, 2))     # (633, 7, 2) coupling matrices
kappa_all = np.zeros(N_obs)              # (633,) scalar coupling

for i in range(N_obs):
    C_all[i] = compute_coupling_matrix(D[i], T_topo[i])
    kappa_all[i] = coupling_scalar(C_all[i])

print(f"  κ range: [{kappa_all.min():.4f}, {kappa_all.max():.4f}], "
      f"mean={kappa_all.mean():.4f}, std={kappa_all.std():.4f}")


# ══════════════════════════════════════════════════════════════════
# EXP-7.1: COUPLING EXISTENCE
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXP-7.1: COUPLING EXISTENCE (3-CLASS)")
print("=" * 70)
print(f"\n  Mathematical motivation: The coupling scalar κ measures the")
print(f"  alignment between dynamical and topological descriptor profiles")
print(f"  across 34 electrodes. If κ exceeds a permutation null, the")
print(f"  temporal and spatial stages of ARSPI-Net are statistically linked.\n")

# ── Step 3: Raw observation — the group-mean coupling matrix ──
mean_C = C_all.mean(axis=0)  # (7, 2)
print(f"  Raw observation — Group-mean coupling matrix C̄ (7 dyn × 2 topo):\n")
print(f"  {'Metric':20s}  {'strength':>10s}  {'clustering':>10s}")
print(f"  {'─' * 45}")
for j, name in enumerate(METRIC_NAMES):
    print(f"  {name:20s}  {mean_C[j, 0]:>+10.4f}  {mean_C[j, 1]:>+10.4f}")

n_negative = (mean_C < 0).sum()
n_total = mean_C.size
print(f"\n  Coupling direction: {n_negative}/{n_total} entries are negative")
print(f"  Mean κ across all observations: {kappa_all.mean():.4f} ± {kappa_all.std():.4f}")

# ── Step 4: Analysis — permutation test ──
print(f"\n  Analysis — Permutation test (5000 shuffles of electrode labels):")
N_PERM = 5000
null_kappa = np.zeros(N_PERM)
rng = np.random.RandomState(RANDOM_STATE)

for p in range(N_PERM):
    # For each observation, shuffle electrode order of topological metrics
    kappas_perm = []
    for i in range(N_obs):
        perm_idx = rng.permutation(N_ch)
        T_perm = T_topo[i, perm_idx, :]
        C_perm = compute_coupling_matrix(D[i], T_perm)
        kappas_perm.append(coupling_scalar(C_perm))
    null_kappa[p] = np.mean(kappas_perm)

observed_kappa = kappa_all.mean()
# (1 + count) / (1 + n). Fixed 2026-11 (docs/v1/DEFECT_FIXES.md, D4).
p_perm = (1 + (null_kappa >= observed_kappa).sum()) / (1 + N_PERM)
print(f"  Observed mean κ: {observed_kappa:.4f}")
print(f"  Null distribution: mean={null_kappa.mean():.4f}, std={null_kappa.std():.4f}")
print(f"  Permutation p = {p_perm:.4f}")
print(f"  z-score = {(observed_kappa - null_kappa.mean()) / (null_kappa.std() + 1e-8):.2f}")

exp71 = {
    'mean_C': mean_C.tolist(),
    'mean_kappa': float(observed_kappa),
    'null_mean': float(null_kappa.mean()),
    'null_std': float(null_kappa.std()),
    'p_perm': float(p_perm),
    'n_negative': int(n_negative),
}
all_results['exp71'] = exp71

# Bonferroni-corrected cell-level tests
print(f"\n  Cell-level significance (Bonferroni α = 0.05/14 = 0.00357):")
alpha_bonf = 0.05 / 14
for j, name in enumerate(METRIC_NAMES):
    for k, tname in enumerate(TOPO_NAMES):
        vals = C_all[:, j, k]
        try:
            _, p = wilcoxon(vals)
        except:
            p = 1.0
        sig = '*' if p < alpha_bonf else ' '
        print(f"    {name:15s} × {tname:12s}: ρ̄={vals.mean():+.4f}, p={p:.5f}{sig}")

# Figure: coupling matrix + null distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
im = ax.imshow(mean_C, aspect='auto', cmap='RdBu_r',
               vmin=-max(0.1, np.abs(mean_C).max()),
               vmax=max(0.1, np.abs(mean_C).max()))
ax.set_yticks(range(N_dyn))
ax.set_yticklabels([n[:12] for n in METRIC_NAMES], fontsize=9)
ax.set_xticks(range(2))
ax.set_xticklabels(TOPO_NAMES, fontsize=9)
plt.colorbar(im, ax=ax, label='Mean Spearman ρ')
ax.set_title('Group-Mean Coupling Matrix', fontweight='bold')

ax = axes[1]
ax.hist(null_kappa, bins=50, color='#dfe6e9', edgecolor='white', label='Null')
ax.axvline(observed_kappa, color='red', linewidth=2, label=f'Observed κ={observed_kappa:.4f}')
ax.axvline(null_kappa.mean(), color='gray', linestyle='--', label=f'Null mean={null_kappa.mean():.4f}')
ax.set_xlabel('Mean κ')
ax.set_ylabel('Count')
ax.set_title(f'Permutation Test (p={p_perm:.4f})', fontweight='bold')
ax.legend(fontsize=8)

fig.suptitle('EXP-7.1: Structure-Function Coupling Existence', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{FIG_DIR}/fig01_coupling_existence.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"\n  → fig01_coupling_existence.pdf")


# ══════════════════════════════════════════════════════════════════
# EXP-7.2: VARIANCE DECOMPOSITION OF κ
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXP-7.2: VARIANCE DECOMPOSITION OF COUPLING STRENGTH")
print("=" * 70)
print(f"\n  Mathematical motivation: The 4-class analysis found 29.2% subject")
print(f"  variance, 0.6% condition variance, and 70.1% residual in κ. At")
print(f"  3-class with 3.6× more condition signal, the condition fraction")
print(f"  may increase.\n")

# ── Step 3: Raw observation ──
# Per-subject, per-condition κ
kappa_sc = np.zeros((N_subj, 3))
for si, sid in enumerate(unique_subjects):
    for c in range(3):
        mask = (subjects == sid) & (y == c)
        if mask.sum() > 0:
            kappa_sc[si, c] = kappa_all[mask].mean()

# Condition means
print(f"  Raw observation — κ by condition:")
for c in range(3):
    print(f"    {COND_NAMES[c]:10s}: κ = {kappa_sc[:, c].mean():.4f} ± {kappa_sc[:, c].std():.4f}")

print(f"\n  Between-subject std of subject-mean κ: {kappa_sc.mean(axis=1).std():.4f}")
print(f"  Mean within-subject std of κ:          {kappa_sc.std(axis=1).mean():.4f}")
ratio = kappa_sc.mean(axis=1).std() / (kappa_sc.std(axis=1).mean() + 1e-8)
print(f"  Between/within ratio:                  {ratio:.2f}")

# ── Step 4: Analysis — formal variance decomposition ──
grand_mean = kappa_all.mean()
ss_total = np.sum((kappa_all - grand_mean) ** 2)

ss_subject = 0.0
for sid in unique_subjects:
    mask = subjects == sid
    sub_mean = kappa_all[mask].mean()
    ss_subject += mask.sum() * (sub_mean - grand_mean) ** 2

ss_condition = 0.0
for c in range(3):
    mask = y == c
    cond_mean = kappa_all[mask].mean()
    ss_condition += mask.sum() * (cond_mean - grand_mean) ** 2

ss_residual = max(0, ss_total - ss_subject - ss_condition)

v_subj = 100 * ss_subject / ss_total if ss_total > 0 else 0
v_cond = 100 * ss_condition / ss_total if ss_total > 0 else 0
v_resid = 100 * ss_residual / ss_total if ss_total > 0 else 0

print(f"\n  Analysis — Variance decomposition:")
print(f"    Subject:   {v_subj:.1f}%")
print(f"    Condition: {v_cond:.1f}%")
print(f"    Residual:  {v_resid:.1f}%")
print(f"\n  4-class reference: Subject=29.2%, Condition=0.6%, Residual=70.1%")
print(f"  Ch5 BSC6 embeddings: Subject=62.6%, Condition=8.7%, Residual=28.7%")

# Permutation test for condition effect
N_PERM_COND = 5000
obs_F = (ss_condition / 2) / (ss_residual / (N_obs - N_subj - 2) + 1e-12)
null_F = np.zeros(N_PERM_COND)
for p in range(N_PERM_COND):
    # Permute condition labels within each subject
    kappa_perm = kappa_all.copy()
    for sid in unique_subjects:
        mask = subjects == sid
        idx = np.where(mask)[0]
        rng.shuffle(idx)
        kappa_perm[mask] = kappa_all[idx]
    ss_c_perm = 0.0
    gm_perm = kappa_perm.mean()
    for c in range(3):
        cm = kappa_perm[y == c].mean()
        ss_c_perm += (y == c).sum() * (cm - gm_perm) ** 2
    ss_r_perm = max(0, np.sum((kappa_perm - gm_perm) ** 2) - ss_subject - ss_c_perm)
    null_F[p] = (ss_c_perm / 2) / (ss_r_perm / (N_obs - N_subj - 2) + 1e-12)

p_cond = (1 + (null_F >= obs_F).sum()) / (1 + N_PERM_COND)
print(f"  Condition F = {obs_F:.2f}, permutation p = {p_cond:.4f}")

# Between-subject share of variance. This is var(subject means) over that plus the mean
# within-subject variance. It is not the ICC(3,1) formula, and was mislabelled as such
# until 2026-11 (docs/v1/DEFECT_FIXES.md, D7). The key name `icc` is kept for old pickles.
subject_means = kappa_sc.mean(axis=1)
icc_between = subject_means.var()
icc_within = kappa_sc.var(axis=1).mean()
icc = icc_between / (icc_between + icc_within + 1e-12)
print(f"  Between-subject variance share (not ICC(3,1)) = {icc:.3f}")

exp72 = {
    'v_subj': v_subj, 'v_cond': v_cond, 'v_resid': v_resid,
    'p_cond': float(p_cond), 'F_cond': float(obs_F), 'icc': float(icc),
    'cond_means': {COND_NAMES[c]: float(kappa_sc[:, c].mean()) for c in range(3)},
}
all_results['exp72'] = exp72

# Figure
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
bars = ax.bar(['Subject', 'Condition', 'Residual'], [v_subj, v_cond, v_resid],
              color=['#74b9ff', '#e17055', '#dfe6e9'], edgecolor='white')
ax.set_ylabel('Variance %')
ax.set_title('Variance Decomposition of κ', fontweight='bold')

ax = axes[1]
for c in range(3):
    ax.hist(kappa_sc[:, c], bins=25, alpha=0.5, color=COND_COLORS[c], label=COND_NAMES[c])
ax.set_xlabel('κ')
ax.set_ylabel('Count')
ax.set_title('κ Distribution by Condition', fontweight='bold')
ax.legend(fontsize=8)

ax = axes[2]
ax.hist(null_F, bins=50, color='#dfe6e9', edgecolor='white', label='Null F')
ax.axvline(obs_F, color='red', linewidth=2, label=f'Observed F={obs_F:.2f}')
ax.set_xlabel('F statistic')
ax.set_title(f'Condition Effect (p={p_cond:.4f})', fontweight='bold')
ax.legend(fontsize=8)

fig.suptitle('EXP-7.2: Variance Decomposition of Coupling Strength', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{FIG_DIR}/fig02_variance_decomposition.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"  → fig02_variance_decomposition.pdf")


# ══════════════════════════════════════════════════════════════════
# EXP-7.3: CLINICAL COUPLING DIFFERENCES
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXP-7.3: CLINICAL COUPLING DIFFERENCES")
print("=" * 70)

if not HAS_CLINICAL:
    print("  SKIPPED: No clinical metadata.")
else:
    print(f"\n  Mathematical motivation: Chapters 5 and 6 independently test")
    print(f"  whether spatial and temporal properties differ by diagnosis.")
    print(f"  This experiment tests whether their RELATIONSHIP — the coupling")
    print(f"  — differs by diagnosis.\n")

    clinical_vars = {'MDD': 'Major Depressive Disorder', 'PTSD': 'PTSD',
                     'GAD': 'Generalized Anxiety', 'SUD': 'Substance Use Disorder',
                     'ADHD': 'ADHD'}
    if 'Psychiatric_Medication' in df_clin.columns:
        clinical_vars['Psychiatric_Medication'] = 'Medication Status'

    exp73 = {}

    for var, label in clinical_vars.items():
        yes_s, no_s, n_yes, n_no = get_clinical_groups(var)
        if yes_s is None or n_yes < 15 or n_no < 15:
            continue

        # Subject-level mean κ
        kappa_yes = np.array([kappa_all[subjects == s].mean() for s in yes_s])
        kappa_no = np.array([kappa_all[subjects == s].mean() for s in no_s])

        # ── Step 3: Raw observation ──
        print(f"  {label} (n={n_no} vs {n_yes}):")
        print(f"    No:  κ = {kappa_no.mean():.4f} ± {kappa_no.std():.4f}")
        print(f"    Yes: κ = {kappa_yes.mean():.4f} ± {kappa_yes.std():.4f}")
        print(f"    Δ  = {kappa_yes.mean() - kappa_no.mean():+.4f}")

        # ── Step 4: Analysis ──
        try:
            _, p = mannwhitneyu(kappa_no, kappa_yes, alternative='two-sided')
        except:
            p = 1.0
        d = cohens_d(kappa_yes, kappa_no)
        sig = '*' if p < 0.05 else ' '
        print(f"    d = {d:+.3f}, p = {p:.4f}{sig}\n")

        exp73[var] = {'d': d, 'p': p, 'n_yes': n_yes, 'n_no': n_no,
                      'mean_yes': float(kappa_yes.mean()),
                      'mean_no': float(kappa_no.mean())}

        # Also test per-condition κ
        for c in range(3):
            k_yes_c = np.array([kappa_all[(subjects == s) & (y == c)].mean()
                                for s in yes_s if ((subjects == s) & (y == c)).sum() > 0])
            k_no_c = np.array([kappa_all[(subjects == s) & (y == c)].mean()
                               for s in no_s if ((subjects == s) & (y == c)).sum() > 0])
            if len(k_yes_c) > 5 and len(k_no_c) > 5:
                try:
                    _, pc = mannwhitneyu(k_no_c, k_yes_c, alternative='two-sided')
                except:
                    pc = 1.0
                dc = cohens_d(k_yes_c, k_no_c)
                sig_c = '*' if pc < 0.05 else ' '
                print(f"      {COND_NAMES[c]:10s}: d={dc:+.3f}, p={pc:.4f}{sig_c}")

    all_results['exp73'] = exp73

    # Figure
    if exp73:
        labels = [clinical_vars[k][:15] for k in exp73]
        d_vals = [exp73[k]['d'] for k in exp73]
        p_vals = [exp73[k]['p'] for k in exp73]

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        colors = ['#e17055' if p < 0.05 else '#74b9ff' for p in p_vals]
        ax.barh(range(len(labels)), d_vals, color=colors, edgecolor='white')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Cohen's d (κ difference)")
        ax.axvline(0, color='black', linewidth=0.5)
        for i, (d, p) in enumerate(zip(d_vals, p_vals)):
            ax.text(d + 0.01 * np.sign(d), i, f'p={p:.3f}', va='center', fontsize=8)
        ax.set_title('EXP-7.3: Clinical Coupling Differences', fontsize=12, fontweight='bold')
        fig.tight_layout()
        fig.savefig(f'{FIG_DIR}/fig03_clinical_coupling.pdf', bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  → fig03_clinical_coupling.pdf")


# ══════════════════════════════════════════════════════════════════
# EXP-7.4: AUGMENTATION ABLATION
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXP-7.4: AUGMENTATION ABLATION")
print("=" * 70)
print(f"\n  Mathematical motivation: If the topological (T) and dynamical (D)")
print(f"  descriptor families carry complementary discriminative information,")
print(f"  combining them improves clinical detection beyond either alone.\n")

if not HAS_CLINICAL:
    print("  SKIPPED: No clinical metadata.")
else:
    # Build subject-level feature matrices
    # T-only: 34 × 2 = 68 features (topological per electrode)
    # D-only: 34 × 7 = 238 features (dynamical per electrode)
    # T+D: 306 features (concatenation)
    # T+D+κ: 307 features (+ coupling scalar)

    exp74 = {}

    # Subject-level averages across conditions
    X_T = np.zeros((N_subj, N_ch * 2))
    X_D = np.zeros((N_subj, N_ch * N_dyn))
    X_kappa = np.zeros((N_subj, 1))
    sub_labels = {}

    for si, sid in enumerate(unique_subjects):
        mask = subjects == sid
        X_T[si] = T_topo[mask].mean(axis=0).flatten()
        X_D[si] = D[mask].mean(axis=0).flatten()
        X_kappa[si] = kappa_all[mask].mean()

    X_TD = np.hstack([X_T, X_D])
    X_TDk = np.hstack([X_TD, X_kappa])

    feature_sets = {
        'T-only (68d)': X_T,
        'D-only (238d)': X_D,
        'T+D (306d)': X_TD,
        'T+D+κ (307d)': X_TDk,
    }

    detection_tasks = ['SUD', 'MDD', 'PTSD', 'GAD', 'ADHD']

    print(f"  Raw observation — balanced accuracy across feature sets:\n")
    print(f"  {'Task':6s}", end='')
    for fs_name in feature_sets:
        print(f"  {fs_name:>15s}", end='')
    print()
    print(f"  {'─' * 70}")

    for var in detection_tasks:
        yes_s, no_s, n_yes, n_no = get_clinical_groups(var)
        if yes_s is None or n_yes < 15 or n_no < 15:
            continue

        task_results = {}
        print(f"  {var:6s}", end='')

        for fs_name, X_fs in feature_sets.items():
            # Build subject-level binary labels
            X_task, y_task, s_task = [], [], []
            for si, sid in enumerate(unique_subjects):
                if sid in set(yes_s):
                    X_task.append(X_fs[si])
                    y_task.append(1)
                    s_task.append(sid)
                elif sid in set(no_s):
                    X_task.append(X_fs[si])
                    y_task.append(0)
                    s_task.append(sid)

            X_t = np.array(X_task)
            y_t = np.array(y_task)
            s_t = np.array(s_task)

            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            accs = []
            for tr, te in cv.split(X_t, y_t, groups=s_t):
                sc = StandardScaler()
                clf = LogisticRegression(C=1.0, class_weight='balanced',
                                         max_iter=1000, random_state=RANDOM_STATE)
                clf.fit(sc.fit_transform(X_t[tr]), y_t[tr])
                accs.append(balanced_accuracy_score(y_t[te], clf.predict(sc.transform(X_t[te]))))
            acc = np.mean(accs)
            task_results[fs_name] = float(acc)
            print(f"  {acc*100:>14.1f}%", end='')

        print()
        exp74[var] = task_results

    all_results['exp74'] = exp74

    # ── Step 4: Analysis ──
    print(f"\n  Analysis — Does T+D outperform T-only and D-only?")
    for var in exp74:
        t_only = exp74[var].get('T-only (68d)', 0)
        d_only = exp74[var].get('D-only (238d)', 0)
        td = exp74[var].get('T+D (306d)', 0)
        best_single = max(t_only, d_only)
        gain = td - best_single
        print(f"    {var}: T-only={t_only*100:.1f}%, D-only={d_only*100:.1f}%, "
              f"T+D={td*100:.1f}%, gain={gain*100:+.1f}pp over best single")

    # Figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    tasks = list(exp74.keys())
    n_tasks = len(tasks)
    n_fs = len(feature_sets)
    x = np.arange(n_tasks)
    width = 0.2
    fs_colors = ['#74b9ff', '#e17055', '#00b894', '#6c5ce7']

    for fi, fs_name in enumerate(feature_sets):
        vals = [exp74[t].get(fs_name, 0.5) * 100 for t in tasks]
        ax.bar(x + fi * width - width * 1.5, vals, width, label=fs_name,
               color=fs_colors[fi], edgecolor='white')

    ax.axhline(50, color='red', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel('Balanced Accuracy (%)')
    ax.set_title('EXP-7.4: Augmentation Ablation — Clinical Detection', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/fig04_augmentation.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"\n  → fig04_augmentation.pdf")


# ══════════════════════════════════════════════════════════════════
# EXP-7.5: CONDITION-CONDITIONED COUPLING STRUCTURE (3-class)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXP-7.5: CONDITION-CONDITIONED COUPLING STRUCTURE")
print("=" * 70)
print(f"\n  Mathematical motivation: The scalar κ compresses a 7×2 matrix")
print(f"  into one number. When κ differs across conditions, the change")
print(f"  could reflect uniform scaling or reorganization concentrated in")
print(f"  specific cells. This experiment examines which dynamical-")
print(f"  topological relationships shift across conditions.\n")

# ── Step 3: Raw observation — per-condition coupling matrices ──
print(f"  Raw observation — condition-mean coupling matrices:\n")
for c in range(3):
    mask = y == c
    mean_Cc = C_all[mask].mean(axis=0)
    print(f"  {COND_NAMES[c]}:")
    print(f"    {'Metric':20s}  {'strength':>10s}  {'clustering':>10s}")
    for j, name in enumerate(METRIC_NAMES):
        print(f"    {name:20s}  {mean_Cc[j, 0]:>+10.4f}  {mean_Cc[j, 1]:>+10.4f}")
    print()

# Paired differences: Neg - Pos
print(f"  Paired differences (Neg − Pos):")
print(f"    {'Metric':20s}  {'Δ strength':>12s}  {'Δ clustering':>12s}")
print(f"    {'─' * 48}")

exp75 = {}
alpha_bonf = 0.05 / 14

for j, name in enumerate(METRIC_NAMES):
    for k, tname in enumerate(TOPO_NAMES):
        neg_vals = C_all[y == 0, j, k]
        pos_vals = C_all[y == 2, j, k]

        # Per-subject paired
        paired_neg = np.array([C_all[(subjects == s) & (y == 0), j, k].mean()
                               for s in unique_subjects])
        paired_pos = np.array([C_all[(subjects == s) & (y == 2), j, k].mean()
                               for s in unique_subjects])

        diff = paired_neg - paired_pos
        try:
            _, p = wilcoxon(diff)
        except:
            p = 1.0
        d_z = diff.mean() / (diff.std() + 1e-12)

        key = f"{name}_{tname}"
        exp75[key] = {'d_z': d_z, 'p': p, 'mean_diff': float(diff.mean())}

for j, name in enumerate(METRIC_NAMES):
    ds = exp75[f"{name}_{TOPO_NAMES[0]}"]['mean_diff']
    dc = exp75[f"{name}_{TOPO_NAMES[1]}"]['mean_diff']
    print(f"    {name:20s}  {ds:>+12.4f}  {dc:>+12.4f}")

# ── Step 4: Analysis ──
print(f"\n  Analysis — Bonferroni-corrected Wilcoxon (α = {alpha_bonf:.5f}):")
n_sig_cells = 0
for key, res in exp75.items():
    if res['p'] < alpha_bonf:
        n_sig_cells += 1
        print(f"    {key:35s}: d_z={res['d_z']:+.3f}, p={res['p']:.5f} *")

if n_sig_cells == 0:
    print(f"    (no cells reach Bonferroni significance)")

# Frobenius norm test
mean_diff_matrix = np.zeros((N_dyn, 2))
for j in range(N_dyn):
    for k in range(2):
        key = f"{METRIC_NAMES[j]}_{TOPO_NAMES[k]}"
        mean_diff_matrix[j, k] = exp75[key]['mean_diff']

obs_frob = np.linalg.norm(mean_diff_matrix, 'fro')

# Sign-flip permutation null for Frobenius norm
N_PERM_FROB = 5000
null_frob = np.zeros(N_PERM_FROB)
for p_idx in range(N_PERM_FROB):
    signs = rng.choice([-1, 1], size=N_subj)
    perm_matrix = np.zeros((N_dyn, 2))
    for j in range(N_dyn):
        for k in range(2):
            paired_neg = np.array([C_all[(subjects == s) & (y == 0), j, k].mean()
                                   for s in unique_subjects])
            paired_pos = np.array([C_all[(subjects == s) & (y == 2), j, k].mean()
                                   for s in unique_subjects])
            diff = (paired_neg - paired_pos) * signs
            perm_matrix[j, k] = diff.mean()
    null_frob[p_idx] = np.linalg.norm(perm_matrix, 'fro')

p_frob = (1 + (null_frob >= obs_frob).sum()) / (1 + N_PERM_FROB)
print(f"\n  Frobenius norm of mean ΔC: {obs_frob:.4f}")
print(f"  Permutation p (pattern-level): {p_frob:.4f}")

exp75['frobenius'] = {'obs': float(obs_frob), 'p': float(p_frob)}
all_results['exp75'] = exp75

# Figure: condition coupling matrices side by side
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for c in range(3):
    ax = axes[c]
    mean_Cc = C_all[y == c].mean(axis=0)
    vmax = max(0.1, np.abs(C_all.mean(axis=0)).max())
    im = ax.imshow(mean_Cc, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(N_dyn))
    ax.set_yticklabels([n[:10] for n in METRIC_NAMES], fontsize=8)
    ax.set_xticks(range(2))
    ax.set_xticklabels(['str', 'clust'], fontsize=8)
    ax.set_title(COND_NAMES[c], fontweight='bold', color=COND_COLORS[c])

ax = axes[3]
im2 = ax.imshow(mean_diff_matrix, aspect='auto', cmap='RdBu_r',
                vmin=-max(0.02, np.abs(mean_diff_matrix).max()),
                vmax=max(0.02, np.abs(mean_diff_matrix).max()))
ax.set_yticks(range(N_dyn))
ax.set_yticklabels([n[:10] for n in METRIC_NAMES], fontsize=8)
ax.set_xticks(range(2))
ax.set_xticklabels(['str', 'clust'], fontsize=8)
ax.set_title(f'Neg−Pos (p_frob={p_frob:.3f})', fontweight='bold')
plt.colorbar(im2, ax=ax, label='Δρ')

fig.suptitle('EXP-7.5: Condition-Conditioned Coupling Structure', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{FIG_DIR}/fig05_condition_coupling.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"  → fig05_condition_coupling.pdf")


# ══════════════════════════════════════════════════════════════════
# SAVE & SUMMARY
# ══════════════════════════════════════════════════════════════════
with open(RESULTS_FILE, 'wb') as f:
    pickle.dump(all_results, f, protocol=4)

print(f"\n{'=' * 70}")
print("COMPLETE EXPERIMENTAL SUMMARY")
print(f"{'=' * 70}")
print(f"\n  Dataset: {N_subj} subjects × 3 conditions = {N_obs} observations")

print(f"\n  EXP-7.1: Coupling existence")
print(f"    Mean κ = {exp71['mean_kappa']:.4f}, permutation p = {exp71['p_perm']:.4f}")
print(f"    {exp71['n_negative']}/14 coupling entries negative")

print(f"\n  EXP-7.2: Variance decomposition of κ")
print(f"    Subject={v_subj:.1f}%, Condition={v_cond:.1f}%, Residual={v_resid:.1f}%")
print(f"    Condition F = {obs_F:.2f}, p = {p_cond:.4f}")

if HAS_CLINICAL:
    print(f"\n  EXP-7.3: Clinical coupling differences")
    for var, res in exp73.items():
        sig = '*' if res['p'] < 0.05 else ''
        print(f"    {var}: d={res['d']:+.3f}, p={res['p']:.4f}{sig}")

    print(f"\n  EXP-7.4: Augmentation ablation")
    for var in exp74:
        td = exp74[var].get('T+D (306d)', 0.5)
        print(f"    {var}: T+D = {td*100:.1f}%")

print(f"\n  EXP-7.5: Condition-conditioned coupling")
print(f"    Frobenius ‖ΔC‖ = {obs_frob:.4f}, p = {p_frob:.4f}")
print(f"    {n_sig_cells}/14 cells Bonferroni-significant")

print(f"\n  Figures: {len(os.listdir(FIG_DIR))} PDFs in {FIG_DIR}/")
print(f"  Results: {RESULTS_FILE}")
print(f"\n  ── 5 EXPERIMENTS COMPLETE ──")
print("=" * 70)
