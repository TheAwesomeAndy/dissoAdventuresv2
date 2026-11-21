#!/usr/bin/env python3
"""
Chapter 7 — Experiment A: Infrastructure + Coupling Existence Test
=================================================================
Optimized version with vectorized permutation null, matrix-based clustering,
and rank-based coupling computation.

Usage:
  python3 run_chapter7_experiment_A.py START_IDX BATCH_SIZE
  python3 run_chapter7_experiment_A.py --analyze

Examples:
  python3 run_chapter7_experiment_A.py 0 8      # Process subjects 0-7
  python3 run_chapter7_experiment_A.py 8 8      # Process subjects 8-15
  python3 run_chapter7_experiment_A.py --analyze  # Generate figures/tables

The script saves intermediate results to chapter7_results/ch7_full_results.pkl
and is crash-resilient: re-running a batch skips already-completed subjects.
"""
import numpy as np
import os
import re
import sys
import pickle
import time
import math
from collections import defaultdict
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import spearmanr, wilcoxon, rankdata

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════
N_CH = 34           # Number of EEG channels
N_RES = 256         # Reservoir neurons
BETA = 0.05         # LIF decay constant
M_TH = 0.5          # Spike threshold
FS = 1024            # Sampling rate (Hz)
SEED = 42            # Reservoir random seed
N_PERM = 5000        # Permutation null iterations
THETA_BAND = (4, 8)  # Hz
CATS = ['Threat', 'Mutilation', 'Cute', 'Erotic']
DYN_NAMES = ['total_spikes', 'mean_firing_rate', 'rate_entropy',
             'rate_variance', 'temporal_sparsity', 'permutation_entropy',
             'tau_ac']
TOPO_NAMES = ['strength', 'clustering']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'chapter7_results', 'ch7_full_results.pkl')
FIGURE_DIR = os.environ.get('ARSPI_FIGURE_DIR', os.path.join(SCRIPT_DIR, 'figures'))

# ═══════════════════════════════════════════════════════════════════
# FILE INVENTORY
# ═══════════════════════════════════════════════════════════════════
def build_file_inventory():
    """Discover all EEG files and return {(sid, cat): filepath} dict."""
    cat_dirs = [os.path.join(SCRIPT_DIR, '..', 'categories', f'categoriesbatch{b}') for b in [1, 2, 3, 4]]
    pat = re.compile(
        r'SHAPE_Community_(\d+)_IAPS(Neg|Pos)_(Threat|Mutilation|Cute|Erotic)_BC\.txt'
    )
    files = {}
    for cd in cat_dirs:
        if not os.path.isdir(cd):
            continue
        for f in os.listdir(cd):
            m = pat.match(f)
            if m:
                sid = int(m.group(1))
                cat = m.group(3)
                if sid != 127:  # Excluded subject
                    files[(sid, cat)] = os.path.join(cd, f)
    return files


def get_sorted_subjects(all_files):
    """Return sorted list of unique subject IDs."""
    return sorted(set(s for s, c in all_files.keys()))


# ═══════════════════════════════════════════════════════════════════
# RESERVOIR
# ═══════════════════════════════════════════════════════════════════
def build_reservoir(seed=SEED):
    """Initialize reservoir weights (deterministic, seed-controlled)."""
    rng = np.random.RandomState(seed)
    Win = rng.randn(N_RES) * 0.3
    mask = (rng.rand(N_RES, N_RES) < 0.1).astype(float)
    np.fill_diagonal(mask, 0)
    Wrec = rng.randn(N_RES, N_RES) * 0.05 * mask
    return Win, Wrec


def run_reservoir(u, Win, Wrec):
    """
    Run LIF reservoir on input signal u (1D array, length T).
    Returns: total_spikes, pop_rate(T), per_neuron_rate(N_RES), mean_membrane(T)
    """
    T = len(u)
    pop_rate = np.zeros(T)
    neuron_rate = np.zeros(N_RES)
    mean_membrane = np.zeros(T)
    m = np.zeros(N_RES)
    s = np.zeros(N_RES)
    total_spikes = 0.0

    for t in range(T):
        I = Win * u[t] + Wrec @ s
        m = (1 - BETA) * m * (1 - s) + I
        sp = (m >= M_TH).astype(float)
        count = sp.sum()
        total_spikes += count
        pop_rate[t] = count / N_RES
        neuron_rate += sp
        mean_membrane[t] = m.mean()
        s = sp

    neuron_rate /= T
    return total_spikes, pop_rate, neuron_rate, mean_membrane


# ═══════════════════════════════════════════════════════════════════
# DYNAMICAL METRICS
# ═══════════════════════════════════════════════════════════════════
def permutation_entropy(x, d=4, tau=1):
    """Bandt-Pompe permutation entropy, normalized to [0, 1]."""
    T = len(x)
    patterns = defaultdict(int)
    for t in range(T - (d - 1) * tau):
        window = x[t:t + d * tau:tau]
        patterns[tuple(np.argsort(window).tolist())] += 1
    n_patterns = T - (d - 1) * tau
    if n_patterns <= 0:
        return 0.0
    probs = np.array(list(patterns.values())) / n_patterns
    H = -np.sum(probs * np.log2(probs + 1e-15))
    H_max = np.log2(math.factorial(d))
    return float(H / H_max) if H_max > 0 else 0.0


def autocorrelation_decay(x, max_lag=100):
    """First lag where autocorrelation drops to e^{-1}."""
    T = len(x)
    xc = x - x.mean()
    var = np.sum(xc ** 2)
    if var <= 0:
        return 0.0
    ml = min(max_lag, T // 2)
    ac = np.array([np.sum(xc[:T - k] * xc[k:]) / var for k in range(ml)])
    threshold = np.exp(-1)
    below = np.where(ac <= threshold)[0]
    return float(below[0]) if len(below) > 0 else float(ml)


def extract_dynamical_metrics(total_spikes, pop_rate, neuron_rate, mean_membrane, T):
    """Extract 7 dual-gate validated metrics from reservoir output."""
    # 1. total_spikes
    ts = total_spikes
    # 2. mean_firing_rate
    mfr = total_spikes / (N_RES * T)
    # 3. rate_entropy: binary entropy of per-neuron rates
    p = neuron_rate
    h = -np.sum(p * np.log2(p + 1e-15) + (1 - p) * np.log2(1 - p + 1e-15)) / N_RES
    # 4. rate_variance
    rv = float(np.var(neuron_rate))
    # 5. temporal_sparsity
    a_temp = float(np.mean(pop_rate < 1.0 / N_RES))
    # 6. permutation_entropy
    pe = permutation_entropy(mean_membrane)
    # 7. tau_ac
    tac = autocorrelation_decay(pop_rate)
    return [ts, mfr, h, rv, a_temp, pe, tac]


# ═══════════════════════════════════════════════════════════════════
# CONNECTIVITY: tPLV (theta-band time-averaged phase concentration)
# ═══════════════════════════════════════════════════════════════════
def compute_tplv(eeg, fs=FS, band=THETA_BAND):
    """
    Compute 34x34 time-averaged phase concentration matrix.
    This is within-epoch phase synchrony, NOT trial-to-trial PLV.
    """
    T, nch = eeg.shape
    nyq = fs / 2.0
    lo = band[0] / nyq
    hi = min(band[1] / nyq, 0.99)

    # Bandpass filter + Hilbert phase extraction
    b, a = butter(3, [lo, hi], btype='band')
    phases = np.zeros((T, nch))
    for ch in range(nch):
        filtered = filtfilt(b, a, eeg[:, ch])
        analytic = hilbert(filtered)
        phases[:, ch] = np.angle(analytic)

    # Pairwise tPLV — fully vectorized
    plv = np.zeros((nch, nch))
    for i in range(nch):
        phase_diff = phases[:, i:i + 1] - phases[:, i + 1:]
        vals = np.abs(np.mean(np.exp(1j * phase_diff), axis=0))
        plv[i, i + 1:] = vals
        plv[i + 1:, i] = vals
    np.fill_diagonal(plv, 0.0)
    return plv


# ═══════════════════════════════════════════════════════════════════
# TOPOLOGY: strength + weighted clustering (OPTIMIZED)
# ═══════════════════════════════════════════════════════════════════
def extract_topology(plv):
    """
    Extract per-node strength and weighted clustering from PLV matrix.
    Uses matrix multiplication on cube-root weights for O(n^2) clustering
    instead of O(n^3) triple-nested loop.
    """
    nch = plv.shape[0]

    # Strength: sum of edge weights per node
    strength = plv.sum(axis=1)

    # Weighted clustering coefficient (Onnela et al.)
    # C_i = (1 / (k_i * (k_i - 1))) * sum over ordered pairs (j, h) of (w_ij * w_ih * w_jh)^{1/3}
    # diag(W^{1/3} @ W^{1/3} @ W^{1/3})_i is that ordered-pair sum: it counts each triangle
    # through node i twice, which the 1/(k(k-1)) normalization already accounts for.
    # Fixed 2026-11: an extra factor 2 doubled every value (docs/v1/DEFECT_FIXES.md, D3).
    W_cbrt = np.cbrt(plv)
    tri = np.diag(W_cbrt @ W_cbrt @ W_cbrt)

    # Degree = number of nonzero neighbors per node
    k = (plv > 0).sum(axis=1).astype(float)
    denom = k * (k - 1)
    clustering = np.where(denom > 0, tri / denom, 0.0)

    return np.column_stack([strength, clustering])


# ═══════════════════════════════════════════════════════════════════
# COUPLING: Spearman C matrix + normalized kappa (OPTIMIZED)
# ═══════════════════════════════════════════════════════════════════
def _rank_columns(X):
    """Rank each column independently, center, and compute norms."""
    R = np.apply_along_axis(rankdata, 0, X)
    R -= R.mean(axis=0)
    norms = np.sqrt((R ** 2).sum(axis=0))
    return R, norms


def compute_coupling(D, T_mat):
    """
    Compute 7x2 Spearman correlation matrix and normalized kappa.
    Uses rank-based matrix multiply instead of 14 individual spearmanr calls.
    D: (34, 7) dynamical profiles
    T_mat: (34, 2) topological profiles
    Returns: C (7x2), kappa (scalar)
    """
    p = D.shape[1]   # 7
    q = T_mat.shape[1]  # 2

    D_ranks, D_norms = _rank_columns(D)
    T_ranks, T_norms = _rank_columns(T_mat)

    # All 14 correlations in one matrix multiply
    denom = np.outer(D_norms, T_norms) + 1e-15
    C = (D_ranks.T @ T_ranks) / denom

    kappa = np.linalg.norm(C, 'fro') / np.sqrt(p * q)
    return C, kappa


def compute_null_kappa(D, T_mat, n_perm=N_PERM, rng_seed=0):
    """
    Vectorized permutation null: shuffle electrode labels in T, recompute kappa.
    Pre-ranks D once, then only re-ranks permuted T columns per iteration.
    Returns array of n_perm null kappa values.
    """
    rng = np.random.RandomState(rng_seed)
    p = D.shape[1]
    q = T_mat.shape[1]

    # Pre-compute D ranks (constant across all permutations)
    D_ranks, D_norms = _rank_columns(D)
    denom_D = D_norms  # (7,)

    null_kappas = np.zeros(n_perm)
    for pi in range(n_perm):
        perm_idx = rng.permutation(T_mat.shape[0])
        T_perm = T_mat[perm_idx]

        T_ranks = np.apply_along_axis(rankdata, 0, T_perm)
        T_ranks -= T_ranks.mean(axis=0)
        T_norms = np.sqrt((T_ranks ** 2).sum(axis=0))

        denom = np.outer(denom_D, T_norms) + 1e-15
        C_perm = (D_ranks.T @ T_ranks) / denom
        null_kappas[pi] = np.linalg.norm(C_perm, 'fro') / np.sqrt(p * q)

    return null_kappas


# ═══════════════════════════════════════════════════════════════════
# PERSISTENCE: load/save results with crash recovery
# ═══════════════════════════════════════════════════════════════════
def load_results():
    """Load existing results or create empty structure."""
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'rb') as f:
            return pickle.load(f)
    return {
        'dyn_profiles': {},
        'topo_profiles': {},
        'plv_matrices': {},
        'coupling_C': {},
        'coupling_kappa': {},
        'null_kappa': {},
        'subjects': [],
        'categories': CATS,
        'dyn_names': DYN_NAMES,
        'topo_names': TOPO_NAMES,
        'n_channels': N_CH,
        'n_permutations': N_PERM,
        'completed_subjects': [],
    }


def save_results(results):
    """Save results to disk."""
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(results, f)


# ═══════════════════════════════════════════════════════════════════
# BATCH PROCESSING: one subject at a time
# ═══════════════════════════════════════════════════════════════════
def process_one_subject(sid, all_files, Win, Wrec, results):
    """
    Process all 4 categories for one subject.
    Computes D, T, PLV, C, kappa, and null kappa for each category.
    Modifies results dict in place.
    """
    for cat in CATS:
        if (sid, cat) not in all_files:
            print(f"    WARNING: Missing ({sid}, {cat})")
            continue

        # Load EEG
        eeg = np.loadtxt(all_files[(sid, cat)])  # (1229, 34)
        T_len = eeg.shape[0]

        # ── Dynamical profiles: 34 channels ──
        D = np.zeros((N_CH, len(DYN_NAMES)))
        for ch in range(N_CH):
            u = eeg[:, ch]
            u = (u - u.mean()) / (u.std() + 1e-10)
            ts, pr, pnr, Mm = run_reservoir(u, Win, Wrec)
            D[ch] = extract_dynamical_metrics(ts, pr, pnr, Mm, T_len)

        # ── Connectivity: tPLV ──
        plv = compute_tplv(eeg)

        # ── Topology: strength + clustering (optimized) ──
        T_mat = extract_topology(plv)

        # ── Coupling (optimized) ──
        C, kappa = compute_coupling(D, T_mat)

        # ── Permutation null (optimized) ──
        null_k = compute_null_kappa(D, T_mat, N_PERM, rng_seed=sid)

        # ── Store ──
        results['dyn_profiles'][(sid, cat)] = D
        results['topo_profiles'][(sid, cat)] = T_mat
        results['plv_matrices'][(sid, cat)] = plv
        results['coupling_C'][(sid, cat)] = C
        results['coupling_kappa'][(sid, cat)] = kappa
        results['null_kappa'][(sid, cat)] = null_k

    results['completed_subjects'].append(sid)
    return results


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS: figures and tables (run after all batches complete)
# ═══════════════════════════════════════════════════════════════════
def run_analysis(results):
    """Generate all Experiment A figures and tables."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    os.makedirs(FIGURE_DIR, exist_ok=True)

    subjects = results['subjects']

    # Collect all kappa and null kappa values
    all_kappa = []
    all_null_median = []
    all_pvals = []
    all_C = []

    for sid in subjects:
        for cat in CATS:
            key = (sid, cat)
            if key not in results['coupling_kappa']:
                continue
            k = results['coupling_kappa'][key]
            nk = results['null_kappa'][key]
            pval = (1 + np.sum(nk >= k)) / (1 + len(nk))
            all_kappa.append(k)
            all_null_median.append(np.median(nk))
            all_pvals.append(pval)
            all_C.append(results['coupling_C'][key])

    all_kappa = np.array(all_kappa)
    all_null_median = np.array(all_null_median)
    all_pvals = np.array(all_pvals)
    all_C = np.array(all_C)  # (844, 7, 2)

    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT A RESULTS")
    print(f"{'=' * 60}")
    print(f"N observations: {len(all_kappa)}")
    print(f"Median kappa:      {np.median(all_kappa):.4f}")
    print(f"Median null kappa: {np.median(all_null_median):.4f}")
    print(f"% with p < 0.05:  {100 * np.mean(all_pvals < 0.05):.1f}%")
    print(f"% with p < 0.01:  {100 * np.mean(all_pvals < 0.01):.1f}%")

    # Group-level Wilcoxon test
    diffs = all_kappa - all_null_median
    stat, p_group = wilcoxon(diffs, alternative='greater')
    d_z = np.mean(diffs) / (np.std(diffs) + 1e-10)

    print(f"\nGroup Wilcoxon (kappa > null_median):")
    print(f"  statistic = {stat:.0f}, p = {p_group:.2e}, d_z = {d_z:.3f}")

    # ── FIGURE 7.A1: Group-mean C matrix ──
    mean_C = np.mean(all_C, axis=0)  # (7, 2)
    sig_mask = np.zeros_like(mean_C, dtype=bool)
    pval_C = np.zeros_like(mean_C)
    for j in range(mean_C.shape[0]):
        for k in range(mean_C.shape[1]):
            vals = all_C[:, j, k]
            try:
                _, p = wilcoxon(vals)
                pval_C[j, k] = p
                sig_mask[j, k] = p < 0.05 / (7 * 2)  # Bonferroni
            except Exception:
                pval_C[j, k] = 1.0

    fig, ax = plt.subplots(1, 1, figsize=(5, 7))
    vmax = max(abs(mean_C.min()), abs(mean_C.max()))
    if vmax == 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(mean_C, cmap='RdBu_r', norm=norm, aspect='auto')
    ax.set_xticks(range(2))
    ax.set_xticklabels(TOPO_NAMES, fontsize=11)
    ax.set_yticks(range(7))
    ax.set_yticklabels(DYN_NAMES, fontsize=10)

    for j in range(7):
        for k in range(2):
            star = '*' if sig_mask[j, k] else ''
            ax.text(k, j, f'{mean_C[j, k]:.3f}{star}', ha='center', va='center',
                    fontsize=9, color='black' if abs(mean_C[j, k]) < vmax * 0.7 else 'white')

    plt.colorbar(im, ax=ax, label='Mean Spearman \u03c1', shrink=0.6)
    ax.set_title(f'Group-Mean Coupling Matrix\n(N={len(all_kappa)}, * = Bonf. p<0.05/{7 * 2})',
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(f'{FIGURE_DIR}/fig7_A1_mean_coupling_matrix.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: fig7_A1_mean_coupling_matrix.pdf")

    # ── FIGURE 7.A2: kappa distribution vs null ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axes

    ax1.hist(all_kappa, bins=40, alpha=0.7, label='Observed \u03ba', color='steelblue', density=True)
    ax1.hist(all_null_median, bins=40, alpha=0.5, label='Null median \u03ba', color='gray', density=True)
    ax1.axvline(np.median(all_kappa), color='steelblue', ls='--', lw=2)
    ax1.axvline(np.median(all_null_median), color='gray', ls='--', lw=2)
    ax1.set_xlabel('\u03ba (normalized coupling strength)')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Observed vs Null \u03ba\nMedian: {np.median(all_kappa):.4f} vs {np.median(all_null_median):.4f}')
    ax1.legend()

    ax2.hist(all_pvals, bins=50, color='steelblue', alpha=0.7)
    ax2.axvline(0.05, color='red', ls='--', lw=2, label='p=0.05')
    ax2.set_xlabel('Per-observation p-value')
    ax2.set_ylabel('Count')
    ax2.set_title(f'p-value distribution\n{100 * np.mean(all_pvals < 0.05):.1f}% significant at 0.05')
    ax2.legend()

    plt.tight_layout()
    fig.savefig(f'{FIGURE_DIR}/fig7_A2_kappa_vs_null.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig7_A2_kappa_vs_null.pdf")

    # ── FIGURE 7.A3: Example subjects ──
    subj_mean_kappa = {}
    for sid in subjects:
        vals = [results['coupling_kappa'].get((sid, c), np.nan) for c in CATS]
        subj_mean_kappa[sid] = np.nanmean(vals)

    sorted_subj = sorted(subj_mean_kappa.keys(), key=lambda s: subj_mean_kappa[s])
    weak_sid = sorted_subj[len(sorted_subj) // 10]
    med_sid = sorted_subj[len(sorted_subj) // 2]
    strong_sid = sorted_subj[-len(sorted_subj) // 10]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, sid, label in zip(axes, [weak_sid, med_sid, strong_sid], ['Weak', 'Medium', 'Strong']):
        C_ex = results['coupling_C'].get((sid, CATS[0]), np.zeros((7, 2)))
        k_ex = results['coupling_kappa'].get((sid, CATS[0]), 0)
        vmax_ex = max(0.5, abs(C_ex).max())
        im = ax.imshow(C_ex, cmap='RdBu_r', vmin=-vmax_ex, vmax=vmax_ex, aspect='auto')
        ax.set_title(f'{label} (S{sid}, \u03ba={k_ex:.3f})')
        ax.set_xticks(range(2))
        ax.set_xticklabels(TOPO_NAMES, fontsize=9)
        ax.set_yticks(range(7))
        ax.set_yticklabels(DYN_NAMES, fontsize=8)
        for j in range(7):
            for k in range(2):
                ax.text(k, j, f'{C_ex[j, k]:.2f}', ha='center', va='center', fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.7)

    plt.suptitle('Example Subject Coupling Matrices (Threat category)', fontsize=13)
    plt.tight_layout()
    fig.savefig(f'{FIGURE_DIR}/fig7_A3_example_subjects.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig7_A3_example_subjects.pdf")

    # ── FIGURE 7.A4: kappa by category ──
    cat_kappas = {c: [] for c in CATS}
    for sid in subjects:
        for cat in CATS:
            k = results['coupling_kappa'].get((sid, cat), np.nan)
            cat_kappas[cat].append(k)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    positions = range(len(CATS))
    data = [cat_kappas[c] for c in CATS]
    ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
    ax.set_xticks(positions)
    ax.set_xticklabels(CATS, fontsize=11)
    ax.set_ylabel('\u03ba (normalized coupling strength)')
    ax.set_title('Coupling Strength by Affective Category')

    # Add paired lines for subset of subjects
    for sid in subjects[::10]:
        vals = [results['coupling_kappa'].get((sid, c), np.nan) for c in CATS]
        ax.plot(positions, vals, '-', color='gray', alpha=0.15, lw=0.5)

    plt.tight_layout()
    fig.savefig(f'{FIGURE_DIR}/fig7_A4_kappa_by_category.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig7_A4_kappa_by_category.pdf")

    # ── Print summary tables ──
    print(f"\n{'=' * 60}")
    print("TABLE 7.A1: Coupling Existence Summary")
    print(f"{'=' * 60}")
    print(f"{'Statistic':<30} {'Observed':>10} {'Null':>10} {'d_z':>8} {'p':>12}")
    print(f"{'Median kappa':<30} {np.median(all_kappa):>10.4f} {np.median(all_null_median):>10.4f} {d_z:>8.3f} {p_group:>12.2e}")
    print(f"{'% p<0.05':<30} {100 * np.mean(all_pvals < 0.05):>9.1f}% {'5.0%':>10}")
    print(f"{'% p<0.01':<30} {100 * np.mean(all_pvals < 0.01):>9.1f}% {'1.0%':>10}")

    print(f"\n{'=' * 60}")
    print("TABLE 7.A2: Group-Mean C Matrix")
    print(f"{'=' * 60}")
    print(f"{'Metric':<22} {'Strength (rho)':>14} {'p':>10} {'Clustering (rho)':>16} {'p':>10}")
    for j, name in enumerate(DYN_NAMES):
        print(f"{name:<22} {mean_C[j, 0]:>14.4f} {pval_C[j, 0]:>10.2e} {mean_C[j, 1]:>16.4f} {pval_C[j, 1]:>10.2e}")

    # Save analysis summary
    analysis = {
        'all_kappa': all_kappa,
        'all_null_median': all_null_median,
        'all_pvals': all_pvals,
        'all_C': all_C,
        'mean_C': mean_C,
        'pval_C': pval_C,
        'sig_mask': sig_mask,
        'group_wilcoxon_stat': stat,
        'group_wilcoxon_p': p_group,
        'group_dz': d_z,
        'cat_kappas': cat_kappas,
    }
    with open(os.path.join(SCRIPT_DIR, 'chapter7_results', 'ch7_expA_analysis.pkl'), 'wb') as f:
        pickle.dump(analysis, f)
    print(f"\nAnalysis saved to ch7_expA_analysis.pkl")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 run_chapter7_experiment_A.py START_IDX BATCH_SIZE")
        print("  python3 run_chapter7_experiment_A.py --analyze")
        sys.exit(1)

    if sys.argv[1] == '--analyze':
        results = load_results()
        print(f"Loaded results: {len(results['completed_subjects'])} subjects completed")
        results['subjects'] = sorted(results['completed_subjects'])
        run_analysis(results)
        return

    start_idx = int(sys.argv[1])
    batch_size = int(sys.argv[2])

    # Build file inventory
    all_files = build_file_inventory()
    all_subjects = get_sorted_subjects(all_files)
    print(f"Total subjects available: {len(all_subjects)}")

    # Load existing results
    results = load_results()
    completed = set(results['completed_subjects'])
    print(f"Already completed: {len(completed)} subjects")

    # Select batch
    batch_subjects = all_subjects[start_idx:start_idx + batch_size]
    batch_subjects = [s for s in batch_subjects if s not in completed]

    if not batch_subjects:
        print("All subjects in this batch already completed. Nothing to do.")
        return

    print(f"\nProcessing batch: subjects index {start_idx} to {start_idx + batch_size - 1}")
    print(f"Subject IDs: {batch_subjects}")
    print(f"Runs: {len(batch_subjects)} subjects x 4 cats x 34 ch = {len(batch_subjects) * 4 * 34}")

    # Build reservoir (same for all)
    Win, Wrec = build_reservoir(SEED)

    # Process each subject
    t0 = time.time()
    for i, sid in enumerate(batch_subjects):
        ts = time.time()
        print(f"\n  [{i + 1}/{len(batch_subjects)}] Subject {sid}...", end=' ', flush=True)
        results = process_one_subject(sid, all_files, Win, Wrec, results)
        elapsed = time.time() - ts
        print(f"done ({elapsed:.1f}s)")

        # Save after EACH subject for crash resilience
        results['subjects'] = sorted(results['completed_subjects'])
        save_results(results)

    total = time.time() - t0
    print(f"\nBatch complete: {len(batch_subjects)} subjects in {total:.0f}s")
    print(f"Total completed: {len(results['completed_subjects'])}/{len(all_subjects)}")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
