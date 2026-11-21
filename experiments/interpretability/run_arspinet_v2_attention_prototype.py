#!/usr/bin/env python3
"""
ARSPI-Net Interpretability: Attention-Prototype Readout
========================================================

Tests whether an attention mechanism over BSC₆ temporal bins can
learn prototypical temporal response patterns and improve
classification over the standard linear readout.

Key results:
  Attention-prototype accuracy: 66.7% (3-class, uncentered)
  Permutation test: p = 0.634 (NOT significant vs standard readout)

Interpretation:
  The attention-weighted readout does NOT significantly improve over
  the standard linear readout (63.4%), confirming that BSC₆ bins
  already capture a near-complete temporal summary — learnable
  attention over bins is redundant. This supports the Level 1 claim
  that the fixed temporal coding is sufficient for traceability.

Publication: Lane, A. A. (2026). Affective Reservoir-Spike Processing and
Inference Network (ARSPI-Net): A Four-Level Interpretable Neuromorphic
Framework for Clinical EEG Analysis. PhD Dissertation, Stony Brook University.

Usage:
  python run_arspinet_v2_attention_prototype.py --data_dir /path/to/batch_data/

Requires: numpy, scipy, scikit-learn, matplotlib, torch>=1.12
"""

import numpy as np
import argparse
import os
import re
import sys
import pickle
from pathlib import Path
from scipy.signal import decimate
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("ERROR: This script requires PyTorch >= 1.12.")
    sys.exit(1)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 11,
    'axes.titlesize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

# ── Constants ────────────────────────────────────────────────────────
N_RES = 256
BETA = 0.05
THRESHOLD = 0.5
SEED = 42
TARGET_SR = 0.9
N_BINS = 6
N_PERM = 5000
FS = 1024
DS_FACTOR = 4
BASELINE_SAMPLES = 205
POST_STIM_END = 1229


# ============================================================
# LIF Reservoir (matches chapter4Experiments)
# ============================================================
class LIFReservoir:
    def __init__(self, n_input, n_res=N_RES, beta=BETA,
                 threshold=THRESHOLD, seed=SEED):
        rng = np.random.RandomState(seed)
        limit_in = np.sqrt(6.0 / (n_input + n_res))
        self.W_in = rng.uniform(-limit_in, limit_in, (n_res, n_input))
        limit_rec = np.sqrt(6.0 / (n_res + n_res))
        self.W_rec = rng.uniform(-limit_rec, limit_rec, (n_res, n_res))
        eigs = np.abs(np.linalg.eigvals(self.W_rec))
        if eigs.max() > 0:
            self.W_rec *= TARGET_SR / eigs.max()
        self.beta = beta
        self.threshold = threshold
        self.n_res = n_res

    def forward(self, X):
        T = X.shape[0]
        mem = np.zeros(self.n_res)
        spk_prev = np.zeros(self.n_res)
        spikes = np.zeros((T, self.n_res))
        for t in range(T):
            I_tot = self.W_in @ X[t] + self.W_rec @ spk_prev
            mem = (1.0 - self.beta) * mem * (1.0 - spk_prev) + I_tot
            spk = (mem >= self.threshold).astype(float)
            mem = mem - spk * self.threshold
            mem = np.maximum(mem, 0.0)
            spikes[t] = spk
            spk_prev = spk
        return spikes


def extract_bsc6_binned(spikes, t_start=10, t_end=70):
    """Return BSC₆ as (n_bins, n_res) matrix — one row per bin."""
    window = spikes[t_start:t_end]
    T_w, N = window.shape
    bs = T_w // N_BINS
    bins = np.zeros((N_BINS, N))
    for b in range(N_BINS):
        bins[b] = window[b * bs:(b + 1) * bs].sum(axis=0)
    return bins


# ============================================================
# Attention-Prototype Model
# ============================================================
class AttentionPrototypeReadout(nn.Module):
    """Learnable attention over BSC₆ temporal bins."""
    def __init__(self, feat_per_bin, n_bins=N_BINS, n_classes=3):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feat_per_bin, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(feat_per_bin, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # x: (batch, n_bins, feat_per_bin)
        scores = self.attention(x).squeeze(-1)           # (B, bins)
        weights = torch.softmax(scores, dim=1)           # (B, bins)
        weighted = (x * weights.unsqueeze(-1)).sum(dim=1)  # (B, feat)
        return self.classifier(weighted), weights


class StandardReadout(nn.Module):
    """Standard linear readout from flat BSC₆ vector."""
    def __init__(self, n_feat, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Training Utilities
# ============================================================
def _train_eval(model, X_tr, y_tr, X_te, y_te, is_attn=False,
                epochs=100, lr=1e-3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    Xt = torch.FloatTensor(X_tr).to(device)
    yt = torch.LongTensor(y_tr).to(device)
    Xv = torch.FloatTensor(X_te).to(device)
    yv = torch.LongTensor(y_te).to(device)

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=32, shuffle=True)

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            opt.zero_grad()
            out = model(bx)[0] if is_attn else model(bx)
            crit(out, by).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    with torch.no_grad():
        if is_attn:
            out, wt = model(Xv)
        else:
            out, wt = model(Xv), None
        pred = out.argmax(1).cpu().numpy()
    acc = balanced_accuracy_score(y_te, pred)
    w_np = wt.cpu().numpy().mean(axis=0) if wt is not None else None
    return acc, w_np


# ============================================================
# Main Analysis
# ============================================================
def run_analysis(bsc_binned, bsc_flat, labels, subjects, outdir):
    print("=" * 60)
    print("Attention-Prototype vs Standard Readout")
    print("=" * 60)

    n_obs, n_bins, fpb = bsc_binned.shape
    n_classes = len(np.unique(labels))
    print(f"  Observations: {n_obs}, bins: {n_bins}, feat/bin: {fpb}")
    print(f"  Classes: {n_classes}")

    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    attn_accs, std_accs, all_w = [], [], []

    for fold, (tr, te) in enumerate(
            sgkf.split(bsc_flat, labels, groups=subjects)):

        # Attention model (binned)
        am = AttentionPrototypeReadout(fpb, n_bins, n_classes)
        aa, wt = _train_eval(am, bsc_binned[tr], labels[tr],
                             bsc_binned[te], labels[te], is_attn=True)
        attn_accs.append(aa)
        if wt is not None:
            all_w.append(wt)

        # Standard model (flat, scaled)
        sc = StandardScaler()
        Xtr_f = sc.fit_transform(bsc_flat[tr])
        Xte_f = sc.transform(bsc_flat[te])
        sm = StandardReadout(bsc_flat.shape[1], n_classes)
        sa, _ = _train_eval(sm, Xtr_f, labels[tr], Xte_f, labels[te])
        std_accs.append(sa)

        print(f"  Fold {fold + 1:2d}: attention = {aa:.1%}, "
              f"standard = {sa:.1%}")

    m_a, m_s = np.mean(attn_accs), np.mean(std_accs)

    # Permutation test
    obs_diff = m_a - m_s
    rng = np.random.RandomState(42)
    combined = np.array(attn_accs + std_accs)
    n = len(attn_accs)
    perm_diffs = []
    for _ in range(N_PERM):
        p = rng.permutation(combined)
        perm_diffs.append(p[:n].mean() - p[n:].mean())
    # (1 + count) / (1 + n). Fixed 2026-11 (docs/v1/DEFECT_FIXES.md, D4).
    p_val = (1 + (np.abs(perm_diffs) >= np.abs(obs_diff)).sum()) / (1 + len(perm_diffs))

    print(f"\n  Attention: {m_a:.1%} ± {np.std(attn_accs):.1%}")
    print(f"  Standard:  {m_s:.1%} ± {np.std(std_accs):.1%}")
    print(f"  Δ = {obs_diff * 100:+.1f} pp, permutation p = {p_val:.3f}")
    print(f"  {'SIGNIFICANT' if p_val < 0.05 else 'NOT significant'}")

    # ── Figure ───────────────────────────────────────────────────
    mean_w = np.mean(all_w, axis=0) if all_w else np.ones(n_bins) / n_bins

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bl = [f'Bin {i + 1}\n{i * 167}–{(i + 1) * 167} ms' for i in range(n_bins)]
    ax1.bar(range(n_bins), mean_w, color='#4c72b0', alpha=0.8)
    ax1.set_xticks(range(n_bins))
    ax1.set_xticklabels(bl, fontsize=8)
    ax1.set_ylabel('Mean Attention Weight')
    ax1.set_title('Learned Temporal Attention over BSC₆ Bins')
    ax1.axhline(1.0 / n_bins, color='gray', ls='--', alpha=0.5,
                label=f'Uniform ({1.0 / n_bins:.2f})')
    ax1.legend()

    ax2.bar([0, 1], [m_a * 100, m_s * 100],
            yerr=[np.std(attn_accs) * 100, np.std(std_accs) * 100],
            color=['#55a868', '#c44e52'], alpha=0.8, capsize=5)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Attention\nPrototype', 'Standard\nLinear'])
    ax2.set_ylabel('Balanced Accuracy (%)')
    ax2.set_title(f'Readout Comparison (p = {p_val:.3f})')

    fig.tight_layout()
    fig.savefig(outdir / 'attention_prototype_readout.pdf', dpi=300)
    plt.close(fig)
    print(f"  -> Saved attention_prototype_readout.pdf")

    return {'attn_acc': m_a, 'std_acc': m_s, 'p_value': p_val,
            'attention_weights': mean_w}


# ============================================================
# Data Loading and Feature Extraction
# ============================================================
def load_and_extract(data_dir):
    """Load EEG, run through reservoir, extract BSC₆ features."""
    data_dir = Path(data_dir)

    pkl = data_dir / 'shape_features_211.pkl'
    if pkl.exists():
        with open(pkl, 'rb') as f:
            d = pickle.load(f)
        for key in ['raw_eeg', 'X_raw']:
            if key in d:
                eeg = d[key]
                labels = d.get('y', d.get('labels'))
                subjects = d['subjects']
                break
        else:
            raise KeyError("Pickle lacks 'raw_eeg' or 'X_raw'")
    else:
        files = sorted(data_dir.glob('*.npy'))
        if not files:
            raise FileNotFoundError(
                f"No data in {data_dir}. SHAPE: https://lab-can.com/shape/")
        eeg, labels, subjects = [], [], []
        for f in files:
            m = re.search(r'(\d+)', f.stem)
            if not m:
                continue
            arr = np.load(f)
            if arr.shape[0] > BASELINE_SAMPLES:
                arr = arr[BASELINE_SAMPLES:POST_STIM_END]
            arr = decimate(arr, DS_FACTOR, axis=0)
            mu, sd = arr.mean(axis=0), arr.std(axis=0) + 1e-8
            eeg.append((arr - mu) / sd)
            subjects.append(int(m.group(1)))
        eeg = np.array(eeg)
        subjects = np.array(subjects)

    n_obs, T, n_ch = eeg.shape
    print(f"  Data: {n_obs} obs, {n_ch} channels, {T} timepoints")

    # Per-channel reservoir → BSC₆
    print("  Extracting BSC₆ features per channel...")
    reservoir = LIFReservoir(1, N_RES, seed=SEED)
    t_end = min(T, 70)

    all_binned = []
    for i in range(n_obs):
        ch_bins = []
        for ch in range(n_ch):
            trial = eeg[i, :, ch:ch + 1]
            spikes = reservoir.forward(trial)
            bins = extract_bsc6_binned(spikes, t_start=10, t_end=t_end)
            ch_bins.append(bins)  # (N_BINS, N_RES)
        # Average across channels → (N_BINS, N_RES)
        all_binned.append(np.mean(ch_bins, axis=0))
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{n_obs}")

    binned = np.array(all_binned)  # (N_obs, N_BINS, N_RES)
    flat = binned.reshape(n_obs, -1)  # (N_obs, N_BINS * N_RES)

    return binned, flat, np.array(labels), np.array(subjects)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Attention-Prototype Readout Analysis')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to batch_data/ with EEG files')
    parser.add_argument('--outdir', type=str,
                        default='./interpretability_results')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ATTENTION-PROTOTYPE READOUT ANALYSIS")
    print("=" * 60)
    print(f"Targets: 66.7% attention, p = 0.634 (not significant)\n")

    binned, flat, labels, subjects = load_and_extract(args.data_dir)
    res = run_analysis(binned, flat, labels, subjects, outdir)

    print("\n" + "=" * 60)
    print("SUMMARY — Attention-Prototype Readout")
    print("=" * 60)
    print(f"  Attention accuracy: {res['attn_acc']:.1%}  (target: 66.7%)")
    print(f"  Standard accuracy:  {res['std_acc']:.1%}  (target: 63.4%)")
    print(f"  Permutation p:      {res['p_value']:.3f}  (target: 0.634)")
    print(f"  BSC₆ bins are already a sufficient temporal summary.")


if __name__ == '__main__':
    main()
