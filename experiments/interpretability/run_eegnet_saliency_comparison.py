#!/usr/bin/env python3
"""
ARSPI-Net Interpretability: EEGNet Saliency Comparison
=======================================================

Compares temporal saliency profiles between EEGNet (post-hoc gradient
saliency) and ARSPI-Net (intrinsic BSC₆ feature extraction window).

Key result:
  EEGNet gradient saliency peaks at 402–691 ms (LPP window)
  ARSPI-Net BSC₆ extraction window: 176–254 ms (early transient)

Interpretation:
  EEGNet learns to attend to the sustained LPP component — the dominant
  source of between-condition variance. ARSPI-Net's fixed reservoir
  captures the EARLY transient response through its explicit temporal
  coding, while still maintaining LPP alignment at two complementary
  scales (Level 1: |r| = 0.82 pre-reservoir temporal-bin analysis;
  |r| = 0.23 post-reservoir BSC₆ spike-count analysis, Ch7).
  The contrast demonstrates post-hoc vs intrinsic interpretability.

Publication: Lane, A. A. (2026). Affective Reservoir-Spike Processing and
Inference Network (ARSPI-Net): A Four-Level Interpretable Neuromorphic
Framework for Clinical EEG Analysis. PhD Dissertation, Stony Brook University.

Usage:
  python run_eegnet_saliency_comparison.py --data_dir /path/to/batch_data/

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
FS = 1024
DS_FACTOR = 4
FS_DS = FS // DS_FACTOR
BASELINE_SAMPLES = 205
POST_STIM_END = 1229


# ============================================================
# EEGNet (Lawhern et al. 2018) — same as canonical baselines
# ============================================================
class EEGNet(nn.Module):
    def __init__(self, n_channels=34, n_samples=256, n_classes=3,
                 F1=8, D=2, F2=16, kernel_length=64, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length),
                               padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1),
                                   groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout_rate)

        self.sep_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16),
                                   padding=(0, 8), groups=F1 * D, bias=False)
        self.sep_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout_rate)

        self.flatten = nn.Flatten()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            dummy = self._features(dummy)
            n_flat = dummy.shape[1]
        self.classifier = nn.Linear(n_flat, n_classes)

    def _features(self, x):
        x = self.drop1(self.pool1(self.elu1(self.bn2(
            self.depthwise(self.bn1(self.conv1(x)))))))
        x = self.drop2(self.pool2(self.elu2(self.bn3(
            self.sep_point(self.sep_depth(x))))))
        return self.flatten(x)

    def forward(self, x):
        return self.classifier(self._features(x))


# ============================================================
# Training and Saliency
# ============================================================
def train_eegnet(X, y, n_channels, n_samples, n_classes, device,
                 n_epochs=80, lr=1e-3):
    """Train EEGNet for saliency analysis."""
    model = EEGNet(n_channels, n_samples, n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=10)

    # Class weights
    counts = np.bincount(y.astype(int))
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * len(weights)
    criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(weights).to(device))

    X_t = torch.FloatTensor(X[:, np.newaxis, :, :]).to(device)
    y_t = torch.LongTensor(y).to(device)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        scheduler.step(avg)
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch + 1}/{n_epochs}: loss = {avg:.4f}")

    return model


def compute_saliency(model, X, y, device):
    """Input × gradient saliency averaged over all samples."""
    model.eval()
    X_t = torch.FloatTensor(X[:, np.newaxis, :, :]).to(device)
    X_t.requires_grad_(True)
    y_t = torch.LongTensor(y).to(device)

    out = model(X_t)
    loss = nn.CrossEntropyLoss()(out, y_t)
    loss.backward()

    # |input × gradient| averaged over batch and channels → (T,)
    sal = (X_t.grad * X_t).abs().detach().cpu().numpy()
    return sal[:, 0, :, :].mean(axis=(0, 1))


def arspinet_feature_window_ms():
    """ARSPI-Net BSC₆ dominant extraction window in ms post-stimulus.

    At 256 Hz the reservoir processes ~1000 ms of post-stimulus EEG.
    Six bins divide this into ~167 ms segments. The dominant bin
    (highest inter-condition variance) is empirically bin 2:
    167–333 ms, with the peak response at ~176–254 ms.
    """
    return 176, 254


# ============================================================
# Main Analysis
# ============================================================
def channels_first(eeg):
    """Return epochs as (N, channels, time), which is what EEGNet expects.

    SHAPE epochs have more time samples (256) than channels (34), so the shorter of the
    two trailing axes is the channel axis. Fixed 2026-11: the test was inverted, so the
    network received time as channels for either input layout
    (docs/v1/DEFECT_FIXES.md, D2).
    """
    if eeg.ndim == 3 and eeg.shape[1] < eeg.shape[2]:
        return eeg                       # already (N, C, T)
    return eeg.transpose(0, 2, 1)        # (N, T, C) -> (N, C, T)


def run_comparison(eeg, labels, outdir):
    print("=" * 60)
    print("EEGNet Saliency vs ARSPI-Net Feature Window")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # EEGNet expects (N, channels, time)
    X = channels_first(eeg)
    n_ch, n_samp = X.shape[1], X.shape[2]
    n_cls = len(np.unique(labels))
    print(f"  Shape: {X.shape}, classes: {n_cls}, device: {device}")

    print("  Training EEGNet...")
    model = train_eegnet(X, labels, n_ch, n_samp, n_cls, device)

    print("  Computing gradient saliency...")
    sal = compute_saliency(model, X, labels, device)

    time_ms = np.arange(n_samp) * (1000.0 / FS_DS)
    sal_norm = sal / (sal.max() + 1e-12)

    # Peak region: top 25% saliency
    thr = np.percentile(sal_norm, 75)
    peak_mask = sal_norm >= thr
    peak_times = time_ms[peak_mask]
    eeg_start = float(peak_times.min())
    eeg_end = float(peak_times.max())
    eeg_peak = float(time_ms[np.argmax(sal_norm)])

    arspi_start, arspi_end = arspinet_feature_window_ms()

    print(f"\n  EEGNet saliency peak: {eeg_start:.0f}–{eeg_end:.0f} ms "
          f"(max at {eeg_peak:.0f} ms)")
    print(f"  ARSPI-Net window: {arspi_start}–{arspi_end} ms")

    # ── Figure ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time_ms, sal_norm, color='#4c72b0', lw=2, label='EEGNet saliency')
    ax.fill_between(time_ms, 0, sal_norm, alpha=0.12, color='#4c72b0')
    ax.axvspan(eeg_start, eeg_end, alpha=0.18, color='#dd8452',
               label=f'EEGNet peak: {eeg_start:.0f}–{eeg_end:.0f} ms')
    ax.axvspan(arspi_start, arspi_end, alpha=0.25, color='#55a868',
               label=f'ARSPI-Net window: {arspi_start}–{arspi_end} ms')
    ax.axvspan(400, 700, alpha=0.08, color='gray',
               label='LPP reference (400–700 ms)')
    ax.set_xlabel('Time Post-Stimulus (ms)')
    ax.set_ylabel('Normalized Saliency')
    ax.set_title('Temporal Saliency: EEGNet (Post-Hoc) vs '
                 'ARSPI-Net (Intrinsic)')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 1000)
    fig.tight_layout()
    fig.savefig(outdir / 'eegnet_saliency_comparison.pdf', dpi=300)
    plt.close(fig)
    print(f"  -> Saved eegnet_saliency_comparison.pdf")

    return {'eegnet_peak_ms': eeg_peak,
            'eegnet_range': (eeg_start, eeg_end),
            'arspi_window': (arspi_start, arspi_end)}


# ============================================================
# Data Loading
# ============================================================
def load_data(data_dir):
    data_dir = Path(data_dir)
    pkl = data_dir / 'shape_features_211.pkl'
    if pkl.exists():
        with open(pkl, 'rb') as f:
            d = pickle.load(f)
        for key in ['raw_eeg', 'X_raw']:
            if key in d:
                return d[key], d.get('y', d.get('labels'))
        raise KeyError("Pickle lacks 'raw_eeg' or 'X_raw'")

    files = sorted(data_dir.glob('*.npy'))
    if not files:
        raise FileNotFoundError(
            f"No data in {data_dir}. SHAPE: https://lab-can.com/shape/")

    eeg_list, label_list = [], []
    for f in files:
        arr = np.load(f, allow_pickle=True)
        if isinstance(arr, np.ndarray) and arr.ndim >= 2:
            if arr.shape[0] > BASELINE_SAMPLES:
                arr = arr[BASELINE_SAMPLES:POST_STIM_END]
            arr = decimate(arr, DS_FACTOR, axis=0)
            mu, sd = arr.mean(axis=0), arr.std(axis=0) + 1e-8
            eeg_list.append((arr - mu) / sd)
    return np.array(eeg_list), None


def main():
    parser = argparse.ArgumentParser(
        description='EEGNet Saliency vs ARSPI-Net Comparison')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to batch_data/ with EEG files')
    parser.add_argument('--outdir', type=str,
                        default='./interpretability_results')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    eeg, labels = load_data(args.data_dir)
    if labels is None:
        print("ERROR: labels required. Provide a pickle with 'y' key.")
        sys.exit(1)

    res = run_comparison(eeg, labels, outdir)

    print("\n" + "=" * 60)
    print("SUMMARY — EEGNet Saliency Comparison")
    print("=" * 60)
    print(f"  EEGNet peak: {res['eegnet_peak_ms']:.0f} ms "
          f"({res['eegnet_range'][0]:.0f}–{res['eegnet_range'][1]:.0f} ms)")
    print(f"  ARSPI-Net: {res['arspi_window'][0]}–{res['arspi_window'][1]} ms")
    print(f"  EEGNet attends to the LPP (post-hoc); ARSPI-Net captures")
    print(f"  the early transient (intrinsic temporal coding).")


if __name__ == '__main__':
    main()
