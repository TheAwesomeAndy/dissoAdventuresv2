#!/usr/bin/env python3
"""Figure: time-resolved decoding under each normalization, ERP CORE N170 and P3.

    python experiments/zscore_artifact/make_figure.py erpcore_results.json zscore_artifact.png
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLE = {"none": ("#1f4e79", "-", "none (baseline-corrected)"),
         "baseline_z": ("#5b9bd5", "--", "divided by pre-stimulus SD"),
         "epoch_z_full": ("#c0392b", "-", "z-scored over whole epoch"),
         "epoch_z_post": ("#e67e22", "--", "z-scored with post-stimulus stats")}


def main():
    R = json.load(open(sys.argv[1]))
    paras = list(R["paradigms"])
    fig, axes = plt.subplots(2, len(paras), figsize=(5.2 * len(paras), 7), sharex=True)
    for j, p in enumerate(paras):
        P = R["paradigms"][p]
        for i, regime in enumerate(("within", "across")):
            ax = axes[i, j]
            for n, (col, ls, lab) in STYLE.items():
                tc = P[regime][n]["time_course"]
                t = tc["bin_center_ms"]
                y = tc["mean_percent"] if regime == "within" else tc["accuracy_percent"]
                ax.plot(t, y, color=col, ls=ls, lw=1.6, label=lab)
                if regime == "within":
                    lo = [a - 1.96 * s for a, s in zip(y, tc["sem_percent"])]
                    hi = [a + 1.96 * s for a, s in zip(y, tc["sem_percent"])]
                    ax.fill_between(t, lo, hi, color=col, alpha=0.12, lw=0)
            ax.axvspan(-200, 0, color="0.9", zorder=0)
            ax.axhline(50, color="0.4", lw=0.8)
            ax.axvline(0, color="0.4", lw=0.8)
            ax.set_title(f"{p}: {' vs '.join(c.split(',')[0] for c in P['contrast'])}, "
                         f"{'single trials within participant' if regime == 'within' else 'participant averages, across participants'}",
                         fontsize=9)
            ax.set_ylabel("balanced accuracy (%)")
            if i == 1:
                ax.set_xlabel("time from stimulus onset (ms)")
    axes[0, 0].legend(fontsize=7.5, loc="upper left", frameon=False)
    fig.text(0.5, 0.005, "Shaded grey: pre-stimulus interval, where accuracy must be at chance (50%). "
             f"ERP CORE, {R['paradigms'][paras[0]]['n_participants']} participants per paradigm.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(sys.argv[2], dpi=200)
    print("wrote", sys.argv[2])


if __name__ == "__main__":
    main()
