"""Small-multiples plot: SSL fine-tuned model, train AUC vs test AUC against number of labelled windows, per participant."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_10s")
few = pd.read_csv(f"{OUT}/results_label_efficiency.csv")
few = few[few.model == "ssl_finetune"]
SERIES = {"train_auc": ("#2a78d6", "Train AUC"), "test_auc": ("#eb6834", "Test AUC")}
SURF, INK, INK2, GRID, AX = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
pids = sorted(few.participant.unique()); n = len(pids); cols = 4; rows = int(np.ceil(n / cols))
fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.6 * rows), sharey=True, facecolor=SURF, squeeze=False)
flat = axes.ravel()
for ax, pid in zip(flat, pids):
    ax.set_facecolor(SURF)
    for key, (col, lbl) in SERIES.items():
        g = few[few.participant == pid].groupby("n_labels")[key]
        mean, sd = g.mean(), g.std()
        ax.plot(mean.index, mean.values, color=col, lw=2, marker="o", ms=4, label=lbl)
        ax.fill_between(mean.index, mean - sd, mean + sd, color=col, alpha=0.12, lw=0)
    ax.axhline(0.5, color=AX, lw=1, ls="--"); ax.set_xscale("log")
    ax.set_title(pid, fontsize=10, color=INK, loc="left"); ax.set_ylim(0.2, 1.02)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(AX)
    ax.set_xticks([5, 10, 20, 50, 100, 200, 500]); ax.set_xticklabels(["5", "10", "20", "50", "100", "200", "500"], fontsize=8)
    ax.minorticks_off()
handles = [Line2D([], [], color=c, lw=2, marker="o", ms=4, label=l) for c, l in SERIES.values()] + \
          [Line2D([], [], color=AX, lw=1, ls="--", label="Chance (AUC 0.5)")]
spare = flat[n:]
for ax in spare: ax.axis("off")
if len(spare):
    spare[0].legend(handles=handles, loc="center", frameon=False, fontsize=9, labelcolor=INK, handlelength=2.2)
else:
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
for i, ax in enumerate(flat[:n]):
    if i + cols >= n: ax.set_xlabel("labelled training windows", color=INK2, fontsize=9)
for r in range(rows): axes[r, 0].set_ylabel("AUC", color=INK2, fontsize=9)
fig.suptitle("ADARP personal stress models (SSL pretrain + fine-tune): train vs test AUC by number of labelled windows\n"
             "Chronological hold-out per participant. Shaded band = ±1 SD over 3 repeats.", x=0.01, ha="left", fontsize=10, color=INK)
fig.tight_layout(rect=(0, 0 if len(spare) else 0.08, 1, 0.93)); fig.savefig(f"{OUT}/label_efficiency_auc.png", dpi=160, facecolor=SURF)
print("saved", f"{OUT}/label_efficiency_auc.png")
