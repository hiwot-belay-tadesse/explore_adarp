"""One plot: train AUC vs test AUC per participant, SSL pretrain + fine-tune model trained on all labels."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_10s")
df = pd.read_csv(f"{OUT}/results_full_labels.csv"); df = df[df.model == "ssl_finetune"].sort_values("participant", ascending=False)
TRAIN, TEST = "#2a78d6", "#eb6834"
SURF, INK, INK2, GRID, AX = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(df) + 1.8), facecolor=SURF); ax.set_facecolor(SURF)
y = np.arange(len(df))
ax.hlines(y, df.train_auc, df.test_auc, color=AX, lw=1.5, zorder=1)
ax.scatter(df.train_auc, y, color=TRAIN, s=60, zorder=3, edgecolor=SURF, lw=1.5)
ax.scatter(df.test_auc, y, color=TEST, s=60, zorder=3, edgecolor=SURF, lw=1.5)
for yi, tr, te in zip(y, df.train_auc, df.test_auc):
    lo, hi = sorted([(tr, "tr"), (te, "te")])
    ax.text(lo[0] - 0.012, yi, f"{lo[0]:.2f}", va="center", ha="right", fontsize=8, color=INK2)
    ax.text(hi[0] + 0.012, yi, f"{hi[0]:.2f}", va="center", ha="left", fontsize=8, color=INK2)
ax.axvline(0.5, color=AX, lw=1, ls="--", zorder=0)
ax.set_yticks(y); ax.set_yticklabels(df.participant, fontsize=9, color=INK)
ax.set_xlim(0.3, 1.0); ax.set_xlabel("AUC", color=INK2, fontsize=9)
ax.grid(axis="x", color=GRID, lw=0.6); ax.tick_params(colors=INK2, labelsize=8)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(AX); ax.tick_params(axis="y", length=0)
handles = [Line2D([], [], color=TRAIN, lw=0, marker="o", ms=7, label="Train AUC"),
           Line2D([], [], color=TEST, lw=0, marker="o", ms=7, label="Test AUC"),
           Line2D([], [], color=AX, lw=1, ls="--", label="Chance (0.5)")]
ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
ax.set_title("ADARP personal stress models: train vs test AUC per participant\nSSL pretrain + fine-tune on all labels, chronological hold-out",
             loc="left", fontsize=10, color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/train_vs_test_auc.png", dpi=160, facecolor=SURF)
print("saved", f"{OUT}/train_vs_test_auc.png")
