"""One plot: mean train AUC vs pooled out-of-fold test AUC per participant, leave-days-out CV, SSL pretrain + fine-tune."""
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
WIN_S = int(sys.argv[1]) if len(sys.argv) > 1 else 10
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"processed_cv_{WIN_S}s")
df = pd.read_csv(f"{OUT}/results_per_person.csv"); df = df[df.participant != "MEAN"].sort_values("participant", ascending=False)
TRAIN, TEST = "#2a78d6", "#eb6834"; SURF, INK, INK2, GRID, AX = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
fig, ax = plt.subplots(figsize=(8, 0.55 * len(df) + 2.4), facecolor=SURF); ax.set_facecolor(SURF); y = np.arange(len(df))
ax.hlines(y, df.train_auc, df.test_auc_pooled, color=AX, lw=1.5, zorder=1)
ax.errorbar(df.test_auc_pooled, y, xerr=df.test_auc_fold_sd.fillna(0), fmt="none", ecolor=TEST, elinewidth=1, alpha=0.5, capsize=2, zorder=2)
ax.scatter(df.train_auc, y, color=TRAIN, s=60, zorder=3, edgecolor=SURF, lw=1.5)
ax.scatter(df.test_auc_pooled, y, color=TEST, s=60, zorder=3, edgecolor=SURF, lw=1.5)
for yi, tr, te in zip(y, df.train_auc, df.test_auc_pooled):
    # train value above its dot, test value below its dot: never collides with error bars or names
    ax.text(tr, yi + 0.22, f"{tr:.2f}", va="bottom", ha="center", fontsize=8, color=INK2)
    ax.text(te, yi - 0.22, f"{te:.2f}", va="top", ha="center", fontsize=8, color=INK2)
ax.axvline(0.5, color=AX, lw=1, ls="--", zorder=0)
ax.set_yticks(y); ax.set_yticklabels(df.participant, fontsize=9, color=INK); ax.tick_params(axis="y", length=0)
ax.set_xlim(0.25, 1.0); ax.set_xlabel("AUC", color=INK2, fontsize=9); ax.grid(axis="x", color=GRID, lw=0.6); ax.tick_params(colors=INK2, labelsize=8)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(AX)
handles = [Line2D([], [], color=TRAIN, lw=0, marker="o", ms=7, label="Train AUC (mean over folds)"),
           Line2D([], [], color=TEST, lw=0, marker="o", ms=7, label="Test AUC (pooled out-of-fold; bar = ±1 SD across folds)"),
           Line2D([], [], color=AX, lw=1, ls="--", label="Chance (0.5)")]
fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False, fontsize=8, labelcolor=INK, bbox_to_anchor=(0.5, 0.0))
ax.set_ylim(-0.6, len(df) - 0.4)
ax.set_title(f"ADARP personal stress models, leave-days-out CV ({WIN_S} s windows)\nSSL pretrain + fine-tune; stress = 10 min before press; per-session z-score",
             loc="left", fontsize=10, color=INK)
fig.tight_layout(rect=(0, 0.16 if len(df) < 6 else 0.10, 1, 1)); fig.savefig(f"{OUT}/train_vs_test_auc_cv.png", dpi=160, facecolor=SURF); print("saved", f"{OUT}/train_vs_test_auc_cv.png")
