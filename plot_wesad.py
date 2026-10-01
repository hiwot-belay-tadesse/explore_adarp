"""WESAD replication figures. Usage: python3 plot_wesad.py [subject]
  left : test RMSE vs number of labelled points, SSL vs purely supervised (paper Fig. 3 analogue), baseline-only run
  right: train and test AUC (stress vs baseline) vs number of labelled points, SSL model, baseline+TSST run"""
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wesad_replication")
SUBJ = sys.argv[1] if len(sys.argv) > 1 else None
SSL, SUP, TRAIN, TEST = "#2a78d6", "#eb6834", "#2a78d6", "#eb6834"; SURF, INK, INK2, GRID, AX = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
def style(ax):
    ax.set_facecolor(SURF); ax.set_xscale("log"); ax.grid(axis="y", color=GRID, lw=0.6); ax.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(AX)
    ax.set_xticks([5, 10, 20, 50, 100, 200, 500, 1000, 7000]); ax.set_xticklabels(["5", "10", "20", "50", "100", "200", "500", "1k", "all"], fontsize=8); ax.minorticks_off()
    ax.set_xlabel("labelled training windows", color=INK2, fontsize=9)
def curve(ax, df, col, color, label):
    g = df.groupby("n_labels")[col]; m, s = g.mean(), g.std().fillna(0)
    ax.plot(m.index, m.values, color=color, lw=2, marker="o", ms=4, label=label); ax.fill_between(m.index, m - s, m + s, color=color, alpha=0.12, lw=0)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURF)
f_base, f_b12 = f"{OUT}/label_curve_base.csv", f"{OUT}/label_curve_blocks12.csv"
subj_title = SUBJ or "all subjects"
if os.path.exists(f_base):
    d = pd.read_csv(f_base); d = d[d.subject == SUBJ] if SUBJ else d
    curve(axes[0], d[d.model == "ssl_finetune"], "test_rmse", SSL, "SSL pretrain + fine-tune"); curve(axes[0], d[d.model == "supervised"], "test_rmse", SUP, "Purely supervised")
    axes[0].set_ylabel("test RMSE (STAI scale 0.25-1)", color=INK2, fontsize=9); axes[0].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
axes[0].set_title(f"Paper's metric: test RMSE, baseline block only ({subj_title})", loc="left", fontsize=10, color=INK); style(axes[0])
if os.path.exists(f_b12):
    d = pd.read_csv(f_b12); d = d[d.subject == SUBJ] if SUBJ else d; d = d[d.model == "ssl_finetune"]
    curve(axes[1], d, "train_auc", TRAIN, "Train AUC"); curve(axes[1], d, "test_auc", TEST, "Test AUC")
    axes[1].axhline(0.5, color=AX, lw=1, ls="--"); axes[1].set_ylim(0.3, 1.02); axes[1].set_ylabel("AUC, stress (TSST) vs baseline", color=INK2, fontsize=9)
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
axes[1].set_title(f"Added metric: AUC, SSL model, baseline + TSST ({subj_title})", loc="left", fontsize=10, color=INK); style(axes[1])
fig.suptitle("WESAD replication of Islam & Washington (2023): six RespiBAN signals, 10 s windows, per-subject models\nShaded band = ±1 SD over 3 random label draws",
             x=0.01, ha="left", fontsize=10, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.90)); out = f"{OUT}/wesad_{SUBJ or 'all'}.png"; fig.savefig(out, dpi=160, facecolor=SURF); print("saved", out)
