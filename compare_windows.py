"""Compare 10 s vs 60 s windows: per-participant train/test AUC of the SSL pretrain + fine-tune model (all labels)."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN, TEST = "#2a78d6", "#eb6834"; SURF, INK, INK2, GRID, AX = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
runs = {}
for w in (10, 60):
    f = f"{HERE}/processed_{w}s/results_full_labels.csv"
    if os.path.exists(f):
        d = pd.read_csv(f); runs[w] = d[d.model == "ssl_finetune"].set_index("participant")[["train_auc", "test_auc", "test_f1", "n_labels", "n_test"]]
pids = sorted(set().union(*[set(r.index) for r in runs.values()]))
tab = pd.concat({f"{w}s": r.reindex(pids) for w, r in runs.items()}, axis=1)
tab.loc["MEAN"] = tab.mean(numeric_only=True)
tab.round(3).to_csv(f"{HERE}/compare_windows_auc.csv"); print(tab[[c for c in tab.columns if "auc" in c[1]]].round(3).to_string())

fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 0.45 * len(pids) + 1.8), sharey=True, facecolor=SURF, squeeze=False)
for ax, (w, r) in zip(axes[0], runs.items()):
    r = r.reindex(pids[::-1]); y = np.arange(len(r)); ax.set_facecolor(SURF)
    ok = r.train_auc.notna()
    ax.hlines(y[ok], r.train_auc[ok], r.test_auc[ok], color=AX, lw=1.5, zorder=1)
    ax.scatter(r.train_auc, y, color=TRAIN, s=55, zorder=3, edgecolor=SURF, lw=1.5)
    ax.scatter(r.test_auc, y, color=TEST, s=55, zorder=3, edgecolor=SURF, lw=1.5)
    for yi, tr, te in zip(y, r.train_auc, r.test_auc):
        if np.isnan(tr): continue
        lo, hi = min(tr, te), max(tr, te)
        ax.text(lo - 0.012, yi, f"{lo:.2f}", va="center", ha="right", fontsize=8, color=INK2)
        ax.text(hi + 0.012, yi, f"{hi:.2f}", va="center", ha="left", fontsize=8, color=INK2)
    ax.axvline(0.5, color=AX, lw=1, ls="--", zorder=0)
    ax.set_yticks(y); ax.set_yticklabels(r.index, fontsize=9, color=INK); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.25, 1.0); ax.set_xlabel("AUC", color=INK2, fontsize=9); ax.grid(axis="x", color=GRID, lw=0.6)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AX)
    m = r.dropna(); ax.set_title(f"{w} s windows   (mean train {m.train_auc.mean():.2f}, mean test {m.test_auc.mean():.2f})", loc="left", fontsize=10, color=INK)
handles = [Line2D([], [], color=TRAIN, lw=0, marker="o", ms=7, label="Train AUC"), Line2D([], [], color=TEST, lw=0, marker="o", ms=7, label="Test AUC"),
           Line2D([], [], color=AX, lw=1, ls="--", label="Chance (0.5)")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("ADARP personal stress models: 10 s vs 60 s windows, SSL pretrain + fine-tune on all labels, chronological hold-out",
             x=0.01, ha="left", fontsize=10, color=INK)
fig.tight_layout(rect=(0, 0.07, 1, 0.94)); fig.savefig(f"{HERE}/compare_windows_auc.png", dpi=160, facecolor=SURF)
print("saved", f"{HERE}/compare_windows_auc.png")
