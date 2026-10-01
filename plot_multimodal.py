"""Two panels: EDA-only vs EDA+HRV, SSL pretrain + fine-tune, train vs pooled test AUC per participant (leave-days-out CV, same windows)."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f"{HERE}/processed_cv_multimodal"
df = pd.read_csv(f"{OUT}/results_per_person.csv"); df = df[(df.participant != "MEAN")].sort_values("participant", ascending=False)
TRAIN, TEST = "#2a78d6", "#eb6834"; SURF, INK, INK2, GRID, AX = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
fig, axes = plt.subplots(1, 2, figsize=(11, 0.55 * len(df) + 2.6), sharey=True, facecolor=SURF)
for ax, name in zip(axes, ("EDA", "EDA+HRV")):
    tr, te = df[f"{name}_train_auc"], df[f"{name}_test_auc"]; y = np.arange(len(df)); ax.set_facecolor(SURF)
    ok = tr.notna()
    ax.hlines(y[ok], tr[ok], te[ok], color=AX, lw=1.5, zorder=1)
    ax.scatter(tr, y, color=TRAIN, s=60, zorder=3, edgecolor=SURF, lw=1.5); ax.scatter(te, y, color=TEST, s=60, zorder=3, edgecolor=SURF, lw=1.5)
    for yi, a, b in zip(y, tr, te):
        if np.isnan(a): ax.text(0.27, yi, "not evaluable (too few stress windows with clean pulse)", va="center", ha="left", fontsize=7, color=INK2); continue
        ax.text(a, yi + 0.22, f"{a:.2f}", va="bottom", ha="center", fontsize=8, color=INK2); ax.text(b, yi - 0.22, f"{b:.2f}", va="top", ha="center", fontsize=8, color=INK2)
    ax.axvline(0.5, color=AX, lw=1, ls="--", zorder=0); ax.set_yticks(y); ax.set_yticklabels(df.participant, fontsize=9, color=INK); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.25, 1.0); ax.set_ylim(-0.6, len(df) - 0.4); ax.set_xlabel("AUC", color=INK2, fontsize=9); ax.grid(axis="x", color=GRID, lw=0.6); ax.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AX)
    m = df.dropna(subset=[f"{name}_test_auc"])
    ax.set_title(f"{name}   (mean train {m[f'{name}_train_auc'].mean():.2f}, mean test {m[f'{name}_test_auc'].mean():.2f})", loc="left", fontsize=10, color=INK)
handles = [Line2D([], [], color=TRAIN, lw=0, marker="o", ms=7, label="Train AUC (mean over folds)"), Line2D([], [], color=TEST, lw=0, marker="o", ms=7, label="Test AUC (pooled out-of-fold)"),
           Line2D([], [], color=AX, lw=1, ls="--", label="Chance (0.5)")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8, labelcolor=INK, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("ADARP personal stress models: EDA vs EDA + HRV (RR tachogram), SSL pretrain + fine-tune, leave-days-out CV\n60 s windows with >= 30 clean beats; stress = 10 min before press; per-session z-score",
             x=0.01, ha="left", fontsize=10, color=INK)
fig.tight_layout(rect=(0, 0.07, 1, 0.92)); fig.savefig(f"{OUT}/eda_vs_eda_hrv_auc.png", dpi=160, facecolor=SURF); print("saved", f"{OUT}/eda_vs_eda_hrv_auc.png")
