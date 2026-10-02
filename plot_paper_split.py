"""Training loss vs held-out (test) loss per epoch for every model in processed_paper_split/history_*.csv.
One small panel per model (pooled + each participant), one row per balancing method. Run any time; plots whatever exists."""
import os, glob, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "processed_paper_split")
TRAIN, TEST = "#2a78d6", "#eb6834"; SURF, INK, INK2, GRID, AX = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
hist = {}
for f in sorted(glob.glob(os.path.join(OUT, "history_*.csv"))):
    scope, bal = os.path.basename(f)[8:-4].rsplit("_", 1); hist[(bal, scope)] = pd.read_csv(f)
if not hist: raise SystemExit("no history files yet")
bals = [b for b in ("under", "smote") if any(k[0] == b for k in hist)]
scopes = ["pooled"] + sorted({k[1] for k in hist} - {"pooled"})
ncol = min(6, len(scopes)); nrow_per = int(np.ceil(len(scopes) / ncol))
fig, axes = plt.subplots(nrow_per * len(bals), ncol, figsize=(max(7, 2.4 * ncol + 1), 2.0 * nrow_per * len(bals) + 1.9),
                         facecolor=SURF, squeeze=False, sharex=True)
ymax = max(max(h.train_loss.max(), h.val_loss.max()) for h in hist.values()) * 1.05
for bi, bal in enumerate(bals):
    for si in range(nrow_per * ncol):
        ax = axes[bi * nrow_per + si // ncol, si % ncol]; ax.set_facecolor(SURF)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color(AX)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.tick_params(colors=INK2, labelsize=7, length=0); ax.set_ylim(0, ymax)
        if si >= len(scopes): ax.axis("off"); continue
        scope = scopes[si]; h = hist.get((bal, scope))
        title = f"{'all participants' if scope == 'pooled' else scope} · {'undersampling' if bal == 'under' else 'SMOTE'}"
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        if h is None:
            ax.text(0.5, 0.5, "not run yet", transform=ax.transAxes, ha="center", va="center", fontsize=8, color=INK2); continue
        ax.plot(h.epoch, h.train_loss, color=TRAIN, lw=2); ax.plot(h.epoch, h.val_loss, color=TEST, lw=2)
        ax.text(h.epoch.iloc[-1], h.train_loss.iloc[-1], f" {h.train_loss.iloc[-1]:.2f}", fontsize=7, color=INK2, va="center")
        ax.text(h.epoch.iloc[-1], h.val_loss.iloc[-1], f" {h.val_loss.iloc[-1]:.2f}", fontsize=7, color=INK2, va="center")
        ax.set_xlim(1, h.epoch.max() * 1.12)
        if si % ncol == 0: ax.set_ylabel("cross-entropy", fontsize=7, color=INK2)
for ax in axes[-1]: ax.set_xlabel("epoch", fontsize=7, color=INK2)
fig.legend(handles=[plt.Line2D([], [], color=TRAIN, lw=2, label="training loss (balanced train set)"),
                    plt.Line2D([], [], color=TEST, lw=2, label="held-out loss (test windows, natural class ratio)")],
           loc="upper left", fontsize=8, frameon=False, ncol=1, bbox_to_anchor=(0.01, 0.955))
fig.suptitle("ADARP paper protocol: random split by window, paper CNN on EDA", fontsize=10, color=INK, x=0.01, ha="left", y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.86)); fig.savefig(os.path.join(OUT, "loss_curves.png"), dpi=150, facecolor=SURF)
print("wrote", os.path.join(OUT, "loss_curves.png"), "models:", len(hist))
