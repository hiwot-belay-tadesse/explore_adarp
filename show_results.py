"""Print processed_paper_split/results.csv as readable tables, next to the paper's numbers.
Usage: python show_results.py [results.csv or directory]     (default: processed_paper_split/)"""
import os, sys, glob, pandas as pd

arg = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_paper_split")
path = os.path.join(arg, "results.csv") if os.path.isdir(arg) else arg
if not os.path.exists(path): raise SystemExit(f"no results yet: {path}")
df = pd.read_csv(path)
pd.set_option("display.width", 160, "display.float_format", lambda v: f"{v:.3f}")
METRICS = ["acc", "precision", "recall", "f1", "auc", "n", "n_stress"]
BAL = {"under": "undersampling", "smote": "SMOTE"}

print(f"\n{path}\n")
for bal in [b for b in ("under", "smote") if (df.balance == b).any()]:
    d = df[df.balance == bal]
    print(f"=== {BAL[bal]} ===")
    pooled = d[(d.scope == "pooled") & d["eval"].isin(["train", "test"])]
    if len(pooled):
        print(f"\nPooled model, all participants  ({int(pooled.epochs.iloc[0])} epochs, "
              f"{int(pooled.train_windows.iloc[0])} training windows, {pooled.minutes.iloc[0]:.0f} min)")
        t = pooled.set_index("eval")[METRICS]
        paper = {"under": dict(train=(0.9972, 0.99, 0.99, 0.99), test=(0.9833, 0.98, 0.98, 0.98)),
                 "smote": dict(train=(0.9992, 0.99, 0.99, 0.99), test=(0.8677, 0.99, 0.74, 0.84))}[bal]
        for k, (a, p, r, f) in paper.items():
            t.loc[f"{k} (paper)"] = [a, p, r, f, float("nan"), float("nan"), float("nan")]
        print(t.loc[["train", "train (paper)", "test", "test (paper)"]].to_string())
    per = d[d["eval"] == "test_per_participant"]
    if len(per):
        print("\nPooled model scored on each participant's test windows")
        print(per.set_index("eval_participant")[METRICS].to_string())
    personal = d[(d.scope != "pooled")]
    if len(personal):
        print("\nPersonal models (one per participant): train AUC vs test AUC")
        w = personal.pivot(index="scope", columns="eval", values="auc")[["train", "test"]]
        w.columns = ["train_auc", "test_auc"]
        extra = personal[personal["eval"] == "test"].set_index("scope")[["acc", "f1", "n", "n_stress"]]
        extra.columns = ["test_acc", "test_f1", "test_n", "test_n_stress"]
        print(w.join(extra).to_string())
    print()

hist = sorted(glob.glob(os.path.join(os.path.dirname(path), "history_*.csv")))
if hist:
    print("=== loss curves (first / best / last epoch) ===")
    rows = []
    for f in hist:
        h = pd.read_csv(f); name = os.path.basename(f)[8:-4]
        rows.append(dict(model=name, epochs=len(h), train_first=h.train_loss.iloc[0], train_last=h.train_loss.iloc[-1],
                         heldout_first=h.val_loss.iloc[0], heldout_min=h.val_loss.min(), heldout_min_epoch=int(h.val_loss.idxmin()) + 1,
                         heldout_last=h.val_loss.iloc[-1]))
    print(pd.DataFrame(rows).set_index("model").to_string(), "\n")
