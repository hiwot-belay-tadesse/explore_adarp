"""
Personal stress models from EDA + HRV features with logistic regression, leave-days-out CV.
  * Windows : 60 s, 30 s stride, per session.
  * Labels  : stress = 10 min before a button press; otherwise within 60 min of a press -> discarded; else not-stress.
  * EDA     : per-session z-scored statistics + phasic features (SCR count / amplitude from a 0.05 Hz high-pass on raw uS).
  * HRV     : from IBI.csv beats inside the window (RR in 0.3-2.0 s): n_beats, meanRR, SDNN, RMSSD, pNN50, meanHR.
              Windows with < MIN_BEATS clean beats are dropped for ALL feature sets so the comparison is on identical windows.
  * Scaling : HRV features z-scored per session (removes day-level baseline), EDA already per-session z-scored.
  * Folds   : 5 contiguous chronological blocks of windows per participant; pooled out-of-fold test AUC.
Outputs -> processed_hrv_eda/{features.csv, results_per_person.csv, results_folds.csv, train_vs_test_auc.png}
"""
import os, numpy as np, pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "Sensor Data"); OUT = os.path.join(HERE, "processed_hrv_eda")
FS = 4; WIN_S, STEP_S = 60, 30; STRESS_BEFORE, BUFFER = 600, 3600; EDA_MIN_US = 0.05; MIN_BEATS = 30; N_FOLDS = 5
os.makedirs(OUT, exist_ok=True)
bh, ah = butter(1, 0.05 / (FS / 2), btype="high")

def load(p):
    with open(p) as f:
        t0 = float(f.readline().split(",")[0]); f.readline(); x = np.loadtxt(f)
    return t0, np.atleast_1d(x)
def load_ibi(p):
    with open(p) as f:
        head = f.readline().split(",")[0].strip()
        if not head: return np.empty(0), np.empty(0)                 # empty IBI file (no clean beats in session)
        t0 = float(head); arr = np.loadtxt(f, delimiter=",", ndmin=2)
    return (t0 + arr[:, 0], arr[:, 1]) if arr.size else (np.empty(0), np.empty(0))

def labels_at(times, tags):
    lab = np.zeros(len(times), np.int8)
    if len(tags):
        tags = np.sort(tags); i = np.searchsorted(tags, times)
        dprev = np.where(i > 0, times - tags[np.clip(i - 1, 0, len(tags) - 1)], np.inf)
        dnext = np.where(i < len(tags), tags[np.clip(i, 0, len(tags) - 1)] - times, np.inf)
        lab[(dprev <= BUFFER) | (dnext <= BUFFER)] = -1; lab[dnext <= STRESS_BEFORE] = 1
    return lab

def eda_feats(raw, z):
    d = np.diff(z); phasic = filtfilt(bh, ah, raw); tonic = raw - phasic
    peaks, props = find_peaks(phasic, height=0.01, distance=FS)
    t = np.arange(len(z)) / FS
    return dict(eda_mean=z.mean(), eda_std=z.std(), eda_range=z.max() - z.min(), eda_slope=np.polyfit(t, z, 1)[0],
                eda_absdiff=np.abs(d).mean(), eda_scr_count=len(peaks), eda_scr_amp_mean=props["peak_heights"].mean() if len(peaks) else 0.0,
                eda_scr_amp_max=props["peak_heights"].max() if len(peaks) else 0.0, eda_phasic_std=phasic.std(), eda_tonic_slope=np.polyfit(t, tonic, 1)[0])
def hrv_feats(rr):
    d = np.diff(rr)
    return dict(hrv_nbeats=len(rr), hrv_meanRR=rr.mean(), hrv_SDNN=rr.std(), hrv_RMSSD=np.sqrt((d ** 2).mean()) if len(d) else np.nan,
                hrv_pNN50=(np.abs(d) > 0.05).mean() if len(d) else np.nan, hrv_meanHR=60 / rr.mean())

rows = []
for part in sorted(os.listdir(ROOT)):
    if not part.startswith("Part"): continue
    pid = part.replace("Part ", "P"); sess = []
    for sid in sorted(os.listdir(f"{ROOT}/{part}")):
        sd = f"{ROOT}/{part}/{sid}"
        if not os.path.isdir(sd): continue
        t0, e = load(f"{sd}/EDA.csv"); bt, rr = load_ibi(f"{sd}/IBI.csv")
        ok = (rr > 0.3) & (rr < 2.0); bt, rr = bt[ok], rr[ok]
        tags = np.array([float(l) for l in open(f"{sd}/tags.csv") if l.strip()])
        sess.append(dict(sid=sid, t0=t0, e=e, bt=bt, rr=rr, tags=tags))
    tags_all = np.concatenate([s["tags"] for s in sess])
    n_win = n_lab = n_hrv = 0
    for s in sess:
        e = s["e"]
        if len(e) < WIN_S * FS: continue
        z = (e - e.mean()) / (e.std() + 1e-8)
        for i0 in range(0, len(e) - WIN_S * FS + 1, STEP_S * FS):
            n_win += 1
            seg = e[i0:i0 + WIN_S * FS]
            if seg.mean() < EDA_MIN_US: continue
            ws = s["t0"] + i0 / FS
            lab = labels_at(ws + np.arange(WIN_S * FS) / FS, tags_all)
            y = 1 if (lab == 1).all() else 0 if (lab == 0).all() else -1
            if y < 0: continue
            n_lab += 1
            m = (s["bt"] >= ws) & (s["bt"] < ws + WIN_S); rr = s["rr"][m]
            if len(rr) < MIN_BEATS: continue
            n_hrv += 1
            rows.append(dict(participant=pid, session=s["sid"], t_start=ws, label=y, **eda_feats(seg, z[i0:i0 + WIN_S * FS]), **hrv_feats(rr)))
    print(f"{pid}: windows {n_win}, labelled {n_lab}, with >= {MIN_BEATS} clean beats {n_hrv} ({100 * n_hrv / max(n_lab, 1):.0f}% HRV coverage)", flush=True)

df = pd.DataFrame(rows).sort_values(["participant", "t_start"]).reset_index(drop=True)
# per-session z-score of HRV features (EDA features are already from a per-session z-scored signal)
hrv_cols = [c for c in df.columns if c.startswith("hrv_")]; eda_cols = [c for c in df.columns if c.startswith("eda_")]
df[hrv_cols] = df.groupby("session")[hrv_cols].transform(lambda c: (c - c.mean()) / (c.std() + 1e-8))
df["fold"] = df.groupby("participant").cumcount() * N_FOLDS // df.groupby("participant")["label"].transform("size")
df.to_csv(f"{OUT}/features.csv", index=False)

SETS = {"EDA": eda_cols, "HRV": hrv_cols, "EDA+HRV": eda_cols + hrv_cols}
person_rows, fold_rows = [], []
for pid, g in df.groupby("participant"):
    rec = dict(participant=pid, n_windows=len(g), n_stress=int(g.label.sum()), n_tags=None)
    for name, cols in SETS.items():
        pp, py, tr_aucs = [], [], []
        for k in range(N_FOLDS):
            tr, te = g[g.fold != k], g[g.fold == k]
            if tr.label.nunique() < 2 or te.label.nunique() < 2:
                fold_rows.append(dict(participant=pid, features=name, fold=k, skipped=True)); continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000, C=0.5)).fit(tr[cols], tr.label)
            ptr, pte = clf.predict_proba(tr[cols])[:, 1], clf.predict_proba(te[cols])[:, 1]
            tr_aucs.append(roc_auc_score(tr.label, ptr)); pp.append(pte); py.append(te.label.values)
            fold_rows.append(dict(participant=pid, features=name, fold=k, n_test=len(te), n_test_stress=int(te.label.sum()),
                                  train_auc=tr_aucs[-1], test_auc=roc_auc_score(te.label, pte), skipped=False))
        rec[f"{name}_train_auc"] = np.mean(tr_aucs) if tr_aucs else np.nan
        rec[f"{name}_test_auc"] = roc_auc_score(np.concatenate(py), np.concatenate(pp)) if pp else np.nan
        rec[f"{name}_folds"] = len(tr_aucs)
    person_rows.append(rec)
pr = pd.DataFrame(person_rows).drop(columns="n_tags"); pr.loc[len(pr)] = dict(participant="MEAN", **pr.drop(columns="participant").mean(numeric_only=True))
pr.to_csv(f"{OUT}/results_per_person.csv", index=False); pd.DataFrame(fold_rows).to_csv(f"{OUT}/results_folds.csv", index=False)
print("\n" + pr[["participant", "n_windows", "n_stress"] + [c for c in pr.columns if "auc" in c]].round(3).to_string(index=False))

# ---- plot: train vs test AUC per person for EDA+HRV
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
p = pr[(pr.participant != "MEAN") & pr["EDA+HRV_test_auc"].notna()].sort_values("participant", ascending=False)
TRAIN, TEST = "#2a78d6", "#eb6834"; SURF, INK, INK2, GRID, AX = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
fig, ax = plt.subplots(figsize=(8, 0.55 * len(p) + 2.4), facecolor=SURF); ax.set_facecolor(SURF); y = np.arange(len(p))
ax.hlines(y, p["EDA+HRV_train_auc"], p["EDA+HRV_test_auc"], color=AX, lw=1.5, zorder=1)
ax.scatter(p["EDA+HRV_train_auc"], y, color=TRAIN, s=60, zorder=3, edgecolor=SURF, lw=1.5)
ax.scatter(p["EDA+HRV_test_auc"], y, color=TEST, s=60, zorder=3, edgecolor=SURF, lw=1.5)
for yi, tr, te in zip(y, p["EDA+HRV_train_auc"], p["EDA+HRV_test_auc"]):
    ax.text(tr, yi + 0.22, f"{tr:.2f}", va="bottom", ha="center", fontsize=8, color=INK2); ax.text(te, yi - 0.22, f"{te:.2f}", va="top", ha="center", fontsize=8, color=INK2)
ax.axvline(0.5, color=AX, lw=1, ls="--", zorder=0); ax.set_yticks(y); ax.set_yticklabels(p.participant, fontsize=9, color=INK); ax.tick_params(axis="y", length=0)
ax.set_xlim(0.25, 1.0); ax.set_ylim(-0.6, len(p) - 0.4); ax.set_xlabel("AUC", color=INK2, fontsize=9); ax.grid(axis="x", color=GRID, lw=0.6); ax.tick_params(colors=INK2, labelsize=8)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(AX)
handles = [Line2D([], [], color=TRAIN, lw=0, marker="o", ms=7, label="Train AUC (mean over folds)"), Line2D([], [], color=TEST, lw=0, marker="o", ms=7, label="Test AUC (pooled out-of-fold)"),
           Line2D([], [], color=AX, lw=1, ls="--", label="Chance (0.5)")]
fig.legend(handles=handles, loc="lower center", frameon=False, fontsize=8, labelcolor=INK, bbox_to_anchor=(0.5, 0.0))
ax.set_title("ADARP personal stress models: EDA + HRV features, logistic regression, leave-days-out CV\n60 s windows; stress = 10 min before press; per-session standardisation", loc="left", fontsize=10, color=INK)
fig.tight_layout(rect=(0, 0.10, 1, 1)); fig.savefig(f"{OUT}/train_vs_test_auc.png", dpi=160, facecolor=SURF); print("saved", f"{OUT}/train_vs_test_auc.png")
