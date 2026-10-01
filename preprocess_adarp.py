"""
ADARP preprocessing for per-participant (personal) stress classification.

Design decisions (agreed with the user):
  * Signal      : EDA only (4 Hz), 2nd-order Butterworth low-pass @ 1.25 Hz (paper).
  * Labels      : stress      = within +/-20 min of any button-press tag
                  discarded   = 20-60 min from the nearest tag (ambiguous buffer)
                  not-stress  = > 60 min from every tag
                  Tags are pooled per participant so a moment tagged on one wrist
                  is labelled identically on a second, concurrently worn device.
  * Windows     : 60 s, 50 % overlap; a window keeps only a pure label
                  (all samples stress, or all samples not-stress); mixed -> dropped.
  * Split       : per participant, random 70/30 by *session group*, seed 42.
                  Sessions that overlap in time (two devices) are always in the
                  same group -> no temporal leakage. Both splits are forced to
                  contain stress windows.
  * Normalising : EDA min-max to [0,1] fitted on TRAIN samples only, per participant.
                  Feature standardisation (StandardScaler) also fitted on train only.
  * Quality     : windows with mean EDA < 0.05 uS (off-body) are dropped.
  * Model       : LogisticRegression(class_weight='balanced') per participant.

Outputs -> processed/
  summary.csv, split_manifest.csv, results_logreg.csv
  P<id>/windows.npz            X_train, y_train, X_test, y_test (+ meta arrays)
  P<id>/features_train.csv     hand-crafted features per window
  P<id>/features_test.csv
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import skew, kurtosis, linregress

ROOT     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Sensor Data")
OUT      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed")
FS       = 4                       # EDA sampling rate
WIN_S    = 60                      # window length (s)
STEP_S   = 30                      # 50 % overlap
STRESS_S = 20 * 60                 # +/-20 min  -> stress
BUFFER_S = 60 * 60                 # 20-60 min  -> discarded
TEST_FRAC = 0.30
SEED     = 42
LP_CUT   = 1.25                    # Hz
LABEL_STRESS, LABEL_NOSTRESS, LABEL_DROP = 1, 0, -1
EDA_MIN_US = 0.05                  # window mean below this = off-body / no skin contact -> dropped (Kleckner et al. 2018)

os.makedirs(OUT, exist_ok=True)

# --------------------------------------------------------------------------- load
def load_eda(path):
    with open(path) as f:
        t0 = float(f.readline().strip())
        fs = float(f.readline().strip())
        x  = np.loadtxt(f, dtype=np.float64)
    assert fs == FS, (path, fs)
    return t0, x

def load_tags(path):
    with open(path) as f:
        return np.array([float(l.strip()) for l in f if l.strip()], dtype=np.float64)

sessions = []   # dicts
for part in sorted(os.listdir(ROOT)):
    if not part.startswith("Part"):
        continue
    pid = part.replace("Part ", "P")
    for sess in sorted(os.listdir(os.path.join(ROOT, part))):
        sd = os.path.join(ROOT, part, sess)
        if not os.path.isdir(sd):
            continue
        t0, eda = load_eda(os.path.join(sd, "EDA.csv"))
        tags = load_tags(os.path.join(sd, "tags.csv"))
        sessions.append(dict(pid=pid, sid=sess, device=sess.split("_")[0].upper(),
                             t0=t0, t1=t0 + len(eda) / FS, eda=eda, tags=tags))
print(f"loaded {len(sessions)} sessions from {len({s['pid'] for s in sessions})} participants")

# --------------------------------------------------------------------------- filter
b, a = butter(2, LP_CUT / (FS / 2), btype="low")
def lowpass(x):
    if len(x) < 3 * max(len(a), len(b)):
        return x.copy()
    return filtfilt(b, a, x)

# --------------------------------------------------------------------------- labels
def sample_labels(t0, n, tags_all):
    """label each EDA sample by distance to nearest participant tag."""
    t = t0 + np.arange(n) / FS
    if len(tags_all) == 0:
        return np.full(n, LABEL_NOSTRESS, dtype=np.int8)
    tags_sorted = np.sort(tags_all)
    idx = np.searchsorted(tags_sorted, t)
    d_prev = np.where(idx > 0, t - tags_sorted[np.clip(idx - 1, 0, len(tags_sorted) - 1)], np.inf)
    d_next = np.where(idx < len(tags_sorted), tags_sorted[np.clip(idx, 0, len(tags_sorted) - 1)] - t, np.inf)
    d = np.minimum(np.abs(d_prev), np.abs(d_next))
    lab = np.full(n, LABEL_NOSTRESS, dtype=np.int8)
    lab[d <= BUFFER_S] = LABEL_DROP
    lab[d <= STRESS_S] = LABEL_STRESS
    return lab

# --------------------------------------------------------------------------- windows
WIN, STEP = WIN_S * FS, STEP_S * FS
def make_windows(x, lab):
    starts = range(0, len(x) - WIN + 1, STEP)
    xs, ys, ss = [], [], []
    for s in starts:
        l = lab[s:s + WIN]
        if (l == LABEL_STRESS).all():
            y = LABEL_STRESS
        elif (l == LABEL_NOSTRESS).all():
            y = LABEL_NOSTRESS
        else:
            continue
        seg = x[s:s + WIN]
        if seg.mean() < EDA_MIN_US:           # off-body artifact
            n_offbody[0] += 1
            continue
        xs.append(seg); ys.append(y); ss.append(s)
    if not xs:
        return np.empty((0, WIN)), np.empty(0, dtype=np.int8), np.empty(0, dtype=np.int64)
    return np.stack(xs), np.array(ys, dtype=np.int8), np.array(ss)
n_offbody = [0]

# --------------------------------------------------------------------------- features
bh, ah = butter(1, 0.05 / (FS / 2), btype="high")   # phasic / tonic decomposition
def features(w):
    """hand-crafted EDA features on one 60 s window (filtered, un-normalised µS)."""
    d1 = np.diff(w)
    d2 = np.diff(d1)
    phasic = filtfilt(bh, ah, w)
    tonic  = w - phasic
    peaks, props = find_peaks(phasic, height=0.01, distance=FS)   # SCR peaks >= 0.01 µS, >= 1 s apart
    slope = linregress(np.arange(len(w)) / FS, w).slope
    return dict(
        eda_mean=w.mean(), eda_std=w.std(), eda_min=w.min(), eda_max=w.max(),
        eda_range=w.max() - w.min(), eda_median=np.median(w),
        eda_q25=np.percentile(w, 25), eda_q75=np.percentile(w, 75),
        eda_skew=np.nan_to_num(skew(w)), eda_kurt=np.nan_to_num(kurtosis(w)), eda_slope=slope,
        eda_d1_mean=d1.mean(), eda_d1_std=d1.std(), eda_d1_absmean=np.abs(d1).mean(),
        eda_d2_mean=d2.mean(), eda_d2_std=d2.std(),
        tonic_mean=tonic.mean(), tonic_std=tonic.std(), tonic_slope=linregress(np.arange(len(w)) / FS, tonic).slope,
        phasic_mean=phasic.mean(), phasic_std=phasic.std(), phasic_max=phasic.max(),
        phasic_auc=np.trapz(np.clip(phasic, 0, None), dx=1 / FS),
        scr_count=len(peaks),
        scr_amp_mean=props["peak_heights"].mean() if len(peaks) else 0.0,
        scr_amp_max=props["peak_heights"].max() if len(peaks) else 0.0,
    )

# --------------------------------------------------------------------------- split
def group_overlapping(sess_list):
    """union-find on time overlap -> group id per session."""
    n = len(sess_list); parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if sess_list[i]["t0"] < sess_list[j]["t1"] and sess_list[j]["t0"] < sess_list[i]["t1"]:
                parent[find(i)] = find(j)
    roots = {}
    return [roots.setdefault(find(i), len(roots)) for i in range(n)]

def split_groups(group_stats, rng):
    """group_stats: {gid: n_stress_windows}. returns set of test gids."""
    with_stress = [g for g, n in group_stats.items() if n > 0]
    without     = [g for g, n in group_stats.items() if n == 0]
    rng.shuffle(with_stress); rng.shuffle(without)
    n_test_s = max(1, int(round(TEST_FRAC * len(with_stress)))) if len(with_stress) >= 2 else 0
    n_test_s = min(n_test_s, len(with_stress) - 1)              # keep >= 1 stress group in train
    n_test_o = int(round(TEST_FRAC * len(without)))
    return set(with_stress[:n_test_s]) | set(without[:n_test_o])

# --------------------------------------------------------------------------- main loop
summary_rows, manifest_rows = [], []
per_pid = {}
for pid in sorted({s["pid"] for s in sessions}):
    sl = [s for s in sessions if s["pid"] == pid]
    tags_all = np.concatenate([s["tags"] for s in sl])
    gids = group_overlapping(sl)
    rng = np.random.RandomState(SEED)
    n_offbody[0] = 0

    # window every session
    for s, g in zip(sl, gids):
        s["gid"] = g
        x_f = lowpass(s["eda"])
        lab = sample_labels(s["t0"], len(x_f), tags_all)
        s["Xw"], s["yw"], s["starts"] = make_windows(x_f, lab)
        s["n_drop_samples"] = int((lab == LABEL_DROP).sum())
    group_stats = {}
    for s in sl:
        group_stats[s["gid"]] = group_stats.get(s["gid"], 0) + int((s["yw"] == 1).sum())
    test_gids = split_groups(group_stats, rng)

    for s in sl:
        s["split"] = "test" if s["gid"] in test_gids else "train"
        manifest_rows.append(dict(participant=pid, session=s["sid"], device=s["device"], group=s["gid"],
                                  split=s["split"], start_utc=pd.to_datetime(s["t0"], unit="s"),
                                  hours=round((s["t1"] - s["t0"]) / 3600, 2), n_tags=len(s["tags"]),
                                  n_windows=len(s["yw"]), n_stress_windows=int((s["yw"] == 1).sum()),
                                  n_nostress_windows=int((s["yw"] == 0).sum())))

    def stack(split):
        ss = [s for s in sl if s["split"] == split and len(s["yw"])]
        X = np.concatenate([s["Xw"] for s in ss]) if ss else np.empty((0, WIN))
        y = np.concatenate([s["yw"] for s in ss]) if ss else np.empty(0, dtype=np.int8)
        sid = np.concatenate([np.repeat(s["sid"], len(s["yw"])) for s in ss]) if ss else np.empty(0, dtype=str)
        t   = np.concatenate([s["t0"] + s["starts"] / FS for s in ss]) if ss else np.empty(0)
        return X, y, sid, t
    Xtr, ytr, sidtr, ttr = stack("train")
    Xte, yte, sidte, tte = stack("test")

    # per-participant min-max fitted on train only
    mn, mx = Xtr.min(), Xtr.max()
    Xtr_n = (Xtr - mn) / (mx - mn + 1e-12)
    Xte_n = (Xte - mn) / (mx - mn + 1e-12)

    pdir = os.path.join(OUT, pid); os.makedirs(pdir, exist_ok=True)
    np.savez_compressed(os.path.join(pdir, "windows.npz"),
                        X_train=Xtr_n.astype(np.float32), y_train=ytr, X_test=Xte_n.astype(np.float32), y_test=yte,
                        session_train=sidtr, session_test=sidte, t_start_train=ttr, t_start_test=tte,
                        norm_min=mn, norm_max=mx, fs=FS, window_s=WIN_S, step_s=STEP_S)
    for name, X, y, sid, t in (("train", Xtr, ytr, sidtr, ttr), ("test", Xte, yte, sidte, tte)):
        df = pd.DataFrame([features(w) for w in X])
        df.insert(0, "label", y); df.insert(0, "t_start_utc", t); df.insert(0, "session", sid)
        df.to_csv(os.path.join(pdir, f"features_{name}.csv"), index=False)

    summary_rows.append(dict(participant=pid, n_sessions=len(sl), n_groups=len(set(gids)),
                             n_test_groups=len(test_gids), n_tags=len(tags_all),
                             hours=round(sum(s["t1"] - s["t0"] for s in sl) / 3600, 1),
                             train_windows=len(ytr), train_stress=int(ytr.sum()),
                             test_windows=len(yte), test_stress=int(yte.sum()),
                             dropped_buffer_hours=round(sum(s["n_drop_samples"] for s in sl) / FS / 3600, 1),
                             dropped_offbody_windows=n_offbody[0]))
    print(f"{pid}: groups={len(set(gids))} test_groups={len(test_gids)} "
          f"train={len(ytr)} ({int(ytr.sum())} stress)  test={len(yte)} ({int(yte.sum())} stress)")

summary = pd.DataFrame(summary_rows); summary.to_csv(os.path.join(OUT, "summary.csv"), index=False)
pd.DataFrame(manifest_rows).to_csv(os.path.join(OUT, "split_manifest.csv"), index=False)
with open(os.path.join(OUT, "config.json"), "w") as f:
    json.dump(dict(fs=FS, window_s=WIN_S, step_s=STEP_S, stress_s=STRESS_S, buffer_s=BUFFER_S,
                   test_frac=TEST_FRAC, seed=SEED, lowpass_hz=LP_CUT, split="random by time-overlap session group",
                   normalisation="per-participant min-max fitted on train", eda_min_us=EDA_MIN_US), f, indent=2)

# --------------------------------------------------------------------------- leakage check
man = pd.DataFrame(manifest_rows)
for pid, g in man.groupby("participant"):
    for gid, gg in g.groupby("group"):
        assert gg["split"].nunique() == 1, f"group {gid} of {pid} spans both splits"
    tr = g[g.split == "train"]; te = g[g.split == "test"]
    assert set(tr.session).isdisjoint(te.session)
print("leakage check passed: every session group is entirely in train or entirely in test")

# --------------------------------------------------------------------------- logistic regression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

res = []
for pid in summary.participant:
    pdir = os.path.join(OUT, pid)
    tr = pd.read_csv(os.path.join(pdir, "features_train.csv")); te = pd.read_csv(os.path.join(pdir, "features_test.csv"))
    feats = [c for c in tr.columns if c not in ("session", "t_start_utc", "label")]
    Xtr, ytr = tr[feats].values, tr.label.values; Xte, yte = te[feats].values, te.label.values
    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0))
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]; yhat = (p >= 0.5).astype(int)
    res.append(dict(participant=pid, n_train=len(ytr), n_test=len(yte), test_stress=int(yte.sum()),
                    accuracy=accuracy_score(yte, yhat), balanced_acc=balanced_accuracy_score(yte, yhat),
                    f1_stress=f1_score(yte, yhat, pos_label=1, zero_division=0),
                    auc=roc_auc_score(yte, p) if len(np.unique(yte)) == 2 else np.nan))
res = pd.DataFrame(res)
res.loc[len(res)] = dict(participant="MEAN", n_train=res.n_train.sum(), n_test=res.n_test.sum(),
                         test_stress=res.test_stress.sum(), accuracy=res.accuracy.mean(),
                         balanced_acc=res.balanced_acc.mean(), f1_stress=res.f1_stress.mean(), auc=res.auc.mean())
res.to_csv(os.path.join(OUT, "results_logreg.csv"), index=False)
print(res.round(3).to_string(index=False))
