"""
Usage: python3 cv_ssl_multimodal.py [n_folds]
Islam & Washington SSL pipeline with TWO modalities (EDA + HRV tachogram), leave-days-out CV per participant.
  * Windows  : 60 s, 30 s stride. Labels: stress = 10 min before press; otherwise within 60 min -> discarded; else not-stress.
  * EDA      : raw 4 Hz, per-session z-score.
  * HRV      : IBI beats (RR in 0.3-2.0 s) -> RR-interval tachogram linearly interpolated to 4 Hz, per-session z-score.
               Windows with < MIN_BEATS clean beats are dropped for both models so they are compared on identical windows.
  * Model    : per modality a 1D-CNN encoder (Table 1, kernels scaled to 240 samples) pretrained with the forecasting pretext task
               on all training windows; fine-tune head = concat(frozen encoders) -> Dense30 -> Dense30 -> sigmoid (late fusion).
               'EDA' model uses the EDA encoder alone; 'EDA+HRV' uses both. Same folds, same windows.
  * Metrics  : per person mean train AUC over folds and pooled out-of-fold test AUC.
Outputs -> processed_cv_multimodal/
"""
import os, sys, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras import layers as L, models as M
from sklearn.metrics import roc_auc_score

N_FOLDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "Sensor Data"); OUT = os.path.join(HERE, "processed_cv_multimodal")
FS = 4; WIN_S = 60; WIN = WIN_S * FS; STEP = WIN // 2; HORIZON = WIN // 10
STRESS_BEFORE, BUFFER, EDA_MIN_US, MIN_BEATS = 600, 3600, 0.05, 30
K = [max(2, round(k * WIN / 40)) for k in (8, 6, 4, 6)]
PRE_EPOCHS, FT_EPOCHS = 8, 15
os.makedirs(OUT, exist_ok=True); tf.random.set_seed(0); np.random.seed(0); tf.config.threading.set_intra_op_parallelism_threads(6)

def load(p):
    with open(p) as f:
        t0 = float(f.readline().split(",")[0]); f.readline(); x = np.loadtxt(f)
    return t0, np.atleast_1d(x)
def load_ibi(p):
    with open(p) as f:
        head = f.readline().split(",")[0].strip()
        if not head: return np.empty(0), np.empty(0)
        t0 = float(head); arr = np.loadtxt(f, delimiter=",", ndmin=2)
    if not arr.size: return np.empty(0), np.empty(0)
    bt, rr = t0 + arr[:, 0], arr[:, 1]; ok = (rr > 0.3) & (rr < 2.0); return bt[ok], rr[ok]
def labels_at(times, tags):
    lab = np.zeros(len(times), np.int8)
    if len(tags):
        tags = np.sort(tags); i = np.searchsorted(tags, times)
        dprev = np.where(i > 0, times - tags[np.clip(i - 1, 0, len(tags) - 1)], np.inf); dnext = np.where(i < len(tags), tags[np.clip(i, 0, len(tags) - 1)] - times, np.inf)
        lab[(dprev <= BUFFER) | (dnext <= BUFFER)] = -1; lab[dnext <= STRESS_BEFORE] = 1
    return lab

# ---------------------------------------------------------------- build windows
recs = []   # per participant: X_eda, Yf_eda, X_hrv, Yf_hrv, y, t
for part in sorted(os.listdir(ROOT)):
    if not part.startswith("Part"): continue
    pid = part.replace("Part ", "P"); sess = []
    for sid in sorted(os.listdir(f"{ROOT}/{part}")):
        sd = f"{ROOT}/{part}/{sid}"
        if not os.path.isdir(sd): continue
        t0, e = load(f"{sd}/EDA.csv"); bt, rr = load_ibi(f"{sd}/IBI.csv")
        sess.append(dict(sid=sid, t0=t0, e=e, bt=bt, rr=rr, tags=np.array([float(l) for l in open(f"{sd}/tags.csv") if l.strip()])))
    tags_all = np.concatenate([s["tags"] for s in sess])
    Xe, Ye, Xh, Yh, Y, T = [], [], [], [], [], []
    n_lab = 0
    for s in sess:
        e = s["e"]; n = len(e)
        if n < WIN + HORIZON or len(s["rr"]) < 2: continue
        te = s["t0"] + np.arange(n) / FS
        ez = (e - e.mean()) / (e.std() + 1e-8)
        tach = np.interp(te, s["bt"], s["rr"])                       # RR tachogram on the EDA time grid
        tz = (tach - s["rr"].mean()) / (s["rr"].std() + 1e-8)
        beat_count = np.searchsorted(s["bt"], te)                     # cumulative beats up to each sample
        lab_all = labels_at(te, tags_all)
        for i0 in range(0, n - WIN - HORIZON + 1, STEP):
            seg = e[i0:i0 + WIN]
            if seg.mean() < EDA_MIN_US: continue
            l = lab_all[i0:i0 + WIN]; y = 1 if (l == 1).all() else 0 if (l == 0).all() else -1
            nb = beat_count[i0 + WIN - 1] - beat_count[i0]
            if nb < MIN_BEATS: continue                                # need real beats inside the window for the tachogram to be meaningful
            if y >= 0: n_lab += 1
            Xe.append(ez[i0:i0 + WIN]); Ye.append(ez[i0 + WIN:i0 + WIN + HORIZON]); Xh.append(tz[i0:i0 + WIN]); Yh.append(tz[i0 + WIN:i0 + WIN + HORIZON])
            Y.append(y); T.append(te[i0])
    recs.append(dict(pid=pid, Xe=np.array(Xe, np.float32)[..., None], Ye=np.array(Ye, np.float32), Xh=np.array(Xh, np.float32)[..., None], Yh=np.array(Yh, np.float32),
                     y=np.array(Y, np.int8), t=np.array(T), n_tags=len(tags_all)))
    print(f"{pid}: {len(Y)} windows with >= {MIN_BEATS} beats, {n_lab} labelled, {int((np.array(Y) == 1).sum())} stress", flush=True)

# ---------------------------------------------------------------- models
def encoder():
    inp = L.Input((WIN, 1)); x = inp
    for filt, k in zip((4, 2, 4, 2), K):
        x = L.Conv1D(filt, k, padding="same")(x); x = L.LeakyReLU()(x)
    return M.Model(inp, x)
def pretrain(X, Yf):
    enc = encoder(); inp = L.Input((WIN, 1)); x = L.Flatten()(enc(inp))
    x = L.Dense(50)(x); x = L.LeakyReLU()(x); x = L.Dense(30)(x); x = L.LeakyReLU()(x)
    m = M.Model(inp, L.Dense(HORIZON)(x)); m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    m.fit(X, Yf, epochs=PRE_EPOCHS, batch_size=256, verbose=0); enc.trainable = False; return enc
def head(encs):
    inps = [L.Input((WIN, 1)) for _ in encs]
    feats = [L.Flatten()(enc(i)) for enc, i in zip(encs, inps)]
    x = L.Concatenate()(feats) if len(feats) > 1 else feats[0]
    x = L.Dense(30)(x); x = L.LeakyReLU()(x); x = L.Dense(30)(x); x = L.LeakyReLU()(x)
    m = M.Model(inps, L.Dense(1, activation="sigmoid")(x)); m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy"); return m

# ---------------------------------------------------------------- CV
person_rows, fold_rows = [], []
for r in recs:
    pid, y = r["pid"], r["y"]; t_start = time.time()
    order = np.argsort(r["t"]); fold = np.minimum(np.arange(len(y)) * N_FOLDS // len(y), N_FOLDS - 1)[np.argsort(order)]  # contiguous chronological blocks
    lab = y >= 0
    res = {"EDA": dict(pp=[], py=[], tr=[]), "EDA+HRV": dict(pp=[], py=[], tr=[])}
    for k in range(N_FOLDS):
        tr, te = fold != k, fold == k; trL, teL = tr & lab, te & lab
        if len(np.unique(y[teL])) < 2 or len(np.unique(y[trL])) < 2: continue
        enc_e = pretrain(r["Xe"][tr], r["Ye"][tr]); enc_h = pretrain(r["Xh"][tr], r["Yh"][tr])
        ytr = y[trL].astype(int); cw = {0: len(ytr) / (2 * (ytr == 0).sum()), 1: len(ytr) / (2 * (ytr == 1).sum())}
        for name, encs, Xs in (("EDA", [enc_e], [r["Xe"]]), ("EDA+HRV", [enc_e, enc_h], [r["Xe"], r["Xh"]])):
            m = head(encs); m.fit([X[trL] for X in Xs], ytr, epochs=FT_EPOCHS, batch_size=128, verbose=0, class_weight=cw)
            ptr = m.predict([X[trL] for X in Xs], batch_size=4096, verbose=0).ravel(); pte = m.predict([X[teL] for X in Xs], batch_size=4096, verbose=0).ravel()
            res[name]["tr"].append(roc_auc_score(ytr, ptr)); res[name]["pp"].append(pte); res[name]["py"].append(y[teL])
            fold_rows.append(dict(participant=pid, model=name, fold=k, n_test=int(teL.sum()), n_test_stress=int((y[teL] == 1).sum()), train_auc=res[name]["tr"][-1], test_auc=roc_auc_score(y[teL], pte)))
        print(f"{pid} fold {k}: EDA test {fold_rows[-2]['test_auc']:.3f} | EDA+HRV test {fold_rows[-1]['test_auc']:.3f}", flush=True)
    rec = dict(participant=pid, n_tags=r["n_tags"], n_stress=int((y == 1).sum()), n_nostress=int((y == 0).sum()), folds=len(res["EDA"]["tr"]))
    for name in res:
        ok = len(res[name]["tr"]) > 0
        rec[f"{name}_train_auc"] = np.mean(res[name]["tr"]) if ok else np.nan
        rec[f"{name}_test_auc"] = roc_auc_score(np.concatenate(res[name]["py"]), np.concatenate(res[name]["pp"])) if ok else np.nan
    person_rows.append(rec); print(f"{pid}: EDA train {rec['EDA_train_auc']:.3f} test {rec['EDA_test_auc']:.3f} | EDA+HRV train {rec['EDA+HRV_train_auc']:.3f} test {rec['EDA+HRV_test_auc']:.3f} ({time.time() - t_start:.0f}s)", flush=True)
    pd.DataFrame(person_rows).to_csv(f"{OUT}/results_per_person.csv", index=False); pd.DataFrame(fold_rows).to_csv(f"{OUT}/results_folds.csv", index=False)
pr = pd.DataFrame(person_rows); pr.loc[len(pr)] = dict(participant="MEAN", **pr.drop(columns="participant").mean(numeric_only=True))
pr.to_csv(f"{OUT}/results_per_person.csv", index=False); print("\n=== per person ===\n" + pr.round(3).to_string(index=False))
