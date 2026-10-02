"""
Exact re-implementation of the authors' pipeline (github.com/rameshKrSah/ADARP_Dataset: data_proc_adarp.py,
data_loader.py, utils.py, stress_models.py), EDA only.

  Stress     : per recording, each tag with >= 30 min of data on both sides; segment = position -/+ 4800 samples (20 min),
               position = int(int(tag - t0) * 4)  [their integer-second truncation]. Butterworth(2, 1.25 Hz) filtfilt,
               then min-max to [0,1] PER SEGMENT. 60 s windows, step 30 s (segment_sensor_reading).
  Not-stress : per recording, the stretches [start, tag1-60min], [tag1+60min, tag2-60min], ..., [tagN+60min, end]
               (whole recording when it has no tags). Filter + min-max PER STRETCH, same windowing.
  Table I    : balance_classes=True  -> not-stress undersampled WITH replacement (np.random.randint) to n_stress BEFORE
               the split; train_test_split(test_size=0.3, random_state=42, stratify).
  Table II   : balance_classes=False -> split first, SMOTE(k=5) on the training set only.
  Model      : get_supervised_full_adarp_model: Conv1D(250,5,same) Conv1D(100,5,same) GlobalMaxPool Dense256 Drop.1
               Dense128 Drop.1 Dense64 Softmax2, categorical CE. Code docstring: Adam lr 0.01157, batch 100 ("code");
               paper text: lr 0.001, batch 32 ("paper"). 50 epochs (evaluate_model default).
  Control    : NORM=recording keeps everything else but min-max scales each whole recording once, so stress and
               not-stress windows share a scale. This isolates the per-segment normalisation artefact.
Env: VARIANTS=comma list of under_code,under_paper,smote_code,smote_paper ; NORM=segment|recording ; EPOCHS.
Outputs -> processed_authors/results.csv, history_<variant>_<norm>.csv, windows_<norm>.npz (cache)
"""
import os, time
import tensorflow as tf
import numpy as np, pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "Sensor Data"); OUT = os.path.join(HERE, "processed_authors")
os.makedirs(OUT, exist_ok=True)
FS, WIN, STEP = 4, 240, 120
NORM = os.environ.get("NORM", "segment"); EPOCHS = int(os.environ.get("EPOCHS", 50))
VARIANTS = os.environ.get("VARIANTS", "under_code,under_paper,smote_code").split(",")
B, A = butter(2, 1.25 / (FS / 2), btype="low")                       # filters.lpf(order=2, fs=4, cutoff=1.25)

def normalization(x):                                                  # preprocessing.normalization
    x = x - np.min(x); return x / (np.max(x) - np.min(x)) if np.max(x) > 0 else x

def segment_sensor_reading(v):                                         # utils.segment_sensor_reading, 60 s, 50 %
    if len(v) < WIN: return np.empty((0, WIN), np.float32)
    s, e, segs = 0, WIN, []
    while True:
        segs.append(v[s:e]); s += STEP; e += STEP
        if s > len(v) or e > len(v): break
    return np.array(segs, np.float32)

def build_windows():
    cache = os.path.join(OUT, f"windows_{NORM}.npz")
    if os.path.exists(cache):
        d = np.load(cache); return d["S"], d["N"], d["S_pid"], d["N_pid"]
    S, N, S_pid, N_pid = [], [], [], []
    for part in sorted(os.listdir(ROOT)):
        if not part.startswith("Part"): continue
        pid = part.replace("Part ", "P")
        for sid in sorted(os.listdir(os.path.join(ROOT, part))):
            sd = os.path.join(ROOT, part, sid)
            if not os.path.isdir(sd): continue
            with open(os.path.join(sd, "EDA.csv")) as f:
                t0 = float(f.readline()); f.readline(); x = np.atleast_1d(np.loadtxt(f)).astype(float)
            n = len(x); tags = [float(l) for l in open(os.path.join(sd, "tags.csv")) if l.strip()]
            if n < WIN: continue
            prep = (lambda seg: normalization(filtfilt(B, A, seg))) if NORM == "segment" else None
            if NORM == "recording":                                    # control: one scale per recording
                xr = normalization(filtfilt(B, A, x)); prep = lambda seg: seg
                xs = xr
            else: xs = x
            end_time = t0 + n / FS
            for tg in tags:                                            # extract_segments_around_tags
                if not (t0 <= tg <= end_time): continue
                pos = int(int(tg - t0) * FS)
                if pos - 1800 * FS < 0 or pos + 1800 * FS > n: continue
                fr, to = pos - 4800, pos + 4800
                if fr < 0 or to > n: continue
                w = segment_sensor_reading(prep(xs[fr:to])); S.append(w); S_pid += [pid] * len(w)
            if len(tags) == 0: stretches = [xs]                        # get_segments_between_timestamps
            else:
                bounds = [t0] + list(tags) + [t0 + n / FS]; stretches = []
                for a, b in zip(bounds[:-1], bounds[1:]):
                    h = int((a - t0) * FS + 3600 * FS); t = int((b - t0) * FS - 3600 * FS)
                    if t - h > 0: stretches.append(xs[h:t])
            for st in stretches:
                if len(st) < WIN: continue
                w = segment_sensor_reading(prep(st)); N.append(w); N_pid += [pid] * len(w)
    S, N = np.concatenate(S), np.concatenate(N); S_pid, N_pid = np.array(S_pid), np.array(N_pid)
    np.savez_compressed(cache, S=S, N=N, S_pid=S_pid, N_pid=N_pid); return S, N, S_pid, N_pid

def smote(X, y, rng, k=5):
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]; mi, ma = (i1, i0) if len(i1) <= len(i0) else (i0, i1)
    Xm = X[mi]; n_new = len(ma) - len(mi)
    nbr = NearestNeighbors(n_neighbors=k + 1).fit(Xm).kneighbors(Xm, return_distance=False)[:, 1:]
    base = rng.integers(0, len(Xm), n_new); pick = nbr[base, rng.integers(0, k, n_new)]
    lam = rng.random((n_new, 1)).astype(np.float32)
    Xa = np.concatenate([X, Xm[base] + lam * (Xm[pick] - Xm[base])]); ya = np.concatenate([y, np.full(n_new, y[mi[0]])])
    p = rng.permutation(len(ya)); return Xa[p], ya[p]

def model(lr):                                                         # stress_models.get_supervised_full_adarp_model
    m = tf.keras.Sequential([tf.keras.Input((WIN, 1)),
        tf.keras.layers.Conv1D(250, 5, activation="relu", padding="same"), tf.keras.layers.Conv1D(100, 5, activation="relu", padding="same"),
        tf.keras.layers.GlobalMaxPool1D(), tf.keras.layers.Dense(256, activation="relu"), tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation="relu"), tf.keras.layers.Dropout(0.1), tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax")])
    m.compile(tf.keras.optimizers.Adam(lr), "categorical_crossentropy"); return m

def score(m, X, y):
    p = m.predict(X, batch_size=1024, verbose=0)[:, 1]; yh = (p >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, yh, average="binary", zero_division=0)
    return dict(acc=accuracy_score(y, yh), precision=pr, recall=rc, f1=f1, auc=roc_auc_score(y, p), n=len(y), n_stress=int(y.sum()))

S, N, S_pid, N_pid = build_windows()
print(f"NORM={NORM}: stress windows {len(S)}, not-stress windows {len(N)}", flush=True)
RES = os.path.join(OUT, "results.csv"); rows = pd.read_csv(RES).to_dict("records") if os.path.exists(RES) else []
for v in VARIANTS:
    if any(r["variant"] == v and r["norm"] == NORM and r["eval"] == "test" for r in rows): print("skip", v); continue
    bal, hp = v.split("_"); lr, bs = (0.01157, 100) if hp == "code" else (0.001, 32)
    np.random.seed(0); rng = np.random.default_rng(0); tf.keras.utils.set_random_seed(0)
    Nn, Np = N, N_pid
    if bal == "under":                                                 # utils.select_random_samples: WITH replacement, before split
        idx = np.random.randint(0, len(N), len(S)); Nn, Np = N[idx], N_pid[idx]
    X = np.concatenate([S, Nn]); y = np.concatenate([np.ones(len(S), int), np.zeros(len(Nn), int)]); pid = np.concatenate([S_pid, Np])
    Xtr, Xte, ytr, yte, _, pte = train_test_split(X, y, pid, test_size=0.3, random_state=42, shuffle=True, stratify=y)
    if bal == "smote": Xtr, ytr = smote(Xtr, ytr, rng)
    Xtr, Xte = Xtr[..., None], Xte[..., None]
    t0 = time.time(); m = model(lr)
    h = m.fit(Xtr, tf.keras.utils.to_categorical(ytr, 2), epochs=EPOCHS, batch_size=bs, verbose=0,
              validation_data=(Xte, tf.keras.utils.to_categorical(yte, 2)), validation_batch_size=1024,
              callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda e, l: print(f"  {v} {NORM} epoch {e+1} loss {l['loss']:.4f} val_loss {l['val_loss']:.4f} ({time.time()-t0:.0f} s)", flush=True) if (e + 1) % 10 == 0 or e == 0 else None)])
    pd.DataFrame(dict(epoch=np.arange(1, EPOCHS + 1), train_loss=h.history["loss"], val_loss=h.history["val_loss"])).to_csv(os.path.join(OUT, f"history_{v}_{NORM}.csv"), index=False)
    base = dict(variant=v, norm=NORM, lr=lr, batch=bs, epochs=EPOCHS, train_windows=len(ytr), minutes=round((time.time() - t0) / 60, 1))
    rows.append(dict(base, eval="train", eval_participant="all", **score(m, Xtr, ytr)))
    rte = score(m, Xte, yte); rows.append(dict(base, eval="test", eval_participant="all", **rte))
    for p in np.unique(pte):
        mk = pte == p
        if len(np.unique(yte[mk])) > 1: rows.append(dict(base, eval="test_per_participant", eval_participant=p, **score(m, Xte[mk], yte[mk])))
    print(f"{v:12s} {NORM:9s} train n={len(ytr)} test acc={rte['acc']:.3f} prec={rte['precision']:.3f} rec={rte['recall']:.3f} f1={rte['f1']:.3f} auc={rte['auc']:.3f} ({base['minutes']} min)", flush=True)
    pd.DataFrame(rows).to_csv(RES, index=False)
print("done", flush=True)
