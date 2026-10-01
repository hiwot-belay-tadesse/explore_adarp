"""
Train the ADARP paper's CNN (Alam et al. 2022, Sec. III-B, Fig. 5) on the random-by-window split made by prep_paper_split.py.

  Model   : Conv1D(250, k5, ReLU) -> Conv1D(100, k5, ReLU) -> GlobalMaxPool -> Dense 256 -> Dropout 0.1 -> Dense 128
            -> Dropout 0.1 -> Dense 64 -> Softmax(2); Adam 1e-3, categorical cross-entropy, 50 epochs, batch 32.
  Balance : (a) majority-class random undersampling, (b) SMOTE (k = 5) on the minority class; both applied to the
            training windows only (the test set keeps its natural class ratio).
  Scope   : one pooled model over all participants (as in the paper) and one model per participant, same test windows.
  Metrics : accuracy, precision, recall, F1 (argmax) and ROC AUC on the balanced training set and on the test set;
            pooled models are additionally scored on each participant's test windows. Per-epoch train loss and
            test ('validation') loss go to history_<scope>_<balance>.csv; the test set is only monitored, never used
            for model selection (the paper has no separate validation split).
Resumable: finished (scope, balance) rows in results.csv are skipped.  Env: EPOCHS, ONLY (e.g. ONLY=P101C), BALANCE.
"""
import os, sys, time, json
import tensorflow as tf                                # import TF before pandas (symbol clash on this machine)
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "processed_paper_split")
EPOCHS, BATCH, LR, SEED = int(os.environ.get("EPOCHS", 50)), 32, 1e-3, 0
ONLY = os.environ.get("ONLY"); BALANCES = os.environ.get("BALANCE", "under,smote").split(",")
RES = os.path.join(OUT, "results.csv")
tf.keras.utils.set_random_seed(SEED)

d = np.load(os.path.join(OUT, "data.npz"))
X, y, pid, split = d["X"][..., None], d["y"].astype(int), d["pid"], d["split"]
pids = sorted(np.unique(pid))

def build():
    m = tf.keras.Sequential([tf.keras.Input((X.shape[1], 1)),
        tf.keras.layers.Conv1D(250, 5, activation="relu"), tf.keras.layers.Conv1D(100, 5, activation="relu"),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(256, activation="relu"), tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation="relu"), tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(64, activation="relu"), tf.keras.layers.Dense(2, activation="softmax")])
    m.compile(tf.keras.optimizers.Adam(LR), "categorical_crossentropy")
    return m

def undersample(Xt, yt, rng):
    i1, i0 = np.where(yt == 1)[0], np.where(yt == 0)[0]
    minority, majority = (i1, i0) if len(i1) <= len(i0) else (i0, i1)
    keep = np.concatenate([minority, rng.choice(majority, len(minority), replace=False)])
    rng.shuffle(keep); return Xt[keep], yt[keep]

def smote(Xt, yt, rng, k=5):
    i1, i0 = np.where(yt == 1)[0], np.where(yt == 0)[0]
    minority, majority = (i1, i0) if len(i1) <= len(i0) else (i0, i1)
    Xm = Xt[minority, :, 0]; n_new = len(majority) - len(minority)
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(Xm))).fit(Xm)
    nbr = nn.kneighbors(Xm, return_distance=False)[:, 1:]
    base = rng.integers(0, len(Xm), n_new); pick = nbr[base, rng.integers(0, nbr.shape[1], n_new)]
    lam = rng.random((n_new, 1)).astype(np.float32)
    Xs = Xm[base] + lam * (Xm[pick] - Xm[base])
    Xa = np.concatenate([Xt, Xs[..., None]]); ya = np.concatenate([yt, np.full(n_new, yt[minority[0]])])
    perm = rng.permutation(len(ya)); return Xa[perm], ya[perm]

def score(m, Xe, ye):
    p = m.predict(Xe, batch_size=1024, verbose=0)[:, 1]; yh = (p >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(ye, yh, average="binary", zero_division=0)
    return dict(acc=accuracy_score(ye, yh), precision=pr, recall=rc, f1=f1, auc=roc_auc_score(ye, p) if len(set(ye)) > 1 else np.nan,
                n=len(ye), n_stress=int((ye == 1).sum()))

done = pd.read_csv(RES) if os.path.exists(RES) else pd.DataFrame()
rows = done.to_dict("records")
def finished(scope, bal):
    return len(done) and ((done.scope == scope) & (done.balance == bal) & (done.eval == "test")).any()

jobs = [(s, b) for b in BALANCES for s in (["pooled"] + pids)]
if ONLY: jobs = [(s, b) for s, b in jobs if s == ONLY]
for scope, bal in jobs:
    if finished(scope, bal): print(f"skip {scope} {bal}", flush=True); continue
    sel = np.ones(len(y), bool) if scope == "pooled" else pid == scope
    tr, te = sel & (split == "train"), sel & (split == "test")
    rng = np.random.default_rng(SEED)
    Xb, yb = (undersample if bal == "under" else smote)(X[tr], y[tr], rng)
    t0 = time.time(); m = build()
    Xte, yte1h = X[te], tf.keras.utils.to_categorical(y[te], 2)
    class Log(tf.keras.callbacks.Callback):
        def on_epoch_end(s, e, logs):
            if (e + 1) % 10 == 0 or e == 0: print(f"  {scope} {bal} epoch {e + 1} loss {logs['loss']:.4f} val_loss {logs['val_loss']:.4f} ({time.time() - t0:.0f} s)", flush=True)
    h = m.fit(Xb, tf.keras.utils.to_categorical(yb, 2), epochs=EPOCHS, batch_size=BATCH, verbose=0, callbacks=[Log()],
              validation_data=(Xte, yte1h), validation_batch_size=1024)
    pd.DataFrame(dict(epoch=np.arange(1, EPOCHS + 1), train_loss=h.history["loss"], val_loss=h.history["val_loss"])).to_csv(
        os.path.join(OUT, f"history_{scope}_{bal}.csv"), index=False)
    base = dict(scope=scope, balance=bal, epochs=EPOCHS, train_windows=len(yb), minutes=round((time.time() - t0) / 60, 1))
    rows.append(dict(base, eval="train", eval_participant="all" if scope == "pooled" else scope, **score(m, Xb, yb)))
    rte = score(m, X[te], y[te]); rows.append(dict(base, eval="test", eval_participant="all" if scope == "pooled" else scope, **rte))
    if scope == "pooled":
        for p in pids:
            mp = te & (pid == p); rows.append(dict(base, eval="test_per_participant", eval_participant=p, **score(m, X[mp], y[mp])))
    print(f"{scope:7s} {bal:5s} train n={len(yb)}  test acc={rte['acc']:.3f} prec={rte['precision']:.3f} rec={rte['recall']:.3f} "
          f"f1={rte['f1']:.3f} auc={rte['auc']:.3f}  ({base['minutes']} min)", flush=True)
    done = pd.DataFrame(rows); done.to_csv(RES, index=False)
print("done", flush=True)
