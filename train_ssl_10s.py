"""
Per-participant stress models following Islam & Washington (2023), adapted to ADARP EDA.
  Stage 1 (Tp): self-supervised forecasting pretrain on ALL train windows (labelled + unlabelled).
  Stage 2 (Td): freeze conv layers, train dense head on stress labels (binary, BCE).
  Baseline    : identical architecture, purely supervised from random init.
  Experiments : full labels, and label-efficiency curve with n in N_LABELS (3 repeats each).
Inputs: z-scored on train statistics per participant. Reports train AUC / test AUC per participant -> processed_10s/results_*.csv
"""
import os, sys, json, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras import layers as L, models as M
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "processed_10s")
WIN, HORIZON = 40, 4
N_LABELS = [5, 10, 20, 50, 100, 200, 500]
REPEATS = 3
PRE_EPOCHS, FULL_EPOCHS, FEW_EPOCHS = 10, 15, 60
tf.random.set_seed(0); np.random.seed(0)
tf.config.threading.set_intra_op_parallelism_threads(8)

def conv_stack(inp):                              # Table 1 conv block, kernels scaled 7000 -> 40
    x = L.Conv1D(4, 8, padding="same")(inp); x = L.LeakyReLU()(x)
    x = L.Conv1D(2, 6, padding="same")(x);   x = L.LeakyReLU()(x)
    x = L.Conv1D(4, 4, padding="same")(x);   x = L.LeakyReLU()(x)
    x = L.Conv1D(2, 6, padding="same")(x);   x = L.LeakyReLU()(x)
    return x

def pretrain_model():
    inp = L.Input((WIN, 1)); x = L.Flatten()(conv_stack(inp))
    x = L.Dense(50)(x); x = L.LeakyReLU()(x); x = L.Dense(30)(x); x = L.LeakyReLU()(x)
    return M.Model(inp, L.Dense(HORIZON, activation="linear")(x))

def classifier(conv_weights=None, freeze=False):
    inp = L.Input((WIN, 1)); feats = conv_stack(inp)
    enc = M.Model(inp, feats)
    if conv_weights is not None:
        enc.set_weights(conv_weights)
    enc.trainable = not freeze
    x = L.Flatten()(enc(inp))
    x = L.Dense(30)(x); x = L.LeakyReLU()(x); x = L.Dense(30)(x); x = L.LeakyReLU()(x)
    m = M.Model(inp, L.Dense(1, activation="sigmoid")(x))
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return m

def evaluate(m, Xtr, ytr, Xte, yte):
    ptr = m.predict(Xtr, batch_size=2048, verbose=0).ravel(); pte = m.predict(Xte, batch_size=2048, verbose=0).ravel()
    return dict(train_auc=roc_auc_score(ytr, ptr) if len(np.unique(ytr)) == 2 else np.nan,
                test_auc=roc_auc_score(yte, pte),
                test_f1=f1_score(yte, pte >= 0.5, zero_division=0),
                test_bal_acc=balanced_accuracy_score(yte, pte >= 0.5))

def fit(m, X, y, epochs, batch, class_weight=None, seed=0):
    tf.random.set_seed(seed)
    m.fit(X, y, epochs=epochs, batch_size=batch, verbose=0, shuffle=True, class_weight=class_weight)
    return m

def stratified_subset(y, n, rng):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    npos = min(len(pos), max(1, n // 2)); nneg = min(len(neg), n - npos)
    return np.concatenate([rng.choice(pos, npos, replace=False), rng.choice(neg, nneg, replace=False)])

full_rows, few_rows = [], []
pids = sorted(d for d in os.listdir(OUT) if d.startswith("P"))
for pid in pids:
    t_start = time.time()
    d = np.load(os.path.join(OUT, pid, "data.npz"), allow_pickle=True)
    # z-score standardisation on TRAIN statistics (per participant); min-max squashed EDA to ~0.02 and the CNN could not fit
    mu, sd = d["X_train"].mean(), d["X_train"].std()
    z = lambda a: ((a - mu) / sd).astype(np.float32)
    Xall = z(d["X_train"])[..., None]; Yf = z(d["Yf_train"])
    lab_tr = d["y_train"] >= 0; Xtr, ytr = Xall[lab_tr], d["y_train"][lab_tr].astype(int)
    lab_te = d["y_test"] >= 0;  Xte, yte = z(d["X_test"])[..., None][lab_te], d["y_test"][lab_te].astype(int)
    cw = {0: len(ytr) / (2 * (ytr == 0).sum()), 1: len(ytr) / (2 * (ytr == 1).sum())}

    # ---- Stage 1: SSL forecasting pretrain on every train window (incl. unlabelled)
    pm = pretrain_model(); pm.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    hist = pm.fit(Xall, Yf, epochs=PRE_EPOCHS, batch_size=256, verbose=0, validation_split=0.1)
    conv_w = M.Model(pm.input, pm.layers[8].output).get_weights()   # weights of the 4 conv layers
    pre_rmse = float(np.sqrt(hist.history["val_loss"][-1]))

    # ---- Stage 2, full labels: SSL fine-tune vs purely supervised
    for name, mk in (("ssl_finetune", lambda: classifier(conv_w, freeze=True)), ("supervised", lambda: classifier())):
        m = fit(mk(), Xtr, ytr, FULL_EPOCHS, 128, class_weight=cw)
        r = evaluate(m, Xtr, ytr, Xte, yte)
        full_rows.append(dict(participant=pid, model=name, n_labels=len(ytr), n_train_stress=int(ytr.sum()),
                              n_test=len(yte), n_test_stress=int(yte.sum()), pretrain_val_rmse=pre_rmse, **r))
        print(f"{pid} {name:13s} full  train_auc={r['train_auc']:.3f} test_auc={r['test_auc']:.3f} f1={r['test_f1']:.3f}", flush=True)

    # ---- label-efficiency curve
    for n in N_LABELS:
        for rep in range(REPEATS):
            rng = np.random.RandomState(1000 * rep + n)
            idx = stratified_subset(ytr, n, rng)
            for name, mk in (("ssl_finetune", lambda: classifier(conv_w, freeze=True)), ("supervised", lambda: classifier())):
                m = fit(mk(), Xtr[idx], ytr[idx], FEW_EPOCHS, min(16, len(idx)), seed=rep)
                r = evaluate(m, Xtr[idx], ytr[idx], Xte, yte)
                few_rows.append(dict(participant=pid, model=name, n_labels=len(idx), repeat=rep, **r))
    pd.DataFrame(full_rows).to_csv(os.path.join(OUT, "results_full_labels.csv"), index=False)
    pd.DataFrame(few_rows).to_csv(os.path.join(OUT, "results_label_efficiency.csv"), index=False)
    print(f"{pid} done in {time.time() - t_start:.0f}s", flush=True)

full = pd.DataFrame(full_rows)
piv = full.pivot(index="participant", columns="model", values=["train_auc", "test_auc"]).round(3)
piv.to_csv(os.path.join(OUT, "results_auc_per_person.csv"))
print("\n=== AUC per person (full labels) ===\n", piv.to_string())
few = pd.DataFrame(few_rows)
curve = few.groupby(["model", "n_labels"])[["train_auc", "test_auc"]].mean().round(3)
curve.to_csv(os.path.join(OUT, "results_label_efficiency_mean.csv"))
print("\n=== mean over participants and repeats: AUC vs number of labels ===\n", curve.unstack(0).to_string())
