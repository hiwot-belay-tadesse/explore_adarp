"""
Usage: python3 cv_ssl.py <window_seconds> [n_folds]      e.g. python3 cv_ssl.py 10 5
Leave-days-out cross-validation of the Islam & Washington SSL pipeline on ADARP EDA, per participant.
  * Labels   : stress = 10 min BEFORE each button press; anything else within 60 min of a press is discarded;
               > 60 min from every press = not-stress. Tags pooled per participant.
  * Scaling  : per-session z-score (each E4 session standardised on its own mean / std) to remove day-level drift.
  * Folds    : sessions grouped by time overlap, ordered chronologically, cut into n_folds contiguous blocks of days.
               Each block is held out once; SSL pretraining and fine-tuning use only the other blocks.
  * Metrics  : per fold train/test AUC; per person mean train AUC and POOLED test AUC over all out-of-fold predictions.
Outputs -> processed_cv_<win>s/
"""
import os, sys, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras import layers as L, models as M
from sklearn.metrics import roc_auc_score

WIN_S = int(sys.argv[1]) if len(sys.argv) > 1 else 10
N_FOLDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "Sensor Data"); OUT = os.path.join(HERE, f"processed_cv_{WIN_S}s")
FS = 4; WIN = WIN_S * FS; STEP = WIN // 2; HORIZON = WIN // 10
STRESS_BEFORE_S, BUFFER_S, EDA_MIN_US = 10 * 60, 60 * 60, 0.05
K = [max(2, round(k * WIN / 40)) for k in (8, 6, 4, 6)]
PRE_EPOCHS, FT_EPOCHS = 6, 12
os.makedirs(OUT, exist_ok=True); tf.random.set_seed(0); np.random.seed(0)
tf.config.threading.set_intra_op_parallelism_threads(6)

# ------------------------------------------------------------------ data
def load_eda(p):
    with open(p) as f:
        t0 = float(f.readline()); fs = float(f.readline()); x = np.loadtxt(f)
    assert fs == FS; return t0, np.atleast_1d(x)

def sample_labels(t0, n, tags):
    """1 = within 10 min before a tag, -1 = otherwise within 60 min of a tag, 0 = far from all tags"""
    t = t0 + np.arange(n) / FS
    lab = np.zeros(n, np.int8)
    if len(tags):
        tags = np.sort(tags); i = np.searchsorted(tags, t)
        dprev = np.where(i > 0, t - tags[np.clip(i - 1, 0, len(tags) - 1)], np.inf)        # time since last tag  (>=0)
        dnext = np.where(i < len(tags), tags[np.clip(i, 0, len(tags) - 1)] - t, np.inf)     # time until next tag (>=0)
        lab[(dprev <= BUFFER_S) | (dnext <= BUFFER_S)] = -1
        lab[dnext <= STRESS_BEFORE_S] = 1
    return lab

def windows(x, lab):
    X, Yf, Lb, S = [], [], [], []
    mu, sd = x.mean(), x.std() + 1e-8                                  # per-session z-score
    for s in range(0, len(x) - WIN - HORIZON + 1, STEP):
        seg = x[s:s + WIN]
        if seg.mean() < EDA_MIN_US: continue
        l = lab[s:s + WIN]; y = 1 if (l == 1).all() else 0 if (l == 0).all() else -1
        X.append((seg - mu) / sd); Yf.append((x[s + WIN:s + WIN + HORIZON] - mu) / sd); Lb.append(y); S.append(s)
    if not X: return np.empty((0, WIN), np.float32), np.empty((0, HORIZON), np.float32), np.empty(0, np.int8), np.empty(0, int)
    return np.array(X, np.float32), np.array(Yf, np.float32), np.array(Lb, np.int8), np.array(S)

def groups_by_overlap(sl):
    n = len(sl); par = list(range(n))
    def f(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if sl[i]["t0"] < sl[j]["t1"] and sl[j]["t0"] < sl[i]["t1"]: par[f(i)] = f(j)
    return [f(i) for i in range(n)]

sessions = []
for part in sorted(os.listdir(ROOT)):
    if not part.startswith("Part"): continue
    pid = part.replace("Part ", "P")
    for sid in sorted(os.listdir(os.path.join(ROOT, part))):
        sd = os.path.join(ROOT, part, sid)
        if not os.path.isdir(sd): continue
        t0, x = load_eda(os.path.join(sd, "EDA.csv"))
        tags = np.array([float(l) for l in open(os.path.join(sd, "tags.csv")) if l.strip()])
        sessions.append(dict(pid=pid, sid=sid, t0=t0, t1=t0 + len(x) / FS, x=x, tags=tags))

# ------------------------------------------------------------------ models (Tables 1 & 2 of the paper, kernels scaled)
def conv_stack(inp):
    x = inp
    for filt, k in zip((4, 2, 4, 2), K):
        x = L.Conv1D(filt, k, padding="same")(x); x = L.LeakyReLU()(x)
    return x
def pretrain_model():
    inp = L.Input((WIN, 1)); x = L.Flatten()(conv_stack(inp))
    x = L.Dense(50)(x); x = L.LeakyReLU()(x); x = L.Dense(30)(x); x = L.LeakyReLU()(x)
    m = M.Model(inp, L.Dense(HORIZON)(x)); m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse"); return m
def finetune_model(conv_w):
    inp = L.Input((WIN, 1)); enc = M.Model(inp, conv_stack(inp)); enc.set_weights(conv_w); enc.trainable = False
    x = L.Flatten()(enc(inp)); x = L.Dense(30)(x); x = L.LeakyReLU()(x); x = L.Dense(30)(x); x = L.LeakyReLU()(x)
    m = M.Model(inp, L.Dense(1, activation="sigmoid")(x)); m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy"); return m

# ------------------------------------------------------------------ CV
fold_rows, person_rows, manifest = [], [], []
for pid in sorted({s["pid"] for s in sessions}):
    t_start = time.time()
    sl = sorted([s for s in sessions if s["pid"] == pid], key=lambda s: s["t0"])
    tags_all = np.concatenate([s["tags"] for s in sl])
    for s, g in zip(sl, groups_by_overlap(sl)):
        s["g"] = g; s["X"], s["Yf"], s["L"], s["S"] = windows(s["x"], sample_labels(s["t0"], len(s["x"]), tags_all))
    gids = sorted({s["g"] for s in sl}, key=lambda g: min(s["t0"] for s in sl if s["g"] == g))
    gwin = {g: sum(len(s["L"]) for s in sl if s["g"] == g) for g in gids}
    # contiguous chronological blocks with ~equal window counts
    cum = np.cumsum([gwin[g] for g in gids]); edges = np.linspace(0, cum[-1], N_FOLDS + 1)[1:-1]
    fold_of_group = {g: int(np.searchsorted(edges, c, side="left")) for g, c in zip(gids, cum)}
    for s in sl:
        s["fold"] = fold_of_group[s["g"]]
        manifest.append(dict(participant=pid, session=s["sid"], group=s["g"], fold=s["fold"], start_utc=pd.to_datetime(s["t0"], unit="s"),
                             n_tags=len(s["tags"]), n_windows=len(s["L"]), n_stress=int((s["L"] == 1).sum()), n_nostress=int((s["L"] == 0).sum())))
    X = np.concatenate([s["X"] for s in sl if len(s["L"])])[..., None]; Yf = np.concatenate([s["Yf"] for s in sl if len(s["L"])])
    y = np.concatenate([s["L"] for s in sl if len(s["L"])]); fold = np.concatenate([np.repeat(s["fold"], len(s["L"])) for s in sl if len(s["L"])])
    pooled_p, pooled_y, train_aucs, n_eval = [], [], [], 0
    for k in range(N_FOLDS):
        te, tr = fold == k, fold != k
        ytr_lab, yte_lab = (y == 1) | (y == 0), (y == 1) | (y == 0)
        trL, teL = tr & ytr_lab, te & yte_lab
        if len(np.unique(y[teL])) < 2 or len(np.unique(y[trL])) < 2:
            fold_rows.append(dict(participant=pid, fold=k, n_test=int(teL.sum()), n_test_stress=int((y[teL] == 1).sum()), skipped=True)); continue
        pm = pretrain_model(); pm.fit(X[tr], Yf[tr], epochs=PRE_EPOCHS, batch_size=256, verbose=0)
        conv_w = M.Model(pm.input, pm.layers[8].output).get_weights()
        ytr = y[trL].astype(int); cw = {0: len(ytr) / (2 * (ytr == 0).sum()), 1: len(ytr) / (2 * (ytr == 1).sum())}
        m = finetune_model(conv_w); m.fit(X[trL], ytr, epochs=FT_EPOCHS, batch_size=128, verbose=0, class_weight=cw)
        ptr = m.predict(X[trL], batch_size=4096, verbose=0).ravel(); pte = m.predict(X[teL], batch_size=4096, verbose=0).ravel()
        tr_auc, te_auc = roc_auc_score(ytr, ptr), roc_auc_score(y[teL], pte)
        train_aucs.append(tr_auc); pooled_p.append(pte); pooled_y.append(y[teL]); n_eval += 1
        fold_rows.append(dict(participant=pid, fold=k, n_train=int(trL.sum()), n_train_stress=int((y[trL] == 1).sum()), n_test=int(teL.sum()),
                              n_test_stress=int((y[teL] == 1).sum()), train_auc=tr_auc, test_auc=te_auc, skipped=False))
        print(f"{pid} fold {k}: train_auc={tr_auc:.3f} test_auc={te_auc:.3f} (test stress windows {int((y[teL]==1).sum())})", flush=True)
    pp, py = np.concatenate(pooled_p), np.concatenate(pooled_y)
    fr = pd.DataFrame([r for r in fold_rows if r["participant"] == pid and not r["skipped"]])
    person_rows.append(dict(participant=pid, folds_evaluated=n_eval, n_tags=len(tags_all), n_stress_windows=int((y == 1).sum()),
                            n_nostress_windows=int((y == 0).sum()), train_auc=np.mean(train_aucs), test_auc_pooled=roc_auc_score(py, pp),
                            test_auc_fold_mean=fr.test_auc.mean(), test_auc_fold_sd=fr.test_auc.std()))
    print(f"{pid}: mean train AUC {np.mean(train_aucs):.3f}  pooled test AUC {roc_auc_score(py, pp):.3f}  ({time.time()-t_start:.0f}s)", flush=True)
    pd.DataFrame(fold_rows).to_csv(f"{OUT}/results_folds.csv", index=False)
    pd.DataFrame(person_rows).to_csv(f"{OUT}/results_per_person.csv", index=False)
    pd.DataFrame(manifest).to_csv(f"{OUT}/fold_manifest.csv", index=False)
pr = pd.DataFrame(person_rows); pr.loc[len(pr)] = dict(participant="MEAN", **pr.drop(columns="participant").mean(numeric_only=True))
pr.to_csv(f"{OUT}/results_per_person.csv", index=False)
print("\n=== per person ===\n", pr.round(3).to_string(index=False))
