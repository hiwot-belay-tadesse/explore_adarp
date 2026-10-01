"""
Usage: python3 wesad_replicate.py [subject ...]    env BLOCKS=1 (baseline only, paper) | BLOCKS=1,2 (baseline+TSST, enables AUC); THREADS=8
Replication of Islam & Washington (2023) on WESAD RespiBAN data, memory-lean: windows are sliced from the raw signal
on the fly from numpy per batch, so only the raw 700 Hz signals (~140 MB) live in memory.
  Signals : chest ECG, EDA, EMG, Resp, Temp, ACC(3ch) at 700 Hz; one encoder per modality (Table 1: kernels 40/30/18/30, filters 4/2/4/2,
            LeakyReLU, Dense 50 -> 30 -> 40 forecast outputs). Windows 7000 samples (10 s), step 100, forecast next 40 samples.
  Split   : baseline only -> first 7000 windows train, last 910 test (paper). baseline+TSST -> last 11.5 % of EACH block is test.
  Labels  : six STAI items of the block mapped 1..4 -> 0.25..1; regression with MSE; one 6-output head.
  SSL     : pretrain each encoder on forecasting over all train windows; fine-tune head (Dense 30 -> 30 -> 6) on frozen encoders.
  Baseline: identical architecture trained purely supervised. Label efficiency: n in N_LABELS random train windows, REPEATS repeats.
  Metrics : train/test RMSE per item (paper); with two blocks also train/test AUC of predicted mean STAI vs stress block.
Outputs -> wesad_replication/label_curve_<tag>.csv (saved after every step), subjects_<tag>.csv, enc_<subj>_<tag>_<mod>.weights.h5 (pretrained encoders, reused on rerun)
  AUC uses the STAI anxiety total: nervous + jittery + worried minus at-ease + relaxed + pleasant (reverse-coded positive items).
"""
import os, sys, time, pickle
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np, tensorflow as tf
from tensorflow.keras import layers as L, models as M
import pandas as pd   # imported after TensorFlow: pyarrow/absl symbol clash otherwise deadlocks TF
from sklearn.metrics import roc_auc_score

ROOT = "/Users/hiwotbelaytadesse/WESAD"; OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wesad_replication"); os.makedirs(OUT, exist_ok=True)
BLOCKS = [int(b) for b in os.environ.get("BLOCKS", "1").split(",")]; BLOCK_NAME = {1: "Base", 2: "TSST", 3: "Fun", 4: "Medi 1"}
TAG = "base" if BLOCKS == [1] else "blocks" + "".join(map(str, BLOCKS))
FS, W, P, O = 700, 7000, 40, 100; N_TRAIN, N_TEST = 7000, 910
MODS = ["ECG", "EDA", "EMG", "Resp", "Temp", "ACC"]
N_LABELS = [5, 10, 20, 50, 100, 200, 500, 1000, N_TRAIN]; REPEATS = 3
PRE_EPOCHS, FT_EPOCHS_FEW, FT_EPOCHS_FULL, BATCH = 5, 30, 8, 32
SUBJECTS = sys.argv[1:] or [f"S{i}" for i in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17)]
tf.random.set_seed(0); np.random.seed(0); tf.config.threading.set_intra_op_parallelism_threads(int(os.environ.get("THREADS", "8")))
AR_W = np.arange(W); AR_P = np.arange(W, W + P)

def stai_labels(subj):
    lines = [l.strip().split(";") for l in open(f"{ROOT}/{subj}/{subj}_quest.csv")]
    order = [c for c in next(l for l in lines if l[0] == "# ORDER")[1:] if c]
    stai = [l[1:7] for l in lines if l[0] == "# STAI"]
    return {blk: np.array([float(v) / 4 for v in row], np.float32) for blk, row in zip(order, stai)}

# ---------------------------------------------------------------- on-the-fly window batches (numpy slicing, no tf.data)
class WindowSeq(tf.keras.utils.PyDataset):
    """yields ((B,W,ch) per modality, target) sliced from the raw signals; mode 'forecast' uses one signal and the next P samples as target"""
    def __init__(self, sigs, starts, targets=None, shuffle=False, seed=0, mode="multi", **kw):
        super().__init__(**kw); self.sigs, self.starts, self.targets, self.shuffle, self.mode = sigs, np.asarray(starts), targets, shuffle, mode
        self.rng = np.random.RandomState(seed); self.order = np.arange(len(self.starts)); self.on_epoch_end()
    def __len__(self): return int(np.ceil(len(self.starts) / BATCH))
    def on_epoch_end(self):
        if self.shuffle: self.rng.shuffle(self.order)
    def __getitem__(self, i):
        idx = self.order[i * BATCH:(i + 1) * BATCH]; st = self.starts[idx]; win = st[:, None] + AR_W
        if self.mode == "forecast":
            return self.sigs[win], self.sigs[st[:, None] + AR_P][..., 0]
        xs = tuple(sig[win] for sig in self.sigs)
        return xs, (self.targets[idx] if self.targets is not None else np.zeros(len(idx), np.float32))
WORKERS = int(os.environ.get("WORKERS", "6")); EVAL_TRAIN_MAX = 2000      # train metrics on a fixed 2000-window sample when the fit set is larger
def win_ds(sigs, starts, targets=None, shuffle=False, seed=0): return WindowSeq(sigs, starts, targets, shuffle, seed, "multi", workers=WORKERS, use_multiprocessing=False, max_queue_size=16)
def forecast_ds(sig, starts, shuffle=True, seed=0): return WindowSeq(sig, starts, None, shuffle, seed, "forecast", workers=WORKERS, use_multiprocessing=False, max_queue_size=16)

# ---------------------------------------------------------------- models
def encoder(ch):
    inp = L.Input((W, ch)); x = inp
    for f, k in zip((4, 2, 4, 2), (40, 30, 18, 30)):
        x = L.Conv1D(f, k, padding="same")(x); x = L.LeakyReLU()(x)
    return M.Model(inp, x)
def pretrain(sig, starts):
    enc = encoder(sig.shape[1]); inp = L.Input((W, sig.shape[1])); x = L.Flatten()(enc(inp))
    x = L.Dense(50)(x); x = L.LeakyReLU()(x); x = L.Dense(30)(x); x = L.LeakyReLU()(x)
    m = M.Model(inp, L.Dense(P)(x)); m.compile(optimizer="adam", loss="mse")
    h = m.fit(forecast_ds(sig, starts), epochs=PRE_EPOCHS, verbose=0); enc.trainable = False
    return enc, float(np.sqrt(h.history["loss"][-1]))
def head(chs, encs=None, n_out=6):
    inps = [L.Input((W, c)) for c in chs]; encs = encs or [encoder(c) for c in chs]
    x = L.Concatenate()([L.Flatten()(e(i)) for e, i in zip(encs, inps)])
    x = L.Dense(30)(x); x = L.LeakyReLU()(x); x = L.Dense(30)(x); x = L.LeakyReLU()(x)
    m = M.Model(inps, L.Dense(n_out, activation="linear")(x)); m.compile(optimizer="adam", loss="mse"); return m

rows, curve = [], []
for subj in SUBJECTS:
    t0 = time.time()
    d = pickle.load(open(f"{ROOT}/{subj}/{subj}.pkl", "rb"), encoding="latin1"); lab = d["label"]; chest = d["signal"]["chest"]; stai = stai_labels(subj)
    starts, blk = [], []
    for b in BLOCKS:
        idx = np.where(lab == b)[0]; st = np.arange(idx[0], idx[-1] + 1 - W - P + 1, O)
        if b == 1: st = st[:N_TRAIN + N_TEST]
        starts.append(st); blk.append(np.full(len(st), b))
    starts = np.concatenate(starts); blk = np.concatenate(blk); o = np.argsort(starts, kind="stable"); starts, blk = starts[o], blk[o]; n = len(starts)
    y6 = np.stack([stai[BLOCK_NAME[b]] for b in blk]); ystress = (blk == 2).astype(int)
    if len(BLOCKS) == 1: tr, te = np.arange(n - N_TEST), np.arange(n - N_TEST, n)
    else:
        te = np.concatenate([np.where(blk == b)[0][-int(round((blk == b).sum() * N_TEST / (N_TRAIN + N_TEST))):] for b in BLOCKS]); tr = np.setdiff1d(np.arange(n), te)
    # raw signals as float32 tensors, z-scored on the span covered by training windows
    sigs = []
    for m in MODS:
        s = chest[m].astype(np.float32); s = s if s.ndim == 2 else s[:, None]
        span = np.zeros(len(s), bool)
        for a in starts[tr]: span[a:a + W] = True
        mu, sd = s[span].mean(0), s[span].std(0) + 1e-8; s = (s - mu) / sd
        sigs.append(s)
    del d, chest
    chs = [s.shape[1] for s in sigs]
    print(f"{subj}: {n} windows ({len(tr)} train / {len(te)} test), blocks {BLOCKS}, stress windows train {int(ystress[tr].sum())} test {int(ystress[te].sum())}", flush=True)

    encs, pre_rmse = [], {}
    for m, s in zip(MODS, sigs):
        ck = f"{OUT}/enc_{subj}_{TAG}_{m}.weights.h5"
        if os.path.exists(ck):
            e = encoder(s.shape[1]); e.load_weights(ck); e.trainable = False; r = np.nan
        else:
            e, r = pretrain(s, starts[tr]); e.save_weights(ck)
        encs.append(e); pre_rmse[m] = r
    print(f"{subj}: pretraining done ({time.time() - t0:.0f}s) forecast RMSE { {k: round(v, 3) for k, v in pre_rmse.items()} }", flush=True)

    def evaluate(model, idx_fit):
        out = {}
        if len(idx_fit) > EVAL_TRAIN_MAX: idx_fit = np.sort(np.random.RandomState(0).choice(idx_fit, EVAL_TRAIN_MAX, replace=False))
        for name, idx in (("train", idx_fit), ("test", te)):
            p = model.predict(win_ds(sigs, starts[idx]), verbose=0)
            out[f"{name}_rmse"] = float(np.sqrt(((p - y6[idx]) ** 2).mean()))
            for q in range(6): out[f"{name}_rmse_q{q + 1}"] = float(np.sqrt(((p[:, q] - y6[idx, q]) ** 2).mean()))
            anx = p[:, [1, 2, 4]].sum(1) - p[:, [0, 3, 5]].sum(1)          # STAI anxiety: nervous+jittery+worried minus at-ease+relaxed+pleasant
            out[f"{name}_auc"] = roc_auc_score(ystress[idx], anx) if len(np.unique(ystress[idx])) == 2 else np.nan
        return out
    for nl in N_LABELS:
        for rep in range(REPEATS if nl < N_TRAIN else 1):
            rng = np.random.RandomState(100 * rep + nl); fit_idx = np.sort(rng.choice(tr, min(nl, len(tr)), replace=False))
            ep = FT_EPOCHS_FULL if nl >= N_TRAIN else FT_EPOCHS_FEW
            for model_name, mk in (("ssl_finetune", lambda: head(chs, encs)), ("supervised", lambda: head(chs))):
                tf.random.set_seed(rep); mdl = mk(); mdl.fit(win_ds(sigs, starts[fit_idx], y6[fit_idx], shuffle=True, seed=rep), epochs=ep, verbose=0)
                curve.append(dict(subject=subj, model=model_name, n_labels=len(fit_idx), repeat=rep, **evaluate(mdl, fit_idx)))
            a, b = curve[-2], curve[-1]
            pd.DataFrame(curve).to_csv(f"{OUT}/label_curve_{TAG}.csv", index=False)
            print(f"{subj} n={len(fit_idx):5d} rep{rep}: SSL test RMSE {a['test_rmse']:.3f} (train {a['train_rmse']:.3f}) | supervised test RMSE {b['test_rmse']:.3f} (train {b['train_rmse']:.3f})"
                  + (f" | AUC ssl train {a['train_auc']:.3f} test {a['test_auc']:.3f}, sup train {b['train_auc']:.3f} test {b['test_auc']:.3f}" if not np.isnan(a["test_auc"]) else ""), flush=True)
    rows.append(dict(subject=subj, n_windows=n, n_train=len(tr), n_test=len(te), seconds=round(time.time() - t0), **{f"pre_rmse_{m}": v for m, v in pre_rmse.items()}))
    pd.DataFrame(rows).to_csv(f"{OUT}/subjects_{TAG}.csv", index=False)
    print(f"{subj} done in {time.time() - t0:.0f}s", flush=True)
