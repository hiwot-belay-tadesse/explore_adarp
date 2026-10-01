"""
ADARP preprocessing replicating the dataset paper's own classification setup (Alam et al. 2022, arXiv:2206.14568, Sec. III-B).

  * Signal   : EDA (4 Hz), 2nd-order Butterworth low-pass 1.25 Hz (zero phase), then min-max to [0,1] per recording.
  * Events   : tags per recording (409 in total). An event is valid if the recording covers the full 30 min before
               and after it (a 60 min buffer zone in total). This reproduces the paper's 181 valid events exactly;
               requiring 60 min on each side gives 168.
  * Labels   : stress = within 30 min of a valid event; not-stress = everything farther than 30 min from every event.
               Samples within 30 min of a dropped (invalid) event are left unlabelled.
  * Windows  : 60 s (240 samples), 50 % overlap (120 samples). A window is kept only if all its samples share a label.
  * Split    : random, by window, stratified 80/20 inside each participant (seed 0). The pooled split is the union,
               so the pooled model and the personal models are evaluated on the same test windows.
Outputs -> processed_paper_split/data.npz, summary.csv, config.json
"""
import os, json
import numpy as np, pandas as pd
from scipy.signal import butter, filtfilt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT, OUT = os.path.join(HERE, "Sensor Data"), os.path.join(HERE, "processed_paper_split")
FS, WIN, STEP, BUFFER_S, TEST_FRAC, SEED = 4, 240, 120, 30 * 60, 0.20, 0
os.makedirs(OUT, exist_ok=True)
B, A = butter(2, 1.25 / (FS / 2), btype="low")

def load_eda(p):
    with open(p) as f:
        t0 = float(f.readline()); fs = float(f.readline()); x = np.loadtxt(f)
    assert fs == FS
    return t0, np.atleast_1d(x).astype(np.float64)

def sample_labels(t0, n, tags, valid):
    """1 stress (within 60 min of a valid tag), 0 not-stress (> 60 min from every tag), -1 near an invalid tag."""
    lab = np.zeros(n, np.int8)
    if len(tags) == 0: return lab
    t = t0 + np.arange(n) / FS
    for tg, ok in zip(tags, valid):
        m = np.abs(t - tg) <= BUFFER_S
        if ok: lab[m] = 1
        else:  lab[m & (lab != 1)] = -1
    for tg, ok in zip(tags, valid):               # a valid tag wins over a neighbouring invalid one
        if ok: lab[np.abs(t - tg) <= BUFFER_S] = 1
    return lab

rows, X, y, pid_w, sid_w, t_w = [], [], [], [], [], []
n_tags = n_valid = 0
for part in sorted(os.listdir(ROOT)):
    if not part.startswith("Part"): continue
    pid = part.replace("Part ", "P")
    for sid in sorted(os.listdir(os.path.join(ROOT, part))):
        sd = os.path.join(ROOT, part, sid)
        if not os.path.isdir(sd): continue
        t0, x = load_eda(os.path.join(sd, "EDA.csv"))
        t1 = t0 + len(x) / FS
        tags = np.array([float(l) for l in open(os.path.join(sd, "tags.csv")) if l.strip()])
        valid = (tags - BUFFER_S >= t0) & (tags + BUFFER_S <= t1)
        n_tags += len(tags); n_valid += int(valid.sum())
        if len(x) < WIN + 10: continue
        xf = filtfilt(B, A, x)
        rng = xf.max() - xf.min()
        xn = (xf - xf.min()) / rng if rng > 0 else np.zeros_like(xf)
        lab = sample_labels(t0, len(x), tags, valid)
        for s in range(0, len(x) - WIN + 1, STEP):
            l = lab[s:s + WIN]
            if (l == 1).all(): lw = 1
            elif (l == 0).all(): lw = 0
            else: continue
            X.append(xn[s:s + WIN].astype(np.float32)); y.append(lw); pid_w.append(pid); sid_w.append(sid); t_w.append(t0 + s / FS)
        rows.append(dict(participant=pid, session=sid, hours=round((t1 - t0) / 3600, 2), tags=len(tags), valid_tags=int(valid.sum()),
                         stress_windows=int(sum(1 for i in range(len(y) - 1, -1, -1) if sid_w[i] == sid and y[i] == 1)),
                         notstress_windows=int(sum(1 for i in range(len(y) - 1, -1, -1) if sid_w[i] == sid and y[i] == 0))))
X, y, pid_w, sid_w, t_w = np.array(X), np.array(y, np.int8), np.array(pid_w), np.array(sid_w), np.array(t_w)
print(f"tags {n_tags}, valid events {n_valid} (paper: 409 / 181)")
print(f"windows {len(y)}: stress {(y == 1).sum()}, not-stress {(y == 0).sum()} (paper: 181 'samples' vs 163884)")

# stratified random 80/20 split by window within each participant
rng = np.random.default_rng(SEED)
split = np.empty(len(y), dtype="<U5")
for p in np.unique(pid_w):
    for c in (0, 1):
        idx = np.where((pid_w == p) & (y == c))[0]
        rng.shuffle(idx)
        n_te = int(round(TEST_FRAC * len(idx)))
        split[idx[:n_te]] = "test"; split[idx[n_te:]] = "train"
assert (split != "").all()

np.savez_compressed(os.path.join(OUT, "data.npz"), X=X, y=y, pid=pid_w, sid=sid_w, t=t_w, split=split)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "sessions.csv"), index=False)
summ = (pd.DataFrame(dict(participant=pid_w, y=y, split=split)).groupby(["participant", "split", "y"]).size()
        .unstack(["split", "y"], fill_value=0))
summ.columns = [f"{s}_{'stress' if c == 1 else 'nostress'}" for s, c in summ.columns]
summ.to_csv(os.path.join(OUT, "summary.csv")); print(summ.to_string())
json.dump(dict(fs=FS, win=WIN, step=STEP, buffer_s=BUFFER_S, test_frac=TEST_FRAC, seed=SEED, filter="butter 2nd order lowpass 1.25 Hz filtfilt",
               norm="min-max [0,1] per recording", split="random by window, stratified 80/20 within participant",
               tags=n_tags, valid_events=n_valid), open(os.path.join(OUT, "config.json"), "w"), indent=1)
