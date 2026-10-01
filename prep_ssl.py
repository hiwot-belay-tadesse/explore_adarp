"""
Usage: python3 prep_ssl.py <window_seconds>   (10 -> processed_10s, 60 -> processed_60s)
ADARP preprocessing following Islam & Washington (2023, arXiv:2307.03337), adapted to ADARP.

  * Signal  : raw EDA (4 Hz) only, no filtering.
  * Windows : WIN_S seconds, 50 % overlap. Forecast target = next WIN/10 samples.
  * Labels  : 1 = all samples within +/-20 min of a tag, 0 = all samples > 60 min from any tag,
              -1 = ambiguous (20-60 min buffer or mixed). -1 windows are kept for SSL only.
              Tags pooled per participant (two wrists, same moment).
  * Quality : windows with mean EDA < 0.05 uS (off-body) dropped everywhere.
  * Split   : per participant, chronological. Time-overlapping sessions form a group; groups are
              ordered by start time; the latest ~20 % of windows are test, guarded so that both
              splits contain stress windows. No window ever straddles sessions or splits.
  * Norm    : min-max [0,1] fitted on train windows only, per participant.
Outputs -> processed_10s/<pid>/data.npz, summary.csv, split_manifest.csv, config.json
"""
import os, json, sys
import numpy as np, pandas as pd
WIN_S = int(sys.argv[1]) if len(sys.argv) > 1 else 10   # window length in seconds (10 or 60)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT, OUT = os.path.join(HERE, "Sensor Data"), os.path.join(HERE, f"processed_{WIN_S}s")
FS = 4; WIN = WIN_S * FS; STEP = WIN // 2; HORIZON = WIN // 10   # 50 % overlap; forecast next 10 % of the window
STRESS_S, BUFFER_S = 20 * 60, 60 * 60
TEST_FRAC, EDA_MIN_US = 0.20, 0.05
os.makedirs(OUT, exist_ok=True)

def load_eda(p):
    with open(p) as f:
        t0 = float(f.readline()); fs = float(f.readline()); x = np.loadtxt(f)
    assert fs == FS
    return t0, np.atleast_1d(x)

sessions = []
for part in sorted(os.listdir(ROOT)):
    if not part.startswith("Part"): continue
    pid = part.replace("Part ", "P")
    for sid in sorted(os.listdir(os.path.join(ROOT, part))):
        sd = os.path.join(ROOT, part, sid)
        if not os.path.isdir(sd): continue
        t0, x = load_eda(os.path.join(sd, "EDA.csv"))
        tags = np.array([float(l) for l in open(os.path.join(sd, "tags.csv")) if l.strip()])
        sessions.append(dict(pid=pid, sid=sid, device=sid.split("_")[0].upper(), t0=t0, t1=t0 + len(x) / FS, x=x, tags=tags))
print(f"{len(sessions)} sessions, {len({s['pid'] for s in sessions})} participants")

def sample_labels(t0, n, tags):
    if len(tags) == 0: return np.zeros(n, np.int8)
    t = t0 + np.arange(n) / FS
    tags = np.sort(tags); i = np.searchsorted(tags, t)
    dprev = np.where(i > 0, t - tags[np.clip(i - 1, 0, len(tags) - 1)], np.inf)
    dnext = np.where(i < len(tags), tags[np.clip(i, 0, len(tags) - 1)] - t, np.inf)
    d = np.minimum(np.abs(dprev), np.abs(dnext))
    lab = np.zeros(n, np.int8); lab[d <= BUFFER_S] = -1; lab[d <= STRESS_S] = 1
    return lab

def windows(x, lab):
    X, Y, L, S = [], [], [], []
    for s in range(0, len(x) - WIN - HORIZON + 1, STEP):
        seg = x[s:s + WIN]
        if seg.mean() < EDA_MIN_US: continue
        l = lab[s:s + WIN]
        y = 1 if (l == 1).all() else 0 if (l == 0).all() else -1
        X.append(seg); Y.append(x[s + WIN:s + WIN + HORIZON]); L.append(y); S.append(s)
    if not X: return np.empty((0, WIN)), np.empty((0, HORIZON)), np.empty(0, np.int8), np.empty(0, int)
    return np.array(X), np.array(Y), np.array(L, np.int8), np.array(S)

def groups_by_overlap(sl):
    n = len(sl); par = list(range(n))
    def f(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if sl[i]["t0"] < sl[j]["t1"] and sl[j]["t0"] < sl[i]["t1"]: par[f(i)] = f(j)
    return [f(i) for i in range(n)]

summary, manifest = [], []
for pid in sorted({s["pid"] for s in sessions}):
    sl = sorted([s for s in sessions if s["pid"] == pid], key=lambda s: s["t0"])
    tags_all = np.concatenate([s["tags"] for s in sl])
    for s, g in zip(sl, groups_by_overlap(sl)):
        s["g"] = g
        s["X"], s["Yf"], s["L"], s["S"] = windows(s["x"], sample_labels(s["t0"], len(s["x"]), tags_all))
    # chronological split by group: walk groups from latest to earliest until ~20 % of windows
    gids = sorted({s["g"] for s in sl}, key=lambda g: min(s["t0"] for s in sl if s["g"] == g))
    gwin = {g: sum(len(s["L"]) for s in sl if s["g"] == g) for g in gids}
    gstr = {g: sum(int((s["L"] == 1).sum()) for s in sl if s["g"] == g) for g in gids}
    total = sum(gwin.values()); test, acc = [], 0
    for g in reversed(gids):
        if acc >= TEST_FRAC * total: break
        test.append(g); acc += gwin[g]
    while sum(gstr[g] for g in test) == 0:                       # guard: test needs stress
        test.append([g for g in gids if g not in test][-1])
    while sum(gstr[g] for g in gids if g not in test) == 0:      # guard: train needs stress
        test.remove(sorted(test, key=gids.index)[0])
    for s in sl:
        s["split"] = "test" if s["g"] in test else "train"
        manifest.append(dict(participant=pid, session=s["sid"], device=s["device"], group=s["g"], split=s["split"],
                             start_utc=pd.to_datetime(s["t0"], unit="s"), hours=round((s["t1"] - s["t0"]) / 3600, 2),
                             n_tags=len(s["tags"]), n_windows=len(s["L"]), n_stress=int((s["L"] == 1).sum()),
                             n_nostress=int((s["L"] == 0).sum()), n_unlabeled=int((s["L"] == -1).sum())))
    def cat(split, key):
        parts = [s[key] for s in sl if s["split"] == split and len(s["L"])]
        return np.concatenate(parts) if parts else np.empty(0)
    out = {}
    for split in ("train", "test"):
        out[f"X_{split}"] = cat(split, "X"); out[f"Yf_{split}"] = cat(split, "Yf"); out[f"y_{split}"] = cat(split, "L")
        out[f"session_{split}"] = np.concatenate([np.repeat(s["sid"], len(s["L"])) for s in sl if s["split"] == split and len(s["L"])])
        out[f"t_{split}"] = np.concatenate([s["t0"] + s["S"] / FS for s in sl if s["split"] == split and len(s["L"])])
    mn, mx = out["X_train"].min(), out["X_train"].max()
    for k in ("X_train", "Yf_train", "X_test", "Yf_test"):
        out[k] = ((out[k] - mn) / (mx - mn)).astype(np.float32)
    out.update(norm_min=mn, norm_max=mx, fs=FS, win=WIN, step=STEP, horizon=HORIZON)
    os.makedirs(os.path.join(OUT, pid), exist_ok=True)
    np.savez_compressed(os.path.join(OUT, pid, "data.npz"), **out)
    ytr, yte = out["y_train"], out["y_test"]
    row = dict(participant=pid, sessions=len(sl), groups=len(gids), test_groups=len(test), tags=len(tags_all),
               train_windows=len(ytr), train_stress=int((ytr == 1).sum()), train_nostress=int((ytr == 0).sum()), train_unlabeled=int((ytr == -1).sum()),
               test_windows=len(yte), test_stress=int((yte == 1).sum()), test_nostress=int((yte == 0).sum()), test_unlabeled=int((yte == -1).sum()),
               test_frac=round(len(yte) / (len(ytr) + len(yte)), 3),
               train_end_utc=pd.to_datetime(out["t_train"].max(), unit="s"), test_start_utc=pd.to_datetime(out["t_test"].min(), unit="s"))
    summary.append(row)
    print(f"{pid}: train {len(ytr)} ({row['train_stress']} S / {row['train_nostress']} NS / {row['train_unlabeled']} U)  "
          f"test {len(yte)} ({row['test_stress']} S / {row['test_nostress']} NS)  test_frac={row['test_frac']}")

pd.DataFrame(summary).to_csv(os.path.join(OUT, "summary.csv"), index=False)
man = pd.DataFrame(manifest); man.to_csv(os.path.join(OUT, "split_manifest.csv"), index=False)
json.dump(dict(fs=FS, win=WIN, step=STEP, horizon=HORIZON, stress_s=STRESS_S, buffer_s=BUFFER_S, test_frac=TEST_FRAC,
               eda_min_us=EDA_MIN_US, split="chronological by time-overlap session group", norm="per-participant min-max on train"),
          open(os.path.join(OUT, "config.json"), "w"), indent=2)
# leakage checks
for pid, g in man.groupby("participant"):
    assert (g.groupby("group").split.nunique() == 1).all()
    assert g[g.split == "train"].start_utc.max() < g[g.split == "test"].start_utc.min(), pid
print("leak check passed: groups never straddle splits; every train session starts before every test session")
