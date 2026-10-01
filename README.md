# ADARP stress classification: leak-free personal models

Pipeline for the ADARP dataset (Alam et al. 2022, arXiv:2206.14568) following the personalized
self-supervised approach of Islam & Washington (2023, arXiv:2307.03337), adapted from WESAD to ADARP.

## Files

| File | Purpose |
|---|---|
| `prep_ssl_10s.py` | Loads all 237 Empatica E4 sessions, labels EDA, windows it, makes the chronological split. Writes `processed_10s/`. |
| `train_ssl_10s.py` | Per participant: SSL forecasting pretrain, fine-tune, purely supervised baseline, label-efficiency curve. Writes `processed_10s/results_*.csv`. |
| `plot_results.py` | Small-multiples figure of test AUC vs number of labels. |
| `preprocess_adarp.py`, `processed/` | Earlier 60 s window / random-session-split / logistic-regression version. Superseded, kept for reference. |
| `prep_ssl.py`, `train_ssl.py` | Window-length-parameterised versions (`python3 prep_ssl.py 60`), used for the 10 s vs 60 s comparison in `compare_windows.py`. |
| `cv_ssl.py` | Leave-days-out CV (5 chronological folds per participant), stress = 10 min before press, per-session z-score, EDA only. Output `processed_cv_<win>s/`, plot `plot_cv.py`. |
| `cv_ssl_multimodal.py` | Same CV with EDA + HRV tachogram (IBI interpolated to 4 Hz), one encoder per modality, late fusion. Output `processed_cv_multimodal/`, plot `plot_multimodal.py`. |
| `hrv_eda_logreg.py` | Logistic-regression baseline on EDA + HRV features, same windows and folds. Output `processed_hrv_eda/`. |
| `prep_paper_split.py`, `train_paper_split.py` | Replication of the ADARP paper's own protocol (Alam et al. 2022, Sec. III-B): random 80/20 split **by window** (not by session), 60 s windows with 50 % overlap, Butterworth 1.25 Hz low-pass, min-max, paper CNN, majority undersampling and SMOTE; one pooled model plus one per participant. Output `processed_paper_split/`. |
| `wesad_replicate.py` | Replication of Islam & Washington on WESAD (`~/WESAD`): `BLOCKS=1` literal paper (RMSE), `BLOCKS=1,2` adds TSST for AUC. Output `wesad_replication/`, plot `plot_wesad.py`. |

## Preprocessing (`prep_ssl_10s.py`)

- **Signal**: raw EDA at 4 Hz only, no filtering (the paper uses raw signals).
- **Labels from button-press tags**, pooled per participant so both wrists agree:
  - within ±20 min of a tag: **stress (1)**
  - 20 to 60 min from the nearest tag: **ambiguous (-1)**, kept for self-supervised pretraining only
  - more than 60 min from every tag: **not-stress (0)**
- **Windows**: 10 s (40 samples), 5 s stride. A window keeps a label only if every sample agrees; mixed windows become -1.
  Forecast target for the pretext task: the next 4 samples (1 s).
- **Quality**: windows with mean EDA below 0.05 µS are dropped (wristband off the skin).
- **Split, per participant, chronological**: sessions that overlap in time (two devices) form one group. Groups are
  ordered by start time and the latest ~20 % of windows form the test set, guarded so both splits contain stress.
  Every training session starts before every test session and no window straddles a session. Checked by assertion.
- **Scaling**: `data.npz` stores min-max on train statistics. The training script re-standardizes to z-scores on train
  statistics because min-max compressed EDA to ~0.02 and the CNN could not fit.

Outputs: `processed_10s/<Pid>/data.npz` (`X_train, Yf_train, y_train, X_test, Yf_test, y_test`, session ids, start
times), `summary.csv`, `split_manifest.csv`, `config.json`.

## Modeling (`train_ssl_10s.py`)

Architecture follows Tables 1 and 2 of the paper, kernel sizes scaled from a 7000-sample input to 40 samples
(8, 6, 4, 6 instead of 40, 30, 18, 30):

1. **Pretext task**: 1D CNN (4 conv layers, filters 4/2/4/2, LeakyReLU) → Dense 50 → Dense 30 → 4 linear outputs,
   MSE, trained on every training window including unlabeled ones.
2. **Fine-tune**: conv layers frozen, Dense 30 → Dense 30 → sigmoid, binary cross-entropy with class weights.
3. **Purely supervised baseline**: same architecture, all layers trainable, random init.
4. **Label efficiency**: both models trained on 5, 10, 20, 50, 100, 200, 500 stratified labeled windows,
   3 repeats each, and on all labels.

Metrics per participant: train AUC, test AUC, test F1 and balanced accuracy at threshold 0.5.

## Reproduce

```bash
python3 prep_ssl_10s.py      # ~15 s
python3 train_ssl_10s.py     # ~2 min per participant on CPU (TensorFlow 2.19)
python3 plot_results.py
```

## Paper-protocol replication (`prep_paper_split.py`, `train_paper_split.py`)

The dataset paper evaluates with a random split over windows, so adjacent 50 %-overlapping windows from the same stress
episode land in both train and test. This is the setting that produces its 98 % test accuracy; it is not leak-free.

- **Events**: 409 tags; an event is valid if the recording covers 30 min before and after it. This reproduces the paper's
  181 valid events exactly (60 min on each side gives 168).
- **Labels**: stress = within 30 min of a valid event, not-stress = more than 30 min from every event; samples near
  dropped events are unlabelled. 60 s windows, 30 s stride, all samples must agree.
- **Preprocessing**: 2nd-order Butterworth low-pass at 1.25 Hz, min-max to [0,1] per recording.
- **Split**: stratified random 80/20 by window within each participant (seed 0); the pooled model uses the union.
- **Model**: paper CNN (Conv 250/100, kernel 5, global max pool, Dense 256/128/64, dropout 0.1, softmax, Adam 1e-3,
  50 epochs, batch 32). Class balance on the training set only: random undersampling or SMOTE (k = 5, own implementation).
- **Outputs**: `processed_paper_split/results.csv` with accuracy, precision, recall, F1 and AUC on train and test for
  the pooled model (also per participant) and for each personal model; `log.txt` has per-epoch losses.

```bash
/usr/bin/python3 prep_paper_split.py     # ~1 min
/usr/bin/python3 train_paper_split.py    # ~5 h on CPU, resumable, ONLY=P101C BALANCE=under EPOCHS=5 for a quick run
```

## Results summary (2026-09-30 / 10-01)

All ADARP variants with leak-free per-participant evaluation land at mean held-out AUC 0.47 to 0.58 (see `compare_windows_auc.csv`,
`processed_cv_10s/results_per_person.csv`, `processed_cv_multimodal/results_per_person.csv`). Day-to-day baseline drift and sparse
button-press labels dominate; HRV adds capacity but not generalisation; the E4 IBI stream has clean beats in only 7 to 27 % of windows.

WESAD S2 (Islam & Washington setup, baseline + TSST, test = last 11.5 % of each block): train AUC 1.0 for every fit, test AUC
unstable across label draws and below 0.5 with all labels for both SSL and supervised. The paper's SSL advantage did not reproduce.
