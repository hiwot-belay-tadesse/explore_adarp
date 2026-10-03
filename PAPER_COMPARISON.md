# Our replication vs the authors' code (github.com/rameshKrSah/ADARP_Dataset)

Compared on 2026-10-02 against `data_proc_adarp.py`, `data_loader.py`, `utils.py`, `filters.py`, `stress_models.py`
(last commit 2022-07-23). The repo holds the preprocessing and model definitions; the training notebook that produced
Tables I and II of the paper is not included, so the exact training call is inferred from the helper defaults.

| step | authors' code | our `prep_paper_split.py` / `train_paper_split.py` | effect |
|---|---|---|---|
| valid event | tag with >= 30 min of recording on each side (`check_threshold = 30*60*sf`) | same (chosen because it reproduces 181) | identical, 181 events |
| stress segment | **+/- 20 min** around the tag (`tag_segment_length_seconds = 40*60`) | +/- 30 min | ours labels 17,393 stress windows, theirs 14,299 (= 181 x 79) |
| not-stress | stretches **> 60 min** from every tag of the recording, plus whole recordings with no tags | > 30 min from every tag | ours includes the 30-60 min zone as not-stress |
| 20-60 min zone | dropped (buffer) | 30-60 min counted as not-stress | label noise differs slightly |
| filter | Butterworth order 2, 1.25 Hz, `filtfilt` | same | identical |
| **normalisation** | **min-max per extracted segment**: each 40 min stress segment and each hours-long not-stress stretch is scaled to [0,1] on its own | min-max per whole recording | **class-dependent artefact, see below** |
| windows | 60 s, 50 % overlap | same | identical |
| split | `train_test_split(test_size=0.3, random_state=42, stratify=y)`, pooled | stratified 80/20 within participant, pooled = union | both random by window; overlapping windows leak across the split in both |
| undersampling | `np.random.randint` **with replacement**, applied **before** the split | without replacement, train only | theirs puts duplicate not-stress windows in train and test |
| SMOTE | imblearn SMOTE on the training set after the split | own SMOTE (k = 5), training set only | equivalent |
| model | Conv 250 k5 **same padding**, Conv 100 k5 same, GlobalMaxPool, Dense 256/128/64, dropout 0.1/0.1, softmax 2 | same but **valid padding** | negligible |
| optimiser | docstring: Adam **lr 0.01157, batch 100** (hyper-parameter search); paper text: lr 0.001, batch 32 | lr 0.001, batch 32 (paper text) | theirs trains ~10x more aggressively per epoch |
| epochs | `evaluate_model` default 50 | 50 | identical |
| window counts | paper: 163,884 not-stress | our rebuild of their rule: 134,526; 161,276 if zero-tag recordings are counted twice (both `not_stressed_data_from_all_files` and `not_stressed_data_from_zero_tags_files` include them) | likely double counting, which also duplicates windows across train and test |

## The normalisation artefact

Rebuilding their pipeline exactly (stress = +/- 20 min, not-stress > 60 min, filter, min-max per segment, 60 s windows)
and looking only at the normalised signal inside each window:

| class | median within-window std | median within-window range | windows with range < 0.01 |
|---|---|---|---|
| stress (40 min segments) | 0.029 | 0.147 | 6 % |
| not-stress (hours-long stretches) | 0.005 | 0.025 | 35 % |

A 40 min segment has a small min-max range, so after scaling its windows swing widely inside [0,1]. A multi-hour
stretch has a large range, so its windows are compressed to a few hundredths. The within-window range alone, with no
model, separates the classes with **AUC 0.78**. A CNN with global max pooling picks this up immediately. Together with
the window-level split, duplicated not-stress windows and overlapping neighbours in train and test, this is sufficient
to explain the paper's 98 % test accuracy without any physiological signal.

Our run normalises per recording, so stress and not-stress windows share the same scale. That removes the artefact and
leaves test AUC 0.64 (`processed_paper_split/results.csv`), consistent with the leak-free results of 0.47 to 0.58.

## Minor code notes

- `segment_sensor_reading` and our windowing yield the same windows (start 0, step 120, drop the tail).
- `select_random_samples` draws indices with `np.random.randint`, so the undersampled majority contains duplicates and
  is smaller than intended; the paper's Table I was produced on this set and then split.
- `load_data_with_preprocessing` and `min_max_scale` rescale to [-1, 1] with a scaler fitted per time index across all
  windows (train and test together). Whether it was applied for the paper is not recoverable from the repo.

## Running their pipeline (`authors_pipeline.py`, 2026-10-02/03)

Same 14,299 stress and 134,526 not-stress windows as their code, 70/30 random window split (seed 42), 50 epochs, EDA only.
"code" = Adam lr 0.01157, batch 100 (model docstring); "paper" = lr 0.001, batch 32 (paper text). AUC is ours; the paper reports none.

| setup | normalisation | optimiser | test acc | precision | recall | F1 | **test AUC** | paper acc / F1 |
|---|---|---|---|---|---|---|---|---|
| Table I (undersample w/ replacement before split) | per segment | code | 0.72 | 0.78 | 0.62 | 0.69 | **0.82** | 0.98 / 0.98 |
| Table I | per segment | paper | 0.74 | 0.77 | 0.70 | 0.73 | **0.83** | 0.98 / 0.98 |
| Table I, control | per recording | code | 0.50 | 0.00 | 0.00 | 0.00 | **0.50** (collapsed to constant) | |
| Table I, control | per recording | paper | 0.62 | 0.62 | 0.59 | 0.61 | **0.68** | |
| Table II (SMOTE on train after split) | per segment | code | 0.67 | 0.20 | 0.79 | 0.32 | **0.79** | 0.87 / 0.84 |
| Table II, control | per recording | code | collapsed by epoch 10 (loss 0.693), stopped | | | | | |
| Table II | per recording | paper | running | | | | | |
| Table II | per segment | paper | queued | | | | | |

Reading: the authors' own pipeline, run as written, gives test AUC 0.79 to 0.83 and accuracy 0.67 to 0.74, not 87 to 98 %.
Changing only the normalisation unit from segment to recording drops Table I from AUC 0.83 to 0.68 with the paper's optimiser
and to a collapsed constant predictor with the code's optimiser. The remaining 0.68 is what the leaky window-level split
leaves on EDA; leak-free per-participant splits give 0.47 to 0.58. Nothing in the repository reproduces a training
accuracy of 99.7 %; the training notebook is not published.
