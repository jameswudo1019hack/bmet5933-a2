# Results Summary

All canonical numbers in one place. Pull from here when writing the paper.
Last updated: 2026-04-29 (Sprint 4 — ConvNeXt V2 on medium; closes the 2×2 grid)

---

## Primary comparison — medium dataset (6,221 images, 70/15/15, seed=42)

Both models evaluated on the **same** 934-image test set. Apples-to-apples.

### Overall metrics

| Model                                          | Accuracy   | Macro-F1   | Weighted F1 | ROC-AUC | Errors      |
| ---------------------------------------------- | ---------- | ---------- | ----------- | ------- | ----------- |
| Classical XGBoost                              | 0.9979     | 0.9976     | 0.9979      | 0.99999 | 2 / 934     |
| EfficientNet-B0 baseline                       | 0.9797     | 0.9745     | 0.9797      | 0.9995  | 19 / 934    |
| EfficientNet-B0 + TTA hflip ← **canonical DL** | 0.9861     | **0.9829** | 0.9861      | 0.9998  | 13 / 934    |
| ConvNeXt V2 Base (Sprint 4)                    | 0.9925     | **0.9898** | 0.9925      | 0.9999  | 7 / 934     |
| Ensemble val-tuned (collapses to classical)    | 0.9979     | 0.9976     | 0.9979      | —       | 2 / 934     |
| Ensemble equal-weight w=0.5 ← **headline**     | **1.0000** | **1.0000** | 1.0000      | —       | **0 / 934** |

### Bootstrap 95% CIs (macro-F1)

| Model | Mean | Lower | Upper |
|---|---|---|---|
| Classical XGBoost | 0.9977 | 0.9939 | 1.0000 |
| EfficientNet-B0 + TTA hflip | 0.9829 | 0.9728 | 0.9918 |

### Per-class F1 — Classical XGBoost

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Cyst | 0.9964 | 0.9964 | 0.9964 | 279 |
| Normal | 1.0000 | 1.0000 | 1.0000 | 381 |
| Stone | 1.0000 | 1.0000 | 1.0000 | 103 |
| Tumor | 0.9942 | 0.9942 | 0.9942 | 171 |

### Per-class F1 — EfficientNet-B0 + TTA hflip

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Cyst | 0.9823 | 0.9964 | 0.9893 | 279 |
| Normal | 0.9894 | 0.9843 | 0.9868 | 381 |
| Stone | 0.9612 | 0.9612 | 0.9612 | 103 |
| Tumor | 1.0000 | 0.9883 | 0.9941 | 171 |

### Confusion matrix — Classical XGBoost

|  | Pred Cyst | Pred Normal | Pred Stone | Pred Tumor |
|---|---|---|---|---|
| **True Cyst** | 278 | 0 | 0 | 1 |
| **True Normal** | 0 | 381 | 0 | 0 |
| **True Stone** | 0 | 0 | 103 | 0 |
| **True Tumor** | 1 | 0 | 0 | 170 |

### Confusion matrix — EfficientNet-B0 + TTA hflip

|  | Pred Cyst | Pred Normal | Pred Stone | Pred Tumor |
|---|---|---|---|---|
| **True Cyst** | 278 | 0 | 1 | 0 |
| **True Normal** | 4 | 375 | 2 | 0 |
| **True Stone** | 0 | 4 | 99 | 0 |
| **True Tumor** | 1 | 0 | 1 | 169 |

---

## TTA ablation — EfficientNet-B0 (same checkpoint, same test set)

| TTA variant | Views | Macro-F1 | Stone F1 | Errors | vs baseline |
|---|---|---|---|---|---|
| Baseline (no TTA) | 1 | 0.9745 | 0.942 | 19 | — |
| hflip ← **canonical** | 2 | **0.9829** | 0.961 | 13 | +6 fixed, 2 broken |
| basic | 4 | 0.9791 | 0.967 | 16 | +3 |
| full | 6 | 0.9811 | 0.981 | 15 | +4 |
| rot | 3 | 0.9711 | 0.957 | 22 | **−3 (hurts)** |

McNemar's baseline vs TTA hflip: discordant=10, **p=0.11** (not significant — low power with only 13 remaining errors).

---

## Data-efficiency sweep — EfficientNet-B0 baseline (no TTA)

Stratified subsets of the medium training split, same val + test set, same training protocol (Sprint 1, commit `270bbd5`). Confirms that the EfficientNet-B0 baseline reaches a capacity ceiling well before the full medium training set is consumed.

| Train fraction | n_train | Macro-F1 [95 % CI] | Stone F1 | Errors / 934 | Wall time (A100) |
|---|---|---|---|---|---|
| 10 % | 436 | 0.7466 [0.713, 0.778] | 0.595 | 65 | 105 s |
| 25 % | 1088 | 0.9092 [0.886, 0.931] | 0.811 | 30 | 199 s |
| 50 % | 2176 | 0.9580 [0.943, 0.972] | 0.915 | 19 | 309 s |
| 100 % | 4353 | 0.9745 [0.963, 0.986] | 0.942 | 19 | 477 s |

Marginal F1 gain from 50 % → 100 % is < 2 percentage points; further data-volume gains require architectural change (see Sprint 2 results below).

---

## Ensemble analysis

### Val weight grid (DL weight w_dl, classical weight = 1 − w_dl)

Val is saturated — classical alone achieves perfect val F1. Grid is flat from w=0.0 to w=0.70, making val-tuning uninformative.

| w_dl | Val macro-F1 | Test macro-F1 (post-hoc) |
|---|---|---|
| 0.00 (val-tuned) | 1.0000 | 0.9976 (= classical alone) |
| 0.05 – 0.35 | 1.0000 | 0.9988 (fixes 1 error) |
| **0.40 – 0.60** | 1.0000 | **1.0000 (0 errors)** |
| 0.75 – 1.00 | 0.9953 – 0.9985 | degrades |

Reported result: **w=0.5 (a-priori default)** — chosen without reference to test.

### The two classical errors (corrected by equal-weight ensemble)

| idx | True class | Classical prediction | Classical confidence | DL-TTA prediction | DL-TTA confidence |
|---|---|---|---|---|---|
| 324 | Cyst | **Tumor** ✗ | 0.667 | Cyst ✓ | 0.773 |
| 891 | Tumor | **Cyst** ✗ | 0.515 | Tumor ✓ | 0.984 |

Both are Cyst ↔ Tumor confusions — clinically the most consequential pair. DL is confident and correct on both; equal weighting tips them correctly.

### McNemar's (ensemble w=0.5 vs components)

| Comparison | Discordant pairs | p-value |
|---|---|---|
| Ensemble vs DL-TTA hflip | 13 | **0.00024** |
| Ensemble vs Classical | 2 | 0.50 |

---

## Sprint 4 — ConvNeXt V2 Base on medium dataset (closes 2×2 matched grid, 2026-04-29)

Same medium split (`split.csv`, n=934 test) as the canonical Sprint 1 medium-scale comparisons. Trained with the *identical* protocol used for ConvNeXt V2 full in Sprint 2 (image size 384, batch 32, AdamW wd=0.05, stochastic depth 0.3, two-stage with stage2_unfreeze_blocks=1) — only the training data differs (4,353 medium vs 8,712 full).

| Metric | Value |
|---|---|
| Accuracy | **0.9925** |
| Macro-F1 | **0.9898 [0.9813, 0.9968]** |
| Weighted-F1 | 0.9925 |
| ROC-AUC OvR | 0.9999 |
| Errors | **7 / 934** |
| Best epoch | (in run_log.json) |

### Per-class F1 — ConvNeXt V2 Base medium

| Class | Precision | Recall | F1 [95 % CI] | Support |
|---|---|---|---|---|
| Cyst | 0.996 | 0.993 | 0.995 [0.988, 1.000] | 279 |
| Normal | 0.995 | 0.992 | 0.993 [0.987, 0.999] | 381 |
| Stone | 0.962 | 0.981 | **0.971** [0.946, 0.991] | 103 |
| **Tumor** | **1.000** | **1.000** | **1.000** | 171 |

### Confusion matrix — ConvNeXt V2 Base medium

|  | Pred Cyst | Pred Normal | Pred Stone | Pred Tumor |
|---|---|---|---|---|
| **True Cyst** | 277 | 0 | 2 | 0 |
| **True Normal** | 1 | 378 | 2 | 0 |
| **True Stone** | 0 | 2 | 101 | 0 |
| **True Tumor** | 0 | 0 | 0 | 171 |

### 2×2 architecture-vs-data matched grid (now complete)

| | Medium (n=934 test) | Full (n=1,867 test) |
|---|---|---|
| EfficientNet-B0 | 0.9829 (TTA hflip), 13 errors, 1.39 % error rate | 0.9819, 23 errors, 1.23 % error rate |
| **ConvNeXt V2 Base** | **0.9898**, 7 errors, **0.75 %** error rate | 0.9953, 6 errors, **0.32 %** error rate |

**Architecture > data at every scale.** ConvNeXt V2 medium beats EffNet-B0+TTA medium; ConvNeXt V2 full beats EffNet-B0 full (Sprint 2, p=0.0021). **Asymmetric data-volume response**: ConvNeXt V2 error rate drops 57 % medium → full; EffNet-B0 stays flat. Data helps the larger architecture; not the smaller one (EffNet-B0 saturated at 4,353 train).

### Paired McNemar's on the medium n=934 test set

`Results/convnextv2_medium_run/sprint4_medium_grid.json`:

| Comparison | Both correct | Only A wrong | Only B wrong | Both wrong | Discordant | p-value |
|---|---|---|---|---|---|---|
| ConvNeXt V2 medium vs EffNet-B0+TTA medium | 916 | 5 | 11 | 2 | 16 | 0.21 (n.s.) |
| **ConvNeXt V2 medium vs Classical medium** | 925 | 7 | 2 | **0** | 9 | 0.18 (n.s.) |
| EffNet-B0+TTA medium vs Classical medium *(Sprint 1 reference)* | 919 | 13 | 2 | **0** | 15 | **0.0074** (sig.) |

**`both_wrong = 0` between classical-medium and ConvNeXt V2-medium** — the disjoint-error pattern that produced the 100 % medium-set ensemble in Sprint 1 *extends to ConvNeXt V2*. The "100 % equal-weight ensemble" finding is paradigm-stable across both CNN backbones at medium scale, not a quirk of EffNet-B0. The Sprint 3 invalidation chain's medium → full failure is therefore **scale-dependent, not architecture-dependent within DL**.

ConvNeXt V2 medium ∩ EffNet-B0 medium = 2 both-wrong (small within-DL-paradigm overlap, dataset-level hard cases).

---

## Supplementary — ConvNeXtV2 Base on full dataset

Different test set (n=1867, full dataset split). **Not directly comparable to medium results above.**

| Metric        | Value                        |
| ------------- | ---------------------------- |
| Accuracy      | 0.9968                       |
| Macro-F1      | **0.9953** [0.991, 0.998]    |
| Weighted F1   | 0.9968                       |
| ROC-AUC       | 0.99998                      |
| Errors        | 6 / 1867                     |
| Best epoch    | 34 (stage 1: 5, stage 2: 29) |
| Training time | ~54 min (A100, bfloat16)     |

### Per-class F1 — ConvNeXtV2 Base

| Class | F1 | Support |
|---|---|---|
| Cyst | 0.9955 | 556 |
| Normal | 0.9993 | 762 |
| Stone | **0.9880** | 207 |
| Tumor | 0.9985 | 342 |

---

## EfficientNet-B0 on full dataset (matched-data control)

Same full-dataset split as ConvNeXt V2, n = 1867 test. Trained with the same two-stage protocol as the medium-set EfficientNet-B0; only training data differs.

| Metric | Value |
|---|---|
| Accuracy | 0.9877 |
| Macro-F1 | 0.9819 [0.975, 0.989] |
| Stone F1 | 0.9496 |
| Errors | 23 / 1867 |
| Wall time (A100) | 18.4 min |
| Best val F1 / epoch | 0.9766 / 35 |

### Paired McNemar's — EfficientNet-B0 full vs ConvNeXt V2 full (same 1867 test set)

| Quantity | Value |
|---|---|
| Both correct | 1839 |
| Both wrong | 1 |
| Only EfficientNet-B0 wrong | 22 |
| Only ConvNeXt V2 wrong | 5 |
| Discordant pairs | 27 |
| **p-value** | **0.0021** (ConvNeXt V2 > EfficientNet-B0) |

Architecture effect is statistically significant at matched training data. **Data-volume effect on EfficientNet-B0 is approximately zero** (1.29 % error rate on medium + TTA → 1.23 % on full).

---

## Classical ML on full dataset (Sprint 3, 2026-04-27)

Same full-dataset split as DL runs, n = 1,867 test. Trained with the same hyperparameter grids and selection criterion as the medium run (`classical/config.py`); only training data differs. Hyperparameters were **not** re-tuned to keep the comparison matched-protocol.

| Metric | Value |
|---|---|
| Best model | XGBoost (`learning_rate=0.1, max_depth=6, n_estimators=200`) |
| Accuracy | 0.9930 |
| Macro-F1 | **0.9897** [0.9835, 0.9948] |
| Weighted-F1 | 0.9930 |
| ROC-AUC OvR | 0.9995 |
| Errors | 13 / 1867 |
| Wall time (local M-series, n_jobs=-1) | 172.7 s train + ~30 s predict + ~30 s sweep |

### Per-classifier val performance — classical full

| Model | Best params | CV macro-F1 | Val macro-F1 |
|---|---|---|---|
| SVM (linear) | `C=1.0` | 0.8388 | 0.8248 |
| Random Forest | `max_depth=20, n_estimators=200, min_samples_split=5` | 0.9734 | 0.9789 |
| **XGBoost (winner)** | `lr=0.1, max_depth=6, n_estimators=200` | **0.9884** | **0.9895** |

SVM's collapse from 0.97 (medium) to 0.82 (full) is itself a finding — at this dataset scale linear SVM with PCA(50) ceiling does not scale. RF and XGB scale; SVM does not.

### Per-class F1 — classical full

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Cyst | 0.988 | 1.000 | 0.994 | 556 |
| Normal | 0.993 | 0.999 | 0.996 | 762 |
| Stone | 0.995 | 0.947 | **0.970** | 207 |
| Tumor | 1.000 | 0.997 | 0.999 | 342 |

### Confusion matrix — classical full (n_test=1867)

|  | Pred Cyst | Pred Normal | Pred Stone | Pred Tumor |
|---|---|---|---|---|
| **True Cyst** | 556 | 0 | 0 | 0 |
| **True Normal** | 0 | 761 | 1 | 0 |
| **True Stone** | 6 | 5 | 196 | 0 |
| **True Tumor** | 1 | 0 | 0 | 341 |

**Note:** the medium-set classical errors were Cyst↔Tumor (1 each). The full-set classical errors are dominated by **Stone → (Cyst, Normal)** (11 / 13). Cyst↔Tumor confusion has *disappeared* at full scale.

### Paired McNemar's at full scale (same 1,867 test set)

`Results/classical_run_full/sprint3_comparison.json`:

| Comparison | Both correct | Only A wrong | Only B wrong | Both wrong | Discordant | p-value |
|---|---|---|---|---|---|---|
| Classical (A) vs EfficientNet-B0-full (B) | 1835 | 9 | 19 | **4** | 28 | **0.089** (n.s.) |
| Classical (A) vs ConvNeXt V2-full (B) | 1850 | 11 | 4 | **2** | 15 | **0.119** (n.s.) |
| EfficientNet-B0-full (A) vs ConvNeXt V2-full (B) | 1839 | 22 | 5 | 1 | 27 | **0.0021** (sig.) |

Critical: `both wrong > 0` between every classical-vs-DL pair. The "disjoint errors" headline from the medium-set Sprint 1 ensemble does **not** survive at full scale. All "both wrong" cases are `Stone → Cyst` confusions.

### Disjoint-error analysis at full scale

Top "only-A-wrong" failure pairs (where A errs, B is correct):

- **Classical-full** (only-classical-wrong): Stone→Normal(5), Stone→Cyst(2–4), Normal→Stone(1), Tumor→Cyst(1)
- **EfficientNet-B0-full** vs classical: **Cyst→Stone(9)**, Stone→Cyst(4), Normal→Stone(3), Normal→Tumor(2), Stone→Normal(1)
- **ConvNeXt V2-full** vs classical: **Cyst→Stone(3)**, Tumor→Normal(1)

**Surviving paradigm-stable claim:** only DL pipelines make `Cyst→Stone` errors at full scale. Classical makes **zero** `Cyst→Stone` errors across all 1,867 test images. This asymmetry is real even though the broader medium-set "disjoint errors" claim is not.

### Data-efficiency sweep — classical on full

`Results/classical_sweep_full/sweep_summary.json`:

| Train fraction | n_train | Val macro-F1 | Test macro-F1 [95 % CI] |
|---|---|---|---|
| 10 % | 871 | 1.0000 | 0.9554 [0.9438, 0.9666] |
| 25 % | 2,178 | 1.0000 | 0.9716 [0.9626, 0.9807] |
| 50 % | 4,355 | 1.0000 | 0.9799 [0.9712, 0.9875] |
| 100 % | 8,712 | 1.0000 | 0.9897 [0.9835, 0.9948] |

Classical reaches 0.96 with only 871 training samples — substantially better than EfficientNet-B0 baseline at any matched fraction on the medium sweep. Classical is highly sample-efficient on this task. Curve plot: `Results/classical_sweep_full/data_efficiency_curve.png`.

---

## Sprint 3 addendum — SVM and RF re-fit on full (same n=1867 test set)

`classical/train.py` grid-searches all three classifiers but only the val-best (XGB) is saved. To enable per-classifier paradigm comparison, SVM and RF were re-fit at full scale on the cached features using their already-discovered best params (`analysis/sprint3_train_svm_rf.py`).

### Per-classifier overall metrics — full classical

| Pipeline | Best params | Accuracy | Macro-F1 [95 % CI] | Errors |
|---|---|---|---|---|
| Classical SVM (linear) | `C=1.0` | 0.8725 | 0.8515 [0.834, 0.868] | 238 / 1867 |
| Classical Random Forest | `max_depth=20, n_estimators=200, min_samples_split=5` | 0.9855 | 0.9801 [0.973, 0.987] | 27 / 1867 |
| **Classical XGBoost** *(matched-protocol winner)* | `lr=0.1, max_depth=6, n_estimators=200` | **0.9930** | **0.9897 [0.984, 0.995]** | 13 / 1867 |

### Per-class F1 — Classical RF (full)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Cyst | 0.979 | 0.996 | 0.988 | 556 |
| Normal | 0.983 | 0.999 | 0.991 | 762 |
| Stone | 0.990 | 0.918 | **0.952** | 207 |
| Tumor | 1.000 | 0.980 | 0.990 | 342 |

### Per-class F1 — Classical SVM (full)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Cyst | 0.912 | 0.908 | 0.910 | 556 |
| Normal | 0.933 | 0.864 | 0.897 | 762 |
| Stone | 0.728 | 0.826 | **0.774** | 207 |
| Tumor | 0.791 | 0.863 | 0.825 | 342 |

### Confusion matrix — Classical RF (full)

|  | Pred Cyst | Pred Normal | Pred Stone | Pred Tumor |
|---|---|---|---|---|
| **True Cyst** | 554 | 0 | 2 | 0 |
| **True Normal** | 1 | 761 | 0 | 0 |
| **True Stone** | 10 | 7 | 190 | 0 |
| **True Tumor** | 1 | 6 | 0 | 335 |

### Confusion matrix — Classical SVM (full)

|  | Pred Cyst | Pred Normal | Pred Stone | Pred Tumor |
|---|---|---|---|---|
| **True Cyst** | 505 | 3 | 23 | 25 |
| **True Normal** | 10 | 658 | 41 | 53 |
| **True Stone** | 24 | 12 | 171 | 0 |
| **True Tumor** | 15 | 32 | 0 | 295 |

### `Cyst → Stone` error tally (the original "DL-exclusive failure" hypothesis, now re-evaluated)

| Pipeline | `Cyst→Stone` errors |
|---|---|
| Classical XGBoost (full) | **0** ← the *only* zero, hence classifier-specific |
| Classical Random Forest (full) | **2** ← within range of ConvNeXt V2 |
| Classical SVM (full, broken) | 23 |
| EfficientNet-B0 (full) | 9 |
| ConvNeXt V2 Base (full) | 3 |

**Conclusion:** the "only DL makes `Cyst→Stone` errors" claim from the XGB-only Sprint 3 analysis is invalidated once RF is in scope. The asymmetry is XGBoost-specific, not paradigm-specific.

### Pairwise McNemar's across all 5 pipelines (10 unordered pairs)

`Results/classical_run_full/sprint3_all_classifiers.json`:

| Pair | Discordant | Both wrong | p-value | Verdict |
|---|---|---|---|---|
| SVM vs RF | 221 | 22 | 2.6e-45 | SVM dominated |
| SVM vs XGB | 233 | 9 | 9.4e-49 | SVM dominated |
| SVM vs EffNet-B0 | 243 | 9 | 6.9e-43 | SVM dominated |
| SVM vs ConvNeXt V2 | 240 | 2 | 2.8e-50 | SVM dominated |
| **RF vs XGB** *(within-classical)* | 18 | 11 | **0.0013** | **Significantly different within paradigm** |
| RF vs EffNet-B0 | 42 | 4 | 0.64 | **Tied** |
| RF vs ConvNeXt V2 | 29 | 2 | 0.0002 | ConvNeXt V2 wins |
| XGB vs EffNet-B0 | 28 | 4 | 0.089 | Tied (Sprint 3) |
| XGB vs ConvNeXt V2 | 15 | 2 | 0.119 | Tied (Sprint 3) |
| EffNet-B0 vs ConvNeXt V2 | 27 | 1 | 0.0021 | ConvNeXt V2 wins (Sprint 2) |

**Performance ranking at full scale**: `ConvNeXt V2 (6 err) > XGBoost (13) > EfficientNet-B0 (23) ~ Random Forest (27) >> SVM (238)`. The "classical vs DL" split does not explain ranking — classifier choice within a paradigm dominates. RF-vs-XGB are significantly different within the classical paradigm, but RF-vs-EffNet-B0 are statistically tied. The two-paradigm framing oversimplifies a 5-model continuum on this saturated dataset.

---

## Sprint 3 second addendum — interpretability + paradigm-coverage check (2026-04-28)

### Classical XGBoost feature importance — per-group permutation importance

Deployed pipeline (`scaler → PCA(50) → XGBClassifier`), permuted on the n=1867 test set, n_repeats=10. Higher drop = more important.

| Feature group | macro-F1 drop ± std | Group size | Share of signal |
|---|---|---|---|
| **LBP** (multi-scale local binary patterns) | **0.568 ± 0.018** | 54 features | **largest** |
| **Gabor** (frequency × orientation responses) | **0.532 ± 0.014** | 32 features | **second** |
| stats (intensity statistics) | 0.236 ± 0.006 | 10 features | third |
| GLCM (Haralick) | 0.163 ± 0.006 | 12 features | smallest |

LBP + Gabor groups jointly carry most of the predictive signal. Figure: `Results/classical_run_full/feature_importance_group.png` (paper Figure 2).

### Top 10 individual features — deployed pipeline permutation importance

| Rank | Feature | Group | macro-F1 drop |
|---|---|---|---|
| 1 | `gabor: f=0.1, θ=π/2, std` | Gabor | 0.0484 |
| 2 | `gabor: f=0.4, θ=3π/4, std` | Gabor | 0.0427 |
| 3 | `gabor: f=0.4, θ=π/2, std` | Gabor | 0.0319 |
| 4 | `lbp: P=24, R=3, b=1` | LBP | 0.0302 |
| 5 | `gabor: f=0.2, θ=3π/4, std` | Gabor | 0.0299 |
| 6 | `lbp: P=24, R=3, b=14` | LBP | 0.0285 |
| 7 | `lbp: P=24, R=3, b=12` | LBP | 0.0259 |
| 8 | `gabor: f=0.4, θ=π/4, std` | Gabor | 0.0247 |
| 9 | `stat: p10` (10th-percentile intensity) | stats | 0.0241 |
| 10 | `gabor: f=0.2, θ=π/2, mean` | Gabor | 0.0235 |

Top features are dominated by *standard deviation of Gabor magnitude responses* at vertical (π/2) and anti-diagonal (3π/4) orientations — consistent with kidney-CT anatomy (renal capsule, vertebral / vascular structures, oblique calculi).

### Raw-XGB sanity check (no PCA)

Parallel XGBoost trained on the unscaled, un-PCA'd 108-dim feature vector with the same `best_params`:

| Pipeline | Test macro-F1 |
|---|---|
| Deployed (scaler → PCA(50) → XGB) | 0.9897 |
| **Raw-XGB (scaler-free, no PCA)** | **0.9950** |

PCA(50) costs ~0.5pp accuracy at full scale. The raw-XGB top-importance features shift to LBP+GLCM (5/5 LBP top, then 3 GLCM, no Gabor in top 10), indicating PCA reshapes the feature weighting. This is a mechanistic explanation for the within-classical-paradigm RF-vs-XGB error disagreement (Sprint 3 addendum, McNemar p=0.0013, both-wrong=11).

### 3-way paradigm-coverage check

3-way disagreement bucket counts on n=1867 test set:

| Bucket | Count |
|---|---|
| classical-XGB right, **both DL wrong** | **0** |
| classical-XGB wrong, both DL right | 8 (all `Stone → Normal/Cyst` classical errors) |
| All three wrong | 1 (idx 790: true=Stone, all → Cyst) |

**This is the fourth step of the invalidation chain.** `classical_right_and_both_DL_wrong = 0` means *there is no test image where adding classical-XGB rescues the joint DL pipeline*. The Sprint 1 "complementary signal" / "100 % ensemble" finding is formally falsified at full scale.

### Cross-paradigm Grad-CAM (Paper Figure 3 candidate)

`Results/gradcam/cross_paradigm_disagreement.png` shows three rows of three columns (original CT slice with classical's verdict in caption / EffNet-B0 Grad-CAM / ConvNeXt V2 Grad-CAM):

- **Rows 1–2** (classical-XGB wrong, DL right): both DL backbones correctly attend to small focal high-density regions (visible bright calcifications); ConvNeXt V2's attention is markedly sharper than EffNet-B0's. Classical's whole-image texture aggregation cannot localise these focal lesions.
- **Row 3** (all three wrong, idx 790): both DL backbones attend to *different* wrong regions with moderate confidence (p=0.56, 0.65) — the dataset's irreducible difficulty.

Mechanistic claim for the paper: DL's spatial attention catches focal lesions that the classical paradigm's whole-image LBP+Gabor aggregation misses. This is direct evidence for *why* classical fails on the Stone class at full scale.

---

## Sprint 3 third addendum — overfitting diagnostics (post-tutor 2026-04-29)

Sandhya raised "the models could be overfitting" at Wednesday's tutor meeting. Four targeted diagnostics; none required retraining of any deployed pipeline.

### Diagnostic 1 — filename-proximity slice-leakage probe (`Results/diagnostics/filename_proximity.{json,png}`)

For each test image with class C and ID *i*, mean cosine similarity in 108-dim handcrafted feature space against (a) K=5 train images of class C with smallest |ID − i|, vs (b) K=5 random train images of class C. One-sided Mann-Whitney U.

| Class | Nearest-by-ID sim | Random sim | Ratio | p-value |
|---|---|---|---|---|
| Cyst | 0.9977 ± 0.0079 | 0.9660 ± 0.0143 | 1.033 | 1.3e-164 |
| Normal | 0.9991 ± 0.0045 | 0.9717 ± 0.0152 | 1.028 | 3.3e-239 |
| Stone | 0.9959 ± 0.0106 | 0.9762 ± 0.0143 | 1.020 | 1.6e-50 |
| Tumor | 0.9987 ± 0.0068 | 0.9707 ± 0.0124 | 1.029 | 3.2e-105 |

**Verdict.** Within-class baseline cosine similarity is already saturated at ~0.97 — kidneys of the same diagnostic class look very alike in this feature space. Nearest-by-ID adds only 2–3 % on top, statistically real but small effect. **The 108-dim handcrafted feature space is too coarse to detect patient identity above class identity.** Not a clear leakage signal in the classical features; a parallel test in DL embedding space remains future work.

### Diagnostic 2 — XGBoost learning curves over n_estimators (`Results/diagnostics/xgb_learning_curves.{json,png}`)

Refit deployed-pipeline XGB on cached train-only-fit scaler+PCA(50) features with `eval_set=[(train, val)]`, ceiling 400 estimators. Test re-evaluated at deployed (n=200) and val-best (n=399).

| Quantity | Value |
|---|---|
| Train mlogloss at deployed | very small (saturated) |
| Val mlogloss at deployed n=200 | 0.0299 |
| Val mlogloss minimum | 0.0218 (at round 399, still slowly decreasing) |
| Train merror at deployed | 0.0000 |
| Val merror at deployed | 0.0070 |
| Train–val merror gap | constant 0.7pp (does not widen with capacity) |
| Test macro-F1 at deployed (n=200) | 0.9871 |
| Test macro-F1 at val-best (n=399) | 0.9874 |
| Δ macro-F1 (val-best − deployed) | +0.0002 (negligible) |

**Verdict.** *Saturation pattern, not classifier overfit.* Train-val gap is fixed; val keeps slowly improving past the deployed point — the opposite of overfitting. Deployed n=200 is essentially indistinguishable from val-best n=399.

### Diagnostic 3 — per-class 5-fold stratified CV (`Results/diagnostics/per_class_cv.{json,png}`)

5-fold stratified CV on train+val combined (n=10,579) with deployed `best_params`. Per-fold per-class precision/recall/F1; compare to held-out test per-class F1.

| Class | CV F1 mean ± std | Held-out test F1 | Test − CV mean | (Test − CV) / std |
|---|---|---|---|---|
| Cyst | 0.9935 ± 0.0012 | 0.9937 | +0.0002 | +0.2 σ |
| Normal | 0.9958 ± 0.0007 | 0.9961 | +0.0003 | +0.4 σ |
| Tumor | 0.9935 ± 0.0023 | 0.9985 | +0.0050 | **+2.2 σ** |
| **Stone** | **0.9792 ± 0.0023** | **0.9703** | **−0.0089** | **−3.9 σ** |

| Aggregate | CV (mean ± std) | Held-out test |
|---|---|---|
| macro-F1 | 0.9905 ± 0.0010 | 0.9897 (within 1 σ) |

**Verdict — the strongest leakage signal in the four diagnostics.** Aggregate macro-F1 looks fine, but **Stone test F1 is 3.9 σ below CV mean** and Tumor is 2.2 σ above. This is per-class structural mismatch between train+val and the held-out test set — exactly what patient-level grouping would produce when the underlying patient population is unevenly distributed across the train+val/test partition and random stratified slice splitting cannot smooth it out. **Quantitative observation of the patient-leakage caveat.**

### Diagnostic 4 — DL learning curves from existing per-epoch logs (`Results/diagnostics/dl_learning_curves.{json,png}`)

Parsed `Results/dl_run_full/run_log.json` and `Results/convnextv2_full_run/run_log.json`. No retraining.

| Pipeline | Total epochs | Best epoch | Best val macro-F1 | Stage-2 val-loss rebound |
|---|---|---|---|---|
| EfficientNet-B0 (full) | 35 | 35 (last) | 0.9766 | +0.000 (none) |
| ConvNeXt V2 Base (full) | 35 | 34 | 0.9979 | +0.000 (none) |

**Verdict.** *Saturation, no rebound.* Both DL pipelines are if anything *under-trained* (still climbing at the last epoch); no val-loss rebound; early-stopping patience never triggered. **DL overfitting is not the cause of the 99 %+ accuracies.**

### Summary verdict for the paper

| Hypothesis tested | Diagnostic | Outcome |
|---|---|---|
| Classical XGB over-trained | XGB learning curves | Rejected — saturation pattern |
| Classical per-class variance hidden by aggregate | Per-class 5-fold CV | Variance rejected (std ≤ 0.0023); BUT structural mismatch *found* |
| DL backbones overfit at end of training | DL learning curves | Rejected — both under-trained, no val-loss rebound |
| Patient-level leakage inflates accuracy | Filename-proximity (Diag 1) + per-class CV (Diag 3) | Diag 1 inconclusive; Diag 3 supports — Stone difficulty mismatch is 3.9 σ |

**The diagnostic answer to Sandhya:** the 99 %+ numbers are not over-trained model artefacts — train and val curves saturate together for every pipeline and there is no overfit rebound. They reflect either genuine dataset signal or shared dataset-level structure (the per-class CV-vs-test mismatch on Stone is most consistent with patient-level grouping). Patient-level resplitting + a parallel filename-proximity test in DL embedding space are the natural follow-ups, deferred as future work.

---

## Interpretability artefacts

| Artefact | Path | Purpose |
|---|---|---|
| Grad-CAM panel (EfficientNet-B0 only) | `Results/gradcam/gradcam_panel.png` | Initial 8-panel Grad-CAM (4 correct + 4 errors) on the medium-set EfficientNet-B0 — Sprint 1 deliverable |
| **Cross-architecture Grad-CAM** | **`Results/gradcam/cross_architecture.png`** | **Paper Figure 1.** 6-row, 3-column comparison of EfficientNet-B0 (full) vs ConvNeXt V2 (full) attention on the same test images, drawn from paired-disagreement buckets |
| Sample manifest | `Results/gradcam/gradcam_manifest.json` | Reproducibility — which test images were selected and why |
| EfficientNet-B0 full predictions | `Results/dl_run_full/dl_predictions.npz` | y_true, y_pred, y_prob — input for paired McNemar's vs ConvNeXt V2 |
| ConvNeXt V2 full predictions | `Results/convnextv2_full_run/dl_predictions.npz` | y_true, y_pred, y_prob — input for paired McNemar's vs EfficientNet-B0 |
| Classical predictions | `Results/classical_run/classical_predictions.npz` | y_true, y_pred, y_prob — for paired McNemar's vs DL on medium set |
| Classical feature importance | *(pending — Person A to extract)* | Paper Figure 2 once available |

---

## State-of-the-art reference — Islam et al. (2022)

Results from the original paper on their balanced subset. **Different split and class distribution — directional comparison only.**

| Model | Accuracy |
|---|---|
| Swin Transformer | 99.30% |
| VGG16 | 98.20% |
| ResNet-50 | 73.80% |
| Inception v3 | 61.60% |
| Our ConvNeXtV2 (full, macro-F1) | 99.53% |
| Our EfficientNet-B0 + TTA (medium, macro-F1) | 98.29% |

---

## Dataset summary

| Version | Images | Cyst | Normal | Stone | Tumor |
|---|---|---|---|---|---|
| Small | 3,110 | ~927 | ~1,269 | ~344 | ~570 |
| Medium (used) | 6,221 | 1,854 | 2,538 | 688 | 1,141 |
| Full | 12,446 | 3,709 | 5,077 | 1,377 | 2,283 |

**Split used**: 70 / 15 / 15 train / val / test, stratified by class, seed=42.
Medium test set breakdown: Cyst 279 · Normal 381 · Stone 103 · Tumor 171.
