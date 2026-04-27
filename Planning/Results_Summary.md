# Results Summary

All canonical numbers in one place. Pull from here when writing the paper.
Last updated: 2026-04-27 (Sprint 3 — classical on full)

---

## Primary comparison — medium dataset (6,221 images, 70/15/15, seed=42)

Both models evaluated on the **same** 934-image test set. Apples-to-apples.

### Overall metrics

| Model | Accuracy | Macro-F1 | Weighted F1 | ROC-AUC | Errors |
|---|---|---|---|---|---|
| Classical XGBoost | 0.9979 | 0.9976 | 0.9979 | 0.99999 | 2 / 934 |
| EfficientNet-B0 baseline | 0.9797 | 0.9745 | 0.9797 | 0.9995 | 19 / 934 |
| EfficientNet-B0 + TTA hflip ← **canonical DL** | 0.9861 | **0.9829** | 0.9861 | 0.9998 | 13 / 934 |
| Ensemble val-tuned (collapses to classical) | 0.9979 | 0.9976 | 0.9979 | — | 2 / 934 |
| Ensemble equal-weight w=0.5 ← **headline** | **1.0000** | **1.0000** | 1.0000 | — | **0 / 934** |

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

## Supplementary — ConvNeXtV2 Base on full dataset

Different test set (n=1867, full dataset split). **Not directly comparable to medium results above.**

| Metric | Value |
|---|---|
| Accuracy | 0.9968 |
| Macro-F1 | **0.9953** [0.991, 0.998] |
| Weighted F1 | 0.9968 |
| ROC-AUC | 0.99998 |
| Errors | 6 / 1867 |
| Best epoch | 34 (stage 1: 5, stage 2: 29) |
| Training time | ~54 min (A100, bfloat16) |

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
