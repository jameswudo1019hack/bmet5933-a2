# Results Summary

All canonical numbers in one place. Pull from here when writing the paper.
Last updated: 2026-04-26

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

## EfficientNet-B0 on full dataset (for reference)

Same full dataset split, n=1867 test.

| Metric | Value |
|---|---|
| Accuracy | 0.9877 |
| Macro-F1 | 0.9819 [0.975, 0.989] |
| Stone F1 | 0.9496 |
| Errors | 23 / 1867 |

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
