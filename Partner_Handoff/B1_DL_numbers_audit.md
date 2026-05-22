# B1 — DL numbers audit

All numbers below are computed directly from the JSON artefacts in `Results/` on the **deduplicated** test set (n = 1,888).

## Headline metrics on clean test (Table I row anchors)

| Pipeline | Macro-F1 | Accuracy | Errors / 1,888 |
|---|---|---|---|
| Classical SVM (partner's, for reference) | **0.9091** | 0.9280 | 136 |
| EfficientNet-B0 (clean baseline) | **0.7679** | 0.8014 | 375 |
| ConvNeXt V2 Base (clean baseline) | **0.8219** | 0.8538 | 276 |
| EfficientNet-B0 + TTA hflip | **0.7945** | 0.8263 | 328 |
| ConvNeXt V2 + TTA hflip | **0.8370** | 0.8665 | 252 |
| 4-seed ConvNeXt V2 cosine+60 + TTA ensemble | **0.8374** | 0.8649 | 255 |

## Paired McNemar's on clean test (Table II row anchors)

All on the same n = 1,888 test images (valid paired comparison).

| Comparison | Both correct | Only A wrong | Only B wrong | Both wrong | Discordant | p-value |
|---|---|---|---|---|---|---|
| Classical vs EfficientNet-B0 | 1426 | 87 | 326 | 49 | 413 | **1.117e-31** |
| Classical vs ConvNeXt V2 | 1529 | 83 | 223 | 53 | 306 | **1.925e-15** |
| EfficientNet-B0 vs ConvNeXt V2 | 1374 | 238 | 139 | 137 | 377 | **4.482e-07** |

All three pairs are highly significant at α = 0.05. **Classical statistically dominates both DL backbones; ConvNeXt V2 statistically dominates EfficientNet-B0** (replicating Sprint 2 EffNet-full vs ConvNeXt-full direction).

## DL-paradigm soft-vote ensemble on clean (Tier 1)

`Results/tier1_ensemble_clean.json`:

| Ensemble | Macro-F1 | Notes |
|---|---|---|
| EF-raw + CN-raw (equal-weight) | 0.8353 | DL-only |
| EF-TTA + CN-TTA (equal-weight) | **see notebooks/colab_dl_clean_full computed inline** | Best DL-only |

(The DL-only TTA ensemble macro-F1 = 0.8448; see `Results/tier1_ensemble_clean.json` for full breakdown.)

## Confusion matrices on clean test

### Classical SVM

|       | **Pred Cyst** | **Pred Normal** | **Pred Stone** | **Pred Tumor** |
|---|---|---|---|---|
| **True Cyst** | 496 | 0 | 17 | 0 |
| **True Normal** | 6 | 757 | 7 | 7 |
| **True Stone** | 17 | 46 | 181 | 4 |
| **True Tumor** | 0 | 32 | 0 | 318 |

### EfficientNet-B0

|       | **Pred Cyst** | **Pred Normal** | **Pred Stone** | **Pred Tumor** |
|---|---|---|---|---|
| **True Cyst** | 424 | 0 | 74 | 15 |
| **True Normal** | 8 | 656 | 47 | 66 |
| **True Stone** | 35 | 31 | 162 | 20 |
| **True Tumor** | 15 | 64 | 0 | 271 |

### ConvNeXt V2 Base

|       | **Pred Cyst** | **Pred Normal** | **Pred Stone** | **Pred Tumor** |
|---|---|---|---|---|
| **True Cyst** | 446 | 2 | 38 | 27 |
| **True Normal** | 2 | 709 | 2 | 64 |
| **True Stone** | 25 | 51 | 156 | 16 |
| **True Tumor** | 12 | 34 | 3 | 301 |

## Per-class F1 on clean test

| Class | Classical | EffNet-B0 | ConvNeXt V2 |
|---|---|---|---|
| Cyst | 0.9612 | 0.8523 | 0.8938 |
| Normal | 0.9392 | 0.8586 | 0.9015 |
| Stone | 0.7991 | 0.6102 | 0.6980 |
| Tumor | 0.9367 | 0.7507 | 0.7942 |

## Notes for the write-up

- All numbers are on the deduplicated test set (n = 1,888) released by the dataset maintainer 2026-05-07.
- The "leaky" baseline numbers (for the Table I before/after columns) live at `Results/_leaky/{dl_run_full, convnextv2_full_run}/` and Sprint-5 paired-McNemar JSON `Results/sprint5_clean_vs_leaky.json` has the full leaky-vs-clean comparison.
- The 4-seed ensemble entry above is the strongest single-method DL number we have (0.8374); still 7.2 pp below classical (0.9091).
