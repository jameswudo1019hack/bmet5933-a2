# B2 — Val-test gap verification

## Headline (the numbers you asked me to confirm)

| Quantity | Value | Source |
|---|---|---|
| ConvNeXt V2 best val macro-F1 | **0.9125** | `Results/convnextv2_full_run/run_log.json` (best over 35 epochs, epoch 17) |
| ConvNeXt V2 test macro-F1 | **0.8219** | `Results/convnextv2_full_run/dl_results.json` |
| **Val − test gap (ConvNeXt V2)** | **0.0907 = 9.1 pp** | derivation: 0.9125 − 0.8219 = 0.0907 |

**Confirmed.** Partner's rounded "0.913 val and 0.822 test" matches the unrounded 0.9125 / 0.8219. The 9.1 pp gap is exact when rounded to one decimal place.

## Derivation steps for the paper

```
best_val_macro_f1   = 0.9125  (from run_log.json's per-epoch trace; best was at epoch 17)
test_macro_f1       = 0.8219  (from dl_results.json, run on the n=1,888 deduplicated test set)
val_test_gap        = 0.9125 − 0.8219
                    = 0.0907
                    ≈ 9.1 percentage points
```

## EfficientNet-B0 comparison (for context)

| Quantity | Value |
|---|---|
| EffNet-B0 best val macro-F1 | **0.7598** (epoch 28) |
| EffNet-B0 test macro-F1 | **0.7679** |
| Val − test gap (EffNet) | **-0.8 pp** *(note sign: test slightly higher than val, no overfit gap)* |

Striking contrast: **ConvNeXt V2 has a +9.1 pp val-test gap (val higher than test), EffNet-B0 has none**. The bigger model is more vulnerable to the post-deduplication train-val distribution shift; the smaller model isn't.

## Interpretation (one-paragraph candidate for the paper)

> *On the deduplicated dataset, ConvNeXt V2 Base showed a 9.1 pp gap between best validation macro-F1 (0.9125) and held-out test macro-F1 (0.8219), while EfficientNet-B0 showed essentially no gap (-0.8 pp). This asymmetric val-test gap is consistent with the larger architecture overfitting to validation-specific structure that the test set does not share — a residual signal of dataset structure that persists even after the maintainer's file-level deduplication. EfficientNet-B0's smaller capacity makes it less susceptible to this effect; the trade-off appears as ConvNeXt's larger 100% advantage on test (Table I) but also as a larger inflation of its val-set number.*
