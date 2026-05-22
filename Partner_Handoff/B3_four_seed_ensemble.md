# B3 — Four-seed ensemble decision

## TL;DR: keep the sentence — we have the ensemble result.

The 4-seed ConvNeXt V2 (cosine LR + 60 stage-2 epochs + 5-epoch warmup, seeds {42, 0, 1, 2}, each with TTA hflip) soft-vote ensemble was trained on Colab Pro+ in Sprint 5 Tier 1A+2C (2026-05-13/14). Final macro-F1 on the deduplicated test:

| Configuration | Macro-F1 | Accuracy | Errors / 1,888 |
|---|---|---|---|
| ConvNeXt V2 baseline (single seed, constant LR) | 0.8219 | 0.8538 | 276 |
| ConvNeXt V2 + TTA hflip (single seed) | 0.8370 | 0.8665 | 252 |
| **ConvNeXt V2 4-seed cosine+60 + TTA hflip ensemble** | **0.8374** | **0.8649** | **255** |

Net improvement of the 4-seed ensemble over the single-seed baseline: **+1.55 pp**. Over single-seed + TTA: +0.03 pp.

## Suggested Methods II-C wording (replaces "either run the ensemble or remove the sentence")

> *"We also report a 4-seed soft-vote ensemble of ConvNeXt V2 Base trained with cosine learning-rate schedule and 60 stage-2 epochs (seed=42 from the protocol-matched baseline plus three additional seeds 0, 1, 2), with horizontal-flip test-time augmentation applied to each before averaging the softmax outputs. The ensemble achieves macro-F1 = 0.8374 on the deduplicated test set, +1.6 pp above the protocol-matched single-seed baseline; still 7.2 pp below the classical SVM (0.9091)."*

Artefacts:
- `Results/convnextv2_4seed_cos_tta_ensemble/dl_predictions.npz` and `dl_results.json`
- Individual seed runs: `Results/convnextv2_full_run_seed{0,1,2}_cos_tta_hflip/`
- Methods detail: `notebooks/colab_3seed_convnextv2_cos.ipynb`
