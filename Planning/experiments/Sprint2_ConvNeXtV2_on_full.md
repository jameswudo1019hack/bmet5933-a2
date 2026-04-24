# Sprint 2 — ConvNeXt V2 supplementary run on the full dataset

**Date**: 2026-04-24
**Status**: infrastructure in progress
Related: [[Sprint2_evaluation_ConvNeXtV2]] (the decision rationale), [[Sprint1_log]], [[DL_Improvements_Analysis]], [[Phase2_Design]], [[Project_Framing_v2]]

> **Framing note (added 2026-04-24).** This run is now read under the paradigm-comparison framing in [[Project_Framing_v2]]. The primary scientific output of the ConvNeXt V2 run is **not its test accuracy** — it is the answer to: *does a much larger, higher-resolution, more modern CNN attend to the same visual structure as EfficientNet-B0, or different structure?* The headline per-epoch and test metrics are secondary; the Grad-CAM attention-map comparison across architectures is the real experiment. Plan the post-run analysis accordingly.

---

## Decision

User chose **Option B** from [[Sprint2_evaluation_ConvNeXtV2]] with a specific scope: **single ConvNeXt V2 training run on the full dataset (12,446 images)**, positioned as a direct scale-validation experiment against Islam et al.'s Swin Transformer baseline (99.30 % accuracy).

Not in scope:
- Rebuilding classical / EfficientNet-B0 / ensemble on the full dataset (preserves the apples-to-apples medium-dataset comparison already in hand)
- 5-seed ensemble of ConvNeXt V2 (deferred to future work)
- Data-efficiency sweep with the new architecture (not needed for the scale-validation framing)

## Paper framing (draft)

Two parallel axes of results:

1. **Primary comparison** — classical vs DL on the **medium** dataset (6,221 images), shared 70/15/15 split:
   - Classical XGBoost: macro-F1 0.9976
   - DL (EfficientNet-B0 + TTA hflip): macro-F1 0.9829
   - Equal-weight soft-vote ensemble: macro-F1 **1.0000** (0 errors / 934)

2. **Scale validation** — single ConvNeXt V2 Base run on the **full** dataset (12,446 images), independent 70/15/15 split:
   - Reports macro-F1 / accuracy / Stone F1 vs Islam et al.'s 99.30 % Swin (on their balanced subset)
   - Provides evidence that our classical-wins result is not a compute-constrained artefact

## Configuration

| Parameter | Value | Justification |
|---|---|---|
| Backbone | ConvNeXt V2 Base | Modern large-capacity CNN; preserves CNN-vs-handcrafted paradigm |
| Pretraining | ImageNet-22k → ImageNet-1k @ 384 | Strongest publicly-available transfer starting point |
| Input resolution | 384×384 | Native pretrain resolution; retains pixel detail for stones |
| Dataset | Full (12,446 images) | Matches Islam et al. size; better overfitting protection for 89M params |
| Split | Stratified 70/15/15, seed=42, `split_full.csv` | Same seed as medium; fresh split CSV because image set differs |
| Optimiser | AdamW, weight decay 0.05, LR 1e-5 (fine-tune) | Proposal defaults; weight decay higher than B0 for the larger capacity |
| Regularisation | Stochastic depth 0.3, dropout 0.3 on head | Mitigates overfitting on ~8,700 training samples |
| Augmentation | Same as EfficientNet-B0 (hflip, rot ±15°, zoom 0.1, brightness/contrast ±0.1) | Consistency with Phase 2 augmentation policy |
| Class weights | Inverse-frequency, same as B0 | Consistency |
| Mixed precision | bfloat16 on A100 | Speed; A100-native format |
| Batch size | 32 (may bump to 64 if VRAM allows) | Conservative start |
| Two-stage | Stage 1 (5 epochs head-only) + Stage 2 (up to 30 with early-stop patience 5) | Same protocol as B0 |

## Expected outcomes and paper handling

| Scenario | Probability (subjective) | Paper handling |
|---|---|---|
| ConvNeXt V2 macro-F1 ≥ 0.995 | ~35 % | "Scale + modern architecture closes most of the gap to classical but does not eliminate it" |
| ConvNeXt V2 0.97–0.99 | ~45 % | "Consistent with EfficientNet-B0 baseline; the classical-DL gap on this dataset is architecture-robust" |
| ConvNeXt V2 < 0.97 (overfits) | ~20 % | Report honestly: "The larger architecture overfits at this dataset scale, underscoring that representational capacity is not the bottleneck on this task" |

All three outcomes strengthen the paper; the framing adjusts accordingly.

## Infrastructure changes needed

### Code

- `shared/split.py` — optionally emit `split_full.csv` alongside the existing `split.csv`, or add a `--dataset-root`/`--output` CLI
- `deep_learning/model.py` — add `build_convnextv2_base()` using `timm`
- `deep_learning/dataset.py` — make `CNN_IMAGE_SIZE` configurable per run; accept a custom `split.csv` path
- `deep_learning/train.py` — new CLI flags `--model`, `--split-csv`, `--image-size`, `--weight-decay`, `--stochastic-depth`
- `deep_learning/predict.py` — mirror the same flags
- `requirements.txt` — add `timm>=1.0.9`
- `notebooks/colab_train.ipynb` — add Sprint 2 section (download full.zip, run ConvNeXt V2 on full)

### User-side (one-time)

- Upload `full.zip` (≈1.6 GB) to `MyDrive/BMET5933/full.zip` (alongside `partial.zip`). ~20 min Drive upload.

## Open questions resolved up front

- **Soft-vote ensemble of ConvNeXt V2 + classical?** No — classical was trained on medium, ConvNeXt V2 is on full. Test sets differ. Reporting each independently.
- **Retraining classical on full for apples-to-apples?** No — preserves partner's existing work; framed as scale-validation only.
- **Islam et al. reported 99.30 % on their balanced 1,300/class subset. Is that comparable to ours?** Not perfectly — they rebalanced; we keep natural imbalance. The comparison is directional, not identity. Paper will note this clearly.

## Iteration sub-log

### 1 — Log + decision

Status: complete (this document).

### 2 — Infrastructure implementation

Status: in progress.

### 3 — Local smoke test

Status: pending.

### 4 — Colab training run

Status: pending (user runs after upload of full.zip).

### 5 — Results interpretation + paper paragraph

Status: pending.
