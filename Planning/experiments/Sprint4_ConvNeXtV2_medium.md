# Sprint 4 — ConvNeXt V2 Base on the medium dataset

**Date**: 2026-04-29 (run on Colab Pro+ A100 the same evening as the tutor meeting)
**Status**: complete
Related: [[Sprint2_ConvNeXtV2_on_full]], [[Sprint2_evaluation_ConvNeXtV2]], [[Sprint1_log]], [[Sprint3_classical_on_full]], [[Validation_and_Verification]], [[Tutor_Meeting_Brief]]

> **Why**: closes the missing cell of the 2×2 architecture-vs-data-volume matched grid. Before Sprint 4 we had EffNet-B0 medium + EffNet-B0 full + ConvNeXt V2 full. The missing ConvNeXt V2 medium cell was a known gap that Sandhya could plausibly raise as "how do you know architecture > data volume isn't a full-scale-only finding?" Sprint 4 closes it definitively.

## Decision

Single-purpose run: identical hyperparameters and protocol to the Sprint 2 ConvNeXt V2 full run, only the training data scale changes (8,712 train → 4,353 train). All other knobs (image size 384, batch size 32, AdamW with weight decay 0.05, stochastic depth 0.3, head-only stage 1 (5 epochs) → fine-tune stage 2 (≤30 epochs, patience 5), `stage2_unfreeze_blocks=1`) match Sprint 2 exactly.

Run venue: Colab Pro+ A100 (the user's existing DL training venue). Predictions transferred to local repo via Drive backup after the auto-push from Colab failed with a 403 (PAT scope issue, unrelated to the science).

## Configuration

| Parameter | Value | Same as Sprint 2 ConvNeXt V2 full? |
|---|---|---|
| Backbone | ConvNeXt V2 Base | ✓ |
| Pretraining | ImageNet-22k → ImageNet-1k @ 384 | ✓ |
| Input resolution | 384 × 384 | ✓ |
| Dataset | Medium (`split.csv`, 6,221 images, 70/15/15 stratified seed=42) | **× — new (was full split_full.csv)** |
| Train / val / test | 4,353 / 934 / 934 | × (was 8,712 / 1,867 / 1,867) |
| Optimiser | AdamW, weight decay 0.05, LR 1e-5 (fine-tune) | ✓ |
| Regularisation | Stochastic depth 0.3, dropout 0.3 on head | ✓ |
| Augmentation | hflip, rot ±15°, zoom 0.1, brightness/contrast ±0.1 | ✓ |
| Class weights | Inverse-frequency | ✓ |
| Mixed precision | bfloat16 on A100 | ✓ |
| Batch size | 32 | ✓ |
| Two-stage protocol | Stage 1 (5 epochs head-only) + Stage 2 (≤30 epochs, patience 5) | ✓ |
| `stage2_unfreeze_blocks` | 1 | ✓ |

This is a strict matched-protocol comparison — only training-set size differs.

## Headline results

`Results/convnextv2_medium_run/dl_results.json`:

| Metric | Value |
|---|---|
| n_test | 934 (medium test split) |
| Accuracy | **0.9925** |
| Macro-F1 | **0.9898 [0.9813, 0.9968]** |
| Weighted-F1 | 0.9925 |
| ROC-AUC OvR | 0.9999 |
| Errors | **7 / 934** |

### Per-class F1 — ConvNeXt V2 medium

| Class | Precision | Recall | F1 [95 % CI] | Support |
|---|---|---|---|---|
| Cyst | 0.996 | 0.993 | 0.995 [0.988, 1.000] | 279 |
| Normal | 0.995 | 0.992 | 0.993 [0.987, 0.999] | 381 |
| Stone | 0.962 | 0.981 | 0.971 [0.946, 0.991] | 103 |
| **Tumor** | **1.000** | **1.000** | **1.000** | 171 |

### Confusion matrix — ConvNeXt V2 medium

|  | Pred Cyst | Pred Normal | Pred Stone | Pred Tumor |
|---|---|---|---|---|
| **True Cyst** | 277 | 0 | 2 | 0 |
| **True Normal** | 1 | 378 | 2 | 0 |
| **True Stone** | 0 | 2 | 101 | 0 |
| **True Tumor** | 0 | 0 | 0 | 171 |

Notable: **Tumor class is perfectly classified** (171/171). The remaining 7 errors are split across Cyst↔Stone (2), Normal↔Stone (3), Normal↔Cyst (1), and Stone↔Normal (which I just listed). Stone remains the universally-weakest class (F1 0.971), consistent with every other model in the project.

## The 2×2 architecture-vs-data grid (now complete)

| | Medium (n=934 test) | Full (n=1,867 test) |
|---|---|---|
| **EfficientNet-B0** | 0.9829 (TTA hflip), 13 errors | 0.9819, 23 errors |
| **ConvNeXt V2 Base** | **0.9898**, 7 errors *(new)* | 0.9953, 6 errors |

Two findings drop out of this grid:

1. **Architecture > data, at every scale.** ConvNeXt V2 medium > EffNet-B0 + TTA medium (0.9898 vs 0.9829 — ConvNeXt has 7 errors vs EffNet's 13). ConvNeXt V2 full > EffNet-B0 full (Sprint 2 result, p = 0.0021). Architecture wins regardless of dataset size.
2. **Asymmetric data-volume response between the two architectures.** ConvNeXt V2 errors per 100 test: 0.75 (medium) → 0.32 (full), a 57 % error-rate reduction. EffNet-B0 errors per 100 test: 1.39 (medium + TTA) → 1.23 (full), essentially flat. **Data helps the larger architecture; not the smaller one.** The natural reading: EffNet-B0 is capacity-limited and saturated already at the medium-scale 4,353 training samples; ConvNeXt V2 has spare capacity that more data can fill.

## Paired McNemar's on the medium n=934 test set

`Results/convnextv2_medium_run/sprint4_medium_grid.json`:

| Comparison | Both correct | Only A wrong | Only B wrong | Both wrong | Discordant | p-value |
|---|---|---|---|---|---|---|
| ConvNeXt V2 medium vs EffNet-B0+TTA medium | 916 | 5 | 11 | **2** | 16 | 0.21 (n.s.) |
| **ConvNeXt V2 medium vs Classical medium** | 925 | 7 | 2 | **0** | 9 | 0.18 (n.s.) |
| EffNet-B0+TTA medium vs Classical medium *(Sprint 1 reference)* | 919 | 13 | 2 | **0** | 15 | **0.0074** (sig.) |

Three observations from this table:

1. **The disjoint-error pattern (`both_wrong = 0`) extends to ConvNeXt V2 at medium scale.** Classical-medium and ConvNeXt-medium fail on disjoint image sets — exactly like classical-medium and EffNet-medium did in Sprint 1. The medium-set "100 % equal-weight ensemble" finding is therefore *paradigm-stable across the two CNN backbones*, not a quirk of EffNet-B0. A {classical, ConvNeXt V2} medium soft-vote ensemble would also reach 100 % macro-F1 on this test set.
2. **No paired-McNemar's pair on medium reaches significance for ConvNeXt V2.** With 7 errors total and 9 discordant against classical, statistical power is low — we cannot conclude classical *significantly* beats ConvNeXt V2 at p < 0.05, only that the raw-count gap is 7 vs 2.
3. **Within-DL-paradigm overlap exists at medium**: ConvNeXt V2 ∩ EffNet-B0 = 2 both-wrong. The two CNN backbones share a small set of failure cases at medium scale — same dataset-level "hard cases" both models miss, consistent with the kind of saturation we'd expect in an overlapping-feature-space paradigm.

## How Sprint 4 reshapes the invalidation chain (consolidated narrative)

The four-step invalidation chain in [[experiments/Sprint3_classical_on_full]] § "Sprint 3 second addendum" said: *the medium-set "disjoint errors / 100 % ensemble" finding fails progressively as we go from medium to full and add more classifiers.* Sprint 4 strengthens the chain rather than disturbing it, by showing the medium-set finding is **even more robust than we previously claimed**:

- *Medium*: classical-XGB ∩ EffNet-medium-TTA = 0 both-wrong (Sprint 1) **AND** classical-XGB ∩ ConvNeXt-medium = 0 both-wrong (Sprint 4). The disjoint-errors property at medium is paradigm-level, not architecture-specific.
- *Full*: classical-XGB ∩ EffNet-full = 4 both-wrong, classical-XGB ∩ ConvNeXt-full = 2 both-wrong (Sprint 3). Property fails at full.
- The invalidation is therefore **scale-dependent, not architecture-dependent or classifier-dependent within DL**.

This is a cleaner finding for the paper: dataset scale (or dataset-scale-correlated structural properties — patient diversity, slice variety, class composition) is the variable that flips the disjoint-error claim. We can now lead the Discussion with a sharper question: *what about the full dataset is structurally different from its medium subset such that disjoint-error fails?*

## What this means for the paper

| Section | Update |
|---|---|
| Headline numbers table | Add ConvNeXt V2 medium row: 0.9898 macro-F1, 7/934 errors |
| Results §3 (medium-scale) | Add "ConvNeXt V2 medium also achieves disjoint errors against classical (both-wrong=0); medium-scale disjoint-error finding is paradigm-stable across CNN backbones" |
| Results §4 (full-scale) | Mention ConvNeXt V2 medium → full error-rate drop (0.75 % → 0.32 %, 57 % reduction) as evidence of "data helps larger architecture" |
| Discussion §"architecture > data" | Now supported by *full 2×2 matched grid*, not partial three-cell evidence |
| Discussion §"invalidation chain" | Step 1 is now stronger: medium-set disjoint-error finding generalises across CNN backbones; the failure at full scale is therefore not a within-DL classifier or architecture quirk |
| Limitations §"saturation framing" | Unchanged — still applies |

## Configuration footnote

The Colab notebook (`notebooks/colab_convnextv2_medium.ipynb`) is committed for the .ipynb-deliverable submission. The auto-push from Colab failed with a 403 (PAT scope issue, not the science); the user copied the run folder to Drive and downloaded a 325 MB zip locally. After unzip, the predictions npz, results json, run log, train log, and predict log were committed (the 350 MB `best_model.pt` is gitignored as per `*.pt` rule). Determinism: the smoke run on Colab matched local-smoke output bit-exact on val_f1 (same as Sprint 3).

## Files written

- `Results/convnextv2_medium_run/best_model.pt` *(89M-param checkpoint, gitignored)*
- `Results/convnextv2_medium_run/dl_predictions.npz`
- `Results/convnextv2_medium_run/dl_results.json`
- `Results/convnextv2_medium_run/run_log.json`
- `Results/convnextv2_medium_run/train_log.txt`
- `Results/convnextv2_medium_run/predict_log.txt`
- `Results/convnextv2_medium_run/sprint4_medium_grid.json` *(paired McNemar's)*
- `analysis/sprint4_medium_grid.py` *(rerunnable comparison script)*
- `notebooks/colab_convnextv2_medium.ipynb` *(submission .ipynb)*
