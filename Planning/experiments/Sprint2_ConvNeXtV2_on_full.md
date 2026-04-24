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

Status: partial — numeric results in. Full interpretation blocked on iteration 6 (matched-data EfficientNet-B0 run).

**2026-04-24 — test-set numbers for ConvNeXt V2 full**:

| Metric | Value |
|---|---|
| Accuracy | 0.9968 |
| Macro-F1 | 0.9953  [0.991, 0.998] |
| ROC-AUC  | 0.99998 |
| Errors   | 6 / 1867 (0.32 %) |
| Per-class F1 | Cyst 0.995 · Normal **1.000** · Stone 0.988 · Tumor 0.999 |
| Dominant error | Cyst ↔ Stone (5/6 of errors) |
| Wall time | 53.8 min on A100 |
| Early stopping | Did NOT fire; ran full 30 stage-2 epochs (under-trained ceiling) |

Ties or beats Islam et al. 2022's Swin (99.30 % accuracy on balanced 1300/class) on every per-class F1, on the harder natural-imbalance test distribution.

### 6 — Matched-data EfficientNet-B0 run (for fair architecture comparison)

Status: complete 2026-04-24.

**Why**: the ConvNeXt V2 full result is confounded — we cannot tell whether its gains over EfficientNet-B0 come from *architecture* (89M params @ 384) or from *data volume* (2× training images). Under framing v2, isolating these is essential for the interpretability analysis.

**What**: retrain EfficientNet-B0 on `split_full.csv` with the same two-stage protocol as the medium-dataset run. Same architecture, same hyperparameters, only the training data changes.

**Decompositions this enables**:

| Comparison | Isolates |
|---|---|
| EffNet medium vs EffNet full | Data-volume effect (fixed architecture) |
| EffNet full vs ConvNeXt V2 full | Architecture effect (fixed data) |
| Classical medium vs EffNet/ConvNeXt full | Paradigm effect with caveat about split |

**Why it matters under framing v2**: without this, the Grad-CAM cross-architecture comparison (next step) would be apples-to-oranges — different training distributions. With matched training data, the attention-map differences measure architecture-induced representational differences.

**Expected artefacts**:
- `Results/dl_run_full/best_model.pt`, `run_log.json`, `dl_results.json`, `dl_predictions.npz`
- Enables paired McNemar's test on the 1867-sample full test set
- Enables same-image Grad-CAM comparison across DL backbones

#### Results (iteration 6)

Colab A100, 18.4 min wall time, ran all 30 stage-2 epochs (early stopping not triggered). Best val macro-F1 = 0.9766 at epoch 35.

**Test-set metrics on full split (n=1867)**:

| Metric | Value |
|---|---|
| Accuracy | 0.9877 |
| Macro-F1 | 0.9819 |
| Per-class F1 | Cyst 0.985 · Normal 0.996 · **Stone 0.950** · Tumor 0.997 |
| Errors | 23 / 1867 (1.23 %) |
| Dominant error | **Cyst ↔ Stone (17/23 = 74 %)** |

**Paired McNemar's on the same 1867 test set (EffNet-full vs ConvNeXt V2-full)**:

| | |
|---|---|
| Both correct | 1839 |
| Only EffNet wrong | **22** |
| Only ConvNeXt V2 wrong | **5** |
| Both wrong | 1 |
| Discordant pairs | 27 |
| **p-value** | **0.0021** |

Architecture effect is **statistically significant**. ConvNeXt V2 fixes 22 of EffNet's 23 errors while only breaking 5 that EffNet got right.

#### Three findings (under framing v2)

1. **Architecture effect is real and large.** On matched data, ConvNeXt V2 reduces DL error rate from 1.23 % → 0.32 % (−74 %). McNemar's p=0.002. The ConvNeXt V2 advantage cannot be attributed to data volume.

2. **Data volume is nearly a no-op for EfficientNet-B0.** Relative error rate barely moves: medium+TTA 1.29 % → full 1.23 %. The 5M-parameter architecture has extracted everything it can from this dataset; adding 4,359 more training images buys essentially nothing. EffNet's capacity, not the training set size, is the bottleneck.

3. **Error-direction convergence across DL on matched data**:

| Model | Dominant error | % of total |
|---|---|---|
| EfficientNet-B0 full | Cyst ↔ Stone | 74 % |
| ConvNeXt V2 full | Cyst ↔ Stone | 83 % |
| Classical (medium) | Cyst ↔ Tumor | 100 % |

Both DL architectures fail on the *same class pair* (Cyst ↔ Stone), regardless of capacity or resolution. Classical texture features fail on a *different* pair (Cyst ↔ Tumor). This is the central empirical observation for the paper's paradigm-comparison thesis.

#### Bonus observation — data volume shifts error direction

The same architecture (EfficientNet-B0) with different training set sizes makes *differently-distributed* errors:

| | Dominant error | Count |
|---|---|---|
| EffNet medium (baseline) | Stone → Normal | 5/19 |
| EffNet medium + TTA hflip | Stone → Normal | ≈5/12 |
| EffNet full | Cyst ↔ Stone | 17/23 |

More training data teaches EffNet to distinguish Stone from Normal, but the spare capacity gets spent confusing Stone with Cyst instead. One sentence for the paper's Discussion: *"increased training data shifts but does not reduce overall DL confusion at this architecture scale."*

### 7 — Grad-CAM cross-architecture interpretability (the central paper figure)

Status: complete 2026-04-24.

**Module**: `deep_learning/gradcam_compare.py` — targets `model.features[-1]` for EffNet-B0 and `model.stages[-1]` for ConvNeXt V2; preprocessing uses each architecture's native resolution (224 vs 384); both heatmaps upsampled to 384 for a fair visual side-by-side.

**Sample selection**: 6 test images drawn automatically from paired disagreement buckets:
- 2 both-correct controls
- 3 only-EffNet-wrong (highest-confidence wrong predictions in EffNet's wrong set — i.e. where EffNet is most confidently mistaken while ConvNeXt V2 gets it right)
- 1 only-ConvNeXt-wrong (reverse direction)

Output: `Results/gradcam/cross_architecture.png`.

#### What the selection itself reveals

All 4 disagreement-bucket samples are **Cysts that one model mistakes for Stone**. This confirms empirically, at the individual-image level, that the Cyst ↔ Stone confusion identified in iteration 6's error breakdown *is* where the DL-architecture comparison lives. Nothing else dominates the disagreement space.

#### What the attention maps show

| Observation | Evidence from figure |
|---|---|
| EfficientNet-B0 attention is **diffuse and dispersed** | Heatmaps fill much of the abdomen, frequently extending outside kidney tissue into bone, body wall, and bowel |
| ConvNeXt V2 attention is **localised and kidney-anchored** | Heatmaps concentrate on rounded kidney-region structures in nearly every example |
| On "only EffNet wrong" cases, **EffNet's attention wanders off-organ** | The three Cyst → Stone errors all show EffNet peaking outside the kidney silhouette, while ConvNeXt V2 correctly focuses on the kidney lesion |
| On the single "only ConvNeXt wrong" case, **ConvNeXt V2 is confidently focused on the wrong local feature** | Tightly-focused heatmap, but on a small hyper-dense structure that shape-wise resembles a stone — a reasonable mistake given density-based cues are ambiguous in this image |

#### Paper-ready paragraph (draft)

> To understand the architectural basis of the paradigm comparison, we generated Grad-CAM attention maps from both EfficientNet-B0 and ConvNeXt V2 on the same six test-set images, drawn from the paired disagreement set. EfficientNet-B0's attention is visibly more dispersed, frequently extending beyond the kidney silhouette into body wall and bowel, whereas ConvNeXt V2's attention is localised and kidney-anchored across all examples. On the three Cyst → Stone misclassifications unique to EfficientNet-B0, the smaller network's attention peaks outside the kidney lesion; ConvNeXt V2 on the same images correctly fixates on the lesion and predicts Cyst. This supports our central claim that the architectural gap we measure quantitatively (74 % fewer errors, p = 0.002) reflects a genuine difference in *where* the networks look, not merely how well they score. It also suggests that the Cyst ↔ Stone failure mode shared by both DL architectures stems from the two classes' shape-level similarity (both rounded, similar size), which a higher-capacity network at higher resolution is better at disambiguating but neither architecture solves completely — consistent with classical texture features being the only approach that achieves perfect Stone F1 on this dataset.

#### Artefacts

- `deep_learning/gradcam_compare.py` — cross-architecture Grad-CAM module
- `Results/gradcam/cross_architecture.png` — the paper figure (18 panels; 6 rows × 3 columns)

---

## Sprint 2 summary

Under framing v2, Sprint 2 produced three empirically-grounded contributions to the paper:

1. **ConvNeXt V2 full-dataset baseline** (iteration 4/5): directly comparable to Islam et al.'s Swin, beats on every per-class F1 on a harder test distribution. Reported but not the headline.

2. **Three-way architecture × data-volume decomposition** (iteration 6):
    - Architecture effect: ConvNeXt V2 −74 % DL error rate vs EffNet-B0 at matched data (p=0.002)
    - Data-volume effect: ~0 (EffNet-B0's 5M-param capacity saturates on medium)
    - Error-direction convergence: both DL architectures fail dominantly on Cyst ↔ Stone

3. **Cross-architecture Grad-CAM** (iteration 7): the interpretability figure that makes the quantitative findings concrete — ConvNeXt V2's higher-capacity attention is kidney-anchored where EfficientNet-B0's is dispersed.

Combined with Sprint 1 (disjoint-error observation, val-saturation in ensemble tuning, equal-weight ensemble reaching 100 %), the paper now has the evidence for its central thesis: **paradigm comparison through interpretability, not score chasing**.
