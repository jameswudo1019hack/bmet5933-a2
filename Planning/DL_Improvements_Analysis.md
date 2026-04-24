# Deep Learning Pipeline — Improvement Analysis

**BMET 5933 Assignment 2 — Person B (deep learning)**
Status: analysis v1, 2026-04-24
Related: [[Phase2_Design]], [[Phase0_Design]], [[pipeline]]

---

## Executive summary

Current DL pipeline (EfficientNet-B0, 2-stage transfer) achieves **macro-F1 0.9745** on the held-out test set (19 errors / 934). Classical ML (XGBoost + handcrafted features) achieves **0.9976** on the same test set, with **disjoint errors** (both-wrong = 0). Empirical analysis of DL's 19 errors reveals that this is not a capacity or architecture problem — it is a **calibration problem on borderline cases**. Specifically:

- On all 19 DL errors, the true class is in the **top-2 predicted classes 84 %** of the time (16/19).
- Mean confidence on wrong predictions is **0.56** (median 0.54), vs **0.95** on correct predictions — DL is explicitly uncertain when it is wrong.
- The error cluster is **Stone ↔ Normal** confusion (8/19 errors); Stone has the lowest per-class recall (0.942 vs Classical's 1.000, Δ = −5.8 pts).

The most promising interventions exploit the top-2 / borderline-error structure: Test-Time Augmentation, multi-seed ensembling, higher input resolution, and targeted minority-class oversampling. Bigger backbones and exotic loss functions would not help — capacity is not the bottleneck.

---

## 1. Current state (test set)


| Metric              | Value      | 95 % CI          |
| ------------------- | ---------- | ---------------- |
| Accuracy            | 0.9797     | —                |
| Macro-F1            | **0.9745** | [0.9628, 0.9857] |
| Weighted F1         | 0.9797     | —                |
| ROC-AUC (OvR macro) | 0.9995     | —                |
| Errors              | 19 / 934   | (2.03 %)         |


### Per-class recall


| Class     | Support | DL recall | Classical recall | Δ          |
| --------- | ------- | --------- | ---------------- | ---------- |
| Cyst      | 279     | 0.989     | 0.996            | +0.007     |
| Normal    | 381     | 0.979     | 1.000            | +0.021     |
| **Stone** | **103** | **0.942** | **1.000**        | **+0.058** |
| Tumor     | 171     | 0.988     | 0.994            | +0.006     |


### Error direction breakdown (DL, 19 total)


| True → Predicted   | Count |
| ------------------ | ----- |
| **Stone → Normal** | **5** |
| Normal → Cyst      | 3     |
| Normal → Stone     | 3     |
| Cyst → Stone       | 2     |
| Normal → Tumor     | 2     |
| Stone → Cyst       | 1     |
| Tumor → Stone      | 1     |
| Cyst → Normal      | 1     |
| Tumor → Normal     | 1     |


Dominant failure mode: **Stone class is under-predicted** (Stone→Normal) and **over-predicted on easier cases** (Normal/Cyst/Tumor → Stone, 6 of 19 errors).

---

## 2. Error structure — the critical insight

### 2.1 DL is uncertain on its errors

Mean softmax probability on the predicted (wrong) class: **0.561** (vs 0.948 on correct predictions). The model "knows" when it is on shaky ground.

### 2.2 True class is usually at rank 2


| Rank of true class among DL predictions | Count / 19 |
| --------------------------------------- | ---------- |
| Rank 2 (top-2 hit)                      | **16**     |
| Rank 3                                  | 2          |
| Rank 4                                  | 1          |


**Top-2 accuracy = 99.6 %.** Every intervention that perturbs DL's decision toward a second-place prediction has a realistic shot at flipping the answer. This is exactly the regime where:

- **Test-Time Augmentation** (averaging softmax over multiple augmented views of the same image) systematically outperforms single-view inference.
- **Multi-seed ensembling** (averaging independent checkpoints' softmax) closes the gap between "confident-wrong" and "confident-right".
- **Temperature scaling / calibration** would not help directly (it changes confidences but not rankings), but it matters if we later report probabilistic outputs.

### 2.3 Classical wins through a different signal

Classical's 2 errors are Cyst ↔ Tumor confusions (the two mass-like abnormalities), on which DL is **perfect**. This suggests:

- Classical features (GLCM / LBP / Gabor) encode **texture + intensity patterns** — likely exploit Islam-dataset-specific acquisition artifacts that give near-perfect separation between most pairs, but falter on the genuinely anatomically-similar Cyst-vs-Tumor pair.
- DL's ImageNet-pretrained features are more **anatomically aware** but less sensitive to dataset artifacts — hence perfect on the clinically-hardest pair, but noisier on the easier pairs classical nails.

Implication: **the two methods are complementary, not redundant.** A simple soft-vote ensemble of classical + DL would likely outperform either alone, and this is a genuinely publishable observation.

---

## 3. Ranked interventions

Each row has a rough estimate of F1 gain based on medical-imaging literature norms, our error structure, and the specific failure mode.

### Tier 1 — Highest ROI (low effort, high expected gain)


| #        | Intervention               | Rationale                                                                                                                                                                           | Estimated gain            | Effort                                       |
| -------- | -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- | -------------------------------------------- |
| **T1.1** | **Test-Time Augmentation** | 16/19 errors have true class at rank 2; TTA averages softmax across augmented views and systematically tips borderline cases. Only needs an inference-time change, zero retraining. | **+0.5 to +1.0 macro-F1** | ~1 h (code only)                             |
| **T1.2** | **Multi-seed ensemble**    | Three or five seeds averaged. Historically +1–2 F1 on medical imaging. Directly addresses the uncertainty-on-errors pattern.                                                        | +1.0 to +2.0 macro-F1     | ~1 h implementation + 3–5 × 8 min Colab runs |


### Tier 2 — Medium ROI (moderate effort, targeted gain)


| #        | Intervention                                         | Rationale                                                                                                                                                                                                         | Estimated gain                               | Effort                                     |
| -------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- | ------------------------------------------ |
| **T2.1** | **WeightedRandomSampler** for Stone                  | Replace loss-weighting with per-epoch oversampling. Buda et al. [11] found oversampling marginally stronger than cost-sensitive loss for CNNs. Targets Stone's 0.942 recall directly.                             | **+1 to +2 on Stone F1** (smaller elsewhere) | ~30 min code + 1 retraining run            |
| **T2.2** | **Higher input resolution (288 × 288 or 320 × 320)** | Kidney stones are small features. Higher resolution retains more pixel-level evidence. EfficientNet-B0 handles 256–320 fine.                                                                                      | +0.3 to +1.0 macro-F1                        | ~15 min code + 1 retraining (slower epoch) |
| **T2.3** | **Focal Loss** (γ = 2)                               | Focuses gradient on hard examples. Given DL's high uncertainty on errors (mean 0.56 confidence), focal loss should penalise these more than the current weighted cross-entropy. Lin et al. 2017 standard setting. | +0.3 to +1.0 macro-F1, mostly on Stone       | ~15 min code + 1 retraining                |


### Tier 3 — Possible ROI (higher effort, uncertain gain)


| #        | Intervention                          | Rationale                                                                                                                | Estimated gain                                                                        | Effort                                                                 |
| -------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **T3.1** | **Label smoothing** (ε = 0.1)         | Regularises over-confident predictions. Since DL is already appropriately uncertain on errors, this may or may not help. | 0 to +0.3                                                                             | Trivial code change, retraining                                        |
| **T3.2** | **Classical + DL soft-vote ensemble** | Error sets are disjoint. Weighted average of softmax from both models; weights tuned on val.                             | **+1 to +3 macro-F1** over DL alone, but competes with classical alone, not beats it. | ~30 min code, no retraining                                            |
| **T3.3** | **RadImageNet backbone**              | Medical-domain pretraining. Mei et al. 2022 show +1–10 AUC over ImageNet on small medical sets.                          | +0.5 to +2.0                                                                          | ~3–4 h integration (replace torchvision weights, adjust normalisation) |


### Tier 4 — Unlikely to help significantly


| #    | Intervention                             | Why NOT                                                                                      |
| ---- | ---------------------------------------- | -------------------------------------------------------------------------------------------- |
| T4.1 | EfficientNet-B2 / B3 / larger backbones  | Capacity is not the bottleneck (error confidence is, not representational failure)           |
| T4.2 | Vision Transformer (Swin)                | Would chase Islam et al.'s 99.3 % number but break the paper's classical-vs-DL narrative     |
| T4.3 | Aggressive augmentation (MixUp / CutMix) | Likely to hurt on small dataset; destroys the diagnostic detail CT classification depends on |


---

## 4. Recommended implementation order

For a realistic 3-week-out budget focused on paper quality (not SOTA chasing):

### Sprint 1 (today, ~2 hours)

1. **T1.1 — Test-Time Augmentation** first. Zero retraining cost, tests the "top-2 at rank-2" hypothesis directly. If TTA moves macro-F1 from 0.9745 → 0.980+ on test, the hypothesis is empirically confirmed.
2. **T3.2 — Soft-vote ensemble** with classical. Another inference-time-only addition; pairs naturally with TTA. Both can share the same results JSON.

### Sprint 2 (if Sprint 1 results justify more work, ~2 hours Colab)

1. **T1.2 — Multi-seed ensemble**. 4 extra seeds × 8 min A100 = ~35 min compute.
2. **T2.1 or T2.3 — Weighted sampler or focal loss**. Pick one (not both) and retrain. If Stone recall climbs, good ablation material.

### Sprint 3 (only if time permits, optional)

1. **T2.2 — Resolution bump to 288**. Single run, probably marginal on top of TTA + ensemble.
2. **T3.3 — RadImageNet**. Most effort, most uncertain payoff — good for a "future work" paragraph even if not implemented.

---

## 5. Decisions locked in

- **Will implement**: T1.1 (TTA), T3.2 (soft-vote with classical) — both non-invasive.
- **Conditional**: T1.2 (multi-seed ensemble), T2.1 (weighted sampler) — only if Sprint 1 shows improvement and time budget allows.
- **Won't implement**: T4.1–T4.3 (bigger backbones / ViT / aggressive aug) — off-thesis.
- **Defer to future-work section**: T3.3 (RadImageNet), T2.2 (higher resolution).

## 6. What this affects in the paper

- **Technical Contributions — Person B**: add a subsection "Inference-time improvements (TTA, ensembling)" to Phase 2 justification.
- **Results**: if Sprint 1 shifts DL from 0.9745 → ≥0.98, the classical-vs-DL comparison gets an interesting nuance — "with inference-time averaging, DL closes most of the gap".
- **Discussion**: the disjoint-error pattern and the soft-vote ensemble result together motivate a concrete clinical suggestion — "deploy classical + DL together, not either alone".
- **Limitations**: update §8 with note that raw performance on this dataset is likely inflated by acquisition-artifact correlation with class (as confirmed by classical's 99.8 % on handcrafted texture features).

---

## 7. Open questions to resolve before implementing

1. **TTA augmentation set**: horizontal flip only, or flip + small rotations (±10°) + crops? Medical-imaging TTA literature varies; I'd default to just flip + ±10° rotation for this run.
2. **Soft-vote weighting for T3.2**: equal weights (0.5 / 0.5), or tune on validation? Val-tuned is more principled.
3. **Multi-seed ensemble budget**: 3 seeds vs 5 seeds. 5 is the standard but 3 fits in ~25 min of Colab time.

None of these block starting T1.1. Defaults are reasonable.

---

## 8. Checklist — before writing any code

- Partner agreement on implementing T1.1 + T3.2 as Person B's additional work (not a joint responsibility)
- Decide TTA augmentation set (see §7.1)
- Decide soft-vote weights (see §7.2)
- Ensure all new results go into `Results/dl_run_tta/` and `Results/ensemble/` — keep the original `Results/dl_run/` as the un-improved baseline for the comparison
- Update `Phase2_Design.md` §11 limitations to note the dataset-artifact concern surfaced by the disjoint-error analysis

