# Tutor Meeting Brief

**For:** BMET5933 Assignment 2 tutor meeting, Wednesday (2026-04-29 or 05-06 — confirm date)
**Team:** Person A (classical ML, XGBoost) + Person B (deep learning, EfficientNet-B0 + ConvNeXt V2)
**Goal of meeting:** validate the reframed thesis, get directional feedback on framing and limitations, and surface anything Sandhya wants us to adjust **before** drafting the paper.

Read [[Project_Framing_v2]] before the meeting for full context.

---

## 1. The reframed thesis (one paragraph, slightly compressed for verbal delivery)

> We compared a classical machine-learning pipeline (handcrafted texture features + XGBoost) with two transfer-learned CNNs (EfficientNet-B0 and ConvNeXt V2 Base) on the Islam et al. 2022 kidney CT dataset. All three achieve > 97 % macro-F1 and an equal-weight soft-vote ensemble achieves 100 %. The interesting finding isn't the score — it's the structure of the *errors*. Classical and DL fail on **disjoint** sets of images (both-wrong = 0), and their dominant failure modes are paradigm-stable: classical fails on Cyst↔Tumor, both DL backbones fail on Cyst↔Stone, regardless of architecture scale. We argue this is evidence that paradigms are exploiting different aspects of the same visual signal — and that on saturated medical-imaging benchmarks, the *direction* of disagreement carries more information than the *magnitude*.

---

## 2. Three key findings (numbered, with caveats)

### Finding 1: paradigm-stable, architecture-stable disagreement
On 934-image medium test set: classical = 0.998 acc, EfficientNet-B0 + TTA = 0.986 acc, ensemble = 1.000 acc. Disjoint errors (both-wrong = 0). On 1,867-image full test set, paired McNemar's confirms ConvNeXt V2 > EfficientNet-B0 (*p* = 0.0021), but **both DL backbones share the Cyst↔Stone failure mode** that classical does not have.

**Caveat 1:** the 100 % ensemble result is on a saturated dataset. Three other groups have approached 99 %+ on this dataset (Islam Swin 99.30 %, Bingol 2023 hybrid 99.37 %, Teke 2025 GLCM-only 99.98 % on a related kidney-CT task). The number is real but the dataset is easy.

### Finding 2: architecture > data volume for EfficientNet-B0
Matched-data control: EfficientNet-B0 trained on full dataset (8,712 train) vs medium (4,353 train). Doubling the data moves error rate from 1.29 % (medium + TTA) to 1.23 % (full) — essentially nothing. ConvNeXt V2 on the same full-dataset training reduces error rate to 0.32 % — 74 % reduction at matched data, *p* = 0.0021.

**Caveat 2:** test sets aren't comparable sample-for-sample (medium-test ≠ full-test), so cross-table comparisons in the paper are flagged as directional only. The paired McNemar's between EfficientNet-B0-full and ConvNeXt V2-full **is** valid (same 1,867 test images).

### Finding 3: cross-architecture Grad-CAM shows attention difference
Grad-CAM on the same six paired-disagreement test images reveals EfficientNet-B0's attention is dispersed and frequently extends beyond the kidney silhouette; ConvNeXt V2's attention is consistently localised to kidney tissue. On the three Cyst → Stone errors unique to EfficientNet-B0, the smaller network peaks off-organ; ConvNeXt V2 fixates on the lesion and gets it right.

**Caveat 3:** Grad-CAM is a local linear approximation of attribution and not a complete explanation of the model's decision process; the qualitative observation is suggestive of representational difference, not proof.

---

## 3. Three specific questions for the tutor

### Q1. Is the dataset-saturation framing defensible for an ISBI-style paper?
We are reporting 100 % test accuracy as our headline ensemble number. Our argument is that this reflects *dataset saturation* (multiple groups have hit 99 %+) and *complementary feature spaces*, not method quality. **Is this an acceptable framing?** Or should we down-weight the 100 % claim and lead with the per-class / failure-mode analysis instead?

### Q2. Is the patient-level-leakage limitation strong enough to invalidate our results?
The Islam dataset has no patient identifiers; we cannot prevent slices from the same patient appearing in both train and test. Yagis et al. 2021 quantified this effect at 29–55 % accuracy inflation in 2D MRI CNN studies; Veetil et al. 2024 replicated at +67 % on Parkinson's data. We are committed to flagging this as the single most important caveat in the paper. **Should we go further** — e.g., contact the dataset authors for patient IDs, switch to KiTS19 for a patient-stratified secondary evaluation, or add a "future work" section specifically on this?

### Q3. Have we over-engineered the methodology relative to the assignment scope?
We have done: TTA ablation (4 view-sets), classical+DL soft-vote ensemble (val-tuned + equal-weight), data-efficiency sweep (10/25/50/100 %), paired McNemar's at multiple stages, cross-architecture Grad-CAM, Sprint 2 ConvNeXt V2 + matched-data EfficientNet-B0 supplementary, ~38 references queued. **Is this the right depth, or should we cut something for the 6-page IEEE ISBI limit?** Specifically — should the data-efficiency sweep stay (it's interesting) or be cut for space?

---

## 4. One thing we explicitly want pushback on

**Our "paradigm-stable error patterns" claim is currently supported by *two* DL backbones.**

We have:
- EfficientNet-B0 (5 M params, 224 × 224, medium + full)
- ConvNeXt V2 (89 M params, 384 × 384, full)

Both share Cyst↔Stone as the dominant DL failure pair. Classical fails on Cyst↔Tumor.

**The push-back we want:** *is 2 backbones enough to claim "paradigm-stable"?* A reviewer might argue we need at least 3 (e.g., add ResNet-50 as a third DL data point, even if Islam et al. report it at only 73.8 % — which would itself be diagnostic if ResNet-50 fails differently). Should we run a third DL backbone for the paper, or is the current evidence sufficient?

---

## 5. Status going into the meeting

| Phase | Status |
|---|---|
| Phase 0: shared infrastructure (split, eval, McNemar's, bootstrap) | ✅ Done |
| Phase 1: classical ML (Person A) | ✅ Done — XGBoost on medium, 0.998 accuracy |
| Phase 2: deep learning (Person B) | ✅ Done — EfficientNet-B0 medium + TTA, 0.986 accuracy |
| Sprint 1: TTA + ensemble (Person B) | ✅ Done — ensemble = 1.000 accuracy |
| Sprint 2: ConvNeXt V2 + matched-data EffNet-B0 (Person B) | ✅ Done — architecture significance *p* = 0.0021 |
| Cross-architecture Grad-CAM | ✅ Figure generated |
| **Paper drafting** | 🔴 **Will start after this meeting** |
| Submission notebooks (one per person) | 🔴 To do |
| In-class demo (~7 minutes) | 🔴 To do |

---

## 6. Repo + vault links (in case Sandhya wants to look)

- GitHub: <https://github.com/jameswudo1019hack/bmet5933-a2>
- Obsidian vault: `Planning/` directory in the repo. Key reading order if cold:
  1. [[Home]] — landing page
  2. [[Project_Framing_v2]] — read this first
  3. [[Results_Summary]] — all canonical numbers
  4. [[Supporting_Literature]] — verified citations per finding
  5. [[Paper_Skeleton]] — IEEE ISBI structure
  6. [[Phase0_Design]] / [[Phase2_Design]] — methodology justifications
  7. [[experiments/Sprint1_log]] / [[experiments/Sprint2_ConvNeXtV2_on_full]] — chronological sprint logs

---

## 7. One-liner if Sandhya asks "what's your contribution?"

> *"We contribute a per-method failure-mode analysis to the active radiomics-vs-deep-learning literature on kidney CT, surfacing a paradigm-stable error structure (classical fails on Cyst↔Tumor, both DL backbones fail on Cyst↔Stone) that prior fusion-ensemble work has not analysed."*

That's the sentence to lead with.
