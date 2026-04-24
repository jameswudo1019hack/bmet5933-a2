# Sprint 2 — Evaluation of the ConvNeXt V2 upgrade proposal

**Date**: 2026-04-24
**Status**: evaluation only — no code changes yet
**Decision**: pending user input
Related: [[Sprint1_log]], [[DL_Improvements_Analysis]], [[Phase2_Design]]

---

## 1. The proposal (external)

A proposal was raised (external reviewer) to swap the deep-learning backbone from **EfficientNet-B0 (5.3M params, 224×224)** to **ConvNeXt V2-Base (89M params, 384×384)** now that A100 GPU access is available. The full proposal argued:

- Remove the compute bottleneck that forced B0
- ConvNeXt V2's local-kernel inductive bias is suited to small features (stones) better than ViTs
- Preserve the paper's "CNN-vs-handcrafted" paradigm comparison
- Use ImageNet-22k pretraining, stochastic depth, weight decay 0.05, bfloat16 AMP
- Run a 5-seed ensemble
- Framing: *"we removed hardware-induced capacity limits to isolate the comparison purely to learning paradigms"*

## 2. Context the proposal did not have

**The current headline test result is macro-F1 = 1.000 via the equal-weight soft-vote ensemble** (Sprint 1 Iteration 2b). Zero errors on 934 test images. No DL-side improvement — architecture, resolution, or seed ensemble — can raise that ceiling. The only thing an upgrade changes is the **DL-alone** number (currently 0.974 with TTA-hflip at 0.983).

## 3. What an upgrade would actually buy us

| Numeric outcome | Probability (my subjective) | Paper impact |
|---|---|---|
| ConvNeXt V2 > 0.99 macro-F1 alone | ~50 % | Modest — still below ensemble |
| ConvNeXt V2 0.97–0.99 (similar to B0) | ~35 % | None — supports current claim |
| ConvNeXt V2 < 0.97 (overfits) | ~15 % | Negative — breaks the upgrade argument |

In no outcome does the ensemble 1.000 number change. The architecture upgrade competes with **write the paper well** for time, not for additional headline marks.

## 4. Cost estimate (honest)

| Task | Time |
|---|---|
| Add `timm` dep, env rebuild, partner re-sync | 15 min |
| Rewrite `deep_learning/model.py` (ConvNeXt V2 loader, new head, GRN details) | 60 min |
| Rewrite `deep_learning/dataset.py` for 384×384 | 30 min |
| Augmentation tuning (384 crops differ) | 30 min |
| Training runs on A100 (5 seeds × ~18 min) | ~90 min Colab |
| Redo TTA on new model (4 view-sets) | 25 min Colab |
| Redo soft-vote ensemble | 5 min |
| Redo data-efficiency sweep (4 fractions × ~18 min) | 75 min Colab |
| Redo Grad-CAM | 20 min |
| Rewrite [[Phase2_Design]] §1, §3, §4 (backbone + resolution + head choices) | 60 min |
| Update [[pipeline]] canvas, [[Sprint1_log]] references, results tables | 30 min |
| Debug first run (near-certain something breaks) | 30–60 min |
| **Total realistic estimate** | **~7–9 hours** |

Calendar cost: one working day with full focus, probably two with interruptions.

## 5. Risk — small-dataset overfitting

ConvNeXt V2-Base: 89M trainable params.
Medium train set: 4,353 images.

≈ 20,400 parameters per training example. Even with heavy regularisation (weight decay 0.05, stochastic depth 0.3, mixed precision), this is in a regime where the model fits dataset-specific noise rather than learning general features. Raghu et al. [16] (cited in [[Phase2_Design]]) already argue ImageNet transfer to medical imaging is modest; ConvNeXt V2 does not change that finding — it just moves more parameters.

**Real scenario**: ConvNeXt V2 achieves strong val F1 (thanks to ImageNet-22k pretraining) but underperforms EfficientNet-B0 on test because the extra capacity overfits to acquisition artifacts. This would be awkward to report.

## 6. The artefacts hypothesis has cheaper tests

The proposal's main epistemic pitch: "ConvNeXt V2 tests whether the classical advantage is truly about learning paradigm or about artefact exploitation." Three cheaper tests of that same hypothesis:

| Test | Effort | What it reveals |
|---|---|---|
| Grad-CAM on DL errors + correct predictions — does attention fall on kidney tissue or image margins? | 30 min | Whether DL learned anatomy or artefacts |
| Classical feature importance ranking — are top features whole-image intensity stats or local texture? | 15 min | Whether classical is using global artefacts |
| Masked-kidney retrain (classical only) — mask everything outside kidney and retrain | 2 hours | Definitive test of "artefacts vs anatomy" |

Any of these is a more direct probe than an architecture swap. They are smaller, can be integrated into the paper's discussion, and do not risk the overfitting scenario.

## 7. Three options

### Option A — Skip the upgrade, focus on paper writing
- Zero methodology work. All time into paper prose.
- Current results (EfficientNet-B0 + TTA + ensemble = 1.000 test F1) are already saturated.

### Option B — One ConvNeXt V2 run as supplementary validation **(recommended)**
- Single training seed, one TTA pass, no sweep redo.
- Report in paper as half a page: *"To validate that our DL results are not an artefact of using a compact architecture under earlier compute constraints, we re-evaluated with ConvNeXt V2 at 384×384. Results were [consistent / divergent] — [paragraph]."*
- Cost: ~3 hours total (1 hour code, ~45 min A100, 1 hour writeup).
- Keeps Phase 2 design intact; treat as scale-validation not methodology change.

### Option C — Full proposal (5-seed ensemble, data-efficiency sweep redo)
- ~7–9 hours total work over 1–2 calendar days
- No upside to the headline number; paper narrative becomes more complex
- Defensible only if Option B shows ConvNeXt V2 substantially beating EfficientNet-B0

## 8. Recommendation

**Option B**. Specifically:

1. Do not swap the primary architecture. Keep EfficientNet-B0 as Person B's reported method.
2. Add one supplementary ConvNeXt V2 run under a new subsection ("Scale-validation with larger backbone").
3. Report the result as a paragraph in the Discussion, not as headline numbers.
4. If ConvNeXt V2 substantially beats B0 and changes the ensemble picture, **only then** escalate to Option C.

This preserves the Phase 2 design doc, keeps paper scope manageable, adds a defensible "we checked scale" paragraph, and leaves time for the actual paper writing — which is the bigger mark contributor under Sandhya's criterion.

## 9. Separately: use Gemini's framing language, even if we skip the upgrade

Two specific phrases from the external proposal are genuinely better than our current draft language and worth borrowing in the Discussion section regardless of whether we do the upgrade:

- *"isolating the comparison purely to the learning paradigms (handcrafted vs. hierarchical representation)"*
- *"even a state-of-the-art CNN cannot overcome the fact that the classical features are exploiting hidden acquisition artefacts in the CT scans"*

Both fit naturally into the paper's discussion of why classical won on this dataset.

## 10. Decision log

- **[awaiting input]** Option A / B / C?
- **[awaiting input]** If B: proceed now, or after drafting the paper's Results section?

Whatever is chosen will produce its own iteration note in this folder.
