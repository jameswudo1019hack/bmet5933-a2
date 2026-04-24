# Project framing v2 — paradigm comparison, not score chasing

**Date**: 2026-04-24
**Status**: canonical framing document
**Supersedes**: the implicit "who wins" framing in [[Phase0_Design]] and [[Phase2_Design]] v1

Related: [[Sprint1_log]], [[Sprint2_ConvNeXtV2_on_full]], [[DL_Improvements_Analysis]]

---

## Why this document exists

When classical ML and deep learning both hit > 99 % macro-F1 on the Islam kidney CT dataset, and their ensemble reaches 100 %, the **score itself conveys no scientific content**. Anyone can train a model, report F1, and move on. The interesting question is not who wins — it is what each paradigm is actually doing, and what the near-identical scores reveal about the dataset that produced them.

This doc anchors that shift so all subsequent writing, analysis, and decisions point in the same direction.

---

## The reframed thesis

> We compare a handcrafted-feature classical ML pipeline (texture features + XGBoost) with a transfer-learned EfficientNet-B0 and a state-of-the-art ConvNeXt V2 on the Islam et al. kidney CT dataset. All three paradigms achieve above 97 % test macro-F1, and their softmax ensemble reaches 100 %. We argue this result is not a positive claim about method quality but a **diagnostic about the dataset**: the task is solvable by simple image-level statistics, and what varies across methods is not what they get right, but what they get wrong. Through per-method failure-mode analysis — feature-importance ranking for classical, Grad-CAM attention maps for DL — we show that the methods exploit **different aspects of the same visual signal**. The classical advantage on Stone and the DL advantage on Cyst ↔ Tumor map to the distinction between global texture statistics and local spatial structure. We conclude that on saturated medical-imaging benchmarks, the comparison between paradigms is informative only when accompanied by interpretability analysis; the score alone conveys no scientific content.

## What this framing implies about our existing results

| Artefact | Score-chasing reading | Paradigm-comparison reading |
|---|---|---|
| Classical XGBoost: 0.9976 macro-F1 | "Classical beats DL on this task" | "108 handcrafted features capture nearly all the discriminating signal — the dataset is texture-solvable" |
| EfficientNet-B0 + TTA: 0.9829 macro-F1 | "DL is worse than classical" | "Even under saturation, DL makes systematically different errors — it is attending to different structure than the texture features do" |
| Equal-weight ensemble: 1.000 macro-F1 | "Our best number" | "Both models correct is both models converge; both models wrong is zero; the methods fail disjointly — strong evidence of complementary signal, not redundant signal" |
| Stone DL recall 0.942 vs classical 1.000 | "DL is weak on Stone" | "Stone is a strongly-textured class; texture features encode it trivially while DL needs to learn the pattern. The recall gap is the measured difference between the two representational strategies" |
| Cyst ↔ Tumor classical errors, DL correct | "DL wins that pair" | "Cyst and Tumor are both rounded soft-tissue masses with similar texture statistics — classical's texture-only representation cannot separate them, but DL's learned spatial filters can" |

## What each upcoming analysis is *for*

- **Classical XGBoost feature importance**: identifies which of the 108 handcrafted features carry the signal. If 2–3 features dominate, the paper's dataset-is-texture-solvable argument gets stronger.
- **Grad-CAM on DL (EfficientNet-B0, ConvNeXt V2)**: asks where the network attends. Does it look at kidney tissue or at image margins (crop boundaries, scanner text)? The answer diagnoses whether DL is learning anatomy or exploiting artefacts.
- **Disjoint-error analysis**: already done; shows the two paradigms fail on different images. Under this framing this is evidence of complementary feature spaces, not "one model is better".
- **Kidney-masked ablation** (optional, higher-effort): mask everything outside the kidney and retrain classical. If classical still hits 99 %, it is using anatomy. If it drops, it is using margins — direct evidence of artefact exploitation.
- **ConvNeXt V2 on full dataset (Sprint 2)**: *not* a test of "can bigger DL beat classical". It is a test of whether a much larger, higher-resolution, more modern CNN attends to the same visual structure as EfficientNet-B0 or to different structure. Grad-CAM comparison is the scientific output; the accuracy number is secondary.
- **Islam et al. 99.30 % Swin baseline**: not a target to beat but a data point showing that SOTA transformers also saturate on this dataset, consistent with the artefact-saturation hypothesis.

## Writing directives — what to foreground, what to background

**Foreground in the paper:**
- Disjoint-error observation
- Per-class failure-mode analysis
- Feature-importance + Grad-CAM attention analysis
- The "what does each paradigm learn" interpretation
- Dataset artefact / patient-leakage limitations (strengthened)

**Background in the paper:**
- Raw accuracy / macro-F1 numbers (report them, but they are not the point)
- The 100 % ensemble result (report it, but frame as "both methods have hit a dataset-imposed ceiling" not "we win")
- Architectural choices (report them; do not oversell ConvNeXt V2 as a novelty)

## Writing directives — tone

- Confident but not triumphalist. A 100 % claim invites scrutiny; the paper earns the right to make it by explaining immediately why the number is not the headline.
- Interpretability-led. Every table of numbers should be followed by a paragraph on what those numbers say about the methods' internal representations.
- Acknowledge the dataset's limitations early and often. The paper benefits more from being honest about dataset saturation than from pretending otherwise.

## Sandhya's rubric under this framing

The rubric explicitly rewards *justification of choices and impact on results*, not *raw classifier accuracy*. Under the reframed thesis, every design decision in Phase 0 / Phase 2 can be justified as serving the interpretability analysis rather than chasing a score:

- Matched preprocessing → interpretability comparison requires the same inputs
- Fixed split, seed-locked → both methods see the same errors, so disjoint-error claim is meaningful
- Bootstrap CIs, McNemar's → uncertainty in the score differences is the input to the "is the gap real" question
- Per-class metrics → the per-class gap pattern *is* the scientific finding

This means Phase 0 / Phase 2 methodology needs no changes — only its *exposition* in the paper needs to shift from "we chose this to win" to "we chose this so the comparison could teach us something".

---

## Open writing questions

- **Abstract / Introduction**: foreground the reframed thesis in the first paragraph, explicitly. Do not let a reader get to "we achieved 100 %" without first encountering "the score is not the point".
- **Section structure**: proposal — dedicate one full page to the per-method interpretability analysis (feature importance + Grad-CAM) rather than treating it as an add-on to the Evaluation section.
- **Conclusion**: resist the temptation to say "our method performs well". Instead: "the score-level tie between paradigms combined with the disjoint-error pattern motivates our central claim about feature-space complementarity".
