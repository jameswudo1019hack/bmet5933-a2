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

> We compare a handcrafted-feature classical ML pipeline (texture features + XGBoost) with a transfer-learned EfficientNet-B0 and a state-of-the-art ConvNeXt V2 on the Islam et al. kidney CT dataset. All three paradigms achieve above 97 % test macro-F1, and their softmax ensemble reaches 100 %. We argue this result is not a positive claim about method quality but a **diagnostic about the dataset**: the task is solvable by simple image-level statistics, and what varies across methods is not what they get right, but what they get wrong. Through per-method failure-mode analysis — feature-importance ranking for classical, Grad-CAM attention maps for DL — we show that the methods exploit **different aspects of the same visual signal**. The classical paradigm's perfect Stone recall and its dominant Cyst ↔ Tumor failure mode are **paradigm-stable**: they persist whether DL is represented by a 5 M-parameter EfficientNet-B0 or an 89 M-parameter ConvNeXt V2 trained on twice the data. The DL paradigm's complementary failure shifts with capacity — at small model / small data, DL fails on Stone → Normal; at larger model / larger data, the residual failure migrates to Cyst ↔ Stone — but never converges with the classical failure pair. We conclude that on saturated medical-imaging benchmarks, the comparison between paradigms is informative only when accompanied by interpretability analysis; the score alone conveys no scientific content, and the **direction** of disagreement between paradigms carries more information than the magnitude.

## What this framing implies about our existing results

| Artefact                                                                       | Score-chasing reading               | Paradigm-comparison reading                                                                                                                                                                                                                                                                          |
| ------------------------------------------------------------------------------ | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Classical XGBoost: 0.9976 macro-F1                                             | "Classical beats DL on this task"   | "108 handcrafted features capture nearly all the discriminating signal — the dataset is texture-solvable"                                                                                                                                                                                            |
| EfficientNet-B0 + TTA: 0.9829 macro-F1                                         | "DL is worse than classical"        | "Even under saturation, DL makes systematically different errors — it is attending to different structure than the texture features do"                                                                                                                                                              |
| Equal-weight ensemble: 1.000 macro-F1                                          | "Our best number"                   | "Both models correct is both models converge; both models wrong is zero; the methods fail disjointly — strong evidence of complementary signal, not redundant signal"                                                                                                                                |
| Stone DL recall 0.942 (medium) → 0.988 (full + ConvNeXt V2) vs classical 1.000 | "DL catches up on Stone with scale" | "Classical's perfect Stone recall is a *hard ceiling* that more capacity / more data approaches asymptotically but does not reach. The Stone class encodes sufficient texture statistics to be trivial for handcrafted features but never trivial for a learned representation, regardless of scale" |
| Cyst ↔ Tumor classical errors (medium), DL correct                             | "DL wins that pair on medium"       | "Cyst and Tumor are both rounded soft-tissue masses with similar texture statistics — classical's texture-only representation cannot separate them, but DL's learned spatial filters can. The advantage is *paradigm-level*, not architecture-level"                                                 |
| Cyst ↔ Stone failure mode shared by both DL backbones (full data)              | "ConvNeXt V2 introduced new errors" | "The DL paradigm's failure pattern shifts with capacity but stays *within* DL: more data did not move the failure mode toward Cyst↔Tumor (where classical fails). Error distributions are paradigm-stable, not architecture-stable, on this dataset"                                                 |

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

---

## Sprint 3 update — 2026-04-27

The framing above was authored on 2026-04-24, when classical-on-full had not yet been run. The thesis paragraph and the rows in the "score-chasing reading vs paradigm-comparison reading" table reflect medium-set data only. Sprint 3 closed the medium/full asymmetry by retraining classical on `split_full.csv` (8,712 train, 1,867 test) — see [[experiments/Sprint3_classical_on_full]].

**What Sprint 3 changed:**

| v2 claim (medium-set) | Sprint 3 finding (full-set) | Verdict |
|---|---|---|
| Both-wrong = 0 between classical and DL → equal-weight ensemble achieves 100 % → "methods fail disjointly" | Classical ∩ EffNet-B0-full = 4 both-wrong, classical ∩ ConvNeXt V2-full = 2 both-wrong (all `Stone→Cyst`) | **Invalidated at full scale.** The 100 % ensemble was a medium-scale artefact, not a paradigm-level property |
| Classical fails on Cyst↔Tumor — paradigm-stable across architectures | Classical-full failures: Stone→Cyst (6) + Stone→Normal (5) + Normal→Stone (1) + Tumor→Cyst (1). Cyst↔Tumor confusion has *disappeared* at full scale | **Invalidated.** Classical's failure pair is not paradigm-stable across dataset scales |
| DL fails on Cyst↔Stone — architecture-stable across backbones | Both DL backbones still fail on Cyst↔Stone at full scale: EffNet `Cyst→Stone(9)`, ConvNeXt V2 `Cyst→Stone(3)`. **Classical makes 0 `Cyst→Stone` errors at full scale.** | **Survives in narrower form.** "Only DL pipelines make `Cyst→Stone` errors" is paradigm-stable and architecture-stable |
| Direction of disagreement carries more information than magnitude | Direction of disagreement is itself dataset-scale-dependent (see rows above) | **Refined**: direction is informative *at a fixed scale* but *changes with scale*. Reporting at one scale alone risks overclaiming |
| Classical's perfect Stone recall is a hard ceiling DL approaches but does not reach | Classical-full Stone recall = 0.947 (11 errors / 207). DL backbones now reach or exceed this — Stone is hard for *every* paradigm at full scale | **Invalidated.** Stone is not classical's privileged class at full scale |

### Sprint 3 same-day addendum (2026-04-27, after RF + SVM refit)

The XGB-only Sprint 3 result above said the *narrower* DL-exclusive `Cyst→Stone` claim survives. Adding RF and SVM to the analysis (3 classical + 2 DL = 5 pipelines) **also kills that claim**:

| Model | Total errors | `Cyst→Stone` errors |
|---|---|---|
| Classical SVM (linear) — broken at scale | 238 | 23 |
| Classical RF | 27 | **2** ← within range of ConvNeXt V2's 3 |
| Classical XGB | 13 | **0** ← classifier-specific zero |
| EfficientNet-B0 | 23 | 9 |
| ConvNeXt V2 | 6 | 3 |

The zero-`Cyst→Stone` is **XGBoost-specific**, not paradigm-specific. Performance ranking at full scale is `ConvNeXt V2 > XGB > EffNet ~ RF >> SVM` — classifier choice within a paradigm dominates the paradigm split. Pairwise McNemar's: RF-vs-XGB is *significantly different within the classical paradigm* (p=0.0013, both-wrong=11), while RF-vs-EffNet is **tied** (p=0.64). The two-paradigm framing oversimplifies a 5-model continuum.

**The replacement thesis** has to acknowledge that paradigm-level claims required deeper analysis to invalidate, and that the *invalidation chain* (medium → full-XGB → full-all-classifiers) is the paper's central methodological contribution.

### Day-2 second addendum (2026-04-28, interpretability + paradigm-coverage check)

Two further analyses (Tasks 15–17 in `2026-04-27-classical-full-dataset-sprint3.md`) extended the invalidation chain to a fourth step:

**Step 4: classical never uniquely succeeds over the joint DL pipeline at full scale.** The 3-way disagreement bucket count: `classical_right_and_both_DL_wrong = 0` images (out of 1,867). The Sprint 1 "complementary signal" / "disjoint errors" framing is now *formally* falsified at full scale, not just statistically weakened. There is no image on which adding the classical paradigm rescues the joint DL ensemble.

**Mechanism (feature importance):** classical XGBoost is dominated by LBP (54 %) and Gabor (53 %) groups; stats (24 %) and GLCM (16 %) contribute substantially less. Top-5 individual features are 4 Gabor *std of magnitude* at vertical / anti-diagonal orientations + 1 LBP at largest scale (P=24, R=3). Raw-XGB without PCA scores higher (0.9950 vs deployed 0.9897) — PCA reshapes the feature weighting (Gabor wins under PCA, LBP wins without), explaining the within-classical RF-vs-XGB error disagreement (p=0.0013 from the addendum).

**Mechanism (Grad-CAM):** on the 8 `Stone → Normal/Cyst` slices that classical misclassifies but DL gets right, both DL backbones correctly attend to small focal high-density regions (calcifications); ConvNeXt V2's attention is sharper than EffNet-B0's. Classical's whole-image LBP+Gabor aggregation cannot localise focal lesions because the calcification occupies a small fraction of the 256×256 frame and gets averaged out. Saved figure: `Results/gradcam/cross_paradigm_disagreement.png` (Paper Figure 3 candidate).

**Updated paper structure (replaces the Sprint 3 addendum implications):**

The Discussion now leads with a **four-step invalidation chain**, with Figure 2 (per-group feature importance) and Figure 3 (cross-paradigm Grad-CAM) doing the mechanistic work for steps 3 and 4. The narrative arc is:

> *"On the medium subset of this benchmark, an equal-weight classical+DL ensemble achieves 100 % and the constituent paradigms appear to fail on disjoint image sets — a finding that suggests complementary feature spaces. On the full dataset (12,446 images, 1,867-image stratified test), this finding does not replicate at any of three increasingly stringent levels of analysis: (i) paired McNemar's between classical-XGB and each DL backbone is no longer statistically significant; (ii) including RF and SVM in the comparison shows that the residual `Cyst→Stone` asymmetry is XGB-specific rather than paradigm-specific; (iii) on a 3-way disagreement check, no test image exists where classical-XGB uniquely succeeds over the joint DL pipeline. We use feature importance and cross-paradigm Grad-CAM to provide a mechanistic explanation: classical XGBoost on this dataset is dominated by multi-scale LBP and Gabor features that aggregate over the whole 256×256 frame, while DL backbones correctly attend to small focal calcifications that the classical aggregation cannot localise. We argue this invalidation chain is itself the paper's most rigorous contribution: paradigm-level claims on saturated medical-imaging benchmarks demand multi-classifier, multi-scale verification with bucket-level coverage analysis before they can be reported."*

This is the thesis to lead with from now on.

**The thesis as it stands cannot survive unrevised.** The replacement thesis is in [[Tutor_Meeting_Brief]] §1 (refreshed 2026-04-27) and is reproduced here for the canonical record:

> We compared a classical machine-learning pipeline (handcrafted texture features + XGBoost) with two transfer-learned CNNs (EfficientNet-B0 and ConvNeXt V2 Base) on the Islam et al. 2022 kidney CT dataset, at *two* dataset scales (n=934 medium test set and n=1,867 full test set). All three achieve > 98 % macro-F1 at every scale, and at the medium scale the equal-weight soft-vote ensemble achieves 100 % — but only at the medium scale. The interesting finding is **scale-dependent**: on the medium dataset, classical and DL fail on disjoint image sets (both-wrong = 0); on the full dataset, classical and the two DL backbones share `Stone→Cyst` failures (both-wrong = 4 vs EffNet, 2 vs ConvNeXt V2), and paired McNemar's classical-vs-each-DL is no longer significant. A *narrower* paradigm-stable asymmetry survives at full scale: only DL pipelines make `Cyst→Stone` errors (9 + 3 across two backbones; classical makes zero). We argue this two-scale comparison — surfacing a result that *does not* replicate from medium to full — is the paper's most rigorous contribution: on saturated medical-imaging benchmarks, both the *magnitude* and *direction* of paradigm disagreement are dataset-scale-dependent, and reporting them at one scale alone risks overclaiming.

**What does NOT change:**

- The interpretability-led framing (feature importance + Grad-CAM) is still load-bearing — arguably more so, because the paper now needs to explain *why* the disjoint-error pattern fails to replicate, and only an interpretability story can do that honestly.
- The dataset-saturation argument (Bingol 2023, Teke 2025, Islam Swin 99.30 %) is now strengthened — the gap-closing at full scale is consistent with that literature.
- The patient-level leakage caveat (Yagis 2021, Veetil 2024) is unchanged and remains the single most important limitation.
- The directives "every table of numbers followed by a paragraph on what those numbers say about the methods' internal representations" still apply.

**What this means for the paper:**

Lead the Discussion with the two-scale result. Frame it as a *positive* methodological contribution — "we ran the comparison at two dataset scales and found the headline result does not replicate" is a more rigorous report than "we ran the comparison at one scale and got a beautiful number". The narrower surviving asymmetry (`Cyst→Stone` is DL-exclusive) becomes the paper's positive paradigm-level finding. Sprint 3's "disjoint errors do not survive" becomes the paper's central methodological cautionary tale.
