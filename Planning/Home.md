# BMET5933 Assignment 2 — Home

Kidney CT Classification · Islam et al. (2022) dataset · Due **Fri 15 May 2026 23:59**
35% of final grade (Demo 15% + Report 20%)

---

## Navigate

### Planning & Design
- [[Phase0_Design]] — shared infrastructure: split, preprocessing, eval harness
- [[Phase2_Design]] — DL pipeline design and literature justification (Person B)
- [[Project_Framing_v2]] — canonical framing: paradigm comparison, not score chasing ← **read this first**

### Analysis
- [[DL_Improvements_Analysis]] — error analysis that motivated TTA and ensemble
- [[experiments/Sprint1_log]] — TTA ablation + ensemble (complete)
- [[experiments/Sprint2_evaluation_ConvNeXtV2]] — ConvNeXtV2 decision rationale
- [[experiments/Sprint2_ConvNeXtV2_on_full]] — ConvNeXtV2 on full dataset
- [[experiments/Sprint3_classical_on_full]] — classical on full + paired McNemar's at matched scale
- [[experiments/Sprint4_ConvNeXtV2_medium]] — ConvNeXt V2 on medium (closes the 2×2 architecture-vs-data matched grid)
- [[Validation_and_Verification]] — overfitting diagnostics + V&V infrastructure (post-tutor 2026-04-29) ← **the answer to Sandhya's overfitting question**

### Paper Writing
- [[Results_Summary]] — all canonical numbers in one place ← start here when writing
- [[Paper_Skeleton]] — IEEE paper draft scaffold (now anchored in radiomics literature)
- [[Supporting_Literature]] — verified citations per finding ← reference when writing each paragraph
- [[Tutor_Meeting_Brief]] — Wednesday meeting prep: thesis, key findings, three questions, one push-back ask

---

## Project status

| Phase | Owner | Status |
|---|---|---|
| Phase 0 — Shared infrastructure | Both | ✅ Complete |
| Phase 1 — Classical ML (XGBoost) | Person A | ✅ Complete |
| Phase 2 — EfficientNet-B0 + TTA | Person B | ✅ Complete |
| Sprint 2 — ConvNeXtV2 (supplementary) | Person B | ✅ Complete |
| Sprint 3 — Classical on full + paired McNemar's | Person B (running) | ✅ Complete |
| Sprint 3 second addendum — Feature importance + cross-paradigm Grad-CAM | Person B | ✅ Complete (2026-04-28) |
| Sprint 3 third addendum — Overfitting diagnostics (post-tutor) | Person B | ✅ Complete (2026-04-29) |
| Sprint 4 — ConvNeXt V2 on medium (closes 2×2 matched grid) | Person B | ✅ Complete (2026-04-29) |
| Grad-CAM figures | Person B | ✅ Figures generated (cross-architecture + cross-paradigm) |
| Data efficiency sweep | Both | ✅ Both pipelines (medium DL + full classical) |
| Feature importance (classical) | Person B (Sprint 3) | ✅ Extracted (deployed-pipeline permutation + raw-XGB sanity) |
| Paper draft | Both | 🔴 Not started |
| Submission notebooks (.ipynb) | Both | 🔴 Not started |
| In-class demo slides | Both | 🔴 Not started |

---

## Headline numbers

Full table → [[Results_Summary]]

| Model                                | Dataset        | Macro-F1   | Errors     |
| ------------------------------------ | -------------- | ---------- | ---------- |
| Classical XGBoost                    | Medium (n=934) | **0.9976** | 2 / 934    |
| EfficientNet-B0 + TTA hflip          | Medium (n=934) | **0.9829** | 13 / 934   |
| ConvNeXtV2 Base (medium)             | Medium (n=934) | **0.9898** | 7 / 934    |
| Ensemble equal-weight w=0.5          | Medium (n=934) | **1.0000** | 0 / 934    |
| Classical SVM (full)                 | Full (n=1867)  | 0.8515     | 238 / 1867 |
| Classical RF (full)                  | Full (n=1867)  | 0.9801     | 27 / 1867  |
| Classical XGBoost (full)             | Full (n=1867)  | **0.9897** | 13 / 1867  |
| EfficientNet-B0 (full, matched data) | Full (n=1867)  | 0.9819     | 23 / 1867  |
| ConvNeXtV2 Base                      | Full (n=1867)  | **0.9953** | 6 / 1867   |

**Key findings — invalidation chain (post-Sprint 3 + 2 addenda, four steps):**
1. **Medium scale** — error sets are **disjoint** (both-wrong = 0 between classical-XGB and DL). Equal-weight ensemble achieves 100 %. Classical fails on Cyst↔Tumor; DL fails on Cyst↔Stone.
2. **Full scale, XGB only** — disjoint-error claim **does not survive**. Classical-XGB and ConvNeXt V2 share 2 `Stone→Cyst` errors; classical-XGB and EffNet-B0 share 4. Paired McNemar's classical-vs-DL is no longer significant (p = 0.089 / 0.119). Narrower surviving claim: "only DL pipelines make `Cyst→Stone` errors".
3. **Full scale, all classifiers (Sprint 3 addendum)** — narrower claim **also fails**. Classical RF makes 2 `Cyst→Stone` errors (within range of ConvNeXt V2's 3). The zero-`Cyst→Stone` is XGBoost-specific, not paradigm-specific. Performance ranking at full scale is `ConvNeXt V2 > XGB > EffNet ~ RF >> SVM` — classifier choice within a paradigm dominates the paradigm split.
4. **Full scale, paradigm-coverage check (Sprint 3 second addendum, 2026-04-28)** — `classical_right_dl_wrong = 0`: no test image (out of 1,867) exists where classical-XGB uniquely succeeds over both DL backbones. **The Sprint 1 "complementary signal between paradigms" claim is formally falsified** at full scale. Cross-paradigm Grad-CAM (new Figure 3) shows DL correctly attends to focal calcifications classical's whole-image texture aggregation cannot localise.

**Mechanism (Sprint 3 second addendum):** classical XGBoost feature importance shows LBP (0.568 macro-F1 drop) + Gabor (0.532) dominate; stats (0.236) and GLCM (0.163) contribute far less. The dataset is solvable by multi-scale local-pattern + frequency-response features. Raw-XGB without PCA scores 0.9950 (vs deployed 0.9897) — PCA(50) costs ~0.5pp at full scale. Top-5 individual features are Gabor std-of-magnitude at vertical/anti-diagonal orientations. See `Results/classical_run_full/feature_importance_group.png` (Paper Figure 2) and `Results/gradcam/cross_paradigm_disagreement.png` (Paper Figure 3).

**Overfitting diagnostics (Sprint 3 third addendum, post-tutor 2026-04-29):** Sandhya's "models could be overfitting" pushback ran into four targeted diagnostics; **all four reject the classical-overfit framing**. XGB train+val curves saturate together (no widening gap, val-best at n=399 is +0.0002 above deployed n=200); DL backbones show no val-loss rebound and are still climbing at last epoch; per-class CV std ≤ 0.0023; classical-feature-space slice-proximity probe shows only 3 % similarity boost over random baseline. **The strongest residual signal is per-class structural mismatch**: held-out test Stone F1 (0.9703) is **3.9 σ below** the 5-fold CV mean (0.9792 ± 0.0023), Tumor is 2.2 σ above — exactly what patient-level grouping would produce that random stratification can't smooth out. The patient-leakage caveat is now a quantitatively-observed signal, not just a literature-backed worry. See `Results/diagnostics/per_class_cv.png` and `Results/diagnostics/{xgb,dl}_learning_curves.png`.

The **four-step invalidation chain + overfitting-diagnostic battery** is now the paper's central methodological contribution. See [[experiments/Sprint3_classical_on_full]] §"Sprint 3 addendum" + §"Sprint 3 second addendum" + §"Sprint 3 third addendum".

---

## Submission checklist

- [ ] Paper — PDF, IEEE ISBI format, **6 pages** (+ 1 page refs/acknowledgements)
- [ ] Person A notebook — `.ipynb` uploaded as comment on submission
- [ ] Person B notebook — `.ipynb` uploaded as comment on submission
- [ ] In-class demo — ~7 minutes, slides or notebook
- [ ] Individual reflection form — separate portal

---

## Known limitations (must appear in paper)

1. No patient IDs → cannot prevent adjacent-slice leakage. All metrics likely inflated vs real-world generalisation.
2. Single random seed per model — variance not characterised.
3. Classical trained on medium, ConvNeXtV2 on full — not directly comparable.
4. Islam et al. comparison is directional only (different splits, different class balance).
