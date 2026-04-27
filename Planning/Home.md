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
| Grad-CAM figures | Person B | ✅ Figures generated |
| Data efficiency sweep | Both | ✅ Both pipelines (medium DL + full classical) |
| Feature importance (classical) | Person A | ⚠️ Not yet extracted |
| Paper draft | Both | 🔴 Not started |
| Submission notebooks (.ipynb) | Both | 🔴 Not started |
| In-class demo slides | Both | 🔴 Not started |

---

## Headline numbers

Full table → [[Results_Summary]]

| Model | Dataset | Macro-F1 | Errors |
|---|---|---|---|
| Classical XGBoost | Medium (n=934) | **0.9976** | 2 / 934 |
| EfficientNet-B0 + TTA hflip | Medium (n=934) | **0.9829** | 13 / 934 |
| Ensemble equal-weight w=0.5 | Medium (n=934) | **1.0000** | 0 / 934 |
| Classical SVM (full) | Full (n=1867) | 0.8515 | 238 / 1867 |
| Classical RF (full) | Full (n=1867) | 0.9801 | 27 / 1867 |
| Classical XGBoost (full) | Full (n=1867) | **0.9897** | 13 / 1867 |
| EfficientNet-B0 (full, matched data) | Full (n=1867) | 0.9819 | 23 / 1867 |
| ConvNeXtV2 Base | Full (n=1867) | **0.9953** | 6 / 1867 |

**Key findings — invalidation chain (post-Sprint 3 + addendum, three steps):**
1. **Medium scale** — error sets are **disjoint** (both-wrong = 0 between classical-XGB and DL). Equal-weight ensemble achieves 100 %. Classical fails on Cyst↔Tumor; DL fails on Cyst↔Stone.
2. **Full scale, XGB only** — disjoint-error claim **does not survive**. Classical-XGB and ConvNeXt V2 share 2 `Stone→Cyst` errors; classical-XGB and EffNet-B0 share 4. Paired McNemar's classical-vs-DL is no longer significant (p = 0.089 / 0.119). Narrower surviving claim: "only DL pipelines make `Cyst→Stone` errors".
3. **Full scale, all classifiers (Sprint 3 addendum)** — narrower claim **also fails**. Classical RF makes 2 `Cyst→Stone` errors (within range of ConvNeXt V2's 3). The zero-`Cyst→Stone` is XGBoost-specific, not paradigm-specific. Performance ranking at full scale is `ConvNeXt V2 > XGB > EffNet ~ RF >> SVM` — classifier choice within a paradigm dominates the paradigm split. RF-vs-XGB are significantly different within the classical paradigm (p=0.0013), but RF-vs-EffNet are *tied* (p=0.64).

The **invalidation chain itself** is now the paper's central methodological contribution. See [[experiments/Sprint3_classical_on_full]] §"Sprint 3 addendum".

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
