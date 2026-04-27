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

### Paper Writing
- [[Results_Summary]] — all canonical numbers in one place ← start here when writing
- [[Paper_Skeleton]] — IEEE paper draft scaffold

---

## Project status

| Phase | Owner | Status |
|---|---|---|
| Phase 0 — Shared infrastructure | Both | ✅ Complete |
| Phase 1 — Classical ML (XGBoost) | Person A | ✅ Complete |
| Phase 2 — EfficientNet-B0 + TTA | Person B | ✅ Complete |
| Sprint 2 — ConvNeXtV2 (supplementary) | Person B | ✅ Complete |
| Grad-CAM figures | Person B | ✅ Figures generated |
| Data efficiency sweep | Both | ⚠️ Code exists, results not yet run |
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
| ConvNeXtV2 Base (supplementary) | Full (n=1867) | **0.9953** | 6 / 1867 |

Key finding: error sets are **disjoint** (both-wrong = 0). Each paradigm fails on different images. This is the scientific content — not the scores themselves.

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
