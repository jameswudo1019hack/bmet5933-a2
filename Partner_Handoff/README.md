# Partner Handoff — DL deliverables for paper write-up

This folder bundles everything James (Person B, DL) owes Person A for the paper.

## Items

| File | Maps to partner's request | Contents |
|---|---|---|
| `B1_DL_numbers_audit.md` | DL numbers audit | All DL rows in Table I + II, paired McNemar p-values, confusion matrices, 4-seed ensemble |
| `B2_val_test_gap.md` | Val-test gap verification | Confirms 0.913 val, 0.822 test, 9.1 pp gap for ConvNeXt V2 + derivation steps + EffNet for comparison |
| `B3_four_seed_ensemble.md` | Four-seed ensemble decision | 4-seed cosine+60 TTA ensemble macro-F1 + the methods sentence to keep or drop |
| `Figure_A1_convnext_gradcam_6case.png` | Figure A1 regeneration | 6-case Grad-CAM panel on the full-dataset ConvNeXt V2 checkpoint (Sprint 5 clean) |
| `Figure_data_efficiency_sweep.png` | (bonus) | DL sample-efficiency curves on clean (both backbones, paper Fig Y) |
| `Figure_cross_paradigm_attribution.png` | (bonus) | Cross-paradigm attribution (classical SVM occlusion + DL Grad-CAM, paper Fig Z) |
| `data_efficiency_sweep_summary.json` | (bonus, raw numbers) | All sweep numbers for tables |

## Provenance

All numbers are computed locally on commit `9dda37b` (or later) from the bug-fixed clean-dataset results. Raw artefacts at `Results/{classical_run_full, dl_run_full, convnextv2_full_run, convnextv2_4seed_cos_tta_ensemble, dl_sweep_clean_v2}` in the repo.

## Status

| Item | Status |
|---|---|
| B1 DL numbers audit | ✅ all numbers verified against source JSON |
| B2 Val-test gap | ✅ confirmed exactly (0.9125 val, 0.8219 test, 9.07 pp gap) |
| B3 4-seed ensemble | ✅ run; macro-F1 0.8374. Keep the Methods II-C sentence; report the ensemble. |
| Figure A1 | ✅ regenerated on Sprint 5 clean ConvNeXt V2 checkpoint, 6 cases from disagreement buckets |
