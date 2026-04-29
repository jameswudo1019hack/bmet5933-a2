# Overfitting Diagnostics Implementation Plan (post-tutor 2026-04-29)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose whether the 99 %+ accuracies surfaced in Sprints 1–3 reflect genuine generalisation or are artefacts of (a) patient-level slice leakage, (b) classical XGB overfitting under PCA(50), (c) DL overfitting on full-set train, or (d) class-imbalance handling. Answers the tutor's "I think your models could be overfitting" pushback (notes from 2026-04-29 meeting).

**Architecture:** Four diagnostic experiments, each ~30–60 min, none requiring full retraining of DL backbones. All use existing cached features (`Results/classical_features_full/`) and existing per-epoch logs (`Results/dl_run_full/run_log.json`, `Results/convnextv2_full_run/run_log.json`). Run sequentially, write outputs to `Results/diagnostics/`, document in a new Sprint 3 third addendum and refreshed Tutor_Meeting_Brief Finding.

**Tech Stack:** numpy, pandas, matplotlib, scikit-learn (StratifiedKFold), XGBoost `evals_result_`, no PyTorch retraining required.

**Out of scope:** Retraining DL with stronger augmentation; pseudo-patient stratified resplit (deferred — decided after diagnostics). Both retain Sprint 3 + addendum framing as the substrate; this work refines the overfitting story.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `analysis/diag_filename_proximity.py` | create | Diagnostic 1: filename-numerical-proximity test for slice-level leakage |
| `analysis/diag_xgb_learning_curves.py` | create | Diagnostic 2: XGBoost train/val mlogloss + merror curves over `n_estimators` |
| `analysis/diag_per_class_cv.py` | create | Diagnostic 3: 5-fold stratified CV on cached features, per-class F1 mean ± std per fold for classical XGB |
| `analysis/diag_dl_learning_curves.py` | create | Diagnostic 4: parse existing per-epoch logs from DL run_log.json, plot train_loss + val_loss + val_macro_f1 for EffNet-B0-full and ConvNeXt V2-full |
| `Results/diagnostics/` | create | Output directory for all diagnostic artefacts (PNG + JSON) |
| `Planning/experiments/Sprint3_classical_on_full.md` | append | "Sprint 3 third addendum — overfitting diagnostics (post-tutor)" |
| `Planning/Tutor_Meeting_Brief.md` | update | New Finding 4 + updated Q1 with diagnostic results |
| `Planning/Home.md` | update | Status row + key findings |

---

## Task 18: Filename-proximity slice-leakage diagnostic

**Files:**
- Create: `analysis/diag_filename_proximity.py`
- Output: `Results/diagnostics/filename_proximity.{json,png}`

**Why first:** directly tests Sandhya's most-likely-implied concern (no patient IDs → adjacent slices in train/test). Result determines whether the rest of the diagnostics are run in "saturation" or "leakage" framing.

**Hypothesis:** Islam dataset filenames are like `Cyst- (3051).jpg`. If numerically adjacent files within the same class are slices from the same patient, then random slice-level stratified split puts adjacent slices in train and test. Models effectively learn patient identity, inflating accuracy.

**Test:** for each test image with class C and numerical ID *i*:
1. Find the K=5 numerically-nearest training images of the same class (smallest |id_train − i|)
2. Compute feature-space cosine similarity between test image's features and those K nearest train images' features
3. Compare to a "random baseline": K random training images of the same class
4. If "nearest-by-ID" similarity ≫ random similarity, that's evidence of patient grouping by filename order

**Outputs:**
- JSON with per-class summary statistics (mean nearest-by-ID similarity, mean random similarity, ratio, p-value from one-sided Mann-Whitney U)
- PNG: density plots of cosine-similarity distributions (nearest-by-ID vs random) per class (4 subplots)
- Console: top-line "leakage suspected" verdict if ratio > 1.5 in any class

- [ ] **Step 1: Skeleton script + load split + features**
- [ ] **Step 2: Parse numerical IDs from filenames per class**
- [ ] **Step 3: K=5 nearest-by-ID lookup per test image**
- [ ] **Step 4: Cosine similarity computation (existing 108-dim features)**
- [ ] **Step 5: Random baseline (10 random train neighbours per test image, per class)**
- [ ] **Step 6: Mann-Whitney U test per class**
- [ ] **Step 7: 4-subplot KDE figure**
- [ ] **Step 8: JSON summary**
- [ ] **Step 9: Run + commit**

---

## Task 19: XGBoost learning curves (train vs val mlogloss + merror over n_estimators)

**Files:**
- Create: `analysis/diag_xgb_learning_curves.py`
- Output: `Results/diagnostics/xgb_learning_curves.{json,png}`

**Why:** XGBoost has built-in `eval_set` tracking; we just didn't use it in the original train.py. Refit the deployed-pipeline winner on cached features with `eval_set=[(X_train_pca, y_train), (X_val_pca, y_val)]` and `eval_metric=["mlogloss", "merror"]`. The train-vs-val gap over `n_estimators` is the textbook overfit signal.

- [ ] **Step 1: Load cached features + saved scaler/PCA from deployed pipeline**
- [ ] **Step 2: Fit XGB with eval_set on (train_pca, val_pca)**
- [ ] **Step 3: Extract evals_result_ → DataFrame**
- [ ] **Step 4: Plot train_mlogloss + val_mlogloss + train_merror + val_merror over n_trees**
- [ ] **Step 5: Diagnose: if val_mlogloss minimum is at n_trees < 200, model is over-trained at deployed `n_estimators=200`. If train_merror → 0 while val_merror plateaus, classical-overfit signature.**
- [ ] **Step 6: Save JSON + PNG, run + commit**

---

## Task 20: Per-class 5-fold CV metrics for classical XGB

**Files:**
- Create: `analysis/diag_per_class_cv.py`
- Output: `Results/diagnostics/per_class_cv.{json,png}`

**Why:** the existing `classical/train.py` does 5-fold stratified CV but only reports aggregate `cv_f1`. Per-class F1 mean ± std across folds tells us whether high-variance classes (Stone) are overfitting more. Sandhya's "do CV class-wise too" note.

- [ ] **Step 1: Reproduce `_cv_xgb` from train.py with the deployed best_params, on cached PCA-projected features**
- [ ] **Step 2: For each of 5 folds, compute per-class F1 + per-class precision + per-class recall on the held-out fold**
- [ ] **Step 3: Aggregate across folds → mean ± std per class**
- [ ] **Step 4: Plot 4-bar per-class F1 with error bars; compare to held-out test set F1 (from classical_results.json)**
- [ ] **Step 5: JSON with full per-fold per-class breakdown**
- [ ] **Step 6: Run + commit**

---

## Task 21: DL learning curves from existing per-epoch logs

**Files:**
- Create: `analysis/diag_dl_learning_curves.py`
- Output: `Results/diagnostics/dl_learning_curves.{json,png}`

**Why:** EfficientNet-B0-full and ConvNeXt V2-full both already logged train_loss + val_loss + val_macro_f1 per epoch in their `run_log.json` files. No retraining needed — just parse and plot.

- [ ] **Step 1: Parse `Results/dl_run_full/run_log.json` and `Results/convnextv2_full_run/run_log.json`**
- [ ] **Step 2: Concatenate stage 1 + stage 2 epochs (both DL pipelines use a 2-stage protocol)**
- [ ] **Step 3: 4-panel figure: (top-left) EffNet train+val loss, (top-right) EffNet val macro-F1, (bottom-left) ConvNeXt train+val loss, (bottom-right) ConvNeXt val macro-F1**
- [ ] **Step 4: Annotate each panel with: best-epoch, stage-1→stage-2 boundary, early-stopping if applicable**
- [ ] **Step 5: Diagnose: if val_loss starts climbing while train_loss continues to fall, that's classical DL overfitting. If both saturate together, that's saturation/leakage.**
- [ ] **Step 6: Save JSON + PNG, run + commit**

---

## Task 22: Document findings in vault

**Files:**
- Modify: `Planning/experiments/Sprint3_classical_on_full.md` (append "Sprint 3 third addendum")
- Modify: `Planning/Tutor_Meeting_Brief.md` (update Q1 + add new Finding 4)
- Modify: `Planning/Home.md` (status row + key findings)
- Modify: `Planning/Results_Summary.md` (new Diagnostics section)

**Why:** vault is the audit trail; Sprint 3 third addendum documents what we did in response to tutor pushback.

- [ ] **Step 1: Append "Sprint 3 third addendum — overfitting diagnostics" to Sprint3 log with results from each of Tasks 18–21**
- [ ] **Step 2: Update Tutor_Meeting_Brief Q1 to reference diagnostic outcomes**
- [ ] **Step 3: Update Home.md status table + headline-numbers caveat**
- [ ] **Step 4: Push everything; final summary to user**

---

## Self-Review Checklist (run before handing off)

- [x] **Spec coverage:** Sandhya's 2 notes from the user → Note 1 (loss curves + class-wise CV) covered by Tasks 19, 20, 21 → Note 2 (data augmentation/sampling) deferred (post-diagnostic decision)
- [x] **Patient-leakage** addressed by Task 18 (the user's notes don't mention this but it's the most likely deepest concern, per my earlier analysis)
- [x] **No retraining DL** — all four diagnostics use existing artefacts
- [x] **All file paths absolute / project-relative consistent**
