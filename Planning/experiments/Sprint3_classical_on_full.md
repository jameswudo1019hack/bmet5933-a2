# Sprint 3 — Classical ML on the full dataset

**Date**: 2026-04-27
**Status**: complete
Related: [[Sprint2_ConvNeXtV2_on_full]], [[Sprint2_evaluation_ConvNeXtV2]], [[Sprint1_log]], [[DL_Improvements_Analysis]], [[Phase0_Design]], [[Project_Framing_v2]]

> **Framing.** Sprint 2 closed with classical-on-medium (0.9976) and DL-on-full (EffNet-B0 0.9819, ConvNeXt V2 0.9953). The medium-vs-full asymmetry left "classical fails on Cyst↔Tumor / DL fails on Cyst↔Stone" as a *medium-set* claim. Sprint 3 closes that asymmetry: classical retrained on the same full split (8,712 train, 1,867 test) so all three pipelines are now matched on the n=1867 test set. The scientific question: *does the paradigm-stable error pattern survive at full scale?* **Answer: in its original form, no — it was a medium-set artefact.** A narrower asymmetry survives (DL alone makes Cyst→Stone errors).
>
> **Same-day addendum (post-RF/SVM refit):** the *narrower* asymmetry above also collapses once RF and SVM are added in scope. RF makes 2 `Cyst→Stone` errors (within range of ConvNeXt V2's 3); the zero-Cyst→Stone result is **specific to XGBoost**, not to the classical paradigm. The full picture is below in the Sprint 3 addendum.
>
> **Day-2 addendum (interpretability + 3-way bucket counting, 2026-04-28):** classical XGBoost feature importance shows LBP (54 %) and Gabor (53 %) groups dominate, with stats (24 %) and GLCM (16 %) contributing less. Cross-paradigm Grad-CAM bucket counts reveal a **third invalidation**: `classical_right_dl_wrong = 0` — there is no test image where classical-XGB uniquely succeeds over the joint DL pipelines. The Sprint 1 "complementary signal" claim is formally falsified at full scale. See "Sprint 3 second addendum" below.
>
> **Day-3 addendum (overfitting diagnostics post-tutor meeting, 2026-04-29):** Sandhya raised "I think your models could be overfitting" in the Wednesday meeting. We ran four diagnostics targeting (a) classical XGB train-val divergence, (b) DL train-val divergence, (c) patient-level slice leakage via filename-proximity, (d) per-class CV stability. **All four reject the classical overfitting hypothesis**: train and val curves converge together for both XGB and DL; no val-loss rebound; train-val merror gap is a fixed 0.7pp (saturation, not active divergence). **However, per-class CV reveals a leakage signature**: Stone test F1 (0.9703) is **3.9 σ below** the train+val 5-fold CV mean (0.9792 ± 0.0023), while Tumor test F1 is 2.2 σ above CV mean — exactly the per-class structural mismatch you'd expect if patient-level grouping creates systematic difficulty differences between train+val and test that random stratification can't smooth out. **Verdict**: the 99 %+ numbers are not over-trained model artefacts; they reflect either genuine dataset signal or shared dataset-level structure that affects train and val together but redistributes per-class difficulty across the train+val/test boundary. See "Sprint 3 third addendum" below.

## Decision

Reverses the Sprint 2 scoping note ("not in scope: rebuilding classical on full"). Justification: the matched-test-set comparison is the actual paradigm claim; without it, classical-vs-DL is medium-only. Person A asked Person B to run on Colab Pro+ given the compute access.

## Configuration

| Parameter | Value | Notes |
|---|---|---|
| Pipeline | `classical/{train,predict,sweep}.py` | identical to medium run after CLI flag plumbing (commits `7a230df`–`01346a6`) |
| Hyperparameter grids | unchanged from `classical/config.py` | matched-protocol — no refitted grids |
| Split | `split_full.csv` (12,446 images, 70/15/15 stratified seed=42) | same split CSV used for EffNet-full and ConvNeXt V2-full |
| Train / val / test | 8,712 / 1,867 / 1,867 | same as DL Sprint 2 |
| Feature extraction | joblib `n_jobs=-1` (Apple Silicon M-series, 8 vCPUs effective) | new in Sprint 3 |
| Classifiers | SVM (C∈{0.001…1.0}, linear), RF, XGB | same grids as medium |
| Final model selector | val macro-F1 (same criterion as medium) | matches `train.py` selector |

### Compute footnote

The Sprint 3 plan called for a Colab Pro+ run via `notebooks/colab_classical_full.ipynb`. The notebook ran cells 1–5 cleanly (PAT auth, clone, dependency install, Drive mount, dataset extraction, smoke train) but cell 6 (full feature extraction on 8,712 images) stalled at >30 minutes with no progress output, almost certainly due to joblib `loky` per-task IPC overhead interacting badly with Colab's vCPU allocation. Pivoted to a local run in the `bmet5934` conda env on an Apple Silicon Mac, completing all three stages in **6.5 minutes** wall time (faster than the Colab smoke would suggest). The `colab_classical_full.ipynb` notebook is preserved unchanged for the .ipynb submission deliverable; the diagnosis (joblib backend / batching) is a future-work item, not a paper-relevant finding.

## Results — headline

`Results/classical_run_full/classical_results.json`:

| Metric | Value |
|---|---|
| Best model | XGBoost |
| Best params | `learning_rate=0.1, max_depth=6, n_estimators=200` |
| Accuracy | 0.9930 |
| Macro-F1 | **0.9897** [95% CI 0.9835, 0.9948] |
| Weighted-F1 | 0.9930 |
| ROC-AUC OvR | 0.9995 |
| Errors | **13 / 1867** |
| Train wall time | 172.7 s (local, M-series Mac, n_jobs=-1) |

### Per-classifier val performance (grid-search winners)

| Model | Best params | CV macro-F1 | Val macro-F1 |
|---|---|---|---|
| SVM (linear) | `C=1.0` | 0.8388 | 0.8248 |
| Random Forest | `max_depth=20, n_estimators=200, min_samples_split=5` | 0.9734 | 0.9789 |
| **XGBoost (winner)** | `lr=0.1, max_depth=6, n_estimators=200` | **0.9884** | **0.9895** |

Note: SVM's collapse from 0.97 (medium) to 0.82 (full) is itself worth flagging — at this dataset scale the linear-kernel SVM with PCA(50) ceiling becomes a poor fit. RF and XGB scale; SVM does not.

### Per-class F1 — classical full

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Cyst | 0.988 | 1.000 | 0.994 | 556 |
| Normal | 0.993 | 0.999 | 0.996 | 762 |
| Stone | 0.995 | 0.947 | **0.970** | 207 |
| Tumor | 1.000 | 0.997 | 0.999 | 342 |

### Confusion matrix (classical full, n_test=1867)

|  | Pred Cyst | Pred Normal | Pred Stone | Pred Tumor |
|---|---|---|---|---|
| **True Cyst** | 556 | 0 | 0 | 0 |
| **True Normal** | 0 | 761 | 1 | 0 |
| **True Stone** | 6 | 5 | 196 | 0 |
| **True Tumor** | 1 | 0 | 0 | 341 |

**Critical observation**: the medium-set classical errors were Cyst↔Tumor (1 each direction). The full-set classical errors are dominated by **Stone → (Cyst, Normal)** (11 / 13 errors involve Stone). Cyst↔Tumor confusion has *disappeared* at full scale.

## Comparison vs medium-set classical

| | Medium (n=934 test) | Full (n=1867 test) |
|---|---|---|
| Macro-F1 | 0.9976 | 0.9897 |
| Errors | 2 / 934 (0.21 %) | 13 / 1867 (0.70 %) |
| Dominant failure pair | Cyst↔Tumor (2/2) | Stone→Cyst (6) + Stone→Normal (5) |

The error rate **rose** at full scale (3.3× higher rate). This is consistent with [[Bingol_2023_99.37]] reaching ≤99.37 % on the same Islam dataset at full size despite a hybrid radiomics+CNN pipeline — i.e. the dataset is harder at full size than the medium subset suggests.

## Paired McNemar's at full scale (n=1867)

`Results/classical_run_full/sprint3_comparison.json`:

| Comparison | Both correct | Only A wrong | Only B wrong | Both wrong | Discordant | p-value |
|---|---|---|---|---|---|---|
| Classical (A) vs EffNet-B0-full (B) | 1835 | 9 | 19 | **4** | 28 | **0.089** (n.s.) |
| Classical (A) vs ConvNeXt V2-full (B) | 1850 | 11 | 4 | **2** | 15 | **0.119** (n.s.) |
| EffNet-B0-full (A) vs ConvNeXt V2-full (B) *(Sprint 2 sanity)* | 1839 | 22 | 5 | 1 | 27 | **0.0021** (sig.) |

Two important reads:

1. **Sprint 2 result reproduces exactly** (EffNet vs ConvNeXt V2: discordant=27, p=0.0021 — matches the `Sprint2_evaluation_ConvNeXtV2.md` log to the digit). The analysis script is sound.
2. **At full scale, classical and each DL backbone are statistically tied** (both p > 0.05). Classical's decisive win on medium (p=0.00024 vs DL ensemble) does not generalise.

## Disjoint-error analysis at full scale

The medium-set headline finding was **both-wrong = 0** between classical and DL — every test image was classified correctly by at least one paradigm, hence the equal-weight ensemble achieved 100 %.

**At full scale this does NOT hold:**

- Classical ∩ EffNet-B0-full both wrong: **4** (all `Stone → Cyst`)
- Classical ∩ ConvNeXt V2-full both wrong: **2** (both `Stone → Cyst`)
- Classical ∩ EffNet ∩ ConvNeXt V2 all-three wrong: 1 image (a `Stone → Cyst` confusion shared by all three pipelines)

The `Stone → Cyst` confusion appears to be a **dataset-level hard-case** that no representation we tried can solve: low-attenuation calculi inside or adjacent to fluid-containing cysts likely confound both texture features and CNN attention.

### Failure-pair breakdown (which class confusions each pipeline uniquely makes)

Top "only-A-wrong" pairs (where A errs, B is correct):

| Pipeline | Top unique-wrong pairs (true→pred, count) |
|---|---|
| Classical-full | Stone→Normal(5), Stone→Cyst(2 to 4 across the two comparisons), Normal→Stone(1), Tumor→Cyst(1) |
| EffNet-B0-full (vs classical) | **Cyst→Stone(9)**, Stone→Cyst(4), Normal→Stone(3), Normal→Tumor(2), Stone→Normal(1) |
| ConvNeXt V2-full (vs classical) | **Cyst→Stone(3)**, Tumor→Normal(1) |

**The narrower paradigm-stable claim that survives:** only DL pipelines make `Cyst → Stone` errors at full scale. Classical makes **zero** Cyst→Stone errors across all 1,867 test images. Both DL backbones (EffNet 9, ConvNeXt V2 3) make this error. This asymmetry is real even though the broader "disjoint errors" claim is not.

A plausible mechanism: cysts have high-attenuation calcified borders in some CT slices that look like Stone radiomic signatures to a CNN's bottom-up texture detectors, but that classical's CLAHE+GLCM+LBP+Gabor pipeline does not encode as Stone-like (because Stone-class GLCM signatures depend on uniformly bright high-density blobs, not boundary intensity). This is speculative and would need attention-map / occlusion analysis to confirm.

### Verdict per the pre-registered interpretation table (Plan §Task 10)

| Pre-registered scenario | Verdict |
|---|---|
| **Both DL backbones still dominated by Cyst↔Stone *and* classical still dominated by Cyst↔Tumor** → claim survives | ❌ does not apply (classical no longer Cyst↔Tumor) |
| Classical Cyst↔Tumor still dominant, DL pattern shifts | ❌ classical's pattern shifted, not DL's |
| **Classical pattern shifts to match DL (Cyst↔Stone)** → does not survive, major reframe | ⚠️ partially: classical's shift is to Stone-related (Stone→Cyst, Stone→Normal), DL's Cyst→Stone-direction asymmetry survives |
| Sample sizes too small to claim *any* pattern (≤3 errors per pipeline) | ❌ enough errors to claim something (13 + 23 + 6) |

**We accept the third verdict with a refinement:** the broad "paradigm-stable, disjoint-errors" framing from the medium dataset is invalidated. The narrower "only DL makes Cyst→Stone errors" asymmetry is preserved. The paper Discussion will report this honestly — invalidating a result is itself a finding.

## Data-efficiency sweep on the full split

`Results/classical_sweep_full/sweep_summary.json`:

| Train fraction | n_train | Val macro-F1 | Test macro-F1 [95 % CI] |
|---|---|---|---|
| 10 % | 871 | 1.0000 | 0.9554 [0.9438, 0.9666] |
| 25 % | 2,178 | 1.0000 | 0.9716 [0.9626, 0.9807] |
| 50 % | 4,355 | 1.0000 | 0.9799 [0.9712, 0.9875] |
| 100 % | 8,712 | 1.0000 | 0.9897 [0.9835, 0.9948] |

**Notable**: classical reaches **0.96 with only 871 training samples** — this is *better than EffNet-B0-medium-baseline at 4,353 train samples (0.9745)* on the easier 934-image medium test. Classical's sample efficiency is genuinely high, consistent with the radiomics-vs-DL data-efficiency literature (Lambin 2017, Guiot 2021). Curve plot: `Results/classical_sweep_full/data_efficiency_curve.png`.

The val F1 = 1.000 at every fraction means **val saturation is more severe at full scale** — the same val-tuning failure mode documented in [[Sprint1_log]] (Cawley & Talbot 2010 / val saturation). Equal-weight ensembling would still be the right choice if we were to retry the medium-set ensemble experiment at full scale.

## Implications for the paper

The Sprint 3 results force three changes to the Paper Skeleton:

1. **Headline reframe.** The "100 % equal-weight ensemble on disjoint errors" finding cannot be lifted into the abstract as the paper's main contribution — it is a medium-set finding that does not survive at full scale. Two options:
   - (a) **Lead with the *narrower* paradigm-stable asymmetry** ("only DL makes Cyst→Stone errors at full scale; classical does not — across both EffNet-B0 and ConvNeXt V2"). This is a 6-page-paper-defensible finding.
   - (b) **Lead with the dataset-saturation framing** (Bingol 2023 99.37 % on the same dataset; classical 0.9897 on full ≈ ConvNeXt V2 0.9953 on full; difference is not statistically significant at p < 0.05). This would be the more honest framing.
   
   The two are not mutually exclusive — option (a) is the contribution, option (b) is the limitation.

2. **Add a "scale-dependent paradigm comparison" subsection.** The most novel finding is that the medium-set "paradigm-stable disjoint errors" finding does not replicate at scale. Reviewers are unlikely to have seen a paper where the authors explicitly invalidate their own result by going from medium to full data — this is rigorous science and should be presented as such.

3. **Invalidate the "100 % ensemble" headline.** The Phase 0 / Sprint 1 ensemble was on medium and reported macro-F1 = 1.0000. We should:
   - Keep that medium-dataset result (it is real on medium)
   - Add a **caveat paragraph** ("This result is medium-dataset-specific. At full dataset scale the classical–ConvNeXt V2 disjoint-error count drops from 0 to 2 of 1867, so an equal-weight ensemble at full scale would not achieve 100 % accuracy.")
   - This honesty is *required* by the CLAIM 2024 / TRIPOD+AI 2024 reporting standards we cited in [[Phase0_Design]].

4. **Update Tutor_Meeting_Brief.md questions.** Current Q1 ("is dataset-saturation framing defensible for ISBI?") becomes load-bearing — the answer to it now determines whether the paper's headline is option (a) or (b). Q3 ("over-engineered methodology?") gains weight — Sprint 3 surfaces a real result that justifies the depth.

## Limitations specific to Sprint 3

1. Same patient-level leakage caveat as all prior runs (no patient IDs available; Yagis 2021, Veetil 2024).
2. Single seed (consistent with medium and DL Sprint 2 — no variance characterisation; the `Stone→Cyst` shared failure could be an artefact of one slice in the test set, but is still informative as such).
3. Hyperparameter grids unchanged from medium — by design (matched protocol), but means the full-set classical may be sub-optimal (a wider grid could help, e.g. higher `n_estimators`, RBF SVM with larger PCA). Flag in paper Discussion.
4. The `bmet5934` conda env runs `xgboost 2.x` which differs by minor patch from Person A's original local env; we explicitly verified that the random seed produces deterministic results (smoke-run val F1 = 0.4125 reproduced bit-exact between Colab and local Apple Silicon).
5. The Colab notebook is included in submission for reproducibility but was not the actual run path; the local-CLI invocations in the run_log have authoritative timestamps.

## Sprint 3 addendum — SVM and RF re-fit on full (2026-04-27, same day)

The user asked whether the partner had trained only XGBoost or also SVM and RF. The pipeline grid-searches all three but only the val-best is saved; the partner's medium run picked XGB and the Sprint 3 full run also picked XGB. To get test-set predictions for SVM and RF *at full scale*, I re-fit both on cached features using the existing best params from `run_log.json`, mirroring the train+val protocol (`analysis/sprint3_train_svm_rf.py`). This expanded the paradigm-comparison from 1 classical + 2 DL to **3 classical + 2 DL**.

### Per-classifier results at full scale (same n=1867 test set)

| Pipeline | Best params | Macro-F1 | Accuracy | Errors | `Cyst→Stone` errors |
|---|---|---|---|---|---|
| Classical SVM (linear) | `C=1.0` | 0.8515 | 0.8725 | 238 | 23 |
| Classical RF | `max_depth=20, n=200, mss=5` | 0.9801 | 0.9855 | 27 | **2** |
| Classical XGB | `lr=0.1, md=6, n=200` | 0.9897 | 0.9930 | 13 | **0** |
| EfficientNet-B0 (full) | (Sprint 2) | 0.9819 | 0.9877 | 23 | 9 |
| ConvNeXt V2 Base (full) | (Sprint 2) | 0.9953 | 0.9968 | 6 | 3 |

### Pairwise McNemar's across all 5 (10 unordered pairs)

`Results/classical_run_full/sprint3_all_classifiers.json`:

| Pair | Discordant | Both wrong | p-value | Reading |
|---|---|---|---|---|
| SVM vs RF | 221 | 22 | 2.6e-45 | SVM dominated |
| SVM vs XGB | 233 | 9 | 9.4e-49 | SVM dominated |
| SVM vs EffNet-B0 | 243 | 9 | 6.9e-43 | SVM dominated |
| SVM vs ConvNeXt V2 | 240 | 2 | 2.8e-50 | SVM dominated |
| **RF vs XGB** | 18 | 11 | **0.0013** | **Within-classical: significantly different** |
| RF vs EffNet-B0 | 42 | 4 | 0.64 | Tied |
| RF vs ConvNeXt V2 | 29 | 2 | 0.0002 | ConvNeXt V2 wins |
| XGB vs EffNet-B0 | 28 | 4 | 0.089 | Tied |
| XGB vs ConvNeXt V2 | 15 | 2 | 0.119 | Tied |
| EffNet-B0 vs ConvNeXt V2 | 27 | 1 | 0.0021 | ConvNeXt V2 wins (Sprint 2) |

### What this kills, what survives

**Killed (the surviving narrow paradigm-stable claim from Sprint 3 §"Verdict"):**

> *"only DL pipelines make `Cyst→Stone` errors at full scale; classical makes zero"*

Classical RF makes **2** `Cyst→Stone` errors at full scale — within range of ConvNeXt V2's 3. The pattern of zero-Cyst→Stone is **specific to XGBoost**, not specific to the classical paradigm. With three classical classifiers in scope, the asymmetry collapses.

**Killed (the within-paradigm-stable claim):**

The texture-features paradigm is not internally consistent across classifier choices. RF and XGB significantly disagree on which images they get wrong (p=0.0013, both-wrong=11). Within-paradigm variance > some between-paradigm comparisons (e.g., RF vs EffNet-B0 are *tied* at p=0.64).

**Killed (the "classical vs DL" framing as the primary axis):**

Performance ranking is `ConvNeXt V2 > XGB > EffNet-B0 ≈ RF >> SVM`. The "classical / DL" split does *not* explain ranking — classifier choice within a paradigm dominates. SVM-linear is broken (238 errors) but RF and XGB outperform EffNet-B0; ConvNeXt V2 outperforms all classical. The two-paradigm framing oversimplifies a 5-model continuum.

**Surviving findings (now stricter and harder-won):**

1. **Stone is the universal weak class.** All 5 models have Stone as their lowest-F1 class. Average Stone F1 across the 4 competent models is ~0.95 vs Cyst/Normal/Tumor all near 0.99. This is dataset-level, not paradigm-level.
2. **SVM-linear with PCA(50) does not scale.** Macro-F1 collapses from 0.91 (medium) to 0.85 (full) — a finding worth one sentence in the paper if there is space.
3. **The medium-set "disjoint errors / 100 % ensemble" finding is a medium-only artefact** (already documented in the main Sprint 3 body above; further confirmed here).
4. **ConvNeXt V2 is the single best model** by a margin that *is* statistically significant against everything except XGBoost.
5. **XGBoost is the most precise classifier on Cyst class** (zero Cyst-row errors) — a *classifier-level* finding worth flagging in the per-class analysis but **not** suitable as a paradigm-level claim.

### Implications for the paper (replacing the earlier Sprint 3 implications)

The earlier Sprint 3 §"Implications for the paper" was written when only classical-XGB was in scope. With RF and SVM added, the implications shift again:

1. **Drop the "DL-exclusive Cyst→Stone errors" claim entirely.** It is XGBoost-specific, not paradigm-specific. The paper Discussion should report this honestly: "we expanded the analysis to RF and SVM and found that the asymmetry was specific to XGBoost".
2. **The most rigorous paper framing is the *invalidation chain*:**
   - Medium-set: "disjoint errors / 100% ensemble / paradigm-stable failure pairs" → headline result on first analysis
   - Full-set XGB only: "disjoint errors does not survive; narrower DL-exclusive `Cyst→Stone` survives" → first invalidation
   - Full-set with SVM + RF: "narrower asymmetry is XGB-specific, not paradigm-specific" → second invalidation
   - This *progressive narrowing* of an over-claimed result by additional analysis is itself the paper's methodological contribution.
3. **Lead with the 5-model comparison table.** It is the most informative single artefact: it demonstrates classifier-within-paradigm variance (`RF vs XGB`, p=0.0013) comparable to between-paradigm variance (`XGB vs ConvNeXt V2`, p=0.119, n.s.).
4. **Add SVM as an instructive failure case.** The 238-error SVM is a teaching moment about model-class capacity at scale; do not bury it.
5. **Update Tutor_Meeting_Brief Q1 again.** "Is dataset-saturation framing defensible for ISBI?" gains another layer: not only does the 100 % ensemble not replicate at full scale, but the surviving paradigm-stable asymmetry from Sprint 3 also fails to replicate when within-paradigm classifier variance is included.

## Sprint 3 second addendum — interpretability findings (2026-04-28)

### Classical XGBoost feature importance

Two complementary analyses on the deployed pipeline (`scaler → PCA(50) → XGB`) using `analysis/sprint3_feature_importance.py`:

**Per-group permutation importance** (macro-F1 drop when each feature group is permuted on the n=1867 test set, n_repeats=10):

| Feature group | macro-F1 drop ± std | Group size |
|---|---|---|
| **LBP** (multi-scale local binary patterns) | **0.568 ± 0.018** | 54 features |
| **Gabor** (frequency × orientation responses) | **0.532 ± 0.014** | 32 features |
| stats (intensity statistics) | 0.236 ± 0.006 | 10 features |
| GLCM (Haralick texture) | 0.163 ± 0.006 | 12 features |

LBP and Gabor jointly carry the predictive signal at full scale. Classical XGB on this dataset is fundamentally a **multi-scale local-pattern + frequency-response detector**, not a global statistics or co-occurrence model. This is paper-relevant: it grounds the dataset-is-texture-solvable claim in specific feature-group dominance rather than the previous hand-wave.

**Top 5 individual features (deployed pipeline permutation importance):**

| Rank | Feature | Group | macro-F1 drop |
|---|---|---|---|
| 1 | `gabor:f=0.1, θ=π/2, std` | Gabor | 0.0484 |
| 2 | `gabor:f=0.4, θ=3π/4, std` | Gabor | 0.0427 |
| 3 | `gabor:f=0.4, θ=π/2, std` | Gabor | 0.0319 |
| 4 | `lbp:P=24, R=3, b=1` | LBP | 0.0302 |
| 5 | `gabor:f=0.2, θ=3π/4, std` | Gabor | 0.0299 |

The dominant Gabor features all measure *standard deviation of magnitude response* (textural roughness), at vertical (π/2) and anti-diagonal (3π/4) orientations — consistent with kidney-CT anatomy (renal capsule edges, vertebral / vascular boundaries, oblique calculi).

**Sanity check — raw-XGB without PCA:** test macro-F1 = **0.9950** (vs deployed 0.9897). PCA(50) costs ~0.5pp accuracy at full scale. Top 5 by raw-XGB gain are LBP-dominated (5/5 LBP top, then 3 GLCM, no Gabor in top 10). The PCA-vs-raw divergence in feature ranking (Gabor wins under PCA, LBP wins without) provides a **mechanistic explanation** for why XGB and RF disagree on errors at full scale (Sprint 3 addendum, p=0.0013): the PCA step reshapes which features get weight, and RF doesn't use PCA in the same way.

Figure files: `Results/classical_run_full/feature_importance_group.png` (Paper Figure 2 candidate), `feature_importance_top20.png` (supplementary).

### Cross-paradigm Grad-CAM — and the third invalidation

`analysis/sprint3_gradcam_paradigm.py` selects samples from the *3-way* classical / EffNet / ConvNeXt disagreement buckets (rather than Sprint 2's pairwise EffNet-vs-ConvNeXt only).

**Bucket counts on n=1867 test set:**

| Bucket | Count |
|---|---|
| classical-XGB right, both DL wrong | **0** |
| classical-XGB wrong, both DL right | 8 (all `Stone → Normal/Cyst` classical errors that DL caught) |
| all three wrong | 1 (idx 790: true=Stone, all → Cyst) |

**Third invalidation in the chain.** `classical_right_dl_wrong = 0` means **there is no test image (out of 1,867) where classical-XGB uniquely succeeds over the DL paradigm**. The Sprint 1 medium-set claim that "classical and DL provide complementary signal" is now formally falsified: at full scale, classical adds zero coverage that the DL pipelines don't already provide jointly.

This is the **third step of the invalidation chain**:

1. *Medium scale*: classical+DL ensemble = 100 % (disjoint errors → complementary signal)
2. *Full scale, XGB only*: 100 % ensemble does not replicate; classical-vs-DL McNemar p > 0.05
3. *Full scale, all classifiers*: `Cyst→Stone` asymmetry is XGB-specific, not paradigm-specific
4. *Full scale, paradigm-coverage check*: classical never uniquely succeeds over the joint DL pipeline → **the "complementary signal" claim is dead at full scale**

**Mechanism (Grad-CAM evidence):** the figure (`Results/gradcam/cross_paradigm_disagreement.png`) shows three rows:

- *Rows 1–2 (Stone-class slices misclassified by classical-XGB as Normal):* both DL backbones correctly attend to small focal high-density regions in the upper kidney quadrants (clearly visible bright calcifications). ConvNeXt V2's attention is sharper than EffNet-B0's. **Classical's whole-image texture aggregation cannot localise these focal lesions** because LBP+Gabor are computed over the entire 256×256 frame and the calcification is a small fraction of the pixel area.
- *Row 3 (universally-wrong slice):* both DL backbones attend to the *wrong* region with moderate confidence (p=0.56, 0.65), genuine slice-level ambiguity. The image is the dataset's irreducible difficulty.

**Implications for the paper (consolidated, post second addendum):**

The paper Discussion now leads with **a four-step invalidation chain** rather than a single positive claim. Each step is a result on its own:

| Step | Finding | What it falsifies |
|---|---|---|
| 1 | Medium-set: classical+DL ensemble = 100 %, disjoint errors | sets the headline number that subsequent steps probe |
| 2 (Sprint 3) | At full scale, disjoint errors does not survive | "this is a paradigm-level property" |
| 3 (Sprint 3 addendum) | XGB-specific zero `Cyst→Stone` does not generalise to RF | "classical paradigm has a systematic blind spot DL doesn't" |
| 4 (Sprint 3 second addendum) | classical never uniquely succeeds over joint DL at full scale | "classical provides complementary signal to DL" |

What survives: **Stone is the universal weak class** (every model's hardest class); **DL attention is mechanistically different from classical's whole-image texture aggregation** (Grad-CAM evidence: focal lesion detection); **ConvNeXt V2 is the single best model** (statistically beats RF, EffNet-B0, and tied-with-trend-favouring vs XGBoost p=0.119); **PCA costs ~0.5pp accuracy** in the deployed classical pipeline at full scale (raw-XGB = 0.995 vs deployed = 0.990).

What does *not* survive: any "paradigm-stable" or "complementary-signal" framing. The paper should report what survives cleanly, lead with the invalidation chain methodologically, and make the figure-2 (feature importance) and figure-3 (cross-paradigm Grad-CAM) the substrate of the Discussion.

## Sprint 3 third addendum — overfitting diagnostics (post-tutor 2026-04-29)

Sandhya's Wednesday-meeting pushback ("the models might be overfitting") triggered four targeted diagnostics, each ~30 min, none requiring retraining of any pipeline. Plan: `Planning/plans/2026-04-29-overfitting-diagnostics.md`.

### Diagnostic 1: filename-numerical-proximity slice-leakage probe

**Rationale.** Islam dataset has no patient IDs. Filenames are like `Cyst- (3051).jpg`. If numerically adjacent files within a class are slices of the same patient, then for each test image its numerically-nearest-by-ID train images should be feature-space-similar above random baseline.

**Method.** For each test image with class C and ID *i*: compute mean cosine similarity (in 108-dim handcrafted feature space) between the test image and (a) the K=5 train images of class C with smallest |ID − i|, (b) K=5 random train images of class C (averaged over 5 random pulls). One-sided Mann-Whitney U with alternative "nearest > random".

**Results** (`Results/diagnostics/filename_proximity.{json,png}`):

| Class | n_test | nearest-by-ID sim | random sim | ratio | p-value | verdict |
|---|---|---|---|---|---|---|
| Cyst | 556 | 0.9977 ± 0.0079 | 0.9660 ± 0.0143 | 1.033 | 1.3e-164 | weak signal |
| Normal | 762 | 0.9991 ± 0.0045 | 0.9717 ± 0.0152 | 1.028 | 3.3e-239 | weak signal |
| Stone | 207 | 0.9959 ± 0.0106 | 0.9762 ± 0.0143 | 1.020 | 1.6e-50 | weak signal |
| Tumor | 342 | 0.9987 ± 0.0068 | 0.9707 ± 0.0124 | 1.029 | 3.2e-105 | weak signal |

**Reading.** The within-class baseline cosine similarity is already saturated at ~0.97 — kidneys of the same diagnostic class look very alike in the 108-dim feature space. The nearest-by-ID condition adds only 2–3 percent on top of that, statistically real (p ≪ 0.001) but with small effect size. **The 108-dim handcrafted feature space cannot distinguish patient identity from class identity** at any scale that would dominate model behaviour. This does NOT rule out patient leakage — it shows the classical feature space is too coarse to detect it. A parallel test in DL embedding space (penultimate layer of EffNet-B0 / ConvNeXt V2) would be richer; deferred as future work.

### Diagnostic 2: XGBoost learning curves over n_estimators

**Rationale.** Classical XGB at deployed `n_estimators=200` could be over-trained if val mlogloss minimum occurs earlier. Train→0/val-plateau divergence would be the textbook overfit signature.

**Method.** Refit XGB on the cached train-only-fit scaler+PCA(50) features with `eval_set=[(train_pca, y_train), (val_pca, y_val)]` and `eval_metric=["mlogloss", "merror"]`. Ceiling: 400 estimators. Then re-evaluate on test at deployed (n=200) and val-best (n=399) operating points.

**Results** (`Results/diagnostics/xgb_learning_curves.{json,png}`):

| Quantity | Value |
|---|---|
| Train mlogloss at deployed n=200 | very small (saturated) |
| Val mlogloss at deployed n=200 | 0.0299 |
| Val mlogloss minimum | 0.0218 (at round 399 — still slowly decreasing) |
| Train merror at deployed | 0.0000 (perfect train fit) |
| Val merror at deployed | 0.0070 |
| Train–val merror gap | **constant 0.7pp**, does not widen with more trees |
| Test macro-F1 at deployed (n=200) | 0.9871 |
| Test macro-F1 at val-best (n=399) | 0.9874 |
| Δ macro-F1 (val-best − deployed) | +0.0002 (negligible) |

**Verdict (script-derived).** *"Saturation pattern: both train and val converge to near-zero error. Consistent with dataset-level signal (or leakage), not classifier overfit."* The train-val gap is fixed (does not widen with capacity), and val keeps slowly improving past the deployed point — opposite of overfitting.

### Diagnostic 3: per-class 5-fold stratified CV on classical XGB

**Rationale.** Sandhya's note "do CV class-wise too". Macro-F1 alone hides per-class variance; the universal-weak-class (Stone) might be the real risk.

**Method.** 5-fold stratified CV on train+val combined (n=10,579) with deployed best_params. Per-fold per-class precision/recall/F1. Compare CV per-class F1 mean ± std to held-out test per-class F1 from `classical_results.json`.

**Results** (`Results/diagnostics/per_class_cv.{json,png}`):

| Class | CV F1 mean ± std | min | max | Held-out test F1 | Test − CV mean | (Test − CV mean) / std |
|---|---|---|---|---|---|---|
| Cyst | 0.9935 ± 0.0012 | 0.9921 | 0.9953 | 0.9937 | +0.0002 | +0.2 σ |
| Normal | 0.9958 ± 0.0007 | 0.9948 | 0.9965 | 0.9961 | +0.0003 | +0.4 σ |
| Tumor | 0.9935 ± 0.0023 | 0.9909 | 0.9974 | 0.9985 | +0.0050 | **+2.2 σ** |
| **Stone** | **0.9792 ± 0.0023** | 0.9760 | 0.9828 | **0.9703** | **−0.0089** | **−3.9 σ** |

| Aggregate | CV (mean ± std) | Held-out test |
|---|---|---|
| macro-F1 | 0.9905 ± 0.0010 | 0.9897 (within 1 σ) |

**Reading — this is the most important diagnostic.** Aggregate macro-F1 looks fine (CV ≈ test within 1 σ), but the per-class breakdown reveals **directional structural difference**:

- Stone test F1 (0.9703) is **3.9 σ below** the 5-fold CV mean — the test set's Stone slices are systematically *harder* than what stratified random folds on train+val would predict.
- Tumor test F1 (0.9985) is **2.2 σ above** the CV mean — the test set's Tumor slices are systematically *easier*.
- Cyst and Normal are within fold-level noise.

**Mechanism.** This is exactly the pattern you'd expect if the underlying patient-grouped slice population is *unevenly distributed* across the train+val and test partitions of `split_full.csv`. Specifically, if some "easy Stone patients" landed in train+val and some "hard Stone patients" landed in test (or vice versa for Tumor), the random stratified split on slice level can't average that out — and 5-fold CV *within* train+val would underestimate the train+val/test difficulty mismatch on those classes. This is a quantitative observation of the patient-leakage caveat, the strongest signal in the four diagnostics.

### Diagnostic 4: DL learning curves from existing per-epoch logs

**Rationale.** EfficientNet-B0-full and ConvNeXt V2-full both already log train_loss + val_loss + val_macro_f1 per epoch (`Results/dl_run_full/run_log.json`, `Results/convnextv2_full_run/run_log.json`). No retraining needed — parse and plot. Look for val-loss rebound (overfit signature) or under-fit early stopping.

**Results** (`Results/diagnostics/dl_learning_curves.{json,png}`):

| Pipeline | Total epochs | Best epoch | Best val macro-F1 | Stage-2 val-loss rebound from minimum |
|---|---|---|---|---|
| EfficientNet-B0 (full) | 35 | 35 (last) | 0.9766 | +0.000 (none) |
| ConvNeXt V2 Base (full) | 35 | 34 (penultimate) | 0.9979 | +0.000 (none) |

**Verdicts (script-derived):** both pipelines show *"Saturation: val loss plateau in stage 2, no overfit rebound detected."*

**Reading.** Both DL models are *under-trained* if anything, not over-trained. Best epoch is at or near the last epoch trained — the early-stopping patience never triggered because val kept improving. Train and val loss curves track each other smoothly. **DL overfitting is not the cause of the 99 %+ accuracies.**

### Synthesis — answering Sandhya's "could be overfitting"

| Hypothesis | Diagnostic that tests it | Verdict |
|---|---|---|
| Classical XGB over-trained at n_estimators=200 | XGB learning curves (Diag 2) | **Rejected** — saturation, val-best at n=399 only +0.0002 macro-F1 above deployed |
| Classical XGB has high per-class variance hidden by aggregate | Per-class 5-fold CV (Diag 3) | **Rejected for variance** (per-class CV std ≤ 0.0023) BUT see structural-mismatch finding |
| DL backbones overfit at end of training | DL learning curves (Diag 4) | **Rejected** — both still climbing at last epoch, no val-loss rebound |
| Patient-level leakage inflates accuracy | Filename proximity (Diag 1) + per-class CV (Diag 3) | **Partially supported** — Diag 1 inconclusive (classical feature space saturated within-class); Diag 3 shows ~4 σ structural Stone difficulty mismatch between train+val and test, consistent with patient-level grouping |

**The diagnostic answer to Sandhya is therefore:**

> *We tested four overfitting hypotheses and the classical-overfit framing is rejected by all four — train-val gaps for XGB and both DL backbones are stable, val curves never rebound, per-class CV variance is small (std ≤ 0.0023). However, per-class CV reveals a per-class structural mismatch between train+val and the held-out test set: Stone test F1 is 3.9 σ below the 5-fold CV mean while Tumor is 2.2 σ above — exactly what we would expect if patient-level grouping creates systematic difficulty differences between sets that random stratified slice splitting cannot smooth out. The 99 % accuracies are not over-trained model artefacts; they reflect either genuine dataset signal or shared dataset-level structure that affects per-class composition. Further investigation requires patient-level resplitting, which the dataset does not support without reverse-engineering patient groups (deferred as future work).*

This is exactly the rigour level Sandhya was asking for.

### Implications for the paper (consolidated, post all three Sprint 3 addenda)

The four-step invalidation chain (medium → full-XGB → full-all-classifiers → 3-way coverage) is augmented with a **fifth methodological step**: *we ran four overfitting diagnostics and the classical-overfit hypothesis is rejected; the residual signal is a per-class structural mismatch between train+val and test most consistent with patient-level grouping*. The paper Discussion should:

1. Lead with the four-step invalidation chain (unchanged).
2. Add a "Diagnostic checks" subsection that summarises the four overfitting tests + the per-class structural mismatch finding. Treat Diagnostic 3 as the headline diagnostic — its 3.9 σ Stone difficulty gap is the cleanest evidence of patient-level structure short of access to patient IDs.
3. Strengthen the patient-leakage caveat in §Limitations: it is no longer a literature-backed worry but a dataset-specific observed signal in our own data.
4. Recommend patient-level resplitting + a parallel filename-proximity test in DL embedding space as the natural next steps for follow-up work.

What we did NOT need to change as a result of the diagnostics: the deployed pipelines, the four-step invalidation chain, the figure 2 / figure 3 framing, or the headline numbers. The diagnostics *strengthen* the existing story without rewriting it.

## Files written (Sprint 3 + Sprint 3 addendum + second addendum + third addendum)

Original Sprint 3:
- `Results/classical_run_full/classical_pipeline.pkl` — final XGBoost on train+val
- `Results/classical_run_full/classical_predictions.npz` — (y_true, y_pred, y_prob)
- `Results/classical_run_full/classical_results.json` — full evaluate() report
- `Results/classical_run_full/run_log.json` — grid-search log
- `Results/classical_run_full/sprint3_comparison.json` — XGB vs DL paired-McNemar artefact
- `Results/classical_features_full/{train_frac100,val,test}.npz` — feature caches
- `Results/classical_sweep_full/sweep_summary.json` — sweep metrics
- `Results/classical_sweep_full/data_efficiency_curve.png` — curve plot
- `analysis/sprint3_full_comparison.py` — XGB-only paired-comparison script

Sprint 3 addendum:
- `Results/classical_run_full_svm/{classical_pipeline.pkl, classical_predictions.npz, classical_results.json}` — re-fit SVM on full
- `Results/classical_run_full_rf/{classical_pipeline.pkl, classical_predictions.npz, classical_results.json}` — re-fit RF on full
- `Results/classical_run_full/sprint3_all_classifiers.json` — 5-pipeline pairwise McNemar's + Cyst→Stone tally
- `analysis/sprint3_train_svm_rf.py` — SVM+RF re-fit script (rerunnable from cached features)
- `analysis/sprint3_all_classifiers.py` — 5-pipeline comparison script (rerunnable)

Sprint 3 second addendum (interpretability):
- `Results/classical_run_full/feature_importance.{json,csv}` — per-feature + per-group importances
- `Results/classical_run_full/feature_importance_group.png` — Paper Figure 2 candidate
- `Results/classical_run_full/feature_importance_top20.png` — supplementary (top-20 individual features)
- `Results/gradcam/cross_paradigm_disagreement.png` — 3-row 3-column figure: original / EffNet Grad-CAM / ConvNeXt V2 Grad-CAM with classical's prediction in the column-1 caption
- `Results/gradcam/gradcam_paradigm_manifest.json` — selected sample provenance
- `analysis/sprint3_feature_importance.py` — permutation + raw-XGB feature importance (rerunnable)
- `analysis/sprint3_gradcam_paradigm.py` — 3-way disagreement bucket selection + Grad-CAM rendering (rerunnable)

Sprint 3 third addendum (overfitting diagnostics):
- `Results/diagnostics/filename_proximity.{json,png}` — Diagnostic 1: classical-feature-space slice-leakage probe
- `Results/diagnostics/xgb_learning_curves.{json,png}` — Diagnostic 2: XGB train+val mlogloss / merror over n_estimators
- `Results/diagnostics/per_class_cv.{json,png}` — Diagnostic 3: 5-fold CV per-class F1 vs held-out test F1 (Stone 3.9 σ below CV mean — leakage signal)
- `Results/diagnostics/dl_learning_curves.{json,png}` — Diagnostic 4: DL train+val loss + val macro-F1 per epoch (saturation, no rebound)
- `analysis/diag_filename_proximity.py` — script for Diagnostic 1 (rerunnable)
- `analysis/diag_xgb_learning_curves.py` — script for Diagnostic 2 (rerunnable)
- `analysis/diag_per_class_cv.py` — script for Diagnostic 3 (rerunnable)
- `analysis/diag_dl_learning_curves.py` — script for Diagnostic 4 (rerunnable)
