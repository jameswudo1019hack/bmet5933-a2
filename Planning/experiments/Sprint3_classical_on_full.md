# Sprint 3 — Classical ML on the full dataset

**Date**: 2026-04-27
**Status**: complete
Related: [[Sprint2_ConvNeXtV2_on_full]], [[Sprint2_evaluation_ConvNeXtV2]], [[Sprint1_log]], [[DL_Improvements_Analysis]], [[Phase0_Design]], [[Project_Framing_v2]]

> **Framing.** Sprint 2 closed with classical-on-medium (0.9976) and DL-on-full (EffNet-B0 0.9819, ConvNeXt V2 0.9953). The medium-vs-full asymmetry left "classical fails on Cyst↔Tumor / DL fails on Cyst↔Stone" as a *medium-set* claim. Sprint 3 closes that asymmetry: classical retrained on the same full split (8,712 train, 1,867 test) so all three pipelines are now matched on the n=1867 test set. The scientific question: *does the paradigm-stable error pattern survive at full scale?* **Answer: in its original form, no — it was a medium-set artefact.** A narrower asymmetry survives (DL alone makes Cyst→Stone errors).

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

## Files written

- `Results/classical_run_full/classical_pipeline.pkl` — final XGBoost on train+val
- `Results/classical_run_full/classical_predictions.npz` — (y_true, y_pred, y_prob)
- `Results/classical_run_full/classical_results.json` — full evaluate() report
- `Results/classical_run_full/run_log.json` — grid-search log
- `Results/classical_run_full/sprint3_comparison.json` — paired McNemar's + failure-pair counts
- `Results/classical_features_full/{train_frac100,val,test}.npz` — feature caches (regenerable; not committed)
- `Results/classical_sweep_full/sweep_summary.json` — sweep metrics
- `Results/classical_sweep_full/data_efficiency_curve.png` — curve plot
- `analysis/sprint3_full_comparison.py` — paired-comparison script (rerunnable)
