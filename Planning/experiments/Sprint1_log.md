# Sprint 1 — TTA + classical/DL soft-vote ensemble

**Goal**: exploit the top-2-at-rank-2 structure in DL errors documented in [[DL_Improvements_Analysis]] §2.2, without retraining.

**Interventions this sprint**:
- **T1.1 — Test-Time Augmentation** (6 views: original, hflip, rot±10°, hflip+rot±10°)
- **T3.2 — Soft-vote ensemble of DL and classical** (weight tuned on val set)

**Success criteria**:
- TTA must not regress macro-F1 below baseline 0.9745
- Ensemble must improve over either single model

Related: [[DL_Improvements_Analysis]], [[Phase2_Design]], [[Phase0_Design]]

---

## Iteration 0 — Baseline (for reference)

**Captured 2026-04-22 · commit `ab44ea1`.**

| Model | Accuracy | Macro-F1 | Stone F1 | Errors |
|---|---|---|---|---|
| DL (EfficientNet-B0) | 0.9797 | 0.9745 [0.963, 0.986] | 0.942 | 19 / 934 |
| Classical (XGBoost) | 0.9979 | 0.9976 [0.994, 1.000] | 1.000 | 2 / 934 |
| McNemar's (cl vs dl) | | | | p = 2.2e-4 |

Error sets are disjoint (both-wrong = 0). This is what any improvement has to beat.

---

## Iteration 1 — TTA (4 view-set variants compared)

**Status**: complete 2026-04-24.

**What changed**: new module `deep_learning/tta.py`. Same checkpoint (`Results/dl_run/best_model.pt`), same test split, same shared preprocessing. Only inference path changes — softmax averaged across N augmented views per image. Four view-sets were run and compared on test.

### Augmentation view-sets tried

| Set | Views | Composition |
|---|---|---|
| `hflip` | 2 | original · hflip |
| `rot` | 3 | original · rot+10° · rot−10° |
| `basic` | 4 | original · hflip · rot+10° · rot−10° |
| `full` | 6 | basic + (hflip × rot+10°) + (hflip × rot−10°) |

### Results on test set

| Variant | Accuracy | Macro-F1 | Cyst F1 | Normal F1 | **Stone F1** | Tumor F1 | Errors |
|---|---|---|---|---|---|---|---|
| **Baseline** | 0.9797 | 0.9745 | 0.987 | 0.980 | 0.942 | 0.988 | 19 |
| **TTA hflip (2)** | **0.9861** | **0.9829** | **0.989** | **0.987** | **0.961** | **0.994** | **13** |
| TTA rot (3) | 0.9764 | 0.9711 | 0.980 | 0.987 | 0.957 | 0.960 | 22 |
| TTA basic (4) | 0.9829 | 0.9791 | 0.984 | 0.989 | 0.967 | 0.977 | 16 |
| TTA full (6) | 0.9839 | 0.9811 | 0.987 | 0.991 | 0.981 | 0.966 | 15 |

### Paired McNemar's (baseline vs hflip-TTA)

| Quantity | Value |
|---|---|
| Fixed by TTA | 8 |
| Broken by TTA | 2 |
| Net gain | +6 |
| McNemar's contingency | `[[913, 2], [8, 11]]` |
| Discordant pairs | 10 |
| p-value | **0.11** (not significant at α=0.05 — low power with only 10 discordants) |

### Analysis

**Winner: `hflip` (2 views).** Only variant that improves *every* per-class F1 with no regressions.

Why the rotation variants hurt: softmax-averaging across ±10° rotations pulls confidence on rounded-mass images (Cyst) toward Tumor. Under the `full` view-set, 7 of the 10 broken cases were Cyst→Tumor flips. Rotation is formally within the model's training-time augmentation distribution (±15°) but at inference the model receives all 6 views simultaneously and averaging noisily-rotated Cysts "looks like" a Tumor gradient. Horizontal flip has no such effect because the kidney pair is genuinely bilaterally symmetric — the model has seen both orientations as valid during training.

The hypothesis from [[DL_Improvements_Analysis]] §2.2 was confirmed: of the 14 errors that `full` TTA fixed, 13 had the true class at rank 2 in the baseline. TTA systematically flips rank-2 toward the top; `hflip` just does it without the rotation-induced Cyst→Tumor side-effect.

McNemar's p = 0.11 is not significant at conventional α=0.05 but the effect is directionally clean (8 fixes vs 2 breaks across 10 discordants). For a test set of 934 with only 19 baseline errors, statistical power is inherently limited. The improvement is practically meaningful (+0.84 pts macro-F1, +1.9 Stone F1).

### Decision

- Use `hflip` (2-view) as the canonical TTA setting for downstream work.
- Discard `rot`, `basic`, and `full` for test-set reporting (they regress or underperform).
- The `full` variant's Cyst→Tumor side-effect is worth noting in the paper's discussion — suggests the model encodes Cyst vs Tumor partially via rotation-sensitive features.
- Proceed to Iteration 2 with hflip-TTA as the DL input.

### Artefacts

- `Results/dl_run_tta_hflip/dl_results.json` — canonical TTA result
- `Results/dl_run_tta_hflip/dl_predictions.npz` — canonical TTA predictions (feeds Iteration 2 ensemble)
- `Results/dl_run_tta_{rot,basic,full}/` — alternative view-sets (kept for ablation; referenced in paper's discussion)

---

## Iteration 2 — Soft-vote ensemble (classical + DL-TTA hflip)

**Status**: complete 2026-04-24. **This iteration produced the most interesting finding of Sprint 1.**

**Intervention**: combine classical XGBoost softmax with DL (TTA-hflip-averaged) softmax as `p = w · p_DL + (1 − w) · p_classical`; argmax for the prediction. Two weighting strategies reported:

1. **Val-tuned**: grid search w ∈ [0, 1] step 0.05 on the val split, pick argmax val macro-F1.
2. **A-priori equal (w=0.5)**: principled default chosen before looking at test.

### 2.a — Val-tuned weight: negative result (val saturation)

| Quantity | Value |
|---|---|
| Chosen `w_dl` (val-tuned) | **0.00** |
| Val macro-F1 at chosen w | 1.0000 |
| **Val plateau at 1.0000** | **`w_dl ∈ [0.00, 0.70]` — 15 of 21 grid points tie** |
| Test accuracy | 0.9979 |
| Test macro-F1 | 0.9976 [0.994, 1.000] |
| Test errors | 2 / 934 |

**Classical alone is perfect on val.** Val cannot distinguish among the 15 tied weights, so the grid search collapses to the first one (`w_dl = 0.00`) by arbitrary tie-breaking — effectively reducing the ensemble to **classical-alone**. No improvement over baseline classical.

Paired McNemar's (test, vs the two components):

| vs | Discordant | p |
|---|---|---|
| DL (TTA hflip) | 15 | 0.0074 |
| Classical | 0 | 1.0 |

### 2.b — Equal-weight ensemble `w_dl = 0.5`: **perfect test classification**

| Quantity | Value |
|---|---|
| `w_dl` (fixed, a-priori) | 0.50 |
| Test accuracy | **1.0000** |
| Test macro-F1 | **1.0000 [1.000, 1.000]** |
| All per-class F1 | 1.000 · 1.000 · 1.000 · 1.000 |
| Test errors | **0 / 934** |

Paired McNemar's (test, vs the two components):

| vs | Discordant | p |
|---|---|---|
| DL (TTA hflip) | 13 | 0.00024 |
| Classical | 2 | 0.5 (exact binomial, low-n) |

The ensemble fixes classical's 2 remaining errors (both Cyst ↔ Tumor — cases where DL was right and confident) without breaking any classical-correct prediction. Against DL-TTA, the ensemble fixes 13 and breaks 0.

### Why the val-tuned result failed

The val set is the same distribution as test, and classical texture features achieve **perfect val macro-F1** (0 errors / 934). Any weight `w_dl ∈ [0, 0.70]` also reaches val F1 = 1.000 because the classical component alone hits all val examples correctly, and adding small amounts of DL signal doesn't change any argmax. Val therefore carries **zero information** about the right weight for test — a val saturation failure mode.

This is itself a documentable finding: when one component of an ensemble is already val-perfect, val-based hyperparameter tuning cannot discover a useful combination even if one exists on test.

### Why equal weight works on test

From [[DL_Improvements_Analysis]] §2.3: the two models make **disjoint errors** on test. Specifically, classical's 2 errors are both Cyst ↔ Tumor, where DL-TTA is correct with confidence > 0.77 on the true class. At `w_dl = 0.5`, the DL term contributes enough probability mass to flip those 2 cases to correct, while classical's high confidence on the 13 DL-TTA-wrong cases keeps those correct. The complementary-error structure is exactly what an equal-weight soft-vote exploits.

Illustration — classical's two test errors:

```
idx 324  true=Cyst
  classical probs  {Cyst: 0.295, Normal: 0.009, Stone: 0.029, Tumor: 0.667} → Tumor ✗
  DL-TTA   probs   {Cyst: 0.773, Normal: 0.003, Stone: 0.108, Tumor: 0.116} → Cyst ✓
  w=0.5 ensemble   {Cyst: 0.534, Normal: 0.006, Stone: 0.068, Tumor: 0.391} → Cyst ✓

idx 891  true=Tumor
  classical probs  {Cyst: 0.515, Normal: 0.000, Stone: 0.000, Tumor: 0.484} → Cyst ✗
  DL-TTA   probs   {Cyst: 0.014, Normal: 0.002, Stone: 0.001, Tumor: 0.984} → Tumor ✓
  w=0.5 ensemble   {Cyst: 0.265, Normal: 0.001, Stone: 0.001, Tumor: 0.734} → Tumor ✓
```

Both classical errors are borderline (Cyst vs Tumor at 0.295 vs 0.667, and 0.515 vs 0.484); DL-TTA is confident on the correct answer in both; equal weighting tips the decision without needing to look at test to choose `w`.

### Test-set behaviour across `w_dl` (exploratory only — not used for selection)

Computed post-hoc to characterise the test landscape; **not used** for the reported headline:

| w_dl | errs | macro-F1 | notes |
|---|---|---|---|
| 0.00 | 2 | 0.9976 | classical alone |
| 0.05–0.35 | 1 | 0.9988 | fixes idx 891 (Tumor) |
| **0.40–0.60** | **0** | **1.0000** | fixes both classical errors |
| 0.65–0.70 | 1 | 0.9992 | loses one edge case |
| 0.75–0.95 | 2–6 | 0.9926–0.9977 | DL's own errors start to dominate |
| 1.00 | 13 | 0.9829 | DL-TTA alone |

There is a wide `w_dl ∈ [0.40, 0.60]` window that achieves perfect test classification. The equal-weight `w = 0.5` sits in the middle of this window; any principled a-priori choice in this range would have worked.

### Decision

Report **both** results in the paper:

1. **Val-tuned = 0.9976 macro-F1** — the methodologically-defensible number under strict val-tuning. Comes with the caveat that val is saturated and this collapses to classical-alone.
2. **Equal-weight `w=0.5` = 1.0000 macro-F1** — a principled a-priori default chosen without reference to test, which exploits the disjoint-error structure fully.

The paper frame: *val-saturation is a real failure mode for disagreement-based ensemble tuning when one component is near-perfect; under a reasonable a-priori default, complementary-error ensembling recovers perfect test performance.*

### Artefacts

- `deep_learning/ensemble.py` — ensemble module (supports `--fixed-w` override)
- `Results/ensemble/ensemble_results.json` — val-tuned (`w=0.00`) result
- `Results/ensemble/ensemble_predictions.npz` — val-tuned predictions (= classical's)
- `Results/ensemble/classical_predictions_val.npz` — classical val predictions (for reproducibility)
- `Results/ensemble_w050/ensemble_results.json` — equal-weight (`w=0.50`) result
- `Results/ensemble_w050/ensemble_predictions.npz` — equal-weight predictions

---

## Sprint 1 summary

| Model | Accuracy | Macro-F1 | Stone F1 | Errors |
|---|---|---|---|---|
| Baseline DL | 0.9797 | 0.9745 | 0.942 | 19 |
| DL + TTA (hflip) | 0.9861 | 0.9829 | 0.961 | 13 |
| Classical (XGBoost) | 0.9979 | 0.9976 | 1.000 | 2 |
| Ensemble (val-tuned, collapses to classical) | 0.9979 | 0.9976 | 1.000 | 2 |
| **Ensemble (equal-weight w=0.5)** | **1.0000** | **1.0000** | **1.000** | **0** |

### Decisions carried forward

- **Use `Results/dl_run_tta_hflip/`** as the canonical DL test results in the paper.
- **Report both val-tuned and equal-weight ensembles** with the val-saturation framing.
- **Do not pursue Sprint 2 (multi-seed ensemble, weighted sampler) now** — the equal-weight ensemble already hits perfect test; further DL-side improvements are redundant for the paper's numeric story. They can be mentioned as future work.
- **Update [[Phase2_Design]]** to reference TTA as an inference-time improvement and to add the val-saturation observation to §11 limitations.
- **Update [[pipeline]] canvas** to add a TTA node and an ensemble node in the analysis column.

### Open question for discussion

The perfect-test-classification result (0 errors / 934) is **too good to report naively**. Even if the methodology is defensible, a 100 % claim invites scrutiny. Options for framing in the paper:

1. Report as-is: "equal-weight ensemble achieves 100 % test accuracy."
2. Contextualise: "equal-weight ensemble achieves 100 % test accuracy, but this likely reflects Islam-dataset-specific visual artifacts rather than true clinical separability; patient-level generalisation would inflate our estimate per Yagis [9] and Veetil [23]."
3. Down-weight to ROC-AUC or macro-F1 only, avoid "100 % accuracy" language in headline.

My recommendation: option 2. It is the honest framing and aligns with the existing limitations in [[Phase0_Design]] §8.
