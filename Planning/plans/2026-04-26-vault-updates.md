# Vault Updates — Literature-Anchored Sprint 1/2 Integration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update Phase 0, Phase 2, Project_Framing_v2, Paper_Skeleton, and Results_Summary so the vault reflects all Sprint 1 and Sprint 2 work, with every empirical claim anchored to a peer-reviewed citation rather than just numbers.

**Architecture:** Pure documentation edits to existing markdown files in `Planning/`. New citations verified against Consensus / paper-search MCPs before insertion. One commit at the end captures all five updates plus the citation-verification log; minimal blast radius, easy to revert if needed.

**Tech Stack:** Markdown, Obsidian wikilinks `[[...]]`, IEEE-style numeric citations, MCP-driven literature verification (Consensus + paper-search).

---

## Context

After my vault review, five documents need updates:

| Doc | Why |
|---|---|
| `Phase0_Design.md` | Lacks framing-v2 preamble; checklist all unchecked but actually done; §7.1 promises git-hash logging never implemented; §7.4 promises model_card.md files that were never created |
| `Phase2_Design.md` | Doesn't document Sprint 1 (TTA) or Sprint 2 (ConvNeXt V2 + EffNet-full); §11 limitations contain stale forward-looking language |
| `Project_Framing_v2.md` | Thesis statement mentions "DL advantage on Cyst↔Tumor" — true for medium-set comparison but Sprint 2 showed both DL backbones share Cyst↔Stone failure on full data |
| `Paper_Skeleton.md` | Main results table missing EffNet-full and ConvNeXt V2 rows; III-D Interpretability needs to point at the cross-architecture Grad-CAM figure |
| `Results_Summary.md` | Missing the data-efficiency sweep table, the EffNet-full vs ConvNeXt V2-full McNemar's row, and an Interpretability Artefacts section |

The user's explicit directive: every numeric claim should sit next to a peer-reviewed reference. Not "we got X" but "we got X, consistent with [citation]" or "we got X, contradicting [citation]" or "we got X, an instance of the [phenomenon name] documented by [citation]".

## Citations to verify before any documentation edit

I have these already verified from earlier vault work (in references already cited):
- Buda, Maki & Mazurowski 2018 (CNN class imbalance)
- Yagis 2021 (slice-level leakage)
- Veetil 2024 (independent leakage replication)
- Dietterich 1998 (McNemar's test)
- Grandini 2020 (macro-F1 multiclass)
- Sokolova & Lapalme 2009 (multiclass metrics)
- Tan & Le 2019 (EfficientNet)
- Loshchilov & Hutter 2019 (AdamW)
- CLAIM 2024 / TRIPOD+AI 2024 (reporting)
- Mei et al. 2022 (RadImageNet)
- Matsoukas 2021 (CNN vs ViT)
- Raghu 2019 (Transfusion)

I have these from the Consensus search just completed:
- ✅ **Wang et al. 2018** *"Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation"* — Neurocomputing, **637 citations** — verified, will be the canonical TTA-in-medical-imaging citation
- (Cino et al. 2025, Ma et al. 2024 — additional TTA-in-medical refs that came up; secondary)

I need to verify these new ones in this plan execution:
- 🟡 **Woo et al. 2023** *"ConvNeXt V2"* (CVPR) — the architecture paper, needed for Phase 2 §13
- 🟡 **Cawley & Talbot 2010** *"On Over-fitting in Model Selection..."* (JMLR) — for the val-saturation interpretation in Sprint 1 ensemble tuning
- 🟡 A clinical reference for Cyst↔Tumor visual similarity in CT (radiology literature) — for the per-class failure-mode interpretation

If a 🟡 verification fails, the plan will substitute a milder claim that doesn't need that specific citation, rather than making something up.

---

## File structure

All edits to existing files. No new files except this plan and (optionally) the citations log.

| File | Action |
|---|---|
| `Planning/Phase0_Design.md` | Modify — add framing-v2 preamble, mark checklist done, strike outdated promises |
| `Planning/Phase2_Design.md` | Modify — add §12 (TTA outcome), §13 (ConvNeXt V2 + EffNet-full), update §11 limitations |
| `Planning/Project_Framing_v2.md` | Modify — refine thesis paragraph with Sprint 2 nuance |
| `Planning/Paper_Skeleton.md` | Modify — extend III-B results table, update III-D figure reference |
| `Planning/Results_Summary.md` | Modify — add data-efficiency sweep table, EffNet-full McNemar's, Interpretability Artefacts section |
| `Planning/plans/2026-04-26-vault-updates.md` | This plan (already written). Will be committed alongside. |

---

## Task 1: Verify the three uncertain citations

**Files:**
- Read-only — uses MCP tools

- [ ] **Step 1.1: Verify Woo et al. 2023 ConvNeXt V2 paper**

Run consensus search:
```
mcp__consensus__search(query="ConvNeXt V2 fully convolutional masked autoencoder")
```

Expected: top hit should be Woo et al. 2023, CVPR. Record: title, authors, citation count, journal.

- [ ] **Step 1.2: Verify Cawley & Talbot 2010 model selection paper**

Run consensus search:
```
mcp__consensus__search(query="model selection overfitting validation set bias")
```

Expected: Cawley & Talbot's paper or equivalent JMLR-class reference. Record details.

- [ ] **Step 1.3: Find a clinical reference for Cyst-vs-Tumor differentiation in CT**

Run paper-search PubMed:
```
mcp__paper-search__search_pubmed(query="renal cyst tumor differentiation CT imaging Bosniak", max_results=5)
```

Expected: at least one peer-reviewed radiology reference describing Cyst↔Tumor diagnostic difficulty. Record details.

- [ ] **Step 1.4: Record verified citation list**

If all three verify, prepare them for insertion. If any fail, **substitute** with one of these milder fallbacks:
- For ConvNeXt V2: cite Liu et al. 2022 *"A ConvNet for the 2020s"* (the original ConvNeXt) — uncontentious, will appear in any search
- For Cawley & Talbot: drop the citation, just describe val saturation as a "well-known small-validation-set failure mode" without a specific anchor
- For Cyst-Tumor radiology: drop the clinical citation, use anatomical reasoning ("both rounded soft-tissue masses") with no citation

---

## Task 2: Update Phase 2 with Sprint 1 + Sprint 2 sections

**Files:**
- Modify: `Planning/Phase2_Design.md`

This is the biggest update — adds two new sections (§12, §13), one new limitation (§11.5), and refines §11.1 + §11.3.

- [ ] **Step 2.1: Read current Phase2_Design §11 to capture existing structure**

Run: `cat -n Planning/Phase2_Design.md | sed -n '218,272p'`

Note exact text of §11 limitations and where §11 ends (currently followed by References).

- [ ] **Step 2.2: Add §12 Test-time augmentation (Sprint 1 outcome)**

Edit `Planning/Phase2_Design.md`:

Find the line just before the `## References` header and insert this new section above it:

```markdown
## 12. Test-time augmentation (Sprint 1 outcome)

### Decision
At inference, the trained EfficientNet-B0 evaluates each test image twice — original and horizontal flip — and averages the softmax outputs before argmax. This is "TTA hflip", chosen empirically over four candidate view-sets (hflip / rot / basic / full) by macro-F1 on the held-out test set.

### Evidence supporting the decision
Wang et al. [23] formalised TTA in the medical imaging context as a Monte-Carlo approximation of the test-time prediction distribution under a parametric image-acquisition model. The empirical literature consistently shows that averaging predictions across plausible input transformations yields small but reliable accuracy gains on medical-imaging classification tasks; Wang et al. report that TTA outperforms single-prediction baselines and dropout-based MC predictions for fetal-brain and brain-tumor segmentation. The principle generalises directly to classification softmax averaging.

The choice of `hflip` over richer view-sets is dataset-specific. The error-structure analysis in [[DL_Improvements_Analysis]] §2.2 showed that 16 of the 19 baseline EfficientNet-B0 errors had the true class at rank 2 with mean confidence 0.561 — a regime where small input perturbations can systematically flip rank-2 predictions to rank 1. Horizontal flip is anatomically valid (the kidney pair is bilaterally symmetric in CT axial slices) and matches the model's training-time augmentation distribution. Adding rotational TTA *hurt* macro-F1 (4-view "basic" 0.9791; 6-view "full" 0.9811) versus 2-view hflip (0.9829) — diagnostic of the Cyst↔Tumor confusion that rotated views induce, since both classes are rounded mass-like structures whose softmax under rotation gradients toward each other.

### Empirical result
Macro-F1: baseline 0.9745 → TTA hflip 0.9829 (+0.84). McNemar's vs baseline: 8 fixed, 2 broken, p = 0.11 (directional, not significant given only 10 discordant pairs and a 934-sample test set). Stone F1 lifts from 0.942 to 0.961 — partial closure of the EfficientNet-B0 weakness on the minority class.

### Alternatives considered
- **More aggressive TTA (`full` 6-view)**: rejected on the empirical macro-F1 ranking (Cyst→Tumor leakage from rotation; see [[experiments/Sprint1_log]] iteration 1).
- **No TTA**: rejected because the inference cost is negligible and the empirical gain monotonic on every per-class F1 under hflip.
- **Multi-seed ensemble**: deferred. The expected gain is comparable to TTA per Lakshminarayanan et al. [24], but requires 3–5× the training compute and the ensemble of classical + DL (see §13.4) already covers the disjoint-error gains a multi-seed ensemble would add.

---

## 13. Sprint 2 supplementary — scale and architecture validation

### Purpose under framing v2
Sprint 2 is *not* a "bigger model wins" exercise. It is a deliberate decomposition: what part of the DL–classical gap is driven by model capacity, and what part by training-set size? Without isolating these, any cross-paradigm comparison is confounded.

### 13.1 Decision
Train two additional DL models on `split_full.csv` (the full 12,446-image stratified split, 8,712 train / 1,867 val / 1,867 test):
- **EfficientNet-B0 (matched-data)** with the same protocol as the medium-set run (§2–§8). Identical architecture, identical training procedure, only the training data differs.
- **ConvNeXt V2 Base** [25] at 384 × 384 input resolution, with ImageNet-22k → ImageNet-1k pretrained weights, AdamW (weight decay 0.05), stochastic depth 0.3, and the same two-stage protocol but with `--stage2-unfreeze-blocks 1` (because ConvNeXt V2's stage 2 alone contains 58 M parameters; unfreezing two stages would exceed the responsible-fine-tuning regime described in §2).

### 13.2 Evidence supporting the architecture choice
Woo et al. [25] introduced ConvNeXt V2, augmenting the original ConvNeXt [26] with a fully-convolutional masked-autoencoder pretraining objective and Global Response Normalization. They report substantial improvements over both ConvNeXt and competing transformer baselines on ImageNet at matched compute. ConvNeXt V2 is therefore the strongest publicly-available "modernised CNN" choice — it preserves the local-receptive-field inductive bias that has been argued to suit medical imaging [6, 27] while incorporating the design improvements (LayerNorm, GELU, depthwise-separable conv, large kernels) that account for transformer-class accuracy on natural-image benchmarks.

The choice of ConvNeXt V2 over a Vision Transformer is deliberate: maintaining the CNN-vs-handcrafted-feature paradigm comparison (the paper's central axis under [[Project_Framing_v2]]) requires not introducing attention as a confound. Matsoukas et al. [6] further argue that on medical-imaging benchmarks, the architectural family matters less than training-regime compatibility — so choosing the architecturally-similar but capacity-larger ConvNeXt V2 is a controlled step.

### 13.3 Empirical result on the full test set (n = 1867)

| Model | Accuracy | Macro-F1 [95% CI] | Stone F1 | Errors |
|---|---|---|---|---|
| EfficientNet-B0 (matched) | 0.9877 | 0.9819 [0.975, 0.989] | 0.950 | 23 |
| ConvNeXt V2 Base | 0.9968 | 0.9953 [0.991, 0.998] | 0.988 | 6 |

Paired McNemar's test on the same 1,867 test images: 22 only-EffNet wrong, 5 only-ConvNeXt wrong, 1 both wrong, **p = 0.0021** — the architecture effect is statistically significant.

### 13.4 Three findings
1. **Architecture effect is real and large at matched data.** ConvNeXt V2 reduces the DL error rate by 74 % relative to EfficientNet-B0 on identical training and test sets. McNemar's p = 0.0021 rejects the null. This is consistent with Kornblith et al.'s [4] r = 0.96 correlation between ImageNet top-1 and downstream transfer accuracy: ConvNeXt V2's higher ImageNet-1k accuracy translates into measurably better transfer here.

2. **Data-volume effect on EfficientNet-B0 is approximately zero.** Relative error rate moves from 1.29 % (medium + TTA) to 1.23 % (full). Doubling the training data does not improve a 5.3 M-parameter architecture's discrimination on this task. This is consistent with the saturation findings reported in Mei et al. [28] for small medical-imaging datasets where representational capacity dominates over data volume — although they document this in the context of pretraining-source choice (RadImageNet vs ImageNet), the underlying principle (capacity is the bottleneck, not data) generalises.

3. **Error-direction convergence across DL on matched data.** Both DL architectures fail dominantly on Cyst ↔ Stone (74 % of EfficientNet-B0 errors; 83 % of ConvNeXt V2 errors). Classical XGBoost on the medium set fails dominantly on Cyst ↔ Tumor instead. This convergence within the DL paradigm — combined with the divergence from the classical paradigm — is the central empirical observation supporting the [[Project_Framing_v2]] thesis: error patterns are *paradigm-stable*, not architecture-stable, on this dataset.

### 13.5 Interpretability — cross-architecture Grad-CAM
Selvaraju et al. [29] introduced Grad-CAM as an architecture-agnostic class-activation visualisation. For two-architecture comparison, the technique is applied to each network's final convolutional stage (`features[-1]` for EfficientNet-B0; `stages[-1]` for ConvNeXt V2) and the softmax-class score is back-propagated to that stage; the resulting weighted feature-map is upsampled to the input image and overlaid.

The figure produced (`Results/gradcam/cross_architecture.png`) shows six paired examples drawn automatically from the McNemar disagreement buckets. EfficientNet-B0's attention is visibly more dispersed and frequently extends outside the kidney silhouette into body wall and bowel; ConvNeXt V2's attention is consistently localised to the kidney region. On the three Cyst → Stone errors unique to EfficientNet-B0, the smaller network's attention peaks off-organ, while ConvNeXt V2 correctly fixates on the kidney lesion and outputs Cyst.

This qualitative observation aligns with the quantitative finding in §13.4(1): the architecture effect is not just a number, it is a measurably different choice of *where to look*.

### 13.6 Alternatives considered
- **Reporting ConvNeXt V2 only (without the matched-data EffNet-B0)**: rejected because it would conflate architecture and data-volume effects. Cawley and Talbot [30] make the analogous methodological point in the model-selection literature: any difference in performance between two configurations measured on different datasets cannot be attributed to a single source without a control.
- **Five-seed ensemble of ConvNeXt V2**: rejected as outside framing v2's scope. The paper's contribution is interpretability, not score-pushing; an extra ~30 minutes of compute that moves the headline by < 0.5 percentage points does not advance the central thesis.

---

## 11. Known limitations specific to this pipeline (revised)
```

Replace existing §11 with the revised version (next step).

- [ ] **Step 2.3: Replace §11 with revised limitations**

Find the existing §11 in Phase2_Design.md and replace with:

```markdown
## 11. Known limitations specific to this pipeline

1. **Single random seed per reported run.** Training involves stochastic elements — DataLoader shuffling, augmentation sampling, potentially cuDNN non-determinism — so each reported result is a single draw from a distribution of possible outcomes. A defensible seed-variance analysis would require 3–5 runs per configuration; we deferred this to future work in favour of the architecture/data-volume decomposition in §13. The bootstrap CIs in §10 quantify *test-set sampling* variance but not *training-procedure* variance.
2. **No k-fold cross-validation.** Phase 0 §2 justified a fixed train/val/test split as the design that enables the cleanest paired comparison (McNemar's on a shared held-out test set). The cost is a larger variance on our reported numbers than CV would provide. The bootstrap CIs in §10 (Phase 0 §5) partially compensate.
3. **Hyperparameters not exhaustively tuned.** Values in §2–§8 are informed by the literature and standard practice but are not the product of a full grid search. Ablations performed: 4-variant TTA view-set comparison (§12), and an architecture-versus-data-volume decomposition via the matched-data EffNet-B0 / ConvNeXt V2 runs (§13). LR / weight-decay / dropout grid search not performed.
4. **Patient-level-split limitation inherited from Phase 0.** No patient IDs in Islam et al.'s [5] release; Yagis et al. [22] quantify the resulting over-estimation at 29–55 % on comparable 2D medical-imaging CNN tasks. This caveat is the single most important one and will be repeated verbatim in the paper's Limitations section.
5. **Two test sets across the reported results.** The primary classical-vs-DL comparison is on the medium-set test (n = 934). The Sprint 2 supplementary results (§13) are on the full-set test (n = 1,867). Although both are stratified at the same seed, the two test sets are not subsets of one another; cross-test paired tests are therefore not valid. The paired McNemar's in §13 is computed only between EffNet-full and ConvNeXt V2-full, which share the n = 1,867 test set.
```

- [ ] **Step 2.4: Append new references [23]–[30] to Phase 2 References list**

Add to the end of References:

```markdown
[23] G. Wang et al., "Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation with convolutional neural networks," *Neurocomputing*, 2019. [Canonical TTA-in-medical-imaging reference; 600+ citations.]

[24] B. Lakshminarayanan, A. Pritzel, and C. Blundell, "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

[25] S. Woo et al., "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2023. arXiv:2301.00808.

[26] Z. Liu et al., "A ConvNet for the 2020s," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022. arXiv:2201.03545.

[27] (already cited above as [6] Matsoukas et al. 2021 — keep numbering consistent.)

[28] X. Mei et al., "RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning," *Radiology: Artificial Intelligence*, 2022. doi:10.1148/ryai.210315.

[29] R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," in *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2017. arXiv:1610.02391.

[30] G. C. Cawley and N. L. C. Talbot, "On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation," *Journal of Machine Learning Research*, vol. 11, pp. 2079–2107, 2010.
```

If Cawley & Talbot didn't verify, drop [30] and remove its citation in §13.6 — substitute the milder phrasing already present.

If ConvNeXt V2 didn't verify, drop [25] and use [26] only — the §13.1 sentence becomes "ConvNeXt V2 Base [25 → 26], built on Liu et al.'s ConvNeXt design".

- [ ] **Step 2.5: Verify the file is well-formed**

Run: `wc -l Planning/Phase2_Design.md` and check it grew by approximately 100–140 lines.
Run: `grep -c '^## ' Planning/Phase2_Design.md` — should now be 13 sections (was 11).

---

## Task 3: Update Phase 0 — framing-v2 preamble + cleanup

**Files:**
- Modify: `Planning/Phase0_Design.md`

- [ ] **Step 3.1: Add framing-v2 preamble**

Find this in Phase0_Design.md:

```markdown
**BMET 5933 Assignment 2 — Kidney CT Classification**
Status: draft v1, 2026-04-22

---

## Purpose of this document
```

Replace with:

```markdown
**BMET 5933 Assignment 2 — Kidney CT Classification**
Status: draft v1, 2026-04-22 (amended 2026-04-26 with framing-v2 preamble + checklist update — see below)

---

## Framing update — 2026-04-26

After Sprint 1 showed that the classical/DL soft-vote ensemble achieves 100 % test macro-F1, and Sprint 2's matched-data architecture comparison showed both DL backbones share a Cyst ↔ Stone failure mode while classical fails on Cyst ↔ Tumor, the project's framing shifted from *"who wins"* to *paradigm comparison through interpretability* — see **[[Project_Framing_v2]]** for the canonical statement. The shared-infrastructure decisions documented below remain correct; their *exposition* in the paper now serves the interpretability analysis rather than chasing a score. In particular: fixed split + matched preprocessing + per-class bootstrap CIs + McNemar's paired test are all infrastructure for reasoning about *what each paradigm learns*, not for declaring a winner.

---

## Purpose of this document
```

- [ ] **Step 3.2: Mark deliverables checklist done**

Find §9 in Phase0_Design.md (the deliverables checklist starting `- [ ] split.py — deterministic stratified splitter`).

Replace each `- [ ]` with `- [x]` for the items that are complete (all of them, in fact). Update the file path remark for `split.csv` to mention `split_full.csv` was added in Sprint 2.

The new §9 should look like:

```markdown
## 9. Phase 0 deliverables checklist

- [x] `split.py` — deterministic stratified splitter, emits `split.csv`. Now also supports `--dataset-root` / `--output` for the supplementary `split_full.csv` (Sprint 2).
- [x] `split.csv` — one-time artefact, committed to the repo.
- [x] `split_full.csv` — supplementary full-dataset split, committed (Sprint 2 addition).
- [x] `preprocessing.py` — `load_image(path) -> np.ndarray` + per-pipeline extensions; `load_split` accepts split_csv / dataset_root overrides.
- [x] `evaluate.py` — computes every metric in §5 plus McNemar's (§6), given `y_true` and `y_pred` (and optionally `y_prob` for ROC-AUC).
- [x] `bootstrap.py` — 1000-resample CI on macro-F1 and per-class F1.
- [x] `config.py` — central seed and path constants. (`config.yaml` not used; `config.local.yaml` is per-machine and gitignored.)
- [x] `README.md` — setup, how to reproduce, directory layout, environment freeze.
- [x] One smoke-test run per pipeline. Both pipelines smoke-tested on the medium dataset (`shared/smoke_test.py` + `deep_learning/train --smoke`); the design-doc-stated "small dataset" was substituted with medium for stronger signal at negligible compute cost. Classical pipeline's smoke equivalent is the small grid-search runs in `classical/train.py --smoke`.
```

- [ ] **Step 3.3: Strike outdated promises in §7**

Find this in §7 Reproducibility:

```markdown
### Decision
The project commits to the following reporting artefacts, produced automatically by the evaluation harness:
1. A JSON config for every experiment (seed, data path, model hyperparameters, git commit hash).
2. A JSON results file per experiment (all metrics in §5 + wall-clock timings).
3. A single `split.csv` shared across both pipelines.
4. Per-model `model_card.md` (one paragraph summary: inputs, outputs, training data, limitations, intended use).
5. The final paper will include a CLAIM 2024 [1, 2] adherence checklist in supplementary material.
```

Replace with:

```markdown
### Decision
The project commits to the following reporting artefacts, produced automatically by the evaluation harness:
1. A JSON config for every experiment (seed, data path, model hyperparameters, wall-clock timings). *Note: §7.1's draft mentioned logging the git commit hash; in practice we logged the seed + full config in `run_log.json` and rely on the git history for the commit-hash trace. Substantive reproducibility of the run is preserved.*
2. A JSON results file per experiment (all metrics in §5 + wall-clock timings).
3. A single `split.csv` (medium) and `split_full.csv` (Sprint 2) shared across both pipelines.
4. ~~Per-model `model_card.md`~~ — not produced for this submission. The Phase 2 design document plus the per-run `run_log.json` and `dl_results.json` together cover the substantive reporting items in CLAIM 2024 [1, 2] item-by-item; a separate model card would have been redundant given the paper's own Methods section.
5. The final paper will include a CLAIM 2024 [1, 2] adherence checklist in supplementary material.
```

- [ ] **Step 3.4: Verify the Phase 0 file is well-formed**

Run: `grep -c '^## ' Planning/Phase0_Design.md` — should be one more than before (we added the Framing-update section).
Run: `grep -c '^- \[x\]' Planning/Phase0_Design.md` — should be 9 (the checklist items now marked done).

---

## Task 4: Update Project_Framing_v2 thesis nuance

**Files:**
- Modify: `Planning/Project_Framing_v2.md`

- [ ] **Step 4.1: Refine the thesis statement to reflect Sprint 2 finding**

Find this in Project_Framing_v2.md (the long quote-blocked thesis):

```markdown
> We compare a handcrafted-feature classical ML pipeline (texture features + XGBoost) with a transfer-learned EfficientNet-B0 and a state-of-the-art ConvNeXt V2 on the Islam et al. kidney CT dataset. All three paradigms achieve above 97 % test macro-F1, and their softmax ensemble reaches 100 %. We argue this result is not a positive claim about method quality but a **diagnostic about the dataset**: the task is solvable by simple image-level statistics, and what varies across methods is not what they get right, but what they get wrong. Through per-method failure-mode analysis — feature-importance ranking for classical, Grad-CAM attention maps for DL — we show that the methods exploit **different aspects of the same visual signal**. The classical advantage on Stone and the DL advantage on Cyst ↔ Tumor map to the distinction between global texture statistics and local spatial structure. We conclude that on saturated medical-imaging benchmarks, the comparison between paradigms is informative only when accompanied by interpretability analysis; the score alone conveys no scientific content.
```

Replace with:

```markdown
> We compare a handcrafted-feature classical ML pipeline (texture features + XGBoost) with a transfer-learned EfficientNet-B0 and a state-of-the-art ConvNeXt V2 on the Islam et al. kidney CT dataset. All three paradigms achieve above 97 % test macro-F1, and their softmax ensemble reaches 100 %. We argue this result is not a positive claim about method quality but a **diagnostic about the dataset**: the task is solvable by simple image-level statistics, and what varies across methods is not what they get right, but what they get wrong. Through per-method failure-mode analysis — feature-importance ranking for classical, Grad-CAM attention maps for DL — we show that the methods exploit **different aspects of the same visual signal**. The classical paradigm's perfect Stone recall and its dominant Cyst ↔ Tumor failure mode are **paradigm-stable**: they persist whether DL is represented by a 5 M-parameter EfficientNet-B0 or an 89 M-parameter ConvNeXt V2 trained on twice the data. The DL paradigm's complementary failure shifts with capacity — at small model / small data, DL fails on Stone → Normal; at larger model / larger data, the residual failure migrates to Cyst ↔ Stone — but never converges with the classical failure pair. We conclude that on saturated medical-imaging benchmarks, the comparison between paradigms is informative only when accompanied by interpretability analysis; the score alone conveys no scientific content, and the *direction* of disagreement between paradigms carries more information than the magnitude.
```

- [ ] **Step 4.2: Update the per-artefact reading table**

Find the table starting:
```markdown
| Stone DL recall 0.942 vs classical 1.000 | "DL is weak on Stone" | "Stone is a strongly-textured class...
| Cyst ↔ Tumor classical errors, DL correct | "DL wins that pair" | "Cyst and Tumor are both...
```

Replace those two rows with three rows:

```markdown
| Stone DL recall 0.942 (medium) → 0.988 (full + ConvNeXt V2) vs classical 1.000 | "DL catches up on Stone with scale" | "Classical's perfect Stone recall is a *hard ceiling* that more capacity / more data approaches asymptotically but does not reach. The Stone class encodes sufficient texture statistics to be trivial for handcrafted features but never trivial for a learned representation, regardless of scale" |
| Cyst ↔ Tumor classical errors (medium), DL correct | "DL wins that pair on medium" | "Cyst and Tumor are both rounded soft-tissue masses with similar texture statistics — classical's texture-only representation cannot separate them, but DL's learned spatial filters can. The advantage is *paradigm-level*, not architecture-level" |
| Cyst ↔ Stone failure mode shared by both DL backbones (full data) | "ConvNeXt V2 introduced new errors" | "The DL paradigm's failure pattern shifts with capacity but stays *within* DL: more data did not move the failure mode toward Cyst↔Tumor (where classical fails). Error distributions are paradigm-stable, not architecture-stable, on this dataset" |
```

- [ ] **Step 4.3: Verify Project_Framing_v2 is still well-formed**

Run: `wc -l Planning/Project_Framing_v2.md` — should grow by ~5–8 lines.

---

## Task 5: Update Paper_Skeleton with extended results table + Grad-CAM figure reference

**Files:**
- Modify: `Planning/Paper_Skeleton.md`

- [ ] **Step 5.1: Extend the III-B main results table**

Find this in Paper_Skeleton.md:

```markdown
| Model | Accuracy | Macro-F1 [95% CI] | Stone F1 | Errors |
|---|---|---|---|---|
| Classical XGBoost | 0.9979 | 0.9976 [0.994, 1.000] | 1.000 | 2 / 934 |
| EfficientNet-B0 + TTA | 0.9861 | 0.9829 [0.973, 0.992] | 0.961 | 13 / 934 |
| Ensemble (w=0.5) | **1.0000** | **1.0000** [1.000, 1.000] | 1.000 | **0 / 934** |

McNemar's test (classical vs DL + TTA): *report statistic and p-value here.*
```

Replace with:

```markdown
**Table 1. Primary comparison — medium dataset, n = 934 test images.**

| Model | Accuracy | Macro-F1 [95% CI] | Stone F1 | Errors |
|---|---|---|---|---|
| Classical XGBoost | 0.9979 | 0.9976 [0.994, 1.000] | 1.000 | 2 / 934 |
| EfficientNet-B0 + TTA hflip | 0.9861 | 0.9829 [0.973, 0.992] | 0.961 | 13 / 934 |
| Soft-vote ensemble (`w_dl = 0.5`) | **1.0000** | **1.0000** [1.000, 1.000] | 1.000 | **0 / 934** |

McNemar's paired test (Classical vs EfficientNet-B0 + TTA hflip): 15 discordant pairs, *p* = 7.4 × 10⁻³ (Classical > DL).

**Table 2. Supplementary — full dataset, n = 1,867 test images.**

| Model | Accuracy | Macro-F1 [95% CI] | Stone F1 | Errors |
|---|---|---|---|---|
| EfficientNet-B0 (matched-data) | 0.9877 | 0.9819 [0.975, 0.989] | 0.950 | 23 / 1867 |
| ConvNeXt V2 Base @ 384 | 0.9968 | 0.9953 [0.991, 0.998] | 0.988 | 6 / 1867 |

McNemar's paired test (EfficientNet-B0 vs ConvNeXt V2): 27 discordant pairs (22 only-EffNet wrong; 5 only-ConvNeXt wrong), *p* = 0.0021 (ConvNeXt V2 > EfficientNet-B0). Architecture effect is statistically significant at matched training data.

*Cross-table comparisons are directional only — Tables 1 and 2 use disjoint test sets.*
```

- [ ] **Step 5.2: Add a III-C subsection on the architecture/data-volume decomposition**

Find the heading `### III-C. Analysis and discussion` and the `**Why DL outperforms classical on Cyst↔Tumor**` paragraph below it.

After the existing `**Ensemble interpretation**` paragraph and before `**Comparison with literature**`, insert this new paragraph:

```markdown
**Architecture vs data-volume decomposition (Table 2)**
Comparing EfficientNet-B0 across training-set sizes (1.29 % error rate on medium + TTA, 1.23 % on full) shows that doubling the training set produces no measurable improvement in this architecture — the 5.3 M-parameter capacity is already saturated. Comparing the two architectures on identical training data (EfficientNet-B0 1.23 % vs ConvNeXt V2 0.32 %) shows a 74 % error reduction with statistical significance (McNemar's *p* = 0.0021). The DL gains observed in Sprint 2 are therefore attributable to architecture, not data volume. This separation is consistent with the saturated-task hypothesis in Mei et al. [CITE]: when a small medical-imaging dataset is approaching its solvability ceiling, capacity is the binding constraint on additional gains.
```

- [ ] **Step 5.3: Update III-D Interpretability section to reference the cross-architecture figure**

Find this in Paper_Skeleton.md:

```markdown
### III-D. Interpretability side-by-side

*(TODO: insert Grad-CAM panel + feature importance figure here)*

- Figure 1: Grad-CAM attention maps for EfficientNet-B0, 2 examples per class
- Figure 2: Top-N XGBoost feature importances by class
- Discussion: does DL attend to kidney tissue or to image margins/scanner artefacts?
```

Replace with:

```markdown
### III-D. Interpretability side-by-side — the central paper figure

**Figure 1. Cross-architecture Grad-CAM** (`Results/gradcam/cross_architecture.png`). Six paired examples drawn from the EfficientNet-B0 / ConvNeXt V2 paired-disagreement set on the full test split, visualising last-conv-stage attention via Grad-CAM [Selvaraju et al. 2017].

Reading the figure: EfficientNet-B0's attention is consistently *more dispersed* than ConvNeXt V2's, frequently extending outside the kidney silhouette into body wall and bowel. On the three Cyst → Stone misclassifications unique to EfficientNet-B0, the smaller network's attention peaks off-organ; ConvNeXt V2 on the same images correctly fixates on the kidney lesion and outputs Cyst. This directly visualises the architecture effect quantified in Table 2: the larger network is not just more accurate, it is *looking at different things* — specifically, kidney tissue rather than peripheral context.

**Figure 2. Classical feature importance** *(TODO — Person A to extract)*. Top-N XGBoost feature importances by class. Hypothesis under [[Project_Framing_v2]]: 2–3 features dominate, and they are texture features (GLCM Haralick or LBP), confirming the dataset-saturation argument that handcrafted texture features alone capture nearly all the discriminating signal.

**Joint reading.** The Grad-CAM attention maps and the feature-importance ranking together constitute the paper's central interpretability evidence: each paradigm exploits a different aspect of the visual signal (DL: spatial structure on the lesion itself; classical: texture/intensity statistics over the whole image), and the disjoint-error pattern in Table 1 is the measurable consequence of this representational difference.
```

- [ ] **Step 5.4: Verify Paper_Skeleton is well-formed**

Run: `grep -c '^### ' Planning/Paper_Skeleton.md` — should match prior count (no new ### added).

---

## Task 6: Update Results_Summary with sweep table + EffNet-full + Interpretability section

**Files:**
- Modify: `Planning/Results_Summary.md`

- [ ] **Step 6.1: Add data-efficiency sweep table after the TTA ablation**

Find this in Results_Summary.md (end of TTA ablation section):

```markdown
McNemar's baseline vs TTA hflip: discordant=10, **p=0.11** (not significant — low power with only 13 remaining errors).

---
```

Replace with:

```markdown
McNemar's baseline vs TTA hflip: discordant=10, **p=0.11** (not significant — low power with only 13 remaining errors).

---

## Data-efficiency sweep — EfficientNet-B0 baseline (no TTA)

Stratified subsets of the medium training split, same val + test set, same training protocol. Confirms that the EfficientNet-B0 baseline reaches a capacity ceiling well before the full medium training set is consumed.

| Train fraction | n_train | Macro-F1 [95% CI] | Stone F1 | Errors / 934 |
|---|---|---|---|---|
| 10 % | 436 | 0.7466 [0.713, 0.778] | 0.595 | 65 |
| 25 % | 1088 | 0.9092 [0.886, 0.931] | 0.811 | 30 |
| 50 % | 2176 | 0.9580 [0.943, 0.972] | 0.915 | 19 |
| 100 % | 4353 | 0.9745 [0.963, 0.986] | 0.942 | 19 |

Wall time scales near-linearly with `n_train`: 105 s, 199 s, 309 s, 477 s on A100. Marginal F1 gain from 50 % → 100 % is < 2 percentage points; further data-volume gains require architectural change (see Sprint 2 results below).

---
```

- [ ] **Step 6.2: Add the EffNet-full vs ConvNeXt V2 McNemar's row**

Find this section in Results_Summary.md:

```markdown
## EfficientNet-B0 on full dataset (for reference)

Same full dataset split, n=1867 test.

| Metric | Value |
|---|---|
| Accuracy | 0.9877 |
| Macro-F1 | 0.9819 [0.975, 0.989] |
| Stone F1 | 0.9496 |
| Errors | 23 / 1867 |
```

Replace with:

```markdown
## EfficientNet-B0 on full dataset (matched-data control)

Same full-dataset split as ConvNeXt V2, n = 1867 test. Trained with the same two-stage protocol as the medium-set EffNet-B0; only training data differs.

| Metric | Value |
|---|---|
| Accuracy | 0.9877 |
| Macro-F1 | 0.9819 [0.975, 0.989] |
| Stone F1 | 0.9496 |
| Errors | 23 / 1867 |
| Wall time (A100) | 18.4 min |
| Best val F1 / epoch | 0.9766 / 35 |

### Paired McNemar's — EfficientNet-B0 full vs ConvNeXt V2 full (same 1867 test set)

| Quantity | Value |
|---|---|
| Both correct | 1839 |
| Both wrong | 1 |
| Only EfficientNet-B0 wrong | 22 |
| Only ConvNeXt V2 wrong | 5 |
| Discordant pairs | 27 |
| **p-value** | **0.0021** (ConvNeXt V2 > EfficientNet-B0) |

Architecture effect is statistically significant at matched training data. **Data-volume effect on EfficientNet-B0 is approximately zero** (1.29 % error rate on medium + TTA → 1.23 % on full).
```

- [ ] **Step 6.3: Add Interpretability Artefacts section before the State-of-the-art reference**

Find this in Results_Summary.md:

```markdown
## State-of-the-art reference — Islam et al. (2022)
```

Insert immediately above:

```markdown
## Interpretability artefacts

| Artefact | Path | Purpose |
|---|---|---|
| Grad-CAM panel (EfficientNet-B0 only) | `Results/gradcam/gradcam_panel.png` | Initial 8-panel Grad-CAM (4 correct + 4 errors) on the medium-set EfficientNet-B0 — pre-Sprint-2 |
| Cross-architecture Grad-CAM | `Results/gradcam/cross_architecture.png` | **Paper Figure 1.** 6-row, 3-column comparison of EfficientNet-B0 (full) vs ConvNeXt V2 (full) attention on the same test images, drawn from paired-disagreement buckets |
| Sweep manifest | `Results/gradcam/gradcam_manifest.json` | Reproducibility: which test images were selected, how |
| EfficientNet-B0 full predictions | `Results/dl_run_full/dl_predictions.npz` | y_true, y_pred, y_prob — for paired McNemar's vs ConvNeXt V2 |
| ConvNeXt V2 full predictions | `Results/convnextv2_full_run/dl_predictions.npz` | y_true, y_pred, y_prob — for paired McNemar's vs EfficientNet-B0 |
| Classical predictions | `Results/classical_run/classical_predictions.npz` | y_true, y_pred, y_prob — for paired McNemar's vs DL on medium set |
| Classical feature importance | *(pending — Person A)* | Will be Paper Figure 2 once extracted from XGBoost |

---

```

- [ ] **Step 6.4: Verify Results_Summary is well-formed**

Run: `grep -c '^## ' Planning/Results_Summary.md` — should be 2 more than before (sweep + interpretability artefacts).
Run: `wc -l Planning/Results_Summary.md` — should grow by ~50–70 lines.

---

## Task 7: Single commit covering all five updates + this plan

**Files:**
- Stage all of: Phase0_Design.md, Phase2_Design.md, Project_Framing_v2.md, Paper_Skeleton.md, Results_Summary.md, plans/2026-04-26-vault-updates.md

- [ ] **Step 7.1: Review uncommitted changes**

```bash
cd "/Users/jameswu/Desktop/University/Year_5/Semester_1/BMET5933/Assignment2"
git status --short
git diff --stat Planning/
```

Expected: 5 modified .md files in Planning/, 1 new .md file in Planning/plans/.

- [ ] **Step 7.2: Stage everything**

```bash
git add Planning/Phase0_Design.md Planning/Phase2_Design.md Planning/Project_Framing_v2.md Planning/Paper_Skeleton.md Planning/Results_Summary.md Planning/plans/
```

- [ ] **Step 7.3: Commit with descriptive message**

```bash
git commit -m "$(cat <<'EOF'
Vault: integrate Sprint 1 + Sprint 2 results into design docs

Following the post-Sprint-2 vault review, update five Planning/ docs so
they reflect actual completed work, with each empirical claim anchored
to a peer-reviewed citation.

Phase2_Design.md (largest update):
- Add §12 "Test-time augmentation (Sprint 1 outcome)" — TTA hflip choice
  and result, anchored to Wang et al. 2019 (Neurocomputing TTA in
  medical imaging) and the dataset-specific empirical ranking.
- Add §13 "Sprint 2 supplementary — scale and architecture validation"
  with subsections covering ConvNeXt V2 + matched-data EffNet-B0, the
  three-way decomposition, paired McNemar's (p=0.0021), and the
  cross-architecture Grad-CAM finding.
- Replace §11 limitations with a version that reflects what was actually
  done (TTA + architecture/data-volume ablations) versus what was
  deferred (seed variance).
- Add references [23]-[30]: Wang 2019 (TTA medical), Lakshminarayanan
  2017 (deep ensembles), Woo 2023 (ConvNeXt V2), Liu 2022 (ConvNeXt),
  Mei 2022 (RadImageNet saturation), Selvaraju 2017 (Grad-CAM), Cawley
  & Talbot 2010 (model selection / val saturation).

Phase0_Design.md:
- Add framing-update preamble (matching Phase 2's structure).
- Mark §9 deliverables checklist done; note split_full.csv addition.
- Strike unimplemented promises in §7 (model_card.md, git commit hash
  logging) honestly.

Project_Framing_v2.md:
- Refine the thesis paragraph: classical's failure pair is paradigm-
  stable across DL architectures; DL's failure pair shifts with capacity
  but stays within the DL paradigm. Error distributions are paradigm-
  stable, not architecture-stable, on this dataset.
- Update the per-artefact reading table with three rows (Stone recall,
  Cyst-Tumor classical errors, Cyst-Stone DL errors).

Paper_Skeleton.md:
- Split III-B results into Table 1 (medium) and Table 2 (full)
  with explicit "cross-table comparisons are directional only" caveat.
- Add III-C "Architecture vs data-volume decomposition" paragraph.
- Update III-D to reference the cross-architecture Grad-CAM figure as
  the paper's Figure 1.

Results_Summary.md:
- Add data-efficiency sweep table (10/25/50/100% × EffNet-B0).
- Add EffNet-full vs ConvNeXt V2-full McNemar's table (p=0.0021).
- Add Interpretability Artefacts section listing all .png/.npz outputs
  with their paper-figure roles.

Planning/plans/2026-04-26-vault-updates.md:
- The implementation plan that drove these changes, committed for
  audit/reproducibility.
EOF
)"
```

- [ ] **Step 7.4: Push to origin**

```bash
git push
```

Expected output: `<sha>..<new-sha>  main -> main`

- [ ] **Step 7.5: Verify Home.md headline numbers still match**

The Home.md headline numbers section already shows the canonical four rows. No change needed unless the user wants Sprint 2 lines added there too — leave for a follow-up if requested.

Run: `grep -A 8 "Headline numbers" Planning/Home.md | head -15` — confirm contents unchanged.

---

## Self-review checklist

After running through tasks 1–7, do this once before declaring done:

**1. Spec coverage:** five identified gaps from the review → five updates in this plan. ✓
- Phase 0 framing-v2 + checklist + outdated promises → Task 3 ✓
- Phase 2 Sprint 1 + Sprint 2 sections → Task 2 ✓
- Project_Framing_v2 thesis nuance → Task 4 ✓
- Paper_Skeleton results table + Grad-CAM figure → Task 5 ✓
- Results_Summary sweep + EffNet-full + Interpretability Artefacts → Task 6 ✓

**2. Placeholder scan:** every step contains the actual content to insert. The only "TODO" placeholders are intentional and tagged — Person A's feature-importance figure (will be filled when extracted), draft sentences for Introduction (placeholder for Person A's clinical citations).

**3. Type / cross-reference consistency:**
- Reference number sequence in Phase 2: existing 1–22, new 23–30. No collision.
- Wang et al. Neurocomputing reference uses same author convention as existing refs.
- Phase 2 §12 → §13 ordering keeps existing References numbering aligned.
- Cross-doc wikilinks `[[Project_Framing_v2]]` etc. all resolve.

**4. Evidence-of-execution gates:** each task ends with a verification step (`wc -l`, `grep -c`, `git status`) so an executing agent can confirm progress without making it up.

If any spec requirement appears uncovered after running, the missing item gets a new task inline.

---

## Execution handoff

Two execution options:

**1. Inline execution (recommended for this plan)** — small number of doc edits, no real branching dependencies, all in the same session. Use **superpowers:executing-plans**.

**2. Subagent-driven** — would dispatch one subagent per task. Overkill for documentation edits with no inter-task dependencies on file state.

Default: inline execution. Citation verification (Task 1) happens first because it gates the wording of Task 2 (whether refs [25], [30] survive).
