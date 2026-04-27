# Paper Skeleton — IEEE ISBI Format

BMET5933 Assignment 2 · 6 pages · Due 15 May 2026
Numbers → [[Results_Summary]] · Framing → [[Project_Framing_v2]]

> **Writing directive**: foreground failure-mode analysis and interpretability. Background raw scores. The 100% ensemble result is not the headline — the disjoint-error pattern is. See [[Project_Framing_v2]] for the full argument.

---

## Author Names

[Surname A], [First A] and Wu, Jimmy
*(alphabetical by surname — assignment requirement)*

---

## I. Introduction (~0.5–1 page)

### What to cover
- Clinical motivation: kidney disease burden, four conditions (cyst, stone, tumor, normal), value of automated CT classification
- Brief overview of the dataset: Islam et al. (2022), PACS-sourced, 12,446 images, 4 classes
- Two paradigms compared: classical handcrafted-feature ML vs transfer-learned CNN
- Key contrast with literature: Islam et al. tested Swin Transformer (99.30%), VGG16 (98.2%), ResNet-50 (73.8%) — we add EfficientNet-B0 (not in their comparison) and a classical XGBoost baseline
- Forward pointer to finding: both paradigms near-ceiling, but disjoint error patterns reveal complementary representational strategies

### Draft sentences
> Kidney disease affects [X] million people worldwide, with conditions including cysts, stones, and tumours requiring accurate and timely diagnosis [CITE]. Automated classification from CT imaging could reduce radiologist workload and support earlier intervention [CITE]. The Islam et al. (2022) dataset [CITE] provides a well-curated benchmark of 12,446 kidney CT images across four classes, enabling reproducible algorithm comparison.
>
> Prior work on this dataset has focused primarily on deep learning: Islam et al. reported 99.30% accuracy with a Swin Transformer and 98.2% with VGG16, but did not evaluate classical handcrafted-feature methods or examine failure-mode complementarity between paradigms. We compare a classical ML pipeline (texture features + XGBoost) with a transfer-learned EfficientNet-B0, asking not only which achieves higher accuracy but what each paradigm learns and where each fails.

### Key references to cite
- Islam et al. 2022 (dataset + Swin baseline)
- LeCun et al. (deep learning overview)
- Litjens et al. 2017 (medical image analysis survey)
- Haralick et al. 1973 (GLCM features)
- Tan & Le 2019 (EfficientNet)

---

## II. Technical Contributions (~1–1.5 pages per person = ~2–3 pages total)

### II-A. Shared Infrastructure *(brief, ~0.25 pages)*

**Split**: Stratified 70/15/15 train/val/test, seed=42, saved as `split.csv`. Both pipelines load from this CSV — neither re-splits. Known limitation: no patient IDs available, so adjacent-slice leakage cannot be prevented (acknowledge, don't fix).

**Preprocessing**: Images resized to 256×256 grayscale via bilinear interpolation. Shared entry point in `shared/preprocessing.py`; each pipeline extends with classifier-appropriate transforms.

**Evaluation harness**: Shared `evaluate.py` — macro-F1 (primary), accuracy, per-class P/R/F1, confusion matrix, ROC-AUC (OvR), bootstrap 95% CIs (1000 resamples), McNemar's paired test. Macro-F1 chosen over accuracy because the dataset is class-imbalanced (Normal 40.8%, Stone 11.1%).

**Class imbalance**: Handled via inverse-frequency class weights in both pipelines. Natural imbalance preserved (matches clinical reality); downsampling rejected because it discards training signal.

---

### II-B. Classical ML — [Person A's name] (~1–1.5 pages)

> *Person A writes this section. Notes below to guide content.*

**Feature extraction** (108-dimensional vector per image):
- First-order intensity statistics: mean, std, skew, kurtosis, entropy, percentiles (10, 25, 50, 75, 90) → 10 features
- GLCM Haralick texture: contrast, correlation, energy, homogeneity, dissimilarity, ASM at 4 angles × 2 distances, rotation-averaged → 12 features. Cite Haralick et al. 1973.
- Multi-scale LBP histograms: 3 scales → 54 features. Cite Ojala et al. 2002.
- Gabor filter bank: 4 frequencies × 4 orientations, mean + std of response magnitude → 32 features. Cite Arivazhagan & Ganesan 2003.
- HOG excluded: pilot experiments showed it alone achieved near-perfect separation due to class-distinctive shapes, dominating the feature space and leaving no signal for texture features. (Worth mentioning — reveals task easiness.)
- CLAHE preprocessing before feature extraction (Pizer et al. 1987) — enhances local contrast for texture feature reliability.

**Dimensionality reduction**: StandardScaler → PCA (50 components, capacity-constrained to reduce slice-memorisation risk).

**Classifier selection**: Grid search over SVM (RBF), Random Forest, XGBoost. Winner selected by val macro-F1. XGBoost won. Cite Chen & Guestrin 2016.

**Retraining**: Winner retrained on train+val combined before final test evaluation.

**Interpretability**: Feature importance from XGBoost / permutation importance — which of the 108 features carry the signal? (TODO: extract and report top features.)

---

### II-B. Deep Learning — Wu, Jimmy (~1–1.5 pages)

**Architecture**: EfficientNet-B0 (Tan & Le, 2019), ImageNet-1K pretrained, 5.3M parameters. Head replaced: `Dropout(0.3) → Linear(1280, 4)`. Chosen over ResNet-50 (Islam et al. report 73.8% on this dataset), VGG16 (138M params, older), and Swin Transformer (requires prohibitive compute for proper fine-tuning; Matsoukas et al. show CNN/ViT performance converges under equivalent training regimes). Kornblith et al. (2019) demonstrate r=0.99 correlation between ImageNet accuracy and transfer performance, justifying the strong pretrained starting point.

**Two-stage training**:
- Stage 1: freeze backbone, train head only. Adam, LR=1e-3, 5 epochs. Rationale: stabilise randomly initialised head before gradients propagate into pretrained features (Howard & Ruder 2018; Yosinski et al. 2014).
- Stage 2: unfreeze last 2 backbone stages, fine-tune. AdamW (Loshchilov & Hutter 2019), LR=1e-5, weight decay=1e-4, patience=5 early stopping on val macro-F1. LR drop of 2 orders of magnitude preserves pretrained representations while allowing domain adaptation (Kornblith et al. 2019).

**Input**: 224×224 RGB (grayscale channel repeated ×3), ImageNet mean/std normalisation.

**Augmentation**: horizontal flip, rotation ±15°, zoom 0.1, brightness/contrast ±0.1. No vertical flip — CT anatomy is not top-bottom symmetric. Matches Islam et al. augmentation for comparability.

**Test-Time Augmentation (TTA)**: At inference, predictions averaged over 2 views (original + horizontal flip). Motivated by error analysis: 16/19 baseline errors had the true class at rank 2 with mean confidence 0.56, making them borderline — averaging softmax over augmented views systematically tips these toward the correct class. Rotation TTA was tested and rejected: rotation-averaged softmax pulled Cyst predictions toward Tumor (7 of 10 broken cases in the full-view ablation), likely because Cyst and Tumor share rounded morphology that rotation makes ambiguous.

**Interpretability**: Grad-CAM on last convolutional block (Selvaraju et al. 2017), 2–3 examples per class, overlay on original CT. Comparison across EfficientNet-B0 and ConvNeXtV2 Base (supplementary).

---

## III. Evaluation (~1.5–2 pages)

### III-A. Experimental protocol

- Test set: 934 images (medium dataset), never seen during training or hyperparameter selection
- Primary metric: macro-F1 (accounts for class imbalance)
- Secondary: accuracy, per-class F1, ROC-AUC (OvR), bootstrap 95% CIs, McNemar's paired test
- Comparison with Islam et al. (2022) is directional — different split, different class balance

### III-B. Main results table

*(Pull from [[Results_Summary]])*

| Model | Accuracy | Macro-F1 [95% CI] | Stone F1 | Errors |
|---|---|---|---|---|
| Classical XGBoost | 0.9979 | 0.9976 [0.994, 1.000] | 1.000 | 2 / 934 |
| EfficientNet-B0 + TTA | 0.9861 | 0.9829 [0.973, 0.992] | 0.961 | 13 / 934 |
| Ensemble (w=0.5) | **1.0000** | **1.0000** [1.000, 1.000] | 1.000 | **0 / 934** |

McNemar's test (classical vs DL + TTA): *report statistic and p-value here.*

### III-C. Analysis and discussion

**Disjoint error pattern (key finding)**
Both-wrong = 0. Classical's 2 errors are Cyst↔Tumor confusions (where DL is confident and correct). DL's 13 errors cluster around Stone↔Normal (where classical's texture features are discriminating). This complementarity is evidence that the paradigms exploit different visual signals, not that one is simply better than the other.

**Why classical outperforms DL on Stone**
Stone is a strongly-textured class — bright calcified deposits with high-contrast GLCM signatures. GLCM, LBP, and Gabor features encode this signal directly. EfficientNet-B0 must learn an equivalent representation from pixels, which requires more data and more training signal than a 103-image Stone test set provides. The DL Stone recall gap (0.961 vs 1.000) is the measured cost of implicit vs explicit texture encoding.

**Why DL outperforms classical on Cyst↔Tumor**
Cysts and tumors are both rounded soft-tissue masses with similar first-order texture statistics. Classical features cannot distinguish them well — both classical errors are in this pair. EfficientNet-B0's learned spatial filters differentiate them via morphological cues (wall thickness, internal structure) that are not captured by rotation-invariant texture statistics.

**Ensemble interpretation**
The equal-weight ensemble achieves 0 errors not because the individual models are perfect, but because their errors are disjoint. This is evidence of complementary feature spaces. The val-tuned ensemble collapsed to classical-alone because classical achieves perfect val F1, making the val set uninformative for weight selection — a documented failure mode of disagreement-based ensemble tuning.

**Comparison with literature**
Islam et al.'s Swin Transformer achieves 99.30% accuracy on their balanced subset. Our ConvNeXtV2 Base (supplementary, full dataset) achieves 99.53% macro-F1, consistent with but not directly comparable to their result (different splits, different class balance). Both results, alongside our classical ML performance, support the interpretation that this dataset is approaching a performance ceiling regardless of architecture — the remaining variance is in which images each paradigm fails on, not whether it fails.

**Compute and practicality**
*(TODO: add training time, inference time, model size table)*

| Model | Params | Train time | Inference (per image) |
|---|---|---|---|
| XGBoost | — | ~X min (CPU) | ~X ms |
| EfficientNet-B0 | 5.3M | ~X min (Colab A100) | ~X ms |

Classical ML's practical advantage: no GPU required, near-instant training, interpretable outputs.

### III-D. Interpretability side-by-side

*(TODO: insert Grad-CAM panel + feature importance figure here)*

- Figure 1: Grad-CAM attention maps for EfficientNet-B0, 2 examples per class
- Figure 2: Top-N XGBoost feature importances by class
- Discussion: does DL attend to kidney tissue or to image margins/scanner artefacts?

---

## IV. Conclusion (~0.5 pages)

### What to cover
- Restate the central finding: both paradigms achieve >97% macro-F1, but fail on different images and for different reasons
- Classical advantage on Stone (texture-driven) vs DL advantage on Cyst↔Tumor (spatial structure)
- Disjoint-error ensemble reaches 100% — interpret as dataset saturation + complementary feature spaces, not as a methodological triumph
- Limitations: adjacent-slice leakage inflates all results; single-seed evaluation; medium dataset only for primary comparison
- Future work: patient-stratified split, multi-seed evaluation, external validation dataset, kidney-mask ablation to test whether classical is using anatomy or image margins

### Draft sentences
> We compared a classical handcrafted-feature pipeline (XGBoost + texture features) with a transfer-learned EfficientNet-B0 on the Islam et al. kidney CT dataset. Both paradigms achieved high macro-F1 (0.9976 and 0.9829 respectively), and their equal-weight soft-vote ensemble reached perfect test classification. The scientific content of this result lies not in the scores but in their failure modes: classical features discriminate Stone perfectly via explicit texture encoding, while learned spatial filters give EfficientNet-B0 an advantage on the morphologically similar Cyst↔Tumor pair. The disjoint error pattern provides empirical evidence that the two paradigms exploit complementary aspects of the visual signal.
>
> These results should be interpreted with caution. The absence of patient identifiers means adjacent CT slices likely appear in both training and test sets, inflating all reported metrics. Future work should incorporate patient-stratified splits and external validation to assess real-world generalisability.

---

## References (IEEE format — additional page, not counted in 6)

Key citations needed:
1. Islam et al. 2022 — dataset + Swin baseline
2. Tan & Le 2019 — EfficientNet
3. Kornblith et al. 2019 — transfer learning
4. Yosinski et al. 2014 — feature transferability
5. Howard & Ruder 2018 — ULMFiT / gradual unfreezing
6. Loshchilov & Hutter 2019 — AdamW
7. Haralick et al. 1973 — GLCM features
8. Ojala et al. 2002 — LBP
9. Selvaraju et al. 2017 — Grad-CAM
10. Chen & Guestrin 2016 — XGBoost
11. Litjens et al. 2017 — medical image analysis survey
12. Matsoukas et al. 2021 — CNN vs ViT on medical imaging
13. Raghu et al. 2019 — ImageNet transfer to medical imaging
14. Pizer et al. 1987 — CLAHE

---

## Page budget (6 pages total)

| Section | Target |
|---|---|
| I. Introduction | 0.5–1 page |
| II-A. Shared infrastructure | 0.25 page |
| II-B. Classical ML (Person A) | 1–1.5 pages |
| II-C. Deep Learning (Person B) | 1–1.5 pages |
| III. Evaluation | 1.5–2 pages |
| IV. Conclusion | 0.5 page |
| **Total** | **~6 pages** |
| References + Acknowledgements | +1 page (not counted) |
