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
- **Position the work within the radiomics-vs-deep-learning literature** [Lambin 2017; Guiot 2021] — not just "we tried two methods", but a *contribution to* an active sub-literature on paradigm comparison
- Brief overview of the dataset: Islam et al. (2022), PACS-sourced, 12,446 images, 4 classes
- Two paradigms compared: classical handcrafted-feature ML vs transfer-learned CNN
- Key contrast with literature: Islam et al. tested Swin Transformer (99.30%), VGG16 (98.2%), ResNet-50 (73.8%); subsequent independent work [Bingol 2023] reports 99.37 % on the same dataset — multiple groups approaching saturation. We add EfficientNet-B0, ConvNeXt V2, and a classical XGBoost baseline, and contribute the **per-method failure-mode analysis** that prior work has not performed.
- Forward pointer to finding: both paradigms near-ceiling, but disjoint error patterns reveal complementary representational strategies that are paradigm-stable, not architecture-stable.

### Draft sentences
> Kidney disease affects [X] million people worldwide, with conditions including cysts, stones, and tumours requiring accurate and timely diagnosis [CITE]. Automated classification from CT imaging could reduce radiologist workload and support earlier intervention [CITE]. Two computational paradigms have been pursued in parallel for medical imaging classification: **radiomics**, the high-throughput extraction of handcrafted quantitative features [Lambin et al. 2017], and **deep learning**, in which features are learned end-to-end from pixels. Both approaches are now standard in the field [Guiot et al. 2021; Zhang et al. 2023]; recent work has begun to combine them in fusion ensembles for cystic renal-mass classification [He et al. 2022; He et al. 2023].
>
> The Islam et al. (2022) kidney CT dataset [CITE] provides a well-curated benchmark of 12,446 images across four classes (cyst, stone, tumour, normal). Prior work on this dataset has focused primarily on deep learning: Islam et al. reported 99.30% accuracy with a Swin Transformer and 98.20% with VGG16; subsequent independent evaluations report comparable saturation (e.g., Bingol et al. 2023, 99.37 %). However, no prior study has compared a handcrafted-feature classical pipeline with a deep model on this dataset, nor analysed the per-method failure-mode complementarity that such a comparison surfaces.
>
> We compare a classical ML pipeline (texture features + XGBoost) with two transfer-learned CNNs (EfficientNet-B0 and ConvNeXt V2 Base), asking not only which achieves higher accuracy but **what each paradigm learns and where each fails**. We find that on the medium-set comparison the two paradigms achieve near-identical scores, with disjoint error sets (both-wrong = 0); that an equal-weight soft-vote ensemble achieves perfect test classification by exploiting this complementarity; and that the classical and DL paradigms' failure modes (Cyst ↔ Tumor for classical, Cyst ↔ Stone for DL) are **paradigm-stable**: they persist when the DL backbone scales from 5 M to 89 M parameters at higher resolution. We argue that on saturated medical-imaging benchmarks, the *direction* of disagreement between paradigms carries more information than the *magnitude* — and that interpretability analysis is essential for distinguishing dataset saturation from method quality.

### Key references to cite
- Lambin et al. 2017 — *Radiomics: the bridge between medical imaging and personalized medicine* (foundational radiomics)
- Guiot et al. 2021 — radiomics review explicitly framing handcrafted-vs-DL as the two approaches
- Islam et al. 2022 (dataset + Swin baseline)
- Bingol et al. 2023 — independent reproduction at 99.37 % on the same dataset
- He et al. 2022 / He et al. 2023 — radiomics + DL ensembles on cystic renal lesions
- Tan & Le 2019 (EfficientNet); Woo et al. 2023 (ConvNeXt V2)
- Haralick et al. 1973 (GLCM features); Ojala et al. 2002 (LBP)

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

### III-B. Main results

*(Numbers pulled from [[Results_Summary]].)*

**Table 1. Primary comparison — medium dataset, n = 934 test images.** Apples-to-apples on a single shared test set.

| Model | Accuracy | Macro-F1 [95 % CI] | Stone F1 | Errors |
|---|---|---|---|---|
| Classical XGBoost | 0.9979 | 0.9976 [0.994, 1.000] | 1.000 | 2 / 934 |
| EfficientNet-B0 + TTA hflip | 0.9861 | 0.9829 [0.973, 0.992] | 0.961 | 13 / 934 |
| Soft-vote ensemble (`w_dl = 0.5`) | **1.0000** | **1.0000** [1.000, 1.000] | 1.000 | **0 / 934** |

McNemar's paired test (Classical vs EfficientNet-B0 + TTA hflip): 15 discordant pairs, *p* = 7.4 × 10⁻³ (Classical > DL).

**Table 2. Supplementary — full dataset, n = 1,867 test images.** Architecture-vs-data-volume decomposition (see [[Phase2_Design]] §13).

| Model | Accuracy | Macro-F1 [95 % CI] | Stone F1 | Errors |
|---|---|---|---|---|
| EfficientNet-B0 (matched-data control) | 0.9877 | 0.9819 [0.975, 0.989] | 0.950 | 23 / 1867 |
| ConvNeXt V2 Base @ 384 | 0.9968 | 0.9953 [0.991, 0.998] | 0.988 | 6 / 1867 |

McNemar's paired test (EfficientNet-B0 vs ConvNeXt V2): 27 discordant pairs (22 only-EffNet wrong; 5 only-ConvNeXt wrong), *p* = 0.0021 (ConvNeXt V2 > EfficientNet-B0). The architecture effect is statistically significant at matched training data.

*Cross-table comparisons are directional only — Tables 1 and 2 use disjoint test sets.*

### III-C. Analysis and discussion

**Disjoint error pattern (key finding)**
Both-wrong = 0. Classical's 2 errors are Cyst↔Tumor confusions (where DL is confident and correct). DL's 13 errors cluster around Stone↔Normal (where classical's texture features are discriminating). This complementarity is evidence that the paradigms exploit different visual signals, not that one is simply better than the other.

**Why classical outperforms DL on Stone**
Stone is a strongly-textured class — bright calcified deposits with high-contrast GLCM signatures. GLCM, LBP, and Gabor features encode this signal directly. EfficientNet-B0 must learn an equivalent representation from pixels, which requires more data and more training signal than a 103-image Stone test set provides. The DL Stone recall gap (0.961 vs 1.000) is the measured cost of implicit vs explicit texture encoding.

**Why DL outperforms classical on Cyst↔Tumor**
Cysts and tumors are both rounded soft-tissue masses with similar first-order texture statistics. Classical features cannot distinguish them well — both classical errors are in this pair. EfficientNet-B0's learned spatial filters differentiate them via morphological cues (wall thickness, internal structure) that are not captured by rotation-invariant texture statistics.

**Ensemble interpretation**
The equal-weight ensemble achieves 0 errors not because the individual models are perfect, but because their errors are disjoint. This is evidence of complementary feature spaces. The val-tuned ensemble collapsed to classical-alone because classical achieves perfect val F1, making the val set uninformative for weight selection — a documented failure mode of disagreement-based ensemble tuning under model-selection bias [Cawley & Talbot 2010].

**Architecture vs data-volume decomposition (Table 2)**
Comparing EfficientNet-B0 across training-set sizes (1.29 % error rate on medium + TTA, 1.23 % on full) shows that doubling the training set produces no measurable improvement in this architecture — the 5.3 M-parameter capacity is already saturated. Comparing the two architectures on identical training data (EfficientNet-B0 1.23 % vs ConvNeXt V2 0.32 %) shows a 74 % error reduction with statistical significance (McNemar's *p* = 0.0021). The DL gains observed in Sprint 2 are therefore attributable to **architecture, not data volume**. This separation is consistent with the saturated-task regime in Mei et al. [RadImageNet 2022]: when a small medical-imaging dataset is approaching its solvability ceiling, capacity is the binding constraint on additional gains. Both DL architectures share the same dominant Cyst ↔ Stone failure mode (74 % and 83 % of total errors respectively), confirming that the DL ↔ classical gap is a paradigm-level distinction, not an architectural one.

**Comparison with literature**
Islam et al.'s Swin Transformer achieves 99.30% accuracy on their balanced subset. Our ConvNeXt V2 Base (supplementary, full dataset) achieves 99.53% macro-F1, consistent with but not directly comparable to their result (different splits, different class balance). Independent work by Bingol et al. (2023) on this same dataset reports 99.37 % accuracy with a hybrid CNN model. Teke et al. (2025) achieve 99.98 % on a kidney CT classification task using **GLCM features alone with a Fine-KNN classifier** — direct corroboration that handcrafted texture features capture nearly all the discriminating signal on kidney CT classification. Together, these results support the interpretation that this benchmark is approaching a performance ceiling regardless of architectural sophistication; the remaining variance is in which images each paradigm fails on, not whether it fails.

**Cross-task evidence — paradigm winners flip across anatomies**
Whereas our results show classical ML edging out DL on kidney CT (medium set), Nishio et al. (2018) report the opposite finding for lung-nodule classification: a deep CNN (68% accuracy) outperformed an LBP+SVM classical pipeline (56%). Shulong Li et al. (2018) similarly found that combining handcrafted (GLCM, intensity, geometric) and CNN features in a fusion classifier beats either alone for lung-nodule malignancy prediction on LIDC/IDRI, with each component's value reflecting its complementary representational scope. The pattern in our data — paradigm-stable error structure with classical winning on texture-dominant minority classes — appears consistent with this broader medical-imaging-paradigm-comparison literature: which paradigm wins is dataset-dependent, but their errors are reliably *complementary*, supporting fusion-ensemble approaches as in He et al. (2022, 2023).

**Compute and practicality**
*(TODO: add training time, inference time, model size table)*

| Model | Params | Train time | Inference (per image) |
|---|---|---|---|
| XGBoost | — | ~X min (CPU) | ~X ms |
| EfficientNet-B0 | 5.3M | ~X min (Colab A100) | ~X ms |

Classical ML's practical advantage: no GPU required, near-instant training, interpretable outputs.

### III-D. Interpretability side-by-side — the central paper figure

**Figure 1. Cross-architecture Grad-CAM** (`Results/gradcam/cross_architecture.png`). Six paired examples drawn automatically from the EfficientNet-B0 / ConvNeXt V2 paired-disagreement set on the full test split, visualising last-conv-stage attention via Grad-CAM [Selvaraju et al. 2017]. For each row: original CT slice (left), EfficientNet-B0 @ 224 attention (centre), ConvNeXt V2 @ 384 attention (right).

**Reading the figure.** EfficientNet-B0's attention is consistently *more dispersed* than ConvNeXt V2's, frequently extending outside the kidney silhouette into body wall and bowel. On the three Cyst → Stone misclassifications unique to EfficientNet-B0, the smaller network's attention peaks off-organ; ConvNeXt V2 on the same images correctly fixates on the kidney lesion and outputs Cyst. This directly visualises the architecture effect quantified in Table 2: the larger network is not just more accurate, it is **looking at different things** — specifically, kidney tissue rather than peripheral context. The qualitative observation aligns with the quantitative finding (McNemar's *p* = 0.0021): ConvNeXt V2's gains are anchored in better attentional discipline.

**Figure 2. Classical feature importance** *(TODO — Person A to extract from the trained XGBoost classifier)*. Top-N feature importances by class. Hypothesis under [[Project_Framing_v2]]: 2–3 features dominate, and they are texture features (GLCM Haralick or LBP), confirming the dataset-saturation argument that handcrafted texture features alone capture nearly all the discriminating signal.

**Joint reading.** The Grad-CAM attention maps and the feature-importance ranking together constitute the paper's central interpretability evidence: each paradigm exploits a different aspect of the visual signal (DL: spatial structure on the lesion itself; classical: texture/intensity statistics over the whole image), and the disjoint-error pattern in Table 1 is the measurable consequence of this representational difference. The clinical relevance is that Cyst ↔ Tumor differentiation — the failure mode classical cannot solve — is exactly the hardest pair under the radiology-standard Bosniak classification framework [Weibl et al. 2017], lending external validity to our paradigm-comparison story.

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

Full master list — pull subset for the actual paper. Cross-reference with [[Supporting_Literature]] for context on what each citation supports. See Phase 0 / Phase 2 design docs for already-verified entries.

**Foundational / framing**
1. Lambin et al. 2017 — *Radiomics: the bridge between medical imaging and personalized medicine*
2. Guiot et al. 2021 — radiomics review (handcrafted vs DL framing)
3. Anyimadu et al. 2025 — classical-vs-DL paradigm comparison on three medical imaging tasks
4. Islam et al. 2022 — dataset + Swin baseline (CITE for the dataset)
5. Bingol et al. 2023 — independent 99.37 % on same dataset (saturation evidence)

**Method — Classical ML (Person A's pipeline)**
6. Haralick et al. 1973 — GLCM features
7. Ojala et al. 2002 — LBP
8. Pizer et al. 1987 — CLAHE
9. Chen & Guestrin 2016 — XGBoost
10. Teke et al. 2025 — GLCM + KNN at 99.98 % on kidney CT (precedent for classical dominance on this anatomy)
11. Sayed et al. 2025 — CNN+XGBoost hybrid for kidney stones (precedent for the architectural choice)

**Method — Deep Learning (Person B's pipeline)**
12. Tan & Le 2019 — EfficientNet
13. Woo et al. 2023 — ConvNeXt V2
14. Liu et al. 2022 — original ConvNeXt
15. Kornblith et al. 2019 — better-ImageNet → better-transfer
16. Yosinski et al. 2014 — feature transferability
17. Howard & Ruder 2018 — ULMFiT / gradual unfreezing
18. Loshchilov & Hutter 2019 — AdamW
19. Raghu et al. 2019 — ImageNet transfer to medical imaging
20. Mei et al. 2022 — RadImageNet (saturation in small medical datasets)
21. Matsoukas et al. 2021 — CNN vs ViT on medical imaging

**Interpretability + ensemble**
22. Selvaraju et al. 2017 — Grad-CAM
23. Wang et al. 2019 — TTA in medical imaging
24. Müller et al. 2022 — ensemble learning analysis for medical image classification
25. Lakshminarayanan et al. 2017 — deep ensembles
26. He et al. 2023 — DL + radiomics blending ensemble for cystic renal lesions
27. He et al. 2022 — DL + radiomics stacking ensemble for cystic renal lesions

**Statistical / reporting**
28. Dietterich 1998 — McNemar's for paired classifiers
29. Grandini et al. 2020 — multiclass metrics (macro-F1)
30. Sokolova & Lapalme 2009 — performance measures for classification
31. Cawley & Talbot 2010 — overfitting in model selection (val saturation)
32. Mongan et al. 2020 + Tejani et al. 2024 — CLAIM / CLAIM 2024
33. Collins et al. 2024 — TRIPOD+AI

**Cross-task evidence**
34. Shulong Li et al. 2018 — lung-nodule malignancy: CNN + handcrafted fusion
35. Nishio et al. 2018 — lung-nodule classification: CNN > LBP+SVM (paradigm flip across tasks)

**Limitations / patient-level leakage**
36. Yagis et al. 2021 — slice-level leakage in 2D CNN brain MRI
37. Veetil et al. 2024 — replication of slice-level-leakage effect

**Clinical context**
38. Weibl et al. 2017 — Bosniak / cystic renal mass differentiation

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
