# Supporting Literature for Paper Findings

**Date compiled:** 2026-04-26
**Purpose:** Anchor every empirical claim in our paper to peer-reviewed evidence, so we can frame results as instances of documented phenomena rather than isolated observations. Verified via Consensus + paper-search MCPs.

Related: [[Project_Framing_v2]] · [[Phase2_Design]] · [[Results_Summary]] · [[Paper_Skeleton]]

> **How to use this file:** when writing a paragraph in the paper, find the relevant finding-section below and pick the cited paper that best supports the specific claim. Do not cite all of them — pick the strongest one for each claim. Reference numbers in this file are LOCAL; the paper's reference list will be reorganised at write-time.

---

## Section 1 — Paradigm comparison (classical handcrafted features vs deep learning) on medical imaging

These citations support the *very existence* of the comparison we are making, and the well-documented complementarity between paradigms in medical imaging.

### [SL-1] Lambin et al. 2017 — *Radiomics: the bridge between medical imaging and personalized medicine* — Nature Reviews Clinical Oncology — **4,313 citations**
*The* canonical radiomics paper. Defines radiomics as "high-throughput mining of quantitative image features from standard-of-care medical imaging." Foundational citation when positioning a paper within the radiomics-vs-deep-learning literature. Use this one in the **Introduction's first sentence on radiomics**.
DOI: 10.1038/nrclinonc.2017.141

### [SL-2] Guiot et al. 2021 — *A review in radiomics: making personalized medicine a reality via routine imaging* — Medicinal Research Reviews — 221 citations
Explicitly frames radiomics as "extracting hand-crafted radiomics features or via deep learning algorithms" — exactly our paradigm dichotomy. Use this when stating that the dichotomy is well-established.

### [SL-3] Zhang et al. 2023 — *Artificial intelligence-driven radiomics study in cancer: the role of feature engineering and modeling* — Military Medical Research — 124 citations
Recent review on radiomics + ML. Useful for the paper's discussion of feature stability, reproducibility, and interpretability — three things our paradigm comparison cares about.

### [SL-4] Anyimadu et al. 2025 — *Deep learning and classical computer vision techniques in medical image analysis* — ArXiv
**Highly relevant.** Systematically evaluates classical-vs-DL on three medical imaging tasks (brain MRI tissue segmentation, lung CT registration, skin lesion classification). They find classical Elastix-based registration **outperforms DL** in lung CT registration (TRE 6.68 mm vs 7.40 mm), while DL ensembles win on skin classification (90–94%). Direct precedent for our finding that paradigm winners flip across tasks.

---

## Section 2 — GLCM / texture features for kidney CT classification

These citations validate that **handcrafted texture features alone capture nearly all the discriminating signal on kidney CT classification** — supporting our dataset-saturation argument.

### [SL-5] Teke et al. 2025 — *Cascading GLCM and T-SNE for detecting tumor on kidney CT images* — European Physical Journal Special Topics — 4 citations
**Critical supporting citation.** Demonstrates that **GLCM features alone** achieve **99.98 % accuracy** on kidney tumor CT classification with Fine KNN. This is the strongest direct precedent for our classical-XGBoost result of 99.79 % accuracy on the same anatomy. *Use this in the paper's Discussion when arguing that handcrafted texture features capture nearly all the kidney-CT signal.*

### [SL-6] Sayed et al. 2025 — *Hybrid Neural Network Framework for Multiclass Classification of Kidney Stones from CT scans* — IEEE INCET — 0 citations (recent)
Builds **CNXGBoost** — a CNN + XGBoost hybrid using GLCM features for kidney-stone classification, achieving 98.71 % accuracy. Direct precedent for Person A's GLCM + XGBoost approach. *Use to anchor the methodological choice in the Classical ML section.*

### [SL-7] Bajpai et al. 2024 — *Intelligent Kidney Abnormality Detection in CT Images Using GLCM and Neural Networks* — ICMNWC
GLCM + KELM achieves 92.54 % accuracy on kidney-stone detection from ultrasound. Slightly off-modality (US, not CT), but useful evidence that GLCM dominates kidney-stone classification across modalities.

---

## Section 3 — Cystic renal mass classification: classical+DL ensembles already published on this exact problem

These citations are **directly precedent** for our soft-vote ensemble approach on the kidney domain.

### [SL-8] Quanhao He et al. 2023 — *Deep learning and radiomic feature-based blending ensemble classifier for malignancy risk prediction in cystic renal lesions* — Insights into Imaging — 24 citations
**Strongest precedent for our paper.** Built a fusion ensemble of **3D-ResNet50 deep features + handcrafted radiomics features** for malignancy risk in cystic renal lesions. Their ensemble achieved AUC 0.934, outperforming the Bosniak-2019 classification. *This is the paper to cite when justifying our soft-vote ensemble of classical + DL on Cyst↔Tumor differentiation.* Use prominently in Discussion.

### [SL-9] Quan He et al. 2022 — *Stratification of malignant renal neoplasms from cystic renal lesions using deep learning and radiomics features based on a stacking ensemble CT machine learning algorithm* — Frontiers in Oncology — 10 citations
Same theme as [SL-8] but uses stacking instead of blending; both deep CNN features and radiomics features fed into an ensemble. Outperforms Bosniak-2019. Use as a second-line citation alongside [SL-8] when establishing the precedent for cross-paradigm ensembles on cystic renal lesions.

### [SL-10] Kang et al. 2024 — *Multiparametric MRI-Based Machine Learning Models for the Characterization of Cystic Renal Masses Compared to the Bosniak Classification* — Academic Radiology — 3 citations
Different modality (MRI not CT), but a valuable cross-modality data point: ML models (LR, RF, SVM) match Bosniak-2019 accuracy (RF AUC 0.907 vs Bosniak 0.893). Supports the broader claim that ML can match or exceed expert clinical classification systems on this problem.

---

## Section 4 — Architecture vs data-volume scaling in medical imaging

These citations support our Sprint 2 finding that **capacity** drives DL gains, not data volume, on saturated medical-imaging datasets.

### [SL-11] Huang et al. 2023 — *STU-Net: Scalable and Transferable Medical Image Segmentation Models Empowered by Large-Scale Supervised Pre-training* — ArXiv — 110 citations
Scaled medical image segmentation models from 14M to 1.4B parameters. **"increasing model size brings a stronger performance gain"**. Direct support for our finding that ConvNeXt V2 (89M) beats EfficientNet-B0 (5M) at matched data. *Use in Discussion when explaining the architecture-vs-data-volume decomposition.*

### [SL-12] Mei et al. 2022 — *RadImageNet* — Radiology: AI (already in Phase 2 references as [28])
Documents the dataset-saturation regime for small medical-imaging datasets. Already cited in [[Phase2_Design]] §13 — keep using it for the saturation argument.

### [SL-13] Piffer et al. 2024 — *Tackling the small data problem in medical image classification with artificial intelligence: a systematic review* — Progress in Biomedical Engineering — 13 citations
Systematic review (147 papers, 77 included) of how medical-imaging studies handle small datasets. 75% use transfer learning, 69% use data augmentation. Useful for framing our methodology choices in the Methods section as standard practice rather than novel.

### [SL-14] Sellergren et al. 2022 — *Simplified Transfer Learning for Chest Radiography Models Using Less Data* — Radiology — 71 citations
Tangential but useful precedent: shows that better pretraining (SupCon) can drastically reduce data requirements (688× fewer labels). Frames our use of ImageNet-22k pretraining as part of an active research direction.

---

## Section 5 — Cross-task: lung-nodule paradigm comparison precedents

These citations establish that the classical-vs-DL paradigm comparison has been studied extensively on related medical-imaging tasks. Useful both for framing and for showing our findings are a general phenomenon, not Islam-dataset-specific.

### [SL-15] Shulong Li et al. 2018 — *Predicting lung nodule malignancies by combining deep convolutional neural network and handcrafted features* — Physics in Medicine & Biology — 64 citations
**Highly relevant.** Combined 29 handcrafted features (intensity, geometric, GLCM texture) with 3D CNN features for lung-nodule malignancy on LIDC/IDRI. Their argument is exactly ours: *"intrinsic CNN features... overcome the disadvantage of the HF that may not fully reflect the unique characteristics of a particular lesion"*. *Use as the canonical precedent for our paradigm-fusion logic in lung-nodule classification.*

### [SL-16] Nishio et al. 2018 — *Computer-aided diagnosis of lung nodule classification* — PLoS ONE — 147 citations
Direct comparison of LBP+SVM vs DCNN on lung-nodule ternary classification. **DCNN wins (68% vs 56% accuracy)**. Useful as a paradigm-winner-flips-across-tasks reference: lung nodules favour DL, kidney stones favour classical. Strengthens our "different signal in different anatomies" argument.

### [SL-17] Fu et al. 2017 — *Automatic detection of lung nodules: false positive reduction using convolution neural networks and handcrafted features* — 35 citations
Combined 88 handcrafted + 864 CNN features for FP reduction in lung-nodule detection. Same paradigm-fusion theme. Secondary citation alongside [SL-15].

---

## Section 6 — Ensemble learning in medical image classification

These citations support our ensemble methodology more broadly.

### [SL-18] Müller et al. 2022 — *An Analysis on Ensemble Learning Optimized Medical Image Classification With Deep CNNs* — IEEE Access — 80 citations
Systematic comparison of augmenting, stacking, and bagging ensembles on four medical-imaging datasets. **Stacking gave +13% F1 max gain.** Notable finding: *"simple statistical pooling functions are equal or often even better than more complex pooling functions"* — supports our **equal-weight w=0.5** soft-vote choice over learned weights.

### [SL-19] Lakshminarayanan, Pritzel, Blundell 2017 — *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles* — NeurIPS (already in Phase 2 references as [24])
Already cited in [[Phase2_Design]] §12 as the reason we deferred multi-seed ensembling — Lakshminarayanan's ensemble framework would be the standard alternative.

---

## Section 7 — Reuse of the Islam et al. dataset by other groups

These citations let us claim the dataset has been independently approached at near-saturation by other groups.

### [SL-20] Bingol et al. 2023 — *Automatic classification of kidney CT images with relief based novel hybrid deep model* — PeerJ Computer Science — 14 citations
**Critical for our dataset-saturation claim.** Uses kidney-CT data with stones, tumors, and cysts (almost certainly the same Islam dataset) and reports **99.37 % accuracy** with their hybrid model. Independent confirmation that this dataset saturates near 99%+ across diverse methods. *Use in the Discussion to support: "multiple groups have approached saturation on this dataset, consistent with our argument that the task is solvable by simple image-level statistics."*

### [SL-21] Mahmud et al. 2023 — *Kidney Cancer Diagnosis and Surgery Selection by Machine Learning from CT Scans Combined with Clinical Metadata* — Cancers — 44 citations
Different dataset (KiTS) and different task (4 RCC subtypes), 85.66% accuracy. Useful as a "what kidney-CT classification looks like on a *non-saturated* dataset" reference — KiTS clinical-grade RCC subtyping is genuinely hard, achieves much lower accuracy than the Islam stones/cysts/tumors task. Supports framing the Islam dataset as a relatively easy benchmark.

### [SL-22] Heller et al. 2019 — *KiTS19 Challenge results* — Medical Image Analysis — 538 citations
Different task (segmentation), but the KiTS19 challenge is the canonical kidney-CT benchmark. Worth citing when contextualising kidney-CT machine-learning literature broadly.

---

## Section 8 — Methodology references already in our docs

These have been verified earlier in the project and are already in [[Phase0_Design]] / [[Phase2_Design]] reference lists. Listed here for completeness:

| Topic | Citation | Already cited in |
|---|---|---|
| McNemar's test for paired classifiers | Dietterich 1998 | Phase 0 [10] |
| Macro-F1 for imbalanced multiclass | Grandini 2020, Sokolova 2009 | Phase 0 [13, 14] |
| TTA in medical imaging | Wang 2019 | Phase 2 [23] |
| Class imbalance in CNNs | Buda 2018, Johnson 2019 | Phase 0 [11, 12] |
| Patient-level leakage | Yagis 2021, Veetil 2024 | Phase 0 [9, 23] |
| Reporting standards | CLAIM 2024, TRIPOD+AI 2024 | Phase 0 [1–3] |
| EfficientNet, ConvNeXt V2, ConvNeXt | Tan & Le 2019, Woo 2023, Liu 2022 | Phase 2 [3, 25, 26] |
| Transfer-learning theory | Yosinski 2014, Howard & Ruder 2018, Kornblith 2019 | Phase 2 [4, 8, 9] |
| Bosniak / cystic renal masses | Weibl 2017 | Paper_Skeleton clinical anchor |
| Model selection / val saturation | Cawley & Talbot 2010 | Phase 2 [30] |

---

## How each finding maps to citations

A quick lookup table for paper-writing time:

| Finding (under framing v2) | Primary supporting citations |
|---|---|
| Disjoint-error pattern between classical and DL | [SL-4] Anyimadu, [SL-15] Shulong Li, [SL-8] Quanhao He |
| Classical wins on Stone via texture features | [SL-5] Teke, [SL-6] Sayed, [SL-7] Bajpai |
| DL wins on Cyst↔Tumor (medium) | [SL-8] Quanhao He, [SL-9] Quan He, [Weibl 2017] |
| Both DL backbones share Cyst↔Stone failure mode (full) | [SL-15] Shulong Li (paradigm-fusion logic), [SL-11] Huang STU-Net |
| Architecture > data volume for EfficientNet-B0 | [SL-11] Huang, [SL-12] Mei (RadImageNet, already cited), [SL-13] Piffer |
| Equal-weight soft-vote ensemble = 100 % | [SL-18] Müller, [SL-8] Quanhao He |
| Val saturation → ensemble tuning fails | [Cawley & Talbot 2010] (already cited) |
| Dataset is artefact-saturable, not "we beat SOTA" | [SL-20] Bingol, [SL-5] Teke, [SL-12] Mei |
| Radiomics-vs-DL framing for Introduction | [SL-1] Lambin, [SL-2] Guiot, [SL-3] Zhang |
| Cross-task evidence (lung) | [SL-15] Shulong Li, [SL-16] Nishio, [SL-17] Fu |

---

## What this changes about how we write the paper

Three things move from "asserted" to "supported":

1. **The radiomics framing** in the Introduction. We now cite Lambin 2017 in the first sentence, Guiot 2021 to define the paradigm dichotomy, and Anyimadu 2025 / Shulong Li 2018 to establish that paradigm comparison is an active sub-literature.

2. **The dataset-saturation argument** in the Discussion. We now have Bingol 2023 (99.37 % on what is almost certainly the same Islam dataset) + Teke 2025 (99.98 % on kidney CT with GLCM alone) as independent corroborations. The framing-v2 claim that "the dataset is texture-solvable" is no longer ours alone.

3. **The cystic-renal-mass ensemble approach** in the Methods/Discussion. He 2023 and He 2022 already built radiomics + DL ensembles on this anatomy and beat the Bosniak classification. We extend their work with the **disjoint-error analysis** and the **paradigm-stable-error-pattern observation**, which neither paper performed.

The paper's framing now reads as: *we contribute a per-method failure-mode analysis to a literature that has been doing radiomics + DL ensembles on kidney CT for several years, and we surface the architecture-stable error structure that prior ensemble papers did not analyse.*

That is a publishable contribution — much stronger than "we tried two methods on a dataset."
