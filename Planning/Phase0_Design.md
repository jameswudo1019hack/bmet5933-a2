# Phase 0 — Shared Infrastructure Design

**BMET 5933 Assignment 2 — Kidney CT Classification**
Status: draft v1, 2026-04-22

---

## Purpose of this document

Phase 0 defines the *shared infrastructure* both classifiers (Person A: classical ML; Person B: deep learning transfer learning) will use. Every design decision below is justified with a reference from the literature, because the unit coordinator has stated that marks are allocated on **justification of choices and their impact on results**, not raw classifier accuracy. The outputs of Phase 0 — a fixed split, a shared preprocessing entry point, and a common evaluation harness — are what make the classical-vs-deep comparison a defensible scientific comparison rather than two separate experiments.

The paper this document supports will also claim conformance with the CLAIM 2024 reporting checklist for AI in medical imaging [1, 2] and the TRIPOD+AI 2024 statement for prediction-model studies [3]. Both are cited downstream as the sources for specific reporting decisions.

---

## 1. Dataset size selection

### Decision
Use the **medium** subset (6,221 images; Normal 2538 / Cyst 1854 / Tumor 1141 / Stone 688) for the main experiments. Optionally repeat on the **full** set (12,446) only if the medium-set pipeline is stable and time permits.

### Rationale
- The unit coordinator has explicitly required the medium set as a minimum for any CNN-based submission.
- Cheplygina & Varoquaux [4] identify **dataset-size over-optimism** as one of the top failure modes in medical-imaging ML: models trained on small, single-centre sets routinely over-fit and fail on external data. 6.2k images is comfortably above the regime in which overfitting dominates for transfer-learned CNNs, while staying computationally tractable on Colab Pro.
- The medium set preserves the native class distribution (roughly 41 / 30 / 18 / 11 %). Preserving imbalance avoids the "balanced subset" design used by the reference paper [5], which discards >7k images and obscures clinically realistic behaviour (§3).

### Alternative considered
The small set (3,110) would allow faster iteration but Stone would contain only 344 images — the 70/15/15 split (§2) would leave ~52 test examples for Stone, giving very wide per-class CIs. The full set (12,446) is attractive but raises the wall-clock cost of the data-efficiency sweep (Phase 3) well beyond what a 3-week timeline can absorb, especially with two classifiers.

---

## 2. Train / Val / Test split strategy

### Decision
- **Ratios**: 70 % train / 15 % validation / 15 % test.
- **Stratification**: stratified by class label to preserve the native class distribution in every split.
- **Seed**: `random_state = 42`, fixed across both classifiers.
- **Artefact**: the split is materialised once as `split.csv` (columns: `filename`, `class`, `split`) and both pipelines load from that CSV. Neither pipeline re-splits.
- **Test set is held out**: touched exactly once per model, at the very end of the study. All hyperparameter tuning uses the training set with stratified 5-fold cross-validation on the training fold only.

### Rationale
- **Why 70/15/15 rather than 80/20 + in-train validation (as in Islam et al.)**: 15 % test (~930 images) gives Stone ~100 test samples, which is the floor for per-class 95 % bootstrap CIs of reasonable width [6]. Keeping validation as an explicit third partition — rather than carving it out of the training fold on each CV iteration — means both models can use an identical validation signal for hyperparameter selection and early stopping, eliminating a confound in the comparison.
- **Why stratified**: with a ~7:1 Normal-to-Stone ratio, random splitting can (and in small runs, does) produce splits with badly skewed per-class test counts. Stratified splitting is the standard recommendation for imbalanced learning [7, 8].
- **Why a fixed seed**: reproducibility is a CLAIM 2024 item (checklist items on data partitioning) [2] and a core TRIPOD+AI recommendation [3]. Seed locking is the cheapest possible action to satisfy it.
- **Why the test set is touched once**: repeatedly evaluating on the test set during development causes "test-set leakage" — a well-documented source of over-optimistic results in ML pipelines [4, 9].

### Alternative considered and why rejected
- **K-fold cross-validation on the full dataset** (no held-out test). Stronger point estimate but prevents a clean final benchmarking of the two models against one another with a single, paired McNemar's test [10]. CV is still used, but only inside the training fold for hyperparameter search.
- **Patient-level splitting** (standard best practice for medical imaging [9]) is **not possible** with this dataset — Islam et al.'s Kaggle release contains no patient identifiers. This is an acknowledged limitation (§8).

---

## 3. Class imbalance handling

### Decision
- **Keep the natural class distribution** in all splits. Do not downsample or upsample in Phase 0.
- Each classifier handles imbalance internally, with matched strategies:
  - **Person A (classical ML)**: `class_weight='balanced'` in the final classifier (SVM / RF / GBM), and stratified CV for tuning.
  - **Person B (deep learning)**: inverse-frequency class weights in the cross-entropy loss, and weighted random sampling in the training DataLoader (one or both — both pipelines to be prototyped and the better-performing one reported).
- **Primary reporting metric is macro-F1**, not accuracy (§5).

### Rationale
- **Why not rebalance the dataset itself**: Islam et al. [5] downsampled to 1,300 per class, discarding >7k images. This has two problems:
  1. It destroys information (especially in the majority Normal class, where retained images cover far less anatomical variation).
  2. It reports metrics on an artificial distribution that does not match deployment reality — in a hospital PACS, Stone is still the rarest finding. Varoquaux & Cheplygina [4] specifically flag this as a "population-shift" failure mode.
- **Why class weights + macro-F1 instead**: Buda, Maki & Mazurowski [11] systematically compared strategies for CNN class imbalance and found oversampling generally dominant, with cost-sensitive learning (class weights) a close and sometimes indistinguishable second. Johnson & Khoshgoftaar [12] reach the same broad conclusion for deep learning more generally. Both strategies preserve the full training set.
- **Why report both micro- and macro-averaged metrics**: Grandini, Bagli & Visani [13] and Sokolova & Lapalme [14] show that in imbalanced multi-class problems, accuracy and micro-F1 are dominated by the majority class and can mask catastrophic failure on the minority class. Macro-F1 gives equal weight per class and is therefore the defensible primary metric.

### Alternative considered
**SMOTE or similar synthetic oversampling on the feature space for Person A**: defensible but introduces a choice (neighborhood size `k`) that is hard to justify without a second literature search. We will mention SMOTE in the paper's discussion as an option but not use it in the main pipeline, because class-weighting gives the same cost-sensitive signal without synthesising examples.

---

## 4. Shared preprocessing entry point

### Decision
A single function `load_image(filepath) -> np.ndarray` used by both pipelines:
1. Load image via `PIL.Image.open`.
2. Convert to grayscale (images are already grayscale CT slices; this is defensive normalisation against any incidental RGB encoding).
3. Resize to **256 × 256** using bilinear interpolation, preserving aspect ratio by centre-cropping to square first.
4. Return as `np.uint8` array.

Everything beyond this is classifier-specific:
- **Person A** reads at 256×256 grayscale and extracts features on that.
- **Person B** reads, then applies `Resize(224×224)` + `ToTensor` + channel replication (grey → 3-channel) + **ImageNet mean/std normalisation** `(μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225])`, matching the pretraining distribution.

### Rationale
- **Why 256×256 at the shared stage**: it is larger than any downstream classifier will use (so no information is destroyed early), and `PIL` resize at 256 is essentially free computationally. Downstream resizes (to 224×224 for the CNN) follow the ImageNet transfer-learning convention that Rajpurkar et al. [15] established for medical-imaging CNNs and that has since become near-universal.
- **Why bilinear, not nearest neighbour**: nearest is appropriate for segmentation masks; bilinear is appropriate for intensity images and is the default across `torchvision`, `PIL`, and `scikit-image` for good reason — it avoids the pixelation artefacts that would change texture statistics (which Person A's GLCM features specifically depend on).
- **Why ImageNet normalisation for Person B**: transfer learning from ImageNet weights requires matching the pretraining input distribution; deviating from it has been shown to reduce the effectiveness of frozen-backbone features [15, 16]. This is a reporting item in CLAIM 2024 [2].
- **Why not apply histogram equalisation at the shared stage**: equalisation alters texture statistics — good for CNNs but may or may not help handcrafted-feature pipelines, and the literature is mixed. We leave it as a per-classifier choice rather than a shared default.

### Alternative considered
**Reading at 224×224 directly** to save memory. Rejected because Person A's GLCM features benefit from at-least-256 resolution (the paper [5] used 168×168 for their transformers; classical features typically want more).

---

## 5. Evaluation metrics

### Decision
For every reported experiment, both classifiers report:

| Metric | Averaging | Role |
|---|---|---|
| **Macro-F1** | macro (equal weight per class) | **Primary** |
| Accuracy | — | Secondary, for direct comparison to Islam et al. |
| Per-class precision, recall, F1 | — | Required — failure-mode analysis |
| Weighted F1 | support-weighted | Reported; sensitive to imbalance |
| ROC-AUC | one-vs-rest, macro-averaged | Threshold-free comparison |
| Confusion matrix | — | Required figure |
| 95 % bootstrap CI on macro-F1 | 1000 resamples of the test set | Uncertainty quantification |

### Rationale
- **Macro-F1 as primary**: Grandini et al. [13] argue that macro-F1 is the appropriate default for imbalanced multi-class problems because it treats the minority class as first-class — exactly what we want clinically, since Stone and Tumor misses are more costly than Normal misses. Sokolova & Lapalme [14] make the same argument more formally.
- **Accuracy reported as secondary, not primary**: Accuracy is dominated by the Normal class (41 % of medium). A model that classified everything as Normal would score ~41 % accuracy — unacceptable clinically, unacceptable as a primary metric, but the Islam et al. comparison requires us to report it.
- **Per-class metrics are mandatory, not optional**: the CLAIM 2024 checklist [2] explicitly requires reporting per-class performance for classification tasks.
- **ROC-AUC one-vs-rest, macro-averaged**: Fawcett [17] establishes this as the default extension of ROC analysis to multi-class. Hand & Till's approach [18] is an alternative but is less commonly reported.
- **Bootstrap CIs on the test set**: Efron & Tibshirani [19] established non-parametric bootstrapping as the default for small-sample CIs. CLAIM 2024 [2] and TRIPOD+AI [3] both require uncertainty quantification around point estimates. 1,000 resamples is the standard "budget" [6, 19].

---

## 6. Statistical testing for model comparison

### Decision
The two final classifiers are compared using **McNemar's test** [10] on paired predictions from the held-out test set. A significance level of α = 0.05 is reported but not used as a gatekeeper — the effect size (difference in macro-F1, with its bootstrap CI) is what the discussion leads with.

### Rationale
- Dietterich [10] provides the definitive guidance on comparing classifiers on a shared test set. He specifically recommends McNemar's for the case where two algorithms are compared on a single held-out test set (as opposed to cross-validated resampling, which would call for the 5×2cv F-test). Our design matches the McNemar case exactly.
- McNemar's test compares *paired* disagreements (cases where classifier A is right and B is wrong vs vice versa), which is more statistically powerful than comparing marginal accuracies [10, 20].

### Alternative considered
**5×2 CV paired F-test** [10]: more powerful but requires restructuring the experimental design around nested CV, which we rejected in §2 to keep a clean held-out test set.

---

## 7. Reproducibility and reporting standards

### Decision
The project commits to the following reporting artefacts, produced automatically by the evaluation harness:
1. A JSON config for every experiment (seed, data path, model hyperparameters, git commit hash).
2. A JSON results file per experiment (all metrics in §5 + wall-clock timings).
3. A single `split.csv` shared across both pipelines.
4. Per-model `model_card.md` (one paragraph summary: inputs, outputs, training data, limitations, intended use).
5. The final paper will include a CLAIM 2024 [1, 2] adherence checklist in supplementary material.

### Rationale
- CLAIM 2024 [2] and TRIPOD+AI [3] are the two most widely adopted reporting frameworks for medical-imaging AI. Conformance is cheap to declare if the infrastructure above is in place from Phase 0. Retrofitting them at paper-writing time is painful and usually incomplete.
- Pineau et al. [21] showed that 40%+ of ML papers at top venues cannot be reproduced from their reported details. Logging configs + seeds + metrics as structured artefacts is the baseline mitigation.

---

## 8. Known limitations documented in Phase 0

These will be repeated in the paper's Limitations section and must not be forgotten:

1. **No patient identifiers**: Islam et al.'s Kaggle release does not include patient IDs. Because CT studies typically contain multiple axial/coronal slices per patient, it is possible that adjacent slices from the same patient fall in both train and test splits. Yagis et al. [9] demonstrated that slice-level splitting in 2D MRI CNN studies over-estimated accuracy by 29–55 % on four public datasets. Our numbers are therefore best treated as an **upper bound** on patient-level generalisation. This is a direct, honest limitation and is probably the single most important caveat in the paper.
2. **Single-source dataset**: all images from hospitals in Dhaka. Varoquaux & Cheplygina [4] repeatedly flag single-centre datasets as the primary driver of over-optimism in medical ML. No external validation is possible within the scope of a 3-week assignment.
3. **Axial + coronal slices mixed**: the reference paper [5] does not document the ratio. Our model will see whatever mix exists in the data; we cannot condition on slice orientation without additional metadata we don't have.

---

## 9. Phase 0 deliverables checklist

- [ ] `split.py` — deterministic stratified splitter, emits `split.csv`.
- [ ] `split.csv` — one-time artefact, committed to the repo.
- [ ] `preprocessing.py` — `load_image(path) -> np.ndarray` + per-pipeline extensions.
- [ ] `evaluate.py` — computes every metric in §5 plus McNemar's (§6), given `y_true` and `y_pred` (and optionally `y_prob` for ROC-AUC).
- [ ] `bootstrap.py` — 1000-resample CI on macro-F1 and per-class F1.
- [ ] `config.py` / `config.yaml` — central seed and path constants.
- [ ] `README.md` — setup, how to reproduce, directory layout, environment freeze.
- [ ] One smoke-test run per pipeline on the **small** dataset, end-to-end, to confirm the harness works before Phase 1/2.

---

## References

[1] J. Mongan, L. Moy, and C. E. Kahn, "Checklist for Artificial Intelligence in Medical Imaging (CLAIM): A Guide for Authors and Reviewers," *Radiology: Artificial Intelligence*, vol. 2, no. 2, e200029, 2020. doi:10.1148/ryai.2020200029.

[2] L. Tejani et al., "Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 Update," *Radiology: Artificial Intelligence*, 2024. doi:10.1148/ryai.240300.

[3] G. S. Collins et al., "TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods," *BMJ*, vol. 385, e078378, 2024. doi:10.1136/bmj-2023-078378.

[4] G. Varoquaux and V. Cheplygina, "Machine learning for medical imaging: methodological failures and recommendations for the future," *npj Digital Medicine*, vol. 5, no. 48, 2022. doi:10.1038/s41746-022-00592-y.

[5] M. N. Islam et al., "Vision transformer and explainable transfer learning models for auto detection of kidney cyst, stone and tumor from CT-radiography," *Scientific Reports*, vol. 12, 11440, 2022. doi:10.1038/s41598-022-15634-4.

[6] A. C. Davison and D. V. Hinkley, *Bootstrap Methods and their Application*, Cambridge University Press, 1997.

[7] H. He and E. A. Garcia, "Learning from Imbalanced Data," *IEEE Transactions on Knowledge and Data Engineering*, vol. 21, no. 9, pp. 1263–1284, 2009.

[8] B. Krawczyk, "Learning from imbalanced data: open challenges and future directions," *Progress in Artificial Intelligence*, vol. 5, pp. 221–232, 2016.

[9] E. Yagis et al., "Effect of data leakage in brain MRI classification using 2D convolutional neural networks," *Scientific Reports*, vol. 11, 22544, 2021. doi:10.1038/s41598-021-01681-w.

[10] T. G. Dietterich, "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms," *Neural Computation*, vol. 10, no. 7, pp. 1895–1923, 1998.

[11] M. Buda, A. Maki, and M. A. Mazurowski, "A systematic study of the class imbalance problem in convolutional neural networks," *Neural Networks*, vol. 106, pp. 249–259, 2018.

[12] J. M. Johnson and T. M. Khoshgoftaar, "Survey on deep learning with class imbalance," *Journal of Big Data*, vol. 6, no. 27, 2019.

[13] M. Grandini, E. Bagli, and G. Visani, "Metrics for Multi-Class Classification: an Overview," arXiv:2008.05756, 2020.

[14] M. Sokolova and G. Lapalme, "A systematic analysis of performance measures for classification tasks," *Information Processing & Management*, vol. 45, no. 4, pp. 427–437, 2009.

[15] P. Rajpurkar et al., "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning," arXiv:1711.05225, 2017.

[16] M. Raghu, C. Zhang, J. Kleinberg, and S. Bengio, "Transfusion: Understanding Transfer Learning for Medical Imaging," *NeurIPS*, 2019.

[17] T. Fawcett, "An introduction to ROC analysis," *Pattern Recognition Letters*, vol. 27, no. 8, pp. 861–874, 2006.

[18] D. J. Hand and R. J. Till, "A Simple Generalisation of the Area Under the ROC Curve for Multiple Class Classification Problems," *Machine Learning*, vol. 45, pp. 171–186, 2001.

[19] B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall, 1993.

[20] A. Benavoli, G. Corani, J. Demšar, and M. Zaffalon, "Time for a Change: a Tutorial for Comparing Multiple Classifiers Through Bayesian Analysis," *Journal of Machine Learning Research*, vol. 18, pp. 1–36, 2017.

[21] J. Pineau et al., "Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)," *Journal of Machine Learning Research*, vol. 22, pp. 1–20, 2021.
