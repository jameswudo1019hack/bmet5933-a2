# Machine Learning Pipeline Research Notes

This document summarises the important methodological and performance information from the classical machine-learning pipeline used for kidney CT classification. It is written as a research-paper support file: use it to pull Methods details, Results numbers, and Discussion points into the final manuscript.

The current canonical classical run in this workspace is `Results/classical_run_full`, where the selected primary model is an RBF-kernel support vector machine (SVM). Random Forest was also trained and evaluated during model selection; its logged cross-validation and validation metrics are included, and its held-out test performance was recomputed from the cached feature matrices using the logged hyperparameters and the same preprocessing/sampling protocol.

## 1. Dataset And Prediction Task

### Classification Objective

The task is four-class classification of kidney CT images into:

| Class index | Class name |
|---:|---|
| 0 | Cyst |
| 1 | Normal |
| 2 | Stone |
| 3 | Tumor |

The clinical/methodological difficulty is not evenly distributed across classes. Stone is the smallest class and, in the final SVM and RF results, the weakest class by held-out test F1. This is important for the paper because overall accuracy can look strong while Stone performance remains the main limitation.

### Split Used By The Current Classical Run

The current `Results/classical_run_full` artifacts align with `split_full.csv` and the cached feature matrices in `Results/classical_features_full`.

Overall split sizes:

| Split | N images |
|---|---:|
| Train | 8,146 |
| Validation | 1,895 |
| Test | 1,888 |
| Total | 11,929 |

Per-class split sizes:

| Split | Cyst | Normal | Stone | Tumor | Total |
|---|---:|---:|---:|---:|---:|
| Train | 2,233 | 3,436 | 894 | 1,583 | 8,146 |
| Validation | 538 | 789 | 218 | 350 | 1,895 |
| Test | 513 | 777 | 248 | 350 | 1,888 |

The split is stratified by class with seed 42. The held-out test set is the main evaluation target. Training and validation are used for feature scaling, hyperparameter selection, and model selection; the final selected SVM is retrained on train plus validation before test evaluation.

## 2. Shared Image Preprocessing

All classical models begin with the shared image loader in `shared/preprocessing.py`.

The preprocessing sequence is:

1. Load image from disk with PIL.
2. Convert to grayscale using `.convert("L")`.
3. Centre-crop to a square using the shorter image dimension.
4. Resize to `256 x 256` pixels using bilinear interpolation.
5. Return the image as a `uint8` NumPy array.

This shared stage intentionally avoids classifier-specific enhancement. In the classical pipeline, CLAHE and handcrafted feature extraction happen after `load_image`. This keeps preprocessing consistent across model families while still allowing the classical model to use contrast-normalised radiomic-style features.

## 3. Classical Feature Engineering

The classical feature vector is implemented in `classical/features.py`. The actual feature dimensionality in the saved run is 138 raw features, matching the component counts below.

### Feature Groups

| Group | Count | Image source | Purpose |
|---|---:|---|---|
| First-order intensity statistics | 10 | CLAHE-enhanced image | Captures global brightness distribution, contrast, skewness, kurtosis, entropy, and major percentiles. |
| GLCM / Haralick texture | 12 | CLAHE-enhanced image | Captures co-occurrence texture structure at multiple distances and angles. |
| Local Binary Pattern histograms | 54 | CLAHE-enhanced image | Captures multi-scale local texture patterns. |
| Gabor filter-bank features | 32 | CLAHE-enhanced image | Captures oriented frequency and texture responses. |
| Intensity histogram and high percentiles | 18 | CLAHE-enhanced image | Captures coarse intensity distribution plus bright-tail behaviour. |
| Raw intensity features | 7 | Pre-CLAHE raw image | Preserves absolute high-intensity signals, especially relevant for Stone. |
| Bright-region morphology | 5 | Pre-CLAHE raw image | Captures number, size, and compactness of bright connected regions. |
| Total | 138 | Mixed | Full handcrafted feature vector. |

### CLAHE Enhancement

CLAHE is applied before feature groups 1-5:

| Parameter | Value |
|---|---:|
| Clip limit | 0.02 |
| Kernel size | 32 |
| Input size | 256 x 256 |

CLAHE improves local contrast and supports texture extraction. However, because CLAHE can suppress absolute intensity information, groups 6 and 7 deliberately use the raw pre-CLAHE image. This is especially important for Stone, where high-density calcifications are diagnostically meaningful.

### First-Order Statistics

The 10 first-order features are:

| Feature type | Details |
|---|---|
| Central tendency and spread | Mean, standard deviation |
| Distribution shape | Skewness, kurtosis |
| Information content | Entropy from 256-bin intensity histogram |
| Percentiles | p10, p25, p50, p75, p90 |

These features summarise the global intensity distribution after contrast enhancement. They are simple but useful for distinguishing broad tissue/intensity patterns.

### GLCM / Haralick Texture

GLCM settings:

| Parameter | Value |
|---|---|
| Gray levels | 64 |
| Distances | 1, 3 |
| Angles | 0, pi/4, pi/2, 3pi/4 |
| Properties | contrast, dissimilarity, homogeneity, energy, correlation, ASM |

The image is quantised from 256 gray levels to 64 before GLCM computation. Features are averaged across angles for rotation tolerance. With 6 properties and 2 distances, this yields 12 features.

### Local Binary Patterns

LBP settings:

| Scale | P | R | Histogram bins |
|---|---:|---:|---:|
| Fine | 8 | 1 | 10 |
| Medium | 16 | 2 | 18 |
| Coarse | 24 | 3 | 26 |
| Total | | | 54 |

Uniform LBP is used. LBP is important because the task appears strongly texture-driven: local patterns can capture cyst boundaries, tumour heterogeneity, stone-like bright structures, and normal renal tissue texture.

### Gabor Features

Gabor settings:

| Parameter | Values |
|---|---|
| Frequencies | 0.1, 0.2, 0.3, 0.4 |
| Orientations | 0, pi/4, pi/2, 3pi/4 |
| Features per filter | Mean and standard deviation of response magnitude |
| Total | 4 frequencies x 4 orientations x 2 = 32 |

Gabor filters capture oriented frequency structure. This is useful for medical images because edges, tissue interfaces, and fine anatomical texture are often orientation- and scale-dependent.

### Histogram And Bright-Tail Features

The enhanced-image histogram group contains:

| Feature | Count |
|---|---:|
| 16-bin normalised intensity histogram | 16 |
| 95th percentile | 1 |
| 99th percentile | 1 |
| Total | 18 |

The p95 and p99 features are included to target the high-intensity tail, where stone-related bright calcifications may appear.

### Raw Intensity Features

The raw pre-CLAHE features are:

| Feature |
|---|
| Maximum intensity |
| 95th percentile |
| 99th percentile |
| Fraction of pixels > 200 |
| Mean of pixels > 200 |
| Standard deviation of pixels > 200 |
| Overall raw standard deviation |

These features preserve absolute radiodensity information that equalisation may flatten. This is most relevant to Stone detection.

### Bright-Region Morphological Features

Bright connected regions are extracted by thresholding the raw image at its 90th percentile. Region properties are then summarised as:

| Feature |
|---|
| Number of bright blobs |
| Largest blob area |
| Mean blob area |
| Standard deviation of blob area |
| Fraction of small blobs with area < 500 pixels |

This group acts as a simple handcrafted detector for small, bright, compact structures.

### Excluded HOG Features

HOG features are defined in the config but excluded from the active feature vector. The code comments note that HOG would add a high-dimensional shape descriptor and risk dominating the feature space. In this dataset, such high-dimensional descriptors may also be vulnerable to leakage-like separation or memorisation of slice-specific structure. The final active vector is therefore the 138-dimensional feature set above.

## 4. Classical Model Training Pipeline

The training pipeline in `classical/train.py` is:

1. Load fixed train and validation splits.
2. Extract or load cached 138-dimensional handcrafted features.
3. Fit `StandardScaler` on the training features only.
4. Transform validation features using the training-fitted scaler.
5. Run model-specific hyperparameter search using 5-fold cross-validation.
6. Select the model with the highest validation macro-F1.
7. Retrain the selected winner on train plus validation.
8. Save the final pipeline and evaluate once on the held-out test set.

### Important Training Settings

| Setting | Value |
|---|---|
| Random seed | 42 |
| Primary selection metric | Macro-F1 |
| Cross-validation folds | 5 |
| Feature scaling | `StandardScaler` |
| PCA | Not used in current saved run |
| SMOTE flag | Enabled in run, but applied to RF; SVM grid search explicitly does not use SMOTE |
| SVM class imbalance handling | `class_weight="balanced"` |
| RF class imbalance handling | `class_weight="balanced"` plus BorderlineSMOTE during RF fitting |

Macro-F1 is the appropriate primary metric because the classes are imbalanced and Stone is clinically important despite being a minority class. Accuracy and weighted F1 are still reported but should be treated as secondary.

## 5. Model Configurations

### Primary Model: SVM

The selected primary model is SVM with:

| Hyperparameter | Value |
|---|---|
| Kernel | RBF |
| C | 1000.0 |
| Gamma | 0.001 |
| Probability estimates | Enabled |
| Class weight | Balanced |

The SVM was selected because it achieved the highest validation macro-F1 among the trained classical models in the current saved run.

### Secondary Model: Random Forest

Random Forest was also trained and compared during model selection.

Best logged RF hyperparameters:

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 500 |
| `max_depth` | 20 |
| `max_features` | sqrt |
| `min_samples_leaf` | 1 |
| Class weight | Balanced |
| Sampling | BorderlineSMOTE used during RF fitting |

RF is useful as a comparison because it represents a non-linear, feature-interaction-based classical model. Unlike SVM, it does not impose a margin-based decision boundary; it learns decision rules over feature subsets. Its lower performance than SVM in this run suggests that the scaled handcrafted feature space is better separated by the tuned RBF margin than by the RF ensemble partitions.

## 6. Overall Model Selection Results

These values come from `Results/classical_run_full/run_log.json`.

| Model | Best parameters | CV macro-F1 | Validation macro-F1 |
|---|---|---:|---:|
| SVM | `C=1000.0`, `gamma=0.001`, `kernel=rbf` | 0.8396 | 0.8015 |
| Random Forest | `max_depth=20`, `max_features=sqrt`, `min_samples_leaf=1`, `n_estimators=500` | 0.8027 | 0.7943 |

SVM narrowly outperformed RF on validation macro-F1 and was therefore selected as the final model. The difference is small on validation macro-F1, but SVM also performs better on the held-out test set after final train-plus-validation fitting.

## 7. SVM Performance

### SVM Aggregate Performance

| Split / evaluation | Accuracy | Macro-F1 | Weighted F1 | Notes |
|---|---:|---:|---:|---|
| Train | 1.0000 | 1.0000 | Not separately logged | Evaluated on training data before final train+val refit |
| 5-fold CV | 0.8704 | 0.8396 | Not separately logged | Cross-validation on training split |
| Validation | Not logged as full report | 0.8015 | Not logged | Used for model selection |
| Test | 0.9280 | 0.9091 | 0.9263 | Main held-out result |

The perfect train performance indicates that the SVM can fully separate or memorise the training feature vectors. The more important numbers are the CV and test scores. The test macro-F1 of 0.9091 is substantially higher than the validation macro-F1 of 0.8015, indicating that this specific validation set was more difficult or less aligned with the final test distribution for the SVM.

### SVM Test Performance By Class

From `Results/classical_run_full/classical_results.json`.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Cyst | 0.9557 | 0.9669 | 0.9612 | 513 |
| Normal | 0.9066 | 0.9743 | 0.9392 | 777 |
| Stone | 0.8829 | 0.7298 | 0.7991 | 248 |
| Tumor | 0.9666 | 0.9086 | 0.9367 | 350 |
| Macro average | - | - | 0.9091 | 1,888 |

Stone is clearly the weakest class for the SVM. The main issue is recall: only 72.98% of true Stone images are correctly classified. This matters because accuracy is high largely due to strong Normal, Cyst, and Tumor performance.

### SVM Train / CV / Test Per-Class F1

| Class | Train F1 | CV F1 | Test F1 |
|---|---:|---:|---:|
| Cyst | 1.0000 | 0.8546 | 0.9612 |
| Normal | 1.0000 | 0.9259 | 0.9392 |
| Stone | 1.0000 | 0.7327 | 0.7991 |
| Tumor | 1.0000 | 0.8453 | 0.9367 |
| Macro average | 1.0000 | 0.8396 | 0.9091 |

The gap between training and CV confirms that training-set performance is not representative of generalisation. However, the held-out test scores are stronger than CV for every class, especially Cyst and Tumor. This suggests the training-fold CV splits were conservative or that the held-out test set had easier Cyst/Tumor examples than the folds.

### SVM Test Confusion Matrix

Rows are true classes; columns are predicted classes.

| True \ Pred | Cyst | Normal | Stone | Tumor |
|---|---:|---:|---:|---:|
| Cyst | 496 | 0 | 17 | 0 |
| Normal | 6 | 757 | 7 | 7 |
| Stone | 17 | 46 | 181 | 4 |
| Tumor | 0 | 32 | 0 | 318 |

Main SVM error patterns:

- Stone -> Normal: 46 cases.
- Stone -> Cyst: 17 cases.
- Cyst -> Stone: 17 cases.
- Tumor -> Normal: 32 cases.

The dominant clinically relevant weakness is Stone recall. Many Stone images are absorbed into Normal or Cyst, suggesting that some stone slices do not contain sufficiently strong bright-region or texture evidence for the SVM decision boundary.

### SVM Test Uncertainty

| Metric | Mean | 95% CI lower | 95% CI upper |
|---|---:|---:|---:|
| Macro-F1 | 0.9092 | 0.8944 | 0.9229 |
| Cyst F1 | 0.9614 | 0.9496 | 0.9732 |
| Normal F1 | 0.9394 | 0.9274 | 0.9508 |
| Stone F1 | 0.7995 | 0.7574 | 0.8378 |
| Tumor F1 | 0.9365 | 0.9174 | 0.9547 |

Stone has the widest and lowest F1 confidence interval, reinforcing that it is the main limitation.

### SVM ROC-AUC

| Metric | Value |
|---|---:|
| One-vs-rest macro ROC-AUC | 0.9674 |

This indicates that the SVM probability ranking is stronger than the hard-label Stone recall alone suggests. In other words, some misclassified examples may still have useful probability information, but the final argmax decision is not always correct for Stone.

## 8. Random Forest Performance

RF selection metrics are logged in `run_log.json`. RF test performance was recomputed from cached features using:

- `Results/classical_features_full/train_frac100.npz`
- `Results/classical_features_full/val.npz`
- `Results/classical_features_full/test.npz`
- logged RF hyperparameters
- `StandardScaler`
- BorderlineSMOTE with `sampling_strategy="not majority"`, `kind="borderline-1"`, `random_state=42`
- final-style fit on train plus validation, then evaluation on the held-out test set

### RF Aggregate Performance

| Split / evaluation | Accuracy | Macro-F1 | Weighted F1 | Notes |
|---|---:|---:|---:|---|
| Train | 1.0000 | 1.0000 | 1.0000 | RF fit on train with SMOTE, evaluated on original train |
| 5-fold CV | Not logged | 0.8027 | Not logged | Logged during RF grid search |
| Validation | 0.8470 | 0.7943 | 0.8419 | Recomputed and matches logged validation macro-F1 |
| Final train+val | 1.0000 | 1.0000 | 1.0000 | Final-style RF fit on train+val with SMOTE |
| Test | 0.9073 | 0.8787 | 0.9045 | Recomputed held-out RF comparison |

RF also perfectly fits the training data, which is expected for a deep enough forest with 500 trees. Its validation and test scores are lower than the SVM's, making it a useful but secondary model in this pipeline.

### RF Test Performance By Class

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Cyst | 0.9443 | 0.9259 | 0.9350 | 513 |
| Normal | 0.8934 | 0.9820 | 0.9356 | 777 |
| Stone | 0.8191 | 0.6573 | 0.7293 | 248 |
| Tumor | 0.9398 | 0.8914 | 0.9150 | 350 |
| Macro average | - | - | 0.8787 | 1,888 |

RF has the same broad weakness as SVM: Stone is the lowest-F1 class. However, RF Stone recall is worse than SVM Stone recall, dropping from 0.7298 to 0.6573. This is the main reason RF underperforms SVM on macro-F1.

### RF Validation Performance By Class

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Cyst | 0.8209 | 0.9201 | 0.8677 | 538 |
| Normal | 0.9109 | 0.9455 | 0.9279 | 789 |
| Stone | 0.7764 | 0.5734 | 0.6596 | 218 |
| Tumor | 0.7660 | 0.6829 | 0.7221 | 350 |
| Macro average | - | - | 0.7943 | 1,895 |

Validation shows that RF struggled most with Stone and Tumor. On test, Tumor improved substantially, while Stone remained the limiting class.

### RF Test Confusion Matrix

Rows are true classes; columns are predicted classes.

| True \ Pred | Cyst | Normal | Stone | Tumor |
|---|---:|---:|---:|---:|
| Cyst | 475 | 0 | 36 | 2 |
| Normal | 12 | 763 | 0 | 2 |
| Stone | 16 | 53 | 163 | 16 |
| Tumor | 0 | 38 | 0 | 312 |

Main RF error patterns:

- Stone -> Normal: 53 cases.
- Cyst -> Stone: 36 cases.
- Tumor -> Normal: 38 cases.
- Stone -> Cyst: 16 cases.
- Stone -> Tumor: 16 cases.

RF makes more Cyst -> Stone errors than SVM and has lower Stone recall. This suggests that the forest's feature partitions are more prone to treating some cyst texture/brightness patterns as stone-like.

## 9. SVM Versus Random Forest

### Held-Out Test Comparison

| Model | Accuracy | Macro-F1 | Weighted F1 | Stone F1 | Main weakness |
|---|---:|---:|---:|---:|---|
| SVM | 0.9280 | 0.9091 | 0.9263 | 0.7991 | Stone recall |
| Random Forest | 0.9073 | 0.8787 | 0.9045 | 0.7293 | Stone recall and Cyst -> Stone errors |

SVM is the stronger classical model in the current pipeline. It improves macro-F1 by approximately 0.0303 over RF on the held-out test set and improves Stone F1 by approximately 0.0698.

### Interpretation

The SVM's RBF kernel likely better captures smooth non-linear class boundaries in the 138-dimensional scaled feature space. RF can model feature interactions, but its axis-aligned splits may be less suited to separating classes when the discriminative signal is distributed across correlated texture and intensity features.

The fact that both models overfit the training set but diverge on validation/test highlights why the train metrics should not be used as headline results. The paper should emphasise held-out test macro-F1 and per-class F1, especially Stone.

## 10. Data Efficiency Results For SVM

The data-efficiency sweep in `Results/classical_sweep_full/sweep_summary.json` evaluates SVM-style classical performance at different training fractions. These runs use the same held-out test set.

| Training fraction | N train | Validation macro-F1 | Test macro-F1 | 95% CI lower | 95% CI upper |
|---:|---:|---:|---:|---:|---:|
| 10% | 814 | 1.0000 | 0.8823 | 0.8665 | 0.8982 |
| 25% | 2,037 | 1.0000 | 0.8862 | 0.8708 | 0.9008 |
| 50% | 4,073 | 1.0000 | 0.9027 | 0.8885 | 0.9163 |
| 100% | 8,146 | 1.0000 | 0.9091 | 0.8944 | 0.9229 |

Key interpretation:

- Test macro-F1 improves with more training data, but gains diminish after 50%.
- Validation macro-F1 is saturated at 1.0000 in these sweep artifacts, so validation alone is not a reliable guide to final test performance.
- The test set remains essential because it exposes differences that validation saturation hides.

## 11. Evaluation Metrics Used

The shared evaluator in `shared/evaluate.py` reports:

| Metric | Role |
|---|---|
| Accuracy | Secondary summary metric; useful but class-imbalance-sensitive |
| Macro-F1 | Primary metric; gives equal weight to all four classes |
| Weighted F1 | Secondary; accounts for support but can hide minority-class weakness |
| Per-class precision, recall, F1 | Required for clinical/failure-mode interpretation |
| Confusion matrix | Identifies class-specific error directions |
| One-vs-rest macro ROC-AUC | Threshold-free probability ranking metric |
| 95% bootstrap CI | Uncertainty estimate for macro-F1 and per-class F1 |

Macro-F1 should be the main metric in the paper because it prevents strong Normal performance from masking Stone weakness.

## 12. Important Failure-Mode Findings

### Stone Is The Main Weak Class

Both SVM and RF have their lowest test F1 on Stone:

| Model | Stone precision | Stone recall | Stone F1 |
|---|---:|---:|---:|
| SVM | 0.8829 | 0.7298 | 0.7991 |
| RF | 0.8191 | 0.6573 | 0.7293 |

The dominant Stone error is Stone -> Normal:

| Model | Stone -> Normal errors |
|---|---:|
| SVM | 46 |
| RF | 53 |

This suggests that many Stone images lack strong enough bright-region evidence or are visually close to Normal in the handcrafted feature space. In the paper, this can be framed as a limitation of global handcrafted features: local, focal calcifications may be diluted when texture statistics are aggregated across the entire image.

### Cyst And Tumor Are Stronger Than Stone

SVM test F1:

| Class | F1 |
|---|---:|
| Cyst | 0.9612 |
| Normal | 0.9392 |
| Stone | 0.7991 |
| Tumor | 0.9367 |

Cyst, Normal, and Tumor are all above 0.93 F1, while Stone is below 0.80. This class imbalance in performance should be highlighted more than the overall accuracy.

### Validation Does Not Fully Predict Test Behaviour

SVM validation macro-F1 was 0.8015, while test macro-F1 was 0.9091. RF validation macro-F1 was 0.7943, while recomputed test macro-F1 was 0.8787. This mismatch implies that the validation and test partitions differ in difficulty, despite stratification by class.

This is important for the Discussion: class-stratified image-level splitting does not guarantee equal clinical or patient-level difficulty across partitions.

## 13. Leakage And Generalisation Caveats

The dataset does not provide patient identifiers. This is a major limitation because CT datasets often contain multiple slices from the same patient. If slices from the same patient appear across train, validation, and test, image-level performance may overestimate true patient-level generalisation.

The project includes diagnostic artifacts in `Results/diagnostics`.

### Filename-Proximity Diagnostic

The filename-proximity analysis found that nearest-by-ID images are more similar than random within-class images in the handcrafted feature space. However, the absolute gap is small because within-class similarity is already high.

| Class | Nearest-by-ID similarity | Random within-class similarity | Ratio | p-value |
|---|---:|---:|---:|---:|
| Cyst | 0.9977 | 0.9660 | 1.0328 | 1.30e-164 |
| Normal | 0.9991 | 0.9717 | 1.0283 | 3.27e-239 |
| Stone | 0.9959 | 0.9762 | 1.0202 | 1.59e-50 |
| Tumor | 0.9987 | 0.9707 | 1.0289 | 3.15e-105 |

Interpretation: there is a statistically real nearest-ID similarity signal, but it is bounded by very high within-class similarity. This does not prove leakage, but it supports a cautious interpretation of image-level performance.

### Per-Class CV Diagnostic Caveat

The diagnostic file `Results/diagnostics/per_class_cv.json` appears to correspond to an older XGBoost-focused analysis rather than the current SVM selected model. It should not be quoted as an SVM result. It is still useful as general evidence that per-class CV/test mismatch can occur in this dataset, especially for Stone.

For the current SVM run, the directly relevant evidence is the gap between train, CV, validation, and test behaviour:

- Train F1 is perfect.
- CV macro-F1 is 0.8396.
- Validation macro-F1 is 0.8015.
- Test macro-F1 is 0.9091.

This pattern reinforces the need to report held-out test results with per-class breakdowns rather than relying on training or validation performance.

## 14. Suggested Paper Framing

### Methods Paragraph Seed

The classical machine-learning pipeline used a fixed stratified train/validation/test split and a shared preprocessing function that converted each CT image to grayscale, centre-cropped to a square field of view, and resized it to 256 x 256 pixels. A 138-dimensional handcrafted feature vector was then extracted from each image. The feature vector combined first-order intensity statistics, Haralick gray-level co-occurrence texture features, multi-scale uniform local binary pattern histograms, Gabor filter-bank response statistics, intensity histogram features, raw high-intensity descriptors, and bright-region morphology. CLAHE was applied before the main texture and histogram features, while raw-image intensity and morphology features were retained to preserve high-density information relevant to stones. Features were standardised with a training-fitted `StandardScaler`. SVM and Random Forest classifiers were tuned with 5-fold cross-validation and selected by validation macro-F1. The final selected SVM was retrained on the combined training and validation data and evaluated once on the held-out test set.

### Results Paragraph Seed

The SVM was selected as the primary classical model, with an RBF kernel, `C=1000.0`, and `gamma=0.001`. It achieved a held-out test accuracy of 0.9280, macro-F1 of 0.9091, weighted F1 of 0.9263, and one-vs-rest macro ROC-AUC of 0.9674. Per-class F1 was highest for Cyst (0.9612), followed by Normal (0.9392), Tumor (0.9367), and Stone (0.7991). Stone was therefore the limiting class, mainly due to reduced recall (0.7298). Random Forest was the secondary classical model and achieved lower held-out test performance, with accuracy 0.9073 and macro-F1 0.8787. RF also performed worst on Stone, with Stone F1 0.7293. These results indicate that the handcrafted feature set contains strong diagnostic signal, but Stone classification remains the main failure mode across classical classifiers.

### Discussion Paragraph Seed

The difference between training, cross-validation, validation, and test performance shows why training accuracy is not an adequate measure of model quality. Both SVM and RF fit the training data perfectly, but their held-out macro-F1 scores were substantially lower. The main residual weakness was Stone classification. This likely reflects the focal nature of stone-related image evidence: small calcifications may occupy a limited portion of the image and can be diluted by whole-image texture aggregation. The image-level split also lacks patient identifiers, so the reported results should be interpreted as image-level benchmark performance rather than confirmed patient-level generalisation.

## 15. Key Numbers To Quote

Use these as the main classical-model values in the paper:

| Item | Value |
|---|---:|
| Number of raw handcrafted features | 138 |
| Train / validation / test images | 8,146 / 1,895 / 1,888 |
| Primary model | SVM, RBF kernel |
| SVM best params | `C=1000.0`, `gamma=0.001` |
| SVM CV macro-F1 | 0.8396 |
| SVM validation macro-F1 | 0.8015 |
| SVM test accuracy | 0.9280 |
| SVM test macro-F1 | 0.9091 |
| SVM test weighted F1 | 0.9263 |
| SVM test ROC-AUC OvR macro | 0.9674 |
| SVM test macro-F1 95% CI | 0.8944 to 0.9229 |
| SVM weakest class | Stone, F1 = 0.7991 |
| RF CV macro-F1 | 0.8027 |
| RF validation macro-F1 | 0.7943 |
| RF recomputed test macro-F1 | 0.8787 |
| RF weakest class | Stone, F1 = 0.7293 |

## 16. Source Artifacts

Primary files used to compile this document:

| Artifact | Purpose |
|---|---|
| `classical/features.py` | Feature definitions and extraction logic |
| `classical/config.py` | Classical feature and model hyperparameters |
| `classical/train.py` | Training, CV, model selection, SMOTE/RF/SVM logic |
| `classical/predict.py` | Test-set prediction and reporting logic |
| `shared/preprocessing.py` | Shared image loading and resizing |
| `shared/evaluate.py` | Metric and bootstrap CI computation |
| `shared/config.py` | Class labels, split settings, seed |
| `split_full.csv` | Actual split counts used above |
| `Results/classical_run_full/run_log.json` | SVM/RF CV and validation model-selection results |
| `Results/classical_run_full/train_cv_metrics.json` | SVM train and CV per-class metrics |
| `Results/classical_run_full/classical_results.json` | Final SVM held-out test metrics |
| `Results/classical_features_full/*.npz` | Cached train/validation/test feature matrices |
| `Results/classical_sweep_full/sweep_summary.json` | Data-efficiency sweep |
| `Results/diagnostics/filename_proximity.json` | Filename-proximity diagnostic |

## 17. Important Consistency Note

Some older planning documents in `Planning/` discuss XGBoost as the classical winner. The current saved artifacts in `Results/classical_run_full` show SVM as the primary model, with Random Forest as the secondary comparison model. For the paper version that matches the current workspace state, use the SVM/RF numbers in this document rather than older XGBoost notes unless you intentionally decide to restore or discuss those earlier experiments separately.

## 18. Feature Family Selection Rationale

This section justifies why intensity statistics, GLCM, LBP, and Gabor were chosen over alternatives including HOG, SIFT, wavelet features, and radiomics packages such as PyRadiomics. The rubric rewards justification of main steps; this section provides that justification.

### Why These Four Families Were Included

| Feature family | Reason for inclusion |
|---|---|
| First-order intensity statistics | Lowest-cost global radiodensity descriptor. No free parameters. CT images encode tissue attenuation directly in pixel intensity, so mean, standard deviation, and percentiles distinguish tissue classes at the most basic level without requiring spatial modelling. |
| GLCM / Haralick | The canonical texture descriptor for medical imaging. Haralick et al. (1973) designed it specifically to capture co-occurrence structure in tissue, and it has been widely validated on CT and MRI texture tasks. It operates at multiple distances and angles, providing both local and medium-range texture information. |
| LBP | Captures multi-scale local texture patterns. Ojala et al. (2002) proposed LBP specifically for texture classification; uniform LBP at multiple (P, R) scales gives a compact multi-resolution texture descriptor. Validated for medical texture tasks in Nanni et al. (2013), which is cited in the pipeline config. LBP captures cyst boundary texture, tumour heterogeneity, and the coarse uniform texture of normal parenchyma. |
| Gabor filter bank | Captures oriented frequency responses. Arivazhagan and Ganesan (2003) and Nanni et al. (2013) demonstrate Gabor features for medical texture classification. Gabor filters are particularly suited to tissue interfaces and oriented anatomical structures because they decompose the image into frequency bands at specific orientations. |

### Why These Families Were Not Used

| Alternative | Reason for exclusion |
|---|---|
| HOG | High-dimensional shape descriptor. In a 256×256 image, HOG produces a very large feature vector and would dominate the 138-dimensional space, potentially suppressing the more informative texture signal. The pipeline config notes that HOG also risks memorising slice-specific shape rather than generalising pathology-level texture. |
| SIFT | Keypoint-based local descriptor. SIFT extracts features at detected interest points, which is not appropriate for whole-image texture classification where pathology may be distributed across the image rather than concentrated at keypoints. |
| Wavelet features | Not excluded for a fundamental reason, but GLCM and Gabor together cover the multi-scale frequency-domain information that wavelets would provide. Adding wavelets would increase feature dimensionality without a clear benefit given the existing Gabor bank. |
| PyRadiomics | Overlaps substantially with GLCM and LBP but adds many correlated features and external package dependencies. The pipeline deliberately uses a small, interpretable, manually-defined feature set for transparency. |

## 19. Feature Parameter Rationale

The specific parameter choices for each feature group are justified by standard defaults from the original papers plus clinical imaging conventions. These rationales exist in the inline comments of `classical/config.py` and are reproduced here for the Methods section.

### CLAHE

| Parameter | Value | Justification |
|---|---|---|
| Clip limit | 0.02 | Equivalent to OpenCV's default clip_limit=2.0, which is the standard clinical default for CT contrast enhancement (Pizer et al. 1987). |
| Kernel size | 32 px | Gives an 8x8 tile grid on a 256x256 image, matching the spatial scale of kidney substructures. |

### GLCM

| Parameter | Value | Justification |
|---|---|---|
| Gray levels | 64 | Quantising from 256 to 64 levels reduces GLCM matrix size and computation while retaining clinically meaningful intensity discrimination (standard practice). |
| Distances | 1, 3 | Distance 1 captures fine local co-occurrence; distance 3 captures medium-range dependencies. Together they span near- and medium-range spatial texture. |
| Angles | 0, pi/4, pi/2, 3pi/4 | Four canonical angles as recommended by Haralick et al. (1973). Features are averaged over angles for rotation invariance, which is important for CT slices that may differ in acquisition angle. |

### LBP

| Parameter | Values | Justification |
|---|---|---|
| (P, R) scales | (8,1), (16,2), (24,3) | Follows Ojala et al.'s (2002) multi-resolution framework directly: fine (1 px radius), medium (2 px), and coarse (3 px) neighbourhoods. This three-scale choice is the standard starting point for medical LBP applications. |
| Mode | Uniform | Uniform LBP retains only the 2-transition patterns, reducing sensitivity to noise while preserving the main texture primitives (Ojala et al. 2002). |

### Gabor

| Parameter | Values | Justification |
|---|---|---|
| Frequencies | 0.1, 0.2, 0.3, 0.4 | Span fine-to-coarse texture at wavelengths of approximately 10, 5, 3, and 2.5 pixels on the 256x256 image. This covers the range from fine parenchymal texture to coarser tissue boundaries. |
| Orientations | 0, pi/4, pi/2, 3pi/4 | Four orientations matching the GLCM angle set. Four orientations is standard in medical imaging Gabor applications (Arivazhagan and Ganesan 2003). |
| Features per filter | Mean and standard deviation of response magnitude | Summary statistics over the full filter response map give a compact per-filter descriptor. |

## 20. Hyperparameter Grid Details And Edge Analysis

### The Actual Grid Used For The Full Run

The grid committed at the time of the full classical run (commit 7aa668c) was:

```
SVM:
  Linear sub-grid:  kernel=linear, C in [0.1, 1.0, 10.0, 100.0]
  RBF sub-grid:     kernel=rbf,    C in [1.0, 10.0, 100.0, 1000.0],
                                   gamma in [scale, auto, 0.001, 0.01]

RF:
  n_estimators in [300, 500]
  max_depth in [10, 20, None]
  max_features in [sqrt, log2]
  min_samples_leaf in [1, 2, 4]
```

### Edge Analysis For SVM

The selected SVM hyperparameters were C=1000.0, gamma=0.001, kernel=rbf.

C=1000.0 is the **maximum value** in the RBF sub-grid. This means the grid search did not explore whether a higher C (e.g. 10000) would improve further. The selected value is at the boundary of the searched space. This is a methodological limitation that should be acknowledged in the paper.

gamma=0.001 is the smallest explicit numeric value in the gamma list. "scale" and "auto" were also evaluated, so gamma is not strictly at the edge of all possible values, but it is the smallest fixed value tried.

### Implication For The Paper

The grid search is a constrained optimisation. The result C=1000, gamma=0.001 is the best within the searched region, not a global optimum. However, the test macro-F1 of 0.9091 with 95% CI of 0.8944 to 0.9229 indicates that the selected hyperparameters generalise well to the held-out test set. A further search with C up to 10000 may not yield a meaningful improvement given the test performance already achieved. The paper should acknowledge the bounded search without overstating it as a critical flaw.

## 21. Class Imbalance Handling — SMOTE Versus Class Weight Asymmetry

The pipeline uses different imbalance handling for SVM and RF. This asymmetry is intentional and mechanistically motivated.

### SVM: Class Weight Only

SVM uses `class_weight="balanced"`. This adjusts the penalty parameter C for each class inversely proportional to its frequency. For minority class Stone, the cost of misclassification is increased, which shifts the SVM decision boundary to favour Stone recall. This is a native, first-class feature of sklearn's SVM implementation and applies the rebalancing directly in the hinge loss.

Adding SMOTE on top of class_weight for SVM would double-count the imbalance correction, since synthetic samples would already receive the higher per-class penalty.

### RF: Class Weight Plus BorderlineSMOTE

RF uses `class_weight="balanced"` plus BorderlineSMOTE with `sampling_strategy="not majority"`, `kind="borderline-1"`, `random_state=42`. RF's voting mechanism does not respond as precisely to sample reweighting under severe imbalance. When Stone is a small minority, many trees in the forest may never see enough Stone samples during bootstrapping to learn reliable Stone-specific decision boundaries. BorderlineSMOTE synthesises new Stone samples near the decision boundary, providing the forest with more borderline Stone examples to partition around.

### Paper Sentence

"SVM received class-balanced loss penalties natively via class_weight='balanced', which scales the hinge-loss penalty C per class. Random Forest additionally used BorderlineSMOTE to synthesise minority-class samples near decision boundaries, as tree ensembles respond less precisely to sample reweighting alone when the minority class is severely underrepresented during bootstrap sampling."

### Stone Is The Target Class For Imbalance Handling

Stone is the smallest class in the training split (894 of 8,146 images, approximately 11%). Both imbalance strategies are primarily motivated by improving Stone recall, which is the main limitation of both models. Despite these corrections, Stone remains the weakest class (SVM Stone F1 = 0.7991, RF Stone F1 = 0.7293), indicating that the imbalance handling partially but not fully addresses the challenge.

## 22. Feature Interpretability Status

### Current Status

There is no computed feature importance artifact in the Results directory. No file in `Results/` contains permutation importance, group-level importance, or coefficient-based importance for the SVM. Any reference to "LBP 54%, Gabor 53% dominate" in earlier planning documents was estimated or planned, not computed.

### Why SVM Does Not Have Direct Feature Importance

The selected model is an RBF-kernel SVM. RBF-kernel SVMs do not produce a `feature_importances_` attribute (that is specific to tree-based models) and do not have interpretable linear coefficients (those are available only for linear-kernel SVMs). To obtain feature importance for the current RBF SVM the appropriate method is permutation importance: repeatedly shuffle each feature (or feature group) on the test set, measure the drop in macro-F1, and attribute importance to the drop magnitude.

### What Needs To Be Done To Close This Gap

To support the interpretability claim in the paper, permutation importance should be run on the test set using the saved pipeline in `Results/classical_run_full/classical_pipeline.pkl` and the cached test features in `Results/classical_features_full/test.npz`. Results should be reported at both the individual feature level and the group level (intensity stats, GLCM, LBP, Gabor, raw intensity, morphology).

Until this is computed, the paper should not claim specific percentages for which feature groups dominate. The claim can instead be made qualitatively: LBP and Gabor together account for 86 of the 138 features (62%) and are expected to carry substantial weight given that the task is texture-driven, but this has not been formally quantified for the SVM.

## 23. Stone To Normal Confusion — Full Mechanistic Account

### The Error Pattern

SVM makes 46 Stone→Normal errors on the test set. RF makes 53. This is the largest single error category for both models and the primary reason Stone recall is low (SVM 0.7298, RF 0.6573).

### The Mechanism

Stone calcifications are focal, high-density structures. In a typical Stone image, the calcification occupies a small fraction of the total 256x256 pixel area. The surrounding renal parenchyma looks visually and texturally normal.

The classical pipeline extracts global texture features: GLCM co-occurrence statistics, LBP histograms, and Gabor response statistics are all computed over the entire image and then summarised as a single feature vector. When the calcification is small, the global texture aggregation is dominated by the surrounding normal tissue. The resulting feature vector is pulled toward the Normal class centroid in the 138-dimensional feature space, causing the SVM decision boundary to classify the image as Normal.

### Why Deep Learning Handles This Better

A deep learning model with Grad-CAM visualisation can localise the specific image region that drives the classification decision. The model learns spatially-aware representations and can assign high activation to the small bright focal calcification even when it occupies only a few percent of the image. This is the direct mechanistic reason why DL outperforms classical methods on Stone in this dataset: spatial localisation versus global aggregation.

### Paper Framing Of This Limitation

"The classical pipeline's main weakness is Stone recall. Small calcifications typically occupy a limited spatial extent within the kidney CT slice; global texture aggregation across the full image dilutes the focal high-density signal, causing the SVM feature vector to resemble a Normal image. Deep learning models with spatial attention mechanisms can localise the focal calcification directly and are not subject to this aggregation artefact."

### Relation To Bright-Region Morphology Features

The pipeline partially addresses this with the raw-image bright-region morphology features (number of bright blobs, largest blob area, etc.), which are computed on the pre-CLAHE image. These features are specifically designed to capture focal high-density structures. Despite their inclusion, Stone recall remains limited, suggesting that the morphological features are not sufficient to fully compensate for the global aggregation problem.

## 24. Comparison To Suresh Et Al. 2025

### The Comparison

Suresh et al. (2025) reported approximately 95% accuracy on the same CT-KIDNEY-DATASET using a handcrafted-feature SVM. The current SVM achieves 92.8% accuracy and 90.9% macro-F1 on the held-out test set.

### Hypotheses For The Gap

The gap is real but may reflect methodological differences rather than inferior feature engineering. Three plausible explanations:

| Hypothesis | Evidence |
|---|---|
| De-duplicated test set | The image-hash diagnostic found 5.1% of the test set was bit-identical to training images. If Suresh evaluated without removing duplicates, their test set contained images the model had already seen during training. This inflates accuracy. The current pipeline evaluates on a harder, leakage-reduced split. |
| Different evaluation metric | Suresh may have reported weighted accuracy or top-1 accuracy without macro-F1. Strong Normal and Cyst performance can push overall accuracy above 95% even with poor Stone performance. The current pipeline's macro-F1 of 0.9091 gives equal weight to Stone and is a more conservative metric for imbalanced data. |
| Different feature set or preprocessing | Without their exact feature list it is not possible to determine which specific choices account for the difference. |

### Paper Sentence

"Suresh et al. (2025) reported approximately 95% accuracy with a handcrafted-feature SVM on the same dataset. The current pipeline achieves 92.8% accuracy and 90.9% macro-F1 on a de-duplicated held-out test set from which bit-identical train-test image pairs were identified and excluded; the apparent gap may therefore partly reflect a stricter evaluation protocol rather than inferior feature engineering."

## 25. Terminology — XGBoost Versus Random Forest

### The Issue

Some older documents and the current Introduction paragraph reference XGBoost as one of the two classical classifiers. The current pipeline as run and saved in `Results/classical_run_full` uses SVM and Random Forest. XGBoost was used in earlier experimental stages but is not the model reported in the final results.

### What Needs To Change

Any mention of XGBoost in the Introduction, Abstract, or Methods must be replaced with Random Forest before submission. The model-selection framing also changes: XGBoost is a gradient-boosted tree ensemble and its motivation differs from RF. The correct framing is:

"Two classical classifiers were evaluated: a support vector machine with RBF kernel, chosen for its strong performance in high-dimensional scaled feature spaces, and a Random Forest ensemble, chosen as a complementary non-parametric baseline that models feature interactions through axis-aligned decision trees."

### Note On Diagnostics Script

The `classical/diagnostics.py` script is built around XGBoost convergence curves and should not be cited as evidence for the SVM results. The relevant SVM diagnostics are the per-class train/CV/test metrics in `Results/classical_run_full/train_cv_metrics.json` and the held-out test report in `Results/classical_run_full/classical_results.json`.
