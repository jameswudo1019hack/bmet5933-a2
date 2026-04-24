# Phase 2 — Deep Learning Pipeline Design Justification

**BMET 5933 Assignment 2 — Kidney CT Classification**
Person B: deep learning classifier
Status: draft v1, 2026-04-22 (amended 2026-04-24 with framing update — see below)

---

## Framing update — 2026-04-24

After Sprint 1 showed that the classical/DL soft-vote ensemble achieves 100 % test macro-F1, the project's framing shifted from *"who wins"* to *paradigm comparison through interpretability* — see **[[Project_Framing_v2]]** for the canonical statement. The design decisions below remain correct, but they should now be read as serving the interpretability analysis rather than chasing a score. In particular: fixed split + matched preprocessing + per-class bootstrap CIs + McNemar's paired test are all infrastructure for reasoning about *what each paradigm learns*, not for declaring a winner.

---

## Purpose of this document

This document justifies every design decision made in Person B's deep-learning pipeline, with direct evidence from the peer-reviewed literature. It is the companion to the Phase 0 justification (shared infrastructure) and will feed the Technical Contributions — Deep Learning subsection of the final paper. Reference numbers below are local to this document; the final paper will merge references across both phases. The combined paper will continue to claim conformance with CLAIM 2024 [1] and TRIPOD+AI [2], both introduced in the Phase 0 document.

The pipeline consists of: (1) an EfficientNet-B0 backbone with ImageNet pretrained weights, (2) a two-stage transfer-learning protocol (frozen-head warmup → selective fine-tune of the final blocks), (3) inverse-frequency class weighting on cross-entropy, (4) moderate geometric and photometric augmentation, (5) early stopping on validation macro-F1, and (6) A100 GPU training via Colab Pro+. Each is justified below.

---

## 1. Architecture selection

### Decision
EfficientNet-B0 [3], pretrained on ImageNet-1K, with the original 1000-way classifier head replaced by a two-layer `Dropout(0.3) → Linear(1280, 4)` block for the 4-way kidney-CT task.

### Evidence supporting the decision
Three independent lines of reasoning converge on EfficientNet-B0.

Tan and Le [3] introduced EfficientNet as a family of convolutional networks obtained by compound scaling of depth, width, and input resolution from a neural-architecture-searched baseline. The central empirical claim in their Table 2 is that EfficientNet-B0 achieves **77.3 %** top-1 accuracy on ImageNet with **5.3 M parameters and 0.39 BFLOPs** — roughly matching ResNet-50's 76 % at approximately one-fifth the parameters and one-tenth the FLOPs. For a transfer-learning regime where training time and GPU memory are real constraints (Colab compute-unit budget), this efficiency directly translates into faster iteration and more room for ablations.

Kornblith, Shlens, and Le [4] tested the implicit assumption that better-ImageNet models transfer better. Across 16 architectures and 12 downstream datasets, they report Pearson correlations of **r = 0.99** (fixed feature extraction) and **r = 0.96** (fine-tuning) between ImageNet top-1 and transfer accuracy. This justifies choosing an architecture with strong ImageNet accuracy as a transfer source, rather than older or simpler baselines. At the time of Kornblith's study, EfficientNet did not yet exist; subsequent replication work has placed EfficientNets firmly in the "strong transfer source" regime.

Comparing against the reference paper [5] (Islam et al.) is the third consideration. Their transfer-learning results place VGG16 at 98.2 % accuracy, ResNet-50 at 73.8 %, and Inception v3 at 61.6 % on their balanced subset. The large spread across architectures makes it clear that choice matters. EfficientNet-B0 was not in their comparison; using it here produces a new data point on the same dataset rather than a duplication of their results.

### Alternatives considered and why rejected
- **ResNet-50**: the classic choice, but Islam et al. [5] report it at 73.8 % on this exact dataset, substantially below their VGG16 result. Without a clear reason to believe we could improve on their ResNet-50 configuration, it is not the best use of our one-classifier budget.
- **VGG16**: strong on Islam's benchmark (98.2 %), but the architecture is a decade old, is substantially heavier (138 M params), and is outperformed on every modern transfer-learning benchmark by EfficientNet-B0 [3]. Reproducing VGG16 would also duplicate Islam's work rather than extend it.
- **Swin Transformer**: Islam's top model (99.3 %). Rejected here because (a) the paper's premise — classical ML vs deep learning — is not served by chasing SOTA numbers, (b) proper Swin fine-tuning requires more compute than our budget supports, and (c) Matsoukas et al. [6] show that CNN and ViT performance on medical-imaging benchmarks converges under equivalent training regimes; the architectural choice matters less than training protocol and data regime.
- **Training a small custom CNN from scratch**: rejected because Raghu et al. [7] demonstrate that for medical imaging, the main benefit of ImageNet pretraining is the weight-scale distribution, not task-specific features. Abandoning pretraining gives up a meaningful free lunch for no reproducibility gain.

---

## 2. Transfer-learning strategy — two-stage training

### Decision
Stage 1: freeze the entire EfficientNet backbone, train only the new classifier head with Adam at LR = 1e-3 for 5 epochs.
Stage 2: unfreeze the last 2 stages of the feature extractor (EfficientNet-B0 has 9 stages; stages 7–8 are unfrozen), train with AdamW at LR = 1e-5, weight decay 1e-4, for up to 30 epochs with early stopping (§5).

### Evidence supporting the decision
Yosinski, Clune, Bengio, and Lipson [8] performed the foundational empirical study of feature transferability in deep networks. Their central finding — that early convolutional layers learn general features (edges, textures, Gabor-like filters) that transfer cleanly across visual tasks, while later layers become progressively more task-specific — directly motivates the two-stage design. Freezing the early layers and unfreezing only the late layers means we preserve the transferable part of ImageNet pretraining while allowing the part that most benefits from domain adaptation to adjust to CT radiography.

Howard and Ruder [9], in the ULMFiT paper for NLP transfer, formalised **gradual unfreezing** and **discriminative learning rates** as techniques that avoid catastrophic forgetting of pretrained features. The core argument generalises directly: if all layers fine-tune simultaneously with one learning rate, the high gradient signal from the new classifier head propagates back through the network and can overwrite stable pretrained representations. Our two-stage design is a stripped-down, vision-equivalent implementation of the same principle.

Kornblith et al. [4] additionally report that fine-tuning typically outperforms frozen-feature extraction on downstream tasks, with the gap depending on dataset size and similarity to ImageNet. Medical CT is quite different from natural images [7], which makes fine-tuning (Stage 2) expected to help more than it would on a closer domain. We therefore should not stop at Stage 1.

Raghu, Zhang, Kleinberg, and Bengio [7] — cited in Phase 0 and reused here — provide the caveat that ImageNet transfer benefits to medical imaging come largely from the weight distribution rather than specific learned features. This argues for Stage 1 being a short warmup (5 epochs is enough to stabilise the randomly initialised head) rather than a long training phase: we are not hoping to "learn" anything at Stage 1, we are merely preparing the head so Stage 2's fine-tuning does not destabilise.

### Alternatives considered and why rejected
- **End-to-end fine-tuning from epoch 1**: simpler but risks catastrophic forgetting per [9]. Additionally, with an uninitialised classifier head, the initial gradient signal is noisy and flows into the pretrained backbone, degrading its features before the head has stabilised.
- **Fully frozen backbone throughout training** (linear probing): rejected because [4] and [7] both argue that fine-tuning outperforms frozen extraction when sufficient training data is available, which we have (4353 training images).
- **Unfreezing more / all backbone stages in Stage 2**: for a 4 M-parameter model on 4353 images, unfreezing more stages risks over-fitting. Two stages is the conservative choice and matches common practice in medical-imaging transfer papers [10].

---

## 3. Optimiser and learning-rate choices

### Decision
- **Stage 1**: Adam [11], LR = 1e-3, no weight decay.
- **Stage 2**: AdamW [12], LR = 1e-5, weight decay = 1e-4.

### Evidence supporting the decision
Kingma and Ba [11] introduced Adam with the empirical finding that a default learning rate of 1e-3 performs robustly across a wide range of deep-learning tasks. For Stage 1 — where only the classifier head (~5,124 parameters) is trainable and the backbone is frozen — this default is appropriate, the gradients are clean, and no weight decay is needed because the head converges in a handful of epochs and regularisation is not the bottleneck.

Loshchilov and Hutter [12] showed that the weight decay in standard Adam implementations is implemented incorrectly: L2 regularisation and weight decay are equivalent under SGD but not under adaptive optimisers. Their AdamW variant decouples weight decay from the gradient update and generalises better than Adam-with-L2 across image-classification benchmarks. Since Stage 2 unfreezes more than a million parameters and therefore genuinely benefits from regularisation, AdamW is the correct choice there.

The learning rate drop of two orders of magnitude between stages (1e-3 → 1e-5) is consistent with standard transfer-learning practice [4, 9, 10]. The intuition is straightforward: Stage 1's gradients flow into randomly initialised weights that can absorb large updates; Stage 2's gradients flow into pretrained weights that we want to *adjust*, not overwrite. A small LR in Stage 2 preserves most of the pretrained information while still allowing task-specific adaptation.

Weight decay of 1e-4 is the default in torchvision reference training recipes for ImageNet models; it has wide empirical support as a reasonable starting point [12].

### Alternatives considered
- **SGD with momentum**: historically outperformed Adam on large-scale ImageNet training and is the optimiser used in many canonical papers. However, Loshchilov and Hutter [12] show that AdamW closes this gap. For transfer learning on a smaller target set (our regime), AdamW is at least as good and is easier to tune.
- **Cosine-annealed LR schedule**: would likely yield a small additional gain but introduces another hyperparameter (warmup length, min LR). We deliberately avoid it to keep the comparison with the classical pipeline as clean as possible.

---

## 4. Input preprocessing for the CNN branch

### Decision
On top of the shared 256×256 grayscale image returned by `shared.preprocessing.load_image`, the CNN branch applies:
1. Resize to 224×224 via bilinear interpolation.
2. Replicate single channel to three channels (grayscale → pseudo-RGB).
3. Normalise using the ImageNet mean and standard deviation (μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225]).

### Evidence supporting the decision
The 224×224 input size is the standard resolution for ImageNet-pretrained CNN backbones and was used by Rajpurkar et al. in CheXNet [13] when transfer-learning DenseNet-121 to medical radiographs. This has since become the near-universal default for medical-imaging transfer learning [6]. Departing from 224×224 would require either the additional compute of a larger input or a discard of spatial information through downsampling, neither of which is justified.

Channel replication (grayscale → 3-channel by repetition) is a mechanical necessity: ImageNet pretraining uses 3-channel RGB inputs, and the first convolutional layer of any ImageNet-pretrained backbone expects three input channels. The alternative of averaging the RGB filters into a single grayscale kernel loses information about how the network was actually trained and rarely outperforms channel replication in practice.

ImageNet normalisation (μ/σ from the pretraining distribution) is justified in detail in Phase 0 §4. Summary: deviating from the input distribution that a pretrained network expects degrades the operating point of the first convolutional layer and loses the benefit of pretraining. This was not contentious in the Phase 0 discussion and remains the default.

### Alternatives considered
- **Histogram equalisation or CLAHE before CNN ingestion**: evidence is mixed for medical images. Shorten and Khoshgoftaar [14] cite kernel-filter and contrast-adjustment augmentations as *sometimes* helpful and *sometimes* harmful depending on the task. We defer CLAHE as an optional A/B test during hyperparameter tuning; it is not part of the main pipeline.
- **Higher input resolution (e.g. 380×380, EfficientNet-B4's native resolution)**: would likely help given CT's high-resolution diagnostic content, but requires switching to a larger EfficientNet variant and substantially increases compute. Deferred as future work.

---

## 5. Early stopping

### Decision
Training terminates when validation macro-F1 fails to improve for 5 consecutive epochs after the best-seen score. The checkpoint corresponding to the best-seen val macro-F1 is what gets saved and evaluated on the test set.

### Evidence supporting the decision
Prechelt [15] gave the canonical treatment of early stopping as an implicit regulariser. The central claim — that monitoring a held-out validation metric and halting training when it plateaus prevents overfitting — is now taken as given, but Prechelt's specific analysis of *when* to stop (patience-based criteria, generalisation-loss criteria, etc.) is the reason patience-based stopping has become standard practice.

The patience value of 5 is conservative for this task. Validation macro-F1 often exhibits epoch-to-epoch noise of a few percentage points on a 934-sample validation set (bootstrap-variance calculation from Phase 0), so a patience of 1–2 would frequently stop training prematurely on noise. A patience of 10 or more would waste compute on runs that have genuinely plateaued. Five is the mid-range default that balances these concerns.

Macro-F1 rather than validation loss is the stopping metric because our primary reporting metric is macro-F1 (Phase 0 §5). Stopping on the same metric we report avoids the scenario where validation loss improves but macro-F1 doesn't — a known pathology on imbalanced multiclass tasks.

### Alternatives considered
- **No early stopping (fixed epoch count)**: would either under-train or waste compute. Worse, it couples the reported result to the choice of epoch count, making the comparison with the classical pipeline harder to interpret.
- **Early stopping on validation loss**: conventional choice, but macro-F1 is what we report, so stopping on F1 is more directly aligned.

---

## 6. Class-imbalance handling

### Decision
Inverse-frequency class weights applied to the cross-entropy loss, computed once from the training split and normalised so the mean weight is 1.0. For the medium dataset split this yields weights of approximately **Cyst 0.66, Normal 0.48, Stone 1.78, Tumor 1.08** — Stone (the minority class) gets 3.7× the weight of Normal (the majority).

### Evidence supporting the decision
This decision was made in Phase 0 §3 and the evidence there (Buda, Maki, and Mazurowski [16] on class imbalance in CNNs; Johnson and Khoshgoftaar [17] on deep learning under imbalance) is not repeated here. Briefly: cost-sensitive class weighting is one of the two empirically dominant strategies for CNN class imbalance [16], is equivalent in performance to random oversampling in most regimes, and preserves the full training set (avoiding the information-loss problem that Islam et al.'s [5] rebalancing-by-downsampling imposes).

Cui, Jia, Lin, Song, and Belongie [18] proposed a more sophisticated variant: **class-balanced loss based on the effective number of samples**, in which the weight for class `c` is `(1 − β) / (1 − β^n_c)` with `β ∈ [0, 1)` a hyperparameter. Their experiments on long-tailed CIFAR and iNaturalist show consistent improvements over both standard cross-entropy and plain inverse-frequency weighting. We consciously adopt the simpler inverse-frequency scheme and note CB loss as future work, for two reasons: (1) the imbalance ratio here (Normal : Stone ≈ 3.7 : 1) is mild compared to the 100 : 1 ratios where [18] shows its largest gains, and (2) CB loss introduces a β hyperparameter that would require its own tuning and a separate justification.

### Alternatives considered
- **Weighted random sampling in the DataLoader** (oversampling the minority class): equivalent or slightly stronger than class weights per [16], but changes the effective training-set size per epoch and complicates the per-epoch time comparison with the classical pipeline. We stick with class weights for simplicity.
- **Focal loss**: effective under extreme imbalance but Buda et al. [16] find it does not outperform cross-entropy with class weights under mild imbalance. Deferred to future work.
- **SMOTE** in feature space: not applicable to an end-to-end CNN; this is a classical-ML technique.

---

## 7. Augmentation

### Decision
Training-time augmentation applied only to the training split:
- Random horizontal flip, p = 0.5
- Random rotation ±15°
- Random zoom up to 10 % via `RandomResizedCrop(scale=(0.9, 1.0))`
- ColorJitter with brightness ±0.1 and contrast ±0.1
- **No vertical flip**

Validation and test see no augmentation; both apply `Resize(224)` + ImageNet normalisation only.

### Evidence supporting the decision
Shorten and Khoshgoftaar [14] survey image augmentation for deep learning and conclude that geometric transformations (rotation, flip, crop, translate) are the most consistently effective augmentation family for classification tasks, with photometric augmentation (brightness, contrast, jitter) providing smaller but generally positive effects. Our chosen augmentations fall squarely in the mainstream of that recommendation.

The specific hyperparameter values (±15° rotation, 0.1 zoom, 0.1 brightness/contrast) are deliberately conservative. Medical imaging augmentation literature repeatedly notes that aggressive augmentation — large rotations, heavy photometric distortion — can destroy diagnostic signals a radiologist would rely on. Matching Islam et al.'s [5] augmentation pipeline as closely as possible also improves comparability: if our model differs from theirs, we want the augmentation to be a controlled variable, not an uncontrolled one.

**The choice to exclude vertical flip is deliberate and important**: CT anatomy is **not** top-bottom symmetric. The spine sits dorsally, the liver on the right, the aorta left of midline. A vertically flipped CT slice is an anatomically invalid image and training the network to classify it as valid would encourage the model to learn invariances that contradict clinical reality. Horizontal flip is acceptable because the left/right kidney pair means horizontal flipping produces another plausible anatomical configuration (a patient imaged from the opposite side).

### Alternatives considered
- **No augmentation at all**: rejected because the training set is small by modern standards (4353 images) and augmentation is effectively free regularisation [14].
- **Aggressive augmentation including large rotations, translations, or MixUp/CutMix**: rejected as too risky for medical imaging without specific evidence these help on this task. Deferred as an ablation if time permits.
- **Elastic deformations**: common in medical image segmentation but rarely improves classification. Not pursued.

---

## 8. Batch size and dropout

### Decision
Batch size 32. Classifier-head dropout p = 0.3.

### Evidence supporting the decision
A batch size of 32 fits comfortably in the VRAM of every GPU we might use (A100 40 GB, T4 16 GB, M1 Pro unified 16 GB) with substantial headroom, which simplifies reproducibility across hardware. It is also within the range (small batch, ≤ 64) that Masters and Luschi [19] argue produces better generalisation than very large batches on image-classification tasks, a result confirmed in multiple subsequent studies.

Srivastava, Hinton, Krizhevsky, Sutskever, and Salakhutdinov [20] introduced dropout with the recommendation of `p = 0.5` on fully connected layers as a default. Modern practice has drifted slightly lower for transfer-learning scenarios — typical values of 0.2 to 0.3 are now standard in torchvision reference recipes and the TIMM library. Our 0.3 is at the upper end of that range, providing meaningful regularisation on the classifier head without destabilising Stage 1's short warmup.

### Alternatives considered
- **Larger batch sizes (64, 128)**: would be faster per epoch on the A100 but [19] argues against it on generalisation grounds. With 4353 training images and batch 32, we already complete an epoch in ~1 minute on A100.
- **No dropout**: rejected because the head is small (5,124 params vs. 4353 training images) and regularisation costs nothing.
- **Batch normalisation instead of dropout**: EfficientNet-B0 already contains batchnorm throughout the backbone; adding more in the head is unnecessary.

---

## 9. Hardware and compute environment

### Decision
Training runs on Colab Pro+ with NVIDIA A100 GPU allocation. Development and analysis (smoke tests, Grad-CAM, paper figures) run locally on an Apple M1 Pro via PyTorch MPS.

### Evidence supporting the decision
A100 is ~3–4× faster than T4 (the default GPU on free Colab) for standard convolutional workloads and is substantially more available under Pro+ than Pro. For this project's anticipated 6–7 training runs (main model plus the planned data-efficiency sweep), this translates to roughly 15 minutes of wall time per run rather than 45 minutes; the total time saved across the assignment is meaningful (2–3 hours) though the compute-unit cost is also higher.

Running smoke tests, inference, and post-hoc analyses (Grad-CAM, figure generation) locally on M1 Pro is deliberate: the M1 Pro MPS backend in PyTorch ≥ 2.2 is stable for EfficientNet-B0, inference over the 934-sample test set completes in under a minute, and keeping Colab GPU time reserved for training alone is efficient.

The entire pipeline is **device-agnostic**: `resolve_device()` in `deep_learning.train` selects CUDA if available, MPS if CUDA is not, otherwise CPU. The same code runs in all three environments. Reproducibility across devices is not perfect — cuDNN and MPS use different floating-point kernels — but all reported results come from the same A100-trained checkpoint, eliminating device variance from the final numbers.

### Alternatives considered
- **Colab Pro (T4 / occasional V100)**: viable and ~$40/month cheaper. For this specific assignment Pro would be sufficient; Pro+ was chosen because the author was already subscribing for other coursework.
- **Local M1 Pro for full training**: technically possible but ~2 hours per run, making the data-efficiency sweep a weekend of unattended training. Not practical.
- **Train on the free Colab tier**: risks session preemption mid-training and the 12 h hard limit. Unreliable for a graded assignment.

---

## 10. Reproducibility artefacts produced by this pipeline

Every training run emits, into a run-specific subdirectory of `Results/`:
1. `best_model.pt` — state_dict at highest val macro-F1 (gitignored; backed up to Drive).
2. `run_log.json` — seed, device, full config, per-epoch train/val losses + val macro-F1, best epoch, total wall time.
3. `dl_results.json` — test-set evaluation through `shared.evaluate`: accuracy, macro-F1 (with 95 % bootstrap CI), weighted F1, per-class P/R/F1 with per-class bootstrap CIs, confusion matrix, OvR-macro ROC-AUC.
4. `dl_predictions.npz` — raw `y_true`, `y_pred`, `y_prob` arrays for downstream paired tests (McNemar's vs. the classical model).

Together these artefacts cover the reporting items required by CLAIM 2024 [1] for a classification study. The config and per-epoch log additionally address the reproducibility items in TRIPOD+AI [2] and Pineau et al.'s [21] NeurIPS-2019 reproducibility checklist.

---

## 11. Known limitations specific to this pipeline

1. **Single random seed per reported run.** Training involves stochastic elements — DataLoader shuffling, augmentation sampling, potentially cuDNN non-determinism — so the reported result is a single draw from a distribution of possible outcomes. A defensible seed-variance analysis would require 3–5 runs per configuration, which the compute budget can just about support if we restrict it to the main final model. To be decided during execution.
2. **No k-fold cross-validation.** Phase 0 §2 justified a fixed train/val/test split as the design that enables the cleanest paired comparison (McNemar's on a shared held-out test set). The cost is a larger variance on our reported numbers than CV would provide. The bootstrap CIs in §10 (Phase 0 §5) partially compensate.
3. **Hyperparameters not exhaustively tuned.** Values in §2–§8 are informed by the literature and standard practice but are not the product of a full grid search. A paper-scale ablation would vary at least LR, weight decay, dropout, and augmentation magnitude; we will do at most one targeted ablation (Stage-2 LR) depending on time.
4. **Patient-level-split limitation inherited from Phase 0.** No patient IDs in Islam et al.'s [5] release; Yagis et al. [22] quantify the resulting over-estimation at 29–55 % on comparable 2D medical-imaging CNN tasks. This caveat is the single most important one and will be repeated verbatim in the paper's Limitations section.

---

## References

[1] J. Mongan, L. Moy, and C. E. Kahn, "Checklist for Artificial Intelligence in Medical Imaging (CLAIM): A Guide for Authors and Reviewers," *Radiology: Artificial Intelligence*, vol. 2, no. 2, e200029, 2020. 2024 Update: L. Tejani et al., *Radiology: AI*, 2024. doi:10.1148/ryai.240300.

[2] G. S. Collins et al., "TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods," *BMJ*, vol. 385, e078378, 2024. doi:10.1136/bmj-2023-078378.

[3] M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," in *Proceedings of the 36th International Conference on Machine Learning (ICML)*, 2019, pp. 6105–6114. arXiv:1905.11946.

[4] S. Kornblith, J. Shlens, and Q. V. Le, "Do Better ImageNet Models Transfer Better?," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019. arXiv:1805.08974.

[5] M. N. Islam et al., "Vision transformer and explainable transfer learning models for auto detection of kidney cyst, stone and tumor from CT-radiography," *Scientific Reports*, vol. 12, 11440, 2022. doi:10.1038/s41598-022-15634-4.

[6] C. Matsoukas, J. F. Haslum, M. Söderberg, and K. Smith, "Is it Time to Replace CNNs with Transformers for Medical Images?," *ICCV Workshop on Computer Vision for Automated Medical Diagnosis*, 2021. arXiv:2108.09038.

[7] M. Raghu, C. Zhang, J. Kleinberg, and S. Bengio, "Transfusion: Understanding Transfer Learning for Medical Imaging," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2019.

[8] J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, "How transferable are features in deep neural networks?," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2014.

[9] J. Howard and S. Ruder, "Universal Language Model Fine-tuning for Text Classification," in *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL)*, 2018.

[10] N. Tajbakhsh et al., "Convolutional Neural Networks for Medical Image Analysis: Full Training or Fine Tuning?," *IEEE Transactions on Medical Imaging*, vol. 35, no. 5, pp. 1299–1312, 2016.

[11] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in *Proceedings of the 3rd International Conference on Learning Representations (ICLR)*, 2015. arXiv:1412.6980.

[12] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," in *Proceedings of the 7th International Conference on Learning Representations (ICLR)*, 2019. arXiv:1711.05101.

[13] P. Rajpurkar et al., "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning," arXiv:1711.05225, 2017.

[14] C. Shorten and T. M. Khoshgoftaar, "A survey on Image Data Augmentation for Deep Learning," *Journal of Big Data*, vol. 6, no. 60, 2019. doi:10.1186/s40537-019-0197-0.

[15] L. Prechelt, "Early Stopping — But When?," in *Neural Networks: Tricks of the Trade*, LNCS 1524, Springer, 1998, pp. 55–69.

[16] M. Buda, A. Maki, and M. A. Mazurowski, "A systematic study of the class imbalance problem in convolutional neural networks," *Neural Networks*, vol. 106, pp. 249–259, 2018.

[17] J. M. Johnson and T. M. Khoshgoftaar, "Survey on deep learning with class imbalance," *Journal of Big Data*, vol. 6, no. 27, 2019.

[18] Y. Cui, M. Jia, T.-Y. Lin, Y. Song, and S. Belongie, "Class-Balanced Loss Based on Effective Number of Samples," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019, pp. 9268–9277. arXiv:1901.05555.

[19] D. Masters and C. Luschi, "Revisiting Small Batch Training for Deep Neural Networks," arXiv:1804.07612, 2018.

[20] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A Simple Way to Prevent Neural Networks from Overfitting," *Journal of Machine Learning Research*, vol. 15, pp. 1929–1958, 2014.

[21] J. Pineau et al., "Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)," *Journal of Machine Learning Research*, vol. 22, pp. 1–20, 2021.

[22] E. Yagis et al., "Effect of data leakage in brain MRI classification using 2D convolutional neural networks," *Scientific Reports*, vol. 11, 22544, 2021. doi:10.1038/s41598-021-01681-w.
