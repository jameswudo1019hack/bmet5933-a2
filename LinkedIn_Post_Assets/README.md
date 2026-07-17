# LinkedIn Post Assets

Visuals summarising the kidney-CT classification project (Normal / Cyst / Stone / Tumour),
built for a LinkedIn write-up. **All metrics are computed from the clean, deduplicated
test set (n = 1,888)** — the honest numbers after an image-level data leak was found and
fixed (the earlier 99–100% figures were leakage-inflated).

## Charts (1200×1200 PNG)
- `chart1_leakage_dumbbell.png` — leaky vs clean macro-F1 for both CNNs (99% → 82%).
- `chart2_data_efficiency.png` — macro-F1 vs training-set size; the classical SVM baseline leads at every fraction.
- `chart3_confusion.png` — ConvNeXt V2 clean-test confusion matrix (Stone is the weak class, 63% recall).

## Animations (1200×1200 GIF, looping)
- `gif1_gradcam_reveal.gif` — Grad-CAM reveal on a kidney CT (ConvNeXt V2).
- `gif2_occlusion_sweep.gif` — occlusion-sensitivity sweep.
- `gif3_attention_compare.gif` — same scan, three models: classical vs EfficientNet-B0 vs ConvNeXt V2.

Numbers recomputed directly from the prediction files under `Results/`.
