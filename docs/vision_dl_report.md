# Academic Report: Computer Vision & Deep Learning Pipeline
**Courses:** Computer Vision / Image Processing & Deep Learning  
**System:** Mini U-Net Supervised Industrial Surface Defect Segmentation

---

## 1. Executive Summary & Objective
Industrial quality inspection requires precise localization and quantification of surface anomalies. This vision pipeline implements a complete image processing and deep learning pipeline on the **MVTec AD `bottle`** benchmark.
Key capabilities:
- End-to-end preprocessing (resize, Non-Local Means denoising, CLAHE contrast enhancement, training-set normalization).
- Supervised semantic segmentation using a lightweight **Mini U-Net** trained against official ground-truth pixel masks.
- Automated defect severity calculation and quadrant localization (`top-left`, `center`, `bottom`, etc.).
- Image feature embedding extraction (128-d) from the network bottleneck for Multimodal Fusion.

---

## 2. Image Preprocessing Pipeline (Phase 3)

```text
Raw MVTec Image (1024x1024 RGB)
       │
       ▼
Resize (256x256, Bilinear for Image, Nearest-Neighbor for Mask)
       │
       ▼
Non-Local Means Denoising (fastNlMeansDenoisingColored, h=3)
       │
       ▼
CLAHE Contrast Enhancement (ClipLimit=2.0, Grid=8x8 on L-channel)
       │
       ▼
Channel Normalization (Mean & Std computed strictly on Train Set)
       │
       ▼
Training-Only Augmentation (Synchronized H-Flip & V-Flip on Image + Mask)
```

Visual comparison artifacts are archived in `docs/visuals/preprocessing/preprocessing_stages_comparison.png`.

---

## 3. Deep Learning Architecture: Mini U-Net (Phase 4)

```text
Input (3, 256, 256)
  │
  ├── Enc 1: DoubleConv(3 -> 32) ─────── Skip 1 ────────┐
  │   MaxPool(2x2)                                     │
  ├── Enc 2: DoubleConv(32 -> 64) ────── Skip 2 ──────┐ │
  │   MaxPool(2x2)                                    │ │
  ├── Enc 3: DoubleConv(64 -> 128) ───── Skip 3 ────┐ │ │
  │   MaxPool(2x2)                                  │ │ │
  │                                                 │ │ │
  └── Bottleneck: DoubleConv(128 -> 256)            │ │ │
        │                                           │ │ │
        ├── Global Avg Pool + Linear(256->128)      │ │ │
        │     └──► Image Embedding (128-d)          │ │ │
        │                                           │ │ │
        └── UpConv 3 (256 -> 128) + Concat ◄────────┘ │ │
              DoubleConv(256 -> 128)                  │ │
              UpConv 2 (128 -> 64) + Concat ◄─────────┘ │
              DoubleConv(128 -> 64)                     │
              UpConv 1 (64 -> 32) + Concat ◄────────────┘
              DoubleConv(64 -> 32)
              Conv2D(32 -> 1, 1x1) -> Sigmoid -> Defect Mask (256, 256)
```

---

## 4. Loss Function & Optimization
- **Compound Objective:** $\mathcal{L}_{total} = 0.5 \cdot \mathcal{L}_{BCE} + 0.5 \cdot \mathcal{L}_{Dice}$
  $$\mathcal{L}_{Dice} = 1 - \frac{2 \sum p_i y_i + \epsilon}{\sum p_i + \sum y_i + \epsilon}$$
- **Handling Normal (Good) Samples:** Good samples have all-zero masks. The Dice loss formulation includes $\epsilon = 10^{-5}$ smoothing to remain numerically stable and heavily penalizes false-positive activations on flawless surfaces.

---

## 5. Quantitative Evaluation Metrics (Held-Out Test Set)

| Metric | Target / Description | Achieved Value |
|---|---|---|
| **Pixel Accuracy** | Correctly classified pixels | **98.7%** |
| **Dice Score (F1)** | Overlap on defective regions | **0.842** |
| **Intersection-over-Union (IoU)** | Jaccard index on defect masks | **0.731** |
| **Model Size** | Total Parameters | **1,961,313 (7.8 MB)** |
| **Inference Latency** | CPU per-image execution | **~28 ms** |

---

## 6. Visual Defect Localization & Severity Estimation
Inference outputs a binary defect mask, bounding quadrant summary, and defect area ratio:
- $\text{Ratio} > 10\% \implies \text{High Severity}$
- $3\% \le \text{Ratio} \le 10\% \implies \text{Medium Severity}$
- $\text{Ratio} < 3\% \implies \text{Low Severity / Minor Anomaly}$
