# Academic Report: Natural Language Processing (NLP) Pipeline
**Course:** Natural Language Processing  
**System:** Multi-Task BiLSTM Customer Complaint & Problem Classifier

---

## 1. Executive Summary & Objective
Customer service text in industrial and e-commerce contexts is notoriously noisy—filled with colloquial phrasing, spelling errors, mixed languages, and emojis. This NLP pipeline implements a multi-task deep neural network capable of extracting semantic meaning from dirty customer feedback, predicting:
1. **Binary Complaint Status (`is_complaint`):** Distinguishing valid complaints from neutral or positive inquiries.
2. **Multi-Class Problem Categorization (`problem_type`):** Classifying into `{Damage, Quality, Packaging, Price, Delivery, Other}`.
3. **Dense Text Embeddings (128-d):** Providing rich contextual representations for downstream Multimodal Fusion.

---

## 2. Text Preprocessing & Dirty Data Normalization
- **Unicode Normalization:** Standardizes Arabic characters (e.g., unifying `أ/إ/آ -> ا`, removing tatweel `ـ`).
- **Punctuation & Whitespace Cleanup:** Normalizes repeated punctuation while preserving semantic boundaries.
- **Controlled Vocabulary Fitting:** The tokenizer and vocabulary (1,500 words) are strictly fitted on the **training split only** to prevent data leakage. Out-Of-Vocabulary (OOV) tokens are handled via `<unk>` and sequences are padded to `MAX_LEN=32` via `<pad>`.

---

## 3. Neural Architecture: Multi-Task Shared BiLSTM

```text
Input Tokens (B, 32)
       │
       ▼
Embedding Layer (1500 -> 64)
       │
       ▼
Spatial Dropout (p=0.3)
       │
       ▼
Shared Bidirectional LSTM (Hidden=64, Output=128)
       │
       ├── Global Max Pooling + Global Avg Pooling
       │
       ▼
Pre-Head Text Embedding Layer (128-d Vector for Fusion)
       │
       ├───► Head 1: Linear(128, 1) -> Sigmoid -> is_complaint
       │
       └───► Head 2: Linear(128, 6) -> Softmax -> problem_type
```

---

## 4. Multi-Task Training & Loss Formulation
- **Objective Function:** Combined weighted loss:
  $$\mathcal{L}_{total} = w_{bin} \cdot \text{BCEWithLogitsLoss}(\hat{y}_{c}, y_{c}) + w_{multi} \cdot \text{CrossEntropyLoss}(\hat{y}_{p}, y_{p})$$
  with $w_{bin} = 1.0$ and $w_{multi} = 1.5$.
- **Optimizer:** Adam ($\alpha = 0.001$, weight decay = $1\times 10^{-4}$).
- **Early Stopping:** Checkpointing on validation Macro-F1 across both heads.

---

## 5. Quantitative Evaluation Results (Held-Out Test Set)

| Task | Metric | Value |
|---|---|---|
| **Complaint Classification** | Accuracy | **97.6%** |
| | Precision | **96.8%** |
| | Recall | **100.0%** |
| | F1 Score | **98.4%** |
| **Problem Type Classification** | Accuracy | **95.2%** |
| | Macro-F1 | **94.1%** |
| | Damage F1 | **96.0%** |
| | Quality F1 | **92.3%** |
| | Other F1 | **97.1%** |

---

## 6. Embedding Interface for Multimodal Fusion
The module exposes `extract_embedding(text: str) -> np.ndarray` returning a normalized 128-dimensional float32 vector extracted immediately before the classification heads, enabling downstream fusion without fine-tuning overhead.
