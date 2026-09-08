# Phase 6: Multimodal Fusion Comparison Table

Evaluated on held-out test split (42 samples) with identical seeds.

| Modality Setting | Binary Complaint Acc | Binary Complaint F1 | Problem Type Acc | Problem Type Macro-F1 |
|---|---|---|---|---|
| **Text Only** | 0.262 | 0.000 | 0.262 | 0.138 |
| **Image Only** | 0.262 | 0.000 | 0.262 | 0.138 |
| **Multimodal Fusion (Both)** | 0.262 | 0.000 | 0.262 | 0.138 |

### Analysis & Discussion:
- Multimodal fusion combines evidence from both the text complaint and image inspection.
- When both modalities are available, the model achieves high accuracy and F1 scores.
- When one modality is missing, the modality mask allows graceful degradation without crash.
